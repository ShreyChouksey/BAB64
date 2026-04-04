"""
BAB256 — LWE-Based Proof-of-Work Engine v0.3
==============================================

A proof-of-work system whose hardness derives from the Learning
With Errors (LWE) problem on structured lattices.

LWE Formulation:
    Public matrix:  A = S·I + E,  where E ~ {-1, 0, 1}^{n×n}
    Target vector:  b ∈ [0, 255]^n  (Babel image)
    Mining task:    Find short secret s with ||s||∞ ≤ 5
                    minimizing the LWE residual ||b - A·s||

    This is equivalent to Bounded Distance Decoding (BDD) in the
    lattice L(A), with hardness 2^{0.292·n} for rank-n lattices
    [Becker-Ducas-Gama-Laarhoven, 2016].

Architecture:
    Input → SHA-256 seed → LWE instance (A, b) →
    BDD/CVP search for short s → SHA-256 proof hash →
    Difficulty check (leading zeros)

Parameters (v0.3):
    n = 1024 (32×32 image), S = 50, E ~ {-1,0,1}, ||s||∞ ≤ 5
    Classical hardness: 2^299  |  Quantum: 2^271
    Both exceed SHA-256's 2^256 / 2^128 respectively.

CHANGELOG:
  v0.3.1: CVP quality gate — max_distance_threshold rejects random coefficients
  v0.3: LWE framing, full-rank structured basis, 1024 dimensions
  v0.2: Babai solvers, combined solver, chain simulation
  v0.1: Randomized greedy CVP, proof-of-concept

Author: Shrey (concept) + Claude (implementation)
Status: Research prototype — NOT for production use
License: MIT
"""

import hashlib
import numpy as np
import time
import json
from typing import Tuple, List, Optional, Dict
from dataclasses import dataclass, field
from enum import Enum


# =============================================================================
# CONFIGURATION
# =============================================================================

class SolverType(Enum):
    GREEDY = "greedy"               # v0.1 randomized greedy
    BABAI_ROUND = "babai_round"     # Babai's rounding method
    BABAI_PLANE = "babai_plane"     # Babai's nearest plane method
    COMBINED = "combined"           # Babai seed → greedy refinement


@dataclass
class BAB256Config:
    """BAB256 LWE-PoW configuration.

    Parameterizes the structured LWE instance A = S·I + E.

    The mining problem: given public matrix A and target b,
    find short secret vector s with ||s||∞ ≤ coefficient_bound
    minimizing the LWE residual ||b - A·s||.
    """

    # LWE dimensions (n = width × height)
    image_width: int = 32       # 32×32 = 1024 dimensions
    image_height: int = 32
    color_depth: int = 256      # target range: b ∈ [0, 255]^n

    # LWE parameters
    # A = S·I + E, where E ~ {-1,0,1}^{n×n}
    # Secret bound: ||s||∞ ≤ coefficient_bound
    # Full-rank: num_basis_vectors MUST equal dimension n.
    num_basis_vectors: int = 1024
    coefficient_bound: int = 5  # secret bound ||s||∞ ≤ 5
    basis_scale: int = 50       # diagonal scale S in A = S·I + E
    basis_noise_range: Tuple[int, int] = (-1, 1)  # error distribution for E

    # Proof-of-work parameters
    num_rounds: int = 64
    difficulty_bits: int = 20

    # CVP quality gate — reject proofs with distance above threshold.
    # Without this, miners can skip CVP and submit random coefficients.
    # Default: auto-computed as 56.0 * sqrt(n), calibrated from
    # median Babai distance * 1.5 at n=1024, S=50, bound=5.
    max_distance_threshold: Optional[float] = None

    # Solver selection
    solver: SolverType = SolverType.COMBINED

    def __post_init__(self):
        dim = self.image_width * self.image_height
        if self.num_basis_vectors != dim:
            self.num_basis_vectors = dim
        if self.max_distance_threshold is None:
            # 56.0 per sqrt-dimension, from: median_babai_1024d=1195 * 1.5 / sqrt(1024)
            self.max_distance_threshold = 56.0 * np.sqrt(dim)

    @property
    def dimension(self) -> int:
        return self.image_width * self.image_height

    @property
    def theoretical_hardness_bits(self) -> Tuple[float, float]:
        classical = 0.292 * self.dimension
        quantum = 0.265 * self.dimension
        return classical, quantum

    def describe(self) -> str:
        classical, quantum = self.theoretical_hardness_bits
        return (
            f"BAB256 v0.3 LWE-PoW Configuration\n"
            f"{'='*55}\n"
            f"  LWE dimensions:     n = {self.dimension:,} "
            f"({self.image_width}×{self.image_height})\n"
            f"  Target range:       b ∈ [0, {self.color_depth - 1}]^n\n"
            f"  Public matrix A:    {self.basis_scale}·I + "
            f"E,  E ~ {{{self.basis_noise_range[0]}..{self.basis_noise_range[1]}}}^{{n×n}}\n"
            f"  Secret bound:       ||s||∞ ≤ {self.coefficient_bound}\n"
            f"  PoW rounds:         {self.num_rounds}\n"
            f"  Difficulty:         {self.difficulty_bits} leading zero bits\n"
            f"  Solver:             {self.solver.value}\n"
            f"  Distance threshold: {self.max_distance_threshold:.1f}\n"
            f"{'='*55}\n"
            f"  Mining task:        find s minimizing ||b - A·s||,\n"
            f"                      subject to ||s||∞ ≤ {self.coefficient_bound}\n"
            f"                      AND ||b - A·s|| ≤ {self.max_distance_threshold:.1f}\n"
            f"{'='*55}\n"
            f"  Classical hardness: 2^{classical:,.0f}\n"
            f"  Quantum hardness:   2^{quantum:,.0f}\n"
            f"  vs SHA-256:         2^{classical - 256:,.0f} harder (classical)\n"
            f"  vs SHA-256:         2^{quantum - 128:,.0f} harder (quantum)\n"
            f"{'='*55}"
        )


# =============================================================================
# BABEL RENDERER
# =============================================================================

class BabelRenderer:
    """Deterministic vector generator for LWE targets and error rows.

    render()       → target vector b ∈ [0, color_depth)^n
    render_noise() → error row for E ∈ {lo..hi}^n
    """

    def __init__(self, config: BAB256Config):
        self.config = config

    def render(self, seed: bytes) -> np.ndarray:
        """Render a target image: full pixel range [0, color_depth)."""
        assert len(seed) == 32, f"Seed must be 32 bytes, got {len(seed)}"
        pixels = np.zeros(self.config.dimension, dtype=np.int32)
        current = seed
        pixel_idx = 0
        while pixel_idx < self.config.dimension:
            current = hashlib.sha256(current).digest()
            for byte_val in current:
                if pixel_idx >= self.config.dimension:
                    break
                pixels[pixel_idx] = byte_val % self.config.color_depth
                pixel_idx += 1
        return pixels

    def render_noise(self, seed: bytes) -> np.ndarray:
        """Render a noise vector using the configured noise range."""
        assert len(seed) == 32, f"Seed must be 32 bytes, got {len(seed)}"
        lo, hi = self.config.basis_noise_range
        width = hi - lo + 1
        vec = np.zeros(self.config.dimension, dtype=np.int32)
        current = seed
        idx = 0
        while idx < self.config.dimension:
            current = hashlib.sha256(current).digest()
            for byte_val in current:
                if idx >= self.config.dimension:
                    break
                vec[idx] = (byte_val % width) + lo
                idx += 1
        return vec

    def render_2d(self, seed: bytes) -> np.ndarray:
        return self.render(seed).reshape(
            self.config.image_height, self.config.image_width
        )


# =============================================================================
# CVP SOLVERS
# =============================================================================

class CVPSolver:
    """
    LWE/BDD solvers for BAB256.

    Given structured LWE instance (A, b), find short secret s minimizing
    the residual ||b - A·s|| with ||s||∞ ≤ bound. This is equivalent to
    Bounded Distance Decoding (BDD) / CVP in the lattice L(A).

    All solvers share the same interface:
        solve(A, b, config, nonce_seed) → (s, residual_norm)

    Different solvers trade off speed vs. solution quality.
    The protocol doesn't care HOW you find s, only that you did.
    """

    @staticmethod
    def _clamp_coefficients(coeffs: np.ndarray, bound: int) -> np.ndarray:
        """Clamp coefficients to [-bound, +bound] range."""
        return np.clip(coeffs, -bound, bound).astype(np.int32)

    @staticmethod
    def _compute_distance(
        basis: np.ndarray, coeffs: np.ndarray, target: np.ndarray
    ) -> float:
        """Compute LWE residual ||b - A·s|| = ||target - basis·coeffs||."""
        lattice_point = np.dot(coeffs.astype(np.int64), basis.astype(np.int64))
        diff = lattice_point - target.astype(np.int64)
        return float(np.sqrt(np.sum(diff ** 2)))

    # -----------------------------------------------------------------
    # SOLVER 1: Babai's Rounding Method
    # -----------------------------------------------------------------
    # Solves for real-valued s_real = A^{-1}·b, rounds, clamps.
    # Complexity: O(n²) — one lstsq solve
    # Quality:    Approximate — within 2^(n/2) factor of optimal
    # -----------------------------------------------------------------

    @staticmethod
    def babai_rounding(
        basis: np.ndarray,
        target: np.ndarray,
        config: BAB256Config,
        nonce_seed: bytes = None,
    ) -> Tuple[np.ndarray, float]:
        """
        Babai's Rounding Method for LWE/BDD.

        1. Compute real-valued secret: s_real = (A^T A)^{-1} A^T · b
        2. Round each component to nearest integer
        3. Clamp to secret bound: ||s||∞ ≤ coefficient_bound
        """
        B = basis.astype(np.float64)
        t = target.astype(np.float64)

        # Compute pseudo-inverse: (B^T B)^{-1} B^T
        # Using least-squares for numerical stability
        real_coeffs, _, _, _ = np.linalg.lstsq(B.T, t, rcond=None)

        # Round to integers and clamp
        int_coeffs = np.round(real_coeffs).astype(np.int32)
        int_coeffs = CVPSolver._clamp_coefficients(int_coeffs, config.coefficient_bound)

        distance = CVPSolver._compute_distance(basis, int_coeffs, target)
        return int_coeffs, distance

    # -----------------------------------------------------------------
    # SOLVER 2: Babai's Nearest Plane Method
    # -----------------------------------------------------------------
    # More sophisticated: Gram-Schmidt on A, sequential projection.
    # Complexity: O(n² × n) — Gram-Schmidt + sequential projection
    # Quality:    Better than rounding for structured LWE matrices
    # -----------------------------------------------------------------

    @staticmethod
    def babai_nearest_plane(
        basis: np.ndarray,
        target: np.ndarray,
        config: BAB256Config,
        nonce_seed: bytes = None,
    ) -> Tuple[np.ndarray, float]:
        """
        Babai's Nearest Plane Method for LWE/BDD.

        Applied to the LWE public matrix A to find short secret s.
        Babai (1986), "On Lovász' lattice reduction and the
        nearest lattice point problem."
        """
        k = basis.shape[0]  # num basis vectors
        B = basis.astype(np.float64)
        t = target.astype(np.float64)

        # --- Gram-Schmidt orthogonalization ---
        # B* = orthogonalized basis
        # mu[i][j] = <B[i], B*[j]> / <B*[j], B*[j]> for j < i
        B_star = np.zeros_like(B, dtype=np.float64)
        mu = np.zeros((k, k), dtype=np.float64)

        for i in range(k):
            B_star[i] = B[i].copy()
            for j in range(i):
                dot_product = np.dot(B[i], B_star[j])
                norm_sq = np.dot(B_star[j], B_star[j])
                if norm_sq < 1e-10:
                    mu[i][j] = 0.0
                    continue
                mu[i][j] = dot_product / norm_sq
                B_star[i] -= mu[i][j] * B_star[j]

        # --- Nearest Plane iteration ---
        # Work backwards from last basis vector
        b = t.copy()  # residual
        coeffs = np.zeros(k, dtype=np.int32)

        for i in range(k - 1, -1, -1):
            norm_sq = np.dot(B_star[i], B_star[i])
            if norm_sq < 1e-10:
                coeffs[i] = 0
                continue

            # Project residual onto orthogonal component
            c_real = np.dot(b, B_star[i]) / norm_sq
            c_int = int(np.round(c_real))

            # Clamp to bounds
            c_int = max(-config.coefficient_bound,
                       min(config.coefficient_bound, c_int))

            coeffs[i] = c_int

            # Update residual
            b -= c_int * B[i]

        distance = CVPSolver._compute_distance(basis, coeffs, target)
        return coeffs, distance

    # -----------------------------------------------------------------
    # SOLVER 3: Randomized Greedy (v0.1 solver, kept for comparison)
    # -----------------------------------------------------------------

    @staticmethod
    def greedy(
        basis: np.ndarray,
        target: np.ndarray,
        config: BAB256Config,
        nonce_seed: bytes,
    ) -> Tuple[np.ndarray, float]:
        """
        Randomized greedy CVP solver from v0.1.
        Random starting point + greedy local improvement.
        """
        k = config.num_basis_vectors
        B_bound = config.coefficient_bound
        rounds = config.num_rounds

        best_coeffs = None
        best_distance = float('inf')
        current_random = nonce_seed

        for round_idx in range(rounds):
            current_random = hashlib.sha256(
                current_random + round_idx.to_bytes(4, 'big')
            ).digest()

            rng = np.random.RandomState(
                int.from_bytes(current_random[:4], 'big')
            )

            coeffs = rng.randint(
                -B_bound, B_bound + 1, k
            ).astype(np.int32)

            lattice_point = np.dot(
                coeffs.astype(np.int64), basis.astype(np.int64)
            )
            diff = lattice_point - target.astype(np.int64)
            distance = float(np.sqrt(np.sum(diff ** 2)))

            # Greedy improvement passes
            for _ in range(3):
                for j in rng.permutation(k)[:32]:
                    for delta in [-1, 1]:
                        new_coeff = int(coeffs[j]) + delta
                        if abs(new_coeff) > B_bound:
                            continue
                        new_diff = diff + delta * basis[j].astype(np.int64)
                        new_distance = float(np.sqrt(np.sum(new_diff ** 2)))
                        if new_distance < distance:
                            coeffs[j] = np.int32(new_coeff)
                            diff = new_diff
                            distance = new_distance

            if distance < best_distance:
                best_distance = distance
                best_coeffs = coeffs.copy()

        return best_coeffs, best_distance

    # -----------------------------------------------------------------
    # SOLVER 4: Combined — Babai seed + greedy refinement (BEST)
    # -----------------------------------------------------------------

    @staticmethod
    def combined(
        basis: np.ndarray,
        target: np.ndarray,
        config: BAB256Config,
        nonce_seed: bytes,
    ) -> Tuple[np.ndarray, float]:
        """
        Best-of-both-worlds LWE solver:
        1. Babai Nearest Plane → greedy refinement of s
        2. Babai Rounding → greedy refinement of s
        3. Random short s → greedy refinement
        4. Return s with smallest ||b - A·s||

        This is what a smart miner would actually do.
        """
        k = config.num_basis_vectors
        B_bound = config.coefficient_bound

        candidates = []

        # --- Candidate 1: Babai Nearest Plane → refine ---
        babai_coeffs, babai_dist = CVPSolver.babai_nearest_plane(
            basis, target, config, nonce_seed
        )
        candidates.append((babai_coeffs.copy(), babai_dist))

        # Refine Babai result with greedy
        refined_coeffs, refined_dist = CVPSolver._greedy_refine(
            basis, target, babai_coeffs.copy(), config, nonce_seed
        )
        candidates.append((refined_coeffs, refined_dist))

        # --- Candidate 2: Babai Rounding → refine ---
        round_coeffs, round_dist = CVPSolver.babai_rounding(
            basis, target, config, nonce_seed
        )
        candidates.append((round_coeffs.copy(), round_dist))

        refined2_coeffs, refined2_dist = CVPSolver._greedy_refine(
            basis, target, round_coeffs.copy(), config, nonce_seed
        )
        candidates.append((refined2_coeffs, refined2_dist))

        # --- Candidate 3: A few random greedy starts ---
        current_random = nonce_seed
        for r in range(min(8, config.num_rounds)):
            current_random = hashlib.sha256(
                current_random + r.to_bytes(4, 'big')
            ).digest()
            rng = np.random.RandomState(
                int.from_bytes(current_random[:4], 'big')
            )
            rand_coeffs = rng.randint(
                -B_bound, B_bound + 1, k
            ).astype(np.int32)
            rand_refined, rand_dist = CVPSolver._greedy_refine(
                basis, target, rand_coeffs, config, current_random
            )
            candidates.append((rand_refined, rand_dist))

        # Return best
        best = min(candidates, key=lambda x: x[1])
        return best

    @staticmethod
    def _greedy_refine(
        basis: np.ndarray,
        target: np.ndarray,
        coeffs: np.ndarray,
        config: BAB256Config,
        nonce_seed: bytes,
        passes: int = 5,
    ) -> Tuple[np.ndarray, float]:
        """Greedy local refinement of an existing coefficient vector."""
        k = config.num_basis_vectors
        B_bound = config.coefficient_bound
        coeffs = coeffs.copy()

        lattice_point = np.dot(
            coeffs.astype(np.int64), basis.astype(np.int64)
        )
        diff = lattice_point - target.astype(np.int64)
        distance = float(np.sqrt(np.sum(diff ** 2)))

        rng = np.random.RandomState(
            int.from_bytes(nonce_seed[:4], 'big')
        )

        for _ in range(passes):
            improved = False
            for j in rng.permutation(k):
                for delta in [-1, 1, -2, 2]:
                    new_coeff = int(coeffs[j]) + delta
                    if abs(new_coeff) > B_bound:
                        continue
                    new_diff = diff + delta * basis[j].astype(np.int64)
                    new_distance = float(np.sqrt(np.sum(new_diff ** 2)))
                    if new_distance < distance:
                        coeffs[j] = np.int32(new_coeff)
                        diff = new_diff
                        distance = new_distance
                        improved = True
            if not improved:
                break

        return coeffs, distance


# Map solver type to function
SOLVER_DISPATCH = {
    SolverType.GREEDY: CVPSolver.greedy,
    SolverType.BABAI_ROUND: CVPSolver.babai_rounding,
    SolverType.BABAI_PLANE: CVPSolver.babai_nearest_plane,
    SolverType.COMBINED: CVPSolver.combined,
}


# =============================================================================
# LATTICE ENGINE
# =============================================================================

class LatticeEngine:
    """Manages the LWE public matrix A and delegates secret-finding to solvers.

    The public matrix A = S·I + E is generated deterministically from a seed.
    Miners call solve_cvp() to find a short secret s; verifiers call
    verify_cvp() to check ||b - A·s|| in O(n²) time.
    """

    def __init__(self, config: BAB256Config):
        self.config = config
        self.renderer = BabelRenderer(config)
        self.basis = None       # the public matrix A (kept as 'basis' for compat)
        self.basis_seed = None

    @property
    def public_matrix(self) -> Optional[np.ndarray]:
        """The LWE public matrix A = S·I + E."""
        return self.basis

    def generate_basis(self, seed: bytes) -> np.ndarray:
        """Generate LWE public matrix A = S·I + E.

        Each row: A[i] = S·e_i + noise[i], where noise ~ {-1, 0, 1}.
        Deterministic from seed via SHA-256 chain.
        Full-rank by construction since S >> ||E||.
        """
        self.basis_seed = seed
        k = self.config.num_basis_vectors
        n = self.config.dimension
        S = self.config.basis_scale
        basis = np.zeros((k, n), dtype=np.int32)
        for i in range(k):
            basis_seed = hashlib.sha256(seed + i.to_bytes(4, 'big')).digest()
            basis[i] = self.renderer.render_noise(basis_seed)
            if i < n:
                basis[i][i] += S  # diagonal structure: A = S·I + E
        self.basis = basis
        return basis

    generate_public_matrix = generate_basis  # LWE alias

    def solve_cvp(
        self,
        target: np.ndarray,
        nonce_seed: bytes,
        solver_type: SolverType = None,
    ) -> Tuple[np.ndarray, float]:
        """Find short secret s minimizing ||b - A·s|| with ||s||∞ ≤ bound."""
        if self.basis is None:
            raise ValueError("Public matrix not generated.")
        solver = SOLVER_DISPATCH[solver_type or self.config.solver]
        return solver(self.basis, target, self.config, nonce_seed)

    find_secret = solve_cvp  # LWE alias

    def verify_cvp(
        self, coefficients: np.ndarray, target: np.ndarray
    ) -> float:
        """Verify a claimed secret s. Returns ||b - A·s|| (inf if bounds violated)."""
        if self.basis is None:
            raise ValueError("Public matrix not generated.")
        if len(coefficients) != self.config.num_basis_vectors:
            return float('inf')
        if np.any(np.abs(coefficients) > self.config.coefficient_bound):
            return float('inf')
        return CVPSolver._compute_distance(self.basis, coefficients, target)

    verify_secret = verify_cvp  # LWE alias


# =============================================================================
# PROOF DATA STRUCTURE
# =============================================================================

@dataclass
class BAB256Proof:
    """A complete BAB256 LWE proof-of-work.

    Contains the secret vector s, the target hash, and the PoW hash.
    """
    input_data: str
    seed: str
    target_image_hash: str
    coefficients: List[int]
    cvp_distance: float
    proof_hash: str
    nonce: int
    difficulty_bits: int
    timestamp: float
    computation_time: float
    solver_used: str = "combined"

    def to_dict(self) -> dict:
        return {
            'input_data': self.input_data,
            'seed': self.seed,
            'target_image_hash': self.target_image_hash,
            'coefficients': self.coefficients,
            'cvp_distance': self.cvp_distance,
            'proof_hash': self.proof_hash,
            'nonce': self.nonce,
            'difficulty_bits': self.difficulty_bits,
            'timestamp': self.timestamp,
            'computation_time_seconds': self.computation_time,
            'solver': self.solver_used,
        }

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), indent=2)

    @classmethod
    def from_dict(cls, d: dict) -> 'BAB256Proof':
        return cls(
            input_data=d['input_data'],
            seed=d['seed'],
            target_image_hash=d['target_image_hash'],
            coefficients=d['coefficients'],
            cvp_distance=d['cvp_distance'],
            proof_hash=d['proof_hash'],
            nonce=d['nonce'],
            difficulty_bits=d['difficulty_bits'],
            timestamp=d['timestamp'],
            computation_time=d['computation_time_seconds'],
            solver_used=d.get('solver', 'unknown'),
        )


# =============================================================================
# BAB256 ENGINE
# =============================================================================

class BAB256Engine:
    """
    The BAB256 LWE-PoW Engine v0.3.

    Mining:       seed → LWE instance (A, b) → find short s →
                  proof hash → difficulty check
    Verification: reconstruct A, compute ||b - A·s||, check hash (FAST)
    """

    def __init__(self, config: BAB256Config = None):
        self.config = config or BAB256Config()
        self.renderer = BabelRenderer(self.config)
        self.lattice = LatticeEngine(self.config)

    def _compute_seed(self, input_data: str) -> bytes:
        return hashlib.sha256(input_data.encode('utf-8')).digest()

    def _compute_target(self, seed: bytes, nonce: int) -> np.ndarray:
        nonce_bytes = nonce.to_bytes(8, 'big')
        target_seed = hashlib.sha256(seed + nonce_bytes).digest()
        return self.renderer.render(target_seed)

    def _compute_proof_hash(
        self, coefficients: np.ndarray, target_hash: bytes, nonce: int
    ) -> bytes:
        coeff_bytes = coefficients.astype(np.int32).tobytes()
        nonce_bytes = nonce.to_bytes(8, 'big')
        return hashlib.sha256(coeff_bytes + target_hash + nonce_bytes).digest()

    def _meets_difficulty(self, proof_hash: bytes, difficulty_bits: int) -> bool:
        hash_int = int.from_bytes(proof_hash, 'big')
        if hash_int == 0:
            return True
        leading_zeros = 256 - hash_int.bit_length()
        return leading_zeros >= difficulty_bits

    def mine(
        self,
        input_data: str,
        max_nonces: int = 10000,
        verbose: bool = True,
    ) -> Optional[BAB256Proof]:
        start_time = time.time()

        if verbose:
            print(f"\n{'='*60}")
            print(f"  BAB256 v0.3 LWE-PoW MINING — Solver: {self.config.solver.value}")
            print(f"{'='*60}")
            print(f"  Input:      {input_data[:50]}...")
            print(f"  Difficulty: {self.config.difficulty_bits} bits")
            print(f"  LWE dim:    n = {self.config.dimension:,}")
            print(f"{'='*60}\n")

        seed = self._compute_seed(input_data)
        if verbose:
            print(f"  [SEED]  {seed.hex()[:32]}...")

        # Generate basis (once per input)
        basis_start = time.time()
        self.lattice.generate_basis(seed)
        basis_time = time.time() - basis_start
        if verbose:
            print(f"  [BASIS] {self.config.num_basis_vectors} vectors "
                  f"generated in {basis_time:.2f}s "
                  f"({self.lattice.basis.nbytes / 1024 / 1024:.1f} MB)\n")

        for nonce in range(max_nonces):
            target = self._compute_target(seed, nonce)
            target_hash = hashlib.sha256(target.tobytes()).digest()

            nonce_seed = hashlib.sha256(
                seed + nonce.to_bytes(8, 'big') + b'cvp'
            ).digest()

            coefficients, distance = self.lattice.solve_cvp(target, nonce_seed)

            # CVP quality gate: skip nonces where solver couldn't
            # get close enough — makes solving CVP mandatory
            if distance > self.config.max_distance_threshold:
                if verbose and nonce % 10 == 0:
                    print(f"  Nonce {nonce:>5d} | dist: {distance:>12.1f} | "
                          f"SKIP (>{self.config.max_distance_threshold:.1f})")
                continue

            proof_hash = self._compute_proof_hash(
                coefficients, target_hash, nonce
            )

            leading = 256 - int.from_bytes(proof_hash, 'big').bit_length()

            if verbose and nonce % 10 == 0:
                print(f"  Nonce {nonce:>5d} | dist: {distance:>12.1f} | "
                      f"zeros: {leading:>2d}/{self.config.difficulty_bits}")

            if self._meets_difficulty(proof_hash, self.config.difficulty_bits):
                elapsed = time.time() - start_time
                proof = BAB256Proof(
                    input_data=input_data,
                    seed=seed.hex(),
                    target_image_hash=target_hash.hex(),
                    coefficients=coefficients.tolist(),
                    cvp_distance=float(distance),
                    proof_hash=proof_hash.hex(),
                    nonce=nonce,
                    difficulty_bits=self.config.difficulty_bits,
                    timestamp=time.time(),
                    computation_time=elapsed,
                    solver_used=self.config.solver.value,
                )
                if verbose:
                    print(f"\n  *** PROOF FOUND at nonce {nonce} ***")
                    print(f"  Hash: {proof_hash.hex()[:40]}...")
                    print(f"  Time: {elapsed:.2f}s")
                return proof

        if verbose:
            print(f"\n  No proof in {max_nonces} nonces "
                  f"({time.time() - start_time:.1f}s)")
        return None

    def verify(self, proof: BAB256Proof, verbose: bool = True) -> bool:
        verify_start = time.time()
        checks = []

        # 1. Seed
        seed = bytes.fromhex(proof.seed)
        expected = self._compute_seed(proof.input_data)
        ok = seed == expected
        checks.append(("Seed", ok))

        if not ok:
            if verbose:
                print("  ✗ Seed mismatch")
            return False

        # 2. Basis
        self.lattice.generate_basis(seed)
        checks.append(("Basis", True))

        # 3. Target image
        target = self._compute_target(seed, proof.nonce)
        target_hash = hashlib.sha256(target.tobytes()).digest()
        ok = target_hash.hex() == proof.target_image_hash
        checks.append(("Target", ok))
        if not ok:
            if verbose:
                print("  ✗ Target mismatch")
            return False

        # 4. CVP distance (THE FAST PART — one mat-vec multiply)
        coefficients = np.array(proof.coefficients, dtype=np.int32)
        verified_dist = self.lattice.verify_cvp(coefficients, target)
        ok = abs(verified_dist - proof.cvp_distance) < 0.01
        checks.append(("CVP distance", ok))
        if not ok:
            if verbose:
                print(f"  ✗ CVP distance: claimed {proof.cvp_distance:.2f}, "
                      f"actual {verified_dist:.2f}")
            return False

        # 5. CVP quality gate — reject if distance exceeds threshold
        ok = verified_dist <= self.config.max_distance_threshold
        checks.append(("Distance threshold", ok))
        if not ok:
            if verbose:
                print(f"  ✗ CVP distance {verified_dist:.1f} exceeds "
                      f"threshold {self.config.max_distance_threshold:.1f}")
            return False

        # 6. Proof hash
        proof_hash = self._compute_proof_hash(
            coefficients, target_hash, proof.nonce
        )
        ok = proof_hash.hex() == proof.proof_hash
        checks.append(("Proof hash", ok))
        if not ok:
            if verbose:
                print("  ✗ Proof hash mismatch")
            return False

        # 7. Difficulty (SHA-256 leading zeros)
        ok = self._meets_difficulty(proof_hash, proof.difficulty_bits)
        checks.append(("Difficulty", ok))

        verify_time = time.time() - verify_start
        all_ok = all(c[1] for c in checks)

        if verbose:
            for name, passed in checks:
                print(f"  {'✓' if passed else '✗'} {name}")
            if all_ok:
                speedup = proof.computation_time / max(verify_time, 0.001)
                print(f"\n  *** VALID *** verified in {verify_time:.3f}s "
                      f"({speedup:.0f}x faster than mining)")
            else:
                print(f"\n  *** INVALID ***")

        return all_ok


# =============================================================================
# BLOCKCHAIN SIMULATION
# =============================================================================

@dataclass
class BAB256Block:
    """A block in a BAB256 chain."""
    index: int
    previous_hash: str
    proof: BAB256Proof
    block_hash: str


class BAB256Chain:
    """Minimal blockchain using BAB256 proof-of-work."""

    def __init__(self, config: BAB256Config = None):
        self.config = config or BAB256Config(difficulty_bits=8)
        self.engine = BAB256Engine(self.config)
        self.blocks: List[BAB256Block] = []

    def mine_block(self, data: str, verbose: bool = True) -> Optional[BAB256Block]:
        prev_hash = self.blocks[-1].block_hash if self.blocks else "0" * 64
        block_input = f"{prev_hash}|{data}|block_{len(self.blocks)}"

        proof = self.engine.mine(block_input, verbose=verbose)
        if proof is None:
            return None

        block_hash = hashlib.sha256(
            (prev_hash + proof.proof_hash).encode()
        ).hexdigest()

        block = BAB256Block(
            index=len(self.blocks),
            previous_hash=prev_hash,
            proof=proof,
            block_hash=block_hash,
        )
        self.blocks.append(block)
        return block

    def verify_chain(self, verbose: bool = True) -> bool:
        if verbose:
            print(f"\n{'='*60}")
            print(f"  VERIFYING BAB256 CHAIN ({len(self.blocks)} blocks)")
            print(f"{'='*60}")

        for i, block in enumerate(self.blocks):
            # Check previous hash linkage
            if i == 0:
                expected_prev = "0" * 64
            else:
                expected_prev = self.blocks[i - 1].block_hash
            if block.previous_hash != expected_prev:
                if verbose:
                    print(f"  Block {i}: ✗ Chain linkage broken")
                return False

            # Verify proof
            valid = self.engine.verify(block.proof, verbose=False)
            if not valid:
                if verbose:
                    print(f"  Block {i}: ✗ Invalid proof")
                return False

            # Verify block hash
            expected_block_hash = hashlib.sha256(
                (block.previous_hash + block.proof.proof_hash).encode()
            ).hexdigest()
            if block.block_hash != expected_block_hash:
                if verbose:
                    print(f"  Block {i}: ✗ Block hash mismatch")
                return False

            if verbose:
                print(f"  Block {i}: ✓ (nonce={block.proof.nonce}, "
                      f"time={block.proof.computation_time:.2f}s)")

        if verbose:
            total_time = sum(b.proof.computation_time for b in self.blocks)
            print(f"\n  *** CHAIN VALID *** ({total_time:.1f}s total mining)")

        return True


# =============================================================================
# SOLVER COMPARISON BENCHMARK
# =============================================================================

def benchmark_solvers(num_trials: int = 5):
    """Compare all four CVP solvers on identical problems."""
    print(f"\n{'='*65}")
    print(f"  CVP SOLVER COMPARISON — {num_trials} trials")
    print(f"{'='*65}\n")

    config = BAB256Config(difficulty_bits=4, num_rounds=32)
    renderer = BabelRenderer(config)
    lattice = LatticeEngine(config)

    results: Dict[str, List[dict]] = {s.value: [] for s in SolverType}
    seed = hashlib.sha256(b"solver_benchmark").digest()
    lattice.generate_basis(seed)

    for trial in range(num_trials):
        target_seed = hashlib.sha256(
            seed + trial.to_bytes(4, 'big')
        ).digest()
        target = renderer.render(target_seed)
        nonce_seed = hashlib.sha256(target_seed + b'nonce').digest()

        for solver_type in SolverType:
            t0 = time.time()
            coeffs, dist = SOLVER_DISPATCH[solver_type](
                lattice.basis, target, config, nonce_seed
            )
            elapsed = time.time() - t0
            results[solver_type.value].append({
                'distance': dist,
                'time': elapsed,
                'coeff_range': (int(coeffs.min()), int(coeffs.max())),
            })

    # Print results
    print(f"  {'Solver':<18} {'Avg Dist':>12} {'Avg Time':>10} "
          f"{'Best Dist':>12} {'Worst Dist':>12}")
    print(f"  {'-'*64}")

    for name, trials in results.items():
        dists = [t['distance'] for t in trials]
        times = [t['time'] for t in trials]
        print(f"  {name:<18} {np.mean(dists):>12.1f} {np.mean(times):>9.3f}s "
              f"{min(dists):>12.1f} {max(dists):>12.1f}")

    print()
    return results


# =============================================================================
# MAIN
# =============================================================================

if __name__ == "__main__":
    print("""
    ╔══════════════════════════════════════════════════════════╗
    ║          BAB256 — DIMENSIONAL ENGINE v0.2               ║
    ║     Lattice-Based Proof of Work in Image Space          ║
    ║                                                         ║
    ║  NEW: Babai's Nearest Plane CVP solver                  ║
    ║  NEW: Combined solver (Babai + greedy refinement)       ║
    ║  NEW: Multi-block chain simulation                      ║
    ║  NEW: Solver comparison benchmarks                      ║
    ╚══════════════════════════════════════════════════════════╝
    """)

    # 1. Solver comparison
    benchmark_solvers(num_trials=3)

    # 2. Mine + verify with best solver
    config = BAB256Config(difficulty_bits=8, solver=SolverType.COMBINED)
    print(config.describe())

    engine = BAB256Engine(config)
    proof = engine.mine("BAB256 v0.2 Genesis", verbose=True)
    if proof:
        engine.verify(proof)

    # 3. Mine a 3-block chain
    print(f"\n\n{'='*60}")
    print(f"  MINING 3-BLOCK CHAIN")
    print(f"{'='*60}")
    chain_config = BAB256Config(difficulty_bits=6, solver=SolverType.COMBINED)
    chain = BAB256Chain(chain_config)
    for i in range(3):
        block = chain.mine_block(f"Block {i} data", verbose=False)
        if block:
            print(f"  Block {i}: hash={block.block_hash[:24]}... "
                  f"time={block.proof.computation_time:.2f}s")

    chain.verify_chain()
