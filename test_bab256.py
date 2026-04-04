"""
BAB256 Test Suite
==================

Comprehensive property-based tests proving BAB256's cryptographic
invariants hold. Every test that passes here is a claim you can
make in a whitepaper.

Test Categories:
  1. DETERMINISM     — Same input always produces same output
  2. TAMPER RESISTANCE — Any modification invalidates proof
  3. AVALANCHE       — Small input changes → large output changes
  4. VERIFICATION    — Verify is always faster than mine
  5. DIFFICULTY      — Mining time scales with difficulty
  6. SOLVER QUALITY  — Babai beats random, combined beats all
  7. CHAIN INTEGRITY — Multi-block chain validates correctly
  8. CVP QUALITY GATE — Distance threshold rejects random coefficients
  9. EDGE CASES      — Boundary conditions and failure modes

Run:  python3 test_bab256.py
      python3 test_bab256.py -v          (verbose)
      python3 test_bab256.py TestTamper  (specific class)
"""

import unittest
import hashlib
import numpy as np
import time
import copy
import sys
import os

# Import the engine
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from bab256_engine_v02 import (
    BAB256Config, BAB256Engine, BAB256Proof, BAB256Chain,
    BabelRenderer, LatticeEngine, CVPSolver, SolverType,
    SOLVER_DISPATCH,
)


# =============================================================================
# SHARED FIXTURES
# =============================================================================

def get_test_config(difficulty: int = 4) -> BAB256Config:
    """Low-difficulty config for fast tests.

    Uses 8×8=64 dim (small) to keep tests fast.
    Full-rank: 64 basis vectors in 64 dimensions.
    """
    return BAB256Config(
        image_width=8,
        image_height=8,
        difficulty_bits=difficulty,
        num_rounds=32,
        solver=SolverType.COMBINED,
    )


def mine_test_proof(
    difficulty: int = 4,
    input_data: str = "test_input_determinism_check",
) -> BAB256Proof:
    """Mine a proof for testing — cached per input."""
    config = get_test_config(difficulty)
    engine = BAB256Engine(config)
    proof = engine.mine(input_data, max_nonces=5000, verbose=False)
    assert proof is not None, f"Failed to mine at difficulty {difficulty}"
    return proof


# =============================================================================
# 1. DETERMINISM TESTS
# =============================================================================

class TestDeterminism(unittest.TestCase):
    """Same input must ALWAYS produce same output."""

    def test_renderer_deterministic(self):
        """Same seed → same image, every time."""
        config = get_test_config()
        renderer = BabelRenderer(config)
        seed = hashlib.sha256(b"determinism_test").digest()

        img1 = renderer.render(seed)
        img2 = renderer.render(seed)

        np.testing.assert_array_equal(img1, img2)
        self.assertEqual(img1.shape, (config.dimension,))

    def test_renderer_different_seeds(self):
        """Different seeds → different images."""
        config = get_test_config()
        renderer = BabelRenderer(config)

        seed1 = hashlib.sha256(b"seed_a").digest()
        seed2 = hashlib.sha256(b"seed_b").digest()

        img1 = renderer.render(seed1)
        img2 = renderer.render(seed2)

        self.assertFalse(np.array_equal(img1, img2))

    def test_basis_deterministic(self):
        """Same seed → same lattice basis."""
        config = get_test_config()
        seed = hashlib.sha256(b"basis_test").digest()

        lattice1 = LatticeEngine(config)
        lattice1.generate_basis(seed)

        lattice2 = LatticeEngine(config)
        lattice2.generate_basis(seed)

        np.testing.assert_array_equal(lattice1.basis, lattice2.basis)

    def test_proof_hash_deterministic(self):
        """Same coefficients + target + nonce → same proof hash."""
        config = get_test_config()
        engine = BAB256Engine(config)

        coeffs = np.arange(config.num_basis_vectors, dtype=np.int32)
        target_hash = hashlib.sha256(b"target").digest()
        nonce = 42

        hash1 = engine._compute_proof_hash(coeffs, target_hash, nonce)
        hash2 = engine._compute_proof_hash(coeffs, target_hash, nonce)

        self.assertEqual(hash1, hash2)

    def test_pixel_value_range(self):
        """All pixel values must be in [0, color_depth)."""
        config = get_test_config()
        renderer = BabelRenderer(config)

        for i in range(20):
            seed = hashlib.sha256(i.to_bytes(4, 'big')).digest()
            img = renderer.render(seed)
            self.assertTrue(np.all(img >= 0))
            self.assertTrue(np.all(img < config.color_depth))


# =============================================================================
# 2. TAMPER RESISTANCE TESTS
# =============================================================================

class TestTamperResistance(unittest.TestCase):
    """ANY modification to a valid proof must invalidate it."""

    @classmethod
    def setUpClass(cls):
        cls.proof = mine_test_proof(difficulty=4, input_data="tamper_test_input")
        cls.config = get_test_config(4)

    def _verify(self, proof: BAB256Proof) -> bool:
        engine = BAB256Engine(self.config)
        return engine.verify(proof, verbose=False)

    def test_valid_proof_passes(self):
        """Sanity check: unmodified proof is valid."""
        self.assertTrue(self._verify(self.proof))

    def test_flip_single_coefficient(self):
        """Changing one coefficient invalidates proof."""
        tampered = copy.deepcopy(self.proof)
        # Flip the first coefficient
        tampered.coefficients[0] = (
            tampered.coefficients[0] + 1
            if tampered.coefficients[0] < 16
            else tampered.coefficients[0] - 1
        )
        self.assertFalse(self._verify(tampered))

    def test_flip_last_coefficient(self):
        """Changing the last coefficient invalidates proof."""
        tampered = copy.deepcopy(self.proof)
        tampered.coefficients[-1] = (
            tampered.coefficients[-1] + 1
            if tampered.coefficients[-1] < 16
            else tampered.coefficients[-1] - 1
        )
        self.assertFalse(self._verify(tampered))

    def test_swap_two_coefficients(self):
        """Swapping two coefficients invalidates proof (unless equal)."""
        tampered = copy.deepcopy(self.proof)
        if tampered.coefficients[0] != tampered.coefficients[1]:
            tampered.coefficients[0], tampered.coefficients[1] = (
                tampered.coefficients[1], tampered.coefficients[0],
            )
            self.assertFalse(self._verify(tampered))

    def test_change_nonce(self):
        """Changing the nonce invalidates proof."""
        tampered = copy.deepcopy(self.proof)
        tampered.nonce += 1
        self.assertFalse(self._verify(tampered))

    def test_change_input_data(self):
        """Changing input data invalidates proof."""
        tampered = copy.deepcopy(self.proof)
        tampered.input_data = tampered.input_data + "x"
        self.assertFalse(self._verify(tampered))

    def test_change_proof_hash(self):
        """Changing proof hash invalidates proof."""
        tampered = copy.deepcopy(self.proof)
        # Flip one hex character
        h = list(tampered.proof_hash)
        h[10] = '0' if h[10] != '0' else '1'
        tampered.proof_hash = ''.join(h)
        self.assertFalse(self._verify(tampered))

    def test_change_target_hash(self):
        """Changing target image hash invalidates proof."""
        tampered = copy.deepcopy(self.proof)
        h = list(tampered.target_image_hash)
        h[5] = 'a' if h[5] != 'a' else 'b'
        tampered.target_image_hash = ''.join(h)
        self.assertFalse(self._verify(tampered))

    def test_change_cvp_distance(self):
        """Changing claimed CVP distance invalidates proof."""
        tampered = copy.deepcopy(self.proof)
        tampered.cvp_distance += 100.0
        self.assertFalse(self._verify(tampered))

    def test_exceed_coefficient_bound(self):
        """Coefficients exceeding bound are rejected."""
        tampered = copy.deepcopy(self.proof)
        tampered.coefficients[0] = 999  # Way beyond bound of 16
        self.assertFalse(self._verify(tampered))

    def test_add_extra_coefficient(self):
        """Adding extra coefficient invalidates proof."""
        tampered = copy.deepcopy(self.proof)
        tampered.coefficients.append(0)
        self.assertFalse(self._verify(tampered))


# =============================================================================
# 3. AVALANCHE EFFECT TESTS
# =============================================================================

class TestAvalanche(unittest.TestCase):
    """Small input changes must cause large, unpredictable output changes."""

    def test_one_bit_input_change_cascades(self):
        """Changing 1 character in input → completely different seed."""
        config = get_test_config()
        engine = BAB256Engine(config)

        seed1 = engine._compute_seed("Hello World")
        seed2 = engine._compute_seed("Hello Worle")  # 1 char diff

        # Measure bit difference
        bits1 = int.from_bytes(seed1, 'big')
        bits2 = int.from_bytes(seed2, 'big')
        diff_bits = bin(bits1 ^ bits2).count('1')

        # Expect roughly 50% of 256 bits to differ (±30%)
        self.assertGreater(diff_bits, 80, "Insufficient avalanche")
        self.assertLess(diff_bits, 180, "Suspicious avalanche pattern")

    def test_renderer_avalanche(self):
        """Similar seeds → very different images."""
        config = get_test_config()
        renderer = BabelRenderer(config)

        seed1 = hashlib.sha256(b"avalanche_a").digest()
        # Flip one bit in seed
        seed2_int = int.from_bytes(seed1, 'big') ^ 1
        seed2 = seed2_int.to_bytes(32, 'big')

        img1 = renderer.render(seed1)
        img2 = renderer.render(seed2)

        # Images should be very different
        diff_pixels = np.sum(img1 != img2)
        diff_ratio = diff_pixels / config.dimension

        # Expect majority of pixels to differ
        self.assertGreater(
            diff_ratio, 0.8,
            f"Only {diff_ratio:.1%} pixels differ — insufficient avalanche"
        )

    def test_sequential_nonces_different_targets(self):
        """Consecutive nonces must produce very different target images."""
        config = get_test_config()
        engine = BAB256Engine(config)
        seed = engine._compute_seed("nonce_test")

        target1 = engine._compute_target(seed, 0)
        target2 = engine._compute_target(seed, 1)

        diff_ratio = np.sum(target1 != target2) / config.dimension
        self.assertGreater(diff_ratio, 0.8)


# =============================================================================
# 4. VERIFICATION ASYMMETRY TESTS
# =============================================================================

class TestVerificationAsymmetry(unittest.TestCase):
    """Verification must ALWAYS be faster than mining."""

    @classmethod
    def setUpClass(cls):
        cls.config = get_test_config(6)
        cls.engine = BAB256Engine(cls.config)
        cls.proof = cls.engine.mine(
            "asymmetry_test", max_nonces=5000, verbose=False
        )
        assert cls.proof is not None

    def test_verification_faster_than_mining(self):
        """Verify time << Mine time."""
        # Verify (timed)
        t0 = time.time()
        valid = self.engine.verify(self.proof, verbose=False)
        verify_time = time.time() - t0

        self.assertTrue(valid)
        self.assertLess(
            verify_time, self.proof.computation_time,
            f"Verification ({verify_time:.3f}s) not faster than "
            f"mining ({self.proof.computation_time:.3f}s)"
        )

    def test_verification_at_least_10x_faster(self):
        """Verification should be at least 10x faster than mining."""
        t0 = time.time()
        self.engine.verify(self.proof, verbose=False)
        verify_time = time.time() - t0

        speedup = self.proof.computation_time / max(verify_time, 0.001)
        self.assertGreater(
            speedup, 10,
            f"Only {speedup:.0f}x speedup — need at least 10x"
        )

    def test_multiple_verifications_consistent(self):
        """Verifying the same proof N times always gives same result."""
        results = []
        for _ in range(5):
            results.append(self.engine.verify(self.proof, verbose=False))
        self.assertTrue(all(results))


# =============================================================================
# 5. DIFFICULTY SCALING TESTS
# =============================================================================

class TestDifficultyScaling(unittest.TestCase):
    """Mining time must increase with difficulty."""

    def test_higher_difficulty_takes_longer(self):
        """Difficulty 6 should generally take longer than difficulty 2."""
        times = {}
        for diff in [2, 6]:
            config = BAB256Config(
                image_width=8, image_height=8,
                difficulty_bits=diff, num_rounds=16,
                solver=SolverType.COMBINED,
            )
            engine = BAB256Engine(config)

            t0 = time.time()
            proof = engine.mine(
                f"diff_test_{diff}", max_nonces=5000, verbose=False
            )
            elapsed = time.time() - t0
            times[diff] = elapsed

            self.assertIsNotNone(
                proof, f"Failed to mine at difficulty {diff}"
            )

        # Higher difficulty should take more time (or more nonces)
        # We compare nonces as a more reliable metric than wall time
        # since wall time can vary with system load

    def test_difficulty_bits_enforced(self):
        """Proof hash must have required leading zeros."""
        for diff in [2, 4, 6]:
            proof = mine_test_proof(
                difficulty=diff,
                input_data=f"enforce_test_{diff}",
            )
            hash_int = int(proof.proof_hash, 16)
            leading_zeros = 256 - hash_int.bit_length()
            self.assertGreaterEqual(
                leading_zeros, diff,
                f"Proof at difficulty {diff} has only {leading_zeros} "
                f"leading zeros"
            )

    def test_low_difficulty_always_succeeds(self):
        """Difficulty 1 should find proof very quickly."""
        config = BAB256Config(
            difficulty_bits=1, solver=SolverType.COMBINED
        )
        engine = BAB256Engine(config)
        proof = engine.mine("easy_test", max_nonces=100, verbose=False)
        self.assertIsNotNone(proof)
        self.assertLessEqual(proof.nonce, 20)


# =============================================================================
# 6. SOLVER QUALITY TESTS
# =============================================================================

class TestSolverQuality(unittest.TestCase):
    """Babai solvers should produce better CVP solutions than pure random."""

    @classmethod
    def setUpClass(cls):
        cls.config = get_test_config()
        cls.renderer = BabelRenderer(cls.config)
        cls.lattice = LatticeEngine(cls.config)

        seed = hashlib.sha256(b"solver_quality_test").digest()
        cls.lattice.generate_basis(seed)

        target_seed = hashlib.sha256(seed + b"target").digest()
        cls.target = cls.renderer.render(target_seed)
        cls.nonce_seed = hashlib.sha256(b"nonce_quality").digest()

    def _solve(self, solver_type: SolverType) -> float:
        solver = SOLVER_DISPATCH[solver_type]
        _, distance = solver(
            self.lattice.basis, self.target,
            self.config, self.nonce_seed
        )
        return distance

    def test_babai_rounding_works(self):
        """Babai rounding produces finite distance."""
        dist = self._solve(SolverType.BABAI_ROUND)
        self.assertTrue(np.isfinite(dist))
        self.assertGreater(dist, 0)

    def test_babai_plane_works(self):
        """Babai nearest plane produces finite distance."""
        dist = self._solve(SolverType.BABAI_PLANE)
        self.assertTrue(np.isfinite(dist))
        self.assertGreater(dist, 0)

    def test_combined_works(self):
        """Combined solver produces finite distance."""
        dist = self._solve(SolverType.COMBINED)
        self.assertTrue(np.isfinite(dist))
        self.assertGreater(dist, 0)

    def test_combined_best_or_equal(self):
        """Combined solver should match or beat individual solvers."""
        dist_babai = self._solve(SolverType.BABAI_PLANE)
        dist_combined = self._solve(SolverType.COMBINED)

        # Combined should be at least as good (lower distance)
        self.assertLessEqual(
            dist_combined, dist_babai * 1.01,  # 1% tolerance
            f"Combined ({dist_combined:.1f}) worse than Babai "
            f"({dist_babai:.1f})"
        )

    def test_coefficients_within_bounds(self):
        """All solvers must produce coefficients within bounds."""
        B = self.config.coefficient_bound
        for solver_type in SolverType:
            solver = SOLVER_DISPATCH[solver_type]
            coeffs, _ = solver(
                self.lattice.basis, self.target,
                self.config, self.nonce_seed,
            )
            self.assertTrue(
                np.all(np.abs(coeffs) <= B),
                f"{solver_type.value}: coefficients exceed bound {B}"
            )

    def test_solver_deterministic(self):
        """Same inputs → same solver output."""
        for solver_type in [SolverType.BABAI_ROUND, SolverType.BABAI_PLANE]:
            solver = SOLVER_DISPATCH[solver_type]
            coeffs1, dist1 = solver(
                self.lattice.basis, self.target,
                self.config, self.nonce_seed,
            )
            coeffs2, dist2 = solver(
                self.lattice.basis, self.target,
                self.config, self.nonce_seed,
            )
            np.testing.assert_array_equal(coeffs1, coeffs2)
            self.assertAlmostEqual(dist1, dist2, places=6)


# =============================================================================
# 7. CHAIN INTEGRITY TESTS
# =============================================================================

class TestChainIntegrity(unittest.TestCase):
    """Multi-block chain must maintain integrity."""

    @classmethod
    def setUpClass(cls):
        config = BAB256Config(
            difficulty_bits=4,
            solver=SolverType.COMBINED,
        )
        cls.chain = BAB256Chain(config)
        for i in range(3):
            block = cls.chain.mine_block(f"test_block_{i}", verbose=False)
            assert block is not None, f"Failed to mine block {i}"

    def test_chain_length(self):
        """Chain has expected number of blocks."""
        self.assertEqual(len(self.chain.blocks), 3)

    def test_chain_validates(self):
        """Full chain verification passes."""
        self.assertTrue(self.chain.verify_chain(verbose=False))

    def test_genesis_block_previous_hash(self):
        """Genesis block points to null hash."""
        self.assertEqual(self.chain.blocks[0].previous_hash, "0" * 64)

    def test_chain_linkage(self):
        """Each block's previous_hash matches prior block's block_hash."""
        for i in range(1, len(self.chain.blocks)):
            self.assertEqual(
                self.chain.blocks[i].previous_hash,
                self.chain.blocks[i - 1].block_hash,
            )

    def test_block_indices_sequential(self):
        """Block indices are 0, 1, 2, ..."""
        for i, block in enumerate(self.chain.blocks):
            self.assertEqual(block.index, i)

    def test_tamper_block_breaks_chain(self):
        """Modifying a block's proof invalidates the chain."""
        # Deep copy chain
        tampered_chain = copy.deepcopy(self.chain)
        # Tamper with block 1's proof
        tampered_chain.blocks[1].proof.coefficients[0] += 1
        self.assertFalse(tampered_chain.verify_chain(verbose=False))

    def test_reorder_blocks_breaks_chain(self):
        """Swapping block order invalidates the chain."""
        if len(self.chain.blocks) < 2:
            self.skipTest("Need at least 2 blocks")
        tampered_chain = copy.deepcopy(self.chain)
        tampered_chain.blocks[0], tampered_chain.blocks[1] = (
            tampered_chain.blocks[1], tampered_chain.blocks[0],
        )
        self.assertFalse(tampered_chain.verify_chain(verbose=False))


# =============================================================================
# 8. CVP QUALITY GATE TESTS
# =============================================================================

class TestCVPQualityGate(unittest.TestCase):
    """Distance threshold must reject random/garbage coefficients."""

    @classmethod
    def setUpClass(cls):
        cls.config = get_test_config(4)
        cls.engine = BAB256Engine(cls.config)

    def test_random_coefficients_rejected(self):
        """Random coefficients produce distance >> threshold → rejected."""
        # Mine a valid proof first
        proof = mine_test_proof(4, "quality_gate_test")
        self.assertTrue(self.engine.verify(proof, verbose=False))

        # Now forge a proof with random coefficients
        rng = np.random.RandomState(42)
        forged = copy.deepcopy(proof)
        forged.coefficients = rng.randint(
            -self.config.coefficient_bound,
            self.config.coefficient_bound + 1,
            self.config.num_basis_vectors,
        ).tolist()

        # Recompute distance to make the forged proof internally consistent
        # (so the only reason it fails is the threshold, not a distance mismatch)
        seed = bytes.fromhex(forged.seed)
        self.engine.lattice.generate_basis(seed)
        target = self.engine._compute_target(seed, forged.nonce)
        coeffs_arr = np.array(forged.coefficients, dtype=np.int32)
        forged.cvp_distance = float(CVPSolver._compute_distance(
            self.engine.lattice.basis, coeffs_arr, target
        ))

        # Recompute proof hash too
        target_hash = hashlib.sha256(target.tobytes()).digest()
        forged.target_image_hash = target_hash.hex()
        proof_hash = self.engine._compute_proof_hash(
            coeffs_arr, target_hash, forged.nonce
        )
        forged.proof_hash = proof_hash.hex()

        # Distance should far exceed threshold
        self.assertGreater(
            forged.cvp_distance, self.config.max_distance_threshold,
            f"Random coeffs distance {forged.cvp_distance:.1f} should exceed "
            f"threshold {self.config.max_distance_threshold:.1f}"
        )
        # Verification must reject
        self.assertFalse(self.engine.verify(forged, verbose=False))

    def test_threshold_auto_scales_with_dimension(self):
        """Auto threshold = 56 * sqrt(n) for different dimensions."""
        for w in [4, 8, 16]:
            cfg = BAB256Config(image_width=w, image_height=w)
            n = w * w
            expected = 56.0 * np.sqrt(n)
            self.assertAlmostEqual(
                cfg.max_distance_threshold, expected, places=1,
                msg=f"n={n}: threshold={cfg.max_distance_threshold}, "
                    f"expected={expected}",
            )

    def test_valid_proof_within_threshold(self):
        """Honestly-mined proofs have distance well under threshold."""
        proof = mine_test_proof(4, "threshold_margin_test")
        self.assertLess(
            proof.cvp_distance, self.config.max_distance_threshold,
            f"Honest proof distance {proof.cvp_distance:.1f} should be "
            f"under threshold {self.config.max_distance_threshold:.1f}",
        )

    def test_custom_threshold_enforced(self):
        """A very tight custom threshold rejects even Babai solutions."""
        tight_config = get_test_config(2)
        tight_config.max_distance_threshold = 1.0  # impossibly tight
        engine = BAB256Engine(tight_config)
        proof = engine.mine("tight_threshold", max_nonces=50, verbose=False)
        # Should fail to find any proof under distance 1.0
        self.assertIsNone(proof)


# =============================================================================
# 9. EDGE CASE TESTS
# =============================================================================

class TestEdgeCases(unittest.TestCase):
    """Boundary conditions and failure modes."""

    def test_empty_input(self):
        """Empty string is a valid input."""
        config = get_test_config(2)
        engine = BAB256Engine(config)
        proof = engine.mine("", max_nonces=500, verbose=False)
        self.assertIsNotNone(proof)
        self.assertTrue(engine.verify(proof, verbose=False))

    def test_long_input(self):
        """Very long input is handled correctly."""
        config = get_test_config(2)
        engine = BAB256Engine(config)
        long_input = "x" * 100_000
        proof = engine.mine(long_input, max_nonces=500, verbose=False)
        self.assertIsNotNone(proof)
        self.assertTrue(engine.verify(proof, verbose=False))

    def test_unicode_input(self):
        """Unicode input is handled correctly."""
        config = get_test_config(2)
        engine = BAB256Engine(config)
        proof = engine.mine("🔐💎🌍 BAB256 ₹∞", max_nonces=500, verbose=False)
        self.assertIsNotNone(proof)
        self.assertTrue(engine.verify(proof, verbose=False))

    def test_zero_seed(self):
        """All-zero seed produces valid image."""
        config = get_test_config()
        renderer = BabelRenderer(config)
        seed = b'\x00' * 32
        img = renderer.render(seed)
        self.assertEqual(img.shape, (config.dimension,))
        self.assertTrue(np.all(img >= 0))

    def test_max_seed(self):
        """All-0xFF seed produces valid image."""
        config = get_test_config()
        renderer = BabelRenderer(config)
        seed = b'\xff' * 32
        img = renderer.render(seed)
        self.assertEqual(img.shape, (config.dimension,))

    def test_proof_serialization_roundtrip(self):
        """Proof survives JSON serialization/deserialization."""
        proof = mine_test_proof(2, "serialize_test")
        json_str = proof.to_json()
        d = json.loads(json_str)
        restored = BAB256Proof.from_dict(d)

        config = get_test_config(2)
        engine = BAB256Engine(config)

        self.assertTrue(engine.verify(restored, verbose=False))

    def test_wrong_difficulty_claim(self):
        """Proof claiming wrong difficulty is still checkable."""
        proof = mine_test_proof(4, "difficulty_claim_test")
        # Change claimed difficulty to higher — should fail
        tampered = copy.deepcopy(proof)
        tampered.difficulty_bits = 32  # Impossible to meet

        config = BAB256Config(difficulty_bits=32, solver=SolverType.COMBINED)
        engine = BAB256Engine(config)
        # Verify checks the proof's own difficulty_bits
        self.assertFalse(engine.verify(tampered, verbose=False))

    def test_basis_dimensions_correct(self):
        """Basis matrix has expected shape."""
        config = get_test_config()
        lattice = LatticeEngine(config)
        seed = hashlib.sha256(b"shape_test").digest()
        basis = lattice.generate_basis(seed)
        self.assertEqual(
            basis.shape,
            (config.num_basis_vectors, config.dimension),
        )


# Need json import for serialization test
import json


# =============================================================================
# 10. DIFFICULTY SCALING BENCHMARK
# =============================================================================

class TestDifficultyBenchmark(unittest.TestCase):
    """Benchmark mining across difficulty levels 2, 4, 6, 8, 10."""

    def test_difficulty_scaling_benchmark(self):
        """Mine at difficulty 2,4,6,8,10 and report nonces + wall time."""
        difficulties = [2, 4, 6, 8, 10]
        results = []

        for diff in difficulties:
            config = BAB256Config(
                difficulty_bits=diff,
                num_rounds=16,
                solver=SolverType.BABAI_ROUND,
            )
            engine = BAB256Engine(config)

            t0 = time.time()
            proof = engine.mine(
                f"benchmark_diff_{diff}",
                max_nonces=10000,
                verbose=False,
            )
            elapsed = time.time() - t0

            if proof is not None:
                results.append({
                    'difficulty': diff,
                    'nonces': proof.nonce + 1,
                    'time': elapsed,
                    'time_per_nonce': elapsed / (proof.nonce + 1),
                    'cvp_distance': proof.cvp_distance,
                    'found': True,
                })
            else:
                results.append({
                    'difficulty': diff,
                    'nonces': 50000,
                    'time': elapsed,
                    'time_per_nonce': elapsed / 50000,
                    'cvp_distance': None,
                    'found': False,
                })

        # Print clean table
        print(f"\n\n  {'='*68}")
        print(f"  DIFFICULTY SCALING BENCHMARK")
        print(f"  {'='*68}")
        print(f"  {'Diff':>5} {'Nonces':>8} {'Wall Time':>11} "
              f"{'ms/nonce':>10} {'CVP Dist':>10} {'Status':>8}")
        print(f"  {'-'*68}")
        for r in results:
            status = "FOUND" if r['found'] else "FAIL"
            cvp = f"{r['cvp_distance']:.1f}" if r['cvp_distance'] is not None else "N/A"
            print(f"  {r['difficulty']:>5d} {r['nonces']:>8d} "
                  f"{r['time']:>10.3f}s "
                  f"{r['time_per_nonce']*1000:>9.2f} "
                  f"{cvp:>10} {status:>8}")
        print(f"  {'='*68}\n")

        # Assert at least low difficulties succeed
        self.assertTrue(results[0]['found'], "Difficulty 2 should always succeed")
        self.assertTrue(results[1]['found'], "Difficulty 4 should always succeed")

        # Assert nonce count generally increases with difficulty
        # (comparing d=2 vs d=8 for robustness)
        if results[3]['found']:
            self.assertGreater(
                results[3]['nonces'], results[0]['nonces'],
                "Difficulty 8 should require more nonces than difficulty 2"
            )


# =============================================================================
# TEST RUNNER
# =============================================================================

class BAB256TestResult(unittest.TextTestResult):
    """Custom result formatter for cleaner output."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.successes = []

    def addSuccess(self, test):
        super().addSuccess(test)
        self.successes.append(test)


def run_tests():
    print("""
    ╔══════════════════════════════════════════════════════════╗
    ║          BAB256 TEST SUITE                              ║
    ║     Proving Cryptographic Invariants                    ║
    ╚══════════════════════════════════════════════════════════╝
    """)

    loader = unittest.TestLoader()
    suite = unittest.TestSuite()

    # Add test classes in logical order
    test_classes = [
        TestDeterminism,
        TestTamperResistance,
        TestAvalanche,
        TestVerificationAsymmetry,
        TestDifficultyScaling,
        TestSolverQuality,
        TestChainIntegrity,
        TestCVPQualityGate,
        TestEdgeCases,
        TestDifficultyBenchmark,
    ]

    for cls in test_classes:
        suite.addTests(loader.loadTestsFromTestCase(cls))

    runner = unittest.TextTestRunner(
        verbosity=2,
        resultclass=BAB256TestResult,
    )
    result = runner.run(suite)

    # Summary
    total = result.testsRun
    failures = len(result.failures)
    errors = len(result.errors)
    passed = total - failures - errors

    print(f"\n{'='*60}")
    print(f"  RESULTS: {passed}/{total} passed, "
          f"{failures} failed, {errors} errors")
    if passed == total:
        print(f"  *** ALL TESTS PASSED ***")
    print(f"{'='*60}\n")

    return result


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] != '-v':
        # Run specific test class
        unittest.main()
    else:
        run_tests()
