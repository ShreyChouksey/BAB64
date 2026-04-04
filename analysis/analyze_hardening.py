"""
BAB256 v0.3 Hardening: Parameter exploration
=============================================
Test different (num_basis_vectors, coefficient_bound) combos to find
where CVP distance becomes meaningfully non-zero.
"""

import hashlib
import numpy as np
import sys
import os
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from bab256_engine_v02 import (
    BAB256Config, BabelRenderer, LatticeEngine, CVPSolver, SolverType,
)


def test_params(num_basis, coeff_bound, num_trials=5):
    """Test a parameter combo and return clamping stats + distances."""
    config = BAB256Config(
        num_basis_vectors=num_basis,
        coefficient_bound=coeff_bound,
    )
    renderer = BabelRenderer(config)
    lattice = LatticeEngine(config)

    seed = hashlib.sha256(b"hardening_analysis").digest()
    lattice.generate_basis(seed)

    distances_round = []
    distances_plane = []
    clamped_round = []
    clamped_plane = []

    for trial in range(num_trials):
        target_seed = hashlib.sha256(seed + trial.to_bytes(4, 'big')).digest()
        target = renderer.render(target_seed)
        nonce_seed = hashlib.sha256(target_seed + b'nonce').digest()

        # Babai Rounding — get unclamped coefficients
        B = lattice.basis.astype(np.float64)
        t = target.astype(np.float64)
        real_coeffs, _, _, _ = np.linalg.lstsq(B.T, t, rcond=None)
        int_coeffs = np.round(real_coeffs).astype(np.int32)
        exceeds = int(np.sum(np.abs(int_coeffs) > coeff_bound))
        clamped_round.append(exceeds)

        coeffs_r, dist_r = CVPSolver.babai_rounding(
            lattice.basis, target, config, nonce_seed)
        distances_round.append(dist_r)

        coeffs_p, dist_p = CVPSolver.babai_nearest_plane(
            lattice.basis, target, config, nonce_seed)
        distances_plane.append(dist_p)
        clamped_plane.append(int(np.sum(np.abs(coeffs_p) == coeff_bound)))

    return {
        'num_basis': num_basis,
        'coeff_bound': coeff_bound,
        'avg_dist_round': np.mean(distances_round),
        'avg_dist_plane': np.mean(distances_plane),
        'avg_clamped_round': np.mean(clamped_round),
        'avg_clamped_plane': np.mean(clamped_plane),
        'pct_clamped_round': np.mean(clamped_round) / num_basis * 100,
    }


if __name__ == "__main__":
    print(f"{'='*80}")
    print(f"  PARAMETER HARDENING EXPLORATION")
    print(f"  Testing (num_basis_vectors, coefficient_bound) combinations")
    print(f"{'='*80}\n")

    configs = [
        # Current params (baseline)
        (128, 16),
        # Reduce coefficient bound
        (128, 8),
        (128, 4),
        (128, 2),
        (128, 1),
        # Reduce basis vectors
        (64, 16),
        (48, 16),
        (32, 16),
        (16, 16),
        # Combined reductions
        (64, 4),
        (48, 3),
        (32, 3),
        (32, 2),
        (16, 8),
        (16, 4),
    ]

    print(f"  {'Basis':>6} {'Bound':>6} {'Dist(Round)':>12} {'Dist(Plane)':>12} "
          f"{'Clamped%':>9} {'Assessment':<20}")
    print(f"  {'-'*75}")

    for num_basis, coeff_bound in configs:
        r = test_params(num_basis, coeff_bound, num_trials=5)

        if r['avg_dist_round'] < 1.0:
            assessment = "TOO EASY"
        elif r['avg_dist_round'] < 1000:
            assessment = "MODERATE"
        elif r['avg_dist_round'] < 5000:
            assessment = "GOOD"
        else:
            assessment = "HARD"

        marker = " ***" if 10 < r['pct_clamped_round'] < 50 else ""
        print(f"  {r['num_basis']:>6d} {r['coeff_bound']:>6d} "
              f"{r['avg_dist_round']:>12.1f} {r['avg_dist_plane']:>12.1f} "
              f"{r['pct_clamped_round']:>8.1f}% {assessment:<20}{marker}")

    # Now test mining feasibility for promising candidates
    print(f"\n\n{'='*80}")
    print(f"  MINING FEASIBILITY TEST (difficulty=4)")
    print(f"{'='*80}\n")

    promising = [(48, 3), (32, 3), (64, 4), (32, 2), (16, 8)]
    print(f"  {'Basis':>6} {'Bound':>6} {'Found':>6} {'Nonces':>8} "
          f"{'Time':>8} {'CVP Dist':>10}")
    print(f"  {'-'*55}")

    for num_basis, coeff_bound in promising:
        config = BAB256Config(
            num_basis_vectors=num_basis,
            coefficient_bound=coeff_bound,
            difficulty_bits=4,
            solver=SolverType.BABAI_ROUND,
            num_rounds=16,
        )
        from bab256_engine_v02 import BAB256Engine
        engine = BAB256Engine(config)
        t0 = time.time()
        proof = engine.mine("hardening_test", max_nonces=2000, verbose=False)
        elapsed = time.time() - t0

        if proof:
            print(f"  {num_basis:>6d} {coeff_bound:>6d} {'YES':>6} "
                  f"{proof.nonce+1:>8d} {elapsed:>7.2f}s "
                  f"{proof.cvp_distance:>10.1f}")
        else:
            print(f"  {num_basis:>6d} {coeff_bound:>6d} {'NO':>6} "
                  f"{'2000+':>8} {elapsed:>7.2f}s {'N/A':>10}")

    print()
