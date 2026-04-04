"""
BAB256 v0.3 Investigation: Babai Coefficient Clamping Analysis
================================================================
Question: How many of the 128 Babai coefficients get clamped by the
bound of 16? If very few get clamped, the lattice is too easy.
"""

import hashlib
import numpy as np
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from bab256_engine_v02 import (
    BAB256Config, BabelRenderer, LatticeEngine, CVPSolver, SolverType,
)


def analyze_clamping(num_trials=20):
    config = BAB256Config()
    renderer = BabelRenderer(config)
    lattice = LatticeEngine(config)

    seed = hashlib.sha256(b"clamping_analysis").digest()
    lattice.generate_basis(seed)

    print(f"{'='*70}")
    print(f"  BABAI COEFFICIENT CLAMPING ANALYSIS")
    print(f"  Config: {config.num_basis_vectors} basis vectors, "
          f"coefficient_bound={config.coefficient_bound}")
    print(f"  Trials: {num_trials}")
    print(f"{'='*70}\n")

    # --- Babai Rounding Analysis ---
    print("  BABAI ROUNDING — unclamped vs clamped coefficients")
    print(f"  {'-'*60}")

    round_clamped_counts = []
    round_unclamped_maxes = []
    round_distances_unclamped = []
    round_distances_clamped = []

    for trial in range(num_trials):
        target_seed = hashlib.sha256(seed + trial.to_bytes(4, 'big')).digest()
        target = renderer.render(target_seed)

        # Get unclamped coefficients via lstsq
        B = lattice.basis.astype(np.float64)
        t = target.astype(np.float64)
        real_coeffs, _, _, _ = np.linalg.lstsq(B.T, t, rcond=None)
        int_coeffs_unclamped = np.round(real_coeffs).astype(np.int32)

        # Count how many exceed the bound
        exceeds = np.sum(np.abs(int_coeffs_unclamped) > config.coefficient_bound)
        max_abs = int(np.max(np.abs(int_coeffs_unclamped)))
        round_clamped_counts.append(int(exceeds))
        round_unclamped_maxes.append(max_abs)

        # Distance with and without clamping
        int_coeffs_clamped = np.clip(int_coeffs_unclamped,
                                      -config.coefficient_bound,
                                      config.coefficient_bound).astype(np.int32)

        dist_unclamped = CVPSolver._compute_distance(
            lattice.basis, int_coeffs_unclamped, target)
        dist_clamped = CVPSolver._compute_distance(
            lattice.basis, int_coeffs_clamped, target)

        round_distances_unclamped.append(dist_unclamped)
        round_distances_clamped.append(dist_clamped)

        if trial < 5:
            print(f"  Trial {trial:>2d}: clamped={exceeds:>3d}/128, "
                  f"max|c|={max_abs:>3d}, "
                  f"dist_unclamped={dist_unclamped:>10.1f}, "
                  f"dist_clamped={dist_clamped:>10.1f}")

    print(f"\n  Summary (Babai Rounding, {num_trials} trials):")
    print(f"    Avg clamped coefficients: "
          f"{np.mean(round_clamped_counts):.1f} / 128 "
          f"({np.mean(round_clamped_counts)/128*100:.1f}%)")
    print(f"    Max |coefficient| before clamping: "
          f"{max(round_unclamped_maxes)}")
    print(f"    Avg |coefficient| max before clamping: "
          f"{np.mean(round_unclamped_maxes):.1f}")
    print(f"    Avg distance (unclamped): {np.mean(round_distances_unclamped):.1f}")
    print(f"    Avg distance (clamped):   {np.mean(round_distances_clamped):.1f}")

    # --- Babai Nearest Plane Analysis ---
    print(f"\n\n  BABAI NEAREST PLANE — unclamped vs clamped coefficients")
    print(f"  {'-'*60}")

    # We need to instrument the nearest plane solver to see pre-clamp values
    # Re-implement with tracking
    k = lattice.basis.shape[0]
    B = lattice.basis.astype(np.float64)

    # Gram-Schmidt (compute once)
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

    plane_clamped_counts = []
    plane_unclamped_maxes = []
    plane_distances = []

    for trial in range(num_trials):
        target_seed = hashlib.sha256(seed + trial.to_bytes(4, 'big')).digest()
        target = renderer.render(target_seed)
        t = target.astype(np.float64)

        # Nearest plane with tracking
        b = t.copy()
        coeffs_raw = np.zeros(k, dtype=np.float64)
        coeffs_clamped = np.zeros(k, dtype=np.int32)

        for i in range(k - 1, -1, -1):
            norm_sq = np.dot(B_star[i], B_star[i])
            if norm_sq < 1e-10:
                continue
            c_real = np.dot(b, B_star[i]) / norm_sq
            c_int = int(np.round(c_real))
            coeffs_raw[i] = c_int

            c_clamped = max(-config.coefficient_bound,
                           min(config.coefficient_bound, c_int))
            coeffs_clamped[i] = c_clamped
            b -= c_clamped * B[i]

        exceeds = int(np.sum(np.abs(coeffs_raw) > config.coefficient_bound))
        max_abs = int(np.max(np.abs(coeffs_raw)))
        plane_clamped_counts.append(exceeds)
        plane_unclamped_maxes.append(max_abs)

        dist = CVPSolver._compute_distance(lattice.basis, coeffs_clamped, target)
        plane_distances.append(dist)

        if trial < 5:
            print(f"  Trial {trial:>2d}: clamped={exceeds:>3d}/128, "
                  f"max|c|={max_abs:>3d}, dist={dist:>10.1f}")

    print(f"\n  Summary (Babai Nearest Plane, {num_trials} trials):")
    print(f"    Avg clamped coefficients: "
          f"{np.mean(plane_clamped_counts):.1f} / 128 "
          f"({np.mean(plane_clamped_counts)/128*100:.1f}%)")
    print(f"    Max |coefficient| before clamping: "
          f"{max(plane_unclamped_maxes)}")
    print(f"    Avg |coefficient| max before clamping: "
          f"{np.mean(plane_unclamped_maxes):.1f}")
    print(f"    Avg distance: {np.mean(plane_distances):.1f}")

    # --- Coefficient distribution histogram ---
    print(f"\n\n  COEFFICIENT DISTRIBUTION (last Babai Rounding trial)")
    print(f"  {'-'*60}")

    # Re-run one trial to get full coefficient vector
    target_seed = hashlib.sha256(seed + (0).to_bytes(4, 'big')).digest()
    target = renderer.render(target_seed)
    B_f = lattice.basis.astype(np.float64)
    t_f = target.astype(np.float64)
    real_coeffs, _, _, _ = np.linalg.lstsq(B_f.T, t_f, rcond=None)
    int_coeffs = np.round(real_coeffs).astype(np.int32)

    abs_coeffs = np.abs(int_coeffs)
    print(f"    Range: [{int_coeffs.min()}, {int_coeffs.max()}]")
    print(f"    Mean |c|: {np.mean(abs_coeffs):.2f}")
    print(f"    Median |c|: {np.median(abs_coeffs):.2f}")
    print(f"    Std |c|: {np.std(abs_coeffs):.2f}")

    # Histogram buckets
    buckets = [(0, 0), (1, 5), (6, 10), (11, 16), (17, 50), (51, 100), (101, 10000)]
    print(f"\n    {'Bucket':<15} {'Count':>6} {'Pct':>7}")
    for lo, hi in buckets:
        count = int(np.sum((abs_coeffs >= lo) & (abs_coeffs <= hi)))
        pct = count / len(abs_coeffs) * 100
        bar = '#' * int(pct / 2)
        print(f"    |c| {lo:>3d}-{hi:>4d}   {count:>4d}   {pct:>5.1f}%  {bar}")

    # --- Lattice density analysis ---
    print(f"\n\n  LATTICE DENSITY ANALYSIS")
    print(f"  {'-'*60}")

    # Compute basis vector norms
    norms = np.array([np.linalg.norm(lattice.basis[i].astype(np.float64))
                       for i in range(k)])
    print(f"    Basis vector norms:")
    print(f"      Min:  {norms.min():.1f}")
    print(f"      Max:  {norms.max():.1f}")
    print(f"      Mean: {norms.mean():.1f}")
    print(f"      Std:  {norms.std():.1f}")

    # Orthogonality check: compute average cosine similarity between basis vectors
    cos_sims = []
    for i in range(min(20, k)):
        for j in range(i + 1, min(20, k)):
            bi = lattice.basis[i].astype(np.float64)
            bj = lattice.basis[j].astype(np.float64)
            cos = np.dot(bi, bj) / (np.linalg.norm(bi) * np.linalg.norm(bj))
            cos_sims.append(abs(cos))
    print(f"\n    Basis vector cosine similarities (first 20 pairs):")
    print(f"      Mean |cos|: {np.mean(cos_sims):.6f}")
    print(f"      Max  |cos|: {np.max(cos_sims):.6f}")
    print(f"      (Near 0 = nearly orthogonal → easier CVP)")

    # Target norm vs lattice reach
    target_norm = np.linalg.norm(target.astype(np.float64))
    max_reach = config.coefficient_bound * np.sum(norms)
    print(f"\n    Target norm: {target_norm:.1f}")
    print(f"    Lattice max reach (sum of B*|b_i|): {max_reach:.1f}")
    print(f"    Ratio (reach/target): {max_reach/target_norm:.1f}x")
    print(f"    → Lattice can reach {max_reach/target_norm:.0f}x further "
          f"than needed")

    # --- Diagnosis ---
    avg_clamped_pct = np.mean(round_clamped_counts) / 128 * 100
    print(f"\n\n{'='*70}")
    print(f"  DIAGNOSIS")
    print(f"{'='*70}")
    if avg_clamped_pct < 5:
        print(f"  ⚠ PROBLEM: Only {avg_clamped_pct:.1f}% of coefficients get clamped.")
        print(f"  The unconstrained least-squares solution already fits within")
        print(f"  [-{config.coefficient_bound}, +{config.coefficient_bound}].")
        print(f"  This means the coefficient bound is NOT constraining the solver —")
        print(f"  the lattice CVP problem is effectively UNCONSTRAINED.")
        print(f"")
        print(f"  RECOMMENDATIONS:")
        print(f"    1. Reduce coefficient_bound from {config.coefficient_bound} to ~3-5")
        print(f"    2. OR reduce num_basis_vectors from {config.num_basis_vectors} to ~32-48")
        print(f"    3. OR both — force the solver to make hard trade-offs")
    elif avg_clamped_pct < 30:
        print(f"  ⚡ MARGINAL: {avg_clamped_pct:.1f}% clamped — some constraint, but weak.")
    else:
        print(f"  ✓ GOOD: {avg_clamped_pct:.1f}% clamped — coefficient bound is binding.")
    print(f"{'='*70}\n")


if __name__ == "__main__":
    analyze_clamping(num_trials=20)
