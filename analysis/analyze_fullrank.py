"""
BAB256 — Full-rank ternary lattice analysis
=============================================
Test: basis_count = dimension (full rank) with ternary entries {-1,0,1}.
Dimensions: 1024 (32x32) and 2048 (32x64).
10 trials each with different basis seeds.
"""

import hashlib
import numpy as np
import time
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from bab256_engine_v02 import BAB256Config, BabelRenderer, CVPSolver


def render_ternary(seed, dim):
    """Generate a ternary {-1,0,1} vector from seed."""
    vec = np.zeros(dim, dtype=np.int32)
    current = seed
    idx = 0
    while idx < dim:
        current = hashlib.sha256(current).digest()
        for byte_val in current:
            if idx >= dim:
                break
            vec[idx] = (byte_val % 3) - 1
            idx += 1
    return vec


def render_target(seed, dim):
    """Generate a [0,255] target vector from seed."""
    vec = np.zeros(dim, dtype=np.int32)
    current = seed
    idx = 0
    while idx < dim:
        current = hashlib.sha256(current).digest()
        for byte_val in current:
            if idx >= dim:
                break
            vec[idx] = byte_val % 256
            idx += 1
    return vec


def test_fullrank(dim, bound=3, num_trials=10):
    """Full-rank ternary test at given dimension."""
    print(f"\n  {'='*70}")
    print(f"  FULL-RANK TEST: dim={dim}, basis={dim}, "
          f"ternary entries, bound={bound}")
    print(f"  Hardness: 2^{0.292 * dim:.0f} classical, "
          f"2^{0.265 * dim:.0f} quantum")
    print(f"  Memory: {dim * dim * 4 / 1024 / 1024:.1f} MB")
    print(f"  {'='*70}\n")

    clamped_pcts = []
    mean_abs_all = []
    max_coeff_all = []
    dists_round = []
    dists_plane = []
    basis_times = []
    lstsq_times = []
    plane_times = []

    for trial in range(num_trials):
        basis_seed = hashlib.sha256(
            b"fullrank_" + dim.to_bytes(4, "big") + trial.to_bytes(4, "big")
        ).digest()

        # Generate full-rank ternary basis
        t0 = time.time()
        basis = np.zeros((dim, dim), dtype=np.int32)
        for i in range(dim):
            bs = hashlib.sha256(basis_seed + i.to_bytes(4, "big")).digest()
            basis[i] = render_ternary(bs, dim)
        basis_time = time.time() - t0
        basis_times.append(basis_time)

        # Generate target
        target_seed = hashlib.sha256(
            basis_seed + (trial + 6666).to_bytes(8, "big")
        ).digest()
        target = render_target(target_seed, dim)

        # lstsq for unclamped coefficients
        B = basis.astype(np.float64)
        t = target.astype(np.float64)

        t0 = time.time()
        real_coeffs, _, _, _ = np.linalg.lstsq(B.T, t, rcond=None)
        lstsq_time = time.time() - t0
        lstsq_times.append(lstsq_time)

        int_c = np.round(real_coeffs).astype(np.int32)
        abs_c = np.abs(int_c)
        exceeds = int(np.sum(abs_c > bound))
        clamped_pcts.append(exceeds / dim * 100)
        mean_abs_all.append(float(np.mean(abs_c)))
        max_coeff_all.append(int(np.max(abs_c)))

        # Babai rounding distance
        config = BAB256Config(
            image_width=dim, image_height=1,
            num_basis_vectors=dim, coefficient_bound=bound,
            basis_entry_range=(-1, 1),
        )
        int_c_cl = np.clip(int_c, -bound, bound).astype(np.int32)
        dist_r = CVPSolver._compute_distance(basis, int_c_cl, target)
        dists_round.append(dist_r)

        # Babai nearest plane (skip if dim > 1024 — O(n^2*n) is too slow)
        if dim <= 1024:
            t0 = time.time()
            coeffs_p, dist_p = CVPSolver.babai_nearest_plane(
                basis, target, config, basis_seed
            )
            plane_time = time.time() - t0
            plane_times.append(plane_time)
            dists_plane.append(dist_p)

        print(f"    Trial {trial:>2d}: clamped={exceeds:>4d}/{dim} "
              f"({exceeds/dim*100:>5.1f}%), "
              f"mean|c|={np.mean(abs_c):>.2f}, max|c|={np.max(abs_c):>3d}, "
              f"dist_r={dist_r:>9.1f}, "
              f"basis={basis_time:.2f}s, lstsq={lstsq_time:.2f}s"
              + (f", plane={plane_time:.2f}s" if dim <= 1024 else ""))

    print(f"\n  --- SUMMARY (dim={dim}) ---")
    print(f"  Clamping:     {np.mean(clamped_pcts):>6.1f}% "
          f"± {np.std(clamped_pcts):.1f}%  "
          f"[{np.min(clamped_pcts):.1f}%, {np.max(clamped_pcts):.1f}%]")
    print(f"  Mean |c|:     {np.mean(mean_abs_all):.2f}")
    print(f"  Max |c|:      {max(max_coeff_all)}")
    print(f"  Dist (round): {np.mean(dists_round):.1f} ± {np.std(dists_round):.1f}")
    if dists_plane:
        print(f"  Dist (plane): {np.mean(dists_plane):.1f} ± {np.std(dists_plane):.1f}")
        print(f"  Plane/Round:  {np.mean(dists_plane)/np.mean(dists_round):.3f} "
              f"(lower = plane is better)")
    print(f"  Basis gen:    {np.mean(basis_times):.2f}s")
    print(f"  lstsq solve:  {np.mean(lstsq_times):.2f}s")
    if plane_times:
        print(f"  Plane solve:  {np.mean(plane_times):.2f}s")
    print(f"  Memory:       {dim * dim * 4 / 1024 / 1024:.1f} MB")

    # Coefficient histogram (last trial)
    print(f"\n  Coefficient histogram (trial {num_trials-1}):")
    print(f"  {'|c|':>5} {'Count':>6} {'Pct':>6}")
    for v in range(min(max(max_coeff_all) + 1, 20)):
        cnt = int(np.sum(abs_c == v))
        pct = cnt / dim * 100
        bar = "#" * min(int(pct), 50)
        if cnt > 0:
            print(f"  {v:>5d} {cnt:>6d} {pct:>5.1f}%  {bar}")
    over = int(np.sum(abs_c > 19))
    if over > 0:
        print(f"  {'>19':>5} {over:>6d} {over/dim*100:>5.1f}%")

    return {
        "dim": dim,
        "clamp_mean": np.mean(clamped_pcts),
        "clamp_std": np.std(clamped_pcts),
        "mean_abs": np.mean(mean_abs_all),
        "max_coeff": max(max_coeff_all),
        "dist_round": np.mean(dists_round),
        "dist_plane": np.mean(dists_plane) if dists_plane else None,
    }


if __name__ == "__main__":
    print(f"{'='*70}")
    print(f"  BAB256 FULL-RANK TERNARY LATTICE ANALYSIS")
    print(f"  Testing whether CVP is genuinely hard at")
    print(f"  cryptographic dimensions with full-rank basis")
    print(f"{'='*70}")

    results = []

    # Test 1024 (32x32) — the target for 2^299 hardness
    r1024 = test_fullrank(1024, bound=3, num_trials=10)
    results.append(r1024)

    # Test 2048 (32x64) if 1024 works
    r2048 = test_fullrank(2048, bound=3, num_trials=5)
    results.append(r2048)

    # Final verdict
    print(f"\n\n{'='*70}")
    print(f"  VERDICT")
    print(f"{'='*70}")
    for r in results:
        dim = r["dim"]
        hardness = 0.292 * dim
        ok_hard = hardness >= 256
        ok_clamp = 8 <= r["clamp_mean"] <= 30 and r["clamp_std"] < 8
        status = "VIABLE" if ok_hard and ok_clamp else "FAIL"
        print(f"  dim={dim:>5d}: 2^{hardness:.0f} hardness, "
              f"{r['clamp_mean']:.1f}% ±{r['clamp_std']:.1f}% clamping → {status}")
    print(f"{'='*70}\n")
