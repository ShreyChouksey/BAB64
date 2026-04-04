"""
BAB256 — Full-rank hardness test
=================================
Critical test: does full-rank (basis_count = dimension) with ternary
entries {-1,0,1} and coefficient_bound=3 give viable clamping?

Key insight: CVP hardness scales with lattice RANK, not ambient dimension.
128 vectors in 4096d → rank 128 → 2^37 hardness (useless).
1024 vectors in 1024d → rank 1024 → 2^299 hardness (exceeds SHA-256).

Test plan:
  1. dim=1024 (32×32), 10 trials, bound=3 → target 10-20% clamping
  2. dim=2048 (if 1024 works), 5 trials → watch memory/compute
"""

import hashlib
import numpy as np
import time
import sys


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


def compute_distance(basis, coeffs, target):
    lp = np.dot(coeffs.astype(np.int64), basis.astype(np.int64))
    diff = lp - target.astype(np.int64)
    return float(np.sqrt(np.sum(diff ** 2)))


def test_fullrank(dim, bound=3, num_trials=10):
    """Full-rank ternary CVP analysis."""
    classical = 0.292 * dim
    quantum = 0.265 * dim
    mem_mb = dim * dim * 4 / 1024 / 1024

    print(f"\n  {'='*70}")
    print(f"  FULL-RANK TEST: dim={dim}, basis={dim}, ternary, bound={bound}")
    print(f"  Classical hardness: 2^{classical:.0f}  |  Quantum: 2^{quantum:.0f}")
    print(f"  Memory estimate: {mem_mb:.1f} MB for basis matrix")
    print(f"  {'='*70}\n")

    clamped_pcts = []
    mean_abs_list = []
    max_coeff_list = []
    dists = []
    basis_times = []
    lstsq_times = []

    for trial in range(num_trials):
        basis_seed = hashlib.sha256(
            b"fullrank_v2_" + dim.to_bytes(4, "big") + trial.to_bytes(4, "big")
        ).digest()

        # Generate full-rank ternary basis (n×n)
        t0 = time.time()
        basis = np.zeros((dim, dim), dtype=np.int32)
        for i in range(dim):
            bs = hashlib.sha256(basis_seed + i.to_bytes(4, "big")).digest()
            basis[i] = render_ternary(bs, dim)
        basis_time = time.time() - t0
        basis_times.append(basis_time)

        # Check rank
        if trial == 0:
            rank = np.linalg.matrix_rank(basis.astype(np.float64))
            print(f"  Rank check (trial 0): {rank}/{dim} "
                  f"({'FULL' if rank == dim else 'DEFICIENT!'})")

        # Generate target in [0, 255]
        target_seed = hashlib.sha256(
            basis_seed + (trial + 9999).to_bytes(8, "big")
        ).digest()
        target = render_target(target_seed, dim)

        # Solve lstsq for unclamped coefficients
        B = basis.astype(np.float64)
        t_vec = target.astype(np.float64)

        t0 = time.time()
        real_coeffs, residuals, rank_out, sv = np.linalg.lstsq(B.T, t_vec, rcond=None)
        lstsq_time = time.time() - t0
        lstsq_times.append(lstsq_time)

        # Analyze coefficients
        int_c = np.round(real_coeffs).astype(np.int32)
        abs_c = np.abs(int_c)
        exceeds = int(np.sum(abs_c > bound))
        clamp_pct = exceeds / dim * 100
        clamped_pcts.append(clamp_pct)
        mean_abs_list.append(float(np.mean(abs_c)))
        max_coeff_list.append(int(np.max(abs_c)))

        # Compute CVP distance after clamping
        clamped_c = np.clip(int_c, -bound, bound).astype(np.int32)
        dist = compute_distance(basis, clamped_c, target)
        dists.append(dist)

        print(f"    Trial {trial:>2d}: clamped={exceeds:>4d}/{dim} "
              f"({clamp_pct:>5.1f}%), "
              f"mean|c|={np.mean(abs_c):>.2f}, "
              f"max|c|={np.max(abs_c):>3d}, "
              f"dist={dist:>9.1f}, "
              f"basis={basis_time:.2f}s, lstsq={lstsq_time:.2f}s")

    # Summary
    print(f"\n  --- SUMMARY (dim={dim}, bound={bound}) ---")
    print(f"  Clamping:     {np.mean(clamped_pcts):>6.1f}% "
          f"± {np.std(clamped_pcts):.1f}%  "
          f"[{np.min(clamped_pcts):.1f}%, {np.max(clamped_pcts):.1f}%]")
    print(f"  Mean |c|:     {np.mean(mean_abs_list):.2f} "
          f"± {np.std(mean_abs_list):.2f}")
    print(f"  Max |c|:      {max(max_coeff_list)}")
    print(f"  CVP distance: {np.mean(dists):.1f} ± {np.std(dists):.1f}")
    print(f"  Basis gen:    {np.mean(basis_times):.2f}s avg")
    print(f"  lstsq solve:  {np.mean(lstsq_times):.2f}s avg")
    print(f"  Memory:       {mem_mb:.1f} MB")
    print(f"  Hardness:     2^{classical:.0f} classical, 2^{quantum:.0f} quantum")
    sha_margin = classical - 256
    print(f"  vs SHA-256:   {'EXCEEDS' if sha_margin > 0 else 'BELOW'} "
          f"by 2^{abs(sha_margin):.0f}")

    # Coefficient histogram (last trial)
    print(f"\n  Coefficient histogram (last trial):")
    print(f"  {'|c|':>5} {'Count':>6} {'Pct':>6}")
    for v in range(min(max(max_coeff_list) + 1, 15)):
        cnt = int(np.sum(abs_c == v))
        pct = cnt / dim * 100
        bar = "#" * min(int(pct * 2), 50)
        if cnt > 0:
            print(f"  {v:>5d} {cnt:>6d} {pct:>5.1f}%  {bar}")
    over = int(np.sum(abs_c >= 15))
    if over > 0:
        print(f"  {'>=15':>5} {over:>6d} {over/dim*100:>5.1f}%")

    return {
        "dim": dim,
        "clamp_mean": np.mean(clamped_pcts),
        "clamp_std": np.std(clamped_pcts),
        "mean_abs": np.mean(mean_abs_list),
        "max_coeff": max(max_coeff_list),
        "dist_mean": np.mean(dists),
        "dist_std": np.std(dists),
        "lstsq_time": np.mean(lstsq_times),
        "basis_time": np.mean(basis_times),
        "classical_hardness": classical,
    }


if __name__ == "__main__":
    print(f"{'='*70}")
    print(f"  BAB256 FULL-RANK HARDNESS TEST")
    print(f"  Key insight: CVP hardness = 2^(0.292 × rank)")
    print(f"  Full-rank → rank = dimension → real hardness")
    print(f"{'='*70}")

    # Test 1: dim=1024, 10 trials
    r1024 = test_fullrank(1024, bound=3, num_trials=10)

    # Test 2: dim=2048 if 1024 looks good
    viable_1024 = (8 <= r1024["clamp_mean"] <= 30 and r1024["clamp_std"] < 8)
    if viable_1024:
        print(f"\n  1024 looks viable — proceeding to 2048...")
        r2048 = test_fullrank(2048, bound=3, num_trials=5)
    else:
        print(f"\n  1024 clamping out of range — skipping 2048")
        r2048 = None

    # Final verdict
    print(f"\n\n{'='*70}")
    print(f"  VERDICT")
    print(f"{'='*70}")
    for r in [r1024] + ([r2048] if r2048 else []):
        dim = r["dim"]
        h = r["classical_hardness"]
        ok_h = h >= 256
        ok_c = 8 <= r["clamp_mean"] <= 30 and r["clamp_std"] < 8
        status = "VIABLE" if ok_h and ok_c else "NEEDS WORK"
        print(f"  dim={dim:>5d}: 2^{h:.0f} hardness, "
              f"{r['clamp_mean']:.1f}% ±{r['clamp_std']:.1f}% clamping, "
              f"lstsq={r['lstsq_time']:.2f}s → {status}")
    print(f"{'='*70}\n")
