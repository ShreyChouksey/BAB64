"""
BAB256 v0.3 — Dimension sweep for hardness/clamping sweet spot
================================================================
Question: What's the minimum dimension (with square system) where
Babai coefficients still get ~10-20% clamped at bound=3?

We need dimension ≥ 878 for 2^256 classical hardness (0.292 * 878).
"""

import hashlib
import numpy as np
import time
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from bab256_engine_v02 import BAB256Config, BabelRenderer, LatticeEngine, CVPSolver


def analyze_dimension(dim, bound=3, num_trials=3):
    """Analyze clamping behavior for a square system at given dimension."""
    # Find image_width x image_height = dim
    # Use factors close to sqrt for reasonable image shape
    w = int(np.sqrt(dim))
    while dim % w != 0:
        w -= 1
    h = dim // w

    config = BAB256Config(
        image_width=w,
        image_height=h,
        num_basis_vectors=dim,
        coefficient_bound=bound,
    )

    renderer = BabelRenderer(config)
    lattice = LatticeEngine(config)

    seed = hashlib.sha256(b"dimension_sweep_v2").digest()

    # Time basis generation
    t0 = time.time()
    lattice.generate_basis(seed)
    basis_time = time.time() - t0
    basis_mb = lattice.basis.nbytes / 1024 / 1024

    clamped_counts = []
    max_coeffs = []
    mean_abs_coeffs = []
    dists_round = []
    dists_plane = []
    lstsq_times = []

    for trial in range(num_trials):
        target_seed = hashlib.sha256(
            seed + (trial + 5000).to_bytes(8, "big")
        ).digest()
        target = renderer.render(target_seed)
        nonce_seed = hashlib.sha256(target_seed + b"sweep").digest()

        B = lattice.basis.astype(np.float64)
        t = target.astype(np.float64)

        # Lstsq to get unclamped coefficients
        t_lstsq = time.time()
        real_coeffs, _, _, _ = np.linalg.lstsq(B.T, t, rcond=None)
        lstsq_time = time.time() - t_lstsq
        lstsq_times.append(lstsq_time)

        int_coeffs = np.round(real_coeffs).astype(np.int32)
        exceeds = int(np.sum(np.abs(int_coeffs) > bound))
        clamped_counts.append(exceeds)
        max_coeffs.append(int(np.max(np.abs(int_coeffs))))
        mean_abs_coeffs.append(float(np.mean(np.abs(int_coeffs))))

        # Babai rounding distance
        coeffs_r, dist_r = CVPSolver.babai_rounding(
            lattice.basis, target, config, nonce_seed
        )
        dists_round.append(dist_r)

        # Babai nearest plane distance (skip for very large dims — too slow)
        if dim <= 1024:
            coeffs_p, dist_p = CVPSolver.babai_nearest_plane(
                lattice.basis, target, config, nonce_seed
            )
            dists_plane.append(dist_p)

    classical_bits = 0.292 * dim
    quantum_bits = 0.265 * dim

    return {
        "dim": dim,
        "image": f"{w}x{h}",
        "bound": bound,
        "basis_time": basis_time,
        "basis_mb": basis_mb,
        "avg_lstsq_time": np.mean(lstsq_times),
        "avg_clamped": np.mean(clamped_counts),
        "pct_clamped": np.mean(clamped_counts) / dim * 100,
        "avg_max_coeff": np.mean(max_coeffs),
        "avg_mean_abs_coeff": np.mean(mean_abs_coeffs),
        "avg_dist_round": np.mean(dists_round),
        "avg_dist_plane": np.mean(dists_plane) if dists_plane else None,
        "classical_bits": classical_bits,
        "quantum_bits": quantum_bits,
    }


if __name__ == "__main__":
    print(f"{'='*85}")
    print(f"  DIMENSION SWEEP — Square System, coefficient_bound=3")
    print(f"  Target: ≥2^256 classical hardness AND ~10-20% clamping")
    print(f"{'='*85}\n")

    # Sweep from small to large
    dimensions = [128, 256, 512, 768, 896, 1024, 2048, 4096]

    results = []
    for dim in dimensions:
        print(f"  Testing dim={dim}...", end="", flush=True)
        t0 = time.time()
        # Fewer trials for large dimensions
        trials = 3 if dim <= 1024 else 2
        r = analyze_dimension(dim, bound=3, num_trials=trials)
        elapsed = time.time() - t0
        results.append(r)
        print(f" done ({elapsed:.1f}s)")

    # Print results table
    print(f"\n{'='*85}")
    print(f"  {'Dim':>5} {'Image':>8} {'Hardness':>10} {'Clamped%':>9} "
          f"{'max|c|':>7} {'mean|c|':>8} {'Dist(R)':>10} {'Dist(NP)':>10} "
          f"{'Basis MB':>9}")
    print(f"  {'-'*83}")

    for r in results:
        hardness = f"2^{r['classical_bits']:.0f}"
        dist_np = f"{r['avg_dist_plane']:.0f}" if r['avg_dist_plane'] is not None else "skipped"
        meets_hardness = "✓" if r["classical_bits"] >= 256 else " "
        meets_clamping = "✓" if 8 <= r["pct_clamped"] <= 25 else " "
        marker = " <<<" if meets_hardness == "✓" and meets_clamping == "✓" else ""

        print(
            f" {meets_hardness}{meets_clamping}{r['dim']:>5d} {r['image']:>8} "
            f"{hardness:>10} {r['pct_clamped']:>8.1f}% "
            f"{r['avg_max_coeff']:>7.0f} {r['avg_mean_abs_coeff']:>8.2f} "
            f"{r['avg_dist_round']:>10.0f} {dist_np:>10} "
            f"{r['basis_mb']:>8.1f}{marker}"
        )

    # Coefficient distribution deep dive for most promising dimension
    print(f"\n\n{'='*85}")
    print(f"  COEFFICIENT DISTRIBUTION DEEP DIVE")
    print(f"{'='*85}")

    for dim in [512, 896, 1024]:
        w = int(np.sqrt(dim))
        while dim % w != 0:
            w -= 1
        h = dim // w

        config = BAB256Config(
            image_width=w, image_height=h,
            num_basis_vectors=dim, coefficient_bound=3,
        )
        renderer = BabelRenderer(config)
        lattice = LatticeEngine(config)
        seed = hashlib.sha256(b"coeff_deep_dive").digest()
        lattice.generate_basis(seed)

        target_seed = hashlib.sha256(seed + (9999).to_bytes(8, "big")).digest()
        target = renderer.render(target_seed)

        B = lattice.basis.astype(np.float64)
        t = target.astype(np.float64)
        real_coeffs, _, _, _ = np.linalg.lstsq(B.T, t, rcond=None)
        int_c = np.round(real_coeffs).astype(np.int32)
        abs_c = np.abs(int_c)

        print(f"\n  dim={dim} ({w}x{h}):")
        print(f"    Range: [{int_c.min()}, {int_c.max()}]")
        print(f"    Mean |c|: {np.mean(abs_c):.3f}")
        print(f"    Std |c|: {np.std(abs_c):.3f}")
        print(f"    Exceeds bound=3: {np.sum(abs_c > 3)}/{dim} "
              f"({np.sum(abs_c > 3)/dim*100:.1f}%)")
        print(f"    Exceeds bound=2: {np.sum(abs_c > 2)}/{dim} "
              f"({np.sum(abs_c > 2)/dim*100:.1f}%)")

        # Histogram
        buckets = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
        print(f"    {'|c|':>5} {'Count':>6} {'Pct':>6}")
        for v in buckets:
            cnt = int(np.sum(abs_c == v))
            pct = cnt / dim * 100
            bar = "#" * min(int(pct), 50)
            print(f"    {v:>5d} {cnt:>6d} {pct:>5.1f}%  {bar}")
        over = int(np.sum(abs_c > 10))
        if over > 0:
            print(f"    {'>10':>5} {over:>6d} {over/dim*100:>5.1f}%")

    # Performance summary
    print(f"\n\n{'='*85}")
    print(f"  PERFORMANCE SUMMARY")
    print(f"{'='*85}")
    for r in results:
        print(f"  dim={r['dim']:>5d}: basis={r['basis_time']:.1f}s, "
              f"lstsq={r['avg_lstsq_time']:.2f}s, "
              f"memory={r['basis_mb']:.1f}MB")

    # Recommendation
    print(f"\n\n{'='*85}")
    print(f"  RECOMMENDATION")
    print(f"{'='*85}")
    sweet = [r for r in results
             if r["classical_bits"] >= 256 and 5 <= r["pct_clamped"] <= 30]
    if sweet:
        best = min(sweet, key=lambda r: r["dim"])
        print(f"  Minimum viable dimension: {best['dim']}")
        print(f"    Classical hardness: 2^{best['classical_bits']:.0f}")
        print(f"    Quantum hardness: 2^{best['quantum_bits']:.0f}")
        print(f"    Clamping: {best['pct_clamped']:.1f}%")
        print(f"    Babai rounding distance: {best['avg_dist_round']:.0f}")
        print(f"    Basis memory: {best['basis_mb']:.1f} MB")
        print(f"    Image shape: {best['image']}")
    else:
        print(f"  No dimension found meeting both criteria!")
        print(f"  Consider adjusting coefficient_bound.")
    print(f"{'='*85}\n")
