"""
BAB256 — Basis entry range sweep
=================================
Hypothesis: if basis entries are small (e.g., ternary {-1,0,1}),
the coefficients must be larger to approximate the [0,255] target,
making the bound constraint meaningful.

We keep 4096d (64x64) for hardness, with 128 basis vectors.
"""

import hashlib
import numpy as np
import time
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from bab256_engine_v02 import BAB256Config, BabelRenderer


def render_small_entry(seed, dim, entry_range):
    """Render a vector with small integer entries from a seed."""
    lo, hi = entry_range
    width = hi - lo + 1
    vec = np.zeros(dim, dtype=np.int32)
    current = seed
    idx = 0
    while idx < dim:
        current = hashlib.sha256(current).digest()
        for byte_val in current:
            if idx >= dim:
                break
            vec[idx] = (byte_val % width) + lo
            idx += 1
    return vec


def test_entry_range(dim, n_basis, bound, entry_range, num_trials=20):
    """Test clamping behavior with a given basis entry range."""
    clamped_pcts = []
    mean_abs_coeffs = []
    max_coeffs = []
    dists = []

    for trial in range(num_trials):
        seed = hashlib.sha256(
            b"entry_range_" + trial.to_bytes(4, "big")
        ).digest()

        # Generate basis with small entries
        basis = np.zeros((n_basis, dim), dtype=np.int32)
        for i in range(n_basis):
            basis_seed = hashlib.sha256(seed + i.to_bytes(4, "big")).digest()
            basis[i] = render_small_entry(basis_seed, dim, entry_range)

        # Target is still full-range [0,255] (the "image" to approximate)
        renderer = BabelRenderer(BAB256Config(
            image_width=64, image_height=64,
        ))
        target_seed = hashlib.sha256(
            seed + (trial + 3333).to_bytes(8, "big")
        ).digest()
        target = renderer.render(target_seed)[:dim]

        B = basis.astype(np.float64)
        t = target.astype(np.float64)

        coeffs, _, _, _ = np.linalg.lstsq(B.T, t, rcond=None)
        int_c = np.round(coeffs).astype(np.int32)

        exceeds = np.sum(np.abs(int_c) > bound)
        clamped_pcts.append(float(exceeds) / n_basis * 100)
        mean_abs_coeffs.append(float(np.mean(np.abs(int_c))))
        max_coeffs.append(int(np.max(np.abs(int_c))))

        # Distance after clamping
        int_c_cl = np.clip(int_c, -bound, bound)
        lp = np.dot(int_c_cl.astype(np.int64), basis.astype(np.int64))
        diff = lp - target[:dim].astype(np.int64)
        dists.append(float(np.sqrt(np.sum(diff ** 2))))

    return {
        "entry_range": entry_range,
        "dim": dim,
        "n_basis": n_basis,
        "bound": bound,
        "clamp_mean": np.mean(clamped_pcts),
        "clamp_std": np.std(clamped_pcts),
        "mean_abs": np.mean(mean_abs_coeffs),
        "max_coeff": max(max_coeffs),
        "dist_mean": np.mean(dists),
        "dist_std": np.std(dists),
    }


if __name__ == "__main__":
    print(f"{'='*90}")
    print(f"  BASIS ENTRY RANGE SWEEP")
    print(f"  dim=4096 (64x64), 128 basis vectors, 20 trials each")
    print(f"  Target: [0,255] pixel values")
    print(f"{'='*90}\n")

    # Test different entry ranges with different bounds
    configs = [
        # (entry_range, bound, label)
        ((-1, 1), 16, "ternary {-1,0,1}, B=16"),
        ((-1, 1), 8, "ternary {-1,0,1}, B=8"),
        ((-1, 1), 4, "ternary {-1,0,1}, B=4"),
        ((-3, 3), 16, "[-3,3], B=16"),
        ((-3, 3), 8, "[-3,3], B=8"),
        ((-3, 3), 4, "[-3,3], B=4"),
        ((-5, 5), 16, "[-5,5], B=16"),
        ((-5, 5), 8, "[-5,5], B=8"),
        ((-10, 10), 8, "[-10,10], B=8"),
        ((0, 255), 16, "[0,255] (original), B=16"),
        ((0, 255), 3, "[0,255] (original), B=3"),
    ]

    print(f"  {'Config':<30} {'Clamp%':>7} {'±σ':>5} {'mean|c|':>8} "
          f"{'max|c|':>7} {'Dist':>10} {'Dist σ':>8}")
    print(f"  {'-'*80}")

    for entry_range, bound, label in configs:
        r = test_entry_range(4096, 128, bound, entry_range, num_trials=20)
        print(f"  {label:<30} {r['clamp_mean']:>6.1f}% {r['clamp_std']:>4.1f} "
              f"{r['mean_abs']:>8.1f} {r['max_coeff']:>7d} "
              f"{r['dist_mean']:>10.0f} {r['dist_std']:>8.0f}")

    # Deep dive on most promising: ternary basis with different bounds
    print(f"\n\n{'='*90}")
    print(f"  TERNARY BASIS DEEP DIVE — dim=4096, 128 basis, 20 trials")
    print(f"{'='*90}")

    for bound in [2, 3, 4, 6, 8, 12, 16, 24, 32]:
        r = test_entry_range(4096, 128, bound, (-1, 1), num_trials=20)
        sweet = "  <<<" if 10 <= r["clamp_mean"] <= 25 and r["clamp_std"] < 5 else ""
        print(f"  bound={bound:>3d}: clamp={r['clamp_mean']:>6.1f}% "
              f"±{r['clamp_std']:>4.1f}  mean|c|={r['mean_abs']:>5.1f}  "
              f"max|c|={r['max_coeff']:>4d}  dist={r['dist_mean']:>8.0f}{sweet}")

    # Coefficient histogram for the sweet spot
    print(f"\n\n{'='*90}")
    print(f"  COEFFICIENT HISTOGRAM — ternary basis, dim=4096, 128 basis")
    print(f"{'='*90}")

    seed = hashlib.sha256(b"histogram_test").digest()
    basis = np.zeros((128, 4096), dtype=np.int32)
    for i in range(128):
        basis_seed = hashlib.sha256(seed + i.to_bytes(4, "big")).digest()
        basis[i] = render_small_entry(basis_seed, 4096, (-1, 1))

    renderer = BabelRenderer(BAB256Config())
    target_seed = hashlib.sha256(seed + b"target_hist").digest()
    target = renderer.render(target_seed)[:4096]

    coeffs, _, _, _ = np.linalg.lstsq(
        basis.astype(np.float64).T, target.astype(np.float64), rcond=None
    )
    int_c = np.round(coeffs).astype(np.int32)
    abs_c = np.abs(int_c)

    print(f"  Range: [{int_c.min()}, {int_c.max()}]")
    print(f"  Mean |c|: {np.mean(abs_c):.2f}")
    print(f"  Std |c|: {np.std(abs_c):.2f}")
    print(f"\n  {'|c|':>5} {'Count':>6} {'Pct':>6}")
    for v in range(max(abs_c) + 1):
        cnt = int(np.sum(abs_c == v))
        pct = cnt / 128 * 100
        bar = "#" * min(int(pct / 2), 40)
        if cnt > 0:
            print(f"  {v:>5d} {cnt:>6d} {pct:>5.1f}%  {bar}")

    # Basis vector norms
    norms = [np.linalg.norm(basis[i].astype(np.float64)) for i in range(128)]
    print(f"\n  Ternary basis vector norms: "
          f"mean={np.mean(norms):.1f}, std={np.std(norms):.1f}")
    print(f"  Target norm: {np.linalg.norm(target.astype(np.float64)):.1f}")
    print(f"  Ratio (target/basis_norm): "
          f"{np.linalg.norm(target.astype(np.float64))/np.mean(norms):.1f}")
    print()
