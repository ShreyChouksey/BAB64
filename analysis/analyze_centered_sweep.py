"""
BAB256 — Centered basis sweep
==============================
Test whether centering the basis (subtracting per-pixel mean across
all basis vectors) stabilizes the coefficient distribution.
"""

import hashlib
import numpy as np
import time
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from bab256_engine_v02 import BAB256Config, BabelRenderer, LatticeEngine, CVPSolver


def sweep_centered(dim, bound=3, num_trials=20):
    """Analyze with centered basis vectors."""
    w = int(np.sqrt(dim))
    while dim % w != 0:
        w -= 1
    h = dim // w

    config = BAB256Config(
        image_width=w, image_height=h,
        num_basis_vectors=dim, coefficient_bound=bound,
    )
    renderer = BabelRenderer(config)
    lattice = LatticeEngine(config)

    clamped_pcts_raw = []
    clamped_pcts_centered = []
    mean_abs_raw = []
    mean_abs_centered = []
    max_coeff_raw = []
    max_coeff_centered = []
    dists_raw = []
    dists_centered = []

    for trial in range(num_trials):
        basis_seed = hashlib.sha256(
            b"centered_sweep_" + trial.to_bytes(4, "big")
        ).digest()
        lattice.generate_basis(basis_seed)

        target_seed = hashlib.sha256(
            basis_seed + (trial + 8888).to_bytes(8, "big")
        ).digest()
        target = renderer.render(target_seed)

        B = lattice.basis.astype(np.float64)
        t = target.astype(np.float64)

        # --- RAW (uncentered) ---
        coeffs_raw, _, _, _ = np.linalg.lstsq(B.T, t, rcond=None)
        int_raw = np.round(coeffs_raw).astype(np.int32)
        clamped_pcts_raw.append(
            np.sum(np.abs(int_raw) > bound) / dim * 100
        )
        mean_abs_raw.append(np.mean(np.abs(int_raw)))
        max_coeff_raw.append(int(np.max(np.abs(int_raw))))

        int_raw_cl = np.clip(int_raw, -bound, bound)
        lp = np.dot(int_raw_cl.astype(np.int64), B.astype(np.int64))
        dists_raw.append(float(np.sqrt(np.sum((lp - t.astype(np.int64))**2))))

        # --- CENTERED ---
        # Compute per-pixel mean across basis vectors
        pixel_mean = B.mean(axis=0)  # shape (dim,)
        B_c = B - pixel_mean  # centered basis
        t_c = t - pixel_mean  # centered target

        coeffs_c, _, _, _ = np.linalg.lstsq(B_c.T, t_c, rcond=None)
        int_c = np.round(coeffs_c).astype(np.int32)
        clamped_pcts_centered.append(
            np.sum(np.abs(int_c) > bound) / dim * 100
        )
        mean_abs_centered.append(np.mean(np.abs(int_c)))
        max_coeff_centered.append(int(np.max(np.abs(int_c))))

        int_c_cl = np.clip(int_c, -bound, bound)
        lp_c = np.dot(int_c_cl.astype(np.float64), B_c)
        dists_centered.append(
            float(np.sqrt(np.sum((lp_c - t_c)**2)))
        )

    return {
        "dim": dim,
        "raw_clamp_mean": np.mean(clamped_pcts_raw),
        "raw_clamp_std": np.std(clamped_pcts_raw),
        "raw_mean_abs": np.mean(mean_abs_raw),
        "raw_max_coeff": max(max_coeff_raw),
        "raw_dist": np.mean(dists_raw),
        "cen_clamp_mean": np.mean(clamped_pcts_centered),
        "cen_clamp_std": np.std(clamped_pcts_centered),
        "cen_mean_abs": np.mean(mean_abs_centered),
        "cen_max_coeff": max(max_coeff_centered),
        "cen_dist": np.mean(dists_centered),
    }


if __name__ == "__main__":
    print(f"{'='*95}")
    print(f"  RAW vs CENTERED BASIS — coefficient_bound=3, 20 trials per dim")
    print(f"{'='*95}\n")

    dimensions = [128, 256, 512, 768, 896, 1024, 2048]

    print(f"  {'':>5} {'--- RAW (uncentered) ---':>40}   {'--- CENTERED ---':>40}")
    print(f"  {'Dim':>5} {'Clamp%':>7} {'±σ':>5} {'mean|c|':>8} {'max|c|':>7} {'Dist':>9}"
          f"   {'Clamp%':>7} {'±σ':>5} {'mean|c|':>8} {'max|c|':>7} {'Dist':>9}")
    print(f"  {'-'*93}")

    for dim in dimensions:
        trials = 20 if dim <= 1024 else 10
        print(f"  dim={dim}...", end="", flush=True)
        r = sweep_centered(dim, bound=3, num_trials=trials)
        print(f"\r", end="")

        print(
            f"  {r['dim']:>5d} {r['raw_clamp_mean']:>6.1f}% {r['raw_clamp_std']:>4.1f} "
            f"{r['raw_mean_abs']:>8.2f} {r['raw_max_coeff']:>7d} {r['raw_dist']:>9.0f}"
            f"   {r['cen_clamp_mean']:>6.1f}% {r['cen_clamp_std']:>4.1f} "
            f"{r['cen_mean_abs']:>8.2f} {r['cen_max_coeff']:>7d} {r['cen_dist']:>9.0f}"
        )

    # Deep dive: coefficient histogram for centered at key dimensions
    print(f"\n\n{'='*95}")
    print(f"  CENTERED COEFFICIENT DISTRIBUTIONS")
    print(f"{'='*95}")

    for dim in [896, 1024]:
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

        # Aggregate across 10 seeds
        all_abs_coeffs = []
        for trial in range(10):
            basis_seed = hashlib.sha256(
                b"hist_" + trial.to_bytes(4, "big")
            ).digest()
            lattice.generate_basis(basis_seed)

            target_seed = hashlib.sha256(
                basis_seed + (trial + 1111).to_bytes(8, "big")
            ).digest()
            target = renderer.render(target_seed)

            B = lattice.basis.astype(np.float64)
            t = target.astype(np.float64)
            pixel_mean = B.mean(axis=0)
            B_c = B - pixel_mean
            t_c = t - pixel_mean

            coeffs, _, _, _ = np.linalg.lstsq(B_c.T, t_c, rcond=None)
            int_c = np.round(coeffs).astype(np.int32)
            all_abs_coeffs.extend(np.abs(int_c).tolist())

        all_abs = np.array(all_abs_coeffs)
        print(f"\n  dim={dim} (aggregated over 10 seeds, {len(all_abs)} coefficients):")
        print(f"    Mean |c|: {np.mean(all_abs):.3f}")
        print(f"    Std |c|: {np.std(all_abs):.3f}")
        print(f"    {'|c|':>5} {'Count':>8} {'Pct':>6}")
        for v in range(11):
            cnt = int(np.sum(all_abs == v))
            pct = cnt / len(all_abs) * 100
            bar = "#" * min(int(pct), 50)
            print(f"    {v:>5d} {cnt:>8d} {pct:>5.1f}%  {bar}")
        over = int(np.sum(all_abs > 10))
        if over > 0:
            print(f"    {'>10':>5} {over:>8d} {over/len(all_abs)*100:>5.1f}%")

    # Bound sensitivity for centered at best dimension
    print(f"\n\n{'='*95}")
    print(f"  BOUND SENSITIVITY — CENTERED, dim=896 (2^261 hardness)")
    print(f"{'='*95}")

    for bound in [1, 2, 3, 4, 5, 6, 8, 10]:
        r = sweep_centered(896, bound=bound, num_trials=20)
        print(f"  bound={bound:>2d}: clamped={r['cen_clamp_mean']:>6.1f}% "
              f"±{r['cen_clamp_std']:>4.1f}  "
              f"mean|c|={r['cen_mean_abs']:.2f}  "
              f"dist={r['cen_dist']:.0f}")

    print()
