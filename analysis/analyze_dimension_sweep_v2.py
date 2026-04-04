"""
BAB256 — Dimension sweep v2: More trials for reliable statistics
================================================================
Focus on dimensions near the 2^256 hardness threshold (dim ≥ 878).
Use 20 trials per dimension with different seeds to get stable averages.
"""

import hashlib
import numpy as np
import time
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from bab256_engine_v02 import BAB256Config, BabelRenderer, LatticeEngine, CVPSolver


def sweep_dimension(dim, bound=3, num_trials=20):
    """Run num_trials with DIFFERENT basis seeds to capture variance."""
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

    clamped_pcts = []
    max_coeffs = []
    mean_abs_coeffs = []
    dists = []

    for trial in range(num_trials):
        # Use DIFFERENT basis seeds to capture basis-dependent variance
        basis_seed = hashlib.sha256(
            b"sweep_v2_basis_" + trial.to_bytes(4, "big")
        ).digest()
        lattice.generate_basis(basis_seed)

        target_seed = hashlib.sha256(
            basis_seed + (trial + 7777).to_bytes(8, "big")
        ).digest()
        target = renderer.render(target_seed)

        B = lattice.basis.astype(np.float64)
        t = target.astype(np.float64)
        real_coeffs, _, _, _ = np.linalg.lstsq(B.T, t, rcond=None)
        int_c = np.round(real_coeffs).astype(np.int32)

        exceeds = int(np.sum(np.abs(int_c) > bound))
        clamped_pcts.append(exceeds / dim * 100)
        max_coeffs.append(int(np.max(np.abs(int_c))))
        mean_abs_coeffs.append(float(np.mean(np.abs(int_c))))

        # Babai rounding distance
        nonce_seed = hashlib.sha256(target_seed + b"n").digest()
        _, dist = CVPSolver.babai_rounding(
            lattice.basis, target, config, nonce_seed
        )
        dists.append(dist)

    return {
        "dim": dim,
        "image": f"{w}x{h}",
        "bound": bound,
        "classical_bits": 0.292 * dim,
        "quantum_bits": 0.265 * dim,
        "clamped_mean": np.mean(clamped_pcts),
        "clamped_std": np.std(clamped_pcts),
        "clamped_min": np.min(clamped_pcts),
        "clamped_max": np.max(clamped_pcts),
        "max_coeff_mean": np.mean(max_coeffs),
        "max_coeff_max": max(max_coeffs),
        "mean_abs_coeff": np.mean(mean_abs_coeffs),
        "dist_mean": np.mean(dists),
        "dist_std": np.std(dists),
        "basis_mb": dim * dim * 4 / 1024 / 1024,
    }


if __name__ == "__main__":
    print(f"{'='*90}")
    print(f"  DIMENSION SWEEP v2 — 20 trials per dimension, varying basis seeds")
    print(f"  Square system (basis_count = dimension), coefficient_bound=3")
    print(f"  Hardness threshold: 2^256 requires dim ≥ 878")
    print(f"{'='*90}\n")

    # Focus on dimensions near and above the threshold
    dimensions = [128, 256, 512, 768, 896, 1024, 1536, 2048]

    results = []
    for dim in dimensions:
        trials = 20 if dim <= 1024 else 10
        print(f"  dim={dim:>5d} ({trials} trials)...", end="", flush=True)
        t0 = time.time()
        r = sweep_dimension(dim, bound=3, num_trials=trials)
        elapsed = time.time() - t0
        results.append(r)
        print(f" {elapsed:.1f}s  clamped={r['clamped_mean']:.1f}% "
              f"(±{r['clamped_std']:.1f}%)")

    # Results table
    print(f"\n{'='*90}")
    print(f"  {'Dim':>5} {'Image':>8} {'Hard':>8} "
          f"{'Clamp%':>7} {'±σ':>5} {'[min':>5} {'max]':>5} "
          f"{'mean|c|':>8} {'max|c|':>7} {'Dist':>10} {'MB':>6}")
    print(f"  {'-'*88}")

    for r in results:
        h_ok = "✓" if r["classical_bits"] >= 256 else " "
        c_ok = "✓" if 8 <= r["clamped_mean"] <= 25 else " "

        print(
            f" {h_ok}{c_ok}{r['dim']:>5d} {r['image']:>8} "
            f"{'2^'+str(int(r['classical_bits'])):>8} "
            f"{r['clamped_mean']:>6.1f}% {r['clamped_std']:>4.1f} "
            f"{r['clamped_min']:>4.1f} {r['clamped_max']:>5.1f} "
            f"{r['mean_abs_coeff']:>8.2f} {r['max_coeff_max']:>7d} "
            f"{r['dist_mean']:>10.0f} {r['basis_mb']:>5.1f}"
        )

    # Also test different bounds at key dimensions
    print(f"\n\n{'='*90}")
    print(f"  BOUND SENSITIVITY at dim=1024 (20 trials each)")
    print(f"{'='*90}")

    for bound in [1, 2, 3, 4, 5, 6, 8]:
        print(f"  bound={bound}...", end="", flush=True)
        r = sweep_dimension(1024, bound=bound, num_trials=20)
        print(f" clamped={r['clamped_mean']:>6.1f}% ±{r['clamped_std']:>4.1f}  "
              f"mean|c|={r['mean_abs_coeff']:.2f}  "
              f"dist={r['dist_mean']:.0f}")

    print(f"\n\n{'='*90}")
    print(f"  BOUND SENSITIVITY at dim=896 (20 trials each)")
    print(f"{'='*90}")

    for bound in [1, 2, 3, 4, 5, 6, 8]:
        print(f"  bound={bound}...", end="", flush=True)
        r = sweep_dimension(896, bound=bound, num_trials=20)
        print(f" clamped={r['clamped_mean']:>6.1f}% ±{r['clamped_std']:>4.1f}  "
              f"mean|c|={r['mean_abs_coeff']:.2f}  "
              f"dist={r['dist_mean']:.0f}")

    print()
