"""
BAB256 — Full-rank bound sweep
================================
Find the coefficient_bound that gives 10-25% clamping
for full-rank ternary lattices at dim=1024.
"""

import hashlib
import numpy as np
import time
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def render_ternary(seed, dim):
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


def measure_coefficients(dim, num_trials=10):
    """Measure the unclamped coefficient distribution for full-rank ternary."""
    all_abs_coeffs = []

    for trial in range(num_trials):
        basis_seed = hashlib.sha256(
            b"bound_sweep_" + dim.to_bytes(4, "big") + trial.to_bytes(4, "big")
        ).digest()

        basis = np.zeros((dim, dim), dtype=np.int32)
        for i in range(dim):
            bs = hashlib.sha256(basis_seed + i.to_bytes(4, "big")).digest()
            basis[i] = render_ternary(bs, dim)

        target_seed = hashlib.sha256(
            basis_seed + (trial + 7777).to_bytes(8, "big")
        ).digest()
        target = render_target(target_seed, dim)

        coeffs, _, _, _ = np.linalg.lstsq(
            basis.astype(np.float64).T, target.astype(np.float64), rcond=None
        )
        int_c = np.round(coeffs).astype(np.int32)
        all_abs_coeffs.extend(np.abs(int_c).tolist())

    return np.array(all_abs_coeffs)


if __name__ == "__main__":
    for dim in [1024, 2048]:
        trials = 10 if dim <= 1024 else 5
        print(f"\n{'='*70}")
        print(f"  dim={dim}: Measuring unclamped coefficient distribution "
              f"({trials} trials)")
        print(f"{'='*70}")

        t0 = time.time()
        abs_coeffs = measure_coefficients(dim, num_trials=trials)
        elapsed = time.time() - t0
        print(f"  Computed in {elapsed:.1f}s")

        # Percentile analysis
        percentiles = [50, 75, 80, 85, 90, 95, 99]
        print(f"\n  Percentile analysis ({len(abs_coeffs)} coefficients):")
        print(f"    Mean |c|:   {np.mean(abs_coeffs):.1f}")
        print(f"    Median |c|: {np.median(abs_coeffs):.1f}")
        print(f"    Max |c|:    {np.max(abs_coeffs)}")
        print()
        for p in percentiles:
            val = np.percentile(abs_coeffs, p)
            print(f"    P{p:>2d}: {val:>8.1f}  "
                  f"(bound={int(np.ceil(val))} → "
                  f"{(1-p/100)*100:.0f}% clamped)")

        # Find bounds for target clamping rates
        print(f"\n  Target clamping → required bound:")
        for target_clamp in [25, 20, 15, 10, 5]:
            target_pct = 100 - target_clamp
            bound = int(np.ceil(np.percentile(abs_coeffs, target_pct)))
            actual_clamp = np.sum(abs_coeffs > bound) / len(abs_coeffs) * 100
            print(f"    {target_clamp:>2d}% clamped → bound={bound:>5d}  "
                  f"(actual: {actual_clamp:.1f}%)")

        # Per-trial variance check
        print(f"\n  Per-trial coefficient magnitude (mean |c|):")
        per_trial_means = []
        per_trial_medians = []
        chunk = dim
        for i in range(trials):
            chunk_data = abs_coeffs[i * chunk:(i + 1) * chunk]
            per_trial_means.append(np.mean(chunk_data))
            per_trial_medians.append(np.median(chunk_data))
        print(f"    Mean of means:   {np.mean(per_trial_means):.1f} "
              f"± {np.std(per_trial_means):.1f}")
        print(f"    Mean of medians: {np.mean(per_trial_medians):.1f} "
              f"± {np.std(per_trial_medians):.1f}")
        print(f"    CV (coeff of variation): "
              f"{np.std(per_trial_means)/np.mean(per_trial_means)*100:.0f}%")

    # Now test specific bounds with clamped distance
    print(f"\n\n{'='*70}")
    print(f"  BOUND SWEEP: dim=1024, full-rank ternary, 10 trials")
    print(f"{'='*70}")

    from bab256_engine_v02 import CVPSolver

    dim = 1024
    abs_coeffs_1024 = measure_coefficients(dim, num_trials=10)

    # Use percentile data to pick bounds
    for target_clamp in [25, 20, 15, 10]:
        target_pct = 100 - target_clamp
        bound = int(np.ceil(np.percentile(abs_coeffs_1024, target_pct)))

        # Measure actual clamping and distance across trials
        actual_clamps = []
        dists = []
        for trial in range(10):
            basis_seed = hashlib.sha256(
                b"bound_sweep_" + dim.to_bytes(4, "big")
                + trial.to_bytes(4, "big")
            ).digest()

            basis = np.zeros((dim, dim), dtype=np.int32)
            for i in range(dim):
                bs = hashlib.sha256(basis_seed + i.to_bytes(4, "big")).digest()
                basis[i] = render_ternary(bs, dim)

            target_seed = hashlib.sha256(
                basis_seed + (trial + 7777).to_bytes(8, "big")
            ).digest()
            target = render_target(target_seed, dim)

            coeffs, _, _, _ = np.linalg.lstsq(
                basis.astype(np.float64).T,
                target.astype(np.float64),
                rcond=None,
            )
            int_c = np.round(coeffs).astype(np.int32)
            clamp_pct = np.sum(np.abs(int_c) > bound) / dim * 100
            actual_clamps.append(float(clamp_pct))

            int_c_cl = np.clip(int_c, -bound, bound).astype(np.int32)
            dist = CVPSolver._compute_distance(basis, int_c_cl, target)
            dists.append(dist)

        print(f"  bound={bound:>5d} → clamp={np.mean(actual_clamps):>5.1f}% "
              f"±{np.std(actual_clamps):>4.1f}%  "
              f"dist={np.mean(dists):>8.1f} ±{np.std(dists):>6.1f}")

    print()
