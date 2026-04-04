"""
BAB256 — Full-rank entry range sweep
======================================
Find the basis entry range that gives STABLE coefficient magnitudes
for full-rank systems, then pick the bound for 15-20% clamping.
"""

import hashlib
import numpy as np
import time
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def render_vec(seed, dim, lo, hi):
    vec = np.zeros(dim, dtype=np.int32)
    current = seed
    width = hi - lo + 1
    idx = 0
    while idx < dim:
        current = hashlib.sha256(current).digest()
        for byte_val in current:
            if idx >= dim:
                break
            vec[idx] = (byte_val % width) + lo
            idx += 1
    return vec


def analyze_entry_range(dim, lo, hi, num_trials=10):
    """Analyze coefficient distribution for full-rank basis with given entry range."""
    per_trial_means = []
    per_trial_medians = []
    per_trial_maxes = []
    all_abs = []

    for trial in range(num_trials):
        basis_seed = hashlib.sha256(
            b"entry_" + lo.to_bytes(4, "big", signed=True)
            + hi.to_bytes(4, "big", signed=True)
            + dim.to_bytes(4, "big")
            + trial.to_bytes(4, "big")
        ).digest()

        basis = np.zeros((dim, dim), dtype=np.int32)
        for i in range(dim):
            bs = hashlib.sha256(basis_seed + i.to_bytes(4, "big")).digest()
            basis[i] = render_vec(bs, dim, lo, hi)

        target_seed = hashlib.sha256(
            basis_seed + (trial + 5555).to_bytes(8, "big")
        ).digest()
        target = render_vec(target_seed, dim, 0, 255)

        coeffs, _, _, _ = np.linalg.lstsq(
            basis.astype(np.float64).T,
            target.astype(np.float64),
            rcond=None,
        )
        int_c = np.round(coeffs).astype(np.int32)
        abs_c = np.abs(int_c)

        per_trial_means.append(float(np.mean(abs_c)))
        per_trial_medians.append(float(np.median(abs_c)))
        per_trial_maxes.append(int(np.max(abs_c)))
        all_abs.extend(abs_c.tolist())

    all_abs = np.array(all_abs)
    cv = np.std(per_trial_means) / np.mean(per_trial_means) * 100

    # Find bound for 15% clamping
    p85 = np.percentile(all_abs, 85)
    bound_15 = int(np.ceil(p85))
    p80 = np.percentile(all_abs, 80)
    bound_20 = int(np.ceil(p80))

    return {
        "entry": f"[{lo},{hi}]",
        "mean": np.mean(per_trial_means),
        "median": np.mean(per_trial_medians),
        "max": max(per_trial_maxes),
        "cv": cv,
        "std_means": np.std(per_trial_means),
        "bound_15": bound_15,
        "bound_20": bound_20,
        "p85": p85,
    }


if __name__ == "__main__":
    dim = 1024

    print(f"{'='*80}")
    print(f"  FULL-RANK ENTRY RANGE SWEEP — dim={dim}, {dim} basis vectors")
    print(f"  10 trials each, target [0,255]")
    print(f"{'='*80}\n")

    entry_ranges = [
        (-1, 1),     # ternary
        (-2, 2),     # 5-ary
        (-3, 3),     # 7-ary
        (-5, 5),     # 11-ary
        (-10, 10),   # 21-ary
        (-20, 20),   # 41-ary
        (-50, 50),   # 101-ary
        (0, 255),    # full pixel (original)
    ]

    print(f"  {'Entries':<12} {'mean|c|':>8} {'med|c|':>7} {'max|c|':>7} "
          f"{'CV%':>6} {'B(15%)':>7} {'B(20%)':>7}")
    print(f"  {'-'*60}")

    results = []
    for lo, hi in entry_ranges:
        print(f"  [{lo},{hi}]...", end="", flush=True)
        t0 = time.time()
        r = analyze_entry_range(dim, lo, hi, num_trials=10)
        elapsed = time.time() - t0
        results.append(r)
        print(f"\r  {r['entry']:<12} {r['mean']:>8.1f} {r['median']:>7.1f} "
              f"{r['max']:>7d} {r['cv']:>5.0f}% "
              f"{r['bound_15']:>7d} {r['bound_20']:>7d}")

    # Find the sweet spot: low CV AND reasonable bound
    print(f"\n  Sweet spot criteria: CV < 30%, bound < 50")
    sweet = [r for r in results if r["cv"] < 30 and r["bound_15"] < 50]
    if sweet:
        for r in sweet:
            print(f"    {r['entry']}: CV={r['cv']:.0f}%, "
                  f"bound(15%)={r['bound_15']}, "
                  f"bound(20%)={r['bound_20']}")

    # Deep dive on promising candidates
    print(f"\n\n{'='*80}")
    print(f"  PER-TRIAL STABILITY CHECK — promising entry ranges")
    print(f"{'='*80}")

    for lo, hi in [(-5, 5), (-10, 10), (-20, 20)]:
        print(f"\n  Entry range [{lo},{hi}]:")
        for trial in range(10):
            basis_seed = hashlib.sha256(
                b"entry_" + lo.to_bytes(4, "big", signed=True)
                + hi.to_bytes(4, "big", signed=True)
                + dim.to_bytes(4, "big")
                + trial.to_bytes(4, "big")
            ).digest()

            basis = np.zeros((dim, dim), dtype=np.int32)
            for i in range(dim):
                bs = hashlib.sha256(basis_seed + i.to_bytes(4, "big")).digest()
                basis[i] = render_vec(bs, dim, lo, hi)

            target_seed = hashlib.sha256(
                basis_seed + (trial + 5555).to_bytes(8, "big")
            ).digest()
            target = render_vec(target_seed, dim, 0, 255)

            coeffs, _, _, _ = np.linalg.lstsq(
                basis.astype(np.float64).T,
                target.astype(np.float64),
                rcond=None,
            )
            int_c = np.round(coeffs).astype(np.int32)
            abs_c = np.abs(int_c)
            p85 = np.percentile(abs_c, 85)

            # Clamping at the suggested bound
            bound = int(np.ceil(np.percentile(
                np.array(results[[r["entry"] for r in results].index(f"[{lo},{hi}]")
                         if f"[{lo},{hi}]" in [r["entry"] for r in results] else 0]["p85"]
                         for _ in [None]), 50)))  # just use overall p85 as bound
            # Actually just use a fixed bound based on the sweep
            for r in results:
                if r["entry"] == f"[{lo},{hi}]":
                    bound = r["bound_15"]
                    break

            clamp = np.sum(abs_c > bound) / dim * 100
            print(f"    trial {trial}: mean|c|={np.mean(abs_c):>6.1f}, "
                  f"med={np.median(abs_c):>5.0f}, max={np.max(abs_c):>5d}, "
                  f"clamp@{bound}={clamp:>5.1f}%")

    print()
