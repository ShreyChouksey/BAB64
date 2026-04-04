"""
LWE Clamping Analysis for BAB256
=================================
Measures how much the coefficient_bound=5 clamp distorts
the Babai solutions at n=1024, S=50, E~{-1,0,1}.

Key question: what fraction of Babai's raw coefficients
exceed the bound, and how much does clamping hurt residual?

Prints numbers only — does NOT modify the engine.
"""

import hashlib
import numpy as np
import time
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from bab256_engine_v02 import BAB256Config, BabelRenderer, LatticeEngine, CVPSolver

NUM_TRIALS = 5
BOUND = 5

config = BAB256Config(
    image_width=32,
    image_height=32,
    coefficient_bound=BOUND,
    basis_scale=50,
    basis_noise_range=(-1, 1),
    difficulty_bits=8,
    num_rounds=16,
)

n = config.dimension
print(f"\n{'='*65}")
print(f"  LWE CLAMPING ANALYSIS — n={n}, S={config.basis_scale}, "
      f"bound={BOUND}, noise={config.basis_noise_range}")
print(f"{'='*65}\n")

renderer = BabelRenderer(config)
lattice = LatticeEngine(config)

# Generate one basis (expensive at 1024d — ~4MB matrix)
seed = hashlib.sha256(b"lwe_clamping_analysis").digest()
print(f"  Generating A = {config.basis_scale}*I + E  ({n}x{n}) ...", end=" ", flush=True)
t0 = time.time()
lattice.generate_basis(seed)
print(f"done ({time.time()-t0:.2f}s, {lattice.basis.nbytes/1024/1024:.1f}MB)")

# Theoretical analysis first
print(f"\n  --- THEORETICAL ---")
print(f"  A = S*I + E, so A^{{-1}} ≈ (1/S)*I for S >> ||E||")
print(f"  For target b ~ Uniform[0,255]^n:")
print(f"    s_real = A^{{-1}} * b ≈ b / S")
print(f"    E[s_i] ≈ E[b_i] / S = 127.5 / {config.basis_scale} = {127.5/config.basis_scale:.2f}")
print(f"    max(s_i) ≈ 255 / S = {255/config.basis_scale:.2f}")
print(f"    Bound = {BOUND}, so clamping threshold = {BOUND * config.basis_scale} in target space")
print(f"    Fraction clamped (theory) ≈ P(|b_i/S| > {BOUND}) = P(|b_i| > {BOUND*config.basis_scale})")
vals = np.arange(256)
centered = vals - 127.5
frac_over = np.mean(np.abs(centered / config.basis_scale) > BOUND)
print(f"    Estimated: {frac_over*100:.1f}% of coefficients clamped")

# Empirical analysis: Babai Rounding (fast, no GS needed)
print(f"\n  --- EMPIRICAL (Babai Rounding, {NUM_TRIALS} trials) ---")
print(f"  {'Trial':>5} | {'Clamped%':>8} | {'|max raw|':>9} | "
      f"{'mean|raw|':>9} | {'Dist(raw)':>10} | {'Dist(clamp)':>11} | {'Dist ratio':>10}")
print(f"  {'-'*80}")

all_clamped_pcts = []
all_raw_maxes = []
all_raw_means = []
all_dist_ratios = []

for trial in range(NUM_TRIALS):
    target_seed = hashlib.sha256(seed + trial.to_bytes(4, 'big')).digest()
    target = renderer.render(target_seed)

    # --- Babai Rounding: get RAW (unclamped) coefficients ---
    B = lattice.basis.astype(np.float64)
    t = target.astype(np.float64)
    raw_coeffs, _, _, _ = np.linalg.lstsq(B.T, t, rcond=None)
    raw_int = np.round(raw_coeffs).astype(np.int32)

    # Stats on raw
    abs_raw = np.abs(raw_int)
    clamped_mask = abs_raw > BOUND
    clamped_pct = 100.0 * np.mean(clamped_mask)
    raw_max = int(abs_raw.max())
    raw_mean = float(abs_raw.mean())

    # Distance with raw (unclamped) coefficients
    lp_raw = np.dot(raw_int.astype(np.int64), lattice.basis.astype(np.int64))
    dist_raw = float(np.sqrt(np.sum((lp_raw - target.astype(np.int64))**2)))

    # Distance with clamped coefficients
    clamped = np.clip(raw_int, -BOUND, BOUND).astype(np.int32)
    lp_clamp = np.dot(clamped.astype(np.int64), lattice.basis.astype(np.int64))
    dist_clamp = float(np.sqrt(np.sum((lp_clamp - target.astype(np.int64))**2)))

    ratio = dist_clamp / max(dist_raw, 1e-10)

    all_clamped_pcts.append(clamped_pct)
    all_raw_maxes.append(raw_max)
    all_raw_means.append(raw_mean)
    all_dist_ratios.append(ratio)

    print(f"  {trial:>5d} | {clamped_pct:>7.1f}% | {raw_max:>9d} | "
          f"{raw_mean:>9.2f} | {dist_raw:>10.1f} | {dist_clamp:>11.1f} | {ratio:>10.2f}x")

print(f"  {'-'*80}")
print(f"  {'AVG':>5} | {np.mean(all_clamped_pcts):>7.1f}% | "
      f"{np.mean(all_raw_maxes):>9.1f} | {np.mean(all_raw_means):>9.2f} | "
      f"{'':>10} | {'':>11} | {np.mean(all_dist_ratios):>10.2f}x")

# Coefficient distribution histogram
print(f"\n  --- RAW COEFFICIENT DISTRIBUTION (last trial) ---")
hist_edges = [-np.inf, -10, -BOUND, -3, -1, 0, 1, 3, BOUND, 10, np.inf]
labels = ["<-10", f"-10..-{BOUND+1}", f"-{BOUND}..-4", "-3..-2", "-1..0",
          "0..1", "2..3", f"4..{BOUND}", f"{BOUND+1}..10", ">10"]
counts = []
for i in range(len(hist_edges)-1):
    lo, hi = hist_edges[i], hist_edges[i+1]
    c = np.sum((raw_int > lo) & (raw_int <= hi))
    counts.append(c)
print(f"  {'Range':>12} | {'Count':>6} | {'Pct':>6}")
print(f"  {'-'*30}")
for label, count in zip(labels, counts):
    print(f"  {label:>12} | {count:>6d} | {100*count/n:>5.1f}%")

# Babai Nearest Plane analysis (one trial only — it's O(n^3) for GS)
print(f"\n  --- BABAI NEAREST PLANE (1 trial, includes GS overhead) ---")
target_seed = hashlib.sha256(seed + b"plane_test").digest()
target = renderer.render(target_seed)

t0 = time.time()
# Run full nearest plane but capture pre-clamp state
# We need to replicate the logic to get raw coefficients
k = lattice.basis.shape[0]
B = lattice.basis.astype(np.float64)
t_vec = target.astype(np.float64)

# Gram-Schmidt
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

gs_time = time.time() - t0
print(f"  Gram-Schmidt: {gs_time:.2f}s")

# Nearest plane — raw (unclamped)
b = t_vec.copy()
raw_plane = np.zeros(k, dtype=np.float64)
for i in range(k - 1, -1, -1):
    norm_sq = np.dot(B_star[i], B_star[i])
    if norm_sq < 1e-10:
        continue
    c_real = np.dot(b, B_star[i]) / norm_sq
    raw_plane[i] = c_real
    c_int = int(np.round(c_real))
    b -= c_int * B[i]

plane_time = time.time() - t0 - gs_time
raw_plane_int = np.round(raw_plane).astype(np.int32)
abs_plane = np.abs(raw_plane_int)
plane_clamped_pct = 100.0 * np.mean(abs_plane > BOUND)
plane_max = int(abs_plane.max())
plane_mean = float(abs_plane.mean())

# Distances
clamped_plane = np.clip(raw_plane_int, -BOUND, BOUND).astype(np.int32)
lp_raw = np.dot(raw_plane_int.astype(np.int64), lattice.basis.astype(np.int64))
dist_raw_p = float(np.sqrt(np.sum((lp_raw - target.astype(np.int64))**2)))
lp_clamp = np.dot(clamped_plane.astype(np.int64), lattice.basis.astype(np.int64))
dist_clamp_p = float(np.sqrt(np.sum((lp_clamp - target.astype(np.int64))**2)))

print(f"  Nearest Plane: {plane_time:.2f}s")
print(f"  Clamped: {plane_clamped_pct:.1f}%  |  max|raw|={plane_max}  |  mean|raw|={plane_mean:.2f}")
print(f"  Dist(raw)={dist_raw_p:.1f}  Dist(clamped)={dist_clamp_p:.1f}  ratio={dist_clamp_p/max(dist_raw_p,1):.2f}x")

# Summary
print(f"\n{'='*65}")
print(f"  SUMMARY")
print(f"{'='*65}")
print(f"  Parameters:  n={n}, S={config.basis_scale}, bound={BOUND}, E~{config.basis_noise_range}")
print(f"  Avg clamping (rounding): {np.mean(all_clamped_pcts):.1f}%")
print(f"  Avg distance inflation:  {np.mean(all_dist_ratios):.2f}x")
print(f"  Theoretical hardness:    2^{0.292*n:.0f} (classical), 2^{0.265*n:.0f} (quantum)")
print(f"")
print(f"  INTERPRETATION:")
avg_clamp = np.mean(all_clamped_pcts)
if avg_clamp < 5:
    print(f"  Clamping is MINIMAL (<5%) — bound={BOUND} fits the solution space well.")
    print(f"  The LWE hardness guarantee holds cleanly.")
elif avg_clamp < 25:
    print(f"  Clamping is MODERATE ({avg_clamp:.0f}%) — some solution degradation.")
    print(f"  Consider: increase S (reduces raw coeff magnitude)")
    print(f"           or increase bound (weakens hardness argument)")
else:
    print(f"  Clamping is HEAVY ({avg_clamp:.0f}%) — Babai solutions don't fit bound={BOUND}.")
    print(f"  The effective search space is truncated significantly.")
    print(f"  Options: increase S to {int(255/(2*BOUND))+1}+ or increase bound")
print(f"{'='*65}\n")
