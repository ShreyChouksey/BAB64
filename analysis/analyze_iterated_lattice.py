"""
Iterated Lattice Transform — v0.4 Design Analysis
===================================================
Standalone analysis of the proposed 16-round iterated scheme:

  For each round r:
    1. Babai solve CVP(A_r, t_r) → coefficients s_r, residual e_r
    2. S-box:   apply AES S-box byte-wise to |residual| mod 256
    3. Permute: shuffle residual pixels by permutation derived from s_r
    4. XOR:     t_{r+1} = t_r ^ sbox_permuted_residual
    5. Basis:   A_{r+1} = transform(A_r, s_r)

Tests:
  A. AVALANCHE — is round 2's target statistically independent
     from round 1's target?
  B. CORRELATION — does solving round 1 give any advantage
     in predicting/solving round 2?

Uses n=64, S=50, bound=5, noise {-1,0,1}.  Prints numbers only.
"""

import hashlib
import numpy as np
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from bab256_engine_v02 import BAB256Config, BabelRenderer, CVPSolver

# ── AES S-Box (Rijndael) ──────────────────────────────────────────
AES_SBOX = np.array([
    0x63,0x7c,0x77,0x7b,0xf2,0x6b,0x6f,0xc5,0x30,0x01,0x67,0x2b,0xfe,0xd7,0xab,0x76,
    0xca,0x82,0xc9,0x7d,0xfa,0x59,0x47,0xf0,0xad,0xd4,0xa2,0xaf,0x9c,0xa4,0x72,0xc0,
    0xb7,0xfd,0x93,0x26,0x36,0x3f,0xf7,0xcc,0x34,0xa5,0xe5,0xf1,0x71,0xd8,0x31,0x15,
    0x04,0xc7,0x23,0xc3,0x18,0x96,0x05,0x9a,0x07,0x12,0x80,0xe2,0xeb,0x27,0xb2,0x75,
    0x09,0x83,0x2c,0x1a,0x1b,0x6e,0x5a,0xa0,0x52,0x3b,0xd6,0xb3,0x29,0xe3,0x2f,0x84,
    0x53,0xd1,0x00,0xed,0x20,0xfc,0xb1,0x5b,0x6a,0xcb,0xbe,0x39,0x4a,0x4c,0x58,0xcf,
    0xd0,0xef,0xaa,0xfb,0x43,0x4d,0x33,0x85,0x45,0xf9,0x02,0x7f,0x50,0x3c,0x9f,0xa8,
    0x51,0xa3,0x40,0x8f,0x92,0x9d,0x38,0xf5,0xbc,0xb6,0xda,0x21,0x10,0xff,0xf3,0xd2,
    0xcd,0x0c,0x13,0xec,0x5f,0x97,0x44,0x17,0xc4,0xa7,0x7e,0x3d,0x64,0x5d,0x19,0x73,
    0x60,0x81,0x4f,0xdc,0x22,0x2a,0x90,0x88,0x46,0xee,0xb8,0x14,0xde,0x5e,0x0b,0xdb,
    0xe0,0x32,0x3a,0x0a,0x49,0x06,0x24,0x5c,0xc2,0xd3,0xac,0x62,0x91,0x95,0xe4,0x79,
    0xe7,0xc8,0x37,0x6d,0x8d,0xd5,0x4e,0xa9,0x6c,0x56,0xf4,0xea,0x65,0x7a,0xae,0x08,
    0xba,0x78,0x25,0x2e,0x1c,0xa6,0xb4,0xc6,0xe8,0xdd,0x74,0x1f,0x4b,0xbd,0x8b,0x8a,
    0x70,0x3e,0xb5,0x66,0x48,0x03,0xf6,0x0e,0x61,0x35,0x57,0xb9,0x86,0xc1,0x1d,0x9e,
    0xe1,0xf8,0x98,0x11,0x69,0xd9,0x8e,0x94,0x9b,0x1e,0x87,0xe9,0xce,0x55,0x28,0xdf,
    0x8c,0xa1,0x89,0x0d,0xbf,0xe6,0x42,0x68,0x41,0x99,0x2d,0x0f,0xb0,0x54,0xbb,0x16,
], dtype=np.uint8)


# ── Config ─────────────────────────────────────────────────────────
N = 64
BOUND = 5
SCALE = 50
NUM_TRIALS = 50

config = BAB256Config(
    image_width=8, image_height=8,
    coefficient_bound=BOUND,
    basis_scale=SCALE,
    basis_noise_range=(-1, 1),
)
renderer = BabelRenderer(config)


# ── Helpers ────────────────────────────────────────────────────────

def make_basis(seed, n=N, scale=SCALE):
    """Structured LWE basis A = S*I + E."""
    A = np.zeros((n, n), dtype=np.int32)
    for i in range(n):
        row_seed = hashlib.sha256(seed + i.to_bytes(4, 'big')).digest()
        idx = 0
        current = row_seed
        while idx < n:
            current = hashlib.sha256(current).digest()
            for b in current:
                if idx >= n:
                    break
                A[i][idx] = (b % 3) - 1  # {-1, 0, 1}
                idx += 1
        A[i][i] += scale
    return A


def babai_solve(A, target, bound=BOUND):
    """Babai rounding → clamp.  Returns (coeffs, residual_vec)."""
    real_coeffs, _, _, _ = np.linalg.lstsq(
        A.astype(np.float64).T, target.astype(np.float64), rcond=None
    )
    coeffs = np.clip(np.round(real_coeffs).astype(np.int32), -bound, bound)
    lattice_pt = np.dot(coeffs.astype(np.int64), A.astype(np.int64))
    residual = (target.astype(np.int64) - lattice_pt).astype(np.int64)
    return coeffs, residual


def diffuse_residual(residual, coeffs):
    """Hash-based diffusion: SHA-256 chain seeded by (residual || coeffs).

    Converts the raw residual into a pseudorandom byte stream before
    S-box, so that ANY change in residual (even 1 bit) avalanches
    across the entire output.  This is the fix for the 3% sensitivity.
    """
    # Seed = H(residual_bytes || coeff_bytes)
    res_bytes = residual.astype(np.int32).tobytes()
    coeff_bytes = coeffs.astype(np.int32).tobytes()
    seed = hashlib.sha256(res_bytes + coeff_bytes).digest()

    # Expand via SHA-256 chain to fill n bytes
    n = len(residual)
    out = np.zeros(n, dtype=np.uint8)
    idx = 0
    current = seed
    while idx < n:
        current = hashlib.sha256(current).digest()
        for b in current:
            if idx >= n:
                break
            out[idx] = b
            idx += 1
    return out


def sbox_transform(diffused_bytes):
    """Apply AES S-box byte-wise to diffused byte stream."""
    return AES_SBOX[diffused_bytes].astype(np.int32)


def permute_by_coeffs(vec, coeffs):
    """Deterministic permutation from coefficient vector."""
    perm_seed = hashlib.sha256(coeffs.astype(np.int32).tobytes()).digest()
    rng = np.random.RandomState(
        int.from_bytes(perm_seed[:4], 'big') % (2**31)
    )
    perm = rng.permutation(len(vec))
    return vec[perm]


def transform_basis(A, coeffs):
    """Transform basis for next round using coefficients.
    A' = A + outer(sbox(coeffs), row_mix), keeping entries int32.
    Deterministic mixing derived from coefficient hash.
    """
    mix_seed = hashlib.sha256(coeffs.tobytes() + b'basis_mix').digest()
    n = A.shape[0]
    # Generate a sparse perturbation: flip noise on a few rows
    rng = np.random.RandomState(
        int.from_bytes(mix_seed[:4], 'big') % (2**31)
    )
    A_new = A.copy()
    # Perturb ~25% of rows by adding a noise vector scaled by coeff sign
    rows_to_perturb = rng.choice(n, size=n // 4, replace=False)
    for row in rows_to_perturb:
        noise_seed = hashlib.sha256(
            mix_seed + int(row).to_bytes(4, 'big')
        ).digest()
        noise_idx = 0
        noise_current = noise_seed
        noise_vec = np.zeros(n, dtype=np.int32)
        while noise_idx < n:
            noise_current = hashlib.sha256(noise_current).digest()
            for b in noise_current:
                if noise_idx >= n:
                    break
                noise_vec[noise_idx] = (b % 3) - 1
                noise_idx += 1
        A_new[row] += noise_vec
    return A_new


def one_round(A, target, bound=BOUND):
    """Execute one round of the iterated lattice transform.
    Returns (A_next, target_next, coeffs, residual, distance).

    Pipeline:  Babai → diffuse(residual, coeffs) → S-box → permute → XOR
    The diffuse step hashes (residual || coeffs) into a pseudorandom
    byte stream, giving full avalanche before the S-box.
    """
    coeffs, residual = babai_solve(A, target, bound)
    distance = float(np.sqrt(np.sum(residual ** 2)))

    # Diffuse: hash residual+coeffs into pseudorandom bytes
    diffused = diffuse_residual(residual, coeffs)

    # S-box on diffused bytes
    sboxed = sbox_transform(diffused)

    # Permute by coefficients
    permuted = permute_by_coeffs(sboxed, coeffs)

    # XOR into target for next round (mod 256 to keep in byte range)
    target_next = ((target.astype(np.int32)) ^ permuted) % 256
    target_next = target_next.astype(np.int32)

    # Transform basis
    A_next = transform_basis(A, coeffs)

    return A_next, target_next, coeffs, residual, distance


# ── Main Analysis ──────────────────────────────────────────────────

print(f"\n{'='*70}")
print(f"  ITERATED LATTICE TRANSFORM — v0.4 DESIGN ANALYSIS")
print(f"  n={N}, S={SCALE}, bound={BOUND}, {NUM_TRIALS} trials")
print(f"{'='*70}")

# =====================================================================
# TEST A: AVALANCHE — t_1 vs t_2 statistical independence
# =====================================================================
print(f"\n  ── A. AVALANCHE (target independence across rounds) ──\n")

bit_diffs = []            # fraction of differing bytes, t1 vs t2
pixel_correlations = []   # Pearson correlation, t1 vs t2
chi2_pvals = []           # chi-squared uniformity of t2

for trial in range(NUM_TRIALS):
    seed = hashlib.sha256(b"avalanche_v04_" + trial.to_bytes(4, 'big')).digest()
    A = make_basis(seed)
    t1 = renderer.render(hashlib.sha256(seed + b"target").digest())

    _, t2, _, _, _ = one_round(A, t1)

    # Byte-level difference
    diff_frac = np.mean(t1 != t2)
    bit_diffs.append(diff_frac)

    # Pearson correlation
    if np.std(t1) > 0 and np.std(t2) > 0:
        corr = np.corrcoef(t1.astype(float), t2.astype(float))[0, 1]
    else:
        corr = 0.0
    pixel_correlations.append(corr)

print(f"  {'Metric':>35} | {'Mean':>8} | {'Std':>8} | {'Min':>8} | {'Max':>8}")
print(f"  {'-'*75}")
print(f"  {'Pixels differing (t1 vs t2)':>35} | "
      f"{np.mean(bit_diffs):>7.3f} | {np.std(bit_diffs):>8.4f} | "
      f"{np.min(bit_diffs):>8.3f} | {np.max(bit_diffs):>8.3f}")
print(f"  {'Pearson corr(t1, t2)':>35} | "
      f"{np.mean(pixel_correlations):>8.4f} | {np.std(pixel_correlations):>8.4f} | "
      f"{np.min(pixel_correlations):>8.4f} | {np.max(pixel_correlations):>8.4f}")

# Ideal: diff ~99%+ (XOR with S-box output), corr ~0
print(f"\n  Target: diff > 0.95, |corr| < 0.10")
avg_diff = np.mean(bit_diffs)
avg_corr_abs = np.mean(np.abs(pixel_correlations))
print(f"  Result: diff = {avg_diff:.3f} {'PASS' if avg_diff > 0.95 else 'FAIL'}, "
      f"|corr| = {avg_corr_abs:.4f} {'PASS' if avg_corr_abs < 0.10 else 'FAIL'}")

# ── 1-bit sensitivity: flip 1 bit in t1, measure change in t2 ──
print(f"\n  ── A2. ONE-BIT SENSITIVITY (flip 1 pixel in t1) ──\n")
one_bit_diffs = []
for trial in range(NUM_TRIALS):
    seed = hashlib.sha256(b"onebit_v04_" + trial.to_bytes(4, 'big')).digest()
    A = make_basis(seed)
    t1 = renderer.render(hashlib.sha256(seed + b"target").digest())

    # Original round
    _, t2_orig, _, _, _ = one_round(A, t1)

    # Flip one pixel value by ±1
    t1_flip = t1.copy()
    t1_flip[0] = (t1_flip[0] + 1) % 256

    _, t2_flip, _, _, _ = one_round(A, t1_flip)

    diff_frac = np.mean(t2_orig != t2_flip)
    one_bit_diffs.append(diff_frac)

print(f"  1-pixel flip in t1 → {np.mean(one_bit_diffs)*100:.1f}% "
      f"± {np.std(one_bit_diffs)*100:.1f}% pixels change in t2")
print(f"  Target: > 40% (full avalanche)")
print(f"  Result: {'PASS' if np.mean(one_bit_diffs) > 0.40 else 'FAIL'}")


# =====================================================================
# TEST B: CORRELATION — does round 1 solution help round 2?
# =====================================================================
print(f"\n  ── B. CORRELATION (round 1 solution → round 2 advantage) ──\n")

r1_dists = []
r2_dists = []
coeff_corrs = []     # correlation between s1 and s2
r2_with_r1_hint = [] # distance if you use s1 as starting point for r2
r2_fresh_dists = []  # distance from fresh Babai on r2

for trial in range(NUM_TRIALS):
    seed = hashlib.sha256(b"corr_v04_" + trial.to_bytes(4, 'big')).digest()
    A1 = make_basis(seed)
    t1 = renderer.render(hashlib.sha256(seed + b"target").digest())

    # Round 1
    A2, t2, s1, res1, d1 = one_round(A1, t1)
    r1_dists.append(d1)

    # Round 2: fresh Babai solve
    s2, res2 = babai_solve(A2, t2)
    d2 = float(np.sqrt(np.sum(res2 ** 2)))
    r2_fresh_dists.append(d2)
    r2_dists.append(d2)

    # Correlation between coefficient vectors
    if np.std(s1.astype(float)) > 0 and np.std(s2.astype(float)) > 0:
        cc = np.corrcoef(s1.astype(float), s2.astype(float))[0, 1]
    else:
        cc = 0.0
    coeff_corrs.append(cc)

    # "Hint" attack: use s1 directly as guess for round 2
    lp_hint = np.dot(s1.astype(np.int64), A2.astype(np.int64))
    d_hint = float(np.sqrt(np.sum((t2.astype(np.int64) - lp_hint) ** 2)))
    r2_with_r1_hint.append(d_hint)

print(f"  {'Metric':>35} | {'Mean':>10} | {'Std':>8}")
print(f"  {'-'*60}")
print(f"  {'Round 1 CVP distance':>35} | {np.mean(r1_dists):>10.1f} | {np.std(r1_dists):>8.1f}")
print(f"  {'Round 2 fresh Babai distance':>35} | {np.mean(r2_fresh_dists):>10.1f} | {np.std(r2_fresh_dists):>8.1f}")
print(f"  {'Round 2 using s1 as hint':>35} | {np.mean(r2_with_r1_hint):>10.1f} | {np.std(r2_with_r1_hint):>8.1f}")
print(f"  {'corr(s1, s2)':>35} | {np.mean(coeff_corrs):>10.4f} | {np.std(coeff_corrs):>8.4f}")

# The hint attack should be no better than random
avg_hint = np.mean(r2_with_r1_hint)
avg_fresh = np.mean(r2_fresh_dists)
print(f"\n  Hint attack vs fresh Babai: {avg_hint:.1f} vs {avg_fresh:.1f} "
      f"(ratio: {avg_hint/avg_fresh:.2f}x)")
print(f"  Target: hint/fresh ratio > 5x (hint useless)")
print(f"  Result: {'PASS' if avg_hint/avg_fresh > 5 else 'FAIL'}")

print(f"\n  Coefficient correlation |corr(s1,s2)|:")
print(f"  Mean = {np.mean(np.abs(coeff_corrs)):.4f}, "
      f"Max = {np.max(np.abs(coeff_corrs)):.4f}")
print(f"  Target: |corr| < 0.15")
print(f"  Result: {'PASS' if np.mean(np.abs(coeff_corrs)) < 0.15 else 'FAIL'}")


# =====================================================================
# TEST C: MULTI-ROUND DISTANCE EVOLUTION
# =====================================================================
print(f"\n  ── C. DISTANCE EVOLUTION OVER 16 ROUNDS ──\n")

NUM_ROUND_TRIALS = 10
ROUNDS = 16
round_dists = [[] for _ in range(ROUNDS)]

for trial in range(NUM_ROUND_TRIALS):
    seed = hashlib.sha256(b"multiround_v04_" + trial.to_bytes(4, 'big')).digest()
    A = make_basis(seed)
    t = renderer.render(hashlib.sha256(seed + b"target").digest())

    for r in range(ROUNDS):
        A, t, s, res, d = one_round(A, t)
        round_dists[r].append(d)

print(f"  {'Round':>7} | {'Mean Dist':>10} | {'Std':>8} | {'Min':>8} | {'Max':>8}")
print(f"  {'-'*50}")
for r in range(ROUNDS):
    ds = round_dists[r]
    print(f"  {r+1:>7d} | {np.mean(ds):>10.1f} | {np.std(ds):>8.1f} | "
          f"{np.min(ds):>8.1f} | {np.max(ds):>8.1f}")

# Check: distances shouldn't collapse to 0 or explode
final_mean = np.mean(round_dists[-1])
first_mean = np.mean(round_dists[0])
print(f"\n  Distance ratio (round 16 / round 1): {final_mean/first_mean:.2f}x")
print(f"  Target: 0.3x — 3.0x (stable, no collapse or explosion)")
stable = 0.3 < final_mean/first_mean < 3.0
print(f"  Result: {'PASS' if stable else 'FAIL'}")


# =====================================================================
# TEST D: BASIS CONDITION NUMBER EVOLUTION
# =====================================================================
print(f"\n  ── D. BASIS HEALTH OVER 16 ROUNDS ──\n")

cond_numbers = [[] for _ in range(ROUNDS)]
ranks = [[] for _ in range(ROUNDS)]

for trial in range(NUM_ROUND_TRIALS):
    seed = hashlib.sha256(b"basis_health_v04_" + trial.to_bytes(4, 'big')).digest()
    A = make_basis(seed)
    t = renderer.render(hashlib.sha256(seed + b"target").digest())

    for r in range(ROUNDS):
        A, t, s, res, d = one_round(A, t)
        cond = np.linalg.cond(A.astype(np.float64))
        rnk = np.linalg.matrix_rank(A.astype(np.float64))
        cond_numbers[r].append(cond)
        ranks[r].append(rnk)

print(f"  {'Round':>7} | {'Cond #':>10} | {'Rank':>6}")
print(f"  {'-'*30}")
for r in range(ROUNDS):
    print(f"  {r+1:>7d} | {np.mean(cond_numbers[r]):>10.2f} | "
          f"{np.mean(ranks[r]):>6.1f}/{N}")

final_cond = np.mean(cond_numbers[-1])
all_full_rank = all(np.mean(ranks[r]) == N for r in range(ROUNDS))
print(f"\n  Final condition #: {final_cond:.2f} (started at ~1.45)")
print(f"  All rounds full rank: {all_full_rank}")
print(f"  Target: cond < 100, always full rank")
print(f"  Result: {'PASS' if final_cond < 100 and all_full_rank else 'FAIL'}")


# =====================================================================
# SUMMARY
# =====================================================================
print(f"\n{'='*70}")
print(f"  SUMMARY")
print(f"{'='*70}")
tests = [
    ("A. Avalanche (pixel diff)", avg_diff > 0.95),
    ("A. Avalanche (correlation)", avg_corr_abs < 0.10),
    ("A2. 1-bit sensitivity", np.mean(one_bit_diffs) > 0.40),
    ("B. Hint attack useless", avg_hint/avg_fresh > 5),
    ("B. Coeff independence", np.mean(np.abs(coeff_corrs)) < 0.15),
    ("C. Distance stability", 0.3 < final_mean/first_mean < 3.0),
    ("D. Basis health", final_cond < 100 and all_full_rank),
]
for name, passed in tests:
    print(f"  {'PASS' if passed else 'FAIL'}  {name}")
print(f"\n  {sum(p for _,p in tests)}/{len(tests)} passed")
print(f"{'='*70}\n")
