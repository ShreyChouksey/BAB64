"""
BAB64 Stress Test — Three Attack Simulations
=============================================
Adversarial analysis of BAB64's image-dependent hash construction.
Tests S-box quality, round reduction resilience, and self-referential shortcuts.
"""

import hashlib
import numpy as np
import time
import sys
from collections import Counter
from bab64_engine import BAB64Config, BabelRenderer, ImageHash


# =============================================================================
# UTILITIES
# =============================================================================

def random_image(config, idx):
    """Generate a random 64x64 image from an index."""
    seed = hashlib.sha256(f"stress_{idx}".encode()).digest()
    renderer = BabelRenderer(config)
    return renderer.render(seed)


def bits_of(hash_bytes):
    """Convert hash bytes to integer for bit ops."""
    return int.from_bytes(hash_bytes, 'big')


def count_differing_bits(h1, h2):
    return bin(bits_of(h1) ^ bits_of(h2)).count('1')


# =============================================================================
# ATTACK 1 — WEAK S-BOX MINING
# =============================================================================

def walsh_hadamard_transform(f, n=8):
    """Compute WHT of a Boolean function f: {0,1}^n -> {-1,+1}."""
    size = 1 << n
    # Convert Boolean function to +1/-1
    table = np.array([1 - 2 * f[i] for i in range(size)], dtype=np.int64)
    # Fast Walsh-Hadamard
    h = 1
    while h < size:
        for i in range(0, size, h * 2):
            for j in range(i, i + h):
                x = table[j]
                y = table[j + h]
                table[j] = x + y
                table[j + h] = x - y
        h *= 2
    return table


def sbox_nonlinearity(sbox):
    """
    Compute nonlinearity of an 8-bit S-box.
    NL = min over all component functions of (2^(n-1) - max|WHT|/2).
    """
    n = 8
    size = 256
    min_nl = size  # will be reduced

    for mask in range(1, size):  # each non-zero output mask = component function
        # Component Boolean function: dot(mask, sbox[x]) mod 2
        f = np.zeros(size, dtype=np.int32)
        for x in range(size):
            f[x] = bin(mask & sbox[x]).count('1') % 2
        wht = walsh_hadamard_transform(f, n)
        max_wht = np.max(np.abs(wht))
        nl = (1 << (n - 1)) - max_wht // 2
        if nl < min_nl:
            min_nl = nl
    return min_nl


def sbox_differential_uniformity(sbox):
    """Compute max entry in the DDT (differential distribution table)."""
    size = 256
    max_count = 0
    for dx in range(1, size):  # non-zero input difference
        counts = np.zeros(size, dtype=np.int32)
        for x in range(size):
            dy = sbox[x] ^ sbox[x ^ dx]
            counts[dy] += 1
        c = np.max(counts)
        if c > max_count:
            max_count = c
    return max_count


def attack_weak_sbox(num_images=10000):
    print(f"\n{'='*70}")
    print(f"  ATTACK 1 — WEAK S-BOX MINING ({num_images:,} images)")
    print(f"{'='*70}")

    config = BAB64Config()
    hasher = ImageHash(config)

    fixed_points_list = []
    worst_fp = (0, -1)  # (count, index)
    identity_like = []

    t0 = time.time()
    for i in range(num_images):
        image = random_image(config, i)
        sbox = hasher._derive_sbox(image)

        # Fixed points
        fp = sum(1 for x in range(256) if sbox[x] == x)
        fixed_points_list.append(fp)
        if fp > worst_fp[0]:
            worst_fp = (fp, i)

        # Near-identity: more than 128 fixed points
        if fp > 128:
            identity_like.append((i, fp))

        if (i + 1) % 2000 == 0:
            elapsed = time.time() - t0
            print(f"    ... {i+1:>6,} images scanned ({elapsed:.1f}s)")

    elapsed = time.time() - t0
    fp_arr = np.array(fixed_points_list)

    print(f"\n  Fixed Points (sbox[x] == x):")
    print(f"    Mean:    {fp_arr.mean():.2f}  (expected ~1.0 for random permutation)")
    print(f"    Std:     {fp_arr.std():.2f}")
    print(f"    Min:     {fp_arr.min()}")
    print(f"    Max:     {fp_arr.max()}  (image #{worst_fp[1]})")
    print(f"    P(0 FP): {np.sum(fp_arr == 0) / len(fp_arr):.3f}  (expected ~0.368)")

    if identity_like:
        print(f"\n  WARNING: {len(identity_like)} near-identity S-boxes found!")
        for idx, fp in identity_like[:5]:
            print(f"    Image #{idx}: {fp} fixed points")
    else:
        print(f"\n  No near-identity S-boxes found (max FP = {fp_arr.max()})")

    # Deep analysis on a sample: nonlinearity + differential uniformity
    sample_size = 50
    print(f"\n  Deep cryptographic analysis on {sample_size} random S-boxes...")
    print(f"  (Computing Walsh transform + DDT — this takes a moment)")

    nl_list = []
    du_list = []
    deep_t0 = time.time()

    # Pick evenly spaced samples
    indices = np.linspace(0, num_images - 1, sample_size, dtype=int)
    for count, i in enumerate(indices):
        image = random_image(config, int(i))
        sbox = hasher._derive_sbox(image)

        nl = sbox_nonlinearity(sbox)
        du = sbox_differential_uniformity(sbox)
        nl_list.append(nl)
        du_list.append(du)

        if (count + 1) % 10 == 0:
            print(f"    ... {count+1}/{sample_size} deep-analyzed "
                  f"({time.time() - deep_t0:.1f}s)")

    nl_arr = np.array(nl_list)
    du_arr = np.array(du_list)

    # AES S-box: NL=112, DU=4. Random permutation: NL~94-100, DU~8-12.
    print(f"\n  Nonlinearity (higher = better, AES=112, random perm ~94-100):")
    print(f"    Mean:  {nl_arr.mean():.1f}")
    print(f"    Min:   {nl_arr.min()}   {'<-- WEAK' if nl_arr.min() < 80 else ''}")
    print(f"    Max:   {nl_arr.max()}")

    print(f"\n  Differential Uniformity (lower = better, AES=4, random perm ~8-12):")
    print(f"    Mean:  {du_arr.mean():.1f}")
    print(f"    Min:   {du_arr.min()}")
    print(f"    Max:   {du_arr.max()}   {'<-- WEAK' if du_arr.max() > 20 else ''}")

    # Verdict
    weak_nl = nl_arr.min() < 70
    weak_du = du_arr.max() > 24
    weak_fp = fp_arr.max() > 10

    if weak_nl or weak_du or weak_fp:
        print(f"\n  VERDICT: POTENTIAL WEAKNESS FOUND")
        if weak_nl:
            print(f"    - Low nonlinearity ({nl_arr.min()}) enables linear approximation attacks")
        if weak_du:
            print(f"    - High diff uniformity ({du_arr.max()}) enables differential attacks")
        if weak_fp:
            print(f"    - Excessive fixed points ({fp_arr.max()}) reduces permutation quality")
    else:
        print(f"\n  VERDICT: S-boxes appear cryptographically adequate")
        print(f"    - Nonlinearity consistent with random permutations")
        print(f"    - Differential uniformity within expected bounds")
        print(f"    - Fixed point distribution matches derangement theory")

    print(f"\n  Total time: {time.time() - t0:.1f}s")


# =============================================================================
# ATTACK 2 — ROUND REDUCTION
# =============================================================================

def hash_with_rounds(hasher, image, num_rounds):
    """Hash an image with a modified round count."""
    # Derive params normally
    rc = hasher._derive_round_constants(image)
    rot = hasher._derive_rotations(image)
    sbox = hasher._derive_sbox(image)
    state = hasher._derive_initial_state(image)

    # Run compression with reduced rounds
    original_rounds = hasher.config.num_rounds
    hasher.config.num_rounds = num_rounds

    image_bytes = image.tobytes()
    block_size = hasher.config.block_size
    num_blocks = (len(image_bytes) + block_size - 1) // block_size

    for b in range(num_blocks):
        start = b * block_size
        block = image_bytes[start:start + block_size]
        if len(block) < block_size:
            block = block + b'\x00' * (block_size - len(block))
        state = hasher._compress(state, block, rc, rot, sbox, round_offset=b)

    hasher.config.num_rounds = original_rounds

    result = b''
    for word in state:
        result += int(word).to_bytes(4, 'big')
    return result


def attack_round_reduction(num_images=100):
    print(f"\n{'='*70}")
    print(f"  ATTACK 2 — ROUND REDUCTION ANALYSIS ({num_images} images)")
    print(f"{'='*70}")

    config = BAB64Config()
    hasher = ImageHash(config)

    round_counts = [1, 2, 4, 8, 16, 24, 32]
    # avalanche_by_rounds[r] = list of avalanche values
    avalanche_by_rounds = {r: [] for r in round_counts}

    t0 = time.time()
    for i in range(num_images):
        image = random_image(config, i + 50000)  # different seed space
        # Create 1-pixel-flipped variant
        modified = image.copy()
        modified[0] = np.uint8((int(modified[0]) + 1) % 256)

        for r in round_counts:
            h_orig = hash_with_rounds(hasher, image, r)
            h_mod = hash_with_rounds(hasher, modified, r)
            diff = count_differing_bits(h_orig, h_mod)
            avalanche_by_rounds[r].append(diff)

        if (i + 1) % 25 == 0:
            print(f"    ... {i+1}/{num_images} images ({time.time() - t0:.1f}s)")

    # Results table
    print(f"\n  {'Rounds':>8} | {'Avg Bits':>10} | {'Avg %':>8} | "
          f"{'Min %':>8} | {'Max %':>8} | {'Std':>6} | {'Status':>10}")
    print(f"  {'-'*8}-+-{'-'*10}-+-{'-'*8}-+-{'-'*8}-+-{'-'*8}-+-{'-'*6}-+-{'-'*10}")

    min_safe_rounds = None
    for r in round_counts:
        vals = np.array(avalanche_by_rounds[r])
        avg = vals.mean()
        avg_pct = avg / 256 * 100
        min_pct = vals.min() / 256 * 100
        max_pct = vals.max() / 256 * 100
        std = vals.std()

        # "Safe" = average avalanche stays above 45%
        status = "PASS" if avg_pct >= 45 else "FAIL"
        if status == "PASS" and min_safe_rounds is None:
            min_safe_rounds = r

        print(f"  {r:>8} | {avg:>10.1f} | {avg_pct:>7.1f}% | "
              f"{min_pct:>7.1f}% | {max_pct:>7.1f}% | {std:>6.1f} | {status:>10}")

    print()
    if min_safe_rounds is not None:
        print(f"  Minimum rounds for >= 45% avalanche: {min_safe_rounds}")
        if min_safe_rounds <= 4:
            print(f"  NOTE: Avalanche achieved quickly — good diffusion design")
        elif min_safe_rounds <= 8:
            print(f"  NOTE: Reasonable — 8 rounds is the safety margin start")
        else:
            print(f"  WARNING: Needs {min_safe_rounds}+ rounds — margin is thin at 32")
    else:
        print(f"  WARNING: Avalanche never reaches 45% — CRITICAL WEAKNESS")

    # Check: does round 32 meet ideal (128 +/- 20 bits)?
    full_vals = np.array(avalanche_by_rounds[32])
    full_avg = full_vals.mean()
    if abs(full_avg - 128) > 20:
        print(f"\n  CONCERN: Full 32-round avalanche avg = {full_avg:.1f} bits")
        print(f"           Ideal = 128.0 bits (50%). Deviation = {abs(full_avg-128):.1f}")
    else:
        print(f"\n  Full 32-round avalanche: {full_avg:.1f}/256 bits — healthy")

    print(f"\n  Total time: {time.time() - t0:.1f}s")


# =============================================================================
# ATTACK 3 — SELF-REFERENTIAL SHORTCUT
# =============================================================================

def attack_self_referential(num_images=1000):
    print(f"\n{'='*70}")
    print(f"  ATTACK 3 — SELF-REFERENTIAL SHORTCUT ({num_images:,} images)")
    print(f"{'='*70}")

    config = BAB64Config()
    hasher = ImageHash(config)

    t0 = time.time()

    # --- Check 1: Round constant repetition ---
    print(f"\n  [1] Round Constant Repetition")
    repeat_count = 0
    max_repeats = 0
    for i in range(num_images):
        image = random_image(config, i + 100000)
        rc = hasher._derive_round_constants(image)
        unique = len(set(int(c) for c in rc))
        repeats = len(rc) - unique
        if repeats > 0:
            repeat_count += 1
            max_repeats = max(max_repeats, repeats)

    print(f"    Images with repeated round constants: {repeat_count}/{num_images}")
    print(f"    Max repeats in single image: {max_repeats}")
    if repeat_count == 0:
        print(f"    Status: SAFE — round constants are unique per image")
        print(f"    (Expected: SHA-256 of disjoint 128-pixel blocks → collision unlikely)")
    else:
        pct = repeat_count / num_images * 100
        print(f"    Status: {'CONCERN' if pct > 5 else 'ACCEPTABLE'} — "
              f"{pct:.1f}% images have repeats")

    # --- Check 2: Identity / near-identity S-boxes ---
    print(f"\n  [2] Identity / Near-Identity S-boxes")
    identity_threshold = 10  # more than 10 fixed points is suspicious
    suspicious = []
    fp_list = []
    for i in range(num_images):
        image = random_image(config, i + 100000)
        sbox = hasher._derive_sbox(image)
        fp = sum(1 for x in range(256) if sbox[x] == x)
        fp_list.append(fp)
        if fp > identity_threshold:
            suspicious.append((i, fp))

    fp_arr = np.array(fp_list)
    print(f"    Fixed-point distribution: mean={fp_arr.mean():.2f}, "
          f"max={fp_arr.max()}")
    if suspicious:
        print(f"    Suspicious (>{identity_threshold} FP): {len(suspicious)} images")
        for idx, fp in suspicious[:5]:
            print(f"      Image #{idx}: {fp} fixed points")
    else:
        print(f"    Status: SAFE — no S-boxes with >{identity_threshold} fixed points")

    # --- Check 3: Information leakage (pixel position → hash bit) ---
    print(f"\n  [3] Information Leakage: Pixel Position → Hash Output")
    # For a subset of pixel positions, compute correlation between
    # pixel value at that position and each bit of the hash output.
    sample_n = min(500, num_images)
    num_positions = 64  # test 64 evenly-spaced pixel positions
    positions = np.linspace(0, config.dimension - 1, num_positions, dtype=int)

    pixel_vals = np.zeros((sample_n, num_positions), dtype=np.float64)
    hash_bits_matrix = np.zeros((sample_n, 256), dtype=np.float64)

    for i in range(sample_n):
        image = random_image(config, i + 200000)
        h = hasher.hash_image(image)
        h_int = bits_of(h)

        for j, pos in enumerate(positions):
            pixel_vals[i, j] = float(image[pos])

        for b in range(256):
            hash_bits_matrix[i, b] = float((h_int >> b) & 1)

    # Compute max absolute correlation across all (position, bit) pairs
    max_corr = 0.0
    max_corr_pos = (0, 0)
    correlations = []

    for j in range(num_positions):
        pv = pixel_vals[:, j]
        pv_centered = pv - pv.mean()
        pv_std = pv.std()
        if pv_std < 1e-10:
            continue
        for b in range(256):
            hb = hash_bits_matrix[:, b]
            hb_centered = hb - hb.mean()
            hb_std = hb.std()
            if hb_std < 1e-10:
                continue
            corr = abs(np.dot(pv_centered, hb_centered) / (sample_n * pv_std * hb_std))
            correlations.append(corr)
            if corr > max_corr:
                max_corr = corr
                max_corr_pos = (positions[j], b)

    corr_arr = np.array(correlations)
    # For N=500 independent samples, expect |corr| < ~0.09 at 95% confidence
    threshold = 3.0 / np.sqrt(sample_n)  # ~0.134 for 500 samples

    print(f"    Tested {num_positions} pixel positions x 256 hash bits "
          f"= {len(correlations):,} pairs")
    print(f"    Max |correlation|: {max_corr:.4f} "
          f"(pixel {max_corr_pos[0]}, bit {max_corr_pos[1]})")
    print(f"    Mean |correlation|: {corr_arr.mean():.4f}")
    print(f"    Statistical threshold (3/sqrt(N)): {threshold:.4f}")

    high_corr = np.sum(corr_arr > threshold)
    expected_high = len(correlations) * 0.003  # ~0.3% expected above 3-sigma
    print(f"    Pairs above threshold: {high_corr} "
          f"(expected by chance: ~{expected_high:.0f})")

    if max_corr > 0.2:
        print(f"    STATUS: LEAKAGE DETECTED — pixel positions influence hash output")
    elif high_corr > expected_high * 3:
        print(f"    STATUS: SUSPICIOUS — more high correlations than expected")
    else:
        print(f"    STATUS: SAFE — no detectable information leakage")

    # --- Check 4: Pixel mean ↔ hash output correlation ---
    print(f"\n  [4] Correlation: Image Pixel Mean → Hash Output")
    means = np.zeros(sample_n)
    hash_ints = np.zeros(sample_n, dtype=np.float64)

    for i in range(sample_n):
        image = random_image(config, i + 200000)
        h = hasher.hash_image(image)
        means[i] = image.mean()
        # Use first 64 bits of hash as float proxy (avoid overflow)
        hash_ints[i] = float(int.from_bytes(h[:8], 'big'))

    # Pearson correlation
    m_c = means - means.mean()
    h_c = hash_ints - hash_ints.mean()
    m_std = means.std()
    h_std = hash_ints.std()
    if m_std > 1e-10 and h_std > 1e-10:
        pearson = np.dot(m_c, h_c) / (sample_n * m_std * h_std)
    else:
        pearson = 0.0

    print(f"    Pearson r (pixel_mean vs hash[0:64 bits]): {pearson:.4f}")
    print(f"    |r| threshold for concern: {threshold:.4f}")
    if abs(pearson) > threshold:
        print(f"    STATUS: CONCERN — global pixel statistics correlate with hash")
    else:
        print(f"    STATUS: SAFE — no detectable global correlation")

    print(f"\n  Total time: {time.time() - t0:.1f}s")


# =============================================================================
# MAIN
# =============================================================================

if __name__ == "__main__":
    print("""
    ================================================================
      BAB64 STRESS TEST — Adversarial Attack Simulations
    ================================================================
      Attack 1: Weak S-box mining (10,000 images)
      Attack 2: Round reduction analysis (100 images x 7 round counts)
      Attack 3: Self-referential shortcut detection (1,000 images)
    ================================================================
    """)

    total_start = time.time()

    attack_weak_sbox(num_images=10000)
    attack_round_reduction(num_images=100)
    attack_self_referential(num_images=1000)

    total = time.time() - total_start
    print(f"\n{'='*70}")
    print(f"  ALL ATTACKS COMPLETE — Total time: {total:.1f}s")
    print(f"{'='*70}\n")
