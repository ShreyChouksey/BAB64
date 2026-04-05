"""
Attack 5: Preimage Structure
==============================
Does the self-referential construction leak information that
helps an attacker work backwards?

In a normal hash, the attacker knows NOTHING about the function.
In BAB64, if you know the target hash output, you know the output
came from SOME image that defined SOME hash function. Does this
circular constraint reduce the search space?

Tests:
  1. OUTPUT BIAS — are BAB64 outputs uniformly distributed?
  2. HASH-TO-PARAMETER LEAKAGE — can hash output reveal image properties?
  3. FIXED-POINT SEARCH — are there exploitable hash↔image relationships?
  4. SECOND-PREIMAGE SHORTCUT — do similar parameters → closer outputs?
"""

import hashlib
import numpy as np
import time
from scipy import stats
from bab64_engine import BAB64Config, BabelRenderer, ImageHash
try:
    import bab64_fast
except ImportError:
    pass


# ─────────────────────────────────────────────────────────────
# TEST 1: OUTPUT BIAS
# ─────────────────────────────────────────────────────────────

def test_output_bias(config, renderer, hasher, num_images=10000):
    print("=" * 60)
    print("  TEST 1: OUTPUT BIAS")
    print("  Are BAB64 outputs uniformly distributed?")
    print("  Chi-squared test on top byte vs SHA-256 baseline.")
    print("=" * 60)

    base_seed = hashlib.sha256(b"output_bias_test").digest()

    bab64_top_bytes = []
    sha256_top_bytes = []

    for i in range(num_images):
        image = renderer.render_from_nonce(base_seed, i)

        # BAB64 self-referential hash
        h = hasher.hash_image(image)
        bab64_top_bytes.append(h[0])

        # SHA-256 of the same image (baseline)
        sha_h = hashlib.sha256(image.tobytes()).digest()
        sha256_top_bytes.append(sha_h[0])

    # Chi-squared test: observed frequency vs uniform expectation
    # Bin into 256 buckets (one per byte value)
    bab64_counts = np.bincount(bab64_top_bytes, minlength=256)
    sha256_counts = np.bincount(sha256_top_bytes, minlength=256)
    expected = num_images / 256.0

    chi2_bab64 = np.sum((bab64_counts - expected) ** 2 / expected)
    chi2_sha256 = np.sum((sha256_counts - expected) ** 2 / expected)

    # Degrees of freedom = 255 (256 bins - 1)
    # Critical value at p=0.01 for df=255 is ~310
    # p-value from survival function
    p_bab64 = stats.chi2.sf(chi2_bab64, df=255)
    p_sha256 = stats.chi2.sf(chi2_sha256, df=255)

    # BAB64 passes if its distribution is not significantly worse than uniform
    passed = p_bab64 > 0.01

    print(f"\n  Images hashed:              {num_images}")
    print(f"  Expected count per bin:     {expected:.1f}")
    print(f"")
    print(f"  BAB64  chi-squared:         {chi2_bab64:.1f}  (p = {p_bab64:.4f})")
    print(f"  SHA256 chi-squared:         {chi2_sha256:.1f}  (p = {p_sha256:.4f})")
    print(f"  Critical value (p=0.01):    ~310")
    print(f"")
    print(f"  BAB64 uniform?              {'YES' if passed else 'NO — output bias detected'}")
    print(f"  Result:                     {'PASS' if passed else 'FAIL'}")
    print()
    return passed


# ─────────────────────────────────────────────────────────────
# TEST 2: HASH-TO-PARAMETER LEAKAGE
# ─────────────────────────────────────────────────────────────

def test_hash_to_parameter_leakage(config, renderer, hasher, num_images=5000):
    print("=" * 60)
    print("  TEST 2: HASH-TO-PARAMETER LEAKAGE")
    print("  Can hash output bytes reveal image properties?")
    print("  Correlation between hash bytes and image statistics.")
    print("=" * 60)

    base_seed = hashlib.sha256(b"leakage_test").digest()

    # Collect hash bytes and image statistics
    hash_bytes_matrix = []   # num_images × 32
    pixel_means = []
    pixel_vars = []
    first_pixels = []
    last_pixels = []

    for i in range(num_images):
        image = renderer.render_from_nonce(base_seed, i)
        h = hasher.hash_image(image)

        hash_bytes_matrix.append(list(h))
        pixel_means.append(float(np.mean(image)))
        pixel_vars.append(float(np.var(image)))
        first_pixels.append(float(image[0]))
        last_pixels.append(float(image[-1]))

    hash_bytes_matrix = np.array(hash_bytes_matrix, dtype=np.float64)
    pixel_means = np.array(pixel_means)
    pixel_vars = np.array(pixel_vars)
    first_pixels = np.array(first_pixels)
    last_pixels = np.array(last_pixels)

    properties = [
        ("pixel mean", pixel_means),
        ("pixel variance", pixel_vars),
        ("first pixel", first_pixels),
        ("last pixel", last_pixels),
    ]

    max_correlations = {}
    any_leak = False

    for prop_name, prop_values in properties:
        max_r = 0.0
        max_byte = -1
        for byte_idx in range(32):
            col = hash_bytes_matrix[:, byte_idx]
            if np.std(col) > 0 and np.std(prop_values) > 0:
                r, _ = stats.pearsonr(col, prop_values)
                if abs(r) > abs(max_r):
                    max_r = r
                    max_byte = byte_idx

        leaked = abs(max_r) > 0.05
        if leaked:
            any_leak = True
        max_correlations[prop_name] = (max_r, max_byte, leaked)

    print(f"\n  Images analyzed:  {num_images}")
    print(f"  Leakage threshold: |r| > 0.05")
    print()

    for prop_name, (max_r, max_byte, leaked) in max_correlations.items():
        status = "LEAK" if leaked else "OK"
        print(f"  {prop_name:20s}  max |r| = {abs(max_r):.4f}  "
              f"(byte {max_byte:2d})  [{status}]")

    passed = not any_leak
    print(f"\n  Any leakage detected?  {'YES' if any_leak else 'NO'}")
    print(f"  Result:                {'PASS' if passed else 'FAIL'}")
    print()
    return passed


# ─────────────────────────────────────────────────────────────
# TEST 3: FIXED-POINT SEARCH
# ─────────────────────────────────────────────────────────────

def test_fixed_point_search(config, renderer, hasher, num_images=10000):
    print("=" * 60)
    print("  TEST 3: FIXED-POINT SEARCH")
    print("  Are there exploitable hash ↔ image relationships?")
    print("  Check for byte-level fixed points and permutations.")
    print("=" * 60)

    base_seed = hashlib.sha256(b"fixed_point_test").digest()

    # (a) Byte-level fixed points: hash byte == corresponding pixel
    # Hash is 32 bytes, image is 4096 bytes. Compare first 32 pixels to hash.
    fixed_point_counts = []

    # (b) Permutation check: is hash a rotation/permutation of first 32 pixels?
    perm_match_counts = []

    for i in range(num_images):
        image = renderer.render_from_nonce(base_seed, i)
        h = hasher.hash_image(image)

        # (a) Count how many of the 32 hash bytes match the corresponding pixel
        matches = sum(1 for j in range(32) if h[j] == image[j])
        fixed_point_counts.append(matches)

        # (b) Check if hash bytes are a permutation of the first 32 image bytes
        hash_sorted = sorted(h)
        image_sorted = sorted(image[:32].tobytes())
        perm_match = (hash_sorted == image_sorted)
        perm_match_counts.append(int(perm_match))

    avg_fixed = np.mean(fixed_point_counts)
    max_fixed = np.max(fixed_point_counts)
    any_perm = sum(perm_match_counts)

    # Random baseline: for each of 32 positions, P(match) = 1/256
    # Expected matches = 32/256 = 0.125
    expected_fixed = 32.0 / 256.0

    # Check if hash byte == corresponding pixel more than random
    # Use one-sample t-test against expected mean
    t_stat, p_value = stats.ttest_1samp(fixed_point_counts, expected_fixed)
    # We care if it's significantly HIGHER than expected (one-sided)
    p_one_sided = p_value / 2 if t_stat > 0 else 1.0 - p_value / 2

    fixed_pass = p_one_sided > 0.01  # Not significantly more fixed points than random
    perm_pass = any_perm == 0  # No permutation matches

    print(f"\n  Images tested:             {num_images}")
    print(f"")
    print(f"  (a) Byte-level fixed points (hash[j] == pixel[j]):")
    print(f"      Avg matches:           {avg_fixed:.3f}  (random baseline: {expected_fixed:.3f})")
    print(f"      Max matches:           {max_fixed}")
    print(f"      p-value (> baseline):  {p_one_sided:.4f}")
    print(f"      Result:                {'PASS' if fixed_pass else 'FAIL — excess fixed points'}")
    print(f"")
    print(f"  (b) Permutation matches (hash = permutation of first 32 pixels):")
    print(f"      Matches found:         {any_perm} / {num_images}")
    print(f"      Result:                {'PASS' if perm_pass else 'FAIL — permutation relationship found'}")

    passed = fixed_pass and perm_pass
    print(f"\n  Overall:                   {'PASS' if passed else 'FAIL'}")
    print()
    return passed


# ─────────────────────────────────────────────────────────────
# TEST 4: SECOND-PREIMAGE SHORTCUT
# ─────────────────────────────────────────────────────────────

def test_second_preimage_shortcut(config, renderer, hasher, num_images=100):
    print("=" * 60)
    print("  TEST 4: SECOND-PREIMAGE SHORTCUT")
    print("  Do similar derived parameters → closer hash outputs?")
    print("  If yes, knowing I1's parameters helps find I2 where")
    print("  BAB64(I2) = BAB64(I1).")
    print("=" * 60)

    base_seed = hashlib.sha256(b"second_preimage_test").digest()

    # Generate images and extract their parameters + hashes
    images = []
    hashes = []
    round_constants_list = []
    sbox_list = []

    for i in range(num_images):
        image = renderer.render_from_nonce(base_seed, i)
        h = hasher.hash_image(image)

        images.append(image)
        hashes.append(h)
        round_constants_list.append(hasher._derive_round_constants(image))
        sbox_list.append(hasher._derive_sbox(image))

    # For each pair, compute:
    #   - parameter similarity (round constant overlap fraction)
    #   - hash distance (Hamming distance in bits)
    # Then check correlation: do more-similar parameters predict closer hashes?

    param_similarities = []
    hash_distances = []

    # Sample pairs (all pairs would be O(n^2); sample for speed)
    rng = np.random.RandomState(123)
    num_pairs = min(2000, num_images * (num_images - 1) // 2)
    pair_indices = set()
    while len(pair_indices) < num_pairs:
        a = rng.randint(0, num_images)
        b = rng.randint(0, num_images)
        if a != b and (a, b) not in pair_indices and (b, a) not in pair_indices:
            pair_indices.add((a, b))

    for a, b in pair_indices:
        # Round constant overlap
        rc_overlap = np.mean(round_constants_list[a] == round_constants_list[b])

        # S-box overlap
        sbox_overlap = np.mean(sbox_list[a] == sbox_list[b])

        # Combined parameter similarity
        param_sim = (rc_overlap + sbox_overlap) / 2.0
        param_similarities.append(param_sim)

        # Hash Hamming distance
        h_a = int.from_bytes(hashes[a], 'big')
        h_b = int.from_bytes(hashes[b], 'big')
        hamming = bin(h_a ^ h_b).count('1')
        hash_distances.append(hamming)

    param_similarities = np.array(param_similarities)
    hash_distances = np.array(hash_distances)

    # Correlation: does higher param similarity → lower hash distance?
    # If r is significantly negative, there's a shortcut.
    if np.std(param_similarities) > 0 and np.std(hash_distances) > 0:
        r, p = stats.pearsonr(param_similarities, hash_distances)
    else:
        r, p = 0.0, 1.0

    # Also check: are the most-similar-parameter pairs closer in hash space?
    # Top 10% most similar params vs bottom 10%
    top_k = max(1, len(param_similarities) // 10)
    sorted_idx = np.argsort(param_similarities)
    bottom_10_dist = np.mean(hash_distances[sorted_idx[:top_k]])
    top_10_dist = np.mean(hash_distances[sorted_idx[-top_k:]])

    # Statistical test on top vs bottom
    top_dists = hash_distances[sorted_idx[-top_k:]]
    bottom_dists = hash_distances[sorted_idx[:top_k]]
    t_stat, t_p = stats.ttest_ind(top_dists, bottom_dists, alternative='less')

    shortcut_exists = (r < -0.05 and p < 0.05) or t_p < 0.01
    passed = not shortcut_exists

    print(f"\n  Images:                    {num_images}")
    print(f"  Pairs sampled:             {num_pairs}")
    print(f"")
    print(f"  Correlation (param_sim vs hash_dist):")
    print(f"    Pearson r:               {r:.4f}  (negative = shortcut)")
    print(f"    p-value:                 {p:.4f}")
    print(f"")
    print(f"  Top 10% similar params:")
    print(f"    Avg hash distance:       {top_10_dist:.1f} bits")
    print(f"  Bottom 10% similar params:")
    print(f"    Avg hash distance:       {bottom_10_dist:.1f} bits")
    print(f"  Ideal (no shortcut):       both ≈ 128 bits")
    print(f"  t-test (top < bottom):     t={t_stat:.3f}, p={t_p:.4f}")
    print(f"")
    print(f"  Shortcut detected?         {'YES' if shortcut_exists else 'NO'}")
    print(f"  Result:                    {'PASS' if passed else 'FAIL'}")
    print()
    return passed


# ─────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────

def main():
    print()
    print("╔══════════════════════════════════════════════════════════╗")
    print("║     Attack 5: Preimage Structure on BAB64               ║")
    print("║     Does self-referential construction leak info?        ║")
    print("╚══════════════════════════════════════════════════════════╝")
    print()

    config = BAB64Config()
    renderer = BabelRenderer(config)
    hasher = ImageHash(config)

    start = time.time()

    r1 = test_output_bias(config, renderer, hasher, num_images=10000)
    r2 = test_hash_to_parameter_leakage(config, renderer, hasher, num_images=5000)
    r3 = test_fixed_point_search(config, renderer, hasher, num_images=10000)
    r4 = test_second_preimage_shortcut(config, renderer, hasher, num_images=100)

    elapsed = time.time() - start

    print("=" * 60)
    print("  SUMMARY")
    print("=" * 60)
    results = [
        ("Output Bias", r1),
        ("Hash-to-Parameter Leakage", r2),
        ("Fixed-Point Search", r3),
        ("Second-Preimage Shortcut", r4),
    ]
    all_pass = True
    for name, passed in results:
        status = "PASS" if passed else "FAIL"
        print(f"  {name:30s} {status}")
        if not passed:
            all_pass = False

    print(f"\n  Total time: {elapsed:.1f}s")
    if all_pass:
        print("  BAB64 resists preimage structure attacks.")
        print("  Self-referential construction does NOT leak exploitable info.")
    else:
        print("  WARNING: Preimage structure weaknesses detected.")
    print("=" * 60)
    print()


if __name__ == "__main__":
    main()
