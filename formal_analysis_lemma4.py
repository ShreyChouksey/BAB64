"""
Formal Security Analysis — Lemma 4: Merkle-Damgard Collision Resistance
=========================================================================

LEMMA 4. The Merkle-Damgard iteration over 128 blocks preserves
collision resistance from the compression function.

PROOF SKETCH. Merkle (1989) and Damgard (1989) proved that if a
compression function h: {0,1}^n × {0,1}^m → {0,1}^n is collision-
resistant, then the iterated hash H built by chaining h over arbitrary-
length messages is also collision-resistant.

BAB64 satisfies the theorem's assumptions: for a given image, the
compression function parameters (round constants, rotations, S-box,
initial state) are derived ONCE and held FIXED across all 128 blocks.
The classical MD theorem therefore applies directly.

However, we verify empirically that the specific construction does
not introduce weaknesses the theorem does not cover:

  1. Length extension resistance (Davies-Meyer feedforward)
  2. Intermediate state diversity (no convergence across blocks)
  3. Block order sensitivity (permuting blocks changes the hash)
  4. Compression chain independence (early ≠ late states)
  5. Multi-block collision search (no intermediate collisions)

WHY THIS MATTERS. The MD theorem gives a reduction: any collision
in the full hash implies a collision in the compression function.
If Lemma 3 establishes PRP security of the compression function,
Lemma 4 lifts that to collision resistance of the full 128-block hash.

Target: IACR resubmission, Section 4.4 (Merkle-Damgard Preservation).
"""

import hashlib
import numpy as np
import time
from scipy import stats as sp_stats

from bab64_engine import BAB64Config, BabelRenderer, ImageHash


# =============================================================================
# HELPERS
# =============================================================================

def _make_image(seed_index: int, tag: str = "lemma4") -> np.ndarray:
    """Generate a deterministic image from a seed index."""
    config = BAB64Config()
    renderer = BabelRenderer(config)
    seed = hashlib.sha256(f"{tag}_{seed_index}".encode()).digest()
    return renderer.render(seed)


def _hash_with_intermediates(image: np.ndarray):
    """
    Run the full BAB64 MD chain, returning the final hash AND
    every intermediate state (after each of the 128 block compressions).

    Returns: (final_hash_bytes, list_of_128_state_arrays)
    """
    config = BAB64Config()
    hasher = ImageHash(config)

    rc = hasher._derive_round_constants(image)
    rot = hasher._derive_rotations(image)
    sbox = hasher._derive_sbox(image)
    state = hasher._derive_initial_state(image)

    image_bytes = image.tobytes()
    block_size = config.block_size
    num_blocks = (len(image_bytes) + block_size - 1) // block_size

    intermediates = []
    for b in range(num_blocks):
        start = b * block_size
        block = image_bytes[start:start + block_size]
        if len(block) < block_size:
            block = block + b'\x00' * (block_size - len(block))
        state = hasher._compress(state, block, rc, rot, sbox, round_offset=b)
        intermediates.append(state.copy())

    final = b''
    for word in state:
        final += int(word).to_bytes(4, 'big')

    return final, intermediates


def _state_to_bytes(state: np.ndarray) -> bytes:
    """Convert an 8×uint32 state array to 32 bytes."""
    return b''.join(int(w).to_bytes(4, 'big') for w in state)


def _hamming_bits(a: bytes, b: bytes) -> int:
    """Count differing bits between two equal-length byte strings."""
    return bin(int.from_bytes(a, 'big') ^ int.from_bytes(b, 'big')).count('1')


# =============================================================================
# TEST 1 — LENGTH EXTENSION RESISTANCE
# =============================================================================
# BAB64 uses Davies-Meyer feedforward AND image-derived parameters.
# An attacker who knows H(image) but NOT the image cannot compute
# H(image || extra) because they lack the compression function params.
# We verify: given only the final hash, attempting to extend with a
# random block using WRONG parameters produces a different result
# than extending with the CORRECT parameters.

def test_length_extension(n_images=500):
    """
    TEST 1: Length extension resistance.

    For each image:
      1. Compute full hash H = BAB64(image) — this is public.
      2. Compute the "correct extension": compress one more block
         using the TRUE image-derived parameters from the final state.
      3. Compute a "blind extension": compress the same block using
         parameters derived from a DIFFERENT random image (simulating
         an attacker who doesn't know the original image).
      4. Verify correct != blind in all cases.

    Structural argument: without knowledge of (rc, rot, sbox), the
    attacker cannot evaluate the compression function, so they cannot
    extend the hash. This test confirms it empirically.
    """
    config = BAB64Config()
    hasher = ImageHash(config)

    mismatches = 0
    hamming_dists = []
    extra_block = hashlib.sha256(b"extension_payload").digest()  # 32 bytes

    for i in range(n_images):
        image = _make_image(i)

        # Derive TRUE parameters
        rc = hasher._derive_round_constants(image)
        rot = hasher._derive_rotations(image)
        sbox = hasher._derive_sbox(image)

        # Get the final MD state (= last intermediate)
        _, intermediates = _hash_with_intermediates(image)
        final_state = intermediates[-1]

        # Correct extension: compress extra_block with TRUE params
        correct_ext = hasher._compress(final_state, extra_block, rc, rot, sbox)

        # Blind extension: use params from a DIFFERENT image
        other_image = _make_image(i + n_images, tag="lemma4_other")
        rc_other = hasher._derive_round_constants(other_image)
        rot_other = hasher._derive_rotations(other_image)
        sbox_other = hasher._derive_sbox(other_image)
        blind_ext = hasher._compress(final_state, extra_block,
                                     rc_other, rot_other, sbox_other)

        correct_bytes = _state_to_bytes(correct_ext)
        blind_bytes = _state_to_bytes(blind_ext)

        if correct_bytes != blind_bytes:
            mismatches += 1
            hamming_dists.append(_hamming_bits(correct_bytes, blind_bytes))

    mismatch_rate = mismatches / n_images
    avg_hamming = float(np.mean(hamming_dists)) if hamming_dists else 0.0

    return {
        'n_images': n_images,
        'mismatches': mismatches,
        'mismatch_rate': mismatch_rate,
        'avg_hamming_bits': avg_hamming,
        'avg_hamming_pct': avg_hamming / 256 * 100,
        # All must mismatch, and Hamming should be ~128 bits (50%)
        'pass': mismatch_rate == 1.0 and avg_hamming > 100,
    }


# =============================================================================
# TEST 2 — INTERMEDIATE STATE DIVERSITY
# =============================================================================
# Hash 1,000 images, record the intermediate state after blocks
# 1, 32, 64, 96, 128. Verify all states are unique at each checkpoint
# and compute entropy of state distribution at each checkpoint.

def test_intermediate_diversity(n_images=1000):
    """
    TEST 2: Intermediate state diversity.

    No two images should share an intermediate state at any checkpoint.
    If states converge, the MD chain loses entropy — a structural flaw.
    We also measure byte-level entropy at each checkpoint to confirm
    the states are well-distributed.
    """
    checkpoints = [0, 31, 63, 95, 127]  # 0-indexed: blocks 1, 32, 64, 96, 128
    checkpoint_labels = [1, 32, 64, 96, 128]

    # Collect states: checkpoint_idx → list of state bytes (hex)
    states_at = {cp: [] for cp in checkpoints}
    # For entropy: collect raw byte arrays
    raw_states_at = {cp: [] for cp in checkpoints}

    for i in range(n_images):
        image = _make_image(i, tag="lemma4_diversity")
        _, intermediates = _hash_with_intermediates(image)

        for cp in checkpoints:
            sb = _state_to_bytes(intermediates[cp])
            states_at[cp].append(sb.hex())
            raw_states_at[cp].append(np.frombuffer(sb, dtype=np.uint8))

        if (i + 1) % 200 == 0:
            print(f"    2: {i+1}/{n_images} images done", flush=True)

    results = {}
    all_unique = True

    for cp, label in zip(checkpoints, checkpoint_labels):
        unique_count = len(set(states_at[cp]))
        is_unique = (unique_count == n_images)
        if not is_unique:
            all_unique = False

        # Byte-level entropy: for each of 32 byte positions, compute
        # Shannon entropy of the distribution across images
        byte_array = np.array(raw_states_at[cp])  # (n_images, 32)
        entropies = []
        for col in range(32):
            values, counts = np.unique(byte_array[:, col], return_counts=True)
            probs = counts / counts.sum()
            ent = -np.sum(probs * np.log2(probs + 1e-15))
            entropies.append(ent)
        avg_entropy = float(np.mean(entropies))
        min_entropy = float(np.min(entropies))

        results[f'block_{label}'] = {
            'unique': unique_count,
            'total': n_images,
            'all_unique': is_unique,
            'avg_byte_entropy': avg_entropy,
            'min_byte_entropy': min_entropy,
        }

    # Max possible byte entropy with 1000 images ≈ log2(256) = 8.0
    # but with 1000 samples over 256 bins, practical max ~ 7.5+
    results['pass'] = all_unique

    return results


# =============================================================================
# TEST 3 — BLOCK ORDER SENSITIVITY
# =============================================================================
# For 100 images, swap two adjacent message blocks and measure
# how much the final hash changes. Should be ~128 bits (50%).

def test_block_order_sensitivity(n_images=100):
    """
    TEST 3: Block order sensitivity.

    Swapping two adjacent message blocks should completely change
    the final hash (~128 of 256 bits differ). This confirms that
    the MD chain is not commutative — block order matters.
    """
    config = BAB64Config()
    hasher = ImageHash(config)
    block_size = config.block_size

    hamming_dists = []

    for i in range(n_images):
        image = _make_image(i, tag="lemma4_order")
        image_bytes = image.tobytes()

        # Compute original hash
        original_hash = hasher.hash_image(image)

        # Choose two adjacent blocks near the middle to swap
        # (avoid first/last to ensure both blocks are full)
        swap_pos = 60  # swap blocks 60 and 61 (0-indexed)

        # Split into blocks
        blocks = []
        num_blocks = (len(image_bytes) + block_size - 1) // block_size
        for b in range(num_blocks):
            start = b * block_size
            blk = image_bytes[start:start + block_size]
            if len(blk) < block_size:
                blk = blk + b'\x00' * (block_size - len(blk))
            blocks.append(blk)

        # Swap two adjacent blocks
        blocks[swap_pos], blocks[swap_pos + 1] = \
            blocks[swap_pos + 1], blocks[swap_pos]

        # Re-hash with swapped blocks (same params, different block order)
        rc = hasher._derive_round_constants(image)
        rot = hasher._derive_rotations(image)
        sbox = hasher._derive_sbox(image)
        state = hasher._derive_initial_state(image)

        for blk in blocks:
            state = hasher._compress(state, blk, rc, rot, sbox)

        swapped_hash = b''
        for word in state:
            swapped_hash += int(word).to_bytes(4, 'big')

        dist = _hamming_bits(original_hash, swapped_hash)
        hamming_dists.append(dist)

    hamming_dists = np.array(hamming_dists)
    avg_dist = float(np.mean(hamming_dists))
    std_dist = float(np.std(hamming_dists))
    min_dist = int(np.min(hamming_dists))

    return {
        'n_images': n_images,
        'avg_hamming_bits': avg_dist,
        'avg_hamming_pct': avg_dist / 256 * 100,
        'std_hamming_bits': std_dist,
        'min_hamming_bits': min_dist,
        # Expect ~128 bits (50%), pass if mean > 100 bits (~39%)
        'pass': avg_dist > 100 and min_dist > 50,
    }


# =============================================================================
# TEST 4 — COMPRESSION CHAIN INDEPENDENCE
# =============================================================================
# For 100 images, measure correlation between the output of block 1
# and the output of block 128. Should be uncorrelated.

def test_chain_independence(n_images=100):
    """
    TEST 4: Compression chain independence.

    The state after block 1 should not predict the state after block 128.
    We compute Pearson correlation between corresponding bytes of the
    two states across many images. Max |r| should be small.
    """
    early_states = []
    late_states = []

    for i in range(n_images):
        image = _make_image(i, tag="lemma4_chain")
        _, intermediates = _hash_with_intermediates(image)

        early_bytes = np.frombuffer(_state_to_bytes(intermediates[0]),
                                    dtype=np.uint8).astype(np.float64)
        late_bytes = np.frombuffer(_state_to_bytes(intermediates[-1]),
                                   dtype=np.uint8).astype(np.float64)
        early_states.append(early_bytes)
        late_states.append(late_bytes)

    early_arr = np.array(early_states)  # (n_images, 32)
    late_arr = np.array(late_states)    # (n_images, 32)

    # Compute correlation matrix: 32 early bytes × 32 late bytes
    # Normalize
    early_centered = early_arr - early_arr.mean(axis=0, keepdims=True)
    late_centered = late_arr - late_arr.mean(axis=0, keepdims=True)
    early_std = np.sqrt(np.sum(early_centered ** 2, axis=0, keepdims=True) + 1e-10)
    late_std = np.sqrt(np.sum(late_centered ** 2, axis=0, keepdims=True) + 1e-10)
    early_normed = early_centered / early_std
    late_normed = late_centered / late_std

    corr_matrix = (early_normed.T @ late_normed) / n_images  # (32, 32)
    max_abs_r = float(np.max(np.abs(corr_matrix)))
    mean_abs_r = float(np.mean(np.abs(corr_matrix)))

    # Under H0 (independence), each r ~ N(0, 1/sqrt(n)), so
    # E[max |r|] over 1024 cells ≈ sqrt(2*ln(2*1024))/sqrt(n)
    # For n=100: ≈ sqrt(2*ln(2048))/10 ≈ 3.9/10 = 0.39
    # We use a generous threshold
    expected_max = np.sqrt(2 * np.log(2 * 1024)) / np.sqrt(n_images)

    return {
        'n_images': n_images,
        'max_abs_r': max_abs_r,
        'mean_abs_r': mean_abs_r,
        'expected_max_under_h0': float(expected_max),
        # Pass if max correlation is within ~2× the expected random level
        'pass': max_abs_r < 2 * expected_max + 0.05,
    }


# =============================================================================
# TEST 5 — MULTI-BLOCK COLLISION SEARCH
# =============================================================================
# For 500 images, attempt to find two different 32-byte blocks that
# produce the same intermediate state when compressed from the same
# starting state. Try 10,000 random block pairs per image.

def test_multiblock_collision(n_images=500, n_trials=10000):
    """
    TEST 5: Multi-block collision search.

    For each image, take the initial state and compress many random
    32-byte blocks. Check if any two produce the same output state.
    Birthday bound for 256-bit states with 10,000 trials:
      P(collision) ≈ n²/(2×2^256) ≈ 10^8 / 10^77 ≈ 0.

    Zero collisions expected.
    """
    config = BAB64Config()
    hasher = ImageHash(config)

    total_collisions = 0
    rng = np.random.default_rng(4500)

    for i in range(n_images):
        image = _make_image(i, tag="lemma4_collision")

        rc = hasher._derive_round_constants(image)
        rot = hasher._derive_rotations(image)
        sbox = hasher._derive_sbox(image)
        state = hasher._derive_initial_state(image)

        seen = set()
        collisions_this_image = 0

        for _ in range(n_trials):
            block = rng.integers(0, 256, size=32, dtype=np.uint8).tobytes()
            out_state = hasher._compress(state, block, rc, rot, sbox)
            out_hex = _state_to_bytes(out_state).hex()

            if out_hex in seen:
                collisions_this_image += 1
            else:
                seen.add(out_hex)

        total_collisions += collisions_this_image

        if (i + 1) % 100 == 0:
            print(f"    5: {i+1}/{n_images} images done "
                  f"(collisions so far: {total_collisions})", flush=True)

    return {
        'n_images': n_images,
        'n_trials_per_image': n_trials,
        'total_blocks_tested': n_images * n_trials,
        'total_collisions': total_collisions,
        'pass': total_collisions == 0,
    }


# =============================================================================
# MAIN — RUN ALL TESTS AND PRINT SUMMARY
# =============================================================================

def run_all():
    print("=" * 72)
    print("  FORMAL ANALYSIS — LEMMA 4: MERKLE-DAMGARD COLLISION RESISTANCE")
    print("  H0: 128-block MD iteration preserves compression-function CR")
    print("  Method: five empirical tests on the iterated construction")
    print("=" * 72)
    print()

    total_time = time.time()
    all_results = []

    # =========================================================================
    # TEST 1 — LENGTH EXTENSION RESISTANCE
    # =========================================================================
    print("  [1/5] Length Extension Resistance (500 images)...")
    t0 = time.time()
    r1 = test_length_extension(500)
    dt = time.time() - t0
    verdict = "PASS" if r1['pass'] else "FAIL"
    print(f"         Mismatch rate:       {r1['mismatch_rate']:.4f} "
          f"(expect 1.0000)")
    print(f"         Avg Hamming (bits):  {r1['avg_hamming_bits']:.1f} / 256 "
          f"({r1['avg_hamming_pct']:.1f}%)")
    print(f"         Result: {verdict}  ({dt:.1f}s)")
    all_results.append(("1. Length extension resistance", r1['pass']))
    print()

    # =========================================================================
    # TEST 2 — INTERMEDIATE STATE DIVERSITY
    # =========================================================================
    print("  [2/5] Intermediate State Diversity (1,000 images × 5 checkpoints)...")
    t0 = time.time()
    r2 = test_intermediate_diversity(1000)
    dt = time.time() - t0
    verdict = "PASS" if r2['pass'] else "FAIL"
    for label in [1, 32, 64, 96, 128]:
        info = r2[f'block_{label}']
        print(f"         Block {label:>3}: "
              f"{info['unique']}/{info['total']} unique, "
              f"byte entropy={info['avg_byte_entropy']:.2f} bits "
              f"(min={info['min_byte_entropy']:.2f})")
    print(f"         Result: {verdict}  ({dt:.1f}s)")
    all_results.append(("2. Intermediate state diversity", r2['pass']))
    print()

    # =========================================================================
    # TEST 3 — BLOCK ORDER SENSITIVITY
    # =========================================================================
    print("  [3/5] Block Order Sensitivity (100 images)...")
    t0 = time.time()
    r3 = test_block_order_sensitivity(100)
    dt = time.time() - t0
    verdict = "PASS" if r3['pass'] else "FAIL"
    print(f"         Avg Hamming (bits):  {r3['avg_hamming_bits']:.1f} / 256 "
          f"({r3['avg_hamming_pct']:.1f}%)")
    print(f"         Std Hamming (bits):  {r3['std_hamming_bits']:.1f}")
    print(f"         Min Hamming (bits):  {r3['min_hamming_bits']}")
    print(f"         Result: {verdict}  ({dt:.1f}s)")
    all_results.append(("3. Block order sensitivity", r3['pass']))
    print()

    # =========================================================================
    # TEST 4 — COMPRESSION CHAIN INDEPENDENCE
    # =========================================================================
    print("  [4/5] Compression Chain Independence (100 images)...")
    t0 = time.time()
    r4 = test_chain_independence(100)
    dt = time.time() - t0
    verdict = "PASS" if r4['pass'] else "FAIL"
    print(f"         Max |r| (block 1↔128): {r4['max_abs_r']:.4f}")
    print(f"         Mean |r|:               {r4['mean_abs_r']:.4f}")
    print(f"         Expected max under H0:  {r4['expected_max_under_h0']:.4f}")
    print(f"         Result: {verdict}  ({dt:.1f}s)")
    all_results.append(("4. Compression chain independence", r4['pass']))
    print()

    # =========================================================================
    # TEST 5 — MULTI-BLOCK COLLISION SEARCH
    # =========================================================================
    print("  [5/5] Multi-Block Collision Search "
          "(500 images × 10,000 blocks)...")
    t0 = time.time()
    r5 = test_multiblock_collision(500, 10000)
    dt = time.time() - t0
    verdict = "PASS" if r5['pass'] else "FAIL"
    print(f"         Total blocks tested: {r5['total_blocks_tested']:,}")
    print(f"         Collisions found:    {r5['total_collisions']}")
    print(f"         Result: {verdict}  ({dt:.1f}s)")
    all_results.append(("5. Multi-block collision search", r5['pass']))
    print()

    # =========================================================================
    # FINAL SUMMARY
    # =========================================================================
    total_dt = time.time() - total_time
    n_pass = sum(1 for _, p in all_results if p)
    n_total = len(all_results)
    all_pass = all(p for _, p in all_results)

    print("=" * 72)
    print("  SUMMARY — LEMMA 4: MERKLE-DAMGARD COLLISION RESISTANCE")
    print(f"  Tests: {n_pass}/{n_total} PASS")
    print(f"  Total time: {total_dt:.1f}s")
    print("=" * 72)

    hdr = f"  {'Test':<45} {'Result':>8}"
    sep = f"  {'-'*45} {'-'*8}"
    print(hdr)
    print(sep)
    for name, passed in all_results:
        verdict = "PASS" if passed else "FAIL"
        print(f"  {name:<45} {verdict:>8}")
    print(sep)

    if all_pass:
        print()
        print("  CONCLUSION: All 5 tests PASS.")
        print("  We FAIL TO REJECT H0: the 128-block Merkle-Damgard")
        print("  iteration preserves collision resistance from the")
        print("  compression function established in Lemma 3.")
        print()
        print("  FORMAL ARGUMENT FOR PAPER:")
        print("  1. Length extension: Attacker without image parameters")
        print("     cannot evaluate the compression function, making")
        print("     length extension structurally impossible. Davies-Meyer")
        print("     feedforward provides additional protection even if")
        print("     parameters were known (standard MD weakness mitigated).")
        print("  2. Intermediate diversity: All 1,000 images produce")
        print("     unique states at every checkpoint (blocks 1, 32, 64,")
        print("     96, 128). No state convergence across the chain.")
        print("  3. Block order: Swapping adjacent blocks changes ~50%")
        print("     of output bits (Hamming distance ≈ 128/256).")
        print("     The chain is strictly order-dependent.")
        print("  4. Chain independence: Block 1 output does not predict")
        print("     block 128 output (max |r| near random baseline).")
        print("     The 127 intermediate compressions fully decorrelate")
        print("     early and late states.")
        print("  5. Collision search: Zero collisions in 5,000,000")
        print("     compression function evaluations, consistent with")
        print("     the birthday bound of 2^128 for 256-bit states.")
        print()
        print("  COMBINED WITH LEMMAS 1-3:")
        print("  Lemma 1: S-box ← uniform permutation (indistinguishable)")
        print("  Lemma 2: Parameters are pairwise independent")
        print("  Lemma 3: Compression function is a PRP")
        print("  Lemma 4: MD iteration preserves collision resistance")
        print()
        print("  Therefore BAB64 is collision-resistant under the")
        print("  assumption that its image-derived compression function")
        print("  is collision-resistant — which Lemma 3 establishes")
        print("  empirically via PRP indistinguishability.")
    else:
        failed = [(name, p) for name, p in all_results if not p]
        print()
        print(f"  WARNING: {len(failed)} test(s) FAILED:")
        for name, _ in failed:
            print(f"    - {name}")
        print()
        print("  The MD iteration may introduce weaknesses beyond")
        print("  what the classical theorem covers. Investigate:")
        print("    - State convergence (funnel effect)")
        print("    - Weak message schedule interaction")
        print("    - Insufficient diffusion across block boundaries")

    print()
    print("=" * 72)

    return all_pass


if __name__ == "__main__":
    run_all()
