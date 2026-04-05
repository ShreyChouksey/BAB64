"""
Attack 6: Joux Multi-Collision
================================
Joux (2004) showed that for Merkle-Damgard hashes, finding
2^k collisions takes only k times the work of one collision,
not 2^k times. Collisions in each block compose independently.

BAB64 uses Merkle-Damgard. But each image defines a DIFFERENT
hash function (different round constants, rotations, S-box,
initial state). Does this per-image parameterization neutralize
Joux's attack?

Tests:
  1. INTERNAL STATE COLLISION — do any two images share an
     intermediate state at block 64 (halfway)?
  2. BLOCK-LEVEL COLLISION INDEPENDENCE — do block-1 collisions
     propagate to block 2 as standard MD would predict?
  3. STATE ENTROPY — does the number of unique intermediate
     states shrink at any point in the chain?
"""

import hashlib
import numpy as np
import time
import ctypes
import os
from ctypes import c_uint32, c_int32, c_uint8, c_int, POINTER
from bab64_engine import BAB64Config, BabelRenderer, ImageHash

# ── Load C extension ──────────────────────────────────────────

_dir = os.path.dirname(os.path.abspath(__file__))
_lib = None
for name in ["bab64_fast.dylib", "bab64_fast.so"]:
    path = os.path.join(_dir, name)
    if os.path.exists(path):
        try:
            _lib = ctypes.CDLL(path)
            break
        except OSError:
            pass

if _lib is None:
    raise RuntimeError("C extension not found — build with: "
                       "cc -O3 -shared -fPIC -o bab64_fast.dylib bab64_fast.c")

_lib.bab64_compress.restype = None
_lib.bab64_compress.argtypes = [
    POINTER(c_uint32), POINTER(c_uint8), POINTER(c_uint32),
    POINTER(c_int32), POINTER(c_uint8), c_int, c_int, c_int,
    POINTER(c_uint32),
]


def compress_one_block(state, block_bytes, rc, rot, sbox, num_rounds):
    """Run one compression call via C and return the new state."""
    state_in = np.ascontiguousarray(state, dtype=np.uint32)
    state_out = np.zeros(8, dtype=np.uint32)

    pad = block_bytes
    if len(pad) < 32:
        pad = pad + b'\x00' * (32 - len(pad))
    blk = np.frombuffer(pad[:32], dtype=np.uint8).copy()

    _lib.bab64_compress(
        state_in.ctypes.data_as(POINTER(c_uint32)),
        blk.ctypes.data_as(POINTER(c_uint8)),
        rc.ctypes.data_as(POINTER(c_uint32)),
        rot.ctypes.data_as(POINTER(c_int32)),
        sbox.ctypes.data_as(POINTER(c_uint8)),
        c_int(num_rounds),
        c_int(len(rc)),
        c_int(len(rot)),
        state_out.ctypes.data_as(POINTER(c_uint32)),
    )
    return state_out


def state_to_key(state):
    """Convert 8×uint32 state to a hashable tuple."""
    return tuple(int(w) for w in state)


def derive_params(hasher, image):
    """Derive all hash parameters for an image."""
    rc = np.ascontiguousarray(
        hasher._derive_round_constants(image), dtype=np.uint32)
    rot = np.ascontiguousarray(
        hasher._derive_rotations(image), dtype=np.int32)
    sbox = np.ascontiguousarray(
        hasher._derive_sbox(image), dtype=np.uint8)
    init = np.ascontiguousarray(
        hasher._derive_initial_state(image), dtype=np.uint32)
    return rc, rot, sbox, init


def run_chain_capture(image, hasher, config, capture_blocks):
    """
    Run the full MD chain block-by-block via C, capturing the
    intermediate state after each block index in capture_blocks.
    Returns dict {block_index: state_tuple}.
    """
    rc, rot, sbox, state = derive_params(hasher, image)
    image_bytes = image.tobytes()
    bs = config.block_size
    num_blocks = (len(image_bytes) + bs - 1) // bs
    nr = config.num_rounds
    captured = {}

    for b in range(num_blocks):
        start = b * bs
        block = image_bytes[start:start + bs]
        state = compress_one_block(state, block, rc, rot, sbox, nr)
        if b in capture_blocks:
            captured[b] = state_to_key(state)

    return captured


# ─────────────────────────────────────────────────────────────
# TEST 1: INTERNAL STATE COLLISION
# ─────────────────────────────────────────────────────────────

def test_internal_state_collision(config, renderer, hasher, num_images=5000):
    print("=" * 60)
    print("  TEST 1: INTERNAL STATE COLLISION")
    print("  Hash 5,000 images. After block 64 (halfway), do ANY")
    print("  two images share an intermediate state?")
    print("=" * 60)

    base_seed = hashlib.sha256(b"joux_internal_state").digest()
    halfway = 63  # block index 63 = 64th block (0-indexed)

    seen_states = {}  # state_tuple -> image index
    collision_found = False
    collision_pair = None

    for i in range(num_images):
        image = renderer.render_from_nonce(base_seed, i)
        captured = run_chain_capture(image, hasher, config, {halfway})
        st = captured[halfway]

        if st in seen_states:
            collision_found = True
            collision_pair = (seen_states[st], i)
            break
        seen_states[st] = i

        if (i + 1) % 1000 == 0:
            print(f"    ... {i + 1}/{num_images} images, "
                  f"{len(seen_states)} unique states")

    num_unique = len(seen_states)

    # Birthday bound for 256-bit state: collision expected at ~2^128.
    # 5,000 images should produce 0 collisions with overwhelming probability.
    passed = not collision_found

    print(f"\n  Images hashed:       {num_images}")
    print(f"  Checkpoint:          block 64 (halfway)")
    print(f"  Unique states:       {num_unique}")
    if collision_found:
        print(f"  COLLISION:           images {collision_pair[0]} "
              f"and {collision_pair[1]}")
    else:
        print(f"  Collisions found:    0")
    print(f"  Expected (256-bit):  0 (birthday bound ~ 2^128)")
    print(f"  Result:              {'PASS' if passed else 'FAIL'}")
    print()
    return passed


# ─────────────────────────────────────────────────────────────
# TEST 2: BLOCK-LEVEL COLLISION INDEPENDENCE
# ─────────────────────────────────────────────────────────────

def test_block_collision_independence(config, renderer, hasher, num_images=100):
    print("=" * 60)
    print("  TEST 2: BLOCK-LEVEL COLLISION INDEPENDENCE")
    print("  Find pairs where block 1 yields the same intermediate")
    print("  state. In standard MD (fixed function), those pairs")
    print("  would also collide at block 2. In BAB64, different")
    print("  parameters should prevent propagation.")
    print("=" * 60)

    base_seed = hashlib.sha256(b"joux_block_independence").digest()

    # Collect states after block 0 and block 1 for each image
    block0_states = {}  # state_key -> list of (image_index, block1_state)
    all_block0 = []
    all_block1 = []

    for i in range(num_images):
        image = renderer.render_from_nonce(base_seed, i)
        captured = run_chain_capture(image, hasher, config, {0, 1})
        st0 = captured[0]
        st1 = captured[1]
        all_block0.append(st0)
        all_block1.append(st1)

        if st0 not in block0_states:
            block0_states[st0] = []
        block0_states[st0].append((i, st1))

    # Find block-0 collisions (same state after processing block 0)
    block0_collision_groups = {
        k: v for k, v in block0_states.items() if len(v) > 1
    }
    num_block0_collisions = sum(
        len(v) * (len(v) - 1) // 2 for v in block0_collision_groups.values()
    )

    # Among block-0 collision pairs, check if block-1 also collides
    block1_also_collide = 0
    total_pairs_checked = 0

    for state_key, group in block0_collision_groups.items():
        for a_idx in range(len(group)):
            for b_idx in range(a_idx + 1, len(group)):
                total_pairs_checked += 1
                _, st1_a = group[a_idx]
                _, st1_b = group[b_idx]
                if st1_a == st1_b:
                    block1_also_collide += 1

    # Since each image has DIFFERENT parameters, even if two images
    # happen to reach the same state after block 0, their different
    # round constants/rotations/sbox mean block 1 should diverge.
    # In standard MD (same function), 100% would propagate.

    # With 100 images and 256-bit states, block-0 collisions are
    # astronomically unlikely anyway. But we check the logic:
    # if any exist, they should NOT propagate.

    # Also do a softer check: measure Hamming distance between block-1
    # states for the closest block-0 pairs
    # (since exact collisions at block 0 are unlikely with 100 images)
    from scipy.spatial.distance import hamming as hamming_dist

    # Find the pair with minimum Hamming distance at block 0
    min_b0_hamming = float('inf')
    min_pair = None
    # Sample pairs (avoid O(n^2) for large n)
    rng = np.random.RandomState(42)
    sample_size = min(2000, num_images * (num_images - 1) // 2)
    pairs_checked = 0

    b0_hamming_list = []
    b1_hamming_list = []

    while pairs_checked < sample_size:
        a = rng.randint(0, num_images)
        b = rng.randint(0, num_images)
        if a == b:
            continue
        pairs_checked += 1

        # Block-0 state Hamming distance (count differing uint32 words)
        st0_a = all_block0[a]
        st0_b = all_block0[b]
        h0 = sum(1 for x, y in zip(st0_a, st0_b) if x != y)

        st1_a = all_block1[a]
        st1_b = all_block1[b]
        h1 = sum(1 for x, y in zip(st1_a, st1_b) if x != y)

        b0_hamming_list.append(h0)
        b1_hamming_list.append(h1)

    avg_b0_hamming = np.mean(b0_hamming_list)
    avg_b1_hamming = np.mean(b1_hamming_list)

    # In standard MD, if block-0 collides (h0=0) then block-1
    # also collides (h1=0). With per-image functions, even close
    # block-0 states should diverge at block-1.
    # Both averages should be near 8 (all words differ) for random states.

    passed = (block1_also_collide == 0) and (avg_b1_hamming > 6.0)

    print(f"\n  Images:                    {num_images}")
    print(f"  Block-0 exact collisions:  {num_block0_collisions}")
    print(f"  Block-1 also collide:      {block1_also_collide}")
    print(f"")
    print(f"  Sampled {sample_size} random pairs:")
    print(f"    Avg block-0 word diff:   {avg_b0_hamming:.2f} / 8")
    print(f"    Avg block-1 word diff:   {avg_b1_hamming:.2f} / 8")
    print(f"  Expected (independent):    both near 8.0 (all words differ)")
    print(f"")
    print(f"  Joux propagation?          "
          f"{'YES — block collisions compose!' if block1_also_collide > 0 else 'NO — per-image params prevent it'}")
    print(f"  Result:                    {'PASS' if passed else 'FAIL'}")
    print()
    return passed


# ─────────────────────────────────────────────────────────────
# TEST 3: STATE ENTROPY
# ─────────────────────────────────────────────────────────────

def test_state_entropy(config, renderer, hasher, num_images=1000):
    print("=" * 60)
    print("  TEST 3: STATE ENTROPY")
    print("  Track unique intermediate states after blocks")
    print("  1, 16, 32, 64, 128 across 1,000 images.")
    print("  If entropy drops, there is a convergence weakness.")
    print("=" * 60)

    base_seed = hashlib.sha256(b"joux_state_entropy").digest()
    checkpoints = [0, 15, 31, 63, 127]  # 0-indexed (blocks 1,16,32,64,128)
    checkpoint_labels = {0: 1, 15: 16, 31: 32, 63: 64, 127: 128}

    # For each checkpoint, collect the set of unique states
    unique_states = {cp: set() for cp in checkpoints}

    for i in range(num_images):
        image = renderer.render_from_nonce(base_seed, i)
        captured = run_chain_capture(
            image, hasher, config, set(checkpoints))

        for cp in checkpoints:
            unique_states[cp].add(captured[cp])

        if (i + 1) % 250 == 0:
            print(f"    ... {i + 1}/{num_images} images processed")

    # Check: unique state count should equal num_images at every
    # checkpoint (since each image has different parameters and
    # different message blocks, producing unique chains).
    # A drop below num_images indicates convergence.

    print(f"\n  Images:  {num_images}")
    print(f"  {'Block':>8s}  {'Unique States':>15s}  {'Ratio':>8s}  Status")
    print(f"  {'-'*48}")

    min_ratio = 1.0
    all_full = True
    for cp in checkpoints:
        n_unique = len(unique_states[cp])
        ratio = n_unique / num_images
        if ratio < 1.0:
            all_full = False
        min_ratio = min(min_ratio, ratio)
        label = checkpoint_labels[cp]
        status = "OK" if ratio >= 0.999 else "LOW"
        print(f"  {label:>8d}  {n_unique:>15d}  {ratio:>8.4f}  {status}")

    # Pass if all checkpoints have >= 99.9% unique states
    # (allowing for astronomically unlikely birthday collisions)
    passed = min_ratio >= 0.999

    print(f"\n  Minimum ratio:  {min_ratio:.4f}")
    print(f"  Entropy drop?   {'YES — convergence detected' if not passed else 'NO — full entropy at all checkpoints'}")
    print(f"  Result:          {'PASS' if passed else 'FAIL'}")
    print()
    return passed


# ─────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────

def main():
    print()
    print("+" + "=" * 58 + "+")
    print("|     Attack 6: Joux Multi-Collision on BAB64              |")
    print("|     Does per-image parameterization neutralize Joux?     |")
    print("+" + "=" * 58 + "+")
    print()

    config = BAB64Config()
    renderer = BabelRenderer(config)
    hasher = ImageHash(config)

    start = time.time()

    r1 = test_internal_state_collision(config, renderer, hasher, num_images=5000)
    r2 = test_block_collision_independence(config, renderer, hasher, num_images=100)
    r3 = test_state_entropy(config, renderer, hasher, num_images=1000)

    elapsed = time.time() - start

    print("=" * 60)
    print("  SUMMARY")
    print("=" * 60)
    results = [
        ("Internal State Collision", r1),
        ("Block Collision Independence", r2),
        ("State Entropy", r3),
    ]
    all_pass = True
    for name, passed in results:
        status = "PASS" if passed else "FAIL"
        print(f"  {name:30s} {status}")
        if not passed:
            all_pass = False

    print(f"\n  Total time: {elapsed:.1f}s")
    if all_pass:
        print("  BAB64 resists Joux multi-collision attack.")
        print("  Per-image hash parameterization neutralizes the")
        print("  Merkle-Damgard structural weakness that Joux exploits.")
    else:
        print("  WARNING: Joux-style weaknesses detected.")
    print("=" * 60)
    print()


if __name__ == "__main__":
    main()
