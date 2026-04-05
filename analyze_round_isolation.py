"""
Round Isolation Analysis — Single-Block Compression Diffusion
=============================================================
Tests the BAB64 compression function in isolation (one block,
no Merkle-Damgard chaining) to determine whether round-level
diffusion exists independent of multi-block chaining.

Key question: does a single _compress call diffuse a 1-bit
message change across the 256-bit state, or does diffusion
only emerge from processing 128 sequential blocks?
"""

import hashlib
import numpy as np
import time
import os
from bab64_engine import BAB64Config, BabelRenderer, ImageHash


def state_to_int(state):
    """Convert 8x uint32 state to a single 256-bit integer."""
    val = 0
    for w in state:
        val = (val << 32) | int(w)
    return val


def count_diff_bits(s1, s2):
    """Count differing bits between two 8-word states."""
    return bin(state_to_int(s1) ^ state_to_int(s2)).count('1')


def flip_bit_in_block(block, bit_index):
    """Flip a single bit in a 32-byte message block."""
    arr = bytearray(block)
    byte_idx = bit_index // 8
    bit_idx = bit_index % 8
    arr[byte_idx] ^= (1 << bit_idx)
    return bytes(arr)


def single_compress(hasher, state, block, rc, rot, sbox, num_rounds):
    """Run _compress with a specific round count."""
    original = hasher.config.num_rounds
    hasher.config.num_rounds = num_rounds
    result = hasher._compress(state, block, rc, rot, sbox)
    hasher.config.num_rounds = original
    return result


def run_analysis(num_trials=200):
    print(f"""
{'='*70}
  ROUND ISOLATION ANALYSIS — Single-Block Compression
{'='*70}
  Trials:  {num_trials} (different images + message blocks)
  Method:  _compress(state, block) with 1 bit flipped in block
  Goal:    isolate round-function diffusion from MD chaining
{'='*70}
""")

    config = BAB64Config()
    renderer = BabelRenderer(config)
    hasher = ImageHash(config)

    round_counts = [1, 2, 4, 8, 16, 32]
    # avalanche[r] = list of bit-diff values
    avalanche = {r: [] for r in round_counts}

    # Also track per-word diffusion: which of 8 state words changed?
    words_touched = {r: np.zeros(8) for r in round_counts}

    t0 = time.time()

    for trial in range(num_trials):
        # Generate image and derive hash params
        seed = hashlib.sha256(f"isolation_{trial}".encode()).digest()
        image = renderer.render(seed)
        rc = hasher._derive_round_constants(image)
        rot = hasher._derive_rotations(image)
        sbox = hasher._derive_sbox(image)
        state = hasher._derive_initial_state(image)

        # Random 32-byte message block
        block = hashlib.sha256(f"block_{trial}".encode()).digest()

        # Flip a random bit (vary across trials to avoid position bias)
        flip_pos = trial % 256  # cycle through all 256 bit positions
        block_flipped = flip_bit_in_block(block, flip_pos)

        for r in round_counts:
            out_orig = single_compress(hasher, state, block, rc, rot, sbox, r)
            out_flip = single_compress(hasher, state, block_flipped, rc, rot, sbox, r)

            diff = count_diff_bits(out_orig, out_flip)
            avalanche[r].append(diff)

            # Track which 32-bit words differ
            for w in range(8):
                if int(out_orig[w]) != int(out_flip[w]):
                    words_touched[r][w] += 1

        if (trial + 1) % 50 == 0:
            print(f"    ... {trial+1}/{num_trials} trials ({time.time()-t0:.1f}s)")

    elapsed = time.time() - t0

    # === Main table ===
    print(f"\n  SINGLE-BLOCK AVALANCHE (1-bit message flip → state diff)")
    print(f"  {'Rounds':>6} | {'Avg Bits':>9} | {'Avg %':>7} | "
          f"{'Min':>5} | {'Max':>5} | {'Std':>6} | {'< 25%':>6} | {'Status':>8}")
    print(f"  {'-'*6}-+-{'-'*9}-+-{'-'*7}-+-"
          f"{'-'*5}-+-{'-'*5}-+-{'-'*6}-+-{'-'*6}-+-{'-'*8}")

    for r in round_counts:
        v = np.array(avalanche[r])
        avg = v.mean()
        pct = avg / 256 * 100
        lo = v.min()
        hi = v.max()
        std = v.std()
        weak = np.sum(v < 64)  # trials with < 25% avalanche

        if pct >= 45:
            status = "GOOD"
        elif pct >= 30:
            status = "WEAK"
        elif pct >= 10:
            status = "POOR"
        else:
            status = "BROKEN"

        print(f"  {r:>6} | {avg:>9.1f} | {pct:>6.1f}% | "
              f"{lo:>5} | {hi:>5} | {std:>6.1f} | {weak:>6} | {status:>8}")

    # === Word-level diffusion ===
    print(f"\n  WORD-LEVEL DIFFUSION (how many of 8 state words changed, out of {num_trials})")
    print(f"  {'Rounds':>6} | ", end="")
    for w in range(8):
        print(f"{'W'+str(w):>6}", end=" | ")
    print(f"{'AvgWords':>9}")

    print(f"  {'-'*6}-+-", end="")
    for w in range(8):
        print(f"{'-'*6}-+-", end="")
    print(f"{'-'*9}")

    for r in round_counts:
        print(f"  {r:>6} | ", end="")
        total_words = 0
        for w in range(8):
            pct = words_touched[r][w] / num_trials * 100
            total_words += words_touched[r][w]
            print(f"{pct:>5.0f}% | ", end="")
        avg_words = total_words / num_trials
        print(f"{avg_words:>8.1f}")

    # === Bit-position analysis at low rounds ===
    print(f"\n  BIT-FLIP POSITION MATTERS? (round=1, grouped by which message word)")
    # At 1 round, only w[0] (bits 0-31) gets XOR'd. Bits in other words
    # should produce zero change if the round function doesn't propagate.
    r1_by_word = {i: [] for i in range(8)}
    for trial in range(num_trials):
        flip_pos = trial % 256
        word_idx = flip_pos // 32  # which 32-bit message word
        r1_by_word[word_idx].append(avalanche[1][trial])

    print(f"  {'MsgWord':>8} | {'Flips':>6} | {'Avg Bits':>9} | {'Avg %':>7} | Note")
    print(f"  {'-'*8}-+-{'-'*6}-+-{'-'*9}-+-{'-'*7}-+------")
    for w in range(8):
        vals = r1_by_word[w]
        if vals:
            v = np.array(vals)
            note = "w[r%8]=w[0] at r=0" if w == 0 else "NOT mixed at round 0"
            print(f"  {'W'+str(w):>8} | {len(vals):>6} | {v.mean():>9.1f} | "
                  f"{v.mean()/256*100:>6.1f}% | {note}")

    # === Diagnosis ===
    r1_avg = np.mean(avalanche[1])
    r1_pct = r1_avg / 256 * 100
    r32_avg = np.mean(avalanche[32])
    r32_pct = r32_avg / 256 * 100

    print(f"\n{'='*70}")
    print(f"  DIAGNOSIS")
    print(f"{'='*70}")

    if r1_pct < 20:
        print(f"""
  Single-block, 1 round: {r1_pct:.1f}% avalanche — POOR
  Single-block, 32 rounds: {r32_pct:.1f}% avalanche

  CONCLUSION: The round function itself provides meaningful diffusion.
  The 32 rounds ARE needed — removing them would break the hash.
  The earlier stress test showing 50% at 1 round was an artifact of
  128-block Merkle-Damgard chaining masking the weak single-block behavior.

  This is actually GOOD NEWS for the design: rounds serve their purpose.
  However, reviewers may ask: how many rounds are truly necessary?
  Look at the table above to find the minimum safe round count.""")
    elif r1_pct < 40:
        print(f"""
  Single-block, 1 round: {r1_pct:.1f}% avalanche — PARTIAL
  Single-block, 32 rounds: {r32_pct:.1f}% avalanche

  CONCLUSION: Some diffusion exists at 1 round but it's incomplete.
  The message-word scheduling (w[r%8]) means only 1 of 8 words
  participates per round. Full message coverage requires >= 8 rounds.
  The 32-round count provides a ~4x safety margin over minimum.""")
    else:
        print(f"""
  Single-block, 1 round: {r1_pct:.1f}% avalanche — HIGH
  Single-block, 32 rounds: {r32_pct:.1f}% avalanche

  CONCLUSION: Even single-block compression achieves good avalanche
  quickly. The round function's combination of S-box, rotation,
  XOR, and modular addition provides strong per-call diffusion.
  The Davies-Meyer feedforward (s[i] + state[i]) contributes.
  Rounds beyond ~4 may be security margin rather than necessity.""")

    # === Davies-Meyer contribution ===
    print(f"\n  DAVIES-MEYER FEEDFORWARD CONTRIBUTION")
    print(f"  The compression does: result[i] = s[i] + state[i]")
    print(f"  If s[] barely changes from a 1-bit flip, the feedforward")
    print(f"  preserves that small change. If s[] changes a lot,")
    print(f"  the addition propagates it. Testing without feedforward:")

    # Quick test: same setup but skip the feedforward
    no_ff_diffs = {r: [] for r in [1, 4, 32]}
    for trial in range(min(50, num_trials)):
        seed = hashlib.sha256(f"isolation_{trial}".encode()).digest()
        image = renderer.render(seed)
        rc = hasher._derive_round_constants(image)
        rot = hasher._derive_rotations(image)
        sbox = hasher._derive_sbox(image)
        state = hasher._derive_initial_state(image)
        block = hashlib.sha256(f"block_{trial}".encode()).digest()
        flip_pos = trial % 256
        block_flipped = flip_bit_in_block(block, flip_pos)

        for r in [1, 4, 32]:
            original = hasher.config.num_rounds
            hasher.config.num_rounds = r

            # Manually run compression without feedforward
            # (just the round loop, no state addition)
            def raw_rounds(blk):
                w = np.zeros(8, dtype=np.uint32)
                for ii in range(min(8, len(blk) // 4)):
                    w[ii] = int.from_bytes(blk[ii*4:(ii+1)*4], 'big')
                s = state.copy()
                for rr in range(r):
                    rcc = int(rc[rr % len(rc)])
                    rott = int(rot[rr % len(rot)])
                    word_bytes = int(s[0]).to_bytes(4, 'big')
                    subst_bytes = bytes([sbox[b] for b in word_bytes])
                    s[0] = np.uint32(int.from_bytes(subst_bytes, 'big'))
                    s[0] = np.uint32(hasher._rotr32(int(s[0]), rott))
                    s[0] = np.uint32(int(s[0]) ^ rcc ^ int(w[rr % 8]))
                    s[0] = np.uint32((int(s[0]) + int(s[1])) & 0xFFFFFFFF)
                    maj = (int(s[0]) & int(s[1])) ^ \
                          (int(s[0]) & int(s[2])) ^ \
                          (int(s[1]) & int(s[2]))
                    s[3] = np.uint32((int(s[3]) + maj) & 0xFFFFFFFF)
                    ch = (int(s[4]) & int(s[5])) ^ \
                         ((~int(s[4]) & 0xFFFFFFFF) & int(s[6]))
                    s[7] = np.uint32((int(s[7]) + ch + rcc) & 0xFFFFFFFF)
                    s = np.roll(s, 1)
                return s

            hasher.config.num_rounds = original
            s_orig = raw_rounds(block)
            s_flip = raw_rounds(block_flipped)
            no_ff_diffs[r].append(count_diff_bits(s_orig, s_flip))

    print(f"  {'Rounds':>6} | {'With FF':>9} | {'Without FF':>11} | {'FF adds':>9}")
    print(f"  {'-'*6}-+-{'-'*9}-+-{'-'*11}-+-{'-'*9}")
    for r in [1, 4, 32]:
        with_ff = np.mean(avalanche[r])
        without_ff = np.mean(no_ff_diffs[r])
        delta = with_ff - without_ff
        print(f"  {r:>6} | {with_ff:>8.1f} | {without_ff:>10.1f} | {delta:>+8.1f}")

    print(f"\n  Total analysis time: {time.time()-t0:.1f}s")
    print(f"{'='*70}\n")


if __name__ == "__main__":
    run_analysis(num_trials=200)
