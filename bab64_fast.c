/*
 * bab64_fast.c — C acceleration for BAB64 compression function
 *
 * Callable via ctypes from Python. Replaces ImageHash._compress
 * which is the hot loop (32 rounds × 128 blocks per hash).
 *
 * Compile:
 *   cc -O3 -shared -fPIC -o bab64_fast.so bab64_fast.c
 *   (macOS: cc -O3 -shared -fPIC -o bab64_fast.dylib bab64_fast.c)
 */

#include <stdint.h>
#include <string.h>

static inline uint32_t rotr32(uint32_t x, int n) {
    return (x >> n) | (x << (32 - n));
}

static inline uint32_t read_be32(const uint8_t *p) {
    return ((uint32_t)p[0] << 24) | ((uint32_t)p[1] << 16) |
           ((uint32_t)p[2] << 8)  | (uint32_t)p[3];
}

static inline void write_be32(uint8_t *p, uint32_t v) {
    p[0] = (uint8_t)(v >> 24);
    p[1] = (uint8_t)(v >> 16);
    p[2] = (uint8_t)(v >> 8);
    p[3] = (uint8_t)(v);
}

/*
 * expand_message:
 *   Expand 8 message words to 32 using sigma-style mixing.
 *   Phase 1: 3 passes of circular pre-mixing (all 8 depend on all 8).
 *   Phase 2: Expand 8 → 32 via sigma functions.
 *
 *   Must exactly match Python ImageHash._expand_message.
 *
 *   Python operator precedence note:
 *     (a ^ b ^ c) + d & MASK  parses as  ((a ^ b ^ c) + d) & MASK
 *     (a ^ b) + c & MASK      parses as  ((a ^ b) + c) & MASK
 *   because + binds tighter than & which binds tighter than ^,
 *   BUT the Python code has explicit parens around the XOR part,
 *   so the effective evaluation is:
 *     Phase 1: ((w[i] ^ rotr(prev,7) ^ rotr(nxt,13)) + prev) & 0xFFFFFFFF
 *     Phase 2: ((w[i-2] ^ rotr(w[i-3],7)) + w[i-8]) & 0xFFFFFFFF
 */
static void expand_message(const uint8_t *message_block, uint32_t *w) {
    /* Parse 8 big-endian words from message block */
    for (int i = 0; i < 8; i++) {
        w[i] = read_be32(message_block + i * 4);
    }

    /* Phase 1: Pre-mix — 3 passes of circular mixing */
    for (int pass = 0; pass < 3; pass++) {
        for (int i = 0; i < 8; i++) {
            uint32_t prev = w[(i + 7) % 8];
            uint32_t nxt  = w[(i + 1) % 8];
            w[i] = (w[i] ^ rotr32(prev, 7) ^ rotr32(nxt, 13)) + prev;
        }
    }

    /* Phase 2: Expand 8 → 32 */
    for (int i = 8; i < 32; i++) {
        w[i] = (w[i - 2] ^ rotr32(w[i - 3], 7)) + w[i - 8];
    }
}

/*
 * bab64_compress:
 *   state_in:        8 × uint32 (native byte order array)
 *   message_block:   32 bytes
 *   round_constants: num_rc × uint32 (native byte order array)
 *   rotations:       num_rot × int32 (native byte order array)
 *   sbox:            256 bytes
 *   num_rounds:      number of compression rounds
 *   state_out:       8 × uint32 (native byte order array)
 */
void bab64_compress(
    const uint32_t *state_in,
    const uint8_t  *message_block,
    const uint32_t *round_constants,
    const int32_t  *rotations,
    const uint8_t  *sbox,
    int             num_rounds,
    int             num_rc,
    int             num_rot,
    uint32_t       *state_out
) {
    /* Expand message: 8 input words → 32 via pre-mix + sigma expansion */
    uint32_t w[32];
    expand_message(message_block, w);

    /* Working copy of state */
    uint32_t s[8];
    for (int i = 0; i < 8; i++) {
        s[i] = state_in[i];
    }

    for (int r = 0; r < num_rounds; r++) {
        uint32_t rc = round_constants[r % num_rc];
        int rot = rotations[r % num_rot];
        uint32_t wr = w[r % 32];

        /* STEP 1: S-box substitution on s[0] bytes */
        uint8_t b0 = (uint8_t)(s[0] >> 24);
        uint8_t b1 = (uint8_t)(s[0] >> 16);
        uint8_t b2 = (uint8_t)(s[0] >> 8);
        uint8_t b3 = (uint8_t)(s[0]);
        s[0] = ((uint32_t)sbox[b0] << 24) | ((uint32_t)sbox[b1] << 16) |
               ((uint32_t)sbox[b2] << 8)  | (uint32_t)sbox[b3];

        /* STEP 2: Rotation */
        s[0] = rotr32(s[0], rot);

        /* STEP 3: XOR with round constant + expanded message word */
        s[0] ^= rc ^ wr;

        /* STEP 4: Modular addition with neighbor */
        s[0] = s[0] + s[1];

        /* STEP 5: Multi-word message injection into s[4] */
        s[4] = s[4] + wr;

        /* STEP 6: Majority function */
        uint32_t maj = (s[0] & s[1]) ^ (s[0] & s[2]) ^ (s[1] & s[2]);
        s[3] = s[3] + maj;

        /* STEP 7: Conditional (choice) function */
        uint32_t ch = (s[4] & s[5]) ^ ((~s[4]) & s[6]);
        s[7] = s[7] + ch + rc;

        /* STEP 8: Rotate state words (roll right by 1) */
        uint32_t tmp = s[7];
        for (int i = 7; i > 0; i--) {
            s[i] = s[i - 1];
        }
        s[0] = tmp;
    }

    /* Davies-Meyer feedforward */
    for (int i = 0; i < 8; i++) {
        state_out[i] = s[i] + state_in[i];
    }
}

/*
 * bab64_hash_image_full:
 *   Full Merkle-Damgård chain over all blocks. Avoids Python↔C overhead
 *   for each of the 128 block calls.
 *
 *   initial_state:   8 × uint32 (native order)
 *   image_bytes:     4096 bytes
 *   image_len:       length of image_bytes
 *   block_size:      bytes per block (32)
 *   round_constants: num_rc × uint32
 *   rotations:       num_rot × int32
 *   sbox:            256 bytes
 *   num_rounds:      rounds per compress call
 *   num_rc:          length of round_constants
 *   num_rot:         length of rotations
 *   state_out:       8 × uint32 result
 */
void bab64_hash_blocks(
    const uint32_t *initial_state,
    const uint8_t  *image_bytes,
    int             image_len,
    int             block_size,
    const uint32_t *round_constants,
    const int32_t  *rotations,
    const uint8_t  *sbox,
    int             num_rounds,
    int             num_rc,
    int             num_rot,
    uint32_t       *state_out
) {
    /* Copy initial state */
    uint32_t state[8];
    for (int i = 0; i < 8; i++) {
        state[i] = initial_state[i];
    }

    int num_blocks = (image_len + block_size - 1) / block_size;
    uint8_t block[64]; /* max block size we'd ever need */

    for (int b = 0; b < num_blocks; b++) {
        int start = b * block_size;
        int remaining = image_len - start;
        int copy_len = remaining < block_size ? remaining : block_size;

        memcpy(block, image_bytes + start, copy_len);
        /* Zero-pad if last block is short */
        if (copy_len < block_size) {
            memset(block + copy_len, 0, block_size - copy_len);
        }

        uint32_t new_state[8];
        bab64_compress(state, block, round_constants, rotations,
                       sbox, num_rounds, num_rc, num_rot, new_state);
        memcpy(state, new_state, sizeof(state));
    }

    for (int i = 0; i < 8; i++) {
        state_out[i] = state[i];
    }
}
