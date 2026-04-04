"""
BAB64 Reference Implementation
Built from BAB64_SPECIFICATION.md v1.0
Independent implementation for specification validation.
"""

import hashlib
import struct

# --- Constants ---
IMG_W = 64
IMG_H = 64
PIXEL_COUNT = IMG_W * IMG_H  # 4096
NUM_ROUNDS = 32
BLOCK_BYTES = 32
HASH_BYTES = 32
NUM_STATE_WORDS = 8
MASK32 = 0xFFFFFFFF


def _sha256_digest(data: bytes) -> bytes:
    return hashlib.sha256(data).digest()


def _u32_from_be(buf: bytes, pos: int) -> int:
    return struct.unpack('>I', buf[pos:pos+4])[0]


def _right_rotate(val: int, amount: int) -> int:
    return ((val >> amount) | (val << (32 - amount))) & MASK32


# --- 2.3 Image Generation ---

def create_image(input_text: str, nonce: int) -> bytes:
    seed_base = _sha256_digest(input_text.encode('utf-8'))
    seed_img = _sha256_digest(seed_base + nonce.to_bytes(8, 'big'))

    pixels = bytearray(PIXEL_COUNT)
    chain = seed_img
    written = 0
    while written < PIXEL_COUNT:
        chain = _sha256_digest(chain)
        for octet in chain:
            if written >= PIXEL_COUNT:
                break
            pixels[written] = octet
            written += 1
    return bytes(pixels)


# --- 2.4 Hash Function Derivation ---

def _derive_round_keys(img: bytes) -> list:
    chunk_len = PIXEL_COUNT // NUM_ROUNDS  # 128
    keys = []
    for rnd in range(NUM_ROUNDS):
        segment = img[rnd * chunk_len:(rnd + 1) * chunk_len]
        digest = _sha256_digest(segment)
        keys.append(_u32_from_be(digest, 0))
    return keys


def _derive_rotations(img: bytes) -> list:
    half = PIXEL_COUNT // 2  # 2048
    rots = []
    for rnd in range(NUM_ROUNDS):
        idx_a = (half + rnd * 2) % PIXEL_COUNT
        idx_b = (half + rnd * 2 + 1) % PIXEL_COUNT
        combined = img[idx_a] * 256 + img[idx_b]
        rots.append((combined % 31) + 1)
    return rots


def _derive_substitution_table(img: bytes) -> list:
    table = list(range(256))
    init_hash = _sha256_digest(img + b'sbox')

    stream = init_hash
    pos = 255
    while pos >= 1:
        if pos % 32 == 0:
            stream = _sha256_digest(stream)
        pick = stream[pos % 32] % (pos + 1)
        table[pos], table[pick] = table[pick], table[pos]
        pos -= 1
    return table


def _derive_starting_state(img: bytes) -> list:
    digest = _sha256_digest(img + b'init_state')
    return [_u32_from_be(digest, i * 4) for i in range(NUM_STATE_WORDS)]


# --- 2.5 Compression Function ---

def _compress_block(state_in: list, msg_bytes: bytes,
                    round_keys: list, rotations: list,
                    sub_table: list) -> list:
    # 2.5.1 Message expansion
    mw = [_u32_from_be(msg_bytes, i * 4) for i in range(NUM_STATE_WORDS)]

    # 2.5.2 Round function
    work = list(state_in)

    for rnd in range(NUM_ROUNDS):
        rk = round_keys[rnd % NUM_ROUNDS]
        ra = rotations[rnd % NUM_ROUNDS]

        # Step 1: S-box on work[0] bytes
        b3 = (work[0] >> 24) & 0xFF
        b2 = (work[0] >> 16) & 0xFF
        b1 = (work[0] >> 8) & 0xFF
        b0 = work[0] & 0xFF
        work[0] = (sub_table[b3] << 24) | (sub_table[b2] << 16) | \
                  (sub_table[b1] << 8) | sub_table[b0]

        # Step 2: Rotate right
        work[0] = _right_rotate(work[0], ra)

        # Step 3: XOR with round key and message word
        work[0] = (work[0] ^ rk ^ mw[rnd % 8]) & MASK32

        # Step 4: Add neighbor
        work[0] = (work[0] + work[1]) & MASK32

        # Step 5: Majority
        majority = (work[0] & work[1]) ^ (work[0] & work[2]) ^ (work[1] & work[2])
        work[3] = (work[3] + majority) & MASK32

        # Step 6: Choice
        choice = (work[4] & work[5]) ^ ((~work[4] & MASK32) & work[6])
        work[7] = (work[7] + choice + rk) & MASK32

        # Step 7: Rotate words
        work = [work[7], work[0], work[1], work[2],
                work[3], work[4], work[5], work[6]]

    # 2.5.3 Davies-Meyer feedforward
    return [(work[i] + state_in[i]) & MASK32 for i in range(NUM_STATE_WORDS)]


# --- 2.6 Full Hash ---

def compute_hash(img: bytes) -> bytes:
    round_keys = _derive_round_keys(img)
    rotations = _derive_rotations(img)
    sub_table = _derive_substitution_table(img)
    state = _derive_starting_state(img)

    total_blocks = PIXEL_COUNT // BLOCK_BYTES  # 128

    for blk in range(total_blocks):
        chunk = img[blk * BLOCK_BYTES:(blk + 1) * BLOCK_BYTES]
        state = _compress_block(state, chunk, round_keys, rotations, sub_table)

    # Serialize state as big-endian bytes
    result = b''
    for word in state:
        result += struct.pack('>I', word)
    return result


# --- 2.7 Mining ---

def mine(input_text: str, difficulty: int, max_attempts: int = 1_000_000):
    seed_base = _sha256_digest(input_text.encode('utf-8'))

    for attempt in range(max_attempts):
        img = create_image(input_text, attempt)
        digest = compute_hash(img)

        # Check leading zero bits
        zero_count = 0
        for octet in digest:
            if octet == 0:
                zero_count += 8
            else:
                mask = 0x80
                while mask and not (octet & mask):
                    zero_count += 1
                    mask >>= 1
                break

        if zero_count >= difficulty:
            img_hash = _sha256_digest(img)
            return {
                'input_data': input_text,
                'base_seed': seed_base.hex(),
                'nonce': attempt,
                'image_hash': img_hash.hex(),
                'bab64_hash': digest.hex(),
                'difficulty_bits': difficulty,
            }
    return None


# --- 2.8 Verification ---

def verify(proof: dict) -> bool:
    # Check 1: base_seed matches input_data
    recomputed_seed = _sha256_digest(proof['input_data'].encode('utf-8')).hex()
    if recomputed_seed != proof['base_seed']:
        return False

    # Check 2: image hash matches
    img = create_image(proof['input_data'], proof['nonce'])
    if _sha256_digest(img).hex() != proof['image_hash']:
        return False

    # Check 3: bab64 hash matches
    digest = compute_hash(img)
    if digest.hex() != proof['bab64_hash']:
        return False

    # Check 4: difficulty met
    zero_count = 0
    for octet in digest:
        if octet == 0:
            zero_count += 8
        else:
            mask = 0x80
            while mask and not (octet & mask):
                zero_count += 1
                mask >>= 1
            break

    return zero_count >= proof['difficulty_bits']
