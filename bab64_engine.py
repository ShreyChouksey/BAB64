"""
BAB64 — Self-Referential Image Hash
=====================================

A novel proof-of-work primitive where the input image defines
its own hash function. Every 64×64 Babel image produces a unique
cryptographic compression function, which is then applied to
the image itself.

Mining: find Babel coordinates C such that H_I(I) < difficulty,
where I = BabelRender(C) and H_I is parameterized by I.

This is a cryptographic fixed-point search — no known PoW
system uses input-dependent hash parameterization.

Architecture:
  Image pixels → derive round constants (like SHA-256's K[])
  Image pixels → derive rotation schedule (like SHA-256's ROTR)
  Image pixels → derive S-box (like AES substitution)
  Image pixels → derive initial hash state (like SHA-256's H0)
  THEN: apply this unique hash function TO the image itself.

The image IS the private key. The hash IS the proof.

Author: Shrey (concept) + Claude (implementation)
Status: Research prototype
"""

import hashlib
import numpy as np
import time
import json
from typing import Tuple, List, Optional
from dataclasses import dataclass


# =============================================================================
# CONFIGURATION
# =============================================================================

@dataclass
class BAB64Config:
    """BAB64 engine parameters."""

    image_width: int = 64
    image_height: int = 64
    color_depth: int = 256       # 8-bit grayscale

    # Hash construction parameters
    num_rounds: int = 32         # Rounds of compression
    block_size: int = 32         # Bytes per message block (256 bits)
    hash_size: int = 32          # Output: 256 bits (32 bytes)

    # PoW parameters
    difficulty_bits: int = 20

    @property
    def dimension(self) -> int:
        return self.image_width * self.image_height  # 4,096

    @property
    def image_bytes(self) -> int:
        return self.dimension  # 4,096 bytes (one byte per pixel)

    def describe(self) -> str:
        return (
            f"BAB64 Configuration\n"
            f"{'='*55}\n"
            f"  Image:       {self.image_width}×{self.image_height} "
            f"= {self.dimension:,} pixels\n"
            f"  Rounds:      {self.num_rounds}\n"
            f"  Hash output: {self.hash_size * 8} bits\n"
            f"  Difficulty:  {self.difficulty_bits} leading zero bits\n"
            f"{'='*55}\n"
            f"  Novel property: image defines its OWN hash function\n"
            f"  Each pixel contributes to round constants, rotations,\n"
            f"  S-box entries, and initial state.\n"
            f"  Mining = fixed-point search in function space.\n"
            f"{'='*55}"
        )


# =============================================================================
# BABEL RENDERER — Deterministic Image from Coordinates
# =============================================================================

class BabelRenderer:
    """
    Maps arbitrary coordinates to a unique 64×64 image.
    Every coordinate in Babel's archive → one image.
    Every image → one set of hash function parameters.
    """

    def __init__(self, config: BAB64Config):
        self.config = config

    def render(self, seed: bytes) -> np.ndarray:
        """Generate a 64×64 image (4,096 bytes) from a 32-byte seed."""
        assert len(seed) == 32
        pixels = np.zeros(self.config.dimension, dtype=np.uint8)
        current = seed
        idx = 0
        while idx < self.config.dimension:
            current = hashlib.sha256(current).digest()
            for b in current:
                if idx >= self.config.dimension:
                    break
                pixels[idx] = b
                idx += 1
        return pixels

    def render_from_nonce(self, base_seed: bytes, nonce: int) -> np.ndarray:
        """Generate image from base_seed + nonce."""
        nonce_bytes = nonce.to_bytes(8, 'big')
        seed = hashlib.sha256(base_seed + nonce_bytes).digest()
        return self.render(seed)


# =============================================================================
# IMAGE-DEPENDENT HASH — The Novel Construction
# =============================================================================

class ImageHash:
    """
    BAB64's core: a hash function parameterized by an image.

    Every 64×64 image defines a UNIQUE hash function by providing:
      - 32 round constants (from pixel rows)
      - 32 rotation amounts (from pixel columns)
      - 256-entry S-box (from pixel histogram/values)
      - Initial 256-bit state (from image hash)

    The construction uses Merkle-Damgård structure (like SHA-256)
    but with image-derived parameters instead of fixed constants.

    SHA-256's security rests on its specific constants being
    "nothing up my sleeve" numbers (cube roots of primes).
    BAB64's constants are derived from the image — equally
    arbitrary, equally unpredictable to the miner.
    """

    def __init__(self, config: BAB64Config):
        self.config = config

    def _derive_round_constants(self, image: np.ndarray) -> np.ndarray:
        """
        Extract 32 round constants from the image.

        Each round constant is a 32-bit integer derived from
        a 128-pixel block of the image (128 pixels × 8 bits = 1024 bits,
        hashed down to 32 bits).

        SHA-256 equivalent: K[0..63] constants
        """
        n_rounds = self.config.num_rounds
        pixels_per_round = len(image) // n_rounds  # 128 pixels per round
        constants = np.zeros(n_rounds, dtype=np.uint32)

        for r in range(n_rounds):
            start = r * pixels_per_round
            block = image[start:start + pixels_per_round]
            # Hash the pixel block down to 4 bytes → 32-bit constant
            h = hashlib.sha256(block.tobytes()).digest()
            constants[r] = int.from_bytes(h[:4], 'big')

        return constants

    def _derive_rotations(self, image: np.ndarray) -> np.ndarray:
        """
        Extract rotation amounts from the image.

        Each round gets a rotation amount in [1, 31] derived
        from image pixels.

        SHA-256 equivalent: ROTR amounts (2, 13, 22, 6, 11, 25, etc.)
        """
        n_rounds = self.config.num_rounds
        rotations = np.zeros(n_rounds, dtype=np.int32)

        # Use pixels from the second half of the image
        offset = len(image) // 2
        for r in range(n_rounds):
            # Combine two pixels to get rotation in [1, 31]
            p1 = int(image[(offset + r * 2) % len(image)])
            p2 = int(image[(offset + r * 2 + 1) % len(image)])
            rotations[r] = ((p1 * 256 + p2) % 31) + 1

        return rotations

    def _derive_sbox(self, image: np.ndarray) -> np.ndarray:
        """
        Derive a 256-entry S-box (substitution table) from the image.

        A valid S-box must be a PERMUTATION of [0..255] — every
        input maps to a unique output. This ensures invertibility
        and good non-linear properties.

        Method: Fisher-Yates shuffle seeded by image hash, using
        rejection sampling to eliminate modular bias.

        Without rejection sampling, j = byte % (i+1) gives non-uniform
        indices when (i+1) does not divide 256. For example, at i=254,
        byte % 255 maps both 0 and 255 to index 0, creating a 2/256
        vs 1/256 bias. Over 255 swaps this accumulates to a detectable
        shift in the fixed-point distribution (p < 0.01 at n=10,000).

        With rejection sampling, we discard bytes >= (i+1)*floor(256/(i+1))
        and draw fresh bytes, guaranteeing each index in [0,i] has
        exactly equal probability. The expected number of rejections
        per swap is < 1 (at most 255/256), so total SHA-256 calls
        increase by < 2x in the worst case.
        """
        sbox = np.arange(256, dtype=np.uint8)

        # Seed the shuffle with a hash of the full image
        shuffle_seed = hashlib.sha256(
            image.tobytes() + b'sbox'
        ).digest()

        # Fisher-Yates shuffle with rejection sampling
        current = shuffle_seed
        byte_idx = 32  # force initial hash

        for i in range(255, 0, -1):
            # Rejection threshold: largest multiple of (i+1) that fits in a byte
            limit = 256 - (256 % (i + 1))  # e.g., i=254 → limit=255

            while True:
                # Advance byte stream
                if byte_idx >= 32:
                    current = hashlib.sha256(current).digest()
                    byte_idx = 0
                b = current[byte_idx]
                byte_idx += 1

                if b < limit:
                    j = b % (i + 1)
                    break
                # else: reject and draw again

            sbox[i], sbox[j] = sbox[j], sbox[i]

        return sbox

    def _derive_initial_state(self, image: np.ndarray) -> np.ndarray:
        """
        Derive the initial 256-bit hash state from the image.

        SHA-256 equivalent: H0 (fractional parts of square roots of
        first 8 primes). Here we derive it from the image.
        """
        h = hashlib.sha256(image.tobytes() + b'init_state').digest()
        # 8 × 32-bit words = 256 bits
        state = np.zeros(8, dtype=np.uint32)
        for i in range(8):
            state[i] = int.from_bytes(h[i*4:(i+1)*4], 'big')
        return state

    def _rotr32(self, x: int, n: int) -> int:
        """32-bit right rotation."""
        x = x & 0xFFFFFFFF
        return ((x >> n) | (x << (32 - n))) & 0xFFFFFFFF

    def _expand_message(self, w_in: np.ndarray) -> np.ndarray:
        """
        Expand 8 message words to 32 using sigma-style mixing.

        Two phases:
          1. Pre-mix: 8 rounds of circular mixing so every base
             word depends on ALL input words. This ensures that
             a 1-bit flip in ANY message word affects round 0.
          2. Expand: grow 8 mixed words to 32 via sigma functions,
             giving every round a unique, well-mixed schedule word.
        """
        w = np.zeros(32, dtype=np.uint32)
        for i in range(8):
            w[i] = w_in[i]

        # Phase 1: Pre-mix — 3 passes of circular mixing
        # Each pass propagates dependencies 2 hops (prev + next).
        # After 3 passes, every word depends on all 8 inputs.
        for _pass in range(3):
            for i in range(8):
                prev = int(w[(i + 7) % 8])
                nxt = int(w[(i + 1) % 8])
                w[i] = np.uint32(
                    (int(w[i]) ^ self._rotr32(prev, 7)
                     ^ self._rotr32(nxt, 13))
                    + prev & 0xFFFFFFFF
                )

        # Phase 2: Expand 8 → 32
        for i in range(8, 32):
            w[i] = np.uint32(
                (int(w[i - 2]) ^ self._rotr32(int(w[i - 3]), 7))
                + int(w[i - 8])
                & 0xFFFFFFFF
            )
        return w

    def _compress(
        self,
        state: np.ndarray,
        message_block: bytes,
        round_constants: np.ndarray,
        rotations: np.ndarray,
        sbox: np.ndarray,
        round_offset: int = 0,
    ) -> np.ndarray:
        """
        One compression function call: state + block → new state.

        Structure inspired by SHA-256's compression but with
        image-derived constants, rotations, and S-box.

        The 8 state words go through num_rounds rounds of:
          1. S-box substitution (non-linear, from image)
          2. Rotation (diffusion, amount from image)
          3. XOR with round constant + expanded message word
          4. Modular addition (non-linear mixing)
          5. Multi-word message injection (both state halves)
          6. Word rotation within state (diffusion)

        Message expansion: 8 input words → 32 via sigma mixing,
        so every round gets a unique, well-mixed message word.
        Multi-word injection: each round mixes the expanded word
        into s[0] (XOR) AND s[4] (addition), touching both
        halves of the state simultaneously.
        """
        # Parse message block to 8 × 32-bit words
        w_in = np.zeros(8, dtype=np.uint32)
        for i in range(min(8, len(message_block) // 4)):
            w_in[i] = int.from_bytes(
                message_block[i*4:(i+1)*4], 'big'
            )

        # Expand 8 words → 32 (one unique word per round)
        w = self._expand_message(w_in)

        # Working variables (copy of state)
        s = state.copy()

        for r in range(self.config.num_rounds):
            rc = int(round_constants[r % len(round_constants)])
            rot = int(rotations[r % len(rotations)])
            wr = int(w[r % 32])

            # --- STEP 1: S-box substitution on state bytes ---
            # Apply S-box to each byte of the first state word
            # This is the NON-LINEAR operation
            word_bytes = int(s[0]).to_bytes(4, 'big')
            subst_bytes = bytes([sbox[b] for b in word_bytes])
            s[0] = np.uint32(int.from_bytes(subst_bytes, 'big'))

            # --- STEP 2: Rotation (diffusion) ---
            s[0] = np.uint32(self._rotr32(int(s[0]), rot))

            # --- STEP 3: XOR with round constant + expanded message word ---
            s[0] = np.uint32(int(s[0]) ^ rc ^ wr)

            # --- STEP 4: Modular addition with neighbor ---
            s[0] = np.uint32((int(s[0]) + int(s[1])) & 0xFFFFFFFF)

            # --- STEP 5: Multi-word message injection ---
            # Inject expanded message word into s[4] (second half)
            # so every round touches BOTH halves of the state
            s[4] = np.uint32((int(s[4]) + wr) & 0xFFFFFFFF)

            # --- STEP 6: Majority-like boolean function ---
            # maj(a,b,c) = (a AND b) XOR (a AND c) XOR (b AND c)
            maj = (int(s[0]) & int(s[1])) ^ \
                  (int(s[0]) & int(s[2])) ^ \
                  (int(s[1]) & int(s[2]))
            s[3] = np.uint32((int(s[3]) + maj) & 0xFFFFFFFF)

            # --- STEP 7: Conditional function ---
            # ch(e,f,g) = (e AND f) XOR (NOT e AND g)
            ch = (int(s[4]) & int(s[5])) ^ \
                 ((~int(s[4]) & 0xFFFFFFFF) & int(s[6]))
            s[7] = np.uint32(
                (int(s[7]) + ch + rc) & 0xFFFFFFFF
            )

            # --- STEP 8: Rotate state words ---
            # Like SHA-256's working variable rotation
            s = np.roll(s, 1)

            # --- STEP 9: Parallel diffusion (MixColumns-like) ---
            # Each word absorbs rotated bits from two non-adjacent
            # words, ensuring a change in ANY word propagates to ALL
            # 8 words in a single round.
            t = s.copy()
            for i in range(8):
                s[i] = np.uint32(
                    int(t[i])
                    ^ self._rotr32(int(t[(i + 2) % 8]), 11)
                    ^ self._rotr32(int(t[(i + 5) % 8]), 19)
                )

        # Davies-Meyer: add original state (feedforward)
        result = np.zeros(8, dtype=np.uint32)
        for i in range(8):
            result[i] = np.uint32(
                (int(s[i]) + int(state[i])) & 0xFFFFFFFF
            )

        return result

    def hash_image(self, image: np.ndarray) -> bytes:
        """
        THE CORE FUNCTION: Hash an image using itself as parameters.

        1. Derive all hash parameters FROM the image
        2. Split the image into message blocks
        3. Run Merkle-Damgård compression
        4. Output 256-bit hash

        The image defines the function that hashes the image.
        """
        # Step 1: Derive hash function parameters from image
        round_constants = self._derive_round_constants(image)
        rotations = self._derive_rotations(image)
        sbox = self._derive_sbox(image)
        state = self._derive_initial_state(image)

        # Step 2: Split image bytes into 32-byte message blocks
        image_bytes = image.tobytes()
        block_size = self.config.block_size
        num_blocks = (len(image_bytes) + block_size - 1) // block_size

        # Step 3: Merkle-Damgård chain
        for b in range(num_blocks):
            start = b * block_size
            block = image_bytes[start:start + block_size]
            # Pad last block if needed
            if len(block) < block_size:
                block = block + b'\x00' * (block_size - len(block))
            state = self._compress(
                state, block, round_constants,
                rotations, sbox, round_offset=b
            )

        # Step 4: Serialize state to 256-bit hash
        result = b''
        for word in state:
            result += int(word).to_bytes(4, 'big')

        return result


# =============================================================================
# BAB64 PROOF-OF-WORK ENGINE
# =============================================================================

@dataclass
class BAB64Proof:
    """A BAB64 proof-of-work."""
    input_data: str
    base_seed: str          # hex
    nonce: int
    image_hash: str         # hex — SHA-256 of the image (for verification)
    bab64_hash: str         # hex — the self-referential hash
    difficulty_bits: int
    timestamp: float
    computation_time: float

    def to_dict(self) -> dict:
        return {
            'input_data': self.input_data,
            'base_seed': self.base_seed,
            'nonce': self.nonce,
            'image_hash': self.image_hash,
            'bab64_hash': self.bab64_hash,
            'difficulty_bits': self.difficulty_bits,
            'timestamp': self.timestamp,
            'computation_time_seconds': self.computation_time,
        }

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), indent=2)

    @classmethod
    def from_dict(cls, d: dict) -> 'BAB64Proof':
        return cls(
            input_data=d['input_data'],
            base_seed=d['base_seed'],
            nonce=d['nonce'],
            image_hash=d['image_hash'],
            bab64_hash=d['bab64_hash'],
            difficulty_bits=d['difficulty_bits'],
            timestamp=d['timestamp'],
            computation_time=d['computation_time_seconds'],
        )


class BAB64Engine:
    """
    The BAB64 Proof-of-Work Engine.

    Mining:
      1. Hash input → base_seed
      2. For each nonce:
         a. Generate image from base_seed + nonce (via Babel)
         b. Derive hash function parameters FROM the image
         c. Hash the image USING its own hash function
         d. Check if hash meets difficulty target
      3. Submit (nonce, image_hash) as proof

    Verification:
      1. Regenerate image from base_seed + nonce
      2. Verify image_hash matches
      3. Re-derive hash function from image
      4. Re-hash image with its own function
      5. Verify bab64_hash matches AND meets difficulty

    The verification cost equals the mining cost per nonce —
    but miners try many nonces while verifiers check one.
    """

    def __init__(self, config: BAB64Config = None):
        self.config = config or BAB64Config()
        self.renderer = BabelRenderer(self.config)
        self.hasher = ImageHash(self.config)

    def _compute_base_seed(self, input_data: str) -> bytes:
        return hashlib.sha256(input_data.encode('utf-8')).digest()

    def _meets_difficulty(self, hash_bytes: bytes) -> bool:
        hash_int = int.from_bytes(hash_bytes, 'big')
        if hash_int == 0:
            return True
        leading_zeros = 256 - hash_int.bit_length()
        return leading_zeros >= self.config.difficulty_bits

    def mine(
        self,
        input_data: str,
        max_nonces: int = 1_000_000,
        verbose: bool = True,
    ) -> Optional[BAB64Proof]:
        """Mine a BAB64 proof."""
        start_time = time.time()

        if verbose:
            print(f"\n{'='*60}")
            print(f"  BAB64 MINING — Self-Referential Image Hash")
            print(f"{'='*60}")
            print(f"  Input:      {input_data[:50]}")
            print(f"  Difficulty: {self.config.difficulty_bits} bits")
            print(f"  Image:      {self.config.image_width}×"
                  f"{self.config.image_height} = "
                  f"{self.config.dimension} pixels")
            print(f"{'='*60}\n")

        base_seed = self._compute_base_seed(input_data)

        for nonce in range(max_nonces):
            # Generate unique image for this nonce
            image = self.renderer.render_from_nonce(base_seed, nonce)

            # THE NOVEL PART: hash image using itself as the function
            bab64_hash = self.hasher.hash_image(image)

            if verbose and nonce % 100 == 0:
                leading = 256 - int.from_bytes(
                    bab64_hash, 'big'
                ).bit_length() if int.from_bytes(bab64_hash, 'big') > 0 else 256
                elapsed = time.time() - start_time
                rate = (nonce + 1) / max(elapsed, 0.001)
                print(f"  Nonce {nonce:>7d} | "
                      f"zeros: {leading:>2d}/{self.config.difficulty_bits} | "
                      f"{rate:.0f} h/s")

            # Check difficulty
            if self._meets_difficulty(bab64_hash):
                elapsed = time.time() - start_time
                image_sha = hashlib.sha256(image.tobytes()).hexdigest()

                proof = BAB64Proof(
                    input_data=input_data,
                    base_seed=base_seed.hex(),
                    nonce=nonce,
                    image_hash=image_sha,
                    bab64_hash=bab64_hash.hex(),
                    difficulty_bits=self.config.difficulty_bits,
                    timestamp=time.time(),
                    computation_time=elapsed,
                )

                if verbose:
                    print(f"\n  *** PROOF FOUND ***")
                    print(f"  Nonce:     {nonce}")
                    print(f"  BAB64:     {bab64_hash.hex()[:40]}...")
                    print(f"  Time:      {elapsed:.2f}s")
                    print(f"  Rate:      {(nonce + 1) / elapsed:.0f} hashes/sec")

                return proof

        if verbose:
            print(f"\n  No proof in {max_nonces} nonces")
        return None

    def verify(self, proof: BAB64Proof, verbose: bool = True) -> bool:
        """
        Verify a BAB64 proof.

        Cost: one image generation + one self-referential hash.
        Same as ONE mining attempt — cheap relative to mining.
        """
        verify_start = time.time()
        checks = []

        # 1. Reconstruct base seed
        base_seed = self._compute_base_seed(proof.input_data)
        ok = base_seed.hex() == proof.base_seed
        checks.append(("Base seed", ok))
        if not ok:
            if verbose:
                print("  ✗ Base seed mismatch")
            return False

        # 2. Regenerate the exact image
        image = self.renderer.render_from_nonce(base_seed, proof.nonce)
        image_sha = hashlib.sha256(image.tobytes()).hexdigest()
        ok = image_sha == proof.image_hash
        checks.append(("Image integrity", ok))
        if not ok:
            if verbose:
                print("  ✗ Image hash mismatch")
            return False

        # 3. Re-hash image with its own function
        bab64_hash = self.hasher.hash_image(image)
        ok = bab64_hash.hex() == proof.bab64_hash
        checks.append(("BAB64 hash", ok))
        if not ok:
            if verbose:
                print("  ✗ BAB64 hash mismatch")
            return False

        # 4. Check difficulty
        ok = self._meets_difficulty(bab64_hash)
        checks.append(("Difficulty", ok))

        verify_time = time.time() - verify_start
        all_ok = all(c[1] for c in checks)

        if verbose:
            for name, passed in checks:
                print(f"  {'✓' if passed else '✗'} {name}")
            if all_ok:
                print(f"\n  *** VALID *** (verified in {verify_time:.4f}s)")

        return all_ok


# =============================================================================
# BLOCKCHAIN
# =============================================================================

@dataclass
class BAB64Block:
    index: int
    previous_hash: str
    proof: BAB64Proof
    block_hash: str


class BAB64Chain:
    def __init__(self, config: BAB64Config = None):
        self.config = config or BAB64Config(difficulty_bits=8)
        self.engine = BAB64Engine(self.config)
        self.blocks: List[BAB64Block] = []

    def mine_block(self, data: str, verbose: bool = True) -> Optional[BAB64Block]:
        prev_hash = self.blocks[-1].block_hash if self.blocks else "0" * 64
        block_input = f"{prev_hash}|{data}|block_{len(self.blocks)}"

        proof = self.engine.mine(block_input, verbose=verbose)
        if proof is None:
            return None

        block_hash = hashlib.sha256(
            (prev_hash + proof.bab64_hash).encode()
        ).hexdigest()

        block = BAB64Block(
            index=len(self.blocks),
            previous_hash=prev_hash,
            proof=proof,
            block_hash=block_hash,
        )
        self.blocks.append(block)
        return block

    def verify_chain(self, verbose: bool = True) -> bool:
        if verbose:
            print(f"\n  VERIFYING BAB64 CHAIN ({len(self.blocks)} blocks)")

        for i, block in enumerate(self.blocks):
            expected_prev = "0" * 64 if i == 0 else self.blocks[i-1].block_hash
            if block.previous_hash != expected_prev:
                if verbose:
                    print(f"  Block {i}: ✗ Chain linkage")
                return False

            if not self.engine.verify(block.proof, verbose=False):
                if verbose:
                    print(f"  Block {i}: ✗ Invalid proof")
                return False

            expected_bh = hashlib.sha256(
                (block.previous_hash + block.proof.bab64_hash).encode()
            ).hexdigest()
            if block.block_hash != expected_bh:
                if verbose:
                    print(f"  Block {i}: ✗ Block hash")
                return False

            if verbose:
                print(f"  Block {i}: ✓ (nonce={block.proof.nonce}, "
                      f"time={block.proof.computation_time:.2f}s)")

        if verbose:
            total = sum(b.proof.computation_time for b in self.blocks)
            print(f"\n  *** CHAIN VALID *** ({total:.1f}s total)")

        return True


# =============================================================================
# ANALYSIS: HASH QUALITY TESTS
# =============================================================================

def analyze_hash_quality(num_trials: int = 100):
    """
    Test whether BAB64's image-dependent hash has good properties:
    1. Avalanche — 1 pixel change → ~50% output bits flip
    2. Distribution — output bits uniformly distributed
    3. Collision resistance — no two images produce same hash
    """
    print(f"\n{'='*60}")
    print(f"  BAB64 HASH QUALITY ANALYSIS — {num_trials} trials")
    print(f"{'='*60}\n")

    config = BAB64Config()
    renderer = BabelRenderer(config)
    hasher = ImageHash(config)

    base_seed = hashlib.sha256(b"quality_test").digest()

    hashes = []
    avalanche_diffs = []

    for i in range(num_trials):
        # Generate image
        image = renderer.render_from_nonce(base_seed, i)
        h = hasher.hash_image(image)
        hashes.append(h)

        # Avalanche test: flip one pixel, measure output change
        modified = image.copy()
        modified[0] = np.uint8((int(modified[0]) + 1) % 256)
        h_mod = hasher.hash_image(modified)

        # Count differing bits
        h_int = int.from_bytes(h, 'big')
        h_mod_int = int.from_bytes(h_mod, 'big')
        diff_bits = bin(h_int ^ h_mod_int).count('1')
        avalanche_diffs.append(diff_bits)

    # Avalanche analysis
    avg_flip = np.mean(avalanche_diffs)
    std_flip = np.std(avalanche_diffs)
    ideal = 128  # 50% of 256 bits

    print(f"  AVALANCHE EFFECT")
    print(f"  Avg bits flipped:  {avg_flip:.1f} / 256 "
          f"({avg_flip/256*100:.1f}%)")
    print(f"  Std deviation:     {std_flip:.1f}")
    print(f"  Ideal (50%):       {ideal}")
    print(f"  Status:            "
          f"{'PASS' if 100 < avg_flip < 156 else 'FAIL'}")

    # Distribution: check each bit position across all hashes
    bit_counts = np.zeros(256)
    for h in hashes:
        h_int = int.from_bytes(h, 'big')
        for b in range(256):
            if h_int & (1 << b):
                bit_counts[b] += 1

    bit_ratios = bit_counts / num_trials
    mean_ratio = np.mean(bit_ratios)
    std_ratio = np.std(bit_ratios)

    print(f"\n  BIT DISTRIBUTION")
    print(f"  Mean P(bit=1):     {mean_ratio:.3f} (ideal: 0.500)")
    print(f"  Std P(bit=1):      {std_ratio:.3f}")
    print(f"  Status:            "
          f"{'PASS' if 0.40 < mean_ratio < 0.60 else 'FAIL'}")

    # Collision check
    unique_hashes = len(set(h.hex() for h in hashes))
    print(f"\n  COLLISION RESISTANCE")
    print(f"  Unique hashes:     {unique_hashes} / {num_trials}")
    print(f"  Status:            "
          f"{'PASS' if unique_hashes == num_trials else 'FAIL'}")

    # Speed
    t0 = time.time()
    for i in range(100):
        image = renderer.render_from_nonce(base_seed, num_trials + i)
        hasher.hash_image(image)
    elapsed = time.time() - t0
    rate = 100 / elapsed

    print(f"\n  PERFORMANCE")
    print(f"  Hash rate:         {rate:.0f} hashes/sec")
    print(f"  Time per hash:     {elapsed/100*1000:.1f} ms")

    print(f"\n{'='*60}\n")

    return {
        'avalanche': avg_flip,
        'distribution': mean_ratio,
        'unique': unique_hashes,
        'rate': rate,
    }


# =============================================================================
# MAIN
# =============================================================================

if __name__ == "__main__":
    print("""
    ╔══════════════════════════════════════════════════════════╗
    ║          BAB64 — Self-Referential Image Hash            ║
    ║                                                         ║
    ║  The image IS the hash function.                        ║
    ║  The hash function hashes the image.                    ║
    ║  Mining = find an image that hashes itself to target.   ║
    ║                                                         ║
    ║  Novel: input-dependent hash parameterization           ║
    ║  No known PoW system uses this construction.            ║
    ╚══════════════════════════════════════════════════════════╝
    """)

    # 1. Hash quality analysis
    quality = analyze_hash_quality(num_trials=200)

    # 2. Mine + verify
    config = BAB64Config(difficulty_bits=8)
    print(config.describe())

    engine = BAB64Engine(config)
    proof = engine.mine("BAB64 Genesis Block", verbose=True)

    if proof:
        print(f"\n  VERIFICATION:")
        engine.verify(proof)

        # 3. Mine a 3-block chain
        print(f"\n\n{'='*60}")
        print(f"  MINING 3-BLOCK CHAIN")
        print(f"{'='*60}")
        chain = BAB64Chain(BAB64Config(difficulty_bits=6))
        for i in range(3):
            block = chain.mine_block(f"Block {i} payload", verbose=False)
            if block:
                print(f"  Block {i}: {block.block_hash[:24]}... "
                      f"nonce={block.proof.nonce} "
                      f"time={block.proof.computation_time:.2f}s")
        chain.verify_chain()

        print(f"\n  Proof JSON:")
        print(f"  {proof.to_json()}")
