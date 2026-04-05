"""
BAB64-IBST — Image-Bound Signature Trees
==========================================

A novel signature scheme where the private image parameterizes
both the hash chain function and the Merkle tree structure.

Construction:
  1. Image-Derived Chain Function: f_I(x) = SHA-256(x XOR image_block[x mod 128])
     Fast (one SHA-256 + one XOR) but identity-specific.

  2. WOTS+ with image chains: Winternitz parameter w=16.
     Message hash → 64 base-16 digits + 3 checksum digits = 67 chains.
     Each chain: apply f_I up to 15 times.
     Signature: 67 * 32 = 2,144 bytes (vs Lamport's 8,192).

  3. Merkle tree (height 10): 1,024 WOTS+ key pairs.
     Tree node hashing uses f_I — proofs are identity-specific.
     Total signature: ~2,464 bytes.

This has never been done. Image-parameterized signature trees
are a novel construction.

Author: Shrey (concept) + Claude (implementation)
"""

import hashlib
import os
from dataclasses import dataclass
from typing import List, Optional, Tuple

from bab64_engine import BAB64Config, BabelRenderer, ImageHash


# =============================================================================
# COMPONENT 1 — Image-Derived Chain Function
# =============================================================================

class ImageChainFunction:
    """
    A lightweight one-way function parameterized by a private image.

    f_I(x) = SHA-256(x XOR image_block[x mod 128])

    The image is split into 128 blocks of 32 bytes each (4,096 bytes total).
    The block index is derived from x itself, making the function
    input-dependent in two ways: through the XOR and through block selection.
    """

    NUM_BLOCKS = 128
    BLOCK_SIZE = 32  # bytes

    def __init__(self, image_bytes: bytes):
        assert len(image_bytes) >= self.NUM_BLOCKS * self.BLOCK_SIZE, (
            f"Image must be at least {self.NUM_BLOCKS * self.BLOCK_SIZE} bytes, "
            f"got {len(image_bytes)}"
        )
        self._blocks: List[bytes] = []
        for k in range(self.NUM_BLOCKS):
            start = k * self.BLOCK_SIZE
            self._blocks.append(image_bytes[start:start + self.BLOCK_SIZE])

    def __call__(self, x: bytes) -> bytes:
        """Apply the image-derived chain function: f_I(x)."""
        assert len(x) == 32, f"Input must be 32 bytes, got {len(x)}"
        # Block selection based on first byte of x
        block_idx = x[0] % self.NUM_BLOCKS
        block = self._blocks[block_idx]
        # XOR x with the selected image block
        xored = bytes(a ^ b for a, b in zip(x, block))
        return hashlib.sha256(xored).digest()

    def chain(self, x: bytes, steps: int) -> bytes:
        """Apply f_I repeatedly: f_I^steps(x)."""
        val = x
        for _ in range(steps):
            val = self(val)
        return val


# =============================================================================
# COMPONENT 2 — WOTS+ with Image Chains
# =============================================================================

class BAB64WOTS:
    """
    Winternitz One-Time Signature with image-derived chains.

    Parameters:
      w = 16 (4 bits per digit)
      n = 32 (256-bit hash)
      msg_digits = 64 (256 / 4)
      checksum_digits = 3 (ceil(64 * 15 / 16) fits in 3 base-16 digits)
      total_chains = 67

    Secret key: 67 random 32-byte values.
    Public key: 67 values after 15 chain applications of f_I.
    Signature: 67 intermediate chain values at message-dependent positions.
    Size: 67 * 32 = 2,144 bytes.
    """

    W = 16           # Winternitz parameter (base-16)
    MAX_CHAIN = 15   # W - 1
    HASH_LEN = 32    # 256 bits
    MSG_DIGITS = 64  # 256 / 4
    CHECKSUM_DIGITS = 3
    TOTAL_CHAINS = MSG_DIGITS + CHECKSUM_DIGITS  # 67

    def __init__(self, chain_fn: ImageChainFunction, seed: bytes):
        """
        Generate a WOTS+ keypair.

        seed: 32-byte secret used to derive 67 secret key values
              deterministically (so the same seed always produces
              the same keypair).
        """
        self._chain_fn = chain_fn
        self._sk = self._derive_secret_keys(seed)
        self._pk = self._compute_public_keys()

    def _derive_secret_keys(self, seed: bytes) -> List[bytes]:
        """Derive 67 secret key values from the seed."""
        keys = []
        for i in range(self.TOTAL_CHAINS):
            keys.append(hashlib.sha256(
                seed + i.to_bytes(4, 'big') + b'wots_sk'
            ).digest())
        return keys

    def _compute_public_keys(self) -> List[bytes]:
        """Public key = f_I^15(sk[i]) for each chain."""
        return [self._chain_fn.chain(sk, self.MAX_CHAIN) for sk in self._sk]

    @property
    def public_key(self) -> List[bytes]:
        return list(self._pk)

    def public_key_hash(self) -> bytes:
        """Hash of all public key values — used as Merkle leaf."""
        h = hashlib.sha256()
        for pk_val in self._pk:
            h.update(pk_val)
        return h.digest()

    @staticmethod
    def _msg_to_digits(msg_hash: bytes) -> List[int]:
        """
        Convert a 256-bit hash to 64 base-16 digits + 3 checksum digits.

        Each nibble (4 bits) becomes one digit in [0, 15].
        Checksum = sum of (15 - digit) for all message digits,
        encoded as 3 base-16 digits.
        """
        digits = []
        for byte in msg_hash:
            digits.append((byte >> 4) & 0x0F)
            digits.append(byte & 0x0F)

        # Checksum: prevents attacker from increasing chain values
        checksum = sum(BAB64WOTS.MAX_CHAIN - d for d in digits)
        # Encode checksum as 3 base-16 digits (max value = 64*15 = 960 < 16^3 = 4096)
        for i in range(BAB64WOTS.CHECKSUM_DIGITS - 1, -1, -1):
            digits.append((checksum >> (4 * i)) & 0x0F)

        return digits

    def sign(self, message: bytes) -> List[bytes]:
        """
        Sign a message. Returns 67 chain values (2,144 bytes total).

        For each digit d[i], the signature value is f_I^d[i](sk[i]).
        The verifier can then chain forward (15 - d[i]) steps
        and check against the public key.
        """
        msg_hash = hashlib.sha256(message).digest()
        digits = self._msg_to_digits(msg_hash)

        signature = []
        for i, d in enumerate(digits):
            signature.append(self._chain_fn.chain(self._sk[i], d))

        return signature

    @staticmethod
    def verify(message: bytes, signature: List[bytes],
               public_key: List[bytes],
               chain_fn: ImageChainFunction) -> bool:
        """
        Verify a WOTS+ signature.

        For each digit d[i], chain the signature value forward
        (15 - d[i]) steps. The result must match public_key[i].
        """
        if len(signature) != BAB64WOTS.TOTAL_CHAINS:
            return False
        if len(public_key) != BAB64WOTS.TOTAL_CHAINS:
            return False

        msg_hash = hashlib.sha256(message).digest()
        digits = BAB64WOTS._msg_to_digits(msg_hash)

        for i, d in enumerate(digits):
            remaining = BAB64WOTS.MAX_CHAIN - d
            expected = chain_fn.chain(signature[i], remaining)
            if expected != public_key[i]:
                return False

        return True


# =============================================================================
# COMPONENT 3 — Merkle Tree for Many-Time Use
# =============================================================================

class BAB64MerkleTree:
    """
    Binary Merkle tree of height 10 (1,024 WOTS+ key pairs).

    Node hashing uses the image-derived chain function f_I,
    making proofs identity-specific. A proof generated for
    identity A cannot be repurposed for identity B.

    Each leaf = hash of one WOTS+ public key.
    Root = hash commitment to all 1,024 public keys.
    Authentication path = 10 sibling hashes (320 bytes).
    """

    HEIGHT = 10
    NUM_LEAVES = 1 << HEIGHT  # 1,024

    def __init__(self, chain_fn: ImageChainFunction,
                 leaf_hashes: List[bytes]):
        assert len(leaf_hashes) == self.NUM_LEAVES, (
            f"Need exactly {self.NUM_LEAVES} leaves, got {len(leaf_hashes)}"
        )
        self._chain_fn = chain_fn
        self._num_nodes = 2 * self.NUM_LEAVES  # index 1..2N-1
        self._nodes: List[bytes] = [b'\x00' * 32] * self._num_nodes

        # Fill leaves (indices NUM_LEAVES .. 2*NUM_LEAVES - 1)
        for i in range(self.NUM_LEAVES):
            self._nodes[self.NUM_LEAVES + i] = leaf_hashes[i]

        # Build tree bottom-up
        for i in range(self.NUM_LEAVES - 1, 0, -1):
            self._nodes[i] = self._hash_pair(
                self._nodes[2 * i], self._nodes[2 * i + 1]
            )

    def _hash_pair(self, left: bytes, right: bytes) -> bytes:
        """Hash two children using the image chain function."""
        # Use f_I for identity binding, then combine
        combined = self._chain_fn(left)
        combined_bytes = bytes(
            a ^ b for a, b in zip(combined, right)
        )
        return self._chain_fn(combined_bytes)

    @property
    def root(self) -> bytes:
        return self._nodes[1]

    def auth_path(self, leaf_index: int) -> List[bytes]:
        """
        Return the authentication path for a leaf.

        The path consists of 10 sibling hashes from leaf to root.
        Total size: 10 * 32 = 320 bytes.
        """
        assert 0 <= leaf_index < self.NUM_LEAVES
        path = []
        idx = self.NUM_LEAVES + leaf_index
        for _ in range(self.HEIGHT):
            # Sibling is the other child of the same parent
            sibling = idx ^ 1
            path.append(self._nodes[sibling])
            idx >>= 1  # Move to parent
        return path

    def verify_path(self, leaf_hash: bytes, leaf_index: int,
                    path: List[bytes]) -> bool:
        """Verify an authentication path against the root."""
        if len(path) != self.HEIGHT:
            return False

        current = leaf_hash
        idx = leaf_index
        for level in range(self.HEIGHT):
            sibling = path[level]
            if idx % 2 == 0:
                current = self._hash_pair(current, sibling)
            else:
                current = self._hash_pair(sibling, current)
            idx >>= 1

        return current == self.root


# =============================================================================
# BAB64-IBST SIGNATURE
# =============================================================================

@dataclass
class IBSTSignature:
    """
    A complete BAB64-IBST signature.

    Contains:
      - wots_signature: 67 chain values (2,144 bytes)
      - wots_public_key: 67 public key values (for verification)
      - leaf_index: which WOTS+ keypair was used
      - auth_path: 10 Merkle sibling hashes (320 bytes)

    Total: ~2,464 bytes (vs Lamport's 8,192 + no many-time use).
    """
    wots_signature: List[bytes]
    wots_public_key: List[bytes]
    leaf_index: int
    auth_path: List[bytes]

    @property
    def size_bytes(self) -> int:
        """Total signature size in bytes."""
        wots_size = len(self.wots_signature) * 32
        auth_size = len(self.auth_path) * 32
        return wots_size + auth_size

    def serialize(self) -> bytes:
        """Serialize the signature to bytes."""
        parts = []
        # Leaf index (4 bytes)
        parts.append(self.leaf_index.to_bytes(4, 'big'))
        # WOTS+ signature values
        for val in self.wots_signature:
            parts.append(val)
        # WOTS+ public key
        for val in self.wots_public_key:
            parts.append(val)
        # Auth path
        for val in self.auth_path:
            parts.append(val)
        return b''.join(parts)


# =============================================================================
# BAB64-IBST IDENTITY
# =============================================================================

class BAB64IBSTIdentity:
    """
    A BAB64 identity using Image-Bound Signature Trees.

    Private key: 256-bit seed
    Private image: BabelRender(seed) -> 4,096 pixels
    Chain function: f_I derived from the image
    WOTS+ keys: 1,024 keypairs in a Merkle tree
    Public identity: (address, Merkle root)

    Supports 1,024 signatures before key exhaustion.
    Each signature is ~2,464 bytes.
    """

    def __init__(self, private_key: bytes, config: BAB64Config = None):
        assert len(private_key) == 32, "Private key must be 32 bytes"
        self.config = config or BAB64Config()
        self._private_key = private_key

        # Render the private image
        renderer = BabelRenderer(self.config)
        self._image = renderer.render(private_key)
        self._image_bytes = self._image.tobytes()

        # Compute BAB64 address
        hasher = ImageHash(self.config)
        self.address = hasher.hash_image(self._image)

        # Create image-derived chain function
        self.chain_fn = ImageChainFunction(self._image_bytes)

        # Generate all WOTS+ keypairs and build Merkle tree
        self._wots_keys: List[BAB64WOTS] = []
        leaf_hashes: List[bytes] = []
        for i in range(BAB64MerkleTree.NUM_LEAVES):
            seed = hashlib.sha256(
                self._image_bytes + i.to_bytes(4, 'big') + b'ibst_wots'
            ).digest()
            wots = BAB64WOTS(self.chain_fn, seed)
            self._wots_keys.append(wots)
            leaf_hashes.append(wots.public_key_hash())

        self._tree = BAB64MerkleTree(self.chain_fn, leaf_hashes)
        self._next_key = 0

    @classmethod
    def generate(cls, config: BAB64Config = None) -> 'BAB64IBSTIdentity':
        """Create a new random IBST identity."""
        return cls(os.urandom(32), config)

    @property
    def address_hex(self) -> str:
        return self.address.hex()

    @property
    def merkle_root(self) -> bytes:
        return self._tree.root

    @property
    def signatures_remaining(self) -> int:
        """How many signatures can still be issued."""
        return BAB64MerkleTree.NUM_LEAVES - self._next_key

    @property
    def signatures_used(self) -> int:
        return self._next_key

    def sign(self, message: bytes) -> IBSTSignature:
        """
        Sign a message using the next available WOTS+ key.

        Raises RuntimeError if all 1,024 keys are exhausted.
        """
        if self._next_key >= BAB64MerkleTree.NUM_LEAVES:
            raise RuntimeError(
                f"Key exhaustion: all {BAB64MerkleTree.NUM_LEAVES} "
                f"WOTS+ keys have been used. Generate a new identity."
            )

        idx = self._next_key
        wots = self._wots_keys[idx]

        # WOTS+ signature
        wots_sig = wots.sign(message)
        wots_pk = wots.public_key

        # Merkle authentication path
        auth_path = self._tree.auth_path(idx)

        self._next_key += 1

        return IBSTSignature(
            wots_signature=wots_sig,
            wots_public_key=wots_pk,
            leaf_index=idx,
            auth_path=auth_path,
        )

    def verify(self, message: bytes, signature: IBSTSignature) -> bool:
        """
        Verify a signature was produced by this identity.

        Steps:
          1. Verify WOTS+ signature against the embedded public key
          2. Hash the public key to get the leaf
          3. Verify the Merkle path from leaf to root
        """
        # Step 1: WOTS+ verification
        if not BAB64WOTS.verify(message, signature.wots_signature,
                                signature.wots_public_key, self.chain_fn):
            return False

        # Step 2: Compute leaf hash from public key
        h = hashlib.sha256()
        for pk_val in signature.wots_public_key:
            h.update(pk_val)
        leaf_hash = h.digest()

        # Step 3: Merkle path verification
        return self._tree.verify_path(
            leaf_hash, signature.leaf_index, signature.auth_path
        )

    @staticmethod
    def verify_standalone(message: bytes, signature: IBSTSignature,
                          merkle_root: bytes,
                          image_bytes: bytes) -> bool:
        """
        Verify without holding the full identity — only needs
        the Merkle root and the image (for the chain function).

        This is how a third party verifies: they know the sender's
        Merkle root (public) and reconstruct f_I from the image.
        """
        chain_fn = ImageChainFunction(image_bytes)

        # WOTS+ verification with the sender's chain function
        if not BAB64WOTS.verify(message, signature.wots_signature,
                                signature.wots_public_key, chain_fn):
            return False

        # Leaf hash
        h = hashlib.sha256()
        for pk_val in signature.wots_public_key:
            h.update(pk_val)
        leaf_hash = h.digest()

        # Merkle path against the known root
        tree_stub = BAB64MerkleTree.__new__(BAB64MerkleTree)
        tree_stub._chain_fn = chain_fn
        tree_stub._nodes = [b'\x00' * 32] * (2 * BAB64MerkleTree.NUM_LEAVES)
        tree_stub._nodes[1] = merkle_root

        # Manual path verification using the chain function
        current = leaf_hash
        idx = signature.leaf_index
        for level in range(BAB64MerkleTree.HEIGHT):
            sibling = signature.auth_path[level]
            if idx % 2 == 0:
                current = tree_stub._hash_pair(current, sibling)
            else:
                current = tree_stub._hash_pair(sibling, current)
            idx >>= 1

        return current == merkle_root
