"""
BAB64 Identity & Transaction Layer
====================================

Bitcoin-like identity system built on BAB64's self-referential hash.

Identity:
  Private key  = 256-bit seed (random Babel coordinates)
  Private image = BabelRender(private_key) → 4,096 pixels
  Public address = BAB64(private_image) → 256-bit hash

The image bridges private key and public address — like a public
key you never reveal. Knowing the address doesn't reveal the image,
and without the image you can't sign.

Signatures:
  Lamport one-time signatures derived from the private image.
  Quantum-resistant — security reduces to SHA-256 preimage hardness.

Author: Shrey (concept) + Claude (implementation)
"""

import hashlib
import os
from dataclasses import dataclass, field
from typing import List, Optional

from bab64_engine import BAB64Config, BabelRenderer, ImageHash


# =============================================================================
# LAMPORT ONE-TIME SIGNATURE
# =============================================================================

class LamportKeyPair:
    """
    Lamport one-time signature derived from a BAB64 private image.

    For each bit position i in [0, 255]:
      sk0[i] = SHA-256(image_bytes || i || 0)
      sk1[i] = SHA-256(image_bytes || i || 1)
      pk0[i] = SHA-256(sk0[i])
      pk1[i] = SHA-256(sk1[i])

    Public key = (pk0[0..255], pk1[0..255]) — 512 hashes.
    Signing reveals one sk per bit position — one-time use only.
    """

    def __init__(self, image_bytes: bytes):
        self._sk0: List[bytes] = []
        self._sk1: List[bytes] = []
        self.pk0: List[bytes] = []
        self.pk1: List[bytes] = []
        self._used: bool = False  # Lamport keys are ONE-TIME use

        for i in range(256):
            i_bytes = i.to_bytes(4, 'big')
            sk0 = hashlib.sha256(image_bytes + i_bytes + b'\x00').digest()
            sk1 = hashlib.sha256(image_bytes + i_bytes + b'\x01').digest()
            self._sk0.append(sk0)
            self._sk1.append(sk1)
            self.pk0.append(hashlib.sha256(sk0).digest())
            self.pk1.append(hashlib.sha256(sk1).digest())

    def sign(self, message: bytes) -> List[bytes]:
        """
        Sign a message using Lamport scheme.

        Returns 256 secret key halves — one per bit of SHA-256(message).
        Raises RuntimeError on second use — Lamport keys are one-time.
        """
        if self._used:
            raise RuntimeError(
                "Lamport key already used — one-time signatures "
                "cannot be reused safely. Generate a new key pair."
            )
        self._used = True
        msg_hash = hashlib.sha256(message).digest()
        signature = []
        for i in range(256):
            byte_idx = i // 8
            bit_idx = 7 - (i % 8)
            bit = (msg_hash[byte_idx] >> bit_idx) & 1
            if bit == 0:
                signature.append(self._sk0[i])
            else:
                signature.append(self._sk1[i])
        return signature

    def verification_key(self) -> List[bytes]:
        """Return the full public verification key (512 hashes)."""
        return self.pk0 + self.pk1

    @staticmethod
    def verify(message: bytes, signature: List[bytes],
               verification_key: List[bytes]) -> bool:
        """
        Verify a Lamport signature.

        For each bit i of SHA-256(message):
          if bit=0: SHA-256(signature[i]) must equal pk0[i]
          if bit=1: SHA-256(signature[i]) must equal pk1[i]
        """
        if len(signature) != 256 or len(verification_key) != 512:
            return False

        pk0 = verification_key[:256]
        pk1 = verification_key[256:]

        msg_hash = hashlib.sha256(message).digest()

        for i in range(256):
            byte_idx = i // 8
            bit_idx = 7 - (i % 8)
            bit = (msg_hash[byte_idx] >> bit_idx) & 1

            expected = pk0[i] if bit == 0 else pk1[i]
            actual = hashlib.sha256(signature[i]).digest()

            if actual != expected:
                return False

        return True


# =============================================================================
# BAB64 SIGNATURE — wraps raw Lamport sig with metadata
# =============================================================================

@dataclass
class BAB64Signature:
    """A Lamport signature with its key index and verification key."""
    raw: List[bytes]
    key_index: int
    verification_key: List[bytes]


# =============================================================================
# BAB64 IDENTITY
# =============================================================================

class BAB64Identity:
    """
    A BAB64 identity — analogous to a Bitcoin keypair.

    Private key:   256-bit seed
    Private image: BabelRender(seed) → 4,096 pixels
    Public address: BAB64(image) → 256-bit hash
    Lamport keys:  derived from image for signing
    """

    def __init__(self, private_key: bytes, config: BAB64Config = None):
        assert len(private_key) == 32, "Private key must be 32 bytes"
        self.config = config or BAB64Config()
        self._private_key = private_key

        # Render the private image from the key
        renderer = BabelRenderer(self.config)
        self._image = renderer.render(private_key)
        self._image_bytes = self._image.tobytes()

        # Compute public address: BAB64 hash of the image
        hasher = ImageHash(self.config)
        self.address = hasher.hash_image(self._image)

        # Lamport key index — each signing derives a fresh keypair
        self._key_index = 0

    @classmethod
    def generate(cls, config: BAB64Config = None) -> 'BAB64Identity':
        """Create a new random identity."""
        private_key = os.urandom(32)
        return cls(private_key, config)

    def _derive_lamport(self, index: int) -> LamportKeyPair:
        """Derive a fresh Lamport keypair for the given index."""
        # Each index produces a unique keypair from the image
        seed = self._image_bytes + index.to_bytes(4, 'big')
        return LamportKeyPair(seed)

    @property
    def address_hex(self) -> str:
        return self.address.hex()

    def current_verification_key(self) -> List[bytes]:
        """Return the verification key for the NEXT signing operation."""
        return self._derive_lamport(self._key_index).verification_key()

    @property
    def verification_key(self) -> List[bytes]:
        """Return the verification key for key_index=0 (backwards compat)."""
        return self._derive_lamport(0).verification_key()

    def sign(self, message: bytes) -> 'BAB64Signature':
        """
        Sign a message, advancing to a fresh Lamport key.

        Returns a BAB64Signature containing the raw sig, key index,
        and verification key needed for standalone verification.
        """
        key_index = self._key_index
        lamport = self._derive_lamport(key_index)
        raw_sig = lamport.sign(message)
        vk = lamport.verification_key()
        self._key_index += 1
        return BAB64Signature(raw=raw_sig, key_index=key_index,
                              verification_key=vk)

    def verify(self, message: bytes, signature) -> bool:
        """
        Verify a signature was produced by THIS identity.

        For BAB64Signature: re-derive the expected verification key at
        the claimed key_index and compare — prevents cross-identity forgery.
        For raw list: check against key_index=0.
        """
        if isinstance(signature, BAB64Signature):
            expected_vk = self._derive_lamport(signature.key_index).verification_key()
            if expected_vk != signature.verification_key:
                return False
            return LamportKeyPair.verify(
                message, signature.raw, expected_vk
            )
        # Legacy: raw list of bytes with key_index=0
        return LamportKeyPair.verify(
            message, signature, self._derive_lamport(0).verification_key()
        )


# =============================================================================
# BAB64 TRANSACTION
# =============================================================================

@dataclass
class BAB64Transaction:
    """
    A signed transaction between two BAB64 identities.

    Fields:
      sender:    sender's public address (hex)
      receiver:  receiver's public address (hex)
      amount:    transfer amount
      nonce:     unique per-transaction counter (replay protection)
      signature: Lamport signature over the transaction hash
      tx_hash:   SHA-256 of (sender || receiver || amount || nonce)
    """
    sender: str
    receiver: str
    amount: int
    nonce: int = 0
    signature: List[bytes] = field(default_factory=list)
    tx_hash: str = ""

    def __post_init__(self):
        self._verification_key: Optional[List[bytes]] = None

    def _compute_hash(self) -> bytes:
        """Compute transaction hash from sender, receiver, amount, nonce."""
        payload = f"{self.sender}:{self.receiver}:{self.amount}:{self.nonce}".encode()
        return hashlib.sha256(payload).digest()

    def sign(self, identity: BAB64Identity) -> None:
        """Sign this transaction with the sender's identity."""
        tx_bytes = self._compute_hash()
        self.tx_hash = tx_bytes.hex()
        sig = identity.sign(tx_bytes)
        if isinstance(sig, BAB64Signature):
            self.signature = sig.raw
            self._verification_key = sig.verification_key
        else:
            self.signature = sig
            self._verification_key = None

    def verify(self, sender_verification_key: List[bytes]) -> bool:
        """Verify transaction signature against sender's public key."""
        if not self.signature or not self.tx_hash:
            return False
        tx_bytes = self._compute_hash()
        if tx_bytes.hex() != self.tx_hash:
            return False
        return LamportKeyPair.verify(
            tx_bytes, self.signature, sender_verification_key
        )

    def verify_self(self) -> bool:
        """Verify using the embedded verification key (from signing)."""
        if self._verification_key is None:
            return False
        return self.verify(self._verification_key)


# =============================================================================
# TRANSACTION POOL — Replay Detection
# =============================================================================

class BAB64TransactionPool:
    """
    Tracks processed transactions to prevent replay attacks.

    A valid transaction can only be accepted ONCE. Submitting the
    same signed transaction again is rejected as a replay.
    """

    def __init__(self):
        self._seen_hashes: set = set()

    def submit(self, tx: BAB64Transaction,
               sender_verification_key: List[bytes]) -> bool:
        """
        Submit a transaction. Returns True if accepted, False if
        the transaction is invalid or a replay.
        """
        if not tx.verify(sender_verification_key):
            return False
        if tx.tx_hash in self._seen_hashes:
            return False  # Replay detected
        self._seen_hashes.add(tx.tx_hash)
        return True

    def is_replay(self, tx: BAB64Transaction) -> bool:
        """Check if a transaction hash has already been processed."""
        return tx.tx_hash in self._seen_hashes


# =============================================================================
# CONVENIENCE FUNCTIONS
# =============================================================================

def create_identity(config: BAB64Config = None) -> BAB64Identity:
    """Create a new random BAB64 identity."""
    return BAB64Identity.generate(config)


def sign_transaction(sender: BAB64Identity, receiver: BAB64Identity,
                     amount: int, nonce: int = 0) -> BAB64Transaction:
    """Create and sign a transaction."""
    tx = BAB64Transaction(
        sender=sender.address_hex,
        receiver=receiver.address_hex,
        amount=amount,
        nonce=nonce,
    )
    tx.sign(sender)
    return tx


def verify_transaction(tx: BAB64Transaction,
                       sender_verification_key: List[bytes] = None) -> bool:
    """Verify a signed transaction."""
    if sender_verification_key is not None:
        return tx.verify(sender_verification_key)
    return tx.verify_self()
