"""
Tests for BAB64 Identity & Transaction Layer
=============================================

20+ tests covering:
  - Identity determinism
  - Signature validity
  - Signature forgery rejection
  - Transaction tamper detection
  - Different messages → different signatures
"""

import hashlib
import pytest
from bab64_identity import (
    BAB64Identity, LamportKeyPair, BAB64Transaction,
    create_identity, sign_transaction, verify_transaction,
)
from bab64_engine import BAB64Config


# Fixed seeds for deterministic tests
SEED_A = hashlib.sha256(b"alice_seed").digest()
SEED_B = hashlib.sha256(b"bob_seed").digest()


# =============================================================================
# IDENTITY DETERMINISM
# =============================================================================

class TestIdentityDeterminism:
    """Same seed must always produce the same identity."""

    def test_same_seed_same_address(self):
        id1 = BAB64Identity(SEED_A)
        id2 = BAB64Identity(SEED_A)
        assert id1.address == id2.address

    def test_same_seed_same_verification_key(self):
        id1 = BAB64Identity(SEED_A)
        id2 = BAB64Identity(SEED_A)
        assert id1.verification_key == id2.verification_key

    def test_different_seeds_different_addresses(self):
        id_a = BAB64Identity(SEED_A)
        id_b = BAB64Identity(SEED_B)
        assert id_a.address != id_b.address

    def test_address_is_256_bits(self):
        identity = BAB64Identity(SEED_A)
        assert len(identity.address) == 32

    def test_address_hex_format(self):
        identity = BAB64Identity(SEED_A)
        assert len(identity.address_hex) == 64
        int(identity.address_hex, 16)  # must be valid hex


# =============================================================================
# LAMPORT SIGNATURE VALIDITY
# =============================================================================

class TestSignatureValidity:
    """Valid signatures must verify; basic properties hold."""

    def test_sign_and_verify(self):
        identity = BAB64Identity(SEED_A)
        msg = b"hello world"
        sig = identity.sign(msg)
        assert identity.verify(msg, sig)

    def test_signature_length(self):
        identity = BAB64Identity(SEED_A)
        sig = identity.sign(b"test")
        assert len(sig) == 256
        assert all(len(s) == 32 for s in sig)

    def test_verification_key_length(self):
        identity = BAB64Identity(SEED_A)
        vk = identity.verification_key
        assert len(vk) == 512
        assert all(len(k) == 32 for k in vk)

    def test_deterministic_signature(self):
        id1 = BAB64Identity(SEED_A)
        id2 = BAB64Identity(SEED_A)
        msg = b"determinism check"
        sig1 = id1.sign(msg)
        sig2 = id2.sign(msg)
        assert sig1 == sig2


# =============================================================================
# FORGERY REJECTION
# =============================================================================

class TestForgeryRejection:
    """Invalid signatures must be rejected."""

    def test_wrong_message_rejected(self):
        identity = BAB64Identity(SEED_A)
        sig = identity.sign(b"original message")
        assert not identity.verify(b"tampered message", sig)

    def test_wrong_key_rejected(self):
        id_a = BAB64Identity(SEED_A)
        id_b = BAB64Identity(SEED_B)
        sig = id_a.sign(b"signed by alice")
        assert not id_b.verify(b"signed by alice", sig)

    def test_truncated_signature_rejected(self):
        identity = BAB64Identity(SEED_A)
        sig = identity.sign(b"test")
        assert not LamportKeyPair.verify(
            b"test", sig[:255], identity.verification_key
        )

    def test_corrupted_signature_rejected(self):
        identity = BAB64Identity(SEED_A)
        msg = b"integrity test"
        sig = identity.sign(msg)
        # Corrupt one element
        sig[0] = b'\x00' * 32
        assert not identity.verify(msg, sig)

    def test_empty_signature_rejected(self):
        identity = BAB64Identity(SEED_A)
        assert not LamportKeyPair.verify(
            b"test", [], identity.verification_key
        )


# =============================================================================
# DIFFERENT MESSAGES → DIFFERENT SIGNATURES
# =============================================================================

class TestSignatureDiversity:
    """Different messages must produce different signatures."""

    def test_different_messages_different_sigs(self):
        identity = BAB64Identity(SEED_A)
        sig1 = identity.sign(b"message one")
        sig2 = identity.sign(b"message two")
        assert sig1 != sig2

    def test_one_bit_message_difference(self):
        identity = BAB64Identity(SEED_A)
        sig1 = identity.sign(b"\x00")
        sig2 = identity.sign(b"\x01")
        # At least some revealed keys must differ
        differences = sum(1 for a, b in zip(sig1, sig2) if a != b)
        assert differences > 0


# =============================================================================
# TRANSACTION SIGNING & VERIFICATION
# =============================================================================

class TestTransactions:
    """Transaction signing, verification, and tamper detection."""

    def test_valid_transaction(self):
        alice = BAB64Identity(SEED_A)
        bob = BAB64Identity(SEED_B)
        tx = sign_transaction(alice, bob, 100)
        assert verify_transaction(tx, alice.verification_key)

    def test_transaction_has_hash(self):
        alice = BAB64Identity(SEED_A)
        bob = BAB64Identity(SEED_B)
        tx = sign_transaction(alice, bob, 50)
        assert len(tx.tx_hash) == 64

    def test_tamper_amount_rejected(self):
        alice = BAB64Identity(SEED_A)
        bob = BAB64Identity(SEED_B)
        tx = sign_transaction(alice, bob, 100)
        tx.amount = 999
        assert not verify_transaction(tx, alice.verification_key)

    def test_tamper_receiver_rejected(self):
        alice = BAB64Identity(SEED_A)
        bob = BAB64Identity(SEED_B)
        charlie = BAB64Identity(hashlib.sha256(b"charlie").digest())
        tx = sign_transaction(alice, bob, 100)
        tx.receiver = charlie.address_hex
        assert not verify_transaction(tx, alice.verification_key)

    def test_tamper_sender_rejected(self):
        alice = BAB64Identity(SEED_A)
        bob = BAB64Identity(SEED_B)
        tx = sign_transaction(alice, bob, 100)
        tx.sender = bob.address_hex
        assert not verify_transaction(tx, alice.verification_key)

    def test_unsigned_transaction_rejected(self):
        tx = BAB64Transaction(sender="aa" * 32, receiver="bb" * 32, amount=10)
        vk = BAB64Identity(SEED_A).verification_key
        assert not verify_transaction(tx, vk)

    def test_wrong_signer_rejected(self):
        alice = BAB64Identity(SEED_A)
        bob = BAB64Identity(SEED_B)
        tx = sign_transaction(alice, bob, 100)
        # Verify with bob's key instead of alice's
        assert not verify_transaction(tx, bob.verification_key)


# =============================================================================
# CONVENIENCE FUNCTIONS
# =============================================================================

class TestConvenienceFunctions:
    def test_create_identity_random(self):
        id1 = create_identity()
        id2 = create_identity()
        assert id1.address != id2.address
