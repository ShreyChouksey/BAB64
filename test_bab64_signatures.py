"""
Tests for BAB64-IBST — Image-Bound Signature Trees
====================================================

25+ tests covering:
  - Chain function determinism and one-way properties
  - WOTS+ sign/verify correctness
  - Forgery rejection
  - Merkle tree construction and path verification
  - Many-time signing (10+ messages)
  - Cross-identity rejection
  - Signature size verification
  - Key exhaustion handling
"""

import hashlib
import os
import pytest

from bab64_engine import BAB64Config, BabelRenderer
from bab64_signatures import (
    ImageChainFunction,
    BAB64WOTS,
    BAB64MerkleTree,
    BAB64IBSTIdentity,
    IBSTSignature,
)


# =============================================================================
# FIXTURES
# =============================================================================

@pytest.fixture
def config():
    return BAB64Config()

@pytest.fixture
def image_bytes(config):
    """A deterministic 4,096-byte image for testing."""
    renderer = BabelRenderer(config)
    seed = hashlib.sha256(b"test_ibst_seed").digest()
    return renderer.render(seed).tobytes()

@pytest.fixture
def chain_fn(image_bytes):
    return ImageChainFunction(image_bytes)

@pytest.fixture
def identity_a():
    """Deterministic identity A."""
    key = hashlib.sha256(b"identity_a_seed").digest()
    return BAB64IBSTIdentity(key)

@pytest.fixture
def identity_b():
    """Deterministic identity B — different from A."""
    key = hashlib.sha256(b"identity_b_seed").digest()
    return BAB64IBSTIdentity(key)


# =============================================================================
# COMPONENT 1 — Image-Derived Chain Function
# =============================================================================

class TestImageChainFunction:

    def test_determinism(self, chain_fn):
        """Same input always produces same output."""
        x = os.urandom(32)
        assert chain_fn(x) == chain_fn(x)

    def test_output_length(self, chain_fn):
        """Output is always 32 bytes."""
        x = os.urandom(32)
        assert len(chain_fn(x)) == 32

    def test_different_inputs_different_outputs(self, chain_fn):
        """Different inputs produce different outputs."""
        x1 = hashlib.sha256(b"input1").digest()
        x2 = hashlib.sha256(b"input2").digest()
        assert chain_fn(x1) != chain_fn(x2)

    def test_different_images_different_functions(self, config):
        """Two images produce different chain functions."""
        renderer = BabelRenderer(config)
        img1 = renderer.render(hashlib.sha256(b"img1").digest()).tobytes()
        img2 = renderer.render(hashlib.sha256(b"img2").digest()).tobytes()
        fn1 = ImageChainFunction(img1)
        fn2 = ImageChainFunction(img2)
        x = hashlib.sha256(b"test").digest()
        assert fn1(x) != fn2(x)

    def test_chain_zero_steps(self, chain_fn):
        """Chaining 0 steps returns the input unchanged."""
        x = os.urandom(32)
        assert chain_fn.chain(x, 0) == x

    def test_chain_one_step(self, chain_fn):
        """Chaining 1 step equals a single call."""
        x = os.urandom(32)
        assert chain_fn.chain(x, 1) == chain_fn(x)

    def test_chain_composition(self, chain_fn):
        """chain(x, 3) == f(f(f(x)))."""
        x = os.urandom(32)
        manual = chain_fn(chain_fn(chain_fn(x)))
        assert chain_fn.chain(x, 3) == manual

    def test_rejects_wrong_input_length(self, chain_fn):
        """Input must be exactly 32 bytes."""
        with pytest.raises(AssertionError):
            chain_fn(b"short")


# =============================================================================
# COMPONENT 2 — WOTS+ with Image Chains
# =============================================================================

class TestBAB64WOTS:

    def test_sign_verify_roundtrip(self, chain_fn):
        """A valid signature verifies."""
        seed = os.urandom(32)
        wots = BAB64WOTS(chain_fn, seed)
        msg = b"hello world"
        sig = wots.sign(msg)
        assert BAB64WOTS.verify(msg, sig, wots.public_key, chain_fn)

    def test_wrong_message_rejected(self, chain_fn):
        """Signature doesn't verify for a different message."""
        seed = os.urandom(32)
        wots = BAB64WOTS(chain_fn, seed)
        sig = wots.sign(b"correct message")
        assert not BAB64WOTS.verify(b"wrong message", sig, wots.public_key, chain_fn)

    def test_signature_length(self, chain_fn):
        """Signature has exactly 67 values of 32 bytes each."""
        seed = os.urandom(32)
        wots = BAB64WOTS(chain_fn, seed)
        sig = wots.sign(b"test")
        assert len(sig) == 67
        assert all(len(v) == 32 for v in sig)

    def test_public_key_length(self, chain_fn):
        """Public key has exactly 67 values."""
        seed = os.urandom(32)
        wots = BAB64WOTS(chain_fn, seed)
        assert len(wots.public_key) == 67

    def test_deterministic_keys(self, chain_fn):
        """Same seed produces same keypair."""
        seed = hashlib.sha256(b"deterministic").digest()
        wots1 = BAB64WOTS(chain_fn, seed)
        wots2 = BAB64WOTS(chain_fn, seed)
        assert wots1.public_key == wots2.public_key

    def test_forgery_wrong_chain_fn(self, config):
        """Signature made with one image's chain function fails
        verification with a different image's chain function."""
        renderer = BabelRenderer(config)
        img1 = renderer.render(hashlib.sha256(b"signer").digest()).tobytes()
        img2 = renderer.render(hashlib.sha256(b"attacker").digest()).tobytes()
        fn1 = ImageChainFunction(img1)
        fn2 = ImageChainFunction(img2)

        seed = os.urandom(32)
        wots = BAB64WOTS(fn1, seed)
        sig = wots.sign(b"message")
        # Verify with wrong chain function
        assert not BAB64WOTS.verify(b"message", sig, wots.public_key, fn2)

    def test_checksum_digits(self):
        """Message-to-digits produces 67 digits, all in [0, 15]."""
        msg_hash = hashlib.sha256(b"test").digest()
        digits = BAB64WOTS._msg_to_digits(msg_hash)
        assert len(digits) == 67
        assert all(0 <= d <= 15 for d in digits)

    def test_public_key_hash_deterministic(self, chain_fn):
        """Public key hash is deterministic."""
        seed = hashlib.sha256(b"pk_hash_test").digest()
        wots = BAB64WOTS(chain_fn, seed)
        assert wots.public_key_hash() == wots.public_key_hash()


# =============================================================================
# COMPONENT 3 — Merkle Tree
# =============================================================================

class TestBAB64MerkleTree:

    def test_root_deterministic(self, chain_fn):
        """Same leaves produce same root."""
        leaves = [hashlib.sha256(i.to_bytes(4, 'big')).digest()
                  for i in range(1024)]
        tree1 = BAB64MerkleTree(chain_fn, leaves)
        tree2 = BAB64MerkleTree(chain_fn, leaves)
        assert tree1.root == tree2.root

    def test_auth_path_length(self, chain_fn):
        """Auth path has exactly 10 hashes (tree height)."""
        leaves = [os.urandom(32) for _ in range(1024)]
        tree = BAB64MerkleTree(chain_fn, leaves)
        path = tree.auth_path(0)
        assert len(path) == 10

    def test_auth_path_verifies(self, chain_fn):
        """Valid auth path verifies against root."""
        leaves = [hashlib.sha256(i.to_bytes(4, 'big')).digest()
                  for i in range(1024)]
        tree = BAB64MerkleTree(chain_fn, leaves)
        for idx in [0, 1, 512, 1023]:
            path = tree.auth_path(idx)
            assert tree.verify_path(leaves[idx], idx, path)

    def test_wrong_leaf_rejected(self, chain_fn):
        """Auth path for wrong leaf value fails."""
        leaves = [hashlib.sha256(i.to_bytes(4, 'big')).digest()
                  for i in range(1024)]
        tree = BAB64MerkleTree(chain_fn, leaves)
        path = tree.auth_path(0)
        wrong_leaf = os.urandom(32)
        assert not tree.verify_path(wrong_leaf, 0, path)

    def test_wrong_index_rejected(self, chain_fn):
        """Auth path for wrong index fails."""
        leaves = [hashlib.sha256(i.to_bytes(4, 'big')).digest()
                  for i in range(1024)]
        tree = BAB64MerkleTree(chain_fn, leaves)
        path = tree.auth_path(0)
        assert not tree.verify_path(leaves[0], 1, path)

    def test_different_chain_fn_different_root(self, config):
        """Different image produces different Merkle root."""
        renderer = BabelRenderer(config)
        img1 = renderer.render(hashlib.sha256(b"tree1").digest()).tobytes()
        img2 = renderer.render(hashlib.sha256(b"tree2").digest()).tobytes()
        fn1 = ImageChainFunction(img1)
        fn2 = ImageChainFunction(img2)
        leaves = [hashlib.sha256(i.to_bytes(4, 'big')).digest()
                  for i in range(1024)]
        tree1 = BAB64MerkleTree(fn1, leaves)
        tree2 = BAB64MerkleTree(fn2, leaves)
        assert tree1.root != tree2.root


# =============================================================================
# FULL IBST IDENTITY — Sign, Verify, Many-Time, Cross-Identity
# =============================================================================

class TestBAB64IBSTIdentity:

    def test_sign_verify_basic(self, identity_a):
        """Basic sign-verify roundtrip."""
        msg = b"basic test message"
        sig = identity_a.sign(msg)
        assert identity_a.verify(msg, sig)

    def test_wrong_message_rejected(self, identity_a):
        """Signature on one message doesn't verify for another."""
        sig = identity_a.sign(b"original")
        assert not identity_a.verify(b"tampered", sig)

    def test_many_time_signing(self, identity_a):
        """Sign 10 different messages, all verify."""
        sigs = []
        for i in range(10):
            msg = f"message number {i}".encode()
            sig = identity_a.sign(msg)
            sigs.append((msg, sig))

        for msg, sig in sigs:
            assert identity_a.verify(msg, sig)

    def test_signatures_remaining(self, identity_a):
        """signatures_remaining decrements correctly."""
        assert identity_a.signatures_remaining == 1024
        identity_a.sign(b"one")
        assert identity_a.signatures_remaining == 1023
        identity_a.sign(b"two")
        assert identity_a.signatures_remaining == 1022

    def test_signatures_used(self, identity_a):
        """signatures_used increments correctly."""
        assert identity_a.signatures_used == 0
        identity_a.sign(b"one")
        assert identity_a.signatures_used == 1

    def test_cross_identity_rejection(self, identity_a, identity_b):
        """Signature from A doesn't verify under B."""
        msg = b"cross identity test"
        sig = identity_a.sign(msg)
        assert not identity_b.verify(msg, sig)

    def test_signature_size_under_2500(self, identity_a):
        """Total signature size < 2,500 bytes."""
        sig = identity_a.sign(b"size test")
        # WOTS sig: 67 * 32 = 2144, auth path: 10 * 32 = 320
        # Total: 2464
        assert sig.size_bytes < 2500
        assert sig.size_bytes == 67 * 32 + 10 * 32  # 2464

    def test_key_exhaustion(self):
        """After 1,024 signatures, signing raises RuntimeError."""
        key = hashlib.sha256(b"exhaustion_test").digest()
        identity = BAB64IBSTIdentity(key)
        # Manually set the counter near exhaustion
        identity._next_key = 1024
        with pytest.raises(RuntimeError, match="Key exhaustion"):
            identity.sign(b"one too many")

    def test_leaf_index_increments(self, identity_a):
        """Each signature uses the next leaf index."""
        sig0 = identity_a.sign(b"msg0")
        sig1 = identity_a.sign(b"msg1")
        sig2 = identity_a.sign(b"msg2")
        assert sig0.leaf_index == 0
        assert sig1.leaf_index == 1
        assert sig2.leaf_index == 2

    def test_deterministic_address(self):
        """Same private key produces same address."""
        key = hashlib.sha256(b"determinism").digest()
        id1 = BAB64IBSTIdentity(key)
        id2 = BAB64IBSTIdentity(key)
        assert id1.address == id2.address

    def test_deterministic_merkle_root(self):
        """Same private key produces same Merkle root."""
        key = hashlib.sha256(b"determinism").digest()
        id1 = BAB64IBSTIdentity(key)
        id2 = BAB64IBSTIdentity(key)
        assert id1.merkle_root == id2.merkle_root

    def test_standalone_verification(self, identity_a):
        """Third-party verification using only Merkle root + image."""
        msg = b"standalone verify"
        sig = identity_a.sign(msg)
        assert BAB64IBSTIdentity.verify_standalone(
            msg, sig, identity_a.merkle_root, identity_a._image_bytes
        )

    def test_standalone_wrong_root_rejected(self, identity_a):
        """Standalone verification fails with wrong root."""
        msg = b"wrong root"
        sig = identity_a.sign(msg)
        fake_root = os.urandom(32)
        assert not BAB64IBSTIdentity.verify_standalone(
            msg, sig, fake_root, identity_a._image_bytes
        )

    def test_standalone_wrong_image_rejected(self, identity_a, identity_b):
        """Standalone verification fails with wrong image."""
        msg = b"wrong image"
        sig = identity_a.sign(msg)
        assert not BAB64IBSTIdentity.verify_standalone(
            msg, sig, identity_a.merkle_root, identity_b._image_bytes
        )

    def test_generate_random(self):
        """generate() produces a working identity."""
        identity = BAB64IBSTIdentity.generate()
        msg = b"random identity test"
        sig = identity.sign(msg)
        assert identity.verify(msg, sig)

    def test_signature_serialization(self, identity_a):
        """Signature serializes to bytes."""
        sig = identity_a.sign(b"serialize me")
        data = sig.serialize()
        # 4 (index) + 67*32 (wots sig) + 67*32 (wots pk) + 10*32 (auth)
        expected = 4 + 67 * 32 + 67 * 32 + 10 * 32
        assert len(data) == expected
