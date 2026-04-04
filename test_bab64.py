"""
BAB64 Test Suite
=================

Comprehensive tests proving BAB64's cryptographic invariants.
BAB64 is a self-referential image hash: the image defines its
own hash function, then is hashed by it.

Test Categories:
  1. DETERMINISM       — Same input → same output, always
  2. TAMPER RESISTANCE — Any modification invalidates proof
  3. AVALANCHE         — Small input changes → large output changes
  4. VERIFICATION      — Verify is always faster than mine
  5. DIFFICULTY        — Mining time scales with difficulty
  6. HASH QUALITY      — Bit distribution, collision resistance, S-box
  7. CHAIN INTEGRITY   — Multi-block chain validates correctly
  8. EDGE CASES        — Boundary conditions and failure modes

Run:  python3 test_bab64.py
      python3 test_bab64.py -v
      python3 test_bab64.py TestAvalanche
"""

import unittest
import hashlib
import numpy as np
import time
import copy
import json
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from bab64_engine import (
    BAB64Config, BAB64Engine, BAB64Proof, BAB64Block, BAB64Chain,
    BabelRenderer, ImageHash,
)


# =============================================================================
# SHARED FIXTURES
# =============================================================================

def get_test_config(difficulty: int = 4) -> BAB64Config:
    """Low-difficulty config for fast tests."""
    return BAB64Config(
        difficulty_bits=difficulty,
        num_rounds=32,
    )


def mine_test_proof(
    difficulty: int = 4,
    input_data: str = "test_input_determinism_check",
) -> BAB64Proof:
    """Mine a proof for testing."""
    config = get_test_config(difficulty)
    engine = BAB64Engine(config)
    proof = engine.mine(input_data, max_nonces=50000, verbose=False)
    assert proof is not None, f"Failed to mine at difficulty {difficulty}"
    return proof


# =============================================================================
# 1. DETERMINISM TESTS
# =============================================================================

class TestDeterminism(unittest.TestCase):
    """Same input must ALWAYS produce same output."""

    def test_renderer_deterministic(self):
        """Same seed → same image, every time."""
        config = get_test_config()
        renderer = BabelRenderer(config)
        seed = hashlib.sha256(b"determinism_test").digest()

        img1 = renderer.render(seed)
        img2 = renderer.render(seed)

        np.testing.assert_array_equal(img1, img2)
        self.assertEqual(img1.shape, (config.dimension,))

    def test_renderer_different_seeds(self):
        """Different seeds → different images."""
        config = get_test_config()
        renderer = BabelRenderer(config)

        seed1 = hashlib.sha256(b"seed_a").digest()
        seed2 = hashlib.sha256(b"seed_b").digest()

        self.assertFalse(np.array_equal(
            renderer.render(seed1), renderer.render(seed2)
        ))

    def test_render_from_nonce_deterministic(self):
        """Same base_seed + nonce → same image."""
        config = get_test_config()
        renderer = BabelRenderer(config)
        base = hashlib.sha256(b"nonce_det").digest()

        img1 = renderer.render_from_nonce(base, 42)
        img2 = renderer.render_from_nonce(base, 42)

        np.testing.assert_array_equal(img1, img2)

    def test_hash_image_deterministic(self):
        """Same image → same BAB64 hash, every time."""
        config = get_test_config()
        renderer = BabelRenderer(config)
        hasher = ImageHash(config)
        seed = hashlib.sha256(b"hash_det").digest()
        image = renderer.render(seed)

        h1 = hasher.hash_image(image)
        h2 = hasher.hash_image(image)

        self.assertEqual(h1, h2)

    def test_derived_constants_deterministic(self):
        """Derived round constants, rotations, S-box, state are stable."""
        config = get_test_config()
        renderer = BabelRenderer(config)
        hasher = ImageHash(config)
        image = renderer.render(hashlib.sha256(b"derive_det").digest())

        rc1 = hasher._derive_round_constants(image)
        rc2 = hasher._derive_round_constants(image)
        np.testing.assert_array_equal(rc1, rc2)

        rot1 = hasher._derive_rotations(image)
        rot2 = hasher._derive_rotations(image)
        np.testing.assert_array_equal(rot1, rot2)

        sb1 = hasher._derive_sbox(image)
        sb2 = hasher._derive_sbox(image)
        np.testing.assert_array_equal(sb1, sb2)

        st1 = hasher._derive_initial_state(image)
        st2 = hasher._derive_initial_state(image)
        np.testing.assert_array_equal(st1, st2)

    def test_pixel_value_range(self):
        """All pixels in [0, 255]."""
        config = get_test_config()
        renderer = BabelRenderer(config)
        for i in range(10):
            seed = hashlib.sha256(i.to_bytes(4, 'big')).digest()
            img = renderer.render(seed)
            self.assertTrue(np.all(img >= 0))
            self.assertTrue(np.all(img <= 255))
            self.assertEqual(img.dtype, np.uint8)


# =============================================================================
# 2. TAMPER RESISTANCE TESTS
# =============================================================================

class TestTamperResistance(unittest.TestCase):
    """ANY modification to a valid proof must invalidate it."""

    @classmethod
    def setUpClass(cls):
        cls.proof = mine_test_proof(difficulty=4, input_data="tamper_test")
        cls.config = get_test_config(4)

    def _verify(self, proof: BAB64Proof) -> bool:
        engine = BAB64Engine(self.config)
        return engine.verify(proof, verbose=False)

    def test_valid_proof_passes(self):
        """Sanity: unmodified proof is valid."""
        self.assertTrue(self._verify(self.proof))

    def test_change_nonce(self):
        """Changing nonce invalidates proof."""
        t = copy.deepcopy(self.proof)
        t.nonce += 1
        self.assertFalse(self._verify(t))

    def test_change_input_data(self):
        """Changing input data invalidates proof."""
        t = copy.deepcopy(self.proof)
        t.input_data += "x"
        self.assertFalse(self._verify(t))

    def test_change_bab64_hash(self):
        """Changing BAB64 hash invalidates proof."""
        t = copy.deepcopy(self.proof)
        h = list(t.bab64_hash)
        h[10] = '0' if h[10] != '0' else '1'
        t.bab64_hash = ''.join(h)
        self.assertFalse(self._verify(t))

    def test_change_image_hash(self):
        """Changing image hash invalidates proof."""
        t = copy.deepcopy(self.proof)
        h = list(t.image_hash)
        h[5] = 'a' if h[5] != 'a' else 'b'
        t.image_hash = ''.join(h)
        self.assertFalse(self._verify(t))

    def test_change_base_seed(self):
        """Changing base seed invalidates proof."""
        t = copy.deepcopy(self.proof)
        h = list(t.base_seed)
        h[0] = 'a' if h[0] != 'a' else 'b'
        t.base_seed = ''.join(h)
        self.assertFalse(self._verify(t))

    def test_change_difficulty_higher(self):
        """Claiming higher difficulty fails."""
        t = copy.deepcopy(self.proof)
        t.difficulty_bits = 32
        config = BAB64Config(difficulty_bits=32)
        engine = BAB64Engine(config)
        self.assertFalse(engine.verify(t, verbose=False))


# =============================================================================
# 3. AVALANCHE EFFECT TESTS
# =============================================================================

class TestAvalanche(unittest.TestCase):
    """Small input changes must cause large output changes."""

    @classmethod
    def setUpClass(cls):
        cls.config = get_test_config()
        cls.renderer = BabelRenderer(cls.config)
        cls.hasher = ImageHash(cls.config)
        cls.base_seed = hashlib.sha256(b"avalanche_suite").digest()

    def _bit_diff(self, h1: bytes, h2: bytes) -> int:
        return bin(
            int.from_bytes(h1, 'big') ^ int.from_bytes(h2, 'big')
        ).count('1')

    def test_one_pixel_flip_avalanche(self):
        """Flipping 1 pixel → ~50% of hash bits change."""
        diffs = []
        for i in range(30):
            img = self.renderer.render_from_nonce(self.base_seed, i)
            h_orig = self.hasher.hash_image(img)

            img_mod = img.copy()
            img_mod[0] = np.uint8((int(img_mod[0]) + 1) % 256)
            h_mod = self.hasher.hash_image(img_mod)

            diffs.append(self._bit_diff(h_orig, h_mod))

        avg = np.mean(diffs)
        # Expect 90..166 bits (35%–65% of 256)
        self.assertGreater(avg, 90, f"Avg {avg:.0f} bits — insufficient avalanche")
        self.assertLess(avg, 166, f"Avg {avg:.0f} bits — suspicious pattern")

    def test_last_pixel_flip_avalanche(self):
        """Flipping last pixel also causes full avalanche."""
        diffs = []
        for i in range(20):
            img = self.renderer.render_from_nonce(self.base_seed, 1000 + i)
            h_orig = self.hasher.hash_image(img)

            img_mod = img.copy()
            img_mod[-1] = np.uint8((int(img_mod[-1]) + 1) % 256)
            h_mod = self.hasher.hash_image(img_mod)

            diffs.append(self._bit_diff(h_orig, h_mod))

        avg = np.mean(diffs)
        self.assertGreater(avg, 90)
        self.assertLess(avg, 166)

    def test_one_char_input_change(self):
        """1 character change in mining input → different base seed."""
        config = get_test_config()
        engine = BAB64Engine(config)

        s1 = engine._compute_base_seed("Hello World")
        s2 = engine._compute_base_seed("Hello Worle")

        bits1 = int.from_bytes(s1, 'big')
        bits2 = int.from_bytes(s2, 'big')
        diff = bin(bits1 ^ bits2).count('1')
        self.assertGreater(diff, 80)
        self.assertLess(diff, 180)

    def test_sequential_nonces_different_images(self):
        """Consecutive nonces → very different images."""
        img1 = self.renderer.render_from_nonce(self.base_seed, 0)
        img2 = self.renderer.render_from_nonce(self.base_seed, 1)

        diff_ratio = np.mean(img1 != img2)
        self.assertGreater(diff_ratio, 0.95)

    def test_sequential_nonces_different_hashes(self):
        """Consecutive nonces → very different BAB64 hashes."""
        img1 = self.renderer.render_from_nonce(self.base_seed, 0)
        img2 = self.renderer.render_from_nonce(self.base_seed, 1)

        h1 = self.hasher.hash_image(img1)
        h2 = self.hasher.hash_image(img2)

        self.assertNotEqual(h1, h2)
        diff = self._bit_diff(h1, h2)
        self.assertGreater(diff, 80)


# =============================================================================
# 4. VERIFICATION ASYMMETRY TESTS
# =============================================================================

class TestVerificationAsymmetry(unittest.TestCase):
    """Verification must be faster than mining."""

    @classmethod
    def setUpClass(cls):
        cls.config = get_test_config(6)
        cls.engine = BAB64Engine(cls.config)
        cls.proof = cls.engine.mine(
            "asymmetry_test", max_nonces=50000, verbose=False
        )
        assert cls.proof is not None

    def test_verification_faster_than_mining(self):
        """Verify time < mine time."""
        t0 = time.time()
        valid = self.engine.verify(self.proof, verbose=False)
        verify_time = time.time() - t0

        self.assertTrue(valid)
        self.assertLess(verify_time, self.proof.computation_time)

    def test_verification_at_least_10x_faster(self):
        """Verify at least 10x faster than mining."""
        t0 = time.time()
        self.engine.verify(self.proof, verbose=False)
        verify_time = time.time() - t0

        speedup = self.proof.computation_time / max(verify_time, 0.0001)
        self.assertGreater(
            speedup, 10,
            f"Only {speedup:.0f}x — need 10x+"
        )

    def test_multiple_verifications_consistent(self):
        """Verifying N times always gives same result."""
        results = [
            self.engine.verify(self.proof, verbose=False)
            for _ in range(5)
        ]
        self.assertTrue(all(results))


# =============================================================================
# 5. DIFFICULTY SCALING TESTS
# =============================================================================

class TestDifficultyScaling(unittest.TestCase):
    """Mining time must increase with difficulty."""

    def test_difficulty_bits_enforced(self):
        """Proof hash has required leading zeros."""
        for diff in [2, 4, 6]:
            proof = mine_test_proof(diff, f"enforce_{diff}")
            hash_int = int(proof.bab64_hash, 16)
            leading = 256 - hash_int.bit_length() if hash_int > 0 else 256
            self.assertGreaterEqual(
                leading, diff,
                f"Difficulty {diff}: only {leading} leading zeros"
            )

    def test_low_difficulty_always_succeeds(self):
        """Difficulty 1 finds proof fast."""
        config = BAB64Config(difficulty_bits=1)
        engine = BAB64Engine(config)
        proof = engine.mine("easy", max_nonces=100, verbose=False)
        self.assertIsNotNone(proof)

    def test_higher_difficulty_more_nonces(self):
        """Difficulty 8 needs more nonces than difficulty 2 (statistical)."""
        nonces = {}
        for diff in [2, 8]:
            config = BAB64Config(difficulty_bits=diff)
            engine = BAB64Engine(config)
            proof = engine.mine(
                f"scale_{diff}", max_nonces=100000, verbose=False
            )
            self.assertIsNotNone(proof, f"Failed at difficulty {diff}")
            nonces[diff] = proof.nonce

        self.assertGreater(
            nonces[8], nonces[2],
            f"d=8 nonce={nonces[8]} should exceed d=2 nonce={nonces[2]}"
        )


# =============================================================================
# 6. HASH QUALITY TESTS
# =============================================================================

class TestHashQuality(unittest.TestCase):
    """BAB64's image-dependent hash must have good cryptographic properties."""

    @classmethod
    def setUpClass(cls):
        cls.config = get_test_config()
        cls.renderer = BabelRenderer(cls.config)
        cls.hasher = ImageHash(cls.config)
        cls.base_seed = hashlib.sha256(b"quality_suite").digest()

        # Pre-compute hashes for 100 images
        cls.images = []
        cls.hashes = []
        for i in range(100):
            img = cls.renderer.render_from_nonce(cls.base_seed, i)
            cls.images.append(img)
            cls.hashes.append(cls.hasher.hash_image(img))

    def test_no_collisions(self):
        """100 distinct images → 100 distinct hashes."""
        unique = len(set(h.hex() for h in self.hashes))
        self.assertEqual(unique, 100)

    def test_bit_distribution(self):
        """Each bit position is ~50% set across many hashes."""
        bit_counts = np.zeros(256)
        for h in self.hashes:
            h_int = int.from_bytes(h, 'big')
            for b in range(256):
                if h_int & (1 << b):
                    bit_counts[b] += 1

        ratios = bit_counts / len(self.hashes)
        mean_ratio = np.mean(ratios)
        # Should be close to 0.5
        self.assertGreater(mean_ratio, 0.40, f"Mean bit ratio {mean_ratio:.3f}")
        self.assertLess(mean_ratio, 0.60, f"Mean bit ratio {mean_ratio:.3f}")

    def test_no_stuck_bits(self):
        """No bit position always 0 or always 1."""
        bit_counts = np.zeros(256)
        for h in self.hashes:
            h_int = int.from_bytes(h, 'big')
            for b in range(256):
                if h_int & (1 << b):
                    bit_counts[b] += 1

        # No bit should be 0% or 100% set across 100 samples
        self.assertTrue(np.all(bit_counts > 0), "Stuck-at-0 bit found")
        self.assertTrue(
            np.all(bit_counts < len(self.hashes)),
            "Stuck-at-1 bit found"
        )

    def test_hash_output_length(self):
        """Hash output is always exactly 32 bytes (256 bits)."""
        for h in self.hashes[:20]:
            self.assertEqual(len(h), 32)

    def test_sbox_is_permutation(self):
        """Derived S-box is always a valid permutation of [0..255]."""
        for i in range(10):
            img = self.renderer.render_from_nonce(self.base_seed, 200 + i)
            sbox = self.hasher._derive_sbox(img)
            self.assertEqual(len(sbox), 256)
            self.assertEqual(len(set(sbox.tolist())), 256)

    def test_sbox_not_identity(self):
        """Derived S-box is never the identity permutation."""
        for i in range(10):
            img = self.renderer.render_from_nonce(self.base_seed, 300 + i)
            sbox = self.hasher._derive_sbox(img)
            identity = np.arange(256, dtype=np.uint8)
            self.assertFalse(np.array_equal(sbox, identity))

    def test_different_images_different_sboxes(self):
        """Two different images produce different S-boxes."""
        img1 = self.renderer.render_from_nonce(self.base_seed, 0)
        img2 = self.renderer.render_from_nonce(self.base_seed, 1)
        sb1 = self.hasher._derive_sbox(img1)
        sb2 = self.hasher._derive_sbox(img2)
        self.assertFalse(np.array_equal(sb1, sb2))

    def test_different_images_different_constants(self):
        """Two different images produce different round constants."""
        img1 = self.renderer.render_from_nonce(self.base_seed, 0)
        img2 = self.renderer.render_from_nonce(self.base_seed, 1)
        rc1 = self.hasher._derive_round_constants(img1)
        rc2 = self.hasher._derive_round_constants(img2)
        self.assertFalse(np.array_equal(rc1, rc2))

    def test_rotations_in_valid_range(self):
        """All rotation amounts are in [1, 31]."""
        for i in range(10):
            img = self.renderer.render_from_nonce(self.base_seed, 400 + i)
            rots = self.hasher._derive_rotations(img)
            self.assertTrue(np.all(rots >= 1))
            self.assertTrue(np.all(rots <= 31))

    def test_initial_state_varies(self):
        """Different images produce different initial states."""
        states = set()
        for i in range(20):
            img = self.renderer.render_from_nonce(self.base_seed, 500 + i)
            st = self.hasher._derive_initial_state(img)
            states.add(st.tobytes())
        self.assertEqual(len(states), 20)


# =============================================================================
# 7. CHAIN INTEGRITY TESTS
# =============================================================================

class TestChainIntegrity(unittest.TestCase):
    """Multi-block chain must maintain integrity."""

    @classmethod
    def setUpClass(cls):
        config = BAB64Config(difficulty_bits=4)
        cls.chain = BAB64Chain(config)
        for i in range(3):
            block = cls.chain.mine_block(f"chain_test_{i}", verbose=False)
            assert block is not None, f"Failed to mine block {i}"

    def test_chain_length(self):
        self.assertEqual(len(self.chain.blocks), 3)

    def test_chain_validates(self):
        self.assertTrue(self.chain.verify_chain(verbose=False))

    def test_genesis_previous_hash(self):
        self.assertEqual(self.chain.blocks[0].previous_hash, "0" * 64)

    def test_chain_linkage(self):
        for i in range(1, len(self.chain.blocks)):
            self.assertEqual(
                self.chain.blocks[i].previous_hash,
                self.chain.blocks[i - 1].block_hash,
            )

    def test_block_indices_sequential(self):
        for i, block in enumerate(self.chain.blocks):
            self.assertEqual(block.index, i)

    def test_tamper_proof_breaks_chain(self):
        """Changing a proof's nonce invalidates the chain."""
        tampered = copy.deepcopy(self.chain)
        tampered.blocks[1].proof.nonce += 1
        self.assertFalse(tampered.verify_chain(verbose=False))

    def test_reorder_blocks_breaks_chain(self):
        """Swapping block order invalidates the chain."""
        tampered = copy.deepcopy(self.chain)
        tampered.blocks[0], tampered.blocks[1] = (
            tampered.blocks[1], tampered.blocks[0],
        )
        self.assertFalse(tampered.verify_chain(verbose=False))

    def test_tamper_block_hash_breaks_chain(self):
        """Changing a block hash breaks linkage."""
        tampered = copy.deepcopy(self.chain)
        h = list(tampered.blocks[0].block_hash)
        h[0] = 'a' if h[0] != 'a' else 'b'
        tampered.blocks[0].block_hash = ''.join(h)
        self.assertFalse(tampered.verify_chain(verbose=False))


# =============================================================================
# 8. EDGE CASE TESTS
# =============================================================================

class TestEdgeCases(unittest.TestCase):
    """Boundary conditions and failure modes."""

    def test_empty_input(self):
        """Empty string is valid."""
        config = get_test_config(2)
        engine = BAB64Engine(config)
        proof = engine.mine("", max_nonces=5000, verbose=False)
        self.assertIsNotNone(proof)
        self.assertTrue(engine.verify(proof, verbose=False))

    def test_long_input(self):
        """Very long input handled correctly."""
        config = get_test_config(2)
        engine = BAB64Engine(config)
        proof = engine.mine("x" * 100_000, max_nonces=5000, verbose=False)
        self.assertIsNotNone(proof)
        self.assertTrue(engine.verify(proof, verbose=False))

    def test_unicode_input(self):
        """Unicode input handled correctly."""
        config = get_test_config(2)
        engine = BAB64Engine(config)
        proof = engine.mine("🔐🌍 BAB64 ₹∞", max_nonces=5000, verbose=False)
        self.assertIsNotNone(proof)
        self.assertTrue(engine.verify(proof, verbose=False))

    def test_proof_json_roundtrip(self):
        """Proof survives JSON serialization."""
        proof = mine_test_proof(2, "json_roundtrip")
        d = json.loads(proof.to_json())
        restored = BAB64Proof.from_dict(d)

        engine = BAB64Engine(get_test_config(2))
        self.assertTrue(engine.verify(restored, verbose=False))

    def test_zero_seed_image(self):
        """All-zero seed produces valid image."""
        config = get_test_config()
        renderer = BabelRenderer(config)
        img = renderer.render(b'\x00' * 32)
        self.assertEqual(img.shape, (config.dimension,))
        self.assertEqual(img.dtype, np.uint8)

    def test_max_seed_image(self):
        """All-0xFF seed produces valid image."""
        config = get_test_config()
        renderer = BabelRenderer(config)
        img = renderer.render(b'\xff' * 32)
        self.assertEqual(img.shape, (config.dimension,))

    def test_all_zero_image_hashes(self):
        """Synthetic all-zero image produces a hash (doesn't crash)."""
        config = get_test_config()
        hasher = ImageHash(config)
        img = np.zeros(config.dimension, dtype=np.uint8)
        h = hasher.hash_image(img)
        self.assertEqual(len(h), 32)

    def test_all_255_image_hashes(self):
        """Synthetic all-255 image produces a hash (doesn't crash)."""
        config = get_test_config()
        hasher = ImageHash(config)
        img = np.full(config.dimension, 255, dtype=np.uint8)
        h = hasher.hash_image(img)
        self.assertEqual(len(h), 32)

    def test_image_dimension(self):
        """Image is exactly 64x64 = 4096 pixels."""
        config = get_test_config()
        renderer = BabelRenderer(config)
        img = renderer.render(hashlib.sha256(b"dim").digest())
        self.assertEqual(len(img), 4096)

    def test_wrong_difficulty_rejected(self):
        """Proof mined at d=4 fails verification at d=32."""
        proof = mine_test_proof(4, "wrong_diff")
        config = BAB64Config(difficulty_bits=32)
        engine = BAB64Engine(config)
        self.assertFalse(engine.verify(proof, verbose=False))


# =============================================================================
# TEST RUNNER
# =============================================================================

class BAB64TestResult(unittest.TextTestResult):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.successes = []

    def addSuccess(self, test):
        super().addSuccess(test)
        self.successes.append(test)


def run_tests():
    print("""
    ╔══════════════════════════════════════════════════════════╗
    ║          BAB64 TEST SUITE                               ║
    ║     Self-Referential Image Hash                         ║
    ╚══════════════════════════════════════════════════════════╝
    """)

    loader = unittest.TestLoader()
    suite = unittest.TestSuite()

    test_classes = [
        TestDeterminism,
        TestTamperResistance,
        TestAvalanche,
        TestVerificationAsymmetry,
        TestDifficultyScaling,
        TestHashQuality,
        TestChainIntegrity,
        TestEdgeCases,
    ]

    for cls in test_classes:
        suite.addTests(loader.loadTestsFromTestCase(cls))

    runner = unittest.TextTestRunner(
        verbosity=2,
        resultclass=BAB64TestResult,
    )
    result = runner.run(suite)

    total = result.testsRun
    failures = len(result.failures)
    errors = len(result.errors)
    passed = total - failures - errors

    print(f"\n{'='*60}")
    print(f"  RESULTS: {passed}/{total} passed, "
          f"{failures} failed, {errors} errors")
    if passed == total:
        print(f"  *** ALL TESTS PASSED ***")
    print(f"{'='*60}\n")

    return result


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] != '-v':
        unittest.main()
    else:
        run_tests()
