"""
Verify that two independent BAB64 implementations produce identical outputs.

If all 20 test cases match, the specification is precise enough to publish.
If any differ, identify which step diverges.
"""

import hashlib
import numpy as np

# Import both implementations
from bab64_engine import BAB64Config, BabelRenderer, ImageHash
from bab64_reference import (
    create_image, compute_hash,
    _derive_round_keys, _derive_rotations,
    _derive_substitution_table, _derive_starting_state,
)


def generate_seed_and_nonce(index: int):
    """Generate a unique (input_text, nonce) pair for test case index."""
    inputs = [
        "hello world",
        "BAB64 Genesis Block",
        "",
        "a",
        "The quick brown fox jumps over the lazy dog",
        "0",
        "test" * 100,
        "\u00e9\u00e8\u00ea",  # unicode
        "12345678901234567890",
        "AAAA",
        "zzz",
        "proof-of-work",
        "self-referential",
        "Library of Babel",
        "SHA-256",
        "bitcoin",
        "nonce_search",
        "pixel_hash",
        "round_constants",
        "final_test",
    ]
    return inputs[index], index


def compare_images(input_text: str, nonce: int) -> bool:
    """Compare image generation between both implementations."""
    config = BAB64Config()
    renderer = BabelRenderer(config)

    base_seed = hashlib.sha256(input_text.encode('utf-8')).digest()
    img_original = renderer.render_from_nonce(base_seed, nonce)
    img_reference = create_image(input_text, nonce)

    return img_original.tobytes() == img_reference


def compare_derivations(img_orig: np.ndarray, img_ref: bytes) -> dict:
    """Compare each derivation step. Returns dict of step -> match."""
    config = BAB64Config()
    hasher = ImageHash(config)

    results = {}

    # Round constants
    orig_rc = hasher._derive_round_constants(img_orig)
    ref_rc = _derive_round_keys(img_ref)
    results['round_constants'] = all(
        int(orig_rc[i]) == ref_rc[i] for i in range(32)
    )

    # Rotations
    orig_rot = hasher._derive_rotations(img_orig)
    ref_rot = _derive_rotations(img_ref)
    results['rotations'] = all(
        int(orig_rot[i]) == ref_rot[i] for i in range(32)
    )

    # S-box
    orig_sbox = hasher._derive_sbox(img_orig)
    ref_sbox = _derive_substitution_table(img_ref)
    results['sbox'] = all(
        int(orig_sbox[i]) == ref_sbox[i] for i in range(256)
    )

    # Initial state
    orig_state = hasher._derive_initial_state(img_orig)
    ref_state = _derive_starting_state(img_ref)
    results['initial_state'] = all(
        int(orig_state[i]) == ref_state[i] for i in range(8)
    )

    return results


def run_verification():
    print("=" * 64)
    print("  BAB64 Specification Verification")
    print("  Comparing bab64_engine.py vs bab64_reference.py")
    print("=" * 64)
    print()

    config = BAB64Config()
    renderer = BabelRenderer(config)
    hasher = ImageHash(config)

    passed = 0
    failed = 0

    for idx in range(20):
        input_text, nonce = generate_seed_and_nonce(idx)
        label = repr(input_text[:30])

        # Step 1: Compare images
        base_seed = hashlib.sha256(input_text.encode('utf-8')).digest()
        img_orig = renderer.render_from_nonce(base_seed, nonce)
        img_ref = create_image(input_text, nonce)

        if img_orig.tobytes() != img_ref:
            print(f"  [{idx+1:2d}/20] FAIL  {label} -- image generation diverges")
            failed += 1
            continue

        # Step 2: Compare derivations
        derivations = compare_derivations(img_orig, img_ref)
        deriv_fail = [k for k, v in derivations.items() if not v]
        if deriv_fail:
            print(f"  [{idx+1:2d}/20] FAIL  {label} -- derivation diverges: {', '.join(deriv_fail)}")
            failed += 1
            continue

        # Step 3: Compare final hash
        hash_orig = hasher.hash_image(img_orig)
        hash_ref = compute_hash(img_ref)

        if hash_orig == hash_ref:
            print(f"  [{idx+1:2d}/20] PASS  {label}")
            passed += 1
        else:
            print(f"  [{idx+1:2d}/20] FAIL  {label} -- hash output differs")
            print(f"           original:  {hash_orig.hex()}")
            print(f"           reference: {hash_ref.hex()}")
            failed += 1

    print()
    print("=" * 64)
    status = "SPECIFICATION IS PUBLISH-READY" if failed == 0 else "SPECIFICATION NEEDS REVISION"
    print(f"  Result: {passed}/20 match -- {status}")
    print("=" * 64)


if __name__ == "__main__":
    run_verification()
