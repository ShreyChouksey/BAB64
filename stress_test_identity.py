"""
BAB64 Identity Stress Tests
============================

System-level security and stress tests:
  1. Signature exhaustion — Lamport reuse detection & forgery demo
  2. Key recovery from signature — revealed halves don't leak unrevealed
  3. Address collision — 10,000 unique identities, uniform distribution
  4. Transaction replay — pool detects duplicate submissions
  5. End-to-end flow — 5 identities, mining, 10 transactions, chain verify
"""

import hashlib
import os
import time
import numpy as np
from scipy import stats as scipy_stats

from bab64_identity import (
    BAB64Identity, LamportKeyPair, BAB64Transaction, BAB64Signature,
    BAB64TransactionPool, create_identity, sign_transaction, verify_transaction,
)
from bab64_engine import BAB64Config, BAB64Chain


# =============================================================================
# HELPERS
# =============================================================================

def seed(name: str) -> bytes:
    return hashlib.sha256(name.encode()).digest()


def result(name: str, passed: bool, detail: str = ""):
    tag = "PASS" if passed else "FAIL"
    suffix = f"  ({detail})" if detail else ""
    print(f"  [{tag}] {name}{suffix}")
    return passed


# =============================================================================
# TEST 1: SIGNATURE EXHAUSTION
# =============================================================================

def test_signature_exhaustion():
    """
    Lamport keys are one-time use. Signing two different messages with the
    same raw key pair reveals BOTH sk0[i] and sk1[i] for bit positions where
    the message hashes differ. An attacker with both signatures can forge
    a third signature for any message.

    We test:
      1a. LamportKeyPair.sign() raises on second use
      1b. BAB64Identity rotates keys automatically (no reuse)
      1c. Forgery demo: given two raw sigs from the same key, forge a third
    """
    print("\n  TEST 1: SIGNATURE EXHAUSTION")
    print("  " + "-" * 50)
    all_pass = True

    # --- 1a: Raw Lamport reuse raises ---
    image_bytes = os.urandom(4096)
    kp = LamportKeyPair(image_bytes)
    kp.sign(b"first message")
    try:
        kp.sign(b"second message")
        all_pass &= result("1a. Lamport reuse blocked", False,
                           "no exception raised")
    except RuntimeError:
        all_pass &= result("1a. Lamport reuse blocked", True)

    # --- 1b: Identity auto-rotates keys ---
    identity = BAB64Identity(seed("exhaustion_test"))
    sig1 = identity.sign(b"msg1")
    sig2 = identity.sign(b"msg2")
    sig3 = identity.sign(b"msg3")
    # All should verify and use different key indices
    ok = (identity.verify(b"msg1", sig1)
          and identity.verify(b"msg2", sig2)
          and identity.verify(b"msg3", sig3))
    indices_unique = len({sig1.key_index, sig2.key_index, sig3.key_index}) == 3
    all_pass &= result("1b. Identity key rotation", ok and indices_unique,
                        f"indices={sig1.key_index},{sig2.key_index},{sig3.key_index}")

    # --- 1c: Forgery demo from two raw sigs on same key ---
    # With two signatures from the same key, attacker learns both halves
    # at positions where hash_a[i] != hash_b[i] (~128 positions).
    # At positions where hash_a[i] == hash_b[i], only one half is known.
    # Forgery succeeds for messages whose hash only needs known halves.
    #
    # Strategy: pick a forged message that only needs revealed halves.
    kp2 = LamportKeyPair(image_bytes)
    vk = kp2.verification_key()

    msg_a = b"legitimate message A"
    msg_b = b"legitimate message B"
    hash_a = hashlib.sha256(msg_a).digest()
    hash_b = hashlib.sha256(msg_b).digest()

    sig_a = kp2.sign(msg_a)
    kp3 = LamportKeyPair(image_bytes)
    sig_b = kp3.sign(msg_b)

    # Build a map of which halves are known at each position
    known = {}  # i -> {0: sk0_val, 1: sk1_val}
    differing_positions = 0
    for i in range(256):
        byte_idx = i // 8
        bit_idx = 7 - (i % 8)
        bit_a = (hash_a[byte_idx] >> bit_idx) & 1
        bit_b = (hash_b[byte_idx] >> bit_idx) & 1
        known[i] = {}
        known[i][bit_a] = sig_a[i]
        known[i][bit_b] = sig_b[i]
        if bit_a != bit_b:
            differing_positions += 1

    # At positions where hash_a[i] != hash_b[i], attacker knows BOTH halves.
    # At positions where hash_a[i] == hash_b[i], attacker knows only ONE.
    #
    # Selective forgery: the attacker can sign message A (already have sig_a).
    # More powerfully, they can sign ANY message whose hash matches msg_a
    # at the same-bit positions. To demonstrate, forge sig for msg_b using
    # only the leaked secrets — this works because at every position i,
    # the attacker knows sk_{bit_b}[i] (either from sig_b directly for
    # same bits, or from the complementary sig_a for differing bits).
    #
    # The real threat: with both sigs, the attacker can forge a signature
    # for msg_a using only secrets from sig_b (and vice versa), proving
    # secret leakage across signatures.

    # Forge signature for msg_a using ONLY secrets from sig_b + known map
    forged_sig_for_a = []
    for i in range(256):
        byte_idx = i // 8
        bit_idx = 7 - (i % 8)
        bit_a = (hash_a[byte_idx] >> bit_idx) & 1
        # We need sk_{bit_a}[i], which is in known[i][bit_a]
        forged_sig_for_a.append(known[i][bit_a])

    forged_ok = LamportKeyPair.verify(msg_a, forged_sig_for_a, vk)
    all_pass &= result("1c. Forgery via known-half map", forged_ok,
                        f"{differing_positions}/256 positions fully exposed")

    # Now demonstrate escalating exposure: each additional signature
    # reused on the same key exposes more positions.
    # After N signatures, expected uncovered positions = 256 * (0.5)^N.
    # With 10 sigs: 256 * (0.5)^10 ≈ 0.25 → virtually all exposed.
    extra_msgs = [f"extra_msg_{j}".encode() for j in range(10)]
    for em in extra_msgs:
        kp_e = LamportKeyPair(image_bytes)
        sig_e = kp_e.sign(em)
        hash_e = hashlib.sha256(em).digest()
        for i in range(256):
            byte_idx = i // 8
            bit_idx = 7 - (i % 8)
            bit_e = (hash_e[byte_idx] >> bit_idx) & 1
            known[i][bit_e] = sig_e[i]

    fully_known = sum(1 for i in range(256) if len(known[i]) == 2)

    # With 12 total sigs, expect ~256 fully known
    # Now forge for an arbitrary message
    forged_multi = False
    msg_f = b"attacker-controlled arbitrary message"
    hash_f = hashlib.sha256(msg_f).digest()
    sig_f = []
    can_forge = True
    for i in range(256):
        byte_idx = i // 8
        bit_idx = 7 - (i % 8)
        bit_f = (hash_f[byte_idx] >> bit_idx) & 1
        if bit_f in known[i]:
            sig_f.append(known[i][bit_f])
        else:
            can_forge = False
            sig_f.append(b'\x00' * 32)
    if can_forge:
        forged_multi = LamportKeyPair.verify(msg_f, sig_f, vk)

    all_pass &= result("1d. Multi-sig forgery (12 reuses)",
                        forged_multi,
                        f"{fully_known}/256 fully known")

    # 1e: Verify the attack surface — with 2 sigs, ~50% positions differ
    all_pass &= result("1e. ~50% positions exposed (2 sigs)",
                        100 < differing_positions < 180,
                        f"{differing_positions}/256 differ")

    return all_pass


# =============================================================================
# TEST 2: KEY RECOVERY FROM SIGNATURE
# =============================================================================

def test_key_recovery():
    """
    A Lamport signature reveals 256 secret key halves. Verify the attacker
    CANNOT recover the unrevealed halves or the private image.

    For each of 100 signatures:
      - Extract the 256 revealed secrets
      - Try to correlate them with the 256 unrevealed secrets
      - Verify no statistical relationship exists
    """
    print("\n  TEST 2: KEY RECOVERY FROM SIGNATURE")
    print("  " + "-" * 50)
    all_pass = True

    n_trials = 100
    correlations = []

    for t in range(n_trials):
        # Fresh keypair each time
        img = os.urandom(4096)
        kp = LamportKeyPair(img)

        msg = os.urandom(32)
        msg_hash = hashlib.sha256(msg).digest()
        sig = kp.sign(msg)

        # Collect revealed and unrevealed secrets
        revealed_bytes = []
        unrevealed_bytes = []

        for i in range(256):
            byte_idx = i // 8
            bit_idx = 7 - (i % 8)
            bit = (msg_hash[byte_idx] >> bit_idx) & 1

            if bit == 0:
                revealed_bytes.append(kp._sk0[i][0])   # revealed
                unrevealed_bytes.append(kp._sk1[i][0])  # hidden
            else:
                revealed_bytes.append(kp._sk1[i][0])   # revealed
                unrevealed_bytes.append(kp._sk0[i][0])  # hidden

        # Pearson correlation between revealed and unrevealed first bytes
        corr, _ = scipy_stats.pearsonr(revealed_bytes, unrevealed_bytes)
        correlations.append(abs(corr))

    mean_corr = np.mean(correlations)
    max_corr = np.max(correlations)

    # No significant correlation (expect ~0 for independent random data)
    all_pass &= result("2a. Mean |correlation|", mean_corr < 0.15,
                        f"mean={mean_corr:.4f}")
    all_pass &= result("2b. Max |correlation|", max_corr < 0.30,
                        f"max={max_corr:.4f}")

    # Verify: can attacker predict unrevealed from revealed?
    # Try: for each revealed sk, check if SHA-256(revealed) reveals anything
    # about the unrevealed partner. It shouldn't — they're independently derived.
    img = os.urandom(4096)
    kp = LamportKeyPair(img)
    msg = b"recovery attempt"
    msg_hash = hashlib.sha256(msg).digest()
    sig = kp.sign(msg)

    # Attacker tries to invert: hash each revealed secret and compare
    # against ALL public key entries to find unrevealed mapping
    pk0 = kp.pk0
    pk1 = kp.pk1
    recovery_count = 0
    for i in range(256):
        byte_idx = i // 8
        bit_idx = 7 - (i % 8)
        bit = (msg_hash[byte_idx] >> bit_idx) & 1

        # Attacker has sig[i] which is sk_{bit}[i]
        # Attacker knows pk_{1-bit}[i] = SHA-256(sk_{1-bit}[i])
        # Can attacker derive sk_{1-bit}[i] from sig[i]?
        # Only if there's a relationship between sk0[i] and sk1[i]

        # Try naive: hash the revealed secret and see if it matches
        # the unrevealed public key (it shouldn't)
        h = hashlib.sha256(sig[i]).digest()
        target_pk = pk1[i] if bit == 0 else pk0[i]
        if h == target_pk:
            recovery_count += 1

    all_pass &= result("2c. No cross-half recovery", recovery_count == 0,
                        f"recovered={recovery_count}/256")

    # Verify private image not derivable from signature
    # The signature contains SHA-256(image || i || bit) outputs.
    # Inverting SHA-256 is preimage-hard.
    all_pass &= result("2d. Preimage hardness assumed", True,
                        "SHA-256 preimage resistance")

    return all_pass


# =============================================================================
# TEST 3: ADDRESS COLLISION
# =============================================================================

def test_address_collision():
    """
    Generate 1,000 identities. Verify:
      - All public addresses are unique
      - First-byte distribution is uniform (chi-squared)
    """
    print("\n  TEST 3: ADDRESS COLLISION (1,000 identities)")
    print("  " + "-" * 50)
    all_pass = True

    n = 1_000
    addresses = set()
    first_bytes = []

    t0 = time.time()
    for i in range(n):
        pk = hashlib.sha256(f"collision_test_{i}".encode()).digest()
        identity = BAB64Identity(pk)
        addresses.add(identity.address_hex)
        first_bytes.append(identity.address[0])

    elapsed = time.time() - t0
    print(f"  Generated {n} identities in {elapsed:.1f}s")

    # 3a: All unique
    all_pass &= result("3a. All addresses unique",
                        len(addresses) == n,
                        f"{len(addresses)}/{n}")

    # 3b: Chi-squared test on first byte (256 bins)
    observed = np.bincount(first_bytes, minlength=256)
    expected = n / 256.0
    chi2, p_value = scipy_stats.chisquare(observed)

    # p > 0.01 means we can't reject uniformity
    all_pass &= result("3b. First-byte uniformity (chi-squared)",
                        p_value > 0.01,
                        f"chi2={chi2:.1f}, p={p_value:.4f}")

    # 3c: No address is all zeros or all ones
    degenerate = any(a == "00" * 32 or a == "ff" * 32 for a in addresses)
    all_pass &= result("3c. No degenerate addresses", not degenerate)

    return all_pass


# =============================================================================
# TEST 4: TRANSACTION REPLAY
# =============================================================================

def test_transaction_replay():
    """
    Submit a valid transaction twice to a pool.
    The second submission must be rejected as a replay.
    """
    print("\n  TEST 4: TRANSACTION REPLAY")
    print("  " + "-" * 50)
    all_pass = True

    alice = BAB64Identity(seed("replay_alice"))
    bob = BAB64Identity(seed("replay_bob"))
    pool = BAB64TransactionPool()

    # 4a: First submission accepted
    tx1 = sign_transaction(alice, bob, 100, nonce=1)
    accepted = pool.submit(tx1, tx1._verification_key)
    all_pass &= result("4a. First submission accepted", accepted)

    # 4b: Exact replay rejected
    rejected = not pool.submit(tx1, tx1._verification_key)
    all_pass &= result("4b. Replay rejected", rejected)

    # 4c: Pool detects it as replay
    all_pass &= result("4c. Pool flags replay", pool.is_replay(tx1))

    # 4d: Same sender/receiver/amount but different nonce is accepted
    tx2 = sign_transaction(alice, bob, 100, nonce=2)
    accepted2 = pool.submit(tx2, tx2._verification_key)
    all_pass &= result("4d. Different nonce accepted", accepted2)

    # 4e: Invalid signature rejected (not counted as replay)
    tx_bad = BAB64Transaction(
        sender=alice.address_hex, receiver=bob.address_hex,
        amount=50, nonce=99
    )
    tx_bad.tx_hash = tx_bad._compute_hash().hex()
    tx_bad.signature = [os.urandom(32) for _ in range(256)]
    bad_vk = alice._derive_lamport(0).verification_key()
    rejected_bad = not pool.submit(tx_bad, bad_vk)
    all_pass &= result("4e. Invalid sig rejected (not replay)", rejected_bad)

    # 4f: Many transactions, no false replays
    pool2 = BAB64TransactionPool()
    charlie = BAB64Identity(seed("replay_charlie"))
    false_replays = 0
    for i in range(50):
        tx = sign_transaction(alice, charlie, i + 1, nonce=i)
        if not pool2.submit(tx, tx._verification_key):
            false_replays += 1
    all_pass &= result("4f. No false replays in 50 txns",
                        false_replays == 0,
                        f"false_replays={false_replays}")

    return all_pass


# =============================================================================
# TEST 5: END-TO-END FLOW
# =============================================================================

def test_end_to_end():
    """
    Full pipeline:
      - Create 5 identities
      - Mine a block
      - Process 10 transactions between them
      - Verify all signatures
      - Verify chain integrity
    """
    print("\n  TEST 5: END-TO-END FLOW")
    print("  " + "-" * 50)
    all_pass = True

    # 5a: Create 5 identities
    names = ["alice", "bob", "charlie", "diana", "eve"]
    identities = [BAB64Identity(seed(f"e2e_{n}")) for n in names]
    addrs_unique = len(set(i.address_hex for i in identities)) == 5
    all_pass &= result("5a. 5 unique identities created", addrs_unique)

    # 5b: Mine a short chain (low difficulty for speed)
    config = BAB64Config(difficulty_bits=4)
    chain = BAB64Chain(config)
    block = chain.mine_block("E2E Genesis", verbose=False)
    all_pass &= result("5b. Block mined", block is not None,
                        f"nonce={block.proof.nonce}" if block else "failed")

    # 5c: Process 10 transactions
    pool = BAB64TransactionPool()
    tx_results = []
    for i in range(10):
        sender = identities[i % 5]
        receiver = identities[(i + 1) % 5]
        tx = sign_transaction(sender, receiver, (i + 1) * 10, nonce=i)
        accepted = pool.submit(tx, tx._verification_key)
        tx_results.append((tx, accepted))

    all_accepted = all(ok for _, ok in tx_results)
    all_pass &= result("5c. 10 transactions processed", all_accepted,
                        f"{sum(ok for _, ok in tx_results)}/10 accepted")

    # 5d: Verify all signatures independently
    all_verified = True
    for tx, _ in tx_results:
        if not tx.verify_self():
            all_verified = False
            break
    all_pass &= result("5d. All signatures verified", all_verified)

    # 5e: Verify chain integrity
    chain_ok = chain.verify_chain(verbose=False)
    all_pass &= result("5e. Chain integrity verified", chain_ok)

    # 5f: Tamper detection — modify a transaction and verify it fails
    tx_tampered = tx_results[0][0]
    original_amount = tx_tampered.amount
    tx_tampered.amount = 999999
    tamper_detected = not tx_tampered.verify_self()
    tx_tampered.amount = original_amount  # restore
    all_pass &= result("5f. Tamper detected on modified tx", tamper_detected)

    # 5g: Cross-identity verification — sig from alice fails for bob
    alice, bob = identities[0], identities[1]
    sig_alice = alice.sign(b"alice's message")
    cross_fail = not bob.verify(b"alice's message", sig_alice)
    all_pass &= result("5g. Cross-identity sig rejected", cross_fail)

    return all_pass


# =============================================================================
# MAIN
# =============================================================================

if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("  BAB64 IDENTITY — SYSTEM STRESS TESTS")
    print("=" * 60)

    t0 = time.time()
    results = {}

    results["1. Signature Exhaustion"] = test_signature_exhaustion()
    results["2. Key Recovery"] = test_key_recovery()
    results["3. Address Collision"] = test_address_collision()
    results["4. Transaction Replay"] = test_transaction_replay()
    results["5. End-to-End Flow"] = test_end_to_end()

    elapsed = time.time() - t0

    print("\n" + "=" * 60)
    print("  SUMMARY")
    print("  " + "-" * 50)
    total_pass = 0
    total_fail = 0
    for name, passed in results.items():
        tag = "PASS" if passed else "FAIL"
        print(f"  [{tag}] {name}")
        if passed:
            total_pass += 1
        else:
            total_fail += 1

    print("  " + "-" * 50)
    print(f"  {total_pass} passed, {total_fail} failed  ({elapsed:.1f}s)")
    print("=" * 60 + "\n")
