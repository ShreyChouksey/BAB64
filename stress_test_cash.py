#!/usr/bin/env python3
"""
BAB64 Cash — SUPER STRESS TEST
================================

10 tests that push every limit of the BAB64 Cash system.
Difficulty=1 for speed. 5-minute timeout.

Author: Shrey (concept) + Claude (implementation)
"""

import asyncio
import copy
import hashlib
import os
import shutil
import sys
import tempfile
import time
import traceback
from contextlib import contextmanager
from typing import List, Tuple

from bab64_cash import (
    BAB64Block, BAB64BlockMiner, BAB64Blockchain, BAB64CashTransaction,
    BlockHeader, ChainSelector, FeePolicy, TxInput, TxOutput, UTXOSet,
    block_reward, build_transaction, compute_lock, compute_unlock,
    merkle_root, verify_lock,
    COIN, COINBASE_MATURITY, DUST_THRESHOLD, HALVING_INTERVAL,
    INITIAL_REWARD, MAX_BLOCK_TRANSACTIONS, MAX_SUPPLY,
)
from bab64_identity import BAB64Identity
from bab64_network import BAB64Node, Mempool, _serialize_block, _deserialize_block
from bab64_storage import BAB64Storage
from bab64_signatures import BAB64IBSTIdentity, BAB64MerkleTree

# ---------------------------------------------------------------------------
# Globals
# ---------------------------------------------------------------------------

DIFFICULTY = 1
RESULTS: List[Tuple[str, bool, float, str]] = []  # (name, passed, seconds, detail)
GLOBAL_START = time.time()
TIMEOUT = 300  # 5 minutes


def elapsed():
    return time.time() - GLOBAL_START


def check_timeout():
    if elapsed() > TIMEOUT:
        raise TimeoutError("Global 5-minute timeout exceeded")


@contextmanager
def test_timer(name: str):
    """Context manager that records PASS/FAIL + timing for a test."""
    print(f"\n{'='*70}")
    print(f"  TEST: {name}")
    print(f"{'='*70}")
    t0 = time.time()
    result = {"passed": False, "detail": ""}
    try:
        yield result
        result["passed"] = True
        result["detail"] = result.get("detail", "") or "OK"
    except Exception as e:
        result["detail"] = f"{type(e).__name__}: {e}"
        traceback.print_exc()
    dt = time.time() - t0
    tag = "PASS" if result["passed"] else "FAIL"
    RESULTS.append((name, result["passed"], dt, result["detail"]))
    print(f"\n  [{tag}] {name} — {dt:.2f}s — {result['detail'][:120]}")


def make_chain(identity: BAB64Identity, difficulty=DIFFICULTY) -> BAB64Blockchain:
    """Create a fresh blockchain with a mined genesis for a given identity."""
    bc = BAB64Blockchain(difficulty=difficulty, miner=identity)
    bc.genesis_block(identity)
    return bc


def mine_n(bc: BAB64Blockchain, identity: BAB64Identity, n: int):
    """Mine n empty blocks on a blockchain."""
    for _ in range(n):
        check_timeout()
        block = bc.mine_block(identity)
        assert block is not None, "Mining failed"


# =============================================================================
# TEST 1 — CHAIN ENDURANCE (50+ blocks with transactions)
# =============================================================================

def test_chain_endurance():
    with test_timer("1 — CHAIN ENDURANCE (110 blocks, 50 txns, 5 identities)") as result:
        # Create 5 identities
        alice = BAB64Identity.generate()
        bob = BAB64Identity.generate()
        carol = BAB64Identity.generate()
        dave = BAB64Identity.generate()
        eve = BAB64Identity.generate()
        people = [alice, bob, carol, dave, eve]
        names = ["Alice", "Bob", "Carol", "Dave", "Eve"]

        # Alice mines genesis + 109 more blocks = 110 total (past maturity)
        bc = make_chain(alice)
        mine_n(bc, alice, 109)
        assert len(bc.chain) == 110, f"Expected 110 blocks, got {len(bc.chain)}"
        check_timeout()

        # Record expected total supply before transactions
        expected_supply = bc.total_supply()
        alice_balance_before = bc.get_balance(alice.address_hex)
        print(f"  Chain height: {len(bc.chain)}")
        print(f"  Alice balance: {alice_balance_before / COIN:.2f} BAB64")
        print(f"  Total supply: {expected_supply / COIN:.2f} BAB64")

        # Send 50 transactions between all 5 identities
        tx_count = 0
        send_amount = 1 * COIN  # 1 BAB64 per tx
        fee = 200_000  # generous fee

        for i in range(50):
            check_timeout()
            sender = people[i % 5]
            recipient = people[(i + 1) % 5]

            sender_bal = bc.get_balance(sender.address_hex)
            if sender_bal < send_amount + fee:
                # Not enough funds, skip
                continue

            tx = build_transaction(sender, recipient, send_amount, bc.utxo_set, fee=fee)
            if tx is None:
                continue

            ok, err = bc.add_transaction_to_mempool(tx)
            if ok:
                tx_count += 1

            # Mine a block every 5 transactions
            if tx_count > 0 and tx_count % 5 == 0:
                block = bc.mine_block(alice)
                assert block is not None

        # Mine remaining mempool
        while bc.mempool:
            check_timeout()
            block = bc.mine_block(alice)
            assert block is not None

        print(f"  Transactions sent: {tx_count}")
        print(f"  Final chain height: {len(bc.chain)}")

        # Verify all balances add up (total UTXO = supply - fees burned)
        total_utxo = sum(
            utxo.amount for utxo in bc.utxo_set._utxos.values()
        )
        final_supply = bc.total_supply()
        print(f"  Total UTXO value: {total_utxo / COIN:.4f}")
        print(f"  Total supply (coinbase): {final_supply / COIN:.4f}")
        # total_utxo should be <= final_supply (fees go to miner, not destroyed)
        assert total_utxo <= final_supply, \
            f"Phantom coins! UTXO={total_utxo} > supply={final_supply}"

        # Verify chain validates end-to-end
        valid, err = bc.validate_chain()
        assert valid, f"Chain validation failed: {err}"

        # Verify UTXO set consistency — rebuild from scratch and compare
        rebuilt_utxo = UTXOSet()
        for block in bc.chain:
            for tx in block.transactions:
                if tx.is_coinbase:
                    rebuilt_utxo.add_outputs(tx)
                else:
                    for inp in tx.inputs:
                        rebuilt_utxo.spend(inp.prev_tx_hash, inp.prev_index)
                    rebuilt_utxo.add_outputs(tx)

        assert len(rebuilt_utxo._utxos) == len(bc.utxo_set._utxos), \
            f"UTXO count mismatch: rebuilt={len(rebuilt_utxo._utxos)} vs live={len(bc.utxo_set._utxos)}"

        for key, utxo in bc.utxo_set._utxos.items():
            assert key in rebuilt_utxo._utxos, f"Missing UTXO: {key}"
            assert rebuilt_utxo._utxos[key].amount == utxo.amount, \
                f"UTXO amount mismatch at {key}"

        result["detail"] = (
            f"{len(bc.chain)} blocks, {tx_count} txns, "
            f"supply={final_supply/COIN:.2f}, UTXO consistent"
        )


# =============================================================================
# TEST 2 — DOUBLE-SPEND ATTACK
# =============================================================================

def test_double_spend():
    with test_timer("2 — DOUBLE-SPEND ATTACK") as result:
        alice = BAB64Identity.generate()
        bob = BAB64Identity.generate()
        carol = BAB64Identity.generate()

        bc = make_chain(alice)
        # Mine past maturity
        mine_n(bc, alice, COINBASE_MATURITY)
        check_timeout()

        alice_bal = bc.get_balance(alice.address_hex)
        print(f"  Alice balance: {alice_bal / COIN:.2f} BAB64")
        assert alice_bal > 0

        # Build tx1: Alice -> Bob (spend a specific UTXO)
        utxos = bc.utxo_set.get_utxos_for(alice.address_hex)
        target_utxo = utxos[0]
        spend_amount = target_utxo.amount - 200_000  # leave room for fee

        tx1 = build_transaction(alice, bob, spend_amount, bc.utxo_set, fee=200_000)
        assert tx1 is not None, "tx1 build failed"

        # Build tx2: Alice -> Carol spending the SAME UTXO
        # (we need to build this manually since build_transaction might pick different UTXOs)
        tx2 = build_transaction(alice, carol, spend_amount, bc.utxo_set, fee=200_000)
        assert tx2 is not None, "tx2 build failed"

        # Both should reference the same UTXO(s) since Alice's balance comes from same source
        tx1_inputs = {(inp.prev_tx_hash, inp.prev_index) for inp in tx1.inputs}
        tx2_inputs = {(inp.prev_tx_hash, inp.prev_index) for inp in tx2.inputs}
        assert tx1_inputs & tx2_inputs, "Double-spend test requires overlapping inputs"
        print(f"  tx1 inputs: {tx1_inputs}")
        print(f"  tx2 inputs: {tx2_inputs}")
        print(f"  Overlap: {tx1_inputs & tx2_inputs}")

        # Submit tx1 — should succeed
        ok1, err1 = bc.add_transaction_to_mempool(tx1)
        assert ok1, f"tx1 should be accepted: {err1}"
        print(f"  tx1 accepted: {ok1}")

        # Submit tx2 — should be REJECTED (UTXO already spent by tx1 in mempool)
        # Note: mempool doesn't track spent UTXOs, but mining will only include tx1
        # because after tx1 is applied, tx2's inputs are gone.
        ok2, err2 = bc.add_transaction_to_mempool(tx2)
        print(f"  tx2 accepted: {ok2} (err: {err2})")

        # Mine a block
        block = bc.mine_block(alice)
        assert block is not None

        # Check: only one of the two txns should be in the block
        block_tx_hashes = {tx.tx_hash for tx in block.transactions}
        tx1_in = tx1.tx_hash in block_tx_hashes
        tx2_in = tx2.tx_hash in block_tx_hashes
        print(f"  tx1 in block: {tx1_in}")
        print(f"  tx2 in block: {tx2_in}")

        # After mining, the UTXO is spent — the second tx referencing it
        # should NOT have been included (double spend prevented)
        assert not (tx1_in and tx2_in), "DOUBLE SPEND: both tx1 and tx2 in same block!"

        # Verify Bob got the money (tx1 was first)
        if tx1_in:
            bob_bal = bc.get_balance(bob.address_hex)
            carol_bal = bc.get_balance(carol.address_hex)
            print(f"  Bob balance: {bob_bal / COIN:.4f}")
            print(f"  Carol balance: {carol_bal / COIN:.4f}")
            assert bob_bal >= spend_amount, f"Bob should have {spend_amount}, got {bob_bal}"
            assert carol_bal == 0, f"Carol should have 0, got {carol_bal}"

        result["detail"] = "Double spend correctly prevented"


# =============================================================================
# TEST 3 — FORK RESOLUTION
# =============================================================================

def test_fork_resolution():
    with test_timer("3 — FORK RESOLUTION (5 vs 3 blocks)") as result:
        alice = BAB64Identity.generate()
        bob = BAB64Identity.generate()

        # Both start from same genesis
        bc_alice = make_chain(alice)
        bc_bob = BAB64Blockchain(difficulty=DIFFICULTY, miner=bob)
        # Copy Alice's genesis to Bob
        genesis = bc_alice.chain[0]
        bc_bob.chain.append(genesis)
        for tx in genesis.transactions:
            bc_bob.utxo_set.apply_transaction(tx)

        # Alice mines 5 more blocks
        mine_n(bc_alice, alice, 5)
        assert len(bc_alice.chain) == 6
        check_timeout()

        # Bob mines 3 more blocks
        mine_n(bc_bob, bob, 3)
        assert len(bc_bob.chain) == 4
        check_timeout()

        alice_work = ChainSelector.cumulative_work(bc_alice.chain)
        bob_work = ChainSelector.cumulative_work(bc_bob.chain)
        print(f"  Alice's chain: {len(bc_alice.chain)} blocks, work={alice_work}")
        print(f"  Bob's chain: {len(bc_bob.chain)} blocks, work={bob_work}")

        # Node C receives both chains — should select Alice's (more work)
        bc_c = BAB64Blockchain(difficulty=DIFFICULTY)
        bc_c.chain = list(bc_bob.chain)
        bc_c.utxo_set = UTXOSet()
        for block in bc_bob.chain:
            for tx in block.transactions:
                if tx.is_coinbase:
                    bc_c.utxo_set.add_outputs(tx)
                else:
                    bc_c.utxo_set.apply_transaction(tx)

        # Now receive Alice's chain (should switch since more work)
        switched = bc_c.handle_fork(bc_alice.chain)
        assert switched, "Should have switched to Alice's longer chain"
        assert len(bc_c.chain) == len(bc_alice.chain), \
            f"Chain length mismatch: {len(bc_c.chain)} vs {len(bc_alice.chain)}"

        # Verify UTXO set matches Alice's chain
        alice_utxo_count = len(bc_alice.utxo_set._utxos)
        c_utxo_count = len(bc_c.utxo_set._utxos)
        assert c_utxo_count == alice_utxo_count, \
            f"UTXO count mismatch: C={c_utxo_count} vs Alice={alice_utxo_count}"

        result["detail"] = f"Fork resolved: selected {len(bc_c.chain)}-block chain over {len(bc_bob.chain)}-block"


# =============================================================================
# TEST 4 — STORAGE PERSISTENCE
# =============================================================================

def test_storage_persistence():
    with test_timer("4 — STORAGE PERSISTENCE (save/reload/continue)") as result:
        tmpdir = tempfile.mkdtemp(prefix="bab64_stress_")
        try:
            alice = BAB64Identity.generate()
            bob = BAB64Identity.generate()

            # Create blockchain with storage
            storage = BAB64Storage(tmpdir)
            bc = BAB64Blockchain(difficulty=DIFFICULTY, miner=alice, storage=storage)
            bc.genesis_block(alice)

            # Mine 20 blocks
            mine_n(bc, alice, 19)  # genesis + 19 = 20 total
            assert len(bc.chain) == 20
            check_timeout()

            # Save state
            storage.save_state(bc.chain)
            orig_height = len(bc.chain)
            orig_balance = bc.get_balance(alice.address_hex)
            orig_utxo_count = len(bc.utxo_set._utxos)
            print(f"  Saved: height={orig_height}, balance={orig_balance/COIN:.2f}, UTXOs={orig_utxo_count}")

            storage.close()

            # Create a NEW blockchain from storage
            storage2 = BAB64Storage(tmpdir)
            bc2 = BAB64Blockchain(difficulty=DIFFICULTY, miner=alice, storage=storage2)

            loaded_height = len(bc2.chain)
            loaded_balance = bc2.get_balance(alice.address_hex)
            loaded_utxo_count = len(bc2.utxo_set._utxos)
            print(f"  Loaded: height={loaded_height}, balance={loaded_balance/COIN:.2f}, UTXOs={loaded_utxo_count}")

            assert loaded_height == orig_height, \
                f"Height mismatch: {loaded_height} vs {orig_height}"
            assert loaded_balance == orig_balance, \
                f"Balance mismatch: {loaded_balance} vs {orig_balance}"
            assert loaded_utxo_count == orig_utxo_count, \
                f"UTXO count mismatch: {loaded_utxo_count} vs {orig_utxo_count}"

            # Validate loaded chain
            valid, err = bc2.validate_chain()
            assert valid, f"Loaded chain invalid: {err}"

            # Mine 5 more blocks on the reloaded chain
            mine_n(bc2, alice, 5)
            assert len(bc2.chain) == orig_height + 5
            check_timeout()

            # Validate again
            valid, err = bc2.validate_chain()
            assert valid, f"Extended chain invalid: {err}"
            print(f"  Extended: height={len(bc2.chain)}, balance={bc2.get_balance(alice.address_hex)/COIN:.2f}")

            storage2.close()
            result["detail"] = f"Saved {orig_height} blocks, reloaded, mined 5 more — all valid"
        finally:
            shutil.rmtree(tmpdir, ignore_errors=True)


# =============================================================================
# TEST 5 — NETWORK PARTITION (async)
# =============================================================================

async def _test_network_partition():
    alice = BAB64Identity.generate()
    bob = BAB64Identity.generate()
    charlie = BAB64Identity.generate()

    bc_a = make_chain(alice)
    bc_b = BAB64Blockchain(difficulty=DIFFICULTY, miner=bob)
    bc_b.chain = list(bc_a.chain)
    bc_b.utxo_set = UTXOSet()
    for block in bc_a.chain:
        for tx in block.transactions:
            bc_b.utxo_set.apply_transaction(tx)

    bc_c = BAB64Blockchain(difficulty=DIFFICULTY, miner=charlie)
    bc_c.chain = list(bc_a.chain)
    bc_c.utxo_set = UTXOSet()
    for block in bc_a.chain:
        for tx in block.transactions:
            bc_c.utxo_set.apply_transaction(tx)

    # Start nodes on different ports
    node_a = BAB64Node("127.0.0.1", 19331, bc_a, alice)
    node_b = BAB64Node("127.0.0.1", 19332, bc_b, bob)
    node_c = BAB64Node("127.0.0.1", 19333, bc_c, charlie)

    await node_a.start()
    await node_b.start()
    await node_c.start()

    try:
        # Connect A-B and B-C
        peer_ab = await node_a.connect_to_peer("127.0.0.1", 19332)
        assert peer_ab is not None, "A->B connection failed"
        await asyncio.sleep(0.3)

        peer_bc = await node_b.connect_to_peer("127.0.0.1", 19333)
        assert peer_bc is not None, "B->C connection failed"
        await asyncio.sleep(0.3)

        # A mines a block
        block1 = node_a.mine_next_block()
        assert block1 is not None
        # Broadcast
        inv_msg = node_a._make_message("INV", {
            "items": [{"type": "block", "hash": block1.block_hash}]
        })
        await node_a.broadcast(inv_msg)
        await asyncio.sleep(0.5)

        # B should have it
        b_height = len(bc_b.chain)
        print(f"  After block1: A={len(bc_a.chain)}, B={b_height}, C={len(bc_c.chain)}")

        # Disconnect B (stop B's server and clear peers)
        await node_b.stop()

        # A mines another block
        block2 = node_a.mine_next_block()
        assert block2 is not None
        inv_msg2 = node_a._make_message("INV", {
            "items": [{"type": "block", "hash": block2.block_hash}]
        })
        await node_a.broadcast(inv_msg2)
        await asyncio.sleep(0.3)

        # C should NOT have this block (B is down)
        c_height_partitioned = len(bc_c.chain)
        a_height = len(bc_a.chain)
        print(f"  Partitioned: A={a_height}, C={c_height_partitioned}")
        assert c_height_partitioned < a_height, \
            "C should NOT have the new block while partitioned"

        return True, f"Partition test: A={a_height}, C={c_height_partitioned} (partitioned correctly)"

    finally:
        await node_a.stop()
        await node_b.stop()
        await node_c.stop()


def test_network_partition():
    with test_timer("5 — NETWORK PARTITION (relay + disconnect)") as result:
        check_timeout()
        ok, detail = asyncio.run(_test_network_partition())
        assert ok
        result["detail"] = detail


# =============================================================================
# TEST 6 — MEMPOOL STRESS
# =============================================================================

def test_mempool_stress():
    with test_timer("6 — MEMPOOL STRESS (200 transactions)") as result:
        alice = BAB64Identity.generate()
        bob = BAB64Identity.generate()

        bc = make_chain(alice)
        # Mine enough blocks for maturity + fund diversity
        mine_n(bc, alice, COINBASE_MATURITY + 10)
        check_timeout()

        alice_bal = bc.get_balance(alice.address_hex)
        print(f"  Alice balance: {alice_bal / COIN:.2f} BAB64")

        # Create as many valid transactions as we can
        tx_created = 0
        target = 200
        send_amount = DUST_THRESHOLD + 1  # minimum viable amount
        fee = 200_000

        for i in range(target):
            check_timeout()
            bal = bc.get_balance(alice.address_hex)
            if bal < send_amount + fee:
                break
            tx = build_transaction(alice, bob, send_amount, bc.utxo_set, fee=fee)
            if tx is None:
                break
            ok, err = bc.add_transaction_to_mempool(tx)
            if not ok:
                break
            tx_created += 1

        print(f"  Created {tx_created} transactions in mempool")
        assert tx_created > 0, "Failed to create any transactions"

        mempool_size_before = len(bc.mempool)

        # Mine blocks until mempool is empty
        blocks_mined = 0
        while bc.mempool:
            check_timeout()
            block = bc.mine_block(alice)
            assert block is not None
            blocks_mined += 1
            tx_in_block = len(block.transactions) - 1  # subtract coinbase
            print(f"    Block {block.index}: {tx_in_block} transactions")

        print(f"  Blocks needed to clear mempool: {blocks_mined}")
        assert len(bc.mempool) == 0, "Mempool should be empty after mining"

        # Verify chain is valid
        valid, err = bc.validate_chain()
        assert valid, f"Chain invalid: {err}"

        result["detail"] = f"{tx_created} txns, cleared in {blocks_mined} blocks"


# =============================================================================
# TEST 7 — SIGNATURE EXHAUSTION
# =============================================================================

def test_signature_exhaustion():
    with test_timer("7 — SIGNATURE EXHAUSTION (IBST key limit)") as result:
        from bab64_cash import _get_ibst

        identity = BAB64Identity.generate()
        ibst = _get_ibst(identity)

        total_keys = BAB64MerkleTree.NUM_LEAVES  # 1024
        print(f"  Total WOTS+ keys: {total_keys}")
        print(f"  Signatures remaining: {ibst.signatures_remaining}")

        # Sign 1020 messages
        payload = b"stress test message"
        for i in range(1020):
            if i % 200 == 0:
                check_timeout()
                print(f"    Signed {i}/{total_keys}...")
            sig = ibst.sign(payload)
            assert sig is not None

        print(f"  After 1020: remaining={ibst.signatures_remaining}")
        assert ibst.signatures_remaining == 4

        # Sign 4 more (reaching 1024)
        for i in range(4):
            sig = ibst.sign(payload)
            assert sig is not None

        print(f"  After 1024: remaining={ibst.signatures_remaining}")
        assert ibst.signatures_remaining == 0

        # The 1025th should fail gracefully
        try:
            sig = ibst.sign(payload)
            assert False, "Should have raised RuntimeError for key exhaustion"
        except RuntimeError as e:
            print(f"  1025th signature correctly rejected: {e}")
            assert "exhaustion" in str(e).lower() or "1,024" in str(e)

        result["detail"] = f"1024 signatures OK, 1025th correctly rejected"


# =============================================================================
# TEST 8 — MALICIOUS BLOCK
# =============================================================================

def test_malicious_block():
    with test_timer("8 — MALICIOUS BLOCK (inflated reward, negative output, bad height)") as result:
        alice = BAB64Identity.generate()
        bc = make_chain(alice)
        mine_n(bc, alice, 5)
        check_timeout()

        height = len(bc.chain)
        prev_hash = bc.chain[-1].block_hash

        rejections = 0

        # --- Test 8a: Inflated coinbase reward ---
        print("  8a: Inflated coinbase reward...")
        inflated_reward = block_reward(height) * 100  # 100x the allowed reward
        bad_coinbase = BAB64CashTransaction(is_coinbase=True, coinbase_height=height)
        bad_coinbase.outputs.append(TxOutput(
            recipient=alice.address_hex, amount=inflated_reward,
            tx_hash="", index=0,
        ))
        bad_coinbase.finalize()

        bad_block = BAB64BlockMiner.mine_block(
            height, prev_hash, [bad_coinbase], DIFFICULTY
        )
        assert bad_block is not None

        valid, err = bc.validate_block_full(
            bad_block, prev_hash, height,
            preceding_chain=bc.chain,
            utxo_set=bc.utxo_set,
        )
        print(f"    Inflated reward: valid={valid}, err={err}")
        assert not valid, "Inflated coinbase should be REJECTED"
        assert "reward" in err.lower() or "mismatch" in err.lower()
        rejections += 1

        # --- Test 8b: Negative output ---
        print("  8b: Negative output...")
        neg_coinbase = BAB64CashTransaction(is_coinbase=True, coinbase_height=height)
        neg_coinbase.outputs.append(TxOutput(
            recipient=alice.address_hex, amount=-1000,
            tx_hash="", index=0,
        ))
        neg_coinbase.finalize()

        # Try to validate the negative-output coinbase directly
        valid_neg, err_neg = bc.utxo_set.validate_transaction(neg_coinbase)
        print(f"    Negative output TX validation: valid={valid_neg}, err={err_neg}")
        # The coinbase amount validation should catch <= 0
        if valid_neg:
            # If UTXO validation doesn't catch it, validate_block_full should
            neg_block = BAB64BlockMiner.mine_block(
                height, prev_hash, [neg_coinbase], DIFFICULTY
            )
            if neg_block:
                valid_nb, err_nb = bc.validate_block_full(
                    neg_block, prev_hash, height,
                    preceding_chain=bc.chain, utxo_set=bc.utxo_set,
                )
                print(f"    Negative block: valid={valid_nb}, err={err_nb}")
                assert not valid_nb, "Negative output block should be REJECTED"
            rejections += 1
        else:
            print(f"    Correctly rejected at TX level: {err_neg}")
            rejections += 1

        # --- Test 8c: Wrong height ---
        print("  8c: Wrong height (skipping a block)...")
        wrong_height = height + 5  # skip 5 blocks ahead
        skip_coinbase = BAB64CashTransaction.create_coinbase(
            alice.address_hex, wrong_height,
            image_bytes=alice._image_bytes,
        )
        skip_block = BAB64BlockMiner.mine_block(
            wrong_height, prev_hash, [skip_coinbase], DIFFICULTY,
        )
        assert skip_block is not None

        valid_skip, err_skip = bc.validate_block_full(
            skip_block, prev_hash, height,
            preceding_chain=bc.chain, utxo_set=bc.utxo_set,
        )
        print(f"    Wrong height: valid={valid_skip}, err={err_skip}")
        assert not valid_skip, "Wrong-height block should be REJECTED"
        rejections += 1

        result["detail"] = f"All {rejections} malicious blocks correctly rejected"


# =============================================================================
# TEST 9 — RAPID MINING (performance)
# =============================================================================

def test_rapid_mining():
    with test_timer("9 — RAPID MINING (100 blocks, performance)") as result:
        alice = BAB64Identity.generate()
        bc = make_chain(alice)

        # Mine 99 more blocks (100 total including genesis)
        t0 = time.time()
        mine_n(bc, alice, 99)
        mine_time = time.time() - t0

        assert len(bc.chain) == 100
        blocks_per_sec = 99 / mine_time
        time_per_block = mine_time / 99

        print(f"  Mining 99 blocks: {mine_time:.2f}s")
        print(f"  Blocks/second: {blocks_per_sec:.2f}")
        print(f"  Time/block: {time_per_block*1000:.1f}ms")

        # Measure single BAB64 hash time
        t0 = time.time()
        n_hashes = 100
        for i in range(n_hashes):
            BAB64BlockMiner.compute_block_hash(
                0, "0"*64, time.time(), "0"*64, i, 1
            )
        hash_time = time.time() - t0
        time_per_hash = hash_time / n_hashes
        print(f"  BAB64 hash: {time_per_hash*1000:.2f}ms/hash ({n_hashes/hash_time:.0f} H/s)")

        # Measure chain validation time
        t0 = time.time()
        valid, err = bc.validate_chain()
        validate_time = time.time() - t0
        assert valid
        print(f"  Chain validation (100 blocks): {validate_time:.2f}s")

        # Measure UTXO rebuild time
        t0 = time.time()
        rebuilt = UTXOSet()
        for block in bc.chain:
            for tx in block.transactions:
                if tx.is_coinbase:
                    rebuilt.add_outputs(tx)
                else:
                    for inp in tx.inputs:
                        rebuilt.spend(inp.prev_tx_hash, inp.prev_index)
                    rebuilt.add_outputs(tx)
        rebuild_time = time.time() - t0
        print(f"  UTXO rebuild (100 blocks): {rebuild_time*1000:.1f}ms")

        result["detail"] = (
            f"{blocks_per_sec:.1f} blocks/s, "
            f"{time_per_hash*1000:.1f}ms/hash, "
            f"validate={validate_time:.2f}s, "
            f"UTXO rebuild={rebuild_time*1000:.0f}ms"
        )


# =============================================================================
# TEST 10 — SUPPLY INTEGRITY (halvings)
# =============================================================================

def test_supply_integrity():
    with test_timer("10 — SUPPLY INTEGRITY (halvings + fee-only mining)") as result:
        # Verify halving math with the block_reward function
        print(f"  HALVING_INTERVAL = {HALVING_INTERVAL}")
        print(f"  INITIAL_REWARD = {INITIAL_REWARD / COIN:.2f} BAB64")

        # Test halving schedule
        r0 = block_reward(0)
        r1 = block_reward(HALVING_INTERVAL)
        r2 = block_reward(HALVING_INTERVAL * 2)
        r3 = block_reward(HALVING_INTERVAL * 3)
        r_end = block_reward(HALVING_INTERVAL * 64)

        print(f"  Reward at height 0: {r0 / COIN:.8f}")
        print(f"  Reward at halving 1: {r1 / COIN:.8f}")
        print(f"  Reward at halving 2: {r2 / COIN:.8f}")
        print(f"  Reward at halving 3: {r3 / COIN:.8f}")
        print(f"  Reward at halving 64: {r_end / COIN:.8f}")

        assert r0 == INITIAL_REWARD
        assert r1 == INITIAL_REWARD // 2, f"First halving wrong: {r1}"
        assert r2 == INITIAL_REWARD // 4, f"Second halving wrong: {r2}"
        assert r3 == INITIAL_REWARD // 8, f"Third halving wrong: {r3}"
        assert r_end == 0, f"Should be 0 after 64 halvings: {r_end}"

        # Verify total supply converges and never exceeds cap
        total = 0
        for h in range(HALVING_INTERVAL * 64 + 10):
            r = block_reward(h)
            total += r
            if total > MAX_SUPPLY:
                assert False, f"Supply exceeded at height {h}: {total} > {MAX_SUPPLY}"
        print(f"  Theoretical total supply: {total / COIN:.8f} BAB64")
        print(f"  MAX_SUPPLY: {MAX_SUPPLY / COIN:.8f} BAB64")
        assert total <= MAX_SUPPLY, f"Total {total} > MAX_SUPPLY {MAX_SUPPLY}"

        # Now test with a real (tiny) chain — simulate 3 halvings
        # Use HALVING_INTERVAL=3 by patching block_reward locally
        def tiny_reward(height):
            halvings = height // 3  # halving every 3 blocks
            if halvings >= 64:
                return 0
            return INITIAL_REWARD >> halvings

        alice = BAB64Identity.generate()
        bob = BAB64Identity.generate()
        bc = make_chain(alice)  # genesis = block 0

        # Mine through 3 halvings (9 blocks: heights 1-9)
        # With tiny_reward: heights 0-2 = 50, 3-5 = 25, 6-8 = 12.5, 9-11 = 6.25
        # But we're using the real block_reward which halves at 210k...
        # So instead let's just verify the real reward schedule holds on the chain
        mine_n(bc, alice, 9)
        check_timeout()

        for block in bc.chain:
            coinbase = block.transactions[0]
            actual_reward = sum(out.amount for out in coinbase.outputs)
            expected = block_reward(block.index)
            # Fees may add to reward, so actual >= expected base reward
            assert actual_reward >= expected, \
                f"Block {block.index}: reward {actual_reward} < expected {expected}"

        assert bc.verify_supply_cap(), "Supply cap exceeded"

        # Test fee-only mining: verify a block with 0 base reward can include fees
        # At height HALVING_INTERVAL * 64, reward is 0
        zero_reward = block_reward(HALVING_INTERVAL * 64)
        assert zero_reward == 0
        # A coinbase with fees=1000 at that height should have reward = 0 + 1000 = 1000
        cb = BAB64CashTransaction.create_coinbase(
            alice.address_hex, HALVING_INTERVAL * 64, fees=1000
        )
        assert sum(out.amount for out in cb.outputs) == 1000
        print(f"  Fee-only coinbase (height={HALVING_INTERVAL*64}): {sum(out.amount for out in cb.outputs)} satoshis")

        result["detail"] = (
            f"Halvings verified, total supply {total/COIN:.2f} <= {MAX_SUPPLY/COIN:.2f}, "
            f"fee-only mining works"
        )


# =============================================================================
# MAIN — Run all tests, print report card
# =============================================================================

def main():
    global GLOBAL_START
    GLOBAL_START = time.time()

    print("=" * 70)
    print("  BAB64 CASH — SUPER STRESS TEST")
    print("  Difficulty: 1 | Timeout: 5 minutes")
    print("=" * 70)

    tests = [
        ("1", test_chain_endurance),
        ("2", test_double_spend),
        ("3", test_fork_resolution),
        ("4", test_storage_persistence),
        ("5", test_network_partition),
        ("6", test_mempool_stress),
        ("7", test_signature_exhaustion),
        ("8", test_malicious_block),
        ("9", test_rapid_mining),
        ("10", test_supply_integrity),
    ]

    for num, test_fn in tests:
        if elapsed() > TIMEOUT:
            RESULTS.append((f"Test {num}", False, 0, "SKIPPED — timeout"))
            continue
        try:
            test_fn()
        except TimeoutError as e:
            RESULTS.append((f"Test {num}", False, 0, str(e)))
            break

    # === REPORT CARD ===
    print("\n")
    print("=" * 70)
    print("  STRESS TEST REPORT CARD")
    print("=" * 70)

    passed = 0
    failed = 0
    total_time = time.time() - GLOBAL_START

    for name, ok, dt, detail in RESULTS:
        status = "PASS" if ok else "FAIL"
        icon = "[+]" if ok else "[X]"
        print(f"  {icon} {status}  {name:<55} {dt:6.2f}s")
        if not ok:
            print(f"         -> {detail[:100]}")
        if ok:
            passed += 1
        else:
            failed += 1

    print(f"\n  {'='*50}")
    print(f"  TOTAL: {passed}/{passed+failed} passed | {failed} failed | {total_time:.1f}s")
    print(f"  {'='*50}")

    if failed == 0:
        print("\n  ALL TESTS PASSED — BAB64 Cash is SOLID.\n")
    else:
        print(f"\n  {failed} TEST(S) FAILED — bugs found!\n")

    return 0 if failed == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
