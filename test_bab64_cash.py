"""
Tests for BAB64 Cash — UTXO-based electronic cash system.

30+ tests covering:
  - UTXO creation and spending
  - Double-spend rejection
  - Insufficient funds rejection
  - Coinbase transaction validity
  - Full block mining and validation
  - Chain of 5 blocks with transactions between 3 identities
  - Balance tracking across blocks
  - Merkle root verification
  - Transaction fee calculation
  - Spending someone else's UTXO rejected
  - Negative amount rejected
  - Empty block (only coinbase) valid
"""

import hashlib
import pytest
from bab64_engine import BAB64Config
from bab64_identity import BAB64Identity
from bab64_cash import (
    COIN, INITIAL_REWARD, HALVING_INTERVAL, MAX_SUPPLY,
    TxOutput, TxInput,
    BAB64CashTransaction, UTXOSet,
    BAB64Block, BAB64BlockMiner,
    BAB64Blockchain,
    block_reward, merkle_root, build_transaction,
    compute_lock, compute_unlock, verify_lock,
)


# =============================================================================
# Fixtures
# =============================================================================

@pytest.fixture
def config():
    return BAB64Config()


@pytest.fixture
def alice(config):
    return BAB64Identity(b'\x01' * 32, config)


@pytest.fixture
def bob(config):
    return BAB64Identity(b'\x02' * 32, config)


@pytest.fixture
def carol(config):
    return BAB64Identity(b'\x03' * 32, config)


@pytest.fixture
def utxo_set():
    return UTXOSet()


@pytest.fixture
def funded_utxo(alice, utxo_set):
    """UTXO set with a coinbase giving Alice 50 BAB64 (hashlocked)."""
    coinbase = BAB64CashTransaction.create_coinbase(
        alice.address_hex, height=0, image_bytes=alice._image_bytes
    )
    utxo_set.apply_transaction(coinbase)
    return utxo_set, coinbase


# =============================================================================
# UTXO Creation and Spending
# =============================================================================

class TestUTXO:
    def test_add_outputs(self, utxo_set, alice):
        """UTXOs are added and retrievable."""
        coinbase = BAB64CashTransaction.create_coinbase(
            alice.address_hex, 0, image_bytes=alice._image_bytes
        )
        utxo_set.add_outputs(coinbase)

        utxo = utxo_set.get(coinbase.tx_hash, 0)
        assert utxo is not None
        assert utxo.recipient == alice.address_hex
        assert utxo.amount == INITIAL_REWARD

    def test_spend_utxo(self, utxo_set, alice):
        """Spending removes the UTXO."""
        coinbase = BAB64CashTransaction.create_coinbase(
            alice.address_hex, 0, image_bytes=alice._image_bytes
        )
        utxo_set.add_outputs(coinbase)

        spent = utxo_set.spend(coinbase.tx_hash, 0)
        assert spent is not None
        assert spent.amount == INITIAL_REWARD
        assert utxo_set.get(coinbase.tx_hash, 0) is None

    def test_spend_nonexistent(self, utxo_set):
        """Spending a non-existent UTXO returns None."""
        assert utxo_set.spend("deadbeef", 0) is None

    def test_balance(self, utxo_set, alice, bob):
        """Balance sums all UTXOs for an address."""
        cb1 = BAB64CashTransaction.create_coinbase(
            alice.address_hex, 0, image_bytes=alice._image_bytes
        )
        cb2 = BAB64CashTransaction.create_coinbase(
            alice.address_hex, 1, image_bytes=alice._image_bytes
        )
        cb3 = BAB64CashTransaction.create_coinbase(
            bob.address_hex, 2, image_bytes=bob._image_bytes
        )
        utxo_set.add_outputs(cb1)
        utxo_set.add_outputs(cb2)
        utxo_set.add_outputs(cb3)

        assert utxo_set.balance(alice.address_hex) == 2 * INITIAL_REWARD
        assert utxo_set.balance(bob.address_hex) == INITIAL_REWARD

    def test_get_utxos_for(self, funded_utxo, alice):
        """Retrieve all UTXOs for a specific address."""
        utxo_set, _ = funded_utxo
        utxos = utxo_set.get_utxos_for(alice.address_hex)
        assert len(utxos) == 1
        assert utxos[0].amount == INITIAL_REWARD


# =============================================================================
# Coinbase Transactions
# =============================================================================

class TestCoinbase:
    def test_coinbase_valid(self, utxo_set, alice):
        """Coinbase transaction passes validation."""
        coinbase = BAB64CashTransaction.create_coinbase(alice.address_hex, 0)
        valid, err = utxo_set.validate_transaction(coinbase)
        assert valid, err

    def test_coinbase_has_no_inputs(self, alice):
        """Coinbase must have no inputs."""
        coinbase = BAB64CashTransaction.create_coinbase(alice.address_hex, 0)
        assert coinbase.is_coinbase
        assert len(coinbase.inputs) == 0

    def test_coinbase_reward_amount(self, alice):
        """Coinbase pays the correct block reward."""
        coinbase = BAB64CashTransaction.create_coinbase(alice.address_hex, 0)
        assert coinbase.outputs[0].amount == INITIAL_REWARD

    def test_coinbase_with_fees(self, alice):
        """Coinbase includes transaction fees."""
        fees = 1000
        coinbase = BAB64CashTransaction.create_coinbase(alice.address_hex, 0, fees)
        assert coinbase.outputs[0].amount == INITIAL_REWARD + fees

    def test_coinbase_tx_hash_set(self, alice):
        """Coinbase has a valid tx_hash after creation."""
        coinbase = BAB64CashTransaction.create_coinbase(alice.address_hex, 0)
        assert len(coinbase.tx_hash) == 64

    def test_coinbase_different_heights_different_hash(self, alice):
        """Coinbase txs at different heights have different hashes."""
        cb0 = BAB64CashTransaction.create_coinbase(alice.address_hex, 0)
        cb1 = BAB64CashTransaction.create_coinbase(alice.address_hex, 1)
        assert cb0.tx_hash != cb1.tx_hash


# =============================================================================
# Block Reward Halving
# =============================================================================

class TestHalving:
    def test_initial_reward(self):
        assert block_reward(0) == 50 * COIN

    def test_first_halving(self):
        assert block_reward(HALVING_INTERVAL) == 25 * COIN

    def test_second_halving(self):
        assert block_reward(2 * HALVING_INTERVAL) == 12 * COIN + 50_000_000

    def test_reward_eventually_zero(self):
        assert block_reward(64 * HALVING_INTERVAL) == 0


# =============================================================================
# Regular Transactions
# =============================================================================

class TestTransactions:
    def test_valid_transaction(self, funded_utxo, alice, bob):
        """A properly signed transaction passes validation."""
        utxo_set, _ = funded_utxo
        tx = build_transaction(alice, bob, 10 * COIN, utxo_set)
        assert tx is not None
        valid, err = utxo_set.validate_transaction(tx)
        assert valid, err

    def test_transaction_with_change(self, funded_utxo, alice, bob):
        """Change is returned to sender."""
        utxo_set, _ = funded_utxo
        send_amount = 10 * COIN
        tx = build_transaction(alice, bob, send_amount, utxo_set)
        assert tx is not None
        assert len(tx.outputs) == 2
        assert tx.outputs[0].amount == send_amount
        assert tx.outputs[0].recipient == bob.address_hex
        assert tx.outputs[1].amount == INITIAL_REWARD - send_amount
        assert tx.outputs[1].recipient == alice.address_hex

    def test_exact_amount_no_change(self, funded_utxo, alice, bob):
        """Spending exact amount produces no change output."""
        utxo_set, _ = funded_utxo
        tx = build_transaction(alice, bob, INITIAL_REWARD, utxo_set)
        assert tx is not None
        assert len(tx.outputs) == 1

    def test_transaction_fee(self, funded_utxo, alice, bob):
        """Transaction fee = input - output."""
        utxo_set, _ = funded_utxo
        fee = 1000
        send_amount = 10 * COIN
        tx = build_transaction(alice, bob, send_amount, utxo_set, fee=fee)
        assert tx is not None
        calculated_fee = tx.fee(utxo_set)
        assert calculated_fee == fee

    def test_apply_transaction(self, funded_utxo, alice, bob):
        """Applying a transaction updates the UTXO set."""
        utxo_set, _ = funded_utxo
        send_amount = 10 * COIN
        tx = build_transaction(alice, bob, send_amount, utxo_set)
        assert tx is not None

        valid, _ = utxo_set.validate_transaction(tx)
        assert valid
        utxo_set.apply_transaction(tx)

        assert utxo_set.balance(bob.address_hex) == send_amount
        assert utxo_set.balance(alice.address_hex) == INITIAL_REWARD - send_amount


# =============================================================================
# Double-Spend Rejection
# =============================================================================

class TestDoubleSpend:
    def test_double_spend_rejected(self, funded_utxo, alice, bob, carol):
        """Cannot spend the same UTXO twice."""
        utxo_set, _ = funded_utxo

        tx1 = build_transaction(alice, bob, 10 * COIN, utxo_set)
        assert tx1 is not None
        valid, _ = utxo_set.validate_transaction(tx1)
        assert valid
        utxo_set.apply_transaction(tx1)

        # Manually craft a double-spend referencing the spent UTXO
        double_spend = BAB64CashTransaction(
            inputs=[tx1.inputs[0]],  # same input as tx1 (already spent)
            outputs=[TxOutput(carol.address_hex, 10 * COIN, "", 0)],
        )
        double_spend.finalize()
        valid, err = utxo_set.validate_transaction(double_spend)
        assert not valid
        assert "non-existent UTXO" in err

    def test_duplicate_input_in_single_tx(self, funded_utxo, alice, bob):
        """Cannot reference the same UTXO twice in one transaction."""
        utxo_set, coinbase = funded_utxo
        utxos = utxo_set.get_utxos_for(alice.address_hex)
        utxo = utxos[0]

        tx = BAB64CashTransaction(
            inputs=[
                TxInput(utxo.tx_hash, utxo.index),
                TxInput(utxo.tx_hash, utxo.index),
            ],
            outputs=[TxOutput(bob.address_hex, 10 * COIN, "", 0)],
        )
        tx.finalize()
        valid, err = utxo_set.validate_transaction(tx)
        assert not valid
        assert "Duplicate input" in err


# =============================================================================
# Insufficient Funds
# =============================================================================

class TestInsufficientFunds:
    def test_insufficient_funds(self, funded_utxo, alice, bob):
        """Cannot spend more than you have."""
        utxo_set, _ = funded_utxo
        tx = build_transaction(alice, bob, INITIAL_REWARD + 1, utxo_set)
        assert tx is None

    def test_insufficient_via_validation(self, funded_utxo, alice, bob):
        """Validation catches output > input."""
        utxo_set, coinbase = funded_utxo
        utxos = utxo_set.get_utxos_for(alice.address_hex)
        utxo = utxos[0]

        tx = BAB64CashTransaction(
            inputs=[TxInput(utxo.tx_hash, utxo.index)],
            outputs=[TxOutput(bob.address_hex, INITIAL_REWARD + 1, "", 0)],
        )
        tx.sign_input(0, alice, lock_nonce=utxo.lock_nonce)
        tx.finalize()
        valid, err = utxo_set.validate_transaction(tx)
        assert not valid
        assert "Insufficient funds" in err


# =============================================================================
# Invalid Signatures / Wrong Owner
# =============================================================================

class TestSignatureValidation:
    def test_wrong_signer_rejected(self, funded_utxo, alice, bob, carol):
        """Bob cannot spend Alice's UTXO — hashlock rejects."""
        utxo_set, _ = funded_utxo
        utxos = utxo_set.get_utxos_for(alice.address_hex)
        utxo = utxos[0]

        tx = BAB64CashTransaction(
            inputs=[TxInput(utxo.tx_hash, utxo.index)],
            outputs=[TxOutput(carol.address_hex, 10 * COIN, "", 0)],
        )
        # Bob signs with his own key and computes owner_proof from his image
        tx.sign_input(0, bob, lock_nonce=utxo.lock_nonce)
        tx.finalize()

        valid, err = utxo_set.validate_transaction(tx)
        assert not valid
        assert "owner" in err.lower()

    def test_missing_signature_rejected(self, funded_utxo, alice, bob):
        """Transaction without signature is rejected."""
        utxo_set, _ = funded_utxo
        utxos = utxo_set.get_utxos_for(alice.address_hex)
        utxo = utxos[0]

        tx = BAB64CashTransaction(
            inputs=[TxInput(utxo.tx_hash, utxo.index)],
            outputs=[TxOutput(bob.address_hex, 10 * COIN, "", 0)],
        )
        tx.finalize()
        valid, err = utxo_set.validate_transaction(tx)
        assert not valid
        assert "owner proof" in err.lower() or "signature" in err.lower()

    def test_missing_owner_proof_rejected(self, funded_utxo, alice, bob):
        """Valid signature but missing owner proof is rejected."""
        utxo_set, _ = funded_utxo
        utxos = utxo_set.get_utxos_for(alice.address_hex)
        utxo = utxos[0]

        tx = BAB64CashTransaction(
            inputs=[TxInput(utxo.tx_hash, utxo.index)],
            outputs=[TxOutput(bob.address_hex, 10 * COIN, "", 0)],
        )
        # Sign but don't provide lock_nonce (no owner_proof)
        tx.sign_input(0, alice)
        tx.finalize()
        valid, err = utxo_set.validate_transaction(tx)
        assert not valid
        assert "owner proof" in err.lower()


# =============================================================================
# Negative / Zero Amounts
# =============================================================================

class TestAmountValidation:
    def test_negative_output_rejected(self, funded_utxo, alice, bob):
        """Negative output amount is rejected."""
        utxo_set, _ = funded_utxo
        utxos = utxo_set.get_utxos_for(alice.address_hex)
        utxo = utxos[0]

        tx = BAB64CashTransaction(
            inputs=[TxInput(utxo.tx_hash, utxo.index)],
            outputs=[TxOutput(bob.address_hex, -100, "", 0)],
        )
        tx.sign_input(0, alice, lock_nonce=utxo.lock_nonce)
        tx.finalize()
        valid, err = utxo_set.validate_transaction(tx)
        assert not valid
        assert "positive" in err

    def test_zero_output_rejected(self, funded_utxo, alice, bob):
        """Zero output amount is rejected."""
        utxo_set, _ = funded_utxo
        utxos = utxo_set.get_utxos_for(alice.address_hex)
        utxo = utxos[0]

        tx = BAB64CashTransaction(
            inputs=[TxInput(utxo.tx_hash, utxo.index)],
            outputs=[TxOutput(bob.address_hex, 0, "", 0)],
        )
        tx.sign_input(0, alice, lock_nonce=utxo.lock_nonce)
        tx.finalize()
        valid, err = utxo_set.validate_transaction(tx)
        assert not valid
        assert "positive" in err


# =============================================================================
# Hashlock Mechanism
# =============================================================================

class TestHashlock:
    def test_lock_unlock_roundtrip(self, alice):
        """Owner can unlock their own lock."""
        lock_hash, lock_nonce = compute_lock(alice._image_bytes)
        proof = compute_unlock(alice._image_bytes, lock_nonce)
        assert verify_lock(proof, lock_hash)

    def test_wrong_image_fails(self, alice, bob):
        """Different image cannot unlock."""
        lock_hash, lock_nonce = compute_lock(alice._image_bytes)
        proof = compute_unlock(bob._image_bytes, lock_nonce)
        assert not verify_lock(proof, lock_hash)

    def test_each_lock_unique(self, alice):
        """Each lock has a unique nonce and hash."""
        lock1 = compute_lock(alice._image_bytes)
        lock2 = compute_lock(alice._image_bytes)
        assert lock1[0] != lock2[0]  # different lock_hash
        assert lock1[1] != lock2[1]  # different nonce


# =============================================================================
# Merkle Root
# =============================================================================

class TestMerkleRoot:
    def test_single_tx(self):
        """Merkle root of one hash is that hash."""
        h = hashlib.sha256(b'tx1').hexdigest()
        assert merkle_root([h]) == h

    def test_two_txs(self):
        """Merkle root of two hashes is their combined hash."""
        h1 = hashlib.sha256(b'tx1').hexdigest()
        h2 = hashlib.sha256(b'tx2').hexdigest()
        expected = hashlib.sha256(
            bytes.fromhex(h1) + bytes.fromhex(h2)
        ).hexdigest()
        assert merkle_root([h1, h2]) == expected

    def test_odd_count_duplicates_last(self):
        """Odd number of hashes duplicates the last one."""
        h1 = hashlib.sha256(b'tx1').hexdigest()
        h2 = hashlib.sha256(b'tx2').hexdigest()
        h3 = hashlib.sha256(b'tx3').hexdigest()
        l1_left = hashlib.sha256(bytes.fromhex(h1) + bytes.fromhex(h2)).digest()
        l1_right = hashlib.sha256(bytes.fromhex(h3) + bytes.fromhex(h3)).digest()
        expected = hashlib.sha256(l1_left + l1_right).hexdigest()
        assert merkle_root([h1, h2, h3]) == expected

    def test_empty_list(self):
        """Empty tx list produces a deterministic root."""
        mr = merkle_root([])
        assert len(mr) == 64

    def test_deterministic(self):
        """Same inputs always produce the same root."""
        hashes = [hashlib.sha256(f'tx{i}'.encode()).hexdigest() for i in range(5)]
        assert merkle_root(hashes) == merkle_root(hashes)


# =============================================================================
# Block Mining and Validation
# =============================================================================

class TestBlocks:
    def test_mine_block(self, alice):
        """A block can be mined with low difficulty."""
        coinbase = BAB64CashTransaction.create_coinbase(alice.address_hex, 0)
        block = BAB64BlockMiner.mine_block(
            index=0, previous_hash="0" * 64,
            transactions=[coinbase], difficulty=1,
        )
        assert block is not None
        assert BAB64BlockMiner.meets_difficulty(block.block_hash, 1)

    def test_block_hash_deterministic(self):
        """Same inputs produce the same block hash."""
        h1 = BAB64BlockMiner.compute_block_hash(0, "0" * 64, 1.0, "abc", 42, 4)
        h2 = BAB64BlockMiner.compute_block_hash(0, "0" * 64, 1.0, "abc", 42, 4)
        assert h1 == h2

    def test_empty_block_valid(self, alice):
        """Block with only coinbase is valid."""
        bc = BAB64Blockchain(difficulty=1, miner=alice)
        genesis = bc.genesis_block()
        assert genesis is not None
        assert len(genesis.transactions) == 1
        assert genesis.transactions[0].is_coinbase
        valid, err = bc.validate_chain()
        assert valid, err

    def test_block_validation(self, alice):
        """validate_block checks hash, difficulty, merkle root."""
        bc = BAB64Blockchain(difficulty=1, miner=alice)
        genesis = bc.genesis_block()
        valid, err = bc.validate_block(genesis, "0" * 64)
        assert valid, err

    def test_tampered_block_rejected(self, alice):
        """Modifying a block's hash is caught."""
        bc = BAB64Blockchain(difficulty=1, miner=alice)
        genesis = bc.genesis_block()
        genesis.block_hash = "f" * 64
        valid, err = bc.validate_block(genesis, "0" * 64)
        assert not valid


# =============================================================================
# Full Blockchain
# =============================================================================

class TestBlockchain:
    def test_genesis_creates_utxo(self, alice):
        """Genesis block creates a UTXO for the miner."""
        bc = BAB64Blockchain(difficulty=1, miner=alice)
        bc.genesis_block()
        assert bc.get_balance(alice.address_hex) == INITIAL_REWARD

    def test_mine_empty_block(self, alice):
        """Mining a block with empty mempool succeeds (coinbase only)."""
        bc = BAB64Blockchain(difficulty=1, miner=alice)
        bc.genesis_block()
        block = bc.mine_block()
        assert block is not None
        assert len(block.transactions) == 1

    def test_chain_validation(self, alice):
        """Multi-block chain validates correctly."""
        bc = BAB64Blockchain(difficulty=1, miner=alice)
        bc.genesis_block()
        bc.mine_block()
        bc.mine_block()
        valid, err = bc.validate_chain()
        assert valid, err

    def test_chain_of_5_blocks_with_3_identities(self, alice, bob, carol):
        """
        Full integration: 5 blocks, 3 identities transacting.

        Block 0: Genesis (Alice mines, gets 50 BAB64)
        Block 1: Alice -> Bob 10 BAB64 (Alice mines, gets reward)
        Block 2: Alice -> Carol 5 BAB64 (Alice mines, gets reward)
        Block 3: Bob -> Carol 3 BAB64 (Alice mines, gets reward)
        Block 4: Carol -> Alice 2 BAB64 (Alice mines, gets reward)
        """
        bc = BAB64Blockchain(difficulty=1, miner=alice)

        # Block 0: Genesis
        bc.genesis_block()
        assert bc.get_balance(alice.address_hex) == INITIAL_REWARD

        # Block 1: Alice -> Bob 10 BAB64
        tx1 = build_transaction(alice, bob, 10 * COIN, bc.utxo_set)
        assert tx1 is not None
        ok, err = bc.add_transaction_to_mempool(tx1)
        assert ok, err
        bc.mine_block()

        alice_bal = bc.get_balance(alice.address_hex)
        bob_bal = bc.get_balance(bob.address_hex)
        assert alice_bal == 2 * INITIAL_REWARD - 10 * COIN
        assert bob_bal == 10 * COIN

        # Block 2: Alice -> Carol 5 BAB64
        tx2 = build_transaction(alice, carol, 5 * COIN, bc.utxo_set)
        assert tx2 is not None
        ok, _ = bc.add_transaction_to_mempool(tx2)
        assert ok
        bc.mine_block()

        assert bc.get_balance(carol.address_hex) == 5 * COIN

        # Block 3: Bob -> Carol 3 BAB64
        tx3 = build_transaction(bob, carol, 3 * COIN, bc.utxo_set)
        assert tx3 is not None
        ok, _ = bc.add_transaction_to_mempool(tx3)
        assert ok
        bc.mine_block()

        assert bc.get_balance(bob.address_hex) == 7 * COIN
        assert bc.get_balance(carol.address_hex) == 8 * COIN

        # Block 4: Carol -> Alice 2 BAB64
        tx4 = build_transaction(carol, alice, 2 * COIN, bc.utxo_set)
        assert tx4 is not None
        ok, _ = bc.add_transaction_to_mempool(tx4)
        assert ok
        bc.mine_block()

        assert bc.get_balance(carol.address_hex) == 6 * COIN
        assert len(bc.chain) == 5

        # Full chain validation
        valid, err = bc.validate_chain()
        assert valid, err

    def test_balance_tracking(self, alice, bob):
        """Balance updates correctly across multiple transactions."""
        bc = BAB64Blockchain(difficulty=1, miner=alice)
        bc.genesis_block()

        for i in range(3):
            tx = build_transaction(alice, bob, 1 * COIN, bc.utxo_set)
            assert tx is not None
            bc.add_transaction_to_mempool(tx)
            bc.mine_block()

        assert bc.get_balance(bob.address_hex) == 3 * COIN

    def test_mempool_cleared_after_mining(self, alice, bob):
        """Mempool is empty after a block is mined."""
        bc = BAB64Blockchain(difficulty=1, miner=alice)
        bc.genesis_block()

        tx = build_transaction(alice, bob, 1 * COIN, bc.utxo_set)
        bc.add_transaction_to_mempool(tx)
        assert len(bc.mempool) == 1

        bc.mine_block()
        assert len(bc.mempool) == 0


# =============================================================================
# Transaction Fees
# =============================================================================

class TestFees:
    def test_fee_goes_to_miner(self, alice, bob):
        """Miner receives block reward + transaction fees."""
        bc = BAB64Blockchain(difficulty=1, miner=alice)
        bc.genesis_block()

        fee = 5000
        tx = build_transaction(alice, bob, 10 * COIN, bc.utxo_set, fee=fee)
        assert tx is not None
        bc.add_transaction_to_mempool(tx)
        bc.mine_block()

        block1 = bc.chain[1]
        coinbase = block1.transactions[0]
        assert coinbase.outputs[0].amount == INITIAL_REWARD + fee


# =============================================================================
# Edge Cases
# =============================================================================

class TestEdgeCases:
    def test_no_inputs_rejected(self, utxo_set, bob):
        """Non-coinbase tx with no inputs is rejected."""
        tx = BAB64CashTransaction(
            inputs=[],
            outputs=[TxOutput(bob.address_hex, 100, "", 0)],
        )
        tx.finalize()
        valid, err = utxo_set.validate_transaction(tx)
        assert not valid

    def test_no_outputs_rejected(self, funded_utxo, alice):
        """Transaction with no outputs is rejected."""
        utxo_set, _ = funded_utxo
        utxos = utxo_set.get_utxos_for(alice.address_hex)
        utxo = utxos[0]

        tx = BAB64CashTransaction(
            inputs=[TxInput(utxo.tx_hash, utxo.index)],
            outputs=[],
        )
        tx.finalize()
        valid, err = utxo_set.validate_transaction(tx)
        assert not valid

    def test_coinbase_with_inputs_rejected(self, utxo_set, alice):
        """Coinbase transaction must not have inputs."""
        tx = BAB64CashTransaction(
            inputs=[TxInput("fake", 0)],
            outputs=[TxOutput(alice.address_hex, INITIAL_REWARD, "", 0)],
            is_coinbase=True,
        )
        tx.finalize()
        valid, err = utxo_set.validate_transaction(tx)
        assert not valid
        assert "no inputs" in err

    def test_multiple_outputs(self, funded_utxo, alice, bob, carol):
        """Transaction can have multiple outputs."""
        utxo_set, _ = funded_utxo
        utxos = utxo_set.get_utxos_for(alice.address_hex)
        utxo = utxos[0]

        tx = BAB64CashTransaction(
            inputs=[TxInput(utxo.tx_hash, utxo.index)],
            outputs=[
                TxOutput(bob.address_hex, 10 * COIN, "", 0),
                TxOutput(carol.address_hex, 10 * COIN, "", 1),
                TxOutput(alice.address_hex, INITIAL_REWARD - 20 * COIN, "", 2),
            ],
        )
        tx.sign_input(0, alice, lock_nonce=utxo.lock_nonce)
        tx.finalize()
        valid, err = utxo_set.validate_transaction(tx)
        assert valid, err


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
