"""
Tests for BAB64 Consensus — Phase 2.

25+ tests covering:
  - Difficulty adjustment (up, down, clamped, minimum)
  - Chain selection (cumulative work, invalid chain rejected)
  - Fork handling (switch chain, UTXO revert)
  - Timestamp validation (no backdating, no future blocks)
  - Full block validation (all 10 rules)
  - Genesis block (valid, correct reward, message)
  - Duplicate transactions rejected
  - Coinbase reward mismatch rejected
  - Block at wrong height rejected
  - Header-only verification
  - 10-block chain with difficulty adjustment
  - BlockHeader extraction
"""

import hashlib
import time
import pytest
from bab64_engine import BAB64Config
from bab64_identity import BAB64Identity
from bab64_cash import (
    COIN, INITIAL_REWARD, ADJUSTMENT_INTERVAL, TARGET_BLOCK_TIME,
    MAX_FUTURE_BLOCK_TIME, GENESIS_TIMESTAMP, GENESIS_ADDRESS,
    GENESIS_MESSAGE,
    TxOutput, TxInput,
    BAB64CashTransaction, UTXOSet,
    BAB64Block, BAB64BlockMiner, BlockHeader,
    BAB64Blockchain, ChainSelector,
    block_reward, merkle_root, build_transaction,
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


def make_chain(miner, n_blocks, difficulty=1, start_time=1_000_000.0,
               block_interval=60.0):
    """Helper: build a chain of n_blocks with controlled timestamps."""
    bc = BAB64Blockchain(difficulty=difficulty, miner=miner)
    bc.genesis_block()
    # Override genesis timestamp
    bc.chain[0] = BAB64Block(
        index=bc.chain[0].index,
        previous_hash=bc.chain[0].previous_hash,
        timestamp=start_time,
        transactions=bc.chain[0].transactions,
        merkle_root_hash=bc.chain[0].merkle_root_hash,
        nonce=bc.chain[0].nonce,
        difficulty=bc.chain[0].difficulty,
        block_hash=bc.chain[0].block_hash,
    )
    # Re-mine genesis with the fixed timestamp
    coinbase = bc.chain[0].transactions[0]
    tx_hashes = [coinbase.tx_hash]
    mr = merkle_root(tx_hashes)
    for nonce in range(10_000_000):
        bh = BAB64BlockMiner.compute_block_hash(
            0, "0" * 64, start_time, mr, nonce, difficulty
        )
        if BAB64BlockMiner.meets_difficulty(bh, difficulty):
            bc.chain[0] = BAB64Block(
                index=0, previous_hash="0" * 64, timestamp=start_time,
                transactions=[coinbase], merkle_root_hash=mr,
                nonce=nonce, difficulty=difficulty, block_hash=bh,
            )
            break

    for i in range(1, n_blocks):
        ts = start_time + i * block_interval
        block = bc.mine_block(timestamp=ts)
        assert block is not None
    return bc


# =============================================================================
# DIFFICULTY ADJUSTMENT
# =============================================================================

class TestDifficultyAdjustment:
    def test_difficulty_increases_when_blocks_too_fast(self, alice):
        """If blocks come faster than target, difficulty goes up."""
        bc = make_chain(alice, ADJUSTMENT_INTERVAL,
                        difficulty=2, block_interval=10.0)
        # Blocks every 10s vs 60s target -> too fast -> increase
        old_diff = bc.difficulty
        new_diff = bc.difficulty_adjustment()
        assert new_diff > old_diff

    def test_difficulty_decreases_when_blocks_too_slow(self, alice):
        """If blocks come slower than target, difficulty goes down."""
        bc = make_chain(alice, ADJUSTMENT_INTERVAL,
                        difficulty=4, block_interval=300.0)
        # Blocks every 300s vs 60s target -> too slow -> decrease
        old_diff = bc.difficulty
        new_diff = bc.difficulty_adjustment()
        assert new_diff < old_diff

    def test_difficulty_clamped_at_4x_increase(self, alice):
        """Difficulty cannot increase more than 4x in one adjustment."""
        bc = make_chain(alice, ADJUSTMENT_INTERVAL,
                        difficulty=2, block_interval=0.001)
        old_diff = bc.difficulty
        new_diff = bc.difficulty_adjustment()
        assert new_diff <= old_diff * 4

    def test_difficulty_clamped_at_quarter_decrease(self, alice):
        """Difficulty cannot decrease below old/4."""
        bc = make_chain(alice, ADJUSTMENT_INTERVAL,
                        difficulty=8, block_interval=10000.0)
        old_diff = bc.difficulty
        new_diff = bc.difficulty_adjustment()
        assert new_diff >= max(1, old_diff // 4)

    def test_difficulty_minimum_is_1(self, alice):
        """Difficulty never goes below 1."""
        bc = make_chain(alice, ADJUSTMENT_INTERVAL,
                        difficulty=1, block_interval=10000.0)
        new_diff = bc.difficulty_adjustment()
        assert new_diff >= 1

    def test_no_adjustment_before_interval(self, alice):
        """No adjustment if chain is shorter than ADJUSTMENT_INTERVAL."""
        bc = make_chain(alice, ADJUSTMENT_INTERVAL - 1,
                        difficulty=4, block_interval=10.0)
        old_diff = bc.difficulty
        new_diff = bc.difficulty_adjustment()
        assert new_diff == old_diff

    def test_adjustment_only_at_interval_boundary(self, alice):
        """Adjustment only triggers when len(chain) % interval == 0."""
        bc = make_chain(alice, ADJUSTMENT_INTERVAL + 1,
                        difficulty=4, block_interval=10.0)
        # 11 blocks -> 11 % 10 != 0 -> no adjustment
        old_diff = bc.difficulty
        new_diff = bc.difficulty_adjustment()
        assert new_diff == old_diff


# =============================================================================
# CHAIN SELECTION
# =============================================================================

class TestChainSelection:
    def test_cumulative_work(self, alice):
        """Cumulative work sums 2^difficulty for each block."""
        bc = make_chain(alice, 3, difficulty=1)
        work = ChainSelector.cumulative_work(bc.chain)
        assert work == 3 * (2 ** 1)

    def test_longer_valid_chain_wins(self, alice, bob):
        """Chain with more cumulative work is selected."""
        chain_a = make_chain(alice, 3, difficulty=1)
        chain_b = make_chain(bob, 5, difficulty=1)

        selected = ChainSelector.select_chain(
            chain_a.chain, chain_b.chain, chain_a.utxo_set
        )
        assert len(selected) == 5

    def test_higher_difficulty_chain_wins(self, alice, bob):
        """Shorter chain with higher difficulty can win."""
        chain_a = make_chain(alice, 5, difficulty=1)  # work = 5 * 2 = 10
        chain_b = make_chain(bob, 3, difficulty=4)    # work = 3 * 16 = 48

        selected = ChainSelector.select_chain(
            chain_a.chain, chain_b.chain, chain_a.utxo_set
        )
        assert len(selected) == 3  # chain_b wins despite fewer blocks

    def test_invalid_chain_rejected(self, alice, bob):
        """Invalid competing chain is not selected even with more work."""
        chain_a = make_chain(alice, 3, difficulty=1)
        chain_b = make_chain(bob, 5, difficulty=1)
        # Corrupt chain_b
        chain_b.chain[2].block_hash = "f" * 64
        selected = ChainSelector.select_chain(
            chain_a.chain, chain_b.chain, chain_a.utxo_set
        )
        assert len(selected) == 3  # chain_a preserved

    def test_equal_work_keeps_current(self, alice, bob):
        """Equal cumulative work keeps chain_a (current)."""
        chain_a = make_chain(alice, 3, difficulty=1)
        chain_b = make_chain(bob, 3, difficulty=1)
        selected = ChainSelector.select_chain(
            chain_a.chain, chain_b.chain, chain_a.utxo_set
        )
        assert selected is chain_a.chain


# =============================================================================
# FORK HANDLING
# =============================================================================

class TestForkHandling:
    def test_switch_to_better_chain(self, alice, bob):
        """handle_fork switches to a valid chain with more work."""
        bc = make_chain(alice, 3, difficulty=1)
        competing = make_chain(bob, 5, difficulty=1)

        old_tip = bc.chain[-1].block_hash
        switched = bc.handle_fork(competing.chain)
        assert switched
        assert len(bc.chain) == 5
        assert bc.chain[-1].block_hash != old_tip

    def test_utxo_reverted_on_fork(self, alice, bob):
        """UTXO set is rebuilt from the new chain after fork."""
        bc = make_chain(alice, 3, difficulty=1)
        alice_balance_before = bc.get_balance(alice.address_hex)

        competing = make_chain(bob, 5, difficulty=1)
        bc.handle_fork(competing.chain)

        # After fork, Alice's UTXOs from old chain are gone
        assert bc.get_balance(alice.address_hex) == 0
        # Bob has UTXOs from new chain
        assert bc.get_balance(bob.address_hex) > 0

    def test_no_switch_to_weaker_chain(self, alice, bob):
        """handle_fork does not switch to a chain with less work."""
        bc = make_chain(alice, 5, difficulty=1)
        competing = make_chain(bob, 3, difficulty=1)

        old_tip = bc.chain[-1].block_hash
        switched = bc.handle_fork(competing.chain)
        assert not switched
        assert bc.chain[-1].block_hash == old_tip


# =============================================================================
# TIMESTAMP VALIDATION
# =============================================================================

class TestTimestampValidation:
    def test_no_backdating(self, alice):
        """Block timestamp must be after median of last 11 blocks."""
        bc = make_chain(alice, 5, difficulty=1, start_time=1_000_000.0,
                        block_interval=60.0)
        prev_hash = bc.chain[-1].block_hash
        backdated_ts = 1_000_000.0  # way before median

        # Create a properly mined block with backdated timestamp
        coinbase = BAB64CashTransaction.create_coinbase(alice.address_hex, 5)
        tx_hashes = [coinbase.tx_hash]
        mr = merkle_root(tx_hashes)

        for nonce in range(10_000_000):
            bh = BAB64BlockMiner.compute_block_hash(
                5, prev_hash, backdated_ts, mr, nonce, 1
            )
            if BAB64BlockMiner.meets_difficulty(bh, 1):
                backdated_block = BAB64Block(
                    index=5, previous_hash=prev_hash,
                    timestamp=backdated_ts,
                    transactions=[coinbase],
                    merkle_root_hash=mr,
                    nonce=nonce, difficulty=1, block_hash=bh,
                )
                break

        valid, err = bc.validate_block_full(
            backdated_block, prev_hash, 5,
            preceding_chain=bc.chain,
            current_time=2_000_000.0,
        )
        assert not valid
        assert "median" in err.lower() or "timestamp" in err.lower()

    def test_no_future_blocks(self, alice):
        """Block timestamp must not be more than 2 hours in the future."""
        bc = make_chain(alice, 3, difficulty=1, start_time=1_000_000.0)
        now = 1_000_200.0
        future_time = now + MAX_FUTURE_BLOCK_TIME + 100

        # Create a properly mined block with future timestamp
        coinbase = BAB64CashTransaction.create_coinbase(alice.address_hex, 3)
        tx_hashes = [coinbase.tx_hash]
        mr = merkle_root(tx_hashes)
        prev_hash = bc.chain[-1].block_hash

        for nonce in range(10_000_000):
            bh = BAB64BlockMiner.compute_block_hash(
                3, prev_hash, future_time, mr, nonce, 1
            )
            if BAB64BlockMiner.meets_difficulty(bh, 1):
                future_block = BAB64Block(
                    index=3, previous_hash=prev_hash,
                    timestamp=future_time,
                    transactions=[coinbase],
                    merkle_root_hash=mr,
                    nonce=nonce, difficulty=1, block_hash=bh,
                )
                break

        valid, err = bc.validate_block_full(
            future_block, prev_hash, 3,
            preceding_chain=bc.chain,
            current_time=now,
        )
        assert not valid
        assert "future" in err.lower()


# =============================================================================
# FULL BLOCK VALIDATION
# =============================================================================

class TestFullBlockValidation:
    def test_valid_block_passes_all_rules(self, alice):
        """A properly mined block passes full validation."""
        bc = make_chain(alice, 3, difficulty=1, start_time=1_000_000.0)
        block = bc.chain[-1]
        prev_hash = bc.chain[-2].block_hash
        valid, err = bc.validate_block_full(
            block, prev_hash, block.index,
            preceding_chain=bc.chain[:-1],
            current_time=2_000_000.0,
        )
        assert valid, err

    def test_wrong_height_rejected(self, alice):
        """Block at wrong height is rejected."""
        bc = make_chain(alice, 3, difficulty=1, start_time=1_000_000.0)
        block = bc.chain[-1]
        prev_hash = bc.chain[-2].block_hash
        # Claim it's at height 99
        valid, err = bc.validate_block_full(
            block, prev_hash, 99,
            preceding_chain=bc.chain[:-1],
            current_time=2_000_000.0,
        )
        assert not valid
        assert "height" in err.lower()

    def test_coinbase_reward_mismatch_rejected(self, alice):
        """Block with wrong coinbase reward is rejected."""
        bc = make_chain(alice, 2, difficulty=1, start_time=1_000_000.0)
        prev_hash = bc.chain[-1].block_hash
        height = 2

        # Create coinbase with wrong reward
        bad_coinbase = BAB64CashTransaction(is_coinbase=True, coinbase_height=height)
        bad_coinbase.outputs.append(TxOutput(
            recipient=alice.address_hex,
            amount=INITIAL_REWARD * 2,  # double reward!
            tx_hash="", index=0,
        ))
        bad_coinbase.finalize()

        tx_hashes = [bad_coinbase.tx_hash]
        mr = merkle_root(tx_hashes)

        for nonce in range(10_000_000):
            bh = BAB64BlockMiner.compute_block_hash(
                height, prev_hash, 1_000_200.0, mr, nonce, 1
            )
            if BAB64BlockMiner.meets_difficulty(bh, 1):
                bad_block = BAB64Block(
                    index=height, previous_hash=prev_hash,
                    timestamp=1_000_200.0,
                    transactions=[bad_coinbase],
                    merkle_root_hash=mr,
                    nonce=nonce, difficulty=1, block_hash=bh,
                )
                break

        valid, err = bc.validate_block_full(
            bad_block, prev_hash, height,
            preceding_chain=bc.chain,
            current_time=2_000_000.0,
        )
        assert not valid
        assert "reward" in err.lower() or "coinbase" in err.lower()

    def test_duplicate_transaction_rejected(self, alice):
        """Block with duplicate transactions is rejected."""
        bc = make_chain(alice, 2, difficulty=1, start_time=1_000_000.0)
        prev_hash = bc.chain[-1].block_hash
        height = 2

        coinbase = BAB64CashTransaction.create_coinbase(alice.address_hex, height)
        # Duplicate the coinbase
        transactions = [coinbase, coinbase]
        tx_hashes = [tx.tx_hash for tx in transactions]
        mr = merkle_root(tx_hashes)

        for nonce in range(10_000_000):
            bh = BAB64BlockMiner.compute_block_hash(
                height, prev_hash, 1_000_200.0, mr, nonce, 1
            )
            if BAB64BlockMiner.meets_difficulty(bh, 1):
                dup_block = BAB64Block(
                    index=height, previous_hash=prev_hash,
                    timestamp=1_000_200.0,
                    transactions=transactions,
                    merkle_root_hash=mr,
                    nonce=nonce, difficulty=1, block_hash=bh,
                )
                break

        valid, err = bc.validate_block_full(
            dup_block, prev_hash, height,
            preceding_chain=bc.chain,
            current_time=2_000_000.0,
        )
        assert not valid
        assert "duplicate" in err.lower()

    def test_merkle_root_mismatch_rejected(self, alice):
        """Block with wrong merkle root is rejected."""
        bc = make_chain(alice, 2, difficulty=1, start_time=1_000_000.0)
        prev_hash = bc.chain[-1].block_hash
        height = 2

        coinbase = BAB64CashTransaction.create_coinbase(alice.address_hex, height)
        wrong_mr = "a" * 64

        for nonce in range(10_000_000):
            bh = BAB64BlockMiner.compute_block_hash(
                height, prev_hash, 1_000_200.0, wrong_mr, nonce, 1
            )
            if BAB64BlockMiner.meets_difficulty(bh, 1):
                bad_block = BAB64Block(
                    index=height, previous_hash=prev_hash,
                    timestamp=1_000_200.0,
                    transactions=[coinbase],
                    merkle_root_hash=wrong_mr,
                    nonce=nonce, difficulty=1, block_hash=bh,
                )
                break

        valid, err = bc.validate_block_full(
            bad_block, prev_hash, height,
            preceding_chain=bc.chain,
            current_time=2_000_000.0,
        )
        assert not valid
        assert "merkle" in err.lower()

    def test_previous_hash_mismatch_rejected(self, alice):
        """Block with wrong previous hash is rejected."""
        bc = make_chain(alice, 2, difficulty=1, start_time=1_000_000.0)
        block = bc.chain[-1]
        valid, err = bc.validate_block_full(
            block, "b" * 64, block.index,
            preceding_chain=bc.chain[:-1],
            current_time=2_000_000.0,
        )
        assert not valid
        assert "previous hash" in err.lower()


# =============================================================================
# GENESIS BLOCK
# =============================================================================

class TestGenesisBlock:
    def test_genesis_is_valid(self):
        """Hardcoded genesis block passes structural validation."""
        genesis = BAB64Blockchain.create_genesis_block()
        assert genesis.index == 0
        assert genesis.previous_hash == "0" * 64
        assert genesis.timestamp == GENESIS_TIMESTAMP
        assert genesis.difficulty == 1
        assert BAB64BlockMiner.meets_difficulty(genesis.block_hash, 1)

    def test_genesis_correct_reward(self):
        """Genesis coinbase pays 50 BAB64 to genesis address."""
        genesis = BAB64Blockchain.create_genesis_block()
        coinbase = genesis.transactions[0]
        assert coinbase.is_coinbase
        assert coinbase.outputs[0].amount == INITIAL_REWARD
        assert coinbase.outputs[0].recipient == GENESIS_ADDRESS

    def test_genesis_message_embedded(self):
        """Genesis block has the BAB64 message embedded."""
        genesis = BAB64Blockchain.create_genesis_block()
        # The message is embedded in the coinbase tx_hash derivation
        h = hashlib.sha256()
        h.update(GENESIS_MESSAGE.encode())
        h.update(GENESIS_ADDRESS.encode())
        h.update(b'coinbase')
        h.update((0).to_bytes(4, 'big', signed=True))
        expected_hash = h.hexdigest()
        assert genesis.transactions[0].tx_hash == expected_hash

    def test_genesis_deterministic(self):
        """Genesis block is the same every time."""
        g1 = BAB64Blockchain.create_genesis_block()
        g2 = BAB64Blockchain.create_genesis_block()
        assert g1.block_hash == g2.block_hash
        assert g1.nonce == g2.nonce

    def test_add_genesis_to_blockchain(self, alice):
        """add_genesis() adds the hardcoded genesis and updates UTXO."""
        bc = BAB64Blockchain(difficulty=4, miner=alice)
        block = bc.add_genesis()
        assert len(bc.chain) == 1
        assert bc.get_balance(GENESIS_ADDRESS) == INITIAL_REWARD


# =============================================================================
# BLOCK HEADER
# =============================================================================

class TestBlockHeader:
    def test_header_extraction(self, alice):
        """Block header contains all metadata, no transactions."""
        bc = make_chain(alice, 2, difficulty=1)
        block = bc.chain[-1]
        header = block.header()

        assert header.index == block.index
        assert header.previous_hash == block.previous_hash
        assert header.timestamp == block.timestamp
        assert header.merkle_root == block.merkle_root_hash
        assert header.difficulty == block.difficulty
        assert header.nonce == block.nonce
        assert header.block_hash == block.block_hash

    def test_header_verification(self, alice):
        """Header-only verification passes for valid block."""
        bc = make_chain(alice, 2, difficulty=1)
        header = bc.chain[-1].header()
        valid, err = BAB64Blockchain.verify_header(header)
        assert valid, err

    def test_tampered_header_rejected(self, alice):
        """Tampered header fails verification."""
        bc = make_chain(alice, 2, difficulty=1)
        header = bc.chain[-1].header()
        header.block_hash = "f" * 64
        valid, err = BAB64Blockchain.verify_header(header)
        assert not valid


# =============================================================================
# INTEGRATION — 10-block chain with difficulty adjustment
# =============================================================================

class TestIntegration:
    def test_10_block_chain_with_adjustment(self, alice):
        """
        Mine 10 blocks (= ADJUSTMENT_INTERVAL), trigger difficulty
        adjustment, verify chain validity throughout.
        """
        bc = make_chain(alice, ADJUSTMENT_INTERVAL,
                        difficulty=2, block_interval=30.0)
        assert len(bc.chain) == ADJUSTMENT_INTERVAL

        # Chain is valid
        valid, err = bc.validate_chain()
        assert valid, err

        # Trigger adjustment (blocks were fast: 30s vs 60s target)
        old_diff = bc.difficulty
        new_diff = bc.difficulty_adjustment()
        assert new_diff > old_diff  # should increase
        assert new_diff >= 1

    def test_full_validation_on_every_block(self, alice):
        """validate_block_full passes on each block in a 5-block chain."""
        bc = make_chain(alice, 5, difficulty=1, start_time=1_000_000.0,
                        block_interval=60.0)

        for i, block in enumerate(bc.chain):
            expected_prev = "0" * 64 if i == 0 else bc.chain[i - 1].block_hash
            preceding = bc.chain[:i] if i > 0 else []
            valid, err = bc.validate_block_full(
                block, expected_prev, i,
                preceding_chain=preceding if preceding else None,
                current_time=2_000_000.0,
            )
            assert valid, f"Block {i}: {err}"

    def test_cumulative_work_increases_with_difficulty(self, alice):
        """Higher difficulty blocks contribute exponentially more work."""
        chain_low = make_chain(alice, 5, difficulty=1)
        chain_high = make_chain(alice, 5, difficulty=3)

        work_low = ChainSelector.cumulative_work(chain_low.chain)
        work_high = ChainSelector.cumulative_work(chain_high.chain)
        assert work_high > work_low
        assert work_high == 5 * (2 ** 3)
        assert work_low == 5 * (2 ** 1)

    def test_mine_with_custom_timestamp(self, alice):
        """mine_block accepts a custom timestamp."""
        bc = BAB64Blockchain(difficulty=1, miner=alice)
        bc.genesis_block()
        ts = 9_999_999.0
        block = bc.mine_block(timestamp=ts)
        assert block is not None
        assert block.timestamp == ts


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
