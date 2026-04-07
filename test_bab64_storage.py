"""
Tests for BAB64 Storage — Persistent SQLite Layer
===================================================

25+ tests covering BlockchainDB, UTXODB, WalletDB, PeerDB,
full roundtrips, persistence across close/reopen, and
integration with BAB64Blockchain.
"""

import os
import shutil
import tempfile
import unittest

from bab64_cash import (
    BAB64Block, BAB64Blockchain, BAB64CashTransaction,
    TxInput, TxOutput, UTXOSet, block_reward, INITIAL_REWARD,
)
from bab64_identity import BAB64Identity
from bab64_storage import (
    BAB64Storage, BlockchainDB, UTXODB, WalletDB, PeerDB,
)


class StorageTestBase(unittest.TestCase):
    """Shared setup: temp directory and identity creation."""

    def setUp(self):
        self.tmpdir = tempfile.mkdtemp(prefix="bab64_test_")

    def tearDown(self):
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    @staticmethod
    def _make_identity():
        return BAB64Identity(os.urandom(32))

    def _make_chain_with_blocks(self, n=3):
        """Create a blockchain with n mined blocks (difficulty=1 for speed)."""
        identity = self._make_identity()
        bc = BAB64Blockchain(difficulty=1, miner=identity)
        bc.genesis_block()
        for _ in range(n - 1):
            bc.mine_block()
        return bc, identity


# =============================================================================
# BLOCKCHAIN DB TESTS
# =============================================================================

class TestBlockchainDB(StorageTestBase):

    def test_save_and_load_single_block(self):
        bc, _ = self._make_chain_with_blocks(1)
        db = BlockchainDB(self.tmpdir)
        db.save_block(bc.chain[0])
        loaded = db.load_chain()
        self.assertEqual(len(loaded), 1)
        self.assertEqual(loaded[0].block_hash, bc.chain[0].block_hash)
        db.close()

    def test_load_full_chain_in_order(self):
        bc, _ = self._make_chain_with_blocks(4)
        db = BlockchainDB(self.tmpdir)
        # Save out of order
        for blk in reversed(bc.chain):
            db.save_block(blk)
        loaded = db.load_chain()
        self.assertEqual(len(loaded), 4)
        for i, blk in enumerate(loaded):
            self.assertEqual(blk.index, i)
            self.assertEqual(blk.block_hash, bc.chain[i].block_hash)
        db.close()

    def test_get_block_by_index(self):
        bc, _ = self._make_chain_with_blocks(3)
        db = BlockchainDB(self.tmpdir)
        for blk in bc.chain:
            db.save_block(blk)
        result = db.get_block(1)
        self.assertIsNotNone(result)
        self.assertEqual(result.block_hash, bc.chain[1].block_hash)
        db.close()

    def test_get_block_by_hash(self):
        bc, _ = self._make_chain_with_blocks(2)
        db = BlockchainDB(self.tmpdir)
        for blk in bc.chain:
            db.save_block(blk)
        target = bc.chain[1].block_hash
        result = db.get_block_by_hash(target)
        self.assertIsNotNone(result)
        self.assertEqual(result.index, 1)
        db.close()

    def test_chain_height(self):
        bc, _ = self._make_chain_with_blocks(5)
        db = BlockchainDB(self.tmpdir)
        self.assertEqual(db.chain_height(), -1)
        for blk in bc.chain:
            db.save_block(blk)
        self.assertEqual(db.chain_height(), 4)
        db.close()

    def test_has_block(self):
        bc, _ = self._make_chain_with_blocks(1)
        db = BlockchainDB(self.tmpdir)
        db.save_block(bc.chain[0])
        self.assertTrue(db.has_block(bc.chain[0].block_hash))
        self.assertFalse(db.has_block("0" * 64))
        db.close()

    def test_duplicate_block_rejected(self):
        bc, _ = self._make_chain_with_blocks(1)
        db = BlockchainDB(self.tmpdir)
        db.save_block(bc.chain[0])
        db.save_block(bc.chain[0])  # should not raise
        loaded = db.load_chain()
        self.assertEqual(len(loaded), 1)
        db.close()

    def test_get_nonexistent_block(self):
        db = BlockchainDB(self.tmpdir)
        self.assertIsNone(db.get_block(999))
        self.assertIsNone(db.get_block_by_hash("deadbeef"))
        db.close()

    def test_transactions_roundtrip(self):
        """Verify transaction data survives serialization."""
        bc, identity = self._make_chain_with_blocks(1)
        db = BlockchainDB(self.tmpdir)
        db.save_block(bc.chain[0])
        loaded = db.load_chain()
        orig_tx = bc.chain[0].transactions[0]
        loaded_tx = loaded[0].transactions[0]
        self.assertEqual(orig_tx.tx_hash, loaded_tx.tx_hash)
        self.assertEqual(orig_tx.is_coinbase, loaded_tx.is_coinbase)
        self.assertEqual(orig_tx.outputs[0].amount, loaded_tx.outputs[0].amount)
        db.close()


# =============================================================================
# UTXO DB TESTS
# =============================================================================

class TestUTXODB(StorageTestBase):

    def test_save_and_get_utxo(self):
        db = UTXODB(self.tmpdir)
        out = TxOutput("aabb", 5000, "tx1", 0)
        db.save_utxo(out)
        got = db.get_utxo("tx1", 0)
        self.assertIsNotNone(got)
        self.assertEqual(got.amount, 5000)
        self.assertEqual(got.recipient, "aabb")
        db.close()

    def test_spend_utxo(self):
        db = UTXODB(self.tmpdir)
        out = TxOutput("aabb", 5000, "tx1", 0)
        db.save_utxo(out)
        db.spend_utxo("tx1", 0)
        got = db.get_utxo("tx1", 0)
        self.assertIsNone(got)
        db.close()

    def test_balance(self):
        db = UTXODB(self.tmpdir)
        db.save_utxo(TxOutput("addr1", 1000, "tx1", 0))
        db.save_utxo(TxOutput("addr1", 2000, "tx2", 0))
        db.save_utxo(TxOutput("addr2", 3000, "tx3", 0))
        self.assertEqual(db.balance("addr1"), 3000)
        self.assertEqual(db.balance("addr2"), 3000)
        self.assertEqual(db.balance("unknown"), 0)
        db.close()

    def test_get_utxos_for(self):
        db = UTXODB(self.tmpdir)
        db.save_utxo(TxOutput("addr1", 1000, "tx1", 0))
        db.save_utxo(TxOutput("addr1", 2000, "tx2", 0))
        db.save_utxo(TxOutput("addr2", 3000, "tx3", 0))
        utxos = db.get_utxos_for("addr1")
        self.assertEqual(len(utxos), 2)
        amounts = {u.amount for u in utxos}
        self.assertEqual(amounts, {1000, 2000})
        db.close()

    def test_spent_utxo_not_in_balance(self):
        db = UTXODB(self.tmpdir)
        db.save_utxo(TxOutput("addr1", 1000, "tx1", 0))
        db.save_utxo(TxOutput("addr1", 2000, "tx2", 0))
        db.spend_utxo("tx1", 0)
        self.assertEqual(db.balance("addr1"), 2000)
        utxos = db.get_utxos_for("addr1")
        self.assertEqual(len(utxos), 1)
        db.close()

    def test_rebuild_from_chain(self):
        bc, identity = self._make_chain_with_blocks(3)
        db = UTXODB(self.tmpdir)
        db.rebuild_from_chain(bc.chain)
        addr = identity.address_hex
        expected = bc.utxo_set.balance(addr)
        self.assertEqual(db.balance(addr), expected)
        db.close()


# =============================================================================
# WALLET DB TESTS
# =============================================================================

class TestWalletDB(StorageTestBase):

    def test_save_and_load_identity(self):
        db = WalletDB(self.tmpdir)
        identity = self._make_identity()
        db.save_identity(identity, "secret123")
        loaded = db.load_identity(identity.address_hex, "secret123")
        self.assertIsNotNone(loaded)
        self.assertEqual(loaded.address_hex, identity.address_hex)
        db.close()

    def test_wrong_passphrase_fails(self):
        db = WalletDB(self.tmpdir)
        identity = self._make_identity()
        db.save_identity(identity, "correct")
        loaded = db.load_identity(identity.address_hex, "wrong")
        self.assertIsNone(loaded)
        db.close()

    def test_list_addresses(self):
        db = WalletDB(self.tmpdir)
        ids = [self._make_identity() for _ in range(3)]
        for i, ident in enumerate(ids):
            db.save_identity(ident, f"pass{i}")
        addrs = db.list_addresses()
        self.assertEqual(len(addrs), 3)
        for ident in ids:
            self.assertIn(ident.address_hex, addrs)
        db.close()

    def test_delete_identity(self):
        db = WalletDB(self.tmpdir)
        identity = self._make_identity()
        db.save_identity(identity, "pass")
        db.delete_identity(identity.address_hex)
        self.assertEqual(len(db.list_addresses()), 0)
        db.close()

    def test_load_nonexistent_address(self):
        db = WalletDB(self.tmpdir)
        self.assertIsNone(db.load_identity("no_such_addr", "pass"))
        db.close()


# =============================================================================
# PEER DB TESTS
# =============================================================================

class TestPeerDB(StorageTestBase):

    def test_save_and_get_peers(self):
        db = PeerDB(self.tmpdir)
        db.save_peer("10.0.0.1", 8333)
        db.save_peer("10.0.0.2", 8334)
        peers = db.get_peers()
        self.assertEqual(len(peers), 2)
        self.assertIn(("10.0.0.1", 8333), peers)
        db.close()

    def test_ban_peer(self):
        db = PeerDB(self.tmpdir)
        db.save_peer("10.0.0.1", 8333)
        db.save_peer("10.0.0.2", 8334)
        db.ban_peer("10.0.0.1", 8333)
        unbanned = db.get_unbanned_peers()
        self.assertEqual(len(unbanned), 1)
        self.assertEqual(unbanned[0], ("10.0.0.2", 8334))
        db.close()

    def test_update_last_seen(self):
        db = PeerDB(self.tmpdir)
        db.save_peer("10.0.0.1", 8333)
        db.update_last_seen("10.0.0.1", 8333)
        # Should not raise, peer still present
        peers = db.get_peers()
        self.assertEqual(len(peers), 1)
        db.close()

    def test_duplicate_save_increments(self):
        db = PeerDB(self.tmpdir)
        db.save_peer("10.0.0.1", 8333)
        db.save_peer("10.0.0.1", 8333)
        peers = db.get_peers()
        self.assertEqual(len(peers), 1)
        db.close()


# =============================================================================
# FULL ROUNDTRIP & PERSISTENCE TESTS
# =============================================================================

class TestFullRoundtrip(StorageTestBase):

    def test_save_chain_reload_verify_balances(self):
        """Create chain -> save to storage -> reload -> balances match."""
        bc, identity = self._make_chain_with_blocks(3)
        storage = BAB64Storage(self.tmpdir)
        storage.save_state(bc.chain)

        chain, utxo_set = storage.load_state()
        self.assertEqual(len(chain), 3)
        addr = identity.address_hex
        self.assertEqual(utxo_set.balance(addr), bc.utxo_set.balance(addr))
        storage.close()

    def test_persistence_across_close_reopen(self):
        """Write DB, close all connections, reopen, data survives."""
        bc, identity = self._make_chain_with_blocks(2)

        # Write
        storage = BAB64Storage(self.tmpdir)
        storage.save_state(bc.chain)
        storage.wallet.save_identity(identity, "mypass")
        storage.peers.save_peer("peer1.example.com", 9999)
        storage.close()

        # Reopen
        storage2 = BAB64Storage(self.tmpdir)
        chain, utxo_set = storage2.load_state()
        self.assertEqual(len(chain), 2)
        self.assertEqual(chain[0].block_hash, bc.chain[0].block_hash)
        self.assertEqual(chain[1].block_hash, bc.chain[1].block_hash)

        loaded_id = storage2.wallet.load_identity(identity.address_hex, "mypass")
        self.assertIsNotNone(loaded_id)
        self.assertEqual(loaded_id.address_hex, identity.address_hex)

        peers = storage2.peers.get_peers()
        self.assertIn(("peer1.example.com", 9999), peers)
        storage2.close()

    def test_utxo_balance_matches_chain(self):
        """UTXO rebuild from persisted chain yields correct balances."""
        bc, identity = self._make_chain_with_blocks(5)
        storage = BAB64Storage(self.tmpdir)
        storage.save_state(bc.chain)
        addr = identity.address_hex
        db_balance = storage.utxos.balance(addr)
        mem_balance = bc.utxo_set.balance(addr)
        self.assertEqual(db_balance, mem_balance)
        storage.close()


# =============================================================================
# INTEGRATION WITH BAB64Blockchain
# =============================================================================

class TestBlockchainIntegration(StorageTestBase):

    def test_blockchain_with_storage_init(self):
        """BAB64Blockchain loads chain from storage on init."""
        identity = self._make_identity()
        bc = BAB64Blockchain(difficulty=1, miner=identity)
        bc.genesis_block()
        bc.mine_block()

        # Save to storage
        storage = BAB64Storage(self.tmpdir)
        storage.save_state(bc.chain)
        storage.close()

        # Create new blockchain backed by storage
        storage2 = BAB64Storage(self.tmpdir)
        bc2 = BAB64Blockchain(difficulty=1, miner=identity, storage=storage2)
        self.assertEqual(len(bc2.chain), 2)
        self.assertEqual(bc2.chain[0].block_hash, bc.chain[0].block_hash)
        self.assertEqual(bc2.chain[1].block_hash, bc.chain[1].block_hash)
        addr = identity.address_hex
        self.assertEqual(bc2.utxo_set.balance(addr), bc.utxo_set.balance(addr))
        storage2.close()

    def test_mine_block_persists_to_storage(self):
        """Mining a block auto-saves it when storage is attached."""
        identity = self._make_identity()
        storage = BAB64Storage(self.tmpdir)
        bc = BAB64Blockchain(difficulty=1, miner=identity, storage=storage)
        bc.genesis_block()
        bc.mine_block()
        # genesis_block doesn't go through mine_block path, save it manually
        storage.blockchain.save_block(bc.chain[0])

        # Verify the mined block was persisted
        self.assertEqual(storage.blockchain.chain_height(), 1)
        loaded = storage.blockchain.get_block(1)
        self.assertIsNotNone(loaded)
        self.assertEqual(loaded.block_hash, bc.chain[1].block_hash)
        storage.close()

    def test_storage_no_effect_when_none(self):
        """BAB64Blockchain works normally with storage=None."""
        identity = self._make_identity()
        bc = BAB64Blockchain(difficulty=1, miner=identity, storage=None)
        bc.genesis_block()
        block = bc.mine_block()
        self.assertIsNotNone(block)
        self.assertEqual(len(bc.chain), 2)


if __name__ == "__main__":
    unittest.main()
