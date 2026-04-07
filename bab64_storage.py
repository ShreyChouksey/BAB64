"""
BAB64 Storage — Persistent SQLite Storage Layer
=================================================

Survives node restarts by persisting blockchain, UTXO set,
wallet keys, and peer list to SQLite databases.

Components:
  1. BlockchainDB — blocks and transactions
  2. UTXODB — unspent transaction outputs
  3. WalletDB — encrypted identity keys
  4. PeerDB — known network peers
  5. BAB64Storage — unified wrapper

Author: Shrey (concept) + Claude (implementation)
"""

import hashlib
import json
import os
import sqlite3
import time
from typing import Dict, List, Optional, Tuple

from bab64_cash import (
    BAB64Block, BAB64Blockchain, BAB64CashTransaction,
    TxInput, TxOutput, UTXOSet,
)
from bab64_network import _serialize_block, _deserialize_block


# =============================================================================
# COMPONENT 1 — BLOCKCHAIN STORAGE
# =============================================================================

class BlockchainDB:
    """Persistent storage for the block chain."""

    def __init__(self, data_dir: str = "bab64_data"):
        db_path = os.path.join(data_dir, "blockchain.db")
        self._conn = sqlite3.connect(db_path)
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._create_tables()

    def _create_tables(self):
        self._conn.execute("""
            CREATE TABLE IF NOT EXISTS blocks (
                idx             INTEGER PRIMARY KEY,
                previous_hash   TEXT NOT NULL,
                timestamp       REAL NOT NULL,
                merkle_root     TEXT NOT NULL,
                nonce           INTEGER NOT NULL,
                difficulty      INTEGER NOT NULL,
                block_hash      TEXT NOT NULL UNIQUE,
                transactions_json TEXT NOT NULL
            )
        """)
        self._conn.commit()

    # -- write --

    def save_block(self, block: BAB64Block):
        """Serialize and store a block. Silently skips duplicates."""
        data = _serialize_block(block)
        txs_json = json.dumps(data["transactions"])
        try:
            self._conn.execute(
                """INSERT INTO blocks
                   (idx, previous_hash, timestamp, merkle_root,
                    nonce, difficulty, block_hash, transactions_json)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
                (block.index, block.previous_hash, block.timestamp,
                 block.merkle_root_hash, block.nonce, block.difficulty,
                 block.block_hash, txs_json),
            )
            self._conn.commit()
        except sqlite3.IntegrityError:
            pass  # duplicate block_hash or index

    # -- read --

    def load_chain(self) -> List[BAB64Block]:
        """Load all blocks ordered by index."""
        cur = self._conn.execute(
            "SELECT idx, previous_hash, timestamp, merkle_root, "
            "nonce, difficulty, block_hash, transactions_json "
            "FROM blocks ORDER BY idx"
        )
        blocks = []
        for row in cur:
            blocks.append(self._row_to_block(row))
        return blocks

    def get_block(self, index: int) -> Optional[BAB64Block]:
        cur = self._conn.execute(
            "SELECT idx, previous_hash, timestamp, merkle_root, "
            "nonce, difficulty, block_hash, transactions_json "
            "FROM blocks WHERE idx = ?", (index,)
        )
        row = cur.fetchone()
        return self._row_to_block(row) if row else None

    def get_block_by_hash(self, block_hash: str) -> Optional[BAB64Block]:
        cur = self._conn.execute(
            "SELECT idx, previous_hash, timestamp, merkle_root, "
            "nonce, difficulty, block_hash, transactions_json "
            "FROM blocks WHERE block_hash = ?", (block_hash,)
        )
        row = cur.fetchone()
        return self._row_to_block(row) if row else None

    def chain_height(self) -> int:
        """Return the highest block index, or -1 if empty."""
        cur = self._conn.execute("SELECT MAX(idx) FROM blocks")
        row = cur.fetchone()
        if row and row[0] is not None:
            return row[0]
        return -1

    def has_block(self, block_hash: str) -> bool:
        cur = self._conn.execute(
            "SELECT 1 FROM blocks WHERE block_hash = ? LIMIT 1",
            (block_hash,),
        )
        return cur.fetchone() is not None

    # -- internal --

    @staticmethod
    def _row_to_block(row) -> BAB64Block:
        idx, prev_hash, ts, mr, nonce, diff, bh, txs_json = row
        txs_data = json.loads(txs_json)
        data = {
            "index": idx,
            "previous_hash": prev_hash,
            "timestamp": ts,
            "merkle_root_hash": mr,
            "nonce": nonce,
            "difficulty": diff,
            "block_hash": bh,
            "transactions": txs_data,
        }
        return _deserialize_block(data)

    def close(self):
        self._conn.close()


# =============================================================================
# COMPONENT 2 — UTXO STORAGE
# =============================================================================

class UTXODB:
    """Persistent storage for unspent transaction outputs."""

    def __init__(self, data_dir: str = "bab64_data"):
        db_path = os.path.join(data_dir, "utxos.db")
        self._conn = sqlite3.connect(db_path)
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._create_tables()

    def _create_tables(self):
        self._conn.execute("""
            CREATE TABLE IF NOT EXISTS utxos (
                tx_hash         TEXT NOT NULL,
                output_index    INTEGER NOT NULL,
                recipient       TEXT NOT NULL,
                amount          INTEGER NOT NULL,
                lock_hash       TEXT NOT NULL DEFAULT '',
                lock_nonce      BLOB NOT NULL DEFAULT x'',
                coinbase_height INTEGER NOT NULL DEFAULT -1,
                spent           INTEGER NOT NULL DEFAULT 0,
                PRIMARY KEY (tx_hash, output_index)
            )
        """)
        self._conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_utxo_recipient "
            "ON utxos(recipient) WHERE spent = 0"
        )
        self._conn.commit()

    def save_utxo(self, out: TxOutput):
        """Insert or replace a UTXO."""
        self._conn.execute(
            """INSERT OR REPLACE INTO utxos
               (tx_hash, output_index, recipient, amount,
                lock_hash, lock_nonce, coinbase_height, spent)
               VALUES (?, ?, ?, ?, ?, ?, ?, 0)""",
            (out.tx_hash, out.index, out.recipient, out.amount,
             out.lock_hash, out.lock_nonce, out.coinbase_height),
        )
        self._conn.commit()

    def spend_utxo(self, tx_hash: str, index: int):
        """Mark a UTXO as spent."""
        self._conn.execute(
            "UPDATE utxos SET spent = 1 WHERE tx_hash = ? AND output_index = ?",
            (tx_hash, index),
        )
        self._conn.commit()

    def get_utxo(self, tx_hash: str, index: int) -> Optional[TxOutput]:
        """Get an unspent UTXO."""
        cur = self._conn.execute(
            "SELECT tx_hash, output_index, recipient, amount, "
            "lock_hash, lock_nonce, coinbase_height "
            "FROM utxos WHERE tx_hash = ? AND output_index = ? AND spent = 0",
            (tx_hash, index),
        )
        row = cur.fetchone()
        return self._row_to_output(row) if row else None

    def get_utxos_for(self, address: str) -> List[TxOutput]:
        """Get all unspent UTXOs for an address."""
        cur = self._conn.execute(
            "SELECT tx_hash, output_index, recipient, amount, "
            "lock_hash, lock_nonce, coinbase_height "
            "FROM utxos WHERE recipient = ? AND spent = 0",
            (address,),
        )
        return [self._row_to_output(row) for row in cur]

    def balance(self, address: str) -> int:
        """Sum unspent amounts for an address."""
        cur = self._conn.execute(
            "SELECT COALESCE(SUM(amount), 0) FROM utxos "
            "WHERE recipient = ? AND spent = 0",
            (address,),
        )
        return cur.fetchone()[0]

    def rebuild_from_chain(self, blocks: List[BAB64Block]):
        """Clear all UTXOs and replay the chain to reconstruct the set."""
        self._conn.execute("DELETE FROM utxos")
        for block in blocks:
            for tx in block.transactions:
                # Add outputs
                for out in tx.outputs:
                    self._conn.execute(
                        """INSERT OR REPLACE INTO utxos
                           (tx_hash, output_index, recipient, amount,
                            lock_hash, lock_nonce, coinbase_height, spent)
                           VALUES (?, ?, ?, ?, ?, ?, ?, 0)""",
                        (out.tx_hash, out.index, out.recipient, out.amount,
                         out.lock_hash, out.lock_nonce, out.coinbase_height),
                    )
                # Spend inputs (non-coinbase)
                if not tx.is_coinbase:
                    for inp in tx.inputs:
                        self._conn.execute(
                            "UPDATE utxos SET spent = 1 "
                            "WHERE tx_hash = ? AND output_index = ?",
                            (inp.prev_tx_hash, inp.prev_index),
                        )
        self._conn.commit()

    @staticmethod
    def _row_to_output(row) -> TxOutput:
        tx_hash, idx, recipient, amount, lock_hash, lock_nonce, cb_height = row
        return TxOutput(
            recipient=recipient,
            amount=amount,
            tx_hash=tx_hash,
            index=idx,
            lock_hash=lock_hash,
            lock_nonce=lock_nonce if isinstance(lock_nonce, bytes) else b"",
            coinbase_height=cb_height,
        )

    def close(self):
        self._conn.close()


# =============================================================================
# COMPONENT 3 — WALLET STORAGE
# =============================================================================

class WalletDB:
    """Persistent storage for encrypted identity keys."""

    def __init__(self, data_dir: str = "bab64_data"):
        db_path = os.path.join(data_dir, "wallet.db")
        self._conn = sqlite3.connect(db_path)
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._create_tables()

    def _create_tables(self):
        self._conn.execute("""
            CREATE TABLE IF NOT EXISTS identities (
                address_hex     TEXT PRIMARY KEY,
                encrypted_key   BLOB NOT NULL,
                ibst_state      INTEGER NOT NULL DEFAULT 0,
                created_at      REAL NOT NULL
            )
        """)
        self._conn.commit()

    @staticmethod
    def _derive_mask(passphrase: str) -> bytes:
        """SHA-256(passphrase) -> 32-byte XOR mask."""
        return hashlib.sha256(passphrase.encode()).digest()

    def save_identity(self, identity, passphrase: str):
        """Encrypt the private key with passphrase and store."""
        from bab64_identity import BAB64Identity
        mask = self._derive_mask(passphrase)
        pk = identity._private_key
        encrypted = bytes(a ^ b for a, b in zip(pk, mask))
        self._conn.execute(
            """INSERT OR REPLACE INTO identities
               (address_hex, encrypted_key, ibst_state, created_at)
               VALUES (?, ?, ?, ?)""",
            (identity.address_hex, encrypted, 0, time.time()),
        )
        self._conn.commit()

    def load_identity(self, address: str, passphrase: str):
        """Decrypt and reconstruct a BAB64Identity. Returns None on wrong passphrase."""
        from bab64_identity import BAB64Identity
        cur = self._conn.execute(
            "SELECT encrypted_key FROM identities WHERE address_hex = ?",
            (address,),
        )
        row = cur.fetchone()
        if not row:
            return None
        encrypted = row[0]
        mask = self._derive_mask(passphrase)
        pk = bytes(a ^ b for a, b in zip(encrypted, mask))
        identity = BAB64Identity(pk)
        # Verify: reconstructed address must match stored address
        if identity.address_hex != address:
            return None  # wrong passphrase
        return identity

    def list_addresses(self) -> List[str]:
        cur = self._conn.execute("SELECT address_hex FROM identities")
        return [row[0] for row in cur]

    def delete_identity(self, address: str):
        self._conn.execute(
            "DELETE FROM identities WHERE address_hex = ?", (address,),
        )
        self._conn.commit()

    def close(self):
        self._conn.close()


# =============================================================================
# COMPONENT 4 — PEER STORAGE
# =============================================================================

class PeerDB:
    """Persistent storage for known network peers."""

    def __init__(self, data_dir: str = "bab64_data"):
        db_path = os.path.join(data_dir, "peers.db")
        self._conn = sqlite3.connect(db_path)
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._create_tables()

    def _create_tables(self):
        self._conn.execute("""
            CREATE TABLE IF NOT EXISTS peers (
                host            TEXT NOT NULL,
                port            INTEGER NOT NULL,
                last_seen       REAL NOT NULL DEFAULT 0,
                times_connected INTEGER NOT NULL DEFAULT 0,
                banned          INTEGER NOT NULL DEFAULT 0,
                PRIMARY KEY (host, port)
            )
        """)
        self._conn.commit()

    def save_peer(self, host: str, port: int):
        """Insert a new peer or bump times_connected."""
        self._conn.execute(
            """INSERT INTO peers (host, port, last_seen, times_connected, banned)
               VALUES (?, ?, ?, 1, 0)
               ON CONFLICT(host, port)
               DO UPDATE SET times_connected = times_connected + 1""",
            (host, port, time.time()),
        )
        self._conn.commit()

    def get_peers(self) -> List[Tuple[str, int]]:
        """Return all known peers (host, port)."""
        cur = self._conn.execute("SELECT host, port FROM peers")
        return [(row[0], row[1]) for row in cur]

    def update_last_seen(self, host: str, port: int):
        self._conn.execute(
            "UPDATE peers SET last_seen = ? WHERE host = ? AND port = ?",
            (time.time(), host, port),
        )
        self._conn.commit()

    def ban_peer(self, host: str, port: int):
        self._conn.execute(
            "UPDATE peers SET banned = 1 WHERE host = ? AND port = ?",
            (host, port),
        )
        self._conn.commit()

    def get_unbanned_peers(self) -> List[Tuple[str, int]]:
        cur = self._conn.execute(
            "SELECT host, port FROM peers WHERE banned = 0"
        )
        return [(row[0], row[1]) for row in cur]

    def close(self):
        self._conn.close()


# =============================================================================
# COMPONENT 5 — UNIFIED STORAGE
# =============================================================================

class BAB64Storage:
    """Wraps all DB components under a single data directory."""

    def __init__(self, data_dir: str = "bab64_data"):
        os.makedirs(data_dir, exist_ok=True)
        self.data_dir = data_dir
        self.blockchain = BlockchainDB(data_dir)
        self.utxos = UTXODB(data_dir)
        self.wallet = WalletDB(data_dir)
        self.peers = PeerDB(data_dir)

    def save_state(self, chain: List[BAB64Block]):
        """Save entire chain and rebuild UTXO index."""
        for block in chain:
            self.blockchain.save_block(block)
        self.utxos.rebuild_from_chain(chain)

    def load_state(self) -> Tuple[List[BAB64Block], UTXOSet]:
        """Load chain from disk and rebuild in-memory UTXO set."""
        chain = self.blockchain.load_chain()
        utxo_set = UTXOSet()
        for block in chain:
            for tx in block.transactions:
                if not tx.is_coinbase:
                    for inp in tx.inputs:
                        utxo_set.spend(inp.prev_tx_hash, inp.prev_index)
                utxo_set.add_outputs(tx)
        return chain, utxo_set

    def close(self):
        self.blockchain.close()
        self.utxos.close()
        self.wallet.close()
        self.peers.close()
