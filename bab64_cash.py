"""
BAB64 Cash — Bitcoin-like Electronic Cash System
==================================================

A UTXO-based electronic cash system built on BAB64 identities.

Architecture:
  - UTXO model (like Bitcoin): coins are unspent transaction outputs
  - Hashlock ownership: UTXOs locked to owner's image via challenge-response
  - Coinbase transactions: block rewards with halving schedule
  - Merkle tree: transaction integrity via hash tree
  - PoW mining: SHA-256 block hashing with difficulty target
  - BAB64 identities: image-derived addresses and Lamport signatures

Ownership model:
  Each UTXO carries a hashlock derived from the recipient's private image.
  To spend, the owner provides a proof that they know the image, plus a
  Lamport signature. This prevents non-owners from spending UTXOs even
  if they can produce valid signatures with their own keys.

Supply:
  - Block reward: 50 BAB64 coins (5,000,000,000 satoshis)
  - Halving: every 210,000 blocks
  - Total supply cap: 21,000,000 BAB64 coins

Author: Shrey (concept) + Claude (implementation)
"""

import copy
import hashlib
import os
import statistics
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

from bab64_identity import BAB64Identity, LamportKeyPair
from bab64_engine import BAB64Config as EngineConfig, BabelRenderer, ImageHash
from bab64_signatures import (
    BAB64IBSTIdentity, BAB64WOTS, BAB64MerkleTree,
    IBSTSignature, ImageChainFunction,
)

# Activate C extension for BAB64 hashing if available
try:
    import bab64_fast
    _has_c_extension = bab64_fast.is_available()
except ImportError:
    _has_c_extension = False

# Shared engine objects (initialized once, reused for all mining/verification)
_engine_config = EngineConfig()
_renderer = BabelRenderer(_engine_config)
_hasher = ImageHash(_engine_config)

# Cache of IBST identities, keyed by BAB64 address hex.
# Avoids re-generating 1,024 WOTS+ keys on every sign.
_ibst_cache: Dict[str, BAB64IBSTIdentity] = {}


def _get_ibst(identity: BAB64Identity) -> BAB64IBSTIdentity:
    """Get or create the cached BAB64IBSTIdentity for a BAB64Identity."""
    addr = identity.address_hex
    if addr not in _ibst_cache:
        _ibst_cache[addr] = BAB64IBSTIdentity(identity._private_key)
    return _ibst_cache[addr]


# =============================================================================
# CONSTANTS
# =============================================================================

COIN = 100_000_000          # 1 BAB64 = 100,000,000 satoshis
INITIAL_REWARD = 50 * COIN  # 50 BAB64 per block
HALVING_INTERVAL = 210_000  # blocks between halvings
MAX_SUPPLY = 21_000_000 * COIN

# Consensus constants
TARGET_BLOCK_TIME = 60          # 60 seconds target
ADJUSTMENT_INTERVAL = 10        # adjust every 10 blocks
MAX_FUTURE_BLOCK_TIME = 7200    # 2 hours
GENESIS_TIMESTAMP = 1712444400  # April 7, 2024
GENESIS_ADDRESS = "0" * 64      # hardcoded genesis recipient
GENESIS_MESSAGE = "BAB64/Genesis/2026"

# Economics constants
DUST_THRESHOLD = 546            # minimum output amount (satoshis)
COINBASE_MATURITY = 100         # blocks before coinbase can be spent
MAX_BLOCK_SIZE = 1_000_000      # 1 MB in estimated bytes
MAX_BLOCK_TRANSACTIONS = 100    # hard cap on tx count (including coinbase)


# =============================================================================
# HASHLOCK — Ownership Proofs
# =============================================================================

# =============================================================================
# FEE POLICY
# =============================================================================

class FeePolicy:
    """Transaction fee policy for relay and mining."""
    MINIMUM_RELAY_FEE = 1000    # 1000 satoshis minimum
    FEE_PER_BYTE = 10           # satoshis per byte of tx

    @staticmethod
    def tx_size(tx: 'BAB64CashTransaction') -> int:
        """Estimate transaction size in bytes.
        Each input: 32 (prev_hash) + 4 (index) + 2144 (WOTS+ sig) + 2144 (WOTS+ pk)
                    + 320 (auth_path) + 4 (leaf_index) + 32 (merkle_root)
                    + image_bytes (~12288) = ~16966
        Each output: 32 (address) + 8 (amount) + 32 (lock_hash) + 32 (lock_nonce) = ~104
        Overhead: 32 (tx_hash) + 1 (is_coinbase) = 33
        """
        size = 33  # overhead
        size += len(tx.inputs) * 16966
        size += len(tx.outputs) * 104
        return size

    @staticmethod
    def minimum_fee(tx: 'BAB64CashTransaction') -> int:
        """Minimum fee based on tx size."""
        return max(FeePolicy.MINIMUM_RELAY_FEE,
                   FeePolicy.tx_size(tx) * FeePolicy.FEE_PER_BYTE)

    @staticmethod
    def fee_rate(tx: 'BAB64CashTransaction', utxo_set: 'UTXOSet') -> float:
        """Fee per byte — used for mempool priority sorting."""
        size = FeePolicy.tx_size(tx)
        if size == 0:
            return 0.0
        return tx.fee(utxo_set) / size


def compute_lock(image_bytes: bytes) -> Tuple[str, bytes]:
    """
    Create a hashlock for a UTXO recipient.

    Returns (lock_hash, lock_nonce).
    The lock_nonce is stored with the UTXO. Only the image owner
    can compute the matching owner_proof to unlock it.
    """
    nonce = os.urandom(32)
    proof = hashlib.sha256(image_bytes + nonce).digest()
    lock_hash = hashlib.sha256(proof).hexdigest()
    return lock_hash, nonce


def compute_unlock(image_bytes: bytes, lock_nonce: bytes) -> bytes:
    """Compute the owner_proof needed to spend a locked UTXO."""
    return hashlib.sha256(image_bytes + lock_nonce).digest()


def verify_lock(owner_proof: bytes, lock_hash: str) -> bool:
    """Check that an owner_proof matches a lock_hash."""
    return hashlib.sha256(owner_proof).hexdigest() == lock_hash


# =============================================================================
# COMPONENT 1 — UTXO: Transaction Outputs and Inputs
# =============================================================================

@dataclass
class TxOutput:
    """An unspent transaction output — a coin."""
    recipient: str       # public address (hex)
    amount: int          # satoshis
    tx_hash: str         # which transaction created this
    index: int           # output index within that transaction
    lock_hash: str = ""  # hashlock: SHA256(owner_proof)
    lock_nonce: bytes = field(default_factory=bytes)  # nonce for lock derivation
    coinbase_height: int = -1  # block height if from coinbase, -1 otherwise


@dataclass
class TxInput:
    """A reference to a UTXO being spent."""
    prev_tx_hash: str   # points to a TxOutput
    prev_index: int     # which output of that transaction
    signature: List[bytes] = field(default_factory=list)        # WOTS+ sig (67 x 32B)
    verification_key: List[bytes] = field(default_factory=list)  # WOTS+ pk (67 x 32B)
    owner_proof: bytes = field(default_factory=bytes)  # proves image knowledge
    ibst_leaf_index: int = 0
    ibst_auth_path: List[bytes] = field(default_factory=list)    # Merkle path (10 x 32B)
    ibst_merkle_root: bytes = field(default_factory=bytes)       # tree root (32B)
    ibst_image_bytes: bytes = field(default_factory=bytes)       # signer's image (for chain fn)


# =============================================================================
# COMPONENT 2 — COINBASE TRANSACTION
# =============================================================================

def block_reward(height: int) -> int:
    """Compute block reward with halving schedule."""
    halvings = height // HALVING_INTERVAL
    if halvings >= 64:
        return 0
    return INITIAL_REWARD >> halvings


# =============================================================================
# COMPONENT 3 — FULL TRANSACTION
# =============================================================================

class BAB64CashTransaction:
    """
    A transaction spending UTXOs and creating new ones.

    Rules:
      - Sum of input values >= sum of output values
      - Difference = transaction fee
      - Each input must reference a valid unspent UTXO
      - Each input must have a valid owner proof (hashlock)
      - Each input must have a valid signature from the UTXO owner
      - No duplicate inputs
    """

    def __init__(self, inputs: List[TxInput] = None,
                 outputs: List[TxOutput] = None,
                 is_coinbase: bool = False,
                 coinbase_height: int = -1):
        self.inputs: List[TxInput] = inputs or []
        self.outputs: List[TxOutput] = outputs or []
        self.is_coinbase = is_coinbase
        self.coinbase_height = coinbase_height
        self.tx_hash: str = ""

    def compute_hash(self) -> str:
        """Hash all inputs and outputs to produce the transaction ID."""
        h = hashlib.sha256()
        for inp in self.inputs:
            h.update(inp.prev_tx_hash.encode())
            h.update(inp.prev_index.to_bytes(4, 'big'))
        for out in self.outputs:
            h.update(out.recipient.encode())
            h.update(out.amount.to_bytes(8, 'big', signed=True))
        if self.is_coinbase:
            h.update(b'coinbase')
            h.update(self.coinbase_height.to_bytes(4, 'big', signed=True))
        return h.hexdigest()

    def finalize(self):
        """Compute tx_hash and stamp it on all outputs."""
        self.tx_hash = self.compute_hash()
        for i, out in enumerate(self.outputs):
            out.tx_hash = self.tx_hash
            out.index = i

    def signing_payload(self) -> bytes:
        """The data that signers commit to (everything except signatures)."""
        h = hashlib.sha256()
        for inp in self.inputs:
            h.update(inp.prev_tx_hash.encode())
            h.update(inp.prev_index.to_bytes(4, 'big'))
        for out in self.outputs:
            h.update(out.recipient.encode())
            h.update(out.amount.to_bytes(8, 'big', signed=True))
        return h.digest()

    def sign_input(self, input_index: int, identity: BAB64Identity,
                   lock_nonce: bytes = None):
        """
        Sign a specific input with the owner's IBST identity.

        Uses WOTS+ signature from the Merkle tree of 1,024 keys.
        If lock_nonce is provided, also computes the owner_proof
        needed to pass the hashlock check.
        """
        payload = self.signing_payload()
        ibst = _get_ibst(identity)
        ibst_sig = ibst.sign(payload)
        inp = self.inputs[input_index]
        inp.signature = ibst_sig.wots_signature
        inp.verification_key = ibst_sig.wots_public_key
        inp.ibst_leaf_index = ibst_sig.leaf_index
        inp.ibst_auth_path = ibst_sig.auth_path
        inp.ibst_merkle_root = ibst.merkle_root
        inp.ibst_image_bytes = identity._image_bytes
        if lock_nonce:
            self.inputs[input_index].owner_proof = compute_unlock(
                identity._image_bytes, lock_nonce
            )

    def fee(self, utxo_set: 'UTXOSet') -> int:
        """Calculate transaction fee (input total - output total)."""
        if self.is_coinbase:
            return 0
        input_total = 0
        for inp in self.inputs:
            utxo = utxo_set.get(inp.prev_tx_hash, inp.prev_index)
            if utxo:
                input_total += utxo.amount
        output_total = sum(out.amount for out in self.outputs)
        return input_total - output_total

    @staticmethod
    def create_coinbase(recipient: str, height: int, fees: int = 0,
                        image_bytes: bytes = None) -> 'BAB64CashTransaction':
        """
        Create a coinbase transaction (block reward + fees).

        If image_bytes is provided, the output is locked to the
        recipient's image via hashlock.
        """
        reward = block_reward(height) + fees
        tx = BAB64CashTransaction(is_coinbase=True, coinbase_height=height)

        lock_hash = ""
        lock_nonce = b""
        if image_bytes:
            lock_hash, lock_nonce = compute_lock(image_bytes)

        tx.outputs.append(TxOutput(
            recipient=recipient,
            amount=reward,
            tx_hash="",
            index=0,
            lock_hash=lock_hash,
            lock_nonce=lock_nonce,
            coinbase_height=height,
        ))
        tx.finalize()
        return tx


# =============================================================================
# COMPONENT 4 — UTXO SET
# =============================================================================

class UTXOSet:
    """
    The set of all unspent transaction outputs.

    Key: (tx_hash, index) -> TxOutput
    """

    def __init__(self):
        self._utxos: Dict[Tuple[str, int], TxOutput] = {}

    def add_outputs(self, tx: BAB64CashTransaction):
        """Add all outputs from a transaction as unspent."""
        for out in tx.outputs:
            self._utxos[(out.tx_hash, out.index)] = out

    def spend(self, tx_hash: str, index: int) -> Optional[TxOutput]:
        """Mark a UTXO as spent (remove it). Returns the spent output."""
        key = (tx_hash, index)
        return self._utxos.pop(key, None)

    def get(self, tx_hash: str, index: int) -> Optional[TxOutput]:
        """Retrieve a UTXO without spending it."""
        return self._utxos.get((tx_hash, index))

    def balance(self, address: str) -> int:
        """Sum all UTXOs owned by an address."""
        return sum(
            utxo.amount for utxo in self._utxos.values()
            if utxo.recipient == address
        )

    def get_utxos_for(self, address: str) -> List[TxOutput]:
        """Get all UTXOs owned by an address."""
        return [
            utxo for utxo in self._utxos.values()
            if utxo.recipient == address
        ]

    def validate_transaction(self, tx: BAB64CashTransaction,
                             current_height: int = None) -> Tuple[bool, str]:
        """
        Validate a transaction against the UTXO set.

        Returns (is_valid, error_message).
        """
        if tx.is_coinbase:
            if tx.inputs:
                return False, "Coinbase must have no inputs"
            if not tx.outputs:
                return False, "Coinbase must have at least one output"
            for out in tx.outputs:
                if out.amount <= 0:
                    return False, "Output amount must be positive"
            return True, ""

        # Regular transaction checks
        if not tx.inputs:
            return False, "Transaction must have at least one input"
        if not tx.outputs:
            return False, "Transaction must have at least one output"

        # Check for negative/zero output amounts
        for out in tx.outputs:
            if out.amount <= 0:
                return False, "Output amount must be positive"

        # Check for duplicate inputs
        seen_inputs = set()
        for inp in tx.inputs:
            key = (inp.prev_tx_hash, inp.prev_index)
            if key in seen_inputs:
                return False, "Duplicate input"
            seen_inputs.add(key)

        # Validate each input
        input_total = 0
        for inp in tx.inputs:
            utxo = self.get(inp.prev_tx_hash, inp.prev_index)
            if utxo is None:
                return False, (
                    f"Input references non-existent UTXO: "
                    f"{inp.prev_tx_hash[:16]}:{inp.prev_index}"
                )

            # Coinbase maturity check
            if utxo.coinbase_height >= 0 and current_height is not None:
                if current_height - utxo.coinbase_height < COINBASE_MATURITY:
                    return False, (
                        f"Coinbase output not mature: need {COINBASE_MATURITY} confirmations, "
                        f"have {current_height - utxo.coinbase_height}"
                    )

            # Check hashlock (ownership proof)
            if utxo.lock_hash:
                if not inp.owner_proof:
                    return False, "Missing owner proof"
                if not verify_lock(inp.owner_proof, utxo.lock_hash):
                    return False, "Invalid owner proof — not the UTXO owner"

            # Verify IBST signature (WOTS+ + Merkle path)
            if not inp.signature or not inp.verification_key:
                return False, "Input missing signature"
            if not inp.ibst_image_bytes or not inp.ibst_merkle_root:
                return False, "Input missing IBST verification data"

            payload = tx.signing_payload()
            ibst_sig = IBSTSignature(
                wots_signature=inp.signature,
                wots_public_key=inp.verification_key,
                leaf_index=inp.ibst_leaf_index,
                auth_path=inp.ibst_auth_path,
            )
            if not BAB64IBSTIdentity.verify_standalone(
                payload, ibst_sig, inp.ibst_merkle_root,
                inp.ibst_image_bytes,
            ):
                return False, "Invalid IBST signature"

            input_total += utxo.amount

        # Check value conservation
        output_total = sum(out.amount for out in tx.outputs)
        if input_total < output_total:
            return False, (
                f"Insufficient funds: inputs={input_total}, "
                f"outputs={output_total}"
            )

        return True, ""

    def validate_relay_policy(self, tx: BAB64CashTransaction) -> Tuple[bool, str]:
        """
        Check relay policy rules (fee minimum, dust threshold).
        These are not consensus rules — they protect the mempool from spam.
        """
        if tx.is_coinbase:
            return True, ""

        # Dust check
        for out in tx.outputs:
            if out.amount < DUST_THRESHOLD:
                return False, f"Output below dust threshold: {out.amount} < {DUST_THRESHOLD}"

        # Minimum relay fee
        fee = tx.fee(self)
        min_fee = FeePolicy.minimum_fee(tx)
        if fee < min_fee:
            return False, f"Fee below minimum: {fee} < {min_fee}"

        return True, ""

    def apply_transaction(self, tx: BAB64CashTransaction) -> bool:
        """Apply a validated transaction: spend inputs, add outputs."""
        if not tx.is_coinbase:
            for inp in tx.inputs:
                self.spend(inp.prev_tx_hash, inp.prev_index)
        self.add_outputs(tx)
        return True


# =============================================================================
# COMPONENT 5 — MERKLE TREE
# =============================================================================

def merkle_root(tx_hashes: List[str]) -> str:
    """Compute the Merkle root of a list of transaction hashes."""
    if not tx_hashes:
        return hashlib.sha256(b'empty').hexdigest()

    level = [bytes.fromhex(h) for h in tx_hashes]

    while len(level) > 1:
        next_level = []
        if len(level) % 2 == 1:
            level.append(level[-1])
        for i in range(0, len(level), 2):
            combined = hashlib.sha256(level[i] + level[i + 1]).digest()
            next_level.append(combined)
        level = next_level

    return level[0].hex()


# =============================================================================
# COMPONENT 6 — FULL BLOCK
# =============================================================================

@dataclass
class BlockHeader:
    """Block header — separates metadata from body for efficient verification."""
    index: int
    previous_hash: str
    timestamp: float
    merkle_root: str
    difficulty: int
    nonce: int
    block_hash: str


@dataclass
class BAB64Block:
    """A block in the BAB64 Cash blockchain."""
    index: int
    previous_hash: str
    timestamp: float
    transactions: List[BAB64CashTransaction]
    merkle_root_hash: str
    nonce: int
    difficulty: int
    block_hash: str

    def header(self) -> BlockHeader:
        """Extract the block header for lightweight verification."""
        return BlockHeader(
            index=self.index,
            previous_hash=self.previous_hash,
            timestamp=self.timestamp,
            merkle_root=self.merkle_root_hash,
            difficulty=self.difficulty,
            nonce=self.nonce,
            block_hash=self.block_hash,
        )


class BAB64BlockMiner:
    """
    Mines blocks using BAB64 self-referential image hashing.

    For each nonce candidate:
      1. Build header string from (index, prev_hash, timestamp, merkle_root, difficulty, nonce)
      2. SHA-256 the header to get a 32-byte seed
      3. BabelRenderer generates a 64x64 image from that seed
      4. Derive hash parameters (S-box, round constants, rotations, initial state) FROM the image
      5. Hash the image using its OWN derived function (self-referential)
      6. Check if the BAB64 hash meets the difficulty target

    Verification costs exactly ONE mining attempt — regenerate image, re-derive, re-hash.
    """

    @staticmethod
    def _header_seed(index: int, previous_hash: str,
                     timestamp: float, merkle_root_hash: str,
                     nonce: int, difficulty: int) -> bytes:
        """Build a 32-byte seed from block header fields."""
        header = (
            f"{index}:{previous_hash}:{timestamp}:"
            f"{merkle_root_hash}:{nonce}:{difficulty}"
        )
        return hashlib.sha256(header.encode()).digest()

    @staticmethod
    def compute_block_hash(index: int, previous_hash: str,
                           timestamp: float, merkle_root_hash: str,
                           nonce: int, difficulty: int) -> str:
        """
        Compute block hash using BAB64 self-referential image hashing.

        header → seed → image → derive H_I from image → H_I(image) → hash
        """
        seed = BAB64BlockMiner._header_seed(
            index, previous_hash, timestamp, merkle_root_hash,
            nonce, difficulty
        )
        image = _renderer.render(seed)
        bab64_hash = _hasher.hash_image(image)
        return bab64_hash.hex()

    @staticmethod
    def meets_difficulty(block_hash: str, difficulty: int) -> bool:
        """Check if hash has enough leading zero bits."""
        hash_int = int(block_hash, 16)
        if hash_int == 0:
            return True
        leading_zeros = 256 - hash_int.bit_length()
        return leading_zeros >= difficulty

    @staticmethod
    def mine_block(index: int, previous_hash: str,
                   transactions: List[BAB64CashTransaction],
                   difficulty: int,
                   max_nonces: int = 10_000_000) -> Optional[BAB64Block]:
        """Mine a block by finding a nonce whose BAB64 hash meets difficulty."""
        tx_hashes = [tx.tx_hash for tx in transactions]
        mr = merkle_root(tx_hashes)
        timestamp = time.time()

        for nonce in range(max_nonces):
            bh = BAB64BlockMiner.compute_block_hash(
                index, previous_hash, timestamp, mr, nonce, difficulty
            )
            if BAB64BlockMiner.meets_difficulty(bh, difficulty):
                return BAB64Block(
                    index=index,
                    previous_hash=previous_hash,
                    timestamp=timestamp,
                    transactions=transactions,
                    merkle_root_hash=mr,
                    nonce=nonce,
                    difficulty=difficulty,
                    block_hash=bh,
                )
        return None


# =============================================================================
# COMPONENT 7 — CHAIN SELECTION
# =============================================================================

class ChainSelector:
    """Select the best chain among competing forks."""

    @staticmethod
    def cumulative_work(chain: List[BAB64Block]) -> int:
        """Sum of 2^difficulty for each block."""
        return sum(2 ** block.difficulty for block in chain)

    @staticmethod
    def select_chain(chain_a: List[BAB64Block], chain_b: List[BAB64Block],
                     utxo_set: UTXOSet) -> List[BAB64Block]:
        """Return the chain with more cumulative work, if it's fully valid."""
        work_a = ChainSelector.cumulative_work(chain_a)
        work_b = ChainSelector.cumulative_work(chain_b)

        if work_b > work_a:
            # Validate chain_b fully
            temp_utxo = UTXOSet()
            for i, block in enumerate(chain_b):
                expected_prev = "0" * 64 if i == 0 else chain_b[i - 1].block_hash
                # Validate block structure
                recomputed = BAB64BlockMiner.compute_block_hash(
                    block.index, block.previous_hash, block.timestamp,
                    block.merkle_root_hash, block.nonce, block.difficulty,
                )
                if recomputed != block.block_hash:
                    return chain_a
                if not BAB64BlockMiner.meets_difficulty(block.block_hash, block.difficulty):
                    return chain_a
                if block.previous_hash != expected_prev:
                    return chain_a
                if block.index != i:
                    return chain_a
                # Apply transactions
                for tx in block.transactions:
                    if tx.is_coinbase:
                        temp_utxo.add_outputs(tx)
                    else:
                        valid, _ = temp_utxo.validate_transaction(tx)
                        if not valid:
                            return chain_a
                        temp_utxo.apply_transaction(tx)
            return chain_b
        return chain_a


# =============================================================================
# COMPONENT 8 — BLOCKCHAIN
# =============================================================================

class BAB64Blockchain:
    """
    The BAB64 Cash blockchain.

    Manages the chain of blocks, UTXO set, and mempool.
    """

    def __init__(self, difficulty: int = 4,
                 miner: BAB64Identity = None,
                 miner_address: str = ""):
        self.difficulty = difficulty
        self.miner = miner
        self.miner_address = miner_address or (miner.address_hex if miner else "")
        self.chain: List[BAB64Block] = []
        self.utxo_set = UTXOSet()
        self.mempool: List[BAB64CashTransaction] = []

    def _miner_image(self) -> Optional[bytes]:
        """Get the miner's image bytes for hashlock creation."""
        if self.miner:
            return self.miner._image_bytes
        return None

    @staticmethod
    def create_genesis_block() -> BAB64Block:
        """
        Create the hardcoded genesis block.

        Fixed timestamp, genesis address, embedded message.
        """
        # Coinbase with genesis message embedded in the hash
        coinbase = BAB64CashTransaction(is_coinbase=True, coinbase_height=0)
        coinbase.outputs.append(TxOutput(
            recipient=GENESIS_ADDRESS,
            amount=INITIAL_REWARD,
            tx_hash="",
            index=0,
        ))
        # Embed genesis message into the tx hash
        h = hashlib.sha256()
        h.update(GENESIS_MESSAGE.encode())
        h.update(GENESIS_ADDRESS.encode())
        h.update(b'coinbase')
        h.update((0).to_bytes(4, 'big', signed=True))
        coinbase.tx_hash = h.hexdigest()
        for i, out in enumerate(coinbase.outputs):
            out.tx_hash = coinbase.tx_hash
            out.index = i

        tx_hashes = [coinbase.tx_hash]
        mr = merkle_root(tx_hashes)

        # Mine with difficulty 1 and fixed timestamp
        for nonce in range(10_000_000):
            bh = BAB64BlockMiner.compute_block_hash(
                0, "0" * 64, GENESIS_TIMESTAMP, mr, nonce, 1
            )
            if BAB64BlockMiner.meets_difficulty(bh, 1):
                return BAB64Block(
                    index=0,
                    previous_hash="0" * 64,
                    timestamp=GENESIS_TIMESTAMP,
                    transactions=[coinbase],
                    merkle_root_hash=mr,
                    nonce=nonce,
                    difficulty=1,
                    block_hash=bh,
                )
        raise RuntimeError("Failed to mine genesis block")

    def genesis_block(self, miner: BAB64Identity = None) -> BAB64Block:
        """Create and add block 0 with initial coinbase."""
        m = miner or self.miner
        addr = m.address_hex if m else self.miner_address
        img = m._image_bytes if m else None

        coinbase = BAB64CashTransaction.create_coinbase(
            addr, height=0, image_bytes=img
        )

        block = BAB64BlockMiner.mine_block(
            index=0,
            previous_hash="0" * 64,
            transactions=[coinbase],
            difficulty=self.difficulty,
        )
        if block is None:
            raise RuntimeError("Failed to mine genesis block")

        self.chain.append(block)
        self.utxo_set.apply_transaction(coinbase)
        return block

    def add_genesis(self) -> BAB64Block:
        """Add the hardcoded genesis block and apply its coinbase."""
        block = BAB64Blockchain.create_genesis_block()
        self.chain.append(block)
        self.difficulty = block.difficulty
        for tx in block.transactions:
            self.utxo_set.apply_transaction(tx)
        return block

    def add_transaction_to_mempool(self, tx: BAB64CashTransaction,
                                   enforce_policy: bool = False) -> Tuple[bool, str]:
        """Validate and add a transaction to the mempool.
        If enforce_policy=True, also checks relay policy (fee/dust)
        and coinbase maturity."""
        current_height = len(self.chain) if enforce_policy else None
        valid, err = self.utxo_set.validate_transaction(tx, current_height=current_height)
        if not valid:
            return False, err
        if enforce_policy:
            valid, err = self.utxo_set.validate_relay_policy(tx)
            if not valid:
                return False, err
        self.mempool.append(tx)
        return True, ""

    def mine_block(self, miner: BAB64Identity = None,
                   timestamp: float = None) -> Optional[BAB64Block]:
        """Mine a new block with mempool transactions, respecting limits."""
        m = miner or self.miner
        addr = m.address_hex if m else self.miner_address
        img = m._image_bytes if m else None
        height = len(self.chain)
        prev_hash = self.chain[-1].block_hash if self.chain else "0" * 64

        # Select transactions by fee_rate descending, respecting limits
        sorted_mempool = sorted(
            self.mempool,
            key=lambda t: FeePolicy.fee_rate(t, self.utxo_set),
            reverse=True,
        )

        selected = []
        # Reserve space for coinbase (1 tx slot + its size)
        coinbase_size = 33 + 104  # overhead + 1 output, no inputs
        block_size = coinbase_size
        tx_count = 1  # coinbase

        for tx in sorted_mempool:
            tx_sz = FeePolicy.tx_size(tx)
            if tx_count + 1 > MAX_BLOCK_TRANSACTIONS:
                break
            if block_size + tx_sz > MAX_BLOCK_SIZE:
                break
            selected.append(tx)
            block_size += tx_sz
            tx_count += 1

        # Calculate total fees from selected transactions
        fees = sum(tx.fee(self.utxo_set) for tx in selected)

        # Coinbase first
        coinbase = BAB64CashTransaction.create_coinbase(
            addr, height, fees, image_bytes=img
        )
        transactions = [coinbase] + selected

        # Use provided timestamp or current time
        tx_hashes = [tx.tx_hash for tx in transactions]
        mr = merkle_root(tx_hashes)
        ts = timestamp if timestamp is not None else time.time()

        block = None
        for nonce in range(10_000_000):
            bh = BAB64BlockMiner.compute_block_hash(
                height, prev_hash, ts, mr, nonce, self.difficulty,
            )
            if BAB64BlockMiner.meets_difficulty(bh, self.difficulty):
                block = BAB64Block(
                    index=height,
                    previous_hash=prev_hash,
                    timestamp=ts,
                    transactions=transactions,
                    merkle_root_hash=mr,
                    nonce=nonce,
                    difficulty=self.difficulty,
                    block_hash=bh,
                )
                break

        if block is None:
            return None

        # Apply all transactions to UTXO set
        for tx in transactions:
            if not tx.is_coinbase:
                for inp in tx.inputs:
                    self.utxo_set.spend(inp.prev_tx_hash, inp.prev_index)
            self.utxo_set.add_outputs(tx)

        self.chain.append(block)
        # Remove only selected transactions from mempool
        selected_hashes = {tx.tx_hash for tx in selected}
        self.mempool = [tx for tx in self.mempool if tx.tx_hash not in selected_hashes]
        return block

    def get_balance(self, address: str) -> int:
        """Query balance from the UTXO set."""
        return self.utxo_set.balance(address)

    def total_supply(self) -> int:
        """Sum of all coinbase rewards across all blocks."""
        total = 0
        for block in self.chain:
            for tx in block.transactions:
                if tx.is_coinbase:
                    total += sum(out.amount for out in tx.outputs)
        return total

    def verify_supply_cap(self) -> bool:
        """Ensure total supply <= MAX_SUPPLY."""
        return self.total_supply() <= MAX_SUPPLY

    def median_time_past(self, chain: List[BAB64Block] = None) -> float:
        """Median timestamp of the last 11 blocks (or fewer if chain is short)."""
        c = chain if chain is not None else self.chain
        if not c:
            return 0.0
        recent = c[-11:]
        timestamps = sorted(b.timestamp for b in recent)
        return statistics.median(timestamps)

    def validate_block(self, block: BAB64Block,
                       expected_prev_hash: str) -> Tuple[bool, str]:
        """Validate a single block (basic structural checks)."""
        if block.previous_hash != expected_prev_hash:
            return False, "Previous hash mismatch"

        recomputed = BAB64BlockMiner.compute_block_hash(
            block.index, block.previous_hash, block.timestamp,
            block.merkle_root_hash, block.nonce, block.difficulty,
        )
        if recomputed != block.block_hash:
            return False, "Block hash mismatch"

        if not BAB64BlockMiner.meets_difficulty(block.block_hash,
                                                 block.difficulty):
            return False, "Block does not meet difficulty"

        tx_hashes = [tx.tx_hash for tx in block.transactions]
        expected_mr = merkle_root(tx_hashes)
        if expected_mr != block.merkle_root_hash:
            return False, "Merkle root mismatch"

        if not block.transactions:
            return False, "Block has no transactions"
        if not block.transactions[0].is_coinbase:
            return False, "First transaction must be coinbase"

        for tx in block.transactions[1:]:
            if tx.is_coinbase:
                return False, "Only first transaction can be coinbase"

        return True, ""

    def validate_block_full(self, block: BAB64Block,
                            expected_prev_hash: str,
                            expected_height: int,
                            preceding_chain: List[BAB64Block] = None,
                            utxo_set: UTXOSet = None,
                            current_time: float = None) -> Tuple[bool, str]:
        """
        Full block validation — all 10 consensus rules.

        1. Block hash meets difficulty target
        2. Previous hash matches chain tip
        3. Timestamp > median of last 11 blocks
        4. Timestamp < current time + 2 hours
        5. First tx is coinbase with correct reward
        6. Coinbase reward = block_reward(height) + sum(fees)
        7. All non-coinbase transactions valid (UTXO rules)
        8. Merkle root matches transaction hashes
        9. No duplicate transactions within block
        10. Block index matches expected height
        """
        now = current_time if current_time is not None else time.time()

        # Rule 10: Block index matches expected height
        if block.index != expected_height:
            return False, f"Wrong block height: expected {expected_height}, got {block.index}"

        # Rule 2: Previous hash matches
        if block.previous_hash != expected_prev_hash:
            return False, "Previous hash mismatch"

        # Rule 1: Block hash meets difficulty
        recomputed = BAB64BlockMiner.compute_block_hash(
            block.index, block.previous_hash, block.timestamp,
            block.merkle_root_hash, block.nonce, block.difficulty,
        )
        if recomputed != block.block_hash:
            return False, "Block hash mismatch"
        if not BAB64BlockMiner.meets_difficulty(block.block_hash, block.difficulty):
            return False, "Block does not meet difficulty"

        # Rule 3: Timestamp > median of last 11 blocks
        if preceding_chain:
            mtp = self.median_time_past(preceding_chain)
            if block.timestamp <= mtp:
                return False, "Timestamp not after median of last 11 blocks"

        # Rule 4: Timestamp < current time + 2 hours
        if block.timestamp > now + MAX_FUTURE_BLOCK_TIME:
            return False, "Block timestamp too far in the future"

        # Rule 8: Merkle root
        tx_hashes = [tx.tx_hash for tx in block.transactions]
        expected_mr = merkle_root(tx_hashes)
        if expected_mr != block.merkle_root_hash:
            return False, "Merkle root mismatch"

        # Rule 9: No duplicate transactions
        seen_tx = set()
        for tx in block.transactions:
            if tx.tx_hash in seen_tx:
                return False, "Duplicate transaction in block"
            seen_tx.add(tx.tx_hash)

        # Block size limits
        if len(block.transactions) > MAX_BLOCK_TRANSACTIONS:
            return False, (
                f"Block exceeds transaction count limit: "
                f"{len(block.transactions)} > {MAX_BLOCK_TRANSACTIONS}"
            )

        total_block_size = sum(FeePolicy.tx_size(tx) for tx in block.transactions)
        if total_block_size > MAX_BLOCK_SIZE:
            return False, (
                f"Block exceeds size limit: {total_block_size} > {MAX_BLOCK_SIZE}"
            )

        # Rules 5, 6: Coinbase checks
        if not block.transactions:
            return False, "Block has no transactions"
        if not block.transactions[0].is_coinbase:
            return False, "First transaction must be coinbase"

        # Calculate fees from non-coinbase txs
        us = utxo_set or self.utxo_set
        total_fees = 0
        for tx in block.transactions[1:]:
            if tx.is_coinbase:
                return False, "Only first transaction can be coinbase"
            total_fees += tx.fee(us)

        expected_reward = block_reward(expected_height) + total_fees
        coinbase = block.transactions[0]
        actual_reward = sum(out.amount for out in coinbase.outputs)
        if actual_reward != expected_reward:
            return False, (
                f"Coinbase reward mismatch: expected {expected_reward}, "
                f"got {actual_reward}"
            )

        # Rule 7: Validate non-coinbase transactions
        for tx in block.transactions[1:]:
            valid, err = us.validate_transaction(tx, current_height=expected_height)
            if not valid:
                return False, f"Invalid transaction: {err}"

        return True, ""

    def validate_chain(self) -> Tuple[bool, str]:
        """Verify the entire chain."""
        for i, block in enumerate(self.chain):
            expected_prev = "0" * 64 if i == 0 else self.chain[i - 1].block_hash
            valid, err = self.validate_block(block, expected_prev)
            if not valid:
                return False, f"Block {i}: {err}"
        return True, ""

    def difficulty_adjustment(self) -> int:
        """
        Adjust difficulty every ADJUSTMENT_INTERVAL blocks.

        new_diff = old_diff * (target_time / actual_time)
        Clamped to [old/4, old*4]. Minimum difficulty: 1.
        """
        if len(self.chain) < ADJUSTMENT_INTERVAL:
            return self.difficulty

        if len(self.chain) % ADJUSTMENT_INTERVAL != 0:
            return self.difficulty

        interval_blocks = self.chain[-ADJUSTMENT_INTERVAL:]
        actual_time = interval_blocks[-1].timestamp - interval_blocks[0].timestamp

        expected_time = TARGET_BLOCK_TIME * (ADJUSTMENT_INTERVAL - 1)

        if actual_time <= 0:
            actual_time = 1.0  # prevent division by zero

        ratio = expected_time / actual_time
        # Clamp ratio to [0.25, 4.0]
        ratio = max(0.25, min(4.0, ratio))

        new_diff = max(1, int(self.difficulty * ratio))
        # Clamp to [old/4, old*4]
        new_diff = max(max(1, self.difficulty // 4), min(self.difficulty * 4, new_diff))
        # Ensure minimum of 1
        new_diff = max(1, new_diff)

        self.difficulty = new_diff
        return new_diff

    def handle_fork(self, competing_chain: List[BAB64Block]) -> bool:
        """
        If competing chain has more cumulative work and is valid,
        switch to it. Revert UTXO set and reapply.

        Returns True if we switched to the competing chain.
        """
        selected = ChainSelector.select_chain(
            self.chain, competing_chain, self.utxo_set
        )
        if selected is competing_chain:
            # Rebuild UTXO set from the new chain
            self.utxo_set = UTXOSet()
            for block in competing_chain:
                for tx in block.transactions:
                    if tx.is_coinbase:
                        self.utxo_set.add_outputs(tx)
                    else:
                        self.utxo_set.apply_transaction(tx)
            self.chain = list(competing_chain)
            if competing_chain:
                self.difficulty = competing_chain[-1].difficulty
            return True
        return False

    @staticmethod
    def verify_header(header: BlockHeader) -> Tuple[bool, str]:
        """Verify a block header without the full block body."""
        recomputed = BAB64BlockMiner.compute_block_hash(
            header.index, header.previous_hash, header.timestamp,
            header.merkle_root, header.nonce, header.difficulty,
        )
        if recomputed != header.block_hash:
            return False, "Header hash mismatch"
        if not BAB64BlockMiner.meets_difficulty(header.block_hash, header.difficulty):
            return False, "Header does not meet difficulty"
        return True, ""


# =============================================================================
# CONVENIENCE — Build and send transactions
# =============================================================================

def build_transaction(sender: BAB64Identity, recipient,
                      amount: int, utxo_set: UTXOSet,
                      fee: int = 0) -> Optional[BAB64CashTransaction]:
    """
    Build, sign, and finalize a transaction.

    recipient: BAB64Identity or str (address hex).
    If BAB64Identity, outputs are hashlocked to the recipient.
    """
    # Resolve recipient
    if isinstance(recipient, BAB64Identity):
        recipient_address = recipient.address_hex
        recipient_image = recipient._image_bytes
    else:
        recipient_address = recipient
        recipient_image = None

    sender_addr = sender.address_hex
    utxos = utxo_set.get_utxos_for(sender_addr)

    # Collect inputs
    collected = 0
    inputs = []
    selected_utxos = []
    for utxo in utxos:
        inputs.append(TxInput(
            prev_tx_hash=utxo.tx_hash,
            prev_index=utxo.index,
        ))
        selected_utxos.append(utxo)
        collected += utxo.amount
        if collected >= amount + fee:
            break

    if collected < amount + fee:
        return None  # Insufficient funds

    # Build outputs
    outputs = []

    # Recipient output
    r_lock_hash, r_lock_nonce = ("", b"")
    if recipient_image:
        r_lock_hash, r_lock_nonce = compute_lock(recipient_image)
    outputs.append(TxOutput(
        recipient=recipient_address, amount=amount,
        tx_hash="", index=0,
        lock_hash=r_lock_hash, lock_nonce=r_lock_nonce,
    ))

    # Change output
    change = collected - amount - fee
    if change > 0:
        c_lock_hash, c_lock_nonce = compute_lock(sender._image_bytes)
        outputs.append(TxOutput(
            recipient=sender_addr, amount=change,
            tx_hash="", index=1,
            lock_hash=c_lock_hash, lock_nonce=c_lock_nonce,
        ))

    tx = BAB64CashTransaction(inputs=inputs, outputs=outputs)

    # Sign all inputs (with owner proofs)
    for i, utxo in enumerate(selected_utxos):
        tx.sign_input(i, sender, lock_nonce=utxo.lock_nonce)

    tx.finalize()
    return tx
