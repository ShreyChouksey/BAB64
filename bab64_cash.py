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

import hashlib
import os
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

from bab64_identity import BAB64Identity, LamportKeyPair


# =============================================================================
# CONSTANTS
# =============================================================================

COIN = 100_000_000          # 1 BAB64 = 100,000,000 satoshis
INITIAL_REWARD = 50 * COIN  # 50 BAB64 per block
HALVING_INTERVAL = 210_000  # blocks between halvings
MAX_SUPPLY = 21_000_000 * COIN


# =============================================================================
# HASHLOCK — Ownership Proofs
# =============================================================================

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


@dataclass
class TxInput:
    """A reference to a UTXO being spent."""
    prev_tx_hash: str   # points to a TxOutput
    prev_index: int     # which output of that transaction
    signature: List[bytes] = field(default_factory=list)
    verification_key: List[bytes] = field(default_factory=list)
    owner_proof: bytes = field(default_factory=bytes)  # proves image knowledge


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
        Sign a specific input with the owner's identity.

        If lock_nonce is provided, also computes the owner_proof
        needed to pass the hashlock check.
        """
        payload = self.signing_payload()
        sig = identity.sign(payload)
        self.inputs[input_index].signature = sig.raw
        self.inputs[input_index].verification_key = sig.verification_key
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

    def validate_transaction(self, tx: BAB64CashTransaction) -> Tuple[bool, str]:
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

            # Check hashlock (ownership proof)
            if utxo.lock_hash:
                if not inp.owner_proof:
                    return False, "Missing owner proof"
                if not verify_lock(inp.owner_proof, utxo.lock_hash):
                    return False, "Invalid owner proof — not the UTXO owner"

            # Verify signature
            if not inp.signature or not inp.verification_key:
                return False, "Input missing signature"

            payload = tx.signing_payload()
            if not LamportKeyPair.verify(payload, inp.signature,
                                         inp.verification_key):
                return False, "Invalid signature"

            input_total += utxo.amount

        # Check value conservation
        output_total = sum(out.amount for out in tx.outputs)
        if input_total < output_total:
            return False, (
                f"Insufficient funds: inputs={input_total}, "
                f"outputs={output_total}"
            )

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


class BAB64BlockMiner:
    """Mines blocks by finding nonces that meet the difficulty target."""

    @staticmethod
    def compute_block_hash(index: int, previous_hash: str,
                           timestamp: float, merkle_root_hash: str,
                           nonce: int, difficulty: int) -> str:
        header = (
            f"{index}:{previous_hash}:{timestamp}:"
            f"{merkle_root_hash}:{nonce}:{difficulty}"
        )
        return hashlib.sha256(header.encode()).hexdigest()

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
        """Mine a block by finding a valid nonce."""
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
# COMPONENT 7 — BLOCKCHAIN
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

    def add_transaction_to_mempool(self, tx: BAB64CashTransaction) -> Tuple[bool, str]:
        """Validate and add a transaction to the mempool."""
        valid, err = self.utxo_set.validate_transaction(tx)
        if not valid:
            return False, err
        self.mempool.append(tx)
        return True, ""

    def mine_block(self, miner: BAB64Identity = None) -> Optional[BAB64Block]:
        """Mine a new block with mempool transactions."""
        m = miner or self.miner
        addr = m.address_hex if m else self.miner_address
        img = m._image_bytes if m else None
        height = len(self.chain)
        prev_hash = self.chain[-1].block_hash if self.chain else "0" * 64

        # Calculate total fees
        fees = sum(tx.fee(self.utxo_set) for tx in self.mempool)

        # Coinbase first
        coinbase = BAB64CashTransaction.create_coinbase(
            addr, height, fees, image_bytes=img
        )
        transactions = [coinbase] + list(self.mempool)

        block = BAB64BlockMiner.mine_block(
            index=height,
            previous_hash=prev_hash,
            transactions=transactions,
            difficulty=self.difficulty,
        )
        if block is None:
            return None

        # Apply all transactions to UTXO set
        for tx in transactions:
            if not tx.is_coinbase:
                for inp in tx.inputs:
                    self.utxo_set.spend(inp.prev_tx_hash, inp.prev_index)
            self.utxo_set.add_outputs(tx)

        self.chain.append(block)
        self.mempool.clear()
        return block

    def get_balance(self, address: str) -> int:
        """Query balance from the UTXO set."""
        return self.utxo_set.balance(address)

    def validate_block(self, block: BAB64Block,
                       expected_prev_hash: str) -> Tuple[bool, str]:
        """Validate a single block."""
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

    def validate_chain(self) -> Tuple[bool, str]:
        """Verify the entire chain."""
        for i, block in enumerate(self.chain):
            expected_prev = "0" * 64 if i == 0 else self.chain[i - 1].block_hash
            valid, err = self.validate_block(block, expected_prev)
            if not valid:
                return False, f"Block {i}: {err}"
        return True, ""

    def difficulty_adjustment(self, target_block_time: float = 600.0,
                              adjustment_interval: int = 2016) -> int:
        """
        Adjust difficulty every `adjustment_interval` blocks.
        Target: one block per `target_block_time` seconds.
        """
        if len(self.chain) < adjustment_interval:
            return self.difficulty

        recent = self.chain[-adjustment_interval:]
        elapsed = recent[-1].timestamp - recent[0].timestamp
        expected = target_block_time * (adjustment_interval - 1)

        ratio = elapsed / expected if expected > 0 else 1.0
        ratio = max(0.25, min(4.0, ratio))

        if ratio < 1.0:
            new_diff = self.difficulty + 1
        elif ratio > 1.0:
            new_diff = max(1, self.difficulty - 1)
        else:
            new_diff = self.difficulty

        self.difficulty = new_diff
        return new_diff


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
