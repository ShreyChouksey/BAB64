"""
BAB64 Network — P2P Networking Layer
======================================

Asyncio-based P2P network for BAB64 Cash nodes.

Components:
  1. Message Protocol — JSON over TCP, newline-delimited
  2. Mempool — Fee-sorted transaction buffer
  3. Peer — Connection state tracking
  4. Node — Full P2P node with message handling
  5. Block Propagation — INV/GETDATA relay
  6. Initial Block Download — Sync from peers

Author: Shrey (concept) + Claude (implementation)
"""

import asyncio
import json
import time
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Set, Tuple

from bab64_cash import (
    BAB64Block, BAB64BlockMiner, BAB64Blockchain, BAB64CashTransaction,
    BlockHeader, TxInput, TxOutput, UTXOSet,
    block_reward, build_transaction, merkle_root,
)
from bab64_identity import BAB64Identity


# =============================================================================
# COMPONENT 1 — MESSAGE PROTOCOL
# =============================================================================

# Message types
VERSION = "VERSION"
VERACK = "VERACK"
INV = "INV"
GETDATA = "GETDATA"
BLOCK = "BLOCK"
TX = "TX"
GETBLOCKS = "GETBLOCKS"
HEADERS = "HEADERS"
PING = "PING"
PONG = "PONG"
ADDR = "ADDR"

# Inventory types
INV_BLOCK = "block"
INV_TX = "tx"


@dataclass
class P2PMessage:
    """A P2P message sent between nodes."""
    msg_type: str
    payload: dict
    sender: str
    timestamp: float

    def serialize(self) -> bytes:
        """Serialize to newline-delimited JSON."""
        data = {
            "msg_type": self.msg_type,
            "payload": self.payload,
            "sender": self.sender,
            "timestamp": self.timestamp,
        }
        return json.dumps(data).encode() + b"\n"

    @staticmethod
    def deserialize(data: bytes) -> 'P2PMessage':
        """Deserialize from JSON bytes."""
        obj = json.loads(data.decode().strip())
        return P2PMessage(
            msg_type=obj["msg_type"],
            payload=obj["payload"],
            sender=obj["sender"],
            timestamp=obj["timestamp"],
        )


# =============================================================================
# COMPONENT 2 — MEMPOOL
# =============================================================================

class Mempool:
    """
    Transaction mempool — holds unconfirmed transactions sorted by fee.
    """

    def __init__(self, utxo_set: UTXOSet = None):
        self._transactions: Dict[str, BAB64CashTransaction] = {}
        self._utxo_set = utxo_set

    def set_utxo_set(self, utxo_set: UTXOSet):
        self._utxo_set = utxo_set

    def add(self, tx: BAB64CashTransaction) -> bool:
        """Validate and add transaction. Reject duplicates."""
        if tx.tx_hash in self._transactions:
            return False
        if self._utxo_set:
            valid, _ = self._utxo_set.validate_transaction(tx)
            if not valid:
                return False
        self._transactions[tx.tx_hash] = tx
        return True

    def remove(self, tx_hash: str):
        """Remove a transaction by hash."""
        self._transactions.pop(tx_hash, None)

    def get_by_fee(self) -> List[BAB64CashTransaction]:
        """Return all transactions sorted by fee descending."""
        if not self._utxo_set:
            return list(self._transactions.values())
        txs = list(self._transactions.values())
        txs.sort(key=lambda t: t.fee(self._utxo_set), reverse=True)
        return txs

    def size(self) -> int:
        return len(self._transactions)

    def contains(self, tx_hash: str) -> bool:
        return tx_hash in self._transactions

    def clear_confirmed(self, block: BAB64Block):
        """Remove transactions that were included in a block."""
        for tx in block.transactions:
            self._transactions.pop(tx.tx_hash, None)

    def get_transactions(self, max_count: int = None) -> List[BAB64CashTransaction]:
        """Get top transactions by fee, up to max_count."""
        txs = self.get_by_fee()
        if max_count is not None:
            txs = txs[:max_count]
        return txs


# =============================================================================
# COMPONENT 3 — PEER
# =============================================================================

CONNECTING = "CONNECTING"
CONNECTED = "CONNECTED"
DISCONNECTED = "DISCONNECTED"


@dataclass
class Peer:
    """Represents a remote peer."""
    host: str
    port: int
    node_id: str
    best_height: int = 0
    last_seen: float = 0.0
    state: str = CONNECTING
    reader: asyncio.StreamReader = field(default=None, repr=False)
    writer: asyncio.StreamWriter = field(default=None, repr=False)
    version_received: bool = False
    verack_received: bool = False

    @property
    def address(self) -> str:
        return f"{self.host}:{self.port}"

    async def send(self, message: P2PMessage):
        """Send a message to this peer."""
        if self.writer and not self.writer.is_closing():
            self.writer.write(message.serialize())
            await self.writer.drain()

    def disconnect(self):
        """Close the connection."""
        self.state = DISCONNECTED
        if self.writer and not self.writer.is_closing():
            self.writer.close()


# =============================================================================
# COMPONENT 4 — NODE
# =============================================================================

def _serialize_block(block: BAB64Block) -> dict:
    """Serialize a block to a JSON-compatible dict."""
    txs = []
    for tx in block.transactions:
        tx_dict = {
            "tx_hash": tx.tx_hash,
            "is_coinbase": tx.is_coinbase,
            "coinbase_height": tx.coinbase_height,
            "inputs": [],
            "outputs": [],
        }
        for inp in tx.inputs:
            tx_dict["inputs"].append({
                "prev_tx_hash": inp.prev_tx_hash,
                "prev_index": inp.prev_index,
                "signature": [b.hex() if isinstance(b, bytes) else b for b in inp.signature],
                "verification_key": [b.hex() if isinstance(b, bytes) else b for b in inp.verification_key],
                "owner_proof": inp.owner_proof.hex() if inp.owner_proof else "",
            })
        for out in tx.outputs:
            tx_dict["outputs"].append({
                "recipient": out.recipient,
                "amount": out.amount,
                "tx_hash": out.tx_hash,
                "index": out.index,
                "lock_hash": out.lock_hash,
                "lock_nonce": out.lock_nonce.hex() if out.lock_nonce else "",
            })
        txs.append(tx_dict)

    return {
        "index": block.index,
        "previous_hash": block.previous_hash,
        "timestamp": block.timestamp,
        "merkle_root_hash": block.merkle_root_hash,
        "nonce": block.nonce,
        "difficulty": block.difficulty,
        "block_hash": block.block_hash,
        "transactions": txs,
    }


def _deserialize_block(data: dict) -> BAB64Block:
    """Deserialize a block from a JSON-compatible dict."""
    txs = []
    for tx_dict in data["transactions"]:
        inputs = []
        for inp_dict in tx_dict["inputs"]:
            sig = [bytes.fromhex(s) if isinstance(s, str) else s for s in inp_dict["signature"]]
            vk = [bytes.fromhex(k) if isinstance(k, str) else k for k in inp_dict["verification_key"]]
            proof = bytes.fromhex(inp_dict["owner_proof"]) if inp_dict["owner_proof"] else b""
            inputs.append(TxInput(
                prev_tx_hash=inp_dict["prev_tx_hash"],
                prev_index=inp_dict["prev_index"],
                signature=sig,
                verification_key=vk,
                owner_proof=proof,
            ))
        outputs = []
        for out_dict in tx_dict["outputs"]:
            nonce_hex = out_dict.get("lock_nonce", "")
            outputs.append(TxOutput(
                recipient=out_dict["recipient"],
                amount=out_dict["amount"],
                tx_hash=out_dict["tx_hash"],
                index=out_dict["index"],
                lock_hash=out_dict.get("lock_hash", ""),
                lock_nonce=bytes.fromhex(nonce_hex) if nonce_hex else b"",
            ))
        tx = BAB64CashTransaction(
            inputs=inputs,
            outputs=outputs,
            is_coinbase=tx_dict["is_coinbase"],
            coinbase_height=tx_dict["coinbase_height"],
        )
        tx.tx_hash = tx_dict["tx_hash"]
        txs.append(tx)

    return BAB64Block(
        index=data["index"],
        previous_hash=data["previous_hash"],
        timestamp=data["timestamp"],
        transactions=txs,
        merkle_root_hash=data["merkle_root_hash"],
        nonce=data["nonce"],
        difficulty=data["difficulty"],
        block_hash=data["block_hash"],
    )


def _serialize_tx(tx: BAB64CashTransaction) -> dict:
    """Serialize a transaction to a JSON-compatible dict."""
    tx_dict = {
        "tx_hash": tx.tx_hash,
        "is_coinbase": tx.is_coinbase,
        "coinbase_height": tx.coinbase_height,
        "inputs": [],
        "outputs": [],
    }
    for inp in tx.inputs:
        tx_dict["inputs"].append({
            "prev_tx_hash": inp.prev_tx_hash,
            "prev_index": inp.prev_index,
            "signature": [b.hex() if isinstance(b, bytes) else b for b in inp.signature],
            "verification_key": [b.hex() if isinstance(b, bytes) else b for b in inp.verification_key],
            "owner_proof": inp.owner_proof.hex() if inp.owner_proof else "",
        })
    for out in tx.outputs:
        tx_dict["outputs"].append({
            "recipient": out.recipient,
            "amount": out.amount,
            "tx_hash": out.tx_hash,
            "index": out.index,
            "lock_hash": out.lock_hash,
            "lock_nonce": out.lock_nonce.hex() if out.lock_nonce else "",
        })
    return tx_dict


def _deserialize_tx(data: dict) -> BAB64CashTransaction:
    """Deserialize a transaction from a JSON-compatible dict."""
    inputs = []
    for inp_dict in data["inputs"]:
        sig = [bytes.fromhex(s) if isinstance(s, str) else s for s in inp_dict["signature"]]
        vk = [bytes.fromhex(k) if isinstance(k, str) else k for k in inp_dict["verification_key"]]
        proof = bytes.fromhex(inp_dict["owner_proof"]) if inp_dict["owner_proof"] else b""
        inputs.append(TxInput(
            prev_tx_hash=inp_dict["prev_tx_hash"],
            prev_index=inp_dict["prev_index"],
            signature=sig,
            verification_key=vk,
            owner_proof=proof,
        ))
    outputs = []
    for out_dict in data["outputs"]:
        nonce_hex = out_dict.get("lock_nonce", "")
        outputs.append(TxOutput(
            recipient=out_dict["recipient"],
            amount=out_dict["amount"],
            tx_hash=out_dict["tx_hash"],
            index=out_dict["index"],
            lock_hash=out_dict.get("lock_hash", ""),
            lock_nonce=bytes.fromhex(nonce_hex) if nonce_hex else b"",
        ))
    tx = BAB64CashTransaction(
        inputs=inputs,
        outputs=outputs,
        is_coinbase=data["is_coinbase"],
        coinbase_height=data["coinbase_height"],
    )
    tx.tx_hash = data["tx_hash"]
    return tx


def _serialize_header(header: BlockHeader) -> dict:
    """Serialize a block header."""
    return {
        "index": header.index,
        "previous_hash": header.previous_hash,
        "timestamp": header.timestamp,
        "merkle_root": header.merkle_root,
        "difficulty": header.difficulty,
        "nonce": header.nonce,
        "block_hash": header.block_hash,
    }


def _deserialize_header(data: dict) -> BlockHeader:
    """Deserialize a block header."""
    return BlockHeader(
        index=data["index"],
        previous_hash=data["previous_hash"],
        timestamp=data["timestamp"],
        merkle_root=data["merkle_root"],
        difficulty=data["difficulty"],
        nonce=data["nonce"],
        block_hash=data["block_hash"],
    )


class BAB64Node:
    """
    A full P2P node in the BAB64 network.

    Handles:
      - TCP server for inbound connections
      - Outbound connections to peers
      - Message routing and handling
      - Block/transaction relay
      - Initial block download
      - Mining and broadcasting
    """

    def __init__(self, host: str, port: int,
                 blockchain: BAB64Blockchain,
                 identity: BAB64Identity = None,
                 node_id: str = None):
        self.host = host
        self.port = port
        self.blockchain = blockchain
        self.identity = identity
        self.node_id = node_id or f"{host}:{port}"
        self.mempool = Mempool(blockchain.utxo_set)
        self.peers: Dict[str, Peer] = {}
        self.server: Optional[asyncio.Server] = None
        self.known_blocks: Set[str] = set()
        self.known_txs: Set[str] = set()
        self._running = False
        self._on_block_received: Optional[Callable] = None
        self._on_tx_received: Optional[Callable] = None
        self._pending_sync: Dict[str, asyncio.Event] = {}
        self._requested_blocks: Dict[str, asyncio.Future] = {}
        self._sync_complete = asyncio.Event()
        self._known_addresses: List[dict] = []

        # Index existing blocks
        for block in blockchain.chain:
            self.known_blocks.add(block.block_hash)

    # -------------------------------------------------------------------------
    # Server lifecycle
    # -------------------------------------------------------------------------

    async def start(self):
        """Start listening for inbound connections."""
        self.server = await asyncio.start_server(
            self._handle_inbound, self.host, self.port
        )
        self._running = True

    async def stop(self):
        """Stop the node and disconnect all peers."""
        self._running = False
        for peer in list(self.peers.values()):
            peer.disconnect()
        self.peers.clear()
        if self.server:
            self.server.close()
            await self.server.wait_closed()

    # -------------------------------------------------------------------------
    # Connection management
    # -------------------------------------------------------------------------

    async def _handle_inbound(self, reader: asyncio.StreamReader,
                              writer: asyncio.StreamWriter):
        """Handle an inbound TCP connection."""
        addr = writer.get_extra_info('peername')
        temp_id = f"{addr[0]}:{addr[1]}"
        peer = Peer(
            host=addr[0], port=addr[1], node_id=temp_id,
            reader=reader, writer=writer, state=CONNECTED,
            last_seen=time.time(),
        )
        self.peers[temp_id] = peer
        await self._read_messages(peer)

    async def connect_to_peer(self, host: str, port: int) -> Optional[Peer]:
        """Initiate an outbound connection and perform handshake."""
        try:
            reader, writer = await asyncio.open_connection(host, port)
        except (ConnectionRefusedError, OSError):
            return None

        peer_id = f"{host}:{port}"
        peer = Peer(
            host=host, port=port, node_id=peer_id,
            reader=reader, writer=writer, state=CONNECTED,
            last_seen=time.time(),
        )
        self.peers[peer_id] = peer

        # Send VERSION
        await self._send_version(peer)

        # Start reading messages in background
        asyncio.ensure_future(self._read_messages(peer))
        return peer

    async def _read_messages(self, peer: Peer):
        """Read and dispatch messages from a peer."""
        try:
            while self._running and peer.state == CONNECTED:
                line = await asyncio.wait_for(
                    peer.reader.readline(), timeout=30.0
                )
                if not line:
                    break
                try:
                    msg = P2PMessage.deserialize(line)
                    await self.handle_message(peer, msg)
                except (json.JSONDecodeError, KeyError):
                    continue
        except (asyncio.TimeoutError, ConnectionError, asyncio.CancelledError):
            pass
        finally:
            peer.state = DISCONNECTED
            # Clean up peer reference (may have been re-keyed)
            for pid, p in list(self.peers.items()):
                if p is peer:
                    del self.peers[pid]
                    break

    # -------------------------------------------------------------------------
    # Message sending helpers
    # -------------------------------------------------------------------------

    def _make_message(self, msg_type: str, payload: dict) -> P2PMessage:
        return P2PMessage(
            msg_type=msg_type,
            payload=payload,
            sender=self.node_id,
            timestamp=time.time(),
        )

    async def _send_version(self, peer: Peer):
        msg = self._make_message(VERSION, {
            "version": 1,
            "best_height": len(self.blockchain.chain) - 1,
            "node_id": self.node_id,
            "address": f"{self.host}:{self.port}",
        })
        await peer.send(msg)

    async def broadcast(self, message: P2PMessage, exclude: str = None):
        """Send a message to all connected peers."""
        for pid, peer in list(self.peers.items()):
            if peer.state == CONNECTED and pid != exclude:
                try:
                    await peer.send(message)
                except (ConnectionError, OSError):
                    peer.state = DISCONNECTED

    # -------------------------------------------------------------------------
    # Message routing
    # -------------------------------------------------------------------------

    async def handle_message(self, peer: Peer, message: P2PMessage):
        """Route a message to the appropriate handler."""
        peer.last_seen = time.time()

        handlers = {
            VERSION: self._handle_version,
            VERACK: self._handle_verack,
            INV: self._handle_inv,
            GETDATA: self._handle_getdata,
            BLOCK: self._handle_block,
            TX: self._handle_tx,
            GETBLOCKS: self._handle_getblocks,
            HEADERS: self._handle_headers,
            PING: self._handle_ping,
            PONG: self._handle_pong,
            ADDR: self._handle_addr,
        }

        handler = handlers.get(message.msg_type)
        if handler:
            await handler(peer, message.payload)

    # -------------------------------------------------------------------------
    # Message handlers
    # -------------------------------------------------------------------------

    async def _handle_version(self, peer: Peer, payload: dict):
        """Handle VERSION: update peer info, send VERACK + our VERSION."""
        peer.best_height = payload.get("best_height", 0)
        remote_id = payload.get("node_id", peer.node_id)

        # Re-key peer if we learned their real node_id
        if remote_id != peer.node_id:
            old_id = peer.node_id
            peer.node_id = remote_id
            if old_id in self.peers:
                del self.peers[old_id]
            self.peers[remote_id] = peer

        peer.version_received = True

        # Send VERACK
        verack = self._make_message(VERACK, {})
        await peer.send(verack)

        # Send our VERSION if we haven't yet (inbound connection)
        if not peer.verack_received:
            await self._send_version(peer)

    async def _handle_verack(self, peer: Peer, payload: dict):
        """Handle VERACK: handshake complete."""
        peer.verack_received = True
        peer.state = CONNECTED

        # Signal sync if waiting
        event = self._pending_sync.get(peer.node_id)
        if event:
            event.set()

    async def _handle_inv(self, peer: Peer, payload: dict):
        """Handle INV: check if we need the announced items."""
        items = payload.get("items", [])
        needed = []
        for item in items:
            inv_type = item["type"]
            inv_hash = item["hash"]
            if inv_type == INV_BLOCK and inv_hash not in self.known_blocks:
                needed.append(item)
            elif inv_type == INV_TX and inv_hash not in self.known_txs:
                needed.append(item)

        if needed:
            msg = self._make_message(GETDATA, {"items": needed})
            await peer.send(msg)

    async def _handle_getdata(self, peer: Peer, payload: dict):
        """Handle GETDATA: send requested blocks/transactions."""
        items = payload.get("items", [])
        for item in items:
            inv_type = item["type"]
            inv_hash = item["hash"]

            if inv_type == INV_BLOCK:
                block = self._find_block(inv_hash)
                if block:
                    msg = self._make_message(BLOCK, {
                        "block": _serialize_block(block),
                    })
                    await peer.send(msg)

            elif inv_type == INV_TX:
                if self.mempool.contains(inv_hash):
                    txs = self.mempool.get_by_fee()
                    for tx in txs:
                        if tx.tx_hash == inv_hash:
                            msg = self._make_message(TX, {
                                "transaction": _serialize_tx(tx),
                            })
                            await peer.send(msg)
                            break

    async def _handle_block(self, peer: Peer, payload: dict):
        """Handle BLOCK: validate and add to chain."""
        block_data = payload.get("block")
        if not block_data:
            return

        block = _deserialize_block(block_data)

        if block.block_hash in self.known_blocks:
            # Resolve pending request if any
            fut = self._requested_blocks.pop(block.block_hash, None)
            if fut and not fut.done():
                fut.set_result(block)
            return

        # Validate the block
        expected_height = len(self.blockchain.chain)
        if block.index != expected_height:
            # Might be for IBD — check requested blocks
            fut = self._requested_blocks.pop(block.block_hash, None)
            if fut and not fut.done():
                fut.set_result(block)
            return

        expected_prev = (
            self.blockchain.chain[-1].block_hash
            if self.blockchain.chain else "0" * 64
        )

        valid, err = self.blockchain.validate_block(block, expected_prev)
        if not valid:
            return

        # Apply block
        for tx in block.transactions:
            if tx.is_coinbase:
                self.blockchain.utxo_set.add_outputs(tx)
            else:
                valid_tx, _ = self.blockchain.utxo_set.validate_transaction(tx)
                if not valid_tx:
                    return
                self.blockchain.utxo_set.apply_transaction(tx)

        self.blockchain.chain.append(block)
        self.known_blocks.add(block.block_hash)
        self.mempool.clear_confirmed(block)

        # Relay to other peers
        inv_msg = self._make_message(INV, {
            "items": [{"type": INV_BLOCK, "hash": block.block_hash}],
        })
        await self.broadcast(inv_msg, exclude=peer.node_id)

        if self._on_block_received:
            self._on_block_received(block)

        # Resolve pending request
        fut = self._requested_blocks.pop(block.block_hash, None)
        if fut and not fut.done():
            fut.set_result(block)

    async def _handle_tx(self, peer: Peer, payload: dict):
        """Handle TX: validate and add to mempool."""
        tx_data = payload.get("transaction")
        if not tx_data:
            return

        tx = _deserialize_tx(tx_data)

        if tx.tx_hash in self.known_txs:
            return

        # Validate against UTXO set
        valid, err = self.blockchain.utxo_set.validate_transaction(tx)
        if not valid:
            return

        added = self.mempool.add(tx)
        if added:
            self.known_txs.add(tx.tx_hash)

            # Relay to other peers
            inv_msg = self._make_message(INV, {
                "items": [{"type": INV_TX, "hash": tx.tx_hash}],
            })
            await self.broadcast(inv_msg, exclude=peer.node_id)

            if self._on_tx_received:
                self._on_tx_received(tx)

    async def _handle_getblocks(self, peer: Peer, payload: dict):
        """Handle GETBLOCKS: send block headers for the requested range."""
        start_height = payload.get("start_height", 0)
        end_height = payload.get("end_height", len(self.blockchain.chain))
        end_height = min(end_height, len(self.blockchain.chain))

        headers = []
        for i in range(start_height, end_height):
            block = self.blockchain.chain[i]
            headers.append(_serialize_header(block.header()))

        msg = self._make_message(HEADERS, {"headers": headers})
        await peer.send(msg)

    async def _handle_headers(self, peer: Peer, payload: dict):
        """Handle HEADERS: request full blocks for headers we don't have."""
        headers_data = payload.get("headers", [])
        needed = []
        for hdr_data in headers_data:
            header = _deserialize_header(hdr_data)
            if header.block_hash not in self.known_blocks:
                needed.append({"type": INV_BLOCK, "hash": header.block_hash})

        if needed:
            msg = self._make_message(GETDATA, {"items": needed})
            await peer.send(msg)

    async def _handle_ping(self, peer: Peer, payload: dict):
        """Handle PING: respond with PONG."""
        pong = self._make_message(PONG, {
            "nonce": payload.get("nonce", 0),
        })
        await peer.send(pong)

    async def _handle_pong(self, peer: Peer, payload: dict):
        """Handle PONG: update last_seen."""
        peer.last_seen = time.time()

    async def _handle_addr(self, peer: Peer, payload: dict):
        """Handle ADDR: learn about new peer addresses."""
        addresses = payload.get("addresses", [])
        for addr in addresses:
            if addr not in self._known_addresses:
                self._known_addresses.append(addr)

    # -------------------------------------------------------------------------
    # Block / TX lookup
    # -------------------------------------------------------------------------

    def _find_block(self, block_hash: str) -> Optional[BAB64Block]:
        """Find a block by hash in our chain."""
        for block in self.blockchain.chain:
            if block.block_hash == block_hash:
                return block
        return None

    # -------------------------------------------------------------------------
    # Sync (Initial Block Download)
    # -------------------------------------------------------------------------

    async def sync_with_peer(self, peer: Peer):
        """Download missing blocks from a peer."""
        my_height = len(self.blockchain.chain)
        if peer.best_height < my_height:
            return

        # Request block hashes
        msg = self._make_message(GETBLOCKS, {
            "start_height": my_height,
            "end_height": peer.best_height + 1,
        })
        await peer.send(msg)

    async def request_block(self, peer: Peer, block_hash: str,
                            timeout: float = 5.0) -> Optional[BAB64Block]:
        """Request a specific block and wait for the response."""
        loop = asyncio.get_event_loop()
        fut = loop.create_future()
        self._requested_blocks[block_hash] = fut

        msg = self._make_message(GETDATA, {
            "items": [{"type": INV_BLOCK, "hash": block_hash}],
        })
        await peer.send(msg)

        try:
            return await asyncio.wait_for(fut, timeout=timeout)
        except asyncio.TimeoutError:
            self._requested_blocks.pop(block_hash, None)
            return None

    # -------------------------------------------------------------------------
    # Mining
    # -------------------------------------------------------------------------

    def mine_next_block(self) -> Optional[BAB64Block]:
        """Collect from mempool, mine a block, return it."""
        # Move mempool transactions to blockchain mempool
        for tx in self.mempool.get_transactions():
            if tx.tx_hash not in {t.tx_hash for t in self.blockchain.mempool}:
                self.blockchain.mempool.append(tx)

        block = self.blockchain.mine_block(self.identity)
        if block:
            self.known_blocks.add(block.block_hash)
            self.mempool.clear_confirmed(block)
            # Update mempool's utxo reference
            self.mempool.set_utxo_set(self.blockchain.utxo_set)
        return block

    async def mine_and_broadcast(self) -> Optional[BAB64Block]:
        """Mine a block and broadcast it to all peers."""
        block = self.mine_next_block()
        if block:
            inv_msg = self._make_message(INV, {
                "items": [{"type": INV_BLOCK, "hash": block.block_hash}],
            })
            await self.broadcast(inv_msg)
        return block

    # -------------------------------------------------------------------------
    # Transaction submission
    # -------------------------------------------------------------------------

    async def submit_transaction(self, tx: BAB64CashTransaction) -> bool:
        """Add a transaction to mempool and broadcast."""
        added = self.mempool.add(tx)
        if added:
            self.known_txs.add(tx.tx_hash)
            inv_msg = self._make_message(INV, {
                "items": [{"type": INV_TX, "hash": tx.tx_hash}],
            })
            await self.broadcast(inv_msg)
        return added

    # -------------------------------------------------------------------------
    # Peer discovery
    # -------------------------------------------------------------------------

    async def share_addresses(self):
        """Share known peer addresses with all connected peers."""
        addresses = []
        for pid, peer in self.peers.items():
            addresses.append({
                "host": peer.host,
                "port": peer.port,
                "node_id": peer.node_id,
            })
        if addresses:
            msg = self._make_message(ADDR, {"addresses": addresses})
            await self.broadcast(msg)

    async def ping_peer(self, peer: Peer, nonce: int = 0):
        """Send a PING to a peer."""
        msg = self._make_message(PING, {"nonce": nonce})
        await peer.send(msg)
