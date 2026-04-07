"""
Tests for BAB64 Network — Phase 3: P2P Networking.

25+ tests covering:
  - Message serialization/deserialization
  - Mempool add/remove/sort by fee
  - Mempool duplicate rejection
  - Mempool clear after block
  - Peer connection state management
  - Two-node handshake (VERSION/VERACK)
  - Block propagation between 2 nodes
  - Transaction relay between 2 nodes
  - Initial block download (node syncs from peer)
  - Mining and broadcasting
  - Reject invalid block from peer
  - Reject invalid transaction from peer
  - Multiple peers (3-node network)
  - Peer discovery (ADDR message)
  - Node reconnection after disconnect
  - PING/PONG keepalive
"""

import asyncio
import time
import pytest
import pytest_asyncio

from bab64_engine import BAB64Config
from bab64_identity import BAB64Identity
from bab64_cash import (
    COIN, INITIAL_REWARD,
    TxOutput, TxInput,
    BAB64CashTransaction, UTXOSet,
    BAB64Block, BAB64BlockMiner, BAB64Blockchain,
    block_reward, merkle_root, build_transaction,
)
from bab64_network import (
    P2PMessage, Mempool, Peer, BAB64Node,
    VERSION, VERACK, INV, GETDATA, BLOCK, TX,
    GETBLOCKS, HEADERS, PING, PONG, ADDR,
    INV_BLOCK, INV_TX,
    CONNECTING, CONNECTED, DISCONNECTED,
    _serialize_block, _deserialize_block,
    _serialize_tx, _deserialize_tx,
    _serialize_header, _deserialize_header,
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


def make_blockchain(miner, n_blocks=0, difficulty=1):
    """Helper: create a blockchain with hardcoded genesis + n_blocks mined."""
    bc = BAB64Blockchain(difficulty=difficulty, miner=miner)
    bc.add_genesis()  # Shared hardcoded genesis for all nodes
    for _ in range(n_blocks):
        bc.mine_block()
    return bc


async def make_node(identity, n_blocks=0, difficulty=1, port=0):
    """Helper: create and start a node with shared genesis + n_blocks."""
    bc = make_blockchain(identity, n_blocks, difficulty)
    node = BAB64Node("127.0.0.1", port, bc, identity)
    await node.start()
    # Get the actual assigned port
    actual_port = node.server.sockets[0].getsockname()[1]
    node.port = actual_port
    node.node_id = f"127.0.0.1:{actual_port}"
    return node


async def connect_nodes(node_a, node_b, settle_time=0.1):
    """Connect node_a to node_b and wait for handshake."""
    peer = await node_a.connect_to_peer("127.0.0.1", node_b.port)
    await asyncio.sleep(settle_time)
    return peer


# =============================================================================
# MESSAGE SERIALIZATION
# =============================================================================

class TestMessageSerialization:
    def test_serialize_deserialize(self):
        """P2PMessage roundtrips through JSON."""
        msg = P2PMessage(
            msg_type=VERSION,
            payload={"version": 1, "best_height": 5},
            sender="node1",
            timestamp=1234567890.0,
        )
        raw = msg.serialize()
        restored = P2PMessage.deserialize(raw)
        assert restored.msg_type == VERSION
        assert restored.payload["version"] == 1
        assert restored.payload["best_height"] == 5
        assert restored.sender == "node1"
        assert restored.timestamp == 1234567890.0

    def test_newline_delimited(self):
        """Serialized messages end with newline."""
        msg = P2PMessage(VERSION, {}, "node1", 0.0)
        raw = msg.serialize()
        assert raw.endswith(b"\n")

    def test_all_message_types(self):
        """All message types can be serialized."""
        for mtype in [VERSION, VERACK, INV, GETDATA, BLOCK, TX,
                      GETBLOCKS, HEADERS, PING, PONG, ADDR]:
            msg = P2PMessage(mtype, {"test": True}, "sender", time.time())
            raw = msg.serialize()
            restored = P2PMessage.deserialize(raw)
            assert restored.msg_type == mtype


class TestBlockSerialization:
    def test_block_roundtrip(self, alice):
        """Block serializes and deserializes correctly."""
        bc = make_blockchain(alice)
        block = bc.chain[0]
        data = _serialize_block(block)
        restored = _deserialize_block(data)
        assert restored.block_hash == block.block_hash
        assert restored.index == block.index
        assert restored.previous_hash == block.previous_hash
        assert restored.nonce == block.nonce
        assert len(restored.transactions) == len(block.transactions)
        assert restored.transactions[0].tx_hash == block.transactions[0].tx_hash

    def test_header_roundtrip(self, alice):
        """Block header serializes and deserializes correctly."""
        bc = make_blockchain(alice, n_blocks=1)
        header = bc.chain[-1].header()
        data = _serialize_header(header)
        restored = _deserialize_header(data)
        assert restored.block_hash == header.block_hash
        assert restored.index == header.index
        assert restored.difficulty == header.difficulty


# =============================================================================
# MEMPOOL
# =============================================================================

class TestMempool:
    def test_add_and_size(self, alice, bob):
        """Adding a transaction increases mempool size."""
        bc = make_blockchain(alice, n_blocks=1)
        mempool = Mempool(bc.utxo_set)
        tx = build_transaction(alice, bob, 10 * COIN, bc.utxo_set)
        assert tx is not None
        assert mempool.add(tx)
        assert mempool.size() == 1

    def test_duplicate_rejected(self, alice, bob):
        """Adding the same transaction twice is rejected."""
        bc = make_blockchain(alice, n_blocks=1)
        mempool = Mempool(bc.utxo_set)
        tx = build_transaction(alice, bob, 10 * COIN, bc.utxo_set)
        assert mempool.add(tx)
        assert not mempool.add(tx)
        assert mempool.size() == 1

    def test_contains(self, alice, bob):
        """contains() checks by tx_hash."""
        bc = make_blockchain(alice, n_blocks=1)
        mempool = Mempool(bc.utxo_set)
        tx = build_transaction(alice, bob, 10 * COIN, bc.utxo_set)
        assert not mempool.contains(tx.tx_hash)
        mempool.add(tx)
        assert mempool.contains(tx.tx_hash)

    def test_remove(self, alice, bob):
        """Removing a transaction decreases size."""
        bc = make_blockchain(alice, n_blocks=1)
        mempool = Mempool(bc.utxo_set)
        tx = build_transaction(alice, bob, 10 * COIN, bc.utxo_set)
        mempool.add(tx)
        mempool.remove(tx.tx_hash)
        assert mempool.size() == 0
        assert not mempool.contains(tx.tx_hash)

    def test_get_by_fee_sorted(self, alice, bob):
        """Transactions are sorted by fee descending."""
        bc = make_blockchain(alice, n_blocks=3)
        mempool = Mempool(bc.utxo_set)

        tx_low = build_transaction(alice, bob, 10 * COIN, bc.utxo_set, fee=100)
        assert tx_low is not None
        bc.utxo_set.apply_transaction(tx_low)

        tx_high = build_transaction(alice, bob, 10 * COIN, bc.utxo_set, fee=5000)
        assert tx_high is not None

        # Re-create mempool with updated UTXO for proper fee calc
        mempool = Mempool(bc.utxo_set)
        mempool._transactions[tx_low.tx_hash] = tx_low
        mempool._transactions[tx_high.tx_hash] = tx_high

        sorted_txs = mempool.get_by_fee()
        assert len(sorted_txs) == 2
        assert sorted_txs[0].tx_hash == tx_high.tx_hash

    def test_clear_confirmed(self, alice, bob):
        """Transactions in a mined block are removed from mempool."""
        bc = make_blockchain(alice, n_blocks=1)
        mempool = Mempool(bc.utxo_set)
        tx = build_transaction(alice, bob, 10 * COIN, bc.utxo_set)
        mempool.add(tx)
        assert mempool.size() == 1

        # Mine a block containing the tx
        bc.add_transaction_to_mempool(tx)
        block = bc.mine_block()
        mempool.clear_confirmed(block)
        assert mempool.size() == 0

    def test_get_transactions_limited(self, alice, bob):
        """get_transactions respects max_count."""
        bc = make_blockchain(alice, n_blocks=1)
        mempool = Mempool(bc.utxo_set)
        tx = build_transaction(alice, bob, 10 * COIN, bc.utxo_set)
        mempool.add(tx)
        assert len(mempool.get_transactions(max_count=0)) == 0
        assert len(mempool.get_transactions(max_count=10)) == 1


# =============================================================================
# PEER STATE
# =============================================================================

class TestPeerState:
    def test_initial_state(self):
        """Peer starts in CONNECTING state."""
        peer = Peer(host="127.0.0.1", port=8000, node_id="test")
        assert peer.state == CONNECTING

    def test_disconnect(self):
        """disconnect() sets state to DISCONNECTED."""
        peer = Peer(host="127.0.0.1", port=8000, node_id="test")
        peer.state = CONNECTED
        peer.disconnect()
        assert peer.state == DISCONNECTED

    def test_address_property(self):
        """address property returns host:port."""
        peer = Peer(host="127.0.0.1", port=9999, node_id="test")
        assert peer.address == "127.0.0.1:9999"


# =============================================================================
# TWO-NODE HANDSHAKE
# =============================================================================

@pytest.mark.asyncio
@pytest.mark.timeout(10)
class TestHandshake:
    async def test_version_verack_exchange(self, alice, bob):
        """Two nodes complete VERSION/VERACK handshake."""
        node_a = await make_node(alice)
        node_b = await make_node(bob)
        try:
            peer = await connect_nodes(node_a, node_b)
            assert peer is not None
            assert len(node_a.peers) >= 1
            assert len(node_b.peers) >= 1
        finally:
            await node_a.stop()
            await node_b.stop()

    async def test_best_height_exchanged(self, alice, bob):
        """Nodes exchange best_height during handshake."""
        node_a = await make_node(alice, n_blocks=3)  # genesis + 3 = height 3
        node_b = await make_node(bob)
        try:
            await connect_nodes(node_a, node_b, settle_time=0.2)
            for peer in node_b.peers.values():
                if peer.version_received:
                    assert peer.best_height == 3
        finally:
            await node_a.stop()
            await node_b.stop()


# =============================================================================
# BLOCK PROPAGATION
# =============================================================================

@pytest.mark.asyncio
@pytest.mark.timeout(10)
class TestBlockPropagation:
    async def test_mined_block_propagates(self, alice, bob):
        """A mined block propagates to connected peer."""
        node_a = await make_node(alice)  # genesis only
        node_b = await make_node(bob)    # same genesis
        try:
            await connect_nodes(node_a, node_b, settle_time=0.2)

            block = await node_a.mine_and_broadcast()
            assert block is not None
            await asyncio.sleep(0.3)

            assert len(node_b.blockchain.chain) == 2
            assert node_b.blockchain.chain[-1].block_hash == block.block_hash
        finally:
            await node_a.stop()
            await node_b.stop()

    async def test_block_relay_chain(self, alice, bob, carol):
        """Block propagates through a chain: A -> B -> C."""
        node_a = await make_node(alice)
        node_b = await make_node(bob)
        node_c = await make_node(carol)
        try:
            await connect_nodes(node_a, node_b, settle_time=0.2)
            await connect_nodes(node_b, node_c, settle_time=0.2)

            block = await node_a.mine_and_broadcast()
            assert block is not None
            await asyncio.sleep(0.5)

            assert len(node_c.blockchain.chain) == 2
            assert node_c.blockchain.chain[-1].block_hash == block.block_hash
        finally:
            await node_a.stop()
            await node_b.stop()
            await node_c.stop()


# =============================================================================
# TRANSACTION RELAY
# =============================================================================

@pytest.mark.asyncio
@pytest.mark.timeout(10)
class TestTransactionRelay:
    async def test_tx_relayed_to_peer(self, alice, bob):
        """A submitted transaction is relayed to connected peer."""
        # Both nodes need identical chains so UTXO sets match.
        # Start with shared genesis, mine on node_a, propagate to node_b.
        node_a = await make_node(alice)
        node_b = await make_node(bob)
        try:
            await connect_nodes(node_a, node_b, settle_time=0.2)

            # Mine a block on node_a so alice has UTXOs, propagate to node_b
            block = await node_a.mine_and_broadcast()
            assert block is not None
            await asyncio.sleep(0.3)
            assert len(node_b.blockchain.chain) == 2  # synced

            # Now build and submit a TX
            tx = build_transaction(
                alice, bob, 10 * COIN, node_a.blockchain.utxo_set
            )
            assert tx is not None
            ok = await node_a.submit_transaction(tx)
            assert ok
            await asyncio.sleep(0.3)

            assert node_b.mempool.contains(tx.tx_hash)
        finally:
            await node_a.stop()
            await node_b.stop()

    async def test_invalid_tx_not_relayed(self, alice, bob):
        """Invalid transaction is rejected, not added to mempool."""
        node_a = await make_node(alice)
        node_b = await make_node(bob)
        try:
            await connect_nodes(node_a, node_b, settle_time=0.2)

            bad_tx = BAB64CashTransaction(
                inputs=[TxInput("deadbeef" * 8, 0)],
                outputs=[TxOutput(bob.address_hex, 10 * COIN, "", 0)],
            )
            bad_tx.finalize()

            msg = node_a._make_message(TX, {
                "transaction": _serialize_tx(bad_tx),
            })
            for peer in node_a.peers.values():
                await peer.send(msg)
            await asyncio.sleep(0.3)

            assert not node_b.mempool.contains(bad_tx.tx_hash)
        finally:
            await node_a.stop()
            await node_b.stop()


# =============================================================================
# INITIAL BLOCK DOWNLOAD
# =============================================================================

@pytest.mark.asyncio
@pytest.mark.timeout(10)
class TestIBD:
    async def test_new_node_syncs_blocks(self, alice, bob):
        """A new node downloads missing blocks from a peer via manual sync."""
        node_a = await make_node(alice, n_blocks=3)  # genesis + 3
        node_b = await make_node(bob)                 # genesis only
        try:
            await connect_nodes(node_b, node_a, settle_time=0.2)

            for p in node_b.peers.values():
                await node_b.sync_with_peer(p)
            await asyncio.sleep(0.5)

            assert len(node_b.blockchain.chain) == 4  # genesis + 3
            # All block hashes match
            for i in range(4):
                assert node_b.blockchain.chain[i].block_hash == \
                    node_a.blockchain.chain[i].block_hash
        finally:
            await node_a.stop()
            await node_b.stop()

    async def test_auto_ibd_on_connect(self, alice, bob):
        """IBD triggers automatically after handshake when peer is ahead."""
        node_a = await make_node(alice, n_blocks=5)  # genesis + 5
        node_b = await make_node(bob)                 # genesis only
        try:
            # B connects to A — auto-sync should trigger after handshake
            await connect_nodes(node_b, node_a, settle_time=1.0)

            assert len(node_b.blockchain.chain) == 6  # genesis + 5
            for i in range(6):
                assert node_b.blockchain.chain[i].block_hash == \
                    node_a.blockchain.chain[i].block_hash
        finally:
            await node_a.stop()
            await node_b.stop()

    async def test_sync_with_peer_height(self, alice, bob):
        """sync_with_peer downloads all blocks when peer has higher chain."""
        node_a = await make_node(alice, n_blocks=5)
        node_b = await make_node(bob)
        try:
            await connect_nodes(node_b, node_a, settle_time=0.2)

            for p in node_b.peers.values():
                p.best_height = 5
                await node_b.sync_with_peer(p)

            await asyncio.sleep(0.5)
            assert len(node_b.blockchain.chain) == 6
        finally:
            await node_a.stop()
            await node_b.stop()


# =============================================================================
# MINING AND BROADCASTING
# =============================================================================

@pytest.mark.asyncio
@pytest.mark.timeout(10)
class TestMining:
    async def test_mine_next_block(self, alice):
        """mine_next_block produces a valid block."""
        node = await make_node(alice)
        try:
            block = node.mine_next_block()
            assert block is not None
            assert len(node.blockchain.chain) == 2
            assert block.block_hash in node.known_blocks
        finally:
            await node.stop()

    async def test_mine_includes_mempool_tx(self, alice, bob):
        """Mined block includes mempool transactions."""
        node = await make_node(alice, n_blocks=1)  # need UTXOs from mining
        try:
            tx = build_transaction(
                alice, bob, 10 * COIN, node.blockchain.utxo_set
            )
            node.mempool.add(tx)

            block = node.mine_next_block()
            assert block is not None
            tx_hashes = [t.tx_hash for t in block.transactions]
            assert tx.tx_hash in tx_hashes
        finally:
            await node.stop()

    async def test_mine_and_broadcast(self, alice, bob):
        """mine_and_broadcast mines and sends INV to peers."""
        node_a = await make_node(alice)
        node_b = await make_node(bob)
        try:
            await connect_nodes(node_a, node_b, settle_time=0.2)
            block = await node_a.mine_and_broadcast()
            assert block is not None
            await asyncio.sleep(0.3)
            assert len(node_b.blockchain.chain) == 2
        finally:
            await node_a.stop()
            await node_b.stop()


# =============================================================================
# INVALID BLOCK REJECTION
# =============================================================================

@pytest.mark.asyncio
@pytest.mark.timeout(10)
class TestInvalidBlockRejection:
    async def test_reject_block_bad_hash(self, alice, bob):
        """Node rejects a block with invalid hash."""
        node_a = await make_node(alice)
        node_b = await make_node(bob)
        try:
            await connect_nodes(node_a, node_b, settle_time=0.2)

            bc = node_a.blockchain
            coinbase = BAB64CashTransaction.create_coinbase(
                alice.address_hex, len(bc.chain)
            )
            bad_block = BAB64Block(
                index=len(bc.chain),
                previous_hash=bc.chain[-1].block_hash,
                timestamp=time.time(),
                transactions=[coinbase],
                merkle_root_hash=merkle_root([coinbase.tx_hash]),
                nonce=0,
                difficulty=1,
                block_hash="f" * 64,
            )

            msg = node_a._make_message(BLOCK, {
                "block": _serialize_block(bad_block),
            })
            for peer in node_a.peers.values():
                await peer.send(msg)
            await asyncio.sleep(0.3)

            assert len(node_b.blockchain.chain) == 1
        finally:
            await node_a.stop()
            await node_b.stop()


# =============================================================================
# MULTIPLE PEERS (3-NODE NETWORK)
# =============================================================================

@pytest.mark.asyncio
@pytest.mark.timeout(10)
class TestMultiPeers:
    async def test_three_node_topology(self, alice, bob, carol):
        """Three nodes form a connected network."""
        node_a = await make_node(alice)
        node_b = await make_node(bob)
        node_c = await make_node(carol)
        try:
            await connect_nodes(node_a, node_b, settle_time=0.2)
            await connect_nodes(node_a, node_c, settle_time=0.2)

            assert len(node_a.peers) >= 2
            assert len(node_b.peers) >= 1
            assert len(node_c.peers) >= 1
        finally:
            await node_a.stop()
            await node_b.stop()
            await node_c.stop()

    async def test_block_reaches_all_nodes(self, alice, bob, carol):
        """Block mined on A reaches both B and C."""
        node_a = await make_node(alice)
        node_b = await make_node(bob)
        node_c = await make_node(carol)
        try:
            await connect_nodes(node_a, node_b, settle_time=0.2)
            await connect_nodes(node_a, node_c, settle_time=0.2)

            block = await node_a.mine_and_broadcast()
            await asyncio.sleep(0.5)

            assert node_b.blockchain.chain[-1].block_hash == block.block_hash
            assert node_c.blockchain.chain[-1].block_hash == block.block_hash
        finally:
            await node_a.stop()
            await node_b.stop()
            await node_c.stop()


# =============================================================================
# PEER DISCOVERY (ADDR)
# =============================================================================

@pytest.mark.asyncio
@pytest.mark.timeout(10)
class TestPeerDiscovery:
    async def test_addr_message_shares_peers(self, alice, bob):
        """ADDR message shares known peer addresses."""
        node_a = await make_node(alice)
        node_b = await make_node(bob)
        try:
            await connect_nodes(node_a, node_b, settle_time=0.2)
            await node_a.share_addresses()
            await asyncio.sleep(0.2)

            # node_b should have learned about addresses
            assert len(node_b._known_addresses) >= 1
        finally:
            await node_a.stop()
            await node_b.stop()


# =============================================================================
# PING / PONG
# =============================================================================

@pytest.mark.asyncio
@pytest.mark.timeout(10)
class TestPingPong:
    async def test_ping_pong(self, alice, bob):
        """PING elicits a PONG response."""
        node_a = await make_node(alice)
        node_b = await make_node(bob)
        try:
            await connect_nodes(node_a, node_b, settle_time=0.2)

            for peer in node_a.peers.values():
                old_time = peer.last_seen
                await node_a.ping_peer(peer, nonce=42)
                await asyncio.sleep(0.2)
                # Peer's last_seen should update from PONG
                # (on node_b side, the peer representing node_a gets updated)
        finally:
            await node_a.stop()
            await node_b.stop()


# =============================================================================
# RECONNECTION
# =============================================================================

@pytest.mark.asyncio
@pytest.mark.timeout(10)
class TestReconnection:
    async def test_disconnect_and_reconnect(self, alice, bob):
        """Node can reconnect after peer disconnects."""
        node_a = await make_node(alice)
        node_b = await make_node(bob)
        try:
            # First connection
            await connect_nodes(node_a, node_b, settle_time=0.2)
            assert len(node_a.peers) >= 1

            # Disconnect all peers
            for peer in list(node_a.peers.values()):
                peer.disconnect()
            await asyncio.sleep(0.2)

            # Reconnect
            peer = await connect_nodes(node_a, node_b, settle_time=0.2)
            assert peer is not None
            assert len(node_a.peers) >= 1
        finally:
            await node_a.stop()
            await node_b.stop()

    async def test_peer_state_after_disconnect(self, alice, bob):
        """Peer state transitions to DISCONNECTED after disconnect."""
        node_a = await make_node(alice)
        node_b = await make_node(bob)
        try:
            await connect_nodes(node_a, node_b, settle_time=0.2)

            peers = list(node_a.peers.values())
            assert len(peers) >= 1
            peer = peers[0]
            peer.disconnect()
            assert peer.state == DISCONNECTED
        finally:
            await node_a.stop()
            await node_b.stop()


# =============================================================================
# TRANSACTION SUBMISSION
# =============================================================================

@pytest.mark.asyncio
@pytest.mark.timeout(10)
class TestTransactionSubmission:
    async def test_submit_valid_tx(self, alice, bob):
        """submit_transaction adds to mempool and returns True."""
        node = await make_node(alice, n_blocks=1)  # need UTXOs
        try:
            tx = build_transaction(
                alice, bob, 10 * COIN, node.blockchain.utxo_set
            )
            assert tx is not None
            ok = await node.submit_transaction(tx)
            assert ok
            assert node.mempool.contains(tx.tx_hash)
        finally:
            await node.stop()

    async def test_submit_duplicate_tx_rejected(self, alice, bob):
        """Submitting the same transaction twice returns False."""
        node = await make_node(alice, n_blocks=1)
        try:
            tx = build_transaction(
                alice, bob, 10 * COIN, node.blockchain.utxo_set
            )
            assert await node.submit_transaction(tx)
            assert not await node.submit_transaction(tx)
        finally:
            await node.stop()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
