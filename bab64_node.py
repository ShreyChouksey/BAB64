#!/usr/bin/env python3
"""
BAB64 Node — Command-line node runner
=======================================

Start, mine, connect to peers, and sync the BAB64 blockchain.

Usage:
  python3 bab64_node.py --port 8333 --mine --data-dir ./node1
  python3 bab64_node.py --port 8334 --connect 127.0.0.1:8333 --data-dir ./node2
  python3 bab64_node.py --port 8335 --connect 127.0.0.1:8333 --mine --data-dir ./node3

Author: Shrey (concept) + Claude (implementation)
"""

import argparse
import asyncio
import getpass
import logging
import os
import signal
import sys
import time

from bab64_cash import BAB64Blockchain, COIN
from bab64_identity import BAB64Identity
from bab64_network import BAB64Node
from bab64_storage import BAB64Storage

logger = logging.getLogger("bab64")


def setup_logging(level: str):
    """Configure logging with timestamped format."""
    numeric = getattr(logging, level.upper(), logging.INFO)
    handler = logging.StreamHandler(sys.stderr)
    handler.setFormatter(logging.Formatter(
        "%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    ))
    logger.setLevel(numeric)
    logger.addHandler(handler)


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description="BAB64 Node — run a BAB64 Cash full node",
    )
    parser.add_argument("--port", type=int, default=8333,
                        help="Listen port (default: 8333)")
    parser.add_argument("--connect", action="append", default=[],
                        metavar="HOST:PORT",
                        help="Connect to a peer on startup (repeatable)")
    parser.add_argument("--mine", action="store_true",
                        help="Enable mining (continuously mine blocks)")
    parser.add_argument("--data-dir", default="./bab64_data",
                        help="Data directory for SQLite DBs (default: ./bab64_data)")
    parser.add_argument("--difficulty", type=int, default=1,
                        help="Starting difficulty (default: 1)")
    parser.add_argument("--wallet", default="",
                        help="Use this wallet address for mining rewards")
    parser.add_argument("--create-wallet", action="store_true",
                        help="Create a new wallet identity and print address")
    parser.add_argument("--passphrase", default=None,
                        help="Passphrase for wallet encryption (default: prompt)")
    parser.add_argument("--log-level", default="INFO",
                        help="Logging level (default: INFO)")
    return parser.parse_args(argv)


def get_or_create_identity(storage, passphrase, wallet_address=None):
    """Load existing identity or create a new one."""
    addresses = storage.wallet.list_addresses()

    if wallet_address and wallet_address in addresses:
        identity = storage.wallet.load_identity(wallet_address, passphrase)
        if identity:
            return identity

    if addresses:
        # Load first available identity
        identity = storage.wallet.load_identity(addresses[0], passphrase)
        if identity:
            return identity

    # Create new identity
    identity = BAB64Identity.generate()
    storage.wallet.save_identity(identity, passphrase)
    return identity


def _mine_block_sync(blockchain, identity):
    """Mine a single block synchronously (no SQLite access).

    We temporarily detach storage so mine_block() doesn't touch SQLite
    from a worker thread, then reattach it afterwards.
    """
    saved_storage = blockchain.storage
    blockchain.storage = None
    try:
        block = blockchain.mine_block(identity)
    finally:
        blockchain.storage = saved_storage
    return block


async def mining_loop(node: BAB64Node, blockchain: BAB64Blockchain,
                      identity: BAB64Identity):
    """Continuously mine blocks and broadcast them."""
    while True:
        try:
            block = await asyncio.get_event_loop().run_in_executor(
                None, _mine_block_sync, blockchain, identity,
            )
            if block:
                # Persist from the main thread (safe for SQLite)
                if blockchain.storage:
                    blockchain.storage.blockchain.save_block(block)

                node.known_blocks.add(block.block_hash)
                node.mempool.clear_confirmed(block)
                node.mempool.set_utxo_set(blockchain.utxo_set)

                n_txns = len(block.transactions)
                logger.info(
                    "Block %d mined | hash: %s... | txns: %d",
                    block.index, block.block_hash[:16], n_txns,
                )
                # Broadcast to peers
                from bab64_network import INV, INV_BLOCK
                inv_msg = node._make_message(INV, {
                    "items": [{"type": INV_BLOCK, "hash": block.block_hash}],
                })
                await node.broadcast(inv_msg)
        except Exception as e:
            logger.error("Mining error: %s", e)

        await asyncio.sleep(1)


async def run_node(args):
    """Main async entry point for the node."""
    data_dir = os.path.abspath(args.data_dir)
    os.makedirs(data_dir, exist_ok=True)

    storage = BAB64Storage(data_dir)

    passphrase = args.passphrase
    if passphrase is None:
        passphrase = "bab64default"

    # Handle --create-wallet
    if args.create_wallet:
        identity = BAB64Identity.generate()
        storage.wallet.save_identity(identity, passphrase)
        print(identity.address_hex)
        if not args.mine and not args.connect:
            storage.close()
            return

    # Load or create identity
    identity = get_or_create_identity(storage, passphrase, args.wallet)
    logger.info("Wallet address: %s", identity.address_hex)

    # Load blockchain
    blockchain = BAB64Blockchain(
        difficulty=args.difficulty,
        miner=identity,
        storage=storage,
    )

    # Init genesis if empty
    if not blockchain.chain:
        blockchain.add_genesis()
        logger.info("Genesis block created")

    # Create P2P node
    node = BAB64Node(
        host="127.0.0.1",
        port=args.port,
        blockchain=blockchain,
        identity=identity,
    )

    # Start TCP server
    await node.start()
    logger.info("Node listening on port %d", args.port)

    # Connect to peers
    for peer_str in args.connect:
        host, port_str = peer_str.rsplit(":", 1)
        port = int(port_str)
        peer = await node.connect_to_peer(host, port)
        if peer:
            logger.info("Connected to peer %s:%d", host, port)
            storage.peers.save_peer(host, port)
        else:
            logger.warning("Failed to connect to %s:%d", host, port)

    # Start mining loop if requested
    mining_task = None
    if args.mine:
        logger.info("Mining enabled")
        mining_task = asyncio.ensure_future(
            mining_loop(node, blockchain, identity)
        )

    # Wait for shutdown signal
    shutdown_event = asyncio.Event()

    def _signal_handler():
        shutdown_event.set()

    loop = asyncio.get_event_loop()
    for sig in (signal.SIGINT, signal.SIGTERM):
        try:
            loop.add_signal_handler(sig, _signal_handler)
        except NotImplementedError:
            pass  # Windows

    await shutdown_event.wait()

    # Graceful shutdown
    logger.info("Shutting down...")
    if mining_task:
        mining_task.cancel()
        try:
            await mining_task
        except asyncio.CancelledError:
            pass

    await node.stop()

    # Save state
    storage.save_state(blockchain.chain)
    balance = blockchain.get_balance(identity.address_hex)
    height = len(blockchain.chain) - 1

    print(f"Shutting down. Chain height: {height}, Balance: {balance / COIN:.8f}")
    storage.close()


def main(argv=None):
    args = parse_args(argv)
    setup_logging(args.log_level)

    if args.create_wallet and not args.mine and not args.connect:
        # Quick path: just create wallet and exit
        data_dir = os.path.abspath(args.data_dir)
        os.makedirs(data_dir, exist_ok=True)
        storage = BAB64Storage(data_dir)
        passphrase = args.passphrase or "bab64default"
        identity = BAB64Identity.generate()
        storage.wallet.save_identity(identity, passphrase)
        print(identity.address_hex)
        storage.close()
        return

    asyncio.run(run_node(args))


if __name__ == "__main__":
    main()
