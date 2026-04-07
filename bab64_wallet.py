#!/usr/bin/env python3
"""
BAB64 Wallet — Command-line wallet for managing identities and coins
======================================================================

Usage:
  python3 bab64_wallet.py create --data-dir ./node1
  python3 bab64_wallet.py balance --address <addr> --data-dir ./node1
  python3 bab64_wallet.py send --to <addr> --amount 10 --fee 0.001 --data-dir ./node1
  python3 bab64_wallet.py list --data-dir ./node1
  python3 bab64_wallet.py info --data-dir ./node1

Author: Shrey (concept) + Claude (implementation)
"""

import argparse
import os
import sys
import time

from bab64_cash import BAB64Blockchain, COIN, build_transaction
from bab64_identity import BAB64Identity
from bab64_storage import BAB64Storage


def _add_common_args(parser, with_defaults=True):
    """Add --data-dir and --passphrase to a parser.

    When with_defaults=False, use SUPPRESS so subparser copies don't
    overwrite values already parsed by the parent parser.
    """
    default_dir = "./bab64_data" if with_defaults else argparse.SUPPRESS
    default_pass = None if with_defaults else argparse.SUPPRESS
    parser.add_argument("--data-dir", default=default_dir,
                        help="Data directory for SQLite DBs (default: ./bab64_data)")
    parser.add_argument("--passphrase", default=default_pass,
                        help="Passphrase for wallet encryption (default: bab64default)")


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description="BAB64 Wallet — manage identities and send coins",
    )
    _add_common_args(parser)

    sub = parser.add_subparsers(dest="command")

    # create
    create = sub.add_parser("create", help="Create a new identity, print address")
    _add_common_args(create, with_defaults=False)

    # balance
    bal = sub.add_parser("balance", help="Show balance for an address")
    bal.add_argument("--address", default="",
                     help="Address to check (default: all)")
    _add_common_args(bal, with_defaults=False)

    # send
    send = sub.add_parser("send", help="Build, sign, and submit a transaction")
    send.add_argument("--to", required=True, help="Recipient address")
    send.add_argument("--amount", type=float, required=True,
                      help="Amount in BAB64 coins")
    send.add_argument("--fee", type=float, default=0.001,
                      help="Fee in BAB64 coins (default: 0.001)")
    send.add_argument("--from", dest="from_addr", default="",
                      help="Sender address (default: first wallet)")
    _add_common_args(send, with_defaults=False)

    # list
    lst = sub.add_parser("list", help="List all wallet addresses")
    _add_common_args(lst, with_defaults=False)

    # info
    info = sub.add_parser("info", help="Show node info: chain height, supply, etc.")
    _add_common_args(info, with_defaults=False)

    return parser.parse_args(argv)


def cmd_create(storage, passphrase):
    identity = BAB64Identity.generate()
    storage.wallet.save_identity(identity, passphrase)
    print(identity.address_hex)


def cmd_balance(storage, address):
    if address:
        bal = storage.utxos.balance(address)
        print(f"{bal / COIN:.8f} BAB64")
    else:
        addresses = storage.wallet.list_addresses()
        if not addresses:
            print("No wallets found")
            return
        for addr in addresses:
            bal = storage.utxos.balance(addr)
            print(f"{addr[:16]}...  {bal / COIN:.8f} BAB64")


def cmd_send(storage, passphrase, from_addr, to_addr, amount_bab64, fee_bab64):
    amount = int(amount_bab64 * COIN)
    fee = int(fee_bab64 * COIN)

    # Load sender identity
    addresses = storage.wallet.list_addresses()
    if not addresses:
        print("Error: no wallets found", file=sys.stderr)
        sys.exit(1)

    sender_addr = from_addr if from_addr else addresses[0]
    identity = storage.wallet.load_identity(sender_addr, passphrase)
    if not identity:
        print(f"Error: cannot load wallet {sender_addr[:16]}...", file=sys.stderr)
        sys.exit(1)

    # Load blockchain state for UTXO set
    chain, utxo_set = storage.load_state()

    tx = build_transaction(identity, to_addr, amount, utxo_set, fee=fee)
    if tx is None:
        print("Error: insufficient funds", file=sys.stderr)
        sys.exit(1)

    print(f"Transaction {tx.tx_hash[:16]} sent | {amount_bab64} BAB64 to {to_addr[:16]}")


def cmd_list(storage):
    addresses = storage.wallet.list_addresses()
    if not addresses:
        print("No wallets found")
        return
    for addr in addresses:
        print(addr)


def cmd_info(storage):
    chain = storage.blockchain.load_chain()
    height = len(chain) - 1 if chain else -1

    difficulty = chain[-1].difficulty if chain else 0
    last_hash = chain[-1].block_hash if chain else "N/A"
    last_time = chain[-1].timestamp if chain else 0

    # Total supply
    total = 0
    for block in chain:
        for tx in block.transactions:
            if tx.is_coinbase:
                total += sum(out.amount for out in tx.outputs)

    peers = storage.peers.get_peers()

    print(f"Chain height:  {height}")
    print(f"Difficulty:    {difficulty}")
    print(f"Total supply:  {total / COIN:.8f} BAB64")
    print(f"Peers known:   {len(peers)}")
    print(f"Last block:    {last_hash[:32]}...")
    if last_time:
        print(f"Last time:     {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(last_time))}")


def main(argv=None):
    args = parse_args(argv)
    if not args.command:
        parse_args(["--help"])
        return

    data_dir = os.path.abspath(args.data_dir)
    os.makedirs(data_dir, exist_ok=True)
    storage = BAB64Storage(data_dir)
    passphrase = args.passphrase or "bab64default"

    try:
        if args.command == "create":
            cmd_create(storage, passphrase)
        elif args.command == "balance":
            cmd_balance(storage, args.address)
        elif args.command == "send":
            cmd_send(storage, passphrase, args.from_addr,
                     args.to, args.amount, args.fee)
        elif args.command == "list":
            cmd_list(storage)
        elif args.command == "info":
            cmd_info(storage)
    finally:
        storage.close()


if __name__ == "__main__":
    main()
