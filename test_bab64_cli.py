"""
Tests for BAB64 CLI — node runner and wallet
=============================================

Subprocess-based tests verifying CLI behavior.
"""

import json
import os
import signal
import subprocess
import sys
import tempfile
import time
import unittest

PYTHON = sys.executable
NODE_SCRIPT = os.path.join(os.path.dirname(__file__), "bab64_node.py")
WALLET_SCRIPT = os.path.join(os.path.dirname(__file__), "bab64_wallet.py")


def run_wallet(args, data_dir, timeout=30):
    """Run bab64_wallet.py with given args, return (stdout, stderr, returncode)."""
    cmd = [PYTHON, WALLET_SCRIPT, "--data-dir", data_dir] + args
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
    return result.stdout.strip(), result.stderr.strip(), result.returncode


def run_node(args, data_dir, timeout=30):
    """Run bab64_node.py with given args, return (stdout, stderr, returncode)."""
    cmd = [PYTHON, NODE_SCRIPT, "--data-dir", data_dir] + args
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
    return result.stdout.strip(), result.stderr.strip(), result.returncode


def start_node(args, data_dir):
    """Start a node process in the background, return Popen."""
    cmd = [PYTHON, NODE_SCRIPT, "--data-dir", data_dir] + args
    proc = subprocess.Popen(
        cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
    )
    return proc


def stop_node(proc, timeout=10):
    """Send SIGINT and wait for clean shutdown."""
    try:
        proc.send_signal(signal.SIGINT)
    except (ProcessLookupError, OSError):
        pass
    try:
        stdout, stderr = proc.communicate(timeout=timeout)
        return stdout.strip(), stderr.strip(), proc.returncode
    except subprocess.TimeoutExpired:
        proc.kill()
        stdout, stderr = proc.communicate()
        return stdout.strip(), stderr.strip(), proc.returncode


class TestWalletCreate(unittest.TestCase):
    """Test wallet create command."""

    def test_create_produces_address(self):
        with tempfile.TemporaryDirectory() as d:
            out, err, rc = run_wallet(["create"], d)
            self.assertEqual(rc, 0)
            # Address should be 64 hex chars
            self.assertEqual(len(out), 64)
            self.assertTrue(all(c in "0123456789abcdef" for c in out))

    def test_create_multiple_addresses(self):
        with tempfile.TemporaryDirectory() as d:
            out1, _, _ = run_wallet(["create"], d)
            out2, _, _ = run_wallet(["create"], d)
            self.assertNotEqual(out1, out2)


class TestWalletList(unittest.TestCase):
    """Test wallet list command."""

    def test_list_shows_created_address(self):
        with tempfile.TemporaryDirectory() as d:
            addr, _, _ = run_wallet(["create"], d)
            out, _, rc = run_wallet(["list"], d)
            self.assertEqual(rc, 0)
            self.assertIn(addr, out)

    def test_list_empty(self):
        with tempfile.TemporaryDirectory() as d:
            out, _, rc = run_wallet(["list"], d)
            self.assertEqual(rc, 0)
            self.assertIn("No wallets", out)


class TestWalletBalance(unittest.TestCase):
    """Test wallet balance command."""

    def test_balance_zero_for_new_address(self):
        with tempfile.TemporaryDirectory() as d:
            addr, _, _ = run_wallet(["create"], d)
            out, _, rc = run_wallet(["balance", "--address", addr], d)
            self.assertEqual(rc, 0)
            self.assertIn("0.00000000", out)

    def test_balance_all_addresses(self):
        with tempfile.TemporaryDirectory() as d:
            run_wallet(["create"], d)
            out, _, rc = run_wallet(["balance"], d)
            self.assertEqual(rc, 0)
            self.assertIn("BAB64", out)


class TestWalletInfo(unittest.TestCase):
    """Test wallet info command."""

    def test_info_empty_chain(self):
        with tempfile.TemporaryDirectory() as d:
            out, _, rc = run_wallet(["info"], d)
            self.assertEqual(rc, 0)
            self.assertIn("Chain height:", out)


class TestNodeCreateWallet(unittest.TestCase):
    """Test node --create-wallet."""

    def test_create_wallet_prints_address(self):
        with tempfile.TemporaryDirectory() as d:
            out, _, rc = run_node(["--create-wallet"], d)
            self.assertEqual(rc, 0)
            self.assertEqual(len(out), 64)
            self.assertTrue(all(c in "0123456789abcdef" for c in out))

    def test_create_wallet_persists(self):
        with tempfile.TemporaryDirectory() as d:
            addr, _, _ = run_node(["--create-wallet"], d)
            out, _, _ = run_wallet(["list"], d)
            self.assertIn(addr, out)


class TestNodeStartStop(unittest.TestCase):
    """Test node starts and stops cleanly."""

    def test_start_and_sigint(self):
        with tempfile.TemporaryDirectory() as d:
            proc = start_node(["--port", "18333", "--difficulty", "1"], d)
            time.sleep(2)
            stdout, stderr, rc = stop_node(proc)
            # Should print shutdown summary
            self.assertIn("Shutting down", stdout)
            self.assertIn("Chain height:", stdout)


class TestNodePeerConnect(unittest.TestCase):
    """Test two nodes connecting."""

    def test_two_nodes_connect(self):
        with tempfile.TemporaryDirectory() as d1, \
             tempfile.TemporaryDirectory() as d2:
            # Start node 1
            node1 = start_node(["--port", "18334", "--difficulty", "1"], d1)
            time.sleep(2)

            # Start node 2 connecting to node 1
            node2 = start_node(
                ["--port", "18335", "--connect", "127.0.0.1:18334",
                 "--difficulty", "1"], d2
            )
            time.sleep(2)

            # Both should have genesis block
            _, stderr2, _ = stop_node(node2)
            _, stderr1, _ = stop_node(node1)

            # Node 2 should have logged a connection
            self.assertIn("Connected to peer", stderr2)


class TestNodeMining(unittest.TestCase):
    """Test node mines a block and persists it."""

    def test_mine_and_persist(self):
        with tempfile.TemporaryDirectory() as d:
            proc = start_node(
                ["--port", "18336", "--mine", "--difficulty", "1"], d
            )
            # Wait for at least one block to be mined
            time.sleep(5)
            stdout, stderr, rc = stop_node(proc)

            # Should have mined at least block 1
            self.assertIn("Block", stderr)
            self.assertIn("mined", stderr)

            # Verify persistence — check chain height via wallet info
            out, _, _ = run_wallet(["info"], d)
            self.assertIn("Chain height:", out)
            # Height should be > 0 (genesis + at least 1 mined)
            for line in out.split("\n"):
                if "Chain height:" in line:
                    height = int(line.split(":")[1].strip())
                    self.assertGreater(height, 0)


class TestNodeMiningBalance(unittest.TestCase):
    """Test wallet balance updates after mining."""

    def test_balance_after_mining(self):
        with tempfile.TemporaryDirectory() as d:
            # Create wallet first
            addr, _, _ = run_node(["--create-wallet"], d)

            # Mine with that wallet
            proc = start_node(
                ["--port", "18337", "--mine", "--difficulty", "1",
                 "--wallet", addr], d
            )
            time.sleep(5)
            stop_node(proc)

            # Check balance via wallet — should be > 0
            out, _, _ = run_wallet(["balance", "--address", addr], d)
            self.assertIn("BAB64", out)
            # The miner reward should give a non-zero balance
            # Parse the balance value
            for part in out.split():
                try:
                    bal = float(part)
                    if bal > 0:
                        break
                except ValueError:
                    continue
            else:
                self.fail("Expected non-zero balance after mining")


class TestWalletSendInsufficientFunds(unittest.TestCase):
    """Test send fails with insufficient funds."""

    def test_send_no_funds(self):
        with tempfile.TemporaryDirectory() as d:
            addr, _, _ = run_wallet(["create"], d)
            _, err, rc = run_wallet(
                ["send", "--to", "a" * 64, "--amount", "10"], d
            )
            self.assertNotEqual(rc, 0)
            self.assertIn("insufficient funds", err.lower())


if __name__ == "__main__":
    unittest.main()
