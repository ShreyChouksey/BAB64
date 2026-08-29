# MAPAE Host-a-Shard Reference Pilot

This directory is the first proof-of-operation for the permission-based **MAPAE Host-a-Shard Network**.

It hosts an exact copy of one previously unexposed, 100-address shard from the fixed Million-Address Perpetual Adversarial Exposure Experiment corpus.

## Reference shard

- Shard ID: `MAPAE-HAS-00001`
- Address count: `100`
- Shard SHA-256: `f6c97c6efc2145bd6db56e783e4efb8c6b624cb5cd88407fcc0cdd7bb16e8f50`
- File: [`MAPAE-HAS-00001.txt`](MAPAE-HAS-00001.txt)
- Canonical one-million-address corpus SHA-256: `5bb9320bc93f07e3129cb6ef5aee4da2c245e0ca11279d4963244bead79a90df`

## Purpose

This is a controlled reference host used to prove the complete workflow:

1. deterministic shard creation;
2. public publication;
3. exact-byte hash verification;
4. public registry entry;
5. later reassignment of different shards to independently administered, consenting websites.

This reference host is controlled by the same operator as MAPAE and therefore does **not** count as an independent-admin host. Its role is to validate the machinery before outside hosts are admitted.

## Safety boundary

The file contains public Bitcoin `bc1q...` addresses only. It contains no private keys, WIFs, seeds, mnemonics, entropy source, recovery material, xprvs, or signing secrets. Publication is not a request to send funds.

Canonical MAPAE project: https://github.com/ShreyChouksey/New/tree/claude/babel-image-archive-generator-mf9Gy/MAPAE

Host-a-Shard program: https://github.com/ShreyChouksey/New/tree/claude/babel-image-archive-generator-mf9Gy/MAPAE/host-a-shard
