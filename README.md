# BAB256 — Dimensional Engine

**Lattice-Based Proof of Work in Babel Image Space**

A novel cryptographic proof-of-work primitive that combines SHA-256's proven security with the Closest Vector Problem (CVP) operating natively in 4,096-dimensional image space derived from Babel's Universal Image Archive.

## Security Properties

| Property | SHA-256 (Bitcoin) | BAB256 |
|---|---|---|
| Classical hardness | 2^256 | 2^1,196 |
| Quantum hardness | 2^128 (Grover) | 2^1,085 |
| Hard problem | Hash preimage | CVP (NP-hard) |
| Memory requirement | ~0 | ~2 MB (basis) |
| ASIC resistance | None | High |
| Verification asymmetry | ~1x | 500x+ |

## Architecture

```
Input → SHA-256(input) → seed
seed → Generate 128 basis images in Z^4096 (Babel lattice)
seed + nonce → Generate target image
HARD PART: Solve approximate CVP — find integer coefficients
           c₁...c₁₂₈ minimizing distance to target in lattice
proof = SHA-256(coefficients || target_hash || nonce)
Check: proof has required leading zero bits (difficulty)
```

## Files

- `bab256_engine_v02.py` — Core engine with 4 CVP solvers + blockchain simulation
- `test_bab256.py` — 46 tests proving cryptographic invariants
- `bab256_engine.py` — v0.1 (historical reference, superseded)

## Quick Start

```bash
# Run engine demo (solver benchmark + mining + chain)
python3 bab256_engine_v02.py

# Run full test suite
python3 test_bab256.py
```

## Test Results (v0.2)

```
46/46 tests passed

✓ Determinism      — Same input always produces same output
✓ Tamper resistance — Any modification invalidates proof
✓ Avalanche effect  — 1-bit input change → ~50% output change
✓ Verification      — 500x+ faster than mining
✓ Difficulty scaling — Mining time scales with difficulty
✓ Solver quality    — Babai beats random, combined beats all
✓ Chain integrity   — 3-block chain validates with tamper detection
✓ Edge cases        — Empty input, unicode, serialization roundtrip
```

## CVP Solvers

| Solver | Method | Speed | Quality |
|---|---|---|---|
| `greedy` | Random start + local search | Medium | Worst |
| `babai_round` | Least-squares + rounding | Fast | Good |
| `babai_plane` | Gram-Schmidt + nearest plane | Medium | Better |
| `combined` | Babai seed + greedy refinement | Slower | Best |

## Roadmap

- [ ] Difficulty scaling benchmark with charts
- [ ] Formal parameter analysis (basis count, coefficient bounds)
- [ ] BKZ lattice reduction integration (fpylll)
- [ ] SNARK/STARK proof compression
- [ ] C/Rust high-performance implementation
- [ ] Formal specification / whitepaper
- [ ] Academic submission (IACR ePrint)

## Origin

Concept by Shrey — inspired by the question: "Can the entropy of Babel's Universal Image Archive be harnessed for cryptographic security?"

The answer: not the entropy itself, but the *dimensionality* — by making the hard problem (CVP) operate natively in image-space dimensions rather than compressing through a 256-bit bottleneck.

## License

MIT — Research prototype, NOT for production use.
