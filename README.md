# BAB64 — Self-Referential Image Hash

**The image defines its own hash function, then is hashed by it.**

A novel proof-of-work primitive where every 64x64 Babel image produces a unique cryptographic compression function — derived from its own pixels — which is then applied to the image itself. Mining becomes a fixed-point search in function space: find an image whose self-generated hash meets the difficulty target.

No known PoW system uses input-dependent hash parameterization.

## What Makes It Novel

In every existing PoW system (Bitcoin, Ethash, Equihash), the hash function is fixed. Miners search for inputs that produce a desired output under that static function.

BAB64 inverts this: **the input defines the function**. Each image's pixels are used to derive:

- **Round constants** (like SHA-256's K[]) — from pixel blocks
- **Rotation schedule** (like SHA-256's ROTR) — from pixel columns
- **S-box** (like AES substitution) — Fisher-Yates shuffle seeded by image
- **Initial state** (like SHA-256's H0) — from image hash

Then this unique function hashes the image that created it.

## Hash Quality

| Metric | Measured | Ideal |
|---|---|---|
| Avalanche (1px flip) | 49.9% bits change | 50.0% |
| Bit distribution P(bit=1) | 0.501 | 0.500 |
| Collisions (200 trials) | 0 | 0 |

## Quick Start

```bash
# Run engine demo (quality analysis + mining + chain)
python3 bab64_engine.py

# Run full test suite (52 tests)
python3 test_bab64.py
```

## How It Works

```
1. Input → SHA-256 → base_seed
2. base_seed + nonce → BabelRender → 64×64 image (4,096 pixels)
3. Image pixels → derive round constants, rotations, S-box, initial state
4. Apply this image-specific hash function TO the image itself
   → If hash has enough leading zeros: proof found
   → Otherwise: increment nonce, goto 2
```

Verification: regenerate the image, re-derive the hash function, re-hash. One nonce check vs. thousands during mining.

## Comparison

| | Bitcoin (SHA-256) | Ethash | Equihash | **BAB64** |
|---|---|---|---|---|
| Hash function | Fixed | Fixed | Fixed | **Input-dependent** |
| Hard problem | Preimage | Memory-hard hash | Birthday bound | **Self-referential fixed point** |
| Input defines function? | No | No | No | **Yes** |
| ASIC resistance | None | Medium | Medium | High (dynamic function) |
| Memory per hash | ~0 | ~2 GB DAG | ~144 MB | ~2 MB (image + params) |
| Construction | Merkle-Damgard | Keccak-based | Wagner's algorithm | **Image-parameterized M-D** |

## BAB256 — Lattice Engine (Historical)

BAB256 was the research path that led to BAB64. It combines SHA-256 with the Closest Vector Problem (CVP) in 4,096-dimensional image space — a lattice-based PoW with quantum resistance claims.

Kept for historical reference:
- `bab256_engine_v02.py` — Core engine with 4 CVP solvers
- `test_bab256.py` — 46 tests

BAB64 is the distilled insight: you don't need the lattice machinery when the image itself can define the hash function.

## Origin

Concept by Shrey — born from the question: *"Can a Babel image be its own cryptographic primitive?"* The answer turned out to be yes: not by using the image as data to be hashed, but by using it as the hash function's parameters. The image is simultaneously the message, the key, and the algorithm.

## License

MIT — Research prototype, NOT for production use.
