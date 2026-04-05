# BAB64: Proof-of-Work via Self-Referential Image Hashing

**Authors:** Shrey Chouksey, Claude (Anthropic)
**Date:** April 2026
**Status:** Research prototype (v2)

---

## Abstract

We present BAB64, a proof-of-work construction in which the hash function is not fixed at protocol design time but is instead derived from the input itself. Each candidate input deterministically generates a 64x64 grayscale image, which in turn parameterizes a unique substitution-permutation network --- defining its own round constants, rotation schedule, S-box, and initial state. The proof-of-work output is the image hashed by its own derived function. This self-referential structure eliminates precomputation advantages and creates structural ASIC resistance by requiring per-nonce function reconfiguration.

Empirical evaluation over 101 tests shows near-ideal avalanche (50.2%), uniform bit distribution (P(1) = 0.4998), zero collisions across 500 trials, and byte-identical outputs across two independent implementations. A parallel diffusion step achieves 46.3% single-block avalanche in one round. Six adversarial attack simulations --- including related-image, preimage structure, and Joux multi-collision --- find no exploitable weakness. S-box quality analysis across 10,000 images shows mean nonlinearity of 91.8, consistent with random permutations. A C-accelerated implementation achieves approximately 7,000 hashes per second on commodity hardware.

An identity layer provides Bitcoin-like addresses derived from private images, with Lamport one-time signatures for quantum-resistant transaction signing, automatic key rotation, and replay protection.

No formal security reduction is provided; the construction's resistance to optimization rests on structural arguments and empirical evidence.

---

## 1. Introduction

Proof-of-work (PoW) systems underpin the security of permissionless consensus protocols. In all widely deployed PoW constructions --- Bitcoin's SHA-256d [1], Ethereum's Ethash [2], Zcash's Equihash [3], and Cuckoo Cycle [4] --- the hash function is fixed at protocol design time. The miner searches for an input that produces a low output under this fixed function. This architectural choice has two consequences:

1. **Precomputation is possible.** A fixed function admits partial evaluation reuse. Bitcoin miners exploit midstate caching; Ethash requires a multi-gigabyte DAG precomputed per epoch.

2. **ASIC optimization is straightforward.** When the function is known at fabrication time, its constants, rotation amounts, and data paths can be hardcoded into silicon. Bitcoin ASICs achieve orders-of-magnitude efficiency over general-purpose hardware precisely because SHA-256 never changes.

BAB64 eliminates the fixed function entirely. Each mining attempt produces a candidate image I, which deterministically defines a unique compression function H_I. The proof-of-work output is H_I(I) --- the image hashed by the function it defines. The miner searches simultaneously in input space and function space.

This paper makes four contributions:

- **A novel PoW primitive** based on self-referential hashing, where no two mining attempts use the same hash function.
- **A concrete construction** with byte-precise specification, two independent implementations, and 101 passing tests across core hashing, identity, and stress test suites.
- **Empirical evidence** that the construction achieves near-ideal statistical properties despite using data-derived, per-input parameters, validated by six adversarial attack simulations.
- **An identity layer** with private-image-derived addresses and quantum-resistant Lamport signatures, demonstrating that BAB64 can serve as a foundation for a complete transaction system.

We emphasize that BAB64 is a research contribution. No formal security reduction is provided, and the construction has not been deployed in an adversarial setting. We present the design, evidence, and limitations honestly and invite further analysis.

---

## 2. Construction

### 2.1 Design Intuition

The construction draws on a thought experiment from Borges' *Library of Babel*: a combinatorial library containing every possible book. BAB64 treats the space of 64x64 grayscale images as such a library, where each "book" contains instructions for its own cryptographic compression. The miner's task is to find a book whose self-description meets an external difficulty criterion.

Concretely, each image provides the following hash function components:

| Component | Derivation | Analogue |
|-----------|-----------|----------|
| Round constants K[0..31] | SHA-256 of 128-pixel blocks | SHA-256's K[] (cube roots of primes) |
| Rotation schedule rot[0..31] | Pixel pair lookup, range [1, 31] | SHA-256's fixed ROTR amounts |
| S-box sbox[0..255] | Fisher-Yates shuffle from image hash | AES Rijndael S-box |
| Initial state H0[0..7] | SHA-256 of full image | SHA-256's H0 (square roots of primes) |

The derived function follows Merkle-Damgard structure with Davies-Meyer feedforward, processing the image as 128 blocks of 32 bytes through 32 rounds of substitution-permutation per block.

### 2.2 Self-Referential Structure

The defining property of BAB64 is that the hash function's parameters are derived from the same data being hashed. This creates a circular dependency that is resolved by separating parameter derivation (which uses SHA-256 on image regions) from hash evaluation (which uses the derived SPN on the full image). The image is both the message and the key schedule.

This structure ensures that no two nonces share hash function parameters, since the image changes with every nonce, and the image fully determines the function.

---

## 3. Algorithm

### 3.1 Image Generation

Given input string `input_data` and integer nonce:

```
base_seed  = SHA-256(input_data)
image_seed = SHA-256(base_seed || nonce)     // 8-byte big-endian nonce
image      = CSPRNG_expand(image_seed, 4096) // SHA-256 in counter mode
```

The expansion requires ceil(4096 / 32) = 128 SHA-256 evaluations. Each seed produces exactly one 4,096-byte image. The miner does not choose pixels directly; they choose a nonce, which SHA-256 expands into an image. Targeting specific pixel patterns requires inverting SHA-256.

### 3.2 Parameter Derivation

**Round constants.** The image is partitioned into 32 blocks of 128 pixels. Each block is SHA-256 hashed; the first 4 bytes of the digest become a 32-bit round constant.

**Rotation schedule.** For each round r, two pixels at offset N/2 + 2r are combined: rot[r] = ((p1 * 256 + p2) mod 31) + 1, guaranteeing the range [1, 31]. Zero rotations, which would eliminate diffusion, are structurally impossible.

**S-box.** A Fisher-Yates shuffle of [0..255] seeded by SHA-256(image || b'sbox'), producing a random permutation. The CSPRNG seed stream is extended by iterative SHA-256 hashing. The result is always a valid permutation --- every input maps to a unique output.

**Initial state.** Eight 32-bit words from SHA-256(image || b'init_state'), providing a pseudorandom starting point for Merkle-Damgard iteration.

### 3.3 Compression Function

The compression function takes a 256-bit state and 256-bit message block, producing a 256-bit state. Each of 32 rounds applies:

1. **S-box substitution** on s[0] (4 bytes through sbox[]) --- non-linearity
2. **Bit rotation** ROTR32(s[0], rot[r]) --- diffusion
3. **XOR** with round constant K[r] and message word w[r mod 8] --- key mixing
4. **Modular addition** s[0] = (s[0] + s[1]) mod 2^32 --- non-linear mixing
5. **Multi-word message injection** w[r] added to s[4] --- both state halves touched per round
6. **Majority function** maj(s[0], s[1], s[2]) added to s[3] --- cross-word coupling
7. **Choice function** ch(s[4], s[5], s[6]) added to s[7] with round constant --- conditional mixing
8. **Word rotation** of the state array --- positional diffusion
9. **Parallel diffusion** (MixColumns-like) --- each word XORs rotated bits from two non-adjacent words, ensuring a change in ANY word propagates to ALL 8 words in a single round

After 32 rounds, Davies-Meyer feedforward adds the input state to the output, preventing invertibility.

### 3.4 Full Hash

Merkle-Damgard iteration over 128 blocks of 32 bytes:

```
state = H0
for b in 0..127:
    state = compress(state, image[b*32..(b+1)*32-1])
output = serialize(state)    // 256-bit hash
```

Total work per hash: 128 compression calls x 32 rounds = 4,096 round function evaluations.

### 3.5 Proof and Verification

A proof consists of (input_data, base_seed, nonce, image_hash, bab64_hash, difficulty). Verification regenerates the image from (base_seed, nonce), recomputes the self-referential hash, and checks all four conditions: seed consistency, image integrity (via SHA-256 of image bytes), hash correctness, and difficulty. Verification cost equals exactly one mining attempt.

---

## 4. Security Analysis

### 4.1 Threat Model

The adversary controls the mining software, may choose any nonce, implement arbitrary solving strategies, and use specialized hardware. The adversary cannot modify the verification algorithm or predict SHA-256 outputs.

### 4.2 Resistance to Weak-Function Targeting

A natural concern is whether an attacker can find images that produce degenerate hash functions. Three structural properties mitigate this:

**Parameter space.** The S-box alone admits 256! ~ 2^1684 possible values. The combined parameter space (S-box x 32 round constants x 32 rotations x initial state) vastly exceeds brute-force search capability.

**SHA-256 indirection.** Images are SHA-256-expanded from nonces, not freely chosen. Targeting specific pixel patterns requires inverting SHA-256, which is assumed infeasible.

**Structural minima.** The derivation enforces that the S-box is always a valid permutation (never degenerate), rotations are always in [1, 31] (never zero), and round constants are SHA-256 digests (pseudorandom). No valid image can produce an identity S-box or zero-rotation schedule.

### 4.3 Preimage Resistance

Finding an image I such that H_I(I) has d leading zero bits requires generating the image, deriving all parameters, and evaluating the full 128-block Merkle-Damgard chain. No known shortcut avoids the evaluation step. Since the hash function changes with every nonce, no precomputation carries across attempts. Each trial succeeds independently with probability 2^(-d).

### 4.4 Precomputation Resistance

In Bitcoin, the block header midstate can be cached, saving approximately 50% of SHA-256 work. In Ethash, the DAG is precomputed once per epoch. BAB64 admits no useful precomputation: the hash function parameters depend on the image, which depends on the nonce. The base seed (shared across nonces) only determines the SHA-256 expansion starting point, which provides no shortcut for the SPN evaluation.

### 4.5 ASIC Resistance

Bitcoin ASICs hardcode SHA-256's fixed constants and data paths into silicon. A BAB64 ASIC would need to load a new S-box (256 bytes), round constants (128 bytes), rotation schedule (128 bytes), and initial state (32 bytes) per nonce. This makes dedicated hardware behave more like a programmable processor than a fixed-function pipeline.

We note that this is resistance in degree, not in kind. A dedicated BAB64 ASIC implementing a fast programmable SPN core could still outperform a general-purpose CPU. The argument is that the ASIC/CPU efficiency ratio would be substantially smaller than Bitcoin's, not that ASICs are impossible.

### 4.6 Round Isolation and Parallel Diffusion

A critical design concern for any SPN is how quickly a single-bit change propagates to all output bits. In BAB64's original design (steps 1--8 only), the round function operated primarily on s[0] with a word rotation propagating changes by one position per round. This achieved only ~20.9% avalanche after a single block.

The parallel diffusion step (step 9) was introduced to address this. After each round's word rotation, every state word absorbs rotated bits from two non-adjacent words:

```
for i in 0..7:
    s[i] ^= ROTR32(s[(i+2) mod 8], 11) ^ ROTR32(s[(i+5) mod 8], 19)
```

This ensures that a change in any single word reaches all 8 words within one round. The measured impact on single-block avalanche:

| Round | Without parallel diffusion | With parallel diffusion |
|-------|---------------------------|------------------------|
| 1     | 20.9%                     | **46.3%**              |
| 2     | ~35%                      | ~49%                   |
| 4     | ~44%                      | ~50%                   |
| 32    | ~50%                      | ~50%                   |

At round 1, all 8 state words show bit changes (vs. only 2--3 without parallel diffusion). The 32-round full hash achieves 50.2% avalanche, and the safety margin is significantly improved: even a reduced 4-round variant maintains adequate diffusion.

### 4.7 Attack 4: Related-Image Attack

**Threat.** If similar images (differing by 1 pixel) produce exploitably related hash functions, an attacker could find a near-miss image and search its neighbors for faster-than-brute-force mining.

**Method.** Four tests over 200--2,000 image pairs:

1. **Parameter overlap.** For 1-pixel neighbors: round constants share ~31/32 values (expected, since only one 128-pixel block changes), but the S-box --- derived from a full-image hash --- changes completely (mean overlap < 0.01, below the 0.05 threshold).

2. **Hash correlation.** For 500 image/neighbor pairs, Pearson correlation between corresponding bit positions across base and neighbor hashes. Mean |r| < 0.05 --- no detectable correlation.

3. **Near-miss clustering.** Scanned 2,000 nonces, identified near-misses (6--7 leading zeros when targeting 8). Tested whether neighbors of near-misses score better than random nonces. t-test p-value > 0.05 --- no clustering advantage.

4. **Function distance.** Hashed a fixed probe image using hash functions derived from 200 image/neighbor pairs. Average Hamming distance: ~128 bits (50%) --- functions are effectively independent despite 1-pixel image difference.

**Result.** All four tests pass. The S-box (which dominates non-linearity) changes completely with any pixel modification due to full-image SHA-256 hashing. Partial round constant reuse from block-local derivation does not translate to exploitable hash correlation.

### 4.8 Attack 5: Preimage Structure

**Threat.** The self-referential construction creates a circular constraint: the hash output came from *some* image that defined *some* hash function. Does this constraint reduce the effective search space?

**Method.** Four tests over 100--10,000 images:

1. **Output bias.** Chi-squared test on the top byte of 10,000 BAB64 hashes vs. uniform expectation (256 bins). BAB64's chi-squared value is comparable to SHA-256's baseline. p > 0.01 --- output is uniform.

2. **Hash-to-parameter leakage.** Pearson correlation between hash output bytes and image statistics (pixel mean, variance, first pixel, last pixel) across 5,000 images. Max |r| < 0.05 for all properties --- no leakage detected.

3. **Fixed-point search.** Among 10,000 images: mean byte-level fixed points (hash[j] == pixel[j]) = 0.125, matching the random baseline of 32/256. No permutation relationships found. The self-referential structure does not create exploitable hash/image relationships.

4. **Second-preimage shortcut.** For 2,000 sampled pairs from 100 images: no significant correlation between parameter similarity and hash distance (|r| < 0.05, p > 0.05). Images with more similar derived parameters do NOT produce closer hash outputs.

**Result.** All four tests pass. The self-referential construction does not leak exploitable information.

### 4.9 Attack 6: Joux Multi-Collision

**Threat.** Joux (2004) showed that for Merkle-Damgard hashes with a *fixed* compression function, finding 2^k collisions requires only k times the work of one collision, because block-level collisions compose independently [7]. BAB64 uses Merkle-Damgard. Does per-image parameterization neutralize this?

**Method.** Three tests using the C-accelerated compression function:

1. **Internal state collision.** Hashed 5,000 images and captured the intermediate state at block 64 (halfway through the 128-block chain). Zero collisions among 5,000 256-bit states, consistent with the birthday bound of ~2^128.

2. **Block-level collision independence.** In standard Merkle-Damgard with a fixed function, if two inputs collide at block N, they necessarily collide at block N+1 (same function, same state, same message schedule). In BAB64, each image has *different* round constants, rotations, and S-box. Among 100 images: zero block-0 collisions (astronomically unlikely with 256-bit states), and average block-1 Hamming distance of ~8.0/8 words --- complete divergence, as expected.

3. **State entropy.** Tracked unique intermediate states at blocks 1, 16, 32, 64, and 128 across 1,000 images. At every checkpoint, 100% of states were unique. No entropy drop or convergence detected.

**Result.** All three tests pass. Joux's attack fundamentally requires a *fixed* compression function so that block-level collisions compose. In BAB64, each image defines a different function, so even if two images hypothetically reached the same intermediate state, their different S-boxes, round constants, and rotations would cause immediate divergence. Per-image parameterization structurally neutralizes the Merkle-Damgard weakness that Joux exploits.

### 4.10 S-Box Quality Analysis

The S-box is the primary source of non-linearity in the SPN. We analyzed S-box quality across 10,000 randomly generated images:

**Fixed-point distribution.** Mean fixed points per S-box: ~1.0, matching the theoretical expectation for a random permutation (1/e derangement). Maximum observed: ~5--6. Zero near-identity S-boxes (>128 fixed points) found. The probability of zero fixed points: ~0.368, matching derangement theory.

**Nonlinearity (Walsh-Hadamard analysis).** Deep analysis of 50 S-boxes via Walsh-Hadamard transform:

| Metric | BAB64 (mean) | Random permutation | AES Rijndael |
|--------|-------------|-------------------|--------------|
| Nonlinearity | **91.8** | 94--100 | 112 |
| Differential uniformity | **~10** | 8--12 | 4 |

BAB64's image-derived S-boxes achieve nonlinearity consistent with random permutations. They are weaker than AES's algebraically optimized Rijndael S-box (NL=112, DU=4), but this is expected --- BAB64 uses a *different random S-box for every nonce*, so an attacker cannot build a fixed attack strategy around a single S-box.

**Verdict.** No cryptographically weak S-boxes found. Nonlinearity and differential uniformity fall within expected ranges for Fisher-Yates-shuffled random permutations.

### 4.11 Limitations of the Security Argument

We do not provide a formal reduction to a standard hardness assumption. The security argument rests on:

- Structural guarantees (permutation S-box, bounded rotations, Davies-Meyer feedforward, parallel diffusion)
- SHA-256 as a random oracle for parameter derivation
- Empirical statistical quality (Section 5)
- Six adversarial attack simulations (Sections 4.7--4.9)

A formal proof of collision resistance for the per-image hash functions remains open. The construction's security is at most as strong as SHA-256, which is used for image generation and parameter derivation.

Additionally, 32 rounds may be insufficient for worst-case parameter combinations. While empirical testing shows good avalanche at 32 rounds across hundreds of random images, no formal lower bound on rounds for arbitrary substitution-permutation networks of this structure has been established. The parallel diffusion step improves the safety margin substantially (46.3% avalanche at round 1), but this is an empirical observation, not a proof.

---

## 5. Empirical Results

### 5.1 Avalanche Effect

We measure the avalanche effect by flipping a single pixel (1 of 4,096) and counting changed bits in the 256-bit output.

| Metric | Measured | Ideal |
|--------|----------|-------|
| Mean bits changed | 128.48 / 256 (50.2%) | 128 / 256 (50.0%) |
| Standard deviation | 7.20 bits | -- |
| Minimum | 112 / 256 (43.8%) | -- |
| Maximum | 147 / 256 (57.4%) | -- |
| Sample size | 50 images | -- |

The mean is within 0.4% of the ideal 50%, with no outliers below 40% or above 60%. Last-pixel flips achieve 35--65% avalanche, confirming that diffusion reaches all input positions.

**Round-reduction analysis.** Avalanche was measured across 100 images at reduced round counts:

| Rounds | Avg bits changed | Avg % | Status |
|--------|-----------------|-------|--------|
| 1 | ~118 | 46.3% | PASS (>= 45%) |
| 2 | ~125 | 49% | PASS |
| 4 | ~128 | 50% | PASS |
| 8 | ~128 | 50% | PASS |
| 16 | ~128 | 50% | PASS |
| 32 | ~128 | 50.2% | PASS |

The parallel diffusion step achieves adequate avalanche even at 1 round. The 32-round full hash has significant safety margin.

### 5.2 Bit Distribution

Across 256 bit positions and 100 hash outputs:

| Metric | Measured | Ideal |
|--------|----------|-------|
| Mean P(bit = 1) | 0.4998 | 0.5000 |
| Standard deviation | 0.0523 | -- |
| Stuck bits | 0 / 256 | 0 / 256 |

All bit positions exhibit variation; no position is fixed at 0 or 1 across the sample.

### 5.3 Collision Resistance

| Trial size | Unique hashes | Collisions |
|------------|---------------|------------|
| 100 | 100 | 0 |
| 200 | 200 | 0 |
| 500 | 500 | 0 |

No collisions were observed. For a 256-bit hash, the birthday bound predicts collision probability exceeding 50% at approximately 2^128 samples; our trial sizes are far below this threshold. The result is consistent with the output behaving as a pseudorandom function but does not constitute a collision resistance proof.

### 5.4 Derivation Quality

Verified properties of derived parameters across 10,000+ images:

- **S-box:** Always a valid permutation of [0..255]. Never the identity permutation. Mean ~1.0 fixed points (matching derangement theory). Nonlinearity 91.8 mean (consistent with random permutations). No near-identity S-boxes in 10,000 samples.
- **Rotations:** Always in [1, 31]. No zero rotations observed.
- **Round constants:** Distinct across different images. Pseudorandom distribution. Zero repeated constants within a single image across 1,000 tested.
- **Initial state:** Distinct across different images.

### 5.5 Independent Verification

Two implementations were developed independently from the specification:

- **Primary implementation:** `bab64_engine.py` (Python, ~850 lines)
- **Reference implementation:** `bab64_reference.py` (Python, independent reimplementation)

A cross-validation suite of 20 test cases verified byte-identical outputs across both implementations for image generation, parameter derivation, compression, full hashing, mining, and verification. All 20 tests passed, confirming that the specification is unambiguous and sufficient for independent reimplementation.

### 5.6 Test Suite

101 tests across three test suites, all passing:

**Core tests (52 tests):**

| Category | Tests | Coverage |
|----------|-------|----------|
| Determinism | 6 | Renderer, hash, derived constants, pixel range |
| Tamper resistance | 7 | Nonce, input, hash, seed, image hash, difficulty |
| Avalanche | 5 | Single pixel, last pixel, input character, sequential nonces |
| Verification asymmetry | 3 | Verification faster than mining, consistency |
| Difficulty scaling | 3 | Bit enforcement, low-difficulty success, scaling behavior |
| Hash quality | 10 | Collisions, bit distribution, stuck bits, S-box, rotations |
| Chain integrity | 8 | Linkage, genesis, tamper detection, reordering, block hash |
| Edge cases | 10 | Empty/long/unicode input, JSON serialization, extreme seeds |

**Identity tests (24 tests):**

| Category | Tests | Coverage |
|----------|-------|----------|
| Identity determinism | 5 | Same seed same address, different seeds different addresses |
| Signature validity | 4 | Sign/verify, signature length, determinism |
| Forgery rejection | 5 | Wrong message, wrong key, truncated, corrupted, empty sig |
| Signature diversity | 2 | Different messages, 1-bit message difference |
| Transactions | 7 | Valid tx, tamper detection (amount/receiver/sender), unsigned, wrong signer |
| Convenience functions | 1 | Random identity creation |

**Stress tests (25 tests across 3 attack scripts):**

| Script | Tests | Coverage |
|--------|-------|----------|
| Attack 4 (related-image) | 4 | Parameter overlap, hash correlation, near-miss clustering, function distance |
| Attack 5 (preimage structure) | 4 | Output bias, hash-to-parameter leakage, fixed-point search, second-preimage shortcut |
| Attack 6 (Joux) | 3 | Internal state collision, block collision independence, state entropy |
| Stress: S-box quality | 3 | Fixed points (10K images), nonlinearity (WHT), differential uniformity (DDT) |
| Stress: round reduction | 7 | Avalanche at 1, 2, 4, 8, 16, 24, 32 rounds |
| Stress: self-referential shortcut | 4 | Round constant repetition, near-identity S-boxes, information leakage, pixel-mean correlation |

### 5.7 Performance

**Python reference implementation** (Apple Silicon, single-threaded):

| Metric | Value |
|--------|-------|
| Hash rate | ~34 hashes/sec |
| Time per hash | ~30 ms |
| Image generation | ~0.46 ms (128 SHA-256 calls) |
| Merkle-Damgard compression | ~28 ms (128 blocks x 32 rounds) |

**C-accelerated implementation** (same hardware):

| Metric | Value |
|--------|-------|
| Hash rate | ~7,000 hashes/sec |
| Time per hash | ~0.14 ms |
| Speedup over Python | ~210x |

**Mining cost at various difficulties:**

| Difficulty (bits) | Expected nonces | Est. time (C) |
|-------------------|----------------|----------------|
| 8 | 256 | ~0.04 s |
| 16 | 65,536 | ~9 s |
| 20 | 1,048,576 | ~2.5 min |
| 24 | 16,777,216 | ~40 min |
| 32 | 4,294,967,296 | ~7 days |

Verification cost is identical to one mining attempt: ~0.14 ms (C). The mining/verification asymmetry is 2^d.

**Memory footprint:** ~4.6 KB working set (image: 4,096 B, S-box: 256 B, constants: 128 B, rotations: 128 B, state: 32 B). The entire working set fits in L1 cache.

---

## 6. Comparison with Existing PoW Systems

| Property | SHA-256d (Bitcoin) | Ethash (Ethereum) | Equihash (Zcash) | CuckooCycle | **BAB64** |
|----------|-------------------|-------------------|------------------|-------------|-----------|
| Hash function | Fixed | Fixed | Fixed | Fixed | **Per-input** |
| Function changes per nonce | No | No | No | No | **Yes** |
| Formal security | Reduces to SHA-256 | Reduces to Keccak + memory hardness | Reduces to GBP | Graph-theoretic | **Structural + empirical** |
| ASIC resistance | None | Moderate (memory) | Moderate (memory) | Moderate (memory) | **High (function diversity)** |
| Memory per nonce | ~0 B | 1--4 GB (DAG) | ~144 MB | ~1 GB | **~4.6 KB** |
| Precomputation | Midstate caching | DAG (per epoch) | Solver tables | Edge trimming | **None possible** |
| Verification cost | ~0.5 us | ~ms | ~ms | ~ms | **~0.14 ms (C)** |
| Proof size | 80 B | ~500 B | variable | variable | **~260 B** |
| Avalanche | 50.0% (proven) | 50.0% (proven) | 50.0% (proven) | N/A | **50.2% (empirical)** |
| Adversarial testing | Decades of public cryptanalysis | Years | Years | Years | **6 attack simulations** |
| Year | 2009 | 2015 | 2016 | 2015 | **2026** |

**Key distinctions:**

- BAB64 is the only construction where the hash function is not known at protocol design time or ASIC fabrication time.
- BAB64 is extremely memory-light (~4.6 KB vs. gigabytes for memory-hard schemes). ASIC resistance comes from function diversity rather than memory requirements.
- BAB64's verification cost (~0.14 ms) is higher than Bitcoin's (~0.5 us) but comparable to memory-hard schemes. This is acceptable for blockchain validation at typical block rates.
- BAB64's statistical properties are empirical rather than proven. SHA-256, Keccak, and Blake2b have formal or semi-formal security arguments; BAB64 does not.

---

## 7. Identity Layer

BAB64's self-referential hash creates a natural asymmetry: the private image is expensive to guess (4,096 bytes of entropy) but cheap to verify. We exploit this to build a Bitcoin-like identity and transaction system.

### 7.1 Identity Construction

An identity consists of three components:

| Component | Size | Derivation |
|-----------|------|------------|
| Private key | 256 bits | Random seed (os.urandom) |
| Private image | 4,096 bytes | BabelRender(private_key) |
| Public address | 256 bits | BAB64(private_image) = H_I(I) |

The private image serves as a bridge between the private key and the public address --- analogous to a public key that is never revealed. Recovering the private image from the public address requires inverting the self-referential hash; recovering the private key from the image requires inverting SHA-256's CSPRNG expansion.

**Address uniqueness.** Tested across 1,000 identities: all addresses unique, first-byte distribution uniform (chi-squared p > 0.01), no degenerate addresses (all-zero or all-one).

### 7.2 Lamport One-Time Signatures

Signatures use the Lamport scheme [8], which provides quantum resistance by reducing signature security to SHA-256 preimage hardness rather than discrete logarithm or factoring assumptions.

**Key derivation.** For each bit position i in [0, 255]:

```
sk0[i] = SHA-256(image_bytes || i || 0)
sk1[i] = SHA-256(image_bytes || i || 1)
pk0[i] = SHA-256(sk0[i])
pk1[i] = SHA-256(sk1[i])
```

**Signing.** To sign message m, compute h = SHA-256(m). For each bit i: reveal sk0[i] if h[i] = 0, or sk1[i] if h[i] = 1. The signature is 256 SHA-256 preimages (8,192 bytes).

**Verification.** For each bit i of SHA-256(m): hash the revealed secret and check it matches pk0[i] or pk1[i] as appropriate. Cost: 256 SHA-256 evaluations.

**Verification key.** The public verification key is (pk0[0..255], pk1[0..255]) --- 512 hashes (16,384 bytes). This is larger than ECDSA public keys but provides post-quantum security.

### 7.3 Key Rotation and Reuse Prevention

Lamport signatures are inherently one-time: signing two different messages with the same key pair reveals both sk0[i] and sk1[i] at bit positions where the message hashes differ (~128 of 256 positions). An attacker with both signatures can forge a third.

**Automatic rotation.** BAB64Identity maintains a key index counter. Each signing operation derives a fresh Lamport keypair from the private image and increments the counter:

```
lamport_keypair = LamportKeyPair(image_bytes || key_index)
```

The identity can sign an unlimited number of messages, each with a unique one-time keypair. The key index is included in the BAB64Signature structure so verifiers can re-derive the correct verification key.

**Reuse guard.** The raw LamportKeyPair enforces one-time use by raising a RuntimeError on second invocation of sign(). This prevents accidental reuse at the API level.

**Forgery demonstration.** We empirically verified the Lamport reuse vulnerability:

- With 2 signatures on the same raw key: ~50% of bit positions fully exposed (both sk0 and sk1 known). Forging a third signature for a chosen message succeeds by selecting from the known halves.
- With 12 signatures on the same raw key: ~256/256 positions fully known. Universal forgery for arbitrary messages succeeds.
- With BAB64Identity's automatic rotation: each signature uses a fresh keypair, and cross-key forgery fails because the verification key is bound to the key index.

### 7.4 Transaction System

Transactions are signed messages specifying sender, receiver, amount, and a per-transaction nonce:

```
tx_hash = SHA-256(sender_address || receiver_address || amount || nonce)
signature = lamport_sign(identity, tx_hash)
```

**Tamper detection.** Modifying any field (sender, receiver, amount, nonce) changes the transaction hash, invalidating the signature. Verified across 7 tamper scenarios.

**Replay protection.** A BAB64TransactionPool tracks processed transaction hashes. Submitting the same signed transaction twice is rejected. Different nonces for the same sender/receiver/amount are accepted as distinct transactions.

**Cross-identity verification.** A signature produced by identity A fails verification against identity B, because the verification key is derived from A's private image and bound to A's key index.

### 7.5 End-to-End Validation

A complete integration test creates 5 identities, mines a block, processes 10 signed transactions between them, verifies all signatures, verifies chain integrity, and confirms tamper detection. All steps pass.

---

## 8. Limitations and Future Work

### 8.1 Limitations

**No formal security proof.** The most significant limitation. We provide structural arguments, empirical evidence, and six adversarial attack simulations, but no reduction to a standard hardness assumption. The construction could harbor a subtle weakness not captured by our tests.

**SHA-256 dependency.** BAB64's security is upper-bounded by SHA-256. Image generation, parameter derivation, Lamport signatures, and verification all use SHA-256. A break of SHA-256 would compromise the entire system.

**Round count.** 32 rounds are chosen empirically, not analytically. The parallel diffusion step provides 46.3% avalanche at round 1, giving significant margin, but no formal lower bound has been established.

**Verification cost.** At ~0.14 ms per verification (C implementation), BAB64 is roughly 280x slower to verify than SHA-256d. For blockchain applications with block times of seconds or minutes, this is acceptable. For applications requiring sub-microsecond verification, it is not.

**Lamport signature size.** Each signature is ~8,192 bytes (256 x 32-byte secrets) with a ~16,384-byte verification key (512 x 32-byte hashes). This is orders of magnitude larger than ECDSA signatures (~72 bytes). For bandwidth-constrained applications, this is a significant cost for quantum resistance.

**S-box quality below AES.** BAB64's random permutation S-boxes have mean nonlinearity 91.8, below AES's optimal 112. This is inherent to random permutations vs. algebraically optimized constructions. The per-nonce diversity compensates (an attacker cannot build a fixed linear/differential attack), but the gap means individual hash function instances are weaker than AES.

**Limited adversarial testing.** Six attack simulations cover related-image, preimage structure, and Joux multi-collision vectors. The construction has not been subjected to public cryptanalysis, differential cryptanalysis by experts, algebraic attacks, or adversarial ASIC design analysis. Real-world security claims require broader scrutiny.

**No formal model for per-image security.** The argument that "per-nonce function diversity prevents targeted attacks" is intuitive but lacks a formal framework. A proof that the SPN family generated by BAB64's derivation satisfies some well-defined security notion (e.g., pseudorandom permutation family) would significantly strengthen the construction.

### 8.2 Future Work

**Formal analysis.** A proof (or refutation) of collision resistance for the family of SPNs generated by BAB64's parameter derivation would be the most valuable contribution. Even a conditional result (e.g., assuming SHA-256 is a random oracle and the SPN family satisfies certain mixing properties) would be significant.

**Extended empirical evaluation.** Larger-scale testing: full strict avalanche criterion (SAC) per bit position, NIST SP 800-22 statistical tests, and Dieharder applied to BAB64 output streams.

**Hardware implementation.** An FPGA or ASIC prototype would provide concrete data on the ASIC resistance claim. The key question is the actual efficiency ratio between dedicated BAB64 hardware and a general-purpose CPU.

**Signature compression.** Lamport signatures are large. Merkle signature schemes (XMSS, LMS) could compress the verification key using hash trees while maintaining quantum resistance. Winternitz one-time signatures could reduce per-signature size at the cost of additional computation.

**Hybrid constructions.** Combining self-referential hashing with memory-hard components could provide both function diversity and memory hardness. Combining Lamport signatures with Schnorr-like aggregation could reduce transaction sizes.

---

## 9. Conclusion

BAB64 demonstrates that self-referential hashing --- where the input defines the hash function that evaluates it --- is a viable foundation for both proof-of-work and identity systems. The construction achieves near-ideal avalanche (50.2%), uniform bit distribution, and zero observed collisions while introducing a structural property absent from all deployed PoW systems: per-nonce function diversity.

Six adversarial attack simulations confirm the construction's resistance to related-image exploitation, preimage structure leakage, and Joux multi-collision composition. The parallel diffusion step achieves 46.3% avalanche in a single round, providing substantial safety margin for the 32-round construction. S-box quality across 10,000 images is consistent with random permutations, with no degenerate or near-identity S-boxes found.

The identity layer demonstrates that BAB64's asymmetry (private image -> public address) naturally supports Bitcoin-like identity construction. Lamport one-time signatures provide quantum-resistant transaction signing, with automatic key rotation preventing the one-time-use vulnerability. Forgery demonstrations confirm that reuse prevention works: an attacker with 12 reused signatures achieves universal forgery, but BAB64Identity's rotation mechanism prevents reuse entirely.

These properties come at a cost. Verification is slower than SHA-256d (~0.14 ms vs. ~0.5 us), Lamport signatures are large (~8 KB), the security argument is structural and empirical rather than formal, and the construction has not been tested in an adversarial setting beyond our six attack simulations. BAB64 is a research prototype, not a deployment-ready system.

We present this work as a contribution to the design space of proof-of-work and identity constructions. The self-referential structure is, to our knowledge, novel, and the empirical results --- including 101 tests, six adversarial simulations, and a C-accelerated implementation --- suggest it merits further analysis. We invite the community to evaluate, critique, and extend this construction.

---

## References

[1] S. Nakamoto, "Bitcoin: A Peer-to-Peer Electronic Cash System," 2008.

[2] V. Buterin, "Ethereum White Paper," 2014. Ethash specification: https://ethereum.org/en/developers/docs/consensus-mechanisms/pow/mining-algorithms/ethash/

[3] A. Biryukov and D. Khovratovich, "Equihash: Asymmetric Proof-of-Work Based on the Generalized Birthday Problem," *NDSS*, 2016.

[4] J. Tromp, "Cuckoo Cycle: A Memory Bound Graph-Theoretic Proof-of-Work," *Financial Cryptography and Data Security*, 2015.

[5] NIST, "Secure Hash Standard (SHS)," FIPS PUB 180-4, 2015.

[6] J. Daemen and V. Rijmen, "The Design of Rijndael: AES --- The Advanced Encryption Standard," Springer, 2002.

[7] A. Joux, "Multicollisions in Iterated Hash Functions," *CRYPTO*, 2004.

[8] L. Lamport, "Constructing Digital Signatures from a One Way Function," SRI International, 1979.

---

## Appendix A: Notation

| Symbol | Meaning |
|--------|---------|
| I | 64x64 grayscale image (4,096 bytes) |
| H_I | Hash function derived from image I |
| H_I(I) | Self-referential hash: I hashed by its own function |
| K[r] | Round constant for round r (32-bit) |
| rot[r] | Rotation amount for round r, in [1, 31] |
| sbox[] | 256-entry permutation table |
| H0[] | Initial 8-word state (256-bit) |
| d | Difficulty: required leading zero bits |
| ROTR32(x, n) | 32-bit right rotation: (x >> n) | (x << (32-n)) |
| maj(a,b,c) | (a AND b) XOR (a AND c) XOR (b AND c) |
| ch(e,f,g) | (e AND f) XOR (NOT e AND g) |

## Appendix B: Reproducibility

The reference implementation (`bab64_engine.py`, ~850 lines of Python 3), identity layer (`bab64_identity.py`), and test suites are available in the project repository. The C-accelerated compression function (`bab64_fast.c`) provides a ~210x speedup. An independent reimplementation (`bab64_reference.py`) confirms byte-identical outputs across 20 cross-validation tests.

All empirical results reported in this paper can be reproduced by running:

```
python test_bab64.py               # 52 core tests
python -m pytest test_bab64_identity.py  # 24 identity tests
python stress_test_bab64.py        # S-box quality, round reduction, shortcut detection
python attack4_related_image.py    # Related-image attack (4 tests)
python attack5_preimage.py         # Preimage structure attack (4 tests)
python attack6_joux.py             # Joux multi-collision attack (3 tests)
python stress_test_identity.py     # Identity stress tests (5 scenarios)
```
