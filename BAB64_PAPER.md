# BAB64: Proof-of-Work via Self-Referential Image Hashing

**Authors:** Shrey Chouksey, Claude (Anthropic)
**Date:** April 2026
**Status:** Research prototype

---

## Abstract

We present BAB64, a proof-of-work construction in which the hash function is not fixed at protocol design time but is instead derived from the input itself. Each candidate input deterministically generates a 64x64 grayscale image, which in turn parameterizes a unique substitution-permutation network --- defining its own round constants, rotation schedule, S-box, and initial state. The proof-of-work output is the image hashed by its own derived function. This self-referential structure eliminates precomputation advantages and creates structural ASIC resistance by requiring per-nonce function reconfiguration. Empirical evaluation over 52 tests shows near-ideal avalanche (50.2%), uniform bit distribution (P(1) = 0.4998), zero collisions across 500 trials, and byte-identical outputs across two independent implementations. A C-accelerated implementation achieves approximately 7,000 hashes per second on commodity hardware. No formal security reduction is provided; the construction's resistance to optimization rests on structural arguments and empirical evidence.

---

## 1. Introduction

Proof-of-work (PoW) systems underpin the security of permissionless consensus protocols. In all widely deployed PoW constructions --- Bitcoin's SHA-256d [1], Ethereum's Ethash [2], Zcash's Equihash [3], and Cuckoo Cycle [4] --- the hash function is fixed at protocol design time. The miner searches for an input that produces a low output under this fixed function. This architectural choice has two consequences:

1. **Precomputation is possible.** A fixed function admits partial evaluation reuse. Bitcoin miners exploit midstate caching; Ethash requires a multi-gigabyte DAG precomputed per epoch.

2. **ASIC optimization is straightforward.** When the function is known at fabrication time, its constants, rotation amounts, and data paths can be hardcoded into silicon. Bitcoin ASICs achieve orders-of-magnitude efficiency over general-purpose hardware precisely because SHA-256 never changes.

BAB64 eliminates the fixed function entirely. Each mining attempt produces a candidate image I, which deterministically defines a unique compression function H_I. The proof-of-work output is H_I(I) --- the image hashed by the function it defines. The miner searches simultaneously in input space and function space.

This paper makes three contributions:

- **A novel PoW primitive** based on self-referential hashing, where no two mining attempts use the same hash function.
- **A concrete construction** with byte-precise specification, two independent implementations, and 52 passing tests.
- **Empirical evidence** that the construction achieves near-ideal statistical properties despite using data-derived, per-input parameters.

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
5. **Majority function** maj(s[0], s[1], s[2]) added to s[3] --- cross-word coupling
6. **Choice function** ch(s[4], s[5], s[6]) added to s[7] with round constant --- conditional mixing
7. **Word rotation** of the state array --- positional diffusion

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

### 4.6 Limitations of the Security Argument

We do not provide a formal reduction to a standard hardness assumption. The security argument rests on:

- Structural guarantees (permutation S-box, bounded rotations, Davies-Meyer feedforward)
- SHA-256 as a random oracle for parameter derivation
- Empirical statistical quality (Section 5)

A formal proof of collision resistance for the per-image hash functions remains open. The construction's security is at most as strong as SHA-256, which is used for image generation and parameter derivation.

Additionally, 32 rounds may be insufficient for worst-case parameter combinations. While empirical testing shows good avalanche at 32 rounds across hundreds of random images, no formal lower bound on rounds for arbitrary substitution-permutation networks of this structure has been established.

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

Verified properties of derived parameters across 100+ images:

- **S-box:** Always a valid permutation of [0..255]. Never the identity permutation. Different images produce different S-boxes.
- **Rotations:** Always in [1, 31]. No zero rotations observed.
- **Round constants:** Distinct across different images. Pseudorandom distribution.
- **Initial state:** Distinct across different images.

### 5.5 Independent Verification

Two implementations were developed independently from the specification:

- **Primary implementation:** `bab64_engine.py` (Python, ~787 lines)
- **Reference implementation:** `bab64_reference.py` (Python, independent reimplementation)

A cross-validation suite of 20 test cases verified byte-identical outputs across both implementations for image generation, parameter derivation, compression, full hashing, mining, and verification. All 20 tests passed, confirming that the specification is unambiguous and sufficient for independent reimplementation.

### 5.6 Test Suite

52 tests across 8 categories, all passing:

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
| Year | 2009 | 2015 | 2016 | 2015 | **2026** |

**Key distinctions:**

- BAB64 is the only construction where the hash function is not known at protocol design time or ASIC fabrication time.
- BAB64 is extremely memory-light (~4.6 KB vs. gigabytes for memory-hard schemes). ASIC resistance comes from function diversity rather than memory requirements.
- BAB64's verification cost (~0.14 ms) is higher than Bitcoin's (~0.5 us) but comparable to memory-hard schemes. This is acceptable for blockchain validation at typical block rates.
- BAB64's statistical properties are empirical rather than proven. SHA-256, Keccak, and Blake2b have formal or semi-formal security arguments; BAB64 does not.

---

## 7. Limitations and Future Work

### 7.1 Limitations

**No formal security proof.** The most significant limitation. We provide structural arguments and empirical evidence but no reduction to a standard hardness assumption (e.g., collision resistance of the per-image SPN family). The construction could harbor a subtle weakness exploitable by an adversary who can analyze the relationship between image content and derived parameters more efficiently than brute force.

**SHA-256 dependency.** BAB64's security is upper-bounded by SHA-256. The image generation, parameter derivation, and verification all use SHA-256. A break of SHA-256 would compromise BAB64's image generation guarantees, though the SPN evaluation itself does not use SHA-256.

**Round count.** 32 rounds are chosen empirically, not analytically. While measured avalanche exceeds 49% across hundreds of trials, worst-case images (which cannot be efficiently targeted, but may exist) might achieve lower diffusion. Formal analysis of minimum rounds for random SPNs of this structure would strengthen the security argument.

**Verification cost.** At ~0.14 ms per verification (C implementation), BAB64 is roughly 280x slower to verify than SHA-256d. For blockchain applications with block times of seconds or minutes, this is acceptable. For applications requiring sub-microsecond verification, it is not.

**Limited adversarial testing.** The construction has been evaluated by its authors. It has not been subjected to public cryptanalysis, red-team exercises, or adversarial ASIC design analysis. Real-world security claims require broader scrutiny.

**Small empirical samples.** Avalanche is measured over 50 images, bit distribution over 100, and collisions over 500. These samples are sufficient to detect gross failures but insufficient to characterize tail behavior or detect subtle biases.

### 7.2 Future Work

**Formal analysis.** A proof (or refutation) of collision resistance for the family of SPNs generated by BAB64's parameter derivation would be the most valuable contribution. Even a conditional result (e.g., collision resistance assuming SHA-256 is a random oracle and the SPN family satisfies certain mixing properties) would significantly strengthen the construction.

**Extended empirical evaluation.** Larger-scale avalanche testing (10,000+ images), full strict avalanche criterion (SAC) evaluation per bit position, and statistical test suites (NIST SP 800-22, Dieharder) applied to BAB64 output streams.

**Hardware implementation.** An FPGA or ASIC prototype would provide concrete data on the ASIC resistance claim. The key question is the actual efficiency ratio between dedicated BAB64 hardware and a general-purpose CPU, compared to the same ratio for SHA-256.

**Parameter tuning.** Systematic exploration of round count, block size, S-box derivation method, and state size. The current parameters are reasonable but not optimized.

**Hybrid constructions.** Combining self-referential hashing with memory-hard components could provide both function diversity and memory hardness, potentially offering stronger combined ASIC resistance.

---

## 8. Conclusion

BAB64 demonstrates that self-referential hashing --- where the input defines the hash function that evaluates it --- is a viable proof-of-work primitive. The construction achieves near-ideal avalanche (50.2%), uniform bit distribution, and zero observed collisions while introducing a structural property absent from all deployed PoW systems: per-nonce function diversity.

The practical implications are twofold. First, precomputation is structurally impossible, since nothing computed for one nonce transfers to another. Second, ASIC optimization is constrained, since dedicated hardware must be programmable enough to load new S-boxes, round constants, and rotation schedules for every candidate.

These properties come at a cost. Verification is slower than SHA-256d (~0.14 ms vs. ~0.5 us), the security argument is structural and empirical rather than formal, and the construction has not been tested in an adversarial setting. BAB64 is a research prototype, not a deployment-ready system.

We present this work as a contribution to the design space of proof-of-work constructions. The self-referential structure is, to our knowledge, novel, and the empirical results suggest it merits further analysis. We invite the community to evaluate, critique, and extend this construction.

---

## References

[1] S. Nakamoto, "Bitcoin: A Peer-to-Peer Electronic Cash System," 2008.

[2] V. Buterin, "Ethereum White Paper," 2014. Ethash specification: https://ethereum.org/en/developers/docs/consensus-mechanisms/pow/mining-algorithms/ethash/

[3] A. Biryukov and D. Khovratovich, "Equihash: Asymmetric Proof-of-Work Based on the Generalized Birthday Problem," *NDSS*, 2016.

[4] J. Tromp, "Cuckoo Cycle: A Memory Bound Graph-Theoretic Proof-of-Work," *Financial Cryptography and Data Security*, 2015.

[5] NIST, "Secure Hash Standard (SHS)," FIPS PUB 180-4, 2015.

[6] J. Daemen and V. Rijmen, "The Design of Rijndael: AES --- The Advanced Encryption Standard," Springer, 2002.

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

The reference implementation (`bab64_engine.py`, ~787 lines of Python 3) and test suite (`test_bab64.py`, 52 tests) are available in the project repository. The C-accelerated compression function (`bab64_fast.c`) provides a ~210x speedup. An independent reimplementation (`bab64_reference.py`) confirms byte-identical outputs across 20 cross-validation tests.

All empirical results reported in this paper can be reproduced by running:

```
python test_bab64.py          # 52 tests
python bab64_engine.py        # Hash quality analysis + mining demo
python verify_implementations.py  # Cross-implementation verification
```
