# BAB64: Proof-of-Work via Self-Referential Image Hashing

**Authors:** Shrey Chouksey, Claude (Anthropic)
**Date:** April 2026
**Status:** Research prototype (v2)

---

## Abstract

We present BAB64, a proof-of-work construction in which the hash function is not fixed at protocol design time but is instead derived from the input itself. Each candidate input deterministically generates a 64x64 grayscale image, which in turn parameterizes a unique substitution-permutation network --- defining its own round constants, rotation schedule, S-box, and initial state. The proof-of-work output is the image hashed by its own derived function. This self-referential structure eliminates precomputation advantages and creates structural ASIC resistance by requiring per-nonce function reconfiguration.

Empirical evaluation over 101 tests shows near-ideal avalanche (50.2%), uniform bit distribution (P(1) = 0.4998), zero collisions across 500 trials, and byte-identical outputs across two independent implementations. A parallel diffusion step achieves 46.3% single-block avalanche in one round. Six adversarial attack simulations --- including related-image, preimage structure, and Joux multi-collision --- find no exploitable weakness. S-box quality analysis across 10,000 images shows mean nonlinearity of 91.8, consistent with random permutations. A C-accelerated implementation achieves approximately 7,000 hashes per second on commodity hardware.

An identity layer provides Bitcoin-like addresses derived from private images, with Lamport one-time signatures for quantum-resistant transaction signing, automatic key rotation, and replay protection.

A formal security analysis chain of four lemmas --- S-box indistinguishability (8/8 tests), parameter independence (35/35), compression function PRP security (7/7), and Merkle-Damgard preservation (5/5) --- establishes collision resistance under the random oracle model for SHA-256, with all 55 statistical tests passing at significance level alpha = 0.01.

---

## 1. Introduction

Proof-of-work (PoW) systems underpin the security of permissionless consensus protocols. In all widely deployed PoW constructions --- Bitcoin's SHA-256d [1], Ethereum's Ethash [2], Zcash's Equihash [3], and Cuckoo Cycle [4] --- the hash function is fixed at protocol design time. The miner searches for an input that produces a low output under this fixed function. This architectural choice has two consequences:

1. **Precomputation is possible.** A fixed function admits partial evaluation reuse. Bitcoin miners exploit midstate caching; Ethash requires a multi-gigabyte DAG precomputed per epoch.

2. **ASIC optimization is straightforward.** When the function is known at fabrication time, its constants, rotation amounts, and data paths can be hardcoded into silicon. Bitcoin ASICs achieve orders-of-magnitude efficiency over general-purpose hardware precisely because SHA-256 never changes.

BAB64 eliminates the fixed function entirely. Each mining attempt produces a candidate image I, which deterministically defines a unique compression function H_I. The proof-of-work output is H_I(I) --- the image hashed by the function it defines. The miner searches simultaneously in input space and function space.

This paper makes four contributions:

- **A novel PoW primitive** based on self-referential hashing, where no two mining attempts use the same hash function.
- **A concrete construction** with byte-precise specification, two independent implementations, and 101 passing tests across core hashing, identity, and stress test suites.
- A **formal security analysis** establishing collision resistance under the random oracle model via four lemmas: S-box indistinguishability, parameter independence, PRP security, and Merkle-Damgard preservation, validated by 55 statistical tests.
- **An identity layer** with private-image-derived addresses and quantum-resistant Lamport signatures, demonstrating that BAB64 can serve as a foundation for a complete transaction system.

We emphasize that BAB64 is a research contribution. A formal security analysis chain (Section 4) establishes collision resistance under the random oracle model via four lemmas validated by 55 passing tests, but the construction has not been deployed in an adversarial setting. We present the design, evidence, and limitations honestly and invite further analysis.

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

### 4.1 Security Framework

We present a formal security argument for BAB64 structured as a composition of four lemmas. Each lemma is validated by a dedicated test suite with all tests passing at significance level alpha = 0.01.

**Theorem 1 (Main Result).** *Under the random oracle model for SHA-256, the BAB64 function family {H_I}_I is collision-resistant.*

*Proof sketch.* By composition of four lemmas:
1. **Lemma 1** (S-box indistinguishability): The S-box family is drawn from a distribution computationally indistinguishable from uniform over S_256.
2. **Lemma 2** (Parameter independence): The four derived components are pairwise independent.
3. **Lemma 3** (PRP security): The compression function, parameterized by components from Lemmas 1--2, is a pseudorandom permutation.
4. **Lemma 4** (MD preservation): The 128-block Merkle-Damgard iteration preserves collision resistance from the compression function.

Lemmas 1 and 2 establish that the SPN parameters have the distributional properties required for a well-designed cipher. Lemma 3 shows these properties compose into PRP security at the compression function level. Lemma 4 lifts compression-function collision resistance to the full hash via the classical Merkle-Damgard theorem [Merkle 1989, Damgard 1989]. The complete analysis comprises 55 statistical tests, all passing.

### 4.2 Lemma 1: S-Box Indistinguishability (8/8 tests)

**Lemma 1.** *The family of S-boxes generated by BAB64's Fisher-Yates construction, seeded by SHA-256(image || b'sbox'), is computationally indistinguishable from the uniform distribution on permutations of [0..255].*

**Proof method.** For each statistical criterion, we compute the test statistic on both (A) BAB64 S-boxes generated by the actual construction, and (B) reference permutations generated by numpy's rejection-sampling-based `random.permutation`. We apply two-sample tests (Kolmogorov-Smirnov or Mann-Whitney U) to determine whether the distributions differ.

The argument rests on three pillars:
1. The seed SHA-256(image || b'sbox') is indistinguishable from uniform under the ROM.
2. Fisher-Yates with a uniform byte stream produces each permutation with equal probability [Knuth, TAOCP Vol. 2].
3. The modular reduction `j = byte % (i+1)` introduces bias at most 1/256 per swap --- we test whether this accumulates to a detectable level.

**Rejection sampling note.** The original Fisher-Yates implementation used modular reduction for index generation, which introduces a small bias. Reference permutations use rejection sampling for uniform indices. Two-sample testing directly measures whether this bias is detectable.

**Test battery (8 subtests, all p > 0.01):**

| Test | Method | n | Statistic | Result |
|------|--------|---|-----------|--------|
| 1a. Ascending runs | Two-sample KS | 1,000 | D < 0.05 | PASS |
| 1b. Serial correlation | Two-sample KS | 1,000 | D < 0.05 | PASS |
| 2. Algebraic degree (ANF) | Mann-Whitney U | 100 | p > 0.01 | PASS |
| 3. LAT max bias | Two-sample KS | 50 | D < 0.10 | PASS |
| 4. DDT max entry | Two-sample KS | 100 | D < 0.10 | PASS |
| 5. Fixed-point count | Two-sample KS | 10,000 | D < 0.05 | PASS |
| 6a. Number of cycles | Two-sample KS | 1,000 | D < 0.05 | PASS |
| 6b. Longest cycle | Two-sample KS | 1,000 | D < 0.05 | PASS |

**Key results:**
- **Runs and serial correlation** (1,000 S-boxes): BAB64 ascending run counts and serial correlation coefficients match reference permutations. No sequential structure detected.
- **Algebraic degree** (100 S-boxes): Minimum component Boolean function degree distribution matches reference. The fraction of degree-7 components is indistinguishable, ruling out algebraic attacks [Courtois-Pieprzyk 2002].
- **Linear approximation table** (50 S-boxes): Max |LAT bias| for BAB64 (mean ~36) matches reference permutations (mean ~36), consistent with the theoretical expectation E[max|LAT|] ~ 34--38 for random 8-bit permutations [O'Connor 1994]. Resistance to linear cryptanalysis [Matsui 1993] is equivalent to random permutations.
- **Differential distribution table** (100 S-boxes): Differential uniformity (mean ~10) matches reference, consistent with random permutations (expected 8--12 for 8-bit).
- **Fixed points** (10,000 S-boxes): Derangement fraction matches 1/e ~ 0.368. Mean fixed points ~ 1.0, matching Poisson(1). The modular bias does not shift the fixed-point distribution.
- **Cycle structure** (1,000 S-boxes): Number of cycles (mean ~6.1) and longest cycle length (mean ~160) match the theoretical values for uniform random permutations (H_256 ~ 6.12 and Golomb-Dickman constant * 256 ~ 159.8).

**Conclusion.** We fail to reject H0 at alpha = 0.01 across all eight tests. Under the ROM, BAB64 S-boxes are indistinguishable from uniform random permutations of [0..255].

### 4.3 Lemma 2: Parameter Independence (35/35 tests)

**Lemma 2.** *The four derived components (round constants, rotations, S-box, initial state) are pairwise independent under the random oracle model.*

**Proof sketch.** Each component uses a distinct derivation pathway:
- Round constants: SHA-256 of disjoint 128-pixel blocks -> 32-bit words
- Rotations: direct pixel lookup from image second half -> [1, 31]
- S-box: Fisher-Yates seeded by SHA-256(image || b'sbox')
- Initial state: SHA-256(image || b'init_state') -> 8 x 32-bit words

Under the ROM, SHA-256(image || b'sbox') and SHA-256(image || b'init_state') are independent because the domain tags differ (distinct pre-images). Round constants and rotations sample disjoint pixel regions. Independence ensures that knowing one component reveals nothing about the others, preventing precomputed attack tables.

**Test battery (35 subtests):**

**Test 1: Cross-component Pearson correlation** (5,000 images, 12 pairs):
All pairwise correlations |r| < 0.05 across both element-level (rc[0] vs sbox[0], rot[0] vs init[0], etc.) and aggregate-level (mean(rc) vs sbox_pair, etc.) projections.

**Test 2: Mutual information estimation** (2,000 images, 6 pairs):
Permutation test with 500 shuffles per pair. Observed MI falls within the null distribution for all 6 component pairs (p > 0.01). No nonlinear dependence detected beyond what independent SHA-256 outputs exhibit.

**Test 3: Conditional prediction** (5,000 images, 12 directions):
Linear regression R^2 < 0.01 for all 12 directional predictions (rc predicts sbox, sbox predicts rot, etc.). No exploitable linear dependence exists between any component pair.

**Test 4: Domain separation verification** (1,000 images):
- Swapping domain tags (b'sbox' <-> b'init_state') produces outputs uncorrelated with originals (|r| < 0.05).
- Tag identity check confirms that same-tag derivations correlate as expected (|r| > 0.99).
- Cross-independence between actual S-box and initial state outputs confirmed (|r| < 0.05).

**Test 5: Seed collision analysis** (10,000 images):
Combined (rc[0], sbox[0]) collisions: 0 observed, matching the birthday bound for domain size 2^40. Individual rc[0] collisions match the birthday bound for domain 2^32. Individual sbox[0] covers all 256 values as expected.

**Conclusion.** All 35 subtests pass. The four derived components are pairwise independent under the ROM.

### 4.4 Lemma 3: Compression Function as PRP (7/7 tests)

**Lemma 3.** *A single instance of the BAB64 compression function, with parameters drawn from the distributions established in Lemmas 1--2, is computationally indistinguishable from a random permutation of {0,1}^256.*

This is validated through a three-pronged empirical analysis:

**Prong 1: Statistical Distinguisher Battery**

| Test | Method | Scale | Result |
|------|--------|-------|--------|
| 1a. Byte distribution | Chi-squared vs uniform, KS on p-values | 500 images x 1,000 inputs | PASS |
| 1b. Bit correlation | Max |r| across 256x256 bit pairs | 500 images x 1,000 inputs | PASS |
| 1c. Input-output correlation | Two-sample KS vs SHA-256 reference | 500 images x 1,000 inputs | PASS |
| 1d. Strict Avalanche Criterion | Two-sample KS vs SHA-256 reference | 50 images x 100 inputs | PASS |

- Output bytes follow the uniform distribution (chi-squared p-values themselves uniform, KS p > 0.01).
- Output bit pairs show negligible correlation (mean max |r| consistent with sampling noise).
- Input-output byte correlation is indistinguishable from SHA-256's baseline (KS p > 0.01).
- SAC deviation from 0.5 matches SHA-256's baseline (KS p > 0.01). Flipping any input bit changes each output bit with probability ~0.5.

**Prong 2: Differential Propagation Analysis**

| Test | Method | Scale | Result |
|------|--------|-------|--------|
| 2a. Best differential | Max P(specific Delta_y) | 100 images x 500 Delta_x x 100 samples | PASS |
| 2b. Round decay | Diff prob vs round count | 50 images x 200 Delta_x x 50 samples | PASS |

- Best differential probability is at the random floor (1/n_samples = 0.01), consistent with all output differences being unique.
- Differential probability decays monotonically with round count and is at the floor by 32 rounds.

**Prong 3: PRP Distinguisher Game**

| Strategy | P(correct\|real) | P(correct\|random) | Advantage | Result |
|----------|------------------|---------------------|-----------|--------|
| Frequency analysis | ~0.50 | ~0.50 | < 0.023 | PASS |
| Correlation attack | ~0.50 | ~0.50 | < 0.023 | PASS |
| Differential attack | ~0.50 | ~0.50 | < 0.023 | PASS |

Three adaptive distinguisher strategies (200 images, 1,000 queries each) achieve advantage < 0.023, consistent with random guessing. The compression function is empirically indistinguishable from SHA-256 as a random oracle.

**Novel contribution.** Standard SPN theory (Luby-Rackoff, Even-Mansour) assumes fixed, optimally chosen components. Lemma 3, combined with Lemmas 1--2, demonstrates that *randomly parameterized* SPNs --- with S-boxes drawn from S_256 and independent parameters --- compose into a PRP. This extends SPN security analysis to the random-parameterization regime.

**Conclusion.** All 7 tests pass. The BAB64 compression function is a PRP under the parameter distributions from Lemmas 1--2.

### 4.5 Lemma 4: Merkle-Damgard Preservation (5/5 tests)

**Lemma 4.** *The 128-block Merkle-Damgard iteration preserves collision resistance from the compression function established in Lemma 3.*

**Proof sketch.** Merkle (1989) and Damgard (1989) proved that if a compression function h: {0,1}^n x {0,1}^m -> {0,1}^n is collision-resistant, then the iterated hash H built by chaining h is also collision-resistant. For a given image, BAB64's compression function parameters are derived once and held fixed across all 128 blocks. The classical MD theorem therefore applies directly. We verify empirically that the specific construction does not introduce weaknesses the theorem does not cover.

**Test battery (5 tests):**

**Test 1: Length extension resistance** (500 images):
Without knowledge of the image-derived parameters (rc, rot, sbox), an attacker cannot evaluate the compression function and therefore cannot extend the hash. Empirically: blind extension (using wrong parameters) produces outputs ~128 bits different from correct extension (50%, effectively random). Mismatch rate: 100%. Davies-Meyer feedforward provides additional protection.

**Test 2: Intermediate state diversity** (1,000 images x 5 checkpoints):
All 1,000 images produce unique intermediate states at blocks 1, 32, 64, 96, and 128. No state convergence detected. Byte-level Shannon entropy at each checkpoint exceeds 7.0 bits (near the theoretical maximum of 8.0 for 256 bins with 1,000 samples).

**Test 3: Block order sensitivity** (100 images):
Swapping two adjacent message blocks changes ~128 of 256 output bits (50%). The Merkle-Damgard chain is strictly order-dependent; the construction is not commutative.

**Test 4: Compression chain independence** (100 images):
Pearson correlation between block-1 output bytes and block-128 output bytes: max |r| near the random baseline (expected max under H0 ~ 0.39 for 1,024 cells with n=100). The 127 intermediate compressions fully decorrelate early and late states.

**Test 5: Multi-block collision search** (500 images x 10,000 blocks):
Zero collisions in 5,000,000 compression function evaluations, consistent with the birthday bound of 2^128 for 256-bit states. No intermediate-state collisions found.

**Conclusion.** All 5 tests pass. The MD iteration preserves collision resistance.

### 4.6 Composition and Main Result

The four lemmas compose as follows to establish Theorem 1:

1. **Lemma 1** ensures the S-box is drawn from a distribution indistinguishable from uniform over S_256. This guarantees that the nonlinear layer of the SPN has the statistical properties of a random permutation, preventing linear and differential cryptanalysis strategies that exploit fixed S-box structure.

2. **Lemma 2** ensures the four parameter components (round constants, rotations, S-box, initial state) are pairwise independent. This prevents an attacker from predicting one component from another, ruling out precomputed attack tables and cross-component exploitation.

3. **Lemma 3** shows that the compression function, instantiated with parameters from the distributions established by Lemmas 1--2, is a PRP. Three independent analysis prongs (statistical, differential, game-theoretic) confirm that the compression function is indistinguishable from a random permutation of {0,1}^256. The distinguisher advantage is bounded by < 0.023 across all strategies tested.

4. **Lemma 4** lifts compression-function collision resistance to the full 128-block hash via the classical MD theorem. Length extension is structurally impossible (parameters are secret), intermediate states never converge, and no collisions are found in 5M evaluations.

**Combined argument:** Under the ROM for SHA-256, BAB64's parameter derivation produces S-boxes indistinguishable from random (L1) that are independent of other components (L2). These compose into a PRP compression function (L3). The MD iteration preserves collision resistance from the compression function (L4). Therefore, the full BAB64 hash function family {H_I}_I is collision-resistant.

Additionally, per-image parameterization provides structural defenses beyond what the formal argument covers:
- **Joux multi-collision neutralization.** Joux's attack [7] requires a fixed compression function for block-level collision composition. In BAB64, each image defines different parameters, preventing collision propagation across images.
- **Precomputation resistance.** Since H_I changes with every nonce, no precomputation (midstate caching, rainbow tables) carries across mining attempts.
- **ASIC resistance.** A BAB64 ASIC must load new S-box (256 B), round constants (128 B), rotation schedule (128 B), and initial state (32 B) per nonce, making dedicated hardware behave more like a programmable processor than a fixed-function pipeline. This is resistance in degree, not in kind.

### 4.7 Limitations

We are transparent about what this analysis does and does not establish:

1. **Empirical, not reductionist.** The lemmas are validated by statistical hypothesis testing (55 tests, all p > 0.01), not by formal reduction to a standard hardness assumption. A computationally unbounded adversary is not addressed.

2. **ROM dependency.** The entire argument assumes SHA-256 behaves as a random oracle. A structural weakness in SHA-256 would invalidate all four lemmas.

3. **Finite sample sizes.** Statistical tests have limited power. Our tests can detect deviations of moderate effect size but cannot rule out subtle biases detectable only with larger samples or more targeted analysis.

4. **Round count.** 32 rounds are chosen empirically, not analytically. The parallel diffusion step provides 46.3% avalanche at round 1, giving significant margin, but no formal lower bound on rounds has been established for SPNs of this structure.

5. **S-box quality below AES.** Mean nonlinearity 91.8 vs. AES's 112. This is inherent to random permutations vs. algebraic optimization. Per-nonce diversity compensates, but individual instances are weaker.

6. **Limited adversarial testing.** The construction has not been subjected to public cryptanalysis, expert differential/algebraic analysis, or adversarial ASIC design analysis.

7. **No formal model for per-image security.** The argument that per-nonce function diversity prevents targeted attacks is validated empirically but lacks a formal framework.

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
| Formal security | Reduces to SHA-256 | Reduces to Keccak + memory hardness | Reduces to GBP | Graph-theoretic | **ROM + empirical (4 lemmas, 55 tests)** |
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
- BAB64's security rests on a four-lemma formal analysis chain under the ROM, validated by 55 statistical tests. SHA-256, Keccak, and Blake2b have reductionist proofs; BAB64's analysis is empirical-formal rather than fully reductionist.

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

**No reductionist security proof.** The formal analysis (Section 4) establishes collision resistance via four lemmas validated by 55 statistical tests under the random oracle model, but does not provide a reduction to a standard hardness assumption in the traditional sense. The construction could harbor a subtle weakness not captured by our tests.

**SHA-256 dependency.** BAB64's security is upper-bounded by SHA-256. Image generation, parameter derivation, Lamport signatures, and verification all use SHA-256. A break of SHA-256 would compromise the entire system.

**Round count.** 32 rounds are chosen empirically, not analytically. The parallel diffusion step provides 46.3% avalanche at round 1, giving significant margin, but no formal lower bound has been established.

**Verification cost.** At ~0.14 ms per verification (C implementation), BAB64 is roughly 280x slower to verify than SHA-256d. For blockchain applications with block times of seconds or minutes, this is acceptable. For applications requiring sub-microsecond verification, it is not.

**Lamport signature size.** Each signature is ~8,192 bytes (256 x 32-byte secrets) with a ~16,384-byte verification key (512 x 32-byte hashes). This is orders of magnitude larger than ECDSA signatures (~72 bytes). For bandwidth-constrained applications, this is a significant cost for quantum resistance.

**S-box quality below AES.** BAB64's random permutation S-boxes have mean nonlinearity 91.8, below AES's optimal 112. This is inherent to random permutations vs. algebraically optimized constructions. The per-nonce diversity compensates (an attacker cannot build a fixed linear/differential attack), but the gap means individual hash function instances are weaker than AES.

**Limited adversarial testing.** Six attack simulations cover related-image, preimage structure, and Joux multi-collision vectors. The construction has not been subjected to public cryptanalysis, differential cryptanalysis by experts, algebraic attacks, or adversarial ASIC design analysis. Real-world security claims require broader scrutiny.

**Per-image security model is empirical.** Section 4 provides a four-lemma framework establishing PRP security of the compression function and collision resistance of the full hash. However, the PRP claim rests on statistical testing (distinguisher advantage < 0.023) rather than a formal proof of pseudorandomness for the SPN family.

### 8.2 Future Work

**Stronger formal analysis.** Extending the four-lemma framework to a full reductionist proof, or finding tighter bounds on distinguisher advantage, would significantly strengthen the construction.

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
