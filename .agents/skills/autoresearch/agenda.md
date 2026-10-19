# Research agenda

The topics the loop surveys. Each names its standing questions, the terms it moves (baseline.md §2 (`research/baseline.md#2-the-cost-model`) defines them, §6 (`research/baseline.md#6-levers-which-technique-class-moves-which-term`) maps technique classes to them), the RFC that fixes today's answer, and query seeds to start a search from. A survey answers a topic's standing questions; an idea moves a term; the sensitivity tables say which term is worth moving at which context. Edit this file when a survey closes a question or opens one.

## Priorities, from the sensitivity tables of 2026-10-16

1. The number of scores proven (`A`) and the streamed passes (`stream`) decide long context: halving `A` is worth 42% of `μ` at 16K and 50% at 128K; removing streaming 32% and 46%.
2. The regeneration passes (`regen`) decide short context: 45% of `μ` at 1K.
3. `c_ℓ` is the largest unit cost at every context (27–29%); `c_c` follows (about 20%); `c_π` is last (at most 7.5%).
4. Proof size (about 2 GB) and verification time (≤ 300 s) trail the best known system by two to three orders of magnitude; `τ_A = 155` sets 0.72 GB of the proof and most of the verifier's hashing.

## Topics

| Key | Topic | Terms | Fixed today by |
|---|---|---|---|
| `lookups` | Per-lookup cost and lookup count | `c_ℓ`, `K_lin` | RFC-0008 |
| `commitments` | Hash-based polynomial commitments | `c_c`, size, verify | RFC-0011, RFC-0012 |
| `proximity` | Proximity gaps, `τ` and proof size | `τ`, size, bits | RFC-0014 |
| `hash` | Hash throughput and hashed volume | `c_c` (RISK-8) | RFC-0004 |
| `sumcheck` | Sumcheck and GKR prover algorithms | glue, `regen`, `stream` | RFC-0010, RFC-0018 |
| `attention` | Per-score cost and the number of scores | `A`, `c_π`, `c_ℓ` | RFC-0009, RFC-0010 |
| `compiler` | Committed-element and lookup counts | `C_lin`, `K_lin` (RISK-2) | RFC-0005, RFC-0006 |
| `passes` | Regeneration, memory and distribution | `regen`, `stream` (RISK-9, RISK-10) | RFC-0018 |
| `zk` | Zero-knowledge overhead | zk overhead, size | RFC-0013 |
| `field` | Field and arithmetic | every unit cost | RFC-0002 |
| `systems` | Proof-of-inference systems and claims | scoreboard | `research/landscape.md` |
| `verifier` | Verification cost, recursion, assurance | verify, size | RFC-0019 |
| `paradigm` | Other paradigms and statements | scoreboard | 00 §6 non-goals |

### `lookups`

1. What does one LogUp-GKR lookup cost on an H100 in field operations and bytes, with base-field first layers, tuple tables and fused fraction trees (source §9 bet 1a targets 15–20 ps)? Where is the 40 ps assumption measured or published?
2. Do one-hot lookup arguments (Lasso, Shout, Twist) beat LogUp-GKR per lookup for 2^24-row tables under hash-based commitments, counting every committed chunk? What does the separation principle [Jo26] forbid, and what does it leave open?
3. Which operations can share one lookup (tuple tables, fused rescale and nonlinearity), and which lookups can become arithmetic (polynomial pieces plus a small range check)?

Seeds: "LogUp-GKR GPU", "Stwo LogUp GKR batching", "Shout lookup argument Setty Thaler 2025", "Twist and Shout", "lookup argument prover cost per lookup", "sparse-dense sumcheck", "Lasso GPU", "logarithmic derivative lookup fraction tree", "Jolt one-hot lookup commitment cost", ePrint 2026/1390.

### `commitments`

1. For 5.8×10^5 rows of width 2^22 at rate 1/2, what is the fastest hash-based commitment per element on a GPU: RS encoding by NTT with Merkle hashing (today), linear-time codes (Brakedown, Blaze, Ligerito), or FRI-family multilinear schemes (BaseFold, WHIR, STIR, DeepFold)? Which admit a one-pass streaming prover (RFC-0011 rejected FRI-family schemes because folding needs the vector resident)?
2. Small-value commitments (NG-5): what do ring switching (Binius), lattice commitments (Greyhound, LaBRADOR, Hyperwolf) or packed encodings cost per committed bit, and are they post-quantum and compatible with hiding?
3. What do Ligerito and "Ligero with logarithmic randomness" buy in proof size and prover cost at the same soundness?

Seeds: "Ligerito", "Blaze code-based polynomial commitment", "Brakedown GPU", "WHIR prover cost", "BaseFold streaming prover", "ring switching Binius small values", "Greyhound lattice polynomial commitment", "LaBRADOR", "Hyperwolf", "Angeris Evans Roh Ligero logarithmic randomness", "DeepFold", "WARP polynomial commitment".

### `proximity`

1. Under which provable bound can `τ_A` fall below 155 at rate 1/2, `n = 2^23`, over `Fp2`, for 64 live bits: unique decoding (today), the Johnson regime, or the 2026 "optimal proximity gaps" results? What are the exact field-size and row-count terms for an interleaved (Ligero-style) test?
2. What is the status of the correlated-agreement and mutual-correlated-agreement conjectures up to capacity after the 2025 negative results (Crites and Stewart) and the 2026 open-problems survey (Arnon)? Which regimes stay usable without a conjecture, as RFC-0014 requires?
3. How do rate, mask size `P` and grinding trade against `τ` and proof size? What do STIR- or WHIR-style query reductions cost the prover?

Seeds: ePrint 2020/654, ePrint 2025/2046, ePrint 2026/680, arXiv 2609.05870, ePrint 2025/870, arXiv 2605.07595, "Diamond Gruen proximity gaps interleaved codes", "mutual correlated agreement WHIR", "list decoding Ligero soundness", `research/tools/proximity.py`.

### `hash`

1. Measured BLAKE3 throughput on an H100 in a column-hashing setting, against the 530 GB/s per GPU the budget implies (RISK-8); the same for Poseidon2, Monolith, Tip5, Skyscraper, Keccak and BLAKE2s (ICICLE, sppark, Plonky3, RISC Zero, Stwo numbers).
2. Can the hashed volume shrink: Merkle arity, caps, column batching, packed field elements, chained digests with wider row blocks?
3. Would a field-native hash make recursion (NG-3) cheap enough to pay for itself in proof size?

Seeds: "BLAKE3 GPU throughput", "Poseidon2 GPU H100 hashes per second ICICLE", "Skyscraper hash", "Monolith hash", "RISC Zero Poseidon2 CUDA", "Plonky3 Poseidon2 GPU", "Merkle tree GPU throughput".

### `sumcheck`

1. Which prover algorithms cut field operations per round for products of multilinears with `eq`: Gruen's univariate skip, the Dao–Thaler `eq` split, the Bagad–Domb–Thaler small-field algorithms, Karatsuba or Toom–Cook for degree-3 rounds? Which apply over Goldilocks with `Fp2` challenges?
2. How many passes over the data does a sumcheck need at minimum (Blendy, space-efficient provers, streaming sumcheck)? This bounds `regen` and `stream`.
3. GKR batching across trees of different heights (Stwo) and across instances: what does it save and cost?

Seeds: "univariate skip sumcheck Gruen", "Dao Thaler more optimizations sumcheck proving", "Bagad Domb Thaler sumcheck small characteristic", "Blendy space-efficient sumcheck", "streaming sumcheck prover passes", "Karatsuba sumcheck degree 3", "GKR batching different heights LogUp".

### `attention`

1. Certified block skipping (source §9 bet 1b): which bound per (query, key block) proves every exp output rounds to zero, and how sparse is Llama-3.1 attention at 16K–128K, measured?
2. Can exp lookups be shared or replaced: piecewise-polynomial exp with a small range check, one lookup per block with a residual, tables shared across heads?
3. Cheaper or amortized max certificates (`c_π`): block-level certificates, other zero proofs with a uniqueness lemma.
4. In bounded mode, what does a proof-aware `D` (windowed, block-sparse or top-k attention) cost in `Û`?

Seeds: "zkAttn", "attention sparsity Llama long context measurement", "block-sparse attention certificate zero knowledge", "softmax zero-knowledge proof lookup-free", "integer exp approximation polynomial pieces", "proof-friendly quantization attention".

### `compiler`

1. How tight is worst-case range analysis against measured ranges at 405B, and what would replace the split gadget (two commitments, two lookups) by one lookup where typical ranges fit a 2^32 table (RISK-2: up to +60% commitments)?
2. Data-dependent certificates: can a per-tensor bound be proven cheaper than per-element range checks without leaking bits (SP-7, zero knowledge)?
3. Quantization co-design: which W8A8 variants (power-of-two scales, per-channel migration) reduce the rescale count or the key width at equal fidelity (RISK-7)?

Seeds: "range analysis quantized neural network zero knowledge", "RequantSplit", "power-of-two INT8 scales deterministic inference", "SmoothQuant integer-only inference proof", "zkML quantization proof cost".

### `passes`

1. What is the minimum number of full-forward-equivalent passes for the witness, the stages and the opening, with checkpointing, pass fusion and streaming-friendly commitments? Which stages can share a pass?
2. Streamed leaf levels: alternatives to `c(S)` streamed rounds (hierarchical folds, partially resident trees, GPU memory tiers, NVLink spill, proof units)?
3. Distributed sumcheck and GKR across eight GPUs or several nodes: communication per round (Pianist, Hekaton, deVirgo, Cirrus, HyperPianist).

Seeds: "streaming GKR prover memory", "distributed sumcheck Pianist Hekaton deVirgo", "checkpoint recomputation witness proof", "HyperPianist", "Cirrus distributed proving", "proof units long context attention".

### `zk`

1. What does Libra-style masking cost per sumcheck and per stage at Tessera's scale; what do BlindFold (Jolt Atlas) and newer zero-knowledge GKR variants cost; can mask rows be packed (OQ-0013-1)?
2. What is known about simulator arguments composing across batched GKR stages (RISK-6)?

Seeds: "zero-knowledge sumcheck low overhead", "BlindFold zero knowledge Jolt", "zero-knowledge GKR masking polynomial cost", "zero-knowledge Ligero hiding rows".

### `field`

1. Goldilocks against M31/QM31, BabyBear/KoalaBear and binary fields on a GPU: multiplications per second, extension costs, NTT throughput, and the rescale-count effect RFC-0002 cites (source §9 problem 5: about 20% either way).

Seeds: "Goldilocks GPU multiplication throughput", "Mersenne 31 GPU Stwo", "KoalaBear field", "Circle STARK GPU", "binary field GPU Binius".

### `systems`

1. New or updated proof-of-inference systems and numbers since the last check: Attestable, DeepProve, Expander and zkPyTorch, EZKL, Jolt Atlas, Seal-then-Sample, VerInf, others; general-purpose provers' GPU figures (SP1 Hypercube, RISC Zero, Plonky3, Stwo, Expander, Ligetron). Each row of `research/landscape.md` with a date older than six months is a question.

Seeds: "zkML LLM proof benchmark H100 2026", "proof of inference 70B", "Attestable benchmark", "DeepProve Llama 70B", "SP1 Hypercube GPU", "Expander GKR GPU benchmark".

### `verifier`

1. The verification budget (SC-3): the cost of hashing 0.72 GB, the inner products, the `Fp2` NTTs and the table MLEs; what parallelism buys; what recursion (NG-3) would cost and save.
2. Formal verification of verifiers: what Attestable's claim consists of, and the route for NG-7.

Seeds: "verifier cost Ligero opening", "formally verified STARK verifier", "recursive proof compression GKR".

### `paradigm`

1. Approximate sumcheck for floating-point statements in bounded mode [BDGIKT25]; folding and accumulation; TEE-plus-proof hybrids; bit-exact recomputation as a competitor; new statements (MoE routing privacy, training).

Seeds: ePrint 2025/2152, "proof of inference TEE hybrid", "MoE routing privacy committed permutation proof".

## Sources

- Venues: IACR ePrint; arXiv cs.CR and cs.LG; CRYPTO, EUROCRYPT, CCS, S&P, USENIX Security, TCC; FOCS and STOC for coding theory; ZKProof workshops; zkSummit talks.
- Teams and repositories: StarkWare (Stwo), Polygon (Plonky3), Succinct (SP1), RISC Zero, a16z crypto (Jolt), Irreducible (Binius), Polyhedra (Expander, zkPyTorch), Lagrange (DeepProve), Attestable (blog), EZKL, Ingonyama (ICICLE), Supranational (sppark), Nexus, Modulus Labs, Giza, Lucid Computing; the Ethereum Research forum; HackMD notes of the zk community.
- Refresh: a topic is stale six months after its survey, or when one of the survey's staleness triggers fires.
