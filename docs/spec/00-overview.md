# 00 — Overview

- Status: Accepted for v1 planning
- Source of intent: [`proofs-of-inference-sota-path-forward.md`](../../proofs-of-inference-sota-path-forward.md) (cited below as "the source", with section numbers)
- Last updated: 2026-10-15

## 1. Thesis

Tessera is a sumcheck/GKR proof system that proves a committed integer language model `D` produced a recorded token stream. The proof is sound against a prover that controls all hardware and software, hides weights, activations, prompts and outputs, and costs a small multiple of the inference it certifies.

The unique claim is structural (source §5, §7). Proving cost under an exact sumcheck/GKR arithmetization is set by two counts, committed non-arithmetic outputs and lookups, not by FLOPs. Tessera therefore:

1. keeps every matmul, residual add, RoPE rotation and attention product virtual (never committed);
2. commits only lookup outputs, per-row scalars, decomposition limbs and weights;
3. pins softmax with a max certificate (one lookup, one commitment, one product-tree leaf per score);
4. batches every layer, head and query block into a constant number of sumcheck stages under one challenge seed, so proof depth is independent of model depth;
5. opens a persistent weight commitment with the same tensor-core pass that folds the weights for the sumcheck.

No published system proves a 400B-class dense model at a cost near inference (source §3). The best open systems trail the best closed system by two to four orders of magnitude; the best published frontier-scale proof (VerInf) has a proof multiplier near 10^5. The source projects Tessera's multiplier for Llama-3.1-405B at 3.2 for 1K tokens and 18.8 for 16K tokens; the corrections in [§9](#9-risks) raise these to about 8.5 and 24, still four orders of magnitude below VerInf.

## 2. Parties and users

| Party | Role | Trust |
|---|---|---|
| Prover operator | Runs the inference cluster, the transcript recorder integration and the audit prover. | Untrusted by the verifier. May deviate arbitrarily (adversary OC5, source §2). |
| Verifier | Counterparty that checks proofs; draws live randomness. | Untrusted by the prover: zero knowledge must hold against a malicious verifier. |
| Inspectorate | Optional third party that verifies transferable (non-interactive) proofs. | Trusts only the public statement and its own verifier build. |
| Recorder | Network tap or attested NIC that digests encrypted transcripts. | Trusted by both parties to record faithfully; its log is signed. |

Software users: protocol engineers integrating Tessera into an audit regime, ML engineers producing `D` from a checkpoint, and contributors implementing the components specified here.

## 3. The statement

Per sampled request, Tessera proves three conjuncts (source §2):

1. **Transcript anchor.** The token streams committed inside the proof equal the streams recorded at generation time. Specified in [RFC-0015](../rfcs/RFC-0015-transcript-anchor.md).
2. **Forward pass.** The committed integer model `D` with committed weights maps those inputs to logits, with no prover freedom anywhere upstream of the logits. Specified in [RFC-0005](../rfcs/RFC-0005-integer-model.md) through [RFC-0012](../rfcs/RFC-0012-weight-commitment.md).
3. **Scoring.** The outputs, scored against the logits, give a public bound `Û` on unexplained information. In exact mode `Û = 0` exactly when outputs equal `D`'s deterministic outputs, including seeded sampling. Specified in [RFC-0017](../rfcs/RFC-0017-scoring.md).

Public: `Û`, sequence lengths, the weight-commitment root `R_W`, the claim list (architecture), and the recorder digests. Hidden: weights, activations, prompts, outputs, sampler keys.

## 4. Two modes, one circuit

- **Exact mode.** The production engine runs `D` bit-exactly on INT8 tensor cores. Integer accumulation is order-independent, so any kernel, tiling or tensor-parallel layout yields the same bits. Decoding is greedy or inverse-CDF sampling from a committed PRF stream. The proof certifies token-for-token equality (`Û = 0`). Requires Hopper- or B200-class serving (source §8; B300 dense INT8 throughput is too low).
- **Bounded mode.** The production stack (FP8/BF16) is unmodified. The proof reports `Û` under a committed predictor derived from `D`'s logits. A poor predictor only inflates `Û`, which hurts the prover (Gibbs' inequality).

Both modes share the compiled circuit for `D`; they differ only in the scoring predictor ([RFC-0017](../rfcs/RFC-0017-scoring.md)). `D` is compiled under the `servable` profile by default so that exact mode is always available ([RFC-0005 §Profiles](../rfcs/RFC-0005-integer-model.md#profiles)).

## 5. Goals

| ID | Goal | Verified by |
|---|---|---|
| G-1 | Prove the three-conjunct statement for dense Llama-3.1-architecture models (8B, 70B, 405B) in exact and bounded modes. | End-to-end tests ([07 §6](07-testing-strategy.md#6-end-to-end-tests)); SC-1. |
| G-2 | Commit only non-arithmetic outputs; keep matmuls and attention products virtual. | Compiler commitment report ([RFC-0006](../rfcs/RFC-0006-compiler-ir.md)). |
| G-3 | Constant proof depth: a fixed number of batched stages regardless of layer count. | Stage schedule tests ([RFC-0010](../rfcs/RFC-0010-proof-schedule.md)). |
| G-4 | Soundness ≥ 100 bits for every non-interactive component; ≥ 64 bits proximity per live audit. | Soundness calculator ([RFC-0014](../rfcs/RFC-0014-interaction-and-soundness.md)). |
| G-5 | Statistical zero knowledge against a malicious verifier; leakage limited to the public statement. | Simulator tests ([RFC-0013](../rfcs/RFC-0013-zero-knowledge.md)). |
| G-6 | An independent verifier that recompiles every constraint from the public claim list and shares no code with the prover. | CI dependency-isolation gate ([RFC-0001](../rfcs/RFC-0001-workspace-and-verifier-independence.md)). |
| G-7 | Measure the unit costs on which every projection rests before building the GPU prover. | v0.1 gate ([RFC-0020](../rfcs/RFC-0020-cost-model.md)). |
| G-8 | Bit-exact, reproducible witness generation across CPU reference, GPU engine and tensor-parallel layouts. | Differential tests ([07 §5](07-testing-strategy.md#5-differential-and-conformance-tests)). |
| G-9 | Commit-then-sample audits whose detection probability is computable from public parameters. | Audit calculator ([RFC-0016](../rfcs/RFC-0016-audit-sampling.md)). |

## 6. Non-goals for v1

| ID | Non-goal | Reason |
|---|---|---|
| NG-1 | Mixture-of-experts models (Llama-4-Maverick) and routing privacy. | Needs the committed-permutation machinery of source §9 problem 2; dense 405B is the primary demo (source §10). |
| NG-2 | Architecture hiding. | Source §10 default: architecture revealed, weights hidden. Universal circuits are research (problem 8). |
| NG-3 | Recursive compression into a STARK; compact transferable proofs. | Optional in source §10; the uncompressed non-interactive mode covers transferability for v1. |
| NG-4 | M31/QM31 or Mersenne-61 backends; Stwo integration. | Goldilocks is the default (source §10). Interfaces stay field-generic ([RFC-0002](../rfcs/RFC-0002-field-arithmetic.md)). |
| NG-5 | Small-value commitments (ring switching, lattice commitments). | Research problem 3; up to 2–4× on commitment cost, not required for v1 correctness. |
| NG-6 | Certified attention block skipping; LogUp micro-architecture beyond the v1 kernels. | Research problem 1; tracked as future work, measured by the cost model. |
| NG-7 | Machine-checked proofs of the verifier. | v1 delivers a formal-verification-ready verifier specification and differential testing; mechanized proofs follow (problem 9). |
| NG-8 | Recorder hardware, production serving-stack integration, compute-coverage anchors. | External integrations. v1 specifies the recorder log format and ships a software reference recorder. |
| NG-9 | Training proofs; per-entry sampled proofs. | Out of the statement. Entry sampling is rejected for inference (source §4). |
| NG-10 | B300-class exact-mode serving. | Dense INT8 throughput is insufficient (source §8); bounded mode covers it. |
| NG-11 | The research paper. | A separate deliverable; this repository holds the specification and implementation. |

## 7. Success criteria

| ID | Milestone | Criterion |
|---|---|---|
| SC-0 | v0.1 | Unit costs `c_c`, `c_ℓ`, `c_π` measured on H100 SXM with the protocol of [RFC-0020](../rfcs/RFC-0020-cost-model.md); go/no-go decision recorded. |
| SC-1 | v1.0 | Exact-mode proof of Llama-3.1-405B batches at 1K, 4K and 16K context on one 8×H100 node, accepted by the independent verifier. |
| SC-2 | v1.0 | Measured proof multiplier `μ(1K) ≤ 12` and `μ(16K) ≤ 45` for 405B. The source projects 3.2 and 18.8; correcting for [RISK-2](#9-risks) and [RISK-9](#9-risks) gives about 8.5 and 24, and the ceilings leave 1.4× and 1.9× headroom over the corrected figures. |
| SC-3 | v1.0 | Verification of a 16K-token batch proof in ≤ 300 s on a 32-core x86-64 CPU with ≤ 64 GiB RAM; proof ≤ 4 GiB at live parameters. |
| SC-4 | v0.3 onward | Soundness calculator reports ≥ 100 bits (non-interactive algebraic parts) and ≥ 64 bits (live proximity) for every shipped parameter set; the tamper corpus is rejected in full. |
| SC-5 | v0.4 onward | Zero-knowledge simulator tests pass; CI redaction scan finds no secret-class data in logs or artifacts. |
| SC-6 | v0.4 | Bounded-mode `Û` measured on held-out prompts with confidence intervals; gate: `Û ≤ 0.88` bits/token (VerInf's published figure); research target `≤ 0.1`. |
| SC-7 | v0.2 onward | Witness engine outputs are bit-identical across CPU reference, PyTorch integer reference and GPU engine for every conformance model. |

## 8. Hardest problems

1. **Unit costs on real hardware.** Every projection rests on 30 ps per committed 64-bit element, 40 ps per LogUp lookup and 12 ps per product-tree leaf on one H100 (source §5). The committed-element budget implies about 530 GB/s of hashing per GPU at rate 1/2. If these do not hold, the design's constants change, not its structure.
2. **Uniqueness at acceptable cost.** No prover freedom upstream of the logits forces every rescale, normalization and softmax to be pinned by lookups or brackets whose keys are true integers within table range. The integer model must stay accurate at 405B scale while keeping key ranges inside tables.
3. **Composition.** Zero knowledge, soundness and the weight-commitment hiding budget must compose across batched constant-depth stages, a persistent commitment opened many times, and a verifier that shares no code with the prover.

## 9. Risks

Each risk names where the spec may be wrong and how it is resolved.

- **RISK-1: unit costs.** The three unit costs are assumptions. Resolution: the v0.1 gate in [RFC-0020](../rfcs/RFC-0020-cost-model.md#decision-rule) measures them. If each is within 2× of the assumption, proceed; otherwise re-plan the GPU work and restate every projection in this corpus.
- **RISK-2: rescale cost exceeds the source count.** The source's 137,216 commitments per layer per token for 405B equals one commitment per element of eight rescaled tensors (16,384 + 16,384 + 1,024 + 1,024 + 16,384 + 16,384 + 53,248 + 16,384). That count requires every rescale to be a single lookup. A single-lookup rescale needs the key's worst-case range to fit a table (≤ 2^32 rows here). Worst-case range analysis puts RoPE-rotated Q/K accumulators near 2^44, the O-projection chain near 2^60, the down-projection accumulator near 2^58 and RMSNorm products near 2^36. Those rescales need the split gadget of [RFC-0009](../rfcs/RFC-0009-gadgets.md#requantsplit), at two commitments and two lookups each. Upper bound: 220,160 commitments per layer per token (+60%), raising the 405B linear-layer multiplier from about 2.1 to at most 3.3. Resolution: the v0.2 compiler commitment report fixes the number, and the cost model consumes it.
- **RISK-3: exact-mode servability.** The source's commitment-minimal `D` feeds the unrescaled gate·up product (about 36 bits) into the down projection and the unrescaled attention output into the output projection. Neither is an INT8 tensor-core GEMM, so exact-mode serving would need multi-limb GEMMs. Resolution: the compiler supports two profiles ([RFC-0005](../rfcs/RFC-0005-integer-model.md#profiles)); `servable` is the default and is required for exact mode; both profiles are costed in the v0.2 report.
- **RISK-4: proximity parameters.** At rate 1/2, τ = 100 opened columns gives 41.5 bits under unique-decoding analysis, below the 64-bit live target. Resolution: [RFC-0014](../rfcs/RFC-0014-interaction-and-soundness.md#parameter-sets) sets τ = 155 for advice and τ = 95 for weights (rate 1/4), which grows proof size about 1.55× over the source figure.
- **RISK-5: LogUp challenge at frontier volume.** The rational-identity error `(N + T)/|F|` is about 2^-90 at 10^11 to 10^12 lookups, below 100 bits if the prover can grind the challenge. Resolution: in live mode the challenge derives from the verifier's coin-tossed seed (a one-shot 2^-90 event); in non-interactive mode seed derivation carries 20 bits of proof-of-work ([RFC-0014](../rfcs/RFC-0014-interaction-and-soundness.md#grinding)).
- **RISK-6: zero-knowledge composition.** Masking across stage boundaries is the least mechanical part of the design. Resolution: a written simulator argument is a v0.4 entry gate ([RFC-0013](../rfcs/RFC-0013-zero-knowledge.md#simulator-argument)).
- **RISK-7: integer-model fidelity.** W8A8 with power-of-two requantization at 405B scale is unmeasured, as is the bounded-mode `Û`. Resolution: the v0.2 fidelity harness ([RFC-0005](../rfcs/RFC-0005-integer-model.md#fidelity-evaluation)); requantization falls back to the split gadget per tensor where fidelity requires it.
- **RISK-8: hash throughput.** BLAKE3 must sustain about 530 GB/s per H100 inside the committed-element budget. Resolution: measured as part of `c_c` at v0.1; the hash is behind a trait so a superseding RFC can swap it.
- **RISK-9: witness regeneration is outside the source cost model.** The source's `μ` counts commitment, lookup and product-tree work, 15% sumcheck glue and the weight pass. It does not count the native compute of regenerating the witness: the witness pass itself, the leaf levels of the fraction and product trees (which exceed memory and are proven by a small-space sumcheck that streams the witness about twice), and the opening pass ([RFC-0011](../rfcs/RFC-0011-advice-commitment.md#opening-protocol)). Each regeneration costs about one native forward pass, so `μ` rises by about 3 to 5 at every context length; with RISK-2, the 405B projection becomes about 8.5 at 1K and 24 at 16K. Resolution: the executable cost model carries an explicit regeneration term ([RFC-0020](../rfcs/RFC-0020-cost-model.md#executable-cost-model)); v0.3 counts passes on CPU; v0.5 measures them on GPU; fusing passes is a performance objective of [RFC-0018](../rfcs/RFC-0018-gpu-prover-runtime.md#pass-structure).

## 10. Three meanings of overhead

This corpus never conflates them (source §2):

- **Proof multiplier `μ`**: proving GPU-time divided by the GPU-time of an ideal teacher-forced INT8 forward pass over the same tokens.
- **Amortized overhead**: `s · μ · η`, where `s` is the sampling rate and `η` is the ratio of ideal prefill time to decode-dominated serving time per token (source default `η = 0.3`).
- **Serving-path overhead**: what production pays continuously: near zero in bounded mode (transcript recording only), the cost of the deterministic INT8 engine in exact mode.

Per-request exact proving is not sub-10% and the corpus never claims it. Sub-10% is an amortized property reached through committed transcripts and unpredictable sampling ([RFC-0016](../rfcs/RFC-0016-audit-sampling.md)).

## 11. Traceability

Every requirement in this corpus traces to a source section or to a derived rationale stated in place. Derived decisions that the source does not make (hash function, encoding, predictor family, profile split, proximity parameters, grinding) are each locked in an RFC with alternatives considered.
