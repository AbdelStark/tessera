# Tessera Specification (v1)

- Status: Accepted for v1 planning
- Date: 2026-10-15
- Source of intent: [`proofs-of-inference-sota-path-forward.md`](proofs-of-inference-sota-path-forward.md)
- Change process: RFCs ([09 §6](docs/spec/09-release-and-versioning.md#6-contribution-workflow))

## Executive summary

Tessera proves that a committed integer language model `D` produced a recorded token stream. The proof is sound against a prover that controls every piece of hardware and software, hides weights, activations, prompts and outputs from a counterparty verifier, and costs a small multiple of the inference it certifies. It is built for verification between mutually distrustful parties, such as states auditing each other's AI inference, and for anyone who needs that statement with the same rigor.

The design is a sumcheck/GKR proof over the Goldilocks field in which matmuls and attention products are never committed. Only lookup outputs, per-row scalars and weights are. Every layer, head and query block is proven in a constant number of batched stages under one seed. Softmax is pinned by a max certificate at one lookup, one commitment and one product-tree leaf per score. A persistent weight commitment is opened by the same pass that folds the weights for the sumcheck. An independent verifier rebuilds every constraint from a public claim list and shares no code with the prover.

The source projects a proof multiplier of 3.2 at 1K tokens and 18.8 at 16K for Llama-3.1-405B on one 8×H100 node. This corpus finds three costs the source's model leaves out, rescale decompositions (RISK-2), witness regeneration (RISK-9) and memory-bound tree levels (RISK-10), and restates the projection as about 10.5 to 11.5 at 1K and 26 to 27 at 16K, or up to 13.8 and 46 with the unmeasured column answers and streaming at their upper estimates. Both remain four orders of magnitude below the best published frontier-scale proof. Every figure is a projection until milestone v0.1 measures the unit costs on H100 and records a go/no-go decision against a rule fixed in advance ([RFC-0020](docs/rfcs/RFC-0020-cost-model.md#decision-rule)).

Sub-10% overhead is an amortized property, reached by recording every request and proving an unpredictable sample in full. It is not a per-request cost, and this corpus never claims it is.

## Reading order

1. [00 Overview](docs/spec/00-overview.md): thesis, statement, goals, non-goals, success criteria, risks.
2. [01 Architecture](docs/spec/01-architecture.md): components, dependency rules, lifecycle.
3. The RFC for the component you are building (index below).
4. [03 Data model](docs/spec/03-data-model.md), [04 Error model](docs/spec/04-error-model.md) and [07 Testing strategy](docs/spec/07-testing-strategy.md) for the contracts every change must respect.

## Specification sections

| Section | Content |
|---|---|
| [00 Overview](docs/spec/00-overview.md) | Thesis, parties, statement, modes, goals, non-goals, success criteria, hardest problems, risk register |
| [01 Architecture](docs/spec/01-architecture.md) | Four lanes, components, dependency rules, artifacts, prover phases, determinism invariants, deployment |
| [02 Public API](docs/spec/02-public-api.md) | Stability tiers, CLIs, exit codes, Rust and Python interfaces, parameter sets, compatibility rules |
| [03 Data model](docs/spec/03-data-model.md) | Scalar, model, circuit and statement types; artifact catalog; invariants; schema versioning |
| [04 Error model](docs/spec/04-error-model.md) | Verdicts, reject reasons, stable error codes, failure modes and responses, retry rules |
| [05 Observability](docs/spec/05-observability.md) | Logging, data classification and redaction, metrics, tracing, run manifests, verifier audit trail |
| [06 Security](docs/spec/06-security.md) | Assets, adversaries, trust boundaries, security properties, threats and mitigations, disclosure policy |
| [07 Testing strategy](docs/spec/07-testing-strategy.md) | Test pyramid, soundness and uniqueness tests, differential tests, ZK and ML tests, CI gates |
| [08 Performance budget](docs/spec/08-performance-budget.md) | Cost model, unit-cost targets, end-to-end targets, memory and proof-size budgets, regression gates |
| [09 Release and versioning](docs/spec/09-release-and-versioning.md) | Release trains, protocol and schema versions, deprecation, changelog, contribution, licensing, platforms |
| [10 Glossary](docs/spec/10-glossary.md) | Canonical terms |

## RFC index

| RFC | Title | Milestone |
|---|---|---|
| [RFC-0001](docs/rfcs/RFC-0001-workspace-and-verifier-independence.md) | Workspace layout, language boundaries, and verifier independence | v0.1 |
| [RFC-0002](docs/rfcs/RFC-0002-field-arithmetic.md) | Goldilocks field and quadratic extension | v0.1 |
| [RFC-0003](docs/rfcs/RFC-0003-canonical-encoding.md) | Canonical encoding and schema-versioned artifacts | v0.1 |
| [RFC-0004](docs/rfcs/RFC-0004-hashing-and-transcript.md) | Hashing, Merkle trees, and the Fiat–Shamir transcript | v0.1 |
| [RFC-0005](docs/rfcs/RFC-0005-integer-model.md) | Integer model D: W8A8 quantization, calibration, and fidelity evaluation | v0.2 |
| [RFC-0006](docs/rfcs/RFC-0006-compiler-ir.md) | Compiler IR: typed integer dataflow, range analysis, and gadget selection | v0.2 |
| [RFC-0007](docs/rfcs/RFC-0007-claim-list.md) | Public claim list and statement schema | v0.2 |
| [RFC-0008](docs/rfcs/RFC-0008-logup-and-tables.md) | LogUp-GKR lookups and the deterministic table registry | v0.1–v0.2 |
| [RFC-0009](docs/rfcs/RFC-0009-gadgets.md) | Uniqueness gadgets | v0.3 |
| [RFC-0010](docs/rfcs/RFC-0010-proof-schedule.md) | Batched constant-depth sumcheck/GKR proof schedule | v0.3 |
| [RFC-0011](docs/rfcs/RFC-0011-advice-commitment.md) | Streaming Ligero-family commitment for per-proof advice | v0.1–v0.3 |
| [RFC-0012](docs/rfcs/RFC-0012-weight-commitment.md) | Persistent weight commitment opened by the folding pass | v0.3–v0.4 |
| [RFC-0013](docs/rfcs/RFC-0013-zero-knowledge.md) | Zero knowledge: masked sumchecks, masked extensions, and hiding openings | v0.4 |
| [RFC-0014](docs/rfcs/RFC-0014-interaction-and-soundness.md) | Interaction model, soundness budget, and parameter selection | v0.3–v1.0 |
| [RFC-0015](docs/rfcs/RFC-0015-transcript-anchor.md) | Transcript anchor and recorder digest log | v0.4 |
| [RFC-0016](docs/rfcs/RFC-0016-audit-sampling.md) | Commit-then-sample audit protocol | v0.4 |
| [RFC-0017](docs/rfcs/RFC-0017-scoring.md) | Scoring and the unexplained-information bound Û | v0.3–v0.4 |
| [RFC-0018](docs/rfcs/RFC-0018-gpu-prover-runtime.md) | Witness engine and GPU prover runtime | v0.2–v1.0 |
| [RFC-0019](docs/rfcs/RFC-0019-verifier.md) | Independent verifier | v0.3–v1.0 |
| [RFC-0020](docs/rfcs/RFC-0020-cost-model.md) | Cost model and unit-cost measurement gate | v0.1 |
| [RFC-0022](docs/rfcs/RFC-0022-saturating-residual.md) | A saturating residual stream and complete norms (amends RFC-0005, RFC-0006, RFC-0009) | v0.2 |
| [RFC-0023](docs/rfcs/RFC-0023-norm-precision.md) | The norm's precision: choosing κ per model (amends RFC-0005) | v0.2 |
| [RFC-0024](docs/rfcs/RFC-0024-wide-requantization.md) | Wide requantization: keys too wide for their shift (amends RFC-0005, RFC-0006, RFC-0009) | v0.2 |
| [RFC-0025](docs/rfcs/RFC-0025-affine-sites.md) | Affine rescale sites in the claim list (amends RFC-0005, RFC-0006, RFC-0007, RFC-0010) | v0.2 |
| [RFC-0026](docs/rfcs/RFC-0026-claim-list-layout.md) | The claim list's commitment layout, scoring and anchor (amends RFC-0007, RFC-0012) | v0.2 |
| [RFC-0027](docs/rfcs/RFC-0027-logits-exponent.md) | The scale of `D`'s logits (amends RFC-0007, RFC-0026) | v0.2 |

All RFCs listed are Accepted; RFC-0021 is reserved for a draft. A decision changes only through a new RFC that supersedes it.

## Release plan

| Milestone | Scope | Exit criteria |
|---|---|---|
| **v0.1** Primitives and unit-cost gate | Workspaces and CI; field, codec, hashing and transcript (prover and verifier sides); multilinear polynomials and sumcheck; LogUp-GKR and product trees; reference advice commitment; GPU microkernels; executable cost model | SC-0: unit costs measured, decision recorded per RFC-0020; conformance vectors pass in both workspaces |
| **v0.2** Integer model and compiler | Python front-end for Llama-3.1 (import, calibration, W8A8, export, integer reference, fidelity); table registry; IR, range analysis, gadget selection, lowering, claim list, commitment report; CPU witness engine | SC-7 bit-exactness; fidelity gates for 8B; commitment report for 8B and 405B shapes resolves RISK-2's count |
| **v0.3** End-to-end reference proof | Gadgets, stage schedule, advice layout and streaming commitment, binding weight commitment, openings, live and non-interactive protocol, soundness calculator, verifier driver, CLIs | Exact-mode greedy proofs of tiny models verify in both modes; tamper corpus rejected in full (SC-4); `tessera-protocol/1` |
| **v0.4** Full statement | Zero knowledge (after the simulator-argument gate), weight hiding budget and refresh, transcript anchor and reference recorder, audit sampling, sampler PRF and mixture scoring, verifier ledger | SC-5 and SC-6; `tessera-protocol/2` |
| **v0.5** GPU prover | CUDA kernels, GPU witness engine, layer-range partitioning, streaming passes, pass planner; 8B and 70B proofs | Measured `μ` for 8B and 70B; model re-fit; RISK-9 and RISK-10 measured |
| **v1.0** Frontier | Llama-3.1-405B at 1K, 4K and 16K on one 8×H100 node; verifier hardening, fuzzing and written specification; live transport; reproducible builds; external review; documentation | SC-1, SC-2, SC-3; protocol frozen |

The implementation tracker, generated from this corpus, is [docs/roadmap/IMPLEMENTATION.md](docs/roadmap/IMPLEMENTATION.md).

## Risk register

Full statements and resolution plans are in [00 §9](docs/spec/00-overview.md#9-risks).

| Risk | Summary | Resolved by |
|---|---|---|
| RISK-1 | Unit costs are assumptions | v0.1 measurement and decision rule |
| RISK-2 | Wide-key rescales need two commitments, or four where the key is too wide for its shift, and a classed residual site two steps; up to +108% linear-layer commitments at 405B in `minimal` and +174% in `servable` | Resolved by the v0.2 commitment reports ([`docs/perf/v0.2/`](docs/perf/v0.2/README.md)) |
| RISK-3 | The source's commitment-minimal `D` is not INT8-servable in exact mode | Profiles `servable` and `minimal` |
| RISK-4 | `τ = 100` gives 41.5 bits, not 64 | `τ_A = 155`, `τ_W = 97` in `live-64` |
| RISK-5 | LogUp error about 2^-90 at frontier volume | Live seed; 20-bit grinding in non-interactive mode |
| RISK-6 | Zero-knowledge composition across stages | Written simulator argument before v0.4 work |
| RISK-7 | W8A8 fidelity and bounded-mode `Û` at 405B unmeasured | v0.2 and v0.4 fidelity studies |
| RISK-8 | BLAKE3 throughput within the committed-element budget | v0.1 measurement |
| RISK-9 | Witness regeneration absent from the source cost model | Explicit `n_regen` term; measured at v0.3 and v0.5 |
| RISK-10 | Tree leaf levels exceed device memory at long context | Pass structure and proof units (RFC-0018); measured at v0.5 |

## Open questions

| ID | Question | Owner | Resolve by |
|---|---|---|---|
| OQ-0001-1 | Move the verifier to its own repository? | @AbdelStark | v1.0 release planning |
| OQ-0004-1 | Reduce hashed volume if BLAKE3 alone exceeds 20 ps per element? | @AbdelStark | v0.1 gate |
| OQ-0005-1 | Deployment-specific calibration for bounded mode? | @AbdelStark | v0.4 fidelity study |
| OQ-0012-1 | Larger `B_W` and `P_W` for longer epochs? | @AbdelStark | v1.0 |
| OQ-0013-1 | Pack mask rows of small tensors? | @AbdelStark | v0.4, with the simulator argument |
| OQ-0014-1 | Live-session deadlines and retry policy | @AbdelStark | v1.0 |
| OQ-0015-1 | Recorder placement when clients use TLS | @AbdelStark | Before the first bilateral pilot (v1.0) |
| OQ-0017-1 | CDF-window predictor for bounded mode? | @AbdelStark | v0.4 fidelity study |
| OQ-0018-1 | Streaming versus smaller proof units at long context | @AbdelStark | v0.5 |
| OQ-0019-1 | Route to mechanized verifier proofs | @AbdelStark | v1.1 planning |
| OQ-0020-1 | B200 as reference hardware for v1.0 claims? | @AbdelStark | v0.5 |
| OQ-SEC-1 | Post-quantum recorder signatures (ML-DSA-65)? | @AbdelStark | v1.0 |

## Conventions

- **Normative language.** In Proposed Design sections and in invariant and requirement tables, "must", "must not" and "may" are normative. Everything else is explanatory.
- **Identifiers.** Goals `G-n`, non-goals `NG-n`, success criteria `SC-n`, invariants `INV-<AREA>-n`, error codes `ERR-*`, reject reasons `REJ-*`, risks `RISK-n`, open questions `OQ-<RFC>-n`, dependency rules `DEP-n`, compatibility rules `COMPAT-n`, observability rules `OBS-n`, security properties `SP-n`, trust boundaries `TB-n`, adversaries `ADV-*`. Identifiers are stable; cite them in code comments, tests and issues.
- **Projections.** Every number derived from the source's cost model is labeled a projection until a measured value with a run manifest replaces it.
- **Source citations.** "Source §n" refers to sections of the source document, [`proofs-of-inference-sota-path-forward.md`](proofs-of-inference-sota-path-forward.md) at the repository root.
