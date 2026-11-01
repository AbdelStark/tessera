# Changelog

All notable changes to the prover release train are recorded here: the crates under `crates/`, the `tessera` command-line tool and the `tessera-quant` Python package. The format follows [Keep a Changelog 1.1](https://keepachangelog.com/en/1.1.0/), and versions follow [Semantic Versioning](https://semver.org/spec/v2.0.0.html) as described in [09 Release and versioning](docs/spec/09-release-and-versioning.md).

## [Unreleased]

### Fixed

- Use checkpoint-derived nonweight tensors and borrow model INT8 weights when their complete replay reservation fits beside the unchanged resident-cell floor; preserve the resident fallback and verify byte-identical independently accepted proofs (RFC-0018, [#641](https://github.com/AbdelStark/tessera/issues/641)).

- Keep telemetry CI from spending its entire deadline refreshing package indexes: reuse installed tools, bound downloads and refresh indexes only after a cached install fails; retain export and syscall-isolation checks ([#641](https://github.com/AbdelStark/tessera/issues/641)).

- Price generated P2 table sources by canonical public count-backing plans when explicitly selected, and bound opening-geometry constructor work including symbolic points and prefix grouping; retain the resident route and its logical-cell floor (RFC-0018, [#641](https://github.com/AbdelStark/tessera/issues/641)).

- Account for retained lowering, runtime selection, advice/leaf/group layouts and native-plan metadata by their nested vector capacities and pinned B-tree node bounds; keep constructors and proof payloads separately reserved (RFC-0006/0018/0045/0054, [#641](https://github.com/AbdelStark/tessera/issues/641)).

- Count graph, layer-map and stage-schedule public metadata by actual nested vector capacities, including empty vectors with spare storage, separately from validation scratch and tensor payloads (RFC-0006/0037/0018, [#641](https://github.com/AbdelStark/tessera/issues/641)).

- Bound stage-driver sharing-map ownership, evaluation and claim scratch, split/leaf outputs and padding reachability work from public geometry; retain source, contraction, engine and encoding reservations separately (RFC-0018/0037, [#641](https://github.com/AbdelStark/tessera/issues/641)).

- Bound canonical matmul, elementwise, rotation, selection and bit-recomposition table construction, retained contractions/public equality tables and borrowed-instance metadata from public geometry; keep original operands and later proof work separate (RFC-0010/0018, [#641](https://github.com/AbdelStark/tessera/issues/641)).

- Account for canonical stage-source callback captures, column/plan/factor vectors and original resident inputs from public geometry; preserve relation values and keep contraction/construction work separately charged (RFC-0018/0055, [#641](https://github.com/AbdelStark/tessera/issues/641)).

- Bound persistent stage-driver claim/part/layer buffers, public challenge maps and constructor metadata from the compiled public schedule; keep pending windows, relation sources and message/evaluation work separately charged (RFC-0018/0037, [#641](https://github.com/AbdelStark/tessera/issues/641)).

- Bound additional window-engine specification, state, message and returned-proof storage from public geometry; preserve the existing grid/fold reservation and charge source/wrapper work separately (RFC-0010/0055/0056, [#641](https://github.com/AbdelStark/tessera/issues/641)).

- Bound ordinary batched sumcheck storage from public instance/term geometry, including table bindings, compiled plans, pinned B-tree metadata, kernel scratch and retained outputs; charge delegated computed-state payloads separately and leave CPU admission unchanged (RFC-0010/0018, [#641](https://github.com/AbdelStark/tessera/issues/641)).

- Bound original advice/weight opening-target capacities and canonical construction work from public geometry, including native-original lists, stable sorting and mapped-point scratch; count repeated and zero-weight obligations without admitting a cached CPU route (RFC-0011/0045/0052, [#641](https://github.com/AbdelStark/tessera/issues/641)).

- Bound protocol/1 regenerated weight trees, construction/answer scratch, canonical model-reference metadata, folds and all ChannelParams combined rows from public layout geometry; measure the existing resident differential and seven-claim fold storage without admitting the cached route (RFC-0012/0040/0052, [#641](https://github.com/AbdelStark/tessera/issues/641)).

- Bound additional computed-score round storage from public geometry, including growing challenge points, boxed state, key folds and four clearing row arrays; measure start/round/bind allocations separately from retained inputs and generic sumcheck work (RFC-0056, [#641](https://github.com/AbdelStark/tessera/issues/641)).

- Bound native prover construction, initial preparation and borrowed-instance metadata from public geometry, including computed/literal score modes, all retained tables/masks, runtime coefficient scratch, equality-cache growth and Arc descriptors; keep rounds/windows, openings and CPU admission separate (RFC-0018/0056, [#641](https://github.com/AbdelStark/tessera/issues/641)).

- Bound resident/cached tile allocation requests, including root comparison polynomials, Softmax witness copies and grouped lookup growth, output/share buffers, padding recounts and runtime keys; charge canonical count backing separately and retain the CPU admission floor (RFC-0018/0050, [#641](https://github.com/AbdelStark/tessera/issues/641)).

- Bound public graph/range validation, runtime source setup, replay input checks and grouped-column transfers from allocation requests, including repeated coverage walks and position-vector overlap; preserve validation and the CPU admission floor (RFC-0006/0018, [#641](https://github.com/AbdelStark/tessera/issues/641)).

- Bound public source/gadget construction metadata from allocator-request counts, including complete root relations, wide requantization, Softmax cores/slack forms and malformed Requant role payloads; keep runtime/replay/tile work and CPU admission separate (RFC-0018/0050, [#641](https://github.com/AbdelStark/tessera/issues/641)).

- Plan computed score query/key/maximum buffers from public tensor geometry, reserve raw INT8 inputs exactly and clear them after padding; account for short contexts, GQA and conversion overlap separately from other native work (RFC-0056, [#641](https://github.com/AbdelStark/tessera/issues/641)).

- Count retained source metadata and public gadget templates, including wide boxes, dense cores, nested constraint vectors and spare capacities; keep construction and tile scratch separate from these measured charges (RFC-0018, [#641](https://github.com/AbdelStark/tessera/issues/641)).

- Plan owned padded model lookup columns, Softmax forests and their constructor descriptors from public domains; reserve nested source metadata exactly while retaining separate tile/gadget work and the existing CPU admission floor (RFC-0018/0050, [#641](https://github.com/AbdelStark/tessera/issues/641)).

- Plan engine table-loader buffers, including id gaps, class clones, RoPE frequencies and replacement overlap; reserve registry and frequency capacities exactly without changing table values or CPU admission ([#641](https://github.com/AbdelStark/tessera/issues/641)).

- Plan owned canonical-token, checkpoint and full nonweight-cache constructor buffers, including repeated leaf copies, public position-vector overlap and checked metadata sizes; use exact reader/operand reservations and reverse topological replay cones without a worklist. Borrowed tables, runtime and other proof phases remain separate (RFC-0006/0018, [#641](https://github.com/AbdelStark/tessera/issues/641)).

- Bound sequential PCS P1 storage, including encoder initialization scratch, all segment chains, active codeword block, merged tree and owned layout; reserve CPU segment descriptors exactly and retain separate caller/whole-route charges (RFC-0011/0018, [#641](https://github.com/AbdelStark/tessera/issues/641)).

- Count every nested public advice-layout allocation, including spare capacities, and bound its owned clone; retain constructor scratch as a separate charge (RFC-0011/0018, [#641](https://github.com/AbdelStark/tessera/issues/641)).

- Bound PCS-owned P5 columns and publication overlap before query sampling, and reserve canonical Merkle siblings exactly from validated public indices. Repeated draws retain their worst-case charge; full CPU admission remains pending (RFC-0004/0011/0018, [#641](https://github.com/AbdelStark/tessera/issues/641)).

- Preflight PCS-owned grouping and advice-opening allocation bounds, including repeated-claim descriptor growth, combination scratch and simultaneous private/public rows; caller storage and the full CPU route remain separate (RFC-0011/0018, [#641](https://github.com/AbdelStark/tessera/issues/641)).

- Stream NI proof and producer Live packet envelopes through a cleared bounded buffer, retaining exclusive 0600 temporary files, synchronized no-overwrite publication and identical manifest entries (RFC-0003/0014/0018, [#641](https://github.com/AbdelStark/tessera/issues/641)).

- Batch generated table/count reads using checked spare public memory, preserving the 256-row fallback and existing window cuts; charge the selected source scratch and retain exact count/proof behavior (RFC-0018/0054, [#641](https://github.com/AbdelStark/tessera/issues/641)).

- Count public proof-byte metrics through the existing TCE encoders without retaining temporary output buffers; preserve reported parts, proof bytes and CPU admission (RFC-0003/0018, [#641](https://github.com/AbdelStark/tessera/issues/641)).

- Derive multiplicity backing plans from public table and runtime metadata through the same dense/weighted storage decision used by allocation; check addressability before reserving storage and run independent padded-column histograms in CI (RFC-0008/0018/0071, [#641](https://github.com/AbdelStark/tessera/issues/641)).

- Preflight materialized lookup-core lengths without allocation, reserve each core once, and report dense-core allocation failures as `ERR-RES-MEMORY`. Table values, digests, parameter bundles and CPU admission remain unchanged (RFC-0008/0018, [#641](https://github.com/AbdelStark/tessera/issues/641)).

- Derive public opening geometry through the CPU producer's original-source traversal, using opaque coordinate origins and retaining zero-weight images and all ChannelParams reads. Counts support future memory planning; CPU admission and proof artifacts remain unchanged (RFC-0011/0012/0018, [#641](https://github.com/AbdelStark/tessera/issues/641)).
- Allow the complete prover CI suite 45 minutes after both checkpoint-source runs exhausted the 25-minute job budget during final NI/Live checks; preserve every test command, filter and protocol deadline ([#641](https://github.com/AbdelStark/tessera/issues/641)).

- Clear completed CPU leaf/native/stage working sets before later proof phases, after transferring their remaining obligations; add checkpoint tensor replay with resident equivalence and allocation checks. The resident-cell memory guard and real-model proof requirements remain unchanged (RFC-0018, [#629](https://github.com/AbdelStark/tessera/issues/629)).

- Derive a common Requant family for incompatible Protocol One model layer sites, with independent verifier selection, complete first-use tables and ordered requests; retain the leaf guard, legacy-compatible artifacts and Protocol Two behavior (RFC-0076, [#617](https://github.com/AbdelStark/tessera/issues/617)). Real 1B/8B proofs and whole-model memory qualification remain pending.

- Stream generated LogUp table/count sources through the CPU prover's ordinary and grouped leaf batches under checked P2 reservations, preserving proof messages and the conservative cell floor while removing dense table-column and negated-count construction (RFC-0008/0054, [#637](https://github.com/AbdelStark/tessera/issues/637)).

- Finish each admitted integer candidate's construction before loading its BF16 held-out reference, separating float64 migration scratch from reference model/window residency while preserving preflight, full-window KL reduction and strength selection (RFC-0005, [#635](https://github.com/AbdelStark/tessera/issues/635)).

- Stream ordinary CPU advice through one reused clearing field row with rank-sized source-index metadata, preserving canonical padding and P1/P4/P5 rows without changing the memory-plan charges (RFC-0011/0047/0050, [#630](https://github.com/AbdelStark/tessera/issues/630)).

- Preflight migration strengths against the existing scale/table bounds before held-out evaluation, then minimize KL over admissible candidates without masking malformed input, scoring or resource errors; reject all-infeasible calibration before BF16 and release each candidate before constructing the next (RFC-0005, [#625](https://github.com/AbdelStark/tessera/issues/625)).

### Added

- Share canonical integer tensor reads between witness tiles and reduction stages, with signed borrowed INT8 weights, owned clearing checkpoint tensors and canonical batch-token copies; run tensor and request-slot regressions in CI (RFC-0018, [#641](https://github.com/AbdelStark/tessera/issues/641)). CPU driver admission and the resident-cell floor remain unchanged pending complete phase accounting.

- Accept common-family Protocol One Requant selection and its explicit one-time unreleased compatibility decision; actual real-model proofs remain pending (RFC-0076, [#619](https://github.com/AbdelStark/tessera/issues/619), parent #617).

- Full pinned Llama-3.1-8B calibration metadata, all 16 existing-policy scale preflights and one fixed diagnostic INT8 artifact with complete readback/Rust geometry checks, with private statistics and weights outside the checkout; no fidelity or real-proof claim (RFC-0005/0025/0038, [#633](https://github.com/AbdelStark/tessera/issues/633)).

- Complete E0 real-1B runtime/planner sizing: the current CPU zero-lookup floor is 2.56–3.80 TiB over nine cases, with independent recount and ten corruptions caught; no production selection, measured peak or proof capability change (EXP-0030, [#627](https://github.com/AbdelStark/tessera/issues/627)).

- A repository tool for bounded, bit-preserving BF16 tied-head checkpoint preparation, with exact configuration lexemes, checked shards, private staged output and complete streaming verification; the [real-model ramp guide](docs/models/real-model-ramp.md) pins the 1B/8B sources and records the unresolved real-proof and fidelity obligations (RFC-0005, TB-5, [#623](https://github.com/AbdelStark/tessera/issues/623)).

- Complete claim-validation and NI-driver fuzz targets, declared tamper-case seeds and a pinned CPU-time campaign runner; the 24 CPU-hour release result remains pending until its report is recorded ([#185](https://github.com/AbdelStark/tessera/issues/185), SP-8, RFC-0019).
- Draft RFC-0075 for bounded full-support weight-mask regeneration and complete core/draw/total cost accounting, backed by SUR-0016 and frozen E0 audit EXP-0028; no GPU measurement or zero-knowledge capability is claimed ([#568](https://github.com/AbdelStark/tessera/issues/568)).
- Proposed separate Lean verifier blueprint, with three surfaces of theorem statements, provisional predicate correspondences, negative-case evidence and open refinement/composition obligations; no mechanized proof or changed acceptance semantics ([#392](https://github.com/AbdelStark/tessera/issues/392), RFC-0019 OQ-0019-1, SP-6).
- Windowed CPU proving uses the resident native representation charged by its plan; ordinary resident proving retains computed score rounds. Generated OneHot/RoPE factors preserve inter-slot padding, with NI/Live budget-window batch coverage (RFC-0055, RFC-0056, RFC-0071, #459).

- Windowed reduction stages on CPU (RFC-0055/0056, #459): grids with factored equality, mixed resident operands, regenerated witness passes, clearing folded tables and budgeted P3b accounting preserve the existing proof bytes. CPU plans retain the resident witness floor and charge stage working sets separately.
- An explicit test-only CPU API acceptance build and required CI job enable the independent verifier dev-dependency's non-certifying parameters; complete one-layer NI/Live proofs are checked without conditional skips (#598, OBS-4).

- Preserve stacked weight, class and multiplier inputs for one-layer models with zero layer variables, so exported one-layer integer models compile and their reduction-stage claims agree with the independent verifier; retain shared-input and partial-sharing refusals (RFC-0037, [#598](https://github.com/AbdelStark/tessera/issues/598)).
- Structured score rounds retain the slot-aware literal path for multi-request batches; the computed single-request representation does not establish batch memory savings (RFC-0056, RFC-0071, #461).

- RFC-0056 native Softmax key and forest sumchecks from INT8 queries, folded key blocks and causal prefix sums, with clearing row tables after the key rounds; the literal prover remains an equivalence oracle. Batched sumchecks share identical relations and borrowed operand tables, including repeated norm sums and root coefficient products, while preserving every transcript message and original opening claim ([#461](https://github.com/AbdelStark/tessera/issues/461)).
- A recorder/serving-engine [integration guide](docs/integration.md) and executable loopback example ([#180](https://github.com/AbdelStark/tessera/issues/180), RFC-0015/0059/0061/0062/0063): TLS placement/trust duties, frame and nonce ownership, sampler pins, exact-engine conformance, private transcript handling, and CI checks against the reference recorder and independent sampler vectors.
- Optional prover CLI telemetry ([#177](https://github.com/AbdelStark/tessera/issues/177), 05 §4–§5): an explicit `--metrics-addr` Prometheus endpoint over the existing registry, and `otel`-gated OTLP/HTTP JSON trace export with `--otlp-endpoint`. Default invocations open no telemetry socket; CI checks both builds in a network namespace. See the [telemetry guide](docs/telemetry.md).
- A private mdBook documentation artifact with the specification, RFCs, guides and separate Rust API references, built and link-checked in CI with tier-B API doctests ([#176](https://github.com/AbdelStark/tessera/issues/176)).
- Prover-side Live directory and optional mutually authenticated TLS 1.3 transport APIs, with bounded length-prefixed envelopes, absolute per-message deadlines and failure termination ([#179](https://github.com/AbdelStark/tessera/issues/179), RFC-0014).
- Tag-triggered draft release workflow and dry runs, with two clean container builds per Linux verifier architecture, byte comparisons, SHA-256 release notes and a scripted second-builder procedure ([#178](https://github.com/AbdelStark/tessera/issues/178), 09 §9).
- `audit batch --plan` and sampled proving split oversized runtime advice layouts with the same slot-class packing rule; a request that cannot fit alone is refused (RFC-0071, #551).

- Conformance vectors and differential prover-to-verifier adapter tests for temperature and mixture scoring checks ([#160](https://github.com/AbdelStark/tessera/issues/160), RFC-0017, RFC-0019, RFC-0030, RFC-0035, RFC-0043, RFC-0050, RFC-0058, RFC-0061):
  - **Specification clarification (`RFC-0017`).** Spelled out `Exp_T`'s precision `sigma = exp_sigma` and deterministic narrowest `key_bits ∈ [min_key_bits, max_log2_rows]` selection (`min_key_bits = max(1, bit_length(hi − lo))` over the lowered LM-head logit interval `[lo, hi]`) under `'D's decoded token'`.
  - **Independent gadget-finals oracles (`spec/vectors/gadget-finals/`).** Added `inverse_cdf.py` (`inverse-cdf.json`: 6 schema cases across `W ∈ [1, 51]` and 17 component cases, 6 honest and 11 isolated alternatives with omission proofs) and `one_sided_log2.py` (`one-sided-log2.json`: exact `c_match(0..=16)`, 6 `VocabMax` cases, 3 runtime table-interning cases, 7 `OneSidedLog2` schema cases, and 15 `OneSidedLog2` component cases, 5 honest and 10 isolated alternatives with omission proofs).
  - **Runtime table-interning parity & differential `DEP-2` adapter tests (`tessera-compiler`, `crates/tessera-prover/tests/`).** Aligned `TemperaturePlan::derive` and `MixturePlan::derive` with `RFC-0017` item 6 (collecting distinct effective temperatures in first-seen output-bearing request order while validating `vocab_exp_scale` on all requests and enforcing `GadgetPolicy.max_log2_rows` on `InverseCdf` `Range` tables), and added `gadget_inverse_cdf` and `gadget_scoring` joining `tessera-scoring`, `tessera-compiler` (`cdf_columns_and_requests`, `log2_columns_and_requests`, `vocab_exp_scale`, `TemperaturePlan`, `MixturePlan`), `tessera-gadgets` (`InverseCdf`, `VocabMax`, `OneSidedLog2`, `c_match`), and `tessera-sumcheck::prove_batch` directly to `tv-core::scoring` and `tv-core::gadgets` across all shared scoring and gadget-finals vectors and `tiny-2-servable.d`.
- Mixture predictor scoring with the one-sided `log2` bound `Û` ([#155](https://github.com/AbdelStark/tessera/issues/155), RFC-0017, RFC-0026, RFC-0035, RFC-0050, RFC-0053, RFC-0058):
  - **`tessera_gadgets::OneSidedLog2` and bounded `Argmax`.** `OneSidedLog2` (`c_match`, `Log2Column`, `Log2Term`, `Log2Relation`, `Log2Constraints`, `Log2Witness`, `Log2BatchColumns`, `BatchedLog2Selector`, `BatchedLog2Evaluation`, `Log2VocabRelations`) proving `ψ_t · (Z_t + V) · 2^{16} ≤ (Y_t + ψ_t) · 2^{I_t} · Exp2Frac[f_t]` over radix `2^{16}` via `Pow2 { max_exp: 63 }`, `Exp2Frac { frac_bits: 8, out_bits: 16 }`, `NonNeg`, a `63`-bit limb box for `P_t`, and `BigIntCompare`, paired with `m_t = 0` at `k = 0` or `m_t(1 − m_t) = 0` and `m_t · (h_{y_t,v} − h_{ŷ_t,v}) = 0` for `1 ≤ k ≤ 16`, a branch-free `63 + 256`-step witness scan (`INV-ARCH-2`), and the degree-3 contraction `Σ_t B_t = Û_r` in `2^{−8}`-bit units (`INV-SC-1..3`). Extended `Argmax` with `with_span`, `sample`, and batched bounded-mode relations (`BatchedArgmaxSelector`, `BatchedArgmaxEvaluation`, `batched_relations`, `batched_sampled_relations`, `batched_suffix_weights`).
  - **Runtime mixture derivation (`tessera-compiler`).** `Log2AdviceColumn`, `log2_columns_and_requests`, `MixtureDecoding` (`Greedy`, `Temperature`), `MixtureRequestSpec`, `MixtureCertificate`, `MixturePlan::derive`, and `Rational::ONE`, interning `Exp_1` / `Exp_T` (`≤ 4` distinct temperatures), `Pow2 { max_exp: 63 }`, `Exp2Frac { frac_bits: 8, out_bits: 16 }`, and physical `Range` tables within `GadgetPolicy.max_log2_rows` without modifying `ClaimList` (`0x0201`).
  - **`tessera_scoring::MixtureScoring`.** `MixtureScoring`, `MixtureRequest`, `MixtureInputs`, `MixtureCertificate`, and `MixtureBatch`, pairing the recorded `OneHot` indicator with a separate committed decoded indicator `h_{ŷ_t}`, certifying matches and mismatches into a single public per-request `Û_r` while clearing all per-row witness columns (`OBS-1`, `OBS-6`, `INV-SC-3`).
- Vocabulary softmax, in-circuit Sampler PRF, runtime `Exp_T` derivation, and exact temperature decoding certificates ([#152](https://github.com/AbdelStark/tessera/issues/152), RFC-0017, RFC-0030, RFC-0035, RFC-0058, RFC-0061):
  - **`tessera_gadgets::VocabMax`.** Vocabulary maximum certificate (`VocabColumn`, `VocabValue`, `VocabTerm`, `VocabRelation`, `VocabConstraints`, `VocabWitness`, `VocabEvaluation`, `VocabSelector`, `VocabRelations`) committing only per-row `Maximum` and per-entry `Exponentials` without a reciprocal or bracket slacks: checks `1 ≤ vocab ≤ 2^18`, `P > 2^30`, `P ≥ 2^{key_bits} + 2^29`, `σ ≤ 32`, `2^σ ≤ (P − 1) / 2` (RFC-0009 Lemma 0), and `Exp_T[K_0] = 0` (RFC-0030), evaluates clearing lookup keys and product-forest leaves across power-of-two row batches, and builds `Elementwise` sumcheck reductions for `Key`, `Leaf`, and `PaddingMaximum` (`(1 − ψ_t)·c_t = 0`).
  - **Batched `tessera_gadgets::InverseCdf`.** `BatchedCdfSelector`, `BatchedCdfEvaluation`, `CdfBatchColumns`, `InverseCdf::range_kinds`, `selector`, `evaluate_batch`, degree-3 `batched_relations` (`Z`, `B`, `S` contractions over `row_vars + vocab_vars`), `batched_sampled_relations`, `batched_suffix_weights`, `evaluate_lookups`, and `equality_relations` (expanding all `CdfConstraints` comparisons and `(ψ_t − 1)·Z_t = 0` into `Elementwise` zero-checks). Resolves OQ-0058-1 in RFC-0058 by keeping `DrawLimb` self-contained rather than coupling `InverseCdf` to AES's `[16, B]` block layout.
  - **Runtime `Exp_T` and `InverseCdf` derivation (`tessera-compiler`).** `MAX_BATCH_TEMPERATURES = 4`, `vocab_exp_scale` (exact rational `2^{logits_exponent} / T` for `0 < T ≤ 4` with power-of-two cancellation), `cdf_columns_and_requests` (RFC-0058 column and `Range` request order for exact and bounded modes), and `TemperaturePlan::derive` (`TemperatureTable`, `TemperatureCertificate`), grouping up to four batch temperatures and selecting the narrowest `key_bits ≤ max_log2_rows` that covers the public logit spread and reaches `Exp_T[K_0] = 0` without altering `ClaimList` (`0x0201`; RFC-0007, RFC-0026).
  - **In-circuit Sampler PRF (`tessera_compiler::anchor`, `tessera_witness::SecretGraphInputs`).** `build_sampler_derive`, `build_sampler_key_block`, `build_sampler_stream_class`, `build_sampler_draws`, `build_standalone_sampler_prf`, and `build_pinned_sampler_prf` lowering RFC-0061 steps 1–4 (`"tessera-sampler-key/v1"` opening, `"tessera-sampler-derive/v1"` key derivation, `"tessera-stream-class/v1"` midstate + prompt class hash for `1 ≤ b ≤ 64`, and AES-128 counter draws tightly range-bounded in `[0, 2^32 − 1]` via binary-tree packing), paired with sealed `Secret<SecretGraphInputs>` builders `SamplerBaseKey::pinned_prf_inputs`, `SamplerKey::standalone_prf_inputs`, and `Engine::evaluate_secret` (`05 §3` OBS-1).
  - **`tessera_scoring::ExactTemperature`.** Exact-mode temperature certificate builder (`TemperatureRequest`, `TemperatureInputs`, `TemperatureCertificate`, `TemperatureRow`, `TemperatureBatch`) reusing the recorded sequence's `OneHot` indicators (`Û = 0`), scanning all real output rows before reporting the first `ERR-WITNESS-EXACT-MISMATCH` (`RequestId`, `TokenPosition`), and assembling power-of-two row batches with unique zero padding witnesses.
  - **Tests.** Adversarial isolation tests for every `VocabMax` and batched `InverseCdf` obligation (`crates/tessera-gadgets/tests/vocab.rs`), `TemperaturePlan` and RFC-0058 parity tests (`crates/tessera-compiler/tests/temperature_runtime.rs`), `10^5` random logit-to-`Exp_T`-to-`InverseCdf` agreement checks plus all 46 `derive.txt` and 369 `prf.txt` oracle vectors through the lowered IR circuits (`crates/tessera-scoring/tests/temperature_certificate.rs`), and end-to-end tiny-model teacher-forcing proofs verified by independent `tv-*` LogUp, product-forest, and sumcheck verifiers (`crates/tessera-prover/tests/temperature_scoring.rs`).
- Conformance vectors for the anchor circuits, from independent oracles written from RFC-0066 and RFC-0072 alone, by an author who read neither implementation ([#150](https://github.com/AbdelStark/tessera/issues/150), slice S1), and the prover workspace's checks against them:
  - **`spec/vectors/anchor-relations/`.** RFC-0072's public factors, each checked against the direct multilinear extension of its table over the padded cube:
    - `ψ̃` and `LT`;
    - `ConstAdd`'s shift `ĉ`, scalar and per channel, with the wrong `c̃(r_c)·ψ̃(r)` beside it on last axes of 3 and 5;
    - the selection kernels `S̃` and `K̃` of the rotations, the value levels' even and odd rows alone and combined, the message assembly, and `Slice`s on three axes and of lengths that are not powers of two, each with its relation on a random tensor;
    - recomposition's weights and zero-extensions for nine widths, two of them with a carry wider in bit variables than the result, the only ones where the result's zero-extension is not 1;
    - the `BitFn` polynomials against their truth tables;
    - recomposition's bit-axis sums, zero-check sum and final value on honest witnesses and on one bit flipped.
  - **`spec/vectors/anchor-graphs/`.**
    - **The graphs.** RFC-0066's SHA-256 graph node by node, in the lowering vectors' format, for the key commitment at `M = 1` and `M = 3`, one and three public messages, and two blocks whose committed bytes cross into the second. Each graph carries its public tensors and per-block counts, which reproduce RFC-0072's sizes table.
    - **The inputs.** The circuit's public inputs from bytes for 81 message sets: FIPS 180-4's examples, CAVP's messages, and `anchor/`'s key commitments and frames.
    - **The check.** An interpreter runs a graph on every set and gives FIPS 180-4's chaining words.
    - **The README** records the twelve readings the graph had to choose, each now quoting the sentence of RFC-0066 'The SHA-256 circuit' that fixes it. The compiler's builder made the same twelve choices, so RFC-0066 states them as they are built (its header's 'Clarified by'). Among them, `e`'s sum is `Add(shared, d)`, as RFC-0066's prose said, and its table now lists `d` last, where it listed `d` first.
  - **`spec/vectors/ranges/anchor.json`.** RFC-0006's ranges of those graphs, and eighteen small graphs that each break one of RFC-0066's and #527's range rules, or none, with the verdict. `ranges.json` is unchanged. `AssertEqual`'s domain is the open interval `(−(P − 1)/2, (P − 1)/2)` of RFC-0006's true-integer rule, which the compiler and the verifier check: the oracle's edge cases are `[0, (P − 1)/2]`, refused, and `[−(P − 3)/2, (P − 3)/2]`, admitted, where it first took the closed interval and wrote `2^{63}`, a constant the IR's `i64` cannot hold.
  - **The checks.**
    - `tessera-compiler`'s `anchor_vectors` test builds each case's graph with `tessera_compiler::anchor` and finds it equal to the oracle's node for node: inputs, nodes, constants, chaining words, public tensors and counts, the zero-checks from the schedule's stage 1. It computes the 81 sets' public inputs from bytes with the compiler's functions, which equal the oracle's, and runs the circuit on each to FIPS 180-4's chaining words. `analyze` and `settle` give the circuits the oracle's ranges, and the small graphs its verdicts.
    - `tessera-prover`'s `anchor_relations` test checks `LT`, `ψ̃`, `ĉ` through `public_value` and the `BitFn` truth tables against `relations.json`. The selection kernels, recomposition's weights and the `BitFn` polynomials over `Fp2` are the stage prover's own, which the stage tests cover.
- A reduction stage's round degree in `tessera-protocol/2`, in the prover workspace ([#565](https://github.com/AbdelStark/tessera/issues/565), RFC-0013 'Masked extensions' 'Degree'):
  - **The bound per protocol.** `tessera_sumcheck::Protocol` (`One`, `Two`) and `Protocol::max_degree(prefix)`: 10 at a reduction stage of `tessera-protocol/2`, `stage/{k}` with `k ≥ 1`, and 7 everywhere else, tree levels, stage 0, the merge and every sumcheck of `tessera-protocol/1`. `RoundPoly` holds degrees 1 to 10, and `prove_batch` proves them, with monomorphized loops for 8, 9 and 10. `tessera-protocol/1`'s messages and transcripts are byte-identical: every existing vector is unchanged.
  - **The keys.** The schedule's native instance carries its kind, `Relation::Native { kind, .. }`, from `Native::kind`, which `model_natives` and the prover's `ModelNativePlan` set alike, so a consumer of the schedule alone keys it. `schedule.json` is unchanged.
  - **The degree.** `tessera_compiler::schedule::degree_row` is RFC-0013's table: a `DegreeKey` (the relation or native kind, the `BoolFn`, the equality zero-check's operation, the `ConstMul`'s `Constant`, the public operands, and for the matmuls the stacking) gives its `DegreeRow`, `d_1`, `s`, `d` and the degree its tiny instance reaches. `protocol2_degree` keys a schedule's instance from the schedule and the graph, and `protocol2_stages` gives each reduction stage's rounds, folds and `d`. `stage_degree` checks a stage against its protocol's bound. An instance that matches no row, and a stage above its bound, are refused, `ERR-INTERNAL`.
  - **The vector.** `spec/vectors/schedule/degrees.json`, written by `cargo xtask gen-vectors` as RFC-0013 pins it: 44 cases, the conformance claim lists at `n` of 1, 4, 5 and `max_context` for the requests `none`, `prefill` and, at `n ≥ 2`, `generate`, on the composed schedules `tessera-protocol/2` proves. The generator checks the RFC's figures: `[10, 7]` in the 22 `servable` cases, `[10, 7, 7]` in the 22 `minimal` and `affine` cases. `tv-core` independently derives and checks all 44 cases.
  - **Tests.**
    - The per-row degree test (`crates/tessera-prover/tests/protocol2_degrees.rs`): each row, relation string, public pattern and constant class on RFC-0013's tiny masked instance, its round polynomials evaluated at 14 nodes; every round within `d`, `d` reached on every tight row and the "Tight" cell's degree on the others; a tight row lowered by 1 fails.
    - The no-row refusal, by key and on fabricated instances: a `MatMulAct` with one public operand and an `Xor3` whose three inputs are public. A stage of degree 11 is refused in `tessera-protocol/2`, one of 8 in `tessera-protocol/1`.
    - The anchor's circuits key every instance to a row, SHA-256's `Xor3` and `Maj` stage at 10; a `BitFn`'s public inputs set its row.
    - `prove_batch` proves batches of degree 8 to 10 at a reduction stage of `tessera-protocol/2`, over Goldilocks and over `F65537`, checked by test-local arithmetic and `tv-core`'s independent protocol/2 verifier; it refuses 11 there, and 8 in `tessera-protocol/1` and at a tree level, with `SumcheckError::Degree`. A batch of degree up to 7 has the same messages under either protocol.
    - The soundness calculator gives a round of degree 10 31,917/256 = 124.68 bits.
- The anchor in a batch's proof, specified ([#146](https://github.com/AbdelStark/tessera/issues/146)):
  - **RFC-0074, 'The Anchor in a Batch's Proof: One Graph and the Token-Byte Binding'** (Draft).
    - **One graph.** The model's lowering, then per anchor session (a `session_id` and the key commitment its requests' periods log) its key commitment, its AES-128-CTR, its plaintext's binding and each frame's SHA-256, in an order both workspaces derive from the statement's batch and the two verified logs of its window. No model node, gadget or table id moves; the exact-greedy certificate's id counts every node, and the anchor's four tables follow the model's.
    - **The token-byte binding.** Each session's plaintext, the bytes AES encrypts, is one committed tensor that `BitDecompose { 8, 0 }` range-checks. One `AssertEqual` binds its prefix and tail bytes to their public values. Another binds its token bytes, recomposed as `Σ_b 2^{8b}·byte_b` by one combined selection, to a gather of the token axis's rows, `OneHot`'s input. Given the tokens, every limb has one passing value.
    - **Batches, token groups, the layout.** RFC-0072's batch refusal no longer reaches through a gather that reads the tokens directly at real rows. The anchor's operations have no token axis, so token groups apply. Every row of a tensor non-zero at padding is committed, decided from public data. INV-PCS-1 now includes the anchor's inputs from the logs.
    - **No claim-list or statement field.** The circuits state nothing a model or deployment chooses, and every statement names its log, so every proof is anchored.
    - **Sizes,** from the compiler on `tiny-2-minimal`: a request of 2 prompt and 2 output tokens in two frames takes 14 stages and 17,555 instances, against 221 for the model alone. The binding is 2 zero-checks, 3 selections and 3 pool claims per session.
    - **The size bound.** A request's frames hold at most `2·n_r + 4` SHA-256 blocks, `n_r` its tokens: one token per frame is the most a token costs, and the 4 admit two frames of no token. Past it is `Reject(Anchor { Frames })`, at RFC-0015's check 2, so frames of no token cannot grow an anchor past what its tokens allow. A serving stack sends at most two frames of no token per request (RFC-0059, RFC-0063). A batch's anchor is then at most `A_b = Σ_r (2·n_r + 5)` SHA-256 blocks, the key commitments included, a function of the statement's lengths.
    - **An interim cap.** At the bound, a 16K-token batch streamed one token per frame would commit about 12 M advice rows (405B commits 0.95 M) and send over 15 GB in opened columns alone, past RFC-0003's 8 GiB decode limit. Until OQ-0074-1 and OQ-0072-3 are resolved, `A_b ≤ 4,096` per batch, checked from the statement (`StatementMismatch { Batch }`) and split by the packer. `ResourceLimits.max_anchor_blocks`, 4,096 by default, decides `Inconclusive(ERR-RES-MEMORY)` alike in every verifier. The cap bounds graph expansion, not proof bytes: production rollout still requires the complete proof to meet the unchanged 4 GiB gate, decode/resource limits and soundness budget across all admitted shapes.
    - **AES-128-CTR's graph to the node,** so that a verifier builds it from the specification alone, as RFC-0066 does for SHA-256: 285 nodes whatever the block count.
    - **Verdicts.** A log or manifest mismatch is the log checks' (`Inconclusive(ERR-CODEC-DIGEST)`, `Reject(Anchor)`), tokens other than the recorded ones fail stage 1 (`Reject(Sumcheck { stage: 1, round: n_1 })`), and 07 §6's `e2e_anchor_mismatch` takes one case of each.
    - **Open questions:** OQ-0074-1, one SHA-256 circuit for a session's frames of one length; OQ-0074-2, the soundness budget of a batch of many streamed frames.
    - It amends, as a draft's amendments, RFC-0006, RFC-0007, RFC-0011, RFC-0015, RFC-0019, RFC-0026, RFC-0031, RFC-0039, RFC-0041, RFC-0043, RFC-0045, RFC-0053, RFC-0054, RFC-0013 (R10's gap list, item 10), RFC-0046, RFC-0059, RFC-0062, RFC-0063, RFC-0066, RFC-0071, RFC-0072, 02 §3.2 and 07 §6, each amendment marked as the draft's.
- The anchor circuits' schedule, fourth part: AES's lookups, padded shapes and public lookup inputs; AES-128-CTR proves end to end on the CPU prover ([#526](https://github.com/AbdelStark/tessera/issues/526), RFC-0072 'Migration' slice 4):
  - **Padded positions.** `tessera_prover::PaddedWitness` gives a witness the padded positions of every tensor that a lookup non-zero at padding reaches (`schedule::tensors_nonzero_at_padding`): what each operation computes there from its inputs' padded positions, such as a padded AES block's `AesSbox(0) = 0x63` and its key-independent consequences, committed with the block's other lookup outputs. It computes over the engine's real values, which each of its tables must hold at its real positions. The stage tables (`WitnessSource::padded`) and the leaf stage's blocks read the same tables, and the stage prover refuses a witness without them, where it refused any lookup non-zero at padding with padded positions. The leaf stage's multiplicities count every block's padded positions.
  - **A gadget reading such values.** The schedule refuses (`ERR-INTERNAL`), before any claim, a gadget other than a lookup, `BitDecompose` or `AssertEqual` (`schedule::reads_padding`), such as a `OneHot`, `SoftmaxRow`, `RsqrtRow`, `Argmax` or `InverseCdf`, that reads a tensor with padded positions which a lookup non-zero at padding reaches: RFC-0035's row relations take a padded row's input as 0. `PaddedWitness` refuses the same graph by the same rule. RFC-0072 'Padded positions' states it, and that it relies on range analysis to pin committed inputs; no Llama graph meets it.
  - **A graph's leaf sites.** `LeafLayout::graph` lays out a graph without a lowering: each `TableMap` and `Bitwise` node is a site in node order with RFC-0008's key, `a + 2^8·b` for `Bitwise8`, the input or `pack(c, z)` for a `TableMap`, and `GraphLeafWitness` lends their tensors over their padded shapes. Its leaf stage runs ungrouped.
  - **Public lookup inputs.** `leaf::Read::public` marks a read of a public tensor, such as AES's counter blocks and round constants: the split sends no value for it (`LeafLayout::split_len`, `split_values`), and the verifier evaluates it at its point (RFC-0045 'The split'). The schedule no longer refuses a leaf site that reads a public tensor: its claim is that one public term, with no part. `GroupedLayout::new` refuses a layout with a public read. RFC-0045 'What is sent' evaluates a public tensor at its point `π_t(P[0..k])`.
  - **A graph's opening targets.** `tessera_prover::GraphOpenings` (`OpeningTarget`) lists, in `ModelOpenings`' order, every committed claim of a graph's proof: the pool's, each lookup output's split claim, on which the schedule leaves none, and the multiplicities'. It refuses a leaf stage with a product forest, whose claims it would not list. RFC-0072 'Messages, rejections and stages' states that order and where a lookup output's claim is owed, and RFC-0045 'The split' follows it. Their mapping to committed rows is the anchor's (#146).
  - **AES-128-CTR proves.** One block of NIST SP 800-38A's CTR vector under a committed key, and three blocks padded to four, range-checked by a `BitDecompose { 8, 0 }` of the key and the plaintext, prove through the real leaf stage and the stage prover in 3 stages. A verifier of the tests, which runs the production LogUp tree driver, accepts them, its openings the committed padded tables' extensions. The schedule is RFC-0072's AES row at `B = 1` and `B = 3`: 97 selections, 30 per-channel `ConstMul`s, 127 lookups, 154 instances, 154 evaluations and 332 split values, the leaf stage's 205 lookup inputs that are not public and 127 outputs, the independent count of RFC-0072's rules stage by stage.
  - **Tests.** An S-box output off its table, and a padded block's S-box outputs taken as 0, are `Reject(Lookup { table })` of the S-box; a changed split value, and a public counter changed on the verifier's side, `Reject(Sumcheck { stage: 0, round: 0 })`; `Gf256Mul { 3 }`'s lookup run on another column than `shifted`'s, its leaf value that column's, `Reject(Sumcheck { stage: 1, round: n_1 })`, which a verifier merging the two lookups' claims on `shifted` accepts; multiplicities that miss the padded positions fail LogUp. Each operation downstream of an S-box proves and verifies over the padded witness, every witness differing from the zero-padded one at padded positions. The tests' verifier computes its public tensors from the public inputs alone, so a round constant changed on either side is `Reject(Sumcheck { stage: 0, round: 0 })`; and a committed lookup output, multiplicity column or input other than the one LogUp and the stages read fails exactly its own opening targets, `Reject(Evaluation { commitment: Advice })`, and is accepted with exactly those removed.
- The weight masks' expansion, specified as a draft ([#147](https://github.com/AbdelStark/tessera/issues/147)):
  - **RFC-0073, 'Expanding an Epoch's Mask Seed into the Weights' Masks and Blinding Rows'** (Draft).
    - **Where its amendments are written:** RFC-0002, RFC-0004, RFC-0012, RFC-0018, RFC-0040 (with its 'Command line'), RFC-0064, 01 §6, 02 §1 and §2.2, 04 §5, 05 §3, 06 §5 and §7, and the simulator's §5 and R6.
    - **Each is marked as a draft's,** and takes effect when the RFC is accepted.
    - **03 §5's rows** for its two artifacts come with their code.
    - **SPEC.md** reserves the number, as it does RFC-0021's.
    - **The gap.** RFC-0064 makes a link a function of the model's refresh key `K`, the epoch and the weights, so that a link remade after a crash is the same bytes. RFC-0040 left the expansion of `masks_seed` into the masks and blinding rows to #147. Until it was fixed, only the build that made a link could remake it.
    - **The stream.** BLAKE3's extended output of `derive_key` under a domain of RFC-0004, read as 64-bit little-endian words. A word below `P` is the next value of `Fp`, and any other is skipped, never reduced. INV-ARCH-2 admits that branch, and the addresses a parallel kernel compacts to, as one narrow exception for the streams of RFC-0004: a public, fixed test that reads only the candidate word. RFC-0002, RFC-0018 and 06 §5 point to it. The skip pattern stays secret-class.
    - **The masks and the blinding rows.** Each committed row's `P_W` mask coefficients come from `weights-mask`, by committed row, and each blinding row's `k_T` values from `weights-blinding`, by blinding row. Both are keyed by `masks_seed` and addressed by the matrix's `commitment_id`, so any row can be drawn alone.
    - **Narrow matrices.** Until #564, `commit-weights` refuses a `tessera-protocol/2` commitment with any committed matrix of `k_T < k_W`, before drawing anything. `τ_W` is computed for `k_W`'s rate:
      - at `live-64`, `k_T = P_W` gives 40.3 bits instead of 64;
      - from `k_T + P_W ≥ n_T` the code is all of `Fp^{n_T}`;
      - past it a combined row passes every column check, with any values on `H` once `P_W ≥ n_T`.

      The verifier's refusal is #564's.
    - **The model entry.**
      - Each model entry has one directory under fixed names: the epoch record, every epoch's state and `.twr`, and the links.
      - In `tessera-protocol/2`, `commit-weights`, `weights refresh` and `prove` take it with `--entry`. The last two also take presented artifacts with `--present`, named by schema.
      - `K` lives in `WeightRefreshKey` (`0x0304`, `.twk`, secret) in the key store, naming the entry by its epoch-0 root.
      - `WeightEpochRecord` (`0x0305`, private) holds every epoch's `R_W`, the latest epoch's per-matrix marks (the next free blinding row and the column count), and a `retired` flag that persists a stale record.
      - The directory itself is locked, and every entry file obeys RFC-0067's path rules: no symbolic link, one hard link. Device and inode are checked before each rename, and a held lock is `ERR-RES-IO`. The lock spans a whole proof from its entry check: one proof at a time per entry.
      - A corrupted entry file is refused, never read as lost, until the operator moves it aside and, before any proof, runs `weights refresh` with the latest artifacts presented.
    - **Fresh and rebuilt states.**
      - A state is fresh only when written right after its epoch's root is appended: epoch 0 by `commit-weights`, `e + 1` by the first link from the latest epoch.
      - Every other state is rebuilt, at `B_T − 4` with no opened column, and serves only its epoch's link.
      - `reserve` runs only at the latest epoch, with the previous epoch closed. It starts from the record's marks as RFC-0073 'The cursor and the count' states, the one place that rule is written, and raises the cursor and the column count together in one record write. `record_queries` writes the state only.
      - Closed and open are taken over matrices with committed rows only.
      - A closed latest state, a file past the record or an authenticated later artifact retires the record.
      - A presented reservation raises the marks, by the same section, and `weights refresh` writes the raise into the record. A ledger counts through this entry's chain only. A kept `Round1` of a later root refuses like the latest `.twr`. A forged later root refuses one command and retires nothing.
      - A lost, stale or retired record fails closed.
      - **The residual, a rule on files,** exact over histories without crashes and conservative after one. The record is behind the truth, the last record written, with every file decodable, and either:
        - across a link, no state or `.twr` of a later epoch and no link from the record's latest epoch on is present, and the record's latest state is absent or open; or
        - within the epoch, the latest state is present and, raised to the record's marks, behind the truth.

        A copy of the entry is the other residual case.
      - **Its bound,** per matrix and per epoch, for each rollback or copy: up to `B_T − 4` rows in epoch 0 (4,092 at 405B) and `B_T − 8` later (4,088), and up to `P_W − τ_link` and `P_W − 2τ_link` columns past `P_W`, in every epoch it reaches. A copy's has no end in time.
      - **What catches it.**
        - A presented later root catches it across a link.
        - Within an epoch only kept `Round1`s catch it, since a verifier's ledger holds accepted proofs only.
        - Copies that prove concurrently are never caught.
        - Verifiers catch reused rows only in part (#158), and reused columns not at all (#569).
      - **The gate.** An exhaustive model of the rules, [`docs/testing/weight-entry-model/`](docs/testing/weight-entry-model/README.md), finds 0 mismatches between the rule and observed harm, either way. It is standard-library Python, run by hand until #147's slice S4 ports it to `tessera-pcs`.
        - **Its state space:** three epochs and two links, a backup after every write, every version of the eight entry files or its loss or corruption.
        - **Its regimes:** rows ending an epoch first, and columns.
        - **What is presented:** nothing, the latest `.twr`, a real ledger beside another model's chain, every kept `Round1` with the `.twr`, and every kept `Round1` alone.
        - **The other cases:** aliases, concurrent copies, forged artifacts, a `B_T = 0` matrix, and every crash and double crash.
        - **Its negative controls:** each earlier or naive reading of a rule shows the harm the rule prevents.
    - **The regeneration checks.**
      - A regenerated commitment is checked against the record's root before anything is written or sent.
      - Each combined row drawn from the seed is checked at one column drawn from the CSPRNG, with the codeword store or through a kept Merkle tree.
      - An evaluated column answer is checked against the root.
    - **Errors.** One code per failure.
      - A held lock or an I/O failure gives `ERR-RES-IO`.
      - The entry's checks and the regeneration checks give `ERR-USAGE`.
      - What `weights refresh` repairs gives `ERR-WITNESS-BUDGET`: a lack of rows or columns, a lost latest state, an unfinished link.
    - **Versions.** This is version 1, `tessera-protocol/2`'s. The expansion is tier A, so a change takes new domains and a new protocol version, and the claim list names it.
    - **Secrets.**
      - `K`, `masks_seed`, every hasher over them and a stream's output reader are key material: heap-held, cleared on drop, never `Clone` or `Debug`.
      - No secret input goes through `tessera_hash::Hasher`, `K`'s derivation included.
    - **Cost at 405B.** 654 GB of stream output per epoch. Under RFC-0012's proximity generator every proof draws all of it, about 1.2 GPU-s; 216 GB if #394 keeps the blinding rows out of the proximity row. The cost model is [#568](https://github.com/AbdelStark/tessera/issues/568) (OQ-0073-5).
    - **The conformance vector** `spec/vectors/weight-masks/` is specified, for #147's implementation to write:
      - streams under `tiny-2-minimal`'s `Embed` and `ChannelParams` identifiers;
      - a skipped word in each domain;
      - `tiny-2-minimal`'s masked commitment (`e_CP = 5`, `B_CP = 64`);
      - `synthetic`, refused at `test-small/1` and committed under a given test layout.
    - **Open questions** for #147:
      - the weights' proximity point in `tessera-protocol/2`;
      - the bytes of `WeightReservation`;
      - paying for the per-proof draws.
    - **Review.** Six independent adversarial readings.
      - **The first found:**
        - the rebuilt state's budget reset;
        - the per-proof cost of the proximity row's blinding rows;
        - INV-ARCH-2;
        - the version rule;
        - the narrow matrices;
        - the index spaces;
        - the regeneration check's reach;
        - the vector's gaps.
      - **The second found:**
        - that narrow matrices lose soundness below `k_W`, with an exact forgery past `n_T` (#564);
        - that a state's freshness could not be decided after a loss;
        - the exception's scope;
        - the unnamed key material.
      - **The third implemented the stream twice from the text, identically, and found the bound tight.** It found:
        - restores and second writers that still reused rows;
        - the vector's missing parameters;
        - the record's digest of `D`;
        - `K` through the public hasher;
        - the documents still contradicting INV-ARCH-2;
        - inconsistent error codes;
        - twelve nits.
      - **The fourth built the exhaustive model.** It found:
        - a closed latest state that should mark the record stale;
        - a lock that a removed file defeated;
        - RFC-0012's `record_queries` row;
        - a residual stated too narrowly and too broadly;
        - `prove`'s free path;
        - error codes not one per failure;
        - an undecidable link precondition.

        The model, with the fixes, became the gate.
      - **The fifth checked the model's fidelity and extended its state space.** It found:
        - that read 4's parameters hid a column under-count when columns end an epoch first, as in production, now counted at `reserve`;
        - that a real ledger holds accepted proofs only;
        - concurrent copies;
        - entry aliases;
        - a residual stated too broadly and an epoch-0 bound stated too low;
        - `B_T = 0`;
        - a stale record with no persistence;
        - corrupted files read as lost;
        - `--present` and `--entry` gaps;
        - missing error rows.

        Its replays, extended, are the gate.
      - **The sixth compared the committed model with every amended document.** The model followed RFC-0073 'The cursor and the count'. Three other documents worded that rule two other ways, and each wording, modelled, over-opens columns outside the rule. It also found:
        - a bound stated per rollback that holds per epoch;
        - "exactly" true only without crashes;
        - the link from `L` left out of the residual's wording;
        - a ledger of several entries refusing every command;
        - a corrupted file's handling against the operator's duty;
        - error-table gaps.

        The rule is now written once and cited elsewhere, and the gate runs both readings as negative controls.
  - RFC-0004's domain table gains `weights-mask` and `weights-blinding`. The headers of RFC-0004 and RFC-0040 now name RFC-0064's amendments too.

- The SHA-256 circuit proves end to end on the CPU stage prover ([#526](https://github.com/AbdelStark/tessera/issues/526), RFC-0072 'Migration' slice 3, 'Testing Strategy' 'The circuits'), tests only: the key commitment, one block with `k_s` and `salt_k` committed and its digest bound, proves and verifies in 14 stages, as do three key commitments (`M = 3`) and three public messages of two blocks. A digest one bit off on the verifier's side, and a round-0 carry bit off committed by the prover, are rejected at stage 1's final check, and each is accepted with exactly the stage-1 instance it breaks removed: the word's `AssertEqual`, or the carry's recomposition with the chain that only its claim reads.
- The anchor circuits' schedule, third part: selections, combined selections and `BitFn` ([#526](https://github.com/AbdelStark/tessera/issues/526), RFC-0072 'Migration' slice 3, OQ-0066-2):
  - **Claims.** A claim through a `Slice` or a `GatherPublic` of a tensor that is not public is a selection's virtual atom, and one through a `BitFn` a virtual atom. Within one claim, two or more selection atoms of one input tensor along one axis, at one point with one set of layer weights, are one atom of coefficient 1 in the first one's place, its members and their coefficients in `Atom::combined` (`schedule::Member`). A single selection is not combined, and no claim merges with another. Llama's atoms have no member, so its schedules and vectors are byte-identical.
  - **Relations.** `Relation::Selection` runs over the selected axis's `⌈log2 n_a⌉` variables alone, of degree 2, against the public kernel `S̃(r_a, ·)`, or `K̃ = Σ_t a_t·S̃_t` for a combination, and leaves one claim on `x` at `r_lo ‖ r'_k ‖ r_hi`. `Relation::BitFn(f)` is RFC-0010's `Elementwise` with RFC-0066's polynomial, of degree 3, or 4 for `Xor3` and `Maj`, a claim per input that is not public. `StageSchedule::atom_of` names the atom an instance proves.
  - **The stage prover** proves both under `stage/{k}/evals`, with no new label. `tessera_sumcheck::Selection` (`Selected`, `SelectionTables`, `SumcheckError::SelectionRows`) contracts `x` at the claim's other coordinates into a clearing buffer and builds the kernel; a combined selection's split value is its whole combination.
  - **SHA-256 derives.** A later block costs 2,945 instances of public bytes and 3,436 with committed bytes, the key commitment 3,431, in 10 and 14 stages whatever the block count, the commitment report's count. They are RFC-0072's sizes table, and an independent count of its rules gives the same instances, evaluations and split values stage by stage. AES-128-CTR maps its gathers too.
  - **Tests.** Each selection (a rotation, even and odd rows, slices on three axes, lengths that are not powers of two, the message's assembly) and each `BitFn` proves through a verifier that computes every kernel and polynomial by RFC-0072's and RFC-0066's closed forms, and is rejected at its stage with a changed input entry. A level and a value's five levels prove as one combined selection per stage, and a verifier with one member's kernel changed rejects them at the stage's final check.
- The anchor circuits' schedule, second part: the bit zero-checks ([#526](https://github.com/AbdelStark/tessera/issues/526), RFC-0072 'Migration' slice 2, OQ-0066-1):
  - **Gadget selection.** `BitDecompose` and `AssertEqual` take gadgets (`Gadget::BitDecompose`, `Gadget::AssertEqual`), numbered in node order with every other, with no table. Their preconditions are checked against the ranges given, `ERR-COMPILE-RANGE` when broken: `low + high ≤ 62`, a decomposed input in `[0, 2^{low + high})` or a committed input whose declared range holds it, and `AssertEqual`'s inputs true integers. `BitFn` takes none. No Llama gadget id moves.
  - **Stage 1's bit zero-checks.** After its other zero-checks, in node order, each node's in RFC-0072's table order: the padding check of the committed graph input a `BitDecompose` reads, directly or moved, when it has padded positions; booleanity of `b` and `h`; their bit padding when `low` or `high` is not a power of two; recomposition, one zero-check over `y` summed over the bit axis's `k = max(k_b, k_h)` variables, the outputs zero-extended on their high bit coordinates, its claims on `x` at `y`'s coordinates and on `b` and `h` at their booleanity checks' points; and `AssertEqual`'s `ψ̃·(ã − b̃)`, each side's evaluation unless public. New `Relation`s `InputPadding`, `BitPadding`, `Recomposition` and `AssertEqual`; booleanity is `Relation::Booleanity`, on the output its instance names. RFC-0014's `zero_checks` entry of each is its instance's variables, `n_y + k` for a recomposition.
  - **The stage prover** proves them under `stage/1/evals`, with no new label, stage 1 of degree 3. `tessera_sumcheck::Recomposition` is recomposition's kernel, `eq(z_y, ·)·Z_b` one multilinear public table. `WitnessSource::padded` lets a source give a tensor's padded positions, which the stage tables otherwise take as 0.
  - **The leaf stage** derives token axes only in token groups: `LeafLayout::model` of a graph whose token axes RFC-0054 does not derive, such as one with RFC-0066's operations, runs ungrouped (`LeafLayout::ungroupable`), and `GroupedLayout::new` refuses to group it.
  - **Tests.** Each check's violation is rejected at `Reject(Sumcheck { stage: 1, round: n_1 })` and accepted with exactly that check removed; the bits, and equal tensors, are the unique passing witnesses; the soundness calibration measures a non-Boolean bit passing booleanity at `1 − (1 − 1/|F|)^n ≤ n/|F|` over F65537.
- The anchor circuits' schedule, first part ([#526](https://github.com/AbdelStark/tessera/issues/526), RFC-0072 'Migration' slice 1):
  - **Graphs without layers.** A graph with no checkpoint has `L = 0`, and `tessera_compiler::derive_graph_schedule` derives its schedule from the graph and its tables, with logits only when it has some. `derive_schedule` is the same code on a lowering. `StageProver::for_graph` proves such a schedule.
  - **The graph's analyses.**
    - `public_tensors` closes publicness over the nodes that are not gadget-backed.
    - `nonzero_at_padding` decides from public data whether a `Requant`, `TableMap` or `Bitwise` node maps zero inputs to a non-zero value: `AesSbox`, and no Llama table.
    - `producers` gives each tensor's producer in `O(1)`.
  - **The refusals** (`ERR-INTERNAL`):
    - a committed output of a node that is not pointwise;
    - a `Sum`, `SumSquares` or `MatMulAct` over a padded axis with no summed input 0 there;
    - a layer template holding a `ConstAdd`, a selection, a per-channel `ConstMul`, a `BitDecompose`, an `AssertEqual`, a lookup non-zero at padding or a public tensor.

    A committed advice graph input counts as non-zero at padding unless a `OneHot` or a `BitDecompose` pins it (RFC-0006).

    The schedule also refuses a matmul with one public operand, which its relations cannot read, and a graph with layers whose schedule has no stage 1 to draw its layer point. No Llama graph is affected.
  - **Public terms.** A claim that reaches a public tensor, and one through a `ConstAdd`, carries `Claim.public` terms (`PublicSource::Tensor`, `PublicSource::Shift`), which the verifier evaluates and subtracts from the claim's value: a ready claim's instance or pool claim, and a split's check, are on the rest, and a claim of public terms alone has no part. `tessera_prover::public_value` evaluates them, a tensor source only when the graph's `PublicTensors` marks it public, the shift `ĉ` as the extension of `ψ·c`: `c·ψ̃(r)` for a scalar, `c̃(r_c)·Π_{a≠last} LT(r_a, n_a)` per channel.
  - **Relations.** A `Mul`'s public operand is read as `Public` and leaves no claim, as is a committed output's in its zero-check. The stage prover proves a `ConstMul` by a constant per channel (`ScaleChannels`) and the zero-checks of a committed `ConstAdd` and a committed per-channel `ConstMul`. A schedule with no reduction stage fills the leaf stage's pool claims after its splits.
  - **Leaf sites.** `BitDecompose` and `AssertEqual` take none, in the schedule (`has_leaf_site`) and in the prover's leaf layout alike.
  - **The commitment report's stage count** follows RFC-0072: selections and `BitFn` are relations, public tensors cost none, and the bit operations' zero-checks count themselves. AES-128-CTR takes 3 stages, and SHA-256 10 per block of public bytes and 14 with committed bytes. Llama's counts are unchanged.
- The anchor circuits' relations, specified ([#526](https://github.com/AbdelStark/tessera/issues/526)):
  - **RFC-0072, 'Proving the Anchor Circuits: Graphs Without Layers, Public Tensors, Selections and the Bit Zero-Checks'.**
    - **The gap.** Neither AES-128-CTR nor SHA-256 could enter a proof. The schedule assumed layers and logits; `GatherPublic`, `Slice` and a `ConstMul` by a constant per channel had no relation; no rule made a tensor public; and RFC-0066 left the zero-checks' messages open (OQ-0066-1, OQ-0066-2).
    - **Graphs without layers.** A lowering with no checkpoint has `L = 0`: every node is outside the layers, as a model's epilogue is, which is where #146 puts the circuits.
    - **Public tensors.** A tensor computed from public tensors alone, by a node that is not gadget-backed, is public. The verifier evaluates its claims, and a lookup that reads one sends no split value for it.
    - **The selection relation.** `ỹ(r_lo ‖ r_a ‖ r_hi) = Σ_k S̃(r_a, k)·x̃(r_lo ‖ k ‖ r_hi)` over the selected axis only, degree 2, with one claim on `x`. It covers `GatherPublic` and `Slice`, so no claim takes a Boolean coordinate. `ConstAdd` moves its constant into the claim's value.
    - **Combined selections.** Within one claim, selections of one tensor along one axis at one point combine into one selection whose kernel weighs theirs by their coefficients. Each level of a SHA-256 sum's value reads its input's even and odd rows, which would otherwise double the instances below it: about 3,400 instances a block of a message with committed bytes, against 71,700. Claims are never merged across a stage. Merging them by value would leave some unchecked (AES's `Gf256Mul { 2 }` and `{ 3 }` lookups read one tensor, SHA-256's round 0 leaves two claims on its first word at one point), and would change Llama's `tessera-protocol/1` messages.
    - **Padded positions** hold what their operation computes from padded inputs, except that a `ConstAdd` adds its constant at real positions only. Its claim shift is the extension of `ψ·c`: `c·ψ̃(r)` for a scalar, and for a constant per channel the channel extension times the other axes' selectors. So a padded position is non-zero only downstream of a lookup node (`Requant`, `TableMap`, `Bitwise`) that maps zero inputs to a non-zero value, such as `AesSbox(0)`; the other gadgets' outputs, `SoftmaxRow`'s exponentials among them, are 0 where their inputs are. SHA-256's padded messages are 0 throughout, and AES's padded blocks run under zero round keys. A summing relation (`Sum`, `SumSquares`, `MatMulAct`) over a padded axis is refused unless an input it sums is free of such a lookup or is an `Expand` or a selection along that axis. So are committed outputs of nodes that are not pointwise, the new operations inside a layer template, and, in a batch of several requests, a `ConstAdd` or `AssertEqual` downstream of the tokens or positions; none occurs in a Llama graph.
    - **The bit zero-checks.** They run in stage 1 and send their evaluations under `stage/1/evals`, with no new label. Recomposition is one zero-check summed over the bit axis, with the outputs zero-extended, so nothing is sent before `γ` and `x`'s claim moves to `y`'s coordinates. `AssertEqual` sends each side that is not public. `BitDecompose` and `AssertEqual` take gadget ids and no leaf site. A false bit fails stage 1's final check, `Reject(Sumcheck { stage: 1, round: n_1 })`, and an S-box output off its table fails LogUp, `Reject(Lookup { table })`.
    - **Sizes,** from the builders and their schedules. A SHA-256 block of a message with committed bytes holds 680 zero-checks, 2,528 selections, 352 `BitFn` products and 96 masked `Mul`s, about 3,400 instances, 4,800 evaluations and 700 split values: 2,441, 335, 79 and about 2,900 instances for a block of public bytes. A frame's SHA-256 takes 14 stages, the leaf stage included, and a public message's 10, whatever the block count. AES-128-CTR holds 97 selections, 30 `ConstMul`s per channel and 127 lookup nodes, in 3 stages. A message of many blocks with committed bytes costs the prover `O(N²)`, which OQ-0072-2's per-block message tensors would make `O(N)`.
    - **The cost RFC-0015's estimate misses.** Each distinct claim point on the anchor's small committed tensors is an RFC-0011 opening group: about 20 to 45 for a graph with the key commitment, two frames and AES, about 2.7 GB for 40 at `k = 2^{22}`. Each committed tensor also takes a row, 368 per SHA-256 block. OQ-0072-3 asks for a reduction to fewer points and packed rows.
    - **OQ-0072-1** asks whether the calculator should bound stage 1's zero-checks by the largest instance rather than their sum, since they share one point.
    - It amends RFC-0006, RFC-0008, RFC-0010, RFC-0014, RFC-0035, RFC-0037, RFC-0045, RFC-0046 and RFC-0066.

- R10's rank check on the compiled schedule, `cargo xtask check-mask-rank` ([#488](https://github.com/AbdelStark/tessera/issues/488); R10 stays open for the leaf stage's `tessera-protocol/2` points):
  - **`tessera_compiler::schedule::masks`.** It extracts every mask row's revealed functionals from the composed schedule and the advice and weight layouts:
    - pool atoms on committed advice tensors, each row its placed tensor's, and a one-layer read's its layer's (RFC-0069);
    - weight matrices, and each `ChannelParams` claim on its own row with its matrix's `β_T` (RFC-0065);
    - virtual tensors' values and one fold per reduction-stage instance, at `tessera-protocol/2`'s rounds.

    Leaf claims share RFC-0045's point. Values merge on their row, point and layer weights, and an advice value on its coefficient when it carries a per-layer constant that varies by layer (#556). A weight claim with such a coefficient is a finding of its own; folds never merge with values. Each indeterminate gets a seeded `Fp2` value, and Gaussian elimination over `Fp2` on product sketches proves generic independence. A `β_T` value with a secret term inside the others' span is refused (RFC-0065).
  - **The gate** runs every claim list of `spec/vectors/claims/` at the schedule vectors' lengths and 3, and the 405B claim list at every `n = 2^j` and `2^j + 1` up to 131,072, for the three requests. It checks RFC-0013's rule for committed tensors' mask columns for every case (#554, below), and a functional whose length is not its row's width is a finding of its own. It joins CI's `checks` job: 38 s on 16 cores, 261 s of CPU.
  - **What it found.**
    - A zero-knowledge leak in RFC-0013 as written. At 405B, for every `n` in `(16384, 32768]`, stage 1 reveals the `SumSquares` outputs at layer positions 0 and 38, of `n_V = 22 = log2 k`, at its last 22 challenges, which is their own folds' point. That reveals `Ṽ`. RFC-0013 now keeps a folded virtual tensor's mask below a full row, `c = min(n_V, log2 k − 1)`, at no cost. The old rule is the gate's regression control, refused at exactly those lengths and tensors.
    - Tensors of fewer than two variables at `n ≤ 2`: the tokens and the per-position sums of squares and their roots. RFC-0013 and RFC-0071 now pad `tessera-protocol/2`'s token axis, and a batch's slot, to at least four rows, which RFC-0047 lays out as a padded axis. The gate checks each length at that shape.
  - **What it does not cover yet** (simulator R10):
    - the leaf stage's `tessera-protocol/2` points (RFC-0068), and gadget outputs' values at their leaf points;
    - `SoftmaxRow`'s reciprocal and `Exp` reads, and its row sums at `1/2`;
    - gadget auxiliary columns, multiplicities, and tree-level and batching masks;
    - native raw openings (RFC-0053's greedy reads), and batches (RFC-0071);
    - committed groups whose layer weights carry per-layer constants, whose mask no document defines ([#556](https://github.com/AbdelStark/tessera/issues/556)).

    The gate checks RFC-0013's rule for committed tensors' mask columns (#554).
  - **Filed:**
    - [#554](https://github.com/AbdelStark/tessera/issues/554). RFC-0013 'Committed tensors' and RFC-0068's degree analysis disagree on whether a stacked or tiled committed tensor's mask takes its layer and tile variables. Neither reading gives a rank finding.
    - [#556](https://github.com/AbdelStark/tessera/issues/556): see above.

- The verifier's log lines, spans and run manifest, specified ([#446](https://github.com/AbdelStark/tessera/issues/446); the verifier's own changelog has the code):
  - **05 §2 'A human line'** states the human format both tools share, which only the prover's rustdoc said. A message follows `error: ` or `warn: `, or at `debug` and `trace` the level and the span path; the fields it does not name follow it.
  - **05 §2 'The verifier's lines'.** `tessera-verify` writes both formats itself, adding no crate (RFC-0001, DEP-1). Every command takes `--log-format` and `--log-level`, read before its other options. Its human lines are today's at the default level. Its three events are `tessera.tv_cli.command_failed`, `check_completed` for each RFC-0019 check that runs, and `verified`.
  - **05 §5.** `verify{proof_digest}`, the proof's envelope digest once it verifies, and `check{kind}`, one per check in the verdict report's names. In `live`, each message reports the checks it runs, repeated ones included.
  - **05 §6 'The verifier's manifest'.** `verify --run-manifest PATH` writes a public-class `tessera.run-manifest/1` after the report, with each input's envelope digest and one `tessera_verifier_seconds` sample per check. The build's commit and `rustc` come from `TESSERA_GIT_COMMIT` and `TESSERA_RUSTC_VERSION` at build time, since RFC-0001 refuses a verifier build script. The schema's description says so.
  - **02 §2 and §2.3** and the reference CLI guide list the options. The release checklist sets the two build-time variables.
  - **RFC-0001 'Enforcement' item 3** now states what `check-isolation` already refused: a build script or a procedural macro in a verifier crate.
  - **`e2e_verifier_json_log_lines_and_run_manifest`** (`tessera-e2e`, `exact_cli`) checks every line of an acceptance, a resealed tampered proof and a corrupt envelope against `log-event-v1`, the checks' order and outcomes, the proof's digest, and the manifest against `run-manifest-v1`. The canary scan now logs the verifier too, at `trace` in both formats.

- A batch of several requests in one proof unit, specified ([#357](https://github.com/AbdelStark/tessera/issues/357); implementation [#551](https://github.com/AbdelStark/tessera/issues/551)):
  - **RFC-0071, 'A Batch of Requests in One Proof Unit: Request Slots on One Token Axis'.**
    - **The gap.** Every document and both workspaces assumed one sequence per proof, so a statement's batch could hold one request only.
    - **Slots.** Each request of a batch of two or more takes a slot of `S = 2^{⌈log2 max n_r⌉}` rows of one token axis, positions first. The rows past a request's end are RFC-0035's padded rows.
    - **What changes.** The graph's nodes, the claim list and the plan do not. Attention reads keys and values within the query's slot. The row selector, the causal selector, `SoftmaxRow`'s root claim and RoPE's positions become sums over the requests. One greedy maximum serves the batch, with a key column and key relation per request with outputs, and `Û_r` sums each request's window.
    - **Statement checks.** The statement refuses an empty batch, a request named twice, a request without a prompt, a batch of several requests under another predictor than exact greedy, and a batch whose layout exceeds the maximum.
    - **The audit's packing** runs per slot class `⌈log2 n⌉`, so that a batch's slots are its requests' own powers of two.
    - **One request is today's proof,** bit for bit.
    - **Cost at 405B,** computed from the compiler's layout: a 16K-token batch of 16 requests of 1K is about 2.07 GB of `tessera-protocol/1` proof, against 25.4 GB as 16 proofs.
    - It amends RFC-0006, RFC-0009, RFC-0010, RFC-0011, RFC-0015, RFC-0016, RFC-0017, RFC-0019, RFC-0030, RFC-0034, RFC-0035, RFC-0037, RFC-0041, RFC-0044, RFC-0046, RFC-0047, RFC-0053, RFC-0054, RFC-0056, RFC-0057, RFC-0061, 02 §2.2 and 03 §4.

- The openings of `tessera-protocol/2`'s mask claims, specified ([#489](https://github.com/AbdelStark/tessera/issues/489)):
  - **RFC-0070, 'Opening Every Mask Claim in One Batching Sumcheck in `tessera-protocol/2`'.**
    - **The gap.** No opening covered a masked sumcheck's `G(r)`, whose functional is not an `eq` vector, a tree level's mask claim, each at its level's own point, or the `SoftmaxRow` leaf mask's. The compiler's 405B schedule has 2,668 such claims: one group per column point would add 38.25 GiB.
    - **The batching.** The claims are linear functionals of one masks' block `M`, which holds the masking polynomials' coefficients and the tree levels' and `SoftmaxRow` leaf's masks in aligned blocks. `α` combines them, and one sumcheck of degree 2 on `M + ρ·S` reduces them to one value `y`, in one more opening group. The pad `S`, laid out like `M`, is committed in Round 1.
    - **Zero knowledge.** Lemma 4a needs `S` to cover `M`'s layout, not a rank condition. `M̃(r*)` and `S̃(r*)` are never sent apart, and `M` holds only claimed masks.
    - **Soundness.** `(N − 1 + 2m + 1)/q`, about `2^{−116.6}` at 405B, which the calculator counts as one more stage.
    - **Cost at 405B.** +65.85 MiB of proof (the proof about 2.94 GB), `S`'s 1,560 committed rows (0.164% of the advice), 32 rounds.
    - **Labels.** Each masked sumcheck's `G(r)` is sent last in its evaluations. `masks/…` follows the last stage, before the blinding claims. `Reject(MaskBatch { round })` joins 04 §2.
    - It amends RFC-0010, RFC-0011, RFC-0013, RFC-0014, RFC-0019, RFC-0032, RFC-0036, RFC-0037, RFC-0057, RFC-0068, 04 §2, 08 §5 and `docs/zk/simulator.md`, where Lemma 4a joins §4.3a and R11 is met.

- The verifier's pending coin toss, specified ([#529](https://github.com/AbdelStark/tessera/issues/529); the verifier's own changelog has the code):
  - **RFC-0067 'The pending toss'.** The verifier keeps its share between the commitment and the reveal in a `VerifierToss` file (schema `0x0702`, `.tvt`, secret-class, verifier-local in 03 §5). It is written, synced and created exclusively before the commitment leaves. The reveal records the sample in the ledger before the sample leaves, and the file is cleared after the record. A verifier that restarts in between keeps its period's audit.
  - **What the file does not bind.** A toss edited and resealed derives another sample, which nothing accepts. A file kept past its record, after a stop, reveals the same sample into any ledger it is given, so one ledger per verifier stays the operator's to keep.
  - **02 §2.3.** `tessera-verify audit commit` and `audit reveal`. The recorder key is 64 hexadecimal digits. `--between` names the zero-length periods, as for `tessera audit sample`. `--new-ledger` creates the ledger, which otherwise must exist. The exit codes of each refusal are listed.
  - **05 §3.** The verifier's share before its reveal is secret-class.
  - **`tessera-codec`'s catalog** lists `VerifierToss` (`schema::VERIFIER_TOSS`), as it lists every artifact of 03 §5. `tessera inspect` renders such a file's envelope only, never its share.

- The OBS-3 canary scan, over the exact CLI cycle ([#164](https://github.com/AbdelStark/tessera/issues/164)):
  - **The canary model**, `spec/vectors/canary/`, from the independent Python oracle: two layers, vocabulary 4,096, three runs of eight INT8 weights whose bytes are distinctive 64-bit values, a prompt of distinctive token ids, the model's greedy continuation and an altered copy of it.
  - **`e2e_redaction_canary_scan`** (`tessera-e2e`, `exact_cli`) runs compile, weight commitment, NI and Live proofs, both verifications, four failures and `inspect` of every artifact, each producer command logged at `trace` in JSON and readable form. It scans every file the run leaves except the secret-class artifacts, and fails on any canary in decimal, hex or little-endian byte form. The same run finds the canaries in the model and token files, and finds one injected into a scanned proof and log.
  - **05 §3 OBS-3** now says how a token id is looked for: as a whole word of text, and in binary artifacts as each recorded sequence of ids. A lone 12-bit id matches public integers there, such as the proof's column indices.

- Claims at one layer in `tessera-protocol/2`, specified ([#535](https://github.com/AbdelStark/tessera/issues/535)):
  - **RFC-0069, 'Claims at One Layer in `tessera-protocol/2`'.**
    - **The leak.** The epilogue reads the last checkpoint at layer `L − 1` alone. At 405B the composed schedule has 33 such claims, on the last position's committed output at the Boolean layer 125, where the stacked kind's mask vanishes. Stage 1 has three: the final norm's sum of squares, its product and RFC-0050's residual padding zero-check. Stage 2 has thirty: the final evaluations of its `SumSquares` instances, which the root's 30 coefficient products start. They sit at two points.
    - **The mask.** Such a claim is on the layer's tensor in its own variables, `X_κ[ℓ]`, which takes an `Fp2` mask row of its own, as RFC-0013 masks every committed tensor. The pool gives the row the coefficient `Z(r)`, in the group the claim already joins.
    - **Cost at 405B:** one mask row, two committed rows; no opening group, round or label.
    - RFC-0037 now states the schedule's one read at one layer, the epilogue's of `c_L`, and that no relation proves a claim at one layer.
    - It amends RFC-0013, RFC-0037 and RFC-0068, and the simulator argument: R8 is met for claims at one layer.

- The leaf stage of `tessera-protocol/2`, specified ([#486](https://github.com/AbdelStark/tessera/issues/486)):
  - **RFC-0068, 'The Leaf Stage in `tessera-protocol/2`: Column Masks and a Layer Sumcheck'.**
    - **The leak.** RFC-0045 evaluates every looked-up tensor of a site in the layers once per real layer, at Boolean layer coordinates, where the masked extension is the tensor. Those values leave as the per-layer split values, 53,489 at 405B, and inside the LogUp columns, which the last tree level reveals unmasked (`docs/zk/simulator.md` §4.3, R8).
    - **Column masks.** Each instance's key column, and its value column when its table has values, takes an `Fp2` mask row: `κ'_I = κ̃_I + Z·R̃_{κ,I}`. The revealed column values are masked by the instance's own rows, whatever tensors and layers it holds.
    - **A layer sumcheck.** One masked sumcheck, after `logup/values`, reduces every instance's key and value claims to one value per site and tensor at a random layer point, where the tensors' own masks act. The column masks' terms are folded in, as a virtual tensor's are. Stage 1 stacks at its point and draws no `stage/1/layer/{c}`.
    - **Its batching.** The instances' sums are secret, so each summand, not each claim, is scaled by `2^{−(n − n_i)}`. Its degree is 4, derived from `λ` alone: a layered tensor whose mask takes the last layer variable reaches it, as the 405B `RsqrtRow` sites' do.
    - **`ChannelParams`.** Sites of two classes or more meet RFC-0065 'What the stages owe', and `tessera-protocol/2` admits them: RFC-0007's and RFC-0039's step 2.4 refusal is lifted.
    - **Cost at 405B,** computed from the compiler's layout: 36 mask rows (72 committed rows), one masked sumcheck of 22 rounds, and one more advice opening group, about 64 MiB, so `G ≤ 17`. The proof is about 2.74 GB in `tessera-protocol/2`, before the masking polynomials' openings (#489). The split values fall from 53,489 to 614.
    - It amends RFC-0007, RFC-0011, RFC-0013, RFC-0014, RFC-0032, RFC-0036, RFC-0037, RFC-0039, RFC-0045 and RFC-0065, the group counts of RFC-0019, RFC-0057 and 08 §5, and the simulator argument: R8 met for the leaf stage, R9 for every `ChannelParams` site.
  - **Found on the way:** the epilogue's claim at a Boolean layer ([#535](https://github.com/AbdelStark/tessera/issues/535)), the calculator's missing layer-point term in `tessera-protocol/1` ([#537](https://github.com/AbdelStark/tessera/issues/537)), and RFC-0013's virtual fold over non-column variables ([#538](https://github.com/AbdelStark/tessera/issues/538)).

- The verifier's audit accounting, its ledger and its audit report ([#151](https://github.com/AbdelStark/tessera/issues/151); the verifier's own changelog has the code):
  - **RFC-0067, 'The Verifier Ledger's Encoding and the Audit Report'.**
    - The ledger's canonical encoding (schema `0x0701`): a `BitSet` is `Bytes`, and each record of `proven` keeps its accepted requests' `Û`, the records sorted by sample.
    - Writes are atomic and locked.
    - What a report from the ledger alone states: `Accepted` or `Unproven(DeadlineMissed)`, exit 10 unless every request is accepted.
    - The verifier's audit plan without a cost model.

    It amends RFC-0019 'Ledger' and RFC-0016 'Audit report' and 'Calculators'.
  - **Prover and verifier select the same sample** over 1,000 seeds (`tessera-protocol`'s `audit_verifier.rs`, DEP-2). The two agree on the seed, every candidate's selection value and the selection, and the verifier decodes the prover's sample and re-encodes it to the same bytes. `tessera-protocol` dev-depends on `tv-codec` for it.
  - **`audit-plan-v1`** admits `null` for the figures that need the cost model.

- A sampled audit driven from the command line: `tessera audit batch` and `tessera prove --sample` ([#157](https://github.com/AbdelStark/tessera/issues/157)):
  - **`tessera audit batch --sample SAMPLE.tas LOGS --out-dir DIR`** packs a sample's selected requests into RFC-0016's batches. `LOGS` is `audit sample`'s log flags, read and checked the same way.
    - **The sample** must be its logs': its manifest is `--manifest`, and its selection is redrawn from its policy and coin toss over the period's candidates. A sample of another period, or one that selects a request the draw does not, exits 64.
    - **The lengths.** A request's `L` is its candidate's length, and its `prompt_len` and `output_len` the tokens of its `In` and `Out` frames over both periods (`tessera_anchor::lengths`).
    - **Greedy only.** Every request is in one group: temperature requests are grouped by RFC-0061's pins, which no crate implements yet.
    - **The output** is `DIR/batches.json`, the new `tessera.audit-batches/1` report, which `--json` prints too. A request whose frames give no lengths is listed without them.
  - **`tessera prove --sample SAMPLE.tas LOGS --transcripts DIR --batch K`** proves batch `K` of that report, in NI (`--out`) or directory Live (`--live DIR`) mode, and replaces the stub that named #157.
    - **One batch per invocation.** `K` is the report's index, from 0, and the batches are read again from the sample and the logs by the same function, never from the report. A `K` past the last batch exits 64. So does a batch of more than one request, naming #357, until multi-request proof units exist.
    - **The transcripts.** `DIR` holds one `SessionTranscript` per session, named `<session_id>.tst` in 32 lowercase hexadecimal digits.
    - **The pre-check** (`tessera_anchor::precheck`) runs before the prover commits anything. A missing or tampered transcript, or a request whose frames give no lengths, exits 20 with `ERR-WITNESS-ANCHOR`, and no `Round1` is sent.
    - **The tokens** are the request's `In` frames' tokens, then its `Out` frames', from the stored frames the log names (`SessionTranscript::request_tokens`). The `PublicStatement` is the reference path's, with the sample's `manifest_digest`.
  - **The specification.**
    - 02 §2.2 gives both command lines, the transcript convention and the exit codes.
    - RFC-0016 'Batching' names the report and the greedy-only grouping.
    - RFC-0015 'Prover pre-check' says where the CLI runs the check.
    - 03 §5 and `spec/schemas/` list `tessera.audit-batches/1`, with an example in `docs/perf/examples/`.
  - **Tests.**
    - `tessera-cli`'s `prove_sample` runs one session through `tessera recorder ref`, then `audit share`, `sample` and `batch`, proves the batch in NI and Live mode, and checks `tessera-verify`'s `Accept` against a statement the test builds with the verifier's own decoders. Required CI runs it.
    - Its refusals: a tampered or missing transcript (`ERR-WITNESS-ANCHOR`, with nothing sent); a batch past the last, and a sample edited to select an unsampled request; a batch of two; malformed samples, transcripts and logs (65); a sample of another period and missing flags (64).
    - A request whose frame has no token count, which the reference recorder refuses to log, so the recorder's core writes the period: `audit batch` lists it alone, first and without lengths, and `prove` refuses its batch with `ERR-WITNESS-ANCHOR`.
    - In `tessera-anchor`: `lengths` against the oracle's candidate lengths and the pre-check, and `request_tokens`' order, window and refusals.

- Bits in the IR and the anchor's SHA-256 circuit ([#143](https://github.com/AbdelStark/tessera/issues/143)):
  - **RFC-0066, 'Bits in the IR, and the SHA-256 Circuit'.** It amends RFC-0006, RFC-0010, RFC-0015 and RFC-0035, and 04's `ERR-COMPILE-RANGE`.
  - **Three operations** in `tessera-ir`, with their shape and kind rules:
    - `BitDecompose { low, high }`, a gadget whose outputs are committed bits: `low` result bits and `high` carry bits of a true integer. Booleanity, recomposition and a check that pins padded bit slots make them its unique digits (Lemma 1).
    - `BitFn { f }`, virtual: `Xor`, `Xor3`, `And`, `Ch` and `Maj`, each its multilinear polynomial, `BoolFn::eval`.
    - `AssertEqual`, an equality with no output.
  - **Range analysis.**
    - A decomposed integer lies in `[0, 2^{low + high})`.
    - A committed graph input may be decomposed instead, its bits being its range check, read directly or moved unchanged by `Reshape`, `Slice`, `Transpose`, `GatherPublic` or `Expand`. That holds only when its declared range, which its other readers trust, holds that interval. A byte declared `[0, 255]` under nine bits is refused: it could hold `256 + k'`.
    - A `BitFn` input is a bit that an operation proves, never a committed input's declared `[0, 1]`. `AssertEqual`'s inputs are true integers.
    - Proving a committed input's declared range through any other operation is [#527](https://github.com/AbdelStark/tessera/issues/527)'s.
  - **The CPU engine** evaluates the three operations. It refuses an integer outside its width and a non-bit `BitFn` input (`ERR-USAGE`), and unequal tensors (`WitnessError::Unequal`, `ERR-WITNESS-ANCHOR`). `Engine::for_graph` now validates its graph (`ERR-USAGE`), so a graph assembled past the builder cannot panic the new operations through its shapes.
  - **Gadget selection** takes `BitDecompose` and `AssertEqual` once their zero-checks are scheduled ([#526](https://github.com/AbdelStark/tessera/issues/526), RFC-0066 OQ-0066-1, above); `BitFn` is virtual. The profile check refuses them in a model (S3).
  - **`tessera_compiler::anchor`.**
    - `sha256` over `M` padded messages of `N` blocks (FIPS 180-4 §6.2), on bit-major words `[32, M]`. It returns the chaining value after every block; `sha256_bind` binds the last to a public digest.
    - Committed bytes, such as a key and its salt, enter through `BitDecompose { 8, 0 }` at the offsets a `Sha256Layout` names, and the public bytes as public bits.
    - A sum's value is a five-level tree of exact intervals, because `Sum` of bits times `2^i` would bound a word by `2^{36}` and fail every carry's width. The carries are 2, 3, 3 and 1 bits.
    - Helpers: `sha256_inputs`, `sha256_pad`, `sha256_blocks`, `sha256_bits`, `SHA256_IV`, `SHA256_K`, `SHA256_MASKS`.
  - **`tessera_compiler::settle`**, public: it gives a built circuit's tensors their analyzed ranges, as the Llama lowering's are.
  - **Cost.** 6,376 committed bits per block (`SHA256_BITS_PER_BLOCK`) and 8 per committed byte, in 6,032 nodes per block. A frame of `n` tokens commits 12,928 bits at `n = 0`, below 1,000 per token from `n = 20`, and about 430 per token at long frames. RFC-0015 now gives these figures in place of "about 1,000".
  - **Tests.**
    - NIST CAVP's 65 short and 64 long messages, the long ones block by block through the initial-value input, and a three-block graph that binds its digest.
    - The key and sampler commitments against `tessera_anchor`'s, and three frames whose committed bytes cross into their second block against the `sha2` crate.
    - A digest one bit off failing its `AssertEqual`.
    - Lemma 1 on every decomposition of a witness, and four tampers that each break booleanity or recomposition.
    - The range rules, one test per rule.
    - The compiler's independent interpreter on FIPS 180-4's `"abc"`.
  - **Review.** Two adversarial reads.
    - **The first** found the following, all fixed here:
      - the committed-input exemption above;
      - the word sum's interval;
      - padded carry slots with two witnesses;
      - `AssertEqual` failing on padded messages, now restricted to real positions;
      - a stage rule that contradicted RFC-0010;
      - the short-frame cost.

      It confirmed the circuit against all 129 vectors from the RFC's text alone. The zero-checks' messages and `GatherPublic`'s relation are open questions for #526.
    - **The second** found the following, fixed here:
      - the exemption's bypass through a `Reshape`;
      - free padded positions of committed bytes, now pinned to 0;
      - the bit-padding check's degree, 3 not 2;
      - the message's assembly and the node order missing from the text, now normative;
      - unvalidated graphs panicking the engine.

      The general rule went to #527.
    - **The third** found no soundness defect. It found:
      - the circuit's helpers handling the key and salt in plain buffers: their output now depends on public bytes only, a caller passes the committed bytes zeroed, and the committed input's values come from the prover's secret-class buffers;
      - the graph's text leaving the gather's row, the inputs' declarations and the binding's operand order open;
      - three untested moves.

      Each is fixed here, and the text no longer claims more than the code proves: completeness at padded positions is the graph's.

- The anchor's AES-128-CTR circuit, lowered into the IR ([#142](https://github.com/AbdelStark/tessera/issues/142)):
  - **`tessera_compiler::anchor`.**
    - `aes128_key_schedule` (FIPS-197 §5.2), `aes128_encrypt` over `B` blocks (§5.1) and `aes128_ctr` (SP 800-38A §6.5), with `aes128_ctr_inputs` for the input declarations and `RCON`.
    - The lookups are `AesSbox`, `Gf256Mul { 2 }`, `Gf256Mul { 3 }` and `Bitwise8 { xor }`, through the IR's `TableMap` and `Bitwise`.
    - ShiftRows and MixColumns' reads are `GatherPublic` over a byte-major state `[16, B]`, and each round key reaches the blocks through `Expand`.
    - Each round key is a `[16]` tensor of four XORs masked to its columns, since the IR has no concatenation and its range analysis would widen a sum of bytes.
    - `aes128_key_schedule` and `aes128_encrypt` serve the sampler PRF too (RFC-0061).
  - **Cost.** 1,072 lookups per block, 268 per four-byte token, and 720 per key (`LOOKUPS_PER_BLOCK`, `LOOKUPS_PER_KEY`). RFC-0015 'Circuits' now gives these figures in place of "about 250", and states that the circuit's inputs are range-checked by their producers: the key by SHA-256's bits (#143), the token bytes by step 1.
  - **Tests** (`tessera-witness`, on the CPU engine):
    - FIPS-197's key expansion and both cipher examples, and SP 800-38A's four-block CTR-AES128 vector;
    - 63 random keys of 1 to 9 blocks against the `aes` crate;
    - the range analysis over the circuit;
    - every honest lookup a row of its table, while one flipped ciphertext bit, or one flipped key bit under the pinned ciphertext, leaves lookups that are no row;
    - the lookup count.

- Masked `ChannelParams` claims, specified ([#487](https://github.com/AbdelStark/tessera/issues/487)):
  - **RFC-0065, 'Masked `ChannelParams` Claims in `tessera-protocol/2`'.**
    - **The leak.** Each claim sits at a point of `Ã_CP` whose padding and type coordinates are Boolean, so RFC-0013's `Z` vanishes there and the masked value is the secret's own (`docs/zk/simulator.md` §4.5). Two claims can also share a column point, and the evaluation rows shared one mask and one pair.
    - **The masks.** Each claim gets its own mask row, with `Z` over its type tensor's `c_i + λ` coordinates, which is its evaluation row's designated block. Each matrix gets one pair `β_T` for its blinding claims.
    - **The reservation.** A proof reserves `2·e_T + 4` rows of a matrix with `e_T` claims: six for `Embed`..`LmHead` as before, and 18 for `ChannelParams` at `e = 7`. `Round1`'s `WeightReservation` names each matrix's first row before the seed, and the verifier's step 5 checks each range.
    - **The budget.** `ChannelParams` has `B_CP = B_W + 2·(e_CP − 1)·⌊(B_W − 4)/6⌋` blinding rows, 12,280 at 405B. It serves as many proofs per epoch as the other matrices, epoch 0's 682 included, for 1.8% more codeword store and column answers.
    - **What the stages owe.** Every revealed value of a type tensor must sit at a point whose coordinates other than the lowest are challenges, and no evaluation row is opened without its mask term.
      - The affine sites' claims, from the reduction stages, meet this.
      - A site of two classes or more does not: the leaf stage sends its class tensor once per real layer before stacking. So until #486's leaf layout removes those values, `tessera-protocol/2` refuses such sites (RFC-0039 step 2.4, `Rescaling { site }`). That includes all of 405B's.
    - **Review.** Two adversarial reads, then the fixes.
      - **The first found:**
        - the leaf stage's per-layer values;
        - a symbol clash, `m_T` used for claims and for rows;
        - `B_W` left in RFC-0064's body;
        - missing pieces: the reservation's declaration, the blinding claims' layout, the step-5 range and the reject payloads;
        - a cheaper design, one pair per matrix with the mask rows as designated blocks, which this text adopts.
      - **The second confirmed** the shared pair's argument, for one proof and for the epoch. It also found:
        - before #486 no 405B proof could pass the evaluation check, and dropping the mask term would leak whole folds through the shared pair;
        - a misstated rank condition for values revealed in the stages, now stated per matrix;
        - documents not yet reconciled with the shared pair: RFC-0013 'Hiding openings', the simulator's R9 text and Lemma 6;
        - stale figures in RFC-0064.
  - **Amended:**
    - RFC-0012: `reserve`, the layout's blinding rows and the store's figures;
    - RFC-0013: the masked extension and 'Cost';
    - RFC-0014: the reservation and blinding claims in the messages, and the field term's `B_T`;
    - RFC-0019 step 5, RFC-0026, RFC-0031, RFC-0040 (the leaves over the blinding rows), RFC-0049, RFC-0052 and RFC-0064 (`B_T` in its body, and `ChannelParams`' 56 data rows).

    `docs/zk/simulator.md` marks R9 met for the reduction stages' claims.
  - **Not yet:** the calculators count `B_T` with #521, #147 commits the masks, and #162 tests them.

- RFC-0016's audit calculators and `tessera audit plan` ([#141](https://github.com/AbdelStark/tessera/issues/141)):
  - **`tessera_protocol::AuditPlan`.** From a policy, each bucket's `BucketTraffic` (requests, mean length, violating requests) and the cost model's `μ`, it gives the plan, per bucket and in total:
    - the detection probability against a prover that pads each violation into the later bucket of lowest rate, `1 − Π_b (1 − s̲_b)^{v_b}`;
    - the detection if none leaves its bucket, labelled as an upper bound;
    - the amortized overhead `Σ n_b L_b s_b μ_b η / Σ n_b L_b`;
    - each bucket's rate for a target overhead, `min(1, O/(μ_b η))`.

    `check_traffic` refuses a profile that is not one well-formed entry per bucket, or has no request. The arithmetic is binary64, with `ln_1p` and `exp_m1`, and none of it reaches a sample or an artifact.
  - **`tessera audit plan`** reads the policy and a `tessera.audit-traffic/1` profile. It takes each bucket's `μ` from the cost model at its mean length: a plan's counts with `--plan`, or the source's with `--source-defaults`. It prints a table, or with `--json` the `tessera.audit-plan/1` report, which `--out` writes too. A refused profile exits 64. It replaces the stub that named #157.
  - **The specification.**
    - RFC-0016 'Calculators' defines the traffic profile, `μ_b`, the plan report and the binary64 arithmetic.
    - RFC-0016 'Audit report' defines `AuditSummary`, `manifest_digest` and who writes the report, the verifier at the deadline (#151). It records a failed pre-check as `ProverReportedAnchorMismatch`.
    - New schemas: `spec/schemas/audit-traffic-v1`, `audit-plan-v1` and `audit-report-v1`, with examples in `docs/perf/examples/`.
    - 02 §2.2 documents the command.
    - **The verifier's `audit plan` (#151)** computes detection from the policy and traffic alone. It reads `μ` for the overhead from a `tessera.cost-model/1` report (`--mu`), since the cost model is the prover's. 02 §2.3's line changes from `--claims` to `[--mu MU.json]`.
  - **Tests.**
    - The source's §8 table reproduced from its `μ` and `η = 0.3`: 10.4%, 5.1%, 1.8% and 0.38%, and 100 violations detected at ≈100%, 99.5%, 83.3% and 31.4%.
    - The closed form within 0.5 points of a Monte Carlo run of `AuditPolicy::selects` over 100,000 samples, violations in place and padded.
    - The lowest later rate behind a higher one, and every refusal.
    - In the CLI: the committed example, `μ` and target rates equal to `bench model`'s, and each refused profile.

- The weight linking proof, specified ([#485](https://github.com/AbdelStark/tessera/issues/485)):
  - **RFC-0064, 'The Weight Linking Proof Opens the Difference'.** It supersedes RFC-0012 'Refresh and linking proof' and 'Failure modes' in part, and RFC-0014's weight-linking row. It amends RFC-0004, RFC-0012, RFC-0013, RFC-0014, RFC-0019, RFC-0040, RFC-0043 and 03 §5.
    - **The leak.** RFC-0012 opened both epochs' commitments at `ρ_T` and compared the values. In `tessera-protocol/2` that reveals `W̃_T(ρ_T)`, a functional of the secret weights (`docs/zk/simulator.md` §6.4).
    - **The difference.** One combined row per matrix covers both commitments: `eq(ρ_T, ·)` on the old rows, its negative on the new, and a blinding pair from each epoch. It is checked against the public claim 0 at the same columns of both commitments. Each commitment keeps its own proximity row. Nothing about the weights is revealed, which is the simulator's R3.
    - **One link per epoch.** A second link would open more than `P_W` columns of an epoch and reuse its proximity pair.
      - Each epoch's `masks_seed` is derived from the model's refresh key `K`, kept apart from the state files. The proof of work takes the least nonce.
      - So every link from an epoch is the same bytes, whether remade after a crash, from a lost or restored record, or on a second node.
      - A `link` transition commits by closing the old epoch.
    - **What RFC-0012 left undefined.**
      - The transcript: `weights-link/1`, the `link/{T}/…` labels and `link/query-pow`.
      - `τ_link = 122` and `g_link = 20`, taken from the weights' layout: 100.54 bits of proximity.
      - `WeightLinkProof`, which carries both public commitments.
      - The verifier's checks, with the claim list as an input. The ledger keeps one successor per root.
      - The four rows at each end of an epoch, counted once.
    - **Size.** About 1.42 GB per link at 405B, 178 MB less than two openings.
    - **Review.** Four adversarial reads, the last finding nothing substantive.
  - **The calculator.** Its weight-linking row omits `θ_T`'s `1/q`, which [#516](https://github.com/AbdelStark/tessera/issues/516) adds.

- The prover's pre-check and its stored transcripts ([#146](https://github.com/AbdelStark/tessera/issues/146), in part):
  - **`SessionTranscript`** (schema `0x0404`, `.tst`, secret-class) in `tessera-anchor`. It holds a session's identifier, `k_s`, `salt_k`, and every frame it sealed, with its counter and plaintext.
    - `SessionTranscript::record` keeps a frame once its header matches the plaintext and its counter is above the last.
    - The decoder refuses counters out of order and unknown direction tags. `envelope::write_file` writes the artifact with mode 0600.
  - **`precheck`.** It reads a batch's requests from the period's verified log and the previous period's, as the verifier will. It checks, in the verifier's order:
    - RFC-0062's session rules in each period, with one key across both;
    - RFC-0063's request rules;
    - the statement's lengths;
    - no counter shared across the window by the session's requests in the batch;
    - the replay of the stored transcript: the key commitment, and each logged frame re-sealed at its counter to its digest.

    The replay returns only whether each digest matches. A failure is `ERR-WITNESS-ANCHOR`, naming the request and the check (`PrecheckKind`).
  - **The specification.** RFC-0015 'Prover pre-check' gives the artifact's TCE and the check order, and 03 §5 lists `SessionTranscript`. RFC-0063 'The request's rules' settles the session rules' order across the two periods, who is named, the kinds under rule 4 and one counter across the window, and RFC-0062 'One verdict' points to it: the readings the verifier's checks (#150) take as well.
  - **Not yet.** The token-byte binding to `OneHot` needs the anchor circuits (#142, #143). The sampler key's checks come with #152.

- Audit sampling, RFC-0016 ([#140](https://github.com/AbdelStark/tessera/issues/140)):
  - **`tessera-protocol`.**
    - `AuditPolicy` with `RateBucket`, and `RateBucket::threshold_of`, which reads a decimal rate exactly as `⌊rate · 2^64⌋`. A policy is valid by construction: `AuditPolicy::new` and the decoder refuse buckets out of order and batch bounds below `2 · min_batch_tokens` (`ProtocolError::Policy`).
    - `CoinTossTranscript` and its check, `sample_seed`, `selection_value`, and `AuditSample` (schema `0x0403`) with `AuditSample::draw`, its selected requests strictly increasing.
    - `ShareRecord`, the prover's side of the coin toss.
      - `ShareRecord::answer` takes one commitment and one policy per manifest. Its fresh share is kept before it is sent, and another commitment or policy is refused with `ProtocolError::Answered` (`ERR-PROTO-ORDER`).
      - `ShareRecord::draw` draws the manifest's only sample from the record and the revealed share.
    - `batches`, first-fit decreasing within each group.
  - **`tessera-anchor`.**
    - `VerifiedLog::check`, RFC-0062's check of a log against its manifest, refusing with `LogMismatch` (`ERR-CODEC-DIGEST`).
    - `candidates`, a period's ended and unterminated requests with their lengths over two periods (RFC-0063), none for a period of zero length, and `frame_tokens`.
  - **`tessera audit share`** answers a verifier's commitment over a manifest under a `tessera.audit-policy/1` file (a new schema, with an example in `docs/perf/examples/`). It keeps its answer under `--state` first: a temporary file of its own, linked into place, then the directory synced.
  - **`tessera audit sample`** draws a period's sample from its log and the previous period's, bound by the chain through any empty zero-length periods, and from that record with the verifier's revealed share. Its `--seed` and floating-point `--rate` are gone, and so is `audit batch`'s `--min-tokens`, which the sample's policy holds.
  - **`tessera inspect`** renders an `AuditSample`: its policy, coin toss, seed and selected requests.
  - **The specification.**
    - RFC-0016 gives the encodings of the policy, the coin toss and the sample.
    - The coin toss.
      - The reveal comes last and the policy first.
      - The prover answers one commitment per manifest and keeps its answer, with the policy, before the share is sent.
      - The share is never a function of the manifest.
      - The transcript is the prover's record with the reveal, and no other.
    - Selection: candidates past every bucket are selected whatever the seed, and a period of zero length has none.
    - Detection: a prover can pad a request into a later bucket, so detection counts each bucket at the lowest rate of it and the buckets after it.
    - It fixes batching's bounds as first-fit decreasing meets them, at most one batch of each group under the minimum.
    - RFC-0004 gives the inputs of `audit-sample` and `audit-select`.
  - **Conformance vectors** in `spec/vectors/audit/`, from an independent oracle: candidates, rates, seeds, selection values, samples with their encodings and refusals, and batches.
- The software reference recorder, `tessera recorder ref` ([#139](https://github.com/AbdelStark/tessera/issues/139)):
  - **The command.** `--listen ADDR --upstream ADDR --log-dir DIR --key FILE --period SECONDS`, the flags of RFC-0015 'Reference recorder' (02 §2.2 now lists them all). It is a TCP proxy for tests and integration work; it does not by itself satisfy A-ANCHOR-2.
  - **Forwarding.** In each direction it cuts frames out of the byte stream by their magic, logs each, then forwards it. Inbound frames are thus sequenced before the upstream reads them (RFC-0063). The bytes forwarded are the bytes received.
  - **Sessions and periods.** When a connection ends, it logs the close of each session seen on it. At each period boundary it writes `<start>.tdl` (the period's log: its entries' TCE encodings, concatenated, now defined in RFC-0062) and `<start>.tlm` (the signed manifest's envelope), then carries open sessions into the next period (RFC-0062). `<start>` is the period's `period_start_ns`, so an empty period keeps its own files rather than sharing its `first_seq`'s with the next ([#512](https://github.com/AbdelStark/tessera/issues/512)). Each file is renamed into place once written, the log before the manifest.
  - **`tessera-anchor::recorder`.**
    - `FrameParser` cuts frames wherever the stream is split.
    - `Recorder` logs entries with their `seq`, closes sessions, and signs and rotates periods with their carries and lapses.
    - `encode_log` and `decode_log` write and read a period's log, and `random_log_id` draws a log's identifier.
    - `read_seed` reads a recorder seed into a heap `Secret`.
  - **Tests.**
    - The parser at every split point.
    - Entries against an offline recomputation of their digests.
    - Four periods forming a valid chain, with each manifest's root over its log, carries, lapses, and sessions that broke rule 1 left uncarried.
    - End to end through the binary: bytes unchanged in both directions, digests, a chain from genesis, and the open, frames, end and close in order.

- Request boundaries in the log ([#496](https://github.com/AbdelStark/tessera/issues/496)):
  - **RFC-0063, 'Request Boundaries in the Log'** (amends RFC-0015, RFC-0016, RFC-0019, RFC-0043, RFC-0061 and RFC-0062).
    - **The gap.**
      - **No end.** No entry recorded where a request ends. A request that crossed a period boundary was read in each period alone: its earlier part passed as the whole request while its tail escaped, and its later part refused an honest prover.
      - **No order.** No rule ordered a request's inbound frames before its outbound ones.
    - **The request's end.** The serving stack sends a cleartext request-end frame, `TRQE ‖ session_id ‖ request_id`, after the request's last frame. The recorder logs it as `RequestEnd`.
    - **Candidates.**
      - **Which requests.** Every request with an end in a period is a candidate of it, and one with a frame and no end in time is an unterminated candidate. Every logged frame therefore lies in a request the sample can draw. A first draft that asked for a frame before the end let an end placed first hide the whole request, found by an exhaustive search of small logs.
      - **The seed.** A candidate's length counts all its frames in its two periods, and the candidates depend only on entries signed before the seed.
    - **A window of two periods.**
      - **Its frames.** A request's frames are all its frames in its end's period and the one before. The verifier holds the previous period's verified log through `prev_manifest`.
      - **Its rules.** One end, every frame before it, each position once, the session's rules in both periods, and inbound frames before outbound ones. The last holds for honest requests because the recorder sequences inbound frames before the cluster reads them.
      - **Who implements it.** The checks are #150's, and the recorder's duties #139's.
  - **`tessera-anchor`.** `RequestEndFrame` and `DigestLogEntry::RequestEnd`, tag 4.
  - **Vectors.** `spec/vectors/anchor/` gives each request's end. In `manifest/`, a request of the first period ends in the second.

- The period log and its sessions ([#498](https://github.com/AbdelStark/tessera/issues/498)):
  - **RFC-0062, 'The Period Log and Its Sessions'** (amends RFC-0015, RFC-0016, RFC-0019, RFC-0043, RFC-0059, RFC-0060, RFC-0061 and 04 §2–3).
    - **The gap.** An inclusion proof shows that an entry is in the log, never that another is not. So no batch could show that its session had one open, that a frame had no duplicate, or where the open of a session from an earlier period was.
    - **The verified log.**
      - **Delivery.** The recorder delivers each period's log to the verifiers by a path that does not pass through the prover. A period not verified by its deadline is reported unaudited.
      - **The check.** The verifier checks the log against the manifest's root, count and `seq` range once, after the signature and before it derives the sample (RFC-0016 read it already). A log that fails is the verifier's input, `Inconclusive(ERR-CODEC-DIGEST)`.
      - **The handle.** The check yields a `VerifiedLog` bound to the manifest's digest, and every later use reads that copy. A batch supplies no entries, the audit needs no inclusion proof, and `AnchorFailure::Inclusion` is retired.
    - **A carry entry.** At the start of each period, the recorder logs a `SessionCarry` with the key commitment of every session still open and not idle for two periods, so each period holds the key of every session it uses and a statement needs no chain of manifests. The audit still links manifests, and a gap or a late log is an unaudited period. The recorder only adds entries.
    - **The session rules.** A session used in a period has exactly one open or carry before its other entries, one frame per position and no counter used twice. A break names the first request of the batch in the session: `Session` or `Frames`. The statement's lengths count all of a request's frames in the period, so none inside it can be left out.
    - **Implementation.** The checks are #150's, the carries #139's and the pre-check #146's.
  - **`tessera-anchor`.** `DigestLogEntry::SessionCarry`, tag 3.
  - **Vectors.** `spec/vectors/anchor/` gives each session's carry. `manifest/`'s second period carries the first period's session and closes it.

- Sampling randomness under a pinned key ([#497](https://github.com/AbdelStark/tessera/issues/497)):
  - **RFC-0061, 'Sampling Randomness Under a Pinned Key'** (amends RFC-0015, RFC-0016, RFC-0017, RFC-0019, RFC-0043, RFC-0059, 04 §2 and 06 §5–6).
    - **The gap.** Exact mode's PRF took a `request_id` and a session's `K_samp`, and the serving stack chooses both after a prompt arrives. Grinding them is best-of-`M` selection among `D`'s samples, and a covert channel, while `Û = 0`: `k` bits of the first draw for `2^k` AES calls on average.
    - **The trade.** Without randomness from another party per request, any per-request input is the stack's choice, its prompt included, while no input makes every request share one stream. `research/tools/sampling_streams.py` measures it from fixed seeds:
      - re-tokenizing two words reaches 256 and 253 of the 256 top bytes of `u_0`, under two keys;
      - one stream for every request lets 10 observed outputs predict the next prompt's first token 91.7% of the time;
      - 256 keyed classes are clustered from 4,096 responses, after which 4 tokens predict the next 92.2% of the time.
    - **The stream.** A keyed class of `b` public bits fills RFC-0017's `request_id` slot, so the stack's choice is at most `b` bits per logged request under one pin. The reference and the verifier's default is `b = 0`, with no per-request input; a verifier allows more only through its policy's `b_max`.
    - **The key.** The prover commits a base key, and the verifier pins the commitment with `b`, the start of a fixed-length window, and a nonce it draws on receipt. Further verifiers adopt the issuer's pins. The PRF's key is derived from all four, so the prover can neither grind it nor keep one it likes.
    - **Pins.**
      - **A request falls within one pin:** every frame lies in the window of its earliest frame's pin, up to a grace `G`. A batch falls within one pin, and RFC-0016 batches by pin.
      - **Checks:** each announcement arrives over TB-1 and is checked on arrival against the policy (`W`, `G`, `b_max`). Each pin keeps its own `W` and `G`, and the list is checked again at load, `Inconclusive(ERR-USAGE)`.
      - **Rejection:** a request outside its batch's pin is `Reject(Anchor { kind: SamplerKey })`. The pins and the check are #150's.
      - **Clocks:** A-ANCHOR-2 gains the recorder's clock bound.
    - **What remains the serving stack's:** the prompt, the output length, request boundaries, the temperature, which windows carry temperature traffic, the pin near a rotation and the pins a held prompt spans, each with its bound.
  - **`tessera-anchor`.** The session-open frame is `TSOP ‖ session_id ‖ key_commitment`, 52 bytes. `DigestLogEntry::SessionOpen` drops its sampler commitment, and `Session::open` draws a session's identifier, key and salt only.
  - **`tessera-scoring`.**
    - `SamplerBaseKey::derive` makes a pin's `SamplerKey`, which carries its class bits. The two are separate types, so the PRF cannot run under the base key.
    - `ClassBits` and `StreamClass`.
    - `Sampler::new(key)` keeps the key block's SHA-256 state on the heap.
    - `Sampler::class(prompt)` and `Sampler::draw(class, t)`. The hashes' state words are cleared where they are held.
  - **Vectors.**
    - `spec/vectors/scoring/prf.txt` has 369 rows, with `b = 0, 1, 8, 63, 64` and SHA-256's block boundaries. Rows whose class equals a `request_id` of #136's file draw its values.
    - `derive.txt` has 46 derivations, each chained through to a class and `u_0`.
    - The inverse-CDF rows now come from their own stream.
    - `anchor/` and `manifest/` follow the 52-byte frame.

- `tessera-anchor`'s log manifests, their chain and inclusion proofs ([#138](https://github.com/AbdelStark/tessera/issues/138)):
  - **RFC-0060, 'Log Manifests, Their Chain and Inclusion Proofs'** (amends RFC-0015). It fixes what RFC-0015 left open:
    - **A manifest's identity** is its envelope digest, which `prev_manifest` and a statement's `manifest_digest` hold.
    - **The chain.** Consecutive manifests share their log's identifier and continue its entries and periods without gap or overlap. The first links to zero, from entry 0.
    - **The tree** is RFC 9162's, `SHA-256("")` for an empty period, and inclusion proofs are RFC 9162's audit paths. On the wire a proof is `Vec<[u8; 32]>`, at most 64 digests, with its index and size taken from the entry and the manifest. That closes RFC-0043's deferral of the encoding.
    - **Ed25519** is RFC 8032's, verified strictly, with a 32-byte key and a 64-byte signature. Scheme 1 and unknown schemes are refused.
  - **`LogManifest`, `ManifestBody` and `SignatureEnvelope`** (schema `0x0402`). A manifest signs with the recorder's seed, held in a heap `Secret`, and checks itself in order:
    - its scheme, then its key's and signature's lengths;
    - its key identifier, then its pinning;
    - its strict signature, then its period;
    - its range of entries without overflow, and an empty period's root.

    `verify_entry` checks an entry at its own `seq` against the manifest's root and size.
  - **The chain checks.** `check_link` checks two consecutive manifests, and `check_chain` a log from its first, each reporting the first break.
  - **`merkle`.** RFC 9162's root, audit paths and verification.
  - **Tests and vectors.**
    - The 98 RFC 6962 inclusion vectors of `spec/vectors/rfc6962/`.
    - Proofs at random positions of random trees, checked also by the verifier's independent `tv-hash`.
    - `spec/vectors/manifest/`, from an independent Python oracle with RFC 8032's reference Ed25519.
    - Each strict Ed25519 rule alone: `S + L`, a small-order `R`, and a small-order key. The last two pass a lax verifier.
    - Every refusal, alone.
  - **Dependencies.** `ed25519-dalek` 2 joins the prover side. It is the crate the verifier's allowlist already names.

- `tessera-anchor`'s frames and digest-log entries ([#137](https://github.com/AbdelStark/tessera/issues/137)):
  - **RFC-0059, 'Frame Headers, Nonces and the Session Counter'** (amends RFC-0015).
    - **The gap.** RFC-0015's CTR nonce used the frame's position in its session, which neither the frame's header nor its log entry carried. A verifier auditing a sampled request could not rebuild its counter blocks, and a prover-chosen counter would have given `2^32` decryptions of each logged frame.
    - **The fix.** The header now carries `frame_seq` and is its own 41-byte TCE encoding; RFC-0015 said 40 bytes, which its fields did not add up to. `DigestLogEntry::Frame` copies the counter, after `length` as in the header. The counter blocks and each plaintext's 22-byte prefix are the circuits' public inputs.
    - **Session identity.** A session's identifier is random and drawn with its keys.
    - **Verifier check 2.** It now specifies the refusal of a repeated counter in a session, a frame length no plaintext has, and a session open after its frames, under RFC-0043's `Frames`. #150 implements it.
    - **The sampler commitment** is now opened in a circuit, binding the in-circuit PRF's key to it. RFC-0015 committed `K_samp` but checked nothing against the commitment. The circuit is #152's.
    - **Left to their own issues.**
      - What the log must show of a session: its opens, duplicate frames and periods (#498).
      - The completeness of a request's frames (#496).
      - The PRF's free inputs: the prompt's `request_id`, and the session that serves it, which the serving stack chooses after the prompt and a prover can grind (#497).
    - **The encodings.** Counter blocks are SP 800-38A's, `session_id[0..8] ‖ u32_be(frame_seq) ‖ u32_be(i)`. The RFC also fixes `TokenFrame`'s encoding and its agreement with the header, and the commitments' exact inputs with 16-byte salts.
  - **The types.** `FrameHeader`, `SealedFrame` and its digest, `SessionOpenFrame`, `TokenFrame` (its tokens in a `Secret`) and `DigestLogEntry` (schema `0x0401`), with decoders that refuse:
    - another magic;
    - a direction other than 0 or 1;
    - a frame of the wrong length;
    - a plaintext that disagrees with its header.
  - **`Session`**, the reference cipher.
    - **At open,** it draws the session's identifier, its key and both salts, keeping the secrets in heap `Secret`s and the key schedule boxed. It returns the session-open frame with the key and sampler commitments.
    - **One counter, one plaintext.** It never encrypts two plaintexts under one counter, and offers no decryption, which the protocol does not need.
    - **Sealing** uses AES-128-CTR under the counter it keeps. It takes no counter or key from its caller and is not `Clone`, so a session never reuses a nonce (INV-ANC-3), and it refuses the frame after `2^32 − 1`.
    - **The commitments** hash one block built in a cleared buffer and passed by reference to `sha2`'s compression function, so no hasher's buffer keeps a copy of a key. The compression function's own stack copies, like every stack copy, are beyond reach (OBS-6).
    - **The decoders** refuse a header `length` that no plaintext has, and report truncated, trailing and oversized inputs with their own codes.
  - **`spec/vectors/anchor/`.** Five sessions and fifteen frames from an independent Python oracle: #136's FIPS-197 AES, CTR checked against SP 800-38A F.5.1, and `hashlib`'s SHA-256. Two sessions start at counters `0x01020304` and `2^32 − 1`, so every byte of the counter word is exercised. The crate's unit tests build those sessions through a test-only constructor, with no test feature.
  - **Dependencies.** `sha2` 0.10 joins the prover side; `aes` 0.8 and `getrandom` were already there.

- `InverseCdf`, the temperature-sampling certificate of RFC-0009 Lemma 10, in `tessera-gadgets`, with RFC-0058 fixing its columns ([#144](https://github.com/AbdelStark/tessera/issues/144)):
  - **RFC-0058, 'InverseCdf Columns and Comparisons'.**
    - **Three vocabulary sums.** The partition `Z`, the prefix below the token `B = C_<` and the token's exponential `S = e_ŷ` are contractions over the vocabulary; `C_≤ = B + S` is never committed.
    - **The limbs.** The draw `u`, `Z` and `B` are decomposed in radix `2^16`, with `k = ⌈W/16⌉` limbs for `W = bit_length(vocab·2^σ)`.
    - **The comparisons.** `B·2^32 ≤ u·Z` and `u·Z + ψ ≤ (B + S)·2^32` take `k + 2` limbs, with coefficients below `2^33`.
    - **The order and counts.** It fixes the committed values' order, the range requests' and their exact counts: 79 values and 82 requests per output token at `σ = 32` and Llama 3's vocabulary, against RFC-0009's estimate of about 20 and 30.
    - **What the gadget proves itself.** Bounded mode's indicator gets OneHot's booleanity, vocabulary and count relations. A padded row's partition gets a zero-check, `(1 − ψ)·Z = 0`, so its limbs stay unique whatever the vocabulary certificate leaves in the exponentials. RFC-0035 lists it.
    - **A correction.** RFC-0009's partition bound `Z < 2^{σ+18}` becomes `Z ≤ vocab·2^σ`: a vocabulary of `2^18` equal logits reaches `2^{σ+18}`.
  - **`InverseCdf`.**
    - **The honest sampler**, `sample`, scans the vocabulary branch-free (INV-ARCH-2) and returns the bounded mode's indicator.
    - **`witness`** gives the row's values in RFC-0058's order. It refuses a token that is not the inverse-CDF sample of its draw with `InverseCdf::MISMATCH`, exact mode's deviation.
    - **`constraints`** gives the row's equalities, the padding zero-check among them, and its range requests.
    - **`evaluate`, `relations` and `suffix_weights`** give the three contractions and the suffix's reduction. `sampled_relations` gives bounded mode's OneHot relations.
  - **Shared code.** `Argmax` and `InverseCdf` share the suffix weights, and `RsqrtRow` and `InverseCdf` the comparison arrays' order.

- `tessera-scoring`'s reference sampler, which serving engines link or port (RFC-0017 'Sampler PRF', [#136](https://github.com/AbdelStark/tessera/issues/136)):
  - **`Sampler::draw(request_id, t)`.** It returns `u_t`, the low four bytes of AES-128 under the session's `SamplerKey` of `u64_le(request_id) ‖ u32_le(t) ‖ 0^4`. The key is secret-class: it lives in a `Secret`, AES's key schedule clears itself on drop, and neither type's `Debug` shows anything. `secret-access.toml` lists the one reader.
  - **`inverse_cdf(weights, u)`.** It returns RFC-0009 Lemma 10's token, the least `v` with `C(v)·2^32 > u·Z`, in exact `u128` sums. It refuses a vocabulary outside RFC-0007's `1..=2^18` and a distribution with no mass.
  - **`spec/vectors/scoring/`.** 144 PRF rows and 228 inverse-CDF rows, written by an independent Python AES-128, which is checked against FIPS-197's examples. The token satisfies the `InverseCdf` certificate's bracket on 10^5 random distributions.

- `docs/zk/simulator.md`, the draft zero-knowledge simulator argument for `tessera-protocol/2` (RFC-0013 'Simulator argument', RISK-6, [#145](https://github.com/AbdelStark/tessera/issues/145)):
  - **The method.** A linear-view lemma, the `Fp2` mask lemma, Ligero's encoding masks, Libra's masked sumcheck, the advice and weight combinations, column answers, aborted sessions, a weight epoch against colluding verifiers, OQ-0013-1's packing, and the error bound.
  - **The finding.** As specified, `tessera-protocol/2` is not zero-knowledge:
    - the leaf stage's per-layer split values and `ChannelParams`' claims sit at Boolean coordinates, where the masked extension equals the tensor;
    - the weight linking proof reveals `W̃_T(ρ_T)`;
    - masked weights have no joint bound across an epoch under the tensor proximity generator.
  - **What the theorems need.** Its §9 lists eleven requirements, R1 to R11, for them to hold. They include a `tessera-protocol/2` leaf layout without per-layer values (R8), masks for `ChannelParams` (R9), a rank check on the compiled schedule (R10), and an opening scheme for the masking polynomials (R11).
  - **Review.** Two independent adversarial readings corrected the first two revisions. A second maintainer's review is still required.

- The nightly workflow runs every night at 03:17 UTC (07 §2, [#96](https://github.com/AbdelStark/tessera/issues/96), `docs/testing/nightly.md`):
  - **Property tests.** Every `proptest` of `tessera-field`, `tessera-codec`, `tessera-compiler` and `tessera-protocol` runs at 65,536 cases (`PROPTEST_CASES`, which proptest 1.11 applies over each test's own case count).
  - **Fuzzing.** An hour across the verifier's six targets: `fuzz.yml`, now also a reusable workflow, ten minutes each.
  - **Calibration.** The soundness calibration of 07 §4, which ran on demand only.
  - **The failure report.** A scheduled run that fails opens an issue labeled `type:bug`, with the failed jobs' log tails, or comments on the one still open.
    - **Checking it.** `gh workflow run nightly.yml -f fail=true` runs the report alone, skipping every other job.
    - **The logs.** It reads each failed job's log with `curl`, which does not forward the token across the endpoint's redirect, and says when a log cannot be read. The first check, #481, came back with empty log blocks.
  - **07 §2's table** says what runs nightly. The 8B CPU subset and range run wait for #93's integer model.

- The CPU backend proves a grouped stage's merge in two passes (RFC-0054 'The prover in two passes', [#134](https://github.com/AbdelStark/tessera/issues/134)):
  - **`MergeTables`** (`tessera-sumcheck`). It now keeps a group's tables from that group's claims and `γ_h` alone, at the end of its pass, before `β` exists: `MergeTables::new(polynomials, groups, j*)` and `add_group(h, claims, γ_h, block)`. It checks those claims as `Merge::new` does. `prove_merge_two_pass` refuses tables kept with another `γ_h` or claim count. `largest_table_rounds` gives the most rounds tables can answer.
  - **`prove_grouped_leaf`** takes a `MergeMode`:
    - `InMemory`, every polynomial at once;
    - `TwoPass { j_star }`: each group's pass cuts its blocks from what the witness lends and keeps its tables, and P2b regenerates every polynomial once after the tables' rounds.
  - **The backend.** It runs `TwoPass` at `balanced_j_star`, the `j*` that balances the tables against the folded state. The polynomials are no longer copied up front.
  - **RFC-0054.** Its Testing Strategy now gives the boundaries the two passes can reach, and its 'Memory' paragraph gives the reference's `j*`.
  - **Tests.**
    - **Equivalence.** On a three-layer model, under four groupings, the two passes at every `j*` from 0 to the largest that fits, and at the balanced `j*`, send the in-memory merge's messages and leave its merged claims.
    - **Acceptance.** `tv-core` accepts the two-pass proofs.
    - **Determinism.** Two forced-budget proofs with one seed are byte-identical.
    - **Pass counters.** A dedicated `tessera-e2e` binary counts one P2 and no P2b for a stage that fits, and one of each for a grouped one.

- The CPU backend proves a grouped leaf stage when the memory budget calls for one ([#134](https://github.com/AbdelStark/tessera/issues/134)):
  - **The planner.** `PassPlan` takes token groups when the stage in one batch exceeds `memory_budget`: the fewest group bits whose largest group's lookup trees fit, with the merge's pass, P2b, after P2 (`PassPlan::leaf_groups`).
    - **The split.** Layers first, keeping at least 256 tokens per group; token bits only past the layer index's.
    - **The refusal.** A budget no grouping fits still fails before P1 with `ERR-RES-MEMORY`.
  - **The witness pass.** `Resident::with_grouped_columns` and `Regenerator::with_grouped_columns` count each table's row-0 multiplicity over every group's padded column (`GroupedLayout::columns`), so the advice commits the multiplicities the table batch proves.
  - **The proof.**
    - `Round1.leaf_groups` carries the grouping, in live sessions through `ProverEngine::leaf_groups`, which defaults to none.
    - P2 runs `prove_grouped_leaf`, then the graph's stage-0 claims at their merged points and the atom splits under `stage/0/split`.
    - `TreeProof` carries the groups, the table roots, the table batch's levels and the merge.
    - The native relations and the openings read the merged claims.
  - **RFC-0054 and RFC-0018.** They state the split policy and the plan's grouped form.
  - **Tests.**
    - **Forced budgets.** Budgets one byte below each plan give `g` = 1, 2 and 3. Each proof carries its grouping and the grouped messages, and the independent verifier accepts it non-interactively and in a live session (`public_cpu_api_groups_the_leaf_stage_under_a_forced_budget`, in CI's `tv-core/test-params` step).
    - **The witness pass.** Under four groupings it counts exactly the prover's grouped multiplicities, and at `g = 0` the stage's own column.
    - **Memory.** For three profiles and three requests, a grouped proof's measured peak stays within its grouped plan.

- The soundness calculator counts the leaf stage in token groups at the most groups a verifier accepts (RFC-0054 'Soundness', RFC-0014 'Soundness calculator', [#454](https://github.com/AbdelStark/tessera/issues/454)):
  - **`StatementSizes.groups`** (`LeafGroupSizes`): `G = 2^{min(14, t + n_ℓ)}`, or 1 for none; the leaf stage's level rounds; the merge's `n_max` rounds; and its `P` claims, the merged polynomials' count.
  - **The rows.** The union row adds `G` more copies of the leaf stage's level terms and the merge's `2·n_max + P + G`. The rounds row takes the merge's degree 2 and its batching `P + G`. Each group's levels, and the table batch's, cost at most the ungrouped stage's, whatever the split of `g` between tokens and layers.
  - **The refusals** (`ERR-USAGE`): `G` not a power of two up to `2^14`; `G > 1` in `tessera-protocol/2`; no grouping with grouped sizes.
  - **The 405B default.** `G = 2^{14}`, 861 level rounds of degree 3, `n_max = 42` and `P = 53,490`. The union row falls from 111.84 to 101.75 bits, and the rounds row from 121.99 to 111.90; both sets still meet every target. `tessera params check --json` prints `groups`, and the committed examples follow.
  - **`--lookups` and `--table-rows`.** The leaf stage's levels now follow the tallest tree: `⌈log2 max(N, T)⌉` levels when above the default's 42, with their rounds and the merge's `n_max`. `N = 2^50` gives 50 levels and a union of 101.36 bits, where the fixed 42 levels gave 101.75.
  - **The figures' source.** `tessera-bench`'s `the_default_statements_merge_sizes_are_the_405b_lowerings` counts the 405B lowering's 53,489 split values and its forest in both workspaces' layouts.
  - **The oracle.** `spec/vectors/soundness/generate.py` states the grouped sizes, their rows and refusals; 89 cases, among them a grouping whose union alone misses 100 bits. The report schema gains `statement.groups`.
  - **Across the workspaces** (DEP-2). The two default statements are equal field for field, and the 1,000-profile differential draws random groupings.
  - **RFC-0054's forest cases.** `StageForest::with_leaves` keeps a forest's shape over other leaves, so a test can plant a cheating prover's leaves. `tv-core` then checks two zeros planted in group 1's share of the forest:
    - **in a real row's tree,** whose root is already 0, the leaf stage cannot see it and carries it to the merged forest claim, which stage 1 rejects;
    - **in a padded row's tree,** of ones, it is rejected at the level where the forest joins, at that level's final check.

- The stages and the openings read the leaf stage's claims where a grouped stage leaves them ([#134](https://github.com/AbdelStark/tessera/issues/134)):
  - **RFC-0054 'The stage's outputs'.** A claim the ungrouped stage leaves at `P[0..n]` on a tensor of `n` variables lies, in a grouped stage, at the merged point `r(n, S)`. `S` is the low `g_T` variables of the tensor's token axis. The same holds for the graph's claims on each gadget input and on the logits, for each tensor's value, for the native relations' points and for the openings'. The forests' leaf claims are their merged claims, the multiplicities' the table batch's, and the merge's rounds bound every such claim.
  - **`LeafClaims`.** The interface the native relations (`ModelNativeProver::new`) and the openings (`ModelOpenings::derive`) now read: a tensor's point from its shape and token axis, a split value's point, the forest and multiplicity claims.
    - **`ProvenLeaf`** implements it with prefixes of `P`, so an ungrouped proof is unchanged.
    - **`GroupedClaims`** implements it over `prove_grouped_leaf`'s result, with `ProvenGroupedLeaf::merged_point` for `r(n, S)`.
  - **Tests.** For two groupings of a model's grouped stage:
    - every split value's point is its merged claim's;
    - every value read over its lookup's whole domain lies at the point its domain's shape and token axis give, which the stages derive without the group layouts;
    - the forest claims are the merged ones;
    - the native relations build their targets on them.

- `tessera-prover` proves a grouped leaf stage up to its atom splits ([#134](https://github.com/AbdelStark/tessera/issues/134)):
  - **The prover.** `prove_grouped_leaf` runs RFC-0054's steps 1 to 4:
    - `logup/alpha` and `logup/beta`, then `logup/roots` and `logup/values` absorbed empty;
    - each group's lookup columns, read through `GroupedLayout::source`, and its share of each product forest (`StageForest`), proven by `prove_group` under `group/{h}/…`;
    - each group's split under `group/{h}/split`, every tensor at the group's leaf point and at its bits on `S`, then `γ_h`;
    - the table batch;
    - `merge/beta` and the in-memory merge.

    It returns one claim per merged polynomial, at its own point (`ProvenGroupedLeaf::merged`).
  - **The multiplicities.** `grouped_multiplicities` counts each instance's lookups and every group's padding at row 0. `ModelWitness::stage_forests` gives the model's forests with their query and layer variables.
  - **RFC-0054.** It now states the orders the verifier must follow:
    - a group's tree schedule: its lookup trees in table order, then its forests;
    - the merge's polynomials: the ungrouped split's tensors, then the forests' leaf tables;
    - a group's claim order `k`;
    - a forest's group variables.
  - **Tests.** A model's leaf stage under three groupings, checked from the recorded messages:
    - every label in RFC-0054's order;
    - each group's key and value claims by RFC-0045's identity over its blocks;
    - RFC-0008's identity over the groups' roots and the table root;
    - the merge's rounds and its final check, with weights computed independently;
    - every merged claim equal to its tensor's or its forest's extension, and a site's layers at one point;
    - two identical runs, and the refusals before the transcript moves.

- `tessera-prover` lays out a grouped leaf stage's groups, from public data ([#134](https://github.com/AbdelStark/tessera/issues/134)):
  - **Token axes.** Each `SiteLookup` carries its domain's token axis, `token`, its lookup tensor's from `token_axes`. `LeafLayout::new` refuses one outside the domain.
  - **The groups.** `GroupedLayout::new(stage, layers, LeafGroups)` derives every group's layout.
    - **Its blocks.** Group `h` holds the blocks at its layer subset's layers, the prologue's and the epilogue's in subset 0. Each is restricted to its token subset: `2^{k − g_T}` entries, the block's index order kept (`GroupedLayout::source`). A block without a token axis is in token subset 0, whole.
    - **Its instances.** One per table it holds a block of, laid out by RFC-0045 'The column'.
    - **Its split.** One value per tensor its blocks read, site by site and layer by layer. Each is a claim on a merged polynomial, one per value of the ungrouped split (`MergePolynomial`): its coordinates in the group's leaf point, and the group's token bits on its group variables.
  - **RFC-0054.** 'What a group holds' and 'Group variables' now state a restricted block's index order, which tensors a group's split sends, its claim's point, and the agreement its lookups owe.
  - **Tests.** Every entry of every stage block in exactly one group, the group of its layer and its token's low bits, decoded independently. Each group block's extension equals its forms at the group's claims. Hand-derived group variables and coordinates. Unlike and empty groups, and each refusal at its rule.

- RFC-0054 defines the token axis every tensor's groups split, and `tessera_compiler::schedule::token_axes` derives it ([#134](https://github.com/AbdelStark/tessera/issues/134)):
  - **The gap.** RFC-0054 split blocks and tensors by "the token axis" but said nowhere how to find it. The IR has no axis semantics, and lengths cannot tell it: a head or channel axis can be as long as the sequence.
  - **The rule.** Structural, from the graph alone, input to output, so that the verifier's lowering derives the same:
    - the tokens and the positions start it at axis 0;
    - elementwise operations, weight products and the one-hot vectors keep it, and `Transpose` and `Expand` move it;
    - `Reshape` keeps it whole, the axes on either side regrouping among themselves;
    - the scores and the attention output keep their queries' rows.

    A graph that breaks a rule is refused at its node.
  - **Tests.** Both profiles at 1 to 129 positions and at 405B's widths: the activations, checkpoints, one-hot vectors and logits at axis 0; the scores and their rows at the query axis; weights, classes and multipliers without one; every token axis as long as the sequence. The axis stays put when a head or a head's width is as long as the sequence. Mutated graphs are refused at their node.

- A grouped leaf stage's trees, in `tessera-sumcheck` and `tessera-logup` ([#134](https://github.com/AbdelStark/tessera/issues/134)):
  - **Scoped labels.** `TreeScope` (`Stage`, `Group(h)`, `Tables`) and `prove_trees_in` run the tree levels under `tree/{ℓ}/…`, `group/{h}/tree/{ℓ}/…` or `tables/tree/{ℓ}/…`, through the new `Prefix::GroupTree` and `Prefix::TableTree`. `prove_trees` is `TreeScope::Stage`, its labels and messages unchanged.
  - **The groups.** `begin_groups` squeezes `logup/alpha` and `logup/beta`, then absorbs `logup/roots` and `logup/values` empty.
    - **One group.** `prove_group` proves a group's lookup trees beside its forests: the roots `(n, d)` in schedule order under `group/{h}/logup/roots`, its levels, and its value evaluations in schedule order under `group/{h}/logup/values`. It returns each instance's key and value claims (`LookupClaims`) and each forest's leaf claim.
    - **The table batch.** `prove_table_batch` proves every table tree under `tables/…`, its multiplicity claims included.
  - **The RFCs.** RFC-0014 and RFC-0054 now state the order of the empty `logup/roots` and `logup/values`, and that a group's values follow its roots' schedule order.
  - **Tests.** Every label in RFC-0014's grouped order; each instance's group fractions and table root summing to RFC-0008's identity; every claim its column's evaluation; a group's levels equal to the ungrouped ones under renamed labels; and the group schedule's refusals.

- `tessera-sumcheck` proves RFC-0054's merge ([#134](https://github.com/AbdelStark/tessera/issues/134)):
  - **The relation.** `Merge` holds the merged polynomials (`MergePolynomial`, with their group variables), each group's claims (`MergeClaim`), `γ_h` and `β`. It checks their shapes, and gives the first claim and each `W_i`, as a table and at a point.
  - **In memory.** `prove_merge` runs the batch engine over `W_i·P̃_i` at weight 1, under the new `Prefix::Merge`: degree 2, rounds under `merge/round/{j}` and `merge/r/{j}`, and every `P̃_i(r_i)` under `merge/evals`. Each polynomial binds its free variables first, then its group variables.
  - **In two passes.** `MergeTables` keeps, in each group's pass, the tables of RFC-0054 'The prover in two passes' for the first `j*` rounds, a class's claims combined by `γ_h^k`. `prove_merge_two_pass` answers those rounds from them, regenerates every polynomial once, folds the variables they bound, and runs the rest in memory.
  - **Tests.** A reference verifier replays the challenges on `tv-hash`'s transcript and checks every round, and the final evaluations against the weights it computes itself; they are the tensors' evaluations at the merge's point. Every false claim fails. The two provers' messages are equal at every `j*` the polynomials allow. Each shape rule refuses with its own rule.
  - **Not yet.** The grouped leaf stage that produces these claims follows in #134.

- `tessera-protocol` carries RFC-0054's messages ([#454](https://github.com/AbdelStark/tessera/issues/454)):
  - **The messages.** `Round1.leaf_groups` (`LeafGroups`, `MAX_LEAF_GROUP_BITS = 14`), absorbed with `Round1`, and `TreeProof`'s `groups` (`GroupProof`), `table_roots` and `merge`. `TreeProof::ungrouped` builds a leaf stage proven in one batch.
  - **The bound.** Decoding refuses more than 14 group bits with `ERR-CODEC-LIMIT`.
  - **The prover.** It keeps `leaf_groups` zero until grouped stages land (#134).
  - **The vectors.** The protocol vectors (an independent oracle) gain the new fields, a grouping at the bound, and the refusals past it. The tamper corpus's base proofs are regenerated, and its oracle gains the `leaf-groups` family: 14 cases and three check-order pairs.

- Run manifests and the metrics registry (05 §4, §6; RFC-0020 'Continuous measurement'; [#95](https://github.com/AbdelStark/tessera/issues/95)). `tessera compile`, `commit-weights`, `prove` and `bench model --out` write `run-manifest.json` (`tessera.run-manifest/1`, `spec/schemas/run-manifest-v1.schema.json`) at their exit point, whether they succeed or fail. It records the tool, git commit and rustc version, the host, the parameter set, every input and output artifact with its envelope digest, public seeds, the metrics snapshot, and the exit and error codes. A secret-class artifact is named without its digest. `tessera_codec::metrics` records phase and stage times, committed elements, lookups, table rows, product leaves and proof bytes, and each run derives its unit costs and glue fraction. `--run-manifest PATH` overrides the location. A test-only `test-rng` feature adds `prove --test-mask-seed` for the OBS-5 reproducibility test.

- Structured logging and tracing spans (05 §2, §5; [#94](https://github.com/AbdelStark/tessera/issues/94)). `tessera --log-format {human,json}` and `--log-level` choose stderr's lines; JSON lines are `tessera.log-event/1` (`spec/schemas/log-event-v1.schema.json`), and the human lines are unchanged. `tessera_codec::event!` and `span!` take only public fields (`Public<T>`, integers, `bool`, `&'static str`) and literal message templates; a `Secret`, a `SecretBuf` or a run-time `String` does not compile. Spans `compile`, `commit_weights`, `prove{batch}`, `witness_pass{layer_tile}`, `round1`, `stage{id}`, `opening{commitment}` and `round3` cover the prover; `trace` events are compiled out of release builds. `cargo xtask lint` rejects other paths to `tracing` and checks every event's and span's name. `tessera-quant --log-format json` writes the same format.

- SC-4 tamper corpus in `spec/vectors/tamper/`: 412 mutations of a valid tiny NI proof and a recorded Live session, one per verifier check from decoding to scoring, each with the verdict an independent Python oracle predicts from RFC-0019's check order. The `tessera-e2e` example `tamper-base` regenerates the base proofs byte for byte, and its `tamper_corpus` test runs every NI case through the independent verifier's public API. Required CI runs the corpus in its own job ([#135](https://github.com/AbdelStark/tessera/issues/135)).

- Exact-mode CLI suite now compiles a synthetic three-layer model, commits weights, produces NI and directory Live proofs, and obtains independent `Û = 0` verdicts for five-token sequences. `tessera prove --reference-tokens` uses clearing input buffers and exact public request metadata; optional compile policy flags retain their defaults. Required CI runs both separately built binaries, exact mismatch and per-round tamper regressions ([#133](https://github.com/AbdelStark/tessera/issues/133)).

- Real CPU proofs now exercise the independent verifier CLI, caller-pin failures, proof tampering, table regeneration, report schemas and output safety. Required CI runs the separately built verifier binary using its own workspace and lockfile ([#131](https://github.com/AbdelStark/tessera/issues/131)).

- Public CPU `Prover` API for one exact-greedy request in NI and Live modes, reusing the protocol state machine and existing witness, tree, stage and streaming PCS kernels. The producer derives its own original opening claims, validates model/plan/epoch/configuration coherence, retains owned public P1 metadata and a commitment-bound P4-to-P5 permit, and clears per-proof witness/randomness on completion or abort. GPU targets and multi-request units fail explicitly; pass planning and zero knowledge remain separate work ([#130](https://github.com/AbdelStark/tessera/issues/130)).

- Protocol/1 NI and one-shot Live drivers now verify real CPU runtime proofs against application-owned statements and epoch weight artifacts. Live sessions draw fresh OS coin-toss/query randomness and clear unrevealed shares. Typed verdicts, exact native-gadget errors and deterministic public JSON include explicit test-parameter/anchor/ledger/ZK boundaries. Real six-case NI/Live integrations, 100 complete replays in each mode and input/report allocation tests cover the accepting path. RFC-0053 is accepted. The current unit is one exact-greedy request; broader batching, full resource hardening and the CLI remain separate work ([#128](https://github.com/AbdelStark/tessera/issues/128)).

- Complete runtime geometry and preset budgets now count shared tree batches, mixed graph/native stages, padded lookup populations, original opening groups and actual matrix rows. Budget checks enforce the unchanged preset name, commitment layout and session mode, with fallible metadata/report reservations. Runtime packet sizes and the prover transcript schedule are checked against actual CPU proofs and independent vectors. Accepting NI/Live session drivers remain under [#128](https://github.com/AbdelStark/tessera/issues/128); RFC-0053 remains Draft.

- Clearing CPU runtime witnesses now supply exact-greedy maxima and independently padded keys, with complete shared-table multiplicities. Producer native token/key/selected relations and prompt-only observations compose with the model graph and canonical advice/weight openings. Same-point weight aliases retain every value check against the original evaluation row and proximity check. Bounded runtime proofs exercise empty/ragged domains and regenerated rows in both transcript contexts; complete budget enforcement and NI/Live verdict drivers remain under [#128](https://github.com/AbdelStark/tessera/issues/128), and RFC-0053 remains Draft.

- Verifier table regeneration now reserves its digest buffer and bounded header fallibly, and streams across hash-block boundaries without growing the buffer. Class copies reserve fallibly; Range/Requant evaluation uses fixed-size public coefficient arrays. Sampled allocation caps return `ERR-RES-MEMORY`, including an implicit greedy Range table, instead of aborting ([#128](https://github.com/AbdelStark/tessera/issues/128)).

- Per-request exact-greedy metadata now preserves the model catalog, reuses or appends its public-span Range table, places a separate maximum before runtime multiplicities, and derives complete leaf counts. The independent verifier composes token/key/selected relations with the existing model batch and retains every original graph/PCS debt, including prompt-only logit observations. Runtime witness generation, complete model proofs, budget enforcement and the accepting driver remain under [#128](https://github.com/AbdelStark/tessera/issues/128); RFC-0053 is Draft.

- Independent exact-greedy boundary components retain every original logit, OneHot, token and certificate-maximum obligation. Literal conformance fixtures and CPU mixed-sumcheck/PCS tests cover ragged domains, public range widths and compensating carry-value forgeries. RFC-0053 remains Draft; complete runtime proofs, full semantic bindings and an accepting session driver remain under [#128](https://github.com/AbdelStark/tessera/issues/128).

- Canonical CPU advice rows now feed the existing commitment, combined-row and column passes from resident or regenerated witnesses. Complete bounded-table model proofs in both modes continue through every original advice/weight opening, with matching replay messages and exact tampering errors. Whole rows of public table-tree padding remain implicit zeros (RFC-0011 rule 4). The protocol driver, semantic token/greedy bindings and final verdict remain under [#128](https://github.com/AbdelStark/tessera/issues/128).

- Complete CPU model native relations (`native::ModelNativePlan` and `ModelNativeProver`): all root quartic coefficients, Softmax key/forest/reciprocal relations and every padding axis join the existing graph stage batch. Actual bounded-table CPU proofs in both modes retain original source identities, points and values. Openings and the complete protocol driver remain under [#128](https://github.com/AbdelStark/tessera/issues/128). Model transcript geometry now includes the residual feature axis required by RFC-0050's simplified root input binding; independent golden transcripts reflect the additional zero-check coordinate.

- Mixed stage composition: the compiler can interleave native relations with graph atoms, defer graph sources produced by native reductions, and project their final points through the existing split/pool routing. `StageProver::prove_stage_composed` uses one batch and one split per stage. Canonical CPU leaf proofs now continue through Softmax key/forest/maximum-padding checks and every graph stage in both modes; every returned graph/native source is checked against the fixture witness. Complete model native checks and canonical openings remain under [#128](https://github.com/AbdelStark/tessera/issues/128).

- Canonical CPU model leaf proofs: `LeafLayout<ModelColumn>::model` reuses the gadget constraints for every Requant, SiLU, root and Softmax lookup, including virtual tops and repeated ranges. Lookup requests now own their domains; existing Requant sites retain `Term` as the default source type. `ModelWitness` assembles canonical tile auxiliaries, virtual inputs and padded Softmax forests in clearing buffers. Complete bounded-table tiny-model leaf proofs verify with the independent production tree/split checks in both modes, and resident/regenerated sources produce identical messages. Native stage reductions, openings and the complete protocol driver remain under [#130](https://github.com/AbdelStark/tessera/issues/130).

- CPU leaf-stage integration: `prepare_leaf` batches lookup trees and product forests, then `PreparedLeaf::finish` binds lookup tensor values followed by graph atom splits in one `stage/0/split` message. Its exclusive transcript borrow prevents an intervening challenge. Standalone `prove_leaf` and `StageProver::leaf_splits` retain their behavior; the full model driver remains under [#130](https://github.com/AbdelStark/tessera/issues/130).

- Prover workspace with the twenty crates of RFC-0001 as skeletons, edition 2024, minimum supported Rust 1.85 and an exact toolchain pin (1.95.0) ([#25](https://github.com/AbdelStark/tessera/issues/25)).
- `cargo xtask` dispatcher, and a test that enforces the unsafe-code policy of RFC-0001: every crate forbids unsafe code except `tessera-field` and `tessera-gpu`, which deny it crate-wide ([#25](https://github.com/AbdelStark/tessera/issues/25)).
- `deny.toml` license, advisory, ban and source policy for the prover workspace ([#25](https://github.com/AbdelStark/tessera/issues/25)).
- `LICENSE-APACHE` and `LICENSE-MIT`, and `cargo xtask lint`, which checks the unsafe-code policy, the SPDX header of every Rust, Python and CUDA source file, and the dual license in every Cargo and Python manifest ([#26](https://github.com/AbdelStark/tessera/issues/26)).
- `cargo xtask lint` checks the verifier crates too: each must forbid unsafe code, with no exemptions ([#28](https://github.com/AbdelStark/tessera/issues/28)).
- `cargo xtask check-isolation`, the verifier-independence gate of RFC-0001 'Enforcement'. It implements the RFC's five checks: no `tessera-*` or out-of-tree path package in the verifier graph; every third-party package from crates.io, allowlisted at a compatible version and scoped to its `direct-dependents`; no file reference or symlink reaching outside `verifier/`; equal toolchain pins; verifier crates only as prover dev-dependencies in the resolved graph. It adds hardening against bypasses found in review: every target of a verifier path package must compile a source inside `verifier/` (`[lib] path` can point anywhere), no build scripts or procedural macros in the verifier, include and `path` arguments must be plain literals, and no `[patch]`/`[replace]`, cargo configuration or alias rewrite that could redirect the verifier's dependencies or neuter the gate ([#36](https://github.com/AbdelStark/tessera/issues/36)).
- Pull-request pipeline (`.github/workflows/ci.yml`): formatting, clippy, rustdoc, tests and an MSRV 1.85 check in both workspaces, `cargo xtask lint` and `cargo xtask check-isolation`, a Python job that activates with the package, and a tier-2 macOS build on `main`. `cargo xtask lint` now also checks every relative Markdown link and heading anchor ([#42](https://github.com/AbdelStark/tessera/issues/42)).
- Test-only cargo features of RFC-0001: `test-rng` (`tessera-prover`, `tessera-pcs`, `tessera-sumcheck`), `test-params` (`tessera-protocol`) and `unsafe-debug-dump` (`tessera-prover`, `tessera-witness`), each with a `compile_error!` guard that fails any release build enabling it. `cargo xtask check-test-features` finds every crate of either workspace that declares such a feature and checks that a release build fails on the guard and a debug build compiles; CI runs it ([#37](https://github.com/AbdelStark/tessera/issues/37)).
- `cargo xtask check-pr` and the `pr-policy` workflow: a pull request that changes `crates/*/src` or `python/tessera_quant/` must add an entry to `CHANGELOG.md`, one that changes `verifier/` must add one to `verifier/CHANGELOG.md`, and one that changes `spec/` must add one to either, unless it carries the `no-changelog` label; every commit must carry a `Signed-off-by:` line naming its author (DCO, 09 §6) ([#51](https://github.com/AbdelStark/tessera/issues/51)).
- `cargo xtask check-deps` and the `dependencies` CI job: `cargo deny check` (licenses, advisories, bans, sources) in both workspaces, `cargo vet --locked` in the verifier, and a self-test on `xtask/fixtures/deny` showing that the verifier policy rejects an MPL-2.0 crate, the prover policy rejects it until an exception with a reason is recorded, and the advisory check rejects a crate under RUSTSEC-2021-0003. `cargo xtask lint` checks the policy files: both allow exactly the licenses of 09 §7, every prover license exception has a `# Reason:` comment and every ignored advisory a `reason`. CI installs cargo-deny from a SHA-256-pinned release binary and cargo-vet 0.10.2 with `--locked` ([#50](https://github.com/AbdelStark/tessera/issues/50)).
- `CONTRIBUTING.md` (workflow, DCO sign-off, changelog, reviews and the second reviewer, RFC lifecycle, issue labels), `SECURITY.md` (the disclosure policy of 06 §8), `CODE_OF_CONDUCT.md` (Contributor Covenant 2.1), issue forms for features, bugs and RFC proposals, and a pull-request template ([#27](https://github.com/AbdelStark/tessera/issues/27)).
- `docs/release/checklist.md`, the milestone release checklist: freeze, CI and release-only gates, the security gates of 06 §9 (threat-model review, simulator argument, fuzzing, external review), versions and changelogs, protocol-version rules, reproducible verifier builds and publication ([#30](https://github.com/AbdelStark/tessera/issues/30)).
- The `gpu` feature of `tessera-gpu`: `build.rs` compiles `kernels/*.cu` with nvcc into fatbins for `sm_90` and `sm_100`, and fails with a message naming CUDA 12.8 when the toolkit is missing or older; host bindings through `cudarc` 0.19 (driver API, CUDA 12.8 interface, loaded at run time); `Device` and `DeviceBuffer` wrappers; a vector-add template kernel with a test against the host reference. `TESSERA_GPU_SKIP_KERNELS=1` builds the host code without CUDA so that CI lints it ([#29](https://github.com/AbdelStark/tessera/issues/29)).
- The error taxonomy of 04 §3 in `tessera-codec::error`: `ErrorCode` with all 25 stable `ERR-...` codes, their classes and the exit codes of the `tessera` command line; `TesseraError { code, context, source }`; and `PublicContext`, whose fields are identifier types only (`SchemaId`, `ByteOffset`, `StageId`, `LayerIndex`, `TensorId`, `TableId`, `GadgetId`, `RequestId`, `TokenPosition`). The golden file `spec/vectors/errors/codes.json` carries the codes for the independent verifier; a test keeps the specification table, the golden file and the enum equal ([#31](https://github.com/AbdelStark/tessera/issues/31)).
- The Goldilocks field and the field traits of RFC-0002 in `tessera-field`: `Field`, `PrimeField`, `batch_inverse` (Montgomery's trick, reporting the first zero and leaving the input unchanged), `Goldilocks` with branch-free `const fn` arithmetic (special-form reduction, a 72-operation inversion chain, roots of unity computed at compile time from `7^((P − 1)/2^32)`), and the test field `F65537` for soundness calibration. `spec/vectors/field/` holds constants and 10,049 operation vectors produced by an independent Python big-integer oracle (`generate.py`), which the verifier will check too ([#33](https://github.com/AbdelStark/tessera/issues/33)).
- The quadratic extension of RFC-0002 in `tessera-field`: the `QuadraticExtension` trait (`new`, `coeffs`, `mul_base`, `conjugate`, `norm`) on a generic `QuadraticExt<B>`, with `GoldilocksExt2 = Fp[X]/(X^2 − 7)` for challenges and `F65537Ext2 = F65537[X]/(X^2 − 3)` for calibration; Karatsuba multiplication; inversion through the norm. `Goldilocks::mul_small` multiplies by a constant below 2^32 with a shorter reduction and serves the multiplication by 7. `spec/vectors/field/goldilocks_ext2.json` adds 4,145 extension cases from the Python oracle ([#39](https://github.com/AbdelStark/tessera/issues/39)).
- `cargo xtask check-isolation` has a seventh check: in the verifier's resolved dependency graph, every edge into the test-support crate `tv-testkit`, whose vector reader panics on malformed input, must be a dev edge ([#40](https://github.com/AbdelStark/tessera/issues/40)).
- `cargo xtask check-isolation` closes ways for code to reach a verifier crate unseen ([#208](https://github.com/AbdelStark/tessera/issues/208)):
  - Both dependency graphs are resolved with every feature enabled, so an optional dependency behind a non-default feature is checked.
  - Verifier sources may not use `include!`, `#[path]` or `#[debugger_visualizer]`. Data includes stay inside their own crate, and macro bodies cannot hide file references.
  - Every crate root must be a `.rs` file of its own package.
  - Every path package must be a workspace member.
  - Names under `verifier/` may not contain control characters.
  - A new eighth check reads rustc's dep-info strictly after `cargo check` in the dev profile (every feature) and the release profile (every feature but the test-only ones). It requires every file a verifier crate compiles to lie inside that crate.
- Hashing in `tessera-hash` (RFC-0004) ([#34](https://github.com/AbdelStark/tessera/issues/34)):
  - `H(domain, bytes)` over BLAKE3 key derivation for the 17 domains, with each domain's context key derived once.
  - Binary Merkle trees with canonical multiproofs: `MerkleTree::open`, and `verify_multiproof`, which rejects a missing, extra or reordered sibling.
  - Chained column digests (`ColumnChains`) with a configurable row block and a short final block.
  - `spec/vectors/hash/` holds domain, Merkle and column vectors from an independent Python BLAKE3, itself checked against the official BLAKE3 vectors vendored in `spec/vectors/blake3/`.
- The Tessera Canonical Encoding (RFC-0003) ([#41](https://github.com/AbdelStark/tessera/issues/41)):
  - `Encode` and `Decode` in `tessera-codec` for every primitive and composite type: integers, `bool`, arrays, `Vec`, `Option`, pairs, `Bytes`, `Ascii`, `Rational` (kept in lowest terms) and `Map` (sorted by key encoding).
  - `#[derive(Encode, Decode)]` for schema structs and enums, from the new `tessera-codec-derive`.
  - `Fp` and `Fp2` encodings in `tessera-field`.
  - Decoding reads through a `Reader` that enforces `DecodeLimits`, in the order RFC-0003 'Limits' now fixes. It checks each declared length against the bytes that remain before reading or reserving anything (INV-TCE-3), and caps reservations by the input. It nests at most 256 levels, rejects trailing bytes (INV-TCE-4), and reports `ERR-CODEC-TRUNCATED`, `ERR-CODEC-NONCANONICAL` or `ERR-CODEC-LIMIT` with the offset of the value at fault. Its debug form never shows the input.
  - `cargo xtask gen-vectors` writes `spec/vectors/codec/`: golden encodings with language-neutral values, a corpus of inputs to reject (each for one reason), and inputs exactly at the limits to accept. A test fails while the files are stale.
  - RFC-0003 'Limits' now specifies the order of checks, what counts as a nesting level, and that sequence elements encode to at least one byte.
- Redaction by type (05 §3) ([#32](https://github.com/AbdelStark/tessera/issues/32)):
  - `Secret<T>` in `tessera-codec` holds secret-class values with no `Debug`, `Display`, `Clone` or TCE `Encode`, and zeroizes them on drop (`zeroize` 1.x).
  - `Public<T>` marks values cleared for telemetry and encodes as the value it wraps.
  - `clippy.toml` denies `expose_secret` everywhere, and the modules that need it opt back in with `#[allow(clippy::disallowed_methods)]`. `cargo xtask lint` accepts that opt-in, and any use of the accessors by name under any `cfg`, only in the files `secret-access.toml` lists. It also rejects every other way found to switch the rule off: wider allowances, `warn(warnings)`, manifest lint tables and a second `clippy.toml`. Prover sources may no longer pull in code through `include!` or `#[path]`, and crate targets must be `.rs` files inside their package, so the scan sees every compiled source.
  - CI's test job installs clippy, which a test of the rule runs.
  - Compile-fail tests pin that a secret cannot be printed, cloned or encoded.
- Artifact envelope and schema registry (RFC-0003 'Envelope', 'Limits'; 03 §5) ([#43](https://github.com/AbdelStark/tessera/issues/43)):
  - `tessera_codec::envelope` seals a TCE payload behind the `TSRA` header with its BLAKE3 digest (`tessera/v1/envelope`).
    - `write` streams an envelope of any size in constant memory, and `verify` checks one read from a file the same way.
    - `open_payload` and `open` check one in memory, and decode only after the digest verifies.
    - Streaming buffers are zeroized, since payloads may be secret-class data.
  - RFC-0003 'Envelope' now fixes the order of a reader's checks: field by field, then the length, the digest and the payload. Every reader returns the same code at the same offset. `spec/vectors/codec/envelope.json` pins the order with 52 cases:
    - one fault per check;
    - every header truncation;
    - pairs of faults that show which check comes first;
    - the RFC-0003 default limits at and one past each bound.
  - `tessera_codec::schema` holds the 21 schemas of 03 §5 with their ids, versions, extensions and classes. A test reads the table from the specification. Readers accept versions `v` and `v − 1`, never 0, which 03 §7 now says too.
  - `DecodeLimits::MESSAGE` (8 GiB, 2^34 elements, depth 32) and `DecodeLimits::INTEGER_MODEL` (1 TiB) are the RFC-0003 defaults. A counting allocator shows that hostile lengths are refused without allocating.
  - A declared length that fits `max_vec_len` but not a 32-bit `usize` is now `ERR-CODEC-TRUNCATED`, as on 64-bit targets.
- The offset of every decoding error is now normative. RFC-0003 'Limits' fixes it, the first byte the failing check reads, together with the order of checks inside a value. `spec/vectors/codec/noncanonical.json` gives each case its offset, which `tessera-codec` and `tessera-field` assert; both already reported these offsets ([#215](https://github.com/AbdelStark/tessera/issues/215)).
- The Fiat–Shamir transcript in `tessera-hash` (RFC-0004 'Transcript'). `Transcript::new` binds a `SessionContext` (protocol, `Mode`, `ParamSetId`, statement digest, session id). `absorb` streams a value's TCE encoding into the hash under a length-prefixed label, collecting a long message's small writes into 64 KiB updates (about three times the throughput on a vector of field elements). `squeeze` hashes the counter and ratchets the state. `challenge_fp`, `challenge_fp2` and `challenge_indices` are exactly uniform, by rejecting words at or above `P` and by masking to a power of two. `grind` finds the smallest nonce with the required leading zero bits, rewriting only the nonce between attempts, and `check_grind` accepts any valid nonce, absorbing only a valid one. The hash sits behind `TranscriptHash`, which tests use to force every rejection branch; `spec/vectors/transcript/` holds vectors from an independent Python oracle, rejections included. RFC-0014 now defines `ParamSetId` as the parameter set's versioned name, and RFC-0004 adds the `statement` domain for the statement digest a session binds ([#45](https://github.com/AbdelStark/tessera/issues/45)).
- Multilinear tables, `eq` tables and round polynomials in `tessera-poly`, following the conventions of the new 'Multilinear extensions' section of RFC-0010 ([#46](https://github.com/AbdelStark/tessera/issues/46)):
  - Tables are slices of `2^n` entries, `x_1` the lowest bit of the index. `bind_lowest` binds an `Fp` table into an `Fp2` table with two base multiplications per pair; `bind_lowest_in_place` binds in one pass without allocating. `evaluate` and `evaluate_base` sum each block of `2^10` entries against one small `eq` table, which measures 1.2 times (`Fp2` tables) and 2.7 times (`Fp` tables) faster than binding one variable at a time.
  - `eq_table` builds the `2^n` weights by doubling, one multiplication per entry; `eq(a, b)` takes two points.
  - `less_than` (`LT(r, a)`), `less_than_points` (`LT2(x, y)`) and `causal_selector` (`C(q, i) = LT2(i, q) + eq(i, q)`) evaluate in `O(n)`, each rustdoc deriving the closed form from its defining sum.
  - `RoundPoly` holds a round message, `p(0), p(2), …, p(d)` for `1 ≤ d ≤ 7`. It evaluates at a challenge from the claim (`p(1) = claim − p(0)`) by Lagrange interpolation, with weights computed at compile time from the modulus and no division.
  - The trait `PolyField` names each field's prime subfield, so the code runs over Goldilocks, `F65537` and their extensions, with challenges in a prime field too for soundness calibration. Every input-dependent failure is a `PolyError`.
  - `spec/vectors/poly/poly.json` (338 KB) comes from an independent Python oracle, `generate.py`, which computes every value from its definition. It holds 22 table evaluations up to `2^12` entries, with bindings up to `2^8`; 11 `eq` tables; `LT` at 14 points of up to 80 coordinates; `LT2` and the causal selector, enumerated over up to `2^20` pairs; and 28 round polynomials.
  - On one Apple M4 Max core (`benches/throughput.rs`), at `2^20` and `2^24` entries: binding takes 0.78–0.87 ns per entry from `Fp` into `Fp2` and 2.1–2.3 ns in place; `eq` tables take 3.1–3.3 ns per entry; evaluation takes 1.0–1.1 ns per entry for an `Fp` table and 3.4–3.7 ns for an `Fp2` table.
- Reed–Solomon encoding with vanishing masks in `tessera-pcs`, following the conventions RFC-0011 'Encoding' now fixes: values in natural order with `ω_m = 7^((P − 1)/m)`, codeword symbol `j` at `7·ω_n^j`, and mask coefficients drawn in the order `m_0, m_1, …` ([#47](https://github.com/AbdelStark/tessera/issues/47)):
  - `NttPlan` precomputes the twiddles of one size, from `2^0` to `2^32` points, and runs `ntt`, `intt`, `coset_ntt` and `coset_intt` in place, in natural order on both sides, without allocating. Two radix-2 kernels share one bit-reversed twiddle table, a prefix of the table of every larger size: Cooley–Tukey from natural to bit-reversed order and Gentleman–Sande back, each with one twiddle per group. The coset shift multiplies each group's twiddle, so no pass scales by its powers, and the bit-reversal permutation moves 16 × 16 tiles.
  - `RowEncoder::new(k, P)` refuses `P > k` and precomputes `2.5·k` table entries. `encode_row` writes the codeword of `g = f + (X^k − 1)·m` into a buffer of `2k` symbols the caller provides: an inverse NTT of size `k` into bit-reversed order, the mask placed there, then a coset NTT of size `2k` back to natural order, with no permutation. `encode_rows` encodes a block of rows with the same tables. Mask coefficients come from a `MaskSource`, which any `FnMut() -> Goldilocks` is, `P` draws per row once every check has passed. Every input-dependent failure is a `PcsError` that carries lengths only, and nothing is drawn or written on error.
  - Tests: the degree test of RFC-0011 (the interpolant through `k + P` random positions predicts every other symbol, and one corrupted symbol fails it), the mask vanishing on `H` and filling `g`'s coefficients `k` to `k + P − 1`, the decoding of noiseless codewords, the transforms against the DFT as a sum up to `2^10` points, round trips up to `2^20` and at `2^23`, and agreement with the verifier's independent NTT in `tv-field`, now a dev-dependency (DEP-2), at every size from `2^1` to `2^18`.
  - `spec/vectors/encoding/encoding.json` (81 KB) comes from an independent Python oracle, `generate.py`, which computes every value from its definition by Horner's rule: 8 transforms of `2^1` to `2^8` points on the subgroup and on `7·L`, and 25 rows of `2^0` to `2^6` values with `P` in `{0, 1, k/2, k}`.
  - On one Apple M4 Max core (`benches/throughput.rs`): an NTT of `2^23` points takes 103 ms (12.3 ns per point), its inverse 101 ms and the coset NTT 117 ms; `encode_row` at `k = 2^22` and `P = 200` takes 135 ms, 32 ns per row value.
- `cargo xtask gen-vectors` also runs every set's independent Python oracle, `spec/vectors/<set>/generate.py`, with `python3 -I -B`, and `--check` reruns them on a copy of `spec/vectors/` and reports each file they write differently, add or remove, and each oracle that fails, leaving the tree untouched. A test runs the check, so `cargo test --workspace` fails when an oracle and its files disagree, as it already did for the codec vectors. The six oracles (`encoding`, `field`, `hash`, `poly`, `sumcheck`, `transcript`) reproduce their files byte for byte in about 5 s ([#226](https://github.com/AbdelStark/tessera/issues/226)).
- The batched sumcheck prover of RFC-0010 'Batching within a stage' and 'Sumcheck messages', in `tessera-sumcheck` ([#53](https://github.com/AbdelStark/tessera/issues/53)):
  - `prove_batch` proves the sums of products of up to 7 multilinear tables over many instances in one round sequence. It squeezes `γ` under `<prefix>/gamma` and weights instance `i` by `γ^i`. It front-pads smaller instances, so that their own variables take the last `n_i` challenges. Round `j` absorbs `p(0), p(2), …, p(d)` as TCE `Vec<Fp2>` under `<prefix>/round/{j}` before squeezing `r_j` under `<prefix>/r/{j}`, and the final evaluations go under `<prefix>/evals`. `Prefix` is `stage/{k}` or `tree/{ℓ}`.
  - An `Instance` is a product of borrowed `Table`s over the prime subfield (`Table::base`) or the extension (`Table::ext`), mixed freely. `Table::public` marks a factor the verifier evaluates itself, such as `eq(r, ·)`, which the proof leaves out. `ProvenBatch` holds the TCE-encodable `SumcheckProof { rounds, final_evals }`, `γ`, the point, and each instance's point and evaluations.
  - The prover never reads a claim. A padding round adds `2^{n_max − n_i − j − 1}·S_i`, where `S_i` comes from the instance's tables in the pass that also computes its first own round; padded tables are never materialized. Tables over the prime subfield multiply there until their first binding, which allocates half of each table once; later bindings run in place.
  - The prover is generic over `PolyField` and over a `SumcheckTranscript` trait, which `tessera_hash::Transcript` implements; tests also run it over `F65537` and its extension. Every input-dependent failure is a `SumcheckError` carrying shapes only, returned before the transcript moves, and debug forms show shapes only.
  - The verifier's `tv_core::sumcheck::verify_batch`, a dev-dependency (DEP-2), checks every proof the tests produce and must draw the same challenges and end in the same transcript state. The cases:
    - products of 1 to 4 tables on 0 to 24 variables;
    - the batch of 3, 7 and 12 variables;
    - instances on no variable;
    - degree 7;
    - public factors;
    - 300 seeded random batches.

    Changing any round value or final evaluation is rejected, and so is a prover that omits the padding scale.
  - `spec/vectors/sumcheck/sumcheck.json` now also records the tables of its 31 stages on at most 5 variables (248 KB). The prover reproduces them message by message, and every product-tree level from the trees' leaves.
  - On one Apple M4 Max core (`benches/throughput.rs`), per entry of all tables:
    - products of 1, 2, 3 and 4 tables over `Fp` take about 4.2, 6.5, 8.3 and 11.5 ns at `2^20` and `2^22` entries per table;
    - over `Fp2`, they take 5.8, 10.1, 12.7 and 17.7 ns at `2^20`;
    - binding one table through every round, alone, takes 3.2 ns per entry.
- The product-tree levels of the leaf stage, over forests of stacked trees (RFC-0010 'Tree levels'), in `tessera-sumcheck` ([#58](https://github.com/AbdelStark/tessera/issues/58)):
  - A `Forest` borrows `2^{s+h}` leaves over the prime subfield: `2^s` trees of height `h`, stacked with the tree index in the high variables. `prove_forests` proves a batch of forests from the roots to the leaves, and forests join where their heights match. Each level runs three steps:
    1. Where forests of more than one tree join, it squeezes `s_max` root challenges under `tree/{ℓ}/root/{c}`, and each joining forest takes the last `s` as its root point.
    2. `prove_batch` runs under `tree/{ℓ}`, of degree 3, over the forests that have joined, in the order given. Each instance is `eq(r_i, ·)`, a public factor, times the two halves of its next level.
    3. It squeezes `ρ` under `tree/{ℓ}/line`, and each next claim is `u_i + ρ·(w_i − u_i)` at `ρ ‖ r'_i`.
  - `ProvenForests` is `#[must_use]`. Each of its levels is a `ProvenLevel`: root challenges, `γ`, `SumcheckProof`, point, `ρ`, instances and next claims. Each forest's leaf claim is a `Claim` on its stacked leaves, and `into_parts` hands the proofs and the leaf claims to the next stage. The prover takes no claim.
  - `SumcheckError` gains `NoForests`, `LeafCount` and `Height`, checked in that order, forest by forest, before the transcript moves. Debug forms show shapes only.
  - Each forest's levels are built once, bottom-up, in the prime subfield. Each level is stored as its two halves, written in one pass, and dropped once its sumcheck has run; each `eq` table is dropped after its level.
  - The verifier's `tv_core::tree::verify_product_level`, a dev-dependency (DEP-2), checks every proof the tests produce. A driver in the tests squeezes the root challenges itself and forms each root claim `Σ_y v_y·eq(r, y)` from the trees' products. The verifier must draw the prover's challenges, and every leaf claim must equal the stacked leaves' multilinear extension. The cases:
    - every forest of 2^0 to 2^10 trees of heights 1 to 12 with at most 2^13 leaves, and 2^10 trees of 2^12 leaves;
    - forests joining at one level, and at several;
    - zero and random roots;
    - 40 seeded random batches;
    - batches over `F65537Ext2` and `F65537`, checked level by level from the RFC;
    - the Python oracle's new `forest_batches` in `spec/vectors/sumcheck/sumcheck.json`, reproduced value by value.

    A row with a non-zero product claimed zero is rejected at the level where its forest joins, and so is a root challenge other than the transcript's. Every changed round value or evaluation is rejected too.
  - On one Apple M4 Max core (`benches/throughput.rs`), 2^22 leaves take 134 ms, 32 ns per leaf, whether as 2^10 trees of 2^12 leaves, as one tree, or as forests of three heights. One degree-3 sumcheck in a tree level's shape (one `Fp2` table, two over `Fp`) over 2^22 entries takes 115 ms. The `eq` tables take 10% of the forests' time.
- The `MatMulWeight` and `MatMulAct` relations of RFC-0010 'Relations', stacked over layers or heads, in `tessera-sumcheck` ([#57](https://github.com/AbdelStark/tessera/issues/57)):
  - `MatMulWeight` proves `Y = X·Wᵀ` for a batch of tokens: `X` holds one row per token and `W` is stored `(o, i)`. `MatMulAct` proves `Z = A·B`, with `B` stored as it is or transposed as `K` in `QKᵀ` (`RightLayout::Direct`, `RightLayout::Transposed`). Operands are borrowed row-major over the prime subfield, with the stacking index highest. Shapes are logs of two (`MatMulWeightShape`, `MatMulActShape`). A count of layers or heads that is not a power of two is padded with zero ones.
  - A relation is proven in three steps around `prove_batch`:
    1. `tables` builds the instance from the input claim's point, never its value. It contracts each operand once with `eq` at the point's coordinates (`W` at `r_o`, `X` at `r_t`), and adds `eq(ρ, ·)` on the stacking variables as a public factor. An operand with nothing to contract is used as it is.
    2. `prove_batch` proves the instance among the stage's others: degree 3 when stacked, 2 for one layer.
    3. `claims` maps the instance's point and evaluations to one claim per operand: `W` at `r_i ‖ r_o ‖ ρ_ℓ` and `X` at `r_i ‖ r_t ‖ ρ_ℓ`; `A` at `r_j ‖ r_a ‖ ρ_b`, and `B` at `r_c ‖ r_j ‖ ρ_b` or `K` at `r_j ‖ r_c ‖ ρ_b`.
  - RFC-0010 'Relations' now gives `MatMulWeight` its token index and states the stacked identities, of degree 3 in the stacking variables. It also says where the claim on `K` lands.
  - `Claim { point, value }`, renamed from `LevelClaim`, is the crate's one claim type, for forests and matmuls alike.
  - `SumcheckError` gains `OperandTooLarge`, `OperandLength`, `PointLength` and `EvaluationCount`, each returned before anything is computed. Debug forms show shapes only.
  - Contractions keep 4 independent sums in registers: panels of 8 rows, 4 columns at a time, for the middle variables, and 4 rows at a time for the low ones. Several loop orders were measured on one Apple M4 Max core. The middle kernel is the fastest on rows of `2^14` entries or more and within 20% of the fastest elsewhere; the low kernel is within 6% of the fastest everywhere.
  - Every proof the tests produce is checked by the verifier's `tv_core::sumcheck::verify_batch`, a dev-dependency (DEP-2). The relations are written in the tests from the RFC, and the claims are computed in `tv-field` from the true products. The cases:
    - the issue's `2^6×2^8` by `2^8×2^5` as a `MatMulAct` with `B` either way, and as a `MatMulWeight` of `2^6` tokens by `2^5` outputs, each alone and stacked;
    - instances on 1 to 24 variables;
    - 2 to 9 layers and 2 to 7 products, zero-padded;
    - single layers;
    - transposed `B`, which proves with the same messages as `B` stored as it is;
    - stages mixing every kind and size, front-padded, and 40 seeded random stages;
    - stages over `F65537Ext2` and `F65537`.

    Every claim equals its operand's multilinear extension at the point the verifier derives. These are all rejected:
    - a product with one entry changed, at its stage's final check;
    - a prover with one operand entry changed;
    - every changed round value or evaluation;
    - a claim routed to the other operand or to another point, at the next stage's final check.
  - On one Apple M4 Max core (`benches/matmul.rs`):
    - A contraction takes 1.14 to 1.29 ns per entry it reads, and the sumcheck 38 to 44 ns per instance entry.
    - A whole relation takes 1.23 to 1.31 ns per operand entry once its operands hold `2^9` times as many entries as its instance. Time is linear in the operands: `2^3` layers of `2^10` tokens by `2^10` inputs and outputs take 20.9 ms, and doubling the inputs or the layers doubles it, to 44.0 and 42.7 ms.
- The reference tensor commitment and opening of RFC-0011, non-streaming, in `tessera-pcs`, for v0.1 measurements and tests ([#54](https://github.com/AbdelStark/tessera/issues/54)):
  - `ReferenceCommitment::commit` takes `CommitmentParams` (`log2 k`, `P_A`, `B`, `commitment_id`), a `RowLayout` whose segments list their non-zero rows in increasing order, the non-zero rows' values in layout order and a `MaskSource`. It draws each row's `P_A` mask coefficients in layout order and encodes the row, chains each segment's columns with `ColumnChains`, merges each column's segment digests into `leaf_j = H("column-merge", d_j^(1) ‖ … ‖ d_j^(S))` for every `S ≥ 1`, and builds the Merkle tree over the `n` leaves. It keeps the rows, masks, codewords, leaves and tree resident.
  - `open` takes the claims grouped by column sub-point, each claim its row point and `γ` but not its value (`ClaimGroup`, `RowClaim`), and `ξ*`. It returns the `G + 1` combined rows, the groups' in group order and the proximity row last (`CombinedRow`). `RowLayout::row_coefficients` gives `c_g(row) = Σ γ·eq(row_point, row)` and `ξ_row = eq(ξ*, row)` at every non-zero row. Consecutive rows share their high bits, so a claim costs about two `Fp2` products per row, and no table of `2^row_vars` entries is built.
  - `answer` takes the queries in draw order, possibly repeated. It returns the columns of their sorted, distinct set, in increasing order, each with one symbol per non-zero row in layout order, and the canonical multiproof of that set (`ColumnAnswer`).
  - The opening pass, `(G + 1)·N·(k + P_A)` products of an `Fp2` coefficient by an `Fp` value, keeps its sums unreduced, as 128 bits and a count of carries, and reduces once per output. It works on tiles of 64 columns, four rows at a time, and allocates nothing in its loop. On one Apple M4 Max core (`benches/commitment.rs`), with 17 combined rows and `P_A = 200`:
    - a product takes 0.48 to 0.60 ns;
    - committing takes 35 ns per committed element at `k = 2^10` and 44 ns at `2^14`;
    - opening takes 8.4 to 10.2 ns per opened element.
  - `PcsError` gains `NoMask`, `RowVars`, `NoSegments`, `RowOrder`, `RowRange`, `MatrixLength`, `ColumnPoint`, `RowPoint`, `XiPoint`, `NoQueries` and `QueryRange`, each returned before anything is drawn or computed. The commitment's `Debug` form shows its shape, and those of combined rows and answers their lengths. Rows, masks and codewords are held in plain vectors and freed without being cleared, as in the encoder, until #229 decides how secret tables are held and cleared.
  - Every opening the tests produce is checked by the verifier's `tv_core::opening`, `check_evaluations` then `check_columns`, a dev-dependency with `tv-hash` (DEP-2), on claims whose values are computed in `tv-field` from the matrix. The cases:
    - 1 to 40 claims in up to 16 groups, a group with no claim and no group at all;
    - one to four segments, empty ones included, with zero rows anywhere, and a matrix with no non-zero row;
    - `B` of 1, 2, 5 and 32, `P_A` from 1 to `k`, `k` from `2^0` to `2^10`;
    - draws that repeat columns and are not increasing.

    Combined rows and row coefficients also equal their definitions, computed in `tv-field`, and every leaf and the root equal `tv-hash`'s column digests, merge and Merkle nodes over all `n` columns.
  - These are all rejected with the reason of RFC-0011's check order:
    - every entry of every combined row and mask, changed by 1 and by `X`;
    - every opened column symbol and every sibling;
    - the other answers to a repeated, unordered draw.
  - A cheating prover commits to one row's codeword corrupted at `⌈δ·n⌉` columns, `δ` in {0.05, 0.25, 0.5}, and sends the honest combined rows. On 10^4 seeded draws of `τ` ∈ {4, 8} queries it is rejected exactly when a draw hits a corrupted column, with `Proximity` at the smallest one, at a rate above `1 − (1 − δ')^τ` less four standard deviations.
  - The prover reproduces every honest opening of `spec/vectors/pcs/opening.json` from its witness, value by value: the root, the leaves, the columns, the siblings, the row coefficients, the combined rows and their values at the queries. Every claim's value equals the matrix's extension at its point.
- The LogUp-GKR prover of RFC-0008 in `tessera-logup`, on sums of products in the sumcheck engine and one level driver for fraction trees and product forests in `tessera-sumcheck` ([#60](https://github.com/AbdelStark/tessera/issues/60)):
  - `prove_logup` runs the LogUp part of the leaf stage (RFC-0014 'Transcript schedule', rows 4 and 5). It squeezes `α` and `β` under `logup/alpha` and `logup/beta`, builds each instance's lookup tree `(1, α − a_i)` and table tree `(−m_j, α − t_j)`, both padded with row 0, and absorbs the roots `n_a, d_a, n_t, d_t` under `logup/roots`. It proves the levels beside the stage's product forests, `λ` after `γ` where a fraction tree is present, and absorbs `valũe(ρ_a)` of each instance with values under `logup/values`. The `#[must_use]` `ProvenLogUp` holds the claims `kẽy(ρ_a) = α − q − β·v`, `valũe(ρ_a) = v` and `m̃(ρ_t) = −p` and the forests' leaf claims.
  - A `LogUpInstance` takes the table's rows and the lookups as `Tuples` of keys and optional values, and the multiplicities, which are committed before `α`. `multiplicities` counts an honest witness's, `m_0` including the lookup tree's padding, and refuses a lookup of no row. A schedule of `TreeId`s orders the trees beside the forests. `LogUpError` refuses malformed shapes before the transcript moves.
  - `Instance::sum_of_products` extends the sumcheck engine to sums of weighted `Term`s over one table list. Factors every term shares multiply once per pair, terms of equal weight share one sum, and a table no term reads is evaluated at the end instead of being bound. A fraction tree's level costs 5 `Fp2` products per pair and node, 3 on the lookup tree's last level. Product instances keep their messages bit for bit.
  - `FractionTree` builds a tree bottom-up, never storing numerators of 1 and keeping the table tree's in the prime subfield. `prove_trees` proves fraction trees and forests in one schedule, each joining where its height matches; `prove_forests` is its case of forests alone, and `check_forests` checks forests' shapes for a caller that moves the transcript first.
  - The prover reproduces all 10 honest `logup_batches` of `spec/vectors/sumcheck/sumcheck.json` value by value, every transcript state included. `tv-core` checks every proof the tests produce, and every claim is compared with its polynomial's extension: dense tables of `2^8` to `2^16` rows with `2^4` to `2^20` lookups, sizes `2^k − 1`, `2^k` and `2^k + 1` on both sides, one lookup into one row, and several instances in any schedule beside forests.
  - `check_roots` rejects with `LookupError::Unbalanced` a lookup outside the table, a wrong value, a multiplicity off by one, and the phantom row `(0, 0)` balanced on a padding row. Every changed root, round value or evaluation is rejected.
  - Calibration over `F65537`, single random-`α` trials:
    - one lookup outside 255 rows, with multiplicities that give the difference 255 chosen roots, accepts 225 of 60,000 trials. Its 99% interval [3.16e-3, 4.45e-3] contains the predicted 3.89e-3, and the bound is 3.91e-3;
    - random bad instances accept 64 of 4·10^6 trials: 1.60e-5, about `1/|F|` and 30 times below the bound.
  - One Apple M4 Max core, one instance into `2^16` rows with values (`benches/throughput.rs`): 185, 113, 93 and 88 ns per lookup at `2^16`, `2^18`, `2^20` and `2^22` lookups. At `2^22`, the levels take 76 ns per lookup, building the lookup tree 11 and the values 1.1.
- The executable cost model of RFC-0020 in `tessera-bench`, and the `tessera` binary with its first command, `tessera bench model` ([#35](https://github.com/AbdelStark/tessera/issues/35)):
  - `mu` evaluates 08 §1 term by term (`MuBreakdown`): the linear layers' commitments and lookups, the attention scores' commitments, lookups and product-tree leaves, the weight pass shared by a batch, the witness regenerations of RISK-9, and the streamed leaf passes of RISK-10, whose count `streaming_passes` gives exactly. `UnitCosts` gains `streaming_pass_ps`, the cost of one streamed pass per score, which RFC-0020's formula needs. `proof_time_s`, `sampling_rate` and `detection_probability` turn `μ` into the columns of the source's §8.
  - `UnitCosts::SOURCE`, `Counts::SOURCE_405B` and `Scenario::SOURCE` are the source's inputs. `ModelReport` projects them at any contexts and writes the `tessera.cost-model/1` report, added to 03 §5. Every input is checked, finite and in range, and every output must be finite: an input the model cannot price is a `ModelError`, never a NaN.
  - `tessera bench model --source-defaults [--context ...] [--gpus G] [--batch-tokens T] [--eta η] [--target-overhead O] [--violations V] [--out MU.json] [--json]` prints the projections as a table, or the report with `--json`, and writes it with `--out` through a temporary file. Usage errors exit 64 (`ERR-USAGE`), an unwritable report 20 (`ERR-RES-IO`) and an internal error 70, as 02 §2.4 says. `--plan` follows the compiler's commitment report.
  - Golden test against the source document, read line by line:
    - The model is within ±0.05 of the source's chart at all eight contexts (worst: linear 0.036, total 0.041) and within ±0.0005 of its weight pass.
    - The calculators, from §8's own `μ`, give its sampling and detection columns within 0.1 percentage point and its proof times within half a unit of their last printed digit.
    - `spec/vectors/cost-model/` holds every figure with its line and every input with its provenance, written by an independent Python oracle that evaluates the same model to 1e-12. One input is reconstructed: `F_lin`, which the source rounds to 8.1×10^11, is two operations per parameter of Llama-3.1-405B, the reading that reproduces the chart.
  - `μ` rises with every unit cost and count of proof work and with the native rate, falls with native work, and each input moves exactly the terms it enters. The streaming term is zero while a batch's leaves fit in memory and grows by one pass per doubling of them beyond it.
- CPU microbenchmarks of the prover's primitives, measured as RFC-0020 measures unit costs, with a baseline for the CI runner class ([#65](https://github.com/AbdelStark/tessera/issues/65)):
  - `cargo bench -p tessera-bench --bench primitives` times `Fp` and `Fp2` products, batch inversion, a `2^20`-point NTT, BLAKE3 column chains, Merkle multiproofs, a degree-3 sumcheck and the LogUp stage at fixed sizes, single-threaded. Each takes 3 warm-up runs and 10 measured runs, and reports median, extremes and coefficient of variation in nanoseconds per unit of work, flagged as noisy above 3%.
  - `tessera_bench::measure`, `Statistics` and `MicrobenchmarkReport` are the harness, which the H100 unit costs will share. `--runner-class NAME --out FILE` writes a `tessera.microbenchmarks/1` baseline, added to 03 §5: schema-versioned, its names sanitized and every number finite.
  - The `bench` workflow runs the benchmark on the GitHub-hosted `ubuntu-24.04` runner when the suite changes, or on demand, and uploads the report. `docs/perf/baselines/github-ubuntu-24.04.json` is its committed baseline, and `docs/perf/README.md` says how to run and read it. The 5% regression gate of 08 §6 runs on a runner with locked clocks later.
  - 08 §6, 07 and RFC-0008 name this harness where they named `criterion`. Its statistic is the one the gate compares, the median of 10 runs, and it writes a schema-versioned baseline and adds no dependency.
- JSON Schemas for the reports and their validation in CI, with an example of the cost model's report ([#49](https://github.com/AbdelStark/tessera/issues/49), in part):
  - `spec/schemas/` holds `microbenchmarks-v1`, `cost-model-v1` and `unit-costs-v1`, named after the reports' `schema` identifiers. The first two describe what `tessera-bench` writes today, and the third is the H100 report of RFC-0020, which has no producer yet.
  - `cargo xtask lint` validates every committed report under `docs/perf/` against its schema, failing on a missing identifier or a schema without a file. Its validator implements a small subset of JSON Schema 2020-12 with no new dependency, and refuses a schema that uses any other keyword, so that no constraint is ignored. Self-tests violate each constraint alone.
  - `docs/perf/examples/cost-model-source-defaults.json` is `tessera bench model --source-defaults --json`'s report. A test keeps it current, comparing numbers to `1e-12` relative, since `ln_1p` and `exp_m1` round differently across platforms.
- The cost model prices the pass that answers the column queries, P5 of RFC-0018 ([#248](https://github.com/AbdelStark/tessera/issues/248)):
  - `UnitCosts::column_answer_ps` and `MuBreakdown::column_answers` price P5's evaluation of every committed element at the queried columns, `C` per token in the linear layers and one per score, with no glue. The source prices none, so the golden test is unchanged.
  - The `tessera.cost-model/1` report, its schema and its example carry both fields, `tessera.unit-costs/1` names a `column_answer` unit, and the table has an `answers` column. 09 §3 now says when a report version freezes: once a release writes it, so these schemas change in place.
  - With five regenerations (08 §1), the corrected projections are `μ(1K) ≈ 9.4` and `μ(16K) ≈ 24.9`, before the column answers, which add up to 1.6 and 6.5 at the commitment's 30 ps per element (08 §3, SC-2).
  - Priced at the commitment's cost, with glue and the other unit costs at zero, the term equals the linear and attention terms: P5 evaluates exactly the elements the witness pass commits. A test holds the projections that 08 §3, SC-2 and RISK-9 quote to the model's, digit for digit.
- The `tessera-quant` Python package's skeleton, under `python/` (RFC-0005 'Python packaging', [#67](https://github.com/AbdelStark/tessera/issues/67)):
  - `pyproject.toml` with RFC-0005's dependencies, locked in `uv.lock` for the platforms of 09 §8, Linux x86-64 and macOS aarch64. PyTorch comes from its own index through the `cpu` or `cu130` extra. Both pin the same release, 2.14.1, and a test fails if they diverge, so CPU and GPU calibration cannot derive different scales through a library upgrade.
  - The package `tessera_quant`, typed (`py.typed`) and versioned by its distribution. `ruff` lints and formats, `pyright` checks types in strict mode, and `pytest` with `hypothesis` runs the tests.
  - The `python` CI job runs on Python 3.11, 3.12 and 3.13, and is now a gate over the three, so branch protection still names one check. Its former `pyright --strict` was not a pyright option: strict mode now comes from `pyproject.toml`.
  - `cargo xtask lint` fails when the workspace `Cargo.toml` and `python/pyproject.toml` state different versions (09 §1), a dynamic version, or one outside `MAJOR.MINOR.PATCH`. 09 §1 maps no pre-release between SemVer and PEP 440, so that last case is refused rather than guessed.
- The arithmetic table kinds in `tessera-tables` (RFC-0008 'Table specifications', [#69](https://github.com/AbdelStark/tessera/issues/69)):
  - `TableSpec`, `TableKind`, `TableId` and `BitOp`. RFC-0008 now fixes their tier-A TCE layout. Tags are declaration indices, so the transcendental kinds are appended later without moving a tag or a pinned digest. 03 §3's `TableSpec` follows the RFC.
  - `GeneratedTable` builds `Range`, `Requant`, `Pow2`, `Bitwise8`, `AesSbox` and `Gf256Mul` from their parameters, refusing those outside RFC-0008's rules. It also regenerates a claim list's specification, refusing a wrong `log2_rows` or digest.
  - `Table` gives the rows, padded with copies of row 0, the row a lookup hits, and its value. Lookups take witness keys, so they compute arithmetically, with no memory access that depends on the key (INV-ARCH-2): the S-box by inversion in GF(2^8), a class's shift by selects.
  - A digest hashes the definition, an empty sequence of piece parameters, and the value column of the kinds the verifier enumerates (RFC-0008 'Generators and digests').
  - `spec/vectors/tables/` holds 20 cases from an independent Python oracle. Before writing, the oracle checks its S-box and products against FIPS-197: the §4.2 examples, Figure 7, and the AES-128 known answers of Appendices B and C.1. It also checks every one of the 65,536 products against logarithms.
- The compiler's dataflow IR in `tessera-ir` (RFC-0006 'Operation set', [#72](https://github.com/AbdelStark/tessera/issues/72)):
  - `Op` holds the 21 operations, each with its `OpClass` (linear, degree 2 or gadget-backed). `TensorSpec`, `NodeKind`, `IntType`, `IntRange`, the dense ids and the references a node names follow 03 §3.
  - `GraphBuilder` adds inputs, weights, public constants, shift-class sets and nodes in topological order. It checks every node against its operation's rules (arity, shapes, the kind of each output and of the inputs that need one), and leaves the graph as it was when it refuses one. Ids are dense in creation order, so two builds from the same calls are equal (INV-DM-5).
  - `Graph::validate` checks the same rules on a graph assembled otherwise: dense ids, one producer per tensor, topological order (so no cycle), and graph inputs that are public or committed. `Graph::producer` walks the virtual DAG that the schedule reads. Errors are `IrError`s.
  - RFC-0006 now states the rules in a table and refines the sketch. Nodes have several outputs, as `SoftmaxRow` gives the exponentials and the reciprocals. `MatMulAct` has a `transpose_rhs` flag, since `QK^T` and `PV` contract different axes and the operation set has no transpose. `Slice` names its axis and range. The graph declares the constants, shift classes and weights its nodes name.
- Checkpoint import in `tessera-quant` (RFC-0005 'Pipeline', step 1; 06 §3, TB-5; [#70](https://github.com/AbdelStark/tessera/issues/70)):
  - `import_checkpoint(path, expected=None)` reads a Llama-3.1 checkpoint from safetensors only and returns a `FloatModel` whose BF16 tensors load on demand. `ArchSpec` and `RopeSpec` (03 §2) come from `config.json` exactly, decimals as rationals.
  - Every shard's header is checked before any tensor is read: bounded size, BF16, shapes, byte ranges tiling the data to the end of the file, and exactly the architecture's tensors where the index places them, in shards of the checkpoint's own directory.
  - A departure raises `ArchitectureError` (`ERR-QUANT-ARCH`), a `TesseraQuantError` (04 §4). A directory holding only pickle files is refused without opening them, and a test scans the package's sources for any pickle-based loading.
  - Tests import a tiny checkpoint, single-file and sharded, bit for bit. They also import a sparse, header-only checkpoint with the published Llama-3.1-8B configuration and shapes, and check each refusal for its reason. A malicious pickle that would create a file if loaded leaves no file.
- The `Exp` and `Exp2Frac` tables in `tessera-tables`, the first of the fixed-point generators of #71 (RFC-0008 'Fixed-point generators'):
  - RFC-0008 now defines the generators to the last integer. `LN2` is pinned as `⌊ln 2 · 2^64⌋`. `exp_neg` reduces by it and runs Horner's rule over 20 Taylor terms in Q64. `exp_pos` works in Q62 and only rounds down. `Exp`'s argument `k · scale` is reduced exactly, with no 256-bit arithmetic. Every intermediate stays below `2^128`, and no floating point is used.
  - `TableKind::Exp { key_bits, sigma, scale }` and `TableKind::Exp2Frac { frac_bits, out_bits }` are appended, as tags 6 and 7. `Exp`'s digest hashes its zero tail's first key as a piece, and the values before it as its core. `Exp2Frac`'s core is all of its values.
  - `Exp2Frac` never exceeds the true `⌊2^out · 2^{f/2^frac}⌋`, as RFC-0017's scoring bound requires. The oracle checks every key, and `out_bits ≤ 56` keeps the rounding inside below one unit: at 62 the reference check found values two units low.
  - `GeneratedTable` holds the generated core of the fixed-point kinds, and `Exp` and `Exp2Frac` lookups read it.
  - The tables oracle checks every `Exp` key of its core, and every `Exp2Frac` key, against 60-digit `decimal` references. That includes the production parameters: 2^22 keys at `σ = 16`, with cores of 36,133 and 144,542 keys. `spec/vectors/tables/constants.json` pins `LN2`.
  - `cargo xtask lint` refuses floating-point types and literals in `tessera-tables` and `tv-tables`. It lexes the sources, so comments and strings do not count.
- The `Silu` table in `tessera-tables`, the second of the fixed-point generators of #71 (RFC-0008 'Fixed-point generators'):
  - RFC-0008's `Silu` row asked for `silu(h · σ_h)` but named no `σ_h`. The kind is now `Silu { b, r_by_class, in_exp, out_exp }`, appended as tag 8. The value is `clamp(round(silu(h · 2^{in_exp}) / 2^{out_exp}), −128, 127)` with `h = k >> r_c`, looked up as `pack(c, k)` like `Requant`.
  - The generator computes `σ(x)` in Q63 from `exp_neg(|x|)`, multiplies by `h`, and rounds half up by a shift of `63 − in_exp + out_exp`. Every intermediate stays below `2^128`.
  - A row's value depends only on its class and `h`, so the digest's core is every `h` of every class, at most `2^24` values in all. That bound becomes a rule, which a 405B-like gate (`b = 30`, shifts 14 to 17) meets with 122,880 values.
  - The tables oracle checks every value against `x / (1 + e^{−x})` at 60 digits, including the 405B-like gate. Rust reproduces every digest.
- RoPE's constants in `tessera-tables`, the last of the fixed-point generators of #71 (RFC-0008 'Fixed-point generators'):
  - `TableKind::Rope { theta, factor, low_freq_factor, high_freq_factor, original_max, bits, max_context, d_head }` is appended as tag 9, with integer parameters, as Llama-3.1's are. Each pair is `(C, S) = (round(2^bits · cos(p·ω_i)), round(2^bits · sin(p·ω_i)))`.
  - RFC-0008 pins `TAU = ⌊2π · 2^64⌋` and `HALF_PI = ⌊π/2 · 2^64⌋` and defines the rest exactly:
    - `ln` by the atanh series;
    - the frequencies `θ^{−2i/d}` by `exp_neg`;
    - Llama-3's frequency bands, decided by exact integer comparisons, and the interpolation weight in Q40;
    - `cos` and `sin` by Horner's rule in Q62 after reduction by `TAU` and quadrants.

    Every intermediate stays below `2^128`.
  - `RopeTable` holds a specification and its digest. The core is every pair, at most `2^24`; Llama-3.1-405B has 8,388,608. `RopeConstants` gives any pair on demand, without the digest, for the witness engine's `Rotate`. `GeneratedTable` refuses a `Rope` kind, and `RopeTable` refuses any other.
  - The tables oracle checks every pair of two test tables, and 4,008 sampled pairs of Llama-3.1's 131,072-position table, against 40-digit references. It also pins the digest of the 8K-context table. A Rust test checks 10^6 more sampled pairs against an f64 implementation of Hugging Face's Llama-3 RoPE, all within one unit.
- `cargo xtask lint` rejects a `|` inside a code span of a Markdown table row: GitHub splits the row's cells there, so the row renders shifted. Of the 18 such rows in six documents, 17 now write it `\|`, and RFC-0005's Scores bound drops its pipes ([#268](https://github.com/AbdelStark/tessera/issues/268)).
- The calibration set of `tessera-quant` (RFC-0005 'Calibration'), the first part of #73:
  - `tessera_quant.calibration.draw` draws 512 sequences of 2,048 tokens and a held-out split of 64 from a seed, the documents and a tokenizer. The draw is exact, through SplitMix64, rejection-sampled integers and a Fisher–Yates order, so the same inputs give the same sequences on any machine and with any library version. RFC-0005 now states it.
  - A BLAKE3 selection digest identifies the sequences, and a content hash identifies each document. `check_disjoint` refuses an evaluation set that shares a document with `ERR-QUANT-CALIB`.
  - `fetch_c4_validation` reads C4's `en` validation split at the pinned revision `1588ec45`. `blake3` joins the package's dependencies.
- Calibration statistics in `tessera-quant` (RFC-0005 'Calibration', 'Determinism'), the second part of #73:
  - `tessera_quant.capture.calibrate` runs each calibration sequence alone through Hugging Face's Llama in BF16, with deterministic algorithms. It records every quantization point's per-channel absolute maxima in float64. The model is built from the checked checkpoint's safetensors files, never through `from_pretrained`, which would unpickle a `.bin` file. Its output equals Hugging Face's own loader's bit for bit.
  - The statistics go to a secret artifact, a safetensors file written with mode 0600 through an atomic rename. A `tessera.calibration/1` report, with its schema in `spec/schemas/` and an example under `docs/perf/examples/`, names them by digest and shape only.
  - A non-finite or all-zero point raises `ERR-QUANT-CALIB`.
- `IntegerModel`'s quantization constants and provenance, field by field (03 §2, [#273](https://github.com/AbdelStark/tessera/issues/273)):
  - `QuantSpec` is the claim list's `PublicQuantSpec`, field for field, beside the hidden per-channel classes and affine multipliers that RFC-0012 commits with the weights.
  - `QuantPoint` names every quantized or rescaled tensor: the embedding, each layer's rescales in RFC-0005's order (`Site`), and the final norm. RFC-0007's lists are ordered by it. `shift_class_sets` gives each site's class shifts, its `Requant` table's `r_by_class`.
  - `RopeSpec` takes the `Rope` table's integer parameters. `Provenance` and `CalibrationIdentity` record the checkpoint's shard digests, the calibration set's identity, the statistics digest, `α` and library versions.
  - The Rust types live in `tessera-ir` with TCE encodings. Debug forms show shapes, never a weight or a channel's class. `spec/vectors/model/integer-model.json`, written by `cargo xtask gen-vectors`, pins a sealed model and its value for the Python writer to check.
  - The claim list's missing public data for affine-mode rescales is [#274](https://github.com/AbdelStark/tessera/issues/274).
- The Tessera Canonical Encoding in `tessera-quant` (RFC-0003, [#74](https://github.com/AbdelStark/tessera/issues/74)):
  - `tessera_quant.tce` reads and writes every TCE type: integers, `bool`, `Fp`, `Fp2`, `Ascii`, `Rational`, arrays, `Vec`, `Option`, pairs, maps, structs and enums. Integer sequences are NumPy arrays: decoding returns read-only views of the input, and encoding writes an array's own memory and casts nothing. For every input the reader reports the code and offset RFC-0003 fixes. It passes the 51 golden, 43 non-canonical, 11 boundary and 52 envelope vectors of `spec/vectors/codec/`, as the Rust readers do.
  - `write_artifact` streams an envelope to a secret file, mode 0600 through an atomic rename. A sizing pass fills the header; the payload is then written and hashed on the way. `read_artifact` maps the file and checks the digest over it before decoding anything. Writing a memory-mapped 1 GiB weight and reading it back takes about 11 and 15 KiB of allocations.
  - The writer refuses a value its readers would refuse under the same limits, before the file exists, with the code and offset they would report. It never converts a value to fit: an array of another dtype, a bool for an integer or a float for a `Rational` is `ERR-INTERNAL`.
  - `tessera_quant.integer_model` defines `IntegerModel` field by field from 03 §2. `TokenFile` and `LogitsFile` live in `tce`. No `repr` shows a weight, a channel parameter, a token or a logit.
  - `spec/vectors/model/integer-model.json` now holds two one-layer models: servable with power-of-two scales, and minimal with affine multipliers. Between them they hold every weight type, rescale site and quantization point, and a map of seven entries, so each enum tag and each canonical order is pinned. Python decodes both to their values and writes them back byte for byte.
- The structured evaluation of lookup tables in `tessera-tables` (RFC-0008 'Structured evaluation', [#75](https://github.com/AbdelStark/tessera/issues/75)):
  - `Table::eval_mle(ρ, β)` gives `t̃(ρ) = kẽy(ρ) + β·valũe(ρ)` over the table tree's padded rows: the value the LogUp check compares with the table tree's claim.
  - `Range` and `Requant` cost `O(n)` per class at any size. Before its clamp, a `Requant` class's value is affine in the bits of its offset key, and such a function's sum over a prefix of the cube has a closed form. A `2^32`-row table of four classes evaluates in 5.5 µs.
  - `Silu` and `Exp` cost their dense core. The runs of equal values at either end of a core, such as `Silu`'s saturation, are differences of `LT`s, and the rest is cut into aligned blocks evaluated with `eq` tables. Nothing divides, so a point with a coordinate 0 or 1 is evaluated like any other. The largest cores, `Silu`'s `2^24` values and `Exp`'s 12.5 million, take 5.2 and 18 ms.
  - A test compares it with the dense sum over every padded row: 30 tables of every kind with `b ≤ 16`, 100 points each, coordinates 0 and 1 included. `cargo bench -p tessera-tables` times every kind at production sizes.
  - RFC-0008 now states two facts it had wrong. `Requant`'s rounding reads bit `r − 1`, so its value does not depend on `x >> r` alone. `Range`, at up to `2^32` rows, is a closed form rather than an enumeration. Its dense core is cut into aligned blocks rather than walked along a Gray code, which divides.
- The table generators in Python, `tessera_quant.tables`: the third implementation of RFC-0008 'Generators and digests', for the integer reference ([#76](https://github.com/AbdelStark/tessera/issues/76)):
  - It generates the kinds `D` uses: `Requant`, `Silu`, `Exp` and RoPE's constants. Each `TableSpec` encodes, digest included, to the same bytes as the independent oracle's vectors in `spec/vectors/tables/`, all 20 of these kinds. Every listed value matches, so do the lookups at the edges of each key range, RoPE's frequencies, and 4,008 pairs of Llama-3.1's 131,072-position table.
  - NumPy has no 128-bit integer, so the vectorized generators compute on pairs of `uint64` words exactly (`tessera_quant._wide`). Division estimates its quotient in floating point and corrects it with exact remainders. Property tests compare it with Python's integers, as they do the vectorized `exp_neg` and `cos_sin` with their definitions, including RoPE's fifth quadrant: `TAU` is `4·HALF_PI + 2`.
  - A `2^24`-key `Exp` core generates in 3.4 s, against the 30 s the issue allows. The RoPE digest of Llama-3.1's 8K table, 524,288 pairs, takes 0.2 s, and the core of a 405B-like `Silu` gate, 122,880 values in Python's integers, 0.3 s.
- Range analysis in `tessera-compiler` (RFC-0006 'Range analysis', [#77](https://github.com/AbdelStark/tessera/issues/77)):
  - `analyze(graph, tables)` bounds every tensor from public data alone: the inputs' declared ranges, the constants, the tables' specifications, and never a weight value. It uses exact `i128` intervals, and an interval that would leave `i128` is `Unbounded`.
  - `MatMulWeight` takes the public bound `127 · fan_in · max|x|`. `Rotate` takes `2^{bits+1} · max|x|`, gadget outputs the ranges their lookups enforce, and the attention output `Mul(MatMulAct(e, v), R)` takes `SoftmaxRow`'s relational fact `R_q · s_q ≤ 2^{2σ}`: RFC-0005's `[−2^39, 127 · 2^32]` rather than a bound about 64 times wider at 32 positions.
  - Every gadget input that must be an integer is checked against its domain. Without a bound it is `ERR-COMPILE-WRAP`; outside its domain it is `ERR-COMPILE-RANGE`. RFC-0006 now lists the domains, the gadget output ranges and the fact's pattern, which the verifier must re-derive identically.
  - An exact interpreter of the IR checks soundness on 11,000 runs of a servable attention and MLP layer: random inputs, and adversarial ones with activations at their bounds and weights of one sign. It also checks every operation alone under random and corner inputs.
- The Llama-3.1 lowering in `tessera-compiler` (RFC-0006 'Llama lowering', [#78](https://github.com/AbdelStark/tessera/issues/78)):
  - `lower(PublicModel, n)` builds `D`'s graph over one sequence of `n` tokens from the public part of the model alone: the architecture, the profile, `PublicQuantSpec` and the `Silu` input exponents. Weights and channel classes enter as committed inputs, so the verifier can re-derive the same graph from the claim list.
  - It returns the graph, its tables and the hooks later stages attach to: the tokens for the anchors, the one-hot tokens and logits for scoring, the positions, and each weight's channel-class vector. Table 0 is RoPE's; each layer's `Exp` and `Silu` tables follow, shared when equal, with key widths from public bounds.
  - Both profiles: `minimal` rescales the source's eight tensors per layer, `servable` three more. Attention runs head-major through `GatherPublic`, so grouped-query attention needs no copy of the committed keys and values. One causal `SoftmaxRow` covers every head, and the attention output takes the softmax fact's pattern.
  - `lower` runs range analysis and `check_profile`, which checks S1 to S3 of RFC-0005 (`ERR-COMPILE-PROFILE`). Every tensor then carries its analyzed range and the narrowest integer type holding it.
  - `validate` checks a whole `IntegerModel`: its public part, then one weight per type and layer with its architecture's shape and entries in `[−127, 127]`, and a `ChannelParams` for each weight a site of several classes rescales. Each refusal names its rule (`ERR-QUANT-ARCH`). Affine rescales wait for [#274](https://github.com/AbdelStark/tessera/issues/274).
  - Seeded tiny models are lowered in both profiles, with grouped-query attention and sites of one to four classes. An exact interpreter of the IR runs each graph to the logits of an independent, token-by-token implementation of RFC-0005's semantics. Every value stays inside its tensor's range, on random and adversarial inputs.
  - Llama-3.1 8B's and 405B's shapes lower from their public part alone. Each validation rule is tested broken alone, and each of 36 bugs planted in the changes fails a test.

- Gain folding, QK equalization and SmoothQuant migration in `tessera_quant.migration` (RFC-0005 'Pipeline', steps 2 and 4, [#79](https://github.com/AbdelStark/tessera/issues/79)):
  - `fold` multiplies each RMSNorm gain into the weights that read its output.
  - `equalize` scales `W_q` and `W_k` rows by `λ` and `1/λ` per KV head and RoPE pair, shared by the group's query heads.
  - `migrate` moves activation outliers into the weights where `D` can absorb the factor: the O projection's input through `W_v`, the down projection's through `W_up`.
  - Each returns the statistics the transformed model computes. Each acts within one layer, so a quantizer can stream a model.
  - `select_alpha` picks the strength of least divergence among `{1/2, 3/5, 7/10, 4/5}`, the smallest of equal ones; `mean_kl` measures the divergence.
  - In float64 on 64 sequences, each transform keeps a tiny Llama's logits within `10^{-12}` relative. An independent recorder finds the maxima each one predicts. 18 planted bugs each fail a test.
  - RFC-0005 now says where each migration factor goes. The norm-fed inputs have no place for one in `D` ([#289](https://github.com/AbdelStark/tessera/issues/289)).

- Gadget selection and table sizing in `tessera-compiler` (RFC-0006 'Gadget selection', [#80](https://github.com/AbdelStark/tessera/issues/80)):
  - `select(graph, ranges, tables, policy)` gives every gadget-backed node its gadget, numbered in node order.
  - A requantization takes `Requant1` when its key and class index fit `B_max` bits, and `RequantSplit` past that.
  - The split needs a low part of at most `B_max` bits, a key of at most 62 bits, and `s ≤ r − 1` for every class's shift; otherwise `ERR-COMPILE-RANGE`. A `TableMap`, `Bitwise` or `SoftmaxRow` table must hold at most `2^{B_max}` rows.
  - Tables are shared by definition and numbered after the lowering's. `GadgetPolicy` (RFC-0007) joins the IR crate, and RFC-0006 now counts the class index's bits and states the split's three conditions.
  - Keys at 32, 33, 62 and 63 bits take `Requant1`, `RequantSplit`, `RequantSplit` and a refusal. A property test checks that the split tables compute `rq(z, r)` across the widths.
  - Llama-3.1 8B's 32 layers read exactly the tables its first two do. `minimal` 8B is refused at its down projection's residual, whose key is too wide for its shift ([#292](https://github.com/AbdelStark/tessera/issues/292)).

- W8A8 quantization in `tessera_quant.quantize` (RFC-0005 'Scale policy', step 5, [#81](https://github.com/AbdelStark/tessera/issues/81)):
  - `quantize(arch, profile, weights, maxima)` turns a migrated model and its statistics into `D`'s INT8 weights and `QuantSpec`, in either profile.
  - Weights take power-of-two scales per output channel: the least that keeps a row inside ±127, rounded to nearest even, so no weight clips. A tensor holds at most four consecutive exponents. RoPE pairs share theirs. The embedding, the LM head and `minimal`'s up projection take one per tensor, since nothing after them applies a per-channel factor.
  - Each site's shift classes follow from the exponents by RFC-0005's semantics, each in `[1, 63]` or `ERR-QUANT-CALIB`, and the hidden `ChannelParams` map channels to them.
  - The residual's scale covers both the calibration's stream and every embedding row, so a token the calibration never saw still enters INT20. `κ` follows RFC-0023. The score scale is the multiple of `2^{−40}` nearest `1/√d_head`, times `2^{e_q + e_k}`. SiLU's input exponent is the finest its table allows.
  - `affine=[...]` gives the named Q, K, V, O, up or down weights per-channel scales and 16-bit multipliers under one shift. The compiler refuses them until [#274](https://github.com/AbdelStark/tessera/issues/274).
  - On a tiny Llama-like model, with weights and activations rounded to their scales in float64, the logits' error is 2% of the largest, KL divergence `2.3 · 10^{−6}` and top-1 agreement 99.2% in both profiles. Every site's shifts equal RFC-0005's formulas, written out independently in the tests. Of 25 planted bugs, 24 fail a test; the 25th is equivalent at the head sizes tested.
  - RFC-0005's scale policy now states the rules its semantics force: the per-tensor weights, the RoPE pairs, the shifts' range, the score scale's rational, SiLU's input exponent and the affine shift.

- `RequantWide` in `tessera-compiler`'s gadget selection, with RFC-0024, accepted (RFC-0006 'Gadget selection', [#292](https://github.com/AbdelStark/tessera/issues/292)):
  - It takes a requantization key too wide for its shift, which `RequantSplit` cannot take. The key splits below every class's rounding bit, keeps a middle part as wide as the output's range needs, and reduces its top part to the saturation `clamp(h1, −2, 1)`. That costs four commitments and five lookups per element, into existing table kinds. RFC-0009 Lemma 11 proves one witness per key.
  - `minimal` now compiles at 8B and 405B. Its down projection's residual key, of 55 and 59 bits against a shift near 17, takes `RequantWide`, with one class or four.
  - The top part's `Range` check is necessary. Without it, a half-integer top part reaches the saturation table's odd keys and passes with another output. An exhaustive search over `F65537` found such outputs in every configuration tried without the check, and none with it.
  - RISK-2's upper bound for `minimal` rises to 252,928 commitments and 269,312 lookups per layer per token at 405B, and its corrected `μ(1K)` from 9.4 to 10.0. 00, 08 and `SPEC.md` restate the projections, which the bench test pins.
  - The tests check the boundary with `RequantSplit` and each precondition at its boundary. A property test checks that the five lookups compute `rq(z, r_c, lo, hi)` for every class, range ends included, with `lo` and `hi` drawn independently. Each of 16 bugs planted in the selection fails a test.

- Tiny conformance models in `spec/vectors/models/` (07 §5, [#83](https://github.com/AbdelStark/tessera/issues/83)):
  - Three Llama-3.1-architecture models with random INT8 weights, each an `IntegerModel` artifact pinned by its digest. tiny-2 (2 layers, `d_model` 64, 4 heads, 2 KV heads, vocabulary 512) comes in both profiles. tiny-4 (4 layers, `d_model` 256) comes in `servable`, with a 16-bit residual.
  - Each model has two token sequences, with the residual stream `D` gives at every layer boundary and its logits.
  - `generate.py` is their independent oracle: `D`'s forward pass in Python's integers, written from RFC-0005 as RFC-0022 and RFC-0023 amend it, with RFC-0008's tables from the tables oracle. Its encoder must first write the pinned `IntegerModel` vectors byte for byte.
  - Each site's shift is the least that keeps its outputs in range. A few sites are pushed past it, so the sequences reach every clamp: an INT8 requantization, the Silu table, the residual at INT20 and at INT16, and a zero exponential. The oracle fails unless each push clamps.
  - The compiler's lowering, run on its exact interpreter, gives every sequence's logits bit for bit, as does its token-by-token reference. That differential test catches 6 of 8 bugs planted in the oracle, and the oracle's own checks refuse the other 2.
  - The prover (`tessera-e2e`), the verifier (`tv-codec`) and `tessera-quant` each load every artifact. `spec/vectors/README.md` maps every vector set and who writes it.

- The CPU reference integer engine in `tessera-witness` (RFC-0018 'Witness engine', [#82](https://github.com/AbdelStark/tessera/issues/82)):
  - `Engine::new(&lowering).run(&model, &tokens)` computes every tensor of `D` on a sequence, by the semantics of RFC-0006's operations, in `i64`. `Engine::for_graph(...).evaluate(inputs)` runs any graph from its inputs' values, and checks that each input is given once, whole and inside its declared range.
  - Debug builds check every value against its tensor's analyzed range (INV-DM-2). The matrix products split rows among threads with `std::thread::scope`, and each value is one thread's exact sum, so the thread count changes nothing.
  - The prompt arrives in a `Secret`, which `secret-access.toml` now lets the engine read. Every tensor lives in a `SecretBuf`, and the logits leave as a `LogitsFile`, the secret-class artifact `0x0602`, which `tessera-ir` now defines with a redacted debug form. `unsafe-debug-dump` adds a dump of every tensor.
  - `tessera-compiler`'s `Lowering` names its checkpoints: the residual after the embedding and after each layer.
  - On the conformance models, the engine reproduces the oracle's logits and every layer-boundary residual bit for bit. 10^3 adversarial runs, with weights at ±127 and tokens at the vocabulary's ends, stay inside every range. One to eight threads give identical tensors.
  - The operations no lowering emits yet (`Sub`, `Slice`, `Sum`, `ConstAdd`, `Bitwise`) give RFC-0006's values on a graph built for them. A zero residual row takes its root from ε alone. Each malformed request, model and input is refused with its code.
  - Each of 23 bugs planted in the engine fails a test, and a no-op control survives.

- The commitment report in `tessera-compiler` (RFC-0006 'Outputs', [#84](https://github.com/AbdelStark/tessera/issues/84)):
  - `report(model, &lowering, &selection, &policy)` counts what a lowered model commits and looks up, by the definition of each selected gadget in RFC-0009 and RFC-0024. Committed elements fall in four classes (lookup outputs, split limbs, row scalars, one-hot entries), counted with the lookups per layer per token and per token. Scores are counted apart, with their product-tree leaves.
  - It also gives each site's requantized elements, each table's rows and lookups, the multiplicities and RFC-0010's stage count, which `stage_count` computes for any graph: 3 in `servable` and 4 in `minimal`, at any depth. `Lowering` now lists each site's output.
  - `CommitmentReport::to_json` writes `tessera.commitment-report/1` (`spec/schemas/commitment-report-v1.schema.json`). It carries the projections of `μ` that `tessera-bench`'s cost model gives for the report's counts, with the unit costs, memory, scenario and regenerations they assume.
  - `docs/perf/examples/commitment-report-405b.json` reports Llama-3.1-405B in `servable`, every site with one shift class. It requantizes RISK-2's 260,096 elements per layer per token, and commits 327,464 elements with 327,612 lookups. That is RISK-2's count with one exponent per tensor at O and down, plus 808 row scalars. At the source's unit costs and five regenerations, `μ(1K)` is 11.0, against 08's 11.5 with the quantizer's per-channel classes at O and down. [#88](https://github.com/AbdelStark/tessera/issues/88) reports those. A test keeps the file current, and `cargo xtask lint` checks it against the schema.
  - The report states what it does not count exactly. `RsqrtRow` costs RFC-0009's estimate of 20 commitments and 30 lookups per row until [#118](https://github.com/AbdelStark/tessera/issues/118) fixes its limbs. The multiplicities leave out the range tables of `SoftmaxRow`'s and `RsqrtRow`'s checks, which the selection does not allocate yet ([#118](https://github.com/AbdelStark/tessera/issues/118), [#122](https://github.com/AbdelStark/tessera/issues/122)). The opening groups `G` and the masks join the report with the stage schedule ([#113](https://github.com/AbdelStark/tessera/issues/113), [#149](https://github.com/AbdelStark/tessera/issues/149), [#154](https://github.com/AbdelStark/tessera/issues/154)).
  - RISK-2's `servable` count of 359,424 per layer per token at 405B holds with two shift classes at the residual sites. With three or four, the down projection's update key, 31 bits and two of classes, splits: 375,808. 00 §9 now says so.
  - Tests check the counts against figures derived by hand from RFC-0005, RFC-0009 and RFC-0024:
    - the source's 137,216 elements for `minimal` at 405B, and RISK-2's `servable` count;
    - each class at 8B in both profiles, the two-step residual update, and `RequantWide` with and without low bits;
    - the embedding's one-hot vector, the final norm, and Llama-3.1-8B's published parameter count without its folded norm gains;
    - each sumcheck relation and claim map on a chain built by hand.
  - RFC-0010's longest `servable` chains are two relations, the attention output's and RoPE's, not one through the committed `o8`; RFC-0010 now says so.
  - Each of 38 bugs planted in the report and in the lowering's record of sites fails a test. A 39th is equivalent, since the first checkpoint's producer is no gadget, and a no-op control survives.

- Affine rescale sites in the claim list, with RFC-0025, accepted ([#274](https://github.com/AbdelStark/tessera/issues/274)):
  - `PublicQuantSpec.affine_sites` names each affine site with its one shift. Every rescale site is in it or in `shift_class_sets`, never both. An affine site follows the Q, K, V, O, up or down projection.
  - From that public data, the verifier can lower an affine site. Its key is `x · M_c`, a `Mul` by the weight's committed 16-bit multipliers broadcast over positions, and the multipliers' public range `[0, 2^16)` bounds it.
  - RFC-0025 also fixes the lowering, the stage count and the soundness argument:
    - An affine Q or K gives `servable` four stages.
    - Uniqueness needs no proof of the multipliers' range, just as it needs none of the weights'.
    - It amends RFC-0005, RFC-0006, RFC-0007, RFC-0010 and 03 §2.
  - The weight before an affine site carries its multipliers, from 1 to `2^16 − 1`, and no classes. `tessera_quant.quantize` writes affine sites this way. It refuses `affine` at `minimal`'s O and down projections, whose keys would exceed 62 bits.
  - The compiler checks RFC-0025's rules on both parts of the model, and refuses to lower an affine site until [#301](https://github.com/AbdelStark/tessera/issues/301) implements the lowering.
  - `PublicQuantSpec`'s layout changes in place, as 09 §3 allows before a release, so each `IntegerModel` grows by the list's 8-byte length.
    - The `IntegerModel` vectors now hold two affine sites, and the models' oracle encodes them byte for byte.
    - The conformance models are regenerated.
  - The tests refuse each rule's breach alone, in the public and the hidden part. Each of 16 bugs planted in the checks and the quantizer fails a test, and a no-op control survives. A 17th survived in an affine branch that nothing reached yet, so the branch was removed.

- Llama-3.1-405B's commitment reports in `docs/perf/v0.2/` ([#88](https://github.com/AbdelStark/tessera/issues/88)):
  - Both profiles, with one, two or four shift classes at every site after a weight, at `B_max` 32 and 30. Each is a `tessera.commitment-report/1` report with the cost model's projections. `tessera-bench`'s test regenerates and compares all nine, and asserts that 30 refuses the other three configurations, each at the table it would need. The single-class `servable` report moves there from `docs/perf/examples/`.
  - They settle RISK-2.
    - At `B_max = 32` the rescales commit 326,656, 359,424 and 375,808 elements per layer per token in `servable` with one, two and four classes, and 252,928 and 285,696 in `minimal`.
    - With two classes, `μ(1K)` is 11.48 and 10.53, as 08 §3 projects. 00 §9 and `SPEC.md` record the resolution.
  - They answer OQ-0006-1: `B_max` stays 32.
    - 30 refuses three configurations, and costs 0.24 to 0.62 more of `μ(1K)` in the others, against at most 0.13 of table side saved.
    - 34 lies outside RFC-0008's tables. By RFC-0006's selection rule, which the test applies, it would change one gadget: the four-class down projection's update.
  - The reports put LogUp's table side at 20 to 28 · 10^9 rows per proof: 76 to 104 GiB of multiplicity counters and 0.15 to 0.20 of `μ`. The cost model leaves that term out, which [#304](https://github.com/AbdelStark/tessera/issues/304) adds. A balanced split point would cut it two- to fourfold ([#303](https://github.com/AbdelStark/tessera/issues/303)), and 08 §4 records the counters.

- The integer reference for `D` in `tessera_quant.reference` (RFC-0005 'Integer semantics', [#89](https://github.com/AbdelStark/tessera/issues/89)):
  - `run_reference(model, tokens)` gives `D`'s logits on a sequence in PyTorch's int64 (02 §4). `trace` also returns every tensor `D` commits: the residual at each layer boundary, each site's output, each norm's root, and each layer's attention maxima, exponentials and reciprocals. Each is keyed by the site's point for the Rust differential test ([#91](https://github.com/AbdelStark/tessera/issues/91)).
  - Before each product, its bound is computed from its operands' largest magnitudes, and a product that could reach `2^63` is refused. The norm's root, whose `d · 2^{2κ}` reaches `2^{94}`, is computed in Python's integers. The tables come from `tessera_quant.tables`, generated once per definition and kept for the next sequence.
  - It refuses affine sites until their lowering ([#301](https://github.com/AbdelStark/tessera/issues/301)), tokens outside the vocabulary, and sequences past the context.
  - On every sequence of the conformance models, it gives the oracle's residual at every layer boundary and its logits, bit for bit. Tests cover `rq`'s ties, negative ties and saturation at shifts 1, 2, 7 and 40, a zero residual row's root, determinism, and each refusal.
  - Each of 16 bugs planted in the reference fails a test, and a no-op control survives. Two more resize the `Exp` and `Silu` tables without changing a value, so they are equivalent.

- The claim list in `tessera-ir` and its emission in `tessera-compiler`, with RFC-0026, accepted ([#85](https://github.com/AbdelStark/tessera/issues/85)):
  - `ClaimList` (schema `0x0201`) holds:
    - the protocol version, 0 until `tessera-protocol/1`;
    - the architecture, the profile and `PublicQuantSpec`;
    - every table, with its digest;
    - the gadget policy;
    - the commitments' layout, `AdviceLayoutParams` and `WeightLayoutParams`;
    - the parameter set.
  - `ClaimList::digest` is `H("claim-list", TCE(ClaimList))`.
  - RFC-0026 defines `WeightLayoutParams` from RFC-0012 and RFC-0014: `log2 k_W`, the rate, `P_W` and `B_W`. It takes scoring and the anchor out of the claim list until their circuits define what each states ([#152](https://github.com/AbdelStark/tessera/issues/152), [#155](https://github.com/AbdelStark/tessera/issues/155), [#137](https://github.com/AbdelStark/tessera/issues/137) to [#146](https://github.com/AbdelStark/tessera/issues/146)), so no field is a guess.
  - `tessera_compiler::claim_list(model, &selection, &policy, layout, params)` builds it from the public part of `D` and the selection's tables. The parameter set is the caller's, `tessera-protocol`'s ([#103](https://github.com/AbdelStark/tessera/issues/103)).
  - `spec/vectors/claims/` holds the conformance models' claim lists for `test-small/1`, sealed, with their digests. `cargo xtask gen-vectors` writes them after the models' oracle, and a test keeps them current.
  - Tests:
    - changing every weight and channel class leaves a claim list's bytes unchanged (INV-CL-1);
    - lowerings of one token and of the whole context give the same claim list;
    - it decodes from its encoding and from its envelope to itself.

- `D`'s assembly and export in `tessera_quant` (RFC-0005 'Pipeline', steps 4 to 6; 02 §4; [#86](https://github.com/AbdelStark/tessera/issues/86)):
  - `calibrate(model, calibration)` returns a `CalibrationResult`, the calibration set and its statistics. `tessera_quant.capture.write_calibration` writes them as before: the secret statistics file and its `tessera.calibration/1` report.
  - `quantize(model, calib, profile=..., scales=...)` chooses the migration strength among RFC-0005's `{1/2, 3/5, 7/10, 4/5}`. It keeps the one whose `D` diverges least from the BF16 reference on the held-out split, the smallest of equal ones. `D` runs through the integer reference, and its logits are taken at their real scale. It then assembles `D` with its `Provenance`:
    - each checkpoint file's BLAKE3, sorted by name;
    - the calibration's identity;
    - the statistics digest;
    - the strength chosen;
    - the versions of Python, torch, transformers, safetensors, NumPy, datasets and BLAKE3.
  - `export(d, path)` writes the artifact as a secret file and returns its digest.
  - `affine` scales are refused until the reference computes affine sites ([#301](https://github.com/AbdelStark/tessera/issues/301)).
  - The module that chooses scales and shifts is now `tessera_quant.scales`, its function `choose`, so that the top-level `quantize` of 02 §4 does not collide with it. 02 §4 gives the interface as built.
  - The tests on a tiny Llama:
    - each candidate's divergence is computed independently, and `quantize` keeps the least divergent candidate's `D`, byte for byte;
    - every provenance field is what it names, and the profile asked for is `D`'s;
    - two runs give the same bytes, the export reads back, and no float reaches `D`;
    - each refusal is tested.
  - Each of 11 bugs planted in the assembly fails a test, and a no-op control survives.

- `D` records the scale of its logits, with RFC-0027, accepted ([#309](https://github.com/AbdelStark/tessera/issues/309)):
  - `PublicQuantSpec.logits_exponent` follows `logits_bits`: a logit `ℓ` stands for `ℓ · 2^{logits_exponent}`, the final norm's exponent plus the LM head's. The quantizer writes it.
  - Fidelity evaluation of an exported `D` needs it for perplexity and KL divergence ([#90](https://github.com/AbdelStark/tessera/issues/90)), and temperature and mixture scoring need it to build `Exp_T` ([#152](https://github.com/AbdelStark/tessera/issues/152), [#155](https://github.com/AbdelStark/tessera/issues/155)).
  - `quantize`'s choice of `α` now reads it from `D`.
  - Every encoding carries it: the Rust and Python types, the models' oracle and the vectors. The layout changes in place, as 09 §3 allows before a release, and the conformance models and claim lists are regenerated.
  - A test pins it to RFC-0005's rule on the folded LM head, in both profiles.

- Fidelity evaluation in `tessera_quant.fidelity` (RFC-0005 'Fidelity evaluation', 02 §4, [#90](https://github.com/AbdelStark/tessera/issues/90)):
  - `evaluate_fidelity(model, reference, spec)` compares `D`, run by the integer reference with its logits at their real scale, with the BF16 reference. It computes:
    - the perplexity ratio, with a 95% bootstrap interval over documents;
    - teacher-forced top-1 agreement, with a 95% Wilson interval. `D` reads each prompt and the reference's greedy continuation, and each continuation token is a trial;
    - the mean per-token `KL(ref ‖ D)`, with a 95% bootstrap interval over prompts.
  - An `EvalSpec` holds the token data with its identities and the calibration's document hashes. An evaluation document that the calibration used is refused (`ERR-QUANT-CALIB`).
  - The `tessera.fidelity/1` report (`spec/schemas/fidelity-v1.schema.json`) records the seed, the data's identities and digests, the library versions and `D`'s provenance. `docs/perf/examples/fidelity-tiny.json` is a tiny model's.
  - `tessera_quant.capture` gains `logits_of` and `continue_greedily` on a loaded reference.
  - Tests: the token metrics on a three-token vocabulary match the hand-computed values, KL in its direction. The Wilson interval of 9 out of 10 is the textbook [0.5958, 0.9821], exactly 0 and 1 at the ends. A bootstrap of one ratio has no width, and its bounds are the 2.5% and 97.5% percentiles by Python's own quantiles. A tiny Llama's report is reproducible, records what it measured, and shows D close to its reference. Each hygiene rule is tested. Each of 12 bugs planted in the module fails a test.
  - Bounded mode's `Û` joins the report with RFC-0017's predictor. The gates themselves are [#93](https://github.com/AbdelStark/tessera/issues/93).

- The CPU engine and the PyTorch reference agree bit for bit on every tensor `D` commits (SC-7, 07 §5, [#91](https://github.com/AbdelStark/tessera/issues/91)):
  - `spec/vectors/models/traces.json` holds the oracle's traces. For each sequence of the conformance models it lists every committed tensor, with the residual before the first layer and the logits, in the order of the forward pass.
  - `crates/tessera-e2e/tests/differential.rs` runs the engine on every sequence and compares each traced tensor. Every committed tensor of the lowering must have its name in the traces.
  - `python/tests/test_reference.py` holds the PyTorch reference to the same traces. Both report the first tensor and index that diverge.
  - `tessera_quant.reference.Trace` gains `updates`: the `u` a two-step residual site commits (RFC-0022).
  - The conformance models now reach the two-step update's clamp, at INT20 and at INT16. tiny-2-minimal's and tiny-4-servable's constants change, and their artifacts, `models.json` and the golden claim lists are regenerated.
  - Tests:
    - a divergence seeded at a drawn index of each tensor is reported there;
    - a channel moved to another class is reported at its site and channel;
    - two bugs planted in each implementation, a reciprocal one too small and an inner clamp at `±2^bits`, fail at the tensor and index where they arise.
  - Llama-3.1-8B is [#312](https://github.com/AbdelStark/tessera/issues/312), once [#93](https://github.com/AbdelStark/tessera/issues/93) gives its `D`.

- The `tessera-quant` command line (02 §2.1, [#92](https://github.com/AbdelStark/tessera/issues/92)), also `python -m tessera_quant`:
  - `compile` draws the calibration set from `--seed`, captures its statistics, quantizes and exports `D`. `--report` writes the calibration report. The same seed gives the same artifact byte for byte.
  - `eval` measures `D` against the checkpoint it was compiled from, checked by `D`'s provenance digests. It redraws `D`'s calibration set from its provenance and the specification's calibration source, checked by the selection digest, and draws evaluation data apart from those documents.
  - `reference` writes the integer logits of every position of every sequence of a `TokenFile` as a `LogitsFile`.
  - Specifications are `tessera.calibration-spec/1` and `tessera.eval-spec/1`, with JSON Schemas in `spec/schemas/`. `docs/perf/examples/calibration-spec-c4.json` is RFC-0005's default calibration.
  - Progress goes to stderr, and `--json` prints one result or error object. The exit status is the error code's (04 §3), checked against `spec/vectors/errors/codes.json`.
  - Supporting changes:
    - `tessera_quant.checkpoint.read_tokenizer` reads a checkpoint's `tokenizer.json` and its `bos_token_id`, through `tokenizers`, now a direct dependency (RFC-0005 'Python packaging'). `Provenance` records its version.
    - `tessera_quant.fidelity.select` draws evaluation sequences as the calibration set is drawn.
    - `tessera_quant.assemble` gains `check_options` and `checkpoint_digests`, and plain JSON-lines files are read as well as gzip-compressed ones.
  - Error codes:
    - `UsageError` (`ERR-USAGE`) is new.
    - `quantize`'s refusals of `affine` scales and unknown profiles become `ERR-USAGE`.
    - The reference's refusal of affine sites becomes `ERR-QUANT-ARCH`, and of tokens outside the vocabulary or bad lengths `ERR-USAGE`.
    - A reference of another architecture in `evaluate_fidelity` becomes `ERR-QUANT-ARCH`. All four were `ERR-INTERNAL`.
  - Tests: an end-to-end run on a tiny checkpoint with its tokenizer, and each failure's exit status. Six bugs planted in the command line each fail a test: hashes not excluded, the selection digest unchecked, the reference unchecked, the seed ignored, the logits reordered, and affine scales accepted.

- The `tessera` command tree of 02 §2.2 ([#68](https://github.com/AbdelStark/tessera/issues/68)):
  - Every command, each with its arguments, value checks and help: `compile`, `commit-weights`, `weights refresh`, `prove`, `audit sample`, `audit batch`, `audit plan`, `params check`, `bench unit-costs`, `bench model` and `recorder ref`. `tessera --help` ends with the whole list, generated from the tree.
  - The commands to come check the envelope of each TCE artifact they read, in constant memory (RFC-0003). They then exit with `ERR-USAGE`, naming the issue that implements them.
  - `--json` is a global flag. A failure prints `{"command", "error": {"code", "message"}}` on stdout, as well as its line on stderr (02 §2).
  - Exit statuses are the error codes' of 04 §3: a malformed artifact exits 65 with its fault's code, an unreadable one 20, a malformed argument 64.
  - `bench model` takes `--plan` or `--source-defaults`, exactly one, and moves to its own module.
  - Every command's outcome leaves through one exit point, where [#95](https://github.com/AbdelStark/tessera/issues/95) will write the run manifest of 05 §6.
  - Tests: through the built binary, the help lists every command of 02 §2.2. Each command to come is refused naming its issue, after its artifacts are checked. Five envelope faults exit 65 with their codes, and ten malformed arguments exit 64.

- The cost model prices the table side of LogUp (RFC-0020, 08 §1, [#304](https://github.com/AbdelStark/tessera/issues/304)):
  - A proof commits one multiplicity per table row and builds the table's fraction tree over every row (RFC-0008). `mu` gains the term `(1 + glue) · M · (c_c + c_ℓ) / T`: each of the `M` rows costs one committed element and one tree leaf, priced as a lookup, once per batch of `T` tokens.
  - `Counts` gains `table_rows`, `MuBreakdown` `tables`, and `tessera.cost-model/1` and `tessera.commitment-report/1` both, in place before a release (09 §3). The compiler's report gives its multiplicities as `table_rows`, and `tessera bench model` prints a `tables` column.
  - The source's counts have no table rows, so the golden test still reproduces the source.
  - With two classes at `B_max = 32`, 405B's tables hold 20.5 · 10^9 rows in `minimal` and 24.7 · 10^9 in `servable`. Shared by a 16K-token batch, that adds 0.15 and 0.18 to `μ(1K)`.
  - The corrected projections become about 10.7 and 26.0 at 1K and 16K for `minimal`, and 11.6 and 26.9 for `servable`. 00 §1, SC-2, RISK-2, RISK-9, 08 §3, `SPEC.md` and the v0.2 reports restate them.
  - With its smaller tables counted, `B_max = 30` costs 0.13 or 0.14 of `μ(1K)` in `servable` and 0.49 in `minimal`, not 0.24 to 0.62.
  - Tests:
    - each table row costs what a token's committed lookup costs, once per batch, and nothing without rows;
    - every input moves exactly the terms it enters, the table rows included;
    - the restated figures are the model's.

- The cost model prices the opening pass's row combination, P4 of RFC-0018 ([#252](https://github.com/AbdelStark/tessera/issues/252)):
  - **The term.** P4 folds every committed element into `u_ξ` and the combined row of each claim group on its tensor. `UnitCosts::row_combination_ps` (`c_r`) and `MuBreakdown::row_combination` price it at `(C + A·S)·c_r` over the native time, with no glue, as the column answers are priced. The source prices none, so the golden test is unchanged. The cost-model, commitment-report and unit-costs schemas, the example, the v0.2 reports and the CLI table carry the term. The unit-costs schema's restated projections also gain `tables`, which #304 left out.
  - **The blocking.** RFC-0018 'Opening pass' combines rows in blocks of the row block `B`, so each accumulator tile is read and written once per block: `32/B` bytes per product instead of 32. On tensor cores a block is one limb-decomposed INT8 GEMM, as the weight fold is. RFC-0020 measures `c_r` at v0.1.
  - **Measured.** `pcs/combine`, a new microbenchmark, times the reference combination: 0.55 ns per product on one Apple M4 Max core, about 3% of encoding and hashing an element that enters three combined rows. Each picosecond of `c_r` adds what one of `c_a` does: 0.064 to `μ(1K)` for `minimal` (08 §3).
  - **Tests.** Every input moves exactly the terms it enters, the row combination included. Priced at `c_c`, the term equals the linear and attention terms, as the column answers' does.

- Affine rescale sites compile and run end to end (RFC-0025, [#301](https://github.com/AbdelStark/tessera/issues/301)):
  - **Lowering.** The weight before each affine site adds its multipliers as a committed input, `[rows]` in `[0, 2^16 − 1]`. An INT8 site rescales `Mul(x, m)` with its one shift. A residual site takes the fused key `acc · m + 2^r · res`, or the two-step update when `r = 63`. `Lowering::multipliers` names the inputs, and the refusal of affine sites is gone.
  - **`Mul`'s broadcast** is a field of the operation: `Mul { broadcast: Trailing }` for one value per row, `Leading` for one per channel. Shapes alone cannot say which applies, since a `[n, n]` operand fits both, as a norm's root does at `n = d_model` positions. RFC-0006 and RFC-0025 say so.
  - **The CPU engine** computes the leading broadcast and binds the multipliers from `ChannelParams`.
  - **The PyTorch reference** computes affine sites. `tessera_quant.quantize`, and `tessera-quant compile --scales affine`, give every weight that can take them a multiplier per channel: Q, K and V, and in `servable` O, up and down.
  - **A conformance model.** `spec/vectors/models/tiny-2-affine` has six affine sites beside `pot` ones, computed independently by the oracle. Its traces and claim list join the vectors.
  - Tests:
    - on random tiny models of both profiles, with weights at ±127 and multipliers at 1 and `2^16 − 1`, and at a residual shift of 63, the lowered graph gives the reference's RFC-0025 semantics, every value in its range;
    - at 405B's shape, each affine key is 16 bits wider than its `pot` one, within RFC-0025's bounds, and takes `RequantSplit` or `RequantWide`;
    - `minimal`'s affine O and down are refused with `ERR-COMPILE-RANGE`;
    - an affine RoPE site gives `servable` four stages;
    - the CPU engine and the PyTorch reference reproduce every traced tensor of the affine conformance model.

- `RequantSplit` splits its key at a balanced point (RFC-0028, accepted, [#303](https://github.com/AbdelStark/tessera/issues/303)):
  - **The rule.** RFC-0006 took the least split that fits the high table, `s = b + ⌈log2 n⌉ − B_max`, so the high table always had `2^{B_max}` rows. The selection now takes `s = clamp(⌈(b + ⌈log2 n⌉)/2⌉, b + ⌈log2 n⌉ − B_max, min(r_min − 1, B_max))`: inside Lemma 2's bounds, the point that balances the two tables' rows, `2^s + 2^{b + ⌈log2 n⌉ − s}`. On a tie, the extra bit goes to the shared `Range { s }`.
  - **What it changes.** Every element still costs two commitments and two lookups. The per-token counts do not change; the tables shrink.
  - **At 405B, with `B_max = 32`.** The commitment reports' table rows fall from 20 to 28 · 10^9 to 5 to 11 · 10^9, and the multiplicity counters from 76 to 104 GiB to 19 to 42 GiB. The table term falls from 0.15 to 0.20 of `μ(1K)` to 0.04 to 0.08.
  - **Restated.** The corrected projections are about 10.6 and 25.9 at 1K and 16K for `minimal`, and 11.5 and 26.8 for `servable`, in 00, 08, `SPEC.md`, the v0.2 reports and OQ-0006-1's analysis. Keeping `B_max = 32` still holds: 34 now gains about 0.18 of `μ(1K)`.
  - **Amended.** RFC-0006 'Gadget selection', RFC-0009 'RequantSplit' and RFC-0023's example. The golden claim lists are regenerated with the conformance models' smaller split tables.
  - **Tests.** The selection tests pin the split at each boundary against the rule computed independently in the test: balanced, clamped by the rounding bit, where Lemma 2's bounds meet, and with classes. Lemma 2's property test checks the split tables at the new points, over shifts up to past the balanced point.

- `commitment_id`, where every column chain starts, is defined (RFC-0031, accepted, [#219](https://github.com/AbdelStark/tessera/issues/219)):
  - **The formula.** `commitment_id = H("commitment-id", TCE(CommitmentDescriptor))`, under a new hash domain between `envelope` and `column-init`.
  - **The advice descriptor** holds the statement digest, the proof unit, `AdviceLayoutParams` and the unit's non-zero rows. `AdviceCommitment.layout_digest` is its `commitment_id`, and the verifier checks it at step 4.
  - **A weight matrix's descriptor** holds `WeightLayoutParams`, the matrix, its message width, its row block, its rows and the epoch.
  - **The weights' row block.** Weight matrices chain their columns over the parameter set's row block, which no document had stated.
  - **Where it lands.** RFC-0004, RFC-0011, RFC-0012 and RFC-0019 are amended. `tessera-hash` gains `Domain::CommitmentId`. `spec/vectors/hash/` lists the domain and four descriptors with their encodings and identifiers, from the oracle.

- The leaf stage's sumchecks have rejection reasons of their own (RFC-0033, accepted, [#225](https://github.com/AbdelStark/tessera/issues/225)):
  - **`TreeLevel { level, round }`, `REJ-TREE`.** A failed tree level was reported as `Reject(Sumcheck { stage, round })`, but a tree level has no `StageId`. The new reason names its `ℓ` and the round within it.
  - **`LogUpIdentity { round }`, `REJ-LOGUP-IDENTITY`.** `tessera-protocol/2`'s identity sumcheck combines every LogUp instance, so its failure names no table.
  - **Amended.** 04 §2, RFC-0010 'Sumcheck messages', RFC-0019 step 7 and RFC-0032. 07 §4's tamper corpus mutates one round coefficient and one evaluation per tree level.
- LogUp batches of any size and height are tested end to end through the independent verifier ([#105](https://github.com/AbdelStark/tessera/issues/105), [#62](https://github.com/AbdelStark/tessera/issues/62)):
  - **Large batches.** Batches of 1 to 20 instances verify, their trees of every height from 4 to 24 between them: one table tree of `2^24` rows, and twenty lookup trees of heights 4 to 23 in one batch.
  - **A dropped instance.** A prover that keeps one instance out of the levels, after absorbing every instance's roots, is rejected at the first level where that instance's taller tree is present, by an `EvaluationCount` failure. The levels before that are the honest ones, and the verifier accepts them.
- Soundness calibration over `F65537` (07 §4) in `tessera-e2e`'s `soundness_calibration`: a prover that lies in one sumcheck round and one with lookups outside its table run on the real engines and transcript, against verifiers written from RFC-0010 and RFC-0008, and each is accepted exactly when its challenge is one of `K = d` or `K = N + T − 1` roots. For degrees 1, 3 and 7 and tables of 16, 255 and 1,023 rows, every 99% Clopper–Pearson interval of the accepted rate contains `K/|F|` and starts at or below the analytic bound. The new `nightly` workflow runs it, with `tessera-logup`'s measurement of random bad lookups, and `docs/testing/soundness-calibration.md` describes it ([#104](https://github.com/AbdelStark/tessera/issues/104)).
- The elementwise relations of RFC-0010 'Relations' and `Linear` claim propagation, in `tessera-sumcheck` ([#99](https://github.com/AbdelStark/tessera/issues/99)):
  - `Elementwise` is a sum of weighted products of inputs against `eq` at the target's point. An input is an operand read at each vertex, at each entry's RoPE partner (`Input::Partner`) or broadcast over some variables (`Input::Broadcast`), or a public table, which the verifier evaluates. The constructors are `mul` (elementwise, scalar-by-row and by-constant products), `rotate` (RoPE on the pairs that differ in one bit), `sum_squares`, `booleanity` and `padding` (RFC-0035's zero-check). `new` takes any terms up to `MAX_DEGREE`, such as SHA-256's majority.
  - A relation is proven in the matmuls' three steps:
    1. `tables` builds the instance from the input claim's point: `eq`, and the partner and broadcast operands laid out in `SecretBuf`s. A single term of weight 1 that multiplies every input once runs on the engine's product loops.
    2. `prove_batch` proves it among the stage's instances.
    3. `claims` gives one claim per operand input: at the instance's point, at that point with the partner's coordinate flipped, or at a broadcast's own coordinates.
  - `zero_check_point` squeezes a stage's zero-check point under `stage/{k}/zero/{c}`.
  - `Linear` turns a claim on a linear target into a `LinearClaim` on its inputs, each with its public coefficient and at the point its `Placement` gives: `Same`, `Broadcast` (the coordinates it has) or `Sum` (`(½, …, ½) ‖ r`, coefficient `2^j`). `substitute` propagates a chain, `evaluate_public` moves public terms to the value, and `combination` evaluates the claim.
  - RFC-0010 'Relations' now states how each input is read and where it leaves its claim, `Rotate`'s identity with its two claims on `X`, and the `Linear` placements.
  - `tessera-compiler`'s stage count now takes a `Sum` over an axis for the `Linear` relation that RFC-0010 and RFC-0030 make it, with no stage of its own: its claim lands at `(½, …, ½)` on the axis. No template uses `Sum`, so no report changes.
  - `SumcheckError` gains `RelationShape`, `InputLength`, `PartnerBit`, `TermInput` and `InputPlacement`, and debug forms show shapes only. `secret-access.toml` lists `elementwise.rs`, whose `Input::secret*` constructors and `Elementwise::secret_rotate` take operands in their `Secret`.
  - The tests run every proof through `tv_core::sumcheck::verify_batch`, with relations written from the RFCs:
    - every kind alone from 1 to 12 variables and at 20, and a product and a rotation at 24;
    - all kinds together, with zero-checks at the stage's point;
    - `rotate` against RoPE as RFC-0005 states it;
    - a residual chain over 8 layers, propagated to its inputs and checked by direct evaluation.

    These are all rejected at the final check: a prover with one operand or public entry changed; every changed round value or evaluation; a non-boolean entry; a value on a padded row.

- RFC-0036, accepted, defines the protocol messages and the session state machines ([#98](https://github.com/AbdelStark/tessera/issues/98)):
  - **Layouts.** Schema version 1 of `SeedCommit`, `ProverSeedShare`, `SeedReveal`, `Round1`, `Round2Response`, `ColumnQueries`, `Round3Response` and `NiProof` is `tessera-protocol/1`, field by field in TCE. It defines the types no document had: `LogUpProof`, `TreeProof`, `WeightCombinedRows`, `ScoringClaims`, `FixedBits`, `MatrixQueries`, `MatrixColumns` and the seed messages. `tessera-protocol/2`'s fields come in schema version 2, with RFC-0013 and RFC-0032.
  - **Transcript.** The bytes each label of RFC-0014's schedule absorbs, and `{T}` spelled as RFC-0031's `WeightMatrix` names.
  - **Sessions.** The prover's and the verifier's live states, each refusal with its code, in RFC-0014's order: 02 §3's calls now follow it.
  - **`ERR-PROTO-QUERIES`**, for column queries of the wrong matrices or count, or past their codeword (INV-PROTO-3), in 04 §3, `tessera-codec`'s `ErrorCode`, `spec/vectors/errors/codes.json` and `tessera-quant`'s exit codes.
  - Amended: RFC-0014 'Messages', 'Session order' and 'Transcript schedule'; 01 §5, 02 §3, 03 §4, 04 §3 and RFC-0010 OQ-0010-1. 01 §5 no longer puts the anchor's key commitment in Round 1: RFC-0015 logs it at session open.
- `tessera-compiler`'s `derive_schedule` derives the stage schedule of a lowered graph from the graph and its checkpoints alone (RFC-0037, INV-SCH-1), toward [#113](https://github.com/AbdelStark/tessera/issues/113):
  - **Layers.** `LayerMap` reads the layers off the checkpoints and checks they are one template, node for node. A graph whose layers differ is refused as `ERR-INTERNAL`, naming the rule (`CompileError::Schedule`).
  - **Claims.** The leaf stage leaves one claim per gadget input, one per layer for a site in the layers, and one on the logits. Each claim maps through `Add`, `Sub`, `ConstMul`, `Sum`, `Reshape`, `Transpose` and `Expand` onto atoms, at points whose coordinates name the challenges that set them. A claim that is neither one virtual atom of coefficient 1 nor committed atoms of one commitment at one point is split, one value per part.
  - **Stages.** Stage `k` holds one instance per virtual part of stage `k − 1`, stacked over the layers, then in stage 1 the equality zero-checks of committed linear outputs. Every committed part reaches the claim pool, the leaf stage's stacked at the layer point.
  - **Tests.**
    - INV-SCH-2 to INV-SCH-5 are checked on tiny Llama models of 2, 3, 4 and 8 layers, in both profiles, with pot, classed and affine sites: 3 stages for `servable` and 4 for `minimal` or an affine RoPE site, as the commitment report counts them, and one pool claim per weight type.
    - On an honest witness at random coordinates, every claim equals its atoms, every pool claim equals its group, and every matmul, product and sum of squares holds at its evaluations' points. This includes a hand-built template of every linear map.
    - 25 mutations of the derivation are caught.
  - RFC-0037 now states that per-layer constants belong to a layer's own template, that a ready virtual atom has coefficient 1, the order of split claims, and the leaf stage's claims on the graph.
- The independent verifier's `tv-tables` is checked against the prover's tables ([#100](https://github.com/AbdelStark/tessera/issues/100)):
  - `tessera-tables`' differential test compares both crates on every kind: specification bytes, digests, rows at the edges, `eval_mle` at seeded points, and which parameters each refuses.
  - `spec/vectors/tables/` gains `silu-b8-wide-arguments`, a `Silu` table whose arguments reach `128·ln 2`, where `exp_neg`'s `n` reaches 128 and the value is 0 by its rule. Both workspaces and `tessera-quant` check it.
- `tessera-prover`'s `StageProver` proves a stage schedule's reduction stages over a `WitnessSource`, which closes [#113](https://github.com/AbdelStark/tessera/issues/113) (RFC-0010 'Stages', RFC-0037):
  - **Stages.** Stage `k` proves one batched sumcheck of its instances over the witness's tables: `MatMulWeight`, `MatMulAct`, `Elementwise` for products, sums of squares and the equality zero-checks, and `Rotate`'s kernel. Each is stacked over the layers where its kind is, at its claim's point.
    - Stage 1 first draws the layer point that stacks the leaf stage's per-layer values, then the zero-check point.
    - After the evaluations, each stage sends the values of the claims it splits, and every committed part's value reaches the pool.
    - Every claim an instance leaves is checked against the schedule's point for it.
  - **The leaf stage** is the gadgets' (#118). Until then, `StageProver::leaf` takes its points and computes its claims' true values.
  - **Tests.**
    - Every stage of the conformance models `tiny-2-servable`, `tiny-2-minimal` and `tiny-4-servable`, and of three hand-built templates (scaled relation outputs, every linear map, per-layer channel products), verifies through a verifier written in the test from the RFCs. The pool it ends with is the prover's and the committed tensors' extensions.
    - A split that does not add up, and a changed round or evaluation, are rejected at their stage. 15 of 16 mutations of the stages are caught; the survivor removes the check of each claim against the schedule, which fires only when a relation and the schedule disagree.
  - `tiny-2-affine` has no schedule: its layer 0 has affine sites that layer 1 has not (#348).
- RFC-0038, accepted, makes every layer of a model one template, for RFC-0037's stacked layers ([#348](https://github.com/AbdelStark/tessera/issues/348)):
  - **The rule.** At each site, every layer is affine or none is, and a `pot` site has one class in every layer or several in every layer. Claim-list validation step 2 checks it, and the compiler refuses a model that breaks it.
  - **The residual's route** is chosen once for all layers: the fused key when every layer's site has one shift of at most 62, two steps otherwise.
  - **The quantizer** can meet the rule without changing `D`, by giving a site unused classes or by turning a `pot` site whose classes span at most 15 bits into the affine site of shift `r_max` with multipliers `2^{r_max − r_c}`.
  - Amended: RFC-0007 'Verifier validation' step 2, RFC-0022 'Proposed Design' and RFC-0037 'Layers and kinds'.
- `tessera-compiler` and `tessera-quant` rescale every layer's sites alike, as RFC-0038 requires ([#348](https://github.com/AbdelStark/tessera/issues/348)):
  - **The rule.** `validate` and `lower` refuse a claim list whose site is affine in some layers and not in others ("each site is affine in every layer or in none"). They also refuse one whose `pot` site has one class in some layers and several in others ("each pot site has one class in every layer, or several in every layer").
  - **The residual's route.** A residual site takes the fused key in every layer when each layer's site has one shift of at most 62, and two steps in every layer otherwise. A shift of 63 in one layer sends every layer of that site through two steps.
  - **The schedule** stacks layers whose sites have different numbers of classes, several in each.
  - **`spec/vectors/models/`.** `tiny-2-affine` is affine at `RopeQ`, `RopeK`, `AttnResidual` and `Up` in both layers, and its layers now lower alike. The claim-list vectors are regenerated with it.
  - **The lowering tests' models** draw one class or several once per site, and their affine sites in every layer.
  - **`tessera-prover`'s stage tests** prove every stage of all four conformance models, `tiny-2-affine` among them.
  - **The quantizer.** A site whose channels share one shift `r` in a layer, while another layer's need several, takes one class more there that no channel uses: `r + 1`, or 62 beside 63. `D` computes what it computed without it. At SiLU that class counts toward the table's core, which can coarsen that layer's input exponent by one. `choose` refuses `affine` named for a weight in some layers only.
  - **The integer reference** and the conformance oracle route each residual site once for all layers, as the compiler lowers it. The oracle refuses a model whose layers would route a site differently.
  - Amended: RFC-0038 'What the quantizer can do' allows any unused shift as a class, and RFC-0005 'Scale policy' points to it.
- RFC-0039, accepted, defines `StatementField`, the payload of `Reject(StatementMismatch)`, and every check that returns it ([#346](https://github.com/AbdelStark/tessera/issues/346)):
  - **One variant per check, in check order.** The claim list's checks are `ProtocolVersion`, `Architecture`, `Quantization`, `Rescaling { site }`, `Profile`, `Table { table }`, `GadgetPolicy`, `Gadget { gadget }`, `Layout` and `Template`. The statement's are `ClaimListDigest`, `WeightsRoot`, `WeightsEpoch`, `ManifestDigest`, `Batch` and `Predictor`, and the advice commitment's is `LayoutDigest`. Every payload is public: a site type, a table's index or a gadget's index.
  - **RFC-0007 'Verifier validation' names each check's field and states every rule.**
    - Step 1 accepts the parameter set first, as RFC-0019 orders it.
    - Step 2 lists the architecture rules and the public quantization rules, which were split between 03 §2, RFC-0025 and the compiler.
    - Step 5 checks `max_log2_rows` first, then each gadget in node order, then that the derived tables are the claim list's, id for id.
  - 04 §2 declares the type.
- `tessera-protocol` implements RFC-0036's messages and live sessions, `tessera-protocol/1` ([#98](https://github.com/AbdelStark/tessera/issues/98)):
  - **Messages.** The TCE schemas of `SeedCommit`, `ProverSeedShare`, `SeedReveal`, `Round1`, `Round2Response`, `ColumnQueries`, `Round3Response` and `NiProof`, field for field, with their parts: `PublicStatement` and `PredictorSpec` (03 §4, RFC-0017), `AdviceCommitment`, `LogUpProof`, RFC-0037's `TreeProof` and `StageProof`, `CombinedRow`, `WeightCombinedRows`, `ScoringClaims`, `FixedBits`, `MatrixQueries`, `MatrixColumns` and `MultiProof`. They reuse `tessera-field`'s elements, `tessera-sumcheck`'s `SumcheckProof` and `tessera-ir`'s `StageId`. `MultiProof` and `CombinedRow` are wire types of their own: `tessera-hash`'s and `tessera-pcs`'s have no TCE encoding. Each message seals and opens its envelope (`Message`), and `WeightMatrix` gives the matrix ids and the names labels spell. `Round1` carries the claim list's protocol version, `tessera_ir::PROTOCOL_VERSION`: 0 until `tessera-protocol/1` is assigned (09 §3).
  - **Decoders.** Every length is checked against the limits and the bytes left before anything is allocated. The entries of `weight_combined`, and of the queries and answers that follow its order, must name a `WeightMatrix` in strictly increasing order: another id is `ERR-CODEC-NONCANONICAL` at its byte, checked before the rest of its entry. The entry types have no `Decode` of their own, so they decode only inside their message, which checks the order.
  - **The coin toss.** `coin_commitment` and `coin_toss`, in RFC-0004's `coin-commit` and `coin-toss` domains, and `PublicStatement::digest`, RFC-0007's statement digest.
  - **Sessions.** `ProverSession` drives a `ProverEngine` through `AwaitCommit`, `AwaitReveal`, `AwaitQueries` and `Done`, and `VerifierSession` runs `VerifierChecks` from `Start` to `Done`, with 02 §3's calls. Each refusal of RFC-0036's tables returns its code:
    - `ERR-PROTO-ORDER` for a message out of order or of another session;
    - `ERR-PROTO-COINTOSS` and an abort for a reveal that does not open its commitment;
    - `ERR-PROTO-REUSE` for a second seed of one `Round1` (INV-PROTO-1);
    - `ERR-PROTO-QUERIES` for queries of other matrices, counts or ranges (INV-PROTO-3), naming a refused query by its position, not its index;
    - `ERR-PROTO-VERSION` for another protocol version.

    The prover's refusals leave its state as it was, except a reveal that does not open its commitment, which aborts the session. So does a failed engine, a session dropped before its end, or one whose engine is taken back with `into_engine`: the engine discards the `Round1`, its commitment and its randomness. The session draws nothing. `s_P` comes from the engine, derived from its per-proof secret seed, so a prover resuming after a crash sends the `Round1` and the share it sent (RFC-0018 'Crash consistency'). The verifier ends at its first failure, and its share stays out of every debug form until `SeedReveal` carries it. The verifier's session id, share and column queries come from the operating system's CSPRNG through `getrandom` 0.4, the queries uniform by rejection, with 1 to `2^16` per codeword.
  - **Vectors.** `spec/vectors/protocol/`, written by an independent Python oracle that checks its output with its own decoder: a small instance of every message with its envelope and value and the offset of every prefix's truncation, the coin toss, 55 refused inputs with their codes and offsets, 10 inputs at their limits, and 6 refused envelopes.
  - **Tests.** One test per refusal of both tables, with its exact code and the state it leaves; both sessions through every state, and a live session between them through envelopes; a session dropped, or its engine taken back, in every state; a resumed prover sending the same bytes. Decoders are tested on hostile lengths under a counting allocator, on the order of their checks and on their `MIN_SIZE`, with `proptest` round trips and edited encodings judged without the decoder. Debug forms are checked to show no protocol-class value, and every error's code across the crate boundary. A mutation run in a scratch copy catches 78 of 79 mutants of the decoders, the coin toss, the queries' check, bounds and draw, both sessions with their discards and version check, the error codes and the debug forms, with a no-op control surviving. The survivor removes the clearing of the verifier's share, which no test can observe.
  - **Specification.** RFC-0036 states the matrix-id rule normatively, the order of the verifier's checks on `Round1`, where `s_P` comes from, and the protocol version 0 until `tessera-protocol/1`. RFC-0019 and RFC-0036 place the drawing of live queries at step 10, before `ColumnQueries`, with steps 11 to 13 after `Round3Response`. 04 §3's `ERR-RES-IO` covers a failure of the operating system's random source, RFC-0010's `StageProof` gains RFC-0037's `splits`, and RFC-0036 names #128 for the verifier's decoders.

- RFC-0040, accepted, defines the weight commitment's two artifacts and closes the gaps in its layout ([#102](https://github.com/AbdelStark/tessera/issues/102)):
  - **`WeightCommitmentPublic`** (`0x0301`, `.twr`) holds the layout, the ten matrix roots and the epoch, and `R_W` is the hash of its encoding. The verifier reads the roots from it, and `tessera-verify` takes it as `--weights WEIGHTS.twr`. A `.twr` that does not hash to the trusted root is `ERR-USAGE`.
  - **`WeightCommitmentState`** (`0x0302`, `.twc`) holds the public part, `masks_seed` and a budget per matrix, with `opened_columns` a strictly increasing set. RFC-0012's two budget fields that repeated `P_W` and `B_W` are dropped. It is written with mode 0600 by atomic rename.
  - **The layout.**
    - `k_T = max(in_pad, min(k_W, out_pad · in_pad))`, so that `test-small`'s 512-wide inputs fit a row.
    - Every entry's position, the committed rows of padded layers and outputs, and `ChannelParams` with no live channel (`C = 1`) are defined.
    - A weight matrix's leaf is its single chain's last digest, with no `column-merge`, and the first epoch is 0.
  - Amended: RFC-0012 'Layout' and 'Budget state machine', RFC-0019 'Inputs' and steps 4 and 11, RFC-0025 'Hidden data', RFC-0026 'Schema', RFC-0039 'The checks' (the weight layout's check, `Layout`, at RFC-0019 step 4), 02 §2.2–2.3 and 03 §5. 00 RISK-4 now says `τ_W = 97`, as RFC-0014 computes it with the masks, rather than 95.
- RFC-0041, accepted, defines the tiles of a sequence, their production order and the execution plan ([#101](https://github.com/AbdelStark/tessera/issues/101)):
  - **Tiles** partition every committed value: the tokens, one tile per committing node, a layer's `SoftmaxRow` split by KV head and query block, and each lookup table's multiplicities.
  - **The production order**, which the advice layout follows: node order, an attention node's tiles by KV head, then query block, and the tables last by id. Within a tile, the node's outputs come first, then its gadget's values.
  - **`ExecutionPlan`** (`0x0202`) holds the claim list and one sequence's tiles, which do not depend on its length. It holds no row ranges, which are the layout's, no stage membership, which is derived, and no devices, which are assigned at prove time.
  - A proof unit of `tessera-protocol/1` covers one request until #357 defines batches of several.
  - Amended: RFC-0006 'Tiling and the execution plan', RFC-0011 rule 2, RFC-0029 'The table set' and 03 §5.
- `cargo xtask lint` rejects a merge-conflict marker in any file of the repository (07 §9). Markdown reads a marker as text, so one left in 03 §5's artifact catalog by a rebase had passed every other lint check before a codec test caught it in CI.
- RFC-0042, accepted, puts each layer's `Silu` input exponent in the claim list ([#358](https://github.com/AbdelStark/tessera/issues/358)):
  - **The defect.** The lowering shares a table among layers whose definitions are equal, and the claim list stated `in_exp` only in the table. Models with `in_exp = (a, b, a)` and `(a, b, b)` therefore had byte-identical claim lists, against INV-CL-3 and INV-DM-3.
  - **`ClaimList.silu_in_exponents`**, after `quant`, holds one exponent per layer, sorted. Claim-list validation step 2 checks its shape, and step 5.3 checks each layer's table against it.
  - **`tessera-compiler`** emits the field, and `PublicModel::of_claims` lowers a claim list from its own fields, as the verifier will. For every conformance model, that gives the graph and tables that lowering `D` gives.
  - **`spec/vectors/claims/claim-lists.json`** is regenerated.
  - Amended: RFC-0007 'Schema' and step 2, RFC-0026 'Schema' and 03 §2. RFC-0039's Non-Goals now say that `rms_eps_int` and `rsqrt_kappa` are stated constants, which validation does not derive from `rms_eps` (RFC-0023).
- RFC-0043, accepted, defines the payloads of `Reject(Malformed)`, `Reject(ProofOfWork)`, `Reject(Anchor)` and `Reject(Params)` ([#353](https://github.com/AbdelStark/tessera/issues/353)):
  - **`ArtifactKind`** is the schema id of an artifact the prover sends (`ProverSeedShare`, `Round1`, `Round2Response`, `Round3Response`, `NiProof`, or a batch's `DigestLogEntry`). One of the verifier's own inputs that fails to decode is `Inconclusive(ERR-CODEC-*)`, as 04 §5 already said and RFC-0019 step 1 now does.
  - **`PowPhase`** is `{ Seed, Queries }`. **`ParamSetName`** is RFC-0014's `ParamSetId`.
  - **`AnchorFailure`** is `{ ManifestSignature, Inclusion, Frames }`, RFC-0015's checks 1 and 2. `Anchor.request_id` is now an `Option<u64>`, `None` for the manifest's signature.
  - **RFC-0016's checks** reject as `StatementMismatch { ManifestDigest }` and `{ Batch }`, a replayed request included.
  - Amended: 04 §2, RFC-0015 and RFC-0016 'Verifier checks', RFC-0019 step 1 and RFC-0039 'The checks'.
- RFC-0044, accepted, states the Llama lowering node by node, and conformance vectors hold the compiler to it ([#362](https://github.com/AbdelStark/tessera/issues/362)):
  - **The statement.** It gives the id rules for tensors, nodes, constants, shift-class sets, tables and gadgets, and the graph inputs and their types. It gives the procedures: `matmul`, a site's key, `requant`, `rescale`, `norm`, RoPE, the head layouts and both residual routes. Then the prologue, a layer and the epilogue in order, and the `Exp` and `Silu` table rules. The verifier's lowering (#107) must reproduce every id.
  - Amended: RFC-0006 'Range analysis' gives the positions' range as `[0, n)`.
  - **The oracle.** `generate.py`, written from RFC-0044 alone, writes the graph each conformance model lowers to at `n` = 1, 4, 5 and `max_context`, every id in creation order: each input with its role, shape, kind, type and range; each node with its operation, parameters, inputs, and outputs' shapes and kinds; the constants, the shift-class sets, the weight list and the tables. `cargo xtask gen-vectors --check` covers it.
  - **Every form.** Two variants of the models take the forms the models miss: affine `V` and `MlpResidual`, two steps at a site of one shift, a `Silu` site of one class, heads with no `Expand`, a shared `Exp` table, an `Exp` table wider than the spread, and one whose last key rounds to 0 before its zero tail.
  - **`tessera-compiler`** lowers every case through its public API and compares the graph field by field (`tests/lowering_vectors.rs`). A difference is reported at its first item: the case, the input, node or entry, and the field. All 24 cases match.
- `tessera-compiler` derives the tiles and the execution plan of RFC-0041 ([#101](https://github.com/AbdelStark/tessera/issues/101)):
  - **Tiles.** `tiles(model, &lowering, &selection)` gives a sequence's tiles in production order:
    - `Tokens` first;
    - then, in node order, a `Node` for each node that commits a value, and one `Attention` per KV head for each layer's `SoftmaxRow`;
    - then a `Table` for each table a gadget's lookups read, by id. RoPE's table, which no lookup reads, has none.
  - **Expansion.** `ExecutionPlan::expand(&lowering)` takes each attention tile at its `N / Q'` query blocks, `Q' = min(Q, N)`. It gives each tile's share of every graph tensor committed in the advice as one range per axis: for an attention tile, its KV head's query heads, its block's queries and every key. The values a gadget commits beyond its node's outputs, such as `SoftmaxRow`'s maxima and bracket limbs, are no graph tensor: they belong to the node's tile, as `Tile` documents.
  - **The plan.** `ExecutionPlan { claims, tiles }` (`0x0202`) encodes as RFC-0003 has it, a tile's tag its declaration index.
    - `plan(model, &claims)` builds it from the public part of `D` and the claim list `claim_list` emits for it. It refuses another claim list, and a `query_block` that is not a power of two (`ERR-USAGE`).
    - `check_plan` refuses a plan whose claim list or tiles are not `D`'s (`ERR-USAGE`).
  - `spec/vectors/codec/execution-plan.json` holds the conformance models' plans for `test-small/1`, sealed, with their tiles and their claim lists' digests. `cargo xtask gen-vectors` writes it after the models' oracle, and a test keeps it current.
  - **Tests**, on the four conformance models at lengths from 1 to the context and query blocks of 512, 4 and 1:
    - every element of every tensor committed in the advice lies in exactly one tile, counted element by element;
    - the tiles follow the production order, with the read tables taken from the commitment report's lookups;
    - a `SoftmaxRow` and an `Argmax` added outside the layers each take a node's tile, the `Argmax`'s holding no graph tensor;
    - each attention tile holds the query heads whose maxima in the oracle's traces read its KV head's keys;
    - node and tensor ids, and the tiles, are the same at every length from 1 to 17 and at the context;
    - two builds give the same bytes, and a model whose weights hold a canary has the same plan, without the canary;
    - a plan with one tile changed, removed, added or moved, or with another claim list, is refused;
    - the golden plans decode to the compiler's, and every proper prefix of their payload is cut short.
  - **Specification.** RFC-0007 step 6 states that `query_block` is a power of two. RFC-0041's plan check compiles for the plan's gadget policy, layout and parameter set. 03 §7 says where each artifact's golden vectors live.
- RFC-0046, accepted, makes the advice layout's segments and short attention tiles exact ([#106](https://github.com/AbdelStark/tessera/issues/106)):
  - **Segments.** The first `L mod S` segments hold `⌈L/S⌉` layers and the others `⌊L/S⌋`, in order. Segment 0 also holds the tokens and the prologue, and segment `S − 1` the epilogue and the tables. A segment's rows run up to the next segment's first tile, and an empty segment's column digests are `d_j[0]`.
  - **Short query blocks.** RFC-0034's attention layout reads RFC-0041's `Q' = min(Q, N)` for `Q`.
  - **Layout.** RFC-0007 step 6 refuses a claim list with `g·min(query_block, N_max) > k`, whose rows could not hold one key against a tile's queries.
- RFC-0045, accepted, lays out the leaf stage's lookups and splits their claims ([#367](https://github.com/AbdelStark/tessera/issues/367)):
  - **Blocks.** An instance's lookup column holds one block per lookup of a site at a layer, over its tensor's padded shape. Blocks are aligned and sorted by size, and row 0's tuple pads the column to a power of two.
  - **Leaves.** The tree levels front-pad, so trees of different heights agree only on their lowest and highest coordinates. The lookup tree holds its column with bits 1 to `h − 1` reversed. Every block of `2^k` entries, in every instance, then lies at `P[0..k]`, with `P` the last tree level's challenges in reverse order. `tessera-logup` and the tree levels are unchanged.
  - **The split.** Under `stage/0/split`, the prover sends each tensor's value at its point, per site and layer. The verifier checks each instance's key and value claims against them (`Reject(Sumcheck { stage: 0, round: 0 })`). The values are RFC-0037's per-layer claims, which stage 1 stacks under its layer point.
- `tessera-gadgets` holds the requantization and range gadgets of RFC-0009 and RFC-0024: their constraints, their witnesses and their LogUp instances ([#118](https://github.com/AbdelStark/tessera/issues/118)):
  - **Gadgets.** `Requant` builds `Requant1`, `RequantSplit` or `RequantWide` from the compiler's `Gadget` and the tables it names. It refuses a table that is not the one the gadget reads in that role, a key over 62 bits, and a `RequantWide` middle part of another width than RFC-0024's `t`. `NonNeg` proves `v ∈ [0, 2^k)` for `k ≤ 62` with limbs of `GadgetPolicy.limb_bits`.
  - **Constraints.** Each gadget's `Constraints` list its committed tensors, its virtual values and its lookups. Each key and value is an `Affine` form over the input `z`, the hidden class column `c`, the committed tensors and constants. A form evaluates at an element, or at a point of its column's hypercube from the terms' extensions, so it turns a LogUp claim on a key or value column into claims on `z`, `c` and the committed tensors. `RequantSplit`'s high key is `z_hi + 2^{b−s}·c`: `z_hi` already is the offset encoding of `⌊z / 2^s⌋`. The class term is present when the gadget has several classes.
  - **Witnesses.** `witness` computes the committed tensors of the one assignment each lemma admits from keys, classes and values its caller lends it, into `SecretBuf`s, without branching on them. An input outside the gadget's domain is `ERR-INTERNAL`, reported after the last element.
  - **LogUp.** `Lookups` evaluates the constraints into the LogUp prover's columns, one instance per table the gadget reads, the lookups into one table one after another. `multiplicities` counts what an honest prover commits, and `instances` pairs the columns with `TableColumns`, a table's dense rows. `secret-access.toml` lists `witness.rs` and `lookups.rs` as borrowers.
  - **Lemma 2's precondition.** A split key of `n` classes needs `n·2^b < P`, since Lemma 2's two sides lie in `[0, n·2^b)`. Four classes at 62 bits break it: `z_lo ± 1` at either end of the key's range wraps the high key to the table's other end, and passes with another output. `Requant` refuses that split, and the compiler's selection gives such a key `RequantWide`, with `s = 32`, `t = 11` and `k = 19` at shifts 33 to 36 and INT8 bounds. Tests show the second witness passing the split of four classes over Goldilocks, and rejected by three classes and by `RequantWide`; over `F65537`, exhaustion shows the bound exact; both selections at 62 bits are complete. Of the two checks' 11 mutations, the 2 that survive are equivalent: `>` for `≥`, since `n·2^b` is even and `P` odd.
  - **Tests.**
    - Completeness on 10^4 keys per gadget, both ends of the range among them, at small widths and at the compiler's production selections, `minimal`'s 8B residual among them. Every tuple is a row, every output is `rq(z, r_c, lo, hi)` and every virtual value is the integer its RFC defines. At small widths the LogUp instances verify through `tv-core` and `tv-tables`.
    - Lemma 1 by exhaustion. Lemma 2 by exhaustion for keys of up to 12 bits at every split point below the rounding bit, over `F65537` and Goldilocks, and `z_lo ± 1` and `z_lo ± 2^s` rejected at exactly the lookups integer arithmetic predicts.
    - Lemma 11 by exhaustion over `F65537` for keys of up to 14 bits, in RFC-0024's five configurations. RFC-0024's odd-key witness is rejected by the top part's `Range { k }` alone; without that lookup it is accepted and returns another `y`.
    - Lemma 3 by exhaustion over every element of `F65537`, its boundaries up to `k = 62`, and a carried limb rejected by its own `Range { L }`.
    - End to end, the verifier's leaf claims are the constraints evaluated at their points, and each tampered witness is rejected, `Unbalanced`, at the one instance whose check it breaks.
    - 107 of 108 mutations of the crate are caught. The survivor refuses a table of exactly `P` rows, which no table kind has.
- `tessera-prover`'s `leaf` module lays out the leaf stage's lookups and proves their split, as RFC-0045 defines them ([#367](https://github.com/AbdelStark/tessera/issues/367)):
  - **Layout.** `LeafLayout` derives everything from public data alone. Sites are gadget nodes in node order (`gadget_nodes`, `Site::requant`). Each site, real layer and lookup is one block, and each table one instance, in table order, its blocks stably sorted larger first at prefix-sum offsets, `h = ⌈log2 m⌉`. Each site's split sends the tensors its lookups read, in the order they first read them, each at `π_t(P[0..k])`. A gadget whose lookups read one tensor over two sets of axes, and a site whose real layers read different tensors or axes, are refused, `ERR-INTERNAL`.
  - **Columns.** `LeafColumns` evaluates each block's forms over its padded domain and writes column entry `q` at leaf `σ_h(q)` (`leaf_index`: bit 0 kept, bits 1 to `h − 1` reversed). Row 0's tuple pads each column, and the multiplicities count the padding: `tessera-logup` receives `2^h` lookups and pads none. `tessera-logup` and `tessera-sumcheck` are unchanged.
  - **Split.** `prove_leaf` proves the instances with `prove_logup` and reads `P = (ρ_{H−1}, r_{n−1}, …, r_0)` from the last tree level (`leaf_point`). It then absorbs the split values under `stage/0/split`, after `logup/values`, as one TCE `Vec<Fp2>`. `secret-access.toml` lists `leaf/prove.rs` as a borrower.
  - **Specification.** RFC-0045 now states that a tensor read over some of a block's axes lies at `π_t(r_s)`, and gives the split's tensor order as a numbered rule. It also states that table order is ascending `TableId`, for the instances, their check and their trees, as RFC-0008's roots now cite, and that every real layer of a site reads the same tensors over the same axes. It sets `tessera-protocol/2`'s masked leaf checks out of scope and makes `σ_0` the identity.
  - **Tests.**
    - `tiny-2-servable`'s requantization sites whose tables hold at most `2^20` rows, positions 13, 14, 32 and 47, take their keys and classes from the CPU engine's forward pass and their witnesses from `tessera-gadgets`, whose outputs are the engine's. A verifier written from RFC-0045 accepts them, on `tv-core`'s LogUp checks, and each split value is its tensor's extension at its point.
    - Each tampered split value is `Reject(Sumcheck { stage: 0, round: 0 })`, and a wrong value in a block fails LogUp's root check.
    - Blocks of one size share `P[0..k]` in instances of heights 11 and 9, where a natural layout would split them and is rejected. Swapping two blocks of one size, or padding with another row, passes LogUp and fails the split's check.
    - A tensor read over an outer axis alone lies at that axis's coordinates of `P`. A gadget whose lookups read one tensor over two sets of axes, a site whose layers read different tensors or axes, and sites out of node order or without their layers, are refused.
    - `σ_h` is an involution, and the verifier's `O(h)` `LT` is its sum of `eq`.
    - Of 29 mutations, 24 of the module and 5 of the test's check, 28 are caught. The survivor pads the keys with 0, which is row 0's key in every table.

- `tessera-protocol` implements RFC-0014's parameter sets and soundness calculator, and `tessera params check` reports on them ([#103](https://github.com/AbdelStark/tessera/issues/103)):
  - **`ParamSet`** holds `live-64/1`, `ni-100/1` and `test-small/1`: the mode, the commitments' layout RFC-0026 writes into the claim list, `τ_A`, `τ_W`, `g_seed`, `g_query`, the row block and segments, whether a release verifier accepts the set, and its targets. `ParamSet::accept` takes a set from its `ParamSetId` (INV-PROTO-5): `test-small/1` only with the `test-params` feature, which no release build has, and any other identifier is `ERR-USAGE`.
  - **`SoundnessReport`** gives every row of RFC-0014's soundness table for a set at a statement's `StatementSizes`: rounds, the union bound, product forests, LogUp, openings, each commitment's queries and field term `ϑ·n/q`, weight linking and hashes. Each row has its exact error bound, its bits `λ = g + max(0, −log2 ε)` as a FixedBits lower bound `r ≤ 256·λ < r + 1 + 2^-20`, its grinding, its target and whether it reaches it, then the minimum.
  - **Integers only.** Bits come from LOG2, a fixed-point `log2` that rounds down at every step, and each decision compares integers, `c·2^T ≤ q` and `A^τ ≥ 2^T·B^τ`, so that `q < 2^128` cannot make a rounded bit decide: `T_g = 2^28` fails a 100-bit target by `6.7·10^-10` bits, `2^28 − 1` passes it. `cargo xtask lint` keeps floating point out of the calculator's sources.
  - **Targets**, from SP-1: 100 bits for every row a prover can retry, and for the rows a proof of work covers, the LogUp identity and the queries, `live-64`'s 64 bits one shot and `ni-100`'s 100. `test-small` is held to none.
  - **The 405B 16K-token, 16K-context statement**, `StatementSizes::llama_405b_16k`, with each size's source: both release sets meet their targets there. The minimum is 64.03 bits at `live-64`, a weight matrix's queries, and 100.09 at `ni-100`, the advice queries. `N + T = 2^50` fails `ni-100`'s LogUp row.
  - **`tessera params check --params PARAMSET [--lookups N] [--table-rows T]`** prints the table, or with `--json` the `tessera.soundness/1` report (`spec/schemas/soundness-v1.schema.json`). `docs/perf/examples/soundness-live-64.json` and `soundness-ni-100.json` are the 405B reports.
  - **Vectors.** `spec/vectors/soundness/`, written by an independent Python oracle, holds 77 reports with each row's true value at 60 digits: the three sets at 405B, statements and parameters that miss exactly one row's target, the exact comparisons' edges, and random sizes under random sets.
  - Amended: RFC-0014 'Soundness calculator', now normative to the integer, and 'Grinding', which names the bounds each proof of work covers: the field terms get no grinding bits, so 'Soundness budget' gives `ni-100` about 100 bits there, not 120. 03 §5 lists `tessera.soundness/1`.
  - Amended: RFC-0012 'Layout' and 'Budget state machine', RFC-0019 'Inputs' and steps 4 and 11, RFC-0025 'Hidden data', RFC-0026 'Schema', 02 §2.2–2.3 and 03 §5. 00 RISK-4 now says `τ_W = 97`, as RFC-0014 computes it with the masks, rather than 95.
- `tessera-pcs` commits the weights of `D` as RFC-0040 defines it for `tessera-protocol/1` ([#102](https://github.com/AbdelStark/tessera/issues/102)):
  - **The rate.** `RowEncoder::new` and `CommitmentParams` take the inverse rate's `log2`: 1 for the advice, as before, and 2 for the weights, rate 1/4. A rate of 1 is `PcsError::InverseRate`. Every advice test and vector passes unchanged.
  - **The matrix commitment.** `ReferenceCommitment::commit_unmasked` encodes a matrix's committed rows with no mask, chains each column once over them, and takes the chain's last digest as the leaf, with no `column-merge`. The advice keeps refusing `P = 0` (`NoMask`).
  - **The layout.** `WeightLayout` places every weight and channel parameter from public data (`WeightShapes`):
    - `k_T`, `a_T` and `h_T`, and each entry's row and column;
    - the structurally zero rows it skips, and `m_T`;
    - the `ChannelParams` tensor flattened as `(t · L_pad + ℓ) · C + c`, with `C = 1` when no weight has one.
  - **The commitment.** `WeightCommitment::commit` lays out and commits the ten matrices. Each starts from the `commitment_id` of its `CommitmentDescriptor::WeightMatrix`, which implements RFC-0031's descriptors, the advice's too. The result is `WeightCommitmentPublic`, whose hash is `R_W`. Each matrix's row layout is built from public data first; its rows are then allocated inside their `Secret` and filled there, so no weight sits in a plain `Vec`.
  - **The artifacts.** `WeightCommitmentPublic` (`0x0301`) encodes and decodes in TCE; `WeightCommitmentState` (`0x0302`) has the TCE encoding of RFC-0040 but no `Encode` or `Decode` of its own.
    - The state's `masks_seed` is private, in a `Secret` of a boxed `[u8; 32]`, and has three sources: `WeightCommitmentState::new` draws it from the operating system's CSPRNG through `getrandom` 0.3, which the lockfile already held; `open` and `read` decode it from a `0x0302` envelope; and, in tests only, `with_seed` pins it. `with_seed` exists in the crate's own test build and, for other crates, under the `test-rng` feature.
    - A private wrapper encodes the state only into its file, by `write`, or, in tests only, as `with_seed` is, into a `SecretBuf` (`sealed`). Release code cannot obtain the seed: a compile-time assertion and two `compile_fail` doctests check that the state implements neither trait and that the field is out of reach.
    - No dev-dependency of the crate on itself enables `test-rng`. That would enable the feature in the crate's benchmarks too, which build without debug assertions and so stop at its guard: `cargo bench -p tessera-pcs` would fail.
    - Its `opened_columns` is a strictly increasing set: decoding rejects an unsorted or repeated index with `ERR-CODEC-NONCANONICAL`.
    - The state is written with mode 0600, the public part with the default mode, both by atomic rename.
  - **Files.** `tessera_codec::envelope::write_file` writes an artifact by atomic rename: a new file beside the target, written, flushed, renamed over it, then the directory flushed. A secret-class artifact has mode 0600 from the moment its file exists, and an interrupted write leaves the previous file whole.
    - The new file is `.<name>.<16 hex digits>.tmp`, 64 bits from the CSPRNG, created exclusively, so never through a link planted there. A name already taken is drawn again, `TEMPORARY_ATTEMPTS` (8) times at most; then the write fails with `TemporaryNamesTaken`, of kind `AlreadyExists`.
    - A crash can leave the new file, with mode 0600, beside the target.
  - **Placement.** `AdviceLayoutParams` and `WeightLayoutParams` move to `tessera-pcs`, and `tessera-ir` re-exports them, so the claim lists' bytes do not change. `tessera-ir` gives each weight type's shape from `ArchSpec` (`ArchSpec::weight_shapes`), and lends an `IntegerModel`'s weights to the commitment (`IntegerModel::weight_shapes`, `IntegerModel::model_weights`).
  - **Domains.** `tessera-ir` checks each value against the domain `D` gives it before it lends the weights, as the prover checks the hidden part before lowering (RFC-0006 'Llama lowering'), so the commitment never binds a value outside it. Each check reads every value of a tensor, so its control flow follows the shapes alone. Each refusal has its own `IrError` and an adversarial test at its boundary:
    - a weight entry of −128, outside `[−127, 127]` (RFC-0005 'Pipeline', RFC-0006 'Range analysis'): `WeightOutOfRange`;
    - a channel class at or above the shift classes of the site after its weight (03 §2 'Channels'): `ClassOutOfRange`;
    - a multiplier of 0 (03 §2 'Channels', RFC-0025): `MultiplierOutOfRange`;
    - a model of no layer, against RFC-0007 'Verifier validation' (`n_layers ≥ 1`): `NoLayers`.
  - **Vectors.**
    - `spec/vectors/weights/generate.py` is an independent oracle in Python's integers. For each conformance model at `test-small/1`, and a synthetic model with no `ChannelParams`, it gives every matrix's layout, `commitment_id` and root, then `R_W` and the `.twr` artifact. It also encodes rows at rate 1/4. `tessera-pcs` and `tessera-ir` reproduce every value.
    - `spec/vectors/codec/weight-commitment.json` holds golden envelopes of both schemas, and states whose opened columns are unsorted or repeated. `cargo xtask gen-vectors` lays out the states' bytes itself, from RFC-0040's field list; a unit test of `tessera-pcs` checks that `sealed` gives each state's value, pinned through `with_seed`, as exactly those bytes.
- RFC-0048, accepted, states what the weights root vouches for ([#369](https://github.com/AbdelStark/tessera/issues/369)):
  - **Uniqueness.** Every gadget's uniqueness argument holds for arbitrary committed weights, classes, multipliers and padding: an injective digit box below `P`, or a table that is a function of its key. A malformed `R_W` cannot let the prover adapt outputs.
  - **The assumption.** `R_W` is the root of a `D` that `commit-weights` accepted. That gives fidelity (`D` is its claim list's Llama) and totality (every input has a proof). A one-time well-formedness proof would buy no uniqueness.
  - **Amendments.** RFC-0009's requantization lemmas split into a uniqueness half and a value half. 00 §3, 06 §5, RFC-0006 and RFC-0040 state the assumption.
- RFC-0009 closes two uniqueness gaps the #369 analysis found, before `RsqrtRow` (#126), `OneHot` and `Argmax` (#123) are implemented ([#377](https://github.com/AbdelStark/tessera/issues/377), [#378](https://github.com/AbdelStark/tessera/issues/378)):
  - **The limb box.** Every decomposition of a field element into range-checked limbs has its box below `P`. `RsqrtRow`'s `u` takes `NonNeg(u, 55)`: three 16-bit limbs and a 7-bit top limb. Four full 16-bit limbs would also represent `u + P`, and at 8B admit a second root `ρ`. RFC-0035's padded rows cite the rule.
  - **Token ids.** `OneHot` gains the zero-check `Σ_x eq(r, x)·(1 − LT(x_v, vocab))·h(x) = 0`, which pins `t ∈ [0, vocab)` at no commitment or lookup. Lemma 8 now proves that. `Argmax` masks its keys at `v ≥ vocab`, so its checks cover every position the one-hot can mark.
- RFC-0047, accepted, fixes the inputs of the advice layout that no document fixed ([#106](https://github.com/AbdelStark/tessera/issues/106)):
  - **Order.** A committed tensor's layout order puts its padded axes first, then the others, each in axis order. A claim's point is permuted into it. At 405B a `[16384, 53248]` MLP tensor commits 208 rows rather than 256.
  - **Size.** `ν = log2 k + ⌈log2 R⌉`. A layout allocates at most `2^{32}` rows at `max_context` (RFC-0007 step 6, `Layout`), so every column answer decodes under the default vector limit.
  - **Heads.** `n_heads / n_kv_heads` is a power of two (RFC-0007 step 2, `Architecture`), as RFC-0034's and RFC-0037's head layouts assume.
  - **Length.** A request is a sequence of `n = prompt_len + output_len` positions. RFC-0019 step 4 refuses a request longer than `max_context` (`Batch`), which RFC-0009 Lemma 6 and the layout's maximum assume.
  - RFC-0034's masked-row conditions read `Q'`.

- RFC-0049, accepted, squeezes each weight matrix's proximity point right after the seed in `tessera-protocol/1` ([#108](https://github.com/AbdelStark/tessera/issues/108)):
  - **One pass.** The folds run inside the `MatMulWeight` stage, and RFC-0014 row 7 squeezed `pcs/weights/{T}/xi` after every stage. The proximity row then needed a second read of every weight: about 0.24 GPU-seconds per 405B unit. Drawn after the seed, `ξ*_T` lets the fold pass compute both rows, as INV-WC-4 requires. `R_W` predates the session, so the proximity argument is unchanged.
  - **Coefficients.** `ξ*_T` indexes the committed rows by position, `⌈log2 m_T⌉` coordinates, as the column answers carry them.
  - **The calculator.** In protocol/1, `ϑ_T = ⌈log2 m_T⌉` and the weights' rate is unmasked. A matrix with no committed rows has no proximity row.
- `tessera-protocol` implements RFC-0014's transcript schedule of `tessera-protocol/1` in both modes ([#125](https://github.com/AbdelStark/tessera/issues/125)):
  - **The registry.** `LabelPattern` holds every tier-A label pattern of `tessera-protocol/1`: RFC-0014's, RFC-0035's zero-check point, RFC-0037's layer point and split values, and the proofs of work's `seed-pow` and `query-pow`. A `Label` is one of them with its indices, and its `Display` is the ASCII label. `WEIGHTS_XI` is the one place that puts each weight matrix's proximity point `pcs/weights/{T}/xi`: right after the seed, for each matrix with committed rows in RFC-0040's order, as RFC-0049 moves it out of row 7. Row 7 keeps `pcs/advice/gamma`, `pcs/advice/xi` and each opened matrix's `pcs/weights/{T}/gamma`.
  - **The expansion.** `TranscriptSchedule::expand` lists every step of one proof unit, a label and its operation, from `ScheduleSizes`: the leaf stage's trees, the reduction stages, the layer point, the advice's and the matrices' proximity points and the opened matrices, in a mode at a parameter set. A step absorbs, squeezes the non-interactive seed, draws challenges in `Fp2` (a proximity point's coordinates one after another under its one label), draws column indices, or grinds. The tree levels follow RFC-0010: a tree of height `h` joins at `H − h`, the forests of several trees that join draw a root point, a level with fraction trees draws `λ`, and a level has as many rounds as its largest instance has variables. Split values and the LogUp messages are absorbed even when empty. `tessera-protocol/2`, whose labels slot in at `Protocol::Two` (#148), a mode the parameter set does not allow, a proof of work above 64 bits and sizes no proof unit has are refused, `ERR-INTERNAL`.
  - **Both modes.** `ScheduledTranscript` holds RFC-0004's transcript to a schedule: each absorb, squeeze, challenge and proof of work must be the next step under its label, and it implements `SumcheckTranscript` for the sumcheck, tree and LogUp provers. An operation out of its place is a `ScheduleFault` that `finish` reports.
    - A live session absorbs the coin-tossed seed under `seed` and the verifier's `ColumnQueries` under `queries` (`seed_live`, `queries_live`).
    - A non-interactive prover grinds `seed-pow` and `query-pow` for the set's `g_seed` and `g_query` bits, then squeezes the seed, and the advice's `τ_A` and each opened matrix's `τ_W` indices (`seed_grind`, `queries_grind`).
    - A verifier counts each nonce's leading zero bits exactly and refuses a short one, unabsorbed, with `PowRejected`, `Reject(ProofOfWork)` with its phase (`seed_check`, `queries_check`). A proof of work out of its place is refused too.
  - **Vectors.** `spec/vectors/transcript-schedule/` is written by an independent Python oracle, which expands the labels from the RFCs and runs `spec/vectors/transcript/`'s transcript. It holds 17 schedules with every label; synthetic cases record complete replay outputs, while conformance-model cases record a prefix through the advice batching challenge:
    - synthetic sizes that reach every rule, in both modes, with proofs of work of up to 10 bits, at `test-small/1`, `live-64/1` and `ni-100/1`'s counts;
    - the four conformance models in both modes.

    `tessera-protocol` expands every case to the oracle's steps, and a prover and a verifier held to each draw its recorded outputs. `tv-hash` replays every recorded step with the verifier's transcript.
  - **The conformance models.** Their sizes are those the derivations give today. The oracle derives the layer point and the weights' matrices from RFC-0040. It states the leaf trees, from the requantization sites' layout (RFC-0045), and the stages, from the stage schedule (RFC-0037), which `tessera-prover`'s `transcript_sizes.rs` derives again and compares. The advice's proximity point waits for the canonical advice layout (#106), so their transcripts run up to `pcs/advice/xi`, and the labels after it are listed. The other gadgets' instances, the `SoftmaxRow` forests and the gadgets' zero-checks change these schedules when they join (#122, #123, #126, #130).
  - **Tests.**
    - A live schedule whose every label is derived by hand from RFC-0014, RFC-0010, RFC-0037, RFC-0045 and RFC-0049; the rows of a non-interactive one; schedules without trees.
    - INV-FS-4: no label twice, on every vector case, the conformance models in both modes and random sizes.
    - INV-FS-1, against RFC-0036's message order: each prover message is absorbed before every challenge that depends on it and after every challenge it is computed from, the seed follows `Round1` and live queries the whole `Round2Response`, on random sizes too. Eleven violations, each moving one step, are caught. No prover message comes between a proof of work and the challenges it covers.
    - A nonce one bit short is refused at each phase, at `ni-100/1`'s 20 bits and at 5 and 11. Against a stub hash whose nonces 0 to 39 fall short of 13 bits, grinding hashes 41 nonces and stops at the first that meets the target, and the check accepts 13 and 16 bits and refuses 12 and 8, with one hash each.
    - Exchanging any two adjacent absorbs changes every later challenge, seed and index, in both modes.
    - Every operation out of its place is a fault at its step, and a session of another mode, set or protocol is refused. The maximum accepted stage count, 65,535, expands without overflow in either mode; the regression panics against the former open-ended `u16` range.
    - Of 72 mutations, of the expansion, the registry, the sizes' checks, the transcript held to its schedule and the vectors, all 72 are caught, and a no-op control survives.

- Clarify RFC-0010 and RFC-0037's stage conventions for independent verification ([#114](https://github.com/AbdelStark/tessera/issues/114)): identity-order final evaluations (`MatMulWeight`: weights before activations), committed split groups include their coefficients, stage degree is the largest instance degree, and layer-weight extensions include the padded cube. Add compiler-generated schedule fixtures and a prover-side harness checked by the independent verifier.

- `tessera-pcs` opens the weight commitment by the folding pass, on CPU, as RFC-0012 and RFC-0049 define it for `tessera-protocol/1`: one claim per matrix, no mask, no blinding row ([#108](https://github.com/AbdelStark/tessera/issues/108)):
  - **The pass.** `WeightFold::new` reads each committed row of a matrix once, in order, through a `MatrixRows`, into the per-layer folds `F_ℓ(o_lo, i) = Σ_{o_hi} eq(r_o^hi, o_hi)·W_ℓ[(o_hi, o_lo), i]` and the proximity row `u_ξ,T = Σ_{j < m_T} eq(ξ*_T, j)·W_T[row_j]`. `ξ*_T` has `⌈log2 m_T⌉` coordinates, one per bit of a committed row's position, and is an input when the pass starts: RFC-0049 squeezes it right after the seed. `WeightLayout::rows` lays a matrix's rows out from `D`'s weights one at a time (`WeightRows`), with the placement code the commitment now uses too.
  - **From the folds, no weight read.** `WeightFold::sumcheck_input` is the matmul sumcheck's `W̃(ℓ, r_o, i)`, and `WeightFold::combined_rows` gives `u_T = θ_T·Σ_ℓ eq(ρ_ℓ, ℓ)·F_ℓ` and `u_ξ,T` once `ρ_ℓ` and `θ_T` are drawn (`WeightCombinedRows`). `ChannelParams` opens its one claim the same way, from its row point; several claims are #386's. A matrix with no committed row has no combined rows: `PcsError::NoRows`. The column answers stay the commitment's, from the codewords held in memory.
  - **The sumcheck input.** `tessera-sumcheck`'s `MatMulWeightTables::folded` builds a `MatMulWeight` instance from `W̃(ℓ, r_o, i)` instead of contracting `W` again, and proves the stage `MatMulWeight::tables` proves, message for message.
  - **Tests.** An in-test verifier, written from RFC-0011, RFC-0012, RFC-0040 and RFC-0049 on `tv-field`, `tv-hash`, `tv-codec` and `tv_core::sumcheck::eq`, checks every opening against `R_W`; `tv_core::opening` checks the advice's, whose leaves merge segments. The fold equals `W̃_ℓ(r_o, i)` from the INT8 weights on the conformance models and on random shapes wider than `k_W`; counting readers show one read of each row and each weight per proof; a claim on each of the ten matrices verifies; one changed entry of each matrix, a padding position and a `ChannelParams` entry are refused at the smallest queried column. Of 17 mutations of the pass and its reader, 17 are caught.
  - **The calculator** counts `tessera-protocol/1`'s weights as RFC-0049 'The calculator' says: `ϑ_T = ⌈log2 m_T⌉`, no mask, and no proximity row for a matrix with no committed row, which is no longer refused; `tessera-protocol/2` keeps `m_T + B_W` and `P_W`. At the 405B statement, a `tessera-protocol/1` statement, each weight matrix's queries give 65.76 bits at `live-64` and 82.72 + 20 at `ni-100` (lower bounds, as the calculator prints them), not 64.03 and 80.53 + 20, and `ϑ_T` is 10 to 17, not 13 to 18: the weights' least field term is 101.91 bits, not 101.82. `live-64`'s minimum becomes the advice queries, 64.32 bits; `ni-100`'s stays them, 80.09 + 20. Under `tessera-protocol/2` every figure is as before. `spec/vectors/soundness/` holds 86 reports, the 405B statement under `tessera-protocol/2` and an empty `ChannelParams` among them, and `docs/perf/examples/soundness-*.json` are regenerated.
- Add `BigIntCompare` carry normalization and subtraction constraints (RFC-0009 Lemma 4; [#121](https://github.com/AbdelStark/tessera/issues/121)), range-checked digits/carries/borrows, zero final carries on both inputs and zero final borrow. Generate batched witnesses in clearing buffers; cover 64–128-bit comparisons, alternative witnesses, exhaustive small-field uniqueness and missing-range Goldilocks/F65537 wrap attacks.
- Independently re-derive public integer ranges and canonical lookup-table definitions in `tv-core` (RFC-0006/0007; [#117](https://github.com/AbdelStark/tessera/issues/117)). Validate gadget domains, centered-field bounds, RequantSplit/RequantWide preconditions and lookup width policy, then reject a noncanonical table sequence with its first differing id. Both workspaces check every tensor against an independent Python interval oracle on all 24 lowering cases.
- Add SoftmaxRow's causal maximum certificate, masked Exp lookups, reciprocal NonNeg bracket and padding zero-checks, with clearing witnesses and explicit pointwise relations (RFC-0009, RFC-0030, RFC-0035; #122). Register all auxiliary Range tables in the compiler and independent verifier, count their lookups in commitment reports, and include padded row instances in the default soundness budget.
- Add OneHot and lowest-index Argmax witnesses and sumcheck relations, binding virtual suffixes to the next-token indicator and masking vocabulary padding to public zero values. Register Argmax's Range(29) in selection and the plan, and enforce its public logit domain (RFC-0009; #123).
- `tessera-pcs` derives the canonical advice layout of RFC-0011, RFC-0034, RFC-0041 and RFC-0046 from tiles given as plain data ([#106](https://github.com/AbdelStark/tessera/issues/106)):
  - **Inputs.** `AdviceLayout::new(params, positions, &tiles)` reads public data alone (INV-PCS-1): `LayoutParams` (`log2 k`, `S`, `L`, `g`, `Q`, `max_context`), the sequence's length, and its tiles in production order. Each `TileShape` names its layer, or none for the prologue and the epilogue, and the shapes of its values: a `DenseShape`, its padded axes with their real lengths, the order of its axes and what its padding holds, or a score tile's `Scores { block }`. `tessera-pcs` precedes the compiler and the gadgets (01 §3), so their caller derives the tiles.
  - **Allocation.** Each tensor of `n_X` variables takes the next offset aligned to `2^{max(n_X, log2 k)}`, and a tensor smaller than a row takes one. A score tile orders its variables `(i_hi, h, q, i_lo)` with `w = min(n, log2 k − log2 Q' − log2 g)` and `Q' = min(Q, N)`; `Attention` gives that geometry, and the `[g, Q']` shape of the tile's per-row tensors.
  - **The map.** `Placement::point` maps a claim `X̃(r)`, in the tensor's row-major order, to the point `r' ‖ base_X` of `Ã`'s `ν = log2 k + ⌈log2 R⌉` variables. `Placement::locate` gives an element's row and column.
  - **Zero rows.** A row is skipped when no tensor holds a real entry in it: padding, alignment gaps, rows past the allocation, and a score tile's rows of masked positions. A tensor whose padding holds RFC-0035's constants commits every row. `rows()` counts the rest, the advice descriptor's `rows`, and `row_layout()` lists them, segment by segment, as the reference commitment takes them.
  - **Segments.** `segment_layers` gives RFC-0046's layer ranges, and each `Segment` runs from its first tile's first row to the next segment's, so the gap before a tile is the segment's before it. A segment without a tile holds no row.
  - **Refusals.** `LayoutParams::check` refuses, as `PcsError::Layout` (RFC-0007 step 6, RFC-0039 `Layout`), `g·min(Q, N_max) > k`, a `query_block` that is no power of two, no segment and `log2 k > 31`; the layout also refuses a matrix of more than `2^32` rows. Malformed data is `LayoutData` or `Tile`, with its rule.
  - **Tests**, with every expectation computed from the RFCs' formulas:
    - on random tiles, each tensor's elements written into a resident `Ã` where the layout places them: a claim at a random point equals `Ã` at the layout's point, no two elements share an entry, and a row is skipped exactly when it holds no real entry; through the reference commitment of the non-zero rows alone, `tv_core::opening` accepts a claim on every tensor;
    - positions computed by hand for a sub-row tensor, a permuted tensor and a score tile, and RFC-0034's committed rows per block, for blocks of padding too;
    - at 405B shapes and 16,384 positions, a `[16384, 53248]` MLP tensor commits 208 rows, its elements' count, and 256 row-major; a layer's score tiles commit `S²/2 + b·S/2` exponentials per head, at 1K and 16K;
    - segments of 16, 16, 16, 16, 16, 16, 15 and 15 layers at `L = 126, S = 8`, every layer once for `L ≤ 130, S ≤ 16`, the rows covered once with each boundary at a tile's first row, and empty segments when `S > L`;
    - `Q' = N` and one row for a sequence shorter than `Q`; each refusal at its rule; each public input that places a tensor moves it when it changes, and `S` moves none.
  - **Heads.** Both validators reject a non-power-of-two query-head group `n_heads / n_kv_heads` (RFC-0047). The independent verifier returns `StatementMismatch { field: Architecture }`; the generic layout refuses such input as `LayoutData`.
  - **Not yet.** Deriving every tile from `ExecutionPlan::expand`, including the additional SoftmaxRow/RsqrtRow committed columns and range tables, depends on #122 and #126. The full claim-list layout and maximum-at-`max_context` validation remain outstanding; #106 stays open.
- Add CPU layer-boundary checkpoints and the witness tile regeneration interface (RFC-0018; [#112](https://github.com/AbdelStark/tessera/issues/112)). Residuals and regenerated tensors use zeroizing buffers, with allocator and redaction tests. Table tiles recount lookup multiplicities including per-axis zero-input padding, broadcast channel classes, masked Exp positions and the combined column's row-0 tail (RFC-0045); independent full-column tests cover every conformance model.


- Verify the folding pass with the independent weight checker on 120 openings of the four conformance models, and reproduce seven independent sparse/scalar/empty opening fixtures. Add cross-row rejection-order and maximal-codeword boundary regressions (RFC-0012/0040/0049; #109).

- `tessera-gadgets::RsqrtRow`: exact multi-limb reciprocal-square-root brackets with bounded product coefficients, the seven-bit input top limb, padded-root zero-check and zeroizing witnesses. Includes independent integer vectors and million-input agreement checks (RFC-0009 Lemma 7, #126).

- Constraint-system digest conformance between the compiler and independent verifier, covering tiny models, variants and 8B/405B shapes. The `constraint-digest` example reports the first differing public constraint, and the named CI job checks it (RFC-0007, #120).

- Specify canonical gadget advice columns and complete auxiliary range-table ordering, with an explicit scalar-reference migration and the original root-bracket polynomial retained (RFC-0050, #407).
- Add independent public model advice-layout vectors for RFC-0011/0047/0050, including complete auxiliary lookup requests and zero rows (#106).
- Preserve complete ordered auxiliary lookup requests in compiler selections and derive plan/witness table sets and exact root report counts from them (RFC-0050, #106).
- Keep inner padded axes' rows in the advice commitment, as RFC-0047 requires, even when their entries are all zero (#106).
- Bind every model advice source and gadget auxiliary to its PCS placement through `tessera-compiler::ModelAdviceLayout`, including weighted images of whole attention claims. Count repeated tiles before expansion; `claim_list` now returns a `Result` and refuses layouts that exceed the row or key capacity at `max_context` (RFC-0011/0046/0047/0050, #106).
- Recalibrate the 405B soundness example with the complete 26-entry catalog, shared Softmax Range counts and the independently checked 22-bit advice row index. Parameter-set values and bound algorithms are unchanged; both sets retain their per-row targets (#106).
- Replay complete conformance-model transcript schedules with the independently derived advice row-index point and the complete catalog's table order (RFC-0050, #106).

- Complete current-model witness tiles and table counts, including every RFC-0050 root/Softmax auxiliary and virtual-top lookup, with canonical scalar root limb counts/order (#112). Wide logical multiplicity columns use storage bounded by public sizes and clearing chunk reads; no secret-dependent sort or row skipping is needed. [CPU allocation measurements](docs/perf/witness-regeneration.md) include checkpoints, public source metadata and replay/gadget scratch. GPU regeneration and the proof driver remain separate work.

- Complete native-model protocol/1 transcript geometry from public claims, including all canonical lookup requests, Softmax forests, gadget zero-checks and their graph dependencies. Independent graph-walk vectors cover all conformance models and sequence/query-block/layer boundaries in both modes; both workspaces replay every step (#125).

- The CPU `StreamingCommitter` commits canonical advice row runs with one initialized secret codeword block and column chain state; rejects misplaced/missing rows and mixed/reordered segment parts; clears blocks on flush and abort; and merges roots without a retained tree. Independent 16-case vectors, randomized batch equality and actual allocator/clearing checks cover S=1/2/8. `AdviceCommitment` is owned by PCS and re-exported at its existing protocol path with unchanged TCE bytes. An allocation-free column-hash entrypoint preserves the existing resident hash path (#115).

- Add independent canonical advice-opening vectors and prover-side cross-checks for complete groups and messages, plus the verifier's model-bound opening API. Conformance counts/descriptors, combination-claim powers, weighted attention images and exact rejection ordering are covered without changing protocol bytes (#116).

- Regenerate grouped advice openings and column answers on CPU: stable original-claim batching, one row block and clearing accumulators, a random-column leaf and combination self-check before combined rows are released, and canonical P5 answers from replayed masks. Public Merkle metadata is retained separately from the root-only committer. Complete model vectors verify through the independent canonical checker, with exact tamper errors, physical clearing and allocator measurements. The shared combination kernel now clears its unreduced secret scratch (OBS-6); no protocol bytes, transcript labels or parameters change (#119).

- Add independent gadget-final fixtures for canonical comparison arrays and narrowed decompositions, consumed by `tv-core` and producer integration tests. The producer's witness columns and lookup forms match the frozen integer and direct-cube oracles (#124).

- Cross-check actual CPU root-product sumchecks and canonical root witnesses against independent coefficient proofs, including padding, constants beyond Goldilocks and a deliberately wrong root whose 119-bit product proof passes but whose complete comparison rejects (#124).

- Freeze independent model-derived requantization sources and lookup rows across single-table, split, wide and four-class cases, with non-power-of-two domains and class-point projections for the independent verifier's binding checks (#124).

- Cross-check actual OneHot witnesses and degree-three CPU batches against independent integer/cube proofs. Each of six alternatives also has a proof adapted to an omitted obligation: true source evaluations pass only that omission control and fail the full independent check (#124).

- Cross-check actual root witness columns and CPU product/padding batches against model-derived independent bindings, including a full integer constant above Goldilocks and a wrong root whose every Range request passes (#124).
- Independent Softmax model-binding fixtures and CPU cross-checks cover causal keys, masked leaves, reciprocal ranges, all NonNeg limbs and padded query/head rows; every round and final evaluation agrees with the independent verifier fixture ([#124](https://github.com/AbdelStark/tessera/issues/124)).
- Independent Argmax fixtures and CPU cross-checks cover lowest-index ties, negative logits, padding, boundary keys and the separate suffix reduction. Actual witness columns, both Argmax stages and canonical teacher-forcing row-copy proofs match the independent verifier fixtures ([#124](https://github.com/AbdelStark/tessera/issues/124)).

- Add exact greedy request scoring in `tessera-scoring`: original teacher-forcing source rows, clearing maximum certificates with zero output padding, per-request `Û = 0` claims, and deterministic `ERR-WITNESS-EXACT-MISMATCH` context after all rows. Proof operands reuse the recorded token's existing OneHot. Actual tiny-model CPU scoring subproofs are checked independently; full model drivers and temperature/mixture scoring remain separate (#127).

- Derive all four OneHot relations in the native stage-one schedule: booleanity, vocabulary padding, row count and the original token index. Proof operands and returned opening claims retain the original indicator/token sources. Independent and CPU proof tests cover isolated alternatives, padded vocabularies, scalar axes and consistent downstream recomputation (#124).
- Exact-greedy request-slot batches (RFC-0071, [#551](https://github.com/AbdelStark/tessera/issues/551)): batch-aware lowering, advice layouts, witness replay, attention reductions and per-request greedy keys; sampled CLI NI/Live proofs, with public omission receipts for failed anchor checks or non-greedy continuations. `Prover::check_witness` checks a candidate within the configured memory budget before any commitment. Independent oracle vectors cover slot selectors and request windows; single-request deterministic proof, claim-list and execution-plan bytes match the pre-batch fixture.


### Changed

- Use bounded exact integer tiles for narrow CPU attention score products, preserving head/batch broadcasting, strided sources, full-trace and wide probability/value products, and whole-result fallback on unsupported kernels (RFC-0005, SC-7, [#621](https://github.com/AbdelStark/tessera/issues/621)).

- Use bounded INT8 tiles and widened integer partials for logits-only servable reference matmuls; preserve the scalar trace and wide/unsupported fallbacks, signed extremes, tails and encoded D artifacts (RFC-0005, SC-7, [#615](https://github.com/AbdelStark/tessera/issues/615)).

- Evaluate only distinct requested SiLU keys in the logits-only integer reference, using the same fixed-point formula and parameter/key checks as full table generation; retain complete cores and digests for trace/conformance consumers (RFC-0008, SC-7, [#613](https://github.com/AbdelStark/tessera/issues/613)).

- Stream quantization's held-out KL scoring by window and retain only per-position divergences; the integer logits-only reference releases layer traces while preserving full-trace conformance consumers, exact rounding and alpha selection (RFC-0005, SC-7, [#611](https://github.com/AbdelStark/tessera/issues/611)).

- Calibration and reference logits transfer BF16 tensors to CPU before widening to float64, avoiding corrupt Metal device/dtype copies; native regression cases cover every finite BF16 pattern (RFC-0005, [#609](https://github.com/AbdelStark/tessera/issues/609)).

- CPU protocol/1 weight commitments retain public Merkle metadata and regenerate column answers in bounded clearing buffers, preserving roots and proof bytes while charging blocks, folds and answers in the host budget (RFC-0012/0040, [#607](https://github.com/AbdelStark/tessera/issues/607)).

- Quantization transforms checkpoint weights one layer at a time instead of retaining whole-model float64 copies; both profiles and scale modes preserve the integer artifact bytes (RFC-0005, [#605](https://github.com/AbdelStark/tessera/issues/605)).

- `tessera_compiler::anchor::sha256` assembles each committed block's message bits as its own `[512, M]` tensor rather than one `[512·N, M]` tensor across the whole message, closing RFC-0072 OQ-0072-2 ([#563](https://github.com/AbdelStark/tessera/issues/563)):
  - **The circuit (RFC-0066 'The message', 'The graph').** For each block `n`, when `K_n = 0` (including trailing FIPS 180-4 padding blocks of a committed message) its sixteen words slice `message` directly and pay the public-bytes block row (`2,945` instances, `6,032` nodes). When `K_n > 0` it slices `bytes_n` from `committed` (unless `K_n = K`), decomposes `bytes_n` with `BitDecompose { 8, 0 }`, gathers 512 rows, multiplies by `place_n` (`Slice` of `place` when `N > 1`, else `place`), `Xor`s with `message_n` (`Slice` of `message` when `N > 1`, else `message`), and slices its sixteen words from that `[512, M]` `Xor`. `Sha256Assembly` and `sha256_with_assembly` retain `WholeMessage` for differential comparison.
  - **Measured trade-off (Apple M4 Max, `--release`, `M ∈ {1, 3}`, `N ∈ {1, 2, 4, 8, 16, 32, 64}`).** `N = 1` (key and sampler key commitments) pays zero extra rows, opening groups, instances or table entries. For `N ≥ 2` every block's 48 word-selection chains run over 512 rows instead of `512·N`, cutting total stage sumcheck table entries by `1.7×` at `N = 2`, `11.4×` at `N = 16` (`8.3×` faster CPU prove: `771 ms` vs `6,430 ms` at `M = 3`) and `43.6×` at `N = 64` (`1.34·10^8` vs `5.85·10^9` entries, `3.12 s` CPU prove), at the cost of `+1` committed row per additional committed block (`369` vs `368`, `+0.27%` at `N = 64`) and `+5` to `+6` bounded opening groups (`17..20` vs `13..14`).
  - **Vectors and tests.** `spec/vectors/anchor-graphs/sha256-graphs.json` and `spec/vectors/ranges/anchor.json` are regenerated from the updated independent oracle, `schedule_anchor` checks the updated stage-by-stage counts (including a trailing padding block paying the public row), and `tessera-prover`'s `sha256` stage tests check differential witness equivalence against `WholeMessage`, non-vacuous rejection of `+256` byte and `Slice(committed)` cheats at stages 1 and 2, and linear table-entry scaling.

- RFC-0013 'Committed tensors' defines `tessera-protocol/2`'s masked extension for a per-layer-scaled committed advice piece, and R10 checks it ([#556](https://github.com/AbdelStark/tessera/issues/556)):
  - **The rule.** A stacked atom `t` on committed advice kind `X_κ` has per-layer coefficients `c_t(0), …, c_t(L − 1)` over the `L` real layers, summed across all paths in the linear node over `Fp2`, with scalar and template `ConstMul` factors reduced in `Fp`. When `c_t(0) = … = c_t(L − 1) = 0`, the atom carries neither data nor mask rows. When `c_t(0) = … = c_t(L − 1) = a ≠ 0`, the atom is uniform and reads `a·V'_{X_κ}(r ‖ ρ)`. When `c_t(ℓ)` differs between real layers, the atom reads `P'_{X_κ}(r ‖ ρ) = D_{X_κ,c_t}(r ‖ ρ) + Z(r ‖ ρ)·R̃_{X_κ}(r_col)`, where `D_{X_κ,c_t}(r ‖ ρ) = Σ_{0 ≤ ℓ < L} eq(ρ, ℓ)·c_t(ℓ)·X̃_{κ,ℓ}(r)`: one advice opening claim in the data's group whose mask row takes `γ_claim·Z(r ‖ ρ)` and `γ_claim·Z(r ‖ ρ)·X` (coefficient 1, RFC-0011 step 2), adding no mask row and no opening group (`G ≤ 20` at 405B). RFC-0068's layer instance reads any linear node's `X'_t(π_t(r_s) ‖ y)` (whether or not RFC-0037 splits the claim; layered claims carry no public terms) as the sum of its substituted atoms' masked extensions at each atom's substituted point `(r_a ‖ y)`, agreeing with `X̃_t` on `{0, 1}^λ`, keeping degree at most 3 in `y` on committed advice atoms, and routing or splitting identically at `ρ'_y` in step 7.
  - **Refusals.** Both workspaces refuse a stacked virtual relation output (`AtomClass::Virtual`) or weight/type atom (`Weight` / `ChannelParams`) whose coefficient carries a template `ConstMul` whose `Fp` constant differs between real layers at any stage (`Layers::Each` or `Layers::Eq`).
  - **R10 (`check-mask-rank`).** `tessera_compiler::schedule::masks` represents a per-layer-scaled advice piece's secret polynomial by its unnormalized `PerLayerScale` (sorted `Expand` `LtFactor` monomials paired with their reduced `Fp` vectors over `0..L`), merges two revealed values at one `(r, ρ)` only when their `PerLayerScale` polynomials agree (`−c_t` or `2·c_t` do not merge, since both put coefficient 1 on `R_{X_κ}`), and refuses an unscaled twin or distinct scaling at the same `(r, ρ)`. R10's gap list in RFC-0013, RFC-0074 and `docs/zk/simulator.md` drops `#556` (now 9 items); `cargo xtask check-mask-rank` self-tests refuse a planted unscaled twin of `tiny-4-servable`'s `2^{r_ℓ}`-scaled `X_in` read.

- RFC-0014's default 405B statement and `tessera-protocol/2` soundness vectors carry the composed schedule's stage, zero-check, runtime greedy lookup/merge, and mask-batching counts ([#566](https://github.com/AbdelStark/tessera/issues/566)):
  - **Default 405B statement (`tessera-protocol/1`).** `StatementSizes::llama_405b_16k()` states the 405B `servable` 16K-token generation batch uniformly across every field: `stages = [123, 98]` (`90` root products in stage 1, `90` `SumSquares` in stage 2; `[122, 97]` for prefill and `[122, 96]` for the model alone), `zero_checks = [42; 14]` (bounding the 14 zero-checks of `Σ n_i = 393`, or 13 of `362` without a request), 27 `lookups` (the commitment report's 26 plus RFC-0053's runtime greedy `Range(29)` at `2^{31}` lookups into `2^{29}` rows), `tree_levels = [55; 42]`, and `groups.merge_claims = 53,491` (`53,489` model split values, the greedy key column and the `SoftmaxRow` forest). Without token groups, the rounds row is `121.05` bits (`FixedBits(30_990)` at `m = 123`, `121.06` at `122`) and the union row `111.83` (`73,168/q`); with `G = 2^{14}`, `rounds` remains `111.90` bits (`P + G = 69,875`) and `union` is `101.72` bits (`80,998,167/q`). `tessera params check --help` states the same defaults.
  - **`tessera-protocol/2` vectors.** Since RFC-0053 holds for `tessera-protocol/1` only (RFC-0013 'Degree'), `405b-protocol-2/{set}` and `edge/protocol-2` in `spec/vectors/soundness/soundness.json` carry the 405B model schedule's 26 tables, `[53; 42]` tree levels, 13 zero-checks (`[42; 13]` in `405b-protocol-2`, exact `Σ n_i = 362` in `edge/protocol-2`), stage 1's 42 rounds at degree 10, the LogUp identity, layer and mask-batching sumchecks (`stages = [122, 96, 26, 36, 2668]`; `Σ count·d = 7,146` in `edge/protocol-2`), and the 20 advice opening groups (RFC-0068's 36-claim group, RFC-0013's `[16, 96]` fold groups and RFC-0070's 1-claim group). The rounds row is governed by RFC-0070's `N = 2,668` mask claims (`116.61` bits); every row meets `live-64/1` and `ni-100/1`.
  - **Checks.** `crates/tessera-bench/tests/commitment_report.rs` verifies the composed stage, zero-check, fold, runtime greedy lookup/merge, column-mask and mask-batching counts on 405B (`servable` and `minimal`), `tv-core`'s independent 405B schedule, and the four conformance claim lists, and checks both calculators' default 405B statements against them. `docs/perf/examples/soundness-{live-64,ni-100}.json` are regenerated to match.

- `tessera_sumcheck::prove_batch` takes the `Protocol` after the transcript, and refuses a degree above `protocol.max_degree(prefix)`; `SumcheckError::Degree` carries that bound, `max`. `MAX_DEGREE`, in `tessera-poly` and `tessera-sumcheck`, is 10, the largest degree of any sumcheck, no longer every sumcheck's bound. `tessera-compiler` depends on `tessera-sumcheck` for the bound ([#565](https://github.com/AbdelStark/tessera/issues/565)).

- RFC-0013 'Committed tensors' fixes a stacked or tiled committed tensor's mask columns ([#554](https://github.com/AbdelStark/tessera/issues/554)):
  - **The rule.** A committed tensor's mask row `R_V` takes its data claim's column sub-point: the low `min(n_P, log2 k)` variables of the tensor the advice layout places, one layer's tensor or one attention tile's share. A stacked kind's layer variables, a tile's KV-head and query-block bits and the variables past `log2 k` are row bits, never mask variables, for `n_P` below, equal to or above `log2 k`. So `V'(p)` stays one claim of the data's opening group, and the mask adds no group. RFC-0011 step 2 masks every committed advice tensor RFC-0013 masks this way: graph tensors, gadget auxiliary columns, multiplicities and the greedy maximum `M`, not mask rows, the pad `S` or blinding rows. The mask term enters once per piece, with `Z` at the piece's full point `(r ‖ ρ)`, never per layer or tile image. The row holds fresh mask-stream values in all `k` entries, as a virtual tensor's does, so every tensor's mask row stays uniform for the proximity row's argument (Lemma 5). RFC-0068's reading, a mask over the stacked tensor's low `min(n_P + λ, log2 k)` variables, is rejected. It put the mask term at another column sub-point, which RFC-0011 cannot open with the data without revealing `Ṽ(p)`: four more column sub-points at 405B's 16K tokens, two in the compiled pool and two among the leaf stage's values, and up to 14 in a conformance claim list's pool.
  - **Hiding.** Two values of one tensor whose points differ only in its layer or tile coordinates share one functional and would leak. R10's check refuses such a pair, and finds none among the values it covers: the pool atoms, the leaf stage's claims on gadget inputs among them, virtual tensors' values and folds, and the weights' claims. RFC-0013 'Committed tensors' now states R10's gap as one list of ten items, the same as `tessera_compiler::schedule::masks` documents: the leaf stage's value forms and partial-axis reads, RFC-0068's column masks and folds, `SoftmaxRow`'s reads and row sums, the natives' raw evaluations on graph tensors and auxiliary columns, multiplicities, the tree levels' masks, #556's per-layer-scaled groups, batches, and the anchor's circuits. RFC-0068 and `docs/zk/simulator.md` R10 refer to it, and it blocks #154 and #159. In `tessera-protocol/2` the prover runs R10's check on the schedule it proves and refuses one with a finding. A stacked or tiled kind's mask holds at most `2^{n_P}` independent functionals across all its layers and tiles below a row: 4 for a per-position kind at `n ≤ 4`.
  - **One site, one point.** `crates/tessera-prover/tests/leaf_read_points.rs` checks that in every conformance claim list's leaf layout one site reads each graph tensor, at one leaf point, sites in and outside the layers together.
  - **RFC-0068's degree** stays `d = 4` when `λ ≥ 1`, derived from `λ` alone, and so do its sizes: at 405B's 16K tokens, 22 rounds, 88 `Fp2` values and a masking polynomial of 222 committed elements, and the calculator's inputs unchanged. Under the rule, a committed tensor never reaches degree 4. The schedule reaches it through virtual tensors, whose folded masks take their layer variables. At 405B those are the two `RsqrtRow` root inputs, not the 376 pairs RFC-0068 counted, and none past 16,384 positions. In the conformance claim lists they are the `RsqrtRow` root inputs at every length, with some `Requant` inputs, virtual matmul, `Rotate` and elementwise-product outputs read directly or through linear maps, at 1 to 5 positions in the lists of one layer variable. RFC-0068's class tensors that do not reach it are seven, of 10, 14 or 16 non-layer variables, where it said five of 10 or 14.
  - **A reduction stage's degree** in `tessera-protocol/2` is #565's ([#565](https://github.com/AbdelStark/tessera/issues/565)), whatever the masks' columns. The rule only lowers a committed stacked factor's degree in its top layer variable to 2, where the rejected reading's columns reached that variable and gave 3.
  - **The documents.** RFC-0013 'Masked extensions' and its testing strategy; RFC-0068 step 4, 'Zero knowledge' and 'Testing Strategy'; RFC-0011 'Opening protocol' step 2 (which committed tensors are masked, the mask row's coefficients in the claim's group, once per piece); RFC-0037 'Claims and atoms' (the layer input `X_in` is one atom with its own mask row, never `c_0` plus a shifted `X_last`) and 'The claim pool'; RFC-0034 'Causal layout' (tiles); RFC-0071 'Soundness and zero knowledge' (batches); the soundness-audit skill's checklist (tile bits and high variables) and the soundness-review skill's zero-knowledge checklist; and `docs/zk/simulator.md` §4.3 (and its layer-input remark: `X_in`'s claims carry its own mask row, not `c_0`'s), §4.4, §7 and R10, with the "Amended by" headers of RFC-0011, RFC-0034, RFC-0037, RFC-0068 and RFC-0071; and the roadmap's dependency of #154 and #159 on R10's list.
  - **`cargo xtask check-mask-rank`** checks the rule alone, `CommittedColumns::RULE`, in place of both readings. Its self-tests plant a value at another layer point of a stacked kind, on tiny-2-servable's layer input and on 405B's `RsqrtRow` outputs, and a value in another attention tile, on tiny-2-servable's `SoftmaxRow` reciprocals. The rule refuses each, and the rejected reading, kept as `CommittedColumns::TensorOrder`, accepts each. `tessera_compiler::schedule::masks::check_schedule` runs R10 on a given schedule of a claim list's lowering.

- The minimum supported Rust version is 1.89, for both workspaces (09 §2, [#534](https://github.com/AbdelStark/tessera/issues/534)): the verifier holds its ledger's directory with `std::fs::File::try_lock`, stable since 1.89. The quality workflow's MSRV job checks 1.89.

- A reduction stage's round degree in `tessera-protocol/2`, specified ([#565](https://github.com/AbdelStark/tessera/issues/565); the code follows in a separate change):
  - **The defect.** The specification bounded a stage's rounds at degree 7, but a relation reads each committed or virtual factor through its masked extension, of degree 3 in the round that binds its last variable. RFC-0050's root coefficient products, `eq` times three committed digits, then reach degree 10 in stage 1, whatever the masks' columns: an honest stage-1 round did not fit a degree-7 message, and a masking polynomial sized at 7 would leave its top coefficients unmasked (simulator R4). RFC-0066's `Xor3` and `Maj` reach 10 too.
  - **The rule.** RFC-0013 'Masked extensions' 'Degree' gives an instance's degree from its relation alone, term by term: the most `d_1 + 2s` over its product terms. `d_1` counts a term's factors: one `eq` factor, which takes in a stacked instance's layer weights and any public factor an RFC writes as part of `eq`, then its public factors as the table counts them, then its reads, a linear combination of one tensor's values being one read. `Rotate` and `Selection` count their kernel as their `eq` factor, a stacked `Rotate`'s layer weights included. `s` counts its reads of committed or virtual tensors. A public operand counts in `d_1` alone, and a fold has 2. The rule's premises are stated: a multilinear `eq` factor and public factors, a product counted as one factor being one only on disjoint variables or as one multilinear extension; no diagonal reads; and a combination's coefficients off its points' variables. A relation outside them states its own degree. The degrees other RFCs give are `tessera-protocol/1`'s.
  - **The table is normative**, keyed by what the schedule decides: the relation, the native kind, the `BoolFn`, the public operands, an equality zero-check's operation and `ConstMul` constant, and, for the matmuls alone, the stacking. An instance that matches no row is refused, and a row that names no public operand holds only when none is public. The table keys RFC-0010's reduction stages from index 1; RFC-0068's layer sumcheck and the tree levels keep their own degrees. The keys are defined: a native's kind is the one its native plan names for its tag, never its `tessera-protocol/1` descriptor, which `Key`, `Forest` and `Reciprocal` share, and the code change carries it in the schedule's instance; stacked means layer weights over `λ ≥ 1` coordinates, a `MatMulAct` over two heads or more, or a batch's requests once they are lowered; a `ConstMul` at a template position has a constant per layer whenever `λ ≥ 1`, whatever its values; and the operands are those the relation's final message lists, `Rotate`'s `X` alone.
  - **Its rows.** It gives every relation and native relation its `relation` string, `d_1`, `s` and `d`, with rows for public operands (`Ch`, `Xor`, `And`, `Xor3`, `Maj`, `Mul`, `AssertEqual`, `Recomposition`, `OneHotIndex`) and every `Equality` the compiler derives: of a scalar `ConstMul`, `ConstAdd`, `Add` or `Sub`, whatever inputs are public, all of them included, or none, 4; of a `ConstMul` by a constant per channel or per layer, 5, or 4 with its input public; of a `Mul`, 7, 5 or 4. A column marks the rows tight on their tiny instance. `Recomposition`, `OneHotIndex` and RFC-0053's tokens and greedy keys overcount on every shape, reaching 4 of their 5, RFC-0053's selected logits 5 of 7, and an observed request's zero summand 0 of a conventional 3: they keep the rule's count. RFC-0053's rows are provisional, as RFC-0053 has no `tessera-protocol/2` form.
  - **Which round rises.** A masked read goes live in the round that binds its own last variable: the last round for a read that holds its instance's top variable, a `Trailing` broadcast operand's included, and an earlier one for one that does not, such as a `Leading` broadcast operand. Simulator R4 and RFC-0013's cost row now say so.
  - **The invariant.** No row exceeds 10. The compiler and the verifier check `d ≤ 10` when they derive a stage's degree in `tessera-protocol/2`, and `d ≤ 7` in `tessera-protocol/1`. The sumcheck engines refuse a batch past their protocol's bound, of degree 11 or 8.
  - **The figures.** Every Llama schedule's stage 1 has `d = 10`, its later reduction stages 7, at 405B and in every conformance claim list. RFC-0070's coefficient block holds at most 8,132 coefficients, not 8,006, still within its 8,192 slots, so `M`'s 780 rows are unchanged. A round of degree 10 keeps 124.68 bits.
  - **The tests.** RFC-0013 'Testing Strategy' specifies:
    - a per-row degree test in both workspaces, which evaluates each row's round polynomials on a specified tiny masked instance at more than `d + 1` nodes: the independent check. It runs every relation, public pattern and constant class, and the matmul row unstacked and stacked over a layer or over heads. It asserts `d` exactly on the tight rows, which its mutation control lowers, and `tv-core` writes the summands of the relations it does not derive yet;
    - a test of the no-row refusal;
    - `spec/vectors/schedule/degrees.json` with its fields, pinned in RFC-0013: `description` and 44 `cases` in a stated order, `model` the claim list's `name`, `length` `max(n, 3)`, `rounds` with `log2 k` when the stage has a fold, `relation` the table's string, `public` per operand, `stacked` layer weights over `λ ≥ 1` or a `MatMulAct` over two heads and not `schedule.json`'s field, `folds` one per virtual part's instance, a combined selection's count left to #145 before the circuits' cases join. It holds the composed schedule `tessera-protocol/2` proves (`derive_composed_schedule` with `model_natives`) for the conformance claim lists at `n` of 1, 4, 5 and `max_context`, requests `none` and `prefill` at every `n` and `generate` at `n ≥ 2`. `gen-vectors` writes it and `tv-core` compares. SHA-256's and AES-128-CTR's schedules join once `tv-core` mirrors their relations (#150, #159);
    - a fabricated degree-11 instance refused by the stage-degree derivation, and one of degree 8 in `tessera-protocol/1`.
  - **The documents.** RFC-0013 'Masked sumcheck' step 1, 'Masked extensions' 'Degree' and 'Testing Strategy' (degrees 1–10, the per-row test, the no-row refusal, the vector, a degree past the protocol's bound refused by the sumcheck engines); RFC-0010 'Sumcheck messages'; RFC-0014's soundness budget and its default statement's protocol/2 use; RFC-0018's round kernels; RFC-0069's degree; RFC-0070's coefficient block; RFC-0072's degree statements, now scoped in its header, and its 'Messages', a stage of `Xor3` or `Maj` at 10 only when one has no public input; `docs/zk/simulator.md` R4; `docs/testing/soundness-calibration.md`; 07 §3, with `tessera-protocol/1` refusing degree 8, and 08 §5; `research/baseline.md`; `spec/vectors/sumcheck/` and `spec/vectors/poly/`'s READMEs; and an erratum line in EXP-0027.
  - **The code** caps the degree at 7 (`tessera-poly`'s `RoundPoly`, `tessera-sumcheck`'s `MAX_DEGREE`, `tv-core`'s round checks, `verifier/fuzz/seeds.py`'s `SUMCHECK_MAX_DEGREE`): #565's code change raises it to 10 in `tessera-protocol/2` and refuses 11, keeps 7 in `tessera-protocol/1`, before #148, #154 and #159 build `tessera-protocol/2`'s stages. The native descriptors' `(1..=7)` checks (the compiler's `schedule/mod.rs`, `tv-core`'s `schedule.rs`) bound `Native.degree`, protocol/1's, and stay.

- RFC-0013 'Virtual tensors' specifies the mask fold of a virtual tensor in a reduction stage ([#538](https://github.com/AbdelStark/tessera/issues/538)):
  - **An instance of its own.** The relation's instance may not run over the mask's variables, as `MatMulWeight`, `MatMulAct` and `SumSquares` do not. The fold takes its relation's power of `γ`, and its summand is scaled by `2^{log2 k − n_i}`, so that the two instances contribute RFC-0010's scaled claim of the `V'(p)` the verifier knows.
  - **Over `log2 k` variables,** reading the row at `(p_col ‖ 0)`, whatever the mask's width. So every fold of a stage ends at one point, and adds at most one opening group per reduction stage: two at 405B. The stage's rounds count the fold's variables.
  - **Its final value** `u` is sent after the instances' evaluations, and is a claim on the mask row in the advice pool. The opening is what binds the fold.
  - **The whole row is random.** A virtual tensor's mask row holds fresh mask-stream values in all `k` entries, past `2^c` too, so the full-width `u` stays independent of the mask's values at points of fewer coordinates: a new requirement on the prover's mask stream.
  - The protocol/2 figures of RFC-0011, RFC-0013, RFC-0019, RFC-0057 and 08 §5 count the fold groups: `G + 1 ≤ 20` combined rows, about 2.87 GB of proof before #489.

- The soundness calculator counts stage 1's layer point ([#537](https://github.com/AbdelStark/tessera/issues/537), RFC-0014):
  - **The gap.** RFC-0037 'The layer point': a false per-layer value survives its stacking at `ρ_1` with probability at most `λ/|Fp2|`. Neither the rounds row nor the union row counted it, so an exact statement's report left out `λ/q`.
  - **The input.** `StatementSizes.layer_vars`, the layer point's `λ` coordinates. The rounds row's `D` takes it as an item and the union row adds it. A calculator refuses, with `ERR-USAGE`, a `layer_vars` other than 0 in `tessera-protocol/2`, whose layer sumcheck is a stage (RFC-0068), or above 32.
  - **A proof's budget** (`tv-core`'s runtime) takes `λ` from the coordinates its own schedule's stage 1 squeezes.
  - **The 405B statement** gives `λ = 7`. Its union numerator grows from 79,621,474 to 79,621,481, and its bits stay the same.
  - **`tessera.soundness/1`** requires `layer_vars` in its statement: a breaking change for readers of the report. The two example reports are regenerated.
  - **The oracle** draws `layer_vars` last in its random statements and adds `edge/layer-point`, whose rows are `7/q` alone. Both calculators reproduce all 93 vectors.

- The soundness calculator counts each weight matrix's own blinding rows in `tessera-protocol/2` ([#521](https://github.com/AbdelStark/tessera/issues/521), RFC-0065):
  - **The input.** RFC-0014's `matrices` gives each matrix's claims `e_T`, and the calculator derives `B_T` from them: `B_W` for `Embed`..`LmHead`, and for `ChannelParams` `B_W + 2·(e_CP − 1)·⌊max(B_W − 4, 0)/6⌋`. It never takes `B_T` as an input.
  - **The refusals** (`ERR-USAGE`, protocol/2): a matrix with no claim, which is not opened, and a matrix other than `ChannelParams` with `e_T ≠ 1`.
  - **The 405B statement.** It gives `ChannelParams` 7 claims, so under protocol/2 its `B_CP` is 12,280 and its `ϑ` 14: 102.19 bits, where one `B_W` gave 13 and 102.30.
  - **`MatrixSizes`** gains `claims` and `blinding_rows`. The `tessera.soundness/1` report gives each matrix's `claims`, a field its schema now requires: a breaking change for its readers.
  - **The oracle** (`spec/vectors/soundness/`) derives `B_T` the same way. It adds three edge cases: two pin `B_CP` exactly at a power of two, and one has fewer than four blinding rows. `tv-core`'s calculator (the verifier's changelog) agrees on all 92 cases and on 1,000 random profiles.
  - **`max(B_W − 4, 0)`** in RFC-0014 and RFC-0065 keeps `N_0` a count for a set of fewer than four blinding rows. Floored division would have given −1, and an implementation could read it either way.

- The soundness calculator's weight-linking row counts the link's combining challenge ([#516](https://github.com/AbdelStark/tessera/issues/516)). RFC-0014's row is now `max (ν_T + 1)/q`, the bound RFC-0064 'Soundness' states: `ν_T/q` for two weight polynomials that agree at `ρ_T`, and `1/q` for `θ_T`, which is drawn after the blinding claim. `tessera-protocol`'s calculator and the soundness oracle follow it, and `spec/vectors/soundness/` and `docs/perf/examples/soundness-*.json` are regenerated. At 405B the row goes from 37 to 38 over `q`, 122.75 bits instead of 122.78.

- Long-lived keys are held on the heap ([#495](https://github.com/AbdelStark/tessera/issues/495), OBS-2):
  - **`tessera_codec::HeapKey<N>`.** `N` secret bytes in a box, cleared on drop even outside a `Secret`, with `zeroed`, `from_box`, `generate` and the `Borrow` traits. It moves from `tessera-anchor`, which re-exports it.
  - **`tessera_scoring::SamplerKey`** holds `Secret<HeapKey>`, and its `secret()` lends the key to the session-open frame's commitment. `SamplerKey::new` now takes the key boxed.
  - **`Sampler`** boxes its AES key schedule, whose first round key is the key itself.

  A `Secret<[u8; 16]>` that moved could leave its bytes behind. OBS-2 now says which keys go on the heap.

- The nightly runs the quality workflow's deferred gates every night ([#492](https://github.com/AbdelStark/tessera/issues/492)). Pull requests still skip them during the fast-build phase (07 §9). `quality.yml` is now also a reusable workflow, which `nightly.yml` calls. A failure in one of its gates opens the nightly's failure issue. The gates are:
  - the dependency policy;
  - MSRV;
  - the macOS build;
  - the GPU host checks;
  - the complete suites;
  - rustdoc.

  Until now they ran only on dispatch. The first dispatch since the token-group PRs, run 40074495804, found `tv-core`'s rustdoc failing on `main`: a link to the private `levels` and a redundant explicit link target, now fixed. `docs/testing/nightly.md` and 07 §9 say what the nightly runs.

- RFC-0057, accepted, proves a batch in one proof unit ([#456](https://github.com/AbdelStark/tessera/issues/456)):
  - **The question.** RFC-0018 split a batch into proof units, sets of requests or layer ranges of one request, each with its own advice commitment, and linked a split request's boundary residuals. No document gave such a proof's messages, labels or checks. #456 asked which budgets still need units.
  - **The answer.** None at the reference statement (405B, 16K tokens at 16K of context, `tessera-protocol/1`).
    - **Memory.** Token groups, windows and RFC-0056's working sets meet every budget. The checkpoints, 136 GB, fit host RAM.
    - **What units cost.** Each adds up to 1.14 GB of combined rows and a table side, and a set of requests adds a weight fold.
    - **What units could not do.** Layer ranges small enough for 405B's largest reduction stage would number at least 33 to 60. Their combined rows alone, 38 to 68 GB, break the 4 GiB gate nine to sixteen times.
  - **The decision.**
    - A proof is one unit holding whole requests, as RFC-0036, RFC-0014 and RFC-0019 describe it. RFC-0031's `unit` is always 0.
    - A batch too large for one proof is several proofs, which nothing links.
    - The planner chooses token groups and windows, not units.
    - No message, label, check or vector changes.
  - **The limits it states.**
    - **Memory.** A whole 405B request fits one proof up to 64K tokens at the cost model's 0.2 TB (`G = 2^{14}`), and about 32K if the multiplicity counters and the keys and values stay in HBM.
    - **Proof size.** Within the gate, applied per proof, it fits up to about 24K tokens: `ModelAdviceLayout`'s rows give 2.26 GB of opened columns at 24,576 tokens.
    - **The verifier.** Its 8 GiB message limit refuses a non-interactive proof of a whole request from about 46K tokens.
  - **What it leaves open.** The corpus's 128K projections model a 16K-token chunk of a longer request whose earlier keys and values are cached. No document binds those to the tokens before them, so the 128K figures assume a statement that does not yet exist. [#470](https://github.com/AbdelStark/tessera/issues/470) and OQ-0057-1 own it. `tessera-protocol/2` keeps `g = 0` (OQ-0054-1), about 90 tokens per 405B proof at 1K.
  - **The documents.**
    - RFC-0018's sections that named units, RFC-0031, RFC-0049, RFC-0054's Alternative 3 and RFC-0056's batch form follow.
    - 08 §3 notes the 128K assumption. 08 §5's opened columns become RFC-0011's 1.18 GB, so the proof is about 2.8 GB: within the gate, above the 2 GiB target.
    - `SPEC.md` lists the RFC and OQ-0057-1, and drops OQ-0018-1.
    - #132's unit criterion is withdrawn, and IDEA-0027 is killed.
  - **Review.** An adversarial review recomputed every figure. It found the long-request and `/2` limits that a first draft's claim, that every memory budget was met, had missed.

- RFC-0056, accepted, counts the prover's native relations in the reduction stages and proves `SoftmaxRow`'s score instances without the score tensor ([#458](https://github.com/AbdelStark/tessera/issues/458)):
  - **The gap.** The CPU prover composes native relations into stage 1, and the compiler's schedule held none of them, so RFC-0055 sized the stages without them:
    - the roots' coefficient products and padding checks;
    - `SoftmaxRow`'s key, forest and reciprocal relations, whose key and forest relations span the score tensor: 8.8 TB at 1K context and 141 TB at 16K for a 405B batch of 16K tokens;
    - a request's token relation and greedy certificate (RFC-0053), 1.17 TB of stage 1 at 16K.
  - **Duplicates.** Each root's thirty coefficient products left thirty identical `SumSquares` instances in stage 2, 38.6 TB in `minimal`.
  - **The score instances, without the score tensor.**
    - **How.** The rounds over the key bits read the INT8 queries, the keys folded over their bound bits block by block, prefix sums over the key blocks and the causal mask's closed form. The rest run on row tables.
    - **The cost.** About 45 GB of residents for 16K tokens, 60 GB at 128K of context. The messages are unchanged.
    - **The evidence.** EXP-0026, pre-registered, checked every round polynomial against a literal prover exactly on 336 runs and caught 9 of 9 planted faults (IDEA-0039, validated).
  - **Shared tables.** Instances of one relation of one tensor at one point hold one set of tables, weighted by the sum of their batching coefficients; a root's coefficient products share theirs. The messages are unchanged.
  - **The windows.** A stage's rounds take one window per eight own variables they bind of an instance, each a regeneration: at the grid's estimated cost, a second window pays only beyond eight at 405B (`MAX_WINDOW_VARS`, a calibration).
  - **The report.**
    - **The natives.** `tessera_compiler::schedule::model_natives(lowering, request)` derives the native relations from the lowering, a request's included. The prover's complete-proof tests hold them, and the composed schedule built from them, equal to `ModelNativePlan`'s for the model alone and each kind of request.
    - **The classes.** `stage_tables` reads the composed schedule with a request with outputs. Each class has a growth (`fixed`, `tokens`, `keys`, `tokens_and_context`), a holding (`windowable`, `resident`, `computed`), a width (`bytes`) and tables per bit of a request (`count_per_bit`), and each stage lists its score instances' `residents`. The stage's entry totals are gone: classes of several growths do not sum to one figure.
  - **The cost model.** `stage_plan` takes the context, grows each class by its growth, sizes the held tables and the residents (`ReductionStage::residents`) at their widths, lets the score instances, but no resident, set the stage's variables, and splits wide windows (`own_vars`, `windows`, `MAX_WINDOW_VARS`). `μ` counts a regeneration and the folds per window. The `tessera.cost-model/1` report states each class's width and each stage's residents.
  - **The projections.** RISK-2, RISK-9, RISK-10, SC-2, 08 §1, §3 and §4, `SPEC.md`, `docs/perf/v0.2/`, RFC-0055, RFC-0054, RFC-0020 and RFC-0018 restate them.
    - **`minimal`.** About 13.8 and 29.1, unchanged: its stages take a window each, of up to eight own variables. The windows' arithmetic adds 0.42 to 1.20 at 1K.
    - **`servable`.** About 15.7 and 31.0, up from 13.7 and 29.0: the roots' products push its stage 2 past memory, and its stage 1 takes two windows. The windows' arithmetic adds 0.13 to 0.47 at 1K.
    - **The batch.** At 2,048 tokens, `servable` would cost 14.3 and 29.5 (OQ-0056-2, [#357](https://github.com/AbdelStark/tessera/issues/357)).
  - **Follow-up.** [#461](https://github.com/AbdelStark/tessera/issues/461) builds the structured score rounds and shared tables in the prover; [#459](https://github.com/AbdelStark/tessera/issues/459) the windows.

- RFC-0055, accepted, proves a reduction stage larger than the memory budget in windowed rounds ([#452](https://github.com/AbdelStark/tessera/issues/452)):
  - **The working sets.** The compiler's schedule gives each reduction stage's tables. At 405B and 16K tokens, `minimal`'s stages 1 and 2 hold 4.1 and 5.5 TB and `servable`'s stage 1 9.6 TB, against 0.2 TB, at every context. Streamed round by round, as RFC-0010 had it, they would cost 12 and 8 regenerations, which no projection counted.
  - **The design.** One pass accumulates each instance's products on a grid over the stage's first `c` rounds, and the rounds read their polynomials from it. A second pass folds the tables at those rounds' challenges, after which they fit. The messages, the verifier and the soundness bound are unchanged. Each such stage costs one more regeneration, counted as P3b (`tessera_passes_total{pass="p3b"}`).
  - **The report.** `tessera.commitment-report/1` lists each reduction stage's tables (`stage_tables`): count, size and whether they grow with the batch. The compiler derives them from the schedules of two lengths and refuses a graph whose tables are not linear in the tokens. Tests hold them to the schedule at every batch, 405B's at 16,384 positions included.
  - **The cost model.** `tessera_bench::stage_plan` sizes each stage at the batch and its window, counting each table's front padding and folding no matrix product. That gives windows of 6, 6 and 8 rounds at 405B where totals alone would give 5, 5 and 6. The regeneration term adds the fold passes, and the streaming term prices the folds at the new unit cost `window_ps`. Both report schemas gain the stages' tables and plans, and `tessera bench model` a `windows` column.
  - **The projections.** RISK-9, RISK-10, SC-2, 08 §3, `SPEC.md` and `docs/perf/v0.2/` restate them.
    - **Corrected `μ(1K)` and `μ(16K)`.** About 13.8 and 29.1 for `minimal`, and 13.7 and 29.0 for `servable`.
    - **SC-2.** Its 1K ceiling of 12 now fails at the corrected projection for both profiles, by 15% and 14%. The 16K ceiling holds.
    - **The windows' arithmetic.** It adds 0.05 to 0.42 to `μ(1K)` for `minimal`, and 0.27 to 0.90 for `servable`, whose window binds eight variables of `a·u`. The bands span SUR-0007's operation count and EXP-0012's CPU measurement, which disagree by about twice; v0.5 measures it (OQ-0055-1).
  - **Not yet counted.** The prover's native gadget relations in stage 1, which the compiler's schedule does not hold. `SoftmaxRow`'s key and forest relations span the score tensor, 8.8 TB at 1K context and 141 TB at 16K, beyond what windows serve economically ([#458](https://github.com/AbdelStark/tessera/issues/458)). The windowed prover itself is [#459](https://github.com/AbdelStark/tessera/issues/459).

- The CPU prover plans its passes before P1 and counts them as they run ([#132](https://github.com/AbdelStark/tessera/issues/132), RFC-0018 'Pass structure', 'Memory scheduler'):
  - **The plan.** `Prover::plan` returns a `PassPlan`: the passes in order (P1, P2, one P3 per reduction stage, P4, P5), the resident cells, and the resident bytes.
  - **The charge.** 64 bytes per cell and the weight commitment's own bytes, replacing a stand-in that charged 4,096 bytes per cell. That stand-in was about 120 times the measured peak, so a host had to budget about 12 GB for a 100 MB proof. A counting-allocator test holds each conformance proof's measured peak within the plan and above a quarter of it.
  - **The refusal.** A budget below the plan fails with `ERR-RES-MEMORY` before P1.
  - **The counters.** `tessera_passes_total{pass}` and `tessera_pass_seconds{pass}` (05 §4) count and time every pass, and the run manifest records them.
  - **Not yet.** Proof units of several layer ranges remain unspecified (#456).

- RFC-0054, accepted, proves a leaf stage larger than the memory budget in token groups ([#329](https://github.com/AbdelStark/tessera/issues/329)):
  - **The accounting it replaces.** RFC-0018 streamed every level of every tree in one batch with shared challenges. Counted for every tree and every level, that is 36 to 55 regeneration passes for a 405B 16K-token batch, +53 to +235 on `μ`. RISK-10 had counted the attention trees' leaf level alone.
  - **The design.** Each group's lookup trees and product forests are proven in memory with challenges of their own. The table trees stream in a batch of their own and regenerate nothing. One merge sumcheck brings the groups' leaf claims back to one point per tensor, so the reduction stages and openings are unchanged. The stage costs two regenerations at any size. `g = 0` is the current leaf stage.
  - **Groups.** They are interleaved token subsets, at least 256 tokens each so that their GEMMs stay compute-bound, and interleaved layer subsets.
  - **Tier-A changes, for `tessera-protocol/1`.**
    - `Round1` gains `leaf_groups`; `TreeProof` gains the groups, the table roots and the merge.
    - New labels: `group/{h}/…`, `tables/…` and `merge/…`.
    - `TreeLevel` gains `group`, and `LeafMerge` is a new rejection.
    - The union bound keeps 103 bits at the most groups a verifier accepts, 2^14.
  - **The cost model.** `tessera_bench::leaf_plan` gives a batch's groups and table rounds, and `shared_streaming_passes` the replaced design. The regeneration term adds the merge's pass. The streaming term prices the merge's folds (new unit cost `merge_ps`) and the table trees' rounds.
  - **The reports.** `tessera.cost-model/1` and `tessera.commitment-report/1` replace `streaming_passes` with `leaf_groups` and `table_rounds`, and add `merge_ps`.
  - **The projections.** RISK-9, RISK-10, SC-2, 08 §3 and `SPEC.md` restate them. For `minimal`, the corrected `μ(1K)` and `μ(16K)` are about 11.8 and 27.1. The merge's folds add 0.16 to 0.32 and 0.82 to 1.63.
  - **Not yet priced.** The reduction stages exceed memory too ([#452](https://github.com/AbdelStark/tessera/issues/452)); RFC-0055 prices them.

- Tree levels sum their terminal products unreduced and reduce once per block (EXP-0013; [#391](https://github.com/AbdelStark/tessera/issues/391)). `tessera_field::Wide192` holds a 192-bit sum of Goldilocks products, `WideExt2` the four schoolbook partial products of an extension product, and `PolyField::Wide` chooses the sum per field; prime fields, the calibration field included, keep reduced sums. A fraction node is weighted as `w·A + (w·λ·D(0))·D(1)`, so both of its outer products are deferred. A block larger than a wide sum's bound (`2^62` pairs) is reduced per partition. Proofs are byte-identical. On one Apple M4 Max core, RFC-0020's protocol measured `prove_forests` 20–21% faster per leaf on the three 2^22-leaf shapes and `prove_trees` 25–31% faster on 2^21-leaf fraction trees. The old kernels stay as the test oracle. `cargo xtask check-branchless` probes the wide sums too, and `tessera-sumcheck` has a `levels` benchmark.

- RFC-0023, accepted, chooses the norms' precision `κ` per model rather than fixing it at 16 ([#287](https://github.com/AbdelStark/tessera/issues/287)):
  - `κ` is the largest value, at most 31, with `⌊d · 2^{2κ} / e⌋ < 2^64`: RFC-0022's range check on the root stays one 32-bit lookup.
  - The root `ρ ≈ 2^κ / rms` keeps about `κ − log2(rms)` bits. At `κ = 16`, a row of `rms = 2^{12}` had a 6% scale error and a saturated row a root of 0. At 8B's `κ = 30`, every row INT20 allows keeps at least 11 bits.
  - No gadget, table or schema changes, and the cost per token is unchanged. `tessera_quant.integer_model.norm_kappa` computes the rule for the quantizer, and a property test checks it.
  - Llama-3.1 8B's and 405B's shapes lower at `κ = 30`, every root's range the full 32 bits.
- RFC-0022, accepted, makes `D`'s residual stream saturate, and bounds its norms' roots over all inputs ([#284](https://github.com/AbdelStark/tessera/issues/284), [#285](https://github.com/AbdelStark/tessera/issues/285)):
  - Each residual update clamps the sum, `res ← clamp(res + ⌊(acc + 2^{r−1}) / 2^r⌋, INT20)`, instead of adding a clamped update. RFC-0005's unsaturated sum reached `2^62` in 8B's norms and passed `P` in 405B's.
  - A residual site of one class proves the sum with the `Requant` it already had, through the key `acc + 2^r · res`: no commitment more. A site of several classes adds one `Requant`, which clamps `2(res + u)`.
  - Wherever the residual stays inside INT20, `D`'s values are unchanged, and no input is left without a proof.
  - `RsqrtRow`'s range check widens from `2^{κ+1}` to the bit length of the largest root any row can have, `⌊√(d · 2^{2κ} / e)⌋`. It replaces #78's refusal of models with `4e ≤ d`, which a residual scale above `2^{−8}` would have met at 8B.
  - The compiler lowers Llama-3.1 8B's shape through all 32 layers, and 405B's through all 126, with every norm's sum at most `d · 2^38` and every root inside its 18-bit check. The residual's width is now at most 31 bits, since a classed site's update clamps within `±(2^{bits} − 1)`, a table value.
- The IR and its range analysis, refined for the lowering ([#78](https://github.com/AbdelStark/tessera/issues/78)):
  - `Rotate` pairs `(x_i, x_{i+m})` of a last axis of `2m`: Llama's `rotate_half` layout, the order of Hugging Face checkpoints. RFC-0005 and RFC-0006 had not said which channels pair.
  - `RsqrtRow` carries its row length `dim`, so `N = dim · 2^{2κ}` no longer depends on the node that produced its input, and requires `κ ≤ 31`.
  - `SoftmaxRow`'s scores must be square in their last two axes, the query and key positions of one sequence. `TableMap` takes the channel classes of a `Silu` table of several classes as a second input. Range analysis requires them exactly when the table has several classes, and bounds them by the class count.
  - The scores' bound is `|x| ≤ 2^21`: INT8 heads of width 128 reach `2^21` at `q = k = −128`. The strict bound that RFC-0005, RFC-0006 and RFC-0009 stated would have refused every Llama model.
  - `OneHot` emits a second relational fact: a one-hot vector selects a column, so the embedding rows keep the weight's range rather than `127 · vocab`.
  - `RsqrtRow` requires `d < 4(u + e)`, the condition under which the honest root passes the gadget's range check (INV-DM-2). With `u` able to reach 0, that is `4e > d`; [#285](https://github.com/AbdelStark/tessera/issues/285) tracks the gadget change that would lift it.
  - RFC-0005 states S1 and S2 for the weight matmuls, the GEMMs on tensor cores: `PV` multiplies 17-bit exponentials in both profiles. `ERR-QUANT-ARCH` now also covers an integer model inconsistent with its architecture, which the compiler raises.
- `IntegerModel`'s `QuantSpec` records each layer's SiLU input exponent, `silu_in_exponents`, in Rust, Python and 03 §2 ([#282](https://github.com/AbdelStark/tessera/issues/282)). The compiler needs it to emit the layer's `Silu` table. The quantizer chooses it with the hidden weight exponents, so no public field gives it, though the claim list states it in the table's specification. 03 §2 now says that a weight's rows are its output channels, so `Embed` holds the transposed embedding, `d_model × vocab`, as RFC-0012 lays it out. The model vectors held it the other way, and are regenerated.
- `tessera_quant.RopeSpec` takes the integer fields of 03 §2, in place of fractions: `theta`, `factor`, `low_freq_factor`, `high_freq_factor`, `original_max` and `bits`. `ArchSpec.family` is the enum `ArchFamily`. `read_config` refuses, with `ERR-QUANT-ARCH`, a RoPE parameter that is not an integer of its field's width, and a size of 2^32 or more ([#74](https://github.com/AbdelStark/tessera/issues/74)).
- The minimum supported Rust version is 1.88, up from 1.85 (09 §2): the field needs `core::hint::select_unpredictable`. The `msrv` CI job, renamed from `msrv (1.85)`, checks both workspaces with 1.88.0 ([#242](https://github.com/AbdelStark/tessera/issues/242)).
- The proximity test's field term carries its tensor generator's factor ([#254](https://github.com/AbdelStark/tessera/issues/254)):
  - **The correction.** RFC-0011 draws the proximity coefficients as `eq(ξ*, row)`, a tensor over the row index's `ϑ` bits, so the false-witness probability is `ϑ·n/|Fp2|` (Diamond–Posen; Diamond–Gruen, Theorems 2 and 3), not the `n/|Fp2|` of independent coefficients.
  - **At 405B.** The term is about `2^{−100.6}` for advice (`ϑ` up to 21) and at most `2^{−101.8}` per weight matrix (`ϑ_T` up to 18). `live-64` keeps about 100 bits one shot, `ni-100` about 120 with the seed's grinding: no parameter changes.
  - **Where it lands.** RFC-0011's soundness and RFC-0014's budget carry the term. RFC-0012 names the weights' generator, `eq(ξ*_T, row)` under `pcs/weights/{T}/xi`. The calculators take each commitment's `ϑ`, and RFC-0014's Johnson-bound alternative notes that no list-decoding bound is known for this generator.
- Tree levels prove about 1.76× faster per leaf, with the same proofs ([#258](https://github.com/AbdelStark/tessera/issues/258)):
  - **The kernel.** `prove_trees`, and so `prove_forests` and `prove_logup`, proves each level by a split `eq` and Gruen's claim recovery (Dao and Thaler, ePrint 2024/1210; Gruen, ePrint 2024/108, §3.2), in `tessera-sumcheck`'s `level` module.
    - It builds no `eq` table: a scalar, the current variable's line and two tables of about `√N` entries, rebuilt each round.
    - It takes two node sums per round, recovering the third from each instance's claim. It sums the third when a public challenge zeroes the recovery's divisor.
  - **The proofs** are bit-identical to the batch prover's over `[eq, …]` instances, which stays as the reference its tests compare to: products on 0 to 12 variables, fraction trees with every kind of numerator, front-padded batches, and the three degenerate challenges.
  - **Measured.** RFC-0020's protocol on one Apple M4 Max core: `trees/forest`, a new microbenchmark of a product forest of `2^22` leaves, goes from 32.96 to 18.70 ns per leaf; `logup/prove` from 119.1 to 86.8 ns per lookup. RFC-0018 'Kernels' records the level kernel.
- The documents' streamed-pass counts are the cost model's, at every context ([#250](https://github.com/AbdelStark/tessera/issues/250)):
  - **The counts.** RFC-0018 'Leaf levels' said `c ≈ 8` at 16K. Its own formula gives `⌈log2 346⌉ = 9` for the 16K-token batch's 2.2×10^12 attention leaves on 0.2 TB, as `streaming_passes` does. It gives 7 at 4K and 5 at 1K, where no document had priced streaming.
  - **The cost.** At RFC-0018's 6 to 8 ps per score, the passes add 0.36 to 0.48 to `μ` at 1K, 2.0 to 2.7 at 4K and 9.6 to 12.8 at 16K: 32–42%, 45–59% and 57–76% of the attention term. RISK-10 had said 30–70% at 16K, and 08 §3 up to about 12.
  - **The ceilings.** Take the upper estimate, with the column answers priced as committing. `minimal` then reaches about 13.0 at 1K and 45.6 at 16K, above both of SC-2's ceilings, and `servable` 14.3 and 46.8. Streaming alone takes `servable` to its 1K ceiling. RISK-10, SC-2, 08 §3, RFC-0018, RFC-0020 and `SPEC.md` restate these figures. The projection tests now take the streaming term from the model, where they added 12.
  - **What stays out.** These counts cover the attention trees' leaf level only. The leaf stage batches every tree, its levels above the leaves stream too, and a pass over the linear layers' lookups regenerates their GEMMs. Under RFC-0018's current rules that is an order of magnitude more, which [#329](https://github.com/AbdelStark/tessera/issues/329) is to price and redesign.
- RFC-0012 specifies how the prover answers the weight column queries ([#251](https://github.com/AbdelStark/tessera/issues/251)):
  - **From a store.** The weights' codewords stay fixed for an epoch. The prover writes them column-major when it commits, and each proof reads its queried columns, checking each against its leaf. At 405B the store takes about 15.2 TB per epoch. A proof reads 0.35 GB at `live-64` (0.44 GB at `ni-100`), overlapped with P5. That costs no GPU time, so the cost model gains no term.
  - **The fallback** evaluates only the queried symbols, as RFC-0018 does for advice rows. At 405B that is about 1.3 GPU-s of tensor-core products, plus 581 GB of masks and blinding rows regenerated from `masks_seed`. Re-encoding every row per proof, about 14 GPU-s, is rejected.
  - **Where it lands.** RFC-0018's pass table gains a WA row, and its 'Column answers' says that only advice codewords are never stored. RFC-0020 says why the model has no term for the answers. 08 §4 budgets the storage, and 06 §7 lists the store among the secret artifacts.
  - **08 §5's weight openings** are about 0.35 GB: `τ_W = 97` columns over 10 matrices, blinding rows included. It had said under 0.3 GB, from 95 columns over 9 types.
- RFC-0034, accepted, takes the masked half of the score tensors out of the proof's work ([#321](https://github.com/AbdelStark/tessera/issues/321)):
  - **The layout.** Within an attention tile, `e`'s variables are ordered `(i_hi, h, q, i_lo)`. Each advice row then holds a block of keys against the tile's queries. The blocks above the diagonal are zero rows, which RFC-0011 rule 4 skips, and only the diagonal blocks' masked halves stay committed.
  - **Constant leaves.** The masked lookups `(K_0, 0)` and tree leaves of 1 form at most `n` dyadic subtrees per row. The tree and lookup provers evaluate them in closed form (RFC-0018 'Kernels'), and no message changes.
  - **The cost.** `Counts::score_block` (`b`) prices it. A token commits `A·(S + min(S, b))` scores, and looks up and multiplies `A·S`; the column answers and the row combination take the committed count. The commitment report gives `b = max(512, 2^{22 − 9 − log2 g})`: 512 at 405B, 2,048 at 8B. The source counts no block, so the golden test is unchanged.
  - **The projections.** 405B's rise by 0.21 at 1K and 0.19 at 16K: `minimal` to about 10.8 and 26.1, `servable` to 11.7 and 27.0. With streaming at its upper estimate, `servable` now passes its 1K ceiling, at about 12.2. Paid in full, the masked half would have added 1.13 and 16.8.
  - **Where it lands.** 08 §1 and §3, SC-2, RISK-9, RISK-10, `SPEC.md` and `docs/perf/v0.2/` restate the projections, and RISK-11 is resolved.
  - **Tests.**
    - The committed scores match a count of the layout's rows, row by row, for key blocks as wide as, wider than and narrower than the query blocks.
    - `score_block` moves exactly the attention, column-answer and row-combination terms.
    - The restated figures are the model's.
- RFC-0035, accepted, makes padded rows satisfy every row gadget ([#322](https://github.com/AbdelStark/tessera/issues/322)):
  - **The gap.** RFC-0011 pads every dimension to a power of two with zeros, and five row relations had no witness on an all-zero row:
    - `RsqrtRow`'s bracket;
    - `SoftmaxRow`'s bracket;
    - `OneHot`'s sum;
    - `InverseCdf`'s certificate and comparison;
    - the scoring bound's constraint 3, whose `b_t` would also have added `k` to `Û` for every padded row.

    The honest prover could not complete a proof for a sequence whose length is not a power of two.
  - **The row selector.** `ψ = [ℓ < n_layers]·[h < n_heads]·[t < n]` is public, and its extension is a product of `LT` predicates. Each failing relation multiplies its constants by `ψ`, so a padded row becomes the relation's instance at zero input.
  - **Uniqueness.** That instance leaves a few values free: `RsqrtRow`'s `ρ`, `SoftmaxRow`'s `c_q` and `R_q`, and `InverseCdf`'s maximum. A padding zero-check `(1 − ψ)·w = 0` pins each of them, so every committed value upstream of the logits keeps one passing witness.
  - **Zero-check points.** No document had said where a zero-check's point came from. Each stage with a `ZeroCheck` now squeezes it under `stage/{k}/zero/{c}` before `γ`, and every `ZeroCheck` runs in stage 1, `OneHot`'s booleanity included.
  - **What it touches.** RFC-0009, RFC-0010, RFC-0011, RFC-0014 and RFC-0017 are amended. No commitment or lookup is added.
- RFC-0012 defines the `ChannelParams` matrix's layout ([#331](https://github.com/AbdelStark/tessera/issues/331)):
  - **The rule.** The matrix holds the tensor `[types, n_layers, C]`, with `C` the widest padded output among the weights that have a `ChannelParams`. Each entry is the channel's class index or affine multiplier, or 0, and the message width is `k_W`.
  - **Why `k_W`.** A narrower message would raise the rate above the one `τ_W` is computed for. A `2^16`-wide message with `P_W = 2^16` has rate 1/2, where `live-64`'s 97 queries give about 40 bits instead of 64.
  - **At 405B.** Rows hold 16 layers of one type, 8 rows per type that has a `ChannelParams`.
  - **The vectors.** The production `ChannelParams` descriptor in `spec/vectors/hash/` follows the rule: `log2_k = 20` and `2 · 8 + 4096` rows for `servable` with two classes. Its encoding and `commitment_id` are regenerated, and both workspaces check them.
- RFC text that disagreed with itself or with every implementation is corrected ([#100](https://github.com/AbdelStark/tessera/issues/100)):
  - RFC-0008's overview of table kinds now follows its specification table: `Requant` has no `classes` field and `|r_by_class| · 2^b` rows, `Pow2` has `max_exp` rows, and `Exp2Frac` has `2^{frac_bits}` rows, its values from below. Its structured evaluation lists `Exp2Frac` among the kinds evaluated by enumeration.
  - RFC-0030's precondition is that the `Exp` table's last key maps to 0. A non-empty zero tail implies it, but it does not imply a non-empty zero tail: values can round to 0 before the tail starts.
  - RFC-0007's validation step 5 refuses a lookup that names a `Rope` table, which RFC-0008 says is no lookup table.
  - RFC-0004's `table` domain hashes the piece parameters and the core values as TCE, as RFC-0008 and every implementation do.

- RFC-0037, accepted, makes the stage schedule a function of the graph, with one claim per weight type ([#113](https://github.com/AbdelStark/tessera/issues/113)):
  - **Layers and kinds.** The lowering's checkpoints bound the layers. A tensor's kind is its node's position in the layer, stacked over the layers, and the layers must follow one template.
  - **`Rotate`** is a product against a public kernel, of degree 2, that leaves one claim on its input. RFC-0010's partner form, which #99 implemented, left two, and so two claims on Q's and K's weights.
  - **The first checkpoint is committed.** The embedding's scaled output is committed advice, proven equal to its `ConstMul` by a zero-check in stage 1. Every layer then reads a committed residual stream, and the embedding has one claim. A linear or degree-2 node other than a matmul may produce a committed tensor this way.
  - **`Transpose` and `Expand`** join RFC-0006's operation set, and attention's head layouts use them in place of `GatherPublic`. Their claims move coordinates, which `GatherPublic`'s do not when `n` is not a power of two.
  - **Splits.** A claim on several tensors splits into one claim per virtual term and one per committed group at one point. The prover sends their values under `stage/{k}/split`, in `TreeProof.splits` and `StageProof.splits`, and they must add up (`Reject(Sumcheck)`).
  - **The layer point.** Stage 1 draws it under `stage/1/layer/{c}` and stacks the leaf stage's per-layer claims with it.
  - **The pool.** It takes combinations of committed tensors at one point, and RFC-0011's grouping admits them.
  - Amended: RFC-0006, RFC-0010, RFC-0011, RFC-0014 row 6 and RFC-0036.
- RoPE is `tessera_sumcheck::Rotate`, RFC-0037's kernel relation, in place of `Elementwise::rotate` ([#113](https://github.com/AbdelStark/tessera/issues/113)):
  - **The relation.** `Ỹ(r) = Σ_x E_r(x)·X̃(x)`, a product of the public kernel `E_r(x) = eq(r_{−b}, x_{−b})·(C·eq(r_b, x_b) + S·(r_b − x_b))` and `X`, of degree 2. It leaves one claim on `X` where the partner form left two: Q's and K's weights now have one claim each.
  - **Construction.** `Rotate::new` takes `X` and the pairs' constants `C` and `S`, `2^{n−1}` entries each, indexed by a pair's other bits. `Rotate::secret` takes `X` in its `Secret`, so `secret-access.toml` lists `rotate.rs`. `tables` builds the kernel from the input claim's point, and `claim` gives the claim on `X`.
  - **Tests.** They run through `tv_core::sumcheck::verify_batch` with RFC-0037's closed form `eq(r_b, r'_b)·⟨C, w⟩ + (r_b − r'_b)·⟨S, w⟩`, which is itself checked against the kernel's extension by enumeration. The cases:
    - every size from 1 to 12 variables at every pair bit, and 20 at RoPE's bit, from RoPE's output as RFC-0005 states it;
    - a stage mixing rotations with elementwise products.

    Each of these is rejected at the final check: one changed entry of `X`, `C` or `S`, every changed round value, and every changed evaluation. The elementwise tests keep their partner reads with a pairwise product.

- The IR and the Llama lowering follow RFC-0037 ([#113](https://github.com/AbdelStark/tessera/issues/113)):
  - **`Transpose` and `Expand`.** These two `tessera-ir` operations are linear. `Transpose { perm }` permutes the input's axes, and `Expand { axis, len }` repeats it along a new axis. The graph checks that a permutation names each axis once, that a new axis sits at most at the rank, and that the output's shape is right. The witness engine and the range analysis implement both, and so does the compiler tests' reference evaluator, element by element.
  - **Committed linear outputs.** A linear or degree-2 node other than a matmul may produce committed advice, which stage 1 proves equal to its relation. The graph refuses any other committed output of such a node.
  - **Head layouts.** Attention's head layouts use `Transpose` and `Expand` in place of `GatherPublic`, whose three index constants are gone from the graph. Every claim on these layouts moves coordinates, even when `n` is not a power of two.
  - **The first checkpoint is committed advice.** This is the residual entering the first layer, the embedding scaled by `2^{s_e}`.
  - **The commitment report.** It counts committed linear outputs in a new class, `checkpoints`: `d_model` per token, 16,384 at 405B. The class is in `commitment-report-v1.schema.json`. The stage count adds each such output's zero-check and its inputs' chain, and stays 3 in `servable` and 4 in `minimal`.
  - **The v0.2 reports**, regenerated, commit 16,384 more elements per token, so `μ` moves by about 0.001. Three figures of the README's table move by 0.01, all at one class:
    - `servable`'s `μ(16K)` at `B_max = 32`, now 26.57;
    - `minimal`'s `μ(1K)` at `B_max = 32`, now 10.31;
    - `minimal`'s `μ(16K)` at `B_max = 30`, now 26.25.

### Removed

- The `TableSetManifest` artifact (schema `0x0203`, `.tts`), which three documents named and none defined (RFC-0029, accepted, [#316](https://github.com/AbdelStark/tessera/issues/316)):
  - **The claim list is the table set.** Its `tables` carry every table's kind, parameters, `log2_rows` and digest. Every party builds each table from its specification and checks the digest. A manifest would have restated them, bound by no statement digest. Materialized rows would take 33 to 57 GiB at 405B, for values that are closed forms or gathers from a core.
  - **What changes.** `tessera compile` writes `claims.tcl`, `plan.tep` and `commitment-report.json`, and `CompileOutput` has no `tables`. `0x0203` and `.tts` leave the catalog and `tessera-codec`'s registry, and stay unassigned. 01, 02 §2.2, 03 §5 and RFC-0006 'Outputs' are amended.
  - **The cost.** A prover rebuilds 405B's tables from their specifications in 2.7 s on one core. RoPE's independent pairs take 2.6 s of it. The verifier's claim-list validation already does the same work.

### Security

- Refuse committed weight matrices narrower than `k_W` (`k_T < k_W`) in `tessera-protocol/2` ([#564](https://github.com/AbdelStark/tessera/issues/564), RFC-0012 'Layout', RFC-0007 step 6, RFC-0039 step 6, RFC-0014 'Soundness budget'):
  - **The gap.** In `tessera-protocol/2`, every committed row of a weight matrix `T` (`m_T > 0`) carries `P_W = layout.weights.mask_coeffs` mask coefficients at codeword length `n_T = k_T · 2^{log2_inv_rate}`, giving effective rate `ρ'_T = (k_T + P_W) / n_T`. `τ_W` is sized for `k_W = 2^{layout.weights.log2_k}` (`ρ'_W = (k_W + P_W) / n_W = 17/64` at `live-64/1` and `ni-100/1`). Any `k_T < k_W` raises `ρ'_T` above `ρ'_W` (`40.3` bits against `64` and `70.6` against `100` at `k_T = P_W = 2^{16}`), reaches full rate `Fp^{n_T}` once `k_T + P_W ≥ n_T`, and admits exact polynomial forgeries `u + (X^{n_T} − 7^{n_T})·h` once `k_T + P_W > n_T` (`d_model = 128` at `live-64/1` gives `k_Q = 2^{14}`, `n_Q = P_W = 2^{16}`).
  - **The rule.** RFC-0012 'Layout', RFC-0007 'Verifier validation' step 6, RFC-0039 'The checks' step 6, RFC-0040 'Committed rows', RFC-0014 'Soundness budget' and draft RFC-0073 state that in every protocol version every weight matrix's codeword satisfies `n_T = k_T · 2^{log2_inv_rate} ≤ 2^{32}`, and in `tessera-protocol/2` the weight layout must also satisfy `P_W > 0` and `k_W + P_W < n_W` and every committed matrix (`m_T > 0`) must satisfy `ρ'_T ≤ ρ'_W`, i.e., `k_T ≥ k_W` (`out_pad · in_pad ≥ k_W` on `Embed`..`LmHead`, and `16 · L_pad · C ≥ k_W` on `ChannelParams` whenever `m_CP > 0`; uncommitted `ChannelParams` with `m_CP = 0` is exempt). `tessera-protocol/1` (`P = 0`) is unchanged for narrow matrices, and Llama-3.1-8B/70B/405B at `live-64/1` and `ni-100/1` and all four conformance models at `test-small/1` meet `k_T ≥ k_W`.
  - **Prover checks.** `tessera_ir::PROTOCOL_VERSION_2` (`2`, re-exported by `tessera_protocol`), `WeightLayout::check_masked` (`PcsError::MaskedWeightRate`, `PcsError::NarrowWeight`), `tessera_compiler::claim_list_for_protocol`, `tessera_compiler::compile_for_protocol`, and `tessera_compiler::plan` enforce the bounds from public data (`ERR-QUANT-ARCH` on oversized weight codewords `n_T > 2^{32}` or narrow committed matrices in `tessera-protocol/2`, `ERR-USAGE` on malformed `WeightLayoutParams` or unknown `protocol_version`). The verifier's independent `tv-core` check is recorded in `verifier/CHANGELOG.md`.

- The CPU prover's per-proof secret streams are named in RFC-0004 and constructed in place on the heap ([#567](https://github.com/AbdelStark/tessera/issues/567), RFC-0004 'Hash function and domains', OBS-2, INV-ARCH-2):
  - **The domains.** `ProverDomain::{AdviceMasks, ProverShare, NiSession}` (`tessera/v1/{advice-masks,prover-share,ni-session}`) replace the ad-hoc `tessera/prover/cpu/*/1` contexts in `crates/tessera-prover/src/cpu/random.rs` and join RFC-0004's domain table, separate from `tessera_hash::Hasher` so secret-class inputs cannot pass through the public hasher. RFC-0004 states INV-ARCH-2's narrow stream-rejection exception (`w < P`, rejected words discarded whole) directly, closing the known violation noted in 01 §6 and RFC-0018.
  - **Heap-only reader.** `blake3::Hasher::finalize_xof` returns `OutputReader` by value on the stack, `OutputReader::fill` for slices shorter than 64 bytes materializes a 64-byte stack output block without clearing unread bytes, and `OutputReader` derives `Clone`. `random.rs` replaces `blake3` in `tessera-prover`'s runtime dependencies with a single-block BLAKE3 extended-output `Stream` allocated zeroed in a `Box`, initialized in place inside `Secret`, and compressed in place on the heap, clearing consumed words immediately and the whole state on drop, with compile-time checks that neither `Seed`, `Masks`, `Heap` nor `Stream` implements `Clone` or `Debug`.
  - **Vectors and tests.** `spec/vectors/hash/hash.json` pins `prover_streams` from the independent Python BLAKE3 oracle, including a seed whose first block hits a rejected word `≥ P`, and `spec/vectors/tamper/` is regenerated from the updated base proofs.

- `Argmax` and `InverseCdf` take their one-hot input from a `OneHot` ([#540](https://github.com/AbdelStark/tessera/issues/540), RFC-0006 'Range analysis', RFC-0009 Lemmas 8 and 9):
  - **The gap.** Range analysis checked only that each entry of the one-hot input lies in `[0, 1]`. Lemma 9's uniqueness argument needs Lemma 8's "exactly one 1", which `OneHot`'s relations prove and bits from `BitDecompose` or a committed input's declared range do not. Greedy decoding is native today (RFC-0053), and the Llama lowering emits neither operation, so no proof was exposed.
  - **The rule.** The input is a `OneHot`'s output, read directly or by a `Slice` of a position axis or a `GatherPublic` of rows, which keep each vocabulary row whole. Anything else is `ERR-COMPILE-RANGE`.
  - **Tests.** Each source, for both operations: a direct read, the next token's slice and a gather pass; a slice of the vocabulary axis, decomposed bits and a committed input are refused.

- Range analysis refuses a committed graph input whose declared range no operation proves (RFC-0006 'Committed graph inputs', [#527](https://github.com/AbdelStark/tessera/issues/527)):
  - **The gap.** A graph input committed in the advice holds whatever the prover commits, and its readers trust its declared range. Nothing checked that an operation proves it. A committed byte under `ConstAdd(0)` then `BitDecompose { 8, 1 }` was proven below 512 only, and a `Bitwise8` lookup taking it as the low operand of `a + 2^8·b` admitted two witnesses. The AES circuit's key and plaintext were bytes only if the graph around it proved them.
  - **The rule.** Every element of every advice-committed graph input is read by an operation that proves its declared range and pins its padded positions. Two operations count:
    - `OneHot`, which proves `[0, vocab)` when the declared range holds it, read directly;
    - `BitDecompose`, read directly or moved by `Reshape`, `Transpose`, `Expand`, or by `Slice` and `GatherPublic` at the elements they keep.

    Weight-committed inputs take the domain their publisher vouches for (RFC-0048).
  - **`analyze`** refuses a graph that breaks it, with the new `CompileError::Unproven { tensor, rule }` (`ERR-COMPILE-RANGE`): a breaking change for code that matches every variant. **`analyze_fragment`** analyzes a circuit whose committed inputs a larger graph proves, such as AES-128-CTR alone, and is what the range tests of synthetic graphs now call.
  - **The Llama lowering** passes: its one committed advice input, the tokens, is `OneHot`'s. So does the SHA-256 circuit, whose committed bytes its `BitDecompose { 8, 0 }` reads. AES-128-CTR alone is refused until its key and plaintext are decomposed.
  - **Tests.**
    - The issue's refusals.
    - `OneHot`'s condition at both ends of the vocabulary, and through a move.
    - Each move's coverage.
    - 3,000 random chains of moves against the interpreter: the analysis accepts exactly the graphs whose decompositions reach every element.

- A sampled audit's proofs use the parameter set its policy names (RFC-0016 'The parameter set', [#530](https://github.com/AbdelStark/tessera/issues/530)):
  - **The gap.** Neither `tessera prove --sample` nor the verifier's accounting compared the proof's parameter set with the sample policy's. A proof under a weaker set, `live-64/1` under a policy of `ni-100/1`, could count toward the audit.
  - **The rule.** RFC-0019 step 2 accepts, for a proof of an audit sample, only the set its policy names: another is `Reject(Params { name })`, naming the claim list's. RFC-0043's payload and RFC-0067's accounting, which runs it first, say so.
  - **`tessera prove --sample`** refuses a plan whose claim list names another set than the sample policy's, with `ERR-USAGE`, before anything is committed. The verifier's half is in its own changelog.

- Reading a secret-class envelope no longer leaves copies of its bytes in uncleared memory (05 §3 OBS-6, [#444](https://github.com/AbdelStark/tessera/issues/444)): `WeightCommitmentState::read` and `tessera inspect` pass the file straight to the envelope reader, whose buffer clears, instead of through a `BufReader`, whose 8 KiB read-ahead held the mask seed or model bytes; the envelope hasher, whose block buffer kept up to 63 payload bytes, is zeroized when dropped. Clippy now refuses `std::io::BufReader` in the prover workspace.

- `RequantSplit` takes a key of `n` classes only when `n·2^b < P` (RFC-0009 Lemma 2, [#118](https://github.com/AbdelStark/tessera/issues/118)):
  - **The gap.** Lemma 2's proof bounded `z_off − z_lo'` by `2^b` and left out the high key's class term `2^b·c`. The two sides it compares lie in `[0, n·2^b)`, so they can differ by `P` once `n·2^b > P`. With four classes at 62 bits and `s = 32`, `z_lo ± 1` wraps the high key from one end of the table to the other, and passes with another output: `127` where the true one is `−128` at shifts 33 to 36. The compiler's selection took `RequantSplit` for such a key.
  - **The fix.** RFC-0009 states the precondition and a corrected proof, and RFC-0006, RFC-0024 and RFC-0028 give such a key `RequantWide`, whose class term reads only values its lookups pin (Lemma 11). The compiler selects it, and `tessera-gadgets` refuses the split. Of keys of at most 62 bits, only four classes at 62 bits change gadget.
  - **Also stated.** The requantization gadgets take the channel classes to be integers in `[0, n)`, RFC-0006's domain for them, which their lookups alone do not establish (RFC-0009 'Requant1').

- INV-ARCH-2 says what the prover's timing may depend on, and a release schedule keeps it from a verifier ([#257](https://github.com/AbdelStark/tessera/issues/257)):
  - **The gap.** The invariant said memory access patterns depend on public data only. Yet multiplicity counting increments the counter of each lookup's row, and `Exp`, `Exp2Frac` and `Silu` read their dense core at a key. RFC-0018 excused those reads because "the verifier observes no timing". It does: a live verifier times every reply, and caches, atomics and power-limited clocks make that work faster or slower with how the keys spread.
  - **The schedule.** In `tessera-protocol/2`, each message a verifier can time leaves at a budget `D` fixed by public data, or at the first of `2D, …, 2^L·D` that its work meets (RFC-0018 'Release schedule'). The work aborts with `ERR-RES-TIME` past the last slot. A release reveals at most which slot it took, `log2(L + 1)` bits, and nothing when the work meets its budget. 02 §3.1 specifies that the `Prover` methods return at the release time, and adds `release` to `ProverConfig`.
  - **Invariant and threat model.** INV-ARCH-2 keeps control flow public and names the two data-dependent accesses. 06's TB-1 and its timing threat, RFC-0013's non-goal and 04 §3's `ERR-RES-TIME` follow. Phase timings in run manifests and metrics stay on the prover's side of TB-1 (05 §4), and a new counter, `tessera_release_overruns_total`, records overruns.
  - **Code.** `tessera_prover::ReleaseSchedule` computes the slots. Its tests check that every duration within a slot leaves at the same instant, and that a release is the first slot the work meets. Two mutants fail them. `multiplicities` and the `Table` readers state the accesses they make, `Silu`'s included.

- Zero knowledge of the tree levels and the LogUp roots is settled for `tessera-protocol/2` (RFC-0032, accepted, [#224](https://github.com/AbdelStark/tessera/issues/224)). No code hides anything yet: zero knowledge is v0.4.
  - **The leaks.**
    - Tree levels sent their round messages unmasked.
    - Their evaluations sit at a Boolean pair bit, where RFC-0013's masked extensions vanish, so a lone LogUp tree revealed both halves of its fraction.
    - RFC-0013's LogUp masking pair shared one `δ` between the two trees, so `d_a/d_t` revealed `Π_i(α − a_i)`, a fingerprint of the secret lookups, and its leaf checks would have revealed the masks.
  - **The design.**
    - Every tree level runs a masked sumcheck (`tree/{ℓ}/zk-sum`, `tree/{ℓ}/zk-rho`).
    - Every forest holds at least two trees, a lone tree taking a public padding tree, and every level below a root is a masked extension that spares its pair bit.
    - LogUp roots are not sent: a masked sumcheck over the two forests' root levels proves the identity (`logup/identity/*`). The masking pair goes.
    - The leaf checks keep RFC-0008's form on the masked keys, values and multiplicities, whose masked extensions now spare their lowest variable.
  - **Amended.** RFC-0008, RFC-0010, RFC-0013, RFC-0014 and RFC-0019 are amended for `tessera-protocol/2`, and OQ-0013-2 is resolved. `tessera-protocol/1` is unchanged.

- `SoftmaxRow` pins its masked exponentials to zero (RFC-0030, accepted, [#279](https://github.com/AbdelStark/tessera/issues/279)):
  - **The gap.**
    - `e` is committed at the scores' full shape, and RFC-0006 says it is 0 past the causal mask. Only the valid positions were looked up, while `PV = MatMulAct(e, v)` sums every key position.
    - A prover could add `Σ_{j > q} e_qj·v_j` to any attention output and pass every check, which breaks SP-3.
    - Nothing implements the gadget yet (#122).
  - **The fix.**
    - Every position of the padded score tensor is looked up: a valid position at its key `c_q − x_qi`, a masked one at the `Exp` table's last key, which the table maps to 0.
    - Every exponential then has exactly one passing value, and the row sum, `PV` and the softmax fact read `e` as written.
    - The causal selector is `[i ≤ q < n]`, so padded rows are masked and their trees' roots are 1.
  - **Also corrected.** The trees' leaf claim reduces by an `Elementwise` relation, `Σ_x eq(r, x)·C(x)·k(x) = L̃(r) − 1 + C̃(r)`. RFC-0009 had written a product of extensions, which holds only on the hypercube.
  - **The compiler.** `SoftmaxRow`'s `Exp` table takes the narrowest width, from the spread's up, whose last key maps to 0. Claim-list validation rejects a table that does not. No model in the repository needs a wider table. RFC-0006, RFC-0007, RFC-0009 and RFC-0010 are amended.
  - **Follow-ups.**
    - RISK-11 and [#321](https://github.com/AbdelStark/tessera/issues/321): the masked half's cost. Paid in full, it would add 1.13 to 405B's `μ(1K)` and 16.8 to `μ(16K)`.
    - [#322](https://github.com/AbdelStark/tessera/issues/322): padded token rows in the row gadgets.

- The field's arithmetic is now branch-free in the code x86_64 runs, not only in its source (INV-ARCH-2, [#242](https://github.com/AbdelStark/tessera/issues/242)):
  - LLVM's x86 cmov conversion had turned the carry and borrow corrections of `Fp2` products inside loops into conditional jumps on the data. That was a timing side channel on witness values, and it cost a misprediction on about every other correction.
  - Every select on values in `tessera-field` now goes through `core::hint::select_unpredictable`: the reductions, `mul_small`, `from_i64` and `to_centered_i128`. The arithmetic is written once and instantiated twice, at run time with these selects and as `const fn`s with masks for the roots of unity computed at compile time. The compiler checks the edge cases of the latter.
  - `cargo xtask check-branchless`, a new CI gate, compiles the 13 probe loops of `tessera-field`'s `codegen-probe` feature for `x86_64-unknown-linux-gnu`, from the field operations to an NTT butterfly and an `Fp2` inner product. It fails on any conditional jump beyond a loop's own two. On main's code, the gate fails with 12 jumps on carries in the `Fp2` product loop.
  - On the GitHub-hosted runner (`docs/perf/baselines/github-ubuntu-24.04.json`, bench run 38944114196), an `Fp2` product went from 27.4 to 8.6 ns. The NTT went from 73 to 30 ns per point, a degree-3 sumcheck from 170 to 65 ns per entry, and the LogUp stage from 945 to 296 ns per lookup. That is on a runner instance whose base products were 1.5 times slower than the earlier baseline's (1.59 ns against 1.08).
- Secret-class tables reach the proving engines through their `Secret`, and every table the sumcheck derives from them is cleared from memory (05 §3 OBS-1, new OBS-6, [#229](https://github.com/AbdelStark/tessera/issues/229)):
  - `SecretBuf` in `tessera-codec` holds values derived from secret-class data. It clears its whole allocation, spare capacity included, when it drops and when it grows; a `Vec` frees the allocation it leaves when it grows without clearing it. The clearing is a fill kept by `zeroize::optimization_barrier`, about 200 GB/s, where `Vec`'s own `Zeroize` writes spare capacity one volatile byte at a time, about 4 GB/s. It has no `Debug`, `Clone` or `Encode`.
  - The field types implement `Zeroize`, which `Field` now requires, so `Secret<Vec<Fp>>` holds a witness table. A recording allocator shows that a `Secret<Vec<_>>` and a `SecretBuf` free zeros only, after dropping and after growing.
  - `tessera-sumcheck` keeps its bound tables, kernel and term rows, contractions and tree levels in `SecretBuf`s. `tessera_poly::bind_lowest` returns one, `bind_lowest_in_place` takes one, and an evaluation keeps its partial sums in one.
  - New entry points take secret-class tables by reference to their `Secret`: `Table::secret_base`, `Table::secret_ext`, `Forest::secret`, `LeafNumerators::secret`, `MatMulWeight::secret` and `MatMulAct::secret`.
  - `secret-access.toml` lists their files as readers, and lists as borrowers (`borrows = true`) the engine files that compute on tables a reader lends them: the sumcheck kernel, the contractions and `tessera-poly`'s evaluation. `cargo xtask lint` rejects a borrower that names `expose_secret`, switches clippy's rule off or is not a prover source.
  - Cost on one Apple M4 Max core: clearing a bound table takes 0.032 ns per entry, against 0.9 ns for the binding pass. The prover's benchmarks stay within 2% of main, the forests within 1.4%. The commitment's and LogUp's own buffers follow in [#264](https://github.com/AbdelStark/tessera/issues/264).
- The commitment and LogUp clear what they hold and derive from secret-class data, as the sumcheck does since #229 (05 §3 OBS-1, OBS-6, [#264](https://github.com/AbdelStark/tessera/issues/264)):
  - `ReferenceCommitment::commit` takes the rows in their `Secret` and keeps them there; its mask coefficients and codewords are `SecretBuf`s. All three are cleared when the commitment drops.
  - `tessera-hash`'s column chains lay each column's symbols out in a `SecretBuf`, where they were a plain vector freed per step.
  - `tessera_logup::multiplicities` takes the lookups' rows in their `Secret` and returns a `SecretBuf`, and counts into one. The prover's negated multiplicities and padded value column are `SecretBuf`s. `Tuples::secret` takes a lookup's keys and values in their `Secret`.
  - `secret-access.toml` lists the commitment and LogUp's instance module as readers, and as borrowers the encoder, the NTT, the opening's combination, the column chains and the LogUp prover.
  - `SecretBuf::filled` builds with `vec!`, so a buffer of zero integers takes memory the allocator has already zeroed.
  - Cost on one Apple M4 Max core: dropping a commitment, which clears it, takes 0.32 ns per committed element, against 34 to 104 ns to commit it. Committing, opening, answering and proving LogUp stay within the noise of main. Counting the multiplicities of 2^16 lookups takes about 10 µs more, the clearing of two 512 KiB buffers.
- `SoftmaxRow`'s reciprocal is unique only while `(2·max_context − 1)·2^{2σ} < P`, which claim-list validation now checks (RFC-0009 Lemma 6, [#376](https://github.com/AbdelStark/tessera/issues/376)):
  - **The gap.** Lemma 6's proof kept `R_q·s_q` below `2^{50}`, which holds at `σ = 16` only, and validation bounded `σ` by RFC-0008's 32 alone: `exp_sigma = 24` compiled at 8B's shape. There a row of 33,280 positions at its maximal score, `s_q = 33,280·2^{24}`, passes relation 3 with `R = 504` and with `R' = 33,038,714`, whose product with `s_q` wraps past `P` once. The second reciprocal changes the attention output, against SP-3. Nothing implements the gadget yet (#122), and every model in the repository uses `σ = 16`.
  - **The condition.** Masked and padded positions add 0 to a row sum, and each exponential is at most `2^σ`, so `s_q ≤ max_context·2^σ`. A reciprocal times the row sum plus its slack then stays below `P + 2^{2σ}`, and so equals `2^{2σ}`, exactly when `(2·max_context − 1)·2^{2σ} < P`: `2σ + ⌈log2 max_context⌉ ≤ 63`, so `σ ≤ 23` at `max_context = 2^{17}` and `σ ≤ 31` at the least context. Past it the largest row sum admits a second reciprocal, so the bound is exact.
  - **The fix.** RFC-0009 states the precondition and a corrected proof, which compares the product without Lemma 0, and says that `NonNeg`'s `σ + 18` bits rest on RFC-0007's `max_context ≤ 2^{17}`. RFC-0007 step 2.2 adds the rule (`Quantization`) and RFC-0006 the lowering's refusal. RFC-0005 bounds the row sum by `2^{σ+17}`, which it reaches, where it said `s_q < 2^{σ+17}`. `tessera-compiler` refuses such a model with `ERR-QUANT-ARCH`, and `tv-core` rejects its claim list.
  - **Tests.** `tessera-gadgets` assembles relation 3 from `Range { σ + 1 }` and two `NonNeg { σ + 18, 16 }` over Goldilocks. Both reciprocals pass at `σ = 24`; exhaustive scans of `R` find the true one alone at `σ = 23` with `2^{17}` positions and at `σ = 24` with `2^{15}`, and two with `2^{15} + 1`. Over `F65537`, exhaustion over `σ`, contexts up to 32, row sums and `R` shows the bound exact. The compiler accepts the largest `σ` and refuses the next at eight contexts from 1 to `2^{17}`, decides as the defining form does at every power of two and its neighbours, and lowers 8B's shape at `σ = 23`.
- Reconcile per-dimension padded `Reshape` coordinate identity in `tessera-compiler` with RFC-0037 'Head layouts' and `tv-core`: in each group of axes that merge or split, every axis after the group's leading axis is a power of two, regardless of axes above the group ([#411](https://github.com/AbdelStark/tessera/issues/411)).
  - Six query heads and three KV heads (`d_model = 96`, `d_head = 16`) now derive stage and transcript schedules, match the verifier's constraint-system digest, and prove and verify across padding boundaries.
  - Non-power-of-two head widths retain exact typed schedule rejection (`Reshape` at `NodeId(12)` when `n_heads > 1`, `Rotate` at `NodeId(13)` when `n_heads = 1`).
- Compare `tessera-protocol`'s soundness calculator against `tv-core::soundness` across 1,000 random statement and parameter profiles in CI, and extend `cargo xtask lint`'s integer-only source check (`no_float`) to `verifier/crates/tv-core/src/soundness.rs` ([#110](https://github.com/AbdelStark/tessera/issues/110)).
- `tessera params check` exits 10 (`02 §2.4`, RFC-0051) when any row of the soundness report misses its target (`meets_targets == Some(false)`), while still printing the human-readable table or `tessera.soundness/1` JSON report on stdout; `live-64`, `ni-100` and `test-small` at the default 405B statement continue to exit 0 ([#375](https://github.com/AbdelStark/tessera/issues/375)).
- Add `tessera compile` (`02 §2.2`, RFC-0006 'Outputs', RFC-0029), `tessera_compiler::compile` (`CompileOutput`), and `tessera bench model --plan` ([#87](https://github.com/AbdelStark/tessera/issues/87)):
  - `tessera compile --model MODEL.d --params PARAMSET --out-dir DIR` validates `D`, lowers it, selects gadgets and tables, and writes `claims.tcl`, `plan.tep` and `commitment-report.json` (`tessera.commitment-report/1`) atomically; with `--json` it prints the commitment report on stdout.
  - `tessera bench model --plan plan.tep` verifies the plan against its claim list and projects `μ` from the model's compiled commitment counts.
- Add `tessera commit-weights` (`02 §2.2`, RFC-0012 'Layout', RFC-0040, RFC-0041 'The plan', RFC-0048) ([#111](https://github.com/AbdelStark/tessera/issues/111)):
  - `tessera commit-weights --model MODEL.d --plan plan.tep --out WEIGHTS.twc` validates `D`'s public and hidden domains, verifies `plan.tep` against `D`, commits the ten weight matrices at epoch 0, writes `WEIGHTS.twc` (`0x0302`, mode `0600`) and `WEIGHTS.twr` (`0x0301`, default mode) by atomic rename, and prints `R_W` (`--json` prints the commitment summary and matrix roots).
- Add `tessera inspect FILE` (`02 §2.2`, RFC-0003 'Non-Goals', `03 §5`, `05 §3`) and `tessera_ir::TokenFile` (`0x0601`) ([#97](https://github.com/AbdelStark/tessera/issues/97)):
  - Decodes any catalog TCE artifact and renders a structured text view (or a JSON summary with `--json`), while secret-class artifacts (`IntegerModel`, `WeightCommitmentState`, `TokenFile`, `LogitsFile`) print shapes and BLAKE3 digests only, never weight entries, channel parameters, mask seeds, token ids or logit values.
- Open several reading-site claims on the `ChannelParams` weight matrix in one opening pass (RFC-0052, accepted; RFC-0012 'Opening'; RFC-0036; RFC-0037 'The claim pool'; RFC-0049; [#386](https://github.com/AbdelStark/tessera/issues/386)):
  - `WeightLayout::channel_point` maps a reading site's `(weight_type, r_c, r_ell)` to its `ChannelParams` point by embedding `weight_type`'s 3-bit index into the top variables and zero-padding `r_c` to `c_vars = ⌈log2 C⌉`.
  - `WeightFold::channel_params` folds any non-empty slice of `(point, gamma)` site claims on `ChannelParams` into `G = |claims|` evaluation rows plus the proximity row in one pass (`PcsError::NoClaims` when empty), and `WeightCombinedRows.evaluations` becomes `Vec<CombinedRow>` in `tessera-pcs` and `tessera-protocol`.
  - `spec/vectors/weights-opening/openings.json` gains `"channel-multi"`, opening three site claims across `Q`, `Gate` and `Down` on `ChannelParams`, and `spec/vectors/protocol/protocol.json` updates for the `Vec<CombinedRow>` wire format.
- Add debug/test-only malicious alternative-witness generators (`tessera_gadgets::malicious`, gated behind `#[cfg(debug_assertions)]` so release builds omit it; OBS-4) and the end-to-end alternative-witness verification suite (`crates/tessera-e2e/tests/alternative_witnesses.rs`) covering `Requant1`, `RequantSplit`, `RequantWide`, `NonNeg`, `BigIntCompare`, `SoftmaxRow`, `RsqrtRow`, `OneHot` and `Argmax` against `tv-core::gadgets`, `tv-core::logup`, `tv-core::tree` and `tv-core::sumcheck` (RFC-0009 'Testing Strategy', `07 §4`, [#124](https://github.com/AbdelStark/tessera/issues/124), [#129](https://github.com/AbdelStark/tessera/issues/129)).
