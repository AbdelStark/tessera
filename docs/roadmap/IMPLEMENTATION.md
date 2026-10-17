# Implementation Tracker — 2026-10-15

Generated from the spec corpus committed in [#1](https://github.com/AbdelStark/tessera/pull/1): the plan as filed on 2026-10-15, every implementable unit of work of the spec at the time. Each issue is independently shippable; cross-issue dependencies are noted inline in each issue body (Depends on / Blocks) and summarized at the end of this document.

This document is the plan, not its progress. Whether an issue is open or closed, and the issues filed since, such as spec corrections and security findings, are on GitHub: each tracking issue's checklist is its subsystem's dashboard.

- Implementation issues: 165; tracking issues: 23; total: 188 (#2–#189).
- By milestone: v0.1 42, v0.2 27, v0.3 42, v0.4 29, v0.5 11, v1.0 14.
- By area: verifier 26, infra 19, compiler 16, quant 13, gpu 12, pcs 9, bench 8, core 8, cli 7, sumcheck 7, anchor 6, docs 6, protocol 6, lookup 5, prover 5, zk 5, scoring 4, witness 3.
- By priority: p0 132, p1 29, p2 4.
- By effort: s 25, m 104, l 36.

## Label taxonomy

| Label family | Meaning |
|---|---|
| `type:*` | Exactly one per issue: feature, bug, test, docs, ci, refactor, security, perf, tracking |
| `area:*` | Exactly one per issue: the subsystem of [01 Architecture](../spec/01-architecture.md#2-components) that owns the work |
| `priority:p0` | Required for its milestone's exit criteria |
| `priority:p1` | Required for v1.0, but not for its own milestone's exit criteria |
| `priority:p2` | Optional; schedule when capacity allows |
| `effort:s` / `effort:m` / `effort:l` | Up to 2 days / up to 1 week / 1 to 3 weeks for one contributor |
| `spec:rfc-NNNN`, `spec:sNN-*` | The RFC or specification section the issue implements (one or more) |
| `tracking`, `blocked`, `good first issue`, `help wanted` | Status labels; `good first issue` only where genuinely self-contained |

Exit criteria per milestone are in [SPEC.md](../../SPEC.md#release-plan).

## Milestone: v0.1 — Primitives and unit-cost gate

| # | Title | Area | Priority | Effort | RFC |
|---|-------|------|----------|--------|-----|
| [#25](https://github.com/AbdelStark/tessera/issues/25) | infra: bootstrap the prover workspace with crate skeletons and toolchain pin | infra | p0 | m | RFC-0001 |
| [#26](https://github.com/AbdelStark/tessera/issues/26) | docs: add license files and the SPDX header policy | docs | p0 | s | s09-release |
| [#27](https://github.com/AbdelStark/tessera/issues/27) | docs: add CONTRIBUTING, SECURITY, code of conduct and templates | docs | p1 | s | s09-release, s06-security |
| [#28](https://github.com/AbdelStark/tessera/issues/28) | infra: bootstrap the verifier workspace with allowlist and toolchain pin | infra | p0 | s | RFC-0001, RFC-0019 |
| [#29](https://github.com/AbdelStark/tessera/issues/29) | infra: integrate the CUDA build for tessera-gpu behind the gpu feature | infra | p0 | m | RFC-0001, RFC-0018 |
| [#30](https://github.com/AbdelStark/tessera/issues/30) | docs: add the milestone release checklist | docs | p2 | s | s09-release, s06-security |
| [#31](https://github.com/AbdelStark/tessera/issues/31) | core: implement the error taxonomy, stable codes and exit-code mapping | core | p0 | s | s04-errors |
| [#32](https://github.com/AbdelStark/tessera/issues/32) | core: implement Secret and Public wrapper types with redaction lints | core | p0 | m | s05-observability, s06-security |
| [#33](https://github.com/AbdelStark/tessera/issues/33) | core: implement the Goldilocks field and the F65537 test field | core | p0 | m | RFC-0002 |
| [#34](https://github.com/AbdelStark/tessera/issues/34) | core: implement BLAKE3 domains, Merkle multiproofs and chained column digests | core | p0 | m | RFC-0004 |
| [#35](https://github.com/AbdelStark/tessera/issues/35) | bench: implement the executable cost model with a golden test | bench | p0 | m | RFC-0020, s08-performance |
| [#36](https://github.com/AbdelStark/tessera/issues/36) | infra: implement the check-isolation gate for the verifier workspace | infra | p0 | m | RFC-0001 |
| [#37](https://github.com/AbdelStark/tessera/issues/37) | infra: reject test-only cargo features in release builds | infra | p1 | s | RFC-0001, s05-observability |
| [#38](https://github.com/AbdelStark/tessera/issues/38) | ci: provision the self-hosted H100 runner and GPU test job | infra | p0 | m | s07-testing, RFC-0020 |
| [#39](https://github.com/AbdelStark/tessera/issues/39) | core: implement the quadratic extension Fp2 = Fp[X]/(X^2 - 7) | core | p0 | s | RFC-0002 |
| [#40](https://github.com/AbdelStark/tessera/issues/40) | verifier: implement independent field arithmetic in tv-field | verifier | p0 | m | RFC-0002, RFC-0019 |
| [#41](https://github.com/AbdelStark/tessera/issues/41) | core: implement the TCE encoder and decoder with derive macro | core | p0 | m | RFC-0003 |
| [#42](https://github.com/AbdelStark/tessera/issues/42) | ci: add the pull-request pipeline for both workspaces | infra | p0 | m | s07-testing, RFC-0001 |
| [#43](https://github.com/AbdelStark/tessera/issues/43) | core: implement the artifact envelope, schema registry and decode limits | core | p0 | s | RFC-0003, s03-data-model |
| [#44](https://github.com/AbdelStark/tessera/issues/44) | verifier: implement independent TCE decoders in tv-codec | verifier | p0 | m | RFC-0003, RFC-0019 |
| [#45](https://github.com/AbdelStark/tessera/issues/45) | core: implement the Fiat-Shamir transcript with rejection sampling and grinding | core | p0 | m | RFC-0004, RFC-0014 |
| [#46](https://github.com/AbdelStark/tessera/issues/46) | sumcheck: implement multilinear tables, eq tables and MLE evaluation | sumcheck | p0 | m | RFC-0010 |
| [#47](https://github.com/AbdelStark/tessera/issues/47) | pcs: implement coset NTT and rate-1/2 RS encoding with vanishing masks | pcs | p0 | m | RFC-0011 |
| [#48](https://github.com/AbdelStark/tessera/issues/48) | gpu: implement the Goldilocks and Fp2 device arithmetic library | gpu | p0 | m | RFC-0002, RFC-0018 |
| [#49](https://github.com/AbdelStark/tessera/issues/49) | bench: implement the unit-cost measurement harness and report schema | bench | p0 | m | RFC-0020 |
| [#50](https://github.com/AbdelStark/tessera/issues/50) | ci: add dependency policy gates with cargo-deny and cargo-vet | infra | p1 | s | s06-security, s09-release, RFC-0001 |
| [#51](https://github.com/AbdelStark/tessera/issues/51) | ci: add changelog files and changelog and DCO checks | infra | p1 | s | s09-release |
| [#52](https://github.com/AbdelStark/tessera/issues/52) | verifier: implement independent hashing, Merkle checks and transcript replay | verifier | p0 | m | RFC-0004, RFC-0019 |
| [#53](https://github.com/AbdelStark/tessera/issues/53) | sumcheck: implement the generic batched sumcheck prover | sumcheck | p0 | m | RFC-0010 |
| [#54](https://github.com/AbdelStark/tessera/issues/54) | pcs: implement the reference tensor commitment and opening | pcs | p0 | m | RFC-0011, RFC-0004 |
| [#55](https://github.com/AbdelStark/tessera/issues/55) | gpu: implement the encoding and column-hashing microkernel for c_c | gpu | p0 | l | RFC-0020, RFC-0011, RFC-0004 |
| [#56](https://github.com/AbdelStark/tessera/issues/56) | gpu: implement the limb-decomposed INT8 fold kernel and weight-pass benchmark | gpu | p1 | m | RFC-0020, RFC-0012 |
| [#57](https://github.com/AbdelStark/tessera/issues/57) | sumcheck: implement the MatMulWeight and MatMulAct relations | sumcheck | p0 | m | RFC-0010 |
| [#58](https://github.com/AbdelStark/tessera/issues/58) | sumcheck: implement the product-tree GKR prover | sumcheck | p0 | m | RFC-0010, RFC-0009 |
| [#59](https://github.com/AbdelStark/tessera/issues/59) | verifier: implement sumcheck and product-tree verification | verifier | p0 | m | RFC-0010, RFC-0019 |
| [#60](https://github.com/AbdelStark/tessera/issues/60) | lookup: implement the LogUp-GKR prover with combined fraction trees | lookup | p0 | l | RFC-0008 |
| [#61](https://github.com/AbdelStark/tessera/issues/61) | verifier: implement tensor-commitment opening checks | verifier | p0 | m | RFC-0011, RFC-0019 |
| [#62](https://github.com/AbdelStark/tessera/issues/62) | verifier: implement LogUp-GKR verification | verifier | p0 | m | RFC-0008, RFC-0019 |
| [#63](https://github.com/AbdelStark/tessera/issues/63) | gpu: implement the LogUp-GKR fraction-tree microkernel for c_l | gpu | p0 | l | RFC-0020, RFC-0008 |
| [#64](https://github.com/AbdelStark/tessera/issues/64) | gpu: implement the product-tree microkernel for c_pi | gpu | p0 | m | RFC-0020, RFC-0010 |
| [#65](https://github.com/AbdelStark/tessera/issues/65) | bench: add CPU microbenchmarks for primitives | bench | p1 | s | s08-performance |
| [#66](https://github.com/AbdelStark/tessera/issues/66) | bench: run the v0.1 H100 unit-cost measurement and record the decision | bench | p0 | m | RFC-0020, s08-performance |

## Milestone: v0.2 — Integer model and compiler

| # | Title | Area | Priority | Effort | RFC |
|---|-------|------|----------|--------|-----|
| [#67](https://github.com/AbdelStark/tessera/issues/67) | infra: bootstrap the tessera-quant Python package | infra | p0 | s | RFC-0001, RFC-0005 |
| [#68](https://github.com/AbdelStark/tessera/issues/68) | cli: implement the tessera CLI skeleton with JSON output and exit codes | cli | p1 | m | s02-api, s04-errors |
| [#69](https://github.com/AbdelStark/tessera/issues/69) | lookup: implement arithmetic table kinds and table digests | lookup | p0 | m | RFC-0008 |
| [#70](https://github.com/AbdelStark/tessera/issues/70) | quant: implement Llama-3.1 checkpoint import and architecture validation | quant | p0 | m | RFC-0005, s06-security |
| [#71](https://github.com/AbdelStark/tessera/issues/71) | lookup: implement integer-only Exp, Silu, Exp2Frac and Rope generators | lookup | p0 | l | RFC-0008, RFC-0005 |
| [#72](https://github.com/AbdelStark/tessera/issues/72) | compiler: implement IR types, graph construction and node kinds | compiler | p0 | m | RFC-0006, s03-data-model |
| [#73](https://github.com/AbdelStark/tessera/issues/73) | quant: implement the deterministic calibration pipeline with hygiene checks | quant | p0 | m | RFC-0005, s07-testing |
| [#74](https://github.com/AbdelStark/tessera/issues/74) | quant: implement the Python TCE writer and reader | quant | p0 | m | RFC-0003, RFC-0005 |
| [#75](https://github.com/AbdelStark/tessera/issues/75) | lookup: implement structured MLE evaluation for piecewise tables | lookup | p0 | m | RFC-0008 |
| [#76](https://github.com/AbdelStark/tessera/issues/76) | quant: implement Python table generators for the integer reference | quant | p0 | m | RFC-0008, RFC-0005 |
| [#77](https://github.com/AbdelStark/tessera/issues/77) | compiler: implement public-only range analysis with relational facts | compiler | p0 | l | RFC-0006, s06-security |
| [#78](https://github.com/AbdelStark/tessera/issues/78) | compiler: implement Llama-3.1 lowering for the servable and minimal profiles | compiler | p0 | l | RFC-0006, RFC-0005 |
| [#79](https://github.com/AbdelStark/tessera/issues/79) | quant: implement SmoothQuant migration, QK equalization and gain folding | quant | p0 | m | RFC-0005 |
| [#80](https://github.com/AbdelStark/tessera/issues/80) | compiler: implement gadget selection and table sizing | compiler | p0 | m | RFC-0006, RFC-0009 |
| [#81](https://github.com/AbdelStark/tessera/issues/81) | quant: implement W8A8 quantization with power-of-two and affine scale policies | quant | p0 | m | RFC-0005, RFC-0007 |
| [#82](https://github.com/AbdelStark/tessera/issues/82) | witness: implement the CPU reference integer engine for D | witness | p0 | l | RFC-0018, RFC-0005 |
| [#83](https://github.com/AbdelStark/tessera/issues/83) | test: generate tiny conformance models and the shared vector layout | infra | p0 | m | s07-testing, RFC-0001 |
| [#84](https://github.com/AbdelStark/tessera/issues/84) | compiler: implement the commitment report | compiler | p0 | m | RFC-0006, RFC-0020 |
| [#85](https://github.com/AbdelStark/tessera/issues/85) | compiler: implement the claim-list schema and emission | compiler | p0 | m | RFC-0007 |
| [#86](https://github.com/AbdelStark/tessera/issues/86) | quant: implement IntegerModel assembly and export with provenance | quant | p0 | s | RFC-0005, RFC-0003 |
| [#87](https://github.com/AbdelStark/tessera/issues/87) | cli: add the tessera compile command | cli | p1 | s | s02-api, RFC-0006 |
| [#88](https://github.com/AbdelStark/tessera/issues/88) | compiler: produce 405B-shape commitment reports to resolve RISK-2 | compiler | p0 | s | RFC-0006, RFC-0020 |
| [#89](https://github.com/AbdelStark/tessera/issues/89) | quant: implement the PyTorch integer reference for D | quant | p0 | l | RFC-0005, s07-testing |
| [#90](https://github.com/AbdelStark/tessera/issues/90) | quant: implement the fidelity evaluation harness with confidence intervals | quant | p0 | m | RFC-0005, s07-testing |
| [#91](https://github.com/AbdelStark/tessera/issues/91) | test: add the Rust versus PyTorch bit-exactness differential test | witness | p0 | s | RFC-0018, RFC-0005, s07-testing |
| [#92](https://github.com/AbdelStark/tessera/issues/92) | cli: implement the tessera-quant command-line interface | cli | p1 | s | s02-api, RFC-0005 |
| [#93](https://github.com/AbdelStark/tessera/issues/93) | quant: run the 8B fidelity gates and add the fidelity CI job | quant | p0 | m | RFC-0005, s07-testing |

## Milestone: v0.3 — End-to-end reference proof (CPU, sound, non-hiding)

| # | Title | Area | Priority | Effort | RFC |
|---|-------|------|----------|--------|-----|
| [#94](https://github.com/AbdelStark/tessera/issues/94) | infra: implement structured logging and tracing spans | infra | p1 | m | s05-observability |
| [#95](https://github.com/AbdelStark/tessera/issues/95) | infra: implement the metrics registry and run manifests | infra | p1 | m | s05-observability, RFC-0020 |
| [#96](https://github.com/AbdelStark/tessera/issues/96) | ci: add nightly jobs for deep property tests, end-to-end subsets and fuzzing | infra | p1 | s | s07-testing |
| [#97](https://github.com/AbdelStark/tessera/issues/97) | cli: add tessera inspect to render artifacts as text | cli | p2 | s | s02-api, RFC-0003 |
| [#98](https://github.com/AbdelStark/tessera/issues/98) | protocol: implement message schemas and session state machines | protocol | p0 | l | RFC-0014, s03-data-model |
| [#99](https://github.com/AbdelStark/tessera/issues/99) | sumcheck: implement Elementwise, SumSquares, ZeroCheck and Linear relations | sumcheck | p0 | m | RFC-0010 |
| [#100](https://github.com/AbdelStark/tessera/issues/100) | verifier: implement independent table generators and structured evaluation | verifier | p0 | l | RFC-0008, RFC-0019 |
| [#101](https://github.com/AbdelStark/tessera/issues/101) | compiler: implement tiling and the execution plan | compiler | p0 | m | RFC-0006, RFC-0018 |
| [#102](https://github.com/AbdelStark/tessera/issues/102) | pcs: implement the weight commitment layout and rate-1/4 commitment | pcs | p0 | m | RFC-0012 |
| [#103](https://github.com/AbdelStark/tessera/issues/103) | protocol: implement parameter sets and the soundness calculator | protocol | p0 | m | RFC-0014 |
| [#104](https://github.com/AbdelStark/tessera/issues/104) | test: add soundness calibration over F65537 for sumcheck and LogUp | sumcheck | p1 | m | RFC-0010, RFC-0008, s07-testing |
| [#105](https://github.com/AbdelStark/tessera/issues/105) | lookup: batch LogUp instances of different heights in one sumcheck per level | lookup | p0 | m | RFC-0008 |
| [#106](https://github.com/AbdelStark/tessera/issues/106) | pcs: implement the canonical advice layout | pcs | p0 | m | RFC-0011, RFC-0006 |
| [#107](https://github.com/AbdelStark/tessera/issues/107) | verifier: implement claim-list validation and independent lowering | verifier | p0 | l | RFC-0007, RFC-0019 |
| [#108](https://github.com/AbdelStark/tessera/issues/108) | pcs: implement the folding-pass opening on CPU | pcs | p0 | m | RFC-0012, RFC-0010 |
| [#109](https://github.com/AbdelStark/tessera/issues/109) | verifier: implement weight opening checks | verifier | p0 | m | RFC-0012, RFC-0019 |
| [#110](https://github.com/AbdelStark/tessera/issues/110) | verifier: implement the independent soundness calculator | verifier | p1 | s | RFC-0014, RFC-0019 |
| [#111](https://github.com/AbdelStark/tessera/issues/111) | cli: add the tessera commit-weights command | cli | p1 | s | s02-api, RFC-0012 |
| [#112](https://github.com/AbdelStark/tessera/issues/112) | witness: implement layer-boundary checkpoints and the regeneration interface | witness | p0 | m | RFC-0018 |
| [#113](https://github.com/AbdelStark/tessera/issues/113) | sumcheck: implement stage schedule derivation with stacked layers | sumcheck | p0 | l | RFC-0010, RFC-0013 |
| [#114](https://github.com/AbdelStark/tessera/issues/114) | verifier: implement independent stage schedule derivation | verifier | p0 | l | RFC-0010, RFC-0019 |
| [#115](https://github.com/AbdelStark/tessera/issues/115) | pcs: implement the streaming committer with segments and zero-row skipping | pcs | p0 | m | RFC-0011 |
| [#116](https://github.com/AbdelStark/tessera/issues/116) | verifier: implement advice layout derivation and grouped opening checks | verifier | p0 | m | RFC-0011, RFC-0019 |
| [#117](https://github.com/AbdelStark/tessera/issues/117) | verifier: implement independent range re-derivation | verifier | p0 | l | RFC-0007, RFC-0006, s06-security |
| [#118](https://github.com/AbdelStark/tessera/issues/118) | gadgets: implement Requant1, RequantSplit, RequantWide and NonNeg | compiler | p0 | m | RFC-0009, RFC-0024 |
| [#119](https://github.com/AbdelStark/tessera/issues/119) | pcs: implement the grouped opening protocol and column answers | pcs | p0 | l | RFC-0011 |
| [#120](https://github.com/AbdelStark/tessera/issues/120) | test: add constraint-system digest conformance between compiler and verifier | compiler | p1 | m | RFC-0007, s07-testing |
| [#121](https://github.com/AbdelStark/tessera/issues/121) | gadgets: implement BigIntCompare | compiler | p0 | m | RFC-0009 |
| [#122](https://github.com/AbdelStark/tessera/issues/122) | gadgets: implement the SoftmaxRow max certificate and reciprocal bracket | compiler | p0 | l | RFC-0009 |
| [#123](https://github.com/AbdelStark/tessera/issues/123) | gadgets: implement OneHot and Argmax | compiler | p0 | m | RFC-0009, RFC-0017 |
| [#124](https://github.com/AbdelStark/tessera/issues/124) | verifier: implement final checks for every uniqueness gadget | verifier | p0 | l | RFC-0009, RFC-0019 |
| [#125](https://github.com/AbdelStark/tessera/issues/125) | protocol: implement the transcript schedule for both modes | protocol | p0 | m | RFC-0014, RFC-0004 |
| [#126](https://github.com/AbdelStark/tessera/issues/126) | gadgets: implement the RsqrtRow bracket | compiler | p0 | m | RFC-0009, RFC-0005 |
| [#127](https://github.com/AbdelStark/tessera/issues/127) | scoring: implement the exact predictor with greedy decoding | scoring | p0 | m | RFC-0017 |
| [#128](https://github.com/AbdelStark/tessera/issues/128) | verifier: implement non-interactive and live drivers with the fixed check order | verifier | p0 | l | RFC-0019, s04-errors |
| [#129](https://github.com/AbdelStark/tessera/issues/129) | test: add alternative-witness tests for every gadget | compiler | p0 | m | RFC-0009, s07-testing |
| [#130](https://github.com/AbdelStark/tessera/issues/130) | prover: implement the CPU prover backend and protocol driver | prover | p0 | l | RFC-0018, s02-api |
| [#131](https://github.com/AbdelStark/tessera/issues/131) | cli: implement the tessera-verify command-line interface | cli | p0 | m | s02-api, RFC-0019 |
| [#132](https://github.com/AbdelStark/tessera/issues/132) | prover: implement the pass planner, proof units and pass counters | prover | p0 | l | RFC-0018, RFC-0020 |
| [#133](https://github.com/AbdelStark/tessera/issues/133) | test: add the exact-mode end-to-end suite on tiny models | infra | p0 | m | s07-testing |
| [#134](https://github.com/AbdelStark/tessera/issues/134) | prover: stream tree levels that exceed the memory budget | prover | p0 | m | RFC-0018, RFC-0010 |
| [#135](https://github.com/AbdelStark/tessera/issues/135) | test: build and maintain the tamper corpus | verifier | p0 | m | s07-testing, RFC-0019 |

## Milestone: v0.4 — Full statement: zero knowledge, anchor, audits, scoring

| # | Title | Area | Priority | Effort | RFC |
|---|-------|------|----------|--------|-----|
| [#136](https://github.com/AbdelStark/tessera/issues/136) | scoring: implement the sampler PRF reference and test vectors | scoring | p0 | s | RFC-0017 |
| [#137](https://github.com/AbdelStark/tessera/issues/137) | anchor: implement frames, session-open frames and digest-log schemas | anchor | p0 | m | RFC-0015 |
| [#138](https://github.com/AbdelStark/tessera/issues/138) | anchor: implement log manifests with Merkle roots and Ed25519 signatures | anchor | p0 | m | RFC-0015 |
| [#139](https://github.com/AbdelStark/tessera/issues/139) | anchor: implement the software reference recorder | anchor | p0 | m | RFC-0015 |
| [#140](https://github.com/AbdelStark/tessera/issues/140) | protocol: implement audit policy, sample derivation and batching | protocol | p0 | m | RFC-0016 |
| [#141](https://github.com/AbdelStark/tessera/issues/141) | protocol: implement the audit report and the detection and overhead calculators | protocol | p1 | m | RFC-0016, RFC-0020 |
| [#142](https://github.com/AbdelStark/tessera/issues/142) | anchor: implement the AES-128-CTR circuit | anchor | p0 | l | RFC-0015, RFC-0009 |
| [#143](https://github.com/AbdelStark/tessera/issues/143) | anchor: implement the SHA-256 circuit | anchor | p0 | l | RFC-0015 |
| [#144](https://github.com/AbdelStark/tessera/issues/144) | gadgets: implement the InverseCdf sampling certificate | compiler | p0 | m | RFC-0009, RFC-0017 |
| [#145](https://github.com/AbdelStark/tessera/issues/145) | docs: write the zero-knowledge simulator argument | zk | p0 | l | RFC-0013, s06-security |
| [#146](https://github.com/AbdelStark/tessera/issues/146) | anchor: implement token-byte binding and the prover pre-check | anchor | p0 | m | RFC-0015 |
| [#147](https://github.com/AbdelStark/tessera/issues/147) | pcs: implement weight masks, blinding rows and the budget state machine | pcs | p0 | l | RFC-0012, RFC-0013, s06-security |
| [#148](https://github.com/AbdelStark/tessera/issues/148) | zk: implement masked sumchecks with committed masking polynomials | zk | p0 | m | RFC-0013 |
| [#149](https://github.com/AbdelStark/tessera/issues/149) | zk: implement blinding rows for advice openings and LogUp root masking | zk | p0 | m | RFC-0013, RFC-0011 |
| [#150](https://github.com/AbdelStark/tessera/issues/150) | verifier: implement anchor checks, manifest signatures and inclusion proofs | verifier | p0 | m | RFC-0015, RFC-0019 |
| [#151](https://github.com/AbdelStark/tessera/issues/151) | verifier: implement audit accounting, the ledger and audit plan | verifier | p0 | m | RFC-0016, RFC-0019 |
| [#152](https://github.com/AbdelStark/tessera/issues/152) | scoring: implement vocabulary softmax and temperature decoding certificates | scoring | p0 | m | RFC-0017, RFC-0009 |
| [#153](https://github.com/AbdelStark/tessera/issues/153) | pcs: implement weight commitment refresh and the linking proof | pcs | p1 | m | RFC-0012 |
| [#154](https://github.com/AbdelStark/tessera/issues/154) | zk: implement masked extensions for stage outputs and committed tensors | zk | p0 | l | RFC-0013 |
| [#155](https://github.com/AbdelStark/tessera/issues/155) | scoring: implement the mixture predictor with the one-sided log2 bound | scoring | p0 | l | RFC-0017 |
| [#156](https://github.com/AbdelStark/tessera/issues/156) | prover: implement crash consistency with the per-proof secret seed | prover | p1 | m | RFC-0018, s06-security |
| [#157](https://github.com/AbdelStark/tessera/issues/157) | cli: add the tessera prove and audit commands | cli | p1 | m | s02-api, RFC-0016 |
| [#158](https://github.com/AbdelStark/tessera/issues/158) | verifier: verify weight link proofs and track epochs in the ledger | verifier | p1 | m | RFC-0012, RFC-0019 |
| [#159](https://github.com/AbdelStark/tessera/issues/159) | verifier: implement the zero-knowledge checks | verifier | p0 | m | RFC-0013, RFC-0019 |
| [#160](https://github.com/AbdelStark/tessera/issues/160) | verifier: implement scoring checks and the Û bound | verifier | p0 | m | RFC-0017, RFC-0019 |
| [#161](https://github.com/AbdelStark/tessera/issues/161) | quant: run the bounded-mode fidelity study with an FP8 deployment proxy | quant | p0 | m | RFC-0017, RFC-0005 |
| [#162](https://github.com/AbdelStark/tessera/issues/162) | test: add the zero-knowledge simulator and distinguisher suite | zk | p0 | m | RFC-0013, s07-testing |
| [#163](https://github.com/AbdelStark/tessera/issues/163) | test: add bounded-mode, anchor, budget and reproducibility end-to-end tests | infra | p0 | m | s07-testing, RFC-0015, RFC-0017, RFC-0012 |
| [#164](https://github.com/AbdelStark/tessera/issues/164) | test: add the canary redaction scan to end-to-end CI | infra | p0 | s | s05-observability, s07-testing |

## Milestone: v0.5 — GPU prover

| # | Title | Area | Priority | Effort | RFC |
|---|-------|------|----------|--------|-----|
| [#165](https://github.com/AbdelStark/tessera/issues/165) | gpu: implement the INT8 witness engine on tensor cores | gpu | p0 | l | RFC-0018, RFC-0005 |
| [#166](https://github.com/AbdelStark/tessera/issues/166) | gpu: implement the production LogUp-GKR and product-tree provers | gpu | p0 | l | RFC-0008, RFC-0018 |
| [#167](https://github.com/AbdelStark/tessera/issues/167) | bench: add the nightly performance regression gate | bench | p1 | m | s08-performance, RFC-0020 |
| [#168](https://github.com/AbdelStark/tessera/issues/168) | gpu: implement the streaming advice commitment on the GPU | gpu | p0 | m | RFC-0011, RFC-0018 |
| [#169](https://github.com/AbdelStark/tessera/issues/169) | bench: measure B200 unit costs with the v0.1 protocol | bench | p2 | m | RFC-0020 |
| [#170](https://github.com/AbdelStark/tessera/issues/170) | quant: produce and evaluate the 70B integer model | quant | p1 | m | RFC-0005 |
| [#171](https://github.com/AbdelStark/tessera/issues/171) | gpu: implement sumcheck round kernels for every relation kind | gpu | p0 | l | RFC-0010, RFC-0018 |
| [#172](https://github.com/AbdelStark/tessera/issues/172) | gpu: implement layer-range partitioning across eight GPUs with NCCL | gpu | p0 | l | RFC-0018 |
| [#173](https://github.com/AbdelStark/tessera/issues/173) | gpu: implement the GPU prover backend with streaming and the memory scheduler | gpu | p0 | l | RFC-0018 |
| [#174](https://github.com/AbdelStark/tessera/issues/174) | test: assert byte-identical proofs from the CPU and GPU backends | gpu | p0 | m | RFC-0018, s07-testing |
| [#175](https://github.com/AbdelStark/tessera/issues/175) | bench: measure end-to-end μ for 8B and 70B and re-fit the cost model | bench | p0 | m | RFC-0020, s08-performance |

## Milestone: v1.0 — Frontier: 405B on one 8xH100 node, hardened verifier, release

| # | Title | Area | Priority | Effort | RFC |
|---|-------|------|----------|--------|-----|
| [#176](https://github.com/AbdelStark/tessera/issues/176) | docs: build the documentation site from docs/ and rustdoc | docs | p1 | m | s09-release, s02-api |
| [#177](https://github.com/AbdelStark/tessera/issues/177) | infra: add an optional Prometheus endpoint and OTLP export | infra | p2 | s | s05-observability |
| [#178](https://github.com/AbdelStark/tessera/issues/178) | ci: add the release workflow with reproducible verifier builds | infra | p0 | l | s09-release, RFC-0019 |
| [#179](https://github.com/AbdelStark/tessera/issues/179) | protocol: implement live stream and directory transports on the prover side | protocol | p1 | m | RFC-0014 |
| [#180](https://github.com/AbdelStark/tessera/issues/180) | docs: write the integration guide for recorders and serving engines | docs | p1 | m | RFC-0015, RFC-0017 |
| [#181](https://github.com/AbdelStark/tessera/issues/181) | verifier: implement tv-transport with mutually authenticated TLS | verifier | p1 | m | RFC-0014, RFC-0019 |
| [#182](https://github.com/AbdelStark/tessera/issues/182) | verifier: harden resource limits and make every input path panic-free | verifier | p0 | m | RFC-0019, s06-security |
| [#183](https://github.com/AbdelStark/tessera/issues/183) | quant: produce and evaluate the 405B integer model | quant | p0 | l | RFC-0005 |
| [#184](https://github.com/AbdelStark/tessera/issues/184) | docs: write the operator guide for compile, commit, prove and verify | docs | p1 | m | s02-api, s01-architecture |
| [#185](https://github.com/AbdelStark/tessera/issues/185) | verifier: add fuzz targets and run the 24 CPU-hour campaign | verifier | p0 | m | RFC-0019, s06-security |
| [#186](https://github.com/AbdelStark/tessera/issues/186) | docs: write the verifier specification | verifier | p1 | l | RFC-0019, RFC-0009 |
| [#187](https://github.com/AbdelStark/tessera/issues/187) | security: obtain the external review of protocol and verifier | verifier | p0 | l | s06-security, RFC-0019, RFC-0013 |
| [#188](https://github.com/AbdelStark/tessera/issues/188) | prover: prove Llama-3.1-405B at 1K, 4K and 16K context on one 8xH100 node | prover | p0 | l | RFC-0018, RFC-0020, s08-performance |
| [#189](https://github.com/AbdelStark/tessera/issues/189) | bench: measure 405B verification time and memory against SC-3 | bench | p0 | s | RFC-0019, s08-performance |

## Tracking Issues

- [#2](https://github.com/AbdelStark/tessera/issues/2) [Tracking] Workspace layout and verifier independence (6 children; RFC-0001)
- [#3](https://github.com/AbdelStark/tessera/issues/3) [Tracking] Goldilocks field and quadratic extension (3 children; RFC-0002)
- [#4](https://github.com/AbdelStark/tessera/issues/4) [Tracking] Canonical encoding and artifacts (4 children; RFC-0003)
- [#5](https://github.com/AbdelStark/tessera/issues/5) [Tracking] Hashing, Merkle trees and Fiat–Shamir transcript (3 children; RFC-0004)
- [#6](https://github.com/AbdelStark/tessera/issues/6) [Tracking] Integer model D: quantization, calibration and fidelity (10 children; RFC-0005)
- [#7](https://github.com/AbdelStark/tessera/issues/7) [Tracking] Compiler IR, range analysis and gadget selection (7 children; RFC-0006)
- [#8](https://github.com/AbdelStark/tessera/issues/8) [Tracking] Public claim list and statement (4 children; RFC-0007)
- [#9](https://github.com/AbdelStark/tessera/issues/9) [Tracking] LogUp-GKR and the table registry (8 children; RFC-0008)
- [#10](https://github.com/AbdelStark/tessera/issues/10) [Tracking] Uniqueness gadgets (8 children; RFC-0009)
- [#11](https://github.com/AbdelStark/tessera/issues/11) [Tracking] Batched constant-depth proof schedule (9 children; RFC-0010)
- [#12](https://github.com/AbdelStark/tessera/issues/12) [Tracking] Streaming advice commitment (8 children; RFC-0011)
- [#13](https://github.com/AbdelStark/tessera/issues/13) [Tracking] Persistent weight commitment (6 children; RFC-0012)
- [#14](https://github.com/AbdelStark/tessera/issues/14) [Tracking] Zero knowledge (6 children; RFC-0013)
- [#15](https://github.com/AbdelStark/tessera/issues/15) [Tracking] Interaction model, soundness and parameter sets (6 children; RFC-0014)
- [#16](https://github.com/AbdelStark/tessera/issues/16) [Tracking] Transcript anchor and recorder log (8 children; RFC-0015)
- [#17](https://github.com/AbdelStark/tessera/issues/17) [Tracking] Commit-then-sample audits (3 children; RFC-0016)
- [#18](https://github.com/AbdelStark/tessera/issues/18) [Tracking] Scoring and the bound Û (6 children; RFC-0017)
- [#19](https://github.com/AbdelStark/tessera/issues/19) [Tracking] Witness engine and GPU prover runtime (19 children; RFC-0018)
- [#20](https://github.com/AbdelStark/tessera/issues/20) [Tracking] Independent verifier (5 children; RFC-0019)
- [#21](https://github.com/AbdelStark/tessera/issues/21) [Tracking] Cost model and unit-cost measurement (8 children; RFC-0020)
- [#22](https://github.com/AbdelStark/tessera/issues/22) [Tracking] Repository infrastructure, CI and release (14 children; s07-testing, s09-release)
- [#23](https://github.com/AbdelStark/tessera/issues/23) [Tracking] Observability, error model and security program (7 children; s04-errors, s05-observability, s06-security)
- [#24](https://github.com/AbdelStark/tessera/issues/24) [Tracking] Command-line interfaces (7 children; s02-api)

## Cross-Cutting Dependencies

Dependencies between subsystems that shape the schedule. Every issue lists its full `Depends on` and `Blocks` sets.

- [#36](https://github.com/AbdelStark/tessera/issues/36) blocks [#42](https://github.com/AbdelStark/tessera/issues/42) because the pull-request pipeline runs the verifier isolation gate, so isolation is enforced from the first merge.
- [#38](https://github.com/AbdelStark/tessera/issues/38) blocks [#49](https://github.com/AbdelStark/tessera/issues/49) because unit-cost measurements run on the self-hosted H100 runner with locked clocks.
- [#55](https://github.com/AbdelStark/tessera/issues/55) blocks [#66](https://github.com/AbdelStark/tessera/issues/66) because the v0.1 go/no-go decision needs the committed-element cost, the largest assumption in the source.
- [#63](https://github.com/AbdelStark/tessera/issues/63) blocks [#66](https://github.com/AbdelStark/tessera/issues/66) because the per-lookup cost drives the attention term at long context.
- [#35](https://github.com/AbdelStark/tessera/issues/35) blocks [#84](https://github.com/AbdelStark/tessera/issues/84) because the commitment report embeds cost-model projections computed from exact counts.
- [#71](https://github.com/AbdelStark/tessera/issues/71) blocks [#82](https://github.com/AbdelStark/tessera/issues/82) because the integer engine evaluates exponentials and SiLU through the generated tables, which define D.
- [#89](https://github.com/AbdelStark/tessera/issues/89) blocks [#91](https://github.com/AbdelStark/tessera/issues/91) because SC-7 bit-exactness compares the PyTorch integer reference with the Rust engine.
- [#101](https://github.com/AbdelStark/tessera/issues/101) blocks [#106](https://github.com/AbdelStark/tessera/issues/106) because the canonical advice layout follows the tiling's production order.
- [#78](https://github.com/AbdelStark/tessera/issues/78) blocks [#102](https://github.com/AbdelStark/tessera/issues/102) because the weight layout follows the lowered model's shapes and channel parameters.
- [#100](https://github.com/AbdelStark/tessera/issues/100) blocks [#107](https://github.com/AbdelStark/tessera/issues/107) because claim-list validation regenerates every table and compares digests.
- [#117](https://github.com/AbdelStark/tessera/issues/117) blocks [#128](https://github.com/AbdelStark/tessera/issues/128) because SP-7: the verifier driver must re-derive ranges before trusting any lookup.
- [#125](https://github.com/AbdelStark/tessera/issues/125) blocks [#145](https://github.com/AbdelStark/tessera/issues/145) because the simulator argument is written against the final transcript schedule.
- [#145](https://github.com/AbdelStark/tessera/issues/145) blocks [#148](https://github.com/AbdelStark/tessera/issues/148) because RFC-0013 gate: no zero-knowledge implementation merges before the reviewed simulator argument.
- [#145](https://github.com/AbdelStark/tessera/issues/145) blocks [#147](https://github.com/AbdelStark/tessera/issues/147) because weight masking consumes the same argument's budget accounting.
- [#142](https://github.com/AbdelStark/tessera/issues/142) blocks [#152](https://github.com/AbdelStark/tessera/issues/152) because the sampler PRF is evaluated in-circuit with the AES-128 gadget.
- [#138](https://github.com/AbdelStark/tessera/issues/138) blocks [#140](https://github.com/AbdelStark/tessera/issues/140) because sampling seeds bind the signed manifest, so sampling cannot start before the freeze.
- [#151](https://github.com/AbdelStark/tessera/issues/151) blocks [#158](https://github.com/AbdelStark/tessera/issues/158) because weight epochs and used blinding rows live in the verifier ledger.
- [#155](https://github.com/AbdelStark/tessera/issues/155) blocks [#161](https://github.com/AbdelStark/tessera/issues/161) because the bounded-mode study scores with the circuit-defined mixture predictor.
- [#134](https://github.com/AbdelStark/tessera/issues/134) blocks [#173](https://github.com/AbdelStark/tessera/issues/173) because the GPU backend implements the streaming plan first validated on CPU.
- [#183](https://github.com/AbdelStark/tessera/issues/183) blocks [#188](https://github.com/AbdelStark/tessera/issues/188) because 405B proofs need the calibrated 405B integer model.
- [#175](https://github.com/AbdelStark/tessera/issues/175) blocks [#188](https://github.com/AbdelStark/tessera/issues/188) because the 405B campaign starts from the re-fit cost model and measured pass plan.
- [#135](https://github.com/AbdelStark/tessera/issues/135) blocks [#185](https://github.com/AbdelStark/tessera/issues/185) because the fuzzing seed corpus includes the tamper corpus.
- [#186](https://github.com/AbdelStark/tessera/issues/186) blocks [#187](https://github.com/AbdelStark/tessera/issues/187) because the external review audits the verifier against its written specification.

## Critical path

The longest dependency chain, weighting effort `s` = 2, `m` = 5, `l` = 12 working days, is 147 contributor-days long. Parallel work elsewhere does not shorten it:

[#25](https://github.com/AbdelStark/tessera/issues/25) → [#33](https://github.com/AbdelStark/tessera/issues/33) → [#39](https://github.com/AbdelStark/tessera/issues/39) → [#46](https://github.com/AbdelStark/tessera/issues/46) → [#53](https://github.com/AbdelStark/tessera/issues/53) → [#60](https://github.com/AbdelStark/tessera/issues/60) → [#105](https://github.com/AbdelStark/tessera/issues/105) → [#113](https://github.com/AbdelStark/tessera/issues/113) → [#119](https://github.com/AbdelStark/tessera/issues/119) → [#145](https://github.com/AbdelStark/tessera/issues/145) → [#148](https://github.com/AbdelStark/tessera/issues/148) → [#154](https://github.com/AbdelStark/tessera/issues/154) → [#171](https://github.com/AbdelStark/tessera/issues/171) → [#172](https://github.com/AbdelStark/tessera/issues/172) → [#173](https://github.com/AbdelStark/tessera/issues/173) → [#175](https://github.com/AbdelStark/tessera/issues/175) → [#188](https://github.com/AbdelStark/tessera/issues/188) → [#189](https://github.com/AbdelStark/tessera/issues/189)

Issues on this chain are the schedule's bottleneck; staff them first.
