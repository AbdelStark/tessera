# 07 — Testing Strategy

- Status: Accepted for v1 planning
- Governs: test pyramid, conformance vectors, cryptographic and ML-specific tests, CI gates
- Related: [04 Error model](04-error-model.md), [08 Performance budget](08-performance-budget.md), each RFC's Testing Strategy section

## 1. Principles

1. **Two implementations, one specification.** The verifier shares no code with the prover. The prover and verifier sides of every primitive are tested against each other and against shared vectors that are data, not code.
2. **Soundness is tested adversarially.** For every check the verifier performs, a test supplies a proof that violates exactly that check and asserts the expected `RejectReason`.
3. **Uniqueness is tested by construction of alternatives.** For every gadget, a test builds an alternative witness that satisfies all other constraints and asserts the gadget rejects it.
4. **Bit-exactness is tested differentially.** The CPU reference engine, the PyTorch integer reference and the GPU engine must agree bit for bit.
5. **ML evaluation follows hygiene rules.** Calibration and evaluation data never overlap; seeds, data versions and library versions are recorded; metrics carry confidence intervals.

## 2. Test pyramid

| Level | Scope | Tooling | Runs |
|---|---|---|---|
| Unit | One function or type | `cargo test`, `pytest` | Every PR |
| Property | Algebraic laws and codec round-trips over random inputs | `proptest` (Rust), `hypothesis` (Python) | Every PR (256 cases), nightly (65,536 cases) |
| Conformance | Shared vectors in `spec/vectors/` checked by prover side, verifier side and Python | Vector runners in each workspace | Every PR |
| Component | One protocol (sumcheck, LogUp, PCS, gadget) end to end, prover against independent verifier | `cargo test` with verifier crates as dev-dependencies | Every PR |
| End to end | Compile, commit, prove, verify on tiny models | CLI-driven tests | Every PR (tiny), nightly (8B CPU subset) |
| Adversarial | Tamper corpus, alternative witnesses, malformed artifacts | Dedicated test crates; `cargo-fuzz` | Every PR (corpus), nightly (fuzz 1 h), release (fuzz 24 CPU-h) |
| GPU | Kernel correctness against CPU reference; determinism across tilings | Self-hosted H100 runner | Nightly and on `gpu`-labeled PRs |
| Performance | Microbenchmarks and end-to-end `μ` | `cargo bench -p tessera-bench`, `tessera bench` | Nightly; regression gate ([08 §6](08-performance-budget.md#6-regression-gates)) |
| ML fidelity | `D` versus reference float model | `tessera-quant eval` | On quantization changes; release |

## 3. Unit and property tests

Required properties, each a named test:

- Field: ring axioms, inverse, `Fp2` multiplication against schoolbook in `Fp[X]/(X^2-7)`, canonical encoding round-trip, rejection of encodings `≥ P` (RFC-0002).
- Codec: decode∘encode = id on random values of every schema; encode∘decode = id on bytes or a canonical-form error; truncation at every byte offset yields `ERR-CODEC-TRUNCATED` (RFC-0003).
- Transcript: identical absorb sequences give identical challenges; any single-byte difference changes all later challenges; challenge distribution passes a chi-square test on 2^20 samples reduced mod 2^16 (RFC-0004).
- Polynomials: MLE evaluation matches brute force on random tables up to 2^12 entries; `eq` table identities.
- Sumcheck: honest proofs verify for degrees 1–4 and 1–24 variables; any single coefficient change is rejected.

## 4. Soundness and uniqueness tests

- **Tamper corpus.** For a valid tiny-model proof, generate one mutated proof per check: flip one round coefficient per stage, one Merkle sibling, one column symbol, one combined-row symbol, one claimed evaluation, one PoW nonce, one anchor digest, one `Û` claim. Each must be rejected with the matching `RejectReason`. The corpus lives in `spec/vectors/tamper/` and is regenerated when the protocol version changes.
- **Alternative witnesses.** For each gadget in RFC-0009: a non-maximal `c_q`; `R_q ± 1`; `rsqrt ± 1`; a wrong split remainder; a non-one-hot token vector; a wrong inverse-CDF index; a wrong argmax under ties. A malicious witness generator substitutes the alternative and runs the honest prover; the verifier must reject.
- **Soundness calibration in a small field.** The field traits admit a test field (`P_test = 2^16 + 1`, two-adicity 16). Over it, a cheating prover that guesses challenges succeeds with empirical frequency within a 99% binomial confidence interval of the analytic bound (`d/|F|` per sumcheck round, `(N+T)/|F|` for LogUp). This checks that the analysis and the implementation agree.
- **Claim-list soundness.** Claim lists with a lookup key whose worst-case range exceeds its table, or that can wrap modulo `P`, must be rejected by the verifier's range re-derivation (SP-7).

## 5. Differential and conformance tests

| Pair | Compared quantity | Models |
|---|---|---|
| Prover-side vs verifier-side codec | Bytes and decode results on all golden vectors | All schemas |
| Prover-side vs verifier-side table generators | Table digests | Every table kind and parameter set in vectors |
| Rust CPU engine vs PyTorch integer reference | Every committed tensor and the logits, bit for bit | Tiny models, Llama-3.1-8B (first 2 layers in CI, full nightly) |
| CPU engine vs GPU engine | Every committed tensor, bit for bit | Tiny models, 8B; tensor-parallel degrees 1, 2, 8 |
| Compiler vs verifier claim-list recompilation | Constraint system digest | All conformance models |
| Python TCE writer vs Rust TCE reader | Decoded `IntegerModel` equality | All conformance models |

**Tiny conformance models** live in [`spec/vectors/models/`](../../spec/vectors/models/README.md): Llama-3.1-architecture models with random INT8 weights (2 layers, `d_model` 64, 4 heads, 2 KV heads, vocab 512, in both profiles; and 4 layers, `d_model` 256, with a 16-bit residual) plus their `IntegerModel` artifacts, inputs, and the residual at each layer boundary and the logits `D` gives. A seeded Python oracle writes them from RFC-0005's semantics, pins them by digest, and makes their inputs reach every clamp `D` has. The compiler's lowering reproduces their logits bit for bit.

## 6. End-to-end tests

- `e2e_exact_tiny`: compile, commit weights, record transcripts with the reference recorder, freeze, sample, prove (live and non-interactive), verify; assert `Accept` and `Û = 0`.
- `e2e_exact_mismatch`: one output token altered before recording; assert the prover returns `ERR-WITNESS-EXACT-MISMATCH`.
- `e2e_bounded_tiny`: outputs from a perturbed float model; assert `Accept` with `Û` equal to an independently computed value.
- `e2e_anchor_mismatch`: transcript altered after recording; assert `Reject(Anchor)`.
- `e2e_budget`: run proofs until the weight hiding budget is exhausted; assert `ERR-WITNESS-BUDGET`, refresh, verify the linking proof, continue.
- `e2e_reproducible`: two runs with pinned randomness give identical artifact digests (OBS-5).
- `e2e_redaction`: canary scan (OBS-3).

## 7. Zero-knowledge tests

- **Simulator test.** For tiny models, a simulator given only the public statement produces transcripts; a distinguisher suite (per-coordinate chi-square on revealed field elements, revealed-column statistics) must not separate simulated from real transcripts at significance 0.001 over 10^4 runs. This tests the implementation of the masking, not the proof of zero knowledge, which is a document ([RFC-0013](../rfcs/RFC-0013-zero-knowledge.md#simulator-argument)).
- **Budget tests.** Opening more distinct weight columns than `P_W`, or reusing a blinding row, is impossible through the public API and raises `ERR-INTERNAL` if forced through internals.

## 8. ML-specific tests

- **Data hygiene.** Calibration and evaluation datasets are specified by name, revision and content digest. A test asserts their document sets are disjoint by content hash.
- **Determinism.** `tessera-quant compile` run twice with the same seed yields the same `IntegerModel` digest (CPU path).
- **Fidelity regression.** For 8B: held-out perplexity of `D` relative to the BF16 reference and greedy token agreement over 512 prompts, with 95% bootstrap confidence intervals. A change that worsens either metric beyond its interval fails the gate.
- **Range soundness.** The witness engine asserts every tensor stays within its declared range during the 8B nightly run.

## 9. CI gates

| Gate | Blocks merge | Command |
|---|---|---|
| Format and lint | Yes | `cargo fmt --check`, `cargo clippy -D warnings`, `ruff check`, `ruff format --check` |
| Types | Yes | `pyright --strict` on `python/` |
| Unit, property, conformance, component | Yes | `cargo test --workspace` (both workspaces), `pytest` |
| Conformance vectors current | Yes | `cargo xtask gen-vectors --check`, also a test of `xtask`: the codec vectors match their cases, and every oracle `spec/vectors/*/generate.py`, rerun on a copy, writes its files byte for byte |
| Tiny end-to-end and tamper corpus | Yes | `cargo test -p tessera-e2e` |
| Verifier isolation | Yes | `cargo xtask check-isolation` ([RFC-0001](../rfcs/RFC-0001-workspace-and-verifier-independence.md#enforcement)) |
| Dependency policy | Yes | `cargo deny check` (both workspaces); `cargo vet` (verifier) |
| MSRV | Yes | Build with the minimum supported Rust version |
| Docs | Yes | `cargo doc -D warnings`; link check on `docs/` |
| GPU correctness | Yes for `gpu`-labeled PRs | Self-hosted runner |
| Performance regression | Nightly, opens an issue on failure | `tessera bench` |
