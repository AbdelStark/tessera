# 05 — Observability

- Status: Accepted for v1 planning
- Governs: logging, metrics, tracing, run manifests, data classification and redaction
- Related: [04 Error model](04-error-model.md), [06 Security](06-security.md), [08 Performance budget](08-performance-budget.md)

## 1. Goals

- Diagnose prover performance and failures per phase, stage, layer tile and device without access to secret data.
- Reproduce any run from its manifest: same inputs, versions, parameters and public seeds.
- Give the verifier an audit trail of every check it ran.
- Make secret leakage through telemetry a type error, and detect any residual leak in CI.

## 2. Logging

- Rust uses the `tracing` crate (0.1.x). Python uses the standard `logging` module with a JSON formatter.
- Two output formats: human-readable to stderr (default) and JSON lines (`--log-format json`). One event per line, with fields `ts`, `level`, `target`, `event`, `span_path` and typed fields.
- Event names are `tessera.<crate>.<event>` in snake case, for example `tessera.pcs.row_block_committed`.
- Levels: `error` for a failed operation; `warn` for a degraded but valid state (for example a hiding budget below 10%); `info` for phase boundaries and results; `debug` for per-stage detail; `trace` for per-tile detail, compiled out of release builds through `tracing`'s `release_max_level_debug` feature.

## 3. Data classification and redaction

| Class | Examples | Logs and metrics | Errors | Artifacts |
|---|---|---|---|---|
| PUBLIC | Shapes, ids, digests, roots, `Û`, parameter-set names, counts, timings, device models | Allowed | Allowed | Allowed |
| PROTOCOL | Sumcheck round messages, openings, column data, combined rows after blinding | Never logged; sizes only | Sizes only | Protocol artifacts only |
| PRIVATE | Execution plan, budget cursor detail, prover configuration | `debug` level only | Identifiers only | Private artifacts |
| SECRET | Weights, activations, prompt and output tokens, logits, lookup multiplicities, sampler and transcript keys, masking randomness, per-proof secret seeds, blinding rows | Never | Never | Secret artifacts only, never in reports |

Enforcement by type:

```rust
/// Holds a secret value. No Debug, Display, Serialize, or TCE Encode impl.
/// Access requires an explicit expose_secret() call, which is denied by a CI lint outside allowlisted modules.
pub struct Secret<T: Zeroize>(T);

/// Marker for values cleared for telemetry. Only Public<T> fields may appear in tracing events and PublicContext.
pub struct Public<T>(pub T);
```

- **OBS-1.** Tracing events and `PublicContext` accept only `Public<T>` and primitive identifier types. A custom lint (`dylint` or a `clippy.toml` disallowed-methods rule) forbids `expose_secret` outside the witness engine, gadget witness generation, commitment encoding and key handling modules.
- **OBS-2.** Key material (`Secret<[u8; 16]>` AES keys, sampler keys, per-proof seeds) is zeroized on drop.
- **OBS-3. Canary scan.** The end-to-end CI job builds a tiny model whose weights and prompts contain canary patterns (distinctive 64-bit values and token ids). After proving and verifying, CI scans every log line, JSON report, protocol artifact and error message for each canary in decimal, hex and little-endian byte forms. Any match outside secret-class artifacts fails the build.
- **OBS-4.** Debug dumps of secret tensors exist only behind the `unsafe-debug-dump` cargo feature, which release builds reject at compile time (`compile_error!` under `cfg(all(feature = "unsafe-debug-dump", not(debug_assertions)))`).

## 4. Metrics

Metrics are collected in-process and written to the run manifest at the end of each run. An optional Prometheus text endpoint (`--metrics-addr`) exposes the same registry for long-running recorder and prover services.

| Name | Type | Labels | Unit |
|---|---|---|---|
| `tessera_phase_seconds` | histogram | `phase` (witness, round1, stage, opening, round3) | s |
| `tessera_stage_seconds` | histogram | `stage` | s |
| `tessera_committed_elements_total` | counter | `class` (advice, weight, mask) | elements |
| `tessera_lookups_total` | counter | `table` | lookups |
| `tessera_product_leaves_total` | counter | — | leaves |
| `tessera_bytes_hashed_total` | counter | `domain` | bytes |
| `tessera_device_memory_peak_bytes` | gauge | `device` | bytes |
| `tessera_host_memory_peak_bytes` | gauge | — | bytes |
| `tessera_unit_cost_ps` | gauge | `kind` (commit, lookup, product_leaf, glue) | ps |
| `tessera_proof_bytes` | gauge | `part` (round1, sumcheck, openings, columns) | bytes |
| `tessera_weight_budget_remaining` | gauge | `weight_type`, `resource` (columns, blinding_rows) | count |
| `tessera_verifier_seconds` | histogram | `check` | s |

Derived unit costs (`tessera_unit_cost_ps`) are computed per run as phase time divided by element counts, so every production run doubles as a unit-cost measurement ([RFC-0020](../rfcs/RFC-0020-cost-model.md#continuous-measurement)).

## 5. Tracing

- Spans: `compile`, `commit_weights`, `prove{batch}`, `witness_pass{layer_tile}`, `round1`, `stage{id}`, `opening{commitment}`, `round3`, `verify{proof_digest}`, `check{kind}`.
- Span fields are PUBLIC class only.
- Optional OTLP export sits behind the `otel` feature for operators who run collectors. The default build has no network telemetry.

## 6. Run manifest

Every `compile`, `commit-weights`, `prove`, `verify` and `bench` invocation writes `run-manifest.json` (`tessera.run-manifest/1`):

- tool name, semantic version, git commit, build profile, enabled cargo features;
- toolchain versions (rustc, CUDA runtime and driver, Python, torch);
- host CPU model and core count, GPU models, memory, ECC state;
- parameter set, protocol version, input and output artifact digests;
- public seeds (coin-toss transcript digests, never per-proof secret seeds);
- per-phase timings and the metrics snapshot;
- exit code and error code, if any.

**OBS-5.** Two runs with identical manifests, apart from timings and host fields, must produce identical output artifact digests, given pinned masking randomness in test builds and identical seeds. CI checks this on the tiny-model end-to-end test.

## 7. Verifier audit trail

`tessera-verify --out VERDICT.json` writes `tessera.verdict/1`: protocol version, parameter set, statement digests, each check in execution order with its outcome and duration, `Û` per request and per batch, and the final verdict. The report contains no protocol-class data, so a verifier can publish it.
