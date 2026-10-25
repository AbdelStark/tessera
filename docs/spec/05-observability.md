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
- Two output formats: human-readable to stderr (default) and JSON lines (`--log-format json`). One event per line, with fields `ts`, `level`, `target`, `event`, `span_path` and typed fields: `tessera.log-event/1`, [`spec/schemas/log-event-v1.schema.json`](../../spec/schemas/log-event-v1.schema.json). `--log-level` selects the most detailed level written, `info` by default.
- Event names are `tessera.<crate>.<event>` in snake case, for example `tessera.pcs.row_block_committed`.
- A message is a literal template, never formatted from values: a `{name}` in it stands for the field `name`, which both formats substitute. Every value an event shows is therefore a typed field.
- Levels: `error` for a failed operation; `warn` for a degraded but valid state (for example a hiding budget below 10%); `info` for phase boundaries and results; `debug` for per-stage detail; `trace` for per-tile detail, compiled out of release builds through `tracing`'s `release_max_level_debug` feature.

## 3. Data classification and redaction

| Class | Examples | Logs and metrics | Errors | Artifacts |
|---|---|---|---|---|
| PUBLIC | Shapes, ids, digests, roots, `Û`, parameter-set names, counts, timings, device models | Allowed | Allowed | Allowed |
| PROTOCOL | Sumcheck round messages, openings, column data, combined rows after blinding | Never logged; sizes only | Sizes only | Protocol artifacts only |
| PRIVATE | Execution plan, budget cursor detail, prover configuration | `debug` level only | Identifiers only | Private artifacts |
| SECRET | Weights, activations, prompt and output tokens, logits, lookup multiplicities, sampler and transcript keys, masking randomness, per-proof secret seeds, blinding rows, a verifier's coin-toss share before its reveal ([RFC-0067](../rfcs/RFC-0067-verifier-ledger-encoding.md#the-verifiers-own-sample) 'The pending toss'); and what the proving engines derive from them that no message carries: bound tables, partial products, contractions, tree levels, codewords | Never | Never | Secret artifacts only, never in reports |

Enforcement by type:

```rust
/// Holds a secret value. No Debug, Display, Serialize, or TCE Encode impl.
/// Access requires an explicit expose_secret() call, which is denied by a CI lint outside allowlisted modules.
pub struct Secret<T: Zeroize>(T);

/// Marker for values cleared for telemetry. Only Public<T> fields may appear in tracing events and PublicContext.
pub struct Public<T>(pub T);

/// Values derived from secret-class data, in a vector that clears every allocation it releases (OBS-6).
/// No Debug, Display, Clone or TCE Encode; lends its values, as a slice, to the code that holds it.
pub struct SecretBuf<T: Copy + Zeroize>(Vec<T>);
```

- **OBS-1.** Tracing events and `PublicContext` accept only `Public<T>` and primitive identifier types. `tessera_codec::event!` and `span!` are the only way the prover's crates emit events and spans: they take [`Field`](../../crates/tessera-codec/src/telemetry.rs) values, `Public<T>`, integers, `bool` and `&'static str`, and a literal message template. `cargo xtask lint` rejects `tracing`'s own macros and a `tracing` dependency elsewhere, and checks every event's and span's name. A `clippy.toml` disallowed-methods rule forbids `expose_secret` outside the witness engine, gadget witness generation, commitment encoding and key handling modules, and outside the proving engines' entry points that take secret-class tables by reference to their `Secret` (`Table::secret_base`, `Forest::secret`, `MatMulWeight::secret`, …). Those modules opt in with `#[allow(clippy::disallowed_methods)]`. `secret-access.toml` lists, each with a reason, the files that compute on secret-class values: the readers, which take values out of `Secret<T>`, and the borrowers (`borrows = true`), engine code that computes on tables a reader lends it by reference for the length of a call, such as the sumcheck kernel and `tessera-poly`'s evaluation. A listed file hands no secret-class value, or reference to one, to code outside the list, beyond arithmetic on single values. `cargo xtask lint` accepts the allowance, and any use of `expose_secret` by name under any `cfg`, only in a reader; it rejects a reader that names no accessor, and a borrower that names one, switches the rule off or is not a prover source. It also rejects lint and clippy configuration that would switch the rule off, and code pulled in other than through `mod` (`include!`, `#[path]`), which its scan could not see.
- **OBS-2.** Key material (AES keys, sampler keys, salts, signing seeds, per-proof seeds) is zeroized on drop. A key that lives as long as a session, or is moved, is held on the heap as `Secret<HeapKey<N>>` (`tessera-codec`), so that a move copies a pointer, not the key. A cipher's key schedule, which holds the key, is boxed for the same reason.
- **OBS-3. Canary scan.** The end-to-end CI job builds a tiny model whose weights and prompts contain canary patterns (distinctive 64-bit values and token ids). After proving and verifying, CI scans every log line, JSON report, protocol artifact and error message for each canary in decimal, hex and little-endian byte forms. Any match outside secret-class artifacts fails the build. A 64-bit canary, a run of eight weights, is looked for anywhere. A token id is short enough for chance to spell it: it is looked for as a whole word of text, and in binary artifacts, where a public integer such as a column index can equal it, as each recorded sequence of ids. The model is in [`spec/vectors/canary/`](../../spec/vectors/canary/README.md), and the scan is `e2e_redaction_canary_scan` ([the exact-CLI suite](../testing/exact-cli.md#the-canary-scan)).
- **OBS-4.** Debug dumps of secret tensors exist only behind the `unsafe-debug-dump` cargo feature, which release builds reject at compile time (`compile_error!` under `cfg(all(feature = "unsafe-debug-dump", not(debug_assertions)))`).
- **OBS-6.** A value derived from secret-class data is secret-class until a protocol message carries it. A proving engine holds the tables it derives, such as bound tables, products of table entries, contractions, tree levels, codewords and multiplicities, in `SecretBuf`s, allocated at their size where it is known; the commitment keeps the rows it commits in their `Secret`. A `SecretBuf` overwrites its whole allocation, spare capacity included, with zeros that an optimization barrier keeps the compiler from removing, when it drops, and the allocation it leaves when it grows; a `Secret<Vec<_>>` clears its vector's whole capacity, with volatile writes, when it drops. A derived value outside a buffer, such as a small array of sums, is cleared where it is held. Clearing cannot reach registers, stack copies, or a copy taken out of the buffer (`to_vec`); a reviewer treats such a copy as a leak.

## 4. Metrics

Metrics are collected in-process and written to the run manifest at the end of each run. An optional Prometheus text endpoint (`--metrics-addr`) exposes the same registry for long-running recorder and prover services.

| Name | Type | Labels | Unit |
|---|---|---|---|
| `tessera_phase_seconds` | histogram | `phase` (witness, round1, stage, opening, round3) | s |
| `tessera_stage_seconds` | histogram | `stage` | s |
| `tessera_passes_total` | counter | `pass` (p1, p2, p2b, p3, p3b, p4, p5) | passes |
| `tessera_pass_seconds` | histogram | `pass` | s |
| `tessera_committed_elements_total` | counter | `class` (advice, weight, mask) | elements |
| `tessera_lookups_total` | counter | `table` | lookups |
| `tessera_product_leaves_total` | counter | — | leaves |
| `tessera_table_rows_total` | counter | `table` | rows |
| `tessera_bytes_hashed_total` | counter | `domain` | bytes |
| `tessera_device_memory_peak_bytes` | gauge | `device` | bytes |
| `tessera_host_memory_peak_bytes` | gauge | — | bytes |
| `tessera_unit_cost_ps` | gauge | `kind` (commit, lookup, product_leaf) | ps |
| `tessera_glue_fraction` | gauge | — | ratio |
| `tessera_proof_bytes` | gauge | `part` (round1, sumcheck, openings, columns) | bytes |
| `tessera_weight_budget_remaining` | gauge | `weight_type`, `resource` (columns, blinding_rows) | count |
| `tessera_verifier_seconds` | histogram | `check` | s |
| `tessera_release_overruns_total` | counter | `message` (round1, round2, round3, proof), `slot` | releases |

Derived unit costs (`tessera_unit_cost_ps`) are computed per run as phase time divided by element counts, so every production run doubles as a unit-cost measurement ([RFC-0020](../rfcs/RFC-0020-cost-model.md#continuous-measurement)):
- `commit` is P1's commitment time, after the witness, per committed advice element.
- `lookup` and `product_leaf` share the leaf stage's time per leaf, over every lookup, table row and product leaf: the stage proves all their trees in one batched sumcheck per level, so one run cannot price them apart.
- `tessera_glue_fraction` is RFC-0020's glue, a ratio and not a time: the reduction stages' time over commitment and leaf time, the factor `1 + glue` the cost model applies.

Passes are [RFC-0018](../rfcs/RFC-0018-gpu-prover-runtime.md#pass-structure)'s, counted as they run.
- **The passes.** `p1` is the witness pass with its commitment, `p2` the leaf stage, `p2b` a grouped leaf stage's merge ([RFC-0054](../rfcs/RFC-0054-token-groups.md#pass-structure)), `p3` one reduction stage, `p3b` a windowed reduction stage's fold ([RFC-0055](../rfcs/RFC-0055-windowed-stages.md#pass-structure)), `p4` the opening pass and `p5` the column answers.
- **On CPU.** The CPU backend computes the witness in P1 and reads it back for the others, so a CPU proof regenerates once and still counts every pass.

The registry is `tessera_codec::metrics`: fixed names, and labels that are enum variants, table identifiers or stage numbers, so no sample carries secret-class data (OBS-1). The CPU prover records phases, stages, passes, committed elements, lookups, table rows, product leaves and proof bytes; the CLI adds the host memory peak where `/proc` reports it. Device memory needs the GPU backend, `tessera_verifier_seconds` the verifier (#446), and the weight budget and release overruns `tessera-protocol/2`, which spends the budget and releases on a schedule. `tessera_bytes_hashed_total` waits for a per-thread count: an atomic per hash would cost the Merkle and column-chain loops. Phase and stage times depend on secret data, so they and the unit costs derived from them stay on the prover's side of TB-1 ([06 §3](06-security.md#3-trust-boundaries)). A verifier sees only when messages are released, on the schedule of INV-ARCH-2 ([RFC-0018 'Release schedule'](../rfcs/RFC-0018-gpu-prover-runtime.md#release-schedule)).

## 5. Tracing

- Spans: `compile`, `commit_weights`, `prove{batch}`, `witness_pass{layer_tile}`, `round1`, `stage{id}`, `opening{commitment}`, `round3`, `verify{proof_digest}`, `check{kind}`.
- Span fields are PUBLIC class only.
- The CPU prover computes its witness as one tile of every layer, so `witness_pass` has one span per proof until token groups regenerate it group by group (#134). The verifier's `verify` and `check` spans are #446; its audit trail is §7's report.
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

The manifest is `tessera.run-manifest/1` ([`spec/schemas/run-manifest-v1.schema.json`](../../spec/schemas/run-manifest-v1.schema.json)), written at the command's one exit point whether it succeeded or failed:
- **Where.** In a directory of the operator's own: `compile`'s output directory, which holds the private plan; beside the secret weight state for `commit-weights` and `prove`; beside the report for `bench model --out`. `--run-manifest PATH` names another. Never in a Live exchange directory.
- **Class.** It is private-class (§3): its timings depend on secret data and stay on the prover's side of TB-1 (§4).
- **Artifacts.** Each is named by path, schema and class, and by its envelope digest, except a secret-class artifact: a digest of secret data is derived secret data (OBS-6), so the manifest names it without one. A JSON report carries its schema identifier.
- **Seeds.** A Live session's coin-toss seed, which both transcripts absorb; the masks' seed never.

**OBS-5.** Two runs with identical manifests, apart from timings and host fields, must produce identical output artifact digests, given pinned masking randomness in test builds and identical seeds. CI checks this on the tiny-model end-to-end test: `tessera-cli`'s `run_manifest` test runs `compile`, `commit-weights` and an NI `prove` twice each, the last with `--test-mask-seed` (feature `test-rng`, refused by release builds).

## 7. Verifier audit trail

`tessera-verify verify --out VERDICT.json` writes `tessera.verdict/1`: protocol version, parameter set, statement digests, each check in execution order with its outcome and duration, `Û` per request and per batch, and the final verdict. The report contains no protocol-class data, so a verifier can publish it. The library core serializer uses null durations so identical inputs yield identical bytes. Its accepted report records all completed protocol/1 phases and explicit false anchor/ledger/zero-knowledge flags; a rejection or operational error carries its exact reason but no partial core check trace.

The reference CLI's optional `execution` context records measured `load`,
`statement_pin` and `verification` phases, the authenticated-pin flag and exit
code. Internal per-check durations remain null; their timings are not measured
by this interface. The overall duration ends at verdict preparation, before a
successful report write. A contained panic can omit the context, and report
allocation failure returns a nonzero status without a report. See the
[reference CLI guide](../verifier/reference-cli.md).
