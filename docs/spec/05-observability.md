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

- The prover's Rust crates use the `tracing` crate (0.1.x). The verifier writes the same lines with its own code, below. Python uses the standard `logging` module with a JSON formatter.
- Two output formats: human-readable to stderr (default) and JSON lines (`--log-format json`). One event per line, with fields `ts`, `level`, `target`, `event`, `span_path` and typed fields: `tessera.log-event/1`, [`spec/schemas/log-event-v1.schema.json`](../../spec/schemas/log-event-v1.schema.json). `--log-level` selects the most detailed level written, `info` by default.
- Event names are `tessera.<crate>.<event>` in snake case, for example `tessera.pcs.row_block_committed`.
- A message is a literal template, never formatted from values: a `{name}` in it stands for the field `name`, which both formats substitute. Every value an event shows is therefore a typed field.
- Levels: `error` for a failed operation; `warn` for a degraded but valid state (for example a hiding budget below 10%); `info` for phase boundaries and results; `debug` for per-stage detail; `trace` for per-tile detail, compiled out of release builds through `tracing`'s `release_max_level_debug` feature.
- **A human line** is the message with its fields substituted. It follows `error: ` or `warn: ` at those levels, and the level and the span path at `debug` and `trace`. The span path is each span from the outermost, as `name{field=value …}` (just `name` when the span has no fields), joined by `:`, as in `debug prove{batch=ab}:stage{id=1}: rounds`. The fields the message does not name follow it, as ` field=value`.
- **The verifier's lines.** Every crate the verifier uses is audit surface (RFC-0001, DEP-1), so `tessera-verify` (`tv-cli`) writes both formats itself, with the same rules, and adds no crate for them: no `tracing`.
  - **Options.** Every command takes `--log-format {human,json}` and `--log-level {error,warn,info,debug,trace}`, `info` by default. Each is given at most once, before or after the command, and both are read before the command's other options, so a refused option is logged in the format asked for. A repeated or invalid value of either is refused, `ERR-USAGE`, in the human format.
  - **Human lines are today's.** Each verifier message starts with its own prefix. So at `error` and `warn` a human line has no level prefix, and at no level does it append the fields its message does not name. At `debug` and `trace` the level and the span path come first, as on the prover's side. With `--json`, whose stdout carries the result, the human format writes no line, as before. The JSON format writes every line its level admits, with or without `--json`. The lines reach stderr when the command ends, as its text does today, each with the time of its event.
  - **The events.** Fields are integers, or the text of identifiers, codes, digests and the command's own refusals, all public:

    | Event | Level | Spans | Fields | Message |
    |---|---|---|---|---|
    | `tessera.tv_cli.command_failed` | `error` | none | `command`, `code`, `detail` | `error ({code}): {detail}` |
    | `tessera.tv_cli.check_completed` | `debug` | `verify`, `check` | `step`, `outcome`, `duration_ns` | `step {step} {outcome} in {duration_ns} ns` |
    | `tessera.tv_cli.verified` | `info` | `verify` | `status`, `code`, `exit_code`, `duration_ns` | `verify: completed in {duration_ns} ns; protocol/1, anchor/ledger/ZK unchecked` |

  - **`command_failed`** is a command that fails before or outside verification: options refused, or an input, output or audit step that fails. It is written wherever today's `error (CODE): …` line is. `command` is the command's name, `tessera-verify` before one is read, as for a refused log option. `code` is the error's code, and `detail` that line's text: a refusal's fixed words, sometimes with the operator's own argument or a typed error's message.
  - **`check_completed`** ends each check of RFC-0019 'Check order' that runs, in its `check` span (§5). `step` is its number, 1 to 13. `outcome` is `pass`, `not_applicable` (step 5 in `tessera-protocol/1`), `reject` (the check that rejects) or `inconclusive` (the verifier's own failure). `duration_ns` is the check's measured time, 0 for step 5.
    - **What a check spans.** It spans the driver's work from its start to the next check's start. A fault judged there is reported in that check, with the verdict's own reason. Before the core runs, the command's reading of its inputs and pins is step 1, and its refusal of the parameter set is step 2.
    - **In `live`**, the session's start reports step 2. Each message's call then reports every check it runs, including those it repeats from earlier messages. A session that accepts reports steps 1 to 6, then 1 to 10, then 1 to 13.
  - **`verified`** ends `verify` and `live`. `status` is `accept`, `reject` or `inconclusive`. `code` is the rejection's `REJ-…` code or the error's `ERR-…` code, absent on `accept`. `exit_code` is the exit code (02 §2.4), and `duration_ns` the time from the command's start to the event.

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
| `tessera_verifier_seconds` | histogram | `check` (the checks of §5's `check` span) | s |
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
- **The verifier's spans** (§2, 'The verifier's lines'):
  - **`verify{proof_digest}`** spans `tessera-verify verify` and `live` from their options' parsing to the verdict. `proof_digest` is the NI proof's envelope digest (RFC-0003 'Envelope') in lowercase hexadecimal, recorded once that envelope's digest verifies, even when its payload then fails to decode. A Live session, which has no proof file, records none, nor does a proof whose envelope fails.
  - **`check{kind}`** spans one check of RFC-0019 'Check order', inside `verify`, in the order the checks run. A check that does not run has no span. `kind` is the check's name in the verdict report (§7): `input`, `params`, `claims`, `statement`, `weight_reservation`, `transcript`, `leaf`, `stages`, `evaluations`, `queries`, `columns`, `exact_bound`, `complete`. In `live`, a check spans the work it does on each message, so steps 1–6 complete before `SeedReveal` (RFC-0019).
- Span fields are PUBLIC class only.
- The CPU prover computes its witness as one tile of every layer, so `witness_pass` has one span per proof until token groups regenerate it group by group (#134). The verifier's audit trail is still §7's report: its spans are diagnostics, and the report's per-check durations stay null.
- Optional OTLP export sits behind the `otel` feature for operators who run collectors. The default build has no network telemetry.

## 6. Run manifest

Every `compile`, `commit-weights`, `prove` and `bench` invocation writes `run-manifest.json` (`tessera.run-manifest/1`), and `tessera-verify verify` writes it when asked:

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

**The verifier's manifest.** `tessera-verify verify --run-manifest PATH` writes one at `PATH`, and only there: a verifier writes no file it is not named.
- **Where.** `PATH` must differ from every input and from `--out`, else `ERR-USAGE` before verification. When either output path is refused, neither file is written. A `PATH` whose directory does not exist fails before verification too, `ERR-RES-IO`, as `--out` does. It is written atomically, as the report is, after the report, so the exit code it records is final. A manifest that cannot be written fails the command, as on the prover's side: stdout's verdict becomes `Inconclusive` with the write's error and the command exits with its code. A report already written stays.
- **Class.** It is public-class: the verifier computes only on public and protocol data, so its timings depend on nothing secret. It names every artifact by its envelope digest.
- **Its fields:**
  - `tool`: `tessera-verify` and `tv-cli`'s version. Then the build's commit, from `TESSERA_GIT_COMMIT` at build time, null when unset or not 40 hexadecimal digits. The profile is `release` when debug assertions are off, else `debug`. The features are the test features the build enables, `tv-core/test-params` when it accepts test parameters;
  - `toolchain`: the `rustc` that built it, from `TESSERA_RUSTC_VERSION` at build time, and null when unset, and null for the rest;
  - **Why build-time variables.** RFC-0001 refuses a build script in a verifier crate, since build-time code could read any file (`cargo xtask check-isolation`). So the release build or CI sets both variables, as `git rev-parse HEAD` and `rustc --version` give them;
  - `host`: the operating system, the architecture and the available parallelism as `cores`. `cpu_model` and `memory_bytes` are what the platform reports without a crate (Linux's `/proc`, macOS's `sysctl`), else null. `gpus` is empty, and `ecc` null;
  - `command` `verify`, `arguments` the command line after the program's name, `protocol` `tessera-protocol/1`, and `params` the claim list's parameter set once it has decoded, else null;
  - `inputs`: the claim list (`ClaimList`, public), the weight commitment (`WeightCommitmentPublic`, public) and the proof (`NiProof`, protocol), each with its envelope digest once its envelope verifies, else null;
  - `outputs`: the report at `--out` once written (`tessera.verdict/1`, public, no digest);
  - `public_seeds`: empty, as an NI proof absorbs none;
  - `metrics`: one `tessera_verifier_seconds` histogram sample per check that ran, its label `check` the check's name (§5), with `count` 1 and its time as `sum`, `min` and `max`;
  - `exit_code`, and `error_code` the verdict's code: `REJ-…` for a rejection, `ERR-…` for an inconclusive verdict, null on accept.

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
