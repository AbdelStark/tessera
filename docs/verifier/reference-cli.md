# Protocol/1 verifier CLI

The v0.3 reference CLI verifies one exact-greedy request in NI or directory Live mode through
the independent verifier. It regenerates table catalogs and retains the existing
parameter calculator. Protocol/1 does not hide weights or tokens. Test-small
parameters certify nothing and are accepted only by an explicit test build.

```sh
cd verifier
cargo build -p tv-cli --locked
target/debug/tessera-verify verify \
  --claims claims.tcl --weights weights.twr --weights-root "$TRUSTED_ROOT" \
  --statement-digest "$TRUSTED_STATEMENT" --proof proof.tnp --bound 0 \
  --out verdict.json --json
```

`TRUSTED_STATEMENT` is the 32-byte, hexadecimal
`H("statement", TCE(PublicStatement))` selected by the application. Obtain it
independently of the proof. It binds the entire expected statement, including
the claim list, epoch, manifest commitment, request identities/lengths and
predictor. The CLI authenticates the decoded statement against this pin before
using its content as the verifier's expected statement. Supplying a digest
extracted from an untrusted proof supplies no independent expectation.
`TRUSTED_ROOT` is the caller's trusted weight root and must agree with the
public epoch artifact. The parameter identity comes from the caller's claim
list. A pin/flag/artifact contradiction is an input-policy usage error, not an
accepted or cryptographically judged proof.

`--bound` is a nonnegative number of bits representable exactly in increments
of 1/256: `0`, `1.5` and `0.00390625` are valid; `0.1`, exponents and rounded
values are refused. No floating-point parsing enters the verification policy.

The command's process exit code is authoritative:

| Code | Meaning |
|---|---|
| 0 | Supported proof accepted; test parameters still certify nothing |
| 10 | Well-formed proof/claim list rejected with the existing exact reason |
| 20 | I/O, resource policy or cooperative deadline prevented a result |
| 64 | Invalid arguments, unsupported capability or contradictory trusted pins |
| 65 | Input artifact failed canonical envelope/payload decoding |
| 70 | Internal invariant failure or contained panic |

Every nonzero status means not certified. A failed report write also returns
a nonzero status, even if the mathematical checks passed. Report writes use a
temporary file and atomic rename; an output path that aliases any input is
refused before writing. Error text contains public codes, not protocol values
or panic payloads. Input files must be regular local files. Help is a separate
invocation (`verify --help`); a help-looking value inside a verification request
cannot turn it into a successful help command.

The default policy allows artifacts up to 8 GiB, a 64 GiB conservative memory
allowance and 300 seconds of cooperative verification time. Override with
`--max-artifact-bytes`, `--max-memory-bytes` and `--timeout-ms`. The CLI reserves
an input/decoded-object allowance of one eighth of the memory policy, checks
lengths before allocation and detects input growth while reading. These are
conservative allowances and phase deadlines, not hard peak-RSS or blocking-I/O
guarantees. Advice and weight openings carry the same deadline through NTTs,
column-chain hashing (between blocks and each 128-symbol piece of a long block),
coefficient construction and consistency folds (at most 4,096 entries per check).
Codeword copying, zero initialization and mask application also poll it.
Merkle multiproofs check before work, at most every 4,096 validation/counting/tree
steps and before releasing successful authentication. These
checks return `ERR-RES-TIME`. NI and Live artifact hashing polls each 64 KiB block;
payload reads and sequence decoding poll at most every 4,096 operations, with
additional checks through long ASCII/map comparisons and before releasing a value.
Protocol size counting, encoding and public-statement hashing also check entry,
at most every 4,096 primitive writes and completion, stopping nested traversal on expiry.
Live message decoding and cached replay retain the session's original deadline;
idle expiry remains `ERR-PROTO-TIMEOUT`. Other verification loops and blocking I/O still need
hardening. Full hardening remains [#182](https://github.com/AbdelStark/tessera/issues/182).

`--json` prints one result. `--out` always writes a
[`tessera.verdict/1`](../../spec/schemas/verdict-v1.schema.json) object. Existing
core per-check durations stay null. The optional `execution` object records
measured CLI `load`, `statement_pin` and `verification` phases, the selected
exit code and whether the input pins were authenticated. Its overall duration
covers evaluation through verdict preparation, before a successful report
write; it does not invent timings for individual internal verifier checks.
On a contained panic the core inconclusive report may omit this context. If
the report itself cannot be allocated, exit 20 is returned without a report.

Logs go to stderr ([05 §2](../spec/05-observability.md#2-logging), 'The
verifier's lines'). The default human format writes the lines above.
`--log-format json` writes one `tessera.log-event/1` object per line
([schema](../../spec/schemas/log-event-v1.schema.json)), even with `--json`.
`--log-level debug` adds one line per check that runs, inside the
`verify{proof_digest}` and `check{kind}` spans, with its outcome and measured
time. Every command takes both options.

`verify --run-manifest PATH` also writes a public-class `tessera.run-manifest/1`
([05 §6](../spec/05-observability.md#6-run-manifest)): the build, the host, the
inputs' envelope digests, each check's `tessera_verifier_seconds` and the exit
code. `PATH` must differ from the inputs and `--out`. It is written after the
report, and a manifest that cannot be written fails the command.

```sh
target/debug/tessera-verify tables --claims claims.tcl --json
target/debug/tessera-verify params --params ni-100 --json
target/debug/tessera-verify audit plan --policy docs/perf/examples/audit-policy.json \
  --traffic docs/perf/examples/audit-traffic.json --mu MU.json --json
target/debug/tessera-verify audit report --ledger LEDGER.tvl --sample SAMPLE.tas \
  --out report.json --json
```

`audit plan` computes a policy's detection from the policy and the traffic alone, and the
amortized overhead and target rates from a `tessera.cost-model/1` report at the traffic's mean
lengths (`--mu`), with its own binary64 arithmetic (`tv_core::audit_plan`); without `--mu` the
overhead figures of `tessera.audit-plan/1` are `null`
([example](../perf/examples/audit-plan-detection.json)). `audit report` reads the ledger and the
sample, the ledger under `tv-codec`'s message limits as every writer keeps it and the sample at
most 1 GiB, and writes `tessera.audit-report/1` atomically: requests the ledger
records accepted, with their `Û`, and the others `DeadlineMissed`. The sample must be one the
ledger's `derived` records as the verifier's own, and the ledger a regular file of one name,
neither a symbolic link nor a file of several hard links, which is refused before it is opened
(exit 64, [RFC-0067](../rfcs/RFC-0067-verifier-ledger-encoding.md)). It exits 0 only when every
selected request is accepted.

`tables` independently regenerates every declared table and compares its
canonical digest and row count, in catalog order. Its JSON is
[`tessera.table-digests/1`](../../spec/schemas/table-digests-v1.schema.json).
This operation accepts no model proof and does not validate the entire model.
The default total work allowance is 16,777,216 padded rows; a larger catalog
returns `ERR-RES-TIME` before generation. `--max-table-rows N` explicitly
selects another allowance. The input file limit is 1 GiB.

`weights link` is a recognized reserved surface and fails explicitly without
accessing or changing a ledger; its implementation remains with
[#158](https://github.com/AbdelStark/tessera/issues/158). The future `--manifest`
anchor interface is not substituted with a different file format. Current
reports explicitly say `anchor_verified`, `ledger_updated` and `zero_knowledge`
are false. These boundaries preserve the v0.4 work rather than claiming it from
a reference proof.

The cross-workspace binary test builds the verifier with its own lockfile and
runs the real producer proof through that binary. It is required by CI:

```sh
(cd verifier && cargo build -p tv-cli --features tv-core/test-params --locked)
TESSERA_VERIFY_BIN="$PWD/verifier/target/debug/tessera-verify" \
  cargo test -p tessera-prover --test model_leaf \
  --features test-rng,tv-core/test-params --locked public_cli_binary -- --ignored
```

These test features fail release builds. The verifier has no producer dependency;
the differential test and repository JSON Schema validator live on the root side.

## Directory Live

```sh
target/debug/tessera-verify live \
  --claims claims.tcl --weights weights.twr --weights-root "$TRUSTED_ROOT" \
  --statement-digest "$TRUSTED_STATEMENT" --bound 0 --connect "$FRESH_DIR" \
  --request-id 23 --request-session "$REQUEST_SESSION_HEX" \
  --prompt-len 2 --output-len 3 --manifest-digest "$MANIFEST_HEX" \
  --out verdict.json --json
```

The application supplies every expected request field before the verifier emits
`SeedCommit`. The independently constructed statement must match the external
whole-statement pin. The verifier waits for each canonical packet, validates it
through the existing one-shot Live core, and publishes the next challenge only
after its checks pass. Only the final Round3 checks can accept.

Both actors require a fresh existing directory. Role markers and atomically
published packets cannot be overwritten; reuse is an input error. Verdict files
must be outside the exchange directory. The default deadlines are 600 seconds
per message and 3,600 seconds for Round2; `--timeout-ms` overrides each wait.
The full session also has a cooperative core deadline. This reference exchange
provides no authenticated TLS channel; full network transport remains #181.
The [exact CLI suite](../testing/exact-cli.md) exercises both real binaries,
ragged geometry, mismatched outputs and tampered packets at every Live round.
