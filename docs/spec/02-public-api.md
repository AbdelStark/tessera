# 02 — Public API

- Status: Accepted for v1 planning
- Governs: every surface a user or integrator touches, its stability tier, and its compatibility promise
- Related: [03 Data model](03-data-model.md), [04 Error model](04-error-model.md), [09 Release and versioning](09-release-and-versioning.md)

## 1. Stability tiers

| Tier | Surfaces | Promise |
|---|---|---|
| **A: protocol** | TCE schemas of every artifact and message, the protocol version string, the claim-list schema, table definitions, parameter-set definitions, verifier verdict semantics and exit codes | Frozen per protocol version. A change requires a new protocol version and a superseding RFC. Readers support the current and previous protocol version ([09 §3](09-release-and-versioning.md#3-protocol-and-schema-versions)). |
| **B: interfaces** | `tessera`, `tessera-verify` and `tessera-quant` CLIs, the Python `tessera_quant` top-level functions, the Rust items re-exported at the root of `tessera-prover` and `tv-core` | SemVer from 1.0. Before 1.0, breaking changes are allowed in minor releases and listed in the changelog. |
| **C: internal** | Everything else | No promise. |

Tier A is the treaty-facing contract: two parties pin a protocol version, not a software version.

## 2. Command-line interface

Every command reads and writes TCE artifacts, prints human-readable progress to stderr, and with `--json` prints one JSON object to stdout describing its result, or `{"command", "error": {"code", "message"}}` when it fails. JSON output is informational and never hashed. With `--log-format json`, stderr's lines are `tessera.log-event/1` objects instead ([05 §2](05-observability.md#2-logging)); `tessera` also takes `--log-level {error,warn,info,debug,trace}`, `info` by default. `compile`, `commit-weights`, `prove` and `bench model --out` write the run manifest of [05 §6](05-observability.md#6-run-manifest), `run-manifest.json` in a directory of the operator's own, or at `--run-manifest PATH`.

### 2.1 `tessera-quant` (Python)

```text
tessera-quant compile  --checkpoint DIR --calibration SPEC.json --profile {servable,minimal}
                       --scales {pot,affine} --out MODEL.d [--report CALIB.json] [--seed INT]
                       [--device DEVICE]
tessera-quant eval     --model MODEL.d --reference DIR --dataset SPEC.json --out FIDELITY.json
                       [--seed INT] [--device DEVICE]
tessera-quant reference --model MODEL.d --tokens TOKENS.ttk --out LOGITS.tlg
```

- **Inputs.** A checkpoint directory holds `config.json`, its safetensors files and the `tokenizer.json` the calibration and evaluation data are tokenized with; `config.json`'s `bos_token_id` is the beginning-of-text token. A `SPEC.json` is a `tessera.calibration-spec/1` or `tessera.eval-spec/1` file, whose JSON Schemas are in [`spec/schemas/`](../../spec/schemas/README.md). Its dataset sources are `"c4"`, the pinned C4 validation split of [RFC-0005](../rfcs/RFC-0005-integer-model.md#calibration), or JSON-lines files with their identity. RFC-0005's default calibration is [`calibration-spec-c4.json`](../perf/examples/calibration-spec-c4.json).
- **`compile`** draws the calibration set from `--seed` (default 0) and captures its statistics on `--device` (default `cpu`). It then quantizes and exports `D`, a secret artifact, and `--report` writes the calibration's `tessera.calibration/1` report. On the CPU, the same seed gives the same artifact byte for byte. `--scales affine` gives every weight that can take it a multiplier per channel ([RFC-0025](../rfcs/RFC-0025-affine-sites.md)).
- **`eval`** measures `D` against the checkpoint it was compiled from, whose file digests `D`'s provenance records; another reference is `ERR-QUANT-ARCH`. It redraws `D`'s calibration set from its provenance and the specification's calibration source, refusing with `ERR-QUANT-CALIB` a source that does not give the recorded selection digest back. It then draws the evaluation documents and prompts from `--seed`, skipping every calibration document (`tessera_quant.fidelity.select`), and writes a `tessera.fidelity/1` report.
- **`reference`** reads a `TokenFile` and writes a `LogitsFile`: the logits of every position of every sequence, in order.
- **Exit codes** are each error code's ([04 §3](04-error-model.md#3-error-codes)): 0 on success, 64 for usage, 65 for a malformed input, 20 for a calibration or I/O failure, 70 for a bug.

### 2.2 `tessera` (Rust, prover side)

```text
tessera compile         --model MODEL.d --params PARAMSET --out-dir DIR
                        # writes claims.tcl, plan.tep, commitment-report.json
tessera commit-weights  --model MODEL.d --plan plan.tep --out WEIGHTS.twc
                        # also writes WEIGHTS.twr, the public commitment, and prints R_W (RFC-0040)
tessera weights refresh --weights WEIGHTS.twc --out WEIGHTS2.twc        # emits linking proof
tessera prove           --plan plan.tep --model MODEL.d --weights WEIGHTS.twc --sample SAMPLE.tas
                        LOGS --transcripts DIR --batch K (--live ENDPOINT | --out PROOF.tnp)
                        [--backend {cpu,gpu}] [--devices 0,1,...] [--memory-budget GIB]
tessera audit share     --manifest MANIFEST.tlm --policy POLICY.json --commitment HEX --state DIR
                        # the prover's share for the coin toss over a manifest, one commitment per manifest
tessera audit sample    --manifest MANIFEST.tlm --log LOG.tdl
                        [--previous-manifest MANIFEST.tlm --previous-log LOG.tdl] [--between MANIFEST.tlm]...
                        --state DIR --verifier-share HEX --out SAMPLE.tas
tessera audit batch     --sample SAMPLE.tas LOGS --out-dir DIR             # writes DIR/batches.json
tessera audit plan      --policy POLICY.json --traffic PROFILE.json (--plan plan.tep | --source-defaults)
                        [--eta η] [--target-overhead O] [--out PLAN.json]           # detection and overhead
tessera params check    --params PARAMSET [--lookups N] [--table-rows T]   # soundness report
tessera bench unit-costs --device INT --out UNIT_COSTS.json
tessera bench model     (--plan plan.tep | --source-defaults) [--context INT,...] [--out MU.json]
                        # cost-model projection
tessera recorder ref    --listen ADDR --upstream ADDR --log-dir DIR --key FILE --period SECONDS
                        # software reference recorder (RFC-0015)
tessera inspect         FILE                                                # renders a TCE artifact as text (RFC-0003)
```

`audit share` is the prover's side of the coin toss over a manifest ([RFC-0016](../rfcs/RFC-0016-audit-sampling.md#sample-derivation) step 2):
- **One answer.** The first commitment for a manifest gets a fresh share. The share is kept in `--state`, with the manifest, the `tessera.audit-policy/1` policy and the commitment, before it is printed. The same commitment and policy again get the same share.
- **Another commitment or policy** for that manifest exits 20 with `ERR-PROTO-ORDER`.
- **The state.** Every responder of one prover shares the `--state` directory.

`audit sample` draws a period's `AuditSample` ([RFC-0016](../rfcs/RFC-0016-audit-sampling.md#sample-derivation)):
- **The logs.** It checks each log against its manifest (RFC-0062), and the chain from the previous period through any empty zero-length `--between` periods ([RFC-0063](../rfcs/RFC-0063-request-boundaries.md#the-previous-periods-log)). A log's first period needs no previous one. A period of zero length has no candidates and is refused.
- **The record.** The coin toss is the record `audit share` kept for the manifest, with the verifier's revealed share. Its policy is the `tessera.audit-policy/1` file ([schema](../../spec/schemas/audit-policy-v1.schema.json)) given there, its rates decimal strings. A manifest with no record exits 64.

A log that is not its manifest's exits 65 with `ERR-CODEC-DIGEST`. A share that does not open its commitment exits 20 with `ERR-PROTO-COINTOSS`. Previous periods that do not bind, or a policy file the reader refuses, exit 64. The recorder's signatures are the verifier's to check.

`LOGS` stands for `audit sample`'s log flags, `--manifest` to `--between`, which `audit batch` and `prove --sample` read and check the same way.

`audit batch` packs a sample's selected requests into proof batches ([RFC-0016](../rfcs/RFC-0016-audit-sampling.md#batching) 'Batching'):
- **The sample** must be its logs': its manifest is `--manifest`, and its policy and coin toss select exactly its requests among the period's candidates.
- **The lengths.** A request's `L` is its candidate's length. Its `prompt_len` and `output_len` are the tokens of its `In` and `Out` frames in both periods.
- **One group.** Every request is packed as greedy, since a temperature request's group is a pin of [RFC-0061](../rfcs/RFC-0061-sampling-randomness.md#which-pin-applies), which no crate implements yet.
- **The output.** `DIR/batches.json`, the `tessera.audit-batches/1` report ([schema](../../spec/schemas/audit-batches-v1.schema.json)), which `--json` prints too. Batches are numbered from 0 in packing order.

`prove --sample` proves batch `--batch K` of a sample, one batch per invocation:
- **The batch** is `audit batch`'s `K`, read again from the sample and `LOGS`, never from the report. A `K` past the last batch exits 64, and so does a batch of more than one request, until multi-request proof units are implemented ([RFC-0071](../rfcs/RFC-0071-request-slots.md), [#551](https://github.com/AbdelStark/tessera/issues/551)).
- **The transcripts.** `--transcripts DIR` holds each session's stored `SessionTranscript` as `DIR/<session_id>.tst`, the identifier in 32 lowercase hexadecimal digits.
- **The pre-check** ([RFC-0015](../rfcs/RFC-0015-transcript-anchor.md#prover-pre-check)) runs before the prover commits anything. A session with no transcript, a stored frame that does not re-seal to its logged digest, or a request whose frames give no lengths, exits 20 with `ERR-WITNESS-ANCHOR`, and no packet is sent.
- **The proof.** The request's tokens are its `In` frames', then its `Out` frames', from the stored frames the log names. Its statement is built as the reference path's below, with the sample's `manifest_digest`.

Both commands refuse `LOGS` with `audit sample`'s codes. A sample of another period, or one that its logs and coin toss do not give, exits 64, and a malformed sample or transcript exits 65.

`audit plan` reports a policy's detection probability and amortized overhead at a traffic profile ([RFC-0016](../rfcs/RFC-0016-audit-sampling.md#calculators) 'Calculators'):
- **The inputs.** The `tessera.audit-policy/1` policy, and a `tessera.audit-traffic/1` profile ([schema](../../spec/schemas/audit-traffic-v1.schema.json)) with one entry per bucket: its requests, their mean length and the violating ones. Each bucket's `μ` is the cost model's at its mean length, from the plan's counts or, with `--source-defaults`, the source's, at `bench model`'s scenario (`--gpus`, `--batch-tokens`).
- **The output.** A table, or with `--json` the `tessera.audit-plan/1` report ([schema](../../spec/schemas/audit-plan-v1.schema.json)), which `--out` writes too. It gives the detection against a prover that pads violations into a later bucket, the detection if none leaves its bucket, the amortized overhead and each bucket's rate for `--target-overhead`.
- **Refusals.** A profile that is not one well-formed entry per bucket, or that has no request, exits 64.

The v0.3 CPU reference path implements one exact-greedy request from an existing
`TokenFile` (one prompt followed by its recorded continuation):

```text
tessera compile --model MODEL.d --params PARAMSET --out-dir DIR
                [--max-log2-rows 32] [--query-block 512]
tessera prove --plan plan.tep --model MODEL.d --weights WEIGHTS.twc
              --reference-tokens TOKENS.ttk --prompt-len N --request-id N
              --request-session HEX --manifest-digest HEX
              (--live DIR | --out PROOF.tnp) [--memory-budget GIB] [--json]
              [--timeout-ms N]  # Live only
```

The optional compile flags select the existing gadget policy; their defaults
remain unchanged. `--live DIR` uses RFC-0014's one-shot directory transport.
A completed producer emits an informational `tessera.proof-run/1` receipt;
only the independent verifier supplies acceptance. A changed recorded output
returns `ERR-WITNESS-EXACT-MISMATCH`. The sampled-audit route, `--sample` in
place of `--reference-tokens` and its four flags, proves a batch of one request
the same way, on the CPU, with `--live DIR` for Live. See the
[exact CLI suite](../testing/exact-cli.md) for reproducible commands and limits.

### 2.3 `tessera-verify` (Rust, verifier side, independent build)

The current v0.3 protocol/1 reference interface verifies one exact-greedy NI
or directory Live request. Its application-owned whole-statement digest authenticates the expected
statement before any proof is evaluated:

```text
tessera-verify verify --claims claims.tcl --weights-root ROOT --weights WEIGHTS.twr
                      --statement-digest HEX --proof PROOF.tnp --bound BITS
                      [--out VERDICT.json] [--json] [--max-artifact-bytes N]
                      [--max-memory-bytes N] [--timeout-ms N]
tessera-verify live --claims claims.tcl --weights-root ROOT --weights WEIGHTS.twr
                    --statement-digest HEX --bound BITS --connect DIR
                    --request-id N --request-session HEX --prompt-len N
                    --output-len N --manifest-digest HEX [--out VERDICT.json] [--json]
                    [--max-artifact-bytes N] [--max-memory-bytes N] [--timeout-ms N]
tessera-verify tables --claims claims.tcl [--json] [--max-table-rows N]
tessera-verify params --params PARAMSET [--lookups N] [--table-rows T] [--json]
```

`--statement-digest` must come from the application independently of the proof;
using the proof's own digest supplies no trust anchor. `--bound` is exact bits
in increments of 1/256. [The reference CLI guide](../verifier/reference-cli.md)
describes resource allowances, atomic report output and the measured CLI trace.
The commands below are the full target interface. `audit plan` and `audit report` are
implemented ([#151](https://github.com/AbdelStark/tessera/issues/151)); anchor integration
in `verify` and `live`, the ledger's weight epochs and `weights link`, and authenticated
network transport remain with
[#137](https://github.com/AbdelStark/tessera/issues/137),
[#158](https://github.com/AbdelStark/tessera/issues/158) and
[#181](https://github.com/AbdelStark/tessera/issues/181).
Reserved commands fail explicitly; they never certify a statement or update a ledger.

```text
tessera-verify verify   --claims claims.tcl --weights-root ROOT --weights WEIGHTS.twr --manifest MANIFEST.tlm
                        --proof PROOF.tnp --bound UHAT [--out VERDICT.json]
tessera-verify live     --claims claims.tcl --weights-root ROOT --weights WEIGHTS.twr --manifest MANIFEST.tlm
                        --connect ENDPOINT --bound UHAT [--out VERDICT.json]
tessera-verify tables   --claims claims.tcl     # regenerates and prints table digests
tessera-verify params   --params PARAMSET       # independent soundness report
tessera-verify weights link --ledger LEDGER.tvl --claims claims.tcl --link LINK.twl   # accept a refreshed weight root (RFC-0064)
tessera-verify audit plan  --policy POLICY.json --traffic PROFILE.json [--mu MU.json]   # detection; overhead with μ
tessera-verify audit report --ledger LEDGER.tvl --sample SAMPLE.tas --out REPORT.json
tessera-verify audit commit --manifest MANIFEST.tlm --recorder-key KEY --policy POLICY.json --toss TOSS.tvt
tessera-verify audit reveal --toss TOSS.tvt --prover-share HEX --manifest MANIFEST.tlm --recorder-key KEY
                            --log LOG.tdl [--previous-manifest M.tlm --previous-log L.tdl] [--between M.tlm]...
                            --ledger LEDGER.tvl [--new-ledger] --out SAMPLE.tas
```

`audit commit` and `audit reveal` are the verifier's half of the coin toss ([RFC-0016](../rfcs/RFC-0016-audit-sampling.md#sample-derivation), [RFC-0067](../rfcs/RFC-0067-verifier-ledger-encoding.md#the-verifiers-own-sample) 'The pending toss').
- **`KEY`** is the recorder's Ed25519 public key, 64 hexadecimal digits, which the verifier pins ([RFC-0015](../rfcs/RFC-0015-transcript-anchor.md#signatures) 'Signatures'). A manifest whose signature fails under it exits 10 with `REJ-ANCHOR`.
- **`audit commit`** checks the manifest's signature. It draws a fresh share, writes the pending toss to `TOSS.tvt` (mode 0600, created exclusively, synced) and only then prints the commitment `c_V`, in hexadecimal or with `--json`. An existing `TOSS.tvt` exits 64.
- **`audit reveal`** checks each log against its manifest, and the chain from the previous period through any zero-length `--between` periods, as `audit sample` does. It reveals the toss against the prover's share and derives the sample. It records the sample in the ledger before writing `SAMPLE.tas`, and then clears and removes `TOSS.tvt`.
  - **The ledger** must exist, else `ERR-RES-IO`, unless `--new-ledger` asks the command to create it: then a file already at the path exits 64. A mistyped path never makes a second ledger.
  - **Refusals.** A toss of another manifest exits 64. A toss whose envelope fails its digest exits 65 with `ERR-CODEC-DIGEST`, and one whose share does not open its commitment exits 20 with `ERR-PROTO-COINTOSS` (04 §3).

`audit plan` reads the policy and the profile as `tessera audit plan` does, and, with `--mu`, a
`tessera.cost-model/1` report whose projections' contexts are the profile's mean lengths, in
order: each bucket's `μ` is its projection's `mu.total`, and `η` and the target overhead are the
report's scenario's ([RFC-0016](../rfcs/RFC-0016-audit-sampling.md#calculators)). Without
`--mu` it reports detection only, and `tessera.audit-plan/1`'s overhead figures are `null`. A
profile, policy or cost model it refuses exits 64.

`audit report` writes the sample's `tessera.audit-report/1` report from the verifier's ledger
([RFC-0016](../rfcs/RFC-0016-audit-sampling.md#audit-report)): a selected request the ledger
records accepted for the sample is `accepted` with its `Û`, and every other one `unproven`,
`DeadlineMissed`. The sample must be the verifier's own derivation, the one sample the ledger's
`derived` records for its manifest when the verifier reveals its toss; any other sample exits 64
with `ERR-USAGE`, however its coin toss opens
([RFC-0067](../rfcs/RFC-0067-verifier-ledger-encoding.md#the-verifiers-own-sample)). A reveal
derives only from a toss that opens, so a sample whose toss does not open can be the ledger's only
if the ledger was edited by hand; such a sample exits 20 with `ERR-PROTO-COINTOSS`. The ledger
must be a regular file of one name, else exit 64, and is read under `tv-codec`'s message limits,
which its writers keep it within. It exits 0 when every selected request is accepted and 10
otherwise.

### 2.4 Exit codes

| Code | `tessera-verify` meaning | `tessera` meaning |
|---|---|---|
| 0 | Accept | Success |
| 10 | Reject: the proof is invalid, `Û` exceeds the bound, or a soundness target is missed ([RFC-0051](../rfcs/RFC-0051-soundness-target-exit-code.md)) | Soundness target missed (`tessera params check`, [RFC-0051](../rfcs/RFC-0051-soundness-target-exit-code.md)) |
| 20 | Inconclusive: the verifier could not evaluate (I/O, resource limit, internal error) | Completeness failure: the honest prover cannot produce a proof (for example a lookup key out of table range or an exact-mode mismatch) |
| 64 | Usage error | Usage error |
| 65 | Malformed input artifact | Malformed input artifact |
| 70 | Internal invariant violation | Internal invariant violation |

**INV-API-1.** Only exit code 0 from `tessera-verify` certifies a statement. Integrations must treat every other code, including 20, as "not certified".

## 3. Rust interfaces (tier B)

### 3.1 Prover

```rust
pub struct ProverConfig {
    pub backend: Backend,
    pub params: ParamSet,            // named, from RFC-0014
    pub mode: Mode,
    pub memory_budget: MemoryBudget, // per device and host, bytes
    pub mask_rng: MaskRng,           // OsRng in production; Seeded(u64) only with the `test-rng` feature
    pub release: ReleaseConfig,      // tessera-protocol/2: budget coefficients and doublings (RFC-0018)
}

pub enum Backend { Cpu { threads: usize }, Gpu { devices: Vec<u32> } }
pub enum Mode { Live, NonInteractive }

pub struct Prover<'a> { /* private */ }

impl<'a> Prover<'a> {
    pub fn new(plan: &'a ExecutionPlan, model: &'a IntegerModel,
               weights: &'a mut WeightCommitmentState, cfg: ProverConfig) -> Result<Self, ProverError>;
    pub fn start(&mut self, batch: &ProofBatch, commit: &SeedCommit)
        -> Result<(Round1, ProverSeedShare), ProverError>;
    pub fn on_seed(&mut self, reveal: &SeedReveal) -> Result<Round2Response, ProverError>;
    pub fn on_queries(&mut self, queries: &ColumnQueries) -> Result<Round3Response, ProverError>;
    pub fn prove_non_interactive(&mut self, batch: &ProofBatch) -> Result<NiProof, ProverError>;
}
```

`Prover` is a state machine: `start → on_seed → on_queries` in live mode, in the session order of [RFC-0014](../rfcs/RFC-0014-interaction-and-soundness.md#session-order-live-mode) and the states of [RFC-0036](../rfcs/RFC-0036-protocol-messages.md#the-provers-live-session). Calling a method out of order returns `ProverError::Protocol(OutOfOrder)` and leaves the state unchanged. `prove_non_interactive` runs all three internally. In `tessera-protocol/2`, each method that yields a message returns at that message's release time, on the schedule of [RFC-0018 'Release schedule'](../rfcs/RFC-0018-gpu-prover-runtime.md#release-schedule), so that no caller can send it early. `WeightCommitmentState` is borrowed mutably because every proof consumes hiding budget ([RFC-0012](../rfcs/RFC-0012-weight-commitment.md#budget-state-machine)).

The current protocol/1 CPU implementation exposes these types from
`tessera_prover`. `ProofBatch` holds one expected `PublicStatement` and
`Secret<Vec<u32>>` containing its requests' prompts and recorded continuations, concatenated in batch order: each request's prompt then outputs, which go to its slot's rows ([RFC-0071](../rfcs/RFC-0071-request-slots.md#the-statement-and-the-batching)). P1 borrows
the tokens to compute the clearing witness; later rounds need no borrow of the
caller's batch, which can then be dropped. The driver checks the full named parameter bundle, execution plan,
actual model commitment and epoch; unsupported GPU targets, predictors and
multi-request units fail explicitly. `MemoryBudget { host: u64 }` supplies a
conservative allowance for this resident reference backend, excluding the
caller's model; weight and runtime estimates are checked before P1. This is not
the tile/pass planner of RFC-0018. `ReleaseConfig` is a unit type for immediate
protocol/1 release; protocol/2 scheduling and hiding are not implemented here.
Only `MaskRng::OsRng` is available in production; `Seeded(u64)` requires
`test-rng` and cannot compile in a release build.

`Prover` reuses `tessera_protocol::ProverSession` for live transitions. A second
reveal after a successful seed response returns `Reuse`, as RFC-0036 requires.
`abort()` ends an unfinished session. Completion, abort, or a backend failure
discards per-proof witness and randomness; retry with a new prover and fresh
seed. Protocol/1 leaves the weight state's hiding budget unchanged. Actual tiny
model acceptance is tested with `test-small/1`, which certifies nothing; it
does not establish production-preset performance or zero knowledge.

### 3.2 Verifier

```rust
pub struct VerifierConfig {
    pub params: ParamSet,
    pub limits: ResourceLimits,  // max artifact bytes, max memory, max wall time
    pub accept_bound: UBound,    // public policy threshold on Û, fixed-point bits
    pub weights: WeightCommitmentPublic, // trusted epoch artifact (RFC-0040)
}

pub fn verify_non_interactive(claims: &ClaimList, statement: &PublicStatement,
                              proof: &NiProof, cfg: &VerifierConfig) -> Verdict;

pub struct LiveVerifier { /* private */ }
impl LiveVerifier {
    pub fn new(claims: ClaimList, statement: PublicStatement, cfg: VerifierConfig) -> Self;
    pub fn start(&mut self) -> Result<SeedCommit, Verdict>;                      // fresh OS randomness may fail
    pub fn on_round1(&mut self, m: &Round1, s: &ProverSeedShare) -> Result<SeedReveal, Verdict>;
    pub fn on_round2(&mut self, m: &Round2Response) -> Result<ColumnQueries, Verdict>;
    pub fn on_round3(&mut self, m: &Round3Response) -> Verdict;
}

pub enum Verdict {
    Accept(AcceptReport),       // includes Û per request and per batch
    Reject(RejectReason),       // typed reason, see 04-error-model
    Inconclusive(VerifierError),
}
```

The current `tv-core` protocol/1 driver composes one exact-greedy request. Other
predictors and multi-request proof units return `Inconclusive(ERR-USAGE)`; no
request is silently omitted. The caller supplies the expected statement and the
trusted epoch's weight artifact, whose hash and epoch must agree. The byte entry
point `verify_non_interactive_bytes` checks the canonical NI envelope before
verification. `LiveVerifier` is one-shot: a valid-phase failure is terminal;
an out-of-order call leaves its state unchanged. Unix hosts obtain fresh entropy
from `/dev/urandom`; unsupported hosts fail with `ERR-RES-IO`. The unrevealed
verifier share is cleared on failure or drop.

`ResourceLimits.max_memory_bytes` currently bounds a conservative work estimate,
not every allocation; `max_wall_time` is checked between phases. Complete
allocation/deadline hardening remains [#182](https://github.com/AbdelStark/tessera/issues/182).
The core report has deterministic bytes and null timings; the CLI's measured
check trace is separate. Protocol/1 acceptance makes no anchor, persistent-ledger
or zero-knowledge claim. `test-small/1` acceptance certifies nothing.

The verifier API never panics on any input. Every malformed byte sequence yields `Reject(Malformed(..))` or `Inconclusive(..)`, never a crash ([RFC-0019](../rfcs/RFC-0019-verifier.md#robustness)).

## 4. Python interface (tier B)

```python
def import_checkpoint(path: Path, *, expected: ArchSpec | None = None) -> FloatModel: ...
def calibrate(model: FloatModel, calibration: CalibrationSet, device: str = "cpu") -> CalibrationResult: ...
def quantize(model: FloatModel, calib: CalibrationResult, *, profile: Literal["servable", "minimal"] = "servable",
             scales: Literal["pot", "affine"] = "pot", alphas: tuple[Fraction, ...] = ALPHAS,
             device: str = "cpu") -> IntegerModel: ...
def export(model: IntegerModel, out: Path) -> bytes: ...           # the artifact's digest
def run_reference(model: IntegerModel, tokens: npt.NDArray[np.uint32]) -> npt.NDArray[np.int64]: ...
def evaluate_fidelity(model: IntegerModel, reference: FloatModel, spec: EvalSpec, *,
                      seed: int = 0, device: str = "cpu") -> FidelityReport: ...
```

- **The calibration set** comes from `tessera_quant.calibration.draw`, which defines it exactly from a seed and the documents ([RFC-0005](../rfcs/RFC-0005-integer-model.md#calibration) 'Calibration'). Its held-out split serves `quantize`.
- **`quantize` chooses the migration strength** among `alphas`: the one whose `D` diverges least from the BF16 reference on the held-out split ([RFC-0005](../rfcs/RFC-0005-integer-model.md#calibration)). `D` runs through `run_reference` for this. It records the strength and its sources in `Provenance`.
- **`export`** writes `D` as a secret artifact and returns the envelope's digest.
- **`evaluate_fidelity`** measures RFC-0005's metrics with their intervals, on an `EvalSpec`: the perplexity documents and the agreement prompts as token sequences with their data identities, the reference continuation's length, and the calibration's document hashes, which no evaluation document may share. Its report is `tessera.fidelity/1`.

All functions are deterministic given their arguments: they set `torch.use_deterministic_algorithms(True)`, fix every RNG from the calibration set's seed, and record library versions in their reports and in `Provenance` ([RFC-0005](../rfcs/RFC-0005-integer-model.md#determinism)).

## 5. Parameter sets (tier A)

Parameters are never passed piecemeal. A `ParamSet` is a named, versioned bundle defined in [RFC-0014](../rfcs/RFC-0014-interaction-and-soundness.md#parameter-sets): `live-64`, `ni-100` and `test-small`. `test-small` is rejected by `tessera-verify` unless built with the `test-params` feature, so test parameters cannot certify anything in production.

## 6. Compatibility rules

- **COMPAT-1.** A `tessera-verify` build for protocol version `N` verifies proofs from any prover build that emits protocol version `N`, regardless of either side's software version.
- **COMPAT-2.** Artifact readers accept schema versions `v` and `v − 1` for each artifact type and reject everything else with `ERR-CODEC-VERSION` ([04](04-error-model.md#3-error-codes)).
- **COMPAT-3.** Removing or changing a tier-B item requires a deprecation period of one minor release after 1.0 ([09 §4](09-release-and-versioning.md#4-deprecation-policy)).
- **COMPAT-4.** Adding a field to a tier-A schema is a schema version bump; TCE has no optional or unknown fields ([RFC-0003](../rfcs/RFC-0003-canonical-encoding.md#no-optional-fields)).
