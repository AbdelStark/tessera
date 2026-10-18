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

Every command reads and writes TCE artifacts, prints human-readable progress to stderr, and with `--json` prints one JSON object to stdout describing its result, or `{"command", "error": {"code", "message"}}` when it fails. JSON output is informational and never hashed.

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
                        --transcripts DIR (--live ENDPOINT | --out PROOF.tnp)
                        [--backend {cpu,gpu}] [--devices 0,1,...] [--memory-budget GIB]
tessera audit sample    --manifest MANIFEST.tlm --seed SEED.tsd --rate FLOAT --out SAMPLE.tas
tessera audit batch     --sample SAMPLE.tas [--min-tokens 16384] --out-dir DIR
tessera audit plan      --policy POLICY.json --traffic PROFILE.json --plan plan.tep   # detection and overhead
tessera params check    --params PARAMSET [--lookups N] [--table-rows T]   # soundness report
tessera bench unit-costs --device INT --out UNIT_COSTS.json
tessera bench model     (--plan plan.tep | --source-defaults) [--context INT,...] [--out MU.json]
                        # cost-model projection
tessera recorder ref    --listen ADDR --log-dir DIR                         # software reference recorder
```

### 2.3 `tessera-verify` (Rust, verifier side, independent build)

```text
tessera-verify verify   --claims claims.tcl --weights-root ROOT --weights WEIGHTS.twr --manifest MANIFEST.tlm
                        --proof PROOF.tnp --bound UHAT [--out VERDICT.json]
tessera-verify live     --claims claims.tcl --weights-root ROOT --weights WEIGHTS.twr --manifest MANIFEST.tlm
                        --connect ENDPOINT --bound UHAT [--out VERDICT.json]
tessera-verify tables   --claims claims.tcl     # regenerates and prints table digests
tessera-verify params   --params PARAMSET       # independent soundness report
tessera-verify weights link --ledger LEDGER.tvl --link LINK.twl   # accept a refreshed weight root
tessera-verify audit plan  --policy POLICY.json --traffic PROFILE.json --claims claims.tcl
tessera-verify audit report --ledger LEDGER.tvl --sample SAMPLE.tas --out REPORT.json
```

### 2.4 Exit codes

| Code | `tessera-verify` meaning | `tessera` meaning |
|---|---|---|
| 0 | Accept | Success |
| 10 | Reject: the proof is invalid or `Û` exceeds the bound | Not used |
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

### 3.2 Verifier

```rust
pub struct VerifierConfig {
    pub params: ParamSet,
    pub limits: ResourceLimits,  // max artifact bytes, max memory, max wall time
    pub accept_bound: UBound,    // public policy threshold on Û, fixed-point bits
}

pub fn verify_non_interactive(claims: &ClaimList, statement: &PublicStatement,
                              proof: &NiProof, cfg: &VerifierConfig) -> Verdict;

pub struct LiveVerifier { /* private */ }
impl LiveVerifier {
    pub fn new(claims: ClaimList, statement: PublicStatement, cfg: VerifierConfig) -> Self;
    pub fn start(&mut self) -> SeedCommit;                                       // verifier commits to its share
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
