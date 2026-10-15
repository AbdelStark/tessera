# 02 — Public API

- Status: Accepted for v1 planning
- Governs: every surface a user or integrator touches, its stability tier, and its compatibility promise
- Related: [03 Data model](03-data-model.md), [04 Error model](04-error-model.md), [09 Release and versioning](09-release-and-versioning.md)

## 1. Stability tiers

| Tier | Surfaces | Promise |
|---|---|---|
| **A: protocol** | TCE schemas of every artifact and message, the protocol version string, the claim-list schema, table definitions, parameter-set definitions, verifier verdict semantics and exit codes | Frozen per protocol version. A change requires a new protocol version and a superseding RFC. Readers support the current and previous protocol version ([09 §3](09-release-and-versioning.md#3-protocol-and-schema-versions)). |
| **B: interfaces** | `tessera` and `tessera-verify` CLIs, the Python `tessera_quant` top-level functions, the Rust items re-exported at the root of `tessera-prover` and `tv-core` | SemVer from 1.0. Before 1.0, breaking changes are allowed in minor releases and listed in the changelog. |
| **C: internal** | Everything else | No promise. |

Tier A is the treaty-facing contract: two parties pin a protocol version, not a software version.

## 2. Command-line interface

Every command reads and writes TCE artifacts, prints human-readable progress to stderr, and with `--json` prints one JSON object to stdout describing its result. JSON output is informational and never hashed.

### 2.1 `tessera-quant` (Python)

```text
tessera-quant compile  --checkpoint DIR --calibration SPEC.json --profile {servable,minimal}
                       --scales {pot,affine} --out MODEL.d [--report CALIB.json] [--seed INT]
tessera-quant eval     --model MODEL.d --reference DIR --dataset SPEC.json --out FIDELITY.json [--seed INT]
tessera-quant reference --model MODEL.d --tokens TOKENS.bin --out LOGITS.bin
```

### 2.2 `tessera` (Rust, prover side)

```text
tessera compile         --model MODEL.d --params PARAMSET --out-dir DIR
                        # writes claims.tcl, plan.tep, tables/, commitment-report.json
tessera commit-weights  --model MODEL.d --plan plan.tep --out WEIGHTS.twc
tessera weights refresh --weights WEIGHTS.twc --out WEIGHTS2.twc        # emits linking proof
tessera prove           --plan plan.tep --model MODEL.d --weights WEIGHTS.twc --sample SAMPLE.tas
                        --transcripts DIR (--live ENDPOINT | --out PROOF.tnp)
                        [--backend {cpu,gpu}] [--devices 0,1,...] [--memory-budget GIB]
tessera audit sample    --manifest MANIFEST.tlm --seed SEED.tsd --rate FLOAT --out SAMPLE.tas
tessera audit batch     --sample SAMPLE.tas --min-tokens 16384 --out-dir DIR
tessera audit plan      --policy POLICY.json --traffic PROFILE.json --plan plan.tep   # detection and overhead
tessera params check    --params PARAMSET [--lookups N] [--table-rows T]   # soundness report
tessera bench unit-costs --device INT --out UNIT_COSTS.json
tessera bench model     --plan plan.tep --context INT --out MU.json         # cost-model projection
tessera recorder ref    --listen ADDR --log-dir DIR                         # software reference recorder
```

### 2.3 `tessera-verify` (Rust, verifier side, independent build)

```text
tessera-verify verify   --claims claims.tcl --weights-root ROOT --manifest MANIFEST.tlm
                        --proof PROOF.tnp --bound UHAT [--out VERDICT.json]
tessera-verify live     --claims claims.tcl --weights-root ROOT --manifest MANIFEST.tlm
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
}

pub enum Backend { Cpu { threads: usize }, Gpu { devices: Vec<u32> } }
pub enum Mode { Live, NonInteractive }

pub struct Prover<'a> { /* private */ }

impl<'a> Prover<'a> {
    pub fn new(plan: &'a ExecutionPlan, model: &'a IntegerModel,
               weights: &'a mut WeightCommitmentState, cfg: ProverConfig) -> Result<Self, ProverError>;
    pub fn start(&mut self, batch: &ProofBatch) -> Result<Round1, ProverError>;
    pub fn on_seed(&mut self, seed: &Seed) -> Result<Round2Response, ProverError>;
    pub fn on_queries(&mut self, queries: &ColumnQueries) -> Result<Round3Response, ProverError>;
    pub fn prove_non_interactive(&mut self, batch: &ProofBatch) -> Result<NiProof, ProverError>;
}
```

`Prover` is a state machine: `start → on_seed → on_queries` in live mode. Calling a method out of order returns `ProverError::Protocol(OutOfOrder)` and leaves the state unchanged. `prove_non_interactive` runs all three internally. `WeightCommitmentState` is borrowed mutably because every proof consumes hiding budget ([RFC-0012](../rfcs/RFC-0012-weight-commitment.md#budget-state-machine)).

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
    pub fn on_round1(&mut self, m: &Round1) -> Result<SeedCommit, Verdict>;     // verifier commits to its share
    pub fn on_prover_share(&mut self, s: &ProverSeedShare) -> Result<SeedReveal, Verdict>;
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
def calibrate(model: FloatModel, spec: CalibrationSpec, *, smoothquant_alpha: float = 0.5,
              seed: int = 0) -> CalibrationResult: ...
def quantize(model: FloatModel, calib: CalibrationResult, *, profile: Literal["servable", "minimal"] = "servable",
             scales: Literal["pot", "affine"] = "pot") -> IntegerModel: ...
def export(model: IntegerModel, out: Path) -> ArtifactDigest: ...
def run_reference(model: IntegerModel, tokens: npt.NDArray[np.uint32]) -> npt.NDArray[np.int64]: ...
def evaluate_fidelity(model: IntegerModel, reference: FloatModel, spec: EvalSpec, *,
                      seed: int = 0) -> FidelityReport: ...
```

All functions are deterministic given their arguments: they set `torch.use_deterministic_algorithms(True)`, fix every RNG from `seed`, and record library versions in their reports ([RFC-0005](../rfcs/RFC-0005-integer-model.md#determinism)).

## 5. Parameter sets (tier A)

Parameters are never passed piecemeal. A `ParamSet` is a named, versioned bundle defined in [RFC-0014](../rfcs/RFC-0014-interaction-and-soundness.md#parameter-sets): `live-64`, `ni-100` and `test-small`. `test-small` is rejected by `tessera-verify` unless built with the `test-params` feature, so test parameters cannot certify anything in production.

## 6. Compatibility rules

- **COMPAT-1.** A `tessera-verify` build for protocol version `N` verifies proofs from any prover build that emits protocol version `N`, regardless of either side's software version.
- **COMPAT-2.** Artifact readers accept schema versions `v` and `v − 1` for each artifact type and reject everything else with `ERR-CODEC-VERSION` ([04](04-error-model.md#3-error-codes)).
- **COMPAT-3.** Removing or changing a tier-B item requires a deprecation period of one minor release after 1.0 ([09 §4](09-release-and-versioning.md#4-deprecation-policy)).
- **COMPAT-4.** Adding a field to a tier-A schema is a schema version bump; TCE has no optional or unknown fields ([RFC-0003](../rfcs/RFC-0003-canonical-encoding.md#no-optional-fields)).
