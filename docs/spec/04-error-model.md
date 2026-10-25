# 04 — Error Model

- Status: Accepted for v1 planning
- Governs: error taxonomy, stable codes, verdict semantics, failure modes and recovery
- Related: [05 Observability](05-observability.md), [RFC-0014](../rfcs/RFC-0014-interaction-and-soundness.md), [RFC-0019](../rfcs/RFC-0019-verifier.md)

## 1. Principles

1. **A rejection is not an error.** The verifier's job is to reject invalid proofs. A rejection is a successful evaluation with a typed reason. An error means the verifier could not evaluate.
2. **Only acceptance certifies.** `Verdict::Accept` is the only outcome that certifies the statement. `Reject` and `Inconclusive` are both "not certified"; they differ in what they imply about the prover.
3. **Completeness failures are evidence, not noise.** When the honest prover cannot produce a proof (for example an exact-mode output mismatch), the failure is reported with its code and never retried into success.
4. **Errors carry public data only.** No error message, context field or source chain may contain a secret-class value ([05 §3](05-observability.md#3-data-classification-and-redaction)).
5. **No panics on input.** Library code returns typed errors for every input-dependent condition. A panic is a bug and maps to `ERR-INTERNAL`.

## 2. Verdicts

```rust
pub enum Verdict {
    Accept(AcceptReport),
    Reject(RejectReason),
    Inconclusive(VerifierError),
}

pub enum RejectReason {
    Malformed { artifact: ArtifactKind, code: ErrorCode },          // REJ-MALFORMED
    StatementMismatch { field: StatementField },                    // REJ-STATEMENT
    Sumcheck { stage: StageId, round: u32 },                        // REJ-SUMCHECK
    TreeLevel { group: u32, level: u32, round: u32 },               // REJ-TREE
    LeafMerge { round: u32 },                                       // REJ-LEAF-MERGE
    LogUpIdentity { round: u32 },                                   // REJ-LOGUP-IDENTITY (tessera-protocol/2)
    Lookup { table: TableId },                                      // REJ-LOOKUP
    Gadget { gadget: GadgetId },                                    // REJ-GADGET
    MerklePath { commitment: CommitmentClass, column: u32 },        // REJ-PCS-MERKLE
    Proximity { commitment: CommitmentClass, column: u32 },         // REJ-PCS-PROXIMITY
    Evaluation { commitment: CommitmentClass },                     // REJ-PCS-EVAL
    ProofOfWork { phase: PowPhase },                                // REJ-POW
    Anchor { request_id: Option<u64>, kind: AnchorFailure },        // REJ-ANCHOR
    UBoundExceeded { request_id: Option<u64>, uhat: FixedBits, bound: FixedBits }, // REJ-UHAT
    Budget { epoch: u32 },                                          // REJ-BUDGET
    Params { name: ParamSetId },                                    // REJ-PARAMS
}

pub enum StatementField {                    // RFC-0039: one variant per check, in check order
    ProtocolVersion, Architecture, Quantization, Rescaling { site: Site }, Profile,
    Table { table: TableId }, GadgetPolicy, Gadget { gadget: GadgetId }, Layout, Template,
    ClaimListDigest, WeightsRoot, WeightsEpoch, ManifestDigest, Batch, Predictor, LayoutDigest,
}

pub struct ArtifactKind(pub SchemaId);       // RFC-0043: the prover's artifact that failed to decode, 03 §5
pub enum PowPhase { Seed, Queries }          // RFC-0043: NiProof's seed_nonce, then its query_nonce
pub enum AnchorFailure { ManifestSignature, Inclusion, Frames, SamplerKey, Session }   // RFC-0043: RFC-0015's checks 1 and 2; SamplerKey from RFC-0061; Inclusion retired and Session added by RFC-0062
```

[RFC-0043](../rfcs/RFC-0043-rejection-payloads.md) defines the other payloads. `Malformed` names an artifact the prover sends; one of the verifier's own inputs that fails to decode is `Inconclusive(ERR-CODEC-*)`. `Anchor` names no request for a manifest-level failure. `Params` holds RFC-0014's `ParamSetId`.

`StatementMismatch` names the first check that fails, as [RFC-0039](../rfcs/RFC-0039-statement-fields.md#the-checks) maps them. The checks are claim-list validation ([RFC-0007](../rfcs/RFC-0007-claim-list.md#verifier-validation)), then the statement and the advice commitment's `layout_digest` ([RFC-0019](../rfcs/RFC-0019-verifier.md#check-order) step 4). Its payload is public: a site type, a table's index or a gadget's index.

`Sumcheck` names a reduction stage by its `stage/{k}/…` index, and `TreeLevel` a level of the leaf stage by its `tree/{ℓ}/…` index ([RFC-0033](../rfcs/RFC-0033-tree-level-rejections.md)). In a stage of token groups, `TreeLevel` also names the group: `h` for group `h`'s levels under `group/{h}/tree/{ℓ}/…`, `G` for the table batch's under `tables/tree/{ℓ}/…`, and 0 in an ungrouped stage. A group's failed split check is at its level count `L_h`, round 0. `LeafMerge` names a round of the grouped stage's merge, `merge/round/{j}` ([RFC-0054](../rfcs/RFC-0054-token-groups.md#messages)). `LogUpIdentity` is `tessera-protocol/2`'s identity sumcheck, which combines every LogUp instance and so names no table. Their `round` counts from 0 as the labels do, with `n_max` for the final evaluations and check ([RFC-0010](../rfcs/RFC-0010-proof-schedule.md#sumcheck-messages)). Every rejection exits 10 ([02 §2.4](02-public-api.md#24-exit-codes)).

The verifier stops at the first failed check and reports it. Check order is fixed ([RFC-0019](../rfcs/RFC-0019-verifier.md#check-order)) so that a given invalid proof always yields the same reason.

## 3. Error codes

Codes are stable tier-A identifiers ([02 §1](02-public-api.md#1-stability-tiers)). Each maps to one Rust variant and one CLI exit code.

| Code | Class | Meaning | Raised by | Exit |
|---|---|---|---|---|
| `ERR-USAGE` | Usage | Invalid CLI arguments or configuration | CLIs, compiler | 64 |
| `ERR-CODEC-MAGIC` | Input | Envelope magic or schema id mismatch | Codec | 65 |
| `ERR-CODEC-VERSION` | Input | Unsupported schema or protocol version | Codec | 65 |
| `ERR-CODEC-NONCANONICAL` | Input | Non-canonical field element, trailing bytes, invalid tag, non-minimal length | Codec | 65 |
| `ERR-CODEC-TRUNCATED` | Input | Input ended inside a value | Codec | 65 |
| `ERR-CODEC-DIGEST` | Input | Envelope digest does not match payload; or a period's log does not match its manifest ([RFC-0062](../rfcs/RFC-0062-period-log-sessions.md#the-periods-log)). The checks run in order: the count, then each `seq`, then the root. The offset is in bytes of the concatenated entries: the log's end for the count or the root, the wrong entry's start for a `seq` | Codec | 65 |
| `ERR-CODEC-LIMIT` | Input | A length exceeds the configured resource limit, or a value exceeds a bound its schema states | Codec | 65 |
| `ERR-QUANT-ARCH` | Model | Checkpoint or integer model outside `ArchSpec` support, or inconsistent with its architecture | Front-end, compiler | 65 |
| `ERR-QUANT-CALIB` | Model | Calibration produced non-finite or degenerate statistics, or its data failed a hygiene check: too few usable documents, a malformed file, or a document shared with evaluation data | Front-end | 20 |
| `ERR-COMPILE-RANGE` | Model | A gadget input's range is not one its gadget takes: a lookup key past any table bound, a bit operation's input that may not be a bit, or an integer past its decomposition's width ([RFC-0066](../rfcs/RFC-0066-bits-and-sha256.md)) | Compiler | 20 |
| `ERR-COMPILE-WRAP` | Model | A lookup key cannot be proven a true integer | Compiler | 20 |
| `ERR-COMPILE-PROFILE` | Model | `servable` profile constraints unsatisfiable | Compiler | 20 |
| `ERR-WITNESS-EXACT-MISMATCH` | Completeness | Exact mode: a recorded output token differs from `D`'s output | Prover | 20 |
| `ERR-WITNESS-ANCHOR` | Completeness | Prover-held transcript does not match the digest log | Prover | 20 |
| `ERR-WITNESS-BUDGET` | Completeness | Weight-commitment hiding budget insufficient; refresh required | Prover | 20 |
| `ERR-PROTO-ORDER` | Protocol | Message out of order for the session state | Both | 20 |
| `ERR-PROTO-VERSION` | Protocol | Peer speaks another protocol version | Both | 20 |
| `ERR-PROTO-COINTOSS` | Protocol | Seed reveal does not match its commitment | Both | 20 |
| `ERR-PROTO-REUSE` | Protocol | Attempt to answer a second seed for the same `Round1` | Prover | 70 |
| `ERR-PROTO-TIMEOUT` | Protocol | Peer did not respond within the session deadline | Both | 20 |
| `ERR-PROTO-QUERIES` | Protocol | Column queries for other matrices, of another count than `τ_A` or `τ_W`, or past the codeword ([RFC-0036](../rfcs/RFC-0036-protocol-messages.md#errors)) | Prover | 20 |
| `ERR-RES-MEMORY` | Resource | Memory budget exceeded | Both | 20 |
| `ERR-RES-DEVICE` | Resource | GPU driver or kernel error | Prover | 20 |
| `ERR-RES-IO` | Resource | Filesystem or network I/O failure, or a failure of the operating system's random source | Both | 20 |
| `ERR-RES-TIME` | Resource | Wall-time limit exceeded, or a message missed the last slot of its release schedule | Both | 20 |
| `ERR-INTERNAL` | Internal | Invariant violation (a bug) | Both | 70 |

A lookup key outside its table range in an honest witness is `ERR-INTERNAL`, not a completeness failure: [INV-DM-2](03-data-model.md#6-invariants) guarantees it cannot happen, so its occurrence is a range-analysis bug.

## 4. Error representation

```rust
pub struct TesseraError {
    pub code: ErrorCode,                    // stable enum, Display prints "ERR-..."
    pub context: PublicContext,             // stage, layer, tensor id, request id; public data only
    pub source: Option<Box<dyn std::error::Error + Send + Sync>>,
}
```

- Each crate defines its own `thiserror` enum and converts into `TesseraError` at crate boundaries.
- `PublicContext` holds only identifiers and shapes. It has no field that can hold field elements, tokens or tensors, so redaction is enforced by the type ([05 §3](05-observability.md#3-data-classification-and-redaction)).
- The verifier workspace defines its own error and verdict types with the same codes. The codes are spelled out in this document, not shared through code ([RFC-0001](../rfcs/RFC-0001-workspace-and-verifier-independence.md)).
- Python raises subclasses of `TesseraQuantError` carrying the same `code` strings.

## 5. Failure modes and system response

| Failure mode | Detection | Response |
|---|---|---|
| Malformed or hostile artifact | Codec checks, size limits before allocation | `Reject(Malformed)` for proofs; `ERR-CODEC-*` for other inputs; no allocation beyond `ResourceLimits` |
| Exact-mode deployment deviated from `D` | Witness pass compares recorded outputs with `D`'s outputs | `ERR-WITNESS-EXACT-MISMATCH` with request id and token index; batch not proven; audit report marks it unproven |
| Bounded-mode deviation | Not an error: `Û` rises | Verifier reports `Û`; `Reject(UBoundExceeded)` if above the bound |
| Recorder log and prover transcripts disagree | Prover pre-check; verifier anchor check | Prover: `ERR-WITNESS-ANCHOR`. Verifier: `Reject(Anchor)` |
| Weight hiding budget exhausted | Budget state machine before `Round1` | `ERR-WITNESS-BUDGET`; operator runs `tessera weights refresh` |
| Selective abort during coin toss | Session state machine | Prover discards the `Round1` and its randomness; the next attempt is a fresh `Round1` (INV-PROTO-1 in [RFC-0014](../rfcs/RFC-0014-interaction-and-soundness.md#invariants)) |
| Prover crash after `Round1` | Restart from persisted per-proof secret seed | Commitments are a deterministic function of inputs and the encrypted per-proof seed, so the prover resumes with identical messages ([RFC-0018](../rfcs/RFC-0018-gpu-prover-runtime.md#crash-consistency)) |
| GPU transient fault | CUDA error codes, ECC counters | Retry the current phase up to two times from its start; then `ERR-RES-DEVICE` |
| Verifier resource exhaustion | `ResourceLimits` checks | `Inconclusive(ERR-RES-*)` |
| Nondeterministic witness (bug) | Differential tests; witness digests compared across backends in CI | `ERR-INTERNAL`; release-blocking |
| Table generator divergence across implementations | Pinned table digests in claim list | Verifier: `Reject(StatementMismatch)` if its regenerated digest differs; CI conformance failure |

## 6. Retry and idempotence

- Every prover phase is restartable from its beginning. Phases have no side effects except writing artifacts through temporary files that are renamed atomically on completion.
- Retrying never changes a sent message. After `Round1` is sent, every later message is a deterministic function of persisted state and verifier messages.
- Completeness failures and rejections are never retried automatically.
