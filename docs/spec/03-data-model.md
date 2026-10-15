# 03 — Data Model

- Status: Accepted for v1 planning
- Governs: core types, artifact schemas, invariants, schema versioning
- Encoding rules: [RFC-0003](../rfcs/RFC-0003-canonical-encoding.md). Every struct below is encoded field by field in declaration order.

## 1. Scalar and algebraic types

```rust
/// Goldilocks element. Encoded as u64 little-endian; decoders reject values >= P.
pub struct Fp(u64);                       // P = 2^64 - 2^32 + 1
/// Quadratic extension element c0 + c1*X with X^2 = 7.
pub struct Fp2 { pub c0: Fp, pub c1: Fp }
pub type Digest = [u8; 32];               // BLAKE3 unless the field name says sha256
pub struct IntType { pub bits: u8, pub signed: bool }   // bits in 1..=64
pub struct IntRange { pub lo: i128, pub hi: i128 }      // inclusive, lo <= hi
```

Signed integers embed into `Fp` as `x mod P` (a negative `x` maps to `P - |x|`). A value is a **true integer** at a point in the circuit when its range lies within `(-(P-1)/2, (P-1)/2)` and no wrap has occurred upstream of that point that the range analysis did not account for ([RFC-0006](../rfcs/RFC-0006-compiler-ir.md#range-analysis)).

## 2. Model types

```rust
pub struct ArchSpec {
    pub family: ArchFamily,            // LlamaV3 only in v1
    pub n_layers: u32, pub d_model: u32, pub n_heads: u32, pub n_kv_heads: u32,
    pub d_head: u32, pub d_ff: u32, pub vocab: u32, pub max_context: u32,
    pub rope: RopeSpec,                // theta, Llama-3 frequency scaling parameters, constant bits (14)
    pub rms_eps: Rational,             // exact rational, never a float
}

pub enum WeightType { Embed, Q, K, V, O, Gate, Up, Down, LmHead }   // 9 types

pub struct IntegerModel {              // artifact "D", schema 0x0101
    pub arch: ArchSpec,
    pub profile: Profile,              // Servable | Minimal (RFC-0005)
    pub quant: QuantSpec,              // scales, shift classes, table parameters
    pub weights: Vec<WeightTensor>,    // int8, row-major, ordered by (WeightType, layer)
    pub provenance: Provenance,        // checkpoint digest, calibration digest, tool versions
}

pub struct WeightTensor { pub ty: WeightType, pub layer: u32, pub rows: u32, pub cols: u32, pub data: Vec<i8> }
```

The source names seven per-layer weight types (source §7). The data model adds `Embed` and `LmHead`: the embedding is applied as a matmul with a committed one-hot token vector, and the LM head produces the logits ([RFC-0006](../rfcs/RFC-0006-compiler-ir.md#llama-lowering)). RMSNorm gains are folded into the following weights at quantization time and do not appear as separate tensors. Per-channel quantities derived from weights (power-of-two shift classes, affine multipliers) are secret: they are committed with the weights as channel-parameter vectors ([RFC-0012](../rfcs/RFC-0012-weight-commitment.md#layout)) and enter lookups through tuple keys, so the claim list never reveals them ([RFC-0007](../rfcs/RFC-0007-claim-list.md#public-and-hidden-quantization-data)).

## 3. Circuit types

```rust
pub struct TensorId(u32); pub struct TableId(u32); pub struct StageId(u16); pub struct GadgetId(u32);

pub enum NodeKind {
    Virtual,                            // never committed; claims reduce through sumchecks
    Committed { commitment: CommitmentClass },  // Advice or Weight
    LookupOutput { table: TableId },    // committed value of a lookup
    Public,                             // known to the verifier (constants, shapes)
}

pub struct TensorSpec {
    pub id: TensorId, pub shape: Vec<u32>, pub ty: IntType,
    pub range: IntRange, pub kind: NodeKind,
}

pub struct TableSpec {
    pub id: TableId, pub kind: TableKind,      // Range | Shift | Exp | Silu | Log2Frac | Bitwise | Tuple ...
    pub params: TableParams,                   // exact integers and rationals only
    pub log2_rows: u8,                         // <= 32
    pub digest: Digest,                        // BLAKE3 over the canonical row encoding
}
```

## 4. Statement and protocol types

```rust
pub struct ClaimList { /* RFC-0007; schema 0x0201 */ }
pub struct PublicStatement {
    pub claim_list_digest: Digest,
    pub weights_root: Digest, pub weights_epoch: u32,
    pub manifest_digest: Digest,               // LogManifest the batch is anchored to
    pub batch: BatchDescriptor,                // request ids, prompt and output lengths
    pub predictor: PredictorSpec,              // RFC-0017
}
pub struct BatchDescriptor { pub requests: Vec<RequestRef> }
pub struct RequestRef { pub request_id: u64, pub session_id: [u8; 16], pub prompt_len: u32, pub output_len: u32 }
```

Protocol messages (`Round1`, `SeedCommit`, `ProverSeedShare`, `SeedReveal`, `Round2Response`, `ColumnQueries`, `Round3Response`, `NiProof`) are defined in [RFC-0014](../rfcs/RFC-0014-interaction-and-soundness.md#messages). Transcript-anchor types (`DigestLogEntry`, `SessionOpen`, `LogManifest`) are defined in [RFC-0015](../rfcs/RFC-0015-transcript-anchor.md#log-format). `AuditSample` is defined in [RFC-0016](../rfcs/RFC-0016-audit-sampling.md#sample-derivation).

## 5. Artifact catalog

| Artifact | Schema id | v1 version | Extension | Class | Defined in |
|---|---|---|---|---|---|
| `IntegerModel` | `0x0101` | 1 | `.d` | Secret | RFC-0005 |
| `ClaimList` | `0x0201` | 1 | `.tcl` | Public | RFC-0007 |
| `ExecutionPlan` | `0x0202` | 1 | `.tep` | Private | RFC-0006 |
| `TableSetManifest` | `0x0203` | 1 | `.tts` | Public | RFC-0008 |
| `WeightCommitmentPublic` | `0x0301` | 1 | `.twr` | Public | RFC-0012 |
| `WeightCommitmentState` | `0x0302` | 1 | `.twc` | Secret | RFC-0012 |
| `WeightLinkProof` | `0x0303` | 1 | `.twl` | Public | RFC-0012 |
| `DigestLogEntry` | `0x0401` | 1 | stream | Public | RFC-0015 |
| `LogManifest` | `0x0402` | 1 | `.tlm` | Public | RFC-0015 |
| `AuditSample` | `0x0403` | 1 | `.tas` | Public | RFC-0016 |
| `SeedCommit`, `ProverSeedShare`, `SeedReveal` | `0x0501`–`0x0503` | 1 | message | Public | RFC-0014 |
| `Round1`, `Round2Response`, `ColumnQueries`, `Round3Response` | `0x0510`–`0x0513` | 1 | message | Protocol | RFC-0014 |
| `NiProof` | `0x0520` | 1 | `.tnp` | Protocol | RFC-0014 |
| `TokenFile` | `0x0601` | 1 | `.ttk` | Secret | RFC-0003 |
| `LogitsFile` | `0x0602` | 1 | `.tlg` | Secret | RFC-0003 |
| `VerifierLedger` | `0x0701` | 1 | `.tvl` | Private (verifier-local) | RFC-0019 |

"Protocol" class means the artifact is sent to the verifier and is zero-knowledge by construction; it may be shared but must not be altered.

JSON reports are not TCE artifacts. Each carries a `"schema"` string (`tessera.commitment-report/1`, `tessera.unit-costs/1`, `tessera.fidelity/1`, `tessera.verdict/1`, `tessera.audit-report/1`, `tessera.run-manifest/1`, `tessera.calibration/1`) and is validated against a JSON Schema file under `spec/schemas/`. JSON reports are never hashed into transcripts or commitments.

## 6. Invariants

| ID | Invariant | Enforced by |
|---|---|---|
| INV-DM-1 | Every TCE artifact has exactly one valid encoding; decode then encode is the identity on bytes. | Codec property tests |
| INV-DM-2 | Every `TensorSpec.range` over-approximates the values the tensor can take for every input admitted by `ArchSpec` and `QuantSpec`. | Range analysis; witness engine asserts ranges in debug builds |
| INV-DM-3 | Every value the verifier uses to rebuild constraints is bound by `claim_list_digest`, `weights_root`, or `manifest_digest`. | Claim-list schema review; verifier recompilation tests |
| INV-DM-4 | `IntegerModel.weights` is exactly the message committed under `weights_root` for the stated epoch. | Commitment construction; linking proof on refresh |
| INV-DM-5 | `TensorId`, `TableId`, `StageId` and `GadgetId` are dense indices assigned deterministically by the compiler from `IntegerModel` and `ParamSet` alone. | Compiler determinism test: two runs, identical bytes |
| INV-DM-6 | Integer-model quantities that the verifier recomputes (scales, table parameters, RoPE constants, epsilon) are exact integers or rationals. Floating-point values never enter a tier-A artifact. | Schema has no float type |
| INV-DM-7 | A `PublicStatement` names exactly one `LogManifest`; every request in the batch has entries in it. | Prover assembly check; verifier anchor check |

## 7. Schema versioning

- Each artifact envelope carries `(schema_id: u16, version: u16)`. Any change to field order, type or meaning increments `version`.
- Readers accept `version` and `version − 1`, never version 0, which no writer emits; writers emit only the current version ([09 §3](09-release-and-versioning.md#3-protocol-and-schema-versions)).
- A schema version bump for any tier-A artifact implies a protocol version bump.
- Golden vectors for every artifact type and version live in `spec/vectors/codec/` and are checked by both the prover-side and verifier-side decoders.
