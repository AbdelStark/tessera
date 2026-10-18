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
    pub rope: RopeSpec,
    pub rms_eps: Rational,             // exact rational, never a float
}

pub enum ArchFamily { LlamaV3 }

pub struct RopeSpec {                  // Llama-3.1's RoPE: the integer parameters of RFC-0008's `Rope` table
    pub theta: u32, pub factor: u16, pub low_freq_factor: u16, pub high_freq_factor: u16,
    pub original_max: u32,             // original_max_position_embeddings
    pub bits: u8,                      // the constants' precision: 14 in D
}

pub enum Profile { Servable, Minimal } // RFC-0005 'Profiles'

pub enum WeightType { Embed, Q, K, V, O, Gate, Up, Down, LmHead }   // 9 types

pub struct IntegerModel {              // artifact "D", schema 0x0101
    pub arch: ArchSpec,
    pub profile: Profile,
    pub quant: QuantSpec,
    pub weights: Vec<WeightTensor>,    // int8, row-major, ordered by (WeightType, layer)
    pub provenance: Provenance,
}

pub struct WeightTensor { pub ty: WeightType, pub layer: u32, pub rows: u32, pub cols: u32, pub data: Vec<i8> }
                                       // rows: output channels; Embed is d_model × vocab, transposed
```

The source names seven per-layer weight types (source §7). The data model adds `Embed` and `LmHead`: the embedding is applied as a matmul with a committed one-hot token vector, and the LM head produces the logits ([RFC-0006](../rfcs/RFC-0006-compiler-ir.md#llama-lowering)). A weight's rows are its output channels and its columns its inputs, as `MatMulWeight` applies it. So `Embed` holds the transposed embedding, `d_model × vocab`, as RFC-0012 lays it out. RMSNorm gains are folded into the following weights at quantization time and do not appear as separate tensors. Per-channel quantities derived from weights (power-of-two shift classes, affine multipliers) are secret: they are committed with the weights as channel-parameter vectors ([RFC-0012](../rfcs/RFC-0012-weight-commitment.md#layout)) and enter lookups through tuple keys, so the claim list never reveals them ([RFC-0007](../rfcs/RFC-0007-claim-list.md#public-and-hidden-quantization-data)).

### Quantization

`QuantSpec` holds every constant of RFC-0005 'Integer semantics', each an integer or a rational (INV-DM-6). Its public part is the claim list's `PublicQuantSpec` ([RFC-0007](../rfcs/RFC-0007-claim-list.md#schema)), field for field; its hidden part is what RFC-0012 commits with the weights as the `ChannelParams` matrix.

```rust
pub struct QuantSpec {
    pub public: PublicQuantSpec,       // RFC-0007, the copy the claim list carries
    pub silu_in_exponents: Vec<(u32, i8)>,  // per layer: in_exp of its Silu table; public, in the claim list (RFC-0042)
    pub channels: Vec<ChannelParams>,  // hidden; one per weight whose outputs a classed site rescales, sorted by (ty, layer)
}

pub struct ChannelParams {             // RFC-0012 'Layout', the ChannelParams matrix
    pub ty: WeightType, pub layer: u32,
    pub class: Vec<u8>,                // per output channel: its index into its site's shift classes; empty at an affine site
    pub multiplier: Vec<u16>,          // per output channel: M_c in affine mode; empty in pot mode
}

pub enum QuantPoint {                  // a quantized or rescaled tensor; sorted as declared
    Embed,                             // the INT8 embedding rows x0
    Layer { layer: u32, site: Site },
    FinalNorm,                         // y before the LM head
}

pub enum Site {                        // the rescales of a layer, in the order of RFC-0005 'Integer semantics'
    AttnNorm,                          // y = rq(res · ρ, r_n)
    RopeQ, RopeK,                      // q and k after RoPE; their classes follow W_Q's and W_K's channels
    V,                                 // v = rq(acc, r_v,c); classes from W_V
    AttnOut,                           // o8 = rq(o, r_o); servable only
    AttnResidual,                      // the O projection's accumulator into the residual; classes from W_O
    MlpNorm,
    Silu,                              // a = T_silu[g]; classes from W_Gate
    Up,                                // u8 = rq(u, r_u); servable only; classes from W_Up
    Product,                           // h = rq(a · u8, r_h); servable only
    MlpResidual,                       // the down projection's accumulator into the residual; classes from W_Down
}
```

- **Logits.** A logit `ℓ` stands for `ℓ · 2^{logits_exponent}`: the final norm's exponent plus the LM head's ([RFC-0027](../rfcs/RFC-0027-logits-exponent.md)).
- **Points.** `PublicQuantSpec.act_exponents` gives the exponent `e` of each INT8 tensor's scale `2^e`, `shift_class_sets` each `pot` rescale site's class shifts `r`, from 1 to 63, at most four (RFC-0005 'Scale policy'), and `affine_sites` each affine site's one shift ([RFC-0025](../rfcs/RFC-0025-affine-sites.md)). Each rescale site is in one list. They are the `r_by_class` of the site's `Requant` table (RFC-0008). A site after a weight takes the classes of that weight's output channels. Every other site has one class. The embedding's shift into the residual is the difference of their exponents: `res = x0 · 2^{e_Embed − residual_exponent}`.
- **SiLU.** Each layer's `Silu` table computes `silu(h · 2^{in_exp})` for `h = g >> r_c`, the gate accumulator `g` shifted by its channel's class shift. The quantizer chooses `in_exp`, and with it each `r_c = in_exp − e_y − e_c` for the hidden weight exponents `e_c`, so public data cannot give it: `silu_in_exponents` records it, sorted by layer. It is public: the claim list states it beside its `PublicQuantSpec`, as `QuantSpec` holds it beside its public part, and the layer's `Silu` table repeats it in its `TableSpec`. A table that several layers share does not say which layers read it, so the table alone would not do ([RFC-0042](../rfcs/RFC-0042-silu-exponents.md)). The table's other parameters come from `PublicQuantSpec`: its `out_exp` is the `Silu` point's exponent, its `r_by_class` the site's shift classes.
- **Channels.** A `ChannelParams` exists for each weight followed by a site with more than one class, or by an affine site. At a site of several classes, `class[c]` indexes the site's shifts and `multiplier` is empty. At an affine site, `multiplier[c]` is `M_c`, from 1 to `2^{16} − 1`, and `class` is empty ([RFC-0025](../rfcs/RFC-0025-affine-sites.md)).

### Provenance

```rust
pub struct Provenance {                // RFC-0005 'Determinism'
    pub tessera_quant: Ascii,          // the version that wrote D
    pub checkpoint: Vec<(Ascii, Digest)>,   // each safetensors file's name and BLAKE3 digest, sorted by name
    pub calibration: CalibrationIdentity,
    pub statistics: Digest,            // the statistics digest of RFC-0005 'Calibration'
    pub alpha: Rational,               // the migration strength chosen
    pub libraries: Map<Ascii, Ascii>,  // name to version: python, torch, transformers, safetensors, numpy, datasets, blake3
}

pub struct CalibrationIdentity {       // RFC-0005 'Calibration'
    pub dataset: Ascii, pub config: Ascii, pub split: Ascii,
    pub revision: Ascii,               // the dataset's commit, 40 hex digits
    pub files: Vec<Ascii>,             // in the order read
    pub seed: u64, pub sequences: u32, pub held_out: u32, pub tokens_per_sequence: u32,
    pub selection: Digest,             // the selection digest
}
```

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

pub struct TableSpec {                         // RFC-0008 'Table specifications'
    pub id: TableId,
    pub kind: TableKind,                       // Range | Requant | Pow2 | Bitwise8 | ..., with its
                                               // parameters: exact integers and rationals only
    pub log2_rows: u8,                         // <= 32
    pub digest: Digest,                        // H("table", ...) over the defining data, not the rows
}
```

## 4. Statement and protocol types

```rust
pub struct ClaimList { /* RFC-0007, RFC-0026; schema 0x0201 */ }
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

Protocol messages (`Round1`, `SeedCommit`, `ProverSeedShare`, `SeedReveal`, `Round2Response`, `ColumnQueries`, `Round3Response`, `NiProof`) are defined in [RFC-0036](../rfcs/RFC-0036-protocol-messages.md), in the session order of [RFC-0014](../rfcs/RFC-0014-interaction-and-soundness.md#session-order-live-mode). Transcript-anchor types (`DigestLogEntry`, `SessionOpen`, `LogManifest`) are defined in [RFC-0015](../rfcs/RFC-0015-transcript-anchor.md#log-format). `AuditSample` is defined in [RFC-0016](../rfcs/RFC-0016-audit-sampling.md#sample-derivation).

## 5. Artifact catalog

| Artifact | Schema id | v1 version | Extension | Class | Defined in |
|---|---|---|---|---|---|
| `IntegerModel` | `0x0101` | 1 | `.d` | Secret | RFC-0005 |
| `ClaimList` | `0x0201` | 1 | `.tcl` | Public | RFC-0007 |
| `ExecutionPlan` | `0x0202` | 1 | `.tep` | Private | RFC-0041 |
| `WeightCommitmentPublic` | `0x0301` | 1 | `.twr` | Public | RFC-0040 |
| `WeightCommitmentState` | `0x0302` | 1 | `.twc` | Secret | RFC-0040 |
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

Schema `0x0203` and the extension `.tts`, once named for a `TableSetManifest`, are retired and stay unassigned: the claim list's `tables` is the table set ([RFC-0029](../rfcs/RFC-0029-table-set.md)).

"Protocol" class means the artifact is sent to the verifier and is zero-knowledge by construction; it may be shared but must not be altered.

JSON reports and specifications are not TCE artifacts. Each carries a `"schema"` string (`tessera.commitment-report/1`, `tessera.cost-model/1`, `tessera.microbenchmarks/1`, `tessera.unit-costs/1`, `tessera.fidelity/1`, `tessera.verdict/1`, `tessera.audit-report/1`, `tessera.run-manifest/1`, `tessera.calibration/1`, and the `tessera-quant` specifications `tessera.calibration-spec/1` and `tessera.eval-spec/1`) and is validated against a JSON Schema file under `spec/schemas/`. JSON reports are never hashed into transcripts or commitments.

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
- Golden vectors for every artifact type and version live under `spec/vectors/`: in `codec/`, or beside the generator that writes them (`model/`, `claims/`, `protocol/`). Every decoder of the type checks them: the prover's, and the verifier's for the artifacts the verifier reads.
