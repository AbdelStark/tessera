# RFC-0007: Public Claim List and Statement Schema

- Status: Accepted
- Authors: @AbdelStark
- Created: 2026-10-15
- Target milestone: v0.2
- Amended by: [RFC-0025](RFC-0025-affine-sites.md) ('Schema': `PublicQuantSpec`'s affine sites; 'Public and hidden quantization data'; 'Verifier validation', step 5), [RFC-0026](RFC-0026-claim-list-layout.md) ('Schema': `scoring` and `anchor` join with their circuits, and `WeightLayoutParams`' fields; 'Verifier validation', step 7), [RFC-0027](RFC-0027-logits-exponent.md) ('Schema': `PublicQuantSpec`'s `logits_exponent`), [RFC-0030](RFC-0030-masked-exponentials.md) ('Verifier validation', step 5: `SoftmaxRow`'s table)

## Summary

The claim list is the compact, public, canonical description of what a proof certifies about `D`: architecture, profile, public quantization parameters, table definitions, gadget policy, commitment layout and protocol parameters. It is not a constraint system. The verifier runs its own independent lowering of the claim list into constraints, re-derives every range, regenerates every table, and rejects claim lists that are not provably sound. The per-batch `PublicStatement` binds a claim list to a weight commitment, a recorder manifest and a batch. This RFC fixes both schemas, the split between public and hidden quantization data, and the validation the verifier performs.

## Motivation

The source keeps VerInf's rule that "the verifier recompiles every constraint from the public claim list" because it yields "a small, auditable trusted base" (source §4 table; §6 principle 12). Two consequences drive this design:

1. The claim list must be high-level enough that a human auditor can read it as "Llama-3.1-405B, servable profile, these scales, these tables" and the verifier can rebuild the circuit with code that shares nothing with the prover.
2. A prover that controls the compiler could emit a claim list whose lookup keys may wrap, admitting false witnesses. The verifier must therefore re-derive soundness-relevant facts rather than trust them (SP-7 in [06](../spec/06-security.md#4-security-properties)).

## Goals

- One canonical encoding and one digest per claim list.
- Everything the verifier needs to build constraints is in the claim list or the `PublicStatement`; nothing else.
- Nothing derived from secret weights appears in it.
- The verifier can reject an unsound claim list before looking at any proof.

## Non-Goals

- Hiding the architecture (NG-2).
- Explicit constraint lists or circuit descriptions; the verifier derives them.

## Proposed Design

### Schema

```rust
pub struct ClaimList {                       // schema 0x0201, version 1
    pub protocol_version: u16,
    pub arch: ArchSpec,                      // 03 §2
    pub profile: Profile,                    // Servable | Minimal
    pub quant: PublicQuantSpec,
    pub silu_in_exponents: Vec<(u32, i8)>,   // each layer's Silu input exponent, sorted (RFC-0042)
    pub tables: Vec<TableSpec>,              // definitions + digests, sorted by TableId
    pub gadget_policy: GadgetPolicy,
    pub layout: CommitmentLayout,
    pub params: ParamSetId,                  // RFC-0014
}                                            // scoring and anchor join with their circuits (RFC-0026)

pub struct PublicQuantSpec {
    pub act_exponents: Vec<(QuantPoint, i8)>,      // per-tensor activation scales 2^e, sorted
    pub residual_exponent: i8, pub residual_bits: u8,   // 20
    pub rsqrt_kappa: u8,                           // RFC-0023: 30 at 8B
    pub rms_eps_int: u64,
    pub exp_sigma: u8,                             // 16
    pub score_scales: Vec<(u32 /*layer*/, Rational)>,   // σ_q σ_k / sqrt(d_head)
    pub shift_class_sets: Vec<(QuantPoint, Vec<i8>)>,   // pot sites: the set of classes per tensor, not the assignment
    pub affine_sites: Vec<(QuantPoint, i8)>,            // affine sites: the one shift r (RFC-0025)
    pub rope: RopeSpec,
    pub logits_bits: u8,                           // 28
    pub logits_exponent: i8,                       // RFC-0027: a logit ℓ stands for ℓ · 2^logits_exponent
}

pub struct GadgetPolicy { pub max_log2_rows: u8 /* B_max = 32 */, pub query_block: u32 /* 512 */,
                          pub limb_bits: u8 /* 16 */ }

pub struct CommitmentLayout {               // the parameter set's values (RFC-0026)
    pub advice: AdviceLayoutParams,          // log2 k, log2 inverse rate, P_A, row block B, segments S (RFC-0011)
    pub weights: WeightLayoutParams,         // log2 k_W, log2 inverse rate, P_W, B_W (RFC-0012, RFC-0026)
}
```

`QuantPoint` names a quantized or rescaled tensor, as [03 §2](../spec/03-data-model.md#quantization) defines it, and orders both lists. `shift_class_sets` gives each `pot` rescale site its class shifts `r`, from 1 to 63: the `r_by_class` of its `Requant` table. `affine_sites` gives each affine site its one shift, from 1 to 63; its key is multiplied by the hidden 16-bit multipliers of the weight before it ([RFC-0025](RFC-0025-affine-sites.md)). Each rescale site is in exactly one of the two lists, and both are sorted. `IntegerModel`'s `QuantSpec` holds this struct as its public part, beside the hidden per-channel classes.

`claim_list_digest = H("claim-list", TCE(ClaimList))` (RFC-0004).

### Public and hidden quantization data

| Data | Public or hidden | Why |
|---|---|---|
| Architecture dimensions, RoPE parameters, `rms_eps` | Public | Architecture is revealed (source §10) |
| Per-tensor activation exponents, residual exponent, score scales | Public | Needed to define tables; derived from calibration statistics, not weight values |
| Set of shift classes per tensor (at most 4 values) | Public | Defines the tuple table's domain |
| Affine sites and their shifts; the multipliers' width, 16 bits | Public | Define the key, its range and its tables ([RFC-0025](RFC-0025-affine-sites.md)) |
| Assignment of each output channel to a shift class; affine multipliers | Hidden | Derived from per-channel weight magnitudes; committed with the weights as channel-parameter vectors ([RFC-0012](RFC-0012-weight-commitment.md#layout)) |
| Weights | Hidden | Committed |

Hidden per-channel classes enter lookups through tuple keys `(class_c, acc)`: the key is the virtual linear combination `acc + 2^b · class_c` of a virtual accumulator and a committed channel parameter, so the lookup itself proves the right shift was applied without revealing it ([RFC-0009](RFC-0009-gadgets.md#requant1)). The quantizer limits each tensor to at most 4 consecutive exponent classes so that tuple tables stay within `2^{B_max}` rows.

### Public statement

```rust
pub struct PublicStatement {              // 03 §4; SessionContext.statement_digest = H("statement", TCE(self))
    pub claim_list_digest: Digest,
    pub weights_root: Digest, pub weights_epoch: u32,
    pub manifest_digest: Digest,
    pub batch: BatchDescriptor,           // request ids, session ids, prompt and output lengths
    pub predictor: PredictorSpec,         // instantiated ScoringTemplate: mode, ε exponent, temperature per request
}
```

Everything a proof depends on is a function of `ClaimList`, `PublicStatement`, the seed and prover messages (INV-DM-3).

### Verifier validation

Before reading any proof message, the verifier runs these checks in order. The first failure is `Reject(Params)` or `Reject(StatementMismatch)`, whose field, given in parentheses, names the check ([RFC-0039](RFC-0039-statement-fields.md#the-checks)):

1. `params` names an allowed parameter set (`test-small` only with the `test-params` feature), else `Reject(Params)`: [RFC-0019](RFC-0019-verifier.md#check-order) runs this check first, as its step 2. `protocol_version` is supported (`ProtocolVersion`).
2. Architecture and quantization, in this order:
   1. Architecture consistency (`Architecture`): `n_layers ≥ 1`; `n_kv_heads ≥ 1` divides `n_heads ≥ 1`; `d_head` is even, from 2 to `2^{16} − 2`; `d_model = n_heads · d_head`; `d_ff ≥ 1`; `vocab` and `max_context` within protocol limits (v1: `1 ≤ vocab ≤ 2^18`, `1 ≤ max_context ≤ 2^17`).
   2. The public quantization data (`Quantization`):
      - `quant.rope` is `arch.rope`;
      - `act_exponents` holds the embedding, each INT8 site of the profile and the final norm, once each and sorted;
      - each rescale site is in exactly one of `shift_class_sets` and `affine_sites`, both sorted;
      - an affine site follows the Q, K, V, O, up or down projection, with a shift from 1 to 63 ([RFC-0025](RFC-0025-affine-sites.md#lowering));
      - a `pot` site has one to four classes, each from 1 to 63, and one class if it follows no weight;
      - `score_scales` holds one positive scale per layer, sorted;
      - for each layer's `SoftmaxRow`, the narrowest `key_bits` from the spread's width, `bitlen(128 · 255 · d_head)`, up whose table `Exp { key_bits, sigma: exp_sigma, scale }`, `scale` the layer's score scale, maps its last key `2^{key_bits} − 1` to 0 exists, and RFC-0008 allows that table: `1 ≤ key_bits ≤ 32`, `σ ≤ 32`, `scale > 0`, its zero tail starting by key `2^{24}` ([RFC-0008](RFC-0008-logup-and-tables.md#table-specifications) 'Table specifications', [RFC-0030](RFC-0030-masked-exponentials.md#range-analysis-and-validation) 'Range analysis and validation');
      - `silu_in_exponents` holds one exponent per layer, sorted ([RFC-0042](RFC-0042-silu-exponents.md));
      - `residual_bits` lies from 2 to 31;
      - the embedding's shift `s_e` into the residual is non-negative, `s_e ≥ 0` ([RFC-0005](RFC-0005-integer-model.md#scale-policy)), and keeps `127 · 2^{s_e} < 2^{residual_bits − 1}`;
      - `rms_eps_int ≥ 1` and `rsqrt_kappa ≤ 31`.
   3. Every layer's sites are rescaled alike ([RFC-0038](RFC-0038-uniform-layers.md)): each site is affine in every layer or in none, and a `pot` site has one class in every layer or several in every layer (`Rescaling { site }`, the first site in 03 §2's declaration order that breaks it).
3. Profile constraints hold for the derived graph (S1–S3 of [RFC-0005](RFC-0005-integer-model.md#profiles)) when the profile claims them (`Profile`).
4. Tables: each `TableSpec` is regenerated from its definition by the verifier's own generator; the digest must match (`Table { table }`, in `TableId` order).
5. Ranges, in this order:
   1. `max_log2_rows` lies from 1 to 32 (`GadgetPolicy`).
   2. The verifier lowers the claim list into its own graph and re-derives every range from public data with its own analysis (INV-IR-5), affine sites included ([RFC-0025](RFC-0025-affine-sites.md)). At each gadget-backed node, in node order (`Gadget { gadget }`):
      - some gadget takes its key within tables of at most `2^{max_log2_rows}` rows;
      - its inputs lie in their domains and in the centered field range;
      - its gadget's preconditions hold, those of a split gadget among them;
      - a `SoftmaxRow`'s `Exp` table maps its last key to 0 ([RFC-0030](RFC-0030-masked-exponentials.md#range-analysis-and-validation)).
   3. The tables the lowering derives are the claim list's, id for id (`Table { table }`, the first id that differs). A lookup names a lookup table, never a `Rope` table ([RFC-0008](RFC-0008-logup-and-tables.md#table-kinds)): no lowering derives one for a lookup.
6. Layout: commitment parameters are those the parameter set prescribes, and the advice layout derives and fits within the protocol's maximum matrix size (`Layout`). Its attention tiles take `log2 query_block` ([RFC-0034](RFC-0034-causal-blocks.md#causal-layout)), so `query_block` is a power of two, and a row holds one key against a tile's queries and heads: `g·min(query_block, N_max) ≤ k` ([RFC-0046](RFC-0046-advice-segments.md#rows-too-short-for-a-key)).
7. Scoring and anchor templates name supported variants, once they join the claim list ([RFC-0026](RFC-0026-claim-list-layout.md)) (`Template`).

The derived constraint system has a canonical digest (`ConstraintSystemDigest`, computed by both the compiler and the verifier with independent code). It is not part of the protocol; CI compares the two digests on every conformance model to detect lowering divergence early.

### Invariants

- **INV-CL-1.** No field of `ClaimList` is a function of secret weight values.
- **INV-CL-2.** The verifier's lowering of a claim list depends only on the claim list and the parameter set.
- **INV-CL-3.** Two claim lists with the same digest produce the same constraint system in any correct verifier.

## Alternatives Considered

1. **An explicit constraint system as the public artifact.** Considered because the verifier would only parse, not lower. Rejected: the artifact would be gigabytes for 405B, unreadable by auditors, and the verifier would still have to check that it encodes the right model, which is the lowering problem again.
2. **Public per-channel shift classes.** Considered because untagged keys need smaller tables. Rejected: classes are derived from per-channel weight magnitudes, about 2 bits per channel, which the source's "weights hidden" rule does not permit. The tuple-key cost is bounded by limiting classes to 4.
3. **Trusting compiler-declared ranges.** Rejected: SP-7. A prover-chosen range that admits wraps is a soundness hole.
4. **A claim list per request.** Rejected: the claim list describes `D` and the proof system, which do not change per request; per-batch data lives in `PublicStatement`.

## Drawbacks

- The verifier must contain a full lowering and range analysis, a large part of the trusted base.
- Limiting shift classes to 4 per tensor costs some precision on channels with unusually small magnitudes.
- Lowering divergence between prover and verifier surfaces as a rejection of honest proofs until the conformance digests catch it.

## Migration / Rollout

The schema lands in v0.2 with the compiler. The verifier's validation lands in v0.3 with the first verifier driver. Version 1 of the schema is frozen with `tessera-protocol/1`.

## Testing Strategy

- Golden claim lists for the conformance models, decoded by both workspaces.
- Constraint-system digests from the compiler and the verifier agree on every conformance model and on 8B and 405B shapes.
- Negative claim lists, each rejected at the expected validation step with its field ([RFC-0039](RFC-0039-statement-fields.md)): inconsistent head counts; a tampered table digest; a table definition whose regenerated digest differs; a `GadgetPolicy` with `max_log2_rows = 40`; a claim list that marks a wide key as `Requant1`; an unsupported parameter set; `test-small` without the feature.
- INV-CL-1: a test perturbs weights (keeping per-tensor statistics fixed) and asserts the claim list bytes are unchanged.

## Open Questions

None.

## References

- Source §4 (verifier recompiles from claim list), §6 principle 12, §7, §10 (architecture revealed).
- [RFC-0006](RFC-0006-compiler-ir.md), [RFC-0008](RFC-0008-logup-and-tables.md), [RFC-0019](RFC-0019-verifier.md).
