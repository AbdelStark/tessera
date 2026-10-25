# RFC-0026: The Claim List's Commitment Layout, Scoring and Anchor

- Status: Accepted
- Authors: @AbdelStark
- Created: 2026-10-17
- Target milestone: v0.2
- Supersedes, on acceptance:
  - [RFC-0007](RFC-0007-claim-list.md#schema) 'Schema': `ClaimList`'s `scoring` and `anchor` fields, and `CommitmentLayout`'s `WeightLayoutParams`; 'Verifier validation', step 7;
  - [RFC-0012](RFC-0012-weight-commitment.md#layout) 'Layout': the fields of `WeightLayoutParams`, which `R_W` hashes.
- Amended by: [RFC-0065](RFC-0065-masked-channel-params.md) (`WeightLayoutParams.blinding_rows` is a single-claim matrix's `B_W`; `ChannelParams`' `B_CP` follows from it)
- Origin: [#85](https://github.com/AbdelStark/tessera/issues/85)
- Amended by: [RFC-0027](RFC-0027-logits-exponent.md) ('Motivation': `PublicQuantSpec` now carries the logits' scale)

## Summary

RFC-0007's `ClaimList` names three types that no RFC defines:

- **`WeightLayoutParams`** (RFC-0012) follows from RFC-0012's layout and RFC-0014's parameter sets, and this RFC defines it.
- **`ScoringTemplate`** (RFC-0017) would hold the model-level parameters of scoring circuits that land in v0.3 and v0.4. None is designed yet. Exact greedy decoding needs none. Temperature sampling and the mixture need the logits' scale, which no public field carries today.
- **`AnchorSpec`** (RFC-0015) is in the same position, for anchor circuits that land in v0.4.

This RFC leaves `scoring` and `anchor` out of the claim list until the issues that implement those circuits define what each states. The compiler can then emit a claim list in v0.2 with no field whose meaning is a guess, and no guessed field becomes tier A at the first release.

## Motivation

#85 implements the claim list's types and their emission. Every field the compiler emits is tier A from the first release that writes a claim list (02 §1, 09 §3). Emitting `ScoringTemplate` or `AnchorSpec` now would mean inventing their fields:

- **Scoring.** RFC-0017's verifier "recompiles the scoring circuit from `ScoringTemplate` (claim list) and `PredictorSpec` (statement)". `PredictorSpec` carries the decoding, temperature and `ε` per batch. What the claim list adds is per model:
  - **Greedy decoding** needs nothing: `Argmax` reads the logits ([RFC-0009](RFC-0009-gadgets.md#argmax-greedy-decoding)).
  - **Temperature sampling and the mixture** need the logits' scale `σ_ℓ = 2^{e_y + e_lm}` to build `Exp_T` ([RFC-0017](RFC-0017-scoring.md#ds-decoded-token)). `e_y` is public, in `act_exponents`, but the LM head's exponent `e_lm` is in no public field.
  - The vocabulary table's own parameters are not fixed yet either.
- **The anchor.** RFC-0015 fixes its circuits whole: token bytes, AES-128-CTR, SHA-256 and the key commitment. It names no parameter a model or deployment chooses.
- **The weights' layout.** `WeightLayoutParams` is also what `R_W` hashes ([RFC-0012](RFC-0012-weight-commitment.md#layout)). RFC-0007 describes it only as "per-type shape, rate, padding, blinding rows", and #102 needs its fields.

## Goals

- A claim list whose every field has a definition, emitted from public data alone (INV-CL-1).
- `WeightLayoutParams`' fields, for the claim list and for `R_W`.
- A stated way for scoring and the anchor to join the claim list later.

## Non-Goals

- Designing `ScoringTemplate` or `AnchorSpec`.
- The parameter sets' implementation (#103).

## Proposed Design

### Schema

```rust
pub struct ClaimList {                       // schema 0x0201, version 1
    pub protocol_version: u16,               // 0 until tessera-protocol/1 is assigned (09 §3)
    pub arch: ArchSpec,                      // 03 §2
    pub profile: Profile,
    pub quant: PublicQuantSpec,              // RFC-0007, RFC-0025
    pub silu_in_exponents: Vec<(u32, i8)>,   // each layer's Silu input exponent, sorted (RFC-0042)
    pub tables: Vec<TableSpec>,              // definitions and digests, sorted by TableId
    pub gadget_policy: GadgetPolicy,
    pub layout: CommitmentLayout,
    pub params: ParamSetId,                  // RFC-0014
}

pub struct CommitmentLayout {
    pub advice: AdviceLayoutParams,          // RFC-0011
    pub weights: WeightLayoutParams,         // RFC-0012
}

pub struct AdviceLayoutParams { pub log2_k: u8, pub log2_inv_rate: u8, pub mask_coeffs: u32,
                                pub row_block: u32, pub segments: u32 }

pub struct WeightLayoutParams {
    pub log2_k: u8,                          // the largest message width k_W = 2^log2_k
    pub log2_inv_rate: u8,                   // 2: rate 1/4
    pub mask_coeffs: u32,                    // P_W, each row's mask coefficients
    pub blinding_rows: u32,                  // B_W, a single-claim matrix's blinding rows; ChannelParams' B_CP follows (RFC-0065)
}
```

- **The weights' layout.** Each weight type's message width is `k_T = max(in_pad, min(2^{log2_k}, out_pad · in_pad))` ([RFC-0040](RFC-0040-weight-commitment-artifacts.md#the-matrices)). Its rows and columns, and the padding of its dimensions to powers of two, follow from `ArchSpec` ([RFC-0012](RFC-0012-weight-commitment.md#layout) 'Layout'), so the layout does not repeat them.
- **The parameter set fixes the layout.** Both parts are the values RFC-0014's table gives the parameter set `params`:

  | | `live-64/1` | `ni-100/1` | `test-small/1` |
  |---|---|---|---|
  | `advice` | 22, 1, 160, 32, 8 | 22, 1, 200, 32, 8 | 10, 1, 16, 4, 1 |
  | `weights` | 20, 2, 65,536, 4,096 | 20, 2, 65,536, 4,096 | 8, 2, 64, 32 |

  The compiler emits them with `params`, and the verifier checks that they are those values ([RFC-0007](RFC-0007-claim-list.md#verifier-validation) step 6).
- **Unchanged.** `claim_list_digest = H("claim-list", TCE(ClaimList))` is as RFC-0007 defines it.

### Scoring and the anchor

RFC-0007's `scoring` and `anchor` fields leave the schema. Each comes back when an implementation defines it, by amending RFC-0007's schema:

- **Scoring.** It comes back with the first predictor that needs a model-level public parameter: temperature sampling (#152) or the mixture (#155). Each needs the logits' scale. Greedy decoding (#127) needs none.
- **The anchor.** It comes back with the anchor circuits (#137 to #146), in v0.4.

While no release has written a claim list, the schema changes in place (09 §3). After that, a change is a new version of the schema and of the protocol. RFC-0007 'Verifier validation' step 7, "Scoring and anchor templates name supported variants", applies once the fields exist.

### Emission

The compiler emits the claim list from four inputs:

- the public part of `D`;
- the tables its lowering and gadget selection define;
- the gadget policy;
- the parameter set's identifier and layout.

It reads nothing hidden, so INV-CL-1 holds by construction. The caller passes the parameter set: it is `tessera-protocol`'s, downstream of the compiler (01 §3).

## Alternatives Considered

1. **Define `ScoringTemplate` and `AnchorSpec` now.** The scoring template would carry the logits' exponent; the anchor's would be one variant naming RFC-0015's construction. But the scoring template's fields depend on how the vocabulary softmax is built, which #152 and #155 design. A guessed field would bind the first release's claim lists, or need a new protocol version to change.
2. **Optional fields**, `Option<ScoringTemplate>` set to `None`, still need the types defined, and add one more state for the verifier to validate.
3. **Per-type shapes in `WeightLayoutParams`.** They repeat what `ArchSpec` fixes, and the verifier would have to check that the two agree.

## Drawbacks

- Until scoring joins, a claim list cannot state the logits' scale. Temperature and mixture proofs wait for it, as they wait for their circuits.
- Adding the fields later changes `claim_list_digest`, before the first release or with a protocol version.

## Migration / Rollout

The schema lands in v0.2 with #85. #152, #155 and the anchor issues add their fields. #102 implements `WeightLayoutParams` in `R_W`, and #103 the parameter sets.

## Testing Strategy

With #85:

- golden claim lists for the conformance models in `spec/vectors/claims/`, with their digests;
- INV-CL-1: perturbing the weights and channel classes leaves a claim list's bytes unchanged.

The verifier decodes the golden lists with its claim-list validation (#107).

## Open Questions

None.

## References

- [RFC-0007](RFC-0007-claim-list.md), [RFC-0011](RFC-0011-advice-commitment.md), [RFC-0012](RFC-0012-weight-commitment.md), [RFC-0014](RFC-0014-interaction-and-soundness.md), [RFC-0015](RFC-0015-transcript-anchor.md), [RFC-0017](RFC-0017-scoring.md).
- [#85](https://github.com/AbdelStark/tessera/issues/85).
