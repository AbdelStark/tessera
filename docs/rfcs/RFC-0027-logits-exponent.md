# RFC-0027: The Scale of `D`'s Logits

- Status: Accepted
- Authors: @AbdelStark
- Created: 2026-10-17
- Target milestone: v0.2
- Supersedes, on acceptance:
  - [RFC-0007](RFC-0007-claim-list.md#schema) 'Schema': `PublicQuantSpec` gains `logits_exponent`;
  - [RFC-0026](RFC-0026-claim-list-layout.md#motivation) 'Motivation': no public field carries the logits' scale.
- Origin: [#309](https://github.com/AbdelStark/tessera/issues/309)

## Summary

`D`'s logits `ℓ = W_lm y` are integers whose real value is `ℓ · 2^{e_y + e_lm}`. That scale is the final norm's exponent plus the LM head's per-tensor exponent ([RFC-0005](RFC-0005-integer-model.md#scale-policy)). `PublicQuantSpec` carries `e_y` but not `e_lm`, so nothing that reads `D` knows what its logits mean. This RFC adds `logits_exponent = e_y + e_lm`: `ℓ` stands for `ℓ · 2^{logits_exponent}`.

## Motivation

- **Fidelity.** RFC-0005 'Fidelity evaluation' compares `D` with the BF16 reference by perplexity and KL divergence. Both take a softmax of the logits, which depends on their scale. Only top-1 agreement does not. `tessera_quant.quantize` gets the scale from the quantizer's own output while it chooses `α` (#86). An exported `D` does not carry it, so `evaluate_fidelity(model, ...)` of 02 §4 could not compute two of its three metrics (#90).
- **Scoring.** Temperature sampling and the mixture build `Exp_T` from `σ_ℓ / T` ([RFC-0017](RFC-0017-scoring.md#ds-decoded-token)). RFC-0026 noted that no public field carries `σ_ℓ`.
- **Integer semantics.** Greedy decoding and every operation upstream of the logits are unaffected: the logits are not rescaled.

## Proposed Design

```rust
pub struct PublicQuantSpec {
    // the fields of RFC-0007 and RFC-0025, unchanged, then:
    pub logits_bits: u8,                           // 28
    pub logits_exponent: i8,                       // ℓ stands for ℓ · 2^logits_exponent
}
```

- **The value.** The quantizer writes `logits_exponent = e_FinalNorm + e_lm`, its LM head's exponent plus the final norm's. RFC-0005 'Scale policy' already gives the LM head one exponent per tensor, so that its logits compare across the vocabulary.
- **Public.** It is a per-tensor statistic of the weights, public like every other exponent (RFC-0007 'Public and hidden quantization data', INV-CL-1). The claim list carries it inside `PublicQuantSpec`, for the scoring template of #152 and #155.
- **No new rule.** The compiler and the verifier do not read it: no constraint depends on it until scoring does. Any `i8` is accepted.
- **Encoding.** It follows `logits_bits`. No release has written an `IntegerModel` or a claim list, so the layout changes in place ([09 §3](../spec/09-release-and-versioning.md#3-protocol-and-schema-versions)), one byte longer. The vectors and the conformance models are regenerated.

## Alternatives Considered

1. **The LM head's exponent alone.** `e_y` is already public, so `e_lm` alone would do. But every consumer wants the sum. The sum also states what the logits mean, without a rule for combining two fields.
2. **A rational scale** in place of an exponent. The LM head's scale is a power of two by RFC-0005, and an exponent has one encoding.
3. **The scale in the scoring template only.** Fidelity evaluation reads `D`, not a claim list, so the scale belongs to `D`.

## Drawbacks

- One more field, which no constraint checks: a wrong exponent mis-scales fidelity and scoring, but not the forward pass. The quantizer's tests pin it to RFC-0005's rule.

## Migration / Rollout

v0.2, before #90's fidelity harness. Scoring reads it with #152 and #155.

## Testing Strategy

- The quantizer writes the LM head's exponent, from RFC-0005's rule on the folded weight, plus the final norm's.
- `quantize`'s divergence uses the field and agrees with an independent computation (#86's test).
- The encodings carry it in every implementation: the `IntegerModel` vectors, `tessera_quant`'s codec, the models' oracle.

## Open Questions

None.

## References

- [RFC-0005](RFC-0005-integer-model.md), [RFC-0007](RFC-0007-claim-list.md), [RFC-0017](RFC-0017-scoring.md), [RFC-0026](RFC-0026-claim-list-layout.md).
- [#309](https://github.com/AbdelStark/tessera/issues/309).
