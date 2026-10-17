# RFC-0023: The Norm's Precision: Choosing κ per Model

- Status: Accepted
- Authors: @AbdelStark
- Created: 2026-10-17
- Target milestone: v0.2
- Supersedes, on acceptance: [RFC-0005](RFC-0005-integer-model.md#integer-semantics) 'Integer semantics': `κ = 16` in the norm row
- Origin: [#287](https://github.com/AbdelStark/tessera/issues/287)

## Summary

`D`'s integer RMSNorm computes `ρ ≈ 2^κ / rms`, so `ρ` keeps about `κ − log2(rms)` bits. RFC-0005 fixed `κ = 16`, which leaves rows near the residual's range with a few bits, or none. This RFC chooses `κ` per model, as large as [RFC-0022](RFC-0022-saturating-residual.md#complete-norms)'s range check on `ρ` stays one 32-bit lookup: `κ = 30` at 8B. The circuit already takes `κ` from the claim list, so no gadget, table or schema changes.

## Motivation

RFC-0005 'Integer semantics' normalizes with `ρ = max{ρ : ρ^2 (s + e) ≤ d · 2^{2κ}}`, so `ρ ≈ 2^κ / rms`, with `rms = √((s + e) / d)` in the residual's integer units. `ρ` is an integer, so the scale it applies to a row is off by up to `1/ρ` relative.

The residual's scale gives the calibration maximum about `2^18` integer units (RFC-0005 'Scale policy'). At `κ = 16`, rows with outlier channels lose most of the precision ([#287](https://github.com/AbdelStark/tessera/issues/287)):

| Row `rms` | `ρ` at `κ = 16` | at `κ = 30` |
|---|---|---|
| `2^8` | 256: 0.4% | `2^22` |
| `2^12` | 16: 6% | `2^18` |
| `2^16` | 1 | `2^14` |
| `2^19`, saturated | 0: the norm outputs zeros | `2^11` |

For example, two channels at `2^18` in a row of 4096 already give `rms ≈ 2^12.5`.

## Goals

- Keep at least about 11 bits in `ρ` for every row INT20 allows, at no cost per token.
- Change nothing in the circuit, the claim list's schema or the compiler.

## Non-Goals

- A change to the norm's formula, to `e`, or to the residual's scale.

## Proposed Design

The quantizer sets `PublicQuantSpec.rsqrt_kappa` to

```text
κ = min(31, max{κ : ⌊d · 2^{2κ} / e⌋ < 2^64})
```

the largest `κ` for which RFC-0022's check `ρ < 2^w`, `w = bitlen(⌊√(d · 2^{2κ} / e)⌋)`, has `w ≤ 32`: one lookup. 31 is the IR's bound on `κ` (RFC-0006). Since `d < 2^32` and `e ≥ 1`, every `κ ≤ 16` fits, so `κ` is never below RFC-0005's 16.

- At 8B with `σ_res = 2^{−7}`: `e = 672`, `d / e ≈ 6.1`, so `κ = 30` and `w = 32`.
- At 405B with the same scale: `e = 2685`, `κ = 30`.
- Where `d / e < 4`, `κ = 31`.

### Cost

None per token. Each row's `BigIntCompare` witnesses of RFC-0009 'RsqrtRow' bound `ρ^2 u` by `2^{2w+55}`, `2^{119}` at `w = 32`: two more 16-bit limbs per comparison, once per row and norm.

The norm's key `res · ρ` reaches `2^{19+w} = 2^{51}`, which RFC-0005's `κ = 16` already split. Its split point is `s = 52 − 32 = 20`, so `RequantSplit` needs `r_n ≥ 21`; `r_n = κ + e_y` is 27 at `e_y = −3`.

### Changes to other documents

- RFC-0005 'Integer semantics': the norm row's `κ` is chosen by this rule.
- RFC-0007 'Schema': `rsqrt_kappa`'s comment.

The compiler accepts any `κ ≤ 31` whose `w` fits one lookup, as RFC-0022 made it. It does not enforce the rule: another `κ` gives a valid `D` of lower precision.

## Alternatives Considered

1. **Keep `κ = 16`, normalize the residual before the root.** That costs a requantization per element per norm, where the larger `κ` costs nothing.
2. **A fixed larger `κ`, say 24.** `d / e` varies with the model and its residual scale, and a fixed value either wastes precision or exceeds one lookup.
3. **A wider range check than 32 bits**, through limbs. More precision than `κ = 30` gives is not needed: the norm's INT8 output keeps 7 bits.

## Drawbacks

- `κ` now depends on `d` and `e`, so models of different residual scales use different `κ`. The claim list states it, as it did.
- The norm's key grows from `2^37` to `2^51`. Both need `RequantSplit`, and `r_n` grows by the same 14 bits.

## Migration / Rollout

No model has been released. The quantizer (#81) writes this `κ`. The PyTorch reference (#89) and the CPU engine (#82) read it from the claim list's data. The conformance vectors of `spec/vectors/model/` pin encodings, and keep their `κ = 16`.

## Testing Strategy

- A property test checks the rule's defining inequalities over `d` and `e`, its cap at 31, and its values at 8B and 405B.
- 8B's and 405B's shapes lower at their `κ`. Every root's range is `[0, 2^32 − 1]`, and every norm key passes range analysis.

## Open Questions

None.

## References

- [RFC-0005](RFC-0005-integer-model.md), [RFC-0009](RFC-0009-gadgets.md), [RFC-0022](RFC-0022-saturating-residual.md).
