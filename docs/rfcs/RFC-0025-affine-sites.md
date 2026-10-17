# RFC-0025: Affine Rescale Sites in the Claim List

- Status: Accepted
- Authors: @AbdelStark
- Created: 2026-10-17
- Target milestone: v0.2
- Supersedes, on acceptance:
  - [RFC-0007](RFC-0007-claim-list.md#schema) 'Schema': `PublicQuantSpec` gains `affine_sites`; 'Public and hidden quantization data': the rows of affine sites; 'Verifier validation', step 5: affine keys;
  - [RFC-0005](RFC-0005-integer-model.md#scale-policy) 'Scale policy': `affine` mode's activations, sites and semantics;
  - [RFC-0006](RFC-0006-compiler-ir.md#operation-set) 'Operation set': `Mul`'s broadcast; 'Llama lowering': affine sites, and the refusal of them;
  - [RFC-0010](RFC-0010-proof-schedule.md#stages) 'Stages': the stage count of `servable` with an affine RoPE site;
  - [03 §2](../spec/03-data-model.md#quantization) 'Quantization': `ChannelParams` of affine sites.
- Origin: [#274](https://github.com/AbdelStark/tessera/issues/274)

## Summary

RFC-0005 lets the quantizer rescale a tensor in `affine` mode: each output channel's accumulator is multiplied by a hidden 16-bit multiplier `M_c`, and the product is shifted by one public shift `r`. The claim list cannot say so. `PublicQuantSpec` lists each site's shift classes, and an affine site's one shift reads as a `pot` site of one class. A verifier lowering that claim list would leave out the multiplier, so it would build the wrong circuit and could not bound the key from public data (SP-7). This RFC adds the affine sites, each with its shift, to `PublicQuantSpec`. It also fixes how the compiler and the verifier lower them. The key is `x · M_c`, a `Mul` by the committed multipliers broadcast over positions, rescaled with the one shift. A residual site fuses it with the residual, as a site of one class does. The multipliers' range is public, `[0, 2^{16})`, so the key's range, its gadget and its tables follow from public data, as for any key.

## Motivation

[RFC-0005](RFC-0005-integer-model.md#scale-policy) 'Scale policy' keeps `affine` as a per-tensor fallback where `pot` misses a fidelity gate: the compiler switches the worst tensors to it until the gate passes ('Evaluation'). `tessera_quant.scales` implements it for the Q, K, V, O, up and down projections (#81). It writes:

- each affine site as `(point, [r])` in `shift_class_sets`;
- the multipliers in the hidden `ChannelParams`.

Public data therefore cannot tell an affine site from a `pot` site of one class, and the compiler refuses any model with a multiplier.

The verifier lowers the claim list with its own code and re-derives every range ([RFC-0007](RFC-0007-claim-list.md#verifier-validation)). For an affine site it needs three facts, none of which `PublicQuantSpec` carries:

- that the site is affine, so the key is `acc · M_c` and not `acc`;
- its shift `r`;
- the multipliers' range, which bounds the key.

Without them, a claim list for a model with one affine tensor cannot state the relation its proofs prove.

## Goals

- The claim list states every affine site and its shift. The verifier derives each affine site's key, range, gadget and tables from public data alone (SP-7).
- `pot` sites keep their claim-list entries and their lowering.
- No new table kind, gadget, transcript label or artifact schema.

## Non-Goals

- Which tensors are affine: RFC-0005's fidelity gates decide.
- A multiplier width other than RFC-0005's 16 bits.
- Affine sites whose key exceeds 62 bits: `minimal`'s O and down projections (below).
- Activation scales that are not powers of two.

## Proposed Design

### Public data

```rust
pub struct PublicQuantSpec {
    pub act_exponents: Vec<(QuantPoint, i8)>,
    pub residual_exponent: i8, pub residual_bits: u8,
    pub rsqrt_kappa: u8,
    pub rms_eps_int: u64,
    pub exp_sigma: u8,
    pub score_scales: Vec<(u32 /*layer*/, Rational)>,
    pub shift_class_sets: Vec<(QuantPoint, Vec<i8>)>,   // pot sites: the set of classes per site
    pub affine_sites: Vec<(QuantPoint, i8)>,            // affine sites: the shift r, sorted by point
    pub rope: RopeSpec,
    pub logits_bits: u8,
}
```

- **One list per site.** Every rescale site of the profile appears exactly once, in `shift_class_sets` or in `affine_sites`. Both lists are sorted by point.
- **Where.** An affine site follows one of the weights RFC-0005 lets take `affine`:

  | Site | Weight |
  |---|---|
  | `RopeQ`, `RopeK` | Q, K |
  | `V` | V |
  | `AttnResidual` | O |
  | `Up` (`servable` only) | up |
  | `MlpResidual` | down |

  The norms, `AttnOut`, `Silu`, `Product` and `FinalNorm` follow no such weight and stay `pot`.
- **The shift.** `r` lies in `[1, 63]`, like a class shift.
- **The multipliers.** Each is a 16-bit unsigned integer, by RFC-0005, so the claim list states no width.

### Hidden data

The weight before an affine site has a `ChannelParams` whose `multiplier` holds one `M_c ∈ [1, 2^{16})` per output channel and whose `class` is empty. The weight before a `pot` site of several classes has `class` and an empty `multiplier`. No other weight has a `ChannelParams` ([03 §2](../spec/03-data-model.md#quantization)). At `RopeQ` and `RopeK`, rows `i` and `i + d_head/2` of each head share their multiplier, as they share a scale ([RFC-0005](RFC-0005-integer-model.md#scale-policy) 'RoPE pairs').

The multipliers are committed with the weights, in the `ChannelParams` matrix of [RFC-0012](RFC-0012-weight-commitment.md#layout), row `(type, layer)`, like the class indices.

### Semantics

[RFC-0005](RFC-0005-integer-model.md#integer-semantics)'s operations change only at affine sites:

- **An INT8 site** computes `y_c = rq(x_c · M_c, r, −128, 127)`. Here `x` is what a `pot` site would rescale: the accumulator, or at `RopeQ` and `RopeK` the rotated accumulator.
- **A residual site** computes `res_c ← clamp(res_c + ⌊(acc_c · M_c + 2^{r−1}) / 2^r⌋, −2^{19}, 2^{19} − 1)`, one rounding, as the residual rows of [RFC-0022](RFC-0022-saturating-residual.md#saturating-residual) state for a shift.

Activation scales stay powers of two in both modes. Every exponent the claim list carries defines tables: `Silu`'s output, `Exp`'s scale, the next site's shifts. In `affine` mode the multiplier carries the weights' real per-channel scales `σ_c = max|w_c| / 127`. As RFC-0005 states, `M_c = round(2^r · f_c)`, with `f_c` the real factor from channel `c`'s accumulator to the site's output and `r` the largest shift at which every `M_c` fits 16 bits.

### Lowering

[RFC-0006](RFC-0006-compiler-ir.md#llama-lowering)'s lowering gains one input and one node per affine site:

- **Input.** The weight's multipliers, `m [rows]`, `Committed { Weight }` with the range `[0, 2^{16} − 1]`.
- **INT8 site.** The key is `Mul(x, m)`, rescaled by a `Requant` with the one shift `r`.
- **Residual site.** It is lowered as a site of one class, with `Mul(acc, m)` in place of `acc`. When `2^r` fits a constant (`r ≤ 62`), the fused key is `Add(Mul(acc, m), ConstMul(res, 2^r))`. Otherwise it takes the two-step update. Both equal the semantics above, since `res · 2^r` is a multiple of `2^r`.
- **`Mul`'s broadcast.** `Mul`'s second operand may have the shape of the first's last axis, broadcast over the leading axes. That is how `m` multiplies `[n, rows]`. Shapes alone cannot say which broadcast applies: for `a [n, n]` and `b [n]`, as a norm's root at `n = d_model` positions, both fit. `Mul` therefore states it, `Mul { broadcast: Trailing }` for one value per row and `Mul { broadcast: Leading }` for one per channel ([RFC-0006](RFC-0006-compiler-ir.md#operation-set) 'Operation set').

Range analysis needs no new rule: `Mul` takes its operands' interval product, so the key's bound is `max|x| · (2^{16} − 1)`. Gadget selection is unchanged ([RFC-0006](RFC-0006-compiler-ir.md#gadget-selection)): the key is 16 bits wider than at a `pot` site, and takes `RequantSplit` or `RequantWide` as its shift allows. At 405B:

| Site | `pot` key bound | Affine key bound |
|---|---|---|
| `V`, `Up` | `127 · 128 · 16,384 < 2^{28}` | `< 2^{44}`, 45 bits |
| `RopeQ`, `RopeK` | `< 2^{44}` ([RFC-0005](RFC-0005-integer-model.md#integer-semantics)) | `< 2^{60}`, 61 bits |
| `AttnResidual`, `servable` | `127 · 128 · 16,384 + 2^{19+r}` | `< 2^{44} + 2^{19+r}` |
| `MlpResidual`, `servable` | `127 · 128 · 53,248 + 2^{19+r}` | `< 2^{46} + 2^{19+r}` |
| `AttnResidual`, `MlpResidual`, `minimal` | about `2^{60}` and `2^{58}` | over `2^{62}` |

In `minimal`, the O and down projections' accumulators are not requantized ([RFC-0005](RFC-0005-integer-model.md#profiles)). With a multiplier their keys exceed 62 bits, so the selection refuses them (`ERR-COMPILE-RANGE`), and the quantizer refuses `affine` there first, as it refuses `minimal`'s up projection.

**Validation.** The lowering refuses, with `ERR-QUANT-ARCH`:

- a rescale site in both lists or in neither, or a list out of order;
- an affine site outside the table above;
- an affine shift outside `[1, 63]`.

The prover's check of the hidden part requires:

- a `ChannelParams` with one multiplier per output channel, each in `[1, 2^{16})`, for each weight before an affine site;
- an empty `class` beside the multipliers, and no multipliers anywhere else.

### Stages

The `Mul` is an `Elementwise` relation ([RFC-0010](RFC-0010-proof-schedule.md#relations)). Its operand `m̃` reads only the channel variables, as a scalar-by-row operand reads only the row variables. Its claim on `m` goes to the weight opening, like a class vector's. It adds one relation to the chain from its key:

- At `V`, `Up` and the `servable` residual sites, the chain becomes `Mul ← MatMulWeight`, two relations, within `servable`'s longest.
- At `RopeQ` and `RopeK` it becomes `Mul ← Rotate ← MatMulWeight`, three relations. A `servable` model with an affine Q or K therefore has four stages, as `minimal` has. `minimal`'s longest chains already have three relations, so it keeps four.

The stage count still does not depend on depth, length or batch (INV-SCH-4). The commitment report gives it.

### Verifier validation

[RFC-0007](RFC-0007-claim-list.md#verifier-validation) step 5 lowers each affine site as above, from the claim list alone:

- the multipliers enter as committed weights with the public range `[0, 2^{16} − 1]`;
- the key's range follows from the analysis;
- the verifier selects the gadget and regenerates its tables like any other ([RFC-0019](RFC-0019-verifier.md#check-order), step 3).

A claim list that breaks a validation rule above is `Reject(StatementMismatch)`.

### Soundness

The proof proves the multiplier's product, not its range.

- **Uniqueness (SP-3).** The key `x · M_c` is a function of committed values: the multipliers are committed with the weights, before any proof. Each requantization gadget's decomposition is injective on its key domain, whose width `b ≤ 62` keeps it below `P` (RFC-0009 Lemmas 1, 2 and 11 via Lemma 0). So for any value the key takes, at most one witness passes.
- **The multipliers' range.** `[0, 2^{16})` is the bound the analysis uses to choose the gadget and size its tables, as the weights' `[−127, 127]` is. The protocol proves neither range. A committed value outside its range changes which function the committed model computes, not whether a proof fixes that function's output. An honest prover's multipliers lie in the range, so honest proofs complete.
- **INV-CL-1.** `r` depends on the weights only through the tensor's largest channel scale, a per-tensor statistic, as a `pot` site's class set does through its exponent window. INV-CL-1's test keeps per-tensor statistics fixed while it perturbs the weights.

### Encoding

`affine_sites` follows `shift_class_sets` in `PublicQuantSpec`'s TCE layout ([RFC-0003](RFC-0003-canonical-encoding.md)). No release has written a claim list or an `IntegerModel`, so the layout changes in place ([09 §3](../spec/09-release-and-versioning.md#3-protocol-and-schema-versions)). A `pot`-only model encodes eight more bytes, the empty list's `u64` length. The vectors that pin `IntegerModel`'s encoding, and the conformance models, are regenerated.

### Changes to other documents

- RFC-0005 'Scale policy': `affine` mode's activations are powers of two, its sites are those above, and its semantics are as above.
- RFC-0006 'Operation set': `Mul`'s shapes. 'Llama lowering': the multipliers' input, the affine key, the validation, and the refusal of affine sites until the lowering implements this RFC.
- RFC-0007: `PublicQuantSpec`, the table of public and hidden data, and step 5.
- RFC-0010 'Stages': `servable` with an affine RoPE site.
- 03 §2: `ChannelParams` of affine sites.

## Alternatives Considered

1. **One list of sites, each with its policy**: `rescales: Vec<(QuantPoint, Rescale)>` with `enum Rescale { Pot(Vec<i8>), Affine(i8) }`. It needs no rule that a site sits in one list. But it rewrites every entry's encoding and every reader of the class sets, for the same information. With two lists, `pot` sites keep their entries, and the one-list rule is a check the lowering already makes for missing and repeated points.
2. **A multiplier width per site.** It would let a narrower multiplier shorten the key. RFC-0005 fixes 16 bits, `ChannelParams` stores `u16`, and no gate has asked for less. Each field is one more value for the verifier to validate.
3. **Public multipliers.** The verifier could then fold them into public constants. But they are derived from per-channel weight magnitudes, which the claim list must not carry (RFC-0007, Alternative 2).
4. **Range-check the multipliers in the proof.** One `Range { 16 }` lookup per channel per proof would prove `M_c < 2^{16}`. Uniqueness does not need it, as for the weights.
5. **Fold the multiplier into the weight**, committing `M_c · w_c`. The weights would become 24-bit, breaking S1 and the INT8 weight pass of RFC-0012.
6. **Encode an affine site as a negative shift in `shift_class_sets`.** It saves a field, but gives one field two meanings, and every reader would have to know the trick.

## Drawbacks

- The verifier's lowering gains a case, and `Mul` a broadcast.
- An affine key is 16 bits wider, so a site that `pot` keeps in one table splits: about one commitment and one lookup more per element. The commitment report gives the exact cost (RFC-0005).
- An affine Q or K adds a stage to `servable`.

## Migration / Rollout

- **This RFC's change.** The claim-list field, the quantizer's output and the encodings change in v0.2.
- **Lowering affine sites.** The compiler lowers them as above since [#301](https://github.com/AbdelStark/tessera/issues/301). The PyTorch reference computes them, and `tessera_quant.quantize` takes `affine` scales.
- **The verifier.** Its lowering of affine sites lands with its claim-list validation ([#107](https://github.com/AbdelStark/tessera/issues/107)).

## Testing Strategy

- **Encoding.** The `IntegerModel` vectors hold two affine sites. `tessera-ir` writes them, and `tessera-quant`'s codec and the conformance models' oracle each encode them byte for byte.
- **The quantizer.** An affine weight's site appears in `affine_sites` with its shift and not in `shift_class_sets`. Its `ChannelParams` has multipliers and no classes. `minimal`'s O and down refuse `affine`.
- **Validation.** Each rule refuses a public part that breaks only it: a site in both lists, a site in neither, an affine `AttnNorm`, an affine shift of 0 or 64. The hidden checks are tested the same way.
- **With the lowering ([#301](https://github.com/AbdelStark/tessera/issues/301)):**
  - the lowered graph, on the compiler's exact interpreter, gives the semantics above at every affine site, ends of the range included;
  - selection takes `RequantSplit` or `RequantWide` at the bounds above;
  - an affine Q gives `servable` four stages;
  - the CPU engine reproduces a conformance model with affine sites.

## Open Questions

None.

## References

- [RFC-0005](RFC-0005-integer-model.md), [RFC-0006](RFC-0006-compiler-ir.md), [RFC-0007](RFC-0007-claim-list.md), [RFC-0009](RFC-0009-gadgets.md), [RFC-0010](RFC-0010-proof-schedule.md), [RFC-0012](RFC-0012-weight-commitment.md), [RFC-0019](RFC-0019-verifier.md), [RFC-0022](RFC-0022-saturating-residual.md).
- [#274](https://github.com/AbdelStark/tessera/issues/274), [#81](https://github.com/AbdelStark/tessera/issues/81).
