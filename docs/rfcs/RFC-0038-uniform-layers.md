# RFC-0038: One Template for Every Layer

- Status: Accepted
- Authors: @AbdelStark
- Created: 2026-10-18
- Target milestone: v0.3
- Supersedes, on acceptance:
  - [RFC-0037](RFC-0037-stage-schedule.md#layers-and-kinds) 'Layers and kinds': the lowering emits one template for a claim list that rescales every layer's sites alike;
  - [RFC-0007](RFC-0007-claim-list.md#verifier-validation) 'Verifier validation' step 2: the claim list rescales every layer's sites alike;
  - [RFC-0022](RFC-0022-saturating-residual.md#proposed-design) 'Proposed Design': a residual site takes one route in every layer.
- Origin: [#348](https://github.com/AbdelStark/tessera/issues/348)

## Summary

RFC-0037 stacks the layers of a lowered graph as one template, node for node. The claim list rescales each site of each layer on its own: a site is `pot` with one to four classes, or `affine` (RFC-0007, RFC-0025). The lowering follows those choices, so layers rescaled differently lower to different nodes, and their graph has no stage schedule. This RFC makes the claim list rescale every layer's sites alike, and gives the quantizer two exact transformations that meet the rule without changing `D`.

## Motivation

The lowering's nodes depend on each site's quantization:

- **An affine site** multiplies its key by the multipliers of the weight before it: one `Mul` more, reading a committed vector (RFC-0025).
- **A `pot` site of several classes** reads the channel-class vector of the weight before it. A site of one class does not.
- **A residual site of several classes**, or of shift 63, takes RFC-0022's two steps: one `Requant` more than the fused key `acc + 2^r·res`.

The conformance model `tiny-2-affine` has four affine sites in layer 0 and none in layer 1. Its layers lower to 56 and 54 nodes, and `derive_schedule` refuses it as `ERR-INTERNAL` (#348).

Stacking groups of identical layers would accept such graphs, but each group has its own stacked kinds, weight claims and masks. When the quantization varies freely by layer, there can be as many groups as layers. Each weight type then has a claim per group, against RFC-0012's one folding pass.

## Goals

- Every claim list the compiler accepts lowers to layers of one template, so that RFC-0037 stacks them.
- The rule is checkable from public data, at claim-list validation.
- A quantizer can meet it without changing `D`.

## Non-Goals

- Choosing classes or modes per site: the quantizer's calibration (RFC-0005).
- Stacking layers of different templates.

## Proposed Design

### The rule

A claim list **rescales every layer's sites alike** when, for each `Site` of its profile:

1. the site is affine in every layer, or in none;
2. as a `pot` site, it has one class in every layer, or several classes in every layer.

The number of classes may differ between layers when it is more than one: the tables differ, and the nodes do not.

Claim-list validation step 2 checks the rule and returns `Reject(StatementMismatch)` on the first site that breaks it. The compiler's `validate` and `lower` refuse such a model with `ERR-QUANT-ARCH`, naming the rule.

### The residual's route

The lowering chooses a residual site's route once for all layers, as RFC-0022 describes it:

- **The fused key**, `acc + 2^r·res` (or `acc·M + 2^r·res` at an affine site, RFC-0025), when the site has one shift in every layer, as a `pot` site of one class or an affine site, and every layer's shift is at most 62;
- **two steps** otherwise.

A layer of one shift whose site takes two steps computes the same output as the fused key would. RFC-0022 shows that the inner clamp binds only where the sum saturates anyway.

### What the quantizer can do

Two transformations meet the rule and leave `D` unchanged:

- **More classes.** A `pot` site of classes `r, …, r + k − 1` takes `r, …, r + k' − 1` for any `k < k' ≤ 4` with `r + k' − 1 ≤ 63`. The class vector keeps every channel's class, so no channel takes the new ones. This lets a site of one class join layers of several. Only the table grows: `k'·2^b` rows instead of `k·2^b`.
- **Affine for `pot`.** Let a `pot` site have classes `r_min` to `r_max`, with `r_max − r_min ≤ 15`. It is exactly the affine site of shift `r_max` whose multiplier for a channel of class `c` is `2^{r_max − r_c}`, in `[1, 2^15]`. The reason is that `rq(x·2^k, r + k, lo, hi) = rq(x, r, lo, hi)` for every `k ≥ 0`: the numerator and the denominator of RFC-0005's rounding scale together. This lets a layer's `pot` site join layers where it is affine.

  The affine key is at most `r_max − r_min` bits wider. It must still pass range analysis: in `minimal`, the O and down projections refuse `affine` in every layer (RFC-0025).

The quantizer chooses per site, over all layers: affine if any layer needs it, otherwise one class if every layer can take one, otherwise several.

## Alternatives Considered

1. **Stacked groups of identical layers.** It needs no rule, but costs a weight claim per group and type, and a mask per group and kind. It was rejected for the reason in 'Motivation'.
2. **Padding in the lowering.** The lowering could give a site of one class a class vector of zeros, so that it reads like a classed site. But the vector is no part of the statement: the verifier would have to know it is zero, and the advice would commit to it. The rule keeps the statement and the circuit one-to-one.
3. **A committed per-channel power `2^{r_c}`** for classed residual sites, so that every residual site takes the fused key. RFC-0022 rejected it, and nothing here changes that.

## Drawbacks

- **Two steps at every layer** of a residual site whose quantization needs several classes in any layer. That is one more committed value and lookup per element (RFC-0022 'Alternatives' 2 counts the cost), so the quantizer should prefer one class at residual sites.
- **Wider keys** in layers that take `affine` for `pot`, and larger tables in layers that take more classes.

## Migration / Rollout

v0.3, for #348:

1. this RFC;
2. the check in `tessera-compiler` and the residual's route chosen per site;
3. the conformance models of `spec/vectors/models/` rescaled alike, `tiny-2-affine` among them, and `tessera-quant`'s quantizer once it chooses classes.

The independent verifier checks the rule with its claim-list validation (#107).

## Testing Strategy

- **The rule.** Each clause refuses a claim list that breaks it alone, with its error, in `tessera-compiler`.
- **The transformations.** Rescaled alike, each conformance model computes the logits it computed before: more classes and `affine` for `pot` change no value.
- **The schedule.** Every conformance model lowers to one template, and `tessera-prover`'s stage tests prove all four.

## Open Questions

None.

## References

- [RFC-0005](RFC-0005-integer-model.md), [RFC-0007](RFC-0007-claim-list.md), [RFC-0022](RFC-0022-saturating-residual.md), [RFC-0025](RFC-0025-affine-sites.md), [RFC-0037](RFC-0037-stage-schedule.md).
