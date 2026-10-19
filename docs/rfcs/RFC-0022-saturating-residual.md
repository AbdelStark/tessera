# RFC-0022: A Saturating Residual Stream and Complete Norms

- Status: Accepted
- Authors: @AbdelStark
- Created: 2026-10-17
- Target milestone: v0.2
- Amended by: [RFC-0050](RFC-0050-gadget-advice-columns.md) ('Complete norms': minimum root width one)
- Supersedes, on acceptance:
  - [RFC-0005](RFC-0005-integer-model.md#integer-semantics) 'Integer semantics': both residual rows, and the range column of the norm row;
  - [RFC-0006](RFC-0006-compiler-ir.md#range-analysis) 'Range analysis': `RsqrtRow`'s output range and domain;
  - [RFC-0006](RFC-0006-compiler-ir.md#llama-lowering) 'Llama lowering': the residual's operations;
  - [RFC-0009](RFC-0009-gadgets.md#rsqrtrow) 'RsqrtRow': the root's range check.
- Origin: [#284](https://github.com/AbdelStark/tessera/issues/284), [#285](https://github.com/AbdelStark/tessera/issues/285)

## Summary

`D`'s residual stream saturates at INT20: each update clamps the sum, not the update alone. A residual site of one class computes the saturating sum with the `Requant` it already has, through the key `acc + 2^r · res`. A site of several classes adds one `Requant`. `RsqrtRow`'s range check widens from `2^{κ+1}` to the largest root any row can have, a public constant. Together they bound every norm's inputs and outputs over all admissible inputs. 8B and 405B then compile through every layer, and no honest witness misses a lookup.

## Motivation

Range analysis bounds every tensor over all admissible inputs, never calibration statistics ([RFC-0006](RFC-0006-compiler-ir.md#range-analysis)). Over all inputs, two of RFC-0005's bounds do not hold.

**The residual is not INT20.** RFC-0005 adds each update unsaturated, `res += rq(acc, r, −2^19, 2^19 − 1)`. After the embedding and `2L` updates, `|res| ≤ 127 · 2^{s_e} + 2L · 2^19`, a bound that updates of one sign at their clamp attain ([#284](https://github.com/AbdelStark/tessera/issues/284)). The norms see `u = Σ res_i^2 ≤ d · res^2`:

| Model | Layers | `max u` | `RsqrtRow`'s `u + e < 2^55` |
|---|---|---|---|
| Llama-3.1-8B | 32 | `2^{62.0}` | 7 bits over |
| Llama-3.1-405B | 126 | `2^{68.0}` | above `P`: not a true integer |

**The norm's range check misses honest roots.** `RsqrtRow` range-checks `ρ ∈ [0, 2^{κ+1})`. The honest root `ρ = ⌊√(N / u)⌋`, with `N = d · 2^{2κ}` and `u` the sum plus `e`, passes iff `d < 4u` ([#285](https://github.com/AbdelStark/tessera/issues/285)). A residual row may be zero, so every row passes only when `4e > d`. With `e = ⌈d · ε / σ_res^2⌉`, that needs `σ_res < 2√ε`, about `2^{−8}` for `ε = 10^{−5}`. RFC-0005's one bit of headroom then caps the calibrated residual at 1024. INV-DM-2 forbids an honest lookup that misses its table, so the compiler refuses a model beyond it.

## Goals

- Bound every norm's sum, root and key over all admissible inputs, so that range analysis compiles 8B and 405B through every layer.
- Keep `D` total: every input has a proof, so that an honest prover never fails and a dishonest one cannot claim that it did.
- Leave the cost per token unchanged where the quantizer allows, and leave `D`'s values unchanged wherever the residual stays inside INT20.

## Non-Goals

- Bounds that use calibration statistics.
- A change to the norm's arithmetic, `ρ = max{ρ : ρ^2 (s + e) ≤ d · 2^{2κ}}`, or to `e`.

## Proposed Design

### Saturating residual

Both residual rows of RFC-0005 'Integer semantics' become

```text
res ← clamp(res + ⌊(acc + 2^{r−1}) / 2^r⌋, −2^19, 2^19 − 1)
```

with `r = r_res,c`, the shift of the output channel's class. The update is no longer clamped on its own: the sum is. Wherever the residual stays inside INT20, `D`'s values are those of RFC-0005, since neither clamp binds there.

The compiler proves the sum one of two ways, chosen from public data so that the verifier derives the same graph. Both compute the formula above exactly:

- **One class: the fused key**, when `r ≤ 62` so that `2^r` is a constant of the IR. With `r` public, `2^r · res` is a multiple of `2^r`, so

  ```text
  rq(acc + 2^r · res, r, −2^19, 2^19 − 1) = clamp(res + ⌊(acc + 2^{r−1}) / 2^r⌋, −2^19, 2^19 − 1)
  ```

  The site's `Requant` takes the key `acc + 2^r · res`, a `ConstMul` and an `Add`, and outputs the new residual. That is one committed value and one lookup per element, as RFC-0005's update had. Its key is one bit wider than `acc`'s at most when `2^{r+19}` is below `acc`'s public bound.
- **Several classes, or `r = 63`: two steps.** A hidden class's shift cannot scale `res` publicly.
  - `u = rq(acc, r_c, −(2^20 − 1), 2^20 − 1)` takes the channel classes, as RFC-0005's update did.
  - Then `res ← rq(2(res + u), 1, −2^19, 2^19 − 1)`: `rq(2t, 1, lo, hi) = clamp(t, lo, hi)` for an integer `t`. Its key lies within `±(2^21 + 2^20 − 2)`, a single lookup of one class.

  A site takes one route in every layer ([RFC-0038](RFC-0038-uniform-layers.md)): the fused key when every layer's site has one shift at most 62, two steps otherwise.

  The inner clamp at `±(2^20 − 1)` changes nothing. It binds only when `|⌊(acc + 2^{r−1}) / 2^r⌋| ≥ 2^20`. Then, since `res ∈ [−2^19, 2^19 − 1]`, the sum reaches a bound of INT20 with or without it: `res + 2^20 − 1 ≥ 2^19 − 1` and `res − (2^20 − 1) ≤ −2^19`. This costs one more committed value and lookup per element per update than the fused key.

The residual after each update is a `Requant` output, so range analysis gives it INT20 by the lookup alone. Every norm then sees `u ≤ d · 2^38`: `2^50` at 8B, `2^52` at 405B. The embedding's `res = x0 · 2^{s_e}` is INT20 by the compiler's rule on `s_e` ([RFC-0006](RFC-0006-compiler-ir.md#llama-lowering)).

In `minimal` at 405B the fused key adds `2^{r+19}` to an accumulator near `2^60`. For a site of one class, `r` is the smallest of its channels' shifts, `e_res − e_in − e_c` with the largest weight exponent. With the accumulator's bound `127 · 2^53` for O, a shift above 41 would take the key past the split gadget's 62 bits, and range analysis would refuse that model with `ERR-COMPILE-RANGE`.

### Complete norms

`RsqrtRow` range-checks `ρ ∈ [0, 2^w)` with

```text
w = max(1, bitlen(⌊√(d · 2^{2κ} / e)⌋))
```

the bit length of the largest root any row can have, clamped to one by RFC-0050 when that root is zero: `u ≥ e` gives `ρ ≤ ⌊√(N / e)⌋ < 2^w`. Every honest root passes, whatever the sum, so the condition `4e > d` falls away. `w` is public: `d`, `κ` and `e` are in the claim list.

`w ≤ 32` holds for any `d < 2^32` at `κ = 16`, so the check stays one lookup. Range analysis refuses a larger `w` with `ERR-COMPILE-RANGE`. At 8B with `σ_res = 2^{−7}`, `e = 672` and `w = 18`, one bit above `κ + 1`. The norm's key `res · ρ` stays below `2^{19+w}`, `2^37` there, as RFC-0005 states. The `BigIntCompare` witnesses of `ρ^2 u ≤ N` and `N + 1 ≤ (ρ + 1)^2 u` reach `2^{2w+55}` rather than `2^89`, a few more limbs where `w` grows.

### Range analysis

- The residual needs no new rule: a `Requant` output takes `[lo, hi]`.
- `RsqrtRow`'s domain is `u ≥ 0` and `u + e < 2^55`, and its output `[0, 2^w − 1]`, with `w ≤ 32` required.

### Changes to other documents

- RFC-0005 'Integer semantics': the residual rows state the saturating sum, and the norm row's range is `s ≤ d · 2^38`, `ρ < 2^w`, key `< 2^{19+w}`.
- RFC-0006: the lowering's residual operations, and `RsqrtRow`'s range and domain as above.
- RFC-0009 'RsqrtRow': the range check `ρ ∈ [0, 2^w)` and the products' bound.

## Alternatives Considered

1. **Range-check the residual at each norm** (#284's option 1). A lookup on the virtual residual enforces INT20 before each norm, with no commitment, and keeps `D` unsaturated. Rejected: an input that drives the residual out of INT20 then has no proof. An honest prover fails on it, and a dishonest one can claim that any input did, since the verifier cannot see the residual. Saturation costs nothing more for a site of one class.
2. **The two-step route everywhere.** Simpler, but it costs one committed value and lookup per element per update at every site. That is `2 · d_model` per layer per token: about a tenth of a servable layer's commitments at 8B, and a quarter of a minimal layer's 137,216 at 405B. Kept for classed sites, where the fused key needs the hidden shift.
3. **A committed per-channel power `2^{r_c}`** for classed sites, so that every site uses the fused key. That needs a hidden vector bound to the classes and a channel-broadcast product in the IR. Rejected for now: the saving applies only to classed residual sites, and the quantizer decides how many there are.
4. **Requantize the residual before each norm** (#284's option 3). A committed value per element per norm, and the norm reads a coarser residual than `D`'s.
5. **Constrain `σ_res < 2√ε`** (#285's option 2). The norms' completeness would then decide the residual's scale, and outliers above 1024 would lose their headroom.
6. **Clamp the root** (#285's option 3). `ρ = min(⌊√(N / u)⌋, 2^{κ+1} − 1)` changes `D` for small rows and needs a gadget of two cases.

## Drawbacks

- `D`'s values change for inputs that saturate the residual, which RFC-0005's headroom keeps outside calibration's range.
- A classed residual site costs one committed value and one lookup per element per update more than RFC-0005's dataflow.
- The fused key is wider than the accumulator by up to one bit, which can move a site from `Requant1` to `RequantSplit` when `r + 19` approaches 31 in `servable`.
- The norm's key and its `BigIntCompare` witnesses grow with `w − κ − 1`, zero to a few bits.
- At RFC-0005's `κ = 16`, a saturated row's root is 0, so the norm after it outputs zeros. [#287](https://github.com/AbdelStark/tessera/issues/287) proposes a `κ` that keeps the root's precision.

## Migration / Rollout

No release has written a claim list or a model, so the change lands in place, as 09 §3 lets an unreleased schema change. The PyTorch integer reference (#89), the CPU engine (#82) and the exact-mode serving engine compute the saturating sum from the start: a clamp after each residual add. The quantizer's shifts and `e` do not change.

## Testing Strategy

- The lowering's differential test runs both routes against an independent reference of the saturating semantics. Its adversarial inputs drive the residual into saturation.
- A model whose residual sites have one class, and one whose sites have several, compute the same logits as the reference.
- Llama-3.1 8B's and 405B's shapes lower through every layer from their public part alone.
- Swept over `d` and `e`, `RsqrtRow`'s analyzed range covers the honest root of a zero row, and its top bit is that root's.

## Open Questions

- **OQ-0022-1.** Should the quantizer prefer residual sites of one class, which cost nothing extra, where fidelity allows? Owner: @AbdelStark. Resolve with the 8B fidelity gates (#93), which measure the cost of a per-tensor exponent for the O and down projections.

## References

- [RFC-0005](RFC-0005-integer-model.md), [RFC-0006](RFC-0006-compiler-ir.md), [RFC-0009](RFC-0009-gadgets.md), [03 §6](../spec/03-data-model.md#6-invariants) INV-DM-2.
