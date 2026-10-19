# IDEA-0036: A canonical public gadget-column contract

| | |
|---|---|
| Status | `triaged` |
| Origin | [SUR-0014](../surveys/SUR-0014-gadget-column-contract.md), #407 / remaining #106 |
| Lever | Exact `C_lin`, `K_lin`, size and verify; correctness prerequisite |
| Spec surfaces | RFC-0009, RFC-0006/0007, RFC-0041 and RFC-0011; tier-A derived offsets/table descriptors require a superseding RFC |
| Priority | 0 performance credit (E0); a blocking format contract takes precedence over performance scoring |

## Mechanism

Fix ordered public roles for the current ClaimList lowering's gadget auxiliaries,
distinguishing graph outputs already allocated by RFC-0041. Preserve the existing
integer relations; make their representation explicit. No witness value decides a
shape, role, limb count or table family.

For Rsqrt, retain three committed low 16-bit input limbs and a virtual 7-bit top
limb (`NonNeg(u,55)`). Decompose the root into 8-bit digits with a virtual narrowed
top digit. Form product sums in byte radix, with the 16-bit input limbs at even
byte positions. Each term has two root-byte factors and one input-limb factor;
their public component boxes bound every sum. For the upper product, replace the
low root-byte factor by `r_0 + psi`. This represents `rho + 1` on a real row and 0
on a padded row, retaining the carry at the maximal root box. It is an auxiliary
representation of the existing bracket, not a new definition of the integer model.

Use two fully specified byte-radix BigIntCompare witnesses. Each has left digits,
left carries, right digits, right carries, difference digits, then borrows, all in
increasing limb order. Keep final normalization carries and final borrows as
committed values constrained to zero. Public constant sides follow the same
normalizer schema; do not optimize their columns away in this proposal.

This is a candidate, not an accepted protocol layout. EXP-0023 must check its
bounds, uniqueness obligations, padding behavior and metadata before promotion.

## Why it could work

[TesseraGadgets26] supplies exact limb, carry, borrow and integer-box conditions;
[TesseraTiles26] supplies graph-output-first allocation; [TesseraAdvice26] requires
the result to depend on public data. The external interface evidence in SUR-0014
supports an explicit schema, not a borrowed arithmetic implementation (E0, primary).

## Expected gain

No timing, proof-size saving or model gain is credited. The unmodified cost-model
baseline remains the reference:

```sh
python3 -I research/tools/cost_model.py --help
```

| Context | Credited delta mu | Credited proof-size gain | Credited verification gain |
|---|---|---|---|
| 1K | 0 (E0) | 0 (E0) | 0 (E0) |
| 4K | 0 (E0) | 0 (E0) | 0 (E0) |
| 16K | 0 (E0) | 0 (E0) | 0 (E0) |
| 128K | 0 (E0) | 0 (E0) | 0 (E0) |

These are credit limits, not measured costs. The experiment must report exact
column/lookup counts and any current-spec disagreement; future implementation must
feed actual counts into the soundness calculator and compiler reports.

## Security gate

| Property | Effect | Pass |
|---|---|---|
| Soundness bits (G-4) | No new proposed challenge; exact lookup/column counts still need protocol accounting | unknown until arithmetic/schema checks; production accounting remains an implementation gate |
| Uniqueness (SP-3) | Preserve sub-P input/root boxes, full products, both final carries and borrows | unknown until EXP-0023; reject if any alternative passes |
| Zero knowledge (G-5) | Only public parameters choose layout; no secret-dependent omission | pass for metadata proposal; existing masking obligations remain |
| Verifier independence (SP-6) | Specify first, independent integer oracle, then separate implementations | pass as a required workflow; no shared production module |
| Post-quantum | No primitive or commitment change | pass, unchanged assumption |

## Prior rejection check

RFC-0009's table/Newton/root-index alternatives and RFC-0011's per-tensor
commitments remain rejected. This keeps the bracket and common commitment.
The baseline's already-rejected mechanisms do not fix the auxiliary-column
ambiguity. This proposal is not a range certificate, altered quantizer or new PCS.

## Kill criterion

Reject if any required limb equation needs a field-wrapped integer, an input or
carry box reaches its forbidden bound, a real/padded witness has another passing
representation, a product loses the root-plus-one carry, or schema output depends
on synthetic witness values rather than public parameters.

## Cheapest experiment

[EXP-0023](../experiments/EXP-0023-gadget-column-contract/README.md): E0 Python
arbitrary-precision arithmetic, public metadata and synthetic rows, with a frozen
pre-registration committed before code. No GPU is needed for the format decision;
it would be needed for any future performance claim.

## Refutations

| Lens | Verdict | Argument |
|---|---|---|
| soundness | survives conditionally | Small-integer equations can preserve the original bracket; incomplete carry, top-bit or padding checks kill the proposal |
| prior art | survives | Explicit schemas are established interface practice; no new cryptographic construction is claimed |
| cost model | weakened | Zero performance credit; approximate historical Rsqrt counts cannot substitute for an exact manifest |
| feasibility | survives | CPU arithmetic can falsify the schema; a tier-A RFC and two independent implementations are still required |

The bounded survey exhausted its single canonical-schema seed. Commitment,
lookup, counts and verifier lenses converge on that seed; attention only needs
its existing Softmax roles. Passes, parameter changes, ZK changes, hardware and
other paradigms introduce no necessary mechanism for this format task and are
out of scope. No extra optimization ideas are created to fill a quota.

## Score

| Gain | Confidence | Cost | Priority |
|---|---|---|---|
| 0 credited gain (E0) | 0.6, sound existing lemmas but untested concrete schema | l / weight 4, tier-A RFC | 0; required public-contract gate overrides performance ordering |

## Decision and links

Triaged for the bounded E0 check. Experiment: EXP-0023. Issue: #407. No accepted
RFC or production change yet. Ledger updated 2026-10-19.
