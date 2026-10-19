# EXP-0023: Canonical gadget columns and exact root-product boxes

| | |
|---|---|
| Status | `planned` |
| Idea | [IDEA-0036](../../ideas/IDEA-0036-canonical-gadget-columns.md) |
| Evidence level | E0, exact arithmetic and finite synthetic checks; no timing claim |
| Hardware | None required for analytic claims; CPU environment recorded with results |
| Pre-registered on | 2026-10-19 |

## Hypothesis

The candidate below deterministically describes every auxiliary column and range
request of the current ClaimList lowering's gadgets from public inputs, while
preserving the exact Rsqrt bracket, full integer products and zero padded rows.

## Success criterion

Freeze and commit this report before experiment code. Do not edit this hypothesis,
criterion or method after that commit; amendments are dated and frozen separately.

`validated` requires all declared cases to pass every exact check below, independent
manifest constructions to agree, and every behavior-changing registered fault to
be caught by a targeted assertion. Report actual case, assertion and fault counts.
Compilation/syntax errors and no-op faults are not caught behavior-changing faults.
Any arithmetic, uniqueness, integer-box, metadata or padding counterexample is
`refuted`. Missing evidence or a harness error not resolved within the registered
method is `inconclusive`. No speed or full-model proof conclusion is permitted.

Required checks:

1. Root widths 1 through 32. For each width use 128 seeded real `u` values in
   `[1,2^55)`, boundary `u` values 1,2,3,255,256,257,`2^55-1`, and one padded row
   `(psi,u,rho)=(0,0,0)`: 4,352 honest rows in total (registered E0 case count).
   Use `N=d*2^(2*kappa)`, `e=1`, with `d=1,kappa=w-1` through width 31 and
   `d=4,kappa=30` at width 32. Independently derive each width from `isqrt(N/e)`.
2. Generate honest roots by whole-integer `isqrt(N//u)`. Validate through separate
   limb-product sums and field-sized equations. Compare both reconstructed products
   to whole-integer `rho*rho*u` and `(rho+psi)*(rho+psi)*u`.
3. Bound each raw coefficient and the entire lookup boxes on both sides of each
   normalizer equation below `2^40` and inside the centered Goldilocks range.
   Compare recurrence-derived carry bounds to a separate weighted-prefix formula.
4. Exercise the full allowed root box, including `rho=2^w-1`, independently of
   whether that root passes the overall bracket. Its increment must retain the
   high carry; products at width 32 must retain all 119 possible bits. No clipped
   limb array or modular whole-product comparison may replace the full product.
5. Both final normalization carries and final borrow of both comparisons are zero
   for every passing witness. Padded rows have zero root and zero auxiliaries;
   `(1-psi)*rho=0` remains required, even though both padded comparisons are zero.
6. Every root-plus/minus-one alternative is rejected with the independently
   predicted range, padding or bracket outcome. Perturb every committed auxiliary
   in representative real and padded rows at every width; a different passing
   auxiliary is a refutation. Cover `u+P`, top-bit, carry, borrow, final-limb and
   padding faults with exact rejection assertions or isolated obligation checks.
7. Build ordered roles and range-family requests by two independent traversals.
   Compare their entire descriptors and first-use-interned families, not only a
   digest. Root and node outputs occur once, outside the auxiliary-role list.
   Different synthetic witness values cannot change either descriptor.
8. Cover Requant1/Lookup/OneHot (no extra columns), RequantSplit (one low part),
   RequantWide (optional low part, middle and saturation value), and Softmax
   (maximum then both NonNeg low-limb arrays). Sweep legal NonNeg widths and public
   limb widths, including a virtual top with no committed low limb.

## Method

### Candidate schema

This is a pre-registered candidate, not an accepted protocol definition.
Graph outputs precede auxiliary roles and are never repeated among them.

For Rsqrt let `r=ceil(w/8)-1`, `m=ceil((2*w+55)/8)`. The auxiliary list is:

1. `u_low[0..3]`, 16 bits each; top `(u-sum(u_low[j]*2^(16*j)))/2^48`
   is virtual with 7 bits.
2. `root_low[0..r]`, 8 bits each; the virtual top has `w-8*r` bits.
3. Lower comparison, then upper comparison. Within each: left normalized digits,
   left carries, right normalized digits, right carries, difference digits,
   borrows. Each array has `m` entries, low digit first. Final carries and borrows
   remain committed and are constrained to zero.

The complete standalone root has `4+r+12*m` committed values including its graph
output, and `6+r+12*m` range requests including virtual top digits (E0 predictions:
187 and 189 at width 32). Auxiliaries alone exclude the graph output root.

Use byte-radix products: root digit indices contribute one byte each; input digit
indices contribute two bytes each. For the upper product only, the low root factor
is `root_lowest+psi`, which may equal 256 and must not be truncated. Lower compares
the full product to `psi*N`; upper compares `psi*(N+1)` to the full incremented
product. Constant sides are byte vectors of length `m`, normalized with the same
committed digit/carry schema; do not eliminate constant columns.

For raw coefficient upper bounds `A_j`, derive `K[-1]=0`,
`K[j]=floor((A_j+K[j-1])/256)`. The carry lookup width is
`max(1,bit_length(K[j]))`, with whole box `C[j]=2^width-1`. Check both
`A_j+C[j-1]` and `255+256*C[j]` against the small-integer limit, taking `C[-1]=0`.
Use actual narrowed top-root and top-input boxes. Compare against the separate
weighted-prefix expression for `K[j]`.

Range requests are root width first; input low limbs then virtual top; root low
limbs then virtual top; then each comparison's arrays in role order. Normalized
digits and differences request Range8, borrows Range1, carries their public widths.
Intern families in first-use order and retain repeated request ids. Default maximum
table width 32 must suffice; a smaller policy that misses a required family is
rejected rather than silently changing the witness representation.

### Independent reference and faults

Python arbitrary-precision whole integers are the semantic reference. Candidate
validation uses coefficient/carry equations, not the reference's computed root or
whole-product inequality as a substitute for them. A second metadata traversal and
weighted-prefix bound formula check descriptor and bound construction separately.
No production crate, prover source or existing witness helper is imported.

Seed: SplitMix64 `0x4071060023`. Inputs are synthetic and public. E0 checks have no
warm-up/timing protocol; elapsed time is not a result. Save environment and raw
JSON outcomes. Reject floats/non-finite artifacts and record the exact reproduction.

Fault controls must cover input/root top widths, direct root range, padded root,
each side's final normalization carry in both comparisons, each final borrow,
increment carry loss, cropped high product limbs, changed/dropped/reordered roles,
graph-output duplication and missing/reordered range-family requests. Use an honest
control and a restored control; do not count observationally equivalent faults.

The current lowering has no scoring/anchor graph nodes (RFC-0026). Their future
columns need their own specified schema; this experiment does not claim to cover
an unimplemented sampler, anchor circuit or full proof driver.

## Reproduction

No experiment code exists at pre-registration. The intended root command is
`python3 -I research/experiments/EXP-0023-gadget-column-contract/main.py`.
Record Python/OS/CPU versions, UTC time and the frozen-plan SHA-256 with results.
The report's frozen bytes must be present identically in git before that command
can be implemented or run.

## Results

Not run. The formulae and counts above are registered E0 predictions.

## Analysis

Finite arithmetic checks do not establish a full-model proof, a G-4 error bound,
Fiat-Shamir security or protocol/2 simulation. Production integration must account
for exact lookups/relations and preserve independent verifier implementation.
No H100, end-to-end inference or performance evidence is sought in this unit.

## Verdict

Pending; status remains `planned`. No adoption or model credit.

## Follow-ups

After a validated result, draft the superseding RFC for #407, compare with the
standalone gadgets only after independent checks pass, and implement both model
layouts in #106 from the accepted specification. A failed criterion is not relaxed
to make existing code pass.
