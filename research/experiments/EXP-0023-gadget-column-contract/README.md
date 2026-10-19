# EXP-0023: Canonical gadget columns and exact root-product boxes

| | |
|---|---|
| Status | `done: validated` |
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

The original registered run passed on Apple M4 Max / Python 3.14.6 (E0 exact arithmetic):
4,352 honest rows, 8,704 root alternatives, 13,680 auxiliary alternatives,
1,984 NonNeg descriptors, 1,024 Softmax descriptors and 1,024 public table-policy
cases. The actual harness made 562,294 assertions; the largest normalization
lookup-box side had 35 bits, below the registered 40-bit limit. The width-32
manifest has 187 committed values and 189 range requests, as predicted (all E0).

Raw original output is `results/original-outcomes.json`. The witness-matrix SHA-256
is `bd728c2db4134fb112103fb801f8ee572f723def60fcc63b0c0403fb85fb833b`.

There are 19 distinct seeded faults plus a separate `u+P` attack control. Of the
check/algorithm controls, direct root-range removal and four final-carry removals
change the first rejection reason; they are not claimed to accept a false witness.
Other controls demonstrate missing-check acceptance or rejection of an honest
increment boundary. Seven metadata mutations fail the actual descriptor gate.
No syntax error or observationally equivalent fault is counted.

After this independent baseline passed, source inspection found a fixed fifteen-
limb scalar reference and interleaved column order. This candidate uses public
width-derived limb counts and grouped arrays, so migration must be explicit.
The dated amendment in `results/README.amendment-preregistered.txt` validates retention
of the original `rho+1` constraint polynomial without changing the original plan.

The amendment passes the same 4,352 honest rows, 8,704 altered roots and 13,680
altered auxiliaries, plus four zero-maximum-root rows, four malformed public
parameter cases and 80 factorization points. Every valid-row column is identical;
the canonical matrix digest remains unchanged. It adds 790,537 assertions; the
combined run has 1,352,832 assertions (E0). The extra frozen-amendment integrity
check makes the repeated original stage 562,295 assertions rather than the
archived original 562,294. Raw final output is `results/outcomes.json`.

## Analysis

Finite arithmetic checks do not establish a full-model proof, a G-4 error bound,
Fiat-Shamir security or protocol/2 simulation. Production integration must account
for exact lookups/relations and preserve independent verifier implementation.
No H100, end-to-end inference or performance evidence is sought in this unit.

## Verdict

`validated` for the bounded original criterion and the pre-registered polynomial-retention amendment. Retain the original `rho+1` relation; canonical data agree only on the required Boolean/input-binding domain. No performance gain, preset fit, cryptographic security proof or full-model credit is inferred.

## Follow-ups

After a validated result, draft the superseding RFC for #407, compare with the
standalone gadgets only after independent checks pass, and implement both model
layouts in #106 from the accepted specification. A failed criterion is not relaxed
to make existing code pass.

## Amendment (2026-10-19)


Date: 2026-10-19. This amendment is frozen and committed before its code.
The original hypothesis, method and success criterion remain unchanged.

## Reason

After the independent original arithmetic/schema checks passed, source inspection
at 5f201fe found that the scalar Rsqrt reference uses fifteen limbs at every root
width, interleaved comparison columns, and the original `(rho+1)^2*u` polynomial.
The candidate's width-derived limb count and grouped roles therefore need an
explicit migration; do not silently inherit the current implementation's schema.

The original experiment uses `(rho+psi)^2*u`, which has the same Boolean-row value
on the required domain `psi in {0,1}`, `u*(1-psi)=0`. The polynomial difference is
`u*(1-psi)*(2*rho+psi+1)`, up to the order of subtraction. They are not identical
polynomials at a sumcheck's non-Boolean final point. To avoid changing the accepted
bracket relation unnecessarily, retain `(rho+1)^2*u` in the RFC if the following
canonical-data equivalence checks pass. No original failing result is reclassified.

## Hypothesis

With all required domain/range/padding checks present, the existing `rho+1`
polynomial produces exactly the same canonical witness columns and metadata as
the registered selector-increment candidate on every valid real/padded row.

## Success criterion

Validated requires exact equality of every column on all 4,352 original honest
rows; successful independent validation through each polynomial; the same 8,704
root alternatives and every representative auxiliary perturbation rejected with
the registered outcome under the original polynomial; unchanged public metadata,
counts, range-family order and full-product carry bounds. Any valid-row column or
outcome difference refutes this amendment. Missing checks are inconclusive.

The test must also demonstrate a point where the two polynomials differ outside
the required domain and verify the stated factorization. Do not claim equality
of arbitrary field evaluations. The RFC must retain the original final-evaluation
formula and require the upstream `u=psi*(sum(res^2)+e)` obligation.

Additional boundary: the scalar reference clamps a zero maximum root's range
width to one. Check `N=1,e=2,w=1` at real u=2,3,2^55-1 and a padded zero row under
both formulations. These four rows are additional to, not replacements for, the
registered matrix. Reject malformed widths 0/33 and invalid table policies; no
silent schema change is permitted.

## Method

Keep the original harness and controls. Add an explicit polynomial choice only
to the witness and small-equation evaluator, defaulting to the original registered
selector form. Generate and validate the constant-increment alternative separately.
Do not change schema construction, original inputs, seed or original success gates.
Preserve the original outcome artifact. Record the amendment's counts and outcomes
separately; this is E0 exact arithmetic, not a retimed or enlarged performance study.

## Reproduction

The same root command runs both units after verifying the original and amendment
frozen bytes against git. No amendment code exists at registration. Record its
SHA-256 and UTC timestamp in preregistration-amendment.txt.

## Arithmetic argument and migration decision

For a true input in the 55-bit box and a root in the public w-bit box, their
range-checked decompositions have boxes below Goldilocks; recomposition modulo
that field is therefore exact. Each product coefficient is a sum of root-byte,
root-byte and input-limb products. The complete lookup boxes on both sides of
all normalization equations fit 35 bits in the declared width sweep (E0), below
2^40 and the centered field range. Small-congruence induction makes normalization
exact as integers. Range-checked normalized digits and borrows, with both final
normalization carries and final borrow zero, compare the full products. The two
brackets then fix `floor(sqrt(N/u))` on a real row. The padded-root check and the
upstream input-selector obligation are still necessary. This is a local arithmetic
argument, not an adaptive protocol proof.

The candidate's normalized product width is public: `ceil((2*w+55)/8)`, never a
function of the actual root or input. The scalar reference at 5f201fe instead uses
fifteen limbs for all widths and interleaves digit/carry and difference/borrow
columns. The new RFC must specify the width-derived bound and grouped role order,
with an explicit per-role adapter/refactor before either model layout uses it.
The difference is not silently treated as an implementation conformance pass.

At Boolean positions, the input-selector equation gives `u*(1-psi)=0`.
Thus the difference between increment formulations factors as
`u*(1-psi)*(2*rho+psi+1)=0`. This does not identify their off-domain polynomials.
The final-evaluation formula remains the original `(rho+1)^2*u`; only its public
column schema is fixed. Each pointwise product still has degree at most three,
or four with the sumcheck equality factor. The existing RFC-0014 calculator must
consume the actual round, lookup, table and advice-row counts and reject any preset
that misses its target. This unit does not assert that a particular full model's
preset fits. Protocol/2 masking and secret-buffer obligations remain unchanged.

## Additional engineering boundary registration (2026-10-19)

The original sweep has N below the field. Before the RFC decision, the separately
frozen `results/README.constants-preregistered.txt` adds twenty scalar row cases
with constants above Goldilocks and five intentional whole-constant-reduction
attacks. This extension is registered, not run; it does not replace any original
case or criterion. The earlier bounded result remains scoped to its inputs.
