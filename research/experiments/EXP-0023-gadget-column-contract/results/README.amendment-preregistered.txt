# EXP-0023 dated amendment: retain the original bracket polynomial

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
