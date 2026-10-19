# EXP-0023 additional boundary registration: constants above the field

Date: 2026-10-19. Freeze and commit before implementing these additional cases.
No original hypothesis, input, success criterion or verdict is replaced.

## Reason

The original width sweep uses N below Goldilocks even though products can exceed
it. Model-sized N and N+1 must also be split as whole integers, not reduced as a
single field element. Add this engineering boundary check before RFC promotion;
it supplies no performance or full-model credit.

## Inputs

Five public primitive parameter profiles `(d,kappa,e)`:

1. `(4096,30,672)`, representative 8B norm metadata;
2. `(16384,30,2685)`, representative 405B norm metadata;
3. `(4,31,2)`, N exactly 2^64;
4. `(2^32-2,31,2^54)`, large primitive constant boundary;
5. `(4,31,2^55-3)`, a small root box with N above the field.

These are scalar parameter tests, not complete claim lists or an assertion that
the extreme dimensions are admitted by full-model range analysis. Derive
`N=d*2^(2*kappa)` and `w=max(1,bit_length(isqrt(N//e)))` independently. All roots
must fit the existing allowed width. For each profile use real u=e,e+1,2^55-1
and one padded zero row: twenty additional row cases.

## Success criterion

Both polynomial formulations must produce identical full canonical columns and
pass the small-equation validator. Reconstructed constant normalization digits
must equal psi*N and psi*(N+1) as whole integers, including their high bytes.
Independent metadata constructions and full carry-box bounds must still agree.

For each profile, deliberately generate a wrong-root witness for N modulo P,
while retaining the original public root width. The normal validator using the
true public N must reject at the first differing constant normalization equation.
An intentionally faulted validator reducing N modulo P must accept that witness;
its root must differ from the whole-integer reference root. This tests a missing
check only in the experimental fault model, not a present production vulnerability.

A failure refutes this boundary extension; unresolved harness failure is
inconclusive. Do not hide it behind the earlier validated result. Report its
cases and assertions separately. Preserve the original outputs and frozen plans.

## Method and reproduction

Extend the same E0 stdlib harness after verifying these committed frozen bytes.
Do not alter original cases, schema, seed or field. Add only the named profiles
and a test-only whole-constant reduction fault. Retain the original final-
evaluation polynomial. The same root command runs all units and saves separate
outcome fields. No code for this extension exists at registration.
