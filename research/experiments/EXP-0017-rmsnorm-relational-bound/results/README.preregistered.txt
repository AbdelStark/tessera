# EXP-0017: RMSNorm relational bound under current precision and counts

| | |
|---|---|
| Status | `planned` |
| Idea | [IDEA-0011](../../ideas/IDEA-0011-rsqrt-fact-norm-key.md) |
| Evidence level | E0 exact integer bounds and compiler-derived counts |
| Hardware | none for the claim; CPU runs exact arithmetic and the compiler probe, without timing |
| Pre-registered on | 2026-10-19 |

## Hypothesis

The verifier-derivable joint bound on an RMSNorm key `res_i * rho` removes one committed split limb and one lookup per element at both layer norms of the current 405B servable fixture, saving exactly 32,768 commitments and 32,768 lookups per layer per token under RFC-0023's current precision policy.

## Success criterion

The hypothesis is the idea's count-saving claim, rather than merely the validity of its algebraic bound. Set `B_max=32` and use the current compiler's signed-width convention, shift classes, balanced split, table sharing and counts. Compare baseline and candidate on the same public fixture.

- `validated`: all mandatory bound/oracle/adversarial checks pass, and both 405B layer norm sites switch from `RequantSplit` to `Requant1`, saving exactly 32,768 commitments and 32,768 lookups per layer per token. The final norm and first layer's embedding-derived input must also be counted explicitly; they cannot be silently treated as an ordinary saturated residual. The candidate's narrowed tables must reproduce every tested integer norm output exactly.
- `refuted`: an accepted integer witness violates the proposed joint bound; or all mandatory checks complete and the exact joint signed widths at both ordinary 405B layer norms remain greater than 32 bits, their selected gadgets remain `RequantSplit` with two commitments and two lookups per element, and the per-layer commitment and lookup reductions are both exactly zero. This fires the current-count kill criterion even if table rows decrease.
- `inconclusive`: all other outcomes, including missing baseline reproduction, incomplete required checks, a partial count reduction, or failure to connect a reported count to the actual current selection rules. A defect in the harness is fixed before judging; a disagreement between the specification and production is reported separately and blocks validation until resolved.

The bound-validity result, exact key intervals, table-row change, and count verdict are recorded separately. No table-row benefit, historical kappa=16 control, conditional cost-model saving or favourable toy case can replace the primary current-count test. Finite checks support the stated arithmetic argument; they do not establish a new full-protocol soundness theorem.

## Method

Preparation is restricted to this preregistration. No harness code, execution or production changes are authorized in this unit. A later, separately authorized implementation must first verify this frozen text in a parent-created commit, as research rule 3 requires. Scope after that authorization is this experiment directory and the IDEA-0011 result update; shared bookkeeping belongs to the parent.

The normative source revision is main `709c37c622caf0351df5ccb9acebcaa6cc09ab6f`. Record the executing checkout and hashes of all source inputs. If relevant norm/range/selection code differs from this revision, describe the difference and amend the registration before running. Reuse the existing public compiler APIs and report fixture wherever possible, with no verifier dependency on prover code. The compiler probe is a separate safe Rust workspace; an independent Python integer oracle reads its public JSON output.

1. **Current baseline.** Read RFC-0005 integer semantics, RFC-0006 range analysis/selection, RFC-0009 RsqrtRow and Lemma 0, RFC-0022 residual saturation and root range, RFC-0023 precision, RFC-0028 balanced splitting and RFC-0035 padding. Reproduce the 405B servable c2/B32 report's `C=45,566,660`, `K=45,440,678`, `M=7,693,467,648` and per-layer `C=360,232`, `K=360,380` from the same public fixture used by `crates/tessera-bench/tests/commitment_report.rs`. Preserve its approximate RsqrtRow cost and omitted range-table multiplicities; they cancel between arms, rather than becoming exact gadget costs. Record every norm node, its input/root/key ranges, dimension, epsilon, kappa, shift classes, selected gadget and table definitions. Verify these against the committed report, not historical EXP-0011's hand table.
2. **Precision and intervals.** Independently derive `e=ceil(d * 2^14 / 100000)` and `kappa=min(31,max{k : floor(d * 2^(2k) / e)<2^64})` for the pinned residual exponent -7. Required current fixtures are 8B (`d=4096`, 32 layers, `e=672`) and 405B (`d=16384`, 126 layers, `e=2685`); both use kappa=30, root-check width 32 and one norm shift class `r=kappa-3=27`. Do not propagate the two weight-site classes into the norms. Record the actual first embedding-derived norm input separately from all later INT20 residuals, including the final norm. Controls use kappa=16 and adjusted norm shift13 on the same dimensions/epsilon, and cap case kappa=31 at public `e=d`. These controls cannot change the primary verdict.
3. **Exact integer bound.** On a real row, `u=sum(res_j^2)+e`, `e>=1`, `N=d*2^(2*kappa)` and the exact root bracket gives `rho^2*u<=N<(rho+1)^2*u`. For rho>=1, `(rho*res_i)^2<=N-rho^2*e<=N-e`; for rho=0 the key is zero. Thus `abs(key)<=isqrt(N-e)` when `N>=e`, otherwise all roots and keys are zero. Also report the weaker strict bound `abs(key)<2^kappa*sqrt(d)` using integers only. Derive the minimal signed width from both asymmetric endpoints, rather than treating an exclusive magnitude bound as a signed width. This already predicts a current 405B width near38 rather than the historical24/25; the harness must compute the exact value.
4. **Attainability and tighter oracle.** For every magnitude `x=0..2^19` of a saturated INT20 coordinate, compute `x*isqrt(N//(x*x+e))`. Other channels can only decrease rho, so their zero assignment maximizes the key at each x. Record the exact extremum and its full one-nonzero-coordinate witness, independently checking both root inequalities. Use `x<=2^19-1` for the positive endpoint and `x<=2^19` for the negative endpoint. Run this complete magnitude enumeration for both dimensions at kappa16 and30 and the kappa31 cap controls. For the narrower first norm use its actual compiler interval and embedding representability; do not call an arbitrary INT20 witness attainable through the embedding. Compare Python `isqrt` against a separately written integer binary-search root routine on every extremizing witness and all boundary controls.
5. **Finite accepted-witness exhaustion.** Enumerate every row of dimensions1,2,3,4 over signed3-bit residuals `[-4,3]`, epsilon1,2,3,4 and kappa0,1,2,3,4. Enumerate every candidate rho from0 through `isqrt(N//e)+1` against the direct pair of integer bracket inequalities. There must be exactly one accepted root, equal to the independent integer-square-root oracle; every accepted key must satisfy both the universal and magnitude-enumerated bound. Include zero rows, maximally negative/positive rows, one nonzero channel, exact-square brackets and immediate neighbours. Test padded `psi=0,u=0,rho=0` and reject a nonzero padded rho by its explicit zero pin. Negative controls must show that dropping epsilon positivity invalidates the zero-row root-uniqueness premise, dropping either root inequality admits an incorrect root, and dropping the padding pin admits a second padded root. These controls are separate from the intact candidate.
6. **No-wrap and semantics.** Establish from the current declared residual/root ranges that each original key is a true centered Goldilocks integer before applying a joint bound. Check `u<2^55`, kappa<=31, root width<=32 and the exact big-integer comparisons; do not infer a small integer from a field congruence. Check zero/nonpositive epsilon rejection and root-width33 rejection. The range refinement is derived from the existing constraints; it introduces no witness, lookup, challenge, opening or transcript change. Report any extra precondition needed by the derivation. A standalone congruence or small-field test cannot substitute for the exact bracket.
7. **Selection and table rows.** Feed baseline and refined public key intervals with their original shifts/classes into production `select`, using small synthetic graphs if the public `Ranges` API prevents replacing an existing range. Cross-check the rule with an independently written integer implementation: Requant1 iff `b+ceil(log2(classes))<=32`; otherwise use all current split preconditions and the balanced split. Reconstruct the full candidate Selection with only norm entries replaced, deduplicate tables by exact definition, and pass it to the production report, or independently reproduce the complete report with a documented node/table mapping. Baseline reproduction is mandatory. Count unique tables over the whole model: shrinking a norm table does not remove an old Range table still used elsewhere. Include the first and final norms and unchanged affine/weight/row gadgets. Record C,K,M deltas, per-layer and per-token totals, and conditional cost-model projections only after these exact counts are known. Projections are E0 and must retain current model omissions; no GPU or wall-clock improvement is claimed.
8. **Additional synthetic controls.** SplitMix64 seed `0x7E55_E7A0_0011_0017` supplies1,000 rows each at dimensions4,16,128 with signed8-bit entries, and100 sparse/boundary-pattern rows each at dimensions4096 and16384 within INT20. Test all endpoints, zeros and alternating signs. Candidate Requant1/RequantSplit output is compared with direct integer rounded-and-saturated requantization at each tested key; include ties, saturation boundaries and adjacent keys. All saved data is synthetic and public. Serialize the parameters, extrema, coverage counts, all failed checks and representative accepted/rejected witnesses to raw JSON.

This is E0. Warmups, timing batches and CV thresholds are inapplicable; no CPU benchmark is run and no timing slot is consumed. Record OS, Python/Rust versions, UTC time, executing SHA and input/harness SHA-256 values. A single deterministic command must reconstruct the verdict from raw count/bound records. The independent checker must fail when an extremum, width, split multiplicity, unique-table row total or verdict is deliberately altered in a scratch result, and pass a no-op control.

## Reproduction

After preregistration commit and separate implementation authorization, from the repository root:

```sh
cargo run --release --locked --manifest-path research/experiments/EXP-0017-rmsnorm-relational-bound/Cargo.toml -- > research/experiments/EXP-0017-rmsnorm-relational-bound/results/compiler.json
python3 -I research/experiments/EXP-0017-rmsnorm-relational-bound/check.py
```

The first command will emit public baseline/candidate selection records; the second will independently check exact bounds, counts and frozen verdict conditions. Those files do not exist at preregistration. `results/README.preregistered.txt` freezes this report, and `results/preregistration.txt` records its SHA-256 and UTC time.

## Results

Pending; no harness has been written or executed.

## Analysis

The old37-to25-bit narrative predates RFC-0023 and is a historical control, not the current baseline. The modern relational bound may reduce table rows while preserving split multiplicity. Exact logical rows are distinct from materialized GPU bytes or measured lookup cost. The toy exhaustion does not prove all protocol security, comparator implementation correctness or verifier integration; the algebraic implication and its current preconditions must be reviewed separately. A full compiled-model count is still partly estimated for row gadgets, as the original report declares.

## Verdict

Pending the fixed checks. IDEA-0011 remains triaged until separately authorized execution and judgment; no idea score changes in this preregistration-only unit.

## Follow-ups

If the count claim is refuted but table rows shrink, record both results and require a new, scoped table-memory/cost experiment before promotion. Any production range-analysis or verifier change needs a superseding specification review and independent verifier implementation. H100 measurements could price a proven table-row reduction; they cannot establish the integer bound or change the exact number of commitments/lookups. No broad parameter sweep or unattended schedule is installed.
