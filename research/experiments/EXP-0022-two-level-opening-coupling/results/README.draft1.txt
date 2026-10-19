# EXP-0022: Two-level opening equation coupling under an ideal PCS

| | |
|---|---|
| Status | `planned` — preregistration draft for review only; not frozen or committed |
| Idea | [IDEA-0035](../../ideas/IDEA-0035-combined-row-second-commitment.md), triaged; experimental dependency [IDEA-0002](../../ideas/IDEA-0002-coefficient-form-rows.md) |
| Evidence level | E0: proposed exact finite algebra and obligation audit, primary repository sources |
| Hardware | None for the mathematical claim; future bounded Python host provenance only |
| Pre-registered on | Not yet: draft prepared 2026-10-19; no implementation or run exists |

## Hypothesis

Under the explicit coefficient-row and ideal-PCS assumptions below, the proposed two-level reduction couples every original evaluation and queried-column equation to the same combined-row tensor, including proximity, component lanes, padding and original encoding masks, and its finite transcript/terminal checker refuses each registered omission or reorder while classifying batching cancellation roots exactly.

## Success criterion

This is a **conditional, bounded algebra/obligation unit**, not a cryptographic extractor, full protocol, G-4 budget, joint-ZK, current-workload count, size or performance result. All conditions are conjunctive:

1. Direct original equations and independently constructed candidate weights agree exactly for every registered honest fixture. The batched sum equals the direct weighted sum of every original equation, including every distinct queried column of every group and proximity, with no missing or duplicate equation.
2. The candidate factored public-weight MLE equals a direct truth-table MLE at every registered point. Every sumcheck round has degree at most2, satisfies its round identity, matches the independent coefficient-polynomial oracle, and terminates at the correct U/W evaluation.
3. Both base-field component-lane claims reconstruct the same Fp2 U value and remain owed until a successful final ideal-PCS opening on the same C_U. The explicit nonrecursive second-level opening's grouping, evaluation, proximity and masked-column algebra agree with direct complete-table references. All dummy positions and skipped public-zero rows have their prescribed logical coordinates.
4. Every registered malformed shape, missing equation/claim, changed group/lane/basis/descriptor/root, mask omission, early challenge and missing final opening is refused for its named local reason by an actual checker invocation, with an unchanged passing control. Each named omission mutant is exposed by its supplied cheating fixture or fails an exact positive comparison; an insensitive fixture makes that arm incomplete.
5. Balanced cross-group attacks are retained. For each attacked equation vector, an independent polynomial oracle computes D(A)=sum_l A^l*d_l, where d_l is the original equation residual. At every registered batch challenge alpha, the checker classifies exactly D(alpha)=0 or nonzero. False individual equations at a cancellation root are recorded as **expected batching exceptions**, not called universally rejected or evidence of soundness. At a nonroot the complete coupling checker must reject. A zero polynomial with a nonzero residual vector is a failure of canonical equation indexing or the oracle.
6. Domain coverage, physical comparisons, any memoized logical coverage, metered algebraic operations, maximum tables/rounds, controls and all attempts are retained. No resource limit or domain is silently relaxed.

`validated` means all six conditions hold in the frozen finite domain and the conditional derivations are internally consistent. `refuted` means one independently checked mismatch, false identity, unclassified false acceptance outside an explicitly recorded batching exception, dropped terminal obligation accepted by the strict checker, or missed mandatory attack. A genuine counterexample takes precedence. `inconclusive` means a cap, incomplete arm, insensitive mutant, unresolved reference discrepancy or missing conditional derivation. Judge original coupling, terminal/level2 algebra, and transcript inventory separately before their conjunction.

IDEA-0035 stays triaged, priority0, zero credited gain and unknown full security gates after finite success. Its complete-budget size kill criterion is **not evaluated** by this unit.

## Method

### Assumptions, exclusions and normative context

The accepted baseline is RFC-0011's current **evaluation-basis** encoding, complete combined-row transmission and original column checks. This experiment instead assumes the proposed IDEA-0002 **coefficient-basis** format at both levels: a row a has f(X)=sum_i a_i*X^i and encoding f(X)+X^k*m(X). The row's multilinear message remains the coefficient vector. It is not the current row's evaluations on H and no existing root or D is reinterpreted. Natural semantic column j is x_j=7*omega_(2k)^j. The proposed basis/column ordering is a fixed experimental descriptor field; the current Z_H=X^k-1/Lagrange route is a negative control, not an interchangeable implementation.

The ideal PCS commits an immutable descriptor and complete base-field tensor and provides an exact, binding, descriptor/root/point-bound MLE opening check. A root is an opaque local identity referencing a stored synthetic object; no collision resistance, Merkle soundness, proximity theorem, hiding or extractor is established by storing that object. Reference code can read the complete tensor. The candidate terminal checker may consume a value only through the ideal opening interface, never because the prover sent it. The explicit ordinary level2 opening is additionally audited for algebra/completeness and obligation order, **not proved a cryptographic implementation of that ideal interface**. Its concrete query/proximity errors and replacement of the ideal assumption remain unresolved. The ideal original-column membership interface similarly checks exact committed symbols; it is not a hash/Merkle implementation.

Read RFC-0011 opening steps1–8, RFC-0013 Hiding openings and malicious-verifier/one-use rules, RFC-0014 transcript/soundness rows, RFC-0031 descriptor, RFC-0036 combined rows/queries/absorption, RFC-0037 claim pool and RFC-0047 allocated R versus committed M. IDEA-0035 and the independent seed review supply the scope. [Draft source pins](results/source-pins.draft.json) record exact bytes. No current group-budget R21 resolution, current G/T_g/R/M/nu, exact serialized payload, tau approval, production verifier acceptance, weight opening, masked sumcheck or joint simulator is inferred. The algebraic blinded fixture below models two-base-row coefficients and q ordering only; its synthetic constants are not fresh random secrets or ZK evidence.

### Exact arithmetic and independent paths

Use Python arbitrary-precision integers for Goldilocks p=2^64-2^32+1 and schoolbook Fp2=a+u*b, u^2=7. Cross-check base arithmetic and Fp2 zero/one, inverse and distributivity against a separately written pair/integer oracle. Use this exact Fraction hand fixture: k1=2,D=1,G=1, row A=(2,-1), mask m=3, p=(1/2), two identical row-point claims v0=v1=1/2 and gamma=2. Then c=6, U=(12,-6), M=18, evaluation target3; at the separate algebra-only point x=3, column26 and corrected target6*26-9*18=-6 equal12-6*3. This hand point is not asserted to be a coset column. Compare every scalar and the canonical rational-to-base embedding (all denominators here are invertible). These are finite controls, not an integer range-proof experiment.

Candidate public weights use factor formulas; the reference enumerates complete Boolean tables and forms direct weighted sums. Candidate sumcheck round polynomials use values at0,1,2 with interpolation and folded factors; the reference independently multiplies affine-factor coefficients and sums the resulting degree<=2 polynomials. Neither reference calls candidate interpolation or its factored W evaluator. Complete input tensors may be shared; acceptance decisions and recurrences may not be echoed.

### Original rows, groups and encoding masks

Let k1 in{1,2,4}, data-row count D in{1,3}, evaluation-group count G in{0,1,2}, and m=G+1; group g=G is the proximity vector and has **no evaluation claim**. Pad the group axis to M=next_power_of_two(m). Every evaluation group contains exactly two ordered claims, t=0,1, at a common column point p_g, with distinct named row points r_(g,t); v_(g,t) is direct MLE evaluation of the base-field data matrix. Set

```text
c_g(r) = sum_(t=0,1) gamma^(t+1)*eq(r_(g,t),r),
c_xi(r) = eq(xi,r),
U_g[i] = sum_r c_g(r)*A[r,i],
M_g(X) = sum_r c_g(r)*m_r(X),
Y_eval,g = sum_(t=0,1) gamma^(t+1)*v_(g,t).
```

Data rows occupy indices0..D-1 in a publicly padded row cube and other data rows are public zero; eq is evaluated on that padded cube and sums include its logical zero rows. P1 in{0,1} is the number of encoding-mask coefficients per physically present original row. P1=0 means the zero polynomial. The P1=1 fixture can have rate1 when k1=1: it is algebra only, never a security parameter set. M_g has exactly P1 coefficients and is transmitted/fixed; it is **not** put into C_U or omitted.

Two modes run. `plain` has no combined-row blinding. `blinded-algebra` appends, for **each** g including proximity, two separate base-field rows Q_g^(0),Q_g^(1), each with its own P1 mask, after the padded data block; a final power-of-two row-cube pad is public zero. Its group coefficient on its own two rows is1,u, on every other group's blinding rows0. The proximity data coefficients still use the data-row cube's xi and its own blinding pair enters1,u. Thus

```text
U_g = sum_data c_g*A + Q_g^(0) + u*Q_g^(1),
M_g = sum_data c_g*m + m_(Q_g^0) + u*m_(Q_g^1),
q_g = <Q_g^(0)+u*Q_g^(1), eq(p_g)>,
Y_eval,g = sum_t gamma^(t+1)*v_(g,t) + q_g.
```

Only evaluation groups have q_g; proximity has no fabricated q/evaluation check. All q_g and v are fixed before gamma. U and M must use exactly the same complete row coefficients as original column equations, including the blinding pairs. No original blinding row is reused in a second combined vector. This fixture tests the algebraic form and inventory; it proves no joint distribution or one-use simulator theorem.

For a queried natural column j, the ideal original-column service supplies every physically present row's

```text
C_j[r] = sum_i A[r,i]*x_j^i + x_j^k1*m_r(x_j).
Y_col,g,j = sum_r c_g(r)*C_j[r] - x_j^k1*M_g(x_j).
<U_g, (x_j^i)_i> = Y_col,g,j.
```

Here c_g includes the two blinding coefficients in blinded-algebra mode. Original column symbols, membership, ordering, group consistency and M_g are all retained; this experiment never substitutes C_U for original columns. Repeated queried indices are answered once, sorted, with exact shape. Skipped rows are public zero with preserved logical row numbers, not compacted into different coefficients.

### Canonical equation list and one batched reduction

The list is: all G evaluation equations in increasing g, then for each **distinct sorted original query j**, all m column equations in increasing g, including proximity. Each equation has a unique contiguous ordinal l beginning0, type, group, descriptor, target, point and vector length. No group order, missing proximity or duplicate equation can be chosen by the prover. E=G+m*tau_distinct. E>=1 because every query list is nonempty; G=0 still has proximity column equations.

Write its vectors w_l[i] as eq(p_g,i) for an evaluation equation or x_j^i for a column equation, with RHS Y_l. After the complete list, all Y_l, C_U, M and original response columns are fixed, draw alpha and set lambda_l=alpha^l:

```text
W[g,i] = sum_(l with group g) lambda_l*w_l[i],
Y = sum_l lambda_l*Y_l,
sum_(g<M,i<k1) U[g,i]*W[g,i] = Y.
```

U and W at g>=m are public zero. Candidate factored weights at (s_g,s_i) use the exact joint tensor

```text
W_MLE(s_g,s_i) = sum_l lambda_l*eq(s_g,g_l)*w_l_MLE(s_i),
(column x^i)_MLE(s_i) = product_b (1-s_i[b]+s_i[b]*x^(2^b)),
(eq(p,i))_MLE(s_i) = product_b ((1-p[b])*(1-s_i[b])+p[b]*s_i[b]).
```

There is no product-of-extensions substitution for overlapping variables. Sumcheck uses U_MLE(z)*W_MLE(z), degree<=2 in each padded group/coefficient bit. The final equality owes U_MLE(s)=v_U and recomputes W_MLE(s) publicly. Round count is exactly log2(M*k1); the zero-variable case still owes its terminal opening and checks the scalar initial/final identity. Challenges/degrees/table dimensions must have equal exact lengths; no zip truncation.

D(A)=sum_l A^l*(<U_g,w_l>-Y_l) is computed directly as a coefficient vector, independently of factored W and sumcheck. A fixed nonzero vector produces a nonzero polynomial because the ordinals are unique. The conditional algebra gives only the familiar polynomial-root exception after fixed messages, not an instantiated probability budget or ROM extractor. alpha0 and alpha1 controls intentionally expose degeneracies. No empirical root fraction on a small catalog is translated to security bits.

### Component lanes, padding and the nonrecursive second opening

Commit the base tensor V[g,a,i] with i least significant, then lane a in{0,1}, then padded group bits; U[g,i]=V[g,0,i]+u*V[g,1,i]. The unpadded payload has2*m*k1 base elements. Public group padding expands the logical cube to F=2*M*k1 base elements. For k2 in{2,4}, extend only with public zero high bits to F2=max(F,k2), then flatten into rows of width k2. R2=F2/k2 is an **allocated toy row count**; M2 counts rows intersecting real group coordinates, never rows guessed nonzero from a sampled value. A wholly public-zero row is skipped and retains its row number. nu2=log2(F2); all these are toy derivations, not current compiler counts. The descriptor binds both-level basis/order, G,m,M,k1,k2, mode, P1,P2, row maps, group identities and root identities.

At the terminal point s, issue both claims V_MLE(s_g,0,s_i)=v0 and V_MLE(s_g,1,s_i)=v1 at the **same** group/coefficient point, with any added high padding bits fixed0. Check v_U=v0+u*v1. Both v0 and v1 can be Fp2: a base-valued tensor has extension-valued off-cube evaluations. Do not take real/imaginary components of an arbitrary off-cube U value as these two MLE claims. Neither terminal claim is fulfilled by sending its value.

For each terminal pair, independently derive the ordinary level2 claim groups by their actual low log2(k2) flattened coordinates. The lane bit may lie in a column or row subpoint, so do not assume one evaluation group: at most2 here. Use gamma2=(2+u) and row coefficients sum_t gamma2^(t+1)*eq(row_point_t,row). Add a distinct proximity vector using xi2. Encode each physical base row of V as coefficient polynomial plus X^k2 times its P2 in{0,1} mask; send every evaluation combined row and the additional proximity row **in full**, and all their combined masks. No third PCS level or recursive verifier is introduced. Check each terminal group's evaluation and every queried column's coefficient-basis consistency, with exact ideal-column membership under the same C_U. Proximity has column checks and no invented evaluation. Original M_g and these level2 masks have different namespaces and neither substitutes for the other.

A level2 opening obligation is consumed only after its descriptor/root, exact groups, full rows/masks, query response, evaluation and column checks pass **and the ideal PCS verifies the requested exact MLE values**. The latter is the explicit binding assumption; the ordinary finite query tests alone do not discharge it. Test this interface separately with one altered value and an absent successful-opening receipt, each against the real ideal checker. For whole-record omission sensitivity, retain the unfinished record at the point before its level2 response: a mutant that accepts merely from the terminal scalar equation must accept the supplied false-value transcript while the unchanged checker refuses its outstanding opening. Do not assert that deleting a redundant direct-oracle comparison from an otherwise completely checked finite all-column fixture admits a forgery. Report ideal-interface and missing-opening controls distinctly, not as a newly proved concrete PCS attack. No hiding mode is claimed for level2, even when the original algebraic blinded mode is selected.

### Finite fixture domain

The exact Cartesian shape domain is k1 in{1,2,4}, D in{1,3}, G in{0,1,2}, mode in{plain,blinded-algebra}, P1 in{0,1}:72 shapes. For each shape run four deterministic synthetic base-field families, with distinct tensor/row/column namespaces: all0; all1 on present entries; alternating signs; and signed residues in{-2,-1,0,1,2} from SplitMix64 seed `0x0022003500000001` (modulo reduction is fixture generation, **not** challenge sampling). Public-zero padding stays0. Mask/blinding families use their own namespace offsets, including nonzero original masks where P1=1. Include exact arrays in representative controls and a deterministic input digest in every fixture record.

For each shape/family run four complete coordinate/query catalogs. Within a catalog enumerate each point's bits in least-significant-first order and use offsets for r_(g,t), p_g, xi, sumcheck s and xi2 to distinguish groups and claims:

1. All coordinate bits0, gamma0; original query draw[0]; level2 draw[0].
2. All coordinate bits1, gamma1; original draw[0,0]; level2 draw[0,0].
3. Coordinate bit j with namespace offset o is `(j+o) mod2`, gamma2; original draw[n1-1,0,n1-1]; level2 draw[n2-1,0,n2-1].
4. Coordinate bit j with offset o cycles through `{1/2,u,1+u,2-u}` at index `(j+o) mod4`, gamma=1+u; original draw[0,...,n1-1]; level2 draw[0,...,n2-1].

Namespace offsets are row point o=2*g+t, group column point o=g+1, xi o=0, sumcheck challenge bits o=1, xi2 o=2. n1=2*k1,n2=2*k2. Empty point tuples use the empty product1. These catalogs are deterministic algebraic controls, not uniform Fiat–Shamir samples. Distinct group IDs remain bound even where chosen points or values coincide.

This gives72*4*4=1,152 original fixtures. Cross each with k2 in{2,4} and P2 in{0,1}:4,608 two-level fixtures. Each runs alpha in{0,1,2,u}, giving18,432 honest batched reductions; the same original reference may be reused with explicit cache counts. Additional mandatory attacks below do not replace these cases. Source pins, all four generators/catalog definitions and exact shape counters must be committed before implementation. No random-only alternative, real model or production workload is substituted.

### Mandatory attack/control corpus

Each item has a stored full input, exact local refusal/classification and a no-op passing control. Apply one fault at a time. These are local research strings, not tv-core RejectReason claims.

- `descriptor/shape`: unknown basis/format, old evaluation-basis discriminator, missing/extra field, bool-as-int, unsupported k1/k2/G/D/m/mode/P1/P2, incorrect power-of-two padding, lane count1 or swapped lane order, wrong group count/order and mismatched original/C_U identity. Refuse before allocation/claims. At least one attack for every listed field/rule.
- `equation-inventory`: drop each evaluation equation and each queried-column equation in a fixed G2/full-query fixture, including each proximity equation; duplicate/reorder one, change its ordinal/type/group or fabricate a proximity evaluation. Strict inventory rejects. In omission mutants the stored altered RHS of the dropped equation can pass the remaining local algebra, exposing the omitted obligation. Use nonzero coefficients/targets so no zero fixture hides it.
- `balanced-cross-group`: in a G2 fixture alter two adjacent canonical equations in different groups by +delta,-delta, delta1 and delta u. The naive unweighted cross-group aggregation cancels. Retain that accepting mutant; the complete checker must reproduce the exact residual polynomial and all four alpha classifications, including the expected alpha1 root and nonroot rejection. Also swap whole U group rows while preserving their aggregate and show the nonconstant group weights/inventory detect it at a chosen nonroot. Include one attack moving errors between evaluation and proximity column equations, not only between two evaluations.
- `masks/columns`: omit or alter original M_g, one original mask coefficient, the blinding-row mask or the X^k1 factor; replace it by X^k1-1. Feed actual altered target equations through the checker. Change one original response symbol/root, omit a column, repeat an answer, unsort distinct indices or compact a skipped row. Named ideal-membership/shape/coupling checks reject; disabling the respective check exposes the supplied fixture. No original column is removed by C_U.
- `blinding-algebra`: omit either base component or assign the pair to the wrong group/proximity row; omit q_g, move it after gamma, reuse one pair in two groups, or give t0 coefficient1. The transcript/ownership inventory rejects the order/reuse cases and the direct algebra catches wrong coefficients. Retain the familiar late-q compensating false-claim algebra as an accepting **order-check-disabled** mutant, with q fixed before gamma in the unchanged control. Do not claim statistical hiding.
- `round/final-weight`: change the initial Y, one degree>2 round, a p(0)+p(1) identity, challenge vector length, W's group/coefficient point, monomial exponent, or use U_MLE(s)*W_MLE(s) at an unrelated terminal point. Invoke actual reduction checker with named rejection. The reference and candidate must still agree in unchanged controls, including zero-variable dimensions.
- `terminal-opening`: drop either lane claim, both claims, final opening or its must-consume flag; change one value, lane/group/coefficient point, zero-padding bit, descriptor or root. Strict obligation/ideal-opening checker rejects. A value-only terminal equality with no opening is retained as an accepting mutant. Correctly reconstruct both off-cube component claims; taking the real/imaginary parts of U_MLE(s) must fail this explicit nonbase fixture: M=k1=2, s_g=1/2,s_i=u, lane0 rows(1,2)/(3,4), lane1 rows(5,6)/(7,8). The two lane MLEs are2+u and6+u, reconstructing9+7u; the latter's base components9/7 are not the lane MLEs.
- `level2`: omit its proximity vector/mask, one evaluation group, combined row, queried column or ideal final-claim check; change one level2 mask, X^k2 factor, row/column split or skip public-zero padding. Complete inventory/column/ideal-opening checks reject; give each omission a nonzero fixture. Include k1=1/k2=4 where a lane is a column bit, and k1=4/k2=2 where it is a row bit, plus G2's padded group. Do not silently open the two lane claims under unrelated roots.
- `transcript-order`: move C_U after first-level queries or alpha; move an M/column response/equation target after alpha; change the equation list after alpha; move a level2 combined row after its queries; reuse first-level alpha/labels as gamma2/xi2; drop/duplicate an event or replay a different descriptor/session. Invoke the finite state checker in both live-style and NI-style traces; expected `order`/`identity`/`label` refusals. Cryptographic challenge derivation, TCE, proof of work and selective-abort simulation remain outside this model.

If a particular chosen attack fixture cancels accidentally, retain it and add a nonzero fixture **within the same registered shape/family/catalog domain**, clearly recording the search and chosen case. If none exercises the check, that arm is inconclusive; do not weaken the control or expand the domain without a committed amendment.

### Transcript partial order and obligations

The local event model has a fresh descriptor/session identity and independent namespaces `e22/l1/*`, `e22/batch/*`, `e22/l2/*`; they are experimental strings, not allocated production labels. In exact order:

1. Fix descriptors, original ideal commitment, public data claims and their order; in blinded-algebra fix/record every q_g and pair ownership before gamma.
2. Record gamma and xi, compute U/M, fix C_U and every original M_g, then permit first-level queries. C_U replaces fixing transmitted U; it does not follow those queries.
3. Fix the exact distinct/sorted original response columns and their ideal membership results. Derive/record the complete canonical equation list and targets. Only then permit alpha.
4. Run batched sumcheck, absorbing each complete round before its dependent challenge; recompute W and issue the exact two terminal lane obligations. Both remain outstanding.
5. Derive level2 terminal claim groups and gamma2/xi2 after their request identities/values are fixed. Fix all level2 combined rows and masks, including proximity, before level2 queries. Check response shapes/membership, equations and the ideal final MLE opening; consume both exact lane obligations.
6. Accept the finite unit only with no outstanding obligation and all checks/classifications correct. A replay, unused terminal claim or changed root is a refusal even when a scalar equality happens to hold.

This is a proposed partial order/obligation audit, not a full TCE transcript or a claim to reuse an accepted version number. Live-style and NI-style traces differ only in the source-of-challenge tag here; neither claims an FS reduction, entropy/uniformity, grinding coverage, live adversarial-query theorem, one-use simulator or retry/abort security. Such claims require a separate full design and review.

### Resources, reporting and scope of counters

Future runner: pure standard-library Python, `python3 -I`, no production edits/imports or third-party code. Bound to15 minutes per full attempt and256MiB whole-process RSS; periodic and final checks enforce caps before validated. Cap100,000,000 judged comparisons and3,000,000,000 metered algebraic operations; stop with `inconclusive` if any cap is hit. Stream counters, failures and representative exact controls rather than millions of identical success records. Retain every attempt with a UTC/pid identifier and its source/pin hashes before corrections.

A judged comparison is one candidate/reference scalar equality, expected refusal/classification, degree/round identity or consumed inventory assertion. Each vector scalar and its explicit length check count separately; an Fp2 value is one semantic comparison. Internal loop/index comparisons, resource/counter bookkeeping and dictionary identity internals are not evidence comparisons. Report physical comparisons and memoized logical judgments separately, with a complete cache key argument before reuse.

One metered operation is an addition/subtraction/negation/multiplication/division or inversion in the actual integer/Fraction/base-field algebraic data path, including expected-value expressions and fixture-value arithmetic. Reduction is part of a base-field operation; Fraction normalization/gcd is internal to its declared operation. Fp2 add/sub2, schoolbook multiplication7 base operations; powers/inverses count their explicit primitive expansion. Loop/index/bit-address arithmetic, counters, serialization, I/O and identity hashing are excluded. Declare separately any fixture PRNG bit operations; do not claim machine instructions, CPU cycles or complete bit complexity.

Public conservative plan: the largest U/W table has M*k1<=16 entries, at most4 variables, E<=2+3*8=26. The largest V table has32 elements, R2<=16, at most2 level2 evaluation groups plus proximity, and columns n2<=8. Reserve32,768 metered operations and4,096 judgments per batched reduction:18,432 times these is603,979,776 operations and75,497,472 judgments. Reserve200,000 operations and2,048 judgments per two-level fixture for original/second-level rows, direct MLEs, masks, columns, component claims and exact references:921,600,000 operations and9,437,184 judgments. Reserve100,000,000 operations and1,000,000 judgments for the fixed corpus/arithmetic/trace controls. Planned allowance1,625,579,776 operations and85,934,656 judgments is below caps; these are conservative planning allowances, not measured counts or permission to skip work. Fold tables/contract factors rather than repeatedly nesting exponential MLE computations. If a public bound is found inadequate before code, revise/freeze the plan first; if execution exceeds a cap, retain it as inconclusive.

Record Python/OS/CPU/UTC/checkout/source hashes. There are no warmups, repeated wall-time samples, speed ratios or GPU extrapolations: E0 means exact computation and derivation only. No current payload byte count or group size is manufactured from these toy shapes.

## Reproduction

**Draft only; no runnable harness or execution exists.** After parent review, freeze the agreed README bytes under results/ with SHA256 and UTC, commit them, and verify exact committed bytes with `git show <commit>:<path>` before writing audit.py (research rule3). These are intended future commands, not commands executed by this draft:

```sh
python3 -I research/experiments/EXP-0022-two-level-opening-coupling/audit.py --preflight
python3 -I research/experiments/EXP-0022-two-level-opening-coupling/audit.py --all
```

Every future run creates an immutable unique attempt directory with inputs/catalog identities, controls, residual polynomials/cancellation classifications, failures/counterexamples, derivation, logical/physical/operation counters, environment and source snapshots. A stable summary points at the judged final attempt without overwriting earlier outputs. Frozen source drift must be disclosed; changing a domain/criterion requires a committed amendment. No implementation is authorized merely by writing this draft.

## Results

| Arm | Direct reference | Candidate | Judgment |
|---|---|---|---|
| Original evaluation/column coupling | not run | not run | unevaluated |
| Factored W / sumcheck / residual polynomial | not run | not run | unevaluated |
| Lane/level2/ideal terminal opening | not run | not run | unevaluated |
| Attacks / transcript / outstanding obligations | not run | not run | unevaluated |

Raw results: none. [Source pins](results/source-pins.draft.json) are draft preparation provenance, not an execution record.

## Analysis

Pending review/commit/implementation. Main threats are the deliberately ideal binding oracle, unaccepted basis, finite deterministic domains, distinct references accidentally sharing logic, and a partial-order model mistaken for a cryptographic FS schedule. Fixed-challenge cancellations must be reported, not hidden by favorable challenge selection. Ordinary level2 algebra under an ideal final check does not prove that concrete column queries implement that ideal check. Full extraction/G-4, all adaptive/joint-ZK views, current joined group/layout counts and real serialized cost remain unknown.

## Verdict

Not judged; planned draft only. IDEA-0035 remains triaged, priority0, zero credited gain. Neither a full security approval nor a size/performance decision is possible from this unit alone.

## Follow-ups

Parent reviews this bounded proof/domain choice before freezing. A successful finite result would justify independent review of the complete two-level extractor and budget, both-mode transcript/NI retry terms and joint simulator, followed by exact current G/T_g/R/M/nu/payload derivation. A concrete PCS implementation needs its own independent acceptance/tamper corpus. H100 and end-to-end measurements would later settle prover/size/verifier tradeoffs; this draft creates no such evidence.
