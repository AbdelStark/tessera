# EXP-0021: Fixed-window selector and selected-row relation audit

| | |
|---|---|
| Status | `done: validated` (bounded E0 selector/transport/local-relation unit only) |
| Idea | [IDEA-0034](../../ideas/IDEA-0034-public-window-attention-shapes.md), independently reviewed with unknown gates and zero credited gain |
| Evidence level | E0: exact integer/rational and Goldilocks/Fp2 arithmetic, finite exhaustive domains plus stated algebraic derivations |
| Hardware | None for the mathematical claim; the bounded Python interpreter run records its host, not CPU timing |
| Pre-registered on | 2026-10-19; committed as `e3a2a51` before code; frozen SHA256 `90b7307ed2a2162bbd7a56a426c959068c83500c8111a6432e72981f641252bc` |

## Hypothesis

For a public all-local window fixed across every row/layer/head of each audit descriptor, the three-borrow bit DP evaluates the exact joint window-selector MLE, and the specified key/max/score/PV reductions and selected-row constraints agree with direct finite-domain integer semantics while rejecting every registered malformed, omitted or altered constraint case.

## Success criterion

This is a bounded **E0 relation audit**, not a complete protocol, ZK, fidelity, compiler-layout or performance validation. `validated` requires **all** the following, with no mismatch or unexplained skipped case:

1. The DP derivation below is checked against direct Boolean predicates on every registered Boolean point and against independent direct weighted-sum MLE oracles on every registered rational/Fp2 off-cube point. It uses at most eight borrow states and at most `32*d` attempted state/bit-choice transitions per evaluation of the q/i selector; separate layer/head LT factors and special-case validation are counted separately.
2. Every registered honest key/max/score/PV reduction agrees exactly with the direct oracle, every round polynomial has the stated per-variable degree, its `p(0)+p(1)` identity holds, and all terminal c/x/Q/K/e/V claims are transported and consumed with the exact layer/head/query/key/channel coordinates stated below.
3. The pointwise selected-row/max/padding audit accepts exactly the oracle's c/e/R/slack witness within the exhaustively registered toy domains. It also passes the stated normative-table fixtures and Goldilocks guard controls, which are explicitly not full-field exhaustion.
4. Every registered mandatory negative control is rejected for its named audit reason, with an unchanged control that passes. The intentionally wrong product-of-MLE selector reproduces the fixed 9/16-versus-1/2 trap and fails comparison; each registered omission mutant must admit its supplied cheating/false-claim fixture or otherwise demonstrably fail its positive comparison. A mutant not exercised by its fixture makes that arm incomplete, not a successful kill.
5. Raw records, exact domain/case counts, maxima of state/transition counts, all negative controls, failures/counterexamples and a structured summary are saved. No runtime, speedup, security-bit, production acceptance or quality number is inferred.

`refuted` means one reproducible selector mismatch, false relation identity/transport, missed mandatory attack, or accepted alternative witness within the registered domains after the harness/oracle are independently checked. A concrete mathematical counterexample has precedence over all other successes. `inconclusive` means a resource cap, incomplete arm, unresolved oracle/implementation discrepancy, insufficient mutation sensitivity or missing derivation prevents judgment. Distinguish selector, transport and row-audit verdicts before the overall conjunction. Finite success does not upgrade IDEA-0034's full soundness/ZK/fidelity gates or its zero gain/priority. An implementation defect corrected before final judgment retains the failed run and correction record; no finite domain or criterion may silently shrink.

## Method

### Scope and source context

Read the current parent-confirmed IDEA-0034, its [independent review](../../ideas/results/seed-review-2026-10-19.json), the [exact MLE trap](../../ideas/results/seed-preflight-2026-10-19.json), RFC-0002 Fp2, RFC-0009 SoftmaxRow/Lemmas5–6, RFC-0010 Relations, RFC-0030 selector/key/leaf reductions, RFC-0035 padding pins, and RFC-0038/0047 boundaries. Source hashes and checkout are saved in results/source-pins.json. This unit fixes an **all-local** policy: no global flag, per-layer window, changed template, secret compaction, full advice layout, serialization, Fiat–Shamir driver or masks are implemented. `D_window` differs from the old dense D where the public window excludes keys.

The proposed Python harness is local research-only pure exact arithmetic, invoked with `python3 -I`; it may use the standard-library Fraction, integer arithmetic, JSON and the repository's pinned table oracle for the explicitly identified normative fixture inputs. It runs no third-party code, benchmark, production build or real model. Candidate DP and direct reference sum must not share a borrow recurrence. Use Python big integers; Fp2 is an independent schoolbook pair modulo `P=2^64-2^32+1`, with `u^2=7` (RFC-0002). Cross-check its zero/one/distributive and rational-to-base embedding controls before using it. The table oracle is input generation, not an independent test of itself.

### Audit descriptor and coordinates

An audit-only immutable descriptor is the exact tuple `(format,d,n,W,L,H)`, with format `e0-local-window/1`, integer d in `[0,3]`, `N=2^d`, integer `0<=n<=N`, integer `1<=W<=2^17`, and integer `1<=L,H<=3`. Python bool is refused as an integer. N is derived, not independently supplied. There is no global/head/layer-specific policy field. Exact expected descriptor identity is carried to every relation/claim check; a changed W/n/L/H or swapped descriptor cannot replay another audit's claims. This is a local audit check, **not** a specified TCE schema, hash binding or proof that the production statement absorbs these fields before challenges.

Layer/head cubes pad L/H separately to their next powers of two; position axes use N even for n=0. Coordinate order in the oracle is low bits first within each named axis, with named axes `(i,q,h,l)` for score-like tensors and `(channel,q,h,l)` for PV outputs. Every claim records its tensor name, descriptor identity and all named coordinates; tuple reordering is never implicit. n=0 is an empty-selector mathematical control, not an assertion that a production empty sequence is accepted.

### Candidate DP and derivation to audit

For n>0, set `w=min(W,N)-1` and `a=n-1`. Clamping W at N is valid because on accepted `q>=i` the difference lies in `[0,N-1]`; W>=N therefore selects every causal key. This is not permission to clamp an invalid W<=0.

Process bit b from least to most significant. State `(b0,b1,b2)` holds incoming borrows for `q-i`, `w-(q-i)`, and `a-q`. Start only state `(0,0,0)` at weight1. For each state and bit choice `(Q,I)` in `{0,1}^2`:

```text
t = Q-I-b0;         D = t mod 2;      b0_next = [t<0]
v = w_b-D-b1;                         b1_next = [v<0]
z = a_b-Q-b2;                         b2_next = [z<0]
weight *= eq(rq_b,Q)*eq(ri_b,I).
```

Add weights of transitions with the same next state. After d bits, accept only state `(0,0,0)`. The candidate's answer is its final weight. n=0 returns0 before computing n-1. d=0/N=1/n=1 returns1 for the positional selector after descriptor validation; all products are empty products1.

Derivation obligation: induction on processed bits shows D's bits are the low-bit subtraction result and b0 is its borrow, b1 is the borrow of w minus that low-bit result, and b2 is the borrow of a-q. Final b0=0 makes D the true nonnegative q-i, so final b1=0 is equivalent to q-i<=w; final b2=0 is q<n. Thus accepted paths are exactly `[i<=q<n][q-i<W]`. The path weight is `eq(rq,q)*eq(ri,i)`; summing accepted paths is the definition of the **joint** MLE. This is a proposed explicit derivation to verify, not a theorem assumed to excuse mismatches. Eight states and four choices per bit give the `32*d` transition upper bound; merging equal states is arithmetic, not extra public randomness.

For full selector `P(l,h,q,i)=[l<L][h<H][i<=q<n][q-i<W]`, multiply by independent LT MLE factors on l and h **because those variables are disjoint**. The q/i comparisons share variables and may not have their extensions multiplied. Reference LT and P are direct truth-table weighted sums, not calls to the candidate DP.

### Frozen selector domains

All domains below are mandatory; no random-only substitution.

- For each d in `{0,1,2,3}`, every n in `{0,...,N}` and W in the deduplicated set `{1,...,N} union {N+1,1024}`, exhaust all Boolean q/i points. W is one public constant across all rows/layers/heads in that case. Cover causal edges, old-window edges, W>N, empty n, and zero-dimensional axes. Compare both rational and Fp2 DP values with 0/1 predicate values.
- For d in `{0,1,2}` and the same n/W set, exhaust every coordinate of rq/ri in rational alphabet `{0,1,1/2,-1}`, and separately every coordinate in Fp2 alphabet `{0,1,u,1+u}`. Compute the reference MLE directly from Boolean table entries. The word exhaustive applies to these finite alphabets only.
- For d=3 with the same n/W set, use eight explicit rq/ri point pairs in each arithmetic: all-zero; all-one; all-half; all-minus-one; rq all-half/ri all-zero; rq all-zero/ri all-half; alternating `(0,1,1/2)`/`(1/2,0,1)`; and `(1/2,-1,2)`/`(2,1/2,-1)`. In Fp2, map the non-Boolean entries 1/2,-1,2 to `(1/2,u)`, `(P-1,1)`, `(2,1)` respectively; this defines fixtures, not a homomorphism from rationals. Record exact coordinates in raw data.
- Full layer/head composition: d=2/N=4, L/H each in `{1,2,3}`, n in `{0,1,3}`, W in `{1,2,4,1024}`. Exhaust all Boolean l/h/q/i coordinates. At four complete off-cube points (all-half, all-u, alternating0/1, and cycling half/u/(1+u)/0 across named axes), compare composed selector to a direct full-table MLE. Half means base-field inverse2 in Fp2. This covers non-power-of-two real counts and padded layer/head coordinates.
- Fixed trap: N=2,n=2,W=1 at rq=ri=1/2 gives exact joint diagonal selector1/2; causal-MLE3/4 times lower-bound-MLE3/4 gives9/16. Preserve the wrong algorithm as a negative control, not a candidate result.

### Key/max transport and complete terminal obligations

Use the full logical cube. Scores `x=QK` are not zeroed outside P. Define Boolean tables `k=c_broadcast-x`, `kappa=P*k+(1-P)*K0` and `leaf=P*k+1-P`. For each target point r, both reductions must compute the same sum

```text
A_r = sum_z eq(r,z)*P(z)*(c(z_row)-x(z))
    = kappa_MLE(r)-K0*(1-P_MLE(r))
    = leaf_MLE(r)-1+P_MLE(r).
```

The tested sumcheck integrand is the product of the separate multilinear extensions `eq(r,z)*P_MLE(z)*(c_MLE(z_row)-x_MLE(z))`; it agrees on the Boolean summation domain and has degree at most3 per variable. Enumerate round polynomials by exact summation over remaining Boolean variables after prior challenge bindings. Check their degrees and `p(0)+p(1)=incoming` identities. Use degree3 evaluation grid0,1,2,3 and exact interpolation; the direct reference summation does not use that interpolation. At final point s, check `eq(r,s)*P_MLE(s)*(c_MLE(s_row)-x_MLE(s))`.

Register and verify **both** terminal claims c at projected row coordinates and x at all score coordinates, with opposite coefficients. If a public coefficient vanishes, keep the required claim inventory or explicitly record a zero-coefficient obligation; it may not hide a missing nonzero claim in another case. Dropping a c/x claim or replacing a row projection/axis order must be caught by a nonzero-coefficient fixture. Direct evaluation from the complete synthetic tables supplies the audit oracle; no commitment/proof-driver acceptance is implied.

For the x claim at `(r_l,r_h,r_q,r_i)`, reduce the full dense MatMulAct identity as

```text
x_MLE(r) = sum_{l,h,t} eq(r_l,l)*eq(r_h,h)
             *Q_MLE(l,h,r_q,t)*K_MLE(l,h,r_i,t).
```

Here t is the padded channel cube; Q/K are zero at padded l/h/positions/channels. Sum over shared l/h as well as t, retaining their equality weights; do not multiply Q/K extensions at the target l/h instead. Check degree<=3 in shared l/h and<=2 in t, then register and consume the exact final Q and K claims. No GQA/merged-head routing theorem is claimed: the bounded fixture has one Q/K head per public head, and swaps of its distinct head/layer identities are mandatory negative controls.

For each PV output coordinate, form the Boolean reference `y(l,h,q,a)=sum_i e(l,h,q,i)*V(l,h,i,a)` over **all** N keys. Its MLE target reduces as

```text
y_MLE(r_l,r_h,r_q,r_a) = sum_{l,h,i} eq(r_l,l)*eq(r_h,h)
                       *e_MLE(l,h,r_q,i)*V_MLE(l,h,i,r_a).
```

Excluded e entries, including omitted public-zero blocks, are defined as zero; V and x at those positions need not be zero. Check the full-domain reduction, degree<=3 in shared l/h and<=2 in i, with final e/V claims consumed. Direct selected-only integer PV and full-domain zero-padded PV must match on Boolean output indices before taking MLEs. “Omit” means a public zero e entry with a known logical index, never shortening the domain or shifting remaining keys.

Transport fixtures: L/H each in `{1,3}`, N=4, n in `{0,1,3,4}`, W in `{1,2,4,1024}`, two real channels (cube2), and four complete deterministic Q/K/V families: all-zero; all-one on real entries; signs `(-1)^(l+h+position+channel)`; and residue pattern `((l+2*h+3*position+channel+offset) mod3)-1`, with offsets0/1/2 for Q/K/V. Padded values are zero. Maxima and e come from the pointwise oracle below. Use eight complete target/challenge catalogs: all0, all1, all-half, all-u, alternating0/1, cycling half/u/(1+u)/0, alternating(2+u)/(1-u), and SplitMix64 seed `0x0021003400000001` reduced canonically to Fp2 components solely to define a deterministic fixture. This is not a challenge-uniformity or soundness-calibration experiment. Both key/leaf reductions and score/PV transports run for every family/descriptor/catalog.

### Pointwise selected/max/padded audit

For N in `{1,2,4}`, n in `{0,...,N}`, W in `{1,...,N} union {N+1,1024}`, every real/padded q, exhaust score rows in `{-1,0,1}^N`. Include ties; only the maximum **value** is unique, and no first-index tie rule is invented. Extend the same rows through L/H in `{1,3}` with every Boolean real/padded l/h combination to check row-selector branches.

The exhaustive toy table is the explicitly synthetic functional map on keys0..3 with values `[4,1,1,0]`, sigma2 and K0=3. It models the structural E(0)>0/E(K0)=0 assumptions; it is not a claim-list-accepted production Exp parameter set. At a selected position, lookup membership checks `(c-x,e)` by exact field-key equality to an enumerated integer key. At masked positions it checks `(K0,e)`. Compute max leaf values as specified and require root0 on every real row, root1 on padded rows. c candidates are every integer in `{-2,-1,0,1,2}` plus non-base Fp2 probes `u`, `1+u` and each direct real maximum plus u. The checker must not assume malicious c is an integer: the zero-product/key argument anchors c to an actual selected score. The finite c probes complement that derivation; they do not enumerate Goldilocks/Fp2.

For each key membership, enumerate e in `{0,1,2,3,4}` at that coordinate and confirm exactly one table output; functional pointwise lookups justify avoiding the Cartesian product of all e candidates. For each surviving c/e, exhaust all R in `{0,...,7}`, compute integer `s=sum_i e_i`, `a=psi*16-R*s`, `b=s-psi-a`, and check `0<=a,b<2^20` (sigma+18) and padded pins `(1-psi)*c=(1-psi)*R=0`. The independent witness oracle is direct selected-set max, table evaluation and `R=floor(16/s)` for real rows, all-zero c/e/R/slacks on padded rows. Check unique outputs/slacks; no limb-decomposition implementation or LogUp proof is tested. Include non-base R probes as range-refusal controls; the range predicate is an explicit idealized true-integer range check, not a new range-proof theorem.

Normative-table controls are separate: the pinned RFC-0008 table oracle supplies `Exp` at rational score scale1, sigma16, key_bits4/K0=15, keys0..15. Verify E(0)=2^16 and E(15)=0 and that narrower widths fail the zero-last-key condition at the candidate spread. For N=4,n in `{1,3,4}`, W in `{1,2,4,1024}`, rows all0, increasing(-1,0,1,1), decreasing(1,1,0,-1) and tied(1,-1,1,-1), compare direct honest c/e/R/slacks and reject R±1 when in range, R=0 and the range maximum unless equal to the true R. This is a fixture check, not exhaustive normative-table/reciprocal enumeration.

Goldilocks obligations: derive the selected-row Lemma5 argument using actual integer keys below P and a nonzero masked leaf. Check the unchanged reciprocal guard `(2*max_context-1)*2^(2*sigma)<P` for `(max_context,sigma)` equal to `(8,2)`, `(8,16)`, `(2^17,16)`, `(2^17,23)` and `(2^17,24)`; the last is a refusal control. This is guard arithmetic, not a claim that every other listed combination satisfies all production schema limits. No Goldilocks wrap result is inferred from a toy modulus or toy sigma.

### Mandatory adversarial controls and mutation sensitivity

Audit reasons are local named strings, not production RejectReason codes. Save each full input, expected reason and actual result; each mutation has an unchanged passing control.

- `descriptor`: refuse missing/extra fields, unknown format, bool-as-int, d<0 or d>3, n<0 or n>N, W<=0 or W>2^17, L/H outside1..3, and any per-layer/global flag. Check mismatched descriptor identity on otherwise valid claims. A “skip validation” mutant must be exposed by a named malformed case.
- `selector`: wrong multiply-the-comparison-MLEs algorithm must fail the 9/16 trap. Drop each final-borrow condition separately and use a Boolean violating exactly its inequality; clamp-W behavior and n=0/d=0 branches have explicit controls. Changing LT layer/head bounds must fail a padded l/h point.
- `excluded-e`: at N=2,n=2,W=1,q=0, selected score0/e4 and masked e1 with V=(0,1) give wrong PV1 instead of0. The full checker rejects the masked lookup; an omitted-masked-lookup mutant admits the supplied local tuple with s5,R3,a1,b3. Other rows remain honest. This is a complete local relation fixture, not a production proof.
- `false-max`: at N=4,n=3,W=3,q=2, selected scores all0 with claimed c1 yield selected e1, s3,R5,a1,b1. The root condition rejects it. Omitting the root check admits it; replacing the masked leaf1 by0 also makes the product vanish and must expose the false c. The remaining padded row is canonical.
- `padding`: at a padded q/l/h row, set c1 or R1 separately with e/s/a/b0. Full checks reject its zero pin; disabling the corresponding pin admits the alternative local witness although PV stays0. Do not mislabel that SP-3 failure as changed output.
- `transport`: alter one key/leaf target, c term, x term, Q/K operand or e/V operand at a point with nonzero public weight; omit each terminal claim separately; swap q/i or two distinct heads/layers; renumber a compacted e coordinate as if the cube shrank. Independent complete-table evaluation/inventory must reject. A no-op mutation passes. An oracle that simply echoes candidate claims cannot count.
- `original-D`: a deterministic W=1/N=2 real-row fixture with an excluded nonzero key/value exhibits a difference between dense and windowed PV; this is expected changed-model evidence, not failure of the window oracle or fidelity approval.

Some final-borrow inequalities imply others for edge constants, so a missing-condition mutant must be exercised on an independently listed descriptor where its violating Boolean point exists. If no such point exists within the frozen cases, report that arm inconclusive and seek an explicit preregistration amendment rather than claim sensitivity.

### Resource bound and measurement protocol

This is E0; no warmups, repeated timings, CV, GPU extrapolation or speed result. The harness records counts and exact values only. The following pre-code revision replaces the first drafts' insufficient 2,000,000-comparison/100,000,000-operation caps without changing any domain, criterion or mathematical relation. The old snapshots remain in results/README.draft1.txt and results/README.draft2.txt.

**Definitions.** A *judged comparison* is one explicit candidate-versus-independent-oracle scalar/rational/Fp2 equality, expected acceptance/rejection comparison, degree/round-identity assertion or terminal-claim/inventory assertion. An Fp2 value is one semantic value for this counter. A vector/tuple's scalar equalities count individually, plus one length/identity judgment when checked. Candidate lookup-output probes and R acceptance probes each count, including expected rejection. Ordinary internal loop bounds, borrow decisions, dictionary comparisons, assertions about counter bookkeeping and resource checks are not additional judged comparisons; they are not empirical security evidence. Do not hide extra audit judgments inside these exclusions. Report logical judgments and any cached physical evaluations separately.

A *primitive exact arithmetic operation* is one addition, subtraction, negation, multiplication or division/inversion on an instrumented integer, Fraction or base-field value used by the candidate or direct reference. Base-field reduction is part of its field operation, not a separate instruction estimate. Fraction normalization/gcd internals are likewise not separately counted: this is a declared algebraic unit, not CPU cycles or bit complexity. Count Fp2 operations through their base-field schoolbook expansion: addition/subtraction2 base operations; multiplication5 base multiplications and2 additions, hence7 operations. If an optimized algorithm is used, record its actual expansion rather than silently retain7. Powers/inverses count the primitive operations in their explicit arithmetic routine, not one opaque pow call. Counter increments, loop/index arithmetic and I/O/formatting are excluded. The pinned repository Exp oracle is a separate input-generation dependency: record its16 requested table entries and source hash; its internal transcendental-certification work is outside this arithmetic counter, but remains inside the process watchdog/memory bound. No source-oracle cost claim is made.

**Public comparison bound (E0 planning arithmetic, not results).** If `a(N)` is the number of distinct W values, then `a(1)=3`, `a(2)=4`, `a(4)=6`, `a(8)=10`. The padded layer/head-coordinate multiplier for the row audit is `(pad(1)+pad(3))^2=25`. Thus the pointwise row contexts are:

| N | `(N+1)*a(N)*N*3^N*25` |
|---|---:|
| 1 | 450 |
| 2 | 5,400 |
| 4 | 243,000 |
| Total | 248,850 |

Budget320 judged comparisons per row context: up to8 c probes; up to `8*4*5=160` per-coordinate e-output probes; up to `8*8=64` R acceptance probes without assuming c uniqueness or any earlier check passes; and88 judgments for root/padding/descriptor/expected scalar-witness checks. Candidate keys, products and row sums are reused within a row context, not recomputed for every e-output probe. This gives79,632,000 row judgments. The selector grids use at most50,000 primary comparisons by direct enumeration: 12,588 Boolean positional comparisons in both arithmetic types;15,756 small off-cube comparisons;1,440 d3 catalog comparisons;18,816 full layer/head Boolean comparisons (conservatively allowing both arithmetic types);864 full off-cube comparisons (same conservative doubling); and the fixed trap. Reserve100,000 for selector judgments including local assertions. Transport has `4*4*4*4*8=2,048` descriptor/family/catalog runs; allow up to512 judged comparisons per run, or1,048,576, and reserve2,000,000 including target and inventory controls. Reserve a further100,000 for normative/malformed/Goldilocks/mutation controls. The combined planning allowance is81,832,000, below the comparison cap below. If a new implementation introduces additional explicit judgments beyond these allowances, count them; it may not suppress them to fit the bound.

**Bounded transport arithmetic.** Precompute complete Boolean factor tables and partial operand contractions once per fixture/target. Maintain folded factor tables for the already bound prefix. For a round polynomial, evaluate each factor's affine interpolation for the current bit and sum the products over the remaining Boolean rows; do not call a fresh nested full-table MLE at every sumcheck vertex. The direct oracle may independently construct exact affine-factor coefficients/weighted Boolean sums or use its own factor-fold traversal; it must not call the candidate DP or echo its claims. Shared immutable truth tables are inputs, not a shared borrow recurrence. The candidate DP evaluates the public selector at actual target/final points; public Boolean P tables for summation are constructed directly from the descriptor predicate, not by rebuilding a DP for every polynomial-evaluation vertex.

A conservative planning allowance for one candidate or reference reduction with at most3 factor tables, initial table size M and b variables is `64*M+256*b` high-level Fp2 arithmetic operations: four degree<=3 round-grid evaluations over the successive halving tables, prefix binding, interpolation and checks. For key and leaf, M<=256,b<=8 each; score M<=32,b<=5; PV M<=64,b<=6. Two independent computations of all four reductions are bounded by `2*7*[2*(64*256+256*8)+(64*32+256*5)+(64*64+256*6)] = 641,536` base-field operations per run, or1,313,865,728 over2,048 runs. Reserve a further100,000 instrumented operations per run for truth-table setup, target contractions, direct terminal MLEs and DP/LT checks:204,800,000. Reserve500,000,000 for all selector grids, based on at most50,000 comparisons and10,000 operations per direct-MLE/DP/LT evaluation pair at these maximum cubes. The row audit allowance is1,024 operations per context (254,822,400), covering cached key/product arithmetic, integer R/slack arithmetic and direct witness checks without assuming acceptance. Reserve50,000,000 for all remaining fixed controls and setup. Total planned primitive-operation allowance is2,323,488,128. These are conservative algebraic bounds for the specified cached/folded method, not timings, measured instruction counts or a license to skip cases. Instrument actual operations and preserve any bound mismatch.

**Hard caps.** At most100,000,000 judged comparisons and3,000,000,000 instrumented primitive exact arithmetic operations; unchanged retained-fixture memory bound256MiB and whole-process watchdog15minutes. The arithmetic cap exceeds the public2,323,488,128 allowance; it is not an expectation that the run will execute that many operations. Stream per-arm/per-descriptor counters, every failure and every mandatory control, with representative exact success fixtures and reproducible deterministic digests/counts rather than millions of redundant success records. Successful caching must retain full logical case coverage and distinguish logical judgments from physical evaluations. If any cap fires or a planning bound is unexpectedly exceeded, retain partial counts and report inconclusive pending review, not a smaller successful domain. No production tests/builds or concurrent timing are needed; coordinate CPU correctness execution with parent if a timing slot is active.

### Preregistration clarification, 2026-10-19 (before review/commit/code)

The transport fixture's two-channel Q/K products can have spread4, exceeding the toy table's K0=3. Therefore its phrase “Maxima and e come from the pointwise oracle below” means the **normative fixture table** at scale1/sigma16/key_bits4/K0=15 for every transport family, not the exhaustive toy table. This includes keys0..4 and honest transport fixtures remain complete. The synthetic sigma2/K0=3 table belongs only to the separate exhaustive score-row alphabet `{-1,0,1}` audit, whose spread is at most2. No table/generator correctness or full sigma16 reciprocal exhaustion is inferred. This clarification changes no success threshold or registered finite domains. The first draft is retained as results/README.draft1.txt; the frozen reviewed candidate includes this clarification.

## Reproduction

After parent review, commit and verification of the frozen text, the intended command from the repository root is:

```sh
python3 -I research/experiments/EXP-0021-public-window-selector/audit.py
```

The frozen preregistration preceded audit.py. The implemented CLI preserves unique attempt directories; `results/run.json`, `results/counterexamples.json`, `results/controls.json`, `results/summary.json` and `results/corrections.md` identify the final run and retained history. Every run preserves the first output before any correction/rerun; full-run source snapshots are retained with each attempt. Environment records checkout, source hashes, Python/OS/CPU and UTC; only exact E0 conclusions are judged. Freeze results/README.preregistered.txt and its SHA256/UTC metadata before code and verify those bytes with `git show HEAD:<path>`; uncommitted or mismatching text stops implementation under research rule3.

## Results

**Validated for the registered bounded E0 unit.** All figures below are exact finite/domain bookkeeping or metered algebraic counts, primary repository evidence; none is wall-time performance. The [final summary](results/summary.json) and [run pointer](results/run.json) name the immutable final author attempt `20261019T105827.081336Z-45083`. Raw events, controls, counterexamples, environment and source snapshot remain in that directory. Python 3.14.6 ran on macOS 26.6.2 arm64; this is host provenance, not a hardware comparison.

| Case | Baseline | Candidate | Ratio |
|---|---|---|---|
| Positional Boolean selector | Direct integer predicate | 12,588 exact comparisons across rational/Fp2 representations; all match | not applicable |
| Small off-cube selector | Direct Boolean-table MLE | 7,878 rational and7,878 Goldilocks/Fp2 points; all match | not applicable |
| d3 off-cube catalogs | Same direct MLE oracle | 1,440 exact points; all match | not applicable |
| Layer/head composition | Full padded-table oracle | 9,408 Boolean and432 off-cube points; all match | not applicable |
| Local max/e/R/padding | Direct finite witness oracle | 248,850 logical row contexts,933 distinct row_core inputs; all match | not applicable |
| Normative sigma16 fixtures | Pinned table oracle plus direct integer reciprocal | 16 table inputs,832 row probes and5 Goldilocks guard checks; all match | not applicable |
| Key/leaf/QK/PV transport | Complete synthetic tables and separate coefficient-expanded round oracle | 2,048 runs,8,192 reductions,38,912 rounds; all identities/degrees/terminal obligations match | not applicable |
| Named adversarial controls | Registered passing controls and named rejection/admission expectations | 69 controls, all sensitive as specified | not applicable |

The known incorrect factor-MLE algorithm yields9/16 at the frozen W1/two-position half-point, versus the joint selector's1/2. It is a rejected shortcut. The DP observed at most7 occupied states and48 attempted transitions; every evaluated call stayed under the proved structural allowances8 and32*d. Those maxima are bookkeeping, not throughput or a claim about every production-sized instance.

### Logical coverage, physical checks and resource accounting

The row domain contains450+5,400+243,000=248,850 contexts. Its pure checker cache key is `(scores, selected, real)`: tuple length gives N; table/sigma/field and c/R candidate rules are fixed for this toy arm; original q/n/W/layer/head coordinates affect the checker only through selected positions and the real-row flag. No cache is reused across normative sigma16 or transport arms. The run executed933 unique row_core inputs and247,917 cache hits. A cached replay is coverage of an equivalent pure-checker input, **not** another physical test.

| Counter | Logical probes | Physically executed probes |
|---|---:|---:|
| Toy c candidates | 1,758,510 | 6,681 |
| Toy per-coordinate e candidates | 34,739,430 | 130,385 |
| Toy integer R candidates | 1,990,800 | 7,464 |
| Toy non-base R refusal probes | 497,700 | 1,866 |

Total42,431,606 logical audit judgments comprise876,535 physically executed judgments and41,555,071 memoized logical judgments. Physical judgments by arm are setup1, arithmetic107, selector39,625, pointwise156,570, normative837, transport678,402 and controls993. Per-coordinate functional e checks justify factorizing the e-candidate product; this is not exhaustive enumeration of all malicious field-valued witnesses.

The final arithmetic meter reports317,682,340 algebraic operations under the frozen definitions. It is not a physical instruction count, bit-complexity measure or benchmark. Arithmetic by arm was not recorded and is not inferred from the judgment counts. The [accounting-only diff](results/accounting-fix.patch) added exactly3,090 operations to the independently reproduced mathematical run:18 expected-half integer product/remainder operations and3,072 family2 parity remainders. The pre-run prediction and actual delta agree, with no judgment/domain/control change. The final source checks all caps immediately before `validated`; resource use remained below100M judgments,3B metered operations,256MiB and the15-minute watchdog. The whole-process RSS check is stronger than the retained-fixture limit; its exact recorded maximum is in the final summary. No elapsed-time gain is reported.

The [attempt history](results/attempt-history.json) and [correction record](results/corrections.md) preserve every preflight/full attempt. The first preflight used an insensitive W1 symmetric q/i fixture; its original raw `refuted` label is retained but does not constitute a selector counterexample. The corrected control uses registered W2 and invokes the actual coordinate/reduction checker. Parent independently reproduced the pre-accounting mathematical result and independently reconstructed all 248,850 row contexts, 933 distinct inputs and the logical/physical row-probe counts in [the domain reconstruction](results/parent-domain-reconstruction.json). [Independent accounting review](results/independent-accounting-review.json) verifies the final three-hunk diff, exact +3,090 operation delta, unchanged judgments/controls and cap check immediately before validation. No failed run was discarded or silently reclassified in its raw file. The first three preflights retain outputs and source hashes but no per-attempt source snapshots; the fourth preflight and every full run retain their source. Independently rerun the domain recount with `python3 -I research/experiments/EXP-0021-public-window-selector/recount.py`.

## Analysis

The borrow-DP induction and direct truth-table oracles agree in all frozen domains, including W>N, n0, d0, full-width n-1/W-1, non-power-of-two real lengths and padded layer/head coordinates. Key and max leaves reduce through the sum of eq*P*(c-x); both c and x claims survive to the expected coordinates. The dense QK and full-domain PV reductions retain shared layer/head equality factors and consume Q/K/e/V obligations. There is no replacement of a sum of products by a target-point product. Altered targets, omitted claims, axis/descriptor changes, excluded exponentials, false maxima and free padded values trigger their registered audit checks.

### Dated range-lift clarification, 2026-10-19

An early source sentence overstated the maximum lemma: score spread<P alone is insufficient. The [independent control](results/independent-spread-control.json) gives `S=(P-3)/2`, scores `(-S,S)` and claimed c=-S. Modular keys0/3, toy e4/0, root0, R4 and slacks0/3 pass, even though c is not the true maximum. This control is outside the frozen score alphabets and current RFC-0009 score bound; it refutes the overbroad sentence, not this registered finite result or production SoftmaxRow.

The scoped derivation requires true integer score span Sspan with **Sspan+K0<P** (or a stronger quantitative Lemma0 bound). A zero selected factor anchors c to a selected score; then delta-k is a multiple of P strictly between -P and P, hence delta=k>=0. The harness checks the quantitative bounds2+3 for the toy rows,4+15 for transport,2+15 for normative fixtures, and2^22+(2^32-1) for the current RFC score/key envelope. No frozen text or domain was edited. The separate outside-domain example is retained as a derivation control, not counted as exhaustion or full-protocol evidence.

### Limits of this result

This audit uses exact pointwise lookups/range predicates and direct complete-table terminal evaluation. It does not produce or accept a serialized production proof, implement the versioned window descriptor in ClaimList, prove a LogUp/range/commitment implementation, establish full-field witness exhaustion or calculate G-4 error. Its terminal inventory checker idealizes the later opening obligation. The rational/Fp2 samples are finite and deterministic, not soundness calibration or evidence about uniform Fiat–Shamir challenges.

Only all-local windows fixed within each descriptor are covered. GQA routing, global/local layer templates, physical advice layout/counts, masked tree/sumcheck simulation, #145/#394 joint views, full independent verifier/driver integration, real-model fidelity and native/H100 proving costs remain outside the unit. D_window is a changed model: the explicit dense/window PV fixture differs as expected. No native-denominator, quality, ZK or performance approval follows.

## Verdict

`validated` for the exact registered finite selector, transport and local row-relation audit. [IDEA-0034](../../ideas/IDEA-0034-public-window-attention-shapes.md) remains `triaged`, priority0, zero credited gain and unknown full soundness/ZK/fidelity gates. This experiment does not validate or promote the entire proposed model/protocol change.

## Follow-ups

The final accounting diff and preserved evidence have independent review; the parent independently reproduced the mathematical run and domain counts. A separate bounded unit would derive physical windowed layout/counts and policy/ClaimList binding, with the changed model/native denominator explicit. Any protocol proposal still needs full claim/extractor/error and joint masking arguments, independent implementation and serialized tamper coverage. Real-weight fidelity/serving and H100/end-to-end performance require separate evidence; no experiment is silently extended here.
