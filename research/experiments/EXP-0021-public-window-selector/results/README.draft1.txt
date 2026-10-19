# EXP-0021: Fixed-window selector and selected-row relation audit

| | |
|---|---|
| Status | `planned` (preregistration only; no implementation or execution) |
| Idea | [IDEA-0034](../../ideas/IDEA-0034-public-window-attention-shapes.md), independently reviewed with unknown gates and zero credited gain |
| Evidence level | E0: exact integer/rational and Goldilocks/Fp2 arithmetic, finite exhaustive domains plus stated algebraic derivations |
| Hardware | None for the mathematical claim; a bounded Python interpreter run will record its host, not CPU timing |
| Pre-registered on | 2026-10-19; frozen text, UTC and SHA256 under results/ before code; parent review/commit required |

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

This is E0; no warmups, repeated timings, CV, GPU extrapolation or speed result. The harness records counts and exact values only. Hard caps: at most2,000,000 distinct candidate/oracle comparisons, at most100,000,000 primitive exact field/rational operations tracked by the harness, at most256MiB of retained in-memory fixture data, and a 15-minute process watchdog. Stream raw failures/summaries instead of retaining every expanded truth table. If a cap fires, retain partial counts and report inconclusive, not a smaller successful domain. No production tests/builds or concurrent timing are needed; coordinate CPU correctness execution with parent if a timing slot is active.

## Reproduction

After parent review, commit and verification of the frozen text, the intended command from the repository root is:

```sh
python3 -I research/experiments/EXP-0021-public-window-selector/audit.py
```

No audit.py exists in this preregistration. Its future CLI writes `results/run.json`, `results/counterexamples.json`, `results/controls.json`, `results/summary.json` and a text derivation/reproduction record. Every run preserves the first output before any correction/rerun. Environment records checkout, source hashes, Python/OS/CPU and UTC; only exact E0 conclusions are judged. Freeze results/README.preregistered.txt and its SHA256/UTC metadata before code and verify those bytes with `git show HEAD:<path>`; uncommitted or mismatching text stops implementation under research rule3.

## Results

| Case | Baseline | Candidate | Ratio |
|---|---|---|---|
| Selector | direct Boolean/direct MLE oracle, not yet run | three-borrow DP, not yet implemented | not applicable (E0 equality) |
| Key/max/score/PV transport | complete-table sum and direct integer outputs, not yet run | registered reductions/terminal inventory, not yet implemented | not applicable |
| Selected/max/padded audit | direct local witness oracle, not yet run | pointwise relation checker, not yet implemented | not applicable |
| Attack controls | fixed expected rejection/admission distinctions above | not yet run | not applicable |

Raw outputs are pending; no result is implied by the preregistration derivation.

## Analysis

Pending. The DP derivation is a finite-state identity argument that must survive independent oracle checks. Finite alphabets/domains and idealized pointwise range/lookups do not establish full protocol soundness, quantified error bounds, masking/ZK or a production-independent verifier. This unit excludes physical layout/count optimization, GQA routing, arbitrary architecture bounds, serialization/Fiat–Shamir, full sumcheck acceptance, full-field malicious-witness exhaustion, normative Exp-generator validation, real-weight fidelity and H100/end-to-end performance. It does not justify changing the old dense D or its native denominator silently.

## Verdict

Pending; status remains `planned`. A successful E0 unit leaves IDEA-0034 `triaged`, priority0 and no credited gain; a counterexample may reject the specified selector/reduction route only as scoped. No promotion or full security-gate closure follows automatically.

## Follow-ups

Only after independent review of this unit: decide whether a versioned statement/ClaimList policy and physical layout/count derivation merit a separate RFC/experiment. Complete #145/#394 joint masking and independent verifier/driver integration remain outside this audit. Fidelity and serving changes need their own evidence; GPU timing alone would settle none of the algebraic or policy gates here.
