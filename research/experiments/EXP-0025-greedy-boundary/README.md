# EXP-0025: Original-source greedy boundary

| | |
|---|---|
| Status | `done: validated` |
| Idea | [IDEA-0038](../../ideas/IDEA-0038-greedy-boundary-bindings.md) |
| Evidence level | E0 exact integer and Goldilocks-extension arithmetic |
| Hardware | none for the analytic result; synthetic CPU execution only |
| Pre-registered on | 2026-10-21 |

## Hypothesis

The masked exact-greedy keys, selected-logit equality and OneHot token identity
admit degree-two, degree-three and degree-two native polynomials on original
model domains, with every unit-shift source bound by weighted original-point
openings and a complete public-span-derived range table.

## Success criterion

Validate only if every registered equality and degree bound below holds and at
least twelve distinct actual experiment-source faults are detected, with an
unchanged and restored control. Any mismatching equality, false predicate or
degree excess refutes the candidate; an interrupted/incomplete run is
inconclusive. No full protocol, G-4, ZK, CPU speed, memory or GPU result follows.
These criteria and the method are frozen before experiment code.

## Method

Use an independent literal-index baseline and the candidate public-kernel / point
decomposition in the same Python program. Goldilocks `Fp2=Fp[X]/(X^2-7)` uses
arbitrary-precision integers. Seed synthetic inputs with SplitMix64
`0x12800370025`; no model weights, tokens or private data enter the experiment.

- Domains: sequence lengths 2 through 9, vocabularies 1 through 9, and every
  prompt length from 1 through sequence-1: 324 domains (E0 finite enumeration).
- Four non-Boolean challenge sets per domain. Compare translated row weights,
  truncated equality and strict bounded suffix coefficients with literal sums;
  test the cyclic increment's original-point decomposition against a literally
  shifted tensor. The original point coordinates and coefficients are part of
  the equality, including wrap and zero-variable vocabulary axes.
- Four source families per domain: honest greedy rows, a changed selected
  token/maximum, changed real or inactive/padded maxima, and arbitrary field
  sources. Check key, selected and token-binding Boolean-cube sums against their
  literal defining relations at each of the four challenge sets. A family unable
  to change a token (vocabulary one) changes its maximum instead. Counts are
  derived from these frozen loops and reported; none may be silently omitted.
- On sequence lengths 2 through 5 and vocabularies 1 through 4, every prompt:
  verify per-coordinate degree by exact finite differences, at five consecutive
  values for the selected relation and four for the key/token relations, using
  the arbitrary-field family. Bounds are respectively 3, 2 and 2 (E0).
- Exhaust logits from `[-2,2]`, vocabularies 1 through 5, every selected index,
  and every claimed maximum in `[-3,3]`: 129,885 cases (E0). Both the original
  29-bit and public-span predicates must hold exactly for the true maximum and
  its lowest-index token. Include explicit near-28-bit/Goldilocks negative-key
  boundaries. The physical range width is `max(1,bit_length(hi-lo))`, based on
  public bounds only. Unknown/incorrect public bounds are not admitted.
- Faults target offsets, interval/vocabulary masks, suffix strictness, carry
  partitions/wrap, weighted source points, maximum padding, equality/range
  predicates and missing token-source binding. Run each edit in a scratch copy;
  failures must execute assertions, not merely fail to parse. Record the exact
  replacements, source hash, outcomes and restored controls.

No timing comparison is made. Record Python/OS/CPU environment for reproducibility
only. Candidate code shares no production implementation. Known hand cases:
one-hot token index weighted by its vocabulary index recovers that token;
`[0,0]` selects index zero; an all-negative row ignores zero vocabulary padding;
one output selects logits at `prompt-1` and the OneHot at `prompt`.

## Reproduction

After verifying the committed frozen plan, from the repository root:

```sh
python3 -I research/experiments/EXP-0025-greedy-boundary/main.py
python3 -I research/tools/check_kb.py
cargo xtask lint
```

No experiment code exists at pre-registration. The frozen copy, digest and UTC
time are under `results/`; the caller commits them before code is written.

## Results

All counts below are E0 finite exact arithmetic, primary own experiment.

| Check | Count | Result |
|---|---|---|
| Public domains | 324 | All registered shapes executed |
| Public kernel equalities | 5,184 | Literal index sums match |
| Original-point shift equalities | 5,184 | Cyclic tensor and weighted original MLEs match |
| Native final-value equalities | 15,552 | Actual candidate final path matches literal coefficients/sources |
| Boolean-cube relation equalities | 15,552 | Key, selected and token identities match |
| Honest selected/token zero checks | 2,592 | All zero |
| Per-coordinate degree checks | 1,704 | Bounds 2/3/2 hold in the registered finite checks |
| Exhaustive integer cases | 129,885 | Both range predicates accept exactly the lowest-index maximum |
| Large-integer/negative-key boundaries | 207 | Public-span completeness and negative-encoding exclusion hold |
| Actual experiment-source faults | 18 | All fail executed assertions; two unchanged/restored controls pass |

Raw output: [exact.json](results/exact.json),
[source-mutations.json](results/source-mutations.json). Python 3.15.0, macOS
26.6.2, arm64; environment is provenance, not a timing/memory measurement.

## Analysis

The E0 experiment can reject incorrect source mappings, kernels and bounds. It
does not run the real leaf/GKR/PCS pipeline, prove extraction, price additional
opening groups, assign a release protocol number or establish a G-4/ZK budget.

## Verdict

`validated` for the frozen finite E0 criteria. No source branch was omitted.
The experiment did not run a complete proof or establish a production security
bound. The idea retains zero performance credit.

## Follow-ups

If validated, specify the complete runtime advice/catalog/order contract in an
RFC draft and implement independent native/opening composition with real CPU
proofs in the existing #128 draft. No accepting path is authorized by an E0 pass.

## Integration refinement after the frozen experiment

The carry identity can be realized with the existing PCS: native final messages
carry all original-point source evaluations, the transcript fixes them before
PCS Gamma, and each retains its own canonical opening obligation. The native
polynomial uses their verifier-derived weighted sum. The experiment's algebra,
source points and success criteria are unchanged. A new whole-linear-claim PCS
interface is unnecessary. Its naive cross-group aggregation variant, retaining
old per-group-reset Gamma powers, is unsafe: false claims +delta and -delta in
different groups have cancelling aggregate errors. Existing per-group checks
reject both. No such construction was added to production.

Analytic per-variable degrees: a shifted source is multilinear because each
carry coefficient uses only the fixed lower source coordinates' challenge bits,
while its original MLE uses the disjoint higher coordinates. The key multiplies
one such source by a public multilinear coefficient. The selected relation
multiplies two sources by one public multilinear coefficient; the maximum term
has lower degree. Token binding multiplies one source by a public coefficient.
These give the E0 bounds 2/3/2; complete masked/dynamic protocol accounting is
not part of this result.

## Post-registration implementation checks

The independent Rust primitive agrees with 32 literal conformance fixtures.
CPU mixed batches pass in 32 NI/Live transcript-context cases. Each uses real
OneHot row encoding, Merkle authentication and PCS column checks; two false
carry values that preserve the native sumcheck are rejected at the original
opening's `Evaluation { group: Some(0) }` check. These are E1 component tests,
not full model proofs, Live coin-toss sessions or additions to the frozen E0
success criteria. Runtime layout/catalog, graph routing, complete budget and
session integration remain outstanding.

The implementation fault-injection record is
[native-mutations.json](results/native-mutations.json): 42 arithmetic/source
faults and four allocation/check-order faults fail executed Rust tests, with
four unchanged/restored controls passing. No compilation failure counts as a
caught fault. The replayable masks and fixture transcript labels establish no
zero-knowledge or protocol-session claim.
