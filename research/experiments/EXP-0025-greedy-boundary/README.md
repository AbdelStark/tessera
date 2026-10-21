# EXP-0025: Original-source greedy boundary

| | |
|---|---|
| Status | `planned` |
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

Pending registered execution. No result or performance credit.

## Analysis

The E0 experiment can reject incorrect source mappings, kernels and bounds. It
does not run the real leaf/GKR/PCS pipeline, prove extraction, price additional
opening groups, assign a release protocol number or establish a G-4/ZK budget.

## Verdict

Pending; status planned.

## Follow-ups

If validated, specify the complete runtime advice/catalog/order contract in an
RFC draft and implement independent native/opening composition with real CPU
proofs in the existing #128 draft. No accepting path is authorized by an E0 pass.
