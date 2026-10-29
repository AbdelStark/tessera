# IDEA-0041: Public per-site requant gadget promotion

| | |
|---|---|
| Status | `triaged` |
| Origin | [#617](https://github.com/AbdelStark/tessera/issues/617): real 1B public geometry and a two-layer synthetic reproduction |
| Lever | Correctness/coverage of stacked leaf reads; `C_lin`, `K_lin`, table rows may increase |
| Spec surfaces | RFC-0006 selection; RFC-0045 same reads; RFC-0050 auxiliary layout. Canonical table/advice changes require a superseding RFC before adoption. |
| Priority | 0 credited performance gain; correctness prerequisite to the authorized real-model proof |

## Mechanism

Candidate for **Protocol One integrity only**; no production code changes yet.
Group Requant nodes by the public layer-template position, leaving prologue/epilogue
nodes separate. Preserve the old selection byte for byte when all layers already
have identical ordered read roles/axes. Otherwise choose the first common feasible
family from Split then Wide, deriving every bound from public metadata. Never
weaken the leaf agreement guard, change integer D, read hidden weights/classes/
multipliers to decide the family, or accept a prover-declared range.

For a common Split, each layer may retain its own split point. Widen its signed
key width to at least 2, so a positive low width and a positive high width exist.
Require `b <= 62`, `n*2^b < P`, and a nonempty interval
`max(1,b+ceil(log2 n)-B) <= s <= min(r_min-1,B,b-1)`.
Choose the balanced point within this interval. Both tables must fit B.

For a common Wide, set each layer's `s=min(r_min-1,B)`. If any layer has s=0,
set s=0 in every layer, keeping low-column presence uniform. Compute each layer's
`t=r_max-s+ceil(log2(max(abs(lo),abs(hi),1)))`. Widen its key width to
`b'=max(b,s+t+1)` so `k=b'-s-t` is positive. Require `b'<=62`, all table widths
within B and all RFC-0009 Lemma 11 preconditions. Width widening enlarges a public
superset; it changes neither the input integer nor its requantization. If either
family cannot meet its conditions in every layer, refuse that family. Refuse the
site when neither works. Different actual class counts above one are allowed;
one class versus several is already prohibited by RFC-0038 and remains refused.

The experiment must test this proposal, including whether it actually handles
#617; these formulas are a hypothesis, not an accepted selection rule.

## Why it could work

The local primary specifications [RFC-0006](../../docs/rfcs/RFC-0006-compiler-ir.md),
[RFC-0009](../../docs/rfcs/RFC-0009-gadgets.md),
[RFC-0038](../../docs/rfcs/RFC-0038-uniform-layers.md),
[RFC-0045](../../docs/rfcs/RFC-0045-leaf-layout.md) and
[RFC-0050](../../docs/rfcs/RFC-0050-gadget-advice-columns.md) already define the
relations and their true-integer preconditions (E0). Read roles depend on the
family and on low/class presence, rather than on every table coefficient. The
production public-only baseline observes one Wide and fifteen Split layers at
the real MLP residual with B=32; the tiny derivative also fails (E0 primary
metadata observations, no proof or performance claim).

## Expected gain

No decrease in mu, proof size or verification work is credited at 1K, 4K, 16K or
128K. Extra auxiliary columns/lookups and possibly larger tables cost work.
`python3 -I research/tools/cost_model.py --profile counted` remains unchanged;
it does not express this real model's heterogeneous layers. The target metric
is elimination of an honest public-layout refusal while preserving exact
integer semantics. Full compiled counts and G-4 must be recalculated before
production adoption. No CPU/GPU throughput projection is made.

## Security gate

| Property | Effect | Pass |
|---|---|---|
| G-4 | Existing relations only; complete actual sizes/budget must be derived after integration | unknown: finite experiment first; no acceptance credit |
| SP-3 | Same requant result; retain range, rounding, class-packing and 62-bit limits | unknown: exact/alternative-witness experiment first |
| G-5 | Scope Protocol One, already non-hiding; Protocol Two canonical selection unchanged/refused until its separate review | pass for this scope; no hiding claim |
| SP-6 | Public specification first, future independent implementations; no shared code | pass for experiment; independent implementation still required |
| Hash commitments | No change to primitive, binding or setup assumptions | pass |

Unknown rows block RFC promotion until settled within their stated scope. A
finite E0 pass will not approve a production verifier or establish G-4 for a
complete proof.

## Prior rejection check

RFC-0038 Alternatives 1 rejects layer groups because of extra weight claims and
masks. This proposal keeps one template and its common roles. Alternative 2's
dummy committed vectors are not introduced. RFC-0024/RFC-0028 choose the cheapest
per-node gadget/split; new evidence is that this may contradict RFC-0045 at one
honest site. Baseline section 7 rejects layer-by-layer GKR; no such schedule is
introduced. This is a compatibility correction needing an RFC, not a silent
reinterpretation of the accepted canonical table rule.

## Alternatives and lenses

| Seed/lens | Decision and reason |
|---|---|
| Remove the leaf guard (schedule) | rejected: unbound/incompatible stacked source claims |
| Partition layers by family (commitment) | not selected: RFC-0038 extra stacked weight claims/masks and routing work |
| Public common-family promotion (lookup/counts) | experiment: existing lemmas, extra columns/counts explicitly retained |
| One common key envelope only (range) | insufficient alone: shifts can still select Split versus Wide |
| Force Wide globally (hardware) | weakened: needless work and may fail at large shifts; promote incompatible sites only |
| New bit-decomposition requant (wild lens) | deferred: new relation and soundness argument, unnecessary before existing families tested |
| Clip scales or change D (quantization) | rejected for this task: no longer proves the fixed exported integer model |
| Exact affine numerator/shift normalization (attention/quantization) | limited: 16-bit multiplier or centered-range bound can overflow; not a general fix |
| Union reads with dummy auxiliary columns (zero knowledge) | deferred: new commitment/routing/padding obligations; RFC-0038 Alternative 2 |
| Vary public table policy (parameters) | measured baseline refusal at all legal feasible tested policies, not a fix |
| Max certificate, regen and stream lenses | no independent change: unrelated to this leaf-role conflict; full resource work remains |

## Kill criterion

Any admitted plan violates a gadget precondition, gives a different integer
output or admits a different passing auxiliary witness in the frozen domain;
or the candidate fails the frozen real-1B and synthetic reproduction cases.

## Cheapest experiment

[EXP-0029](../experiments/EXP-0029-uniform-requant/README.md), E0 exact integer
metadata/lookup audit, with public real-model metadata and seeded synthetic
cases. No GPU needed; runtime budgets, production verifier implementation,
full proof resources and independent review remain separate.

## Refutations

| Lens | Verdict | Argument |
|---|---|---|
| Soundness, ADV-P/ADV-V | weakened | Must check widening, class packing, rounding and low presence; no secret/message changes in E0, no production soundness credit |
| Prior art | survives | Reuses current lemmas; keeps the template rather than RFC-0038's rejected layer groups |
| Cost model | weakened | Correctness need, zero performance gain; tables/auxiliaries may grow |
| Feasibility | survives conditionally | Public metadata experiment feasible here; canonical production rule requires RFC and independent verifier review |

## Score

| Gain | Confidence | Cost | Priority |
|---|---|---|---|
| 0 credited performance gain | 0.3 pending experiment | l: canonical selection/RFC and independent implementation | 0; authorized correctness prerequisite |

## Decision and links

Self-triaged for a bounded experiment, not a sweep or independent review.
Experiment: EXP-0029. Issue: #617. Ledger updated 2026-10-29.
