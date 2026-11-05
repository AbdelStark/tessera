# EXP-0031: Bounded canonical grouped source feasibility

| | |
|---|---|
| Status | `done: validated` |
| Idea | [IDEA-0042](../../ideas/IDEA-0042-bounded-grouped-source.md), #687 |
| Evidence level | E0 public geometry/value equivalence; E1 CPU allocation counts, no timing |
| Hardware | Apple M1 Max, 64 GiB unified memory; existing immutable model jobs remain live |
| Pre-registered on | 2026-11-05 |

## Hypothesis

A stateless canonical synthetic source can regenerate the held lookup columns,
forest leaves and original polynomial tables of the frozen finite grouped cases
without retaining every source column, while matching current resident grouped
constructors and both merge kernels exactly. Its finite source/merge allocation
peaks can be bounded from public geometry with every observed owner explicit.

## Success criterion

`validated` means every item below passes, only in this finite prototype scope.
It never means production grouped admission, complete `ModelWitness` replacement,
all-input soundness, real-model acceptance, fidelity or whole-proof memory/time.

1. Pin production commit `646f7addc7bc737f2fbb8970c865df8333877093`, Rust 1.95,
   and all source files used by the reference. Verify the committed frozen plan's
   SHA-256 before any harness code. No real model, tokens, statistics or private
   artifact is opened; no original process, binary, environment or input changes.
2. Use synthetic fixture lengths `[1]`, `[3]`, `[5]`, `[2,1]`, `[3,2]`,
   `[5,1,3]`, with real layer counts respectively `2,3,5,2,3,5`. Slots use the
   next power of two of the longest request and a padded request axis. Test
   group bit pairs `(g_T,g_L) = (0,1),(1,0),(1,1),(2,1)` and forest heights
   `1,2`. Every geometrically legal combination runs; an impossible token/layer
   split is an explicit negative case, never a silent skip. Include key-only
   and key/value tables, multiple tree heights, token-restricted and no-token
   reads, held channel roles, prologue/epilogue subset-zero roles and stacked
   forest query/layer bits. Enumerate every group and emitted row.
3. The reference materializes seeded source tensors and calls the pinned current
   resident grouped column and forest-share constructors. The candidate computes
   source entries from descriptor/index requests, without full source arrays.
   Compare all table IDs, lengths, ordered keys/values, row-zero padding, forest
   entries, original indices and split/free/full coordinate transport. Count
   callbacks and verify public request domains. Synthetic seed is
   `0x5eed0031`; generation is deterministic integer SplitMix64 with explicitly
   bounded table domains. A hand-computed one-variable table `[3,5]`, token
   subset selection and non-token held-role control must reproduce first.
4. Regenerate original merged polynomials and compare the current in-memory and
   two-pass merge proof messages/final points/evaluations under identical
   deterministic test transcripts, for every applicable fixture and every legal
   cut `0..=largest_table_rounds`. Claims include empty group-variable sets,
   different polynomial dimensions, repeated point classes and retained forest
   claims. Compare against the materialized reference; do not compare only two
   copies of the candidate. Explicitly meter original table, permutation,
   weights/equality, folded-state, callback-block and retained-proof lifetimes.
5. Count actual requested allocations with borrowed reference inputs outside
   the counting scope and exclusions reported. Measure baseline source
   construction separately. Candidate source/merge measured peaks must never
   exceed their public prototype bills. In at least one admitted multigroup
   fixture, candidate maximum simultaneously held **source payload** must be
   strictly below baseline retained full source payload; no whole-proof ratio
   is inferred. Charge vector capacity/growth and earlier outputs, not length
   alone. Record all source, callback, layout and retained owners. Never price
   an unknown production owner as zero. Give exact arithmetic/addressability
   and one-byte-under prototype-budget refusals before candidate payload work.
6. An independent Python integer audit reads saved public shape/accounting and
   index records and recounts slot/group restrictions, padding, forest indices,
   payload bytes and phase maxima. It must not import candidate calculation
   functions or treat a producer's final verdict as its oracle. A complete
   P1-P5 ownership inventory must identify which production obligations this
   prototype does not implement. Both production grouped physical guards stay
   byte-identical; no complete route is admitted by the experiment.
7. Catch these twelve isolated, buildable physical candidate/audit faults on
   assertions: swapped token/layer bits; omitted request-slot gap; missing row-0
   padding; no-token role repeated in every token subset; misplaced
   prologue/epilogue layer subset; wrong forest query bits; wrong forest layer
   bits; wrong split coordinate order; omitted earlier-group output overlap;
   omitted original P2b table beside its copy; omitted nested metadata capacity;
   wrapping size arithmetic. A compile failure is not detection. Require passing
   original and final restored controls, exact source restoration and saved
   first-failure evidence. Each fault must violate the relevant registered check.
8. Keep production and verifier sources unchanged. The reference may use a
   separate scratch checkout with visibility-only exposure of existing grouped
   constructors; every overlay edit and source hash must be recorded, and the
   candidate must not replace the oracle. No behavior patch, private security
   patch, changed feature guard, unsafe code, new verifier dependency or secret
   access exemption is allowed. Prototype-derived buffers use clearing storage.
9. Supply an E0 all-input argument for the proposed abstract source contract:
   interleaved bit insertion/restriction are inverse on the declared domain;
   held-role selection and padding retain each canonical obligation exactly;
   forest query/layer insertion and split/merged point transport preserve the
   original polynomial evaluations; regenerated values refer to the same
   canonical source identity and full index; public bounds cover simultaneous
   owners by checked sums/maxima, with unknown owners infeasible. State the
   canonical source callback's assumptions explicitly. This does not prove a
   future gadget/checkpoint adapter meets them. An unresolved in-scope soundness
   or uniqueness contract gate prevents RFC promotion even if finite tests pass.

`refuted` means a reproducible mismatch, escaped registered fault, underpriced
allocation or forbidden ownership/secret-dependent decision in this candidate.
`inconclusive` means environmental/tooling inability or an unimplemented required
case prevents judgment. Retain failed runs; do not drop cases, loosen budgets or
count compilation errors as killed mutants. A necessary method amendment is
dated, frozen and committed before revised experiment code or measurements.

## Method

An isolated Rust workspace under this directory uses pinned prover crates and
public synthetic fixtures. Reference and candidate run in the same harness.
Reference-only visibility overlay lives outside both production workspaces.
Candidate source values are deterministic from public synthetic identifiers;
results contain public geometry, counts, digests and verdicts, not payload dumps.
Run correctness/allocation controls serially in a dedicated Cargo target. No
wall-clock speed experiment, warm-up cohort, GPU estimate or repeated model run.

The merge comparison uses a deterministic allocation-free test transcript. Its
hash/reduction fixture is not the production TCE/Fiat-Shamir implementation and
establishes no challenge-uniformity or NI/Live acceptance claim. Both current
merge kernels must emit identical labels/messages/points under that same fixture.

Review added two extra controls beyond the twelve registered faults: an eager
unrestricted-original materialization during group preparation must fail its
phase counter even when outputs agree. A component message-dimension declaration below any public tree height must fail the independent audit even if allocation observations fit. A selected block with no group variables
is legitimately whole and remains charged. The counter tracks calls to this
prototype's unrestricted builder, not arbitrary work in a future factory.

The output bills and phase inventory are experimental contracts. Later production
work still owes canonical gadget/tile integration, checkpoint/KV replay and its
scratch, original opening targets, table/count/cache ownership, every P1-P5 and
publication lifetime, public shape authentication and NI/Live independent
acceptance with exact tamper refusals. Protocol/2 remains ungrouped.

## Reproduction

Before code exists, verify registration and repository consistency:

```sh
git show HEAD:research/experiments/EXP-0031-bounded-grouped-source/results/README.preregistered.txt
python3 -I research/tools/check_kb.py
cargo xtask lint
```

From this study's commit, use a fresh scratch directory and Python 3.13+ with
`blake3==1.0.11` for the independent audit and mutation runner:

```sh
python3 -B -I research/experiments/EXP-0031-bounded-grouped-source/run.py prepare /tmp/exp0031-reproduce
python3 -B -I research/experiments/EXP-0031-bounded-grouped-source/run.py build /tmp/exp0031-reproduce
python3 -B -I research/experiments/EXP-0031-bounded-grouped-source/run.py source /tmp/exp0031-reproduce
python3 -B -I research/experiments/EXP-0031-bounded-grouped-source/run.py merge /tmp/exp0031-reproduce
python3 -B -I research/experiments/EXP-0031-bounded-grouped-source/audit.py /tmp/exp0031-reproduce/source-partial.json /tmp/exp0031-reproduce/audit.json --merge /tmp/exp0031-reproduce/merge-partial.json
python3 -B -I research/experiments/EXP-0031-bounded-grouped-source/mutate.py /tmp/exp0031-reproduce
python3 -B -I research/experiments/EXP-0031-bounded-grouped-source/run.py clippy /tmp/exp0031-reproduce
```

The runner refuses a changed registration or production/verifier source. It
archives the pinned production base outside the repository, applies only the
recorded reference visibility/doc exposure, and copies the standalone harness
there. It never patches production files in this worktree. The committed lock
file and dedicated target isolate builds. Receipts retain each command/log;
repeated stage outputs are archived rather than discarded.

## Results

| Check | Observed result | Evidence |
|---|---|---|
| Source fixtures | 42 legal cases, 164 groups, 43,372 index records; six impossible split cases explicitly refused | E0 finite coverage, current resident constructors and independent integer/value-digest audit |
| Source payload | Largest group source payload strictly below retained full source payload in 36 of 42 cases | E0 exact public payload counts; not a whole-proof ratio |
| Requested allocations | Every measured source/input/table/P2b peak fits its public prototype composition; exact and one-byte-under source allowances checked before payload work | E1 CPU requested-allocation observations, including baseline source construction |
| Merge | 42 cases, all 285 legal cuts; complete round messages, evaluations, points and test-transcript digests equal the materialized in-memory reference | E0 finite equality over current kernels |
| P2b regeneration | Each original polynomial regenerated exactly once at each cut; no unrestricted-builder call during group preparation | E0 counters and phase regression control |
| Faults | All twelve registered compiled faults caught; two extra phase/geometry faults caught; original and restored controls pass | E0 finite negative controls; no compile failure counted |

Measured on Apple M1 Max, 64 GiB, macOS 15.6.1, Rust 1.95.0; exact environment
is in [environment.json](results/environment.json). No experiment code existed
in registration commit `ede585df1fc290df27c32ba2a7b4a6672a69cc74`; the frozen
plan's SHA-256 remains `972efa1bdf13f49e271ab58f41edde3a6b54d17605d00e4b9d39bd8e02bea2c0`.

Raw source and merge records are compressed under `results/`. The independent
[audit](results/audit.json), [fault receipts](results/mutations.json),
[source provenance](results/provenance.json) and
[reference overlay](results/reference-overlay.json) identify the exact controls.
The final restored controls supply the published records. Earlier partial runs
remain in the private scratch evidence; the initial public fixture-order failure
is retained in [its log](results/initial-source-order-failure.log).

## Analysis

The frozen finite criteria pass, together with the conditional all-input
[contract argument](contract.md) and [complete phase inventory](phase-inventory.md).
The synthetic callbacks are not a production checkpoint/gadget factory. Layer
maps here are hand-built layout fixtures, not complete compiled model graphs.
Future factory/cache/replay and native handoffs remain unimplemented and cannot
be priced as zero by this study.

Four faults are detected by the independent Python audit, including a slot-gap
fault that affects the shared synthetic value factory and would agree between
the two Rust arms. Identity/schema controls test source bookkeeping as well as
value generation; they are not fourteen cryptographic forgery demonstrations.
The initial site-order constructor error was corrected without changing the
registered domains. Review then replaced full-polynomial group split evaluation
with restricted-block evaluation and added the eager-materialization control.
An empty group-variable set still keeps its whole source, as required.

The Python audit independently reconstructs public indices, values/digests,
shapes, claim/cut coverage and composition arithmetic. Rust ABI sizes, allocator
observations and pinned production MergeMemory/GroupLayout component totals are
producer inputs; this is not a second implementation of all their internals.
Raw stage records say `not judged` because each stage alone is insufficient; the
verdict here joins both stages, the independent audit, faults and contract review.

Requested allocation counts exclude stack, allocator bookkeeping, runtime and
the explicitly borrowed/materialized oracle and reporting allocations. Production
phase composition must add all real owners. Existing model jobs prevent a
controlled timing study, which is outside scope. No H100, RSS, real-model,
forward-equivalent, fidelity or whole-proof speed/memory gain is credited.

## Verdict

`validated` for the registered finite E0/E1 source/merge feasibility and abstract
contract scope only. No complete factory, production admission, independent
real-model acceptance or performance qualification is established.

## Follow-ups

Accepted [RFC-0077](../../../docs/rfcs/RFC-0077-bounded-grouped-checkpoint-sources.md)
records the canonical source contract and full unknown-owner/refusal rule. Its
security argument preserves the existing accepted relation and is conditional
on a canonical factory; the implementation must establish that contract.
Do not promote this prototype as production-ready.
Accept the runtime amendment before a separate production admission change.
Real-model fidelity, independent proof acceptance and whole-process memory/time
remain obligations of #617/#629/#641. An H100 run would assess device storage and
throughput only after production correctness gates pass.
