# EXP-0031: Bounded canonical grouped source feasibility

| | |
|---|---|
| Status | `planned` |
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

The implementation will add exact build/run/audit commands here after verifying
the committed frozen plan. Build targets must be unique to this experiment.
Record OS, CPU, `rustc --version`, Python version, source/lock hashes and UTC time.

## Results

Not run. No experiment code exists at registration. Raw receipts will live under
`results/`; the frozen registration remains unchanged.

## Analysis

Not judged. Finite synthetic equality cannot establish all-input ModelWitness
correctness or an accepted physical route. Requested allocation counts exclude
stack, allocator bookkeeping, runtime and any borrowed input documented outside
the counter. Existing model jobs prevent a controlled timing study, which is
outside scope. No H100 or whole-model extrapolation is permitted.

## Verdict

Pending: `planned`. No validated design, production implementation or gain claim.

## Follow-ups

If the finite criteria and all-input abstract contract argument pass, write a
Draft RFC with that canonical source contract and the full unknown-owner/refusal
rule; separate future implementation obligations from settled design arguments.
Unresolved in-scope security gates block promotion. Do not promote this prototype
as production-ready.
Accept the runtime amendment before a separate production admission change.
Real-model fidelity, independent proof acceptance and whole-process memory/time
remain obligations of #617/#629/#641. An H100 run would assess device storage and
throughput only after production correctness gates pass.
