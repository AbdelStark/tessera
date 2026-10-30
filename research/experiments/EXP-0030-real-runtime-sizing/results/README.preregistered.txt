# EXP-0030: Complete real-model runtime sizing

| | |
|---|---|
| Status | `planned` |
| Idea | [IDEA-0041](../../ideas/IDEA-0041-uniform-requant.md), already triaged and finite-E0 validated; follow-up for [#627](https://github.com/AbdelStark/tessera/issues/627) / #617 |
| Evidence level | E0: exact public geometry and charged planner bytes, not measured resident memory |
| Hardware | Apple M1 Max, 64 GiB unified memory; CPU metadata only, no timing criterion |
| Pre-registered on | 2026-10-30 |

## Hypothesis

The complete public runtime of the unchanged fixed 1B integer candidate, with the
previously studied common-Wide metadata supplied in scratch, has a charged CPU
resident floor above 64 GiB at every tested table policy and sequence length,
even before stage grids and after eliminating all lookup-tree cells. Exact
recounts can establish this without allocating a witness or any dense table.

## Success criterion

`validated` only if all the following frozen criteria pass:

1. Pin production base `51fbd9353a42259e0fdc07973aae6f3857a90f6a`, the public
   real-1B ClaimList SHA-256 `af7d31a9e853cde6fdf4e2a8a195e6d312ae3ae6fc2fdb2e8700cb963ab3d5aa`,
   and EXP-0029 diagnostic plan SHA-256
   `ead57bd4b08b94e5ba4e44f33264684a9772fba68b01c000440404259062c279`.
   The unmodified baseline must reproduce the exact RFC-0045 read-role refusal
   at B=30,31,32. No real weight, statistics, token, seed or witness artifact is opened.
2. For every pair B in {30,31,32}, n in {2,512,2048}, derive the entire
   exact-greedy runtime with prompt n-1 and one continuation. Supply only the
   EXP-0029 fixed public MLP residual plans. Finalize all table ids by first use
   from the lowering's base catalog, rebuild the full ordered read cache and
   retain every leaf/native/advice guard. No graph or integer-model change.
3. Save all padded tensor shapes, complete runtime lookup instances and their
   used table kinds/row counts, split-value count, advice placement dimensions,
   native stage/window shapes, and every weight working-set input. Exercise
   the original PassPlan new/staged/for_budget/check functions through
   scratch-only visibility access. Include exact decisions for 16,64,256,1024 GiB.
4. Save the resident charge and the strict base floor with lookup cells set to
   zero. This latter value is a lower bound for every legal leaf grouping,
   not a claim that zero lookup cells is a valid proof layout. It must exceed
   64 GiB for every real case. Grouping must never remove table, tensor,
   advice or weight terms. No budget, 64-byte coefficient or planner rule changes.
5. A separate Python arbitrary-precision audit must recount tensor padding,
   lookup/table sums, advice packing, weight working-set terms, the base floor,
   peak aggregation and saved budget decisions from public records. Original
   WindowMemory stage charges remain source-pinned producer values; this audit
   does not independently validate their internal allocation model. Include a
   hand-computed Cells(1,2,3,4), weight_bytes=1024 control: 1664 bytes fits,
   1663 refuses. Include the unchanged canonical tiny-2-servable model as a
   control with no candidate promotion.
6. Catch at least eight isolated saved-record accounting corruptions, including
   a missing tensor, unpadded lookup count, missing runtime greedy lookup,
   omitted table rows, shortened advice storage, omitted weight fold term,
   removed table charge under grouping, changed cell coefficient and a false
   budget decision. Preserve passing original/restored controls and first failures.
7. Verify that all production leaf, native, advice and planner algorithm bytes
   remain unchanged in the repository; scratch modifications are limited to
   metadata injection/visibility, reusing the frozen experimental adapter.
   Save source hashes and exact patch receipts. No production/verifier source,
   lockfile, canonical usable claim artifact, witness or proof is changed.

`refuted` if a correct complete real case has that zero-lookup base floor at
or below 64 GiB, or the stated grouping lower-bound argument is false.
`inconclusive` if any required metadata case cannot be derived, the independent
recount or controls disagree, a meaningful corruption survives, or source
preservation cannot be demonstrated. Keep the failure; do not shrink the domain
or weaken a guard after seeing it.

## Method

Use the same Rust adapter for baseline and candidate; baseline failure precedes
metadata injection. Reuse only EXP-0029's fixed public plans; this is not an
implementation or adoption of draft RFC-0076. Unchanged runtime, leaf, native,
advice and stage metadata APIs drive the complete measurement. Scratch exposes
Selection.reads and the existing private PassPlan/weight-memory entry points;
function bodies and all guards stay identical. The input is public metadata
from the pinned actual model, plus the canonical tiny control and exact hand
case. No secret synthetic/real tensors are evaluated; no random seed is needed.

Counts, integer equalities and original planner decisions are the measurements.
No warm-ups, throughput, RSS, GPU projection or claimed soundness bits are used.
Complete soundness-budget evaluation, original opening groups, independent
verifier integration, full fidelity and actual proof acceptance remain owed.

## Reproduction

After this registration is committed and verified, commands from repository root:

```sh
python3 -I research/experiments/EXP-0030-real-runtime-sizing/run.py
python3 -I research/experiments/EXP-0030-real-runtime-sizing/audit.py
python3 -I research/tools/check_kb.py
cargo xtask lint
```

The runner must verify the committed frozen registration before building code.
Record the actual Rust/Python/OS environment and immutable input/source hashes.
Use a private scratch checkout and a separate Cargo target directory so no
experimental build can contaminate ordinary validation. Keep all result logs.

## Results

Not run. Raw outputs will be under `results/`; no memory-floor number is credited yet.

## Analysis

This measures a frozen candidate's charged metadata geometry. It neither makes
that candidate canonical nor proves every allocation bound. A planner charge is
not observed RSS. Any counterfactual grouping floor must remain labeled as such.

## Verdict

Pending; `planned`. No production capability, G-4 pass or performance credit.

## Follow-ups

If validated, use the exact dominant charged terms to choose the next prover
memory/scheduling unit or hardware requirement; keep the separate RFC review
and complete-budget obligations. A measured whole proof is still required.
An H100 run would address a separately implemented GPU backend; no CPU byte
count predicts its speed or establishes GPU feasibility.
