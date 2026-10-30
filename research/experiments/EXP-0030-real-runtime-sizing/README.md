# EXP-0030: Complete real-model runtime sizing

| | |
|---|---|
| Status | `done: validated` |
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

**E0, primary, charged metadata quantities.** All nine real cases and the
unchanged tiny control completed. Every baseline real case reproduced the exact
leaf refusal before metadata injection. All algorithm-source restoration and
untouched-guard checks passed. The separate integer audit agrees on every saved
case and catches ten isolated accounting corruptions; original/restored controls
pass. The raw JSON is losslessly compressed (30,954,860 to 1,201,273 bytes).

Each row below applies identically to B=30,31,32 on this fixed candidate:

| Positions | Tensor cells | Lookup-tree cells | Used table rows | Advice cells | Zero-lookup base floor (bytes) | Resident plan peak (bytes) |
|---|---:|---:|---:|---:|---:|---:|
| 2 | 1,514,951,816 | 3,434,752 | 5,536,481,870 | 36,733,714,432 | 2,814,533,396,320 | 2,814,903,721,648 |
| 512 | 2,961,852,416 | 962,854,912 | 5,536,481,870 | 36,796,628,992 | 2,911,161,566,560 | 3,062,420,989,584 |
| 2,048 | 10,537,656,320 | 5,662,834,688 | 5,536,481,870 | 49,073,356,800 | 4,181,723,596,128 | 5,143,204,545,168 |

The original planner returns exactly `ERR-RES-MEMORY` at every tested real
budget: 16, 64, 256 and 1,024 GiB. The counterfactual zero-lookup floors are
2.559803, 2.647686 and 3.803255 TiB at 2, 512 and 2,048 positions, respectively.
The actual unwindowed charged peaks are 2.560140, 2.785256 and 4.677717 TiB.
These are byte counts divided by 2^40, not measured resident memory.

At two positions, NI-100's advice layout has 8,758 nonzero rows, each charged at
its full width of 4,194,304 field cells: 36,733,714,432 advice cells. At 64 bytes
per cell this contributes 2,350,957,723,648 bytes to the base charge. Table rows
contribute 354,334,839,680 bytes, graph tensors 96,956,916,224 bytes, and the
separate original weight working-set charge is 12,283,916,768 bytes. The terms
sum exactly to the zero-lookup floor of 2,814,533,396,320 bytes.

The tiny canonical control was also evaluated with the *named NI-100 layout*:
its zero-lookup floor is 465,528,360,736 bytes, and its original planner fits
1,024 GiB. This is a layout/control observation; it is not a tiny proof run.
The hand case has the exact 1,664-byte boundary required by registration.

Evidence: [counts](results/counts.json.gz), [independent audit and corruptions](results/audit.json),
[execution/source receipt](results/run-receipt.json), [environment/input hashes](results/environment.json),
and [raw run](results/run.log). The first compile exposed two private metadata
methods; the following setup had a wrong module path. Both failures are retained
([compile](results/initial-build-run.log), [setup](results/initial-visibility-setup.log)).
Adding scratch visibility resolved them; no criterion, guard or algorithm changed.
The reused scratch target is separate from ordinary validation targets.

## Analysis

The adapter reads only the already-public fixed-alpha=1/2 diagnostic candidate's
ClaimList and EXP-0029 plans. Its underlying 1B model is not yet alpha-selected
or fidelity-qualified. These results must be recomputed for a final chosen D.
They do not describe every 1B/8B model or the projected 405B GPU backend.

`ModelAdviceLayout::runtime` executes the same `allocated_runtime_rows`
counter used by the early batch-advice guard, then checks the constructed
layout against that counter. Thus supplying the candidate selection directly
does not omit its allocation bound; the ordinary early wrapper would reselect
the unchanged baseline and cannot validate unadopted candidate catalogs.

This is a complete *current CPU planner* cell count at the frozen public
candidate metadata, including exact-greedy runtime requests, native stages,
canonical first-use table finalization and every original weight term. It is
not a complete soundness budget: opening groups and the full runtime calculator,
independent verifier implementation and acceptance remain owed. It does not
adopt draft RFC-0076 or produce a canonical usable candidate ClaimList.

The grouping lower bound is strict: the existing `PassPlan::grouped` replaces
only lookup cells; stage windows leave the cell base unchanged. Even giving
that term the impossible ideal value zero and charging no stage working memory
cannot reach any tested real budget. Thus the present grouping/window choices
cannot make this frozen candidate fit the 64 GiB host's current CPU plan.

This is **not** an observed out-of-memory event or a physical impossibility
result. The original actual proof attempt still failed at its leaf-layout guard,
before this experimental candidate's resource plan. The 64-byte cell reserve
and full advice-row charges are conservative allocations in the current CPU
contract. A different, validated memory schedule may require much less memory.
Original public table-generator cores used by lowering are retained unchanged;
the adapter allocates no full logical table or witness. WindowMemory's internal
allocation formula remains source-pinned producer evidence, not independently
proved by the Python recount.

## Verdict

`validated` for the frozen E0 resource hypothesis: all required cases and
recounts pass, all ten accounting corruptions are caught, and every real
zero-lookup floor exceeds 64 GiB. No production capability, complete G-4,
physical peak, performance or hiding credit follows.

## Follow-ups

The next real-model implementation decision is a memory plan that addresses
advice rows, table trees and resident tensors together, validated on complete
NI-100 proofs. Current grouping/windows alone are insufficient. Retain canonical
artifacts, every opening/range/read check and an actual-count soundness budget;
any changed parameter or canonical layout needs the repository's review/RFC
process. Keep the independent RFC-0076 acceptance and implementation obligations.
Do not repeat the unchanged long real proof attempt expecting table-policy or
leaf grouping choices alone to solve this planner refusal.

An H100 run requires an implemented GPU backend and its separate resource model;
these CPU charges provide neither GPU feasibility nor a speed projection.
