# SUR-0015: Bounded multiplicity storage for witness regeneration

| | |
|---|---|
| Status | `current (2026-10-20)` |
| Agenda topic | `passes`: bounded CPU implementation prerequisite for #112 |
| Questions | "What is the minimum number of full-forward-equivalent passes for the witness, the stages and the opening, with checkpointing, pass fusion and streaming-friendly commitments? Which stages can share a pass?"; "Streamed leaf levels: alternatives to `c(S)` streamed rounds (hierarchical folds, partially resident trees, GPU memory tiers, NVLink spill, proof units)?" |
| Period covered | SC20 generalized histograms and current Rust/CUB primary documentation, checked 2026-10-19 UTC; Tessera at `017ab6b` |
| Produced by | `autoresearch survey passes`, 2026-10-20 |

## Method

This bounded unit addresses storage needed to finish the checkpoint/regeneration
interface, not the agenda's full distribution or pass-fusion question. Read
RFC-0018's Witness engine/Pass structure, RFC-0041's table tiles, RFC-0045's lookup
padding and RFC-0050's complete range requests, plus current `tessera-witness`
counter/source code. The baseline retains a dense `u64` count per padded table
row while replaying contributing node/attention tiles. Unsupported auxiliary
families are still explicitly refused.

Searches: Rust slice in-place sorting; CUB histogram and reduce-by-key; generalized
histogram sorting and memory. Read the Rust API's allocation/complexity contract,
CUB's ReduceByKey contract and relevant sections of Henriksen et al.'s primary
paper (abstract, global-memory inspector and sort/reduce comparison). No GPU
benchmark was reproduced. The versioned Rust 1.88 web endpoint failed; the current
stable documentation was accessible. The pinned toolchain must be tested in the
experiment. These three primary sources settle available implementation patterns;
none establishes a Tessera memory bound or protocol/2 timing guarantee.

## Findings

All numeric statements below are analytic E0; no CPU/GPU timing is claimed.

| Source | What it is | Mechanism | Numbers | Term | Applicability to Tessera | Caveats |
|---|---|---|---|---|---|---|
| [TesseraWitness26] | Current implementation and accepted tile contract | Dense counters retain canonical padding and row-0 tails | A Range(32) requires `2^32 * 8 = 32 GiB` for a dense `u64` array, even in a small model (E0) | memory, `stream` | direct: any replacement must expose exactly the same logical column | Existing refusal is honest; completing the catalog makes this storage issue actionable |
| [RustSlice26] | Standard-library API | In-place unstable key sorting without another heap allocation | Worst-case `O(M log M)` work, no measured constant (E0, source contract) | memory, `regen` | direct: sort weighted updates in clearing storage | Comparison/memory access is data dependent; a fixed public capacity is necessary but insufficient for timing ZK |
| [CubReduce26] | CUDA primitive contract | Reduce consecutive equal keys to weighted aggregates | No transferred numeric claim | memory, `regen` | adapt: exact integer run sums specify a CPU reference | CUB uses separate output/temporary storage; no CUDA dependency or GPU speed credit is proposed |
| [Henriksen20] | Generalized GPU histograms | Compare atomic accumulation, partitioned passes and sorting/reduction | No transferred timing or speedup | memory, `stream` | adapt: retain dense accumulation when its public bound is cheaper | The paper's sampled sparsity inspector depends on values and is unsuitable for selecting a secret witness's allocation/shape |

## Synthesis

Canonical multiplicities are a logical column, not permission to change or skip
committed rows. A compact physical representation must still enumerate every
public table position, including implicit zero counts and table-tree padding.
It must preserve each repeated request, per-axis padding, masked Exp keys and
the final lookup-column row-0 tail. Counts remain exact integers below Goldilocks;
no modular accumulation is allowed.

Dense storage is appropriate when the public table domain is small relative to
the public number of updates. For a wide domain and few updates, a fixed-capacity
clearing array of weighted `(row, amount)` entries can be sorted and read as the
same canonical column. Choose the representation from public domain/update
bounds only. Neither capacity, mode nor debug output may expose the secret
number of distinct/nonzero bins. Do not substitute a secret-sized map or a
sampled sparsity decision. No protocol rule, layout, encoding or transcript
changes are implied.

The literal returned-buffer size is not a peak-memory bound: the current
regenerator also retains replay inputs and temporaries. #112 must measure and
account for the complete working set of one requested tile, checkpoints and
public metadata separately. A compact counter alone does not satisfy that
acceptance criterion. Existing baseline cost-model timings remain historical
projections; this survey credits no change to `mu`, pass count or proof size.

## Candidate ideas

- Publicly bounded dense-or-sorted weighted counters, with fixed logical row
  access and clearing storage. Term: CPU working memory; credited gain zero
  until measured. First killers: a missing canonical count, a secret-dependent
  allocation/shape, unbounded updates, a count reaching the field modulus, or
  a replay/counter peak that cannot fit a public one-tile working-set bound.
- Bounded row-window recounting is a fallback when neither representation fits:
  it trades repeated replay for memory. No separate performance idea yet;
  repeated full forwards may dominate. The same-harness experiment must include
  this alternative before adopting it.

## Staleness triggers

An implemented complete witness path; a changed canonical padding/table contract;
a measured allocator result rejecting the bounds; a changed pinned sorting API;
or a protocol/2 release schedule that cannot cover the data-dependent work.

## References

Keys resolve in [bibliography.md](../bibliography.md). No E2 or E3 evidence exists.
