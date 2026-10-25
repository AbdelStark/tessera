# RFC-0057: One Proof Unit per Proof

- Status: Accepted
- Authors: @AbdelStark
- Created: 2026-10-24
- Target milestone: v0.3
- Amended by: [RFC-0068](RFC-0068-leaf-stage-zero-knowledge.md) ('What a unit costs': 18 combined rows in `tessera-protocol/2`)
- Supersedes:
  - [RFC-0018](RFC-0018-gpu-prover-runtime.md) 'Witness engine' (the checkpoints), 'Partitioning', 'Pass structure' (its 'Proof units' paragraph and the passes' count), 'Memory scheduler', 'Release schedule', 'Drawbacks' and 'Testing Strategy', wherever they name proof units: a batch is not split into units, and a request's layers are never split into layer-range units;
  - [RFC-0031](RFC-0031-commitment-id.md#the-descriptor) 'The descriptor': its `unit` is always 0;
  - [RFC-0054](RFC-0054-token-groups.md) Alternative 3 and [RFC-0056](RFC-0056-native-working-sets.md) 'The batch's form', where they rely on units or on a chunk of a longer sequence;
  - [RFC-0049](RFC-0049-weight-proximity.md) 'Motivation': its weight read is per proof;
  - [08 §3 and §5](../spec/08-performance-budget.md#3-end-to-end-targets-llama-31-405b-one-8h100-node): what a proof holds at long context, and its proof size.
- Scope: which statements one proof can hold, and the prover's planning. `tessera-protocol/1` and `/2` already describe one advice commitment per proof: no message, label, check or soundness term changes.
- Origin: [#456](https://github.com/AbdelStark/tessera/issues/456)

## Summary

RFC-0018 split a batch into proof units when it did not fit: sets of requests at short context, and layer ranges of one request at long context. Each unit had its own advice commitment. A split request's boundary residuals were committed in both adjacent units and linked by an equality claim. No document specified the messages, labels or checks of such a proof (#456).

This RFC drops units. A proof is one unit: whole requests under one session seed and one advice commitment, as RFC-0036, RFC-0014 and RFC-0019 already describe it. A batch the prover cannot hold is proven as several proofs, which nothing links.

- **Why units go.** In `tessera-protocol/1`, every memory budget at the reference statement is met in one unit: token groups (RFC-0054), windowed rounds (RFC-0055) and native working sets (RFC-0056). Units never shrink a proof. Each adds its combined rows, up to 1.14 GB, and costs time: its table side, and for a set of requests its weight fold. At the sizes that would need layer ranges, their combined rows alone break 08 §5's 4 GiB gate many times over.
- **What one proof holds.** At 405B on one 8×H100 node, under `live-64`:
  - a whole request fits in memory up to 64K tokens at the cost model's 0.2 TB of free HBM, about 32K if the multiplicity counters and the keys and values stay in HBM;
  - the proof-size gate, applied to each proof, holds a whole request up to about 24K tokens, since its opened columns grow with its committed scores.
- **What this leaves open.** A longer request, and every 128K figure of the corpus, needs a proof of a chunk of one sequence whose earlier keys and values are cached. Binding those to the tokens before them is unspecified, so the 128K projections assume a link no RFC defines. Layer-range units would not supply it: they shrink neither the proof nor the scores' commitments. [#470](https://github.com/AbdelStark/tessera/issues/470) and OQ-0057-1 own the question.

## Motivation

[#456](https://github.com/AbdelStark/tessera/issues/456) asked which budgets still need layer-range units after RFC-0054 and #452 (RFC-0055, RFC-0056), and for their messages if any does.

### The reference statement

At 08's reference statement: Llama-3.1-405B, a batch of 16K tokens at 16K of context, one 8×H100 node with 0.2 TB of free HBM and 1 TiB of host RAM, `tessera-protocol/1`, `live-64`.

| Budget | What units would have done | What meets it now |
|---|---|---|
| Leaf stage memory | Split the batch until its trees fit | Token groups, up to `2^{14}`: two regenerations ([RFC-0054](RFC-0054-token-groups.md)) |
| Reduction stages' memory | Split the layers until a stage's tables fit | Windowed rounds, one regeneration per window ([RFC-0055](RFC-0055-windowed-stages.md)); score instances from the INT8 queries and keys, 45 GB of residents ([RFC-0056](RFC-0056-native-working-sets.md)) |
| Checkpoints | Hold a layer range's residuals | Host RAM: 4 bytes × 16,384 × 127 checkpoints, 8.3 MB per token, 136 GB for the batch, against 1 TiB (08 §4) |
| Keys and values | — | 4.2 GB for the batch (RFC-0054 'Pass structure') |
| Proof size | — | About 2.8 GB: 1.18 GB of opened columns ([RFC-0011](RFC-0011-advice-commitment.md#sizes-at-the-default-parameters-16k-tokens-16k-context-405b)), at most 1.14 GB of combined rows, 0.35 GB of weight openings, and about 120 MB of the groups' messages. Within 08 §5's 4 GiB gate, above its 2 GiB target |

### What a unit costs

Each unit adds, whatever its size:
- **Proof size.** Its advice opening's combined rows, at most 17 × `2^{22}` × 16 bytes = 1.14 GB, and 18, 1.21 GB, in `tessera-protocol/2` with RFC-0068's column masks' group. The row length is the parameter set's message width, not the unit's. Its opened columns are its share of the batch's, so splitting adds nothing there.
- **Time.**
  - Its table side: its own multiplicities and table trees, `6.7×10^9` rows at 405B, about 0.5 GPU-s (RFC-0054 'Motivation').
  - For a set of requests, its weight fold, 0.24 GPU-s, since it reads every weight. A layer range reads only its layers', so its folds add nothing in total.
- **For a layer range, the boundary.** Its boundary residuals committed twice, and a linking claim with its own soundness term.

So at the reference statement no budget needs units. They could not replace a mechanism either:
- **The reduction stages.** 405B's largest holds 6.6 to 11.9 TB at 16K tokens (RFC-0056 'At 405B'). Layer ranges small enough to fit 0.2 TB number at least 33 to 60: more where a stage's tables do not stack over the layers, as stage 1's 1.17 TB of greedy relations do not. Their combined rows alone, at least 38 to 68 GB, exceed the 4 GiB gate nine to sixteen times.
- **The leaf stage.** It holds 36 to 200 TB of leaves at 40 bytes each (RFC-0054 'When groups apply'). Units would number 180 to 1,000, and at 16K of context one layer alone holds 1.2 TB.

### Beyond the reference statement

- **Memory, a whole request.**
  - **What fits.** Run on `docs/perf/v0.2`'s commitment reports, `tessera-bench`'s plans fit a whole request of 64K tokens (`T = S = 65,536`) in both profiles, the leaf stage at `G = 2^{14}`, the cap. That is the model's best case: its 0.2 TB of free HBM counts neither the multiplicity counters (19 to 42 GiB, 08 §4) nor the keys and values (17 GB at 64K). At 0.15 TB the plans fit 32K tokens and refuse 48K.
  - **What does not.** Just above 64K tokens the reduction stages, padded to `2^{17}` positions, no longer fit, and from 72K no grouping of the leaf stage does. At 128K tokens:
    - the leaf stage would need `2^{16}` groups;
    - stage 1's residents would come to about 360 GB, RFC-0056's 2.6 MB per token and 0.13 MB per key position summed at `T = S`, against 0.2 TB;
    - the checkpoints would take 1,016 GiB of the 1 TiB of host RAM, before the merge's tables and the weights' staging.
  - **Layer ranges.** They would shrink the leaf stage's and the residents' share per unit, but not the proof (below).
- **Proof size, at long context.** 08 §5 states its gate at the reference statement. This RFC applies it to each proof, at every statement. The opened columns are `τ_A` × 8 bytes per non-zero committed row (RFC-0011), and each valid score is one commitment (`docs/perf/v0.2`'s `per_score`). So the columns grow with the tokens and, through the scores, with the context. `ModelAdviceLayout`'s rows for a whole `servable` request, which reproduce RFC-0011's 953,580 at 16K tokens exactly:

  | Tokens | Non-zero rows | Opened columns | Proof, about |
  |---:|---:|---:|---:|
  | 16,384 | 953,580 | 1.18 GB | 2.8 GB |
  | 24,576 | 1,819,563 | 2.26 GB | 4.2 GB, at the 4 GiB gate |
  | 32,768 | 2,889,129 | 3.58 GB | 5.6 GB |

  - **A whole request** fits the gate up to about 24K tokens.
  - **A 16K-token chunk at 128K of context** would open about 5 to 10 GB of columns, by the chunk's position: a fit to the layout's rows, its later chunks attending more keys.
  - **The verifier's message limit.** Beyond the gate, `tv-codec`'s message limit of 8 GiB (RFC-0003) refuses a non-interactive proof of a whole request from about 46K tokens, and of the late 128K chunks.

  Units cannot help here: a layer range's columns are its share of the request's, and each unit adds combined rows. Every proof that keeps these columns under the gate is smaller: fewer tokens, or less context.
- **The 128K figures.** 08 §3's stretch row, RFC-0054's `G` at 128K, RFC-0056's 60 GB of residents and `docs/perf/v0.2`'s `μ(128K)` model a 16K-token batch at 128K of context. That is, by RFC-0056 'The batch's form', a chunk of one sequence whose earlier keys are cached. Proving such a chunk needs a binding of the cached keys and values to the tokens before it, across proofs, which no document specifies. Without it they would be unproven prover input.
- **`tessera-protocol/2`.** It keeps `g = 0` until OQ-0054-1 is resolved, so a 405B leaf stage fits only `F/(L_tok·(B + 8))`, about 90 tokens at 1K of context. Layer-range units would not change that within the gate: a 1K-token request would take about 12, and 14 GB of combined rows.

## Goals

- A decision on proof units that #132's planner, #357's batching and the verifier can build on.
- The limits of one proof, stated, with every document that names units or a chunk in agreement.

## Non-Goals

- **Chunked requests.** A request longer than one proof holds, and the 128K targets: [#470](https://github.com/AbdelStark/tessera/issues/470) (OQ-0057-1).
- **Several requests in one proof.** That is one unit holding several requests, [#357](https://github.com/AbdelStark/tessera/issues/357) (v0.4).
- **Multi-node proving.** As in RFC-0054.
- **Layer-range partitioning across one node's GPUs** ([RFC-0018](RFC-0018-gpu-prover-runtime.md#partitioning) 'Partitioning', #172). It splits one advice commitment into segments, one per GPU (RFC-0011), and is not a unit. It is unchanged.

## Proposed Design

- **One unit.** A proof holds one proof unit: whole requests, with one session seed, one advice commitment and one stage structure. RFC-0036's messages, RFC-0014's transcript schedule and RFC-0019's checks are that unit's, as written. RFC-0031's descriptor keeps its `unit` field, always 0, so no digest changes.
- **No layer-range units.** A request's layers are never split across proofs or across advice commitments. Within the proof, RFC-0018 'Partitioning' assigns each GPU a range of layers and its advice segment.
- **Whole requests.** A request is proven whole, in one proof, from its first token: its statement and its anchor cover its token stream (RFC-0007, RFC-0015). A proof of part of a request, with earlier tokens' keys and values taken as given, is not a `tessera-protocol/1` statement until #470 specifies one.
- **The planner** ([RFC-0018](RFC-0018-gpu-prover-runtime.md#memory-scheduler) 'Memory scheduler', #132). From `ProverConfig.memory_budget` and before P1, it chooses the leaf stage's token groups (RFC-0054 'When groups apply') and each reduction stage's windows (RFC-0055), not units. A plan that no grouping and no windows make feasible fails before `Round1` with `ERR-RES-MEMORY`.
- **A batch the prover cannot hold.** The operator proves fewer requests per proof. Each proof is a complete session, with its own statement, seed and advice commitment, and nothing links them.
  - **What it costs.** Each proof pays its own weight openings, about 0.35 GB, and its own blinding rows and columns of the epoch's weight budget, about 670 proofs per epoch (RFC-0012 'Budget state machine').
  - **The knob.** The batch size takes over from units as the planner's lever (OQ-0056-2).
- **The passes** run once per proof: RFC-0018's pass table, its weight fold included, reads "per proof" where it read "per proof unit".

## Alternatives Considered

1. **Layer-range units with boundary-residual linking** (RFC-0018 as written, and #456's first branch). It would need:
   - per-unit `Round1` commitments and openings under a unit-indexed label scope, with the seed and stage structure shared;
   - each boundary residual committed in both units, and an equality claim at a random point opened in both;
   - a soundness term per boundary, the verifier's checks and reasons, and, for zero knowledge, each unit's masks and blinding rows.

   **Rejected.** At the reference statement it buys no budget. Beyond it, it shrinks memory per unit but neither the proof nor the scores' commitments, and each unit adds combined rows.
2. **Sets of requests as units in one proof**, several advice commitments under one session. It pays the same per-unit costs for no budget. Rejected; a proof of several requests holds them in one unit (#357).
3. **Token-range chunks of one request**, linked by their cached keys and values. This is the partition that would serve both memory and proof size at long context: each chunk's columns shrink with its tokens. It needs a cross-proof binding with its own soundness argument. Deferred to #470, since it is a new statement rather than a unit of today's.
4. **A batch-shared table side across units** ([IDEA-0027](../../research/ideas/IDEA-0027-batch-shared-table-side-units.md)). It would remove one per-unit cost, but units go. Moot.

## Drawbacks

- **Long requests.** A 405B request above about 24K tokens has no proof within the gate on one node, and above 64K none at all, until #470 lands.
- **The 128K figures** remain projections for a statement that is not yet defined.
- **Several proofs** cost their weight openings and their share of the epoch's budget once each, where units would have shared them.

## Migration / Rollout

v0.3, with this RFC:
- **RFC-0018.** Its amendment line names this RFC. 'Witness engine', 'Partitioning', 'Pass structure', 'Memory scheduler', 'Release schedule', 'Drawbacks' and 'Testing Strategy' say "per proof". 'Pass structure' replaces its 'Proof units' paragraph with a pointer here, and 'Testing Strategy' drops the proof-unit test.
- **RFC-0031.** `unit` is always 0.
- **RFC-0049.** Its weight read is per proof.
- **RFC-0054 and RFC-0056.** Alternative 3 no longer says RFC-0018 keeps units. 'The batch's form' says a chunk needs #470.
- **08 §3 and §5.** The 128K row assumes #470. §5's opened columns are RFC-0011's 1.18 GB, and its total about 2.8 GB.
- **#132.** Its criterion "splitting one request into two units verifies; a tampered boundary residual is rejected" is withdrawn. The planner's grouping and windowing are #134's and #459's.
- **`SPEC.md`.** The RFC index lists this RFC. OQ-0018-1, which RFC-0054 resolved, leaves the open questions, and OQ-0057-1 joins them.
- **No message, label, check or vector changes:** both workspaces already prove and verify one unit.

## Testing Strategy

- **Nothing new to test for units.** RFC-0018's proof-unit test is dropped with them.
- **The planner's refusal** is RFC-0054's and RFC-0055's: under a forced memory budget, a plan that no grouping or window makes feasible fails before `Round1` with `ERR-RES-MEMORY` (#134, #459).
- **The limits.** The 64K and 32K figures are `tessera-bench`'s `leaf_plan` and `stage_plan` on the committed v0.2 reports. The proof-size rows are `ModelAdviceLayout`'s, exact and seconds to compute. #470 brings a proof-size model that holds them in a test.

## Open Questions

- **OQ-0057-1: requests longer than one proof** ([#470](https://github.com/AbdelStark/tessera/issues/470)). Chunks linked by their cached keys and values, a long-context proof-size budget, or restated targets. Owner: @AbdelStark. Resolve before v1.0's 128K target.

## References

- [#456](https://github.com/AbdelStark/tessera/issues/456), [#470](https://github.com/AbdelStark/tessera/issues/470), [#132](https://github.com/AbdelStark/tessera/issues/132), [#357](https://github.com/AbdelStark/tessera/issues/357), [#452](https://github.com/AbdelStark/tessera/issues/452).
- RFC-0007, RFC-0011, RFC-0012, RFC-0014, RFC-0015, RFC-0018, RFC-0019, RFC-0031, RFC-0036, RFC-0049, RFC-0054, RFC-0055, RFC-0056; 08 §3, §4 and §5.
- [IDEA-0027](../../research/ideas/IDEA-0027-batch-shared-table-side-units.md): the table side per unit.
