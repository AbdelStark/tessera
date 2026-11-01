# RFC-0018: Witness Engine and GPU Prover Runtime

- Status: Accepted
- Authors: @AbdelStark
- Created: 2026-10-15
- Target milestone: v0.2 (CPU witness engine), v0.3 (CPU prover), v0.5 (GPU prover), v1.0 (405B)
- Amended by: [RFC-0034](RFC-0034-causal-blocks.md) ('Kernels': constant subtrees in closed form; 'Testing Strategy'); [RFC-0054](RFC-0054-token-groups.md) ('Pass structure': P2, 'Leaf levels', 'Proof units' and 'Fusion' for the leaf stage; Alternative 2; OQ-0018-1; 'Drawbacks'; 'Testing Strategy'); [RFC-0055](RFC-0055-windowed-stages.md) ('Pass structure': P3 and P3b for a reduction stage larger than memory; 'Drawbacks'); [RFC-0056](RFC-0056-native-working-sets.md) ('Pass structure': what stage 1 holds for `SoftmaxRow`'s score instances); [RFC-0057](RFC-0057-one-proof-unit.md) ('Witness engine', 'Partitioning', 'Pass structure', 'Memory scheduler', 'Release schedule', 'Drawbacks', 'Testing Strategy': one proof unit per proof, no layer-range units); [RFC-0013](RFC-0013-zero-knowledge.md#masked-extensions) ('Kernels', in `tessera-protocol/2`: round kernels up to degree 10 at a reduction stage, [#565](https://github.com/AbdelStark/tessera/issues/565)); [RFC-0004](RFC-0004-hashing-and-transcript.md#hash-function-and-domains) ('Backend interface', 'Crash consistency', 'Determinism': per-proof stream domains `advice-masks`, `prover-share` and `ni-session`, heap-constructed readers, and INV-ARCH-2's stream rejection exception, [#567](https://github.com/AbdelStark/tessera/issues/567)); draft [RFC-0073](RFC-0073-weight-mask-expansion.md) ('Crash consistency': the weights' masks and blinding rows come from the epoch's `masks_seed`; in effect when it is accepted)

## Summary

The prover has two backends behind one interface: a CPU reference backend, which is also the bit-exact test oracle, and a CUDA backend for H100, H200 and B200 nodes. The witness engine executes `D` exactly: INT8 GEMMs with INT32 accumulation on tensor cores, integer gadget arithmetic on CUDA cores. Field-heavy work (encoding, hashing, sumcheck rounds, fraction trees) runs in custom kernels; folds that are matrix products run as limb-decomposed INT8 GEMMs. Work is partitioned across the GPUs of one node by layer range. Every proof is a fixed sequence of passes over a regenerated witness; this RFC fixes that pass structure, how passes are counted and fused, crash consistency, and the determinism rules. A leaf stage that exceeds memory is proven in token groups ([RFC-0054](RFC-0054-token-groups.md)), and a reduction stage that does in windowed rounds ([RFC-0055](RFC-0055-windowed-stages.md)).

## Motivation

Source principles 8–10 (§6): tile everything, recompute instead of storing, and run matrix-product-shaped work on tensor cores. Source §7 names the phases (witness pass, seed, sumchecks, openings) and §5 the unit costs the kernels must hit. Two gaps found while specifying the corpus land here: regeneration passes cost native compute the source's model omits (RISK-9), and the leaf-adjacent levels of fraction and product trees exceed device memory (RISK-10).

## Goals

- One `ProverBackend` interface; CPU and GPU backends produce byte-identical messages (INV-ARCH-1).
- Unit costs within the targets of [08 §2](../spec/08-performance-budget.md#2-unit-cost-targets).
- An explicit, measured count of regeneration passes per proof.
- Restartable phases with no change to sent messages.

## Non-Goals

- Multi-node proving (NVL72 scaling is future work; the source's minimum is one 8-GPU node).
- Serving-stack integration (NG-8); the witness engine can serve exact-mode demonstrations but is not a production server.
- B300 support (NG-10).

## Proposed Design

### Backend interface

```rust
// The phase boundary used by the existing RFC-0036 session machine.
pub use tessera_protocol::ProverEngine as ProverBackend;
```

Its concrete methods are `commit`, `prover_share`, `prove`, `answer` and
`discard`, with the messages and query shape specified by RFC-0036. The CPU
adapter uses this existing state machine instead of implementing another one.
The earlier six-method sketch described the computation's decomposition; the
reference implementation composes those existing kernels inside the phase
adapter rather than publishing placeholder `ProofUnit`, `TreeRequest` or
`OpeningGroups` types before their planner exists.

`tessera-prover` drives the protocol and owns the transcript; backends compute.
The CPU adapter runs the exact witness engine, streams canonical rows through
P1, proves tree levels and mixed graph/native stages, derives its own original
opening targets, streams P4, and regenerates P5 columns. Public P1 metadata owns
its layout across network rounds. Successful P4 releases an opaque permit tied
to that same commitment; P5 cannot consume a permit from another P1. Per-proof
randomness is a fresh OS seed held on the heap in `Secret`; BLAKE3 streams under
RFC-0004's `advice-masks`, `prover-share` and `ni-session` domains, constructed
in place on the heap inside `Secret` without touching the stack (OBS-2,
[#567](https://github.com/AbdelStark/tessera/issues/567)), produce replayable
masks, the prover coin-toss share and the NI session id. Every mask coefficient
uses rejection sampling (`w < P`, rejected words discarded whole), covered by
INV-ARCH-2's RFC-0004 stream exception. Completion and
abort clear the witness and private seed/PRG state. The CPU backend uses scoped standard-library threads and is the reference for every kernel. The current driver composes one
exact-greedy request; pass planning and protocol/2 are separate work.

### Witness engine

- **Exact integer semantics** of [RFC-0005](RFC-0005-integer-model.md#integer-semantics). INT8 × INT8 GEMMs with INT32 accumulation use cuBLASLt from CUDA ≥ 12.8; integer accumulation is exact in any order, so results do not depend on the library's tiling or split-K choices (S2 bounds every accumulator below 2^31).
- Gadget arithmetic (requantization, RoPE rotations, softmax maxima, brackets, AES, SHA-256) runs in custom kernels with 64-bit integer intermediates.
- **Checkpoints.** Layer-boundary residuals (INT20 stored as INT32) are kept for the proof's tokens, so any layer's tensors can be regenerated without rerunning earlier layers.
- The CPU engine implements the same semantics with plain `i64` arithmetic and is the oracle for differential tests ([07 §5](../spec/07-testing-strategy.md#5-differential-and-conformance-tests)).

### Kernels

| Kernel | Used by | Notes |
|---|---|---|
| Goldilocks and `Fp2` arithmetic library | All | 32-bit limb multiplies, branch-free reduction ([RFC-0002](RFC-0002-field-arithmetic.md)) |
| Coset NTT (sizes 2^10–2^23) and RS encode with vanishing mask | Advice and weight commitments | Batched rows; canonical output |
| BLAKE3 column-chain and Merkle kernels | Commitments | Byte-identical to the CPU reference ([RFC-0004](RFC-0004-hashing-and-transcript.md#column-digests)) |
| Fraction-tree build and level sumcheck | LogUp-GKR | Bottom-up build fused with leaf generation; the level sumcheck as the product tree's; a subtree of constant leaves, such as the masked scores', in closed form ([RFC-0034](RFC-0034-causal-blocks.md#constant-leaves)) |
| Product-tree build and level sumcheck | Max certificate | `eq` never materialized: per instance a scalar, the current variable's line and two tables of about `√N` entries, in shared memory; two node sums per round, the third recovered from the instance's claim (Gruen), about 15 `Fp`-multiplication equivalents per leaf against 25.5 with an `eq` table ([#258](https://github.com/AbdelStark/tessera/issues/258)); a subtree of constant leaves in closed form (RFC-0034) |
| Sumcheck round kernels (degree ≤ 7; ≤ 10 at a reduction stage of `tessera-protocol/2`, [RFC-0013](RFC-0013-zero-knowledge.md#masked-extensions) 'Degree') with `eq` tables | Stages | Front-padded batching |
| Table gather | Witness and leaf generation | Piecewise tables keep the dense core in L2 where it fits |
| Limb-decomposed folds | Weight fold, tensor folds over tokens, query folds, the opening's row combination | `Fp2` coefficients split into 8-bit limbs; INT8 GEMMs; exact recombination |

### Partitioning

A node with `G` GPUs (8 by default) splits the model's layers into `G` contiguous ranges, one per GPU, matching the advice segments of [RFC-0011](RFC-0011-advice-commitment.md#canonical-advice-layout). Each GPU holds its layers' weights (405 GB / 8 ≈ 51 GB for 405B) and commits its own segment.

- Witness pass: pipeline-parallel over micro-batches of tokens; residuals move between GPUs over NVLink.
- Stages and tree levels: each GPU computes its layers' contribution to every round; round messages are summed by an all-reduce of at most a few hundred `Fp2` elements per round (NCCL through `cudarc` 0.x, pinned in the lockfile).
- Embedding and LM head are assigned to the first and last GPU.

### Pass structure

Every proof runs these passes, each regenerating the parts of the witness it needs from checkpoints:

| Pass | When | Regenerates | Native cost |
|---|---|---|---|
| P1 witness and commit | Before `Round1` | Everything; streams rows to the committer; counts multiplicities; writes checkpoints | 1× forward |
| P2 leaf stage | After the seed | Lookup keys and values and softmax keys: once, group by group when the stage is grouped, and once more for the merge (P2b, [RFC-0054](RFC-0054-token-groups.md#pass-structure)) | 1× forward; 2× when grouped |
| P3 stage regeneration | Per reduction stage | Folds of the tensors the stage reads; most are elementwise from checkpoints, not GEMMs. A stage larger than memory accumulates its window's grid instead ([RFC-0055](RFC-0055-windowed-stages.md#the-window)) | ≤ 1× forward in total for the stages that fit; ≤ 1× per windowed stage |
| P3b window fold | Per window of a reduction stage larger than memory, after its rounds | The stage's windowed tables, bound at the window's challenges ([RFC-0055](RFC-0055-windowed-stages.md#pass-structure)); a further window's grid rides the fold before it ([RFC-0056](RFC-0056-native-working-sets.md#the-windows)) | ≤ 1× forward per window |

| P4 opening | After the last stage | Every committed row, which it combines into the opening's combined rows ('Opening pass') | 1× forward, plus `c_r` per committed element |
| P5 column answers | After the column queries | Every committed row, whose masked codeword it evaluates at the queried columns, hashing nothing | 1× forward, plus `c_a` per committed element |
| W weight fold | Once per proof | Every weight | 0.24 GPU-s per 405B proof (source §5) |
| WA weight column answers | After the column queries, beside P5 | Nothing: the queried columns of the epoch's stored codewords ([RFC-0012 'Column answers'](RFC-0012-weight-commitment.md#column-answers)) | None: 0.35 GB of reads per 405B proof at `live-64`, overlapped with P5 |

`n_regen` in the cost model ([08 §1](../spec/08-performance-budget.md#1-cost-model)) counts the forward passes of P1 to P5, five. The model adds the merge's pass P2b when the leaf stage is grouped, and a fold pass P3b per window: one per stage larger than memory, and one per eight own variables its rounds bind of an instance ([RFC-0056](RFC-0056-native-working-sets.md#the-windows)).

Stage 1's P3 also keeps what `SoftmaxRow`'s score instances need, which never hold the score tensor ([RFC-0056](RFC-0056-native-working-sets.md#score-instances-without-the-score-tensor)): the INT8 queries and keys and two row tables, about 45 GB for a 405B batch of 16K tokens and 60 GB at 128K of context. The merge's folds, the table trees' streamed rounds and the windows' folds, the combination of P4, the evaluation of P5 and the weight fold are the model's streaming, row-combination, column-answer and weight terms ([RFC-0020](RFC-0020-cost-model.md#executable-cost-model)). WA costs no GPU time, so the model has no term for it. A prover without the store evaluates the weight answers instead, which adds about 1.3 GPU-s per 405B proof to the weight term. The run manifest records every pass and its time.

**Leaf levels** ([RFC-0054](RFC-0054-token-groups.md)).
- **When it fits.** A leaf stage whose lookup trees and product forests fit in the memory budget is proven in memory after one regeneration. Its table trees' levels that do not fit beside them stream: each round recomputes the table leaves from the tables' rows and the stored multiplicities, and regenerates nothing.
- **When it does not.** A larger stage is proven in token groups, each in memory with its own challenges. One merge sumcheck brings their leaf claims back to one point per tensor. That is two regenerations whatever its size: the group pass and the merge's pass.
- **What this section first said.** It streamed every level of one shared-challenge batch instead: `c(c + 1)/2` regeneration passes, 36 to 55 for a 405B 16K-token batch on 0.2 TB, which add 53 to 235 to `μ` (RFC-0054 'Motivation').

**Proof units** ([RFC-0057](RFC-0057-one-proof-unit.md)). A proof is one unit: the statement's batch, under one session seed and one advice commitment. This section first split a batch into proof units, sets of requests or layer ranges of one request, each with its own advice commitment. At the reference statement no budget needs them since token groups (RFC-0054), windows (RFC-0055) and RFC-0056's working sets, and each unit only added combined rows, a table side and, for a set of requests, a weight fold. A batch too large for one proof is proven as several proofs, which nothing links. A request longer than one proof holds, about 24K tokens at 405B within the proof-size gate and at best 64K within memory, needs [#470](https://github.com/AbdelStark/tessera/issues/470).

**Fusion.** P2 builds each group's trees bottom-up as it regenerates their leaves; P3 regenerations are fused into P2b or P4 when the stage's point is known before the pass begins. A windowed stage's P3 and P3b fuse with neither, since each needs challenges the pass before it fixes (RFC-0055 'Pass structure'). The pass planner's objective is the fewest full-forward equivalents; v0.5 publishes the achieved count per context length.

**Opening pass.** P4 folds every committed element into `u_ξ`, and into the `u_g` of each claim group on its tensor ([RFC-0011](RFC-0011-advice-commitment.md#opening-protocol) step 4). That is one product of an `Fp2` coefficient by the element per combined row it enters; every row's `P_A` mask coefficients are combined the same way.
- **Blocking.** The `G + 1` accumulators hold `k` values of `Fp2` each, 64 MiB at `k = 2^22`: too much to stay on chip. The pass therefore combines rows in blocks of the parameter set's row block `B`. It holds a block's rows as it regenerates them, multiplies them into each accumulator tile in one sweep, and reads and writes each tile once per block. Accumulator traffic is then `32/B` bytes per product, 1 byte at `B = 32`; combining row by row would move 32.
- **Tensor cores.** A block's products are one limb-decomposed INT8 GEMM of the `(G + 1) × B` coefficients by the block's rows, as the weight fold is ([RFC-0012](RFC-0012-weight-commitment.md#opening-by-the-folding-pass)). A coefficient's sixteen 8-bit limbs make `16·⌈b/8⌉` products per element of `b` bits and per combined row.
- **Cost.** The model prices the combination at `c_r` per committed element ([RFC-0020](RFC-0020-cost-model.md#executable-cost-model)). The CPU reference multiplies every row into all `G + 1` combined rows. It costs 0.55 ns per product on one Apple M4 Max core (`pcs/combine`).

**Column answers.** The answer to the column queries ([RFC-0011](RFC-0011-advice-commitment.md#opening-protocol) step 7) holds every committed row's symbol in each distinct queried column. The indices are drawn after the combined rows that P4 produces are absorbed ([RFC-0014](RFC-0014-interaction-and-soundness.md#transcript-schedule), orders 8 and 10), and must be: a prover that knew them first could fit `u_g` to the queried columns. Advice codewords are freed in P1: storing them, or the witness they encode, is rejected ([RFC-0011](RFC-0011-advice-commitment.md#alternatives-considered) Alternative 1, Alternative 1 here). So P5 regenerates every committed row once more and evaluates its masked codeword `g = f + Z_H · m` ([RFC-0011 'Encoding'](RFC-0011-advice-commitment.md#encoding)) at the `τ' ≤ τ_A` distinct queried positions. Two mitigations bear on its cost:

- **Fusion** cannot remove it. No pass that runs before the queries can know them, and no pass but P5 runs after them, so P5's regeneration is a pass of its own.
- **Evaluating only the queried symbols** can cut its `c_a`. Re-encoding a row whole costs an inverse NTT of size `k` and a coset NTT of size `2k`, about 34 butterflies per element at `k = 2^22`, to keep `τ'` symbols. An output-pruned coset NTT computes only those. Alternatively, symbol `j` of row `r` is `⟨A_r, w_j⟩ + Z_H(x_j) · m_r(x_j)`, where `w_j` holds the barycentric weights `ω_k^i · Z_H(x_j) / (k · (x_j − ω_k^i))` of `x_j`, shared by every row. The answers are then one matrix product of the rows with `τ'` weight vectors. That is `τ'` multiply-adds per element, several times the encoder's butterflies on CUDA cores, but it runs on tensor cores as limb-decomposed INT8 products, like the weight fold ([RFC-0012](RFC-0012-weight-commitment.md#opening-by-the-folding-pass)): `⌈b/8⌉ · 8` products per queried column for a row of `b`-bit integers.

The planner picks whichever evaluation measures cheaper; [RFC-0020](RFC-0020-cost-model.md#measurement-protocol-v01-gate) measures `c_a`.

The weights' codewords are fixed for an epoch, so their rows are answered from a store instead ([RFC-0012 'Column answers'](RFC-0012-weight-commitment.md#column-answers)): `answer_columns` reads the queried columns beside P5, and evaluates a weight row only where the store is missing or damaged.

### Memory scheduler

Budgets per device and host come from `ProverConfig.memory_budget`. The planner assigns tiles, token groups, windows and streaming decisions before P1 (RFC-0054, RFC-0055, RFC-0057) so that no allocation fails mid-proof; an infeasible plan fails fast with `ERR-RES-MEMORY` before `Round1`.

- **The CPU reference's plan** (`tessera_prover::PassPlan`, `Prover::plan`) lists its passes and retains a conservative resident-cell floor for graph tensors, lookups, table rows and advice entries, even when generated table fractions stream.
  - **The charge.** The floor remains 64 bytes per cell plus the weight commitment's own bytes; any P2 working set beyond its reserved portion is additional. Earlier resident conformance proofs measured 32 to 34 bytes per cell. Counting-allocator tests hold every measured peak within its plan and check refusal before P1. Streaming can use less than a quarter of the reservation; there is no minimum-utilization requirement. The borrowed execution plan, weight state and input batch token/request vectors, including spare capacities, remain charged beside the proof, and the reported peak also covers checked session preparation, including the canonical maximum-context row preflight and regenerated weight construction. Caller-owned model storage is excluded from the host allowance and reported separately in whole-process measurements. The [table-batch allocation model](../perf/v0.3/table-batch-memory.md) records the component bounds and measurements; whole-model qualification remains separate.
  - **The limit.** A proof whose plan exceeds the budget takes token groups (RFC-0054): the plan then holds its largest group's lookup trees in place of the stage's, and runs the merge's pass, P2b, after P2. The fewest group bits that fit win. A proof that no grouping fits fails before P1 with `ERR-RES-MEMORY`.
- **Checked checkpoint fallback.** For an ungrouped protocol/1 proof that no resident plan admits, the CPU may replace that route's logical-cell floor with a complete physical lifetime bound. This route borrows checked INT8 model tensors, owns clearing tokens/checkpoints and nonweight cached tensors, and selects generated table sources and positive reduction-window cuts before P1. Its checked maximum covers session/unit/public planning, caller plan/state/batch capacities, table cores/count backing, replay, original native/leaf/stage claims, retained messages, target transfer, opening accumulators, weight work and streaming packet publication. Any missing owner, unsupported shape or sizing overflow refuses before P1. It preserves the resident implementation as the first choice and oracle; this exception does not lower its 64-byte floor. Caller model storage, stack, allocator overhead and external transport/runtime allocations are reported separately. The standard Live directory reader caps seed/query envelopes from public message shapes before allocating. This route changes no proof, transcript or parameter semantics. Whole-model time/memory measurements and independent acceptance remain qualification requirements; a tiny accepting proof is not a real-model result.
- **CPU row workers.** `Backend::Cpu { threads }` and `tessera prove --threads N` bound witness and row-encoding participants, including the caller; the default is one. Encoders share public tables, borrow disjoint codeword slices and draw masks serially. The plan adds an explicit configured NTT stack/control reserve before weight construction and P1. Resident and physical plans retain their existing charges; the physical chooser spends only the remaining allowance. Failed OS thread creation computes unclaimed tasks serially, and all created workers join before return or panic propagation. [The CPU row inventory](../perf/cpu-row-ntt.md#parallel-cpu-row-execution-659) records the pinned control sizes, process overhead scope and cross-worker proof controls. This scheduling choice changes no protocol bytes.
- **Pass counters.** The run manifest records every pass and its time as `tessera_passes_total{pass}` and `tessera_pass_seconds{pass}` ([05 §4](../spec/05-observability.md#4-metrics)).

The [CPU lifetime inventory](../perf/v0.3/cpu-memory-lifetimes.md) records which
backing allocations coexist across these passes, and which completed reduction
buffers are cleared before openings. The separate checked checkpoint fallback is
qualified against that inventory and whole-route controls; real-model memory and
runtime qualification still require an actual execution and independent acceptance.

### Crash consistency

- Before `Round1`, the prover writes a per-proof secret seed (from the OS CSPRNG) to a 0600 file via atomic rename. Every masking value, mask polynomial and blinding choice of the proof's advice, along with its coin-toss share and NI session id, derives from it through RFC-0004's per-purpose domains (`advice-masks`, `prover-share`, `ni-session`). The weights' masks and blinding rows derive from the epoch's `masks_seed` instead (INV-ZK-3, draft [RFC-0073](RFC-0073-weight-mask-expansion.md)), and the rows a proof uses are its reservation's.
- Every phase is a deterministic function of inputs, the per-proof seed, the session seed and verifier messages. After a crash, the prover reruns completed phases to rebuild state and resumes; messages already sent are reproduced byte for byte and compared before continuing.
- The per-proof seed file is deleted after `Round3Response` is sent.

### Release schedule

A verifier can time the prover by when its messages arrive:
- in live mode, `Round1`, `Round2Response` and `Round3Response`, each timed from the verifier's message before it;
- in either mode, the proof, timed from the request.

The work behind each message depends on secrets beyond its values. Multiplicity counting increments the counter of each lookup's row, and `Exp`, `Exp2Frac` and `Silu` read their dense core at a key ('Determinism'). Caches, atomic contention and memory banks make that work faster or slower with how the keys spread, and power limits move the clocks with the data. `tessera-protocol/2` promises zero knowledge, so none of these durations may reach a verifier. Each message a verifier can time is therefore released on a schedule that public data fixes:

1. **Budget.** Before the work behind a message starts, the prover computes its budget `D = d_0 + Σ_u d_u · n_u` from public data alone:
   - the message's work counts `n_u` (committed elements, lookups, scores and tokens), from the claim list and the batch's shapes;
   - the coefficients in `ProverConfig.release`, which come from calibration runs on public inputs, never from the timings of a session that proved secret data.
2. **Release.** Timed from its trigger, a message is released at `D` if its work is done by then. The trigger is the receipt of the verifier's preceding message or of the request, at the time the transport records on a monotonic clock. It is not the time the prover reaches the message, which other work on the node could delay. Otherwise it is released at the first slot `2^j · D` that the work meets, for `j` from 1 to `L = ProverConfig.release.doublings`.
   - The `Prover` method that yields the message returns at that time ([02 §3.1](../spec/02-public-api.md#31-prover)), so a caller cannot send it early.
   - A message is complete before its first byte leaves, so its transfer time depends only on its length, which is public, and on the network.
3. **Overrun.** Work that misses the last slot aborts the session with `ERR-RES-TIME`. Each release past `D`, and each abort, increments `tessera_release_overruns_total` ([05 §4](../spec/05-observability.md#4-metrics)), so that operators recalibrate before overruns become common.

A release time therefore reveals only the public statement and which of the `L + 1` slots the message took. A message whose work meets its budget reveals nothing; one that overruns reveals at most `log2(L + 1)` bits.

- **Phase records.** The prover's own records of how long its phases took stay on its side of TB-1 ([06 §3](../spec/06-security.md#3-trust-boundaries)). These are the run manifest's `tessera_phase_seconds` and the unit costs derived from it. Published, they would undo the schedule.
- **Verifier deadline.** A verifier's session deadline (`ERR-PROTO-TIMEOUT`) must exceed each message's last slot.
- **Cost.** The schedule costs latency, not throughput: the node works on other proofs while a message waits.
- **`tessera-protocol/1`** promises no hiding, and releases each message when it is ready.

### Determinism

- Field reductions are exact, so the order of GPU reductions and all-reduces does not change results.
- Integer GEMMs are exact under S2.
- Nothing in a kernel's control flow depends on secret values (INV-ARCH-2), except a stream of RFC-0004 rejecting its words, the one exception INV-ARCH-2 admits ([RFC-0004](RFC-0004-hashing-and-transcript.md#hash-function-and-domains), draft [RFC-0073](RFC-0073-weight-mask-expansion.md#the-stream)). Memory addresses do, in two places: multiplicity counting, which increments the counter of each lookup's row (RFC-0008), and the reads of the `Exp`, `Exp2Frac` and `Silu` dense cores at a key. A kernel that compacts a stream's accepted words is the same exception. None of these reaches a verifier, which times the prover only through the release schedule.

## Alternatives Considered

1. **Store the witness on NVMe instead of regenerating.** Considered for long context. Rejected as the default: about 35 TB of `Fp2` leaves at 16K context, and read bandwidth of a large NVMe array (tens of GB/s) makes each pass slower than regeneration on tensor cores. Kept as a fallback for linear-layer leaves, whose regeneration needs full GEMMs.
2. **Independent per-tile challenges for leaf levels.** Considered to make each tile's tree provable in one in-memory pass. Rejected at first: tile-specific final points multiply opening groups, and each group costs a combined row (64 MB at `k = 2^22`). [RFC-0054](RFC-0054-token-groups.md) adopts it as token groups, whose merge brings the points back to one at the cost of one regeneration.
3. **Multi-variable sumcheck rounds** (binding several variables per round to cut streamed passes). Considered. Rejected for v1: message size and prover work grow as `(d + 1)^w` per round; reconsider if RISK-10 measurements exceed 50% overhead.
4. **Tensor-parallel witness generation** (split each layer across GPUs, as in serving). Rejected for proving: it scatters each committed row across GPUs, while layer ranges map one segment to one GPU.
5. **CUTLASS kernels instead of cuBLASLt for INT8 GEMMs.** Both are exact for integers; cuBLASLt ships with the toolkit and avoids a source dependency. CUTLASS remains an option for fused limb-decomposed folds.

## Drawbacks

- A leaf stage larger than memory pays the merge's regeneration (RISK-10, RFC-0054), which at 405B means a 16K-token batch at every context length. A reduction stage larger than memory pays its window's fold pass (RFC-0055): two for `minimal`, one for `servable` at 405B and 16K tokens.
- Openings alone cost two of the five regenerations: P4 forms the combined rows, and P5 answers the column queries that depend on them.
- Two backends double the kernel test burden.
- The pass planner is a new optimization component with its own failure modes.

## Migration / Rollout

v0.2: CPU witness engine. v0.3: CPU prover backend with the pass structure and counters, in memory for tiny models and with forced token groups in tests, and the release schedule's slots. v0.4: the release schedule in every `tessera-protocol/2` session. v0.5: GPU backend, partitioning, streaming, measured on 8B and 70B. v1.0: 405B with the pass planner tuned against SC-2.

## Testing Strategy

- Witness engine: CPU and GPU engines bit-identical on every conformance model and on 8B, across micro-batch sizes and GPU counts 1, 2 and 8.
- Kernels: each GPU kernel against its CPU reference on random inputs, including NTT sizes 2^10–2^23 and BLAKE3 chains with partial blocks.
- Backends: full proofs from CPU and GPU backends are byte-identical with pinned masking randomness.
- Token groups and the table batch: RFC-0054 'Testing Strategy'. A tiny model under forced memory budgets takes 1, 2, 4 and 8 groups and verifies. Its table trees' streamed rounds give the in-memory messages byte for byte, and the pass counter counts P2 and P2b once each.
- Constant subtrees: the lookup and tree provers' messages equal the dense provers' byte for byte when they evaluate the masked scores' subtrees in closed form, for sequence lengths that are and are not powers of two ([RFC-0034](RFC-0034-causal-blocks.md#testing-strategy)).
- Column answers: for every evaluation the planner can pick, P5's symbols equal those of each row's whole codeword at the queried columns, repeated and boundary indices included; the pass counter counts P5 once per proof.
- Crash consistency: fault injection kills the prover after each phase; the resumed prover sends byte-identical messages.
- Release schedule:
  - work that meets its budget is released at the budget;
  - every duration within a later slot is released at that slot's instant;
  - work past the last slot aborts with `ERR-RES-TIME`;
  - on a tiny model in `tessera-protocol/2`, each message leaves at its budget whatever the witness, including witnesses whose keys concentrate on one table row or spread over all of them.
- Performance: `tessera bench` reports pass counts and per-pass times nightly on the GPU runner.

## Open Questions

- **OQ-0018-1** (resolved by [RFC-0054](RFC-0054-token-groups.md)). Which is cheaper at 405B: streaming leaf levels, or splitting the batch into more, smaller proof units? Neither.
  - **Streaming.** Every level of every tree streaming costs 36 to 55 regenerations.
  - **Units.** Each unit carries every table tree.
  - **The answer.** Token groups cost two regenerations.

## References

- Source §5 (unit costs, weight pass), §6 principles 8–10, §7 (phases, streaming commitments), §9 problem 6 (hardware drift).
- Cormode, Mitzenmacher, Thaler, "Practical verified computation with streaming interactive proofs", ITCS 2012 (streaming sumcheck provers).
- [RFC-0010](RFC-0010-proof-schedule.md), [RFC-0011](RFC-0011-advice-commitment.md), [RFC-0012](RFC-0012-weight-commitment.md), [RFC-0020](RFC-0020-cost-model.md).
