# RFC-0054: Token Groups: a Leaf Stage Larger Than Memory

- Status: Accepted
- Authors: @AbdelStark
- Created: 2026-10-23
- Target milestone: v0.3 (CPU prover and verifier), v0.5 (GPU backend)
- Supersedes:
  - [RFC-0018](RFC-0018-gpu-prover-runtime.md#pass-structure) 'Pass structure', 'Leaf levels' and 'Proof units' as they bear on the leaf stage, its P2 row, Alternative 2 and OQ-0018-1;
  - [RFC-0010](RFC-0010-proof-schedule.md#tree-levels) 'Tree levels' and 'Sumcheck messages' for a leaf stage that is grouped: its instances, labels and order;
  - [RFC-0008](RFC-0008-logup-and-tables.md#batching) 'Batching' and 'Fraction tree' for a grouped leaf stage: one lookup tree per table and group, the roots and their check;
  - [RFC-0014](RFC-0014-interaction-and-soundness.md#transcript-schedule) 'Transcript schedule', rows 4 and 5, and the soundness calculator's union row;
  - [RFC-0036](RFC-0036-protocol-messages.md#round1) `Round1`, `LogUpProof` and `TreeProof`;
  - [RFC-0037](RFC-0037-stage-schedule.md#splits) 'Splits' and [RFC-0045](RFC-0045-leaf-layout.md#the-split) 'The split', for a grouped stage: RFC-0045 applies to each group, and the leaf stage's values come from the merge;
  - [RFC-0020](RFC-0020-cost-model.md#executable-cost-model) 'Executable cost model': the streaming term.
- Scope: `tessera-protocol/1`. `tessera-protocol/2` keeps one group until OQ-0054-1 is answered.
- Origin: [#329](https://github.com/AbdelStark/tessera/issues/329)

## Summary

The leaf stage proves every LogUp fraction tree and every product forest in one batched sumcheck per level. At 405B its trees hold `K + 2·A·S` leaves per token, 5.5×10^7 at 1K context and 3.0×10^8 at 16K: 9×10^11 and 5×10^12 for a 16K-token batch, against 0.2 TB of free HBM. RFC-0018 streams every level that does not fit, one regeneration pass per round. Counted over every tree and every level, that is 36 passes at 1K and 55 at 16K, and adds 53 to 235 to `μ`.

This RFC proves such a stage in **token groups** instead. When the stage's lookup trees and product forests do not fit, the prover splits the batch's tokens into `G = 2^g` groups whose trees do. Each group's trees are proven in memory, with challenges of its own, one group after another. The table trees, whose leaves come from the tables' rows and the stored multiplicities, are proven in a batch of their own and stream without regenerating anything. A **merge** sumcheck then reduces the groups' leaf claims to one point per tensor, so the reduction stages, the claim pool and the openings are unchanged.

The stage then costs two regenerations at any size: the group pass, which is RFC-0018's P2, and the merge's pass. At 405B that is one more regeneration than a stage that fits, and about 0.2 to 1.6 on `μ` for the merge's arithmetic. The price is in the transcript: `G` sets of level messages, about 120 KB each at 405B, and a union bound about 5 bits lower. With `g = 0` the stage is RFC-0010's, byte for byte, but for one byte of `Round1`.

## Motivation

[#329](https://github.com/AbdelStark/tessera/issues/329) counted what RFC-0018 'Leaf levels' costs when every rule of the corpus applies:

- **Every tree streams.** A level's working set is every tree batched at it: the linear layers' lookups (`K` per token), the scores' `Exp` lookups and `SoftmaxRow` leaves (`2·A·S` per token), and the table trees' `M` rows. RISK-10 and `streaming_passes` counted the scores' leaf level alone. At 1K the linear layers' lookups are 69% of the leaves.
- **Every level streams.** Each level above the leaves holds half the entries of the one below. A leaf level that streams `c` rounds therefore costs `c + (c − 1) + … + 1 = c(c + 1)/2` passes. No pass serves two levels: a level's first round needs the level above's last challenge.
- **Every pass regenerates.** The linear layers' lookup keys are GEMM outputs, so a pass over them costs a forward pass, plus about 6 to 8 ps per leaf to fold.

With `minimal`'s corrected counts (285,696 commitments and 302,080 lookups per layer per token), a 16K-token batch and 32 bytes per leaf on 0.2 TB, `shared_streaming_passes` in the executable model gives:

| Context | Leaf level `c` | Passes, every level | Added to `μ` at 6 to 8 ps per leaf |
|---|---:|---:|---:|
| 1K | 8 | 36 | +53 to +59 |
| 4K | 9 | 45 | +85 to +99 |
| 16K | 10 | 55 | +190 to +235 |

SC-2's ceilings are 12 and 45, and the corrected `μ` before any streaming is about 11 and 26. RFC-0018's own alternatives do not close the gap:

- **Proof units** split the batch until its levels fit. Every unit holds every table tree, though: `M = 6.7×10^9` rows, 213 GB at 32 bytes, more than the memory. At 1K, each token adds 1.7 GB of lookup and score leaves. Small units also pay the table side once each, about 0.5 GPU-s, which a batch of 64 tokens cannot amortize.
- **Windowed rounds.** One pass can answer `w` rounds of a level from a `3^w`-point grid. That saves passes but not levels: no shared-challenge schedule can prove a level that does not fit without regenerating its leaves after the level above has finished. Any such schedule pays at least one regeneration per oversized level, 8 to 10 at 405B, against SC-2's whole budget of about 1 to 19.

Proving each group with challenges of its own removes the level-to-level dependency across groups. RFC-0018 rejected exactly that (Alternative 2), because each tile's own final point would multiply the opening groups. The merge answers that objection with one regeneration.

## Goals

- A leaf stage of any size proven within a memory budget, at a fixed number of regenerations: two.
- The leaf stage's outputs unchanged: one claim per tensor, site and layer at one point, so RFC-0037's layer point, the reduction stages and the openings stay as they are.
- `g = 0` is RFC-0010's leaf stage, so every batch that fits keeps its proof.
- Soundness within RFC-0014's targets at every group count this RFC admits.

## Non-Goals

- **The reduction stages' working sets.** They exceed the budget at 405B too ('Reduction stages'); [#452](https://github.com/AbdelStark/tessera/issues/452) addresses them.
- **Zero knowledge for groups** (OQ-0054-1). `tessera-protocol/2` proofs keep `g = 0`.
- **Multi-node proving.**

## Proposed Design

### When groups apply

The prover's planner chooses `g` before `Round1`, from public sizes and its memory budget `F` (`ProverConfig.memory_budget`, RFC-0018 'Memory scheduler'):

- `B` is the bytes a level holds per node: 32, two `Fp2` values.
- `L_tok` is a token's leaves: `K` lookups, plus one `Exp` lookup and one `SoftmaxRow` leaf per valid score, `K + 2·A·S`. Masked scores are constant subtrees (RFC-0034) and hold none.
- **One batch.** While the batch's lookup trees and product forests fit, `T·L_tok·B ≤ F`, `g = 0`. The stage is RFC-0010's single batch. Its levels that do not fit with the table trees stream those alone ('The table batch').
- **Groups.** Otherwise `g` is the least value with `(T/2^g)·L_tok·B ≤ F`. A plan with `g` above 14, or with a group of fewer than one token, fails before `Round1` with `ERR-RES-MEMORY`.

The rule is the planner's, and it runs before P1: the witness pass counts each group's column padding in row 0's multiplicity. The verifier reads `g` from `Round1` and checks only its bounds ('Messages').

At 405B on 0.2 TB, a 16K-token batch takes `g` = 8, 9, 10 and 13 at 1K, 4K, 16K and 128K context: groups of 64, 32, 16 and 2 tokens.

### Groups

- **Tokens.** The batch's token axis, per-dimension padded to `T_pad = 2^t` ([RFC-0011](RFC-0011-advice-commitment.md#canonical-advice-layout) rule 1), splits into `G = 2^g` ranges of `2^{t−g}` consecutive tokens. Group `h` holds the tokens whose top `g` index bits are `h`. `g ≤ t`.
- **A group's instances.** Group `h` holds, for each LogUp instance, one lookup tree over the lookups of its tokens. It also holds the `SoftmaxRow` forest of its tokens' query rows. Every other rule of RFC-0008 and RFC-0010 applies within the group:
  - **Blocks.** A block of [RFC-0045](RFC-0045-leaf-layout.md#blocks) restricted to the group's tokens is the group's block, of `2^{k−g}` entries. The column, the leaf order, the leaf point and the split apply to the group's blocks as RFC-0045 states them for the stage's.
  - **No token axis.** A block whose domain has no token axis belongs to group 0 whole.
  - **Forests.** The group's forest is `2^{s−g}` trees with their public root values.
  - **Same shape.** Every group's instances have the same heights and shapes, so the groups' messages have the same lengths.
- **Table trees** are not grouped: one per table, as RFC-0008 states, in the table batch.
- **The token axis in a tensor.** In a tensor read by a group's lookups, the group is fixed by the token index's top `g` bits. They are `g` of the tensor's variables, its group variables `S`. Group `h`'s block of the tensor is the tensor with `S` set to `h`.

### The leaf stage with groups

With `g ≥ 1`, the leaf stage runs in this order:

1. **Challenges.** `logup/alpha` and `logup/beta`, as RFC-0008 states.
2. **Roots.** Under `logup/roots`, for each LogUp instance in schedule order: the `G` lookup-tree roots `(n_h, d_h)` in group order, then the table-tree root `(n_t, d_t)`.
   - The verifier checks every `d ≠ 0`, else `Reject(Lookup)`.
   - It then sums the instance's `G + 1` fractions by `(n, d) ⊕ (n', d') = (n·d' + n'·d, d·d')`, in the order sent, and checks that the sum's numerator is 0, else `Reject(Lookup)`. That is RFC-0008's identity: the lookups' fractions and the rows' fractions sum to zero.
3. **Each group `h`, in order from 0.** RFC-0010 'Tree levels' for the group's instances, under `group/{h}/tree/{ℓ}/root/{c}`, `group/{h}/tree/{ℓ}/gamma`, `group/{h}/tree/{ℓ}/lambda`, `group/{h}/tree/{ℓ}/round/{j}`, `group/{h}/tree/{ℓ}/r/{j}`, `group/{h}/tree/{ℓ}/evals` and `group/{h}/tree/{ℓ}/line`.
   - `ℓ` counts the group's levels from its root.
   - Each lookup tree enters at its group level with its root as its claims, as RFC-0008 states.
   - After its last level, the group sends its lookup trees' value evaluations under `group/{h}/logup/values` and its split values under `group/{h}/split` ([RFC-0045](RFC-0045-leaf-layout.md#the-split)).
   - The verifier checks them as RFC-0008 and RFC-0045 check the stage's.
4. **The table batch.** RFC-0010 'Tree levels' for the table trees alone, under `tables/tree/{ℓ}/…`. Their leaf claims go to RFC-0008's final checks, the multiplicities' to the pool, as now.
5. **The merge** ('The merge'), under `merge/gamma`, `merge/beta`, `merge/round/{j}`, `merge/r/{j}` and `merge/evals`. Its final evaluations are the stage's values: one per tensor, site and real layer, at the merge's point. They take the place of the ungrouped stage's lookup tensor values.
6. **The atom splits.** [RFC-0037](RFC-0037-stage-schedule.md#splits)'s atom split values follow under `stage/0/split`, for the merged claims.

With `g = 0`, the stage is RFC-0010's, under its labels.

### The table batch

The table trees' leaves are `(−m_j, α − t_j)`: the table's row from its generator, and the multiplicity that P1 counted and stores (08 §4: 19 to 42 GiB at 405B). A round over them regenerates no witness. Their levels run as RFC-0010's, with RFC-0018's streaming rule for this batch alone. A level that does not fit streams one round per halving, each recomputing the table leaves, until its folded state fits. That is `c_T(c_T + 1)/2` rounds over the levels, with `c_T = ⌈log2(M·B/F)⌉` for the tables' `M` rows. At 405B, `c_T` = 1: 6.7×10^9 rows hold 213 GB, which halves to fit 0.2 TB, so the table trees stream one round.

With `g = 0`, the table trees stay in the stage's one batch. Its levels that do not fit stream the same way. The lookup trees and forests, which fit, stay resident.

### The merge

The groups end with each tensor's claims at `G` points: group `h` claims `P̃(z_h ‖ h) = v_h` on tensor `P`. Here `z_h` assigns its point to `P`'s variables outside `S`, and the coordinates in `S` are the bits of `h`. The merge reduces every tensor's `G` claims to one, at one point, in one sumcheck.

**Claims.** Each value of a group's split is one claim on its tensor `P_i`, on `n_i` variables, at the group's point. A tensor without a token axis has `S_i` empty and claims from every group that reads it, at `G` points over all of its variables.

**The sumcheck.**
- **Batching.** `γ` is squeezed under `merge/gamma`, then `β` under `merge/beta`. The merge proves `Σ_i γ^i·2^{n_max − n_i}·Σ_h β^h·v_{i,h}`, front-padded, of `Σ_x Σ_i γ^i·W_i(x)·P̃_i(x)`.
- **Weights.** For a tensor with group variables, `W_i(x) = Σ_h β^h·eq(z_{i,h}, x_{S̄_i})·eq(h, x_{S_i})`. Its restriction to the Boolean cube is `β^h·eq(z_{i,h}, x_{S̄})` on group `h`'s block and 0 elsewhere. For a tensor without group variables, `W_i(x) = Σ_h β^h·eq(z_{i,h}, x)`.
- **Variable order.** The rounds bind each tensor's variables outside `S_i` first, lowest first, then those in `S_i`, lowest first, and `P̃_i` and `W_i` are read in that order. Front-padding gives a tensor of `n_i` variables the last `n_i` challenges, so tensors of one size and one `S` lie at one point.
- **Messages.** The degree is 2. Round `j` sends `p_j(0)` and `p_j(2)` under `merge/round/{j − 1}` and squeezes `merge/r/{j − 1}`, as [RFC-0010](RFC-0010-proof-schedule.md#sumcheck-messages) 'Sumcheck messages' states.
- **The final check.** The prover sends `P̃_i(r_i)` for every tensor in claim order under `merge/evals`, at its point `r_i`. The verifier computes `W_i(r_i)` itself, a sum over the groups, and checks `Σ_i γ^i·W_i(r_i)·P̃_i(r_i)` against the last claim. A failed round or final check is `Reject(Sumcheck { stage: 0, round })`, its round counted from 0 as the labels are and the final check at `n_max`.
- **The result.** `P̃_i(r_i)` is tensor `P_i`'s claim at `r_i`: the value RFC-0037 takes from the stage for that tensor, site and layer.

**Points share weights.** Within a group, RFC-0045 puts every block of one size at one point, `P_h[0..k]`. Tensors read over the same axes therefore share `z_{i,h}` in every group, and so share `W_i`. The merge's sum then holds one weight per class of such tensors, `W_c·(Σ_{i∈c} γ^i·P̃_i)`.
- The prover merges one combination per class, and the verifier evaluates one weight per class.
- At 405B there are a few tens of classes, against about 3,000 tensor claims per group.

**The prover in two passes** (informative). Write a class's combination `Q` over `(x, y)`, its variables outside `S` and in `S`, and `m = n − g`. Summing `W·Q̃` over the Boolean `y` and over the variables after round `j` collapses each group's share to an evaluation. Round `j ≤ m`'s message is then:

```text
p_j(Z) = Σ_h β^h · eq(z_{h,<j}, c_{<j}) · eq(z_{h,j}, Z) · Q̃(c_{<j}, Z, z_{h,>j}, h)
```

- **Group pass.** While group `h` is resident, the prover keeps, for each `j ≤ j*`, the table `T_{h,j}(a, b) = Q̃(a, b, z_{h,>j}, h)` of `2^j` entries. It is `Q`'s block `h` folded at the group's own coordinates. Round `j ≤ j*` reads `Q̃(c_{<j}, Z, z_{h,>j}, h)` from `T_{h,j}` in `2^j` operations per group.
- **Merge pass.** After round `j*`, one pass regenerates every merged tensor and folds its first `j*` variables at `c_{≤j*}`. That is `2^{n−j*}` values per class, and the remaining rounds run in memory on them and on `W`'s restriction.
- **Memory.** The tables hold `2^{j*+1}` entries per class and group. The folded state is per tensor, since every tensor's evaluation is sent: at most about `N_m·32/2^{j*}` bytes for `N_m` merged elements. `j*` is chosen so that both fit.
  - At 405B, 16K-token batch, 16K context: `N_m ≈ 4.9×10^12`, so `j* ≥ 10`. The tables then take about 1 GB for 30 classes and 1,024 groups.
  - At 128K the folded state can sit in host memory, or `j*` rises to 13 with about 64 GB of tables.
- **Lower bound.** No merge to one point avoids the second pass. Its point must be drawn after every group's claims are fixed, and the claim at a fresh random point is a new linear combination of every entry of `Q`, which a prover that cannot store `Q` must read again.

### Pass structure

- **P2, the group pass.** For each group in order, the prover regenerates the group's tokens at every layer from the checkpoints, builds and proves the group's trees in memory, sends its values and splits, and keeps its merge tables.
  - Attention needs every earlier token's keys and values, so P1 keeps them for the whole batch as well as the residuals: `2·n_layers·d_kv` bytes per token, 4.2 GB for 16K tokens at 405B.
- **P2b, the merge pass.** One regeneration of every tensor the leaf stage's lookups read, after the merge's first `j*` rounds.
- The table batch's streamed rounds regenerate nothing.

A leaf stage that fits costs P2 alone. A grouped one costs P2 and P2b, whatever `G`: six regenerations in RFC-0018's count instead of five.

### Soundness

Each check is RFC-0010's or RFC-0008's, applied per group, so RFC-0014's rows keep their form. They gain items:

- **Rounds.** The groups' levels and the table batch's levels are levels, and the merge is a stage of `n_max` rounds of degree 2 batching `P + G` claims. Its batching polynomial in `γ` and `β` has total degree below `P + G`.
- **The union row** sums every one of them: about `G` times the leaf stage's level terms.
  - At RFC-0014's default 405B statement, the leaf stage's levels add about 5·10^3 to the union's `7.3·10^4`.
  - With `G = 2^10` they add about `2.4·10^6`, and the union row falls from 111.8 bits to about 106.7.
  - At `G = 2^14`, the most this RFC admits, it is about 103.1 bits, above the 100-bit target.
- **The product-forest roots** keep `s/|Fp2|`, with each group's `s − g`.
- **The LogUp identity** keeps `v·max(N + T)/|Fp2|`. The groups' roots and the table root sum to one rational identity in `α` of the same degree.
- **The splits** are exact checks. A false value in any group's split is a false claim on its tensor. The merge's sumcheck rejects it except within its bound, or carries it to a false merged claim, which the next stage rejects within its own.

The soundness calculators of both workspaces take `G`, the merge's `n_max` and its claims `P` as sizes, and the verifier rejects `g > 14` ('Messages'), so the union row's bound covers every accepted proof.

### Messages

[RFC-0036](RFC-0036-protocol-messages.md)'s layouts change in place: `tessera-protocol/1` is still unassigned, `protocol_version` 0.

```rust
pub struct Round1 { pub protocol_version: u16, pub session_id: [u8; 16], pub statement: PublicStatement,
                    pub advice: AdviceCommitment, pub leaf_groups: u8 }   // g
pub struct TreeProof { pub levels: Vec<SumcheckProof>, pub groups: Vec<GroupProof>,
                       pub merge: SumcheckProof, pub splits: Vec<Fp2> }
pub struct GroupProof { pub levels: Vec<SumcheckProof>, pub values: Vec<Fp2>, pub splits: Vec<Fp2> }
```

- **`leaf_groups`.** Absorbed with `Round1`, before any challenge, so a prover cannot choose its grouping after seeing a challenge. The verifier rejects `leaf_groups > 14` or `leaf_groups > t` with `Reject(Malformed { artifact: Round1, code: ERR-CODEC-LIMIT })`.
- **`g = 0`.** `groups` is empty, `merge` has no rounds and no evaluations, nothing is absorbed for either, and `levels` and `splits` are as now.
- **`g ≥ 1`.**
  - `levels` holds the table batch's levels, and `groups` one entry per group.
  - `merge` holds the merge.
  - `splits` holds RFC-0037's atom splits.
  - `LogUpProof.roots` holds `2·(G + 1)` values per instance, in the order 'The leaf stage with groups' states, and `LogUpProof.values` is empty: each group's are in its `GroupProof`.
- **Rejections.** A level of a group or of the table batch rejects as RFC-0033's `TreeLevel { level, round }`, with its own level. The first failing check in the order above is the reason.

### Reduction stages

#329 asked for the reduction stages' working sets against the same budget. Instances whose relation is linear are answered by splits ([RFC-0037](RFC-0037-stage-schedule.md#splits)) and hold no tables, and the matrix products' tables are small after the folds P3 computes. The elementwise products, such as these, run their sumchecks over a stacked tensor of every layer and token:

| Instance (405B, 16K tokens) | Entries | At 32 B | Streamed rounds on 0.2 TB |
|---|---:|---:|---:|
| `a·u`, the MLP's gate and up | `T·d_ff·n_layers ≈ 1.1×10^11` | 3.5 TB | 5 |
| the norms' scaling, two per layer | `2·T·d·n_layers ≈ 6.8×10^10` | 2.2 TB | 4 |
| `R_q·PV`, the attention output's rescale | `T·d·n_layers ≈ 3.4×10^10` | 1.1 TB | 3 |

They do not depend on context. Under RFC-0010 'Sumcheck messages' they stream, one regeneration per round, and RFC-0018's P3 estimate of at most one forward pass in total does not hold. This RFC does not change the stages. Groups and a merge per stage, or groups carried through the stages to the openings, are the candidates. [#452](https://github.com/AbdelStark/tessera/issues/452) prices them and fixes the design. Until then the cost model does not count them, and every projection that quotes it says so.

### The executable model

`tessera-bench`'s `leaf_plan` gives a batch's groups and its table rounds, and `shared_streaming_passes` the design this RFC replaces. `mu`'s regeneration term adds the merge's pass when it counts regenerations. Its streaming term becomes the merge's folds, at `c_m` (`merge_ps`) per merged element, about one per leaf, plus the table rounds at `c_s` per table leaf ([RFC-0020](RFC-0020-cost-model.md#executable-cost-model)).

At 405B, `minimal`, a 16K-token batch, the corrected `μ(1K)` and `μ(16K)` become about 11.8 and 27.1 with the merge's regeneration. At an estimated 2 to 4 ps per element, the folds add 0.16 to 0.32 at 1K and 0.82 to 1.63 at 16K. The table rounds add under 0.005.

## Alternatives Considered

1. **RFC-0018 as written: one batch, every level streamed.** The table in 'Motivation' gives 36 to 55 regenerations. Rejected.
2. **Shared challenges with windowed rounds.** One pass can answer `w` rounds of a level from a `3^w`-point grid, at about `(3/2)^w` times a round's arithmetic per leaf. Later windows are cheaper, since a pass folds the bound variables first. The windows cut a level's passes to two or three, but not below one per level. The leaf stage would still pay 13 to 19 regenerations at 405B, and at 16K a window's arithmetic outweighs the passes it saves. Rejected. It remains a fallback for the table batch, which regenerates nothing.
3. **Smaller proof units.** Each unit carries every table tree and pays the table side once, and its own advice commitment adds combined rows ('Motivation'). Rejected for the leaf stage. RFC-0018 keeps units for other reasons.
4. **Store the leaves, on NVMe or in host memory.** The keys alone take about 2.5 TB per batch at 4 bytes. An NVMe pass over them is slower than regenerating, and 1 TiB of host memory holds less than half. Rejected. Host memory still helps the merge's tables and its folded state at 128K.
5. **Committed LogUp helper columns for the linear layers' lookups.** One sumcheck instead of a tree, but it commits every helper element at `c_c`, which costs more than the trees' arithmetic. And the zero-check over them streams as a tree's level does. Rejected.
6. **Groups whose claims stay at `G` points through the stages.** This would save the merge's pass. But a weight's claims would land at `G` points, and each costs a pass over the resident weights. And each committed tensor's `G` claim groups cost a combined row each, 64 MB at `k = 2^22`. Rejected for the leaf stage. It is one of #452's candidates for the reduction stages.
7. **Groups by layer instead of by token.** A layer's leaves at 16K are about 1.2 TB, so groups must split tokens anyway. And regenerating a set of layers for every token repeats each layer's earlier sites in every group. Rejected.

## Drawbacks

- **One more regeneration** than a leaf stage that fits, at every size. At 405B that is +1.0 on `μ`, which puts `minimal`'s 1K projection at its ceiling and `servable`'s 6% above it before the column answers.
- **Proof size.** About 120 KB of messages per group at 405B: 32 levels of at most 32 rounds, 27 instances, and about 2,500 split values. That is about 30 MB at 1K, 120 MB at 16K and 1 GB at 128K, against 08 §5's 2 GiB target and 4 GiB gate.
- **Soundness.** The union row loses about 5 to 9 bits and stays above 100.
- **Verification.** It checks `G` sets of levels, about `10^5` field operations each, and one weight per class and group for the merge. Milliseconds to a second.
- **No zero knowledge yet** for grouped proofs (OQ-0054-1).
- **The reduction stages are still unpriced** (#452).

## Migration / Rollout

- **v0.3.** The CPU prover plans `g` from `ProverConfig.memory_budget` and proves grouped stages. `tv-core` verifies them. Both soundness calculators take `G` and the merge's sizes, and the transcript, protocol and tamper vectors are regenerated.
  - Until each part lands, a prover keeps `g = 0` and a verifier rejects `g ≥ 1` with `ERR-CODEC-LIMIT`.
  - [#134](https://github.com/AbdelStark/tessera/issues/134) implements the groups and the table batch's streaming in place of RFC-0018's per-level streaming. [#132](https://github.com/AbdelStark/tessera/issues/132)'s planner chooses `g`.
- **v0.5.** The GPU backend proves the groups. The merge tables live in HBM or host memory, as the planner fits them.

## Testing Strategy

- **`g = 0`.** The proof of every batch that fits is RFC-0010's, byte for byte, but for `Round1`'s `leaf_groups` byte. Every existing leaf-stage test passes on it unchanged.
- **Groups.**
  - **Verify.** A tiny model, forced to a memory budget that takes `g` = 1, 2 and 3, produces proofs that the independent verifier accepts, its leaf claims at the merge's point.
  - **Equivalence.** The merged claims equal the tensors' evaluations at that point.
  - **Determinism.** Two runs with the same seed are byte-identical.
- **Adversarial.**
  - **The identity.** A false group root, a zero denominator, and a group root moved to another group each fail RFC-0008's identity or its check.
  - **Levels.** A changed level message in any group or in the table batch is rejected at that level.
  - **Splits.** A changed split value in any group is rejected by its group's check or by the merge.
  - **The merge.** A changed merge round or evaluation, or a merged claim swapped between two tensors of one class, is rejected.
  - **`leaf_groups`.** Above 14 or above `t`, it is `ERR-CODEC-LIMIT`.
- **The merge prover.**
  - **Two passes.** The table-backed rounds and the merge pass give the same messages as an in-memory merge on every class.
  - **Boundaries.** That holds at `j*` = 0, at the largest `j*` that fits, and at `j* = m`, which needs no merge pass.
  - **Classes.** A tensor without group variables merges in memory.
- **Memory.** Under a forced budget, the group pass and the merge pass never exceed it, and the pass counter counts P2 and P2b once each.
- **The model.** `leaf_plan` doubles its groups per doubling of the leaves beyond the memory and counts the table batch's rounds. `shared_streaming_passes` counts `c(c + 1)/2`. The documents' figures are held by `tessera-bench`'s tests.

## Open Questions

- **OQ-0054-1: zero knowledge.** The groups' roots are partial LogUp sums, and their split values are evaluations at points with Boolean coordinates in `S`. A masked extension vanishes there ([RFC-0013](RFC-0013-zero-knowledge.md), OQ-0013-2). `tessera-protocol/2` needs:
  - each group's levels masked as [RFC-0032](RFC-0032-zero-knowledge-tree-levels.md) masks the stage's;
  - the identity proved over all groups' roots without sending them;
  - the splits masked by values that cancel in the merge.

  Owner: @AbdelStark. Resolve before v0.4 grouped proofs.
- **OQ-0054-2: fusing the merge's pass.** The merge's pass is the only regeneration this RFC adds. A schedule that carries the groups' claims into stage 1's folds, where #452's design regenerates anyway, could share it. Resolve with #452.

## References

- [#329](https://github.com/AbdelStark/tessera/issues/329), its first-order table, and its three directions; this RFC is its first.
- RFC-0008, RFC-0010, RFC-0011, RFC-0013, RFC-0014, RFC-0018, RFC-0020, RFC-0032, RFC-0034, RFC-0036, RFC-0037, RFC-0045.
- Gruen, "Some Improvements for the PIOP for ZeroCheck", ePrint 2024/108; Dao and Thaler, "More Optimizations to Sum-Check Proving", ePrint 2024/1210: the split-`eq` level kernel the groups keep.
- Baweja, Chiesa, Fedele, Fenzi, Mishra, Mopuri and Zitek-Estrada, "Time-Space Trade-Offs for Sumcheck", ePrint 2025/1473; Bagad, Dao, Domb and Thaler, "Speeding Up Sum-Check Proving", ePrint 2025/1117: the windowed rounds of Alternative 2.
- `research/surveys/SUR-0003`, `SUR-0007`: the streaming and round-batching surveys, and the tile-fold idea this RFC's merge makes concrete.
