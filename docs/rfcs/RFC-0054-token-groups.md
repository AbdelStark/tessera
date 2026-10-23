# RFC-0054: Token Groups: a Leaf Stage Larger Than Memory

- Status: Accepted
- Authors: @AbdelStark
- Created: 2026-10-23
- Target milestone: v0.3 (CPU prover and verifier), v0.5 (GPU backend)
- Amended by: [RFC-0055](RFC-0055-windowed-stages.md) ('Reduction stages', 'The executable model', 'Drawbacks', OQ-0054-2: the reduction stages larger than memory, in windowed rounds); [RFC-0056](RFC-0056-native-working-sets.md) ('Reduction stages': stage 1's score instances without the score tensor)
- Supersedes:
  - [RFC-0018](RFC-0018-gpu-prover-runtime.md#pass-structure) 'Pass structure', 'Leaf levels' and 'Proof units' as they bear on the leaf stage, its P2 row, Alternative 2 and OQ-0018-1;
  - [RFC-0010](RFC-0010-proof-schedule.md#tree-levels) 'Tree levels' and 'Sumcheck messages' for a leaf stage that is grouped: its instances, labels and order;
  - [RFC-0008](RFC-0008-logup-and-tables.md#batching) 'Batching' and 'Fraction tree' for a grouped leaf stage: one lookup tree per table and group, the roots and their check;
  - [RFC-0014](RFC-0014-interaction-and-soundness.md#transcript-schedule) 'Transcript schedule', rows 4 and 5, and the soundness calculator's union row;
  - [RFC-0036](RFC-0036-protocol-messages.md#round1) `Round1`, `LogUpProof` and `TreeProof`;
  - [RFC-0033](RFC-0033-tree-level-rejections.md) and [04 §2](../spec/04-error-model.md#2-verdicts): `TreeLevel` gains `group`, and `RejectReason` gains `LeafMerge`, both with the verifier's implementation;
  - [RFC-0037](RFC-0037-stage-schedule.md#splits) 'Splits' and [RFC-0045](RFC-0045-leaf-layout.md#the-split) 'The split', for a grouped stage: RFC-0045 applies to each group, and the leaf stage's values come from the merge;
  - [RFC-0020](RFC-0020-cost-model.md#executable-cost-model) 'Executable cost model': the streaming term.
- Scope: `tessera-protocol/1`. `tessera-protocol/2` keeps one group until OQ-0054-1 is answered.
- Origin: [#329](https://github.com/AbdelStark/tessera/issues/329)

## Summary

The leaf stage proves every LogUp fraction tree and every product forest in one batched sumcheck per level. At 405B its trees hold `K + 2·A·S` leaves per token, 5.5×10^7 at 1K context and 3.0×10^8 at 16K: 9×10^11 and 5×10^12 for a 16K-token batch, against 0.2 TB of free HBM. RFC-0018 streams every level that does not fit, one regeneration pass per round. Counted over every tree and every level, that is 36 passes at 1K and 55 at 16K, and adds 53 to 235 to `μ`.

This RFC proves such a stage in **token groups** instead. When the stage's lookup trees and product forests do not fit, the prover splits them into `G = 2^g` groups whose trees do: interleaved subsets of the batch's tokens, at least 256 tokens each so that their GEMMs stay compute-bound, and interleaved subsets of the layers. Each group's trees are proven in memory, with challenges of its own, one group after another. The table trees, whose leaves come from the tables' rows and the stored multiplicities, are proven in a batch of their own and stream without regenerating anything. A **merge** sumcheck then reduces the groups' leaf claims, their tensors' values and their forests' leaves, to one point per polynomial. The reduction stages, the claim pool and the openings are unchanged.

The stage then costs two regenerations at any size: the group pass, which is RFC-0018's P2, and the merge's pass. At 405B that is one more regeneration than a stage that fits, and about 0.2 to 1.6 on `μ` for the merge's arithmetic. The price is in the transcript: `G` sets of level messages, about 120 KB each at 405B, and a union bound about 9 bits lower at the most groups a verifier accepts. With `g = 0` the leaf stage is RFC-0010's.

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

- **The reduction stages' working sets.** They exceed the budget at 405B too ('Reduction stages'); [RFC-0055](RFC-0055-windowed-stages.md) proves them in windowed rounds.
- **Zero knowledge for groups** (OQ-0054-1). `tessera-protocol/2` proofs keep `g = 0`.
- **Multi-node proving.**

## Proposed Design

### When groups apply

The prover's planner chooses the groups before P1, from public sizes and its memory budget `F` (`ProverConfig.memory_budget`, RFC-0018 'Memory scheduler'):

- **Bytes.** `B` is the bytes a level holds per node, 32: two `Fp2` values. A leaf also keeps the tensor value it reads, one integer of 8 bytes. That value stays until the leaf point is known and the tensors' values at it are computed (RFC-0045 'The split').
- **Leaves.** `L_tok` is a token's leaves: `K` lookups, plus one `Exp` lookup and one `SoftmaxRow` leaf per valid score, `K + 2·A·S`. Masked scores are constant subtrees (RFC-0034) and hold none.
- **One batch.** While the batch's lookup trees and product forests fit, `T·L_tok·(B + 8) ≤ F`, the stage takes one group (`g = 0`). It is RFC-0010's single batch. Its levels that do not fit with the table trees stream those alone ('The table batch').
- **Groups.** Otherwise the stage takes the fewest groups `G = 2^g` whose lookup trees and forests fit, `(T·L_tok/G)·(B + 8) ≤ F`, split as 'Groups' states.
- **Infeasible plans.** A plan that needs `g` above 14, or a group smaller than its split allows, fails before `Round1` with `ERR-RES-MEMORY`.

The plan runs before P1 because the witness pass counts each group's column padding in row 0's multiplicity. The verifier reads the plan from `Round1` and checks only its bounds ('Messages').

At 405B on 0.2 TB, a 16K-token batch takes `G` = 256, 512, 1,024 and 8,192 groups at 1K, 4K, 16K and 128K context.

### Groups

A group fixes `g_T` bits of the token index and `g_L` bits of the layer index, `g = g_T + g_L`. Group `h` sets them to `h`'s bits: the token bits first, then the layer bits, each lowest first.

- **Tokens, interleaved.** The batch's token index is its requests' positions in claim-list order, per-dimension padded to `2^t` ([RFC-0011](RFC-0011-advice-commitment.md#canonical-advice-layout) rule 1). Token subset `s` holds the tokens whose low `g_T` index bits are `s`.
  - **Why interleaved.** Every subset then holds queries at every position, so causal attention loads every group alike. A contiguous range of queries would put about twice the average scores into the last group.
  - **Score tensors.** In a score tensor the token axis is the query axis. Keys are not split.
- **Layers, interleaved.** The template's layer index, padded to `2^{n_ℓ}`, splits the same way: layer subset `o` holds the layers whose low `g_L` bits are `o`. On a node partitioned by layer range (RFC-0018 'Partitioning'), every GPU then holds a share of every group's layers.
  - Prologue and epilogue blocks belong to layer subset 0.
- **Compute-bound groups.** A group's GEMMs multiply the weights of its layers by its tokens. An INT8 GEMM of `n` tokens performs `2n` operations per weight byte, against about 355 for H100 at 60% of its rate over 3.35 TB/s. The planner therefore keeps at least 256 tokens per group and splits layers instead.
  - Each layer's weights are then read once per token subset, `2^{g_T}` times in all.
  - At 405B every plan above takes 64 token subsets of 256 tokens: about 1 s of weight reads per GPU, under the pass's 1.4 to 1.5 s of compute.
- **What a group holds.** Group `h` holds the blocks of [RFC-0045](RFC-0045-leaf-layout.md#blocks) at its layer subset's layers, restricted to its token subset's tokens.
  - **No token axis.** A block whose domain has no token axis belongs to token subset 0.
  - **Lookup trees.** For each LogUp instance, the group's lookups form one lookup tree, laid out by RFC-0045 over the group's blocks: the column, the leaf order, the leaf point and the split. A group that holds no block of an instance has no tree for it and sends no root.
  - **The forest.** The group's `SoftmaxRow` forest is the trees of its layers' heads and its tokens' query rows, `2^{s−g}` trees with their public root values.
  - **Shapes.** Groups differ in shape where their blocks differ: prologue, epilogue and padded layers, and blocks without a token axis. The verifier derives every group's layout from public data, as it derives the stage's.
- **Table trees** are not grouped: one per table, as RFC-0008 states, in the table batch.
- **Group variables.** In a tensor read by a group's lookups, or in a forest's leaf table, the group fixes some of its variables: its group variables `S`.
  - These are the token axis's low `g_T` bits.
  - For a table stacked over layers, as the forest's leaf table is, they also include the layer index's low `g_L` bits.
  - A per-layer tensor is read by the `2^{g_T}` groups whose layer subset holds its layer, and group `h`'s block of it is the tensor with `S` set to `h`'s token bits.

### The leaf stage with groups

With `g ≥ 1`, the leaf stage runs in this order:

1. **Challenges.** `logup/alpha` and `logup/beta`, as RFC-0008 states. `logup/roots`, then `logup/values`, are absorbed empty, right after `logup/beta`.
2. **Each group `h`, in order from 0.**
   - **Roots.** The roots `(n, d)` of the group's lookup trees, in schedule order, under `group/{h}/logup/roots`. The group's value evaluations follow the same order. The verifier checks every `d ≠ 0`, else `Reject(Lookup)`.
   - **Levels.** RFC-0010 'Tree levels' for the group's instances, under `group/{h}/tree/{ℓ}/root/{c}`, `group/{h}/tree/{ℓ}/gamma`, `group/{h}/tree/{ℓ}/lambda`, `group/{h}/tree/{ℓ}/round/{j}`, `group/{h}/tree/{ℓ}/r/{j}`, `group/{h}/tree/{ℓ}/evals` and `group/{h}/tree/{ℓ}/line`. `ℓ` counts the group's levels from its root, and each lookup tree enters with its root as its claims.
   - **Values and splits.** The group's lookup trees' value evaluations under `group/{h}/logup/values`, then its split values under `group/{h}/split`. The verifier checks the leaf claims against them as RFC-0008 and RFC-0045 check the stage's.
   - **Merge challenge.** `γ_h` is squeezed under `group/{h}/merge/gamma` ('The merge').
3. **The table batch.**
   - **Roots.** The table trees' roots under `tables/logup/roots`, with `d ≠ 0` checked.
   - **The identity.** For each instance, the verifier sums its groups' fractions and its table root's by `(n, d) ⊕ (n', d') = (n·d' + n'·d, d·d')`, in group order then the table root. It checks that the sum's numerator is 0, else `Reject(Lookup)`. That is RFC-0008's identity over every group's lookups.
   - **Levels.** RFC-0010 'Tree levels' for the table trees alone, under `tables/tree/{ℓ}/…`. Their leaf claims go to RFC-0008's final checks, the multiplicities' to the pool, as now.
4. **The merge** ('The merge'), under `merge/beta`, `merge/round/{j}`, `merge/r/{j}` and `merge/evals`. It takes every group's split values and every group's forest leaf claims. Its final evaluations are the stage's outputs at the merge's point:
   - one value per tensor, site and real layer, in place of the ungrouped stage's lookup tensor values;
   - one claim per forest leaf table, the leaf claim stage 1 consumes ([RFC-0010](RFC-0010-proof-schedule.md#tree-levels), RFC-0030 relation 2).
5. **The atom splits.** [RFC-0037](RFC-0037-stage-schedule.md#splits)'s atom split values of the merged claims follow under `stage/0/split`.

Each group's roots are absorbed before its own levels, and `α` before every root. A group's roots therefore depend only on its own leaves, which its regeneration produces: no pass is needed to compute them in advance.

With `g = 0`, the leaf stage's messages, labels and checks are RFC-0010's and RFC-0008's.

### The table batch

The table trees' leaves are `(−m_j, α − t_j)`: the table's row from its generator, and the multiplicity that P1 counted and stores (08 §4: 19 to 42 GiB at 405B). A round over them regenerates no witness.
- **Streaming.** Their levels run as RFC-0010's, with RFC-0018's streaming rule for this batch alone. A level that does not fit streams one round per halving, each recomputing the table leaves, until its folded state fits.
- **Count.** That is `c_T(c_T + 1)/2` rounds over the levels, with `c_T = ⌈log2(M·B/F)⌉` for the tables' `M` rows.
- **At 405B,** `c_T` = 1: 6.7×10^9 rows hold 213 GB, which halves to fit 0.2 TB, so the table trees stream one round.

With `g = 0`, the table trees stay in the stage's one batch, and its levels that do not fit stream the same way. The lookup trees and forests, which fit, stay resident.

### The merge

The groups end with each merged polynomial's claims at the points of the groups that read it. A merged polynomial is a tensor that a group's split values evaluate, or a forest's leaf table. Group `h` claims `P̃(z_h ‖ h_S) = v_h` on polynomial `P`.
- `z_h` assigns its point to `P`'s variables outside `S`.
- `h_S` sets the coordinates in `S` to `h`'s bits.
- A polynomial with `S` empty, a tensor without a token axis, has its claims at `z_h` over all of its variables, from the groups that read it.

The merge reduces every polynomial's claims to one, at one point, in one sumcheck.

**Batching.**
- `γ_h` is squeezed after group `h`'s splits, under `group/{h}/merge/gamma`, and `β` after the table batch, under `merge/beta`.
- With claim `k` of group `h` on polynomial `P_{i(k)}` of `n_i` variables, the merge proves `Σ_h β^h·Σ_k γ_h^k·2^{n_max − n_i}·v_{k,h}`, front-padded, of `Σ_x Σ_i W_i(x)·P̃_i(x)`.
- **Weights.** `W_i` sums over the groups `h` that read `P_i`, its claim there being the `k`-th:
  - with group variables, `W_i(x) = Σ_h β^h·γ_h^k·eq(z_{i,h}, x_{S̄_i})·eq(h_S, x_{S_i})`;
  - with `S_i` empty, `W_i(x) = Σ_h β^h·γ_h^k·eq(z_{i,h}, x)`.
- **Why it holds.** On the Boolean cube, `W_i` is `β^h·γ_h^k·eq(z_{i,h}, ·)` on group `h`'s block and 0 elsewhere, so `Σ_x W_i·P̃_i` is `Σ_h β^h·γ_h^k·P̃_i(z_{i,h} ‖ h_S)`.
- **Soundness of the batching.** If any `v_{k,h}` is false, group `h`'s combination is a nonzero polynomial of degree below `P_h` in `γ_h`, which was fixed after its claims. The outer sum is then nonzero in `β` but for degree below `G`: an error of `(P + G)/|Fp2|` at most, `P` the largest group's claims.

**Rounds.**
- **Variable order.** The rounds bind each polynomial's variables outside `S_i` first, lowest first, then those in `S_i`, lowest first, and `P̃_i` and `W_i` are read in that order. Front-padding gives a polynomial of `n_i` variables the last `n_i` challenges. Polynomials of one size and one `S` therefore lie at one point, as RFC-0037's layer point needs a site's layers to.
- **Messages.** The degree is 2. Round `j` sends `p_j(0)` and `p_j(2)` under `merge/round/{j − 1}` and squeezes `merge/r/{j − 1}`, as [RFC-0010](RFC-0010-proof-schedule.md#sumcheck-messages) 'Sumcheck messages' states.
- **The final check.** The prover sends `P̃_i(r_i)` for every polynomial, in claim order, under `merge/evals`. The verifier computes every `W_i(r_i)` itself and checks `Σ_i W_i(r_i)·P̃_i(r_i)` against the last claim.
  - A failed round, or the final check at round `n_max`, is `Reject(LeafMerge { round })`.
  - The verifier's work is one `eq` product per point class and group, and one sum per polynomial over the groups that read it: about `10^7` multiplications at 405B and 16K.
- **The result.** `P̃_i(r_i)` is `P_i`'s claim at `r_i`. For a tensor, it is the value RFC-0037 takes from the stage for that tensor, site and layer. For a forest's leaf table, it is the leaf claim the consuming stage proves.

**Point classes.** Within a group, RFC-0045 puts every block of one size at one point, `P_h[0..k]`. Polynomials read over the same axes therefore share `z_{i,h}`: a point class `c`. At 405B there are a few tens of classes, against about 3,000 claims per group.

**The prover in two passes** (informative). For a class `c` and group `h`, write `Q_{c,h} = Σ_{k∈c} γ_h^k·P_{i(k)}`, over its variables outside `S` and in `S`. Summing over the Boolean `S` variables and over the variables after round `j` collapses group `h`'s share of round `j ≤ m` to an evaluation:

```text
p_j(Z) = Σ_c Σ_h β^h · eq(z_{c,h,<j}, c_{<j}) · eq(z_{c,h,j}, Z) · Q̃_{c,h}(c_{<j}, Z, z_{c,h,>j}, h_S)
```

- **Group pass.** `γ_h` and `z_{c,h}` are known at the end of group `h`, while its tensors are still resident. The prover then keeps, for each `j ≤ j*`, the table `T_{c,h,j}(a, b) = Q̃_{c,h}(a, b, z_{c,h,>j}, h_S)` of `2^j` entries. Round `j ≤ j*` reads its share from `T_{c,h,j}` in `2^j` operations.
- **Merge pass.** After round `j*`, one pass regenerates every merged polynomial and folds its first `j*` variables at `c_{≤j*}`. The remaining rounds run in memory, and the folded state is per polynomial, since each one's evaluation is sent.
- **Memory.** The tables take `2^{j*+1}` entries per class and group, and the folded state at most about `N_m·32/2^{j*}` bytes for `N_m` merged elements. `j*` is chosen so that both fit.
  - At 405B, 16K-token batch, 16K context: `N_m ≈ 4.9×10^12`, so `j* ≥ 10`, and the tables take about 1 GB for 30 classes and 1,024 groups.
  - At 128K, `j* = 13` with about 64 GB of tables in host memory.
- **Lower bound.** No merge to one point avoids a second pass. Its point must be drawn after every group's claims are fixed, and the claim at a fresh random point is a new linear combination of every entry of `Q`, which a prover that cannot store `Q` must read again.

### Pass structure

- **P2, the group pass.** For each group in order, the prover regenerates the group's tokens at its layers from the layer checkpoints. It then builds and proves the group's trees in memory, sends its roots, values and splits, and keeps its merge tables.
  - Attention needs every earlier token's keys and values, so P1 keeps them for the batch beside the residuals: `2·n_layers·d_kv` bytes per token, 4.2 GB for 16K tokens at 405B, 0.5 GB per GPU.
  - The groups' GEMMs stay compute-bound ('Groups'), so P2 costs one regeneration.
- **P2b, the merge pass.** One regeneration of every merged polynomial, after the merge's first `j*` rounds.
- The table batch's streamed rounds regenerate nothing.

A leaf stage that fits costs P2 alone. A grouped one costs P2 and P2b, whatever `G`: six regenerations in RFC-0018's count instead of five.

### Soundness

Each check is RFC-0010's or RFC-0008's, applied per group, so RFC-0014's rows keep their form. They gain items:

- **Rounds.** The groups' levels and the table batch's levels are levels. The merge is a stage of `n_max` rounds of degree 2, batching with error `(P + G)/|Fp2|` ('Batching').
- **The union row** sums every one of them: about `G` times the leaf stage's level terms.
  - At RFC-0014's default 405B statement, the leaf stage's levels add about 5·10^3 to the union's `7.3·10^4`.
  - With `G = 2^10` they add about `2.4·10^6`, and the union row is about 106.7 bits.
  - With `G = 2^14`, the most a verifier accepts, it is about 103.1 bits: about 102.2 at RFC-0014's conservative degree 7. Both are above the 100-bit target.
- **The bound covers every accepted proof.** The prover chooses `g`, and a cheating prover chooses the largest. The soundness calculators of both workspaces therefore evaluate the union row at the largest `G` the verifier accepts for the statement, `2^{min(14, t + n_ℓ)}`. They take the merge's `n_max` and `P` as sizes.
- **The product-forest roots** keep `s/|Fp2|`, with each group's `s − g`.
- **The LogUp identity** keeps `v·max(N + T)/|Fp2|`: the groups' roots and the table root sum to one rational identity in `α` of the same degree. Each group's roots follow `α`, and each precedes its group's own challenges.
- **The splits** are exact checks. A false value in any group's split is a false claim on its polynomial. The merge rejects it except within its bound, or carries it to a false merged claim, which the next stage rejects within its own.
- **Moving a claim.** A lookup or a claim moved to another group lands on the wrong block of its polynomial, which `eq(h_S, x_S)` in `W_i` separates.

### Messages

[RFC-0036](RFC-0036-protocol-messages.md)'s layouts change in place: `tessera-protocol/1` is still unassigned, `protocol_version` 0.

```rust
pub struct Round1 { pub protocol_version: u16, pub session_id: [u8; 16], pub statement: PublicStatement,
                    pub advice: AdviceCommitment, pub leaf_groups: LeafGroups }
pub struct LeafGroups { pub token_bits: u8, pub layer_bits: u8 }          // g_T, g_L
pub struct TreeProof { pub levels: Vec<SumcheckProof>, pub groups: Vec<GroupProof>,
                       pub table_roots: Vec<Fp2>, pub merge: SumcheckProof, pub splits: Vec<Fp2> }
pub struct GroupProof { pub roots: Vec<Fp2>, pub levels: Vec<SumcheckProof>, pub values: Vec<Fp2>,
                        pub splits: Vec<Fp2> }
```

- **`leaf_groups`.** It is absorbed with `Round1`, before any challenge, so a prover cannot choose its grouping after seeing one.
  - The verifier rejects, with `Reject(Malformed { artifact, code: ERR-CODEC-LIMIT })`, `artifact` being `Round1` in live mode and `NiProof` otherwise: `token_bits + layer_bits > 14`, `token_bits > t`, or `layer_bits > n_ℓ`. The first is a decoding failure; the others need the claim list and the statement. In `tessera-protocol/2` it also rejects any `leaf_groups` other than zero (OQ-0054-1).
- **`g = 0`.** `groups` and `table_roots` are empty, `merge` has no rounds and no evaluations, nothing is absorbed for them, and `levels`, `splits` and `LogUpProof` are as now.
  - The leaf stage is RFC-0010's, but the proof is not byte-identical to today's. `Round1`'s new bytes change every challenge, and `TreeProof`'s new fields its encoding.
- **`g ≥ 1`.**
  - `levels` holds the table batch's levels, `groups` one entry per group, and `table_roots` the table trees' roots.
  - `merge` holds the merge, and `splits` RFC-0037's atom splits.
  - `LogUpProof.roots` and `LogUpProof.values` are empty: each group's are in its `GroupProof`.
- **Rejections.** These change [RFC-0033](RFC-0033-tree-level-rejections.md) and 04 §2; no artifact encodes `RejectReason` yet, so they change in place.
  - `TreeLevel` gains `group`: `TreeLevel { group, level, round }`. `group` is `h` for group `h`'s levels, `G` for the table batch's, and 0 for an ungrouped stage.
  - A group's split mismatch is `TreeLevel { group: h, level: L_h, round: 0 }`, `L_h` the group's level count.
  - The merge's is the new `LeafMerge { round }` (`REJ-LEAF-MERGE`, exit 10).
  - The atom splits keep `Sumcheck { stage: 0, round: 0 }`.
  - Each reason names one check, and the first failing check in the order above is the one reported.

### Reduction stages

#329 asked for the reduction stages' working sets against the same budget. They exceed it at 405B too. The compiler's schedule gave `minimal`'s stages 1 and 2 4.1 and 5.5 TB of tables at 16K tokens, and `servable`'s stage 1 9.6 TB, against 0.2 TB. With the native relations counted ([RFC-0056](RFC-0056-native-working-sets.md#at-405b)), they hold 6.4, 6.6 and 11.9 TB at 16K of context, stage 1 a little more at longer contexts. This RFC does not change the stages. [RFC-0055](RFC-0055-windowed-stages.md) proves such a stage in windowed rounds, with RFC-0010's messages and one more regeneration per stage. It sizes the stages from the compiler's report, which replaced the first-order table, from the model's shape, that this section first gave. Stage 1's native relations, whose score instances read the same scores as this RFC's forests, are counted, and those instances proven without the score tensor, by [RFC-0056](RFC-0056-native-working-sets.md).

### The executable model

`tessera-bench`'s `leaf_plan` gives a batch's groups and its table rounds, sizing a group by its level tables and its leaves' tensor values (`TENSOR_BYTES_PER_LEAF`). `shared_streaming_passes` counts the design this RFC replaces.

`mu` uses the plan in two terms ([RFC-0020](RFC-0020-cost-model.md#executable-cost-model)):
- **The regeneration term** adds the merge's pass when it counts regenerations.
- **The streaming term** prices the merge's folds, at `c_m` (`merge_ps`) per merged element, about one per leaf, and the table rounds, at `c_s` per table leaf.

The plan counts `G` alone, not how the planner splits it: every 405B plan above keeps 256-token groups.

At 405B, `minimal`, a 16K-token batch, the corrected `μ(1K)` and `μ(16K)` become about 11.8 and 27.1 with the merge's regeneration. RFC-0055's windows add two more, for about 13.8 and 29.1, with every native relation counted (RFC-0056). At an estimated 2 to 4 ps per element, the folds add 0.16 to 0.32 at 1K and 0.82 to 1.63 at 16K. The table rounds add under 0.005.

## Alternatives Considered

1. **RFC-0018 as written: one batch, every level streamed.** The table in 'Motivation' gives 36 to 55 regenerations. Rejected.
2. **Shared challenges with windowed rounds.** One pass can answer `w` rounds of a level from a `3^w`-point grid, at about `(3/2)^w` times a round's arithmetic per leaf. Later windows are cheaper, since a pass folds the bound variables first. The windows cut a level's passes to two or three, but not below one per level. The leaf stage would still pay 13 to 19 regenerations at 405B, and at 16K a window's arithmetic outweighs the passes it saves. Rejected. It remains a fallback for the table batch, which regenerates nothing.
3. **Smaller proof units.** Each unit carries every table tree and pays the table side once, and its own advice commitment adds combined rows ('Motivation'). Rejected for the leaf stage. RFC-0018 keeps units for other reasons.
4. **Store the leaves, on NVMe or in host memory.** The keys alone take about 2.5 TB per batch at 4 bytes. An NVMe pass over them is slower than regenerating, and 1 TiB of host memory holds less than half. Rejected. Host memory still helps the merge's tables and its folded state at 128K.
5. **Committed LogUp helper columns for the linear layers' lookups.** One sumcheck instead of a tree, but it commits every helper element at `c_c`, which costs more than the trees' arithmetic. And the zero-check over them streams as a tree's level does. Rejected.
6. **Groups whose claims stay at `G` points through the stages.** This would save the merge's pass. But a weight's claims would land at `G` points, and each costs a pass over the resident weights. And each committed tensor's `G` claim groups cost a combined row each, 64 MB at `k = 2^22`. Rejected for the leaf stage. It is one of #452's candidates for the reduction stages.
7. **Groups by layer alone, or by tokens alone.**
   - **By layer.** A layer's leaves at 16K are about 1.2 TB, so groups must split tokens anyway.
   - **By tokens.** Groups of tokens alone are small at 405B: 16 tokens at 16K, 2 at 128K. Their GEMMs would then be bound by weight bandwidth, and the group pass would read every weight once per group, about ten forward passes at 16K.
   - **The choice.** Splitting both keeps 256 tokens per group.
8. **Contiguous token ranges.** Causal attention gives the last queries about twice the average scores, so the last group would need about twice the memory of the plan. Rejected for interleaved subsets.

## Drawbacks

- **One more regeneration** than a leaf stage that fits, at every size. At 405B that is +1.0 on `μ`, which put `minimal`'s 1K projection at its ceiling and `servable`'s 6% above it before the column answers. With RFC-0055's windows, both are 14 to 15% above it.
- **Proof size.** About 120 KB of messages per group at 405B: 32 levels of at most 32 rounds, 27 instances, and about 2,500 split values. That is about 30 MB at 1K, 120 MB at 16K and 1 GB at 128K, against 08 §5's 2 GiB target and 4 GiB gate.
- **Soundness.** At the most groups a verifier accepts, the union row loses about 9 bits and stays above 100.
- **Verification.** It checks `G` sets of levels, about `10^5` field operations each. For the merge, it computes one `eq` product per class and group and one sum per polynomial. Milliseconds to a second.
- **No zero knowledge yet** for grouped proofs (OQ-0054-1).
- **The reduction stages exceed memory too.** [RFC-0055](RFC-0055-windowed-stages.md) windows them, one more regeneration each.

## Migration / Rollout

- **v0.3.** The CPU prover plans `g` from `ProverConfig.memory_budget` and proves grouped stages. `tv-core` verifies them. Both soundness calculators take `G` and the merge's sizes, and the transcript, protocol and tamper vectors are regenerated.
  - Until each part lands, a prover keeps `g = 0` and a verifier rejects `g ≥ 1` with `ERR-CODEC-LIMIT`: `Reject(Malformed { artifact, code: ERR-CODEC-LIMIT })` at RFC-0019 step 1, `artifact` being `Round1` in live mode and `NiProof` otherwise. An ungrouped proof with a group, a table root or a merge round or evaluation is `Malformed` with `ERR-CODEC-NONCANONICAL` there too, its `Round2Response` in live mode: the grouping fixes those lengths ('Messages'). Both checks follow decoding, so a proof that fails to decode is `Malformed` with the codec's code first, and the grouping comes before the ungrouped fields.
  - [#134](https://github.com/AbdelStark/tessera/issues/134) implements the groups and the table batch's streaming in place of RFC-0018's per-level streaming. [#132](https://github.com/AbdelStark/tessera/issues/132)'s planner chooses `g`.
- **v0.5.** The GPU backend proves the groups. The merge tables live in HBM or host memory, as the planner fits them.

## Testing Strategy

- **`g = 0`.** A batch that fits proves RFC-0010's leaf stage. Its messages and checks are unchanged, though `Round1`'s and `TreeProof`'s new fields change its bytes. Every existing leaf-stage test passes on it once its vectors are regenerated.
- **Groups.**
  - **Verify.** A tiny model, forced to a memory budget that takes `g` = 1, 2 and 3, produces proofs that the independent verifier accepts, its leaf claims at the merge's point.
  - **Equivalence.** The merged claims equal the tensors' evaluations at that point.
  - **Determinism.** Two runs with the same seed are byte-identical.
- **Adversarial.**
  - **The forests.** A zero leaf in a forest of group `h ≥ 1`, at a valid position, is rejected: the merged leaf claim reaches stage 1 as the ungrouped stage's does.
  - **The identity.** A false group root, a zero denominator, and a group root moved to another group each fail RFC-0008's identity or its check.
  - **Levels.** A changed level message in any group or in the table batch is rejected at that level.
  - **Splits.** A changed split value in any group is rejected by its group's check or by the merge.
  - **The merge.** A changed merge round or evaluation, or a merged claim swapped between two tensors of one class, is rejected.
  - **`leaf_groups`.** Above 14, above `t` or `n_ℓ`, or nonzero in `tessera-protocol/2`, it is `ERR-CODEC-LIMIT`.
  - **Reasons.** Every grouped check's failure has its own reason: `TreeLevel` with its group, the split at its group's last level, and `LeafMerge` with its round.
- **Layouts.**
  - **Unlike groups.** Groups whose blocks differ, such as layer subset 0 with the prologue and the padded layers, verify.
  - **Empty groups.** An instance absent from a group has no tree there and no root, and its identity sums the groups that hold it.
  - **Interleaving.** A causal batch's groups hold their valid scores within one per group of each other.
- **The merge prover.**
  - **Two passes.** The table-backed rounds and the merge pass give the same messages as an in-memory merge on every class.
  - **Boundaries.** That holds at `j*` = 0, at the largest `j*` that fits, and at `j* = m`, which needs no merge pass.
  - **Classes.** A tensor without group variables merges in memory.
- **Memory.** Under a forced budget, the group pass and the merge pass never exceed it, and the pass counter counts P2 and P2b once each.
- **The model.** `leaf_plan` doubles its groups per doubling of the leaves beyond the memory and counts the table batch's rounds. `shared_streaming_passes` counts `c(c + 1)/2`. The documents' figures are held by `tessera-bench`'s tests.
- **The calculator.** It evaluates the union row at the largest `G` the verifier accepts, and a statement whose bound misses the target at that `G` fails the calculator's check.

## Open Questions

- **OQ-0054-1: zero knowledge.** The groups' roots are partial LogUp sums, and their split values are evaluations at points with Boolean coordinates in `S`. A masked extension vanishes there ([RFC-0013](RFC-0013-zero-knowledge.md), OQ-0013-2). `tessera-protocol/2` needs:
  - each group's levels masked as [RFC-0032](RFC-0032-zero-knowledge-tree-levels.md) masks the stage's;
  - the identity proved over all groups' roots without sending them;
  - the splits masked by values that cancel in the merge.

  Owner: @AbdelStark. Resolve before v0.4 grouped proofs.
- **OQ-0054-2: fusing the merge's pass.** The merge's pass is the only regeneration this RFC adds. [RFC-0055](RFC-0055-windowed-stages.md)'s windows do not share it: stage 1's grid pass needs the claims the merge leaves. A schedule that carries the groups' claims into stage 1, groups through the stages (RFC-0055 Alternative 3), could, at the costs that alternative lists. Open.

## References

- [#329](https://github.com/AbdelStark/tessera/issues/329), its first-order table, and its three directions; this RFC is its first.
- RFC-0008, RFC-0010, RFC-0011, RFC-0013, RFC-0014, RFC-0018, RFC-0020, RFC-0032, RFC-0034, RFC-0036, RFC-0037, RFC-0045.
- Gruen, "Some Improvements for the PIOP for ZeroCheck", ePrint 2024/108; Dao and Thaler, "More Optimizations to Sum-Check Proving", ePrint 2024/1210: the split-`eq` level kernel the groups keep.
- Baweja, Chiesa, Fedele, Fenzi, Mishra, Mopuri and Zitek-Estrada, "Time-Space Trade-Offs for Sumcheck", ePrint 2025/1473; Bagad, Dao, Domb and Thaler, "Speeding Up Sum-Check Proving", ePrint 2025/1117: the windowed rounds of Alternative 2.
- `research/surveys/SUR-0003`, `SUR-0007`: the streaming and round-batching surveys, and the tile-fold idea this RFC's merge makes concrete.
