# RFC-0010: Batched Constant-Depth Sumcheck/GKR Proof Schedule

- Status: Accepted
- Authors: @AbdelStark
- Created: 2026-10-15
- Target milestone: v0.3
- Amended by: [RFC-0025](RFC-0025-affine-sites.md) ('Stages': `servable` with an affine RoPE site), [RFC-0030](RFC-0030-masked-exponentials.md) ('Multilinear extensions': the causal selector's sequence length; 'Tree levels': `SoftmaxRow`'s root values), [RFC-0032](RFC-0032-zero-knowledge-tree-levels.md) ('Tree levels', in `tessera-protocol/2`: forests of two trees at least, masked levels), [RFC-0033](RFC-0033-tree-level-rejections.md) ('Sumcheck messages': a failed tree level's reason), [RFC-0035](RFC-0035-padded-rows.md) ('Relations': the padding zero-check; 'Batching within a stage': the zero-check point)

## Summary

After all advice is committed and the seed is fixed, the prover runs a fixed sequence of stages. Each stage is one batched sumcheck (or one batched GKR tree level) covering every layer, head and query block, with shared challenges. Claims flow from lookup and product-tree leaves back through virtual relations until they land on committed tensors, where they are batched into openings. Because every non-arithmetic output is committed, the longest virtual chain inside a layer is a handful of relations, so the number of stages is independent of the number of layers. Shared challenges put every layer's claims on one weight type at a single point, which is what lets the weight commitment open with one folding pass.

## Motivation

The source's "constant-depth proof" (source §7) is what removes the sequential dependency of layer-by-layer GKR: "the longest chain inside a layer is about six stages ... Each stage runs as one sumcheck batched over all layers, heads and query blocks with shared challenges. Proof depth no longer grows with model depth." The weight-commitment design depends on it: "Batching every layer's instance of a weight type with shared challenges lands all its weight claims on one common point per type" (source §7, "Weights opened by the folding pass"). This RFC turns those sentences into a schedule both sides derive identically.

## Goals

- A deterministic stage schedule derived from the constraint system by prover and verifier independently.
- Stage count independent of `n_layers`.
- Every claim on a committed tensor of a given kind lands on one point per stage.
- Sumcheck messages and their verification fully specified.

## Non-Goals

- Zero-knowledge masking of stages (RFC-0013).
- Kernel-level scheduling (RFC-0018).

## Proposed Design

### Multilinear extensions

A table of `2^n` values `f[0], …, f[2^n − 1]` defines the multilinear polynomial `f̃(x_1, …, x_n)` that equals `f[Σ_i x_i·2^{i−1}]` on `{0, 1}^n`: `x_1` is the least significant bit of the index. A point `r = (r_1, …, r_n)` sets `x_i = r_i`, and

```text
eq(r, x) = Π_i (r_i·x_i + (1 − r_i)·(1 − x_i))        f̃(r) = Σ_{x ∈ {0,1}^n} f[x]·eq(r, x)
```

- Tensors are flattened row-major, so their last index takes the lowest variables, and a stacked tensor's layer index the highest. Function notation such as `W̃(ℓ, o, i)` names the indices in row-major order. As a coordinate vector, low variables first, the same point is `r_i ‖ r_o ‖ ρ_ℓ`, as RFC-0011 writes `r ‖ base_X`.
- Sumcheck round `j` binds `x_j` to its challenge `r_j`: each pair `(f[2k], f[2k+1])` becomes `f[2k] + r_j·(f[2k+1] − f[2k])`. A stage's final point therefore lists its challenges in round order.
- Two public predicates have `O(n)` closed forms, which the prover and the verifier each evaluate:
  - `LT(r, a) = Σ_{h < a} eq(r, h)`, for an integer `0 ≤ a ≤ 2^n` ([RFC-0008](RFC-0008-logup-and-tables.md#structured-evaluation));
  - the causal selector of a sequence of `S` positions, `C(q, i) = Σ_{i' ≤ q' < S} eq(q, q')·eq(i, i')`, over pairs of `n`-bit positions ([RFC-0009](RFC-0009-gadgets.md#softmaxrow), [RFC-0030](RFC-0030-masked-exponentials.md#the-causal-selector)). Its closed form is a recursion over the bits, most significant first, that carries two comparisons: `q'` against `S`, and `i'` against `q'`. A tensor stacked over layers and heads multiplies it by `LT` of each, which vanishes on padded layers and heads.

### Relations

Every constraint in the derived constraint system is one of these relation kinds, each a sumcheck identity over the Boolean hypercube:

| Kind | Identity (claim on target at point `r`) | Degree per variable |
|---|---|---|
| `MatMulWeight` | `Ỹ(r_t, r_o) = Σ_i W̃(r_o, i) X̃(r_t, i)` | 2 |
| `MatMulAct` | `Z̃(r_a, r_c) = Σ_j Ã(r_a, j) B̃(j, r_c)` | 2 |
| `Elementwise` (`Mul`, scalar-by-row) | `Z̃(r) = Σ_x eq(r, x) Ã(x) B̃(x)` (constants evaluated by the verifier) | 3 |
| `Rotate` ([RFC-0037](RFC-0037-stage-schedule.md#rotate)) | `Ỹ(r) = Σ_x E_r(x)·X̃(x)`, `E_r` the public kernel of RoPE's pairs | 2 |
| `SumSquares` | `Ũ(r) = Σ_x eq(r, x_row) X̃(x)^2` | 3 |
| `Linear` (sums, adds, residual chains, selectors) | Evaluated directly from input claims at the same point; no sumcheck | — |
| `ZeroCheck` (`OneHot` booleanity) | `0 = Σ_x eq(z, x) h̃(x)(1 − h̃(x))` | 3 |
| `ZeroCheck` (padding, [RFC-0035](RFC-0035-padded-rows.md#the-padding-zero-check)) | `0 = Σ_x eq(z, x)·(1 − ψ̃(x))·w̃(x)`, `ψ` the public row selector | 3 |
| `ProductTreeLevel` | `V_ℓ(r) = Σ_x eq(r, x) V_{ℓ+1}(x, 0) V_{ℓ+1}(x, 1)` | 3 |
| `FractionTreeLevel` | LogUp-GKR level (RFC-0008) | 3 |

In `MatMulWeight`, `t` is the token index: `X` and `Y` hold one row per token of the batch, and `W` is stored `(o, i)`, so `Y = X·Wᵀ`. The prover contracts each operand once, `W` at `r_o` and `X` at `r_t`, so a matmul costs `O(|W| + |X|)`, or `O(|A| + |B|)`, beyond the product.

The matmuls' rows give their degree per variable when they are not stacked. Stacked over layers or heads ('Batching within a stage'), they also sum over the stacking index, weighted by `eq` at the input claim's coordinate `ρ` of that index:
- `MatMulWeight`: `Ỹ(ρ, r_t, r_o) = Σ_{ℓ, i} eq(ρ, ℓ)·W̃(ℓ, r_o, i)·X̃(ℓ, r_t, i)`, a sumcheck over `(i, ℓ)` with `i` low.
- `MatMulAct`: `Z̃(ρ, r_a, r_c) = Σ_{b, j} eq(ρ, b)·Ã(b, r_a, j)·B̃(b, j, r_c)`, a sumcheck over `(j, b)` with `j` low.

Either has degree 3 in the stacking variables and 2 in the others. It ends with one claim on each operand at the shared final point. For `MatMulWeight`, with the point `r_i ‖ ρ_ℓ`, these are `W̃` at `r_i ‖ r_o ‖ ρ_ℓ` and `X̃` at `r_i ‖ r_t ‖ ρ_ℓ`.

In `QKᵀ` the second operand is `Kᵀ`, with `K` stored row-major as `(b, c, j)`. Since `B̃(j, c) = K̃(c, j)`, the claim on `B` lands on `K` with its coordinates reordered, at `r_j ‖ r_c ‖ ρ_b`.

`Elementwise`, `SumSquares` and `ZeroCheck` are sums of weighted products of inputs against `eq`: on `n` variables, `T̃(r) = Σ_x eq(r, x_high)·Σ_t w_t·Π_{f ∈ t} I_f(x)`, with `x_high` the target's variables. `SumSquares` sums its row's low `m` variables, so its target and `eq` have the other `n − m`; the others have all `n`. Each input is read at every vertex `x` in one of four ways. The final evaluations are those of the operand inputs, in the order the relation lists its inputs, and each is a claim at the instance's point `r'` or at a point derived from it:
- **Operand** `A(x)`: the claim is at `r'`.
- **Partner** `X(x ⊕ 2^b)`, the entry whose index differs in bit `b`: the claim is on `X̃` at `r'` with coordinate `b` replaced by `1 − r'_{b+1}`, since `X̃(x ⊕ 2^b)` is `X̃` with `x_{b+1}` replaced by `1 − x_{b+1}`. No relation of the Llama lowering reads a partner: `Rotate` is the kernel of [RFC-0037](RFC-0037-stage-schedule.md#rotate), which leaves one claim on `X`.
- **Broadcast** `R(x_{k+1}, …, x_{k+j})`, constant in the other variables: the claim is at the coordinates `k` to `k + j − 1` of `r'`. A scalar-by-row product broadcasts over the low variables, as `R_q` over the rows of `PV` (`Mul`'s `Trailing`, `k = n − j`); a channel's value broadcasts over the rows (`Leading`, `k = 0`).
- **Public**, such as RoPE's constants or a selector: the verifier evaluates it at `r'`, and the proof sends nothing for it.

`ZeroCheck`'s `h·(1 − h)` and `(1 − ψ)·w` each send one evaluation, `h̃(r')` or `w̃(r')`.

Claims on a `Linear` target become claims on its inputs with public coefficients, so linear chains (residual stream sums, `c_q − x_qi`, offsets) cost nothing. An input with the target's variables takes the same point. A broadcast input, such as `c_q` in `c_q − x_qi`, takes the coordinates it has. A sum over an input's `j` lowest variables, such as a row sum `s_q = Σ_i e_qi`, takes `(½, …, ½) ‖ r` with its coefficient times `2^j`: for multilinear `Ĩ`, `Σ_{x ∈ {0,1}^j} Ĩ(x ‖ y) = 2^j·Ĩ(½, …, ½ ‖ y)`. A public input, an offset's constants, is evaluated by the verifier and moves to the claim's value. `Transpose`, `Expand` and `Reshape` move coordinates ([RFC-0037](RFC-0037-stage-schedule.md#head-layouts)). The result is a claim on a linear combination of tensors, which [RFC-0037](RFC-0037-stage-schedule.md#splits) splits into one claim per virtual term and one per group of committed terms at one point.

### Stages

- **Leaf stage.** All LogUp fraction trees and all product trees, level by level from the root, batched across instances. Output: claims on lookup keys (virtual), lookup values and multiplicities (committed), product-tree leaves (virtual softmax keys).
- **Reduction stages.** A virtual claim is assigned to the stage equal to its distance from the leaf stage in the dependency DAG of virtual tensors. Stage `k` proves every relation instance whose target claims arose in stage `k − 1`, across all layers, heads and blocks, in one batched sumcheck. Its outputs are claims on inputs: committed tensors go to the opening pool; virtual tensors go to stage `k + 1`.
- **Termination.** The schedule ends when no virtual claim remains. Every chain ends at committed tensors because every non-arithmetic output is committed (INV-IR-3).

For the `servable` Llama template the longest chains have two relations, so it has three stages:

- the attention output's requantization key ← `R_q · PV` (`Elementwise`) ← `PV` (`MatMulAct`) ← committed `e`, `v`;
- RoPE's key ← `Rotate` ← the Q or K projection (`MatMulWeight`).

The O projection reads the committed `o8`, so the residual update's key is one relation from committed tensors. The `minimal` template has three relations and one more stage. Its attention output is not committed, so the residual update's key ← O-projection (`MatMulWeight`) ← `R_q · PV` ← `PV` makes three. So does the down projection's, through `a · u` (`Elementwise`) and the up projection. An affine RoPE site adds its multipliers' `Mul` to RoPE's chain ([RFC-0025](RFC-0025-affine-sites.md#stages)), so a `servable` model with an affine Q or K has four stages too. The compiler's commitment report lists the stage count; it must be identical for 2-, 4-, 8- and 126-layer models of the same template.

### Tree levels

The leaf stage proves each product tree level by level from its root, level 0, to its leaves, level `h`. Level `ℓ + 1` has twice the entries of level `ℓ`, and the pair `2x, 2x + 1` combines into entry `x`: `V_ℓ[x] = V_{ℓ+1}[2x]·V_{ℓ+1}[2x + 1]`. In the relation `ProductTreeLevel`, `V_{ℓ+1}(x, b)` is therefore `Ṽ_{ℓ+1}(b ‖ x)`: the pair bit is the lowest variable. Fraction trees pair their entries the same way, under the rule of [RFC-0008](RFC-0008-logup-and-tables.md#fraction-tree).

Product trees of one height that the schedule lists together are stacked into a forest and proven as one instance, such as the `SoftmaxRow` trees of every query row, head and layer ([RFC-0009](RFC-0009-gadgets.md#softmaxrow)).
- **Layout.** Level `ℓ` of a forest of `2^s` trees is a table of `2^{s+ℓ}` entries: entry `y·2^ℓ + x` is entry `x` of tree `y`'s level `ℓ`. The tree index takes the high `s` variables, as a stacked tensor's layer index does ('Multilinear extensions'). The forest's levels therefore pair their entries by the rule above, and the pair bit stays the lowest variable. A single tree is a forest with `s = 0`. When the schedule's trees are not a power of two, it pads the forest with trees whose root values are public as well.
- **Joining the batch.** Forests of different heights join the batch where their heights match, as RFC-0008 'Batching' says. With `H` the tallest forest's height, a forest of height `h` takes its root level at level `H − h` of the batch, and at level `ℓ` of the batch it has `s + ℓ − (H − h)` variables.
- **Root claim.** A forest's root level holds one value per tree. These values are public. For `SoftmaxRow` they are zero for the trees of a sequence's rows, and one for the trees of its padded rows and for those that pad the forest, whose leaves are all 1 ([RFC-0030](RFC-0030-masked-exponentials.md#relation-2-the-trees)).
  - At a level `ℓ` where forests of more than one tree join, before `γ`, the challenges `c = 0, 1, …, s_max − 1` are squeezed in `Fp2` under `tree/{ℓ}/root/{c}`. Here `s_max` is the largest `s` among the joining forests.
  - Each joining forest's root point `r = (r_1, …, r_s)` is the last `s` of these challenges, in the order squeezed. The forest enters the level with the root claim `Ṽ_0(r) = Σ_y v_y·eq(r, y)`, where `v_y` are its public root values. For `SoftmaxRow`, stacked over layers and heads, that is `1 − LT(r_ℓ, n_layers)·LT(r_h, n_heads)·LT(r_q, S)`.
  - A forest of one tree takes no challenge. Its root claim is its tree's root value, on the root level's zero variables.
  - If a forest's root level differs from its public values, its root claim holds with probability at most `s/|Fp2|` over the root point.

The fraction trees of LogUp ([RFC-0008 'Fraction tree'](RFC-0008-logup-and-tables.md#fraction-tree)) run in the same levels, under the same labels: each instance's lookup tree and table tree. In `tessera-protocol/2`, every forest holds at least two trees, a lone tree taking a public padding tree, and every level runs masked ([RFC-0032](RFC-0032-zero-knowledge-tree-levels.md)).
- Each fraction tree is an instance of its own, a single tree. It joins at level `H − h` like a forest of one tree.
- Its root `(n, d)`, sent under `logup/roots`, gives the claims on its level 0.
- It holds two claims at one point: `Ñ_t(r) = p` on its numerators and `D̃_t(r) = q` on its denominators.

`H` is the height of the tallest forest or tree, and instances of both kinds are numbered from 0 in schedule order.

At level `ℓ`, each instance `i` of height `h_i` holds its claims on its own level `t_i = ℓ − (H − h_i)`, over `n_i` variables: `Ṽ_{t_i}(r_i) = v_i` for a forest, and `Ñ_{t_i}(r_i) = p_i`, `D̃_{t_i}(r_i) = q_i` for a fraction tree.

1. The challenge `γ` is squeezed under `tree/{ℓ}/gamma`. When a fraction tree is present at the level, the challenge `λ` follows under `tree/{ℓ}/lambda`. Each instance enters with the claim `c_i`: `v_i` for a forest, `p_i + λ·q_i` for a fraction tree. The instances are combined and front-padded as in a stage ('Batching within a stage').
2. A sumcheck of degree 3 over `n_max` variables proves `Σ_x eq(r_i, x)·g_i(x)` for every instance, with the messages of 'Sumcheck messages' under `tree/{ℓ}/round/{j − 1}` and `tree/{ℓ}/r/{j − 1}`.
   - For a forest, `g_i(x) = Ṽ_{t_i+1}(0 ‖ x)·Ṽ_{t_i+1}(1 ‖ x)`.
   - For a fraction tree, with `N` and `D` its numerators and denominators on level `t_i + 1`, `g_i(x) = N(0 ‖ x)·D(1 ‖ x) + N(1 ‖ x)·D(0 ‖ x) + λ·D(0 ‖ x)·D(1 ‖ x)`.
3. The prover sends each instance's evaluations at its point `r'_i`, the last `n_i` challenges, instance by instance in schedule order, as one TCE `Vec<Fp2>` under `tree/{ℓ}/evals`:
   - for a forest, `u_i = Ṽ_{t_i+1}(0 ‖ r'_i)` and `w_i = Ṽ_{t_i+1}(1 ‖ r'_i)`;
   - for a fraction tree, `N_0`, `N_1`, `D_0`, `D_1`, the values of `N(0 ‖ r'_i)`, `N(1 ‖ r'_i)`, `D(0 ‖ r'_i)` and `D(1 ‖ r'_i)`.

   The verifier checks `Σ_i γ^i · eq(r_i, r'_i) · g_i` against the sumcheck's last claim, with `g_i = u_i·w_i` for a forest and `g_i = N_0·D_1 + N_1·D_0 + λ·D_0·D_1` for a fraction tree.
4. The challenge `ρ` is squeezed under `tree/{ℓ}/line`. Each instance's claims on its next level are on `n_i + 1` variables:
   - for a forest, `Ṽ_{t_i+1}(ρ ‖ r'_i) = u_i + ρ·(w_i − u_i)`;
   - for a fraction tree, `N(ρ ‖ r'_i) = N_0 + ρ·(N_1 − N_0)` and `D(ρ ‖ r'_i) = D_0 + ρ·(D_1 − D_0)`.

The claims after a forest's last level are the leaf claims the next stage consumes. Each is a claim on the forest's leaf table of `2^{s+h}` entries, its trees' leaves stacked as above, every entry included. A gadget whose trees hold fewer than `2^h` meaningful leaves, as SoftmaxRow's rows do, defines the others in its leaf formula ([RFC-0009](RFC-0009-gadgets.md#softmaxrow): the constant 1 at masked positions). The consuming stage proves the table's extension with that formula. A fraction tree's leaf claims go to the final checks of RFC-0008 'Fraction tree', after the prover sends the lookup trees' value evaluations under `logup/values`. A level with no fraction tree squeezes no `λ`, so the levels of a leaf stage of product forests alone run exactly as before. Such a stage still squeezes `α` and `β` and absorbs `logup/roots` and `logup/values`, empty.

### Batching within a stage

- Instances of one stage are combined with powers of a batching challenge `γ` drawn after the stage's input claims are fixed (label `stage/{k}/gamma`). Numbered from 0 in schedule order, instance `i` with claim `c_i` on `n_i` variables enters the stage's claim as `γ^i · 2^{n_max − n_i} · c_i`.
- Instances with fewer variables than the stage maximum `n_max` are front-padded: each is treated as constant in the extra leading variables, the lowest `n_max − n_i`, which the first `n_max − n_i` rounds bind, hence the scaling of its claim by `2^{n_max − n_i}`. Its own variables take the last `n_i` challenges, so all instances share one round sequence, and the high coordinates of every instance, a stacked tensor's layer coordinate among them, are the stage's last challenges.
- **Stacked layers.** Per-layer instances of one relation are stacked: the layer index becomes `⌈log2 n_layers⌉` extra sumcheck variables (padded layers are zero). [RFC-0037](RFC-0037-stage-schedule.md#layers-and-kinds) defines the layers and kinds, the stacked instances' layer weights, the layer point that stacks the leaf stage's per-layer claims in stage 1, and the order of a stage's instances. A stage therefore ends with one claim per stacked tensor, at one point that includes a layer coordinate `ρ_ℓ`. For weights this yields a single claim `W̃_T(ρ_ℓ, r_o, r_i)` per weight type, opened with one folding pass ([RFC-0012](RFC-0012-weight-commitment.md#opening-by-the-folding-pass)); for zero knowledge it means one masked evaluation per tensor kind per proof rather than one per layer ([RFC-0013](RFC-0013-zero-knowledge.md#stacked-layers)).
- All instances share the stage's round challenges `r_1 … r_{n_max}`; different relations and tensors of different shapes are combined with powers of `γ` as above.
- **Zero-check point.** A stage that holds `ZeroCheck` instances squeezes one point just before `γ`, after `ρ` in `tessera-protocol/2`: coordinates `z_c`, `c = 0, 1, …`, under `stage/{k}/zero/{c}`, one per variable of its largest `ZeroCheck`. An instance of `n_i` variables takes the last `n_i` coordinates. Every `ZeroCheck` runs in stage 1, `OneHot`'s booleanity among them ([RFC-0035](RFC-0035-padded-rows.md#zero-check-points)). Each adds `n_i/|Fp2|` to its stage's error over its point.

### Sumcheck messages

- Round `j`, from 1 to `n_max`, sends the round polynomial `p_j` of the stage's degree `d`, the largest degree among its instances, as its evaluations at `0, 2, 3, …, d` (`p_j(0)` alone when `d = 1`): a `Vec<Fp2>` of `d` entries, absorbed as TCE under `stage/{k}/round/{j − 1}`. A message of another length is rejected (`Reject(Sumcheck)`). The verifier derives `p_j(1) = claim_j − p_j(0)`, squeezes the challenge `r_j ∈ Fp2` under `stage/{k}/r/{j − 1}`, and continues with `claim_{j+1} = p_j(r_j)`, `claim_1` being the stage's claim.
- At the end, the prover sends the claimed evaluations of every input polynomial at the final point: instance by instance in schedule order, each at its own point, the last `n_i` challenges, and within an instance in the order its relation's identity in 'Relations' names its polynomials: `MatMulWeight`'s `W̃` then `X̃`, and an equality zero-check's output `Ỹ`, then `f`'s inputs in the order `f` names them. They are absorbed as one TCE `Vec<Fp2>` under `stage/{k}/evals`. The verifier checks that the stage's batched combination of these evaluations, with public factors (`eq`, selectors, constants) it computes itself, equals `claim_{n_max + 1}`; a stage of `n_max = 0` has no rounds and checks its claim directly.
- The verifier checks a stage's messages in this order and stops at the first failure, a `Reject(Sumcheck)` whose round counts from 0 as the labels do: for `j` from 1 to `n_max`, round `j` is present and holds `d` values (round `j − 1`); no round follows round `n_max` (round `n_max`); the final evaluations are as many as the instances' relations list, and the final check holds (both round `n_max`). A tree level is checked in the same order, a `Reject(TreeLevel { level, round })` whose `level` is its `ℓ` ([RFC-0033](RFC-0033-tree-level-rejections.md)).

```rust
pub struct SumcheckProof { pub rounds: Vec<Vec<Fp2>>, pub final_evals: Vec<Fp2> }
pub struct StageProof { pub stage: StageId, pub sumcheck: SumcheckProof, pub splits: Vec<Fp2> }  // splits: RFC-0037 'Splits'
pub struct Claim { pub poly: PolyRef, pub point: Vec<Fp2>, pub value: Fp2 }

pub fn derive_schedule(cs: &ConstraintSystem) -> StageSchedule;    // prover and verifier each implement this
pub fn prove_stage(stage: &Stage, w: &mut dyn WitnessSource, tr: &mut Transcript)
    -> (StageProof, Vec<Claim>);
```

`WitnessSource` regenerates the evaluation tables a stage needs from committed data and weights on demand (recompute, never store; RFC-0018). Stages whose tables exceed device memory (the leaf levels of fraction and product trees at production sizes) are proven by streaming their early rounds over the regenerated witness, one pass per streamed round, until the folded state fits in memory ([RFC-0018](RFC-0018-gpu-prover-runtime.md#pass-structure)); the messages are identical to those of the in-memory prover.

### Claim pool and openings

All claims on committed advice are collected in a pool keyed by polynomial. A claim may be a combination of committed tensors at one point, such as a stacked tensor's layers ([RFC-0037](RFC-0037-stage-schedule.md#the-claim-pool)). Claims on the same polynomial at different points are reduced to one point by a final batched sumcheck over the advice matrix (RFC-0011). Claims on weights are combined per weight type (RFC-0012). Claims on channel parameters (hidden shift classes) join the weight openings.

### Soundness of the schedule

For a stage with `n` rounds of degree `d` and `m` batched instances, the error is at most `(n·d + m)/|Fp2|`. Each tree level contributes `(n·d + m + 1)/|Fp2|`, with `m` its instances: the line challenge `ρ` that joins each instance's two evaluations into one claim adds 1, since two distinct lines agree at one point at most. At a level with fraction trees, the combination `Σ_i γ^i·c_i` is a polynomial of total degree at most `m` in `γ` and `λ`, so `λ` adds nothing to that bound. Each forest's root claim adds at most `s/|Fp2|`. The total over all stages and levels is summed by the soundness calculator (RFC-0014); for 405B at 16K tokens it is below 2^-110.

### Invariants

- **INV-SCH-1.** The schedule is a function of the constraint system only; prover and verifier derive it independently and must agree.
- **INV-SCH-2.** Every virtual claim is consumed by exactly one relation instance in the next stage.
- **INV-SCH-3.** Every claim produced at a committed polynomial reaches the opening pool.
- **INV-SCH-4.** The number of stages does not depend on `n_layers`, sequence length or batch size.

## Alternatives Considered

1. **Layer-by-layer GKR** (zkLLM, classic GKR). Rejected: proof depth grows with the 126 layers, each layer's challenges differ, and weight claims land on 126 different points per type.
2. **Commit every intermediate** (VerInf, DeepProve). Rejected: the cost driver the design exists to remove (source §4).
3. **Independent challenges per layer within a stage.** Rejected: loses the common point per weight type and multiplies weight openings by the layer count.
4. **One monolithic sumcheck over all relations.** Rejected: relations depend on claims produced by earlier relations, so a single round sequence cannot serve them all.
5. **Sending full round-polynomial coefficients.** Rejected: one redundant element per round; the derived `p(1)` form is standard.

## Drawbacks

- Shared challenges couple all layers: the prover must hold or regenerate every layer's evaluation tables for the current round, which is why stages recompute witness values on tensor cores.
- Front-padding wastes rounds for small instances in large stages.

## Migration / Rollout

The sumcheck engine, matmul relation and product tree land in v0.1 (needed for unit-cost measurement); the full schedule lands in v0.3 with the CPU prover and verifier.

## Testing Strategy

- Each relation kind: honest proofs verify for random instances with 1–24 variables; flipping any round evaluation or final evaluation is rejected.
- Batching: stages mixing instances of 3, 7 and 12 variables verify; dropping the `2^{n_max − n_i}` scaling is rejected.
- Product forests: forests of 1 to 2^10 trees with 2 to 2^12 leaves each verify, including forests of several heights in one batch. A forest with a tree whose product differs from its public root value is rejected.
- Schedule derivation: prover and verifier schedules (stage count, relation kinds, variable counts) are equal on all conformance models.
- INV-SCH-4: stage count equal for tiny models with 2, 4 and 8 layers of the same template, and in the 405B-shaped commitment report.
- Common point: after the `MatMulWeight` stage, each weight type has exactly one claim, at a point whose layer coordinate is shared by all stacked tensors of that stage (asserted in tests).
- Soundness calibration over `F65537`: a prover lying about one round succeeds at the rate `d/|F|` within confidence bounds.

## Open Questions

- **OQ-0010-1** (resolved). How does one level order fraction trees and product trees, lay out their evaluations under `tree/{ℓ}/evals`, and combine a fraction tree's numerator and denominator claims? 'Tree levels' answers it. Instances of both kinds are numbered in schedule order. A fraction tree sends `N_0, N_1, D_0, D_1` where a forest sends `u, w`. A level with fraction trees squeezes `λ` after `γ`, and each fraction tree enters with the claim `p + λ·q`. [RFC-0036](RFC-0036-protocol-messages.md#round2response) gives the types of `Round2Response` that carry them, `TreeProof` and `LogUpProof`.

## References

- Source §6 principle 8, §7 ("Constant-depth proof", "Weights opened by the folding pass").
- Goldwasser, Kalai, Rothblum, STOC 2008; Thaler, CRYPTO 2013 (matmul sumcheck); Lund, Fortnow, Karloff, Nisan, 1992 (sumcheck).
- [RFC-0008](RFC-0008-logup-and-tables.md), [RFC-0009](RFC-0009-gadgets.md), [RFC-0011](RFC-0011-advice-commitment.md), [RFC-0012](RFC-0012-weight-commitment.md).
