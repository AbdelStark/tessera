# RFC-0037: Stage Schedule: Stacked Layers, Split Claims and the Claim Pool

- Status: Accepted
- Authors: @AbdelStark
- Created: 2026-10-17
- Target milestone: v0.3
- Amended by: [RFC-0052](RFC-0052-channel-params-claims.md) ('The claim pool' and 'Invariants': `ChannelParams` claims in the pool), [RFC-0068](RFC-0068-leaf-stage-zero-knowledge.md) (in `tessera-protocol/2`: 'The layer point', `ρ_1` is the leaf stage's layer sumcheck point; 'Splits' and 'Messages and transcript', the leaf stage's values and labels), [RFC-0069](RFC-0069-one-layer-claims.md) ('Layers and kinds' and 'Stages and their instances': the one read at one layer, and no relation proving such a claim; 'The claim pool', in `tessera-protocol/2`: a piece at one layer takes its layer tensor's own mask row), [RFC-0070](RFC-0070-mask-batching.md) (INV-SCH-3, in `tessera-protocol/2`: the mask claims reach the masks' batching), [RFC-0071](RFC-0071-request-slots.md) ('`Rotate`': a row's position in a batch is its place in its request), [RFC-0072](RFC-0072-anchor-relations.md) ('Layers and kinds': a graph with no checkpoint has no layers, and the operations a template may not hold; 'The committed first checkpoint': the pointwise nodes whose committed outputs stage 1 proves; 'Claims and atoms': public tensors, `ConstAdd`, `Slice`, `GatherPublic` and a per-channel `ConstMul` as relations, selections of one tensor combined, and one point; 'Splits': a combined selection's value; 'Stages and their instances': RFC-0066's zero-checks in stage 1, after the others, in node order, and the stage count; 'The layer point': the gadgets with leaf sites, and the logits when there are some), [RFC-0013](RFC-0013-zero-knowledge.md#masked-extensions) ('The claim pool', in `tessera-protocol/2`: each piece's mask row, once per piece, at `r`'s column sub-point, [#554](https://github.com/AbdelStark/tessera/issues/554))
- Supersedes, on acceptance:
  - [RFC-0010](RFC-0010-proof-schedule.md#relations) 'Relations': `Rotate`'s identity and its one claim on its input, and the claims of the linear maps added here;
  - [RFC-0010](RFC-0010-proof-schedule.md#stages) 'Stages' and 'Batching within a stage': how a stage stacks layers, splits claims and orders its instances, and the layer point;
  - [RFC-0010](RFC-0010-proof-schedule.md#claim-pool-and-openings) 'Claim pool and openings', and [RFC-0011](RFC-0011-advice-commitment.md#opening-protocol) 'Opening protocol' steps 1 and 2: a claim on committed tensors as a combination at one point;
  - [RFC-0006](RFC-0006-compiler-ir.md#operation-set) 'Operation set' and 'Llama lowering': `Transpose` and `Expand`, the committed first checkpoint, and committed outputs of linear and degree-2 nodes;
  - [RFC-0014](RFC-0014-interaction-and-soundness.md#transcript-schedule) 'Transcript schedule' row 6, and [RFC-0036](RFC-0036-protocol-messages.md#round2response)'s `TreeProof` and `StageProof`: the layer point and the split values.
- Origin: [#113](https://github.com/AbdelStark/tessera/issues/113)

## Summary

RFC-0010 derives the stages from the constraint system and stacks each relation's per-layer instances. It does not say how a stage stacks them, what happens to a claim that names several tensors, or what the opening receives. Deriving the schedule of the lowered Llama graph for #113 found three of RFC-0010's and RFC-0012's properties false:

- **Q and K would have two claims each.** `Rotate` reads every entry and its RoPE partner, so it leaves two claims on its input.
- **The embedding would have two.** The residual stream that enters the first layer is the embedding's virtual output, and three nodes read it.
- **The residual key has no stage.** It is `acc + 2^r·res`: a claim on a virtual accumulator and a committed stream together, which no instance proves as it stands.

This RFC makes the schedule a function of the graph alone (INV-SCH-1), with one claim per weight type:

- **Layers.** The lowering's checkpoints bound the layers. A tensor's kind is its node's position in the layer, stacked over the layers with the layer index highest.
- **`Rotate`** becomes one product against a public kernel, `Ỹ(r) = Σ_x E_r(x)·X̃(x)`, of degree 2, with one claim on its input.
- **The first checkpoint is committed**, proven by a zero-check in stage 1. Every layer then reads a committed residual stream, and the embedding has one claim.
- **Head layouts.** Attention uses `Transpose` and `Expand`, whose claims move coordinates. They replace `GatherPublic`, whose claims do not when the sequence length is not a power of two.
- **Splits.** A claim on several tensors splits into one claim per virtual term and one per group of committed terms. The prover sends their values under `stage/{k}/split`, and they must add up.
- **The layer point.** Stage 1 stacks the leaf stage's per-layer claims with a layer point drawn after them.
- **The pool** receives combinations of committed tensors at one point, which RFC-0011's row coefficients already admit.

## Motivation

#113 derives the stages from the lowered graph. Tiny Llama models of 2 layers in both profiles, lowered by RFC-0006, show the following.

- **`Rotate`'s partner.**
  - RFC-0010 'Relations' writes RoPE as `Ỹ(r) = Σ_x eq(r, x)·(X̃(x)·C(x) + X̃(x ⊕ 2^b)·S_±(x))`.
  - The sumcheck ends with two evaluations of `X̃`, at `r'` and at `r'` with coordinate `b` replaced by `1 − r'_b`.
  - `X` is the Q or K projection, so each runs two `MatMulWeight` instances and leaves two claims on its weights. RFC-0012's folding pass opens one claim per weight type.
- **The first layer's input.**
  - `c_0 = 2^{s_e}·x_0`, with `x_0` the embedding's output. It is virtual, and the first layer's `SumSquares`, its norm's `Mul` and its residual all read it.
  - Their claims land at different points in different stages, so `MatMulWeight` on `Embed` runs twice.
  - The stacked layer input mixes a virtual slice, layer 0's, with committed ones, the other layers' checkpoints.
- **Linear claims on several tensors.**
  - The leaf stage's claim on the residual's lookup key is a claim on `acc + 2^r·res`.
  - The next stage needs `acc`'s own value for its `MatMulWeight` instance, and the opening needs `res`'s.
  - RFC-0010 says such chains "cost nothing", but defines no instance or message that separates them.
- **Head layouts.**
  - The lowering reorders attention's rows with `Reshape`, `GatherPublic` and `Reshape`.
  - With `n` not a power of two, the gathered tensor's rows `h·n + t` are not a concatenation of bits. A claim on it then names no single point of its input.
- **Stacking.**
  - RFC-0010 stacks per-layer instances, and RFC-0013 calls this "the batching of RFC-0010 with `eq(ρ_ℓ, ℓ)` in place of powers of `γ`".
  - No document says where `ρ_ℓ` comes from when the claims arrive one layer at a time. The leaf stage's do: a layer's `Exp` and `Silu` tables are its own (RFC-0006 'Llama lowering'), so its lookups form their own LogUp instances (RFC-0008 'Batching').

## Proposed Design

### Layers and kinds

The lowering's checkpoints `c_0, …, c_L` are the residual stream after the embedding and after each layer (RFC-0006).

- **Layers.**
  - Layer `ℓ` holds the nodes after `c_ℓ`'s producer, up to and including `c_{ℓ+1}`'s, in node order.
  - The nodes up to `c_0`'s producer are the prologue, and those after `c_L`'s producer the epilogue.
- **The template.** Every layer has the same nodes, in the same order. Their operations are equal up to their layer's own weights, constants, shift classes and tables, and their shapes are equal. Each input is one of:
  - the same position's output in its own layer;
  - its layer's checkpoint `c_ℓ`;
  - a graph input, such as the positions or the layer's own weight, classes and multipliers.

  The lowering emits such layers for a claim list that rescales every layer's sites alike ([RFC-0038](RFC-0038-uniform-layers.md)). The schedule refuses any other graph with `ERR-INTERNAL`, and a template that holds a `ConstAdd`, a selection, a per-channel `ConstMul`, a `BitDecompose`, an `AssertEqual`, a lookup non-zero at padding or a node whose output is public ([RFC-0072](RFC-0072-anchor-relations.md#graphs-without-layers)).
- **No checkpoint.** A lowering with no checkpoint has no layers, `L = 0` and `λ = 0`. Every node is outside them, every tensor a kind without a layer index, and stage 1's layer point has no coordinate, as for the anchor's circuits ([RFC-0072](RFC-0072-anchor-relations.md#graphs-without-layers)). A single checkpoint bounds no layer, and the schedule refuses it (`ERR-INTERNAL`).
- **Kinds.**
  - The outputs at position `p` form one stacked kind, `X_κ[ℓ, …]`.
  - The layer input `X_in[ℓ] = c_ℓ` is a stacked kind.
  - A weight type is stacked over its layers.
  - Prologue and epilogue tensors are kinds without a layer index.
  - **The last checkpoint.** A node outside the layers reads `c_L` as the last position's output at layer `L − 1`: that kind with the layer weights `[ℓ = L − 1]`, `Layers::One(L − 1)`. Every other tensor it reads is a kind without a layer index. So no other read outside the layers has the weights of one layer ([RFC-0069](RFC-0069-one-layer-claims.md)). The leaf stage's per-layer claims, `Layers::Each`, are the layers' own, which stage 1 stacks ('The layer point'; RFC-0068 in `tessera-protocol/2`).
- **Layer variables.**
  - A stacked kind has `λ = ⌈log2 L⌉` layer variables, the highest of its tensor ([RFC-0010](RFC-0010-proof-schedule.md#multilinear-extensions) 'Multilinear extensions').
  - Layers `L` to `2^λ − 1` are padded. Each of their tensors is what its node computes on zero inputs ([RFC-0035](RFC-0035-padded-rows.md#what-a-padded-row-holds)), and their weights are zero. In the Llama templates every virtual tensor of a padded layer is therefore zero, and its `X_in` is zero.

### The committed first checkpoint

- **Lowering.** The `ConstMul` that forms `c_0 = 2^{s_e}·x_0` produces committed advice ([RFC-0006](RFC-0006-compiler-ir.md#llama-lowering) 'Llama lowering').
- **Committed outputs.** A pointwise linear or degree-2 node, `ConstMul`, `ConstAdd`, `Add`, `Sub` or `Mul`, may produce a committed tensor `Y`; a committed output of any other node but a gadget is refused, a matmul's by the IR and any other's when the schedule is derived (`ERR-INTERNAL`, [RFC-0072](RFC-0072-anchor-relations.md#linear-operations-that-are-not-coordinate-maps)). Stage 1 proves its relation `f` as an equality zero-check at the stage's zero-check point ([RFC-0035](RFC-0035-padded-rows.md#zero-check-points)): `0 = Σ_x eq(z, x)·(Ỹ(x) − f(x))`. For `c_0` it is `Ỹ − 2^{s_e}·x̃_0`, of degree 2.
- **Effect.** Every layer reads a committed input. `x_0` has one reader, so `MatMulWeight` on `Embed` runs once.
- **Cost.**
  - `n·d_model` more committed elements per sequence: `2^{28}` at 405B's 16K tokens, about 0.01% of the advice.
  - One zero-check instance in stage 1.

### Head layouts

RFC-0006 gains two linear operations:

| Operation | Inputs → outputs | Shapes | Outputs and inputs |
|---|---|---|---|
| `Transpose { perm }` | `x` → `y` | `y`'s axis `i` is `x`'s axis `perm[i]`, `perm` a permutation | |
| `Expand { axis, len }` | `x` → `y` | `y` is `x` with an axis of length `len ≥ 1` inserted before axis `axis`, `x` repeated along it | |

On per-dimension padded extensions ([RFC-0011](RFC-0011-advice-commitment.md#canonical-advice-layout) rule 1), a claim on `y` is one claim on `x`:
- **`Transpose`** permutes the coordinate blocks of the axes.
- **`Expand`** drops the inserted axis's coordinates `r_a`, with coefficient `LT(r_a, len)`, which is 1 when `len` is a power of two.
- **`Reshape`** maps coordinates as the identity when, in each group of axes that merge or split, every axis but the group's leading axis has a length that is a power of two, axes of length 1 aside. When `d_head` is a power of two (`Rotate` below), the lowering uses no other `Reshape`: `[n, heads · d_head] ↔ [n, heads, d_head]` has `heads` as its group's leading axis and `d_head` as its inner axis, and `[kv, g, n, d_head] → [n_heads, n, d_head]` has `kv` as its group's leading axis and `g = n_heads / n_kv_heads` as its inner axis. A non-power-of-two `d_head` is rejected when deriving the stage schedule (`Reshape` when `heads > 1`, `Rotate` when `heads = 1`).

The lowering's attention uses them in place of `GatherPublic`:
- Q, K and V reshape to `[n, heads, d_head]` and transpose to `[heads, n, d_head]`.
- K and V then expand to `[kv, g, n, d_head]` and reshape to `[n_heads, n, d_head]`, with `g = n_heads / n_kv_heads` a power of two, as it is in Llama-3.1.
- The attention output transposes back to `[n, n_heads, d_head]` and reshapes to `[n, d_model]`.

`GatherPublic` stays in the operation set. RFC-0006's lowering no longer emits it.

### `Rotate`

`Rotate` pairs `x_i` with `x_{i+m}` along the last axis: the entries whose flattened indices differ in bit `b`, `m = 2^b`. Each pair is multiplied by the matrix `M = [[C, −S], [S, C]]`, whose constants `C` and `S` depend on the other coordinates `x_{−b}` only (RFC-0005 'RoPE').

The extension of `M` in its row bit `y` and column bit `c` is `C·eq(y, c) + S·(y − c)`. So

```text
Ỹ(r) = Σ_x E_r(x)·X̃(x),        E_r(x) = eq(r_{−b}, x_{−b})·(C(x_{−b})·eq(r_b, x_b) + S(x_{−b})·(r_b − x_b))
```

- **The sumcheck.** It has degree 2. `E_r` is its one public factor, of `2^n` entries, built from the input claim's point `r` as `eq` is.
- **The claim.** The relation leaves one claim, on `X̃` at the instance's point `r'`.
- **The verifier's evaluation.** `Ẽ_r(r') = eq(r_b, r'_b)·⟨C, w⟩ + (r_b − r'_b)·⟨S, w⟩`, with `w = eq(r_{−b}, ·)·eq(r'_{−b}, ·)` over `x_{−b}`.
  - RoPE's constants depend on the position and the frequency, not the head, so the head's coordinates contribute the factor `eq(r_h, r'_h)`. In a batch of several requests a real row's position is its place in its request, `τ mod S`, and the constants are zero on every padded row, as past `n` today ([RFC-0071](RFC-0071-request-slots.md#the-public-selectors)). The verifier derives both from the batch's lengths, and its inner products run over each request's real rows.
  - The inner products run over `n·d_head/2` entries.

This replaces RFC-0010's partner form, of degree 3 with two claims, which #99 implemented.

### Claims and atoms

A claim names a kind and a point, `X̃_κ(r ‖ ρ) = v` for a stacked kind with layer coordinates `ρ`. More generally it carries public layer weights `w`: `Σ_ℓ w(ℓ)·X̃_{κ,ℓ}(r) = v`, where `w = eq(ρ, ·)` for a point.

Through a linear node, a claim on its output becomes terms on its inputs, as [RFC-0010](RFC-0010-proof-schedule.md#relations) 'Relations' places them:
- `Transpose`, `Expand` and `Reshape` map coordinates as above.
- A `ConstMul` by one constant multiplies the coefficient: a constant of one value is a scalar, whatever the last axis's length ([RFC-0072](RFC-0072-anchor-relations.md#linear-operations-that-are-not-coordinate-maps)). A constant per layer multiplies the layer weights.
- `ConstAdd` subtracts the extension `ĉ` of its constant at real positions, times the atom's coefficient, from the claim's value: `ỹ(r) = v` becomes `x̃(r) = v − ĉ(r)` for an atom of coefficient 1. `ĉ(r)` is `c·ψ̃(r)` for one constant and `c̃(r_c)·Π_{a≠last} ψ̃_a(r_a)` for one per channel of the last axis, with `ψ` 1 on the tensor's real positions, those inside its shape ([RFC-0072](RFC-0072-anchor-relations.md#linear-operations-that-are-not-coordinate-maps)); in a batch of several requests, a `ConstAdd` downstream of the tokens or the positions is refused (RFC-0072 'Real positions'). A `ConstMul` by a constant per channel, `Slice` and `GatherPublic` are relations, not maps: their outputs are virtual atoms ([RFC-0072](RFC-0072-anchor-relations.md#linear-operations-that-are-not-coordinate-maps)).
- A claim on the layer input reads its slices: `Σ_ℓ w(ℓ)·X̃_in[ℓ](r) = w(0)·c̃_0(r) + Σ_{ℓ≥1} w(ℓ)·X̃_last[ℓ − 1](r)`, with `X_last` the kind of the layer's last output. Every slice is committed.

**Atoms.** After substitution, a claim is a sum of atoms. An atom is a coefficient, a kind, a point and layer weights.
- Two points are one when they name the same challenges and constants in the same order, as both parties derive them. Atoms on one kind at one point merge, their layer weights adding. Within one claim, two or more virtual atoms that select rows of one input tensor along one axis at one point with one set of layer weights, outputs of `Slice` or `GatherPublic`, combine into one atom of coefficient 1, a selection whose kernel weighs theirs by their coefficients ([RFC-0072](RFC-0072-anchor-relations.md#combined-selections)).
- An atom is virtual (a sumcheck relation's output), committed (advice or a weight) or public (the verifier computes it). A tensor is public when it is a public input or a constant, or the output of a node that is not gadget-backed and whose inputs are all public, unless it is declared committed ([RFC-0072](RFC-0072-anchor-relations.md#public-tensors)).
- Atoms are listed in the order substitution reaches them, depth first, inputs in their node's order. A merged atom keeps its first place.

### Splits

The verifier computes a claim's public atoms and moves them to its value. The claim is then ready for the next stage when it is one virtual atom of coefficient 1, or committed atoms of one commitment at one point. Otherwise the stage `k` that produced it splits it:

- **What is sent.** The prover sends one value for each virtual atom, and one for each group of committed atoms of one commitment at one point, in atom order. A group takes the place of its first atom.
  - **A part's value.** A virtual atom's value is its tensor's evaluation, and the check weights it by the atom's coefficient. A combined selection's value is its whole combination `Σ_t a_t·ỹ_t(r)`, which the check counts with weight 1 ([RFC-0072](RFC-0072-anchor-relations.md#combined-selections)). A committed group's value is its combination `Σ_t a_t·P̃_t(r)`, coefficients included, and the check counts it with weight 1. That combination is the claim the pool receives.
  - **Where.** The values go under `stage/{k}/split`, after the stage's final evaluations, as one `Vec<Fp2>` covering every claim the stage splits, claim by claim.
  - **The order of claims.** A reduction stage's claims come by instance, then by evaluation. The leaf stage's come by site, then by layer, as 'The layer point' orders them.
  - **The leaf stage** is stage 0, and its values follow `logup/values` in one `Vec<Fp2>` under `stage/0/split` (`TreeProof.splits`): first the lookup tensor values—one per tensor, site and real layer ([RFC-0045](RFC-0045-leaf-layout.md#the-split))—and then the atom split values of any leaf claim that splits into virtual and committed parts.
  - **In `tessera-protocol/2`**, `stage/0/split` holds the sites outside the layers only. The sites in the layers send one value per site and tensor after the leaf stage's layer sumcheck, and their atom values follow under `stage/0/layer/split` ([RFC-0068](RFC-0068-leaf-stage-zero-knowledge.md#the-layer-sumcheck) 'The layer sumcheck').
- **The check.** The verifier checks that the values, weighted by their coefficients, give the claim's value. A mismatch is `Reject(Sumcheck { stage: k, round: n_k })`, with `n_k` the stage's number of rounds (0 for the leaf stage in `tessera-protocol/1`; in `tessera-protocol/2`, RFC-0068 'The layer sumcheck' steps 6 and 7 give the leaf stage's checks their rounds).
- **The result.** Each value becomes one claim: a virtual atom's for the next stage, a committed group's for the pool.
- **Example.** The residual key's claim `acc + 2^r·X_in` at `(r ‖ ρ)` splits into `acc`'s virtual atom and `X_in`'s committed group: two values.
- **Soundness.** The values are fixed before any later challenge. If the claim is false, any split leaves a false claim, which its own check rejects except with the probability that check allows.

### Stages and their instances

- **Stage index.** The leaf stage is stage 0, and an instance's stage is one more than its claim's.
- **Instances.** Stage `k` runs, in order:
  - one instance per virtual atom that stage `k − 1` leaves after its splits, in atom order: the atom's producing relation, stacked as its kind, at the atom's point, with the atom's value as its claim;
  - in stage 1, after them, the zero-checks in node order: RFC-0035's padding zero-checks, `OneHot`'s booleanity, and the equality zero-checks of committed outputs. RFC-0066's follow, in node order, each node's checks in [RFC-0072](RFC-0072-anchor-relations.md#the-bit-zero-checks-oq-0066-1)'s table order.
- **Stacked instances.** A stacked instance takes the atom's layer weights `w` as a public factor over the layer variables, in place of `eq(ρ, ·)`. The verifier evaluates `w̃(ρ') = Σ_ℓ eq(ρ', ℓ)·w(ℓ)` in `O(L)`, the sum running over the whole cube of `2^λ` layers: a padded layer's weight, such as `eq(ρ, ℓ)`, need not be 0.
- **No claim at one layer is proven by a relation.** A virtual atom with `Layers::One` weights is refused (`ERR-INTERNAL`): in `tessera-protocol/2` it would be revealed at a Boolean layer coordinate, where its kind's mask vanishes. A claim at one layer is a committed atom's, which reaches the pool with its layer tensor's own mask ([RFC-0069](RFC-0069-one-layer-claims.md)).
- **INV-SCH-2.** Each virtual atom is one instance.
- **Stage count.** The count, the leaf stage included, is one more than the longest chain of sumcheck relations from a leaf site's input, or from a zero-check, which counts itself, down to committed or public tensors ([RFC-0072](RFC-0072-anchor-relations.md#messages-rejections-and-stages)). Linear maps cost no stage, so the count does not depend on `L`, `n` or the batch (INV-SCH-4). The compiler's commitment report counts it the same way.

### The layer point

The leaf stage leaves its gadgets' claims one layer at a time. Each lookup site's layers sit at one point `r_s`: `P[0..k]` for its tensors of `2^k` entries ([RFC-0045](RFC-0045-leaf-layout.md#the-leaf-point)).

- **Its claims on the graph.**
  - The leaf stage leaves one claim on each input of each gadget, at a point of its site, but `BitDecompose` and `AssertEqual`, whose relations are stage 1's zero-checks and which have no site ([RFC-0072](RFC-0072-anchor-relations.md#gadgets-and-leaf-sites)). When the graph has logits it leaves one claim on them, which the scoring certificates read (RFC-0017).
  - Its sites come in node order, a template position once for all its layers, and the logits last.
  - The gadgets' own committed values and relations are RFC-0009's (#118).
- **Drawing it.** Stage 1 draws the layer point `ρ_1`: `λ` coordinates under `stage/1/layer/{c}`, after the leaf stage's splits and before the zero-check point and `γ_1`.
  - **In `tessera-protocol/2`**, `ρ_1` is the leaf stage's layer sumcheck point `ρ'_y`, and stage 1 draws none. The leaf stage leaves its stacked claims at `(π_t(r_s) ‖ ρ'_y)` directly, with no per-layer `v_ℓ` ([RFC-0068](RFC-0068-leaf-stage-zero-knowledge.md) 'The layer sumcheck', step 7).
- **Stacking.** A site's claims `v_ℓ` become the stacked claim `X̃(r_s ‖ ρ_1) = Σ_ℓ eq(ρ_1, ℓ)·v_ℓ`. A padded layer's `v_ℓ` is 0.
- **Soundness.** A false `v_ℓ` survives with probability at most `λ/|Fp2|` over `ρ_1`.
- **Native gadget targets.** First apply each real layer's affine checks and move its public terms into the native target; then stack those fixed targets. For example, a Softmax lookup leaves `κ̃_ℓ(r) − K_{0,ℓ}·(1 − C̃_local(r))`, whose stacked target is the zero-padded `C·(c−x)` relation. This retains each layer's original Exp lookup and value claim, even when its table and `K_{0,ℓ}` differ. A forest leaf already has its own layer coordinates: its target is `L̃(r) − 1 + C̃(r)`, at that full point. Every final source claim keeps its original source family and the final layer coordinates; no later per-layer values replace a fixed stacked claim.
- **Native source routing.** A native relation's raw final evaluations may encode fewer or more original graph claims (for example root digits that reconstruct `S`). Transform those same evaluations using the prescribed public equations, then apply the graph's ordinary linear substitution, splits and pool routing at that native instance's final point. Project the source's local coordinates after the complete stage round count is known, keeping its highest layer coordinates. A graph input or selected-logit claim supplied this way starts after its native reduction; it cannot also be initialized from an unrelated leaf value. Retain every remaining native auxiliary, lookup and opening obligation, including repeated claims at different points. Raw native evaluation counts and outgoing graph claim counts are distinct.
- **Later stages.** Their claims are stacked at their own points' layer coordinates.

### The claim pool

A claim ready for the pool is a combination `Σ_t a_t·P̃_t(r) = v` at one point `r`. Each `P_t` is a committed kind with its layer weights, or a committed prologue or epilogue tensor.

- **Advice.** RFC-0011's grouping takes the combination as one claim. Its row coefficients add `γ_claim·a_t·w_t(ℓ)·eq(row part of (r ‖ base_{t,ℓ}), row)` for each piece, and its column sub-point is `r`'s. In `tessera-protocol/2` each piece is read through its masked extension ([RFC-0013](RFC-0013-zero-knowledge.md#masked-extensions) 'Committed tensors', [#554](https://github.com/AbdelStark/tessera/issues/554)). A piece of layer weights `eq(ρ, ·)` adds its kind's mask row with `γ_claim·a_t·Z(r ‖ ρ)`, and a piece of a kind without layers its tensor's with `γ_claim·a_t·Z(r)`, once per piece whatever its tile images: the mask's column sub-point is `r`'s, so the group is unchanged. A piece whose layer weights carry per-layer constants has no mask defined yet ([#556](https://github.com/AbdelStark/tessera/issues/556)). A `Layers::One(ℓ)` piece, the read of `c_L`, adds its layer tensor's own mask row with `γ_claim·a_t·Z(r)` when `λ ≥ 1`, and the kind's own row takes nothing. When `λ = 0` it is an ordinary claim on the kind, whose row takes `γ_claim·a_t·Z(r)` ([RFC-0069](RFC-0069-one-layer-claims.md#the-pool)). The rule follows the schedule's `Layers::One` atoms, never the shape of a weight.
- **Weights.** A weight type's claim is `W̃_T(r_i ‖ r_o ‖ ρ)` with `eq(ρ, ·)` its layer weights: RFC-0012's one claim per type. Each site that reads channel classes or affine multipliers leaves one claim on `ChannelParams` in the pool, opened together in one pass ([RFC-0052](RFC-0052-channel-params-claims.md)).
- **INV-SCH-3.** Every claim on a committed tensor reaches the pool. In `tessera-protocol/2`, the mask claims reach [RFC-0070](RFC-0070-mask-batching.md)'s batching instead, whose claim `y` reaches the pool.

The complete current-model advice opening orders original combinations before
assigning the per-group powers of `gamma`: graph pool claims first in pool
order; fixed non-native leaf claims in actual graph-node order across layers;
fixed native families in template-site order, then real layer order; remaining
raw native sources in stage/instance order; finally each multiplicity at its
own LogUp table-leaf point. Fixed Requant claims take outputs followed by real
auxiliaries in canonical first-read order. Root claims take the output,
input-low limbs, root-low bytes and canonical comparison columns. Softmax takes
the reciprocal at its row point, the Exp row sum at key coordinates all `1/2`
followed by that row point, the low A then low B slack limbs, then Exp at its
original full score-leaf point. Virtual top digits are graph reconstructions,
not extra committed sources. Retain repeated obligations, including Softmax's
Exp at its native reciprocal-final point. An implementation may split a row
coefficient among equivalent layout images, but the complete Boolean row
coefficient of each original claim must agree; neither witness equality nor
equality of the final batched sum substitutes for that check.

### Messages and transcript

- **`TreeProof`** ([RFC-0036](RFC-0036-protocol-messages.md#round2response)) is `{ levels, splits }`: the leaf stage's levels, then its split values.
- **`StageProof`** is `{ stage, sumcheck, splits }`.

Schema version 1 changes in place: no release has written it ([09](../spec/09-release-and-versioning.md)).

[RFC-0014](RFC-0014-interaction-and-soundness.md#transcript-schedule)'s row 6 gains two labels:
- `stage/1/layer/{c}`, squeezed before `stage/1/zero/{c}`, in `tessera-protocol/1` only: `tessera-protocol/2` takes RFC-0068's layer point;
- `stage/{k}/split`, absorbed after `stage/{k}/evals`. Stage 0's comes after `logup/values`. In `tessera-protocol/2` the leaf stage's layered atom values come after its layer sumcheck, under `stage/0/layer/split` (RFC-0068).

Stage 1 runs, in order:
1. `zk-sum` and `zk-rho`, in `tessera-protocol/2`;
2. the layer point, in `tessera-protocol/1`;
3. the zero-check point;
4. `γ`;
5. the rounds;
6. the evaluations;
7. the splits.

### Invariants

- **INV-SCH-5.** In the Llama templates every weight type `Embed`..`LmHead` the proof opens has exactly one claim in the pool; `ChannelParams` has one claim per reading site ([RFC-0052](RFC-0052-channel-params-claims.md)).

## Alternatives Considered

1. **One combined instance per linear claim.** Each virtual atom's relation, and a pass-through `eq·P` for each committed group, would be front-padded into one sum of products. This sends no value. It was rejected: such instances mix relation kinds and variable counts, which both the prover and the independent verifier would have to assemble exactly. A few field elements per stage cost less.
2. **A virtual first checkpoint, with claims merged.** The three readers' claims on `c_0` land at unrelated points, so merging them needs a reduction sumcheck. That costs one more stage on the embedding's chain.
3. **The partner form of `Rotate`, with its two claims merged** at a challenge on coordinate `b`. This needs one more challenge and keeps degree 3. The kernel has degree 2, one public factor and one claim.
4. **Composing `Reshape`–`GatherPublic` chains into coordinate maps.** This means simulating the chain per element, at `O(n·d)`, in the prover's and the verifier's derivations. `Transpose` and `Expand` say the map directly.
5. **Per-layer instances in stage 1, batched by `γ`.** They would leave `L` claims per weight type.

## Drawbacks

- **Revealed values.** Split values are revealed evaluations, a few per stage. `tessera-protocol/2` masks them as it masks final evaluations ([RFC-0013](RFC-0013-zero-knowledge.md#masked-extensions)).
- **Two more operations.** RFC-0006's operation set grows by two, and the witness engine, the range analysis and the verifier's lowering each implement them.
- **Churn.** #99's `Rotate` is replaced.

## Migration / Rollout

v0.3, for #113, in this order:
1. this RFC;
2. `Rotate`'s kernel;
3. `Transpose`, `Expand` and the committed first checkpoint, in the IR, the lowering and the witness engine;
4. the schedule's derivation, then its stages, its splits and the pool.

The verifier's derivation (#114), the gadgets' leaf layout (#118) and the opening (#119) follow this RFC.

## Testing Strategy

- **Schedule.** On tiny models of 2, 4 and 8 layers in both profiles:
  - equal stage counts, 3 and 4, equal to the commitment report's;
  - one claim per weight type (INV-SCH-5);
  - every claim on a committed tensor in the pool (INV-SCH-3).
- **Stages.**
  - Reduction stages proven from leaf claims, true per-layer values at random points, verify through a verifier written from this RFC.
  - Every pool claim equals its tensors' extensions.
  - A split that does not add up is rejected at its stage.
- **`Rotate`.** The verifier's `Ẽ_r(r')` equals the kernel's extension by enumeration. `Ỹ` equals RoPE as RFC-0005 states it, and a changed input entry is rejected.
- **Layouts.** `Transpose` and `Expand` claims equal direct evaluation at `n` not a power of two.

## Open Questions

None. The leaf layout that puts a lookup site's layers at one point is the gadgets' (#118).

## References

- [RFC-0005](RFC-0005-integer-model.md), [RFC-0006](RFC-0006-compiler-ir.md), [RFC-0008](RFC-0008-logup-and-tables.md), [RFC-0010](RFC-0010-proof-schedule.md), [RFC-0011](RFC-0011-advice-commitment.md), [RFC-0012](RFC-0012-weight-commitment.md), [RFC-0013](RFC-0013-zero-knowledge.md), [RFC-0014](RFC-0014-interaction-and-soundness.md), [RFC-0035](RFC-0035-padded-rows.md), [RFC-0036](RFC-0036-protocol-messages.md).
