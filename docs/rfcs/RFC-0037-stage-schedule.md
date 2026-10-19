# RFC-0037: Stage Schedule: Stacked Layers, Split Claims and the Claim Pool

- Status: Accepted
- Authors: @AbdelStark
- Created: 2026-10-17
- Target milestone: v0.3
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

  The lowering emits such layers for a claim list that rescales every layer's sites alike ([RFC-0038](RFC-0038-uniform-layers.md)). The schedule refuses any other graph with `ERR-INTERNAL`.
- **Kinds.**
  - The outputs at position `p` form one stacked kind, `X_κ[ℓ, …]`.
  - The layer input `X_in[ℓ] = c_ℓ` is a stacked kind.
  - A weight type is stacked over its layers.
  - Prologue and epilogue tensors are kinds without a layer index.
- **Layer variables.**
  - A stacked kind has `λ = ⌈log2 L⌉` layer variables, the highest of its tensor ([RFC-0010](RFC-0010-proof-schedule.md#multilinear-extensions) 'Multilinear extensions').
  - Layers `L` to `2^λ − 1` are padded. Each of their tensors is what its node computes on zero inputs ([RFC-0035](RFC-0035-padded-rows.md#what-a-padded-row-holds)), and their weights are zero. In the Llama templates every virtual tensor of a padded layer is therefore zero, and its `X_in` is zero.

### The committed first checkpoint

- **Lowering.** The `ConstMul` that forms `c_0 = 2^{s_e}·x_0` produces committed advice ([RFC-0006](RFC-0006-compiler-ir.md#llama-lowering) 'Llama lowering').
- **Committed outputs.** A linear or degree-2 node other than a matmul may produce a committed tensor `Y`. Stage 1 proves its relation `f` as an equality zero-check at the stage's zero-check point ([RFC-0035](RFC-0035-padded-rows.md#zero-check-points)): `0 = Σ_x eq(z, x)·(Ỹ(x) − f(x))`. For `c_0` it is `Ỹ − 2^{s_e}·x̃_0`, of degree 2.
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
- **`Reshape`** maps coordinates as the identity when it merges or splits only axes whose lengths are powers of two, the leading axis aside. The lowering uses no other `Reshape`.

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
  - RoPE's constants depend on the position and the frequency, not the head, so the head's coordinates contribute the factor `eq(r_h, r'_h)`.
  - The inner products run over `n·d_head/2` entries.

This replaces RFC-0010's partner form, of degree 3 with two claims, which #99 implemented.

### Claims and atoms

A claim names a kind and a point, `X̃_κ(r ‖ ρ) = v` for a stacked kind with layer coordinates `ρ`. More generally it carries public layer weights `w`: `Σ_ℓ w(ℓ)·X̃_{κ,ℓ}(r) = v`, where `w = eq(ρ, ·)` for a point.

Through a linear node, a claim on its output becomes terms on its inputs, as [RFC-0010](RFC-0010-proof-schedule.md#relations) 'Relations' places them:
- `Transpose`, `Expand` and `Reshape` map coordinates as above.
- A `ConstMul` by one constant multiplies the coefficient. A constant per layer multiplies the layer weights.
- A claim on the layer input reads its slices: `Σ_ℓ w(ℓ)·X̃_in[ℓ](r) = w(0)·c̃_0(r) + Σ_{ℓ≥1} w(ℓ)·X̃_last[ℓ − 1](r)`, with `X_last` the kind of the layer's last output. Every slice is committed.

**Atoms.** After substitution, a claim is a sum of atoms. An atom is a coefficient, a kind, a point and layer weights.
- Atoms on one kind at one point merge, their layer weights adding.
- An atom is virtual (a sumcheck relation's output), committed (advice or a weight) or public (the verifier computes it).
- Atoms are listed in the order substitution reaches them, depth first, inputs in their node's order. A merged atom keeps its first place.

### Splits

The verifier computes a claim's public atoms and moves them to its value. The claim is then ready for the next stage when it is one virtual atom of coefficient 1, or committed atoms of one commitment at one point. Otherwise the stage `k` that produced it splits it:

- **What is sent.** The prover sends one value for each virtual atom, and one for each group of committed atoms of one commitment at one point, in atom order. A group takes the place of its first atom.
  - **A part's value.** A virtual atom's value is its tensor's evaluation, and the check weights it by the atom's coefficient. A committed group's value is its combination `Σ_t a_t·P̃_t(r)`, coefficients included, and the check counts it with weight 1. That combination is the claim the pool receives.
  - **Where.** The values go under `stage/{k}/split`, after the stage's final evaluations, as one `Vec<Fp2>` covering every claim the stage splits, claim by claim.
  - **The order of claims.** A reduction stage's claims come by instance, then by evaluation. The leaf stage's come by site, then by layer, as 'The layer point' orders them.
  - **The leaf stage** is stage 0, and its values follow `logup/values`: one per tensor, site and layer ([RFC-0045](RFC-0045-leaf-layout.md#the-split)).
- **The check.** The verifier checks that the values, weighted by their coefficients, give the claim's value. A mismatch is `Reject(Sumcheck { stage: k, round: n_k })`, with `n_k` the stage's number of rounds (0 for the leaf stage).
- **The result.** Each value becomes one claim: a virtual atom's for the next stage, a committed group's for the pool.
- **Example.** The residual key's claim `acc + 2^r·X_in` at `(r ‖ ρ)` splits into `acc`'s virtual atom and `X_in`'s committed group: two values.
- **Soundness.** The values are fixed before any later challenge. If the claim is false, any split leaves a false claim, which its own check rejects except with the probability that check allows.

### Stages and their instances

- **Stage index.** The leaf stage is stage 0, and an instance's stage is one more than its claim's.
- **Instances.** Stage `k` runs, in order:
  - one instance per virtual atom that stage `k − 1` leaves after its splits, in atom order: the atom's producing relation, stacked as its kind, at the atom's point, with the atom's value as its claim;
  - in stage 1, after them, the zero-checks in node order: RFC-0035's padding zero-checks, `OneHot`'s booleanity, and the equality zero-checks of committed outputs.
- **Stacked instances.** A stacked instance takes the atom's layer weights `w` as a public factor over the layer variables, in place of `eq(ρ, ·)`. The verifier evaluates `w̃(ρ') = Σ_ℓ eq(ρ', ℓ)·w(ℓ)` in `O(L)`, the sum running over the whole cube of `2^λ` layers: a padded layer's weight, such as `eq(ρ, ℓ)`, need not be 0.
- **INV-SCH-2.** Each virtual atom is one instance.
- **Stage count.** The count is one more than the longest chain of sumcheck relations from a gadget's input or a zero-check down to committed tensors. Linear maps cost no stage, so the count does not depend on `L`, `n` or the batch (INV-SCH-4). The compiler's commitment report counts it the same way.

### The layer point

The leaf stage leaves its gadgets' claims one layer at a time. Each lookup site's layers sit at one point `r_s`: `P[0..k]` for its tensors of `2^k` entries ([RFC-0045](RFC-0045-leaf-layout.md#the-leaf-point)).

- **Its claims on the graph.**
  - The leaf stage leaves one claim on each input of each gadget, at a point of its site, and one on the logits, which the scoring certificates read (RFC-0017).
  - Its sites come in node order, a template position once for all its layers, and the logits last.
  - The gadgets' own committed values and relations are RFC-0009's (#118).
- **Drawing it.** Stage 1 draws the layer point `ρ_1`: `λ` coordinates under `stage/1/layer/{c}`, after the leaf stage's splits and before the zero-check point and `γ_1`.
- **Stacking.** A site's claims `v_ℓ` become the stacked claim `X̃(r_s ‖ ρ_1) = Σ_ℓ eq(ρ_1, ℓ)·v_ℓ`. A padded layer's `v_ℓ` is 0.
- **Soundness.** A false `v_ℓ` survives with probability at most `λ/|Fp2|` over `ρ_1`.
- **Later stages.** Their claims are stacked at their own points' layer coordinates.

### The claim pool

A claim ready for the pool is a combination `Σ_t a_t·P̃_t(r) = v` at one point `r`. Each `P_t` is a committed kind with its layer weights, or a committed prologue or epilogue tensor.

- **Advice.** RFC-0011's grouping takes the combination as one claim. Its row coefficients add `γ_claim·a_t·w_t(ℓ)·eq(row part of (r ‖ base_{t,ℓ}), row)` for each piece, and its column sub-point is `r`'s.
- **Weights.** A weight type's claim is `W̃_T(r_i ‖ r_o ‖ ρ)` with `eq(ρ, ·)` its layer weights: RFC-0012's one claim per type.
- **INV-SCH-3.** Every claim on a committed tensor reaches the pool.

### Messages and transcript

- **`TreeProof`** ([RFC-0036](RFC-0036-protocol-messages.md#round2response)) is `{ levels, splits }`: the leaf stage's levels, then its split values.
- **`StageProof`** is `{ stage, sumcheck, splits }`.

Schema version 1 changes in place: no release has written it ([09](../spec/09-release-and-versioning.md)).

[RFC-0014](RFC-0014-interaction-and-soundness.md#transcript-schedule)'s row 6 gains two labels:
- `stage/1/layer/{c}`, squeezed after `stage/1/zk-rho` in `tessera-protocol/2`, and before `stage/1/zero/{c}`;
- `stage/{k}/split`, absorbed after `stage/{k}/evals`. Stage 0's comes after `logup/values`.

Stage 1 runs, in order:
1. `zk-sum` and `zk-rho`, in `tessera-protocol/2`;
2. the layer point;
3. the zero-check point;
4. `γ`;
5. the rounds;
6. the evaluations;
7. the splits.

### Invariants

- **INV-SCH-5.** In the Llama templates every weight type the proof opens has exactly one claim in the pool.

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
