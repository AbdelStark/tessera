# RFC-0072: Proving the Anchor Circuits: Graphs Without Layers, Public Tensors, Selections and the Bit Zero-Checks

- Status: Accepted
- Authors: @AbdelStark
- Created: 2026-10-26
- Target milestone: v0.4
- Amends:
  - [RFC-0066](RFC-0066-bits-and-sha256.md) '`BitDecompose`' (recomposition's relation and its claim on `x`), '`BitFn`' (`¬a` at padded positions), '`AssertEqual`' (a public input pads with zeros, and the selector's example), 'Completeness there is the graph's' (a `ConstAdd`'s padded positions), 'Where the zero-checks run' (their messages), 'Drawbacks' (stacking), and its open questions OQ-0066-1 and OQ-0066-2, which this RFC closes;
  - [RFC-0010](RFC-0010-proof-schedule.md#relations) 'Relations' (recomposition's row, which sums its bit axis, the selection relation, `ConstMul` by a constant per channel, the refusal of a sum over a padded axis), 'Stages' (termination at public tensors) and 'Batching within a stage' (a recomposition's zero-check point);
  - [RFC-0037](RFC-0037-stage-schedule.md#layers-and-kinds) 'Layers and kinds' (a graph with no checkpoint, and the operations a layer template may not hold), 'The committed first checkpoint' (the committed outputs stage 1 proves), 'Claims and atoms' (public tensors, `ConstAdd`'s shift, `Slice`, `GatherPublic` and a per-channel `ConstMul` as relations, combined selections, one point), 'Splits' (a combined selection's value), 'Stages and their instances' (stage 1's new zero-checks, the stage count) and 'The layer point' (no site for `BitDecompose` and `AssertEqual`, and the logits when there are some);
  - [RFC-0035](RFC-0035-padded-rows.md#zero-check-points) 'Zero-check points' (a recomposition's `eq` takes the highest of its instance's coordinates) and 'What a padded row holds' (`ConstAdd` adds its constant at real positions only, and `SoftmaxRow` and `InverseCdf` look up their masked key there);
  - [RFC-0045](RFC-0045-leaf-layout.md#the-split) 'Sites' (none for `BitDecompose` and `AssertEqual`) and 'The split' (a public tensor's value is not sent);
  - [RFC-0006](RFC-0006-compiler-ir.md#gadget-selection) 'Gadget selection': `BitDecompose` and `AssertEqual` take gadgets; 'Operation set': only a pointwise node's committed output is proven;
  - [RFC-0046](RFC-0046-advice-segments.md#segments) 'Segments': a graph with no layers;
  - [RFC-0008](RFC-0008-logup-and-tables.md#table-kinds) 'Table kinds': `Bitwise8` serves AES-128 alone (RFC-0066);
  - [RFC-0014](RFC-0014-interaction-and-soundness.md#soundness-calculator) 'Soundness calculator': a recomposition's `zero_checks` entry.
- Origin: [#526](https://github.com/AbdelStark/tessera/issues/526)

## Summary

RFC-0015's anchor proves the transcript with two circuits: AES-128-CTR (#142) and SHA-256 over committed bits (#143, RFC-0066). Both exist as IR graphs and as witnesses, but neither can enter a proof. Gadget selection refuses RFC-0066's operations, and the schedule cannot derive their stages:
- it assumes layers, checkpoints and logits;
- it has no relation for `GatherPublic`, `Slice` or a `ConstMul` by a constant per channel;
- it cannot claim a tensor the verifier knows;
- RFC-0066 left the zero-checks' messages open.

This RFC fills each gap with RFC-0010's existing machinery and adds no new kind of message, label or opening group:
- **A graph with no checkpoint has no layers.** Every node is outside them, as a model's prologue and epilogue are.
- **Public tensors.** A tensor computed from public tensors alone, by a node that is not gadget-backed, is public: the verifier evaluates its claims itself.
- **The selection relation.** `GatherPublic` and `Slice` reduce a claim through one sumcheck over the selected axis, against a public 0/1 kernel, of degree 2 (OQ-0066-2).
- **`ConstMul` by a constant per channel** is RFC-0010's `Elementwise` with the constant a public operand.
- **Combined selections.** Within one claim, selections of one tensor along one axis at one point are one selection, whose kernel weighs theirs by their coefficients. Without that, the two selections that read each level of a SHA-256 sum's value, its even and odd rows, would double the instances below it at every level. Claims are never merged across a stage, so no claim goes unchecked and no Llama schedule changes.
- **Padded positions** hold what their operation computes from their inputs' padded positions, except that a `ConstAdd` adds its constant at real positions only. They are 0 except downstream of a lookup non-zero at padding, such as AES's S-box, and a sum over a padded axis that such values reach is refused.
- **The bit zero-checks** run in stage 1 and send their evaluations under `stage/1/evals` (OQ-0066-1). Recomposition is one zero-check summed over the bit axis, which needs no value before `γ`. `BitDecompose` and `AssertEqual` take gadget ids. A false bit fails stage 1's final check, `Reject(Sumcheck)`.

A SHA-256 block of a message with committed bytes holds 680 zero-checks, 2,528 selections, 352 `BitFn` products and 96 masked `Mul`s: about 3,400 instances once selections combine, against about 71,700 if they did not. A frame's SHA-256 takes 14 stages, the leaf stage included, whatever its block count. AES-128-CTR takes 3.

## Motivation

- **RFC-0066 OQ-0066-1** leaves open where the zero-checks' claims are sent and under which label. It also leaves open how recomposition's two sums batch with `x̃(z)`, what `AssertEqual` sends, the gadget ids, the reject reasons and stacking. Until they are settled, gadget selection refuses the operations (`ERR-QUANT-ARCH`).
- **OQ-0066-2** names a candidate relation for `GatherPublic`. `ROTR`, `SHR`, the message's assembly and AES's `ShiftRows` and MixColumns all reduce claims through it. RFC-0037 'Head layouts' removed `GatherPublic` from the Llama lowering, so no relation proves it.
- **The schedule assumes a model.**
  - RFC-0037 'Layers and kinds' refuses any graph whose layers do not follow the template.
  - The leaf stage always leaves a claim on the logits.
  - Neither circuit has a layer, a checkpoint or a logit.
- **Claims through `Slice`, `ConstAdd` and public tensors.**
  - SHA-256's words are `Slice`s of the message's bits and of the initial value.
  - Its sums add their round constant with `ConstAdd`.
  - The `SHR` masks, the message's public bits, the digest, AES's counters and its round constants are public.
  - RFC-0037 'Claims and atoms' names public atoms but says nothing of how a tensor becomes one.
- **AES's key schedule** masks each round key to its columns with a `ConstMul` whose constant has one value per byte. No RFC states its relation.

## Goals

- Every operation of both circuits has a relation or moves claims without one, and every claim has a destination: the pool, an instance of the next stage, or the verifier.
- No new kind of message, transcript label or opening group. A claim one of these relations leaves names one point. The anchor's committed claims form RFC-0011 groups at their own column sub-points, as any claims do ('Drawbacks').
- The Llama graphs' schedules, vectors and proofs are unchanged.

## Non-Goals

- **`tessera-protocol/2`.** The circuits' committed bits are advice, masked under RFC-0013's rules. Their sums and some of their carries at `M ≤ 2` have fewer than the two variables a masked tensor needs, so #145 pads them, or R10's check refuses the schedule (#488). Their virtual tensors take RFC-0013's masks and folds, in #145's argument. The selection relation leaves its claims at random points, so it adds no Boolean coordinate that R8 forbids.
- **The anchor's wiring.** #146 puts the circuits into a batch's graph, its inputs taken from the log and the session, and its statement's sizes; #163 tests it end to end.
- **The verifier's mirror** of these relations is #150's and #159's.
- **Token groups.** RFC-0054 refuses a `Bitwise` site, and the other operations its token axis lists, in a leaf stage in token groups. RFC-0054's token axis has no rule for RFC-0066's operations either. So a batch that carries the anchor circuits runs its leaf stage without token groups, or #146 amends RFC-0054.
- **Stacking instances of one kind** ('Alternatives Considered' 3).

## Proposed Design

### Graphs without layers

- **No checkpoint, no layers.** A lowering with no checkpoint has `L = 0` layers and `λ = 0` layer variables (RFC-0037 'Layers and kinds').
  - Every node is outside the layers, and every tensor is a kind without a layer index, like a prologue's or an epilogue's.
  - Stage 1's layer point has no coordinate, so `stage/1/layer/{c}` squeezes nothing.
- **No logits.** A graph with no logits leaves no claim on them and has no scoring site.
- **Advice segments.** With `L = 0`, every tile is in segment 0, and any further segments are empty ([RFC-0046](RFC-0046-advice-segments.md#segments) 'An empty segment').
- **In a model's graph.** #146 adds the circuits to a batch's graph after the model's nodes. Their nodes are then epilogue nodes, outside the layers, under the same rules. They read no checkpoint.
- **Not inside a layer.** A layer template holding a `ConstAdd`, a selection, a per-channel `ConstMul`, a `BitDecompose`, an `AssertEqual`, a lookup node non-zero at padding ('Padded positions'), or a node whose output is public is refused (`ERR-INTERNAL`): their stacked forms, a per-layer constant on a padded layer, stacked bit zero-checks, and the values of a padded layer that are not 0 among them, are not specified. `BitFn` and `Bitwise` stack as any relation and lookup do. The Llama lowering emits none of them.

### Public tensors

- **What is public.**
  - A graph input declared public.
  - A constant.
  - The output of a node that is not gadget-backed and whose inputs are all public, such as a `GatherPublic` of the `SHR` masks or a `Slice` of the digest.
  - A tensor declared committed is committed, whatever its inputs.
- **Their claims.** An atom on a public tensor is public (RFC-0037 'Claims and atoms'). The verifier computes the tensor from public data and evaluates its extension at the atom's point, `O(size)`, then moves the atom to the claim's value, as RFC-0037 'Splits' does. The atom reaches no instance and no pool. A claim whose atoms are all public has no part: the verifier computes its value, and nothing is sent or checked for it. Such claims arise in the leaf stage, on a lookup's public input, whose value RFC-0045 'The split' does not send.
- **Once per point.** The verifier evaluates a public tensor, or a selection's kernel, once per distinct point. A stage's instances of one relation and one size end at its last challenges, so their reads of one public operand share a point.
- **As an operand.** A relation's operand that is public is read as `Public` (RFC-0010 'Relations'): the verifier evaluates it at the instance's point, a broadcast operand such as an `SHR` mask at its own coordinates of that point, and the proof sends nothing for it. The same holds for a public input of a committed output's equality zero-check. A matmul cannot read an operand as public: a `MatMulWeight` or `MatMulAct` with a public operand is refused when the schedule is derived (`ERR-INTERNAL`). Neither circuit nor Llama has one.
- **In the leaf stage.** A lookup that reads a public tensor, such as AES's counter blocks, sends no split value for it (RFC-0045 'The split'). The verifier evaluates it at the block's point.

### Padded positions

**Real positions.** A tensor's real positions are those inside its shape `(n_a)`. Its selector `ψ(x) = Π_a [x_a < n_a]` is 1 on them, with extension `ψ̃(r) = Π_a LT(r_a, n_a)`: the selector of `ConstAdd`, of `AssertEqual` and of the input padding check alike.
- **In a batch of several requests** (RFC-0071), the rows of each slot past its request's length are RFC-0071's padded rows, although they are real positions here, inside the token axis's shape ('Real and padded rows'). So a tensor downstream of the tokens or the positions, by any path through the graph, would need RFC-0071's row factor on its token axis.
- **Refused until #146.** A `ConstAdd` or an `AssertEqual` with an input so downstream, in a graph of more than one request, is refused when the schedule is derived (`ERR-INTERNAL`). Reachability decides it on any graph, without RFC-0054's token axis. #146 binds the anchor to the tokens and states that selector. The circuits read neither the tokens nor the positions, and Llama has neither operation.

Every tensor's padded positions hold what its operation computes from its inputs' padded positions, as RFC-0035 'What a padded row holds' says. `ConstAdd` is the exception, and selections, `Expand` and `Rotate` are 0 on their padded rows by their relations. That holds for virtual and public tensors too:
- graph inputs and constants pad with zeros, and a per-channel constant is 0 at padded channels;
- **a `ConstAdd` adds its constant at real positions only**, `y = x + ψ·c` with `ψ` 1 on the tensor's real positions, so its padded positions are its input's;
- a selection's padded rows on its selected axis are 0, and its other padded positions are its input's;
- an `Expand` is 0 past `len`, as its coefficient `LT(r_a, len)` says;
- a `Rotate`'s padded rows are 0, as its kernel is (RFC-0037);
- a lookup's are the lookup of what its node computes there (RFC-0045), such as `AesSbox(0) = 0x63`.

Every other operation maps zeros at padded positions to zeros, and so does every gadget but a lookup:
- RFC-0035's padding zero-checks pin `RsqrtRow`'s `ρ`, `SoftmaxRow`'s `c_q` and `R_q` and `InverseCdf`'s `c` to 0 on a padded row;
- `OneHot`'s output is the zero vector on a padded row (RFC-0035) and 0 at padded vocabulary entries (RFC-0009 'OneHot');
- `SoftmaxRow`'s exponentials `e` are 0 at every padded position: each looks up the masked key `K_0`, which its table maps to 0 (RFC-0009, RFC-0030);
- `BitDecompose`'s bits are its input's, 0 where the input is 0.

So a padded position is non-zero only downstream of a lookup node that is **non-zero at padding**: a `Requant`, `TableMap` or `Bitwise` node whose function maps zero inputs, in any of its classes, to a value other than 0. That is decided from public data: `Requant`'s `rq(0)` from its shifts and bounds, a table's value at the key that packs input 0 with each class. Of the lowerings' nodes only a `TableMap` into `AesSbox` is (`0x63`): `rq(0) = 0` at every Llama site, and SiLU, `Gf256Mul` and `Bitwise8` map 0 to 0. RFC-0035's `Exp2Frac` and `Pow2` are lookups inside the scoring bound, not graph nodes, and a `Range` lookup has no output.

So the prover's witness and stage tables, and the verifier's public tensors, are computed over the padded shapes. With `M` messages or `B` blocks not a power of two:
- SHA-256's padded messages are 0 throughout: their bits, sums and carries.
- AES's padded blocks run on zero counters under zero round keys, since each round key's `Expand` is 0 past `B`. Their S-box outputs are then `AesSbox(0) = 0x63` and its consequences, key-independent constants, committed with the block's other lookup outputs, not `AES_k(0)`.

A committed graph input's padded positions are computed by nothing. RFC-0006 'Committed graph inputs' admits a graph only when a `BitDecompose` or a `OneHot` reads each committed advice input, and their checks pin its padded positions: RFC-0066's input padding check, or `OneHot`'s row selector. AES-128-CTR alone is a fragment, whose key and plaintext a larger graph proves (#146).

### Linear operations that are not coordinate maps

- **`ConstAdd`** moves its constant into the claim's value: `ỹ(r) = x̃(r) + ĉ(r)`, where `ĉ` is the multilinear extension of `ψ·c`, the constant at real positions only ('Padded positions'). Its constant is a public atom: a claim `a·ỹ(r) = v` becomes `a·x̃(r) = v − a·ĉ(r)`. A `ConstAdd` is never inside a layer ('Graphs without layers'), so no layer weights arise.
  - **The selector.** `ψ̃(r) = Π_a ψ̃_a(r_a)` with `ψ̃_a = LT(r_a, n_a)`, one factor per axis of length `n_a` ('Real positions').
  - **A constant of one value** contributes `ĉ(r) = c·ψ̃(r)`.
  - **A constant per channel**, one value per position of the last axis, contributes `ĉ(r) = c̃(r_c)·Π_{a≠last} ψ̃_a(r_a)`, with `r_c` the last axis's coordinates and `c̃` the constant's extension, zero at padded channels. The constant is already 0 at padded channels, so the last axis's factor does not appear: `c̃(r_c)·ψ̃(r)` agrees with `ĉ` on the cube only, and is not multilinear when the last axis's length is not a power of two.
  - The verifier evaluates each `LT` factor in `O(log)` time, and `c̃(r_c)` in `O(channels)`.
- **`ConstMul` by a constant per channel** is RFC-0010's `Elementwise`: `Ỹ(r) = Σ_x eq(r, x)·c̃(x_c)·X̃(x)`, degree 3, the constant a public operand broadcast over the other axes. It leaves one claim on `X`. A scalar `ConstMul` stays a coefficient.
- **Scalar or per channel.** A constant of one value is a scalar, whatever the last axis's length: a `ConstMul` or `ConstAdd` on a last axis of length 1, such as SHA-256's value levels at `M = 1`, is a scalar's. A constant per channel has as many values as the last axis, and more than one.
- **`Slice` and `GatherPublic`** are selections, below.
- **A committed output.** RFC-0037's equality zero-check proves a committed output whose node computes it pointwise: `ConstMul` (scalar or per channel), `ConstAdd`, `Add`, `Sub` and `Mul`, with `f` the operation itself as a multilinear polynomial in each input: `ConstAdd`'s `f = x̃ + ĉ`, `ĉ` the extension above, so that both workspaces' round polynomials agree. A committed output of any other node but a gadget is refused: a matmul's by the IR when the graph is built (RFC-0006 'Operation set'), and that of any other linear or degree-2 node, a selection, `Sum`, `SumSquares`, `Rotate`, `Transpose`, `Reshape` or `Expand` among them, when the schedule is derived (`ERR-INTERNAL`), although the IR admits the declaration. `BitFn`'s output is virtual (RFC-0066). Neither circuit declares one, and the Llama lowering's only one is `c_0`, a scalar `ConstMul`'s.
- **Summing relations.** `Sum`, `SumSquares` and `MatMulAct` sum over an axis of an input, `MatMulAct` over its inner axis. Their relations are the operations only when an input they sum is 0 at the axis's padded positions, since RFC-0010 sums over the whole padded cube. `MatMulWeight` sums over its input axis against weights that are 0 there.
  - **The rule.** Such a relation over an axis whose length is not a power of two is refused when the schedule is derived (`ERR-INTERNAL`), unless one input it sums over that axis is 0 at the axis's padded positions: no lookup node non-zero at padding ('Padded positions') lies upstream of it in the graph, or it is an `Expand` or a selection along that axis, which are 0 there whatever their input.
  - **Committed graph inputs.** A committed advice graph input counts as non-zero at padding unless RFC-0006 'Committed graph inputs' pins it: a `OneHot` reads it directly, or a `BitDecompose` reads it directly or moved. On a graph that range analysis admits every such input is pinned, so the clause changes no verdict there. It keeps the rule sound on a graph whose schedule is derived without range analysis.
  - **Which input.** `Sum` and `SumSquares` have one. `MatMulAct` sums both of its inputs over its inner axis, and one such input suffices, since a product is 0 where either factor is.
  - **Where it applies.** No circuit sums over a padded axis. Llama's lookup nodes, `Requant` and SiLU's `TableMap`, map 0 to 0, and its gadgets' outputs, `SoftmaxRow`'s `e` among them, are 0 at padded positions ('Padded positions'), so no Llama schedule changes: `PV` sums `e` over the padded key axis. The compiler's schedule tests sum an `Expand` along its own axis.

### The selection relation

A selection takes rows of `x` along one axis `a`. `GatherPublic` selects the outermost axis's rows `idx_j`, and `Slice { axis, start, len }` its axis's rows `start + j`. Let `n_a` be `x`'s length on that axis, `len` the number of rows selected, and `S(j, k) = [j < len]·[idx_j = k]`. Then:

```text
ỹ(r_lo ‖ r_a ‖ r_hi) = Σ_{k ∈ {0,1}^{⌈log2 n_a⌉}} S̃(r_a, k)·x̃(r_lo ‖ k ‖ r_hi)
```

Points list low variables first (RFC-0010 'Multilinear extensions'). Here `r_a` is the claim's coordinates of `y`'s axis `a`, `r_lo` its coordinates of the axes after `a`, which hold the lower variables, and `r_hi` those of the axes before it. The relation keeps both.
- **The sumcheck.** It runs over `k` alone: `⌈log2 n_a⌉` variables, of degree 2, with `S̃(r_a, ·)` its one public factor, as `Rotate`'s kernel is (RFC-0037 '`Rotate`'). Its variables take the stage's last `⌈log2 n_a⌉` challenges `r'_k`.
- **Its claim.** It leaves one claim on `x`, at `r_lo ‖ r'_k ‖ r_hi`.
- **The verifier's evaluation.** `S̃(r_a, r'_k) = Σ_{j < len} eq(r_a, j)·eq(r'_k, idx_j)`, in `O(len·(⌈log2 len⌉ + ⌈log2 n_a⌉))`: 32 terms for a rotation, 16 for AES's.
- **Padded rows.** `S` is 0 on `y`'s padded rows, so they are 0, which is what the operation computes there: no index selects them. `x`'s padded rows are never selected.
- **Why not a coordinate map.**
  - A gather's indices are not a concatenation of bits in general (RFC-0037 'Head layouts').
  - A `Slice` aligned to a power of two would map coordinates, but its claim on `x` would then have Boolean coordinates on the axis, which `tessera-protocol/2`'s masks cannot cover (R8).

### Combined selections

Each level of a SHA-256 sum's value is `Add(even, ConstMul(odd))`: two selections of one tensor, its even and its odd rows. A claim on the level's output substitutes to two virtual atoms, the selections, at one point. RFC-0037 makes one instance per virtual atom, and each instance leaves its own claim on the tensor below, so each level would double the instances below it.

- **The rule.** Within one claim, two or more virtual atoms that select rows of one input tensor along one axis, at one point with one set of layer weights, are one atom: `Σ_t a_t·y_t` with `a_t` their coefficients. A single selection atom is not combined: its coefficient weighs its split value, as RFC-0037 says. It is a selection of that tensor with the kernel `K = Σ_t a_t·S_t`, and the relation above proves it with `K̃` in place of `S̃`. "One point" is RFC-0037's, under which atoms of one kind already merge: the same challenges and constants, in the same order, as both parties derive them.
- **Its value.** When the combination is the claim's only part, the claim is ready. Otherwise its split sends one value for it, weighted by 1, at its first atom's place (RFC-0037 'Splits').
- **The verifier's evaluation.** `K̃(r_a, r'_k) = Σ_t a_t·S̃_t(r_a, r'_k)`, each term as above.
- **Soundness.** The combination is a linear functional of `x`, so its identity is one sumcheck, of degree 2. The members' own values are never sent and never needed.
- **No merging across claims.** Two claims on one tensor at one point stay two claims, each with its instance. One instance for both would prove one claim's value and leave the other's unchecked: AES's `Gf256Mul { 2 }` and `Gf256Mul { 3 }` lookups read one tensor, ShiftRows' output, at one leaf point, and SHA-256's round 0 leaves two claims on its first message word at one point. Llama's composed schedule holds such pairs too, so a merge would also change `tessera-protocol/1`'s messages.
- **Llama** has no selection, so this rule never applies to it.

### The bit zero-checks (OQ-0066-1)

They run in stage 1, after its other zero-checks, in node order, each node's checks in the table's order (RFC-0066 'Where the zero-checks run'). Each is one instance at the zero-check point `z` (RFC-0035 'Zero-check points'): an instance of `n_i` variables takes `z`'s last `n_i` coordinates, and its `eq` the highest of them.

| Check | Identity | Instance variables | `eq` variables | Evaluations, in order | Degree |
|---|---|---|---|---|---|
| Input padding, of the committed graph input `u` a `BitDecompose` reads, directly or moved (RFC-0066), when `u` has padded positions | `0 = Σ_x eq(z, x)·(1 − ψ̃(x))·ũ(x)`, `ψ` 1 on `u`'s real positions | `n_u` | `n_u` | `ũ` | 3 |
| Booleanity of `b`, of `h` | `0 = Σ_x eq(z, x)·b̃(x)·(1 − b̃(x))` | `n_y + k_b`, `n_y + k_h` | all | `b̃`, `h̃` | 3 |
| Bit padding of `b`, of `h`, when its `low` or `high` bits are not a power of two | `0 = Σ_x eq(z, x)·(1 − LT(x_bit, low))·b̃(x)`, `high` for `h̃` | `n_y + k_b`, `n_y + k_h` | all | `b̃`, `h̃` | 3 |
| Recomposition | below | `n_y + k` | `n_y` | `x̃` unless public, `b̃`, `h̃` | 2 |
| `AssertEqual` | `0 = Σ_x eq(z, x)·ψ̃(x)·(ã(x) − b̃(x))` | `n_a` | `n_a` | `ã`, `b̃`, each unless public; none when both are | 3 |

`n_y` is the variables of `x`'s shape, `n_u` those of `u`'s, and `n_a` those of `AssertEqual`'s tensors. Every `ψ̃` is that of 'Real positions'. `k_b = ⌈log2 low⌉` and `k_h = ⌈log2 high⌉` are the bit axes' variables, which are the lowest.

**Recomposition.** It is one zero-check over `y`, summed over the bit axis, `k = max(k_b, k_h)` variables, `k_b` when `high = 0`:

```text
0 = Σ_{(i, y) ∈ {0,1}^{k + n_y}} eq(z_y, y)·[2^{−k}·x̃(y) − W_b(i)·b̂(i ‖ y) − 2^{low}·W_h(i)·ĥ(i ‖ y)]
```

- **Zero-extended outputs.** An output of fewer bit variables than `k` takes the *high* ones: `b̂(i ‖ y) = b̃(i_{≥ k − k_b} ‖ y)·Π_{t < k − k_b}(1 − i_t)`, and `ĥ` likewise with `k_h`. So bit `j` of `b` sits at `i = j·2^{k − k_b}`. `h` is absent when `high = 0`.
- **The weights.** `W_b(i) = 2^j` at `i = j·2^{k − k_b}` for `j < low`, and 0 otherwise. `W_h` is the same with `k_h` and `high`. They vanish where the extension does, so its factor changes no sum, but both parties use this polynomial.
- **Why the high coordinates.** Recomposition's claims on `b` and `h` then fall at the last `n_y + k_b` and `n_y + k_h` challenges, the points of their booleanity checks, so stage 1 opens them in the same groups.
- **Why it holds.** For each `y` the sum over `i` is `x(y) − Σ_{i < low} 2^i b_i − 2^{low}·Σ_{j < high} 2^j h_j`, RFC-0066's recomposition, and the `eq` over `y` zero-checks it, with error `n_y/|Fp2|`.
- **Its claims.** At the instance's final point `r' = r'_i ‖ r'_y`:
  - `x̃(r'_y)`, a claim on `x` at `y`'s coordinates. It goes to the next stage when `x` is virtual, as a SHA-256 sum's value is, and to the pool when it is committed, as the key commitment's bytes are. When `x` is public the verifier evaluates it and the proof sends nothing.
  - `b̃(r'_{i, ≥ k − k_b} ‖ r'_y)` and `h̃(r'_{i, ≥ k − k_h} ‖ r'_y)`, claims on committed outputs, for the pool, at their booleanity checks' points.
- **The verifier's check** computes the public factors itself: `eq(z_y, r'_y)`, the zero-extensions' products, and the weights on `k` variables. With `s = (r_{k − k_b}, …, r_{k − 1})`, the high `k_b` coordinates, low first, `W̃_b(r) = Π_{t < k − k_b}(1 − r_t)·Σ_{j < low} 2^j·eq(s, j)`, in `O(low·k)`. When `low = 2^{k_b}` the sum is `Π_{t < k_b}(1 − s_t + s_t·2^{2^t})`. `W̃_h` is the same with `high` and `k_h`.
- **Why one instance.** Stage 1 batches its instances with `γ`, after each instance's claim is fixed (RFC-0010 'Batching within a stage'). RFC-0066 wrote recomposition as `x̃(z)` minus two sums of `n_y + k_b` and `n_y + k_h` variables. As three instances, the two sums' values, unknown to the verifier, would have to be sent before `γ`, a new message. Summed over the bit axis, the instance's claim is 0, and `x`'s claim moves from `z` to `r'_y`.
- **Degree.** Each variable is of degree at most 2: `eq·x̃` in the `y` variables, and `W·b̂` in the bit variables. Stage 1's degree is 3 either way, from booleanity.

**`AssertEqual` with neither side public** sends `ã(r')` then `b̃(r')`, two claims at one point on two tensors, each to its destination. With one side public, the proof sends only the other.

**Which variables count.** A zero-check errs over its `eq`'s variables, `n_y` for recomposition. RFC-0014's `zero_checks` lists each instance's variables, `n_y + k` for recomposition, as the calculators count `OneHot`'s row-summed checks today: an overcount, never an undercount. Stage 1's zero-check point has as many coordinates as its largest zero-check instance (RFC-0035).

### Gadgets and leaf sites

- **Gadget ids.** `BitDecompose` is gadget-backed (RFC-0066). `AssertEqual`, a relation with no output, is of the IR's gadget class too (`OpClass::Gadget`). So each takes a gadget, numbered in node order with every other ([RFC-0006](RFC-0006-compiler-ir.md#gadget-selection) 'Gadget selection'). Their preconditions are statement checks ([RFC-0039](RFC-0039-statement-fields.md)): `low + high ≤ 62`, and each input's range, which the verifier re-derives (SP-7). `BitFn` is virtual and takes none. No Llama graph holds either operation, so no Llama gadget id moves.
- **No leaf site.** The leaf stage gives them none: RFC-0045's sites are every gadget node but these two. A site carries the leaf stage's claims on its gadget's inputs, and every claim these two leave comes from stage 1's zero-checks. `OneHot` keeps its site, whose claim on its token input its native index binding replaces, so no Llama site moves.
- **AES's lookups.** `TableMap` into `AesSbox` and `Gf256Mul`, and `Bitwise` into `Bitwise8`, are lookup gadgets. Each is a site outside the layers in the leaf stage, in node order (RFC-0045), with RFC-0008's keys: the input for a `TableMap`, `a + 2^8·b` for a `Bitwise8`.

### Messages, rejections and stages

- **An empty leaf stage.** A graph with no lookup and no product tree, such as SHA-256 alone, still runs the leaf stage: it squeezes `α` and `β` and absorbs empty `logup/roots`, `logup/values` and `stage/0/split` (RFC-0010 'Tree levels').
- **No reduction stage.** A graph whose leaf stage leaves no virtual part has no stage 1, unless it has zero-checks. Its pool then holds the committed parts of the leaf stage's claims, after the leaf stage's splits.

- **Messages.** No label is added.
  - Stage 1's new instances send their evaluations under `stage/1/evals`, instance by instance, in the table's order within each.
  - The reduction stages' selections, `BitFn`s, `Mul`s and `ConstMul`s send theirs under `stage/{k}/evals`.
  - Splits go under `stage/{k}/split` (RFC-0037).
  - A stage that holds a `BitFn` of degree 4, `Xor3` or `Maj`, sends round polynomials of 4 values (RFC-0010 'Sumcheck messages').
- **Rejections.**
  - A false bit, a false carry, a padded slot set to 1, or an input whose decomposition does not recompose: the prover commits a value no honest witness has. With a digest one bit off, either the binding's zero-check is false or some committed value is. Either way stage 1's final check fails, `Reject(Sumcheck { stage: 1, round: n_1 })`, with `n_1` stage 1's rounds.
  - A prover that lies in a round instead fails the stage's final check, a split's check or a later stage's, or the opening (`Reject(Evaluation)`, RFC-0019 step 9). A round only derives `p(1)` and checks its message's length.
  - None is `Reject(Gadget)`: a batched nonlinear relation names no gadget ([gadget finals](../testing/gadget-finals.md) 'Rejection order and alternatives').
  - An S-box, `Gf256Mul` or `Bitwise8` output off its table fails LogUp: `Reject(Lookup { table })` in `tessera-protocol/1` (RFC-0008), `Reject(LogUpIdentity { round })` in `tessera-protocol/2` (RFC-0019 step 7).
- **Stages.** The stage count, the leaf stage included, is one more than the longest chain of sumcheck relations down to committed or public tensors (RFC-0037 'Stages and their instances'). A chain starts at a leaf site's input, or at a zero-check, which counts itself. `Add`, `Sub`, scalar `ConstMul`, `ConstAdd`, `Reshape`, `Transpose`, `Expand` and `Sum` cost none.

**The circuits' sizes,** counted from `tessera_compiler::anchor`'s builders and their schedules under these rules:

| Circuit | Zero-checks | Selections | `BitFn` | `Mul` | `ConstMul` per channel | Lookups | Instances | Evaluations | Split values | Stages |
|---|---|---|---|---|---|---|---|---|---|---|
| SHA-256, per block of public bytes | 680 | 2,441 | 335 | 79 | 0 | 0 | 2,945 | 4,237 | 627 | 10 |
| SHA-256, per block of a message with committed bytes | 680 | 2,528 | 352 | 96 | 0 | 0 | 3,436 | 4,764 | 697 | 14 |
| SHA-256, the key commitment (one block, 32 committed bytes, bound) | 690 | 2,523 | 349 | 97 | 0 | 0 | 3,431 | 4,739 | 681 | 14 |
| AES-128-CTR, per circuit | 0 | 97 | 0 | 0 | 30 | 127 | 154 | 154 | 332 | 3 |

- **The columns.** Evaluations are the reduction stages' final evaluations (`stage/{k}/evals`). Split values are every `stage/{k}/split`, the leaf stage's tensor values included: AES's 332 are its lookups' 205 non-public inputs and 127 outputs (RFC-0045 'The split').
- **Per block.** The 680 zero-checks are 368 booleanity checks, 128 bit-padding checks for the round carries and 184 recompositions. The per-block rows count a block after the first, at `M = 1`. A first block reads the initial value, part of which is public: 2,415, 331 and 79 for public bytes. A frame's last block can hold padding alone, and still costs the committed row, since its words are `Slice`s of the assembled bits.
- **Without combined selections** a block would hold about 55,000 instances of public bytes and 71,700 with committed bytes, 85,000 and 103,000 evaluations, 31,000 and 34,000 split values.
- **Per graph.** The binding adds 8 `AssertEqual`s. Each `BitDecompose { 8, 0 }` of committed bytes adds its booleanity, its recomposition and, when it reads a committed graph input with padded positions, that input's padding check.
- **Chain length.**
  - A public message's chain is a word's `SHR` (a selection, then a `Mul`), its `σ`'s `Xor3`, and the sum's five levels of selections. That is 8 relations below the zero-check, 9 with it, so 10 stages.
  - Committed bytes add a message word's `Slice`, `Xor`, `Mul` and selection: 12 relations below the zero-check, so 14 stages.
  - Each block's carries and words are committed, so the chain does not grow with the block count (INV-SCH-4).
- **AES.** Its lookups' keys reach committed or public tensors through at most two relations, a selection and a `ConstMul` or MixColumns' two selections, so 3 stages.

### Soundness

- **The new relations** are sumcheck identities of RFC-0010's form, with public factors the verifier computes: the selection kernel, the weights `W`, the zero-extensions and `ψ`. Each instance counts in its stage's `m` and its stage's rounds in `n·d`, as the calculator counts every instance (RFC-0014).
- **Each zero-check** errs with probability at most its `eq`'s variables over `|Fp2|`, and its `zero_checks` entry counts at least as many.
- **Public atoms** are computed by the verifier, with no error.
- **Uniqueness** is RFC-0066's: Lemma 1 for the bits, given each decomposition's input, and the lookups' outputs, unique for their keys. The committed inputs' padded positions are pinned ('Padded positions'). `BitFn` and the selections' outputs are virtual.
- **Combined selections.** A combination is one linear functional of its tensor, proven by its own sumcheck; no claim is merged with another ('Combined selections').

## Alternatives Considered

1. **Recomposition as three instances**, `x̃(z)` and the two sums.
   - **The cost.** Each decomposition sends the two sums' values before `γ`, under a new tier-A label, and leaves `x`'s claim at `z`.
   - **Why rejected.** The summed instance sends nothing more.
2. **An aligned `Slice` as a coordinate map.**
   - **The advantage.** It is free: every SHA-256 slice is 32 rows at a multiple of 32.
   - **Why rejected.** Its claim on `x` fixes Boolean coordinates on the axis, where `tessera-protocol/2`'s masks vanish (R8). It also needs a rule for unaligned slices.
3. **Stacking instances of one kind.** Nodes with one operation and shapes would become one instance on `⌈log2 K⌉` more variables, each member sending its evaluations.
   - **What it saves.** The proof's size and the prover's work stay the same. It saves the verifier's per-instance work and lowers the calculator's union of zero-checks.
   - **The union.** The union sums every zero-check's variables, 680 entries a block.
   - **A tighter bound.** Every zero-check of stage 1 shares the point `z`. A cheating prover must make each false relation's zero-check vanish there, and each does so with probability at most its own variables over `|Fp2|`. So the largest entry bounds the event, not their sum (OQ-0072-1).
   - **Why rejected.** Stacking needs a layout of identical members that the circuits' sequential rounds do not have.
4. **A committed copy of each gathered tensor.** It needs no relation but commits 2,528 more tensors per block.
5. **Merging equal claims across a stage**, by sending a value once.
   - **The advantage.** It would also cut Llama's composed stage 2, whose natives repeat claims.
   - **Why rejected.** It is sound only for claims that read one sent value through one map, and the leaf stage's per-site values, derived values and native claims do not. Merged otherwise, it leaves claims unchecked ('Combined selections'). It would also change `tessera-protocol/1`'s messages for Llama.
6. **Each sum's value as one relation.** `x̃(r) = Σ_k 2^k·rows̃(r ‖ k)` over the row axis is a selection-like sumcheck with the public kernel `2^k`.
   - **What it saves.** The value levels would cost one stage instead of five, so the chains would be 4 and 8.
   - **Why not now.** The IR has no operation for it. RFC-0066 builds the value in five levels so that range analysis bounds it exactly, and a new operation would need that range rule.

## Drawbacks

- **Many small instances.** A block adds about 3,000 to 3,400 instances to its stages. Each is front-padded to its stage's rounds (RFC-0010).
- **The calculator's union.** A block adds those instances and about 2,400 zero-check variables at `M = 1`, so 500 blocks add about `2^{−106.5}` to the soundness error (OQ-0072-1).
- **Public tensors.** The verifier evaluates every public tensor once per distinct point, `O(size)` each. Its largest is the message's public bits, `512·N·M` entries.
- **A message of many blocks costs the prover `O(N²)`.** RFC-0066 assembles a message's bits as one `[512·N, M]` tensor. So each block's words, when the message has committed bytes, reach it through four instances that each run over all `512·N` rows: a `Slice`, the `Xor`, the placing `Mul` and the assembly's selection.
  - **The size.** Their tables sum to about `2·10^5·N²·M` entries, against about `70,000·N·M` for the rest, stage 1's zero-checks included. That is about 6 times the rest at a frame's two blocks, and about 180 times at 64 blocks, a frame of about 1,000 tokens.
  - **The verifier** stays linear, each distinct kernel and public operand evaluated once per stage.
  - **Per-block tensors.** OQ-0072-2 asks for them.
- **Opening groups.** Every anchor tensor is smaller than an advice row, so a claim on it has column sub-point `r ‖ 0^{log2 k − n_X}` (RFC-0011 rule 3). Each distinct claim point is therefore an RFC-0011 group of its own.
  - **The count.** About 8 to 14 per SHA-256 circuit shape, and 7 to 12 for AES with its multiplicities' table-tree points. A graph with the key commitment, two frames' hashes and AES holds about 20 to 45, depending on the frames' block and message counts, counted from the schedules under these rules. The high-coordinate extension of recomposition's outputs saves two to six of them, two per distinct message count.
  - **The cost at `k = 2^{22}`.** Each group is a combined row of about 64 MiB, so 40 groups are about 2.7 GB of proof, against 1.14 GB for 405B's own (RFC-0011's `G ≤ 16` counts Llama's). The verifier runs about 40 more coset NTTs of `2^{23}`, and `tessera-protocol/2` adds about 80 blinding rows.
  - **Rows.** RFC-0011 rule 2 also gives every committed tensor a row of its own: 368 a SHA-256 block, a word's bits and its carry. At 500 blocks that is 184,000 rows, 19% of 405B's, and about 228 MB of opened columns at `τ_A = 155`.
  - **Not small.** RFC-0015's estimate counts committed elements, not groups or rows. OQ-0072-3 asks how to bring the anchor's claims to fewer points and pack its tensors.
- **Deeper schedules.** A batch that carries a frame's SHA-256 has 14 stages, against the Llama templates' 3 or 4, both counting the leaf stage. Every stage has its own rounds and messages.

## Migration / Rollout

1. **This RFC.**
2. **#526, in slices:**
   - graphs without layers, public tensors, `ConstAdd`, a per-channel `ConstMul`, the non-zero-at-padding analysis, and the summing-relation, committed-output and layer-template refusals, in the schedule and the stage prover. The commitment report's stage count follows 'Messages, rejections and stages', tested equal to the schedule's on graphs of these operations. Their padded positions are the zero padding the witness and stage tables already use: no graph this slice proves is non-zero at padding. The batch refusal of 'Real positions' is tested once #551's lowering builds a graph of two requests;
   - the bit zero-checks: gadget selection, stage 1's instances and their alternative witnesses. The leaf stage gives `BitDecompose` and `AssertEqual` no site, and derives token axes only in token groups;
   - the selection relation, combined selections, `BitFn` and the SHA-256 block, whose stage count the report then matches;
   - AES's lookups and its block, with the witness and stage tables over padded shapes: a padded block's `AesSbox(0)` and its consequences, as the leaf stage's multiplicities already count them.
3. **#146** puts the circuits into a batch's graph and statement, and **#150** and **#159** mirror the relations in the verifier.

## Testing Strategy

- **Combined selections.** The SHA-256 schedule's instances, evaluations and split values per stage equal an independent count's. A combined selection with one member's kernel changed is rejected at its stage's final check.
- **No claim unchecked.** Each of these is rejected:
  - AES-128-CTR's `Gf256Mul { 3 }` lookup run on a key column other than `shifted`'s, with its leaf value that column's;
  - a round-0 carry off in the SHA-256 key commitment;
  - in the Llama composed schedule, one `RsqrtRow` native's sum-of-squares value off.
- **Padded positions.** SHA-256 with `M = 3` messages and AES with `B = 3` blocks prove and verify.
- **AES's graphs.** AES-128-CTR alone is a fragment (RFC-0006). Its tests run it in a graph that range-checks its key and its plaintext with a `BitDecompose { 8, 0 }` of each, the plaintext's input padding check included at `B = 3`.
- **Relations.** Each proves and verifies, through a verifier written from this RFC, with a changed input entry rejected:
  - the selection relation for a rotation, the value levels' even and odd rows, the message's assembly, a `Slice` on axes 0 and 1, and a `len` that is not a power of two;
  - each `BitFn` on random bits;
  - `ConstMul` by a constant per channel;
  - `ConstAdd` by a scalar and by a constant per channel on a last axis of 3, virtual and committed, its shift `ĉ(r)` equal to an independent extension of `ψ·c`, with `c̃(r_c)·ψ̃(r)` in its place rejected;
  - a `Mul` with a public operand.
- **The bit zero-checks.** Each of these is rejected, `Reject(Sumcheck { stage: 1, round: n_1 })`, and accepted with exactly its check removed:
  - a wrong carry bit;
  - a carry made up by a non-Boolean bit;
  - a padded carry slot set to 1;
  - a digest one bit off;
  - a committed byte of 256.
- **The circuits.**
  - The SHA-256 key commitment, one block with committed `k_s` and `salt_k`, proves on the CPU prover and verifies.
  - So does one AES-128-CTR block under a committed key.
  - An S-box output off its table is `Reject(Lookup { table })`.
- **Calibration.** Over F65537, a witness with one non-Boolean bit passes stage 1's booleanity check at the rate `n/|F|`, within confidence bounds, with scripted challenges. There `low + high ≤ 16` (RFC-0066 Lemma 1).
- **No change elsewhere.** The Llama schedules' vectors (`spec/vectors/schedule/`), each Llama stage's evaluation count, `cargo xtask gen-vectors --check` and the verifier's stage tests are unchanged, and no Llama schedule meets the summing-relation, committed-output or layer refusals.
- **The refusals.** Each is `ERR-INTERNAL` when the schedule is derived: a `Sum` over a padded axis downstream of a `TableMap` into `AesSbox`; a `MatMulAct` whose two inputs both have one upstream, while one with a single such input is admitted; a layer template holding a `BitDecompose`, an `AssertEqual`, a `ConstAdd`, a selection, a per-channel `ConstMul`, a lookup non-zero at padding or a node whose output is public; a single checkpoint; a sum over a padded axis downstream of a committed input that nothing pins; a committed output of a `Slice` and of a `Sum`; in a graph of two requests, a `ConstAdd` and an `AssertEqual` downstream of the tokens, once #551 builds one.

## Open Questions

- **OQ-0072-1. The zero-checks' bound.** RFC-0014's union row sums the variables of every `ZeroCheck` sharing stage 1's point. The largest is a bound, since one false relation's check vanishes at `z` with probability at most its own variables over `|Fp2|`, whatever the others do. Should the calculator count it so? It would remove the zero-checks' share of the union, about 2,400 variables per block at `M = 1` as counted (1,480 of them `eq` variables). The stages' instances add about 3,400 per block.
- **OQ-0072-2. Per-block message tensors.** Should RFC-0066's builder assemble each block's bits from that block's own committed bytes? That means a `Slice` of the committed bytes per block, its `BitDecompose { 8, 0 }`, and the block's selection, `Mul` and `Xor` over `[512, M]`, with the block's words `Slice`s of it. Each chain would then run over 512 rows, and the prover's work would grow as `N`. #526's SHA-256 slice measures both before choosing.
- **OQ-0072-3. The anchor's opening groups and rows.** Should one reduction sumcheck, as RFC-0070 does for the masks, bring the anchor's claims on small committed tensors to one column sub-point? Should RFC-0011's layout pack such tensors into shared rows, rather than one row each? Either would remove most of 'Drawbacks' cost. #146, which builds the anchor's graph, measures the groups first.

## References

- [RFC-0006](RFC-0006-compiler-ir.md), [RFC-0008](RFC-0008-logup-and-tables.md), [RFC-0009](RFC-0009-gadgets.md), [RFC-0010](RFC-0010-proof-schedule.md), [RFC-0013](RFC-0013-zero-knowledge.md), [RFC-0014](RFC-0014-interaction-and-soundness.md), [RFC-0015](RFC-0015-transcript-anchor.md), [RFC-0019](RFC-0019-verifier.md), [RFC-0030](RFC-0030-masked-exponentials.md), [RFC-0035](RFC-0035-padded-rows.md), [RFC-0037](RFC-0037-stage-schedule.md), [RFC-0039](RFC-0039-statement-fields.md), [RFC-0045](RFC-0045-leaf-layout.md), [RFC-0054](RFC-0054-token-groups.md), [RFC-0066](RFC-0066-bits-and-sha256.md), [RFC-0071](RFC-0071-request-slots.md); [gadget finals](../testing/gadget-finals.md).
- [#142](https://github.com/AbdelStark/tessera/issues/142), [#143](https://github.com/AbdelStark/tessera/issues/143), [#146](https://github.com/AbdelStark/tessera/issues/146), [#526](https://github.com/AbdelStark/tessera/issues/526).
