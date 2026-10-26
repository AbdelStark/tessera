# RFC-0072: Proving the Anchor Circuits: Graphs Without Layers, Public Tensors, Selections and the Bit Zero-Checks

- Status: Accepted
- Authors: @AbdelStark
- Created: 2026-10-26
- Target milestone: v0.4
- Amends:
  - [RFC-0066](RFC-0066-bits-and-sha256.md) '`BitDecompose`' (recomposition's relation and its claim on `x`), '`AssertEqual`' (a public input pads with zeros), 'Where the zero-checks run' (their messages), 'Drawbacks' (stacking), and its open questions OQ-0066-1 and OQ-0066-2, which this RFC closes;
  - [RFC-0010](RFC-0010-proof-schedule.md#relations) 'Relations' (recomposition's row, which sums its bit axis, the selection relation, `ConstMul` by a constant per channel), 'Stages' (termination at public tensors) and 'Batching within a stage' (a recomposition's zero-check point);
  - [RFC-0037](RFC-0037-stage-schedule.md#layers-and-kinds) 'Layers and kinds' (a graph with no checkpoint), 'The committed first checkpoint' (no committed selection output), 'Claims and atoms' (public tensors, `ConstAdd`'s shift, `Slice`, `GatherPublic` and a per-channel `ConstMul` as relations, combined selections, one point), 'Splits' (a combined selection's value), 'Stages and their instances' (stage 1's new zero-checks, the stage count) and 'The layer point' (no site for `BitDecompose` and `AssertEqual`, and the logits when there are some);
  - [RFC-0035](RFC-0035-padded-rows.md#zero-check-points) 'Zero-check points': a recomposition's `eq` takes the highest of its instance's coordinates;
  - [RFC-0045](RFC-0045-leaf-layout.md#the-split) 'Sites' (none for `BitDecompose` and `AssertEqual`) and 'The split' (a public tensor's value is not sent);
  - [RFC-0006](RFC-0006-compiler-ir.md#gadget-selection) 'Gadget selection': `BitDecompose` and `AssertEqual` take gadgets;
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
- **Public tensors.** A tensor computed from public tensors alone is public: the verifier evaluates its claims itself.
- **The selection relation.** `GatherPublic` and `Slice` reduce a claim through one sumcheck over the selected axis, against a public 0/1 kernel, of degree 2 (OQ-0066-2).
- **`ConstMul` by a constant per channel** is RFC-0010's `Elementwise` with the constant a public operand.
- **Combined selections.** Within one claim, selections of one tensor along one axis at one point are one selection, whose kernel weighs theirs by their coefficients. Without that, the two selections that read each level of a SHA-256 sum's value, its even and odd rows, would double the instances below it at every level. Claims are never merged across a stage, so no claim goes unchecked and no Llama schedule changes.
- **Padded positions** hold what their operation computes from padded inputs, for virtual and public tensors too.
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

- **`tessera-protocol/2`.** The circuits' committed bits are advice, masked as any other tensor. Their virtual tensors take RFC-0013's masks and folds, in #145's argument. The selection relation leaves its claims at random points, so it adds no Boolean coordinate that R8 forbids.
- **The anchor's wiring.** #146 puts the circuits into a batch's graph, its inputs taken from the log and the session, and its statement's sizes; #163 tests it end to end.
- **The verifier's mirror** of these relations is #150's and #159's.
- **Token groups.** RFC-0054 refuses a `Bitwise` site, and the other operations its token axis lists, in a leaf stage in token groups. A batch that carries AES-128-CTR runs its leaf stage in one batch, or #146 amends RFC-0054.
- **Stacking instances of one kind** ('Alternatives Considered' 3).

## Proposed Design

### Graphs without layers

- **No checkpoint, no layers.** A lowering with no checkpoint has `L = 0` layers and `λ = 0` layer variables (RFC-0037 'Layers and kinds').
  - Every node is outside the layers, and every tensor is a kind without a layer index, like a prologue's or an epilogue's.
  - Stage 1's layer point has no coordinate, so `stage/1/layer/{c}` squeezes nothing.
- **No logits.** A graph with no logits leaves no claim on them and has no scoring site.
- **Advice segments.** With `L = 0`, every tile is in segment 0, and any further segments are empty ([RFC-0046](RFC-0046-advice-segments.md#segments) 'An empty segment').
- **In a model's graph.** #146 adds the circuits to a batch's graph after the model's nodes. Their nodes are then epilogue nodes, outside the layers, under the same rules. They read no checkpoint.

### Public tensors

- **What is public.**
  - A graph input declared public.
  - A constant.
  - The output of a node that is not gadget-backed and whose inputs are all public, such as a `GatherPublic` of the `SHR` masks or a `Slice` of the digest.
  - A tensor declared committed is committed, whatever its inputs.
- **Their claims.** An atom on a public tensor is public (RFC-0037 'Claims and atoms'). The verifier computes the tensor from public data and evaluates its extension at the atom's point, `O(size)`, then moves the atom to the claim's value, as RFC-0037 'Splits' does. The atom reaches no instance and no pool.
- **Once per point.** The verifier evaluates a public tensor, or a selection's kernel, once per distinct point. A stage's instances of one relation and one size end at its last challenges, so their reads of one public operand share a point.
- **As an operand.** A relation's operand that is public is read as `Public` (RFC-0010 'Relations'): the verifier evaluates it at the instance's point, a broadcast operand such as an `SHR` mask at its own coordinates of that point, and the proof sends nothing for it.
- **In the leaf stage.** A lookup that reads a public tensor, such as AES's counter blocks, sends no split value for it (RFC-0045 'The split'). The verifier evaluates it at the block's point.

### Padded positions

Every tensor's padded positions hold what its operation computes from its inputs' padded positions, as RFC-0035 'What a padded row holds' says. That holds for virtual and public tensors too:
- graph inputs and constants pad with zeros, and a per-channel constant is 0 at padded channels;
- a selection's padded rows on its selected axis are 0, and its other padded positions are its input's;
- an `Expand` is 0 past `len`, as its coefficient `LT(r_a, len)` says;
- a `ConstAdd`'s padded positions are its input's plus its constant;
- a lookup's are the lookup of what its node computes there (RFC-0045), such as `AesSbox(0) = 0x63`.

So the prover's witness and stage tables, and the verifier's public tensors, are computed over the padded shapes. With `M` messages or `B` blocks not a power of two, SHA-256's padded messages run the compression on zero bits and their sums' constants. AES's padded blocks run on zero counters under zero round keys, since each round key's `Expand` is 0 past `B`, and give a public constant, not `AES_k(0)`.

A committed graph input's padded positions are computed by nothing. RFC-0006 'Committed graph inputs' admits a graph only when a `BitDecompose` or a `OneHot` reads each committed advice input, and their checks pin its padded positions: RFC-0066's input padding check, or `OneHot`'s row selector. AES-128-CTR alone is a fragment, whose key and plaintext a larger graph proves (#146).

### Linear operations that are not coordinate maps

- **`ConstAdd`** moves its constant into the claim's value: `ỹ(r) = x̃(r) + c̃(r)`. Its constant is a public atom, with the atom's coefficient `a` and layer weights `w`, so a claim `a·Σ_ℓ w(ℓ)·ỹ_ℓ(r) = v` becomes `a·Σ_ℓ w(ℓ)·x̃_ℓ(r) = v − a·Σ_ℓ w(ℓ)·c̃_ℓ(r)`.
  - A constant of one value contributes `c`.
  - A constant per channel contributes `c̃(r_c)`, its channel coordinates' extension with zeros at padded channels.
  - At padded positions the operation computes `x + c` ('Padded positions'), which is what this extension gives.
- **`ConstMul` by a constant per channel** is RFC-0010's `Elementwise`: `Ỹ(r) = Σ_x eq(r, x)·c̃(x_c)·X̃(x)`, degree 3, the constant a public operand broadcast over the other axes. It leaves one claim on `X`. A scalar `ConstMul` stays a coefficient.
- **`Slice` and `GatherPublic`** are selections, below.
- **A committed output.** A `ConstAdd` or a per-channel `ConstMul` whose output is declared committed is proven as RFC-0037's committed outputs are, by an equality zero-check in stage 1. A selection whose output is declared committed, which RFC-0006 and the IR admit, is refused when the schedule is derived (`ERR-INTERNAL`): its relation is a sum over the selected axis, not the pointwise `f` that equality zero-check reads. `BitFn`'s output is virtual (RFC-0066). Neither circuit declares one.
- **`Sum`.** RFC-0010 maps a `Sum` claim to `2^j·x̃(½, …, ½ ‖ r)`, a sum over the whole padded cube of the summed axes, which is the operation's only when its input is 0 there. Padded positions need not be 0 ('Padded positions'). From zero inputs only a `ConstAdd` and a lookup into a table that is non-zero at key 0, `Exp2Frac`, `Pow2` or `AesSbox` (RFC-0035 'What a padded row holds'), make them non-zero. Every other operation keeps them 0, the gadgets' padded rows included, which RFC-0035's checks pin. So a `Sum` over an axis whose length is not a power of two is refused when the schedule is derived (`ERR-INTERNAL`) when a `ConstAdd` or such a lookup lies upstream of its input in the graph. An `Expand` along the summed axis is the exception, being 0 past `len` whatever its input. No circuit or Llama graph has a `Sum`, and the compiler's schedule tests sum an `Expand` along its own axis.

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
- **No merging across claims.** Two claims on one tensor at one point stay two claims, each with its instance. One instance for both would prove one claim's value and leave the other's unchecked: AES's `Gf256Mul { 2 }` and `Gf256Mul { 3 }` lookups read one tensor, ShiftRows' output, at one leaf point, and SHA-256's first round reduces three claims on its first message word to one point. Llama's composed schedule holds such pairs too, so a merge would also change `tessera-protocol/1`'s messages.
- **Llama** has no selection, so this rule never applies to it.

### The bit zero-checks (OQ-0066-1)

They run in stage 1, after its other zero-checks, in node order, each node's checks in the table's order (RFC-0066 'Where the zero-checks run'). Each is one instance at the zero-check point `z` (RFC-0035 'Zero-check points'): an instance of `n_i` variables takes `z`'s last `n_i` coordinates, and its `eq` the highest of them.

| Check | Identity | Instance variables | `eq` variables | Evaluations, in order | Degree |
|---|---|---|---|---|---|
| Input padding, of a committed input a `BitDecompose` reads, when it has padded positions | `0 = Σ_x eq(z, x)·(1 − ψ̃(x))·x̃(x)` | `n_x` | `n_x` | `x̃` | 3 |
| Booleanity of `b`, of `h` | `0 = Σ_x eq(z, x)·b̃(x)·(1 − b̃(x))` | `n_y + k_b`, `n_y + k_h` | all | `b̃`, `h̃` | 3 |
| Bit padding of `b`, of `h`, when its `n` bits are not a power of two | `0 = Σ_x eq(z, x)·(1 − LT(x_bit, n))·b̃(x)` | `n_y + k_b`, `n_y + k_h` | all | `b̃`, `h̃` | 3 |
| Recomposition | below | `n_y + k` | `n_y` | `x̃` unless public, `b̃`, `h̃` | 2 |
| `AssertEqual` | `0 = Σ_x eq(z, x)·ψ̃(x)·(ã(x) − b̃(x))` | `n` | `n` | `ã`, `b̃`, each unless public; none when both are | 3 |

`n_y` is the variables of `x`'s shape. `k_b = ⌈log2 low⌉` and `k_h = ⌈log2 high⌉` are the bit axes' variables, which are the lowest.

**Recomposition.** It is one zero-check over `y`, summed over the bit axis, `k = max(k_b, k_h)` variables:

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
- **Per graph.** The binding adds 8 `AssertEqual`s. Each `BitDecompose { 8, 0 }` of committed bytes adds its booleanity, its recomposition and, when its input has padded positions, its padding check.
- **Chain length.**
  - A public message's chain is a word's `SHR` (a selection, then a `Mul`), its `σ`'s `Xor3`, and the sum's five levels of selections. That is 8 relations below the zero-check, 9 with it, so 10 stages.
  - Committed bytes add a message word's `Slice`, `Xor`, `Mul` and selection: 12 relations below the zero-check, so 14 stages.
  - Each block's carries and words are committed, so the chain does not grow with the block count (INV-SCH-4).
- **AES.** Its lookups' keys reach committed or public tensors through at most two relations, a selection and a `ConstMul` or MixColumns' two selections, so 3 stages.

### Soundness

- **The new relations** are sumcheck identities of RFC-0010's form, with public factors the verifier computes: the selection kernel, the weights `W`, the zero-extensions and `ψ`. Each instance counts in its stage's `m` and its rounds in `n·d`, as the calculator counts every instance (RFC-0014).
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
  - **The count.** About 8 to 14 per SHA-256 circuit shape, and 7 to 12 for AES with its multiplicities' table-tree points. A graph with the key commitment, two frames' hashes and AES holds about 20 to 45, depending on the frames' block and message counts, counted from the schedules under these rules. The high-coordinate extension of recomposition's outputs saves four to six of them.
  - **The cost at `k = 2^{22}`.** Each group is a combined row of about 64 MiB, so 40 groups are about 2.7 GB of proof, against 1.14 GB for 405B's own (RFC-0011's `G ≤ 16` counts Llama's). The verifier runs about 40 more coset NTTs of `2^{23}`, and `tessera-protocol/2` adds about 80 blinding rows.
  - **Rows.** RFC-0011 rule 2 also gives every committed tensor a row of its own: 368 a SHA-256 block, a word's bits and its carry. At 500 blocks that is 184,000 rows, 19% of 405B's, and about 228 MB of opened columns at `τ_A = 155`.
  - **Not small.** RFC-0015's estimate counts committed elements, not groups or rows. OQ-0072-3 asks how to bring the anchor's claims to fewer points and pack its tensors.
- **Deeper schedules.** A batch that carries a frame's SHA-256 has 14 stages, against the Llama templates' 3 or 4, both counting the leaf stage. Every stage has its own rounds and messages.

## Migration / Rollout

1. **This RFC.**
2. **#526, in slices:**
   - graphs without layers, public tensors, `ConstAdd`, combined selections, and the witness and stage tables over padded shapes, in the schedule and the stage prover. The commitment report's stage count follows 'Messages, rejections and stages', tested equal to the schedule's on the circuits;
   - the bit zero-checks: gadget selection, stage 1's instances and their alternative witnesses;
   - the selection relation, `BitFn` and the SHA-256 block;
   - AES's lookups and its block.
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
- **No change elsewhere.** The Llama schedules' vectors (`spec/vectors/schedule/`), each Llama stage's evaluation count, `cargo xtask gen-vectors --check` and the verifier's stage tests are unchanged.

## Open Questions

- **OQ-0072-1. The zero-checks' bound.** RFC-0014's union row sums the variables of every `ZeroCheck` sharing stage 1's point. The largest is a bound, since one false relation's check vanishes at `z` with probability at most its own variables over `|Fp2|`, whatever the others do. Should the calculator count it so? It would remove the zero-checks' share of the union, about 2,400 variables per block at `M = 1` as counted (1,480 of them `eq` variables). The stages' instances add about 3,400 per block.
- **OQ-0072-2. Per-block message tensors.** Should RFC-0066's builder assemble each block's bits from that block's own committed bytes? That means a `Slice` of the committed bytes per block, its `BitDecompose { 8, 0 }`, and the block's selection, `Mul` and `Xor` over `[512, M]`, with the block's words `Slice`s of it. Each chain would then run over 512 rows, and the prover's work would grow as `N`. #526's SHA-256 slice measures both before choosing.
- **OQ-0072-3. The anchor's opening groups and rows.** Should one reduction sumcheck, as RFC-0070 does for the masks, bring the anchor's claims on small committed tensors to one column sub-point? Should RFC-0011's layout pack such tensors into shared rows, rather than one row each? Either would remove most of 'Drawbacks' cost. #146, which builds the anchor's graph, measures the groups first.

## References

- [RFC-0006](RFC-0006-compiler-ir.md), [RFC-0008](RFC-0008-logup-and-tables.md), [RFC-0009](RFC-0009-gadgets.md), [RFC-0010](RFC-0010-proof-schedule.md), [RFC-0013](RFC-0013-zero-knowledge.md), [RFC-0014](RFC-0014-interaction-and-soundness.md), [RFC-0015](RFC-0015-transcript-anchor.md), [RFC-0019](RFC-0019-verifier.md), [RFC-0035](RFC-0035-padded-rows.md), [RFC-0037](RFC-0037-stage-schedule.md), [RFC-0039](RFC-0039-statement-fields.md), [RFC-0045](RFC-0045-leaf-layout.md), [RFC-0054](RFC-0054-token-groups.md), [RFC-0066](RFC-0066-bits-and-sha256.md); [gadget finals](../testing/gadget-finals.md).
- [#142](https://github.com/AbdelStark/tessera/issues/142), [#143](https://github.com/AbdelStark/tessera/issues/143), [#146](https://github.com/AbdelStark/tessera/issues/146), [#526](https://github.com/AbdelStark/tessera/issues/526).
