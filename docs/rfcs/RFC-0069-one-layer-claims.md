# RFC-0069: Claims at One Layer in `tessera-protocol/2`

- Status: Accepted
- Authors: @AbdelStark
- Created: 2026-10-25
- Target milestone: v0.4
- Amends, in both protocols: [RFC-0037](RFC-0037-stage-schedule.md#layers-and-kinds) 'Layers and kinds' and 'Stages and their instances': they state the schedule's one read outside the layers with the weights of one layer, and that no relation proves a claim at one layer. Both are rules the compiler and the verifier already follow, so `tessera-protocol/1` is unchanged.
- Amends, in `tessera-protocol/2`:
  - [RFC-0013](RFC-0013-zero-knowledge.md#masked-extensions) 'Masked extensions' and 'What the verifier sees': a claim at one layer of a stacked committed kind is a claim on that layer's tensor, masked by a row of its own;
  - [RFC-0037](RFC-0037-stage-schedule.md#the-claim-pool) 'The claim pool': that row's term in the claim's functional;
  - [RFC-0068](RFC-0068-leaf-stage-zero-knowledge.md#the-layer-sumcheck) 'The layer sumcheck' step 1, 'Zero knowledge' and 'Non-Goals': an outside value at one layer.
- Origin: [#535](https://github.com/AbdelStark/tessera/issues/535)
- Amended by: [RFC-0013](RFC-0013-zero-knowledge.md#masked-extensions) ('The mask' 'Degree', in `tessera-protocol/2`: a stage counts a masked factor at 3, [#565](https://github.com/AbdelStark/tessera/issues/565))

## Summary

The epilogue reads the last checkpoint `c_L` as the last layer's output: RFC-0037 gives the claim the layer weights `[ℓ = L − 1]`, one layer alone. In `tessera-protocol/2` a stacked tensor's masked extension equals the tensor at a Boolean layer coordinate, where RFC-0013's `Z` vanishes, so the claim's value is revealed unmasked (`docs/zk/simulator.md` R8).

This RFC masks it: a claim at one layer `ℓ` of a committed kind `κ` is a claim on the layer's tensor `X_κ[ℓ]` in its own variables, which takes an `Fp2` mask row of its own, as any committed tensor does under RFC-0013. At 405B that is one mask row, two committed rows, and no opening group. `tessera-protocol/1` is unchanged.

## Motivation

- **Where the claims are.** The schedule the prover proves composes the graph's relations with the native ones (`derive_composed_schedule`, RFC-0050 'Binding the simplified input'). On the 405B servable lowering with two classes, at 16K positions, it has 33 claims at one layer, all on the last position's output, the committed kind `c_L` belongs to, at layer 125:
  - **stage 1, three, at its final point:** the final norm's sum of squares (`SumSquares`), its product with the root (`Mul`), and RFC-0050's residual padding zero-check on the final norm;
  - **stage 2, thirty, at its final point:** one `SumSquares` instance for each of the root's 30 coefficient-product relations (root width `w = 32`, `2·⌈(2·32 + 55)/8⌉ = 30`).
  - Each claim's point is `(r ‖ e_{125})`: `r`, the 28 coordinates of `c_L`'s variables, all challenges, and the Boolean layer 125. The claims of one stage share `r` and their value, so two values are revealed, at two points. Each claim goes to the pool unsplit, one committed atom.
  - The graph's relations alone (`derive_schedule`) give stage 1's first two.
- **Why it leaks.** The kind's mask, `Z(x ‖ y)·R̃_κ`, vanishes at `y = e_{125}`. So the revealed value is `X̃_κ(r ‖ e_{125})`, a linear functional of the last layer's output alone.
- **The other weights.**
  - `Layers::Each`, the leaf stage's per-layer claims, are RFC-0068's.
  - `Layers::One` arises only from the epilogue's read of `c_L`, a rule of the schedule that this RFC writes into RFC-0037 'Layers and kinds'.
  - The prologue's `c_0` is a tensor of its own (`Kind::Tensor`), with its own mask, and no layer-input claim reaches `c_L`'s rows: layer `ℓ`'s input is `c_ℓ`, for `ℓ < L`.

## Goals

- No claim of the schedule revealed at a Boolean layer coordinate without a uniform mask (R8).
- No change to the schedule, the claim pool's grouping or `tessera-protocol/1`.

## Non-Goals

- **Virtual atoms at one layer.** No relation proves a claim at one layer: the schedule refuses one with `ERR-INTERNAL`, as RFC-0037 now states. Its virtual atom would be revealed at a Boolean layer, where its kind's mask vanishes. Every claim at one layer is therefore a committed atom's.

## Proposed Design

**Notation.** A stacked committed kind `κ` has tensor `X_κ`, its layer variables `y` highest and its other variables `x` (RFC-0037). `X_κ[ℓ]` is layer `ℓ`'s tensor, `X_κ[ℓ](x) = X_κ(x ‖ e_ℓ)` on the hypercube, with `e_ℓ` the bits of `ℓ`.

### The mask

For each committed kind `κ` and layer `ℓ` of an atom the schedule gives `Layers::One(ℓ)`, the epilogue's read of `c_L`. The rule follows the schedule's atoms, never the shape of a weight: a stacked claim whose layer weights happen to vanish but at one layer is not at one layer. With `λ = 0` there is no layer coordinate: the kind's claims are at points of challenges, its own mask acts, and this RFC adds nothing.
- **The row.** `X_κ[ℓ]` takes an `Fp2` advice mask row `R_{κ,ℓ}`, two committed rows, over its column variables, the low `min(n_x, log2 k)` of `x`. It is drawn from the per-proof seed (INV-ZK-3).
- **The extension.** The claim, and every instance that reads `X_κ[ℓ]`, in any stage, RFC-0050's residual padding zero-check included, use
  ```text
  X'_{κ,ℓ}(x) = X̃_κ(x ‖ e_ℓ) + Z(x)·R̃_{κ,ℓ}(x_col),     Z(x) = Π_{i ≥ 2} x_i·(1 − x_i)
  ```
  This is RFC-0013's masked extension of a committed tensor, for the tensor in its own variables. The stacked kind's own mask `R_κ` takes no part in it.
- **On the hypercube** `Z = 0`, so every relation holds unchanged with `X'_{κ,ℓ}` in place of `X̃_κ(· ‖ e_ℓ)`.
- **Degree.** `X'_{κ,ℓ}` has degree at most 3 in each variable, as RFC-0013's masked extensions have. A stage reading it counts it at 3 in its degree (RFC-0013 'Masked extensions' 'Degree'), within RFC-0014's bound.

### The pool

A claim at one layer is a pool claim on `κ` with `w = [ℓ' = ℓ]` (RFC-0037 'The claim pool').
- **Data rows.** RFC-0037's row coefficients for the data are unchanged: `γ_claim·a·eq(row part of (r ‖ base_{κ,ℓ}), row)`.
- **The mask row.** When `λ ≥ 1`, `R_{κ,ℓ}`'s row takes `γ_claim·a·Z(r)`, with `eq(r_col, ·)` its column functional, and `R_κ`'s row takes nothing, its `Z(r ‖ e_ℓ)` being 0. When `λ = 0` the claim is an ordinary claim on the kind, and `R_κ`'s row takes `γ_claim·a·Z(r)`.
- **The group.** The claim's column sub-point is `r`'s, so it joins the group it joins today. No group is added.

### Zero knowledge

- **What is revealed.** The claim's value `X'_{κ,ℓ}(r)`, at each point `r` the schedule claims `X_κ[ℓ]` at, and the instances' round messages, which `G` masks (Lemma 4).
- **Why it is uniform.**
  - `r`'s coordinates are challenges, so `Z(r) ≠ 0` except with probability at most `2(n_x − 1)/|Fp2|`.
  - `R_{κ,ℓ}` enters no other revealed value: the stacked kind's claims are masked by `R_κ`, and `R_{κ,ℓ}` is not in its extension.
  - Claims at one point give one functional and equal values, a public relation. At 405B the 33 claims sit at two points.
  - Values at `e` distinct points are jointly uniform when their `eq(r_col, ·)` vectors are independent, which fails with probability at most `e²/|Fp2|` (Lemma 2). R10's check (#488) covers these rows like every other.
- **The combined rows** that hold `R_{κ,ℓ}`'s row are blinded by `Q_g` (Lemma 5), and its opened columns are covered by Lemma 3, as every mask row's.
- **R8** is met for claims at one layer.

### Soundness

Unchanged. The value is bound to the committed rows of layer `ℓ` and to `R_{κ,ℓ}`'s row by the opening, as any committed tensor's is (RFC-0011, RFC-0013 'Hiding openings'). A false claim fails at the opening, `Reject(Evaluation { commitment: Advice })`. No challenge is added, so the calculator is unchanged.

### The leaf stage

An outside site whose gadget reads `c_L` directly would send RFC-0068's outside value at `(π(r_s) ‖ e_{L−1})`, a claim at one layer. That value is `X'_{κ,L−1}(π(r_s))`, masked by this RFC's row. At 405B no outside site does: the final norm's site reads its sum of squares and its own auxiliaries.

### Cost at 405B

| Item | Before | With this RFC |
|---|---|---|
| Claims at one layer | 33, at two points: 3 in stage 1, 30 in stage 2 | the same |
| Mask rows | — | 1 `Fp2` row, 2 committed rows |
| Opening groups | — | none added |
| Rounds, messages, labels | — | none added |

## Alternatives Considered

1. **An epilogue copy of `c_L`.** A virtual epilogue tensor with its own mask, produced from the stacked kind by a layer-selection sumcheck. It hides the value too, but adds a relation, an instance per claim and a fold. Rejected: the layer's committed rows already form a tensor of their own, which RFC-0013 masks like any other.
2. **Summing the epilogue's relations over the layer variables**, with the public weight `eq(e_{L−1}, ·)`. The final evaluations would then sit at a random layer point, but every relation reading `c_L` would gain `λ` variables. Rejected for its reach.
3. **A mask on the stacked kind weighted to layer `ℓ`,** `Z(x)·eq(e_ℓ, y)·R̃(x_col)`. It hides the claim, but its term enters every claim on the kind, at every layer point. Rejected: the tensor's own mask is local to its claims.

## Drawbacks

- One more mask row for each kind and layer claimed at one layer: one at 405B.
- The prover holds one more masked extension for the instances that read `X_κ[ℓ]`.

## Migration / Rollout

1. **This RFC.** It amends RFC-0013, RFC-0037 and RFC-0068, and `docs/zk/simulator.md` §1, §4.3, R8 and §10.
2. **The prover** (#154): the mask row and the extension; the advice layout's mask rows (#149).
3. **The verifier** (#159): the mask row's coefficient in the opening.
4. **The distinguisher suite** (#162). As the negative control, with `R_{κ,ℓ}` set to zero, it tells apart two witnesses that differ in the last layer's output. With the row in place it does not.

## Testing Strategy

- **Soundness.** A false claim at one layer is rejected at the opening, `Reject(Evaluation { commitment: Advice })`.
- **Zero knowledge.** The negative control above, in #162.
- **The schedule.** A test counts the claims at one layer on the 405B composed schedule: 33, on the last position's output at layer 125, at two points, stage 1's and stage 2's final points.

## Open Questions

None.

## References

- [RFC-0013](RFC-0013-zero-knowledge.md), [RFC-0037](RFC-0037-stage-schedule.md), [RFC-0065](RFC-0065-masked-channel-params.md) (a mask per claim at Boolean coordinates), [RFC-0068](RFC-0068-leaf-stage-zero-knowledge.md).
- `docs/zk/simulator.md` §4.3, R8 and R10.
