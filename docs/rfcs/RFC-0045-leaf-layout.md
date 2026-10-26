# RFC-0045: The Leaf Layout

- Status: Accepted
- Authors: @AbdelStark
- Created: 2026-10-18
- Target milestone: v0.3
- Amended by: [RFC-0052](RFC-0052-channel-params-claims.md) ('Open Questions': opening `ChannelParams` claims), [RFC-0068](RFC-0068-leaf-stage-zero-knowledge.md) (in `tessera-protocol/2`: 'The leaf point', `P` without the tree index's coordinate; 'The split', the values of sites in the layers sent once per site and tensor, after a layer sumcheck), [RFC-0072](RFC-0072-anchor-relations.md) ('The split': a public tensor's value is not sent)
- Supersedes, on acceptance:
  - [RFC-0008](RFC-0008-logup-and-tables.md#batching) 'Batching' and 'Fraction tree': how an instance lays out its lookups, and the order of its lookup tree's leaves;
  - [RFC-0037](RFC-0037-stage-schedule.md#the-layer-point) 'The layer point' and 'Splits': the point `r_s` of a site's tensors, and the leaf stage's split values.
- Origin: [#367](https://github.com/AbdelStark/tessera/issues/367)

## Summary

RFC-0008 makes all lookups into one table one LogUp instance, across every layer and site. RFC-0037 expects one claim on each gadget input per layer, at one point per site. Nothing said how an instance lays out its lookups, or how its leaf claims become claims on the gadgets' tensors. This RFC fixes both:

- **Blocks.** An instance's lookup column is a sequence of blocks, one per lookup of a site at a layer. Blocks are aligned and sorted by size.
- **Leaves.** The lookup tree holds its column with bits 1 to `h − 1` of the index reversed. Every block of `2^k` entries, in every instance, then lies at `P[0..k]`, the first `k` coordinates of one point `P`. `P` is the last tree level's challenges in reverse order.
- **The split.** The prover sends each tensor's value at its point, per site and layer, and the verifier checks each instance's leaf claims against them. The values are RFC-0037's per-layer claims.

## Motivation

The tree levels front-pad their instances ([RFC-0010](RFC-0010-proof-schedule.md#tree-levels) 'Tree levels'). At level `ℓ`, an instance on `n_i` variables takes the last `n_i` of the level's challenges, and its next claims lie at `ρ_ℓ ‖ r'_i`. With `H` the tallest tree's height and `r_0, …, r_{n−1}` the `n` round challenges of the last level, `H − 1`, a tree of height `h` therefore ends at

```text
(ρ_{H−1}, r_{n−h+1}, r_{n−h+2}, …, r_{n−1})
```

Two trees of different heights agree on their lowest coordinate and on their highest ones, not on the others. #118's probe measured it: a taller instance ends at the shorter one's point with a coordinate inserted at index 1. Laid out naturally, blocks of one size would then lie at different points in instances of different heights. That happens whenever a site's layers read different tables, as the `Silu` tables of layers with different exponents do ([RFC-0042](RFC-0042-silu-exponents.md)). A site's layers would no longer sit at one point, and RFC-0037's layer point could not stack them.

## Goals

- One point for every block of one size, whatever its instance's height.
- One claim per layer on each tensor a site's lookups read, at one point.
- No change to `tessera-logup`, the tree levels or their transcript.

## Non-Goals

- The gadgets' lookups themselves, which RFC-0009 lists.
- The claims of the product forests, which RFC-0009 and RFC-0030 route.
- `tessera-protocol/2`'s masked leaf checks ([RFC-0032](RFC-0032-zero-knowledge-tree-levels.md#leaf-checks) 'Leaf checks'). This layout is `tessera-protocol/1`'s, and [RFC-0068](RFC-0068-leaf-stage-zero-knowledge.md) states what changes in `tessera-protocol/2`.

## Proposed Design

### Blocks

A block is one lookup of one gadget site at one layer, over its domain: the tensor the lookup runs over, per-dimension padded ([RFC-0011](RFC-0011-advice-commitment.md#canonical-advice-layout) rule 1), in that tensor's index order. It has `2^k` entries, with `k` the sum of the axes' `⌈log2 len⌉`.
- **Padded positions.** A position of the padding holds the lookup of what the node computes there ([RFC-0035](RFC-0035-padded-rows.md#what-a-padded-row-holds)). A masked score is the constant tuple `(K_0, 0)` ([RFC-0030](RFC-0030-masked-exponentials.md#relation-1-every-position-looked-up)).
- **Sites.** A site is a gadget node. A template position is one site for all its layers, with one block per real layer and lookup; a padded layer has none. A prologue or epilogue site has one block per lookup.

### The column

Each instance lays out the blocks of its table:

1. **Order.**
   - The sites come in node order, then each site's layers in order, then the gadget's lookups into the table in RFC-0009's order.
   - A stable sort then puts larger blocks first.
2. **Offsets.** Each block starts where the previous one ends. The sizes are powers of two and never increase, so a block of `2^k` entries starts at a multiple of `2^k`, its offset `o_b`.
3. **Height.** With `m` the total of the blocks' entries, the column has `2^h` entries, `h = ⌈log2 m⌉`. Entries `m` to `2^h − 1` hold row 0's tuple, which the multiplicity of row 0 counts ([RFC-0008](RFC-0008-logup-and-tables.md#fraction-tree)).

The table column and the multiplicities stay one per instance, in the table's row order.

### The leaves

The lookup tree's leaf `i` holds column entry `σ_h(i)`, where `σ_h` keeps bit 0 and reverses bits 1 to `h − 1`:

```text
σ_h(i) = (i mod 2) + 2·rev_{h−1}(⌊i / 2⌋)
```

- **Height 0.** `σ_0` is the identity: the column has one entry, leaf 0.
- **Inverse.** `σ_h` is its own inverse.
- **Padding.** The column's padding entries are explicit lookups. `tessera-logup` receives `2^h` lookups and pads none.
- **The table tree** keeps its rows in order.

### The leaf point

`P = (ρ_{H−1}, r_{n−1}, r_{n−2}, …, r_0)` is the last tree level's challenges in reverse order of squeezing: its line challenge, then its `n` round challenges from the last to the first. `H` is the tallest tree's height, product forests included, and `n` is the level's round count, which a forest of several trees can make larger than `H − 1`. `P` is empty when `H = 0`.

- **An instance's column** of height `h` has its leaf claims at `P[0..h]`. These are the claims of RFC-0008 'Fraction tree' on its key and value columns, read in column order.
  - The tree's leaf point is `(ρ_{H−1}, r_{n−h+1}, …, r_{n−1})`: leaf bit `j ≥ 1` takes `r_{n−h+j}`, and holds column bit `h − j`, which `P` gives `r_{n−(h−j)}`.
  - So `Σ_i eq(leaf point, i)·leaf_i = Σ_q eq(P[0..h], q)·column_q`.
- **A block** of `2^k` entries has its claims at `P[0..k]`, in every instance:

  ```text
  column~(P[0..h]) = Σ_b eq(P[k_b..h], o_b / 2^{k_b})·block~_b(P[0..k_b]) + (1 − LT(P[0..h], m))·t_0
  ```

  `t_0` is row 0's key or value, and `LT(r, m) = Σ_{x<m} eq(r, x)` over `h` bits, which the verifier computes in `O(h)`.
- **A site's point.** A site's blocks of `2^k` entries lie at `r_s = P[0..k]`, at every layer: RFC-0037's `r_s`.
  - A tensor the site's lookups read over all of a block's axes lies at `r_s`.
  - A tensor read over some of them lies at those axes' coordinates of `r_s`, `π_t(r_s)` ('The split').
  - Blocks of one size share their point whatever their table, so a site's lookups into several tables read their tensors at one point.
- **The multiplicities** keep their claim at the table tree's leaf point, in row order. They are committed, and their claim goes to the pool.

### The split

**A block's claim.**
- A lookup's key and value are affine forms of tensors: `Σ_t a_t·X_t + a_0` (RFC-0009).
- Each `X_t` is read over the block's axes or over some of them. A channel's class, for example, is read over the channel axis alone.
- At the block's point the form is `Σ_t a_t·X~_t(π_t(P[0..k])) + a_0`, with `π_t` the coordinates of `X_t`'s axes.

**What is sent.**
- The values go under `stage/0/split`, after `logup/values`, followed in the same `Vec<Fp2>` (`TreeProof.splits`) by the atom split values of any split leaf claim ([RFC-0037](RFC-0037-stage-schedule.md#splits)).
- For each site in node order, and each of its real layers in order (once for a site outside the layers), the prover sends one value per tensor its lookups read. It sends none for a public tensor, which the verifier evaluates at the block's point ([RFC-0072](RFC-0072-anchor-relations.md#public-tensors)).
- Each value is the tensor's evaluation at its point.
- The tensors of a site at a layer come in this order:
  1. lookup by lookup, in RFC-0009's order;
  2. within a lookup, the key's tensors, then the value's;
  3. within a form, in the order RFC-0009's formula names them, so the input before the class for `pack(c, z) = (z + 2^{b−1}) + 2^b·c`;
  4. each tensor once per site and layer, at its first read.
- Public tensors are not sent: constants, the causal selector and the row selector are the verifier's.

**The check.**
- Table order is ascending `TableId`. The instances take it in `logup/roots`, in `logup/values` and in this check, and their trees take it in the levels' schedule, each instance's lookup tree before its table tree.
- For each instance in table order, the verifier evaluates its blocks' key forms from the sent values.
- It checks that they give the key claim by the identity above. It checks the value claim the same way when the table has values.
- A mismatch is `Reject(Sumcheck { stage: 0, round: 0 })`.

**The result.** Each value is a per-layer claim of RFC-0037 'The layer point'.
- Stage 1 stacks a site's tensor over its layers under `ρ_1`.
- The stacked claim of a virtual tensor is a stage-1 instance.
- The stacked claim of a committed tensor goes to the pool.

**One claim per input.**
- Every tensor a site's lookups read is read over one set of axes, so it has one point and one value per layer.
- The schedule refuses, with `ERR-INTERNAL`, a gadget whose lookups read one tensor over two sets of axes.
- Every real layer of a site reads the same tensors over the same axes, so stage 1 stacks each of them over the layers. The schedule refuses, with `ERR-INTERNAL`, a site whose layers' gadgets read different ones.

**Soundness.**
- The values are absorbed before `ρ_1` and every later challenge.
- The identity is exact for honest values. A false leaf claim therefore forces a false value, and so does a true leaf claim with any false value sent beside it.
- Each value's own check rejects a false one, except with the probability that check allows (RFC-0037 'Splits').

### The verifier's derivation

The verifier derives the sites, the blocks, their offsets and the instances' heights from the claim list. It uses its own lowering (RFC-0044) and gadget selection (RFC-0006), and nothing the prover sends. It reads `P` from the transcript.

## Alternatives Considered

1. **Back-padded tree levels.** Shorter instances would take the first challenges of each level, and a natural layout would put every block of `2^k` entries at one point. But it changes RFC-0010's batching, which both workspaces implement for trees and stages alike.
2. **Pairing on the highest bit.** Pairing entries `x` and `x + 2^ℓ` aligns the trees' high coordinates, with a strided layout. It changes the tree levels of both workspaces too.
3. **One instance per table and block size.** Equal heights would still not follow, and instances would multiply.
4. **Padding every instance to the tallest height.** It would align every point at the cost of the small tables' trees, up to `2^H` leaves each.
5. **A sumcheck per instance over its column.** A stage-1 relation would reduce each leaf claim to its blocks. Front-padded stages would misalign the blocks again, and it costs a stage the split avoids.

## Drawbacks

- The prover writes each lookup tree's leaves in the permuted order: one index computation per lookup.
- The leaf stage sends one value per tensor, site and layer. At 405B that is 53,489 `Fp2` elements, 856 KB, against the openings' gigabytes. In `tessera-protocol/2`, RFC-0068 sends 614.

## Migration / Rollout

v0.3, for #367:
1. this RFC and its amendments;
2. the leaf stage's layout and split, with a test of a conformance model through `tessera-logup` (#367, after #118);
3. the prover's leaf integration (#130) and the verifier's gadget checks (#124).

## Testing Strategy

- **One point per size.** Two instances of different heights, each holding a block of `2^k` entries, give both blocks the point `P[0..k]`.
- **End to end.**
  - A conformance model's leaf stage, through `tessera-logup` and the split, is accepted by an in-test verifier written from this RFC.
  - A tampered split value is `Reject(Sumcheck { stage: 0, round: 0 })`.
  - A tampered lookup in a block fails LogUp's root check.
- **One claim per input.** Every tensor a gadget's lookups read has exactly one value per layer.
- **The layout.** Offsets are aligned and the padding is exactly the column's tail. Changing the order of two blocks of one size changes the split's check.

## Open Questions

- **Classes.** Resolved by [RFC-0052](RFC-0052-channel-params-claims.md): each site that reads channel classes or affine multipliers leaves one claim on `ChannelParams` in the pool, and `ChannelParams` opens all of its claims in one pass over its committed rows.

## References

- [RFC-0008](RFC-0008-logup-and-tables.md), [RFC-0009](RFC-0009-gadgets.md), [RFC-0010](RFC-0010-proof-schedule.md), [RFC-0011](RFC-0011-advice-commitment.md), [RFC-0030](RFC-0030-masked-exponentials.md), [RFC-0035](RFC-0035-padded-rows.md), [RFC-0037](RFC-0037-stage-schedule.md), [RFC-0042](RFC-0042-silu-exponents.md), [RFC-0044](RFC-0044-lowering-nodes.md).
