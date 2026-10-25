# RFC-0034: Causal Blocks: the Score Tensors Without Their Masked Half

- Status: Accepted
- Authors: @AbdelStark
- Created: 2026-10-18
- Target milestone: v0.3
- Amended by: [RFC-0071](RFC-0071-request-slots.md) ('Causal layout' and 'Zero rows': in a batch, a slot's keys and a query block's own request's length)
- Supersedes, on acceptance:
  - [RFC-0011](RFC-0011-advice-commitment.md#canonical-advice-layout) 'Canonical advice layout', rules 1, 3 and 4: a causal tensor's variable order, and its masked blocks as zero rows;
  - [RFC-0006](RFC-0006-compiler-ir.md#tiling-and-the-execution-plan) 'Tiling and the execution plan': the order of `e`'s variables within an attention tile;
  - [RFC-0030](RFC-0030-masked-exponentials.md#cost) 'Cost': who pays for the masked half;
  - [RFC-0018](RFC-0018-gpu-prover-runtime.md#kernels) 'Kernels': the tree and lookup provers evaluate constant subtrees in closed form;
  - [RFC-0020](RFC-0020-cost-model.md#executable-cost-model) 'Executable cost model' and [08 §1](../spec/08-performance-budget.md#1-cost-model): the committed scores per token;
  - [00 §9](../spec/00-overview.md#9-risks) RISK-11, which this RFC resolves.
- Origin: [#321](https://github.com/AbdelStark/tessera/issues/321)

## Summary

The cost model prices attention by the valid scores, the causal half of each score tensor. The specified layout paid for the whole square: `e` was committed at the scores' shape with every row mixing valid and masked positions, and `SoftmaxRow` looks up every position and builds every tree over the padded square ([RFC-0030](RFC-0030-masked-exponentials.md)). This RFC takes the masked half out of the work and keeps the proof as it is:

- **Layout.** Within an attention tile, `e`'s key index splits into high and low bits, and the high bits index advice rows. A row then holds one block of keys against all the tile's queries. A block whose keys all follow its queries holds only masked positions, zeros that RFC-0011 rule 4 skips. Only the diagonal blocks' masked halves remain committed.
- **Constant leaves.** Every masked lookup is the tuple `(K_0, 0)`, and every masked tree leaf is 1. The tree and lookup provers evaluate each subtree whose leaves are all constant in closed form, so their work follows the valid positions. Their messages do not change.
- **Cost.** A token commits `A·(S + min(S, b))` scores and looks up and multiplies `A·S`, where `b = 512` at 405B. The model gains `b` as `score_block`, and 405B's projections rise by about 0.2 on `μ`.

## Motivation

RISK-11 ([00 §9](../spec/00-overview.md#9-risks)) found the gap while #279 was resolved: if the masked half cost what the valid half does, 405B's attention term would double, adding 1.13 to `μ(1K)` and 16.8 to `μ(16K)`, and `servable` would miss SC-2's 1K ceiling. Three things paid for the masked half:

- RFC-0011 skips only rows that hold nothing but padding, and every row of `e` held valid and masked positions.
- RFC-0030 looks up every position of the padded square, its masked ones at `Exp`'s last key.
- `SoftmaxRow`'s trees have `2^h` leaves per row, the masked ones the constant 1.

The advice layout is not implemented yet (#106), so the layout can still change.

## Proposed Design

### Causal layout

An attention tile `(layer, kv_head, query_block)` holds `e[h, q, i]` for three ranges ([RFC-0006](RFC-0006-compiler-ir.md#tiling-and-the-execution-plan)):

- the `g = n_heads / n_kv_heads` query heads of its KV head;
- the `Q' = min(Q, N)` positions of its query block, with `Q = 512` the policy's `query_block` ([RFC-0046](RFC-0046-advice-segments.md#short-query-blocks));
- the `N = 2^n` positions of the padded sequence; in a batch of several requests, the `2^b` positions of the query block's slot, with `n = b` below ([RFC-0071](RFC-0071-request-slots.md#the-plan-and-the-layout)).

Row-major, the key index takes the lowest variables ([RFC-0010](RFC-0010-proof-schedule.md#multilinear-extensions)), so each advice row of `k` entries spans its queries against every key, and no row is all masked.

The tile's layout splits the key index into `i = i_hi·2^w + i_lo`, with `w = min(n, log2 k − log2 Q' − log2 g)`, and orders the tile's variables `(i_hi, h, q, i_lo)`, most significant first.

- **Rows.** The row index within the tile is `i_hi`. Each row holds `2^w` keys against all the tile's queries and heads, `g·Q'·2^w = k` entries. When `log2 g + log2 Q' + n ≤ log2 k`, `w = n` and the tile fits in one row, as RFC-0011 rule 2 places any tensor smaller than a row.
- **405B.** `g = 16` and `w = 9`: a row holds the block's 512 queries against 512 keys, for all 16 heads.
- **8B.** `g = 4` and `w = 11`: 512 queries against 2,048 keys.
- **Claims.** A claim `ẽ(r_h, r_q, r_i)` names its indices as before. RFC-0011 rule 3 maps it to `Ã` with the tile's coordinates in layout order: low variables first, the column sub-point `(r_{i,lo} ‖ r_q ‖ r_h)`, then `r_{i,hi}` and the tile's base bits. Relations name indices, not variable positions, so none changes.

### Zero rows

Row `i_hi` of the tile of query block `qb` holds only masked positions in either of two cases:

- every key of the row follows every query of the block: `i_hi·2^w > (qb + 1)·Q' − 1`;
- its keys or its queries are all padding: `i_hi·2^w ≥ S` or `qb·Q' ≥ S`, for a sequence of `S` positions. In a batch of several requests, `S` is the length `n_r` of the query block's request, which holds the whole block, and `qb` counts within its slot ([RFC-0071](RFC-0071-request-slots.md#the-public-selectors)).

RFC-0030 pins `e` to 0 at every masked position, so such a row holds the honest values, zeros. RFC-0011 rule 4 skips it like padding: it is not encoded, hashed or opened, and the verifier computes it from the public sequence lengths. A cheating prover gains nothing. A skipped row is zero by definition, and the masked positions of committed rows are still pinned by their lookups (RFC-0030 relation 1).

### Constant leaves

The leaves at masked positions are public constants:

- `(1, α − K_0)` in the `Exp` instance's lookup tree, the tuple `(K_0, 0)` compressed;
- 1 in `SoftmaxRow`'s trees.

The lookups and the trees list positions in the tensor's index order, the key index lowest ([RFC-0030](RFC-0030-masked-exponentials.md#relation-1-every-position-looked-up)). For a valid row `q`, the masked positions `i > q` are therefore a suffix of its leaves: a union of at most `n` dyadic subtrees. Every leaf of a padded row is masked.

A subtree of `2^t` equal leaves has nodes the prover computes in closed form. For lookup leaves `(1, d)` the subtree's root is `(2^t·d^{2^t − 1}, d^{2^t})`, by RFC-0008's combination, and for product leaves of 1 it is 1. In each level's sumcheck, a node inside a constant subtree contributes a known product to every node sum. The `eq` weights over a subcube sum in closed form, so the prover adds the subtree's contribution without visiting its nodes ([RFC-0018](RFC-0018-gpu-prover-runtime.md#kernels) 'Kernels'). Its work follows the valid positions, plus `O(n)` per row.

Every message is unchanged, byte for byte: a closed form is only another way to compute the same sums. In `tessera-protocol/2`, the masks of [RFC-0032](RFC-0032-zero-knowledge-tree-levels.md) enter each level separately from the leaves, and are computed as before.

### Cost

Per token at context `S`, with `A = n_heads · n_layers / 2`:

- **Committed scores.** `A·(S + min(S, b))`, with `b = max(Q, 2^w)`: the valid half plus the diagonal blocks' masked halves.
  - For `S ≥ b` the excess is `A·b` per token, whatever the context. A tile of query block `qb` commits `⌊((qb + 1)·Q − 1)/2^w⌋ + 1` rows of `2^w` keys, which sum to `S²/2 + b·S/2` per layer and head over the sequence.
  - For `S < b` no row is all masked, and the whole square is committed: `2A·S`.
- **Lookups and leaves.** `A·S` each, as the model counts them: the masked ones are constant subtrees.
- **Openings.** The column answers and the row combination take every committed element: `C + A·(S + min(S, b))`.
- **405B.** `b = 512`: 4.1×10^6 committed elements per token beyond the valid half, 11% of `minimal`'s `C`. The executable model gives:
  - +0.21 on `μ(1K)` and +0.19 on `μ(16K)`;
  - `μ(1K) ≈ 10.8` and `μ(16K) ≈ 26.1` for `minimal`;
  - `μ(1K) ≈ 11.7` and `μ(16K) ≈ 27.0` for `servable` ([08 §3](../spec/08-performance-budget.md#3-end-to-end-targets-llama-31-405b-one-8h100-node)).
- **Proof size.** No change: `e`'s claims already formed claim groups of their own.

### Changes to other documents

- **RFC-0011 'Canonical advice layout'.** Rule 1 orders a causal tensor's variables as above. Rule 3 maps its claims through that order. Rule 4 skips its masked blocks.
- **RFC-0006 'Tiling and the execution plan'.** `e`'s order within an attention tile.
- **RFC-0030 'Cost'.** The layout and the constant subtrees settle who pays for the masked half.
- **RFC-0018 'Kernels'.** The tree and lookup provers' constant subtrees, and its testing strategy.
- **RFC-0020 and 08 §1.** `Counts::score_block` (`b`) and the committed scores. 08 §3, SC-2, RISK-9, RISK-10 and `SPEC.md` restate the projections.
- **00 §9.** RISK-11 is resolved.

## Alternatives Considered

1. **All heads in a row.** Tiles of `(layer, query_block)` with blocks of 256 positions would put all 128 heads in a row and halve `b` to 256, saving about 0.1 on `μ`. Rejected for now: the KV head's tile is the unit the witness engine computes (RFC-0006), and the gain is small. A later layout can take it, since nothing in the protocol depends on `b`.
2. **Lookups and leaves at valid positions only.** The instances would cover the causal half alone. But that index set is not a product of ranges, so the leaf claims would need a reduction over an enumeration of blocks, and the masked positions of committed rows would need another pin. Rejected: constant subtrees reach the same cost without changing a message.
3. **Pay for the masked half.** The status quo, priced: attention doubles, +1.13 on `μ(1K)` and +16.8 on `μ(16K)` (RISK-11). Rejected.

## Drawbacks

- The diagonal blocks' masked halves are still committed: `A·b` per token, about 0.2 on `μ` at 405B.
- The tree and lookup provers need a constant-subtree path beside the dense one, and tests that hold the two to the same messages.

## Migration / Rollout

v0.3, with the advice layout (#106), `SoftmaxRow` (#122) and the CPU tree provers. Nothing is implemented yet that the layout would have to migrate.

## Testing Strategy

- **Layout.** On a tiny model with several query blocks, the committed rows of `e` are exactly the blocks that hold a valid position. A claim on `e` opens through the permuted coordinates.
- **Zero rows.** A commitment that holds a row the layout skips is rejected by the verifier's row count. A sequence whose length is not a multiple of `Q` or of `2^w` commits its partly padded blocks and skips the rest.
- **Constant subtrees.** The lookup and tree provers' messages equal the dense provers' byte for byte, for sequence lengths that are and are not powers of two. A wrong closed form fails the comparison.
- **Cost.** A count of the layout's rows, row by row, matches the closed form `A·(S + min(S, b))` that the commitment report and the model use.

## Open Questions

None.

## References

- [RFC-0006](RFC-0006-compiler-ir.md), [RFC-0008](RFC-0008-logup-and-tables.md), [RFC-0010](RFC-0010-proof-schedule.md), [RFC-0011](RFC-0011-advice-commitment.md), [RFC-0018](RFC-0018-gpu-prover-runtime.md), [RFC-0020](RFC-0020-cost-model.md), [RFC-0030](RFC-0030-masked-exponentials.md), [RFC-0032](RFC-0032-zero-knowledge-tree-levels.md).
- Tri Dao, "FlashAttention-2: Faster Attention with Better Parallelism and Work Partitioning" (2023), whose causal kernels skip the blocks above the diagonal on the native path.
