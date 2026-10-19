# RFC-0047: The Advice Layout's Order, Size and Sequence Length

- Status: Accepted
- Authors: @AbdelStark
- Created: 2026-10-19
- Target milestone: v0.3
- Supersedes, on acceptance:
  - [RFC-0011](RFC-0011-advice-commitment.md#canonical-advice-layout) 'Canonical advice layout' rules 1 and 3: the order of a tensor's variables, and `ν`;
  - [RFC-0007](RFC-0007-claim-list.md#verifier-validation) 'Verifier validation' steps 2 and 6: heads grouped by a power of two, and the protocol's maximum layout;
  - [03 §4](../spec/03-data-model.md#4-statement-and-protocol-types) 'Statement and protocol types': a request's sequence length;
  - [RFC-0019](RFC-0019-verifier.md#check-order) 'Check order' step 4: a request longer than the context;
  - [RFC-0034](RFC-0034-causal-blocks.md#zero-rows) 'Zero rows': `Q'` for `Q`.
- Origin: [#106](https://github.com/AbdelStark/tessera/issues/106)

## Summary

Implementing the canonical advice layout (#106) found five inputs that no document fixed. Two implementations would derive different layouts, and the verifier recomputes the layout to locate every claim and every segment's chains:

- **Order.** RFC-0011 orders "padded dimensions most significant", but does not say how padded and unpadded axes interleave, and its rule 3 reads a claim's point in the tensor's own order.
- **`ν`**, the advice polynomial's variable count, is named and never defined.
- **The maximum layout** of RFC-0007 step 6 is defined nowhere.
- **Heads.** RFC-0034 and RFC-0037 take `log2 g` of a KV head's query heads, while RFC-0007 admits any divisor.
- **Sequence length.** No document says how a request's prompt and output give the sequence length `n`.

This RFC fixes each.

## Motivation

- **Order.** RFC-0011 Alternative 6 rejects laying tensors out row-major. Row-major, a `[16384, 53248]` MLP tensor at 405B pads in place and commits 256 rows. With the padded axis most significant it commits 208, exactly its elements over `k`.
- **Two readings of rule 1.**
  - "Padded dimensions ordered most significant" with "`r` taken in the order of rule 1" (rule 3).
  - RFC-0010's `r_i ‖ r_o ‖ ρ_ℓ`, which reads a claim in the tensor's natural order.
  - They give different `base_X` bits and different points.
- **Undefined size and length.** A layout without a defined maximum, or a sequence length without a definition, leaves `layout_digest` (RFC-0031) to each implementation's choice.

## Goals

- One layout for every claim list and batch, and one point for every claim.
- A maximum the verifier checks before it allocates anything for the layout.

## Non-Goals

- The committed values of `RsqrtRow`, which #126 fixes with its limbs (#377).
- Laying out a tensor with two padded axes so that both pad in whole rows.

## Proposed Design

### The order of a tensor's variables

A committed tensor's **layout order** lists its axes from most to least significant:
1. its padded axes, those whose length is not a power of two, in axis order;
2. its other axes, in axis order.

An attention tile's score tensor takes RFC-0034's `(i_hi, h, q, i_lo)` instead.

- **Claims.** A claim `X̃(r) = v` names `r`'s coordinates per axis, as the relations do ([RFC-0010](RFC-0010-proof-schedule.md#multilinear-extensions)). Rule 3 maps it to `Ã` with `r`'s coordinates permuted into layout order. The extension is the same multilinear polynomial, with its variables renamed, so no relation changes.
- **Padding.** The outermost padded axis's padding fills whole blocks of the inner axes' entries. Those blocks are whole rows when the inner entries number at least `k`, and rule 4 skips them. A second padded axis pads within those blocks, and its rows are committed.
- **Example.** At 405B, `[16384, 53248]` lays out as `[53248 → 65536, 16384]`: 53,248 slices of `2^{14}` entries in use, 208 rows of `2^{22}`, and 48 skipped rows of padding.

### `ν`

`R` is the number of rows the layout allocates, the last tile's end included. Then `ν = log2 k + ⌈log2 R⌉`: the row index has `⌈log2 R⌉` bits, and RFC-0014's `ϑ` for advice is `⌈log2 R⌉`.

### The maximum layout

A layout fits when it allocates at most `2^{32}` rows: a row index of at most 32 bits.
- **Column answers.** A column answer then carries at most `2^{32}` symbols, below RFC-0003's default vector limit of `2^{34}`, so every honest answer decodes.
- **The bound** holds at `max_context`, the longest sequence a claim list admits. RFC-0007 step 6 refuses a claim list whose layout at `n = max_context` exceeds it (`Layout`). Every shorter sequence lays out within it, because rows grow with `n`.
- **Scale.** The complete 405B `servable`, two-class, Q=512 layout at 16K tokens allocates 2,328,576 rows and spans a 22-bit row index (RFC-0014, RFC-0050).

### Heads

`g = n_heads / n_kv_heads` is a power of two. RFC-0037's head layouts expand K and V by `g`, and reshape them as the identity only over axes whose lengths are powers of two. RFC-0034 takes `log2 g`. RFC-0007 step 2's architecture rule gains it (`Architecture`).

### A request's sequence length

A request of `prompt_len` prompt tokens and `output_len` output tokens is a sequence of `n = prompt_len + output_len` positions.
- **Every committed token is a position.** The anchor commits every token (RFC-0015), and each feeds `OneHot`.
- **Scoring.** It reads the logits at positions `prompt_len − 1` to `n − 2`, one per output token (RFC-0017). The last position's logits are computed and unused.
- **The bound.** `n ≤ max_context`, as the context window holds every token. RFC-0019 step 4 refuses a batch holding a request with `n > max_context` (`StatementMismatch { field: Batch }`). The bounds of RFC-0009 Lemma 6 and the advice layout's maximum assume it.

### RFC-0034's zero rows

Its two masked-row conditions read `Q'` for `Q`, as RFC-0046 made its layout do: `i_hi·2^w > (qb + 1)·Q' − 1`, and `qb·Q' ≥ S`.

## Alternatives Considered

1. **Row-major tensors.** RFC-0011 Alternative 6 rejected them: padding commits rows.
2. **Every padded axis outermost, in decreasing padding.** It skips more rows when two axes pad, but the order then depends on lengths beyond "padded or not", and claims permute by a rule that changes with the shape.
3. **A 64-bit row index.** It admits layouts no column answer could carry under the default limits.
4. **`n = prompt_len + output_len − 1`.** It saves one position, but the last output token would feed the anchor and not `OneHot`, and it would need its own range check.

## Drawbacks

- A tensor with two padded axes commits some padding rows. For example, the conformance models' `d_ff` of 192 at a sequence length that is not a power of two.
- The last position's forward pass is computed for no logit.

## Migration / Rollout

v0.3, for #106:
1. this RFC and its amendments;
2. `tessera-pcs`'s layout implements the order, `ν` and the maximum, with the tiles' derivation;
3. both validators refuse a `g` that is not a power of two (RFC-0007 step 2) and an oversized layout (step 6).

## Testing Strategy

- **Order.**
  - A claim on any tensor equals the claim on `Ã` at its permuted point, with `base_X`, for random shapes with zero, one and two padded axes.
  - At 405B's shapes the MLP tensors commit 208 rows.
- **`ν`** equals `log2 k + ⌈log2 R⌉`. It is the length of the opening's proximity point.
- **The maximum.**
  - A claim list whose layout at `max_context` allocates `2^{32} + 1` rows is refused (`Layout`).
  - `2^{32}` is accepted.
- **Heads.** `n_heads = 24, n_kv_heads = 8` is refused (`Architecture`) by both validators.
- **Length.**
  - A request's layout uses `n = prompt_len + output_len`.
  - A batch holding a request with `n = max_context + 1` is refused (`Batch`), and `n = max_context` is accepted.

## Open Questions

None.

## References

- [RFC-0003](RFC-0003-canonical-encoding.md), [RFC-0007](RFC-0007-claim-list.md), [RFC-0010](RFC-0010-proof-schedule.md), [RFC-0011](RFC-0011-advice-commitment.md), [RFC-0014](RFC-0014-interaction-and-soundness.md), [RFC-0015](RFC-0015-transcript-anchor.md), [RFC-0017](RFC-0017-scoring.md), [RFC-0031](RFC-0031-commitment-id.md), [RFC-0034](RFC-0034-causal-blocks.md), [RFC-0037](RFC-0037-stage-schedule.md), [RFC-0046](RFC-0046-advice-segments.md).
