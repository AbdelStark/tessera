# RFC-0046: Advice Segments and Short Query Blocks

- Status: Accepted
- Authors: @AbdelStark
- Created: 2026-10-18
- Target milestone: v0.3
- Supersedes, on acceptance:
  - [RFC-0011](RFC-0011-advice-commitment.md#canonical-advice-layout) 'Canonical advice layout' rule 5: which tiles and rows each segment holds;
  - [RFC-0034](RFC-0034-causal-blocks.md#causal-layout) 'Causal layout': an attention tile's query count;
  - [RFC-0007](RFC-0007-claim-list.md#verifier-validation) 'Verifier validation' step 6: a layout whose attention rows cannot hold a key.
- Origin: [#106](https://github.com/AbdelStark/tessera/issues/106)

## Summary

The advice layout must be a function of public data, because the verifier recomputes every segment's column chains from a queried column (RFC-0011 'Column digests and root'). Two of its inputs were not exact:
- **Segments.** RFC-0011 partitions the rows into `S` segments "aligned to layer boundaries", and RFC-0018 gives each GPU a contiguous range of layers. Neither says which layers each segment holds.
- **Query blocks.** RFC-0034 sizes an attention tile with `Q = 512` queries. RFC-0041's tiles hold `Q' = min(Q, N)`. For a sequence shorter than `Q` the two give different layouts.

This RFC fixes both. It also refuses a layout whose rows cannot hold one key against an attention tile's queries.

## Motivation

The verifier checks a queried column by recomputing each segment's chain over that column's rows ([RFC-0004](RFC-0004-hashing-and-transcript.md#column-digests)). If two implementations split the rows differently, they hash different chains, and the honest prover's proof fails against a correct verifier. RFC-0041's `Q'` and RFC-0034's `Q` disagree on every sequence of fewer than 512 positions, which includes every conformance model.

## Goals

- One segment assignment for every row, from the claim list and the batch alone.
- One attention tile shape, as RFC-0041 defines it.

## Non-Goals

- How a prover maps segments to devices beyond RFC-0018's one segment per GPU.

## Proposed Design

### Segments

`S` is the claim list's `AdviceLayoutParams.segments`, at least 1, and `L` its `n_layers`. The first `L mod S` segments hold `⌈L / S⌉` layers each and the others `⌊L / S⌋`, in layer order:

```text
segment j holds layers  j·⌊L/S⌋ + min(j, L mod S)  to  (j+1)·⌊L/S⌋ + min(j+1, L mod S) − 1
```

- **A layer's tiles.** Layer `ℓ`'s tiles are its nodes' tiles ([RFC-0037](RFC-0037-stage-schedule.md#layers-and-kinds) 'Layers and kinds' defines its nodes). They are contiguous in production order ([RFC-0041](RFC-0041-execution-plan.md#production-order)).
- **The rest.**
  - Segment 0 also holds every tile before the first layer's: the tokens and the prologue.
  - Segment `S − 1` also holds every tile after the last layer's: the epilogue and the tables.
  - RFC-0018 assigns the embedding and the LM head the same way.
- **Rows.** A segment's rows run from its first tile's first row up to the next segment's first row. The alignment gap before a tile therefore belongs to the segment before it. Gap rows are zero rows, which no chain hashes (RFC-0011 rule 4).
- **An empty segment.** When `S > L`, a segment can hold no tile. Its chain for column `j` hashes nothing, and its digest is `d_j[0]` (RFC-0004).
- **Example.** At 405B, `L = 126` and `S = 8` give 16, 16, 16, 16, 16, 16, 15 and 15 layers. Segment 7 holds one layer fewer than the first six, beside the LM head.

### Short query blocks

An attention tile holds the `Q' = min(Q, N)` queries of its block (RFC-0041), with `Q = GadgetPolicy.query_block`. RFC-0034's layout reads `Q'` for `Q`:

- `w = min(n, log2 k − log2 Q' − log2 g)`, and a row holds `g·Q'·2^w` entries;
- a tile of `g·Q'·N ≤ k` entries fits in one row, as RFC-0011 rule 2 places any tensor smaller than a row.

### Rows too short for a key

The layout needs `w ≥ 0`: a row holds at least one key against all of a tile's queries and heads.
- **The rule.** For every sequence the claim list admits, `g·min(Q, N) ≤ k`. Since `min(Q, N)` grows with `N`, it suffices at `N_max`, the padded `max_context`.
- **Where it is checked.** RFC-0007 step 6 refuses a claim list with `g·min(Q, N_max) > k` (`Layout`, [RFC-0039](RFC-0039-statement-fields.md#the-checks)).

## Alternatives Considered

1. **`⌊j·L/S⌋` boundaries.** Their segment sizes also differ by at most one. But they place larger segments last, beside the LM head, and when `S > L` they scatter empty segments between the layers.
2. **Segments by rows**, equal row counts regardless of layers. A layer's tiles would straddle two GPUs, against RFC-0018's partitioning.
3. **A negative `w` handled by splitting rows by query.** It adds a second tile layout for parameter sets that no production model needs.

## Drawbacks

- A model with fewer layers than segments has segments with no tiles.

## Migration / Rollout

v0.3, for #106: this RFC and its amendments. The advice layout of `tessera-pcs` implements it, with the verifier's derivation (#124).

## Testing Strategy

- **Segments.** For `L` from 1 to 130 and `S` from 1 to 16, every layer lies in exactly one segment, in order, and sizes differ by at most one. At 405B the sizes are those above.
- **Rows.** A conformance model's segments cover its rows exactly once, each boundary at a tile's first row.
- **Short blocks.** A sequence of `n < Q` lays out its attention tiles with `Q' = N`. The tile of `g·N·N ≤ k` entries takes one row.
- **Layout.** A claim list with `g·min(Q, N_max) > k` is refused with `StatementMismatch { field: Layout }`.

## Open Questions

None.

## References

- [RFC-0004](RFC-0004-hashing-and-transcript.md), [RFC-0007](RFC-0007-claim-list.md), [RFC-0011](RFC-0011-advice-commitment.md), [RFC-0018](RFC-0018-gpu-prover-runtime.md), [RFC-0034](RFC-0034-causal-blocks.md), [RFC-0037](RFC-0037-stage-schedule.md), [RFC-0039](RFC-0039-statement-fields.md), [RFC-0041](RFC-0041-execution-plan.md).
