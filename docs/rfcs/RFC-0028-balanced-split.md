# RFC-0028: A Balanced Split Point for `RequantSplit`

- Status: Accepted
- Authors: @AbdelStark
- Created: 2026-10-17
- Target milestone: v0.2
- Supersedes, on acceptance:
  - [RFC-0006](RFC-0006-compiler-ir.md#gadget-selection) 'Gadget selection': the split point `s = b + ⌈log2 n⌉ − B_max`;
  - [RFC-0009](RFC-0009-gadgets.md#requantsplit) 'RequantSplit': the split point `s = b − B_max`.
- Origin: [#303](https://github.com/AbdelStark/tessera/issues/303)

## Summary

`RequantSplit` cuts a key of `b` bits into a low part of `s` bits, checked by `Range { s }`, and a high part, looked up with its class in `Requant { b − s, r_c − s, lo, hi }`. RFC-0006 takes the least `s` that fits the high table. That table then always has `2^{B_max}` rows, while the low one has a few. This RFC takes the `s` that balances the two, inside the bounds Lemma 2 already allows. Every element still costs two commitments and two lookups. The tables shrink, and with them the multiplicities each proof commits: at 405B, from 20 to 28 · 10^9 rows to 5 to 11 · 10^9.

## Motivation

Each table costs its rows once per proof: a committed multiplicity and a fraction-tree leaf per row, however few lookups the table serves ([RFC-0008](RFC-0008-logup-and-tables.md)). The cost model prices them ([RFC-0020](RFC-0020-cost-model.md#executable-cost-model), [#304](https://github.com/AbdelStark/tessera/issues/304)). At 405B with `B_max = 32`, the commitment reports of [#88](https://github.com/AbdelStark/tessera/issues/88) held 20 to 28 · 10^9 table rows per proof. Most of them were split high tables of exactly `2^{32}` rows. That came to 76 to 104 GiB of multiplicity counters at 4 bytes each ([08 §4](../spec/08-performance-budget.md#4-memory-budgets)), and 0.15 to 0.20 of `μ(1K)`.

## Proposed Design

### The split point

For a key of signed width `b` with `n · 2^b < P`, `n` shift classes from `r_min` up, and `c = ⌈log2 n⌉` class bits, RFC-0009 Lemma 2 holds for every `s` with:

- `s ≥ b + c − B_max`, so that the high table, `n · 2^{b − s}` rows padded to `2^{b + c − s}`, fits `2^{B_max}`;
- `s ≤ B_max`, so that `Range { s }` fits;
- `s ≤ r_min − 1`, so that every class's rounding bit lies above the low part.

The two tables commit `2^s + 2^{b + c − s}` rows, padded to powers of two as the LogUp table tree is. That sum is least at `s = (b + c)/2`, and grows on either side. The split point is therefore

```text
s = clamp(⌈(b + c) / 2⌉, b + c − B_max, min(r_min − 1, B_max))
```

where `clamp(x, lo, hi) = max(lo, min(x, hi))`. The lower bound never binds: a non-empty interval needs `b + c − B_max ≤ B_max`, and then `⌈(b + c)/2⌉ ≥ b + c − B_max`. The rule states it anyway, as Lemma 2's own bound. When `b + c` is odd, `⌊·⌋` and `⌈·⌉` give the same sum. The rule takes `⌈·⌉`, which puts the extra bit in `Range { s }`: that table depends on `s` alone, so other sites share it, while the high table also depends on the shifts and the bounds. When the interval is empty, `b + c − B_max > min(r_min − 1, B_max)`, the key takes `RequantWide`, as before ([RFC-0024](RFC-0024-wide-requantization.md)). So does a key with `n · 2^b ≥ P`, whatever its interval: Lemma 2 also needs `n · 2^b < P`, for every `s` ([RFC-0009](RFC-0009-gadgets.md#requantsplit)). Of keys of at most 62 bits, that is four classes at `b = 62`.

Everything the rule reads is public: `b` from range analysis, the shifts from the claim list, `B_max` from the gadget policy. A verifier that re-derives the selection ([RFC-0007](RFC-0007-claim-list.md#verifier-validation) step 5) finds the same `s` and regenerates the same tables.

### Unchanged

- **The gadget.** Its committed values, its lookups and Lemma 2's proof are RFC-0009's for any `s` inside the bounds above.
- **The costs per element.** Two commitments and two lookups; the per-token counts of the commitment report do not change.
- **`Requant1` and `RequantWide`.** Their conditions and their tables are as before.

### Effect at 405B

The commitment reports of [`docs/perf/v0.2/`](../perf/v0.2/README.md), regenerated:

| Configuration, `B_max = 32` | Table rows, least split | Balanced | Counters | Balanced | Table term at 1K | Balanced |
|---|---|---|---|---|---|---|
| `servable`, 1 class | 22.0 · 10^9 | 5.0 · 10^9 | 82 GiB | 19 GiB | 0.16 | 0.04 |
| `servable`, 2 classes | 24.7 · 10^9 | 7.7 · 10^9 | 92 GiB | 29 GiB | 0.18 | 0.06 |
| `servable`, 4 classes | 27.9 · 10^9 | 7.7 · 10^9 | 104 GiB | 29 GiB | 0.20 | 0.06 |
| `minimal`, 1 class | 24.2 · 10^9 | 6.6 · 10^9 | 90 GiB | 25 GiB | 0.17 | 0.05 |
| `minimal`, 2 classes | 20.5 · 10^9 | 6.7 · 10^9 | 76 GiB | 25 GiB | 0.15 | 0.05 |
| `minimal`, 4 classes | 23.9 · 10^9 | 11.2 · 10^9 | 89 GiB | 42 GiB | 0.17 | 0.08 |

`minimal` with four classes keeps the most. Its down projection's residual shifts are small, and `s ≤ r_min − 1` clamps the split below the balanced point.

## Alternatives Considered

1. **Minimize the rows of the whole selection.** Tables are shared between sites with equal parameters, so the best split of one site depends on the others'. The global problem is a search over every site's split, which the verifier would have to repeat exactly. The per-site rule is a closed form any implementation computes alike.
2. **Choose the split to share tables.** Fixing `s` to a few values would let more sites share `Range { s }`, but the high tables, which differ by shift and bounds anyway, hold most of the rows.
3. **Keep the least split.** Its high table is always full. The cost is per proof, not per element, so it grows with the number of distinct tables, not with the batch.

## Drawbacks

- One more rule in the verifier's re-derivation of the selection, which a mistake there would turn into a mismatched table digest. Claim-list validation catches that as `Reject(StatementMismatch)`, not a soundness failure.
- `Range { s }` tables grow: up to `2^{⌈(b + c)/2⌉}` rows, never past `2^{B_max}`, the bound the high tables always reached before.

## Migration / Rollout

v0.2, before any claim list is released: the selection changes in place. The 405B commitment reports and the projections that read their table rows are restated.

## Testing Strategy

- The selection tests pin the split at each boundary, against the rule computed independently in the test:
  - balanced and unclamped;
  - clamped by the rounding bit;
  - where Lemma 2's two bounds meet;
  - with classes taking their bits first.
- Lemma 2's property test checks the split's two tables at RFC-0028's split points, over shifts from just above the least split to past the balanced point.
- The 405B reports' per-token counts do not change, and their multiplicities fall as above.

## Open Questions

None.

## References

- [RFC-0006](RFC-0006-compiler-ir.md), [RFC-0008](RFC-0008-logup-and-tables.md), [RFC-0009](RFC-0009-gadgets.md), [RFC-0020](RFC-0020-cost-model.md), [RFC-0024](RFC-0024-wide-requantization.md).
- [#303](https://github.com/AbdelStark/tessera/issues/303), [#304](https://github.com/AbdelStark/tessera/issues/304), [#88](https://github.com/AbdelStark/tessera/issues/88).
