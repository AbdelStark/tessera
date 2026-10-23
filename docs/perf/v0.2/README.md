# v0.2 commitment reports: Llama-3.1-405B

These are the compiler's current commitment reports ([RFC-0006](../../rfcs/RFC-0006-compiler-ir.md#outputs)) for Llama-3.1-405B's shape in both profiles. Each file is a `tessera.commitment-report/1` report ([schema](../../../spec/schemas/commitment-report-v1.schema.json)), named `commitment-report-405b-{profile}-c{classes}-b32.json`. The counts include [RFC-0050](../../rfcs/RFC-0050-gadget-advice-columns.md)'s complete root columns and lookup requests. The `μ` values are cost-model projections using the source's unit costs.

## How they are made

`crates/tessera-bench/tests/commitment_report.rs` builds each model's public part, lowers it for 16 positions, selects its gadgets and projects their counts. The test requires all six `B_max = 32` configurations to compile and all six `B_max = 30` configurations to fail at the first root. It also checks every committed report against the current compiler, and the two-class reports' stage tables against the schedule lowered at 16,384 positions.

```sh
cargo test -p tessera-bench --test commitment_report --locked
TESSERA_WRITE_EXAMPLES=1 cargo test -p tessera-bench --test commitment_report --locked
```

The reports assume:

- **Model.** 126 layers, `d_model = 16384`, 128 heads and 8 KV heads of width 128, `d_ff = 53248`, and vocabulary 128,256. The residual scale is `2^−7`; shifts are those of the compiler's lowering tests.
- **Classes.** Every site after a weight takes one, two or four shift classes. A site's shifts change its gadget where a split would reach a rounding bit. Three classes also use two class-index bits, but their actual table row counts differ from four classes.
- **Projection inputs.** Source unit costs, memory and scenario; five regenerations, and the merge's sixth, since each batch's leaf stage takes token groups ([RFC-0054](../../rfcs/RFC-0054-token-groups.md)): 256, 512, 1,024 and 8,192 at the contexts 1K, 4K, 16K and 128K, which each projection's `leaf_groups` records. Each reduction stage larger than memory adds its window's fold pass ([RFC-0055](../../rfcs/RFC-0055-windowed-stages.md)), which each projection's `reduction_stages` records. The table side is shared by a 16K-token batch ([#304](https://github.com/AbdelStark/tessera/issues/304)). Masked halves of score diagonal blocks are included, with block size 512 ([RFC-0034](../../rfcs/RFC-0034-causal-blocks.md)). Requant splits use [RFC-0028](../../rfcs/RFC-0028-balanced-split.md)'s balanced point.

## Counts

All rows use `B_max = 32`. Committed elements include lookup outputs, split limbs and row scalars; per-token totals also include the one-hot vector, the first checkpoint and the final norm. Table rows count each used lookup family once, at its actual size before padding; RoPE has no multiplicity tensor.

| Profile | Classes | Committed, per layer/token | Lookups, per layer/token | Committed, per token | Lookups, per token | `μ(1K)` | `μ(16K)` | Table rows |
|---|---|---|---|---|---|---|---|---|
| `servable` | 1 | 327,798 | 327,930 | 41,480,143 | 41,352,137 | 13.28 | 28.60 | 9.5 · 10^9 |
| `servable` | 2 | 360,566 | 360,698 | 45,608,911 | 45,480,905 | 13.78 | 29.07 | 12.2 · 10^9 |
| `servable` | 4 | 376,950 | 377,082 | 47,673,295 | 47,545,289 | 14.03 | 29.29 | 12.2 · 10^9 |
| `minimal` | 1 | 254,070 | 270,586 | 32,190,415 | 34,126,793 | 13.34 | 28.73 | 11.1 · 10^9 |
| `minimal` | 2 | 286,838 | 303,354 | 36,319,183 | 38,255,561 | 13.83 | 29.18 | 11.1 · 10^9 |
| `minimal` | 4 | 286,838 | 303,354 | 36,319,183 | 38,255,561 | 13.83 | 29.18 | 11.4 · 10^9 |

Every configuration requantizes [RISK-2](../../spec/00-overview.md#9-risks)'s elements per layer/token: 260,096 in `servable` and 137,216 in `minimal`. Stage counts remain 3 and 4. The row scalars are now exact: two roots at 187 columns each, plus 768 softmax row values, for 1,142 per layer/token. Each root has 189 lookup requests; repeated table families still count as separate reads.

## Reduction stages

Each report lists its reduction stages' sumcheck tables (`stage_tables`, [RFC-0055](../../rfcs/RFC-0055-windowed-stages.md#working-sets)). Each class is `count` tables of `2^{log2_entries}` entries, per token or fixed. They are the same at every class count.

| Profile | Stage | Entries per token | Fixed entries | At 16K tokens, 16 B per entry | Window over 0.2 TB |
|---|---|---|---|---|---|
| `minimal` | 1 | 15,548,416 | 33,587,200 | 4.08 TB | 6 rounds |
| `minimal` | 2 | 20,971,520 | 8,650,752 | 5.50 TB | 6 rounds |
| `minimal` | 3 | 32,768 | 4,194,304 | 8.7 GB | none |
| `servable` | 1 | 36,519,936 | 37,781,504 | 9.57 TB | 8 rounds |
| `servable` | 2 | 32,768 | 8,650,752 | 8.7 GB | none |

Every fixed class is a matrix product's, which no window folds. These classes cover the compiler's stage schedule; the native gadget relations the prover composes into stage 1, `SoftmaxRow`'s score instances among them, are not counted yet ([#458](https://github.com/AbdelStark/tessera/issues/458)).

A window counts each table's front padding. `servable`'s stage 1 holds 4.4 TB unpadded and 4.9 TB padded by two variables, so it takes 8 rounds where its total alone would suggest 6.

## RISK-2

At `B_max = 32`, subtracting the 1,142 row scalars leaves the same rescale commitments:

- **`servable`:** 326,656 with one class, 359,424 with two, and 375,808 with four. With four classes the down-projection update's 31-bit key and two class bits require a split.
- **`minimal`:** 252,928 commitments and 269,312 lookups with one class; 285,696 and 302,080 with two or four.

The complete reports additionally count the root bracket representation and every auxiliary table. Earlier versions of these snapshots used estimated root counts and omitted required families; their `B_max = 30` files have been removed.

## OQ-0006-1: the table bound

[RFC-0006](../../rfcs/RFC-0006-compiler-ir.md#open-questions) asks whether the default should be 30, 32 or 34.

**30 is infeasible for this quantization.** Its norm precision is `κ = 30`, with integer epsilon 2,685. The first norm root needs `Range(32)` under RFC-0050, exceeding a 30-bit table policy before the projection or SiLU choices are reached. This applies to both profiles and all class configurations above. The former 30-versus-32 cost comparison omitted that lookup.

**32 supports every configuration.** Each proof commits multiplicities and builds fraction-tree leaves for the used tables. The cost model prices the table term as `(1 + glue) · M · (c_c + c_ℓ)`, shared by the batch. Under 08 §4's four-byte-counter assumption:

| Profile, classes | Table contribution to `μ(1K)` | Counter footprint |
|---|---|---|
| `servable`, 1 | 0.07 | 35 GiB |
| `servable`, 2 | 0.09 | 45 GiB |
| `servable`, 4 | 0.09 | 45 GiB |
| `minimal`, 1 | 0.08 | 41 GiB |
| `minimal`, 2 | 0.08 | 41 GiB |
| `minimal`, 4 | 0.08 | 42 GiB |

**34 requires a superseding RFC.** RFC-0008 limits tables to `2^32` rows. A hypothetical 34-bit policy would change one requantization in these configurations: `servable`'s down-projection update with four classes would use `Requant1`, saving 16,384 commitments and lookups per layer/token. Its new table would have `2^33` rows, or 32 GiB of four-byte counters for that table alone. The test checks the changed-element count; it emits no report for this unsupported policy.

**Keep `B_max = 32`.** With this quantization it is the feasible choice among the three within RFC-0008. The reports retain the existing unit-cost and memory assumptions; complete prover timing and runtime memory measurements remain separate validation work.
