# v0.2 commitment reports: Llama-3.1-405B

These are the compiler's commitment reports ([RFC-0006](../../rfcs/RFC-0006-compiler-ir.md#outputs) 'Outputs') for Llama-3.1-405B's shape in both profiles. They settle the count of [RISK-2](../../spec/00-overview.md#9-risks) and answer OQ-0006-1, the table bound `B_max`. Each file is a `tessera.commitment-report/1` report ([schema](../../../spec/schemas/commitment-report-v1.schema.json)), named `commitment-report-405b-{profile}-c{classes}-b{B_max}.json`.

## How they are made

`crates/tessera-bench/tests/commitment_report.rs` builds each model's public part and lowers it for 16 positions. It then selects the gadgets and writes the report with the cost model's projections. The test fails when a committed file differs from what the compiler gives now.

```sh
cargo test -p tessera-bench --test commitment_report                          # check the files
TESSERA_WRITE_EXAMPLES=1 cargo test -p tessera-bench --test commitment_report # rewrite them
```

What the reports assume:

- **The model.** The shape of Llama-3.1-405B: 126 layers, `d_model` 16,384, 128 heads and 8 KV heads of 128, `d_ff` 53,248, a vocabulary of 128,256. The residual's scale is `2^{−7}`, and the shifts are those of the compiler's lowering tests.
- **Classes.** Every site after a weight takes one, two or four shift classes: RoPE's `q` and `k`, `V`, the residual sites, SiLU and, in `servable`, `Up`.
  - One is RISK-2's variant with one exponent per tensor.
  - Four is the most the quantizer allows ([RFC-0005](../../rfcs/RFC-0005-integer-model.md#scale-policy)). Three classes commit what four do, since the class index takes two bits either way.
  - A site's shifts change its gadget only where a split would reach a class's rounding bit. Here, that is at the residual sites.
- **Projections.** `μ` at 1K, 4K, 16K and 128K uses the source's unit costs, memory and scenario, with five regenerations, as 08 §3 does. It includes the table side: each row's multiplicity and fraction-tree leaf, shared by the 16K-token batch ([#304](https://github.com/AbdelStark/tessera/issues/304), below).

## Counts

Per layer per token, then per token. Committed elements are lookup outputs, split limbs and row scalars; per token they also include the embedding's one-hot vector and the final norm. Table rows are per proof.

| Profile | Classes | `B_max` | Committed, per layer | Lookups, per layer | Committed, per token | Lookups, per token | `μ(1K)` | `μ(16K)` | Table rows |
|---|---|---|---|---|---|---|---|---|---|
| `servable` | 1 | 32 | 327,464 | 327,612 | 41,421,508 | 41,311,910 | 11.16 | 26.49 | 22.0 · 10^9 |
| `servable` | 2 | 32 | 360,232 | 360,380 | 45,550,276 | 45,440,678 | 11.66 | 26.95 | 24.7 · 10^9 |
| `servable` | 4 | 32 | 376,616 | 376,764 | 47,614,660 | 47,505,062 | 11.93 | 27.20 | 27.9 · 10^9 |
| `servable` | 1 | 30 | 343,848 | 343,996 | 43,485,892 | 43,376,294 | 11.30 | 26.62 | 8.1 · 10^9 |
| `servable` | 2 | 30 | 376,616 | 376,764 | 47,614,660 | 47,505,062 | 11.79 | 27.07 | 8.6 · 10^9 |
| `minimal` | 1 | 32 | 253,736 | 270,268 | 32,131,780 | 34,086,566 | 10.22 | 25.62 | 24.2 · 10^9 |
| `minimal` | 2 | 32 | 286,504 | 303,036 | 36,260,548 | 38,215,334 | 10.68 | 26.04 | 20.5 · 10^9 |
| `minimal` | 4 | 32 | 286,504 | 303,036 | 36,260,548 | 38,215,334 | 10.70 | 26.07 | 23.9 · 10^9 |
| `minimal` | 1 | 30 | 286,504 | 319,420 | 36,260,548 | 40,279,718 | 10.71 | 26.07 | 5.8 · 10^9 |

`B_max = 30` refuses three configurations, each for one table:

- **`servable` with four classes**: SiLU's table of `4 · 2^{29}` rows, since a table lookup has no split.
- **`minimal` with two or four classes**: the classed O-projection update. Its 61-bit key would split past 30 bits, and `RequantWide`'s reduced table would need `t + 2 + ⌈log2 n⌉ = 31` bits.

Every configuration requantizes RISK-2's elements per layer per token: 260,096 in `servable` and the source's 137,216 in `minimal`. Each has RFC-0010's stage count, 3 and 4.

## RISK-2

At the default `B_max = 32`, the rescales commit RISK-2's figures: the committed counts above less 808 row scalars per layer, the norms' roots and the softmax rows' scalars.

- **`servable`**: 326,656 with one class and RISK-2's 359,424 with two. With four, 375,808: the down projection's update key is 31 bits plus two class bits, so it splits.
- **`minimal`**: 252,928 commitments and 269,312 lookups with one class, and RISK-2's upper bound with more: 285,696 and 302,080.

The quantizer's real class counts at O and down decide which line applies. With two classes, `μ(1K)` is 11.66 in `servable` and 10.68 in `minimal`, the table side included. 08 §3's 11.6 and 10.7 count the rescales alone.

## OQ-0006-1: the table bound

[RFC-0006](../../rfcs/RFC-0006-compiler-ir.md#open-questions) asks whether `B_max = 32` is the right default, against 30 or 34. Larger tables cost prover memory and table time per proof. Smaller ones force more splits.

**The table side.** Each proof commits one multiplicity per table row and builds the table's fraction tree over every row ([RFC-0008](../../rfcs/RFC-0008-logup-and-tables.md)).

- The cost model prices it as one committed element and one tree leaf per row, `(1 + glue) · M · (c_c + c_ℓ)`, shared by the 16K-token batch ([#304](https://github.com/AbdelStark/tessera/issues/304)). The projections above include it. It adds this much to `μ(1K)`.
- Its multiplicity counters take 4 bytes per row (08 §4).

| Configuration | Table term at 30 | at 32 | Counters at 30 | at 32 |
|---|---|---|---|---|
| `servable`, 1 class | 0.06 | 0.16 | 30 GiB | 82 GiB |
| `servable`, 2 classes | 0.06 | 0.18 | 32 GiB | 92 GiB |
| `servable`, 4 classes | refused | 0.20 | — | 104 GiB |
| `minimal`, 1 class | 0.04 | 0.17 | 21 GiB | 90 GiB |
| `minimal`, 2 classes | refused | 0.15 | — | 76 GiB |

**Against 30.** At 30, one site per layer changes gadget.

- In `servable`, the down projection's residual key of 31 bits splits: 16,384 more commitments and lookups per layer per token. `μ(1K)` rises by 0.13 or 0.14: the split costs 0.24 or 0.25, and the smaller tables save 0.10 to 0.12.
- In `minimal`, the O projection's 61-bit residual key takes `RequantWide` instead of `RequantSplit`, since its low part would exceed 30 bits. That costs 32,768 more commitments and 49,152 more lookups per layer per token. `μ(1K)` rises by 0.49: the wider gadget costs 0.62, and the smaller tables save 0.13.
- Where 30 compiles, it costs more than it saves. It also refuses three of the six configurations.

**Against 34.** RFC-0008 bounds every table at `2^{32}` rows, so 34 needs a superseding RFC.

- Under RFC-0006's selection rule it would change one gadget in these configurations: with four classes, `servable`'s down-projection update. Its 33-bit key would take `Requant1`, saving 16,384 commitments and lookups per layer per token, 0.25 of `μ(1K)`.
- In return, a table of `2^{33}` rows replaces one of `2^{32}`: about 0.03 more table term and 16 GiB more counters.
- Nothing else changes: no other key is 33 or 34 bits, and no `RequantWide` key would split. The test checks this.

**Recommendation: keep `B_max = 32`.** It is the only bound of the three that compiles every configuration within RFC-0008. Moving to 34 gains about 0.2 of `μ(1K)`, and only with three or four classes at the down projection. Revisit it if two conditions hold together: the quantizer's measured class counts at the down projection are three or four, and the v0.5 unit costs price the table side low enough. The counters weigh more than the bound. At 32, splits' high tables of exactly `2^{32}` rows hold most of the 76 to 104 GiB of counters. A balanced split point would cut them two- to fourfold at no cost per element ([#303](https://github.com/AbdelStark/tessera/issues/303)).
