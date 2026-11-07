# EXP-0036: How the epoch rank loss's span term scales with the field

| | |
|---|---|
| Status | `planned` |
| Idea | [IDEA-0044](../../ideas/IDEA-0044-tensor-generator-epoch-rank.md) |
| Evidence level | E0 (exact arithmetic over small prime fields; a measured rate, not a proof) |
| Hardware | Apple M4 Max CPU, macOS; no GPU needed |
| Pre-registered on | 2026-11-07 |

## Hypothesis

At `N = 64` sessions, where [EXP-0035](../EXP-0035-epoch-rank-slope/README.md)'s faster-than-linear span term dominates, the tensor generator's epoch rank-loss rate falls with the prime as `p^{-a}` with `a` near 2, the `1/q` order, rather than near 1.

## Success criterion

The model is EXP-0033's, unchanged: G-T, `Dn = 3`, the same layout and the same event E. The cells:

| `N` | Trials per cell `T_N` |
|---:|---:|
| 16 | 2,000,000 |
| 32 | 1,000,000 |
| 64 | 400,000 |

Each `N` runs at `p ∈ {7, 11, 19}`.

**The statistic.** `a_N` is minus the least-squares slope of `ln f` against `ln p`, over the three primes. Its 95% interval comes from 2,000 parametric bootstrap resamples (binomial at each cell's observed rate, seed `0x0036`). The decision rests on `a_64`. `a_16` and `a_32` are reported beside it.

**The verdicts.**
- **Validated** when `a_64`'s interval lies entirely above 1.5. The span term then falls like `1/q`, or faster.
- **Refuted** when `a_64`'s interval lies entirely below 1.5. The span term then falls like `1/p`, and at 405B's `N = 681` the tensor generator would miss the epoch target: R2 is necessary.
- **Inconclusive** otherwise, or when any cell has fewer than 20 failures.

**The controls** are EXP-0033's three. A failed control makes the run invalid.

## Method

- **The harness.** EXP-0033's `main.rs`, with its grid and trial counts changed. Each cell is split into chunks of `10^4` trials, so that one cell's work spreads over all cores. Chunk `c` of cell `k` uses the stream `0x0036 + (k·1000 + c)·0x9E3779B97F4A7C15`.
- **The fit.** `fit.py` computes `a_N` and the bootstrap.

## Reproduction

```sh
cd research/experiments/EXP-0036-epoch-rank-p-scaling
cargo run --release > results/run.txt
python3 -I fit.py results/run.txt > results/fit.txt
```

Environment: Apple M4 Max, macOS 26, `rustc`, Python 3. Not run.

## Results

Not run.

## Analysis

Not run.

## Verdict

Not run.

## Follow-ups

Not run.
