# EXP-0036: How the epoch rank loss's span term scales with the field

| | |
|---|---|
| Status | `done: inconclusive` |
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

Environment: Apple M4 Max (16 cores), macOS 26.6.2, `rustc 1.95.0`, Python 3.15.2, 2026-11-07; source `main` at `779d9d5` plus this directory. One run, 741 s wall. Raw output: [results/run.txt](results/run.txt); the fit's output and its failure: [results/fit.txt](results/fit.txt).

## Results

The controls all pass.

The counts are G-T at `Dn = 3` (E0, exact counts):

| `N` | Trials | p = 7 | p = 11 | p = 19 |
|---:|---:|---:|---:|---:|
| 16 | 2,000,000 | 401 | 30 | 0 |
| 32 | 1,000,000 | 576 | 32 | 0 |
| 64 | 400,000 | 638 | 29 | 0 |

**The registered fit did not run to completion.** `fit.py` takes `ln f` of every cell, and it stopped on the zero counts at `p = 19` (`ValueError`, recorded in `results/fit.txt`). The script was not changed after the run. The criterion's own clause, any cell below 20 failures, already decides the verdict.

## Analysis

**As registered, the run is `inconclusive`.** The three `p = 19` cells have no failures, and `a_64` cannot be computed.

**The zeros still carry information.** The following is post hoc and decides nothing as registered.

| `N` | Two-point exponent, 7 → 11 | Expected at 19 under `1/q` | Under `1/p` | Observed | `P(0)` under `1/q` |
|---:|---:|---:|---:|---:|---:|
| 16 | 5.7 | 54 | 148 | 0 | 2×10^-24 |
| 32 | 6.4 | 78 | 212 | 0 | 1×10^-34 |
| 64 | 6.8 | 87 | 235 | 0 | 3×10^-38 |

- **Faster than `1/q`.** From `p = 7` to `p = 11`, the rate falls by a factor of 13 to 22 at every `N`. That is an exponent near 6, far steeper than 2. At `p = 19`, extrapolating `p = 7` by `1/q` predicts 54 to 87 failures per cell, and none appeared. So the epoch rate falls faster than `1/q` in this range. That includes [EXP-0035](../EXP-0035-epoch-rank-slope/README.md)'s faster-than-linear growth in `N`, which is largest at `p = 7`.
- **A small-field effect.** The likeliest reading is that the span term found at `p = 7` comes from events whose probability falls as a high power of `1/p`. For example, several coordinates of `ξ*` land in a set of a few field elements, which only a field this small makes common.
- **Bounds at `p = 19`.** The 95% upper bounds are below `1.5×10^-6`, `3×10^-6` and `7.5×10^-6` at `N = 16`, `32` and `64`. Next to [EXP-0033](../EXP-0033-tensor-generator-epoch-rank/README.md)'s `N = 8` rate at `p = 19` (0 in `10^6`), this shows no visible growth in `N` at the largest prime tested.

**Threats to validity.**
- **The registered statistic was never computed.** The verdict above is the criterion's. The reading in this section is post hoc.
- **Three primes, all small.** The orders in `1/p` that dominate here may not be the ones that dominate at `2^{64}`. The analytic `1/q` straddle term of EXP-0033 is the floor any lemma must carry.
- **A measurement, not a proof.** EXP-0033's other threats still apply.

## Verdict

`inconclusive` as registered: there were zero failures at `p = 19`, and the registered fit stopped on them. The post-hoc reading is that the faster-than-linear span term EXP-0035 measured at `p = 7` falls much faster than `1/q` with the field: it vanishes at `p = 19` against 54 to 87 expected failures. That lifts EXP-0035's objection to a per-session bound, but only as evidence. Until a lemma bounds the span term analytically, #394 should still treat R2 as the safe choice.

## Follow-ups

- **The lemma (queue item 0).** It must carry the exact `1/q` straddle term and bound the span-dependent events analytically. The data say those events are of high order in `1/p`, but that is no substitute for a bound.
- **A rerun at larger primes.** To measure an exponent rather than bound it: `p ∈ {11, 13, 17}` at `N = 64`, with enough trials for 20 failures per cell, or an exponent fit that admits zero counts by a likelihood model registered in advance.
