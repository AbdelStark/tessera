# EXP-0035: The slope in sessions of the tensor generator's epoch rank loss

| | |
|---|---|
| Status | `planned` |
| Idea | [IDEA-0044](../../ideas/IDEA-0044-tensor-generator-epoch-rank.md) |
| Evidence level | E0 (exact arithmetic over a small prime field; a measured rate, not a proof) |
| Hardware | Apple M4 Max CPU, macOS; no GPU needed |
| Pre-registered on | 2026-11-07 |

## Hypothesis

In [EXP-0034](../EXP-0034-epoch-rank-scaling/README.md)'s tightest configuration (G-T, `Dn = 3`, `p = 7`), the epoch rank-loss rate grows at most linearly in the number of sessions `N`: the fitted exponent of `f` against `N` is at most 1.

## Success criterion

The model is EXP-0033's, unchanged: G-T at `Dn = 3` and `p = 7`, with `N ∈ {8, 16, 32, 64}` and `T = 10^5` trials per `N`. The fit is the least-squares slope `s` of `ln f` against `ln N`, over the four `N`. The 95% interval of `s` comes from 2,000 parametric bootstrap resamples of the four counts (binomial at the observed rates, seed `0x0035`).

- **Validated** when `s ≤ 1.0`, or when the bootstrap interval contains 1.0 and its upper end is at most 1.5.
- **Refuted** when the bootstrap interval's lower end exceeds 1.0: growth faster than linear in `N`. A per-session bound summed over sessions would then not describe the epoch, and IDEA-0044's lemma would need a span-dependent term.
- **Inconclusive** otherwise, or when any `N` has fewer than 10 failures.

**The controls** are EXP-0033's three: rank against brute force, G-T at real points, and G-U at `p = 127`. A failed control makes the run invalid.

## Method

- **The harness.** EXP-0033's `main.rs`, copied with only its grid changed (`SESSIONS = [8, 16, 32, 64]`, `OFFSETS = [3]`, `PRIMES = [7]`, generator G-T, `T = 10^5`) and seed `0x0035`.
- **The fit.** `fit.py` reads the counts and computes the slope and the bootstrap interval with Python's `random` seeded `0x0035`.

## Reproduction

```sh
cd research/experiments/EXP-0035-epoch-rank-slope
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
