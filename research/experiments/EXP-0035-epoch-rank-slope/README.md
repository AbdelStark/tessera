# EXP-0035: The slope in sessions of the tensor generator's epoch rank loss

| | |
|---|---|
| Status | `done: inconclusive` |
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

Environment: Apple M4 Max (16 cores), macOS 26.6.2, `rustc 1.95.0`, Python 3.15.2, 2026-11-07; source `main` at `30f66dd` plus this directory. One run, 451 s wall. Raw output: [results/run.txt](results/run.txt), [results/fit.txt](results/fit.txt).

## Results

The controls all pass:
1. The rank routine agrees with brute force.
2. G-T at real points fails on every trial.
3. G-U at `p = 127` fails on none.

The counts are G-T at `Dn = 3` and `p = 7`, with `10^5` trials per `N` (E0, exact counts):

| `N` | Failures | Rate `f` | Per session `f/N` |
|---:|---:|---:|---:|
| 8 | 6 | 6.0×10^-5 | 7.5×10^-6 |
| 16 | 12 | 1.2×10^-4 | 7.5×10^-6 |
| 32 | 52 | 5.2×10^-4 | 1.6×10^-5 |
| 64 | 167 | 1.67×10^-3 | 2.6×10^-5 |

The registered fit gives slope `s = 1.651`, with a 95% bootstrap interval of `[1.358, 2.178]`.

## Analysis

**By the registered criterion, the run is `inconclusive`.** At `N = 8` it had 6 failures, fewer than the 10 the criterion requires at every `N`. The criterion also names a refutation, the interval's lower end above 1.0, and that one holds too. The text does not say which clause takes precedence. So the stricter reading is reported: `inconclusive`, not `refuted`.

**The signal is growth faster than linear in `N`.** These points are post hoc and decide nothing:
- The per-session rate is flat from 8 to 16 sessions, then rises about 3.5 times by 64.
- Without `N = 8`, the slope over 16, 32 and 64 is 1.90.
- [EXP-0034](../EXP-0034-epoch-rank-scaling/README.md)'s same cell sat at 1.9 times its linear expectation (17 against 9.1), and its criterion allowed up to 2 times. So its `validated` verdict stands as registered, but its tolerance hid this trend.

**What it means for IDEA-0044's lemma.** A bound that sums one fixed per-session rate over the epoch is not supported at `p = 7`. Each session's chance of falling into the span of the earlier sessions' vectors appears to grow with that span's dimension. So the lemma needs a term that depends on the span, roughly `j` for session `j`, or about `N^2` over an epoch. The question is how that term scales with `p`:
- **If it scales as `p^{-2}`,** for example `ϑ·j/q` per session, then 405B's `N = 681` gives about `ϑ·N^2/q ≈ 2^{-105}` per matrix. That is near the `2^{-100}` target, but not comfortably below it.
- **If it scales as `1/p`,** the tensor generator fails the epoch target, and R2 is needed after all.

At a single prime this run cannot tell the two apart.

**Threats to validity.**
- One prime, `p = 7`, and one layout. At `p = 7` events of every order in `1/p` are frequent, so the slope in `N` may mix terms of different orders in `p`.
- A small count at `N = 8`.
- EXP-0033's other threats still apply: a measurement, not a proof; E is necessary for a leak, not sufficient; only the vector views are modeled.

## Verdict

`inconclusive` as registered: the `N = 8` count is below 10. The registered refutation signal also holds: the bootstrap interval `[1.36, 2.18]` lies above 1. So the epoch rank loss grows faster than linearly in the number of sessions at `p = 7`. IDEA-0044 stays `validated` for its EXP-0033 claim, that no layout fails at a `1/p` rate at fixed `N`. But the summed per-session form of its lemma is now in doubt, and the queue item is amended to need a span-dependent term.

## Follow-ups

- **The growth across primes.** Measure the `N`-growth at three or more primes (`p = 7, 11, 19`), with `N = 16` to `128` and enough trials for at least 30 failures per cell. That separates a `j/q` term from a `j/p` term. It decides between "near target at 405B" and "R2 needed".
- **The R2 decision.** Until then, the evidence no longer suggests that R2 is unnecessary at the epoch level. #394's decision should treat R2 as the safe choice.
