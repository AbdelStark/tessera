# EXP-0034: Does the tensor generator's epoch rank loss grow faster than the number of sessions?

| | |
|---|---|
| Status | `done: validated` |
| Idea | [IDEA-0044](../../ideas/IDEA-0044-tensor-generator-epoch-rank.md) |
| Evidence level | E0 (exact arithmetic over small prime fields; a measured rate, not a proof) |
| Hardware | Apple M4 Max CPU, macOS; no GPU needed |
| Pre-registered on | 2026-11-07 |

## Hypothesis

With `N` from 16 to 64 sessions per epoch, the tensor generator's rank-loss rate grows at most linearly in `N` from EXP-0033's measured rate at `N = 8`, both without R1 and with it. So a per-session bound summed over sessions, the form IDEA-0044's lemma takes, is not contradicted at these sizes.

## Success criterion

**The model** is [EXP-0033](../EXP-0033-tensor-generator-epoch-rank/README.md)'s, unchanged:
- the same layout of `B = 6N + 8` blinding rows after `Dn` data rows;
- the same reservations;
- the same event E (Fp-rank of the `4N × B` blinding-coefficient matrix below `4N`);
- the same generators G-T and G-T1.

`ϑ` grows with `N`: 7, 8 and 9 row bits at `N = 16, 32, 64`.

**The grid.**
- G-T and G-T1;
- `N ∈ {16, 32, 64}` and `Dn ∈ {0, 3}`;
- `p ∈ {7, 11, 19}`.

That makes 36 cells of `T = 10^4` trials each.

**The reference** is linear growth from EXP-0033's `N = 8` failures out of `10^6` (`f_8`). The expected failures in a cell are `x = f_8 · (N/8) · T / 10^6`. The `f_8` values:

| Generator, `Dn` | p = 7 | p = 11 | p = 19 |
|---|---:|---:|---:|
| G-T, 0 | 268 | 19 | 0 |
| G-T, 3 | 114 | 6 | 0 |
| G-T1, 0 | 0 | 0 | 0 |
| G-T1, 3 | 0 | 0 | 0 |

**The verdicts.**
- **Validated** when every cell has at most `max(10, 2x)` failures, and the controls pass.
- **Refuted** when some cell has at least 20 failures and more than `10·max(1, x)`. That is growth well beyond linear in `N`, and a sum of per-session bounds would not describe it.
- **Inconclusive** otherwise.

**The controls.** A failed control makes the run invalid. They are EXP-0033's three controls, plus a fourth:
4. G-U at `N = 64` and `p = 7` has at most 10 failures in `10^4`.

## Method

- **The harness.** EXP-0033's `main.rs`, copied unchanged except for the grid, `T`, and control 4.
- **Inputs.** SplitMix64 with seed `0x0034`, one stream per cell as in EXP-0033.
- **Output.** The failure count per cell, then the controls. Wall time is reported for information only.

## Reproduction

```sh
cd research/experiments/EXP-0034-epoch-rank-scaling
cargo run --release > results/run.txt
```

Environment: Apple M4 Max (16 cores), macOS 26.6.2, `rustc 1.95.0`, 2026-11-07; source `main` at `2267668` plus this directory. One run, 107.5 s wall, 15 MB peak RSS. Raw output: [results/run.txt](results/run.txt).

## Results

The controls all pass:
1. The rank routine agrees with brute force on all 10,000 matrices.
2. G-T at real points fails on all 10,000 trials.
3. G-U at `p = 127, N = 2` fails on none.
4. G-U at `p = 7, N = 64` fails on none.

Each cell below gives the failures out of `10^4` trials, then the linear expectation `x`, then the pre-registered limit `max(10, 2x)` (E0, exact counts):

| Generator | `Dn` | `p` | `N = 16` | `N = 32` | `N = 64` |
|---|---:|---:|---|---|---|
| G-T | 0 | 7 | 5 / 5.4 / 10.7 | 9 / 10.7 / 21.4 | 13 / 21.4 / 42.9 |
| G-T | 0 | 11 | 0 / 0.4 / 10.0 | 0 / 0.8 / 10.0 | 3 / 1.5 / 10.0 |
| G-T | 0 | 19 | 0 / 0.0 / 10.0 | 0 / 0.0 / 10.0 | 0 / 0.0 / 10.0 |
| G-T | 3 | 7 | 3 / 2.3 / 10.0 | 7 / 4.6 / 10.0 | 17 / 9.1 / 18.2 |
| G-T | 3 | 11 | 0 / 0.1 / 10.0 | 2 / 0.2 / 10.0 | 1 / 0.5 / 10.0 |
| G-T | 3 | 19 | 0 / 0.0 / 10.0 | 0 / 0.0 / 10.0 | 0 / 0.0 / 10.0 |
| G-T1 | 0, 3 | 7, 11, 19 | 0 | 0 | 0 |

## Analysis

**The criterion is met.** Every cell is within its limit, and G-T1 had no failure in `1.8×10^5` trials, at `ϑ` up to 9 and up to 64 sessions.

**The tightest cell.** G-T at `N = 64`, `Dn = 3`, `p = 7` had 17 failures against `x = 9.1` and a limit of 18.2. At `Dn = 0` the same column is below its expectation (13 against 21.4). One cell at 1.9× its linear expectation is within this run's counting noise, but it is not evidence of sublinear growth either. A sharper look at `p = 7`, `Dn = 3` with `10^5` trials per `N` would settle the slope.

**Threats to validity.**
- Everything EXP-0033 lists still applies: a measurement, not a proof; E is necessary for a leak, not sufficient; only the vector views are modeled.
- `10^4` trials per cell resolves only the smallest primes. At `p ≥ 19` both generators had no failures, which bounds the rate only loosely.
- `N = 64` and `ϑ = 9` are still far from 405B's `N = 681` and `ϑ_T = 18`.

## Verdict

`validated`. Up to 64 sessions, the tensor generator's rank loss grows no faster than linearly in `N`, within the registered tolerance. This is the form a per-session bound summed over sessions needs. Under R1 there were no failures. IDEA-0044 stays `validated`; its lemma remains queue item 0.

## Follow-ups

- **The slope at `Dn = 3`.** The tightest cell (`N = 64`, `Dn = 3`, `p = 7`) with `10^5` trials per `N`, to measure the slope rather than bound it.
- **The lemma** (queue item 0), which is what decides #394.
