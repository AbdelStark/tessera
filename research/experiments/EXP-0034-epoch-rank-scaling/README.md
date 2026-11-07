# EXP-0034: Does the tensor generator's epoch rank loss grow faster than the number of sessions?

| | |
|---|---|
| Status | `planned` |
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

Environment: Apple M4 Max, macOS 26, `rustc`. Not run.

## Results

Not run.

## Analysis

Not run.

## Verdict

Not run.

## Follow-ups

Not run.
