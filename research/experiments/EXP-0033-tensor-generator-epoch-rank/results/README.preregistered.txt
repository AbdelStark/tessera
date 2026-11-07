# EXP-0033: How often does the tensor proximity generator lose rank on an epoch's blinding rows?

| | |
|---|---|
| Status | `planned` |
| Idea | [IDEA-0044](../../ideas/IDEA-0044-tensor-generator-epoch-rank.md) |
| Evidence level | E0 (exact arithmetic over small prime fields; a measured rate, not a proof) |
| Hardware | Apple M4 Max CPU, macOS; no GPU needed |
| Pre-registered on | 2026-11-07 |

## Hypothesis

On RFC-0012's and RFC-0065's blinding-row layout, the tensor proximity generator `eq(ξ*, row)` loses rank on an epoch's revealed blinding coefficients with probability that falls at least as fast as `p^{-2}`, not as `p^{-1}` as the simulator argument's §6.3 bound allows.

## Success criterion

**The model.** It is one weight matrix over `Fp2 = Fp[X]/(X^2 + 1)`, for primes `p ≡ 3 (mod 4)`, where `X^2 + 1` is irreducible.
- **Rows.** `Dn` data rows come first, then `B = 6N + 8` blinding rows. An `eq` vector has `ϑ = ⌈log2(Dn + B)⌉` row bits.
- **Reservations.** Session `i` (from 0) reserves blinding rows `4 + 6i` to `9 + 6i`: the evaluation pair `β_i`, the mask row `R_W`, and the proximity pair `F_i` (RFC-0065 'The reservation'). The first four and last four rows belong to the linking proofs, and none of the sessions reveals them.
- **Revealed rows.** Each session reveals two `Fp2` combined rows. The evaluation row puts `(1, X)` on `β_i` and nothing on any other blinding row. The proximity row puts a generator coefficient on every blinding row.
- **The three generators.**
  - **G-T**, the tensor generator that RFC-0011 step 3, RFC-0012 and RFC-0049 specify: `eq(ξ*_i, Dn + r)` on blinding row `r`, with `ξ*_i` uniform in `Fp2^ϑ`.
  - **G-T1**, G-T under R1: `(1, X)` on `F_i`, and `eq` everywhere else.
  - **G-U**, the R2 control: `(1, X)` on `F_i`, and an independent uniform `Fp2` coefficient on every other blinding row.
- **The event E.** Each `Fp2` coefficient on an `Fp` row gives two `Fp` entries, so the `N` sessions' revealed rows form a `4N × B` `Fp` matrix of blinding coefficients. E is that matrix having rank below `4N`. Outside E, the revealed rows are uniform whatever the data, by Lemma 1 of the simulator argument. E is necessary for a leak.

**The grid.** Every combination of:
- the three generators;
- `N ∈ {1, 2, 4, 8}` and `Dn ∈ {0, 3}`;
- `p ∈ {7, 11, 19, 23, 31, 43, 59, 67, 83, 103, 127}`.

Each of these 264 cells runs `T = 10^6` independent trials. A cell's failure rate is `f = (trials in E) / T`.

**The exponent.** For a given generator and `(N, Dn)`, it is `e = −slope` of the least-squares line of `ln f` against `ln p`, over the primes where the cell has at least 10 failures. It is reported only when at least three primes qualify.

**The verdicts.**
- **Validated** when all three hold:
  1. For G-T and G-T1, every `(N, Dn)` has either `e ≥ 2.0`, or fewer than 10 failures at every `p ≥ 19`.
  2. G-U has no failures at any `p ≥ 31`.
  3. All three controls pass.
- **Refuted** when, for G-T or G-T1, some `(N, Dn)` has at least 100 failures at some `p ≥ 31` and `e ≤ 1.3`. That is a structural failure at rate about `1/p`, which would make R2, or exclusion with its own soundness argument, necessary (IDEA-0044's kill criterion).
- **Inconclusive** otherwise.

**The controls.** A failed control makes the run invalid, not a result.
1. **Rank against brute force.** At `p = 3`, the rank routine must agree with a brute-force null-space count on 10,000 random matrices of up to 4 × 6.
2. **A planted real point.** G-T with `ξ*_i ∈ Fp^ϑ` must be in E on every one of 10,000 trials at `p = 31`. Every `eq` vector is then real, so its imaginary half is zero.
3. **A full-rank witness.** G-U at `p = 127` must be outside E in at least 99.9% of 10,000 trials.

## Method

- **The harness.** A standalone Rust binary in this directory, with no dependencies. It runs all three generators in one program, on the same layouts and the same random streams.
- **Inputs.** SplitMix64 with seed `0x0033`. Each cell has its own stream: the seed plus the cell's index, times `0x9E3779B97F4A7C15`. Field elements are drawn by rejection.
- **Rank.** Gaussian elimination over `Fp` in `u64`, on the `4N × B` matrix.
- **Output.** For each cell: the trial count and the failure count. For each generator and `(N, Dn)`: the qualifying primes and `e`. Then the three controls. Wall time is reported for information only: this is E0, and no number here is a timing.

## Reproduction

```sh
cd research/experiments/EXP-0033-tensor-generator-epoch-rank
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
