# EXP-0033: How often does the tensor proximity generator lose rank on an epoch's blinding rows?

| | |
|---|---|
| Status | `done: validated` |
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

Environment: Apple M4 Max (16 cores), macOS 26.6.2, `rustc 1.95.0`, 2026-11-07; source `main` at `74c75f9` plus this directory. One run, 123.4 s wall on 16 threads, 3.6 MB peak RSS. Raw output: [results/run.txt](results/run.txt).

## Results

The controls all pass:
1. The rank routine agrees with brute force on all 10,000 matrices at `p = 3`.
2. G-T at real points is in E on all 10,000 trials.
3. G-U at `p = 127` is in E on none of its 10,000.

The rows below give the failures out of `10^6` trials per cell (E0, exact counts).

| Generator | `N` | `Dn` | p = 7 | 11 | 19 | 23 | 31 | 43 | 59 | 67 | 83 | 103 | 127 | `e` |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| G-T | 1 | 0 | 1,000 | 164 | 20 | 6 | 0 | 1 | 0 | 0 | 0 | 0 | 0 | 3.92 |
| G-T | 1 | 3 | 20,624 | 8,500 | 2,783 | 1,886 | 1,034 | 492 | 278 | 212 | 145 | 94 | 55 | 2.03 |
| G-T | 2 | 0 | 3,258 | 488 | 57 | 19 | 8 | 2 | 1 | 0 | 0 | 0 | 0 | 4.23 |
| G-T | 2 | 3 | 1,750 | 234 | 19 | 8 | 3 | 1 | 1 | 0 | 0 | 0 | 0 | 4.53 |
| G-T | 4 | 0 | 336 | 38 | 1 | 1 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | n/a |
| G-T | 4 | 3 | 11,090 | 1,945 | 211 | 113 | 39 | 16 | 2 | 1 | 1 | 0 | 0 | 3.67 |
| G-T | 8 | 0 | 268 | 19 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | n/a |
| G-T | 8 | 3 | 114 | 6 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | n/a |
| G-T1 | 2 | 0 | 2 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | n/a |
| G-T1 | other seven | | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | n/a |
| G-U | all eight | | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | n/a |

`e` is the pre-registered least-squares exponent, over primes with at least 10 failures, reported when at least three qualify.

## Analysis

**The criterion is met.**
1. Every G-T row has `e ≥ 2.03` or fewer than 10 failures at every `p ≥ 19`. G-T1 has two failures in all 88 of its cells, both at `p = 7`.
2. G-U has none at any prime.
3. The controls pass.

No layout shows the `1/p` rate that §6.3's Schwartz–Zippel bound allows.

**The one `p^{-2}` row has an exact cause.** This explanation is post hoc. It is derived from the layout, and the run was not designed to test it.
- **The layout.** At `N = 1, Dn = 3`, the blinding rows are 3 to 16, and `ϑ = 5`. Row 16, a link row, is the only one with bit 4 set.
- **The event.** If `ξ*_4 = 1`, every other row's `eq` coefficient carries the factor `1 − ξ*_4 = 0`. The proximity row then has one non-zero `Fp2` coefficient, on one `Fp` row, so its two `Fp` vectors are dependent. That has probability exactly `1/q = 1/p^2`.
- **The fit.** The measured `f·p^2` is 1.01 at 7, 0.99 at 31 and 0.89 at 127.
- **In general.** A blinding block that crosses a power-of-two boundary by a few rows has such a coordinate.
- **At 405B,** it is a `1/q = 2^{-128}` event per session and per coordinate: about `ϑ_T·N/q ≈ 2^{-114}` per matrix and epoch. The other rows fall faster, `e` from 3.7 to 4.5.

**R1 removes the failures.** With `(1, X)` on its own proximity pair, each session's proximity row always covers its own two `Fp` directions. G-T1 failed twice in `8.8×10^7` trials, both at `p = 7`.

**Threats to validity.**
- **A measurement, not a proof.** An exponent fitted over eleven small primes says nothing about `2^{64}`. Nor do zero counts at `p ≥ 31` prove a rate. The argument still has to be written.
- **Toy sizes.** `N ≤ 8` and `ϑ ≤ 6`, against `N = 681` and `ϑ_T = 18` at 405B. A failure mode that needs many sessions, or many row bits, would not appear here. The sequential bound of IDEA-0044 adds over sessions, so a per-session rate is what matters. But the span that a session must avoid grows with `N`, and only `N ≤ 8` is tested.
- **E is necessary, not sufficient, for a leak.** The data rows' coefficients in a dependent combination were not checked. That makes the measured rate an upper bound on the leak rate.
- **Scope of the model.** Only the vector views are in it. The scalars `q_T` and `W'(p)`, the column relations of §6.1, `ChannelParams`' `2e + 4` rows and the link rows' own revelation are left to the simulator argument, as Lemma 7 leaves them.
- **One field basis.** `X^2 + 1` replaces `X^2 − 7`. Fp-rank does not depend on the basis of `Fp2`, so this does not change E.

## Verdict

`validated`. The tensor generator's rank loss falls at least as fast as `p^{-2}` in every tested layout, and the only `p^{-2}` term has an exact `1/q` cause. Under R1 the failures all but vanish. IDEA-0044 moves to `validated`: a sharper Lemma 7 for the specified generator is worth writing, and §6.3's `2^{-49}` does not describe the event. The zero-knowledge gate stays `unknown` until that lemma exists and is reviewed (#145, #394).

## Follow-ups

- **The lemma.** Write Lemma 7 for G-T1 using the spare blinding rows, with the sequential bound per session and the `1/q` straddle term above. Then propose it as a replacement for §6.3 in the simulator argument (#145), and record in #394 that R2 is sufficient but not necessary for zero knowledge. R1 still stands as a soundness requirement.
- **A larger grid.** Run `N` up to 64 and `ϑ` up to 10 at a few primes, as a check on the per-session rate as the span grows. Exact arithmetic on a CPU is enough: no H100 run applies.
