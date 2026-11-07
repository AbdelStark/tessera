# IDEA-0044: Keep the tensor proximity generator for masked weights by proving its epoch rank directly

| | |
|---|---|
| Status | `validated` |
| Origin | [#394](https://github.com/AbdelStark/tessera/issues/394); [simulator argument](../../docs/zk/simulator.md) §6.3 and requirement R2; [#145](https://github.com/AbdelStark/tessera/issues/145) |
| Lever | zero knowledge across a weight epoch (G-5); `regen` (the proximity row's PRG draws, [RFC-0075](../../docs/rfcs/RFC-0075-weight-mask-cost.md)); no Δμ credited |
| Spec surfaces | None if it holds: RFC-0011 step 3, RFC-0012 and RFC-0049 keep `eq(ξ*_T, row)` over data and blinding rows, and the simulator's Lemma 7 gains a proof for that generator. R2 then becomes one sufficient condition, not a requirement |
| Priority | correctness prerequisite: Theorem 2 of the simulator argument, and IDEA-0043's context commitments, need an epoch argument under some generator |

## Mechanism

Nothing in the protocol changes. The idea is an argument.

The simulator argument's Lemma 7 shows that an epoch's combined weight rows hide the weights when the proximity generator gives the blinding rows independent uniform `Fp2` coefficients (R2). Under the generator the specification has, `eq(ξ*_T, row)` over every committed row, its only bound is Schwartz–Zippel over `Fp` on one determinant: `2Nϑ_T/p`, about `2^{-49}` at `N = 681` and `ϑ_T = 18`. That is far above the `2^{-100}` target. §6.3 says that a sharper argument for tensor coefficients may exist.

The sharper argument would use more of the epoch than §6.3 does.
- **Spare blinding rows.** §6.3's determinant uses only the proximity pairs `F(i)`. But each proximity row reaches every blinding row of the epoch: the mask rows `R_W`, the unreserved rows and the link rows. An epoch has `B_W = 6N + 8` of them against `4N` revealed `Fp` vectors.
- **Rank instead of one determinant.** The view leaks only if the revealed vectors' blinding coefficients lose rank: some non-zero `Fp`-combination of them has no blinding term. With `2N + 8` spare columns, the event needs many polynomial conditions at once. For `eq` vectors, all of them come down to ratios falling into `Fp`.
- **Sequential independence.** Each `ξ*_i` is a random-oracle output drawn after the earlier sessions' views are fixed. The seed is uniform from `V*`'s view in Live sessions too (§4.1). So the event can be bounded one session at a time, against the span of the earlier sessions' vectors, with no union over which sessions complete.

If the event has probability `p^{-e}` per session with `e ≥ 2`, then the epoch term is at most `N·p^{-e}`. That is `2^{-118}` at `N = 681` and `e = 2`, and the tensor generator then needs no R2.

## Why it could work

- Lemma 2.2 of the simulator argument already finds the pattern for one row: a pair of `eq` coefficients falls on one `Fp`-line with probability about `1/p`, but three rows that differ in two bits need two coordinates in `Fp`, about `p^{-2}` (E0, primary: the argument's own lemma).
- The obvious structured failure, every coordinate of `ξ*` in `Fp`, makes each `eq` vector real and its pair rank one. Over the `a` row bits that vary across the blinding block, it has probability about `p^{-a}` (E0).
- §6.3's `2^{-49}` comes from one `2N × 2N` determinant, which ignores the `2N + 8` spare columns. A rank bound over a wider matrix is usually far smaller than any one of its minors' (E0).

None of this is a proof. The cheapest test is to measure the event's rate over small primes and read off its exponent.

## Expected gain

There is no Δμ. Two things change if it holds:
- **Theorem 2 is proved without R2.** The simulator argument's epoch theorem would hold for the generator that RFC-0011, RFC-0012 and RFC-0049 specify.
- **The R2 decision gets one fewer reason.** #394's choice between R2 and RFC-0021's free-row stream would no longer be forced by zero knowledge. [SUR-0001](../surveys/SUR-0001-proximity.md) records an independent soundness reason for uniform coefficients: the Johnson-regime bounds.

On the `regen` side, RFC-0075 already accounts for full-support regeneration, with no blinder-exclusion credit, so no cost change is credited here.

## Security gate

| Property | Effect | Pass |
|---|---|---|
| Soundness bits (G-4, RFC-0014) | None: the generator is unchanged | pass |
| Uniqueness upstream of the logits (SP-3, RFC-0009 Lemma 0) | None | pass |
| Zero knowledge (G-5, RFC-0013) | The idea is a ZK argument. Until it is proved, the tensor generator has only §6.3's bound | unknown |
| Verifier independence (SP-6) | None | pass |
| Post-quantum (hash-based commitments) | None | pass |

## Prior rejection check

- **Simulator argument §6.3.** It states "A sharper argument for tensor-structured coefficients may exist; none is known here", so the question is open, not rejected.
- **Draft RFC-0021.** Its free-row stream meets R2 by changing the generator. This idea keeps the generator.
- **RFC-0075.** It takes no blinder-exclusion credit, so the question has no effect on cost there.

## Kill criterion

A structured failure at rate about `1/p`: a layout where the tensor generator's revealed blinding coefficients lose rank with probability that scales as `p^{-1}`. That would show §6.3's bound is essentially tight, and R2 (or exclusion with its own soundness argument) necessary.

## Cheapest experiment

[EXP-0033](../experiments/EXP-0033-tensor-generator-epoch-rank/README.md): exact rank counts over `Fp2 = Fp[X]/(X^2 + 1)` for eleven primes from 7 to 127, on RFC-0012's and RFC-0065's blinding-row layout. Each run compares the tensor generator with and without R1 against uniform coefficients (R2) in the same harness, and fits the failure exponent. E0 (exact arithmetic), effort `s`. A measured exponent is not a proof, and toy sizes are not 405B's. If the data supports the idea, the next unit is the lemma itself.

## Result

[EXP-0033](../experiments/EXP-0033-tensor-generator-epoch-rank/README.md) is `done: validated` (E0). Over 264 cells of `10^6` exact trials:
- **G-T.** The tensor generator's rank loss falls as `p^{-2}` or faster in every layout, with fitted exponents 2.03 to 4.53.
- **Where `p^{-2}` comes from.** The one `p^{-2}` row is the exact `1/q` event of a row coordinate equal to 1 where the blinding block straddles a power of two. At 405B that is about `2^{-114}` per matrix and epoch.
- **G-T1.** Under R1 there were 2 failures in `8.8×10^7` trials, both at `p = 7`.
- **G-U.** Uniform coefficients had none.

So §6.3's `2Nϑ/p` does not describe the event. The zero-knowledge gate stays `unknown` until the lemma is written and reviewed.

**Amended by [EXP-0035](../experiments/EXP-0035-epoch-rank-slope/README.md)** (`done: inconclusive`, E0). At `p = 7` the epoch rate grows faster than linearly in the number of sessions: slope 1.65, 95% interval `[1.36, 2.18]`. A fixed per-session rate summed over the epoch is therefore not supported. The lemma needs a term that grows with the span of earlier sessions, and whether that term scales as `1/q` or `1/p` decides between 'near target at 405B' and 'R2 needed'. Until that is measured, R2 is the safe choice.

**Further amended by [EXP-0036](../experiments/EXP-0036-epoch-rank-p-scaling/README.md)** (`done: inconclusive` as registered, E0). The span term falls far faster than `1/q`. From `p = 7` to `11` the rate drops 13 to 22 times. At `p = 19` there are no failures in any cell, against 54 to 87 expected under `1/q`. So the faster-than-linear growth is a small-field effect of high order in `1/p`, a post-hoc reading. A per-session bound remains plausible, but only an analytic lemma settles it. R2 stays the safe choice until then.

## Refutations

None yet.
