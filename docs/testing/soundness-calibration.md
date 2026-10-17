# Soundness calibration over `F65537`

Over the 17-bit test field, a prover that cheats by guessing a challenge succeeds often enough to measure. [07 §4](../spec/07-testing-strategy.md#4-soundness-and-uniqueness-tests) asks that its empirical rate lie within a 99% binomial interval of the analytic bound: `d/|F|` per sumcheck round ([RFC-0010 'Testing Strategy'](../rfcs/RFC-0010-proof-schedule.md#testing-strategy)), `(N + T)/|F|` for a LogUp instance ([RFC-0008 'Testing Strategy'](../rfcs/RFC-0008-logup-and-tables.md#testing-strategy)). This checks that the analysis and the implementation agree. The harness is [`crates/tessera-e2e/tests/soundness_calibration.rs`](../../crates/tessera-e2e/tests/soundness_calibration.rs), with its parts in [`calibration/`](../../crates/tessera-e2e/tests/calibration/).

## Running it

```sh
# Quick tests, in every `cargo test`: the strategies, the verifiers and the intervals, in about a second.
cargo test -p tessera-e2e --test soundness_calibration
# The measurement: about 2·10^7 trials on every core, with the report on stderr.
cargo test --locked -p tessera-e2e --test soundness_calibration -- --ignored --nocapture
```

The `soundness calibration` job of [`.github/workflows/nightly.yml`](../../.github/workflows/nightly.yml) runs the measurement every night, on pull requests that change the harness, and on demand. Its step fails after 10 minutes. A second step runs the measurement of random bad lookups in `tessera-logup`'s own calibration test (see 'Limits'), with the same limit. It runs in the dev profile, which keeps overflow checks and debug assertions on, at about 1.1 times the release profile's time.

## The cheating provers

Each prover cheats once and is honest otherwise, so that the verifier accepts exactly when one challenge falls in a set `Z` of `K` elements. Every trial checks that its verdict is this event. Over many trials, the rate must then be `K/|F|`, if the challenge is uniform, and `K` must be the analysis's count.

### A lie in one sumcheck round

The instance is a product of `d` random tables over `n = 2` variables, whose sum `S` the engine (`tessera_sumcheck::prove_batch`) proves under the labels of a stage. The prover claims `S + δ(0) + δ(1)` instead, with `δ = Π_{k=1..d} (X − (P − k))`, and alters what the engine absorbs:

- **Round 0** sends `p_0 + δ`: the honest round polynomial plus `δ`, consistent with the false claim. The verifier's next claim is `p_0(r_0) + δ(r_0)`, the honest one exactly when `r_0` is a root of `δ`. The engine's later messages are then honest, and the proof is accepted.
- **After a miss**, the verifier's claim exceeds the honest one by `Δ ≠ 0`. Each later round sends the honest values plus `Δ/2`, which the verifier reads as the polynomial `p_j + Δ/2`, since it derives `p_j(1) = claim − p_j(0)`. The excess halves and never vanishes, a constant having no root, so the final check fails.

The proof is therefore accepted exactly when `r_0 ∈ {P − 1, …, P − d}`: `K = d`, and the rate is `d/|F|`. No lie does better in one round. A message is `d` values, so `p_0' − p_0` has degree at most `d` and is not zero, since the claims differ. The roots sit at the top of the field, next to where the challenge sampler starts rejecting words, so that a sampler that mishandles its largest values shows (see 'Mutations').

### Lookups outside the table

An instance has `T` rows and `N` lookups, keys without values, and the multiplicities the prover commits. The prover is `tessera_logup::prove_logup` itself: it builds honest fraction trees over the bad leaves, padding included. Each tree level is then a true statement, and the verifier accepts exactly when the roots balance: when `α` is no pole and

```text
R(α) = Σ_i 1/(α − a_i) − Σ_j m_j/(α − t_j) = 0.
```

With the `N` keys `T, …, T + N − 1` outside the rows `0, …, T − 1`, `R` has `N + T` distinct poles and a residue of 1 at each key, so it is not zero. Its numerator has degree at most `N + T − 1`, which bounds the accepting values of `α` ([RFC-0008 'Relation'](../rfcs/RFC-0008-logup-and-tables.md#relation)). The multiplicities are chosen to reach that bound. The numerator is `Z·G`, with `T` chosen roots `Z = Π_s (α − (P − 1 − c − s))` and `G` of degree `N − 1`. `G` interpolates the values that make the keys' residues 1. A row's multiplicity is the opposite of its residue, plus the lookup tree's padding at row 0. The first `c` for which `G`'s `N − 1` roots lie in the field, apart from the poles and the chosen roots, gives the instance. A brute force over the 65,537 values of `α` then counts the accepting set, from the leaves alone: `K = N + T − 1` in every setting, against the bound `N + T`.

The levels' own sumchecks are not attacked. They are honest, so they pass at every challenge and add nothing to the rate, which measures the identity alone. Their per-round error is the degree-3 case of the sumcheck lie above.

## Challenges and verifiers

- **Challenges** come from RFC-0004's transcript (`tessera_hash::Transcript`: BLAKE3, labels as RFC-0014's schedule has them). They are drawn in `F65537` as `challenge_fp` draws in `Fp`, but from 17-bit words: of a squeeze's four little-endian words, each masked to its low 17 bits, the challenge is the first below `P`; if none is, the transcript squeezes again. Every element is drawn with probability `1/P`, if the hash's output is uniform. The engines are generic over `PolyField`, so `r_0` and `α` lie in `F65537` itself, and `|F| = 65537` in every bound. Over `F65537Ext2`, `|F| ≈ 2^32`, and a guess would almost never succeed.
- **Sessions.** Each trial is a session of its own, with a fresh session id (INV-PROTO-4), in live mode: the prover gets one attempt at its challenge, as in interactive soundness. Grinding, which non-interactive proofs pay for with `g_seed` (RFC-0014 'Grinding'), is outside the measurement.
- **Order.** The prover's claim and tables, or its lookups and multiplicities, are absorbed under `round1` before any challenge, as their commitments are (RFC-0014 'Transcript schedule', row 2; INV-FS-1).
- **Verifiers.** `tv-core` checks Goldilocks proofs only, so each proof is checked by a verifier written in the harness from RFC-0010 'Sumcheck messages' and 'Tree levels' and RFC-0008 'Fraction tree': round lengths, `p(1) = claim − p(0)`, the final check, the levels' `γ`, `λ`, front-padding, `eq` and `ρ`, the roots' and leaves' checks. In place of openings, it reads the committed columns directly: the tables, the keys and the multiplicities. Its polynomial arithmetic (multilinear extensions, `eq`, Lagrange interpolation) is written in the harness, not taken from `tessera-poly`; the field's is `tessera-field`'s.
- **Scripted challenges.** The quick tests also run the provers on a transcript that sets a challenge. At every root of `δ`, and at every value of `α` that the brute force accepts, the verifier accepts; at every pole and at other values, it rejects. The rates themselves come only from the real transcript.

## Statistics

- **Intervals.** Two-sided 99% Clopper–Pearson intervals, which invert the two one-sided binomial tests and cover every rate with probability at least 99%. They are computed in the harness, with no dependency: binomial tails summed in logarithms, endpoints by bisection. A quick test compares them, to `10^-9` relative, with an independent oracle: [`calibration/oracle.py`](../../crates/tessera-e2e/tests/calibration/oracle.py), which sums the tails term by term in mpmath at 50 digits. Its points include the intervals reported below. Another checks, by direct summation for up to 120 trials, that each endpoint leaves 0.5% in its tail and that the intervals cover 99% of the outcomes.
- **Checks.** For each setting:
  1. the construction attains the analysis's count: `K = d`, or `K = N + T − 1`;
  2. `K/|F|` lies inside the interval, so that the implementation agrees with the analysis;
  3. the interval's lower end, a one-sided 99.5% lower bound, is at or below the bound `d/|F|` or `(N + T)/|F|`.
- **Trials.** Each setting runs `⌈200·|F|/K⌉` trials, for 200 expected accepts. The interval's half-width is then about 18% of the rate, so that an analytic rate off by a factor of 2 falls outside it in every setting.
- **Seeds** are fixed, so every run gives the same counts. A change to anything a transcript absorbs, such as a label, an encoding or an engine's messages, draws new challenges. With six settings at 99% each, one of them then fails by chance up to about 6% of the time. A fluke lands just outside one interval and goes away with other seeds; a defect persists.

## Results

The nightly job's log has the latest table. This run is from 2026-10-17, in the dev profile, on an Apple M4 Max with 16 threads: 13.3 s, 142 CPU-seconds. The counts do not depend on the number of threads.

| Setting | `K` | Trials | Accepted | Rate | 99% interval | `K/\|F\|` | Bound |
|---|---|---|---|---|---|---|---|
| sumcheck, `d = 1`, `n = 2` | 1 | 13,107,400 | 204 | 1.5564e-5 | [1.2900e-5, 1.8597e-5] | 1.5259e-5 | 1.5259e-5 |
| sumcheck, `d = 3`, `n = 2` | 3 | 4,369,134 | 203 | 4.6462e-5 | [3.8493e-5, 5.5541e-5] | 4.5776e-5 | 4.5776e-5 |
| sumcheck, `d = 7`, `n = 2` | 7 | 1,872,486 | 197 | 1.0521e-4 | [8.6904e-5, 1.2610e-4] | 1.0681e-4 | 1.0681e-4 |
| LogUp, `N = 1`, `T = 16` | 16 | 819,213 | 194 | 2.3681e-4 | [1.9532e-4, 2.8422e-4] | 2.4414e-4 | 2.5940e-4 |
| LogUp, `N = 2`, `T = 255` | 256 | 51,201 | 174 | 3.3984e-3 | [2.7723e-3, 4.1186e-3] | 3.9062e-3 | 3.9214e-3 |
| LogUp, `N = 3`, `T = 1023` | 1,025 | 12,788 | 205 | 1.6031e-2 | [1.3311e-2, 1.9117e-2] | 1.5640e-2 | 1.5655e-2 |

Every interval contains `K/|F|` and starts below the bound. The LogUp bound `(N + T)/|F|` lies inside each interval too. That is not asserted: an instance reaches `(N + T − 1)/|F|` at most, and enough trials would separate the two for a small table.

The degrees are the smallest, a tree level's and the largest that RFC-0014 allows (`d ≤ 7`). The tables have 16, 255 and 1,023 rows, with lookup trees of heights 0, 1 and 2, the last padded with row 0.

On the GitHub-hosted `ubuntu-24.04` runner, the pull request's run of the `nightly` workflow took 81.6 s for the measurement and 105.4 s for `tessera-logup`'s random lookups, with the same counts, and about 4 minutes for the whole job, building included: both steps stay well under their limit of 10 minutes.

## Mutations

The harness must be able to fail. Each mutation below was applied to the source, run and reverted:

| Mutation | Effect | Result |
|---|---|---|
| The field taken as half its size | Every predicted rate doubled | All six settings fail: `K/\|F\|` outside the interval |
| The field taken as twice its size | Every predicted rate halved | All six fail, outside the interval and with the lower end above the bound |
| One root too many: `d + 1`, `N + T` | The predicted count off by one | `d = 1` (a factor of 2) and `d = 3` (4/3) fail; `d = 7` (8/7) and LogUp (`1 + 1/K`) pass, finer than the 18% resolution |
| Challenge words masked to 16 bits | `P − 1` never drawn | `d = 1` fails with no accepts, `d = 3` with 139, of 200 expected |
| Challenge words reduced mod `P` | `P − 1` and `P − 2` drawn at half the rate | `d = 1` fails with 104 accepts, `d = 3` with 130, of 200 expected |
| The sumcheck verifier without its final check | Every lie accepted | The quick tests fail: a lie is accepted where `r_0` is no root |
| The LogUp verifier without the roots' balance | Bad lookups accepted wherever `α` is no pole | The quick tests fail: an `α` that is no root is accepted |
| The level verifier without the front-padding scale `2^{n_max − n_i}` | Trees of different heights misbatched | The quick tests fail: honest lookups are rejected, and so are bad ones at their roots |

## Limits

- **Small field only.** The bounds of RFC-0014 'Soundness budget' are stated over `|Fp2| ≈ 2^128`. The calibration checks their form, `K/|F|` with the analysis's `K`, where it can be measured, not the production numbers (#103).
- **One challenge per prover.** Batching over `γ` (`m/|F|`), forests' root claims (`s/|F|`), the compression of tuples by `β`, zero-check points and `tessera-protocol/2`'s masked identity each add terms of their own, which this harness does not measure. A prover that lies in every round reaches about `n·d/|F|`, the union bound.
- **Other instances.** Random bad lookups, which accept far below the bound, are measured by `tessera-logup`'s own calibration test ([`crates/tessera-logup/tests/calibration.rs`](../../crates/tessera-logup/tests/calibration.rs)): 4·10^6 trials, which the nightly job's second step runs.
