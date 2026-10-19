# EXP-0009: Does a forest prover that answers the power-of-two padding analytically reproduce today's padded transcript bit for bit and recover the padding factor in wall clock?

| | |
|---|---|
| Status | `done: inconclusive` (run 2026-10-16: every arm bit-identical to the production prover and the padded-versus-analytic time ratio within 10% of the padding factor on all four judged configurations, but the candidate's cost per real leaf exceeds the unpadded forest's by 9.5–14.4% on the three ragged-row configurations against the pre-registered 5%) |
| Idea | [IDEA-0019](../../ideas/IDEA-0019-analytic-padding-blocks.md) |
| Evidence level | E1 (CPU wall clock here, one Apple M4 Max performance core) for the time ratios and the bit-identity; E0 for the padding-factor table and the cost-model translation; whether GPU tiles align with the blocks stays E2 |
| Hardware | Apple M4 Max (12 performance cores, 128 KiB L1D and 16 MiB L2 per performance cluster), macOS 26.6.2 |
| Pre-registered on | 2026-10-16 |

## Hypothesis

In one harness on one M4 Max performance core, a product-forest prover that never materializes, builds, binds or streams a padded leaf or node, and instead adds each padded block's contribution to every round message in closed form (the padded leaves of a `SoftmaxRow` row are the public constant 1 of issue #247, `leaf(q, i) = C(q, i)·k_qi + (1 − C(q, i))`; a padded tree of [RFC-0010 'Tree levels'](../../../docs/rfcs/RFC-0010-proof-schedule.md#tree-levels) is a tree of ones with the public root 1), produces round messages, final evaluations, challenges and leaf claims bit-identical to today's `prove_forests` of `crates/tessera-sumcheck` on the materialized power-of-two slice, and is faster than the same loops over the materialized padding by the padding factor: about 2.0× for causal rows at one height, 1.33× for rows grouped by height, 1.6× for the 70B forest shape and 2.13× for both paddings together, each within 10%, with a cost per real leaf within 5% of an unpadded forest's cost per leaf.

## Success criterion

Written before any code. Five arms run in the same program on the same seeded inputs; two carry the verdict, one anchors them to the production code, two give context.

Arms:

- `production` (anchor and correctness reference): `tessera_sumcheck::prove_forests` on `Forest::new(padded_slice, h)` per forest, the code as it is today, which takes a materialized power-of-two leaf slice and builds every level over it.
- `dense` (baseline of the verdict): the harness's own forest prover with every leaf declared real: the same algorithm as the production prover (levels built bottom-up as even and odd halves, one `eq` table per instance per level, the first own round in the prime subfield, `γ`-weighted and front-padded batching with the padding rounds' `2^{n_max − n_i − j − 1}·S_i`, three message nodes `p(0), p(2), p(3)`, folds after every round), written with a loop per tree so that the candidate and the baseline share every inner loop. It must reproduce `production`'s output bit for bit.
- `analytic` (candidate of the verdict): the same prover told which leaves are real. Per tree and per level only the real prefix of each half is built, materialized, bound and folded (`⌈R/2⌉` even and `⌊R/2⌋` odd entries for `R` real nodes, `R_t = ⌈R_{t+1}/2⌉`); the `eq` table is materialized over the real prefix of each tree only (`eq(ρ, (x, y)) = eq_low(x)·eq_high(y)`, one multiplication per real entry, folded like the other tables); a pair with one real and one padded entry is formed with the constant; the fully padded pairs of a tree contribute `κ²·C_j·eq(ρ_j, z)·TAIL(ρ_{j+1..}, P)` in round `j`, with `TAIL(ρ, P) = Σ_{x ≥ P} eq(ρ, x) = 1 − Σ_{i: P_i = 1} (1 − ρ_i)·Π_{l > i} eq(ρ_l, P_l)` (the bit recursion of [RFC-0008 'Structured evaluation'](../../../docs/rfcs/RFC-0008-logup-and-tables.md), `O(n)` per block, memoized per distinct `P` per round); the padded trees of a forest contribute `κ²·C_j·eq(ρ_j, z)·TAIL(ρ_tree, N_trees)` in the position rounds and are the constant tail of the tree-level tables in the tree rounds. `κ = 1` at every level. Nothing in the transcript's labels or order changes.
- `dense-shared` and `analytic-shared` (context): the same two provers with the `eq` table not materialized per tree but read from the shared `eq_low` table of the level and scaled by `eq_high(y)` per tree after the sum (IDEA-0010's mechanism 1, which the idea note says this idea composes with). Not judged.

Configurations, each a batch of forests proven in one `prove_forests` call, with the padding factor `f = padded leaves / real leaves`:

| Configuration | Shape | Padded leaves | Real leaves | `f` |
|---|---|---|---|---|
| `S1` one height | `T = 4096` causal rows `q = 0 … T − 1`, one forest of `2^12` trees of height 12, row `q` has `q + 1` real leaves and `T − q − 1` ones | 16,777,216 | 8,390,656 | 1.99951 |
| `S2` grouped by height | the same rows in forests of heights 1 … 12: height 1 holds rows 0 and 1 (2 trees), height `h ≥ 2` the `2^{h−1}` rows with `2^{h−1} < q + 1 ≤ 2^h` | 11,184,812 | 8,390,656 | 1.33301 |
| `S3` 70B forest | 640 = 1.25 × 2^9 full trees of `2^14` leaves padded to `2^10` trees of ones (RFC-0020's row height) | 16,777,216 | 10,485,760 | 1.60000 |
| `S4` 70B rows grouped | `T = 2048` causal rows for 5 = 1.25 × 4 head-layer pairs, grouped by height as `S2`, so each height's tree count `5·2^{h−1}` pads to `2^{h+2}` | 22,369,632 | 10,490,880 | 2.13229 |
| `U23` unpadded | `2^9` full trees of `2^14` leaves (the real count of `S1`/`S2` within 0.03%) | 8,388,608 | 8,388,608 | 1 |
| `U24` unpadded (context) | `2^10` full trees of `2^14` leaves | 16,777,216 | 16,777,216 | 1 |
| `S1-1K`, `S2-1K` (context) | `S1` and `S2` at `T = 1024` | 1,048,576 / 699,052 | 524,800 / 524,800 | 1.99805 / 1.33204 |

Every real tree has one zero leaf at a seeded real position, as `SoftmaxRow` rows have (the key `c_q − x_qi` vanishes at the maximum), so every real root is 0 and every padded root is 1. `count.py` prints the exact integers of this table.

Quantities, medians over the measured iterations:

- `R_same(c) = T_dense(c) / T_analytic(c)`, the padded-versus-analytic time ratio of the idea note's criterion, in the same code.
- `R_prod(c) = T_production(c) / T_analytic(c)`, against today's prover.
- `a(c)`, the candidate's nanoseconds per real leaf on `c`; `u`, the `dense` arm's nanoseconds per leaf on `U23`.

Verdicts:

- `validated`: for every `c ∈ {S1, S2, S3, S4}`, `0.90 ≤ R_same(c) / f_c ≤ 1.10` and `a(c) / u ≤ 1.05`; every arm's output equals `production`'s bit for bit on every configuration; every coefficient of variation on the arms a ratio needs (`dense(c)`, `analytic(c)`, `dense(U23)`) is at most 3% after at most one rerun.
- `refuted`: `R_same(c) ≤ 1 + (f_c − 1)/2` for some `c ∈ {S1, S2, S3, S4}` (less than half of the padding's cost is recovered: at most 1.4998, 1.1665, 1.3000, 1.5661 for `S1 … S4`).
- `inconclusive`: anything else: a ratio inside `(1 + (f_c − 1)/2, 0.90·f_c)` or above `1.10·f_c` for some `c`; `a(c) / u > 1.05` for some `c`; or a coefficient of variation above 3% after the rerun on an arm a ratio needs.

A candidate whose proofs, claims or transcript state differ from `production`'s is a defect in the harness, fixed and rerun before any number is read (the skill's rules of evidence), never a verdict.

Secondary quantities, reported but not judged: `R_prod(c)` and `T_production / T_dense` per configuration (the loop-quality factor between the production kernels and the harness's loops; expected within ±10%); the two shared-`eq` arms and `T_dense-shared / T_analytic-shared`; the context configurations; the E0 table of padding factors per table instance and per `SoftmaxRow` forest for 405B and 70B at 1K, 4K and 16K tokens (`count.py`); and the cost-model translation under the `counted` profile with the measured ratios in place of the idea note's arithmetic factors: `leaf_ps = 73 × R_same(S2)` with `lookups_per_token = 3.1e7` (405B lower bound), `leaf_ps = 73 × R_same(S1)` with the same (405B upper bound), and `leaf_ps = 73 × R_same(S4)` with `lookup_ps = 161` (70B proxy), against the baseline `leaf_ps = 73`.

Nothing above this line is edited after pre-registration; amendments go below it with a date.

## Method

- **Harness.** One Rust program, `padblocks`, its own Cargo workspace in this directory (empty `[workspace]` table), `#![forbid(unsafe_code)]`, depending by path on `tessera-sumcheck` (the production `prove_forests`, the `Prefix` labels and the `SumcheckTranscript` trait), `tessera-poly` (`eq_table`, `PolyField`), `tessera-field`, `tessera-hash` (RFC-0004's transcript) and `tessera-codec`, and on `tv-core`, `tv-field` and `tv-hash` for the independent checks (research rule 2: this crate is a dependency of nothing). The harness's prover drives the production `Transcript` through the `SumcheckTranscript` trait with the production labels (`tree/{ℓ}/root/{c}`, `tree/{ℓ}/gamma`, `tree/{ℓ}/round/{j}`, `tree/{ℓ}/r/{j}`, `tree/{ℓ}/evals`, `tree/{ℓ}/line`), so that a message that differs by one bit changes every later challenge and the final state.
- **Inputs.** Synthetic, SplitMix64: the real leaves of every tree from seed `0x7E55_E7A0_0009_0001`, each word reduced into `Fp` by `from_u64_reduced`, with the zero leaf's position drawn from the same stream; the hand case and the `TAIL` check from `0x7E55_E7A0_0009_0002`. The session context is the bench's (`tessera-protocol/1`, non-interactive, `test-small/1`, statement digest `0x5a` × 32, session id `7` × 16). The `analytic` arm reads the real prefix of each tree from the same padded slice the other arms take (a view, no copy), together with the per-tree real counts.
- **Protocol** as RFC-0020's gate and EXP-0006/EXP-0007: 3 warm-ups and 10 measured iterations per arm and configuration, one full `prove_forests` of the configuration on a fresh transcript per iteration (level build, `eq` tables and all levels inside the timed region, as in the production function); median, minimum, maximum and coefficient of variation; a coefficient above 3% reruns the arm once and the rerun is reported, flagged if still above. Times are reported in milliseconds per proof and in nanoseconds per real leaf and per padded leaf.
- **Correctness checks**, each aborting the run on failure:
  - the hand case: two trees of height 2, rows `[2, 3]` and `[5, 7, 11]`, padded slice `[2, 3, 1, 1, 5, 7, 11, 1]`: the production proof is accepted by the `tv-core` driver of EXP-0006 (written around `tv_core::tree::verify_product_level`) from the public roots `(6, 385)` and rejected against `(6, 386)`; the `analytic` and `dense` arms equal `production`; and the roots with the constant 1 are the products of the real leaves while a padding of 0 would make every root 0 (the soundness point of #247);
  - `TAIL(ρ, P)` equals the brute-force suffix sum of `eq_table(ρ)` for every `P` at 6 coordinates;
  - per configuration, before timing: the production proof is accepted by the `tv-core` driver from the true roots computed in `tv-field` arithmetic from the padded slice (0 for every real tree, 1 for every padded tree), the two transcripts end in the same state, and every leaf claim equals the padded leaves' multilinear extension computed from its definition in the verifier's field; then every other arm's proofs (`Vec<SumcheckProof>`), leaf claims and transcript state equal `production`'s (structural equality of every field element), and the final transcript state is printed as the configuration's digest.
- **Counts (E0).** `count.py`: the padding factors of the eight RISK-2 table instances per model under two assumptions about shared table definitions (one instance per rescaled tensor; norms, residual rescales and RoPE Q+K each sharing one definition), the `Exp` instance, and the `SoftmaxRow` forest under two groupings (one height; grouped by height), for 405B (126 layers, 128 heads, hidden 16,384, KV 1,024, intermediate 53,248) and 70B (80 layers, 64 heads, hidden 8,192, KV 1,024, intermediate 28,672) at 1K, 4K and 16K tokens, as exact integers; and the exact leaf counts of the harness's configurations above.

## Reproduction

```sh
# from the repository root
cd research/experiments/EXP-0009-analytic-padding-blocks
cargo run --release --locked -- all results/run.json | tee results/run.txt   # first run without --locked writes Cargo.lock
python3 -I count.py | tee results/count.txt
python3 -I ../../tools/cost_model.py --profile counted | tee results/cost_model_baseline.txt
python3 -I ../../tools/cost_model.py --profile counted --set lookups_per_token=3.1e7 --set leaf_ps=<73 x R_same(S2)> | tee results/cost_model_405b_grouped.txt
python3 -I ../../tools/cost_model.py --profile counted --set lookups_per_token=3.1e7 --set leaf_ps=<73 x R_same(S1)> | tee results/cost_model_405b_one_height.txt
python3 -I ../../tools/cost_model.py --profile counted --set lookup_ps=161 --set leaf_ps=<73 x R_same(S4)> | tee results/cost_model_70b_grouped.txt
shasum -a 256 README.md                                   # the pre-registered text: results/preregistration.txt
```

Environment: Apple M4 Max, 16 cores (12 performance, 4 efficiency), 48 GiB; macOS 26.6.2 (Darwin 25.6.0); rustc 1.95.0 (59807616e 2026-04-14), cargo 1.95.0; Python 3.14.6; 2026-10-16. One process, one thread, no other load; macOS schedules performance cores first.

## Results

Every correctness check passed before any arm was timed: `TAIL(ρ, P)` equals the suffix sum of `eq_table(ρ)` for all 65 values of `P` at 6 coordinates; the hand case `[2, 3 | 1, 1] [5, 7, 11 | 1]` is accepted by the `tv-core` driver from the roots `(6, 385)` and rejected against `(6, 386)` (`FinalMismatch { round: 1 }` at level 0), all four harness arms equal the production on it, and a padding of 0 gives the roots `(0, 0)` (the second witness #247 closes); on five small ragged shapes (rows of one leaf, trees with a single real node at a level, padded trees, forests joining at different levels, a single tree) and on all eight configurations, the production proof is accepted by the driver from the true roots (0 for every real tree, 1 for every padded tree), the transcripts end in the same state, every leaf claim equals the padded leaves' multilinear extension, and the four harness arms reproduce the production's round messages, final evaluations, leaf claims and transcript state bit for bit (the configurations' digests are in [results/run.txt](results/run.txt) and [results/run.json](results/run.json)). No arm a ratio needs exceeded a coefficient of variation of 0.93%, so no rerun was needed; the one flag of the run (`analytic` on the context configuration `S2-1K`, 3.35% after its rerun, a 22 ms case) touches no judged number. Raw output: [results/run.txt](results/run.txt), [results/run.json](results/run.json) (the pre-registered run), [results/run2.txt](results/run2.txt), [results/run2.json](results/run2.json) (the reproduction run), [results/count.txt](results/count.txt), [results/preregistration.txt](results/preregistration.txt) and the cost-model files below.

**The verdict's table** (E1, primary; one performance core; medians of 10 after 3 warm-ups, milliseconds per proof of the whole configuration, nanoseconds per leaf and the coefficient of variation; `u` = 36.250 ns per leaf, the `dense` arm on `U23`):

| Case | Baseline: `dense` (ms; ns per padded leaf; CV) | Candidate: `analytic` (ms; ns per real leaf; CV) | `R_same` (`R_same / f`); `a(c) / u` |
|---|---|---|---|
| `S1` one height, `f` = 1.9995 | 627.2 (37.39; 0.26%) | 342.4 (40.81; 0.28%) | **1.832** (0.916); 1.126 |
| `S2` grouped by height, `f` = 1.3330 | 419.6 (37.52; 0.24%) | 333.2 (39.71; 0.44%) | **1.259** (0.945); 1.095 |
| `S3` 70B forest, `f` = 1.6000 | 608.0 (36.24; 0.40%) | 380.9 (36.33; 0.33%) | **1.596** (0.997); 1.002 |
| `S4` 70B rows grouped, `f` = 2.1323 | 860.7 (38.47; 0.20%) | 435.0 (41.47; 0.61%) | **1.978** (0.928); 1.144 |
| `U23` unpadded, `f` = 1 | 304.1 (36.25; 0.16%) | 305.0 (36.36; 0.93%) | 0.997 (0.997); 1.003 |

The share of the padding's cost the candidate recovers, `(R_same − 1) / (f − 1)`: 83% (`S1`), 78% (`S2`), 99% (`S3`), 86% (`S4`).

**Against today's prover** (E1, primary; the production `prove_forests` on the materialized slices):

| Configuration | `production` (ms; ns per padded leaf; CV) | `R_prod` = production / analytic | production / dense |
|---|---|---|---|
| `S1` | 541.5 (32.28; 0.66%) | 1.581 | 0.863 |
| `S2` | 361.3 (32.30; 0.22%) | 1.084 | 0.861 |
| `S3` | 540.6 (32.22; 0.29%) | 1.419 | 0.889 |
| `S4` | 721.3 (32.25; 0.21%) | 1.658 | 0.838 |
| `U23` | 270.6 (32.25; 0.49%) | 0.887 | 0.890 |
| `U24` | 543.3 (32.38; 0.67%) | 0.891 | 0.891 |

**The shared-`eq` arms** (context, E1; the `eq` table read from the level's shared position table and scaled per tree, IDEA-0010's mechanism 1):

| Configuration | `dense-shared` (ms; ns per padded leaf) | `analytic-shared` (ms; ns per real leaf) | dense-shared / analytic-shared | production / analytic-shared |
|---|---|---|---|---|
| `S1` | 486.4 (28.99) | 264.0 (31.47) | 1.842 | 2.051 |
| `S2` | 326.0 (29.15) | 262.3 (31.27) | 1.243 | 1.377 |
| `S3` | 473.5 (28.22) | 295.1 (28.14) | 1.605 | 1.832 |
| `S4` | 667.7 (29.85) | 341.2 (32.52) | 1.957 | 2.114 |
| `U23` | 235.7 (28.09) | 235.7 (28.10) | 1.000 | 1.148 |

**Context sizes** (E1): `U24` dense 609.96 ms (36.36 ns per leaf), analytic 609.87 (36.35): the per-leaf cost of both arms is flat from 2^23 to 2^24 leaves. `S1-1K`: production 33.82 ms, dense 39.31 (37.49 ns per padded leaf), analytic 22.70 (43.25 ns per real leaf), `R_same` 1.732 (0.867 `f`), `a / u` 1.193. `S2-1K`: production 23.16, dense 27.06 (38.71), analytic 22.20 (42.30; CV 3.35%, flagged), `R_same` 1.219 (0.915 `f`), `a / u` 1.167.

**Reproduction run** ([results/run2.txt](results/run2.txt), [results/run2.json](results/run2.json); the same binary and inputs, run immediately after the first): `R_same` 1.834 / 1.277 / 1.556 / 1.959 on `S1 … S4` (0.917 / 0.958 / 0.973 / 0.919 of `f`), `a(c) / u` 1.137 / 1.101 / 1.030 / 1.155 with `u` = 36.267 ns, `R_prod` 1.568 / 1.081 / 1.394 / 1.639, the production prover at 32.20–32.75 ns per padded leaf across the eight configurations, the shared-`eq` ratios 1.829 / 1.243 / 1.594 / 1.970; every arm again bit-identical to the production and every production proof accepted. Four arms were rerun under the 3% rule (first-pass coefficients above 3% on `S1` dense-shared and analytic-shared and on `S3` dense and dense-shared) and every reported coefficient is at most 2.90%. The medians agree with the pre-registered run within 1.1% on every arm but `S2` dense (+1.9%) and `S3` analytic (+2.8%, 391.7 against 380.9 ms). The rule returns `inconclusive` on this run too; the pre-registered run decides. A third run of the committed source ([results/run3.txt](results/run3.txt), [results/run3.json](results/run3.json); the harness as checked in, formatted and clippy-clean after the two runs with no semantic change) reproduces all thirteen transcript digests of the pre-registered run exactly, with `R_same` 1.835 / 1.249 / 1.596 / 1.976 and `a(c) / u` 1.121 / 1.100 / 1.000 / 1.140.

**Padding factors** (E0, primary, [count.py](count.py), exact; [results/count.txt](results/count.txt)): at 405B the five power-of-two tensors and the two 1,024-wide ones pad by 1.0159 (126 layers = 63/64 × 2^7), SiLU (53,248 = 13 × 2^12) by 1.2503, a shared RoPE Q+K definition (17,408) by 1.9122, all lookups weighted by 1.1068 per tensor and 1.2206 with shared definitions, the `Exp` instance by 1.0149–1.0158, the `SoftmaxRow` forest by 2.0298–2.0316 at one height and 1.3532–1.3544 grouped by height (`T` = 1K–16K); at 70B (80 layers = 5/8 × 2^7) every tensor pads by 1.6000 (SiLU 1.8286; weighted 1.6914 per tensor, 1.6686 shared), the `Exp` instance by 1.5984–1.5999, the `SoftmaxRow` forest by 3.1969–3.1998 at one height and 2.1313–2.1332 grouped. The model-independent row factor `Σ_q 2^⌈log2(q+1)⌉ / (T(T+1)/2)` is 1.33204 / 1.33301 / 1.33325 at 1K / 4K / 16K (one height: 1.99805 / 1.99951 / 1.99988), the idea note's 4/3 and 2.

**Cost model** (E0, `tools/cost_model.py`, `counted` profile, `leaf_ps = 73 × R_same` in place of the note's arithmetic factors; files [results/cost_model_baseline.txt](results/cost_model_baseline.txt), [results/cost_model_405b_grouped.txt](results/cost_model_405b_grouped.txt), [results/cost_model_405b_one_height.txt](results/cost_model_405b_one_height.txt), [results/cost_model_70b_grouped.txt](results/cost_model_70b_grouped.txt)):

| Scenario | `leaf_ps` | μ(1K) | μ(4K) | μ(16K) | μ(128K) | Padded prover over the baseline | The idea's gain, `−Δ/(1 + Δ)` |
|---|---|---|---|---|---|---|---|
| Baseline (`counted`) | 73 | 17.2 | 29.8 | 78.6 | 373.3 | — | — |
| 405B, rows grouped by height (`lookups_per_token` 3.1e7) | 91.9 (73 × 1.2595) | 18.0 | 31.4 | 83.0 | 393.8 | +4.7% / +5.3% / +5.6% / +5.5% | −4.5% / −5.0% / **−5.3%** / −5.2% |
| 405B, one height (`lookups_per_token` 3.1e7) | 133.7 (73 × 1.8316) | 18.6 | 33.7 | 91.6 | 438.4 | +8.1% / +12.9% / +16.5% / +17.4% | −7.5% / −11.4% / **−14.2%** / −14.8% |
| 70B proxy, rows grouped (`lookup_ps` 161) | 144.4 (73 × 1.9784) | 21.8 | 39.7 | 108.2 | 515.2 | +26.8% / +33.2% / +37.6% / +38.0% | −21.1% / −24.9% / **−27.3%** / −27.5% |

The idea note's arithmetic figures were −6.4% / −16.4% / −28.7% at 16K; the measured ratios give −5.3% / −14.2% / −27.3%, the difference being the per-tree overhead described below. The lookup side of each scenario (SiLU's 1.25, the 70B 1.6 on `c_ℓ`) is the note's E0 arithmetic, not measured here.

## Analysis

**Against the criterion.** Bit-identity holds on every configuration and arm, so the closed-form blocks compute exactly what the materialized padding computes, including the padded rounds' `S_i` and the leaf claims over the padded leaves. The time-ratio clause passes with margin on all four judged configurations: `R_same / f` is 0.916, 0.945, 0.997 and 0.928, inside `[0.90, 1.10]`, and the refutation thresholds (1.4998, 1.1665, 1.3000, 1.5661) are cleared by 0.33, 0.09, 0.30 and 0.41. The per-real-leaf clause fails on the three ragged-row configurations: `a(c) / u` is 1.126 (`S1`), 1.095 (`S2`) and 1.144 (`S4`) against the pre-registered 1.05, and passes only on the forest-padding configuration `S3` (1.002). Every coefficient of variation the rule needs is at most 0.93%. The pre-registered rule returns `inconclusive`.

**Where the cost per real leaf goes.** `S3` isolates the closed-form machinery: its real trees are full, its padding is 384 whole trees answered by one `TAIL` term per round, and the candidate's cost per real leaf equals the unpadded forest's within 0.2% while recovering 99% of the padding's cost. The ragged-row configurations add a per-tree, per-round constant: every real tree is its own group with its own loop over a short real prefix, about one pair in two straddles the real boundary and is formed by the scalar path, the three prefixes fold an odd tail about every other round, and each group pays the closed-form term and its bookkeeping. The excess over `u` divides evenly by the number of (tree, position-round) pairs: `S1` 38.3 ms over 270,336 of them (4,096 trees × Σ_{t=1}^{12} (t − 1)), 142 ns each; `S2` 29.0 ms over 229,374, 126 ns; `S4` 54.7 ms over 471,030, 116 ns; `S1-1K` 3.7 ms over 46,080, 80 ns. About 100 ns per tree per round, some thirty `Fp2` operations' worth of loop start-up, scalar boundary pairs and folds, is the whole discrepancy, and it is amortized over the tree's real entries: 4.6 ns per real leaf at `T = 4096` (`S1`), 7.0 ns at `T = 1024`. Extrapolated at the measured constant, `S1` at `T = 16384` (16,384 trees, 91 position rounds each, 134 million real leaves) would carry 1.5 million tree-rounds, about 210 ms over 4.87 s, `a / u ≈ 1.04` (E0, from the E1 constant): the clause is scale-dependent and would pass at the contexts where the attention term dominates `μ`, but that is an extrapolation, not a measurement.

**Against today's prover.** The production kernels are 11–14% faster per leaf than the harness's per-tree loops (`production / dense` 0.84–0.89; `U23` 32.25 against 36.25 ns per leaf, the production figure agreeing with EXP-0006's 31.2–31.7 and the bench's 31.9 within 3%), so against today's code the candidate is 1.58× (`S1`), 1.08× (`S2`), 1.42× (`S3`) and 1.66× (`S4`) faster, and an implementation of the blocks inside the production kernels would land at `R_same`, 1.26–1.98×, since the harness's two arms share every inner loop. The shared-`eq` arms give the same picture (dense-shared / analytic-shared 1.842, 1.243, 1.605, 1.957, that is 0.92–1.00 of `f`) and show the composition the note claims with IDEA-0010's mechanism 1: reading `eq` from the level's shared position table saves 22% of the dense leaf cost (28.09 against 36.25 ns, matching EXP-0006's count of the `eq` build and its folds at 6.9 of 31.1 ns), and `analytic-shared` is 2.05× (`S1`), 1.38× (`S2`), 1.83× (`S3`) and 2.11× (`S4`) faster than today's production prover.

**What the E0 table adds.** The padding factors are as the idea note computed them, now as exact integers: the `SoftmaxRow` forest is the largest multiplier at every model size (1.35 grouped, 2.03 at one height at 405B; 2.13 and 3.20 at 70B, where every instance also pads by 1.6), the lookup instances at 405B pad by 1.6% except SiLU (1.25) and a shared RoPE Q+K definition (1.91), and the `Exp` instance by 1.5%. Which of 1.35 and 2.03 applies is the schedule sentence #247 left to #122.

Threats to validity:

- **CPU against GPU, the main one.** On an H100 the padded blocks of a level coincide with whole tiles or not (RFC-0018 'Kernels'); a block boundary inside a tile costs a partial tile, and a ragged row inside a tile row costs a partly idle warp per round, which is the GPU form of the per-tree constant measured here. Whether the saving is the full factor or the factor less partial tiles is E2. The per-tree constant itself (short scalar loops, boundary pairs) is a CPU loop-structure effect and does not transfer as a number.
- **Loop quality.** The verdict's ratio is same-code by construction; the harness's loops are 11–14% slower per leaf than the production kernels, so `R_prod` understates what the blocks inside the production kernels would give, and `R_same` is the estimate that carries.
- **Operation counts against wall clock.** No count model is involved: the candidate's work per real entry equals the baseline's by construction, and `R_same` measures the padding's share of wall clock less the per-tree overhead. The overhead is characterized by one constant fitted to four configurations (80–142 ns per tree-round), not by a count.
- **Cache.** The dense arm's tables are 1.33–2.13× larger than the candidate's and both exceed the 16 MiB L2 at the leaf levels; the per-leaf cost of every arm is flat from 2^23 to 2^24 leaves (`U23` against `U24`, within 0.3%), and EXP-0006 found no size dependence from 2^18 to 2^22, so the ratios are not cache artefacts. The candidate's packed layout streams fewer bytes, in its favour; the measured overhead is despite it.
- **Input distribution.** Control flow depends on shapes only (INV-ARCH-2): the real counts `q + 1`, the tree counts and the heights, which are the specification's causal rows and RFC-0010's forest rule, not a sampling choice; the one zero leaf per row is a value and costs nothing. Two shape choices are the experiment's: rows grouped by height against one height (the schedule sentence RFC-0009 and RFC-0010 do not write; both measured), and the 70B head-pair count scaled from 5,120 = 1.25 × 2^12 to 5 = 1.25 × 4 in `S4`, which keeps the tree-padding factor at exactly 1.6 but shortens the forests.
- **The `eq` materialization.** The candidate materializes `eq` over each tree's real prefix so that its work per real entry equals the production's; the shared variant that does not materialize it gives the same ratios (0.92–1.00 of `f`), so the conclusion does not rest on that choice.
- **One machine, one day, one toolchain.** The reproduction run is reported below the pre-registered run's tables.

## Verdict

`inconclusive`, by the letter of the pre-registered criterion: the mechanism is confirmed (every arm bit-identical to today's prover on every configuration, and the padded-versus-analytic time ratio within 10% of the padding factor on all four judged configurations, 0.92–1.00 of `f`, recovering 78–99% of the padding's cost), but the hypothesis's second number, a cost per real leaf within 5% of an unpadded forest's, is missed by 4.5–9.4 points on the three ragged-row configurations at `T ≤ 4096` by this implementation's per-tree constant of about 100 ns per tree per round, and met only where the padding is whole trees. IDEA-0019 moves from `triaged` to `inconclusive`. Its score changes to: gain −5.3% μ(16K) as the lower bound (rows grouped by height, `counted`, 405B; −14.2% at one height; −27.3% on the 70B grouped shape), from the measured `R_same` in place of the arithmetic 4/3, 2 and 2.13 (E0 from E1 ratios); confidence 0.6 → 0.7 (the multiplier is measured rather than inferred from the specification, the transcript is proven bit-identical, and the residual uncertainty is the GPU tile alignment and whether the per-tree overhead can be amortized, both E2 or a short E1); cost `s` unchanged, plus the amortization work the follow-up names; priority 5.3 × 0.7 / 1 = 3.7 (9.9 at one height; 19 on the 70B model). Next action: the amortized ragged loop of the first follow-up as a dated amendment to this experiment, then `promote` as a kernel note for RFC-0018 'Kernels' (tile skipping with the closed form) and the `CommitmentReport` counts of RFC-0006.

## Follow-ups

- **Amortize the per-tree constant (E1, `s`).** Rows of one height class sorted by real length put equal-length trees side by side: consecutive trees then share `P`, the `TAIL` term and the boundary pair's position, so a run of equal-length trees is one loop over a strided block with one boundary column, and the odd-tail folds vectorize across the run. Pre-register the same criterion with the amended loop; the E0 extrapolation above says the 5% clause then passes at every `T`, which the amendment would turn into a measurement.
- **E2 (RFC-0020 runner).** The `c_π` microbenchmark in the `SoftmaxRow` shape (causal rows grouped by height) with the padded blocks skipped per tile: the pre-registered prediction from this report is a ratio of 1.26–1.33 against the padded kernel for rows grouped by height and 1.83–2.0 at one height, less the partial tiles at the ragged boundary; a measurement inside it settles whether the GPU saving is the full factor.
- **The schedule sentence (#122).** The grouping decides 1.35 against 2.03 (405B) and 2.13 against 3.20 (70B) in the E0 table; the one-height layout costs 1.5× the grouped one in product-leaf work at every model size (E0, exact) and 1.50× here (E1, `S1` against `S2`, production: 541.5 against 361.3 ms), so the grouped layout should be the schedule's default unless the extra levels' round count argues otherwise.
- **`CommitmentReport` (RFC-0006).** Padded against real leaves per table instance and per forest, the additive field the idea note proposes, so that the cost model's `K_lin` and `A` carry the multipliers `count.py` computes by hand.
- **IDEA-0010, mechanism 1.** The shared-`eq` arms quantify it on the forest prover at 22% of the dense leaf cost (E1) against EXP-0006's 7.5% on the fraction proxy; an issue against `tessera-sumcheck` regardless of this idea.
- **The fraction-tree padding (mechanism item 1 of the note)** was not measured: its blocks carry the level constant `κ_m = (2^m·n·d^{2^m − 1}, d^{2^m})` instead of 1, which the harness's per-table constant supports but its instances (two halves, one `eq`) do not; EXP-0006's fraction proxy is the shape for it.


## Amendment 2026-10-19: amortized ragged loop

Status: `planned`. The original hypothesis, criterion, methods, results and verdict above remain the historical experiment. This amendment tests a new `analytic-amortized` arm; the four original harness arms and the production anchor remain runnable without selecting the new arm.

### Hypothesis and unchanged criterion

Sorting the processing order of each height class by real prefix length makes equal-length trees adjacent. Sharing their prefix/boundary metadata and closed-form `TAIL` contribution, and folding odd tails together across such runs, removes enough per-tree bookkeeping that `analytic-amortized` satisfies both original clauses on S1, S2, S3 and S4. Tree indices in the transcript and the tree-coordinate polynomial keep their original order; sorting is an internal traversal permutation, reversed before the tree rounds.

The baseline remains `dense`; only the candidate is replaced by `analytic-amortized` in `R_same`, `R_prod` and `a(c)`. The unpadded reference remains `u = dense(U23)` nanoseconds per leaf. **Validated** means every judged configuration has `0.90 <= R_same / f <= 1.10` and `a(c) / u <= 1.05`, every proof/message/evaluation/leaf claim and final transcript state equals production exactly, and all arms needed for those ratios have CV at most 3% after at most one rerun. The original refutation thresholds `R_same <= 1 + (f - 1)/2` remain reported; a failed or noisy amendment leaves EXP-0009 and IDEA-0019 inconclusive, as the handover specifies. A mismatched output is a harness defect to fix before judging, never a performance result. No threshold, configuration or seed is selected after measurement.

### Method and reproduction plan

The new arm retains the original materialized per-tree `eq` prefixes (it does not silently substitute the contextual shared-`eq` optimization). At each position round, groups are traversed in runs with equal current even, odd and `eq` prefix lengths. One run shares its real-pair count, boundary column and `TAIL` coefficient. The fully padded contribution is accumulated with the sum of its tree scales. Full pairs retain the original arithmetic; odd final folds are traversed across a run with its parity decided outside that traversal. The original order is used again when assembling the tree-coordinate tables. Sorting, metadata construction, allocations, level building and folds remain inside the timed proof.

Inputs and session context are exactly those above: SplitMix64 leaves seed `0x7E55_E7A0_0009_0001`, checks seed `0x7E55_E7A0_0009_0002`; S1-S4 and U23 decide the result, with U24 and the two 1K cases as context. All original hand cases, small ragged shapes, independent `tv-core` acceptance and direct leaf-MLE checks run before timing. The new arm must equal the production proof and digest on all these shapes. Extra repeated-length and unsorted real-length shapes exercise traversal permutation and odd tails.

The protocol is 3 warmups and 10 measured samples per arm, median/minimum/maximum/CV, rerunning an arm once if CV exceeds 3%; each sample repeats the complete fresh-transcript proof until elapsed time is at least 20 ms, reporting time per proof. Raw sample durations and repetition counts, including a first attempt that triggers a rerun, are retained. Only one experiment benchmarks at a time. CPU/OS/toolchain/date/git SHA are captured with the raw output. Evidence is E1 CPU wall clock and exact transcript agreement; `count.py --amendment` will emit E0 counts of tree-rounds, equal-length runs, odd-tail folds and shared tail terms for all configurations. No GPU number is inferred.

Commands from this directory (after committing this preregistration):

```sh
cargo test --locked
cargo clippy --all-targets --locked -- -D warnings
cargo fmt --all --check
cargo run --release --locked -- amendment results/amendment-run.json > results/amendment-run.txt
python3 -I count.py --amendment > results/amendment-count.txt
```

The `all` command preserves the original arms. The new `amendment` command includes all original arms plus `analytic-amortized`, applies the unchanged criterion to the new arm, and writes separate raw files. H100 follow-up: run the same causal-row shapes under RFC-0020 with block-skipping and grouped odd boundary columns, to establish whether tile alignment and partially idle warps preserve the CPU saving. No production crate, verifier, transcript label or acceptance condition changes.
