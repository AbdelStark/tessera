# EXP-0006: Does an operation count, calibrated on the CPU product-tree prover, exclude the 12 ps assumption for `c_π`?

| | |
|---|---|
| Status | `done: inconclusive` (run 2026-10-16: the pre-registered coefficient-of-variation clause fired on the level-build kernel; every substantive condition of `validated` held) |
| Idea | [IDEA-0008](../../ideas/IDEA-0008-reprice-product-leaf-cost.md) |
| Evidence level | E1 for the fit (CPU wall clock here), E0 for the counts and the H100 interval; the absolute `c_π` stays E2 until the RFC-0020 v0.1 gate |
| Hardware | Apple M4 Max (12 performance cores, 128 KiB L1D and 16 MiB L2 per performance cluster), macOS 26.6.2 |
| Pre-registered on | 2026-10-16 |

## Hypothesis

The operation count of the product-tree leaf stage as specified ([RFC-0010 'Tree levels'](../../../docs/rfcs/RFC-0010-proof-schedule.md#tree-levels): one degree-3 sumcheck of `eq(r, x)·Ṽ(0 ‖ x)·Ṽ(1 ‖ x)` per level, levels halving from the leaves to the roots, one `eq` table per instance per level, every table folded after every round) and as implemented (`prove_forests`, `prove_batch` and the kernel rules of [`crates/tessera-sumcheck/src/kernel.rs`](../../../crates/tessera-sumcheck/src/kernel.rs)), when it is checked against the CPU implementation of those kernels in one harness, places `c_π` on an H100 above the 24 ps trigger of [08 §2](../../../docs/spec/08-performance-budget.md#2-unit-cost-targets) at every point of the counted interval, that is `ρ_π = c_π / 12 ps > 2` even for the lowest count at the integer pipeline's peak, and the same counts explain the measured CPU kernels within a factor 2. IDEA-0008's own figures, written before this experiment: 29–46 Goldilocks multiplications per leaf (E0, counted at four nodes per round) and 31.9 ns per leaf on one M4 Max core (E1).

## Success criterion

Two quantities decide, both defined before any code:

- **The fit `F`.** For each measured kernel, the harness predicts the time per leaf (or per table entry) from unit costs it measures itself on in-cache inputs, times the operation counts of `count.py`; `F` passes when every ratio predicted/measured lies in `[0.5, 2]`. The kernels: `prove_forests` on 2^22 leaves in three shapes (2^8 trees of 2^14 leaves, the row shape of the `c_π` microbenchmark of [RFC-0020](../../../docs/rfcs/RFC-0020-cost-model.md#measurement-protocol-v01-gate); 2^10 trees of 2^12 leaves, RFC-0010's largest test case; one tree of 2^22 leaves); the `eq` tables of every level alone; the level build alone (`V_t[z] = V_{t+1}[2z]·V_{t+1}[2z + 1]` over the whole tree); `prove_batch` of degree 3 on the leaf level's instance alone (`eq` public over `Fp2`, the even and odd halves over `Fp`, 2^21 entries); `prove_batch` of degree 3 on three `Fp2` tables and on three `Fp` tables of 2^21 entries; `bind_lowest` and binding every round over 2^22 entries; and the fraction-tree-shaped proxy defined under Method. The unit costs, each from an in-place loop over 2^12 elements (inside the 128 KiB L1D): nanoseconds per Goldilocks multiplication, per Goldilocks addition, per `Fp2` multiplication (Karatsuba with `mul_small(7)`, as `tessera-field` implements it), per `Fp2`-by-`Fp` multiplication (`mul_by_base`) and per `Fp2` addition. The counts per kernel are the operations of the code as written, read from `forest.rs`, `prover.rs`, `kernel.rs` and `tessera-poly`'s `eq.rs` and `multilinear.rs`: three nodes per round message (`p(0), p(2), p(3)`; `p(1)` is never computed), `k − 1` multiplications per node across `k` tables with the values along the line formed by addition, the prime-subfield first round with `mul_by_base` into the `eq` factor, `bind_base` then in-place folds, one `eq` table per instance per level built by doubling.
- **The interval `I = [lo, hi]`**, in picoseconds per leaf of the whole tree, from `count.py` alone (E0): `lo = max(lowest operation count / 1.67×10^13 op/s, lowest byte count / 3.35×10^12 B/s)` and `hi = max(highest operation count / (0.6 × 1.67×10^13), highest byte count / (0.8 × 3.35×10^12))`. The rates are the H100 SXM int32 peak (132 SMs × 64 lanes × 1.98 GHz) and HBM3 peak, with the 60% integer utilization that [VYTA25] measured for field kernels and 80% of HBM, exactly as [EXP-0002](../EXP-0002-commit-cost-calibration/README.md) and IDEA-0008 step 3 state them. The operation counts are the as-written algorithm's Goldilocks multiplications, additions and subtractions per leaf (the `Fp2` operations expanded into them: a Karatsuba product is 3 multiplications, one `mul_small(7)` and 5 additions or subtractions; `mul_by_base` is 2 multiplications; an `Fp2` addition is 2), translated to int32 instructions per operation with EXP-0002's low (lazy reduction: 18 per multiplication, 4 per addition or subtraction) and high (canonical: 26, 8, 5) figures; `mul_small(7)` counts as a shortened multiplication whose figure `count.py` derives from `goldilocks.rs` (low 8, high 13). The byte counts are a fused kernel (each level's values read once, the rounds and the `eq` table on chip) at the low end and every round through memory with the `eq` table materialized over `Fp2` at the high end.

Verdicts:

- `validated`: `F` passes and `lo > 24 ps` (`ρ_π > 2` at the floor: the "proceed" band of RFC-0020's [decision rule](../../../docs/rfcs/RFC-0020-cost-model.md#decision-rule) is excluded at every point of the interval). The report then names which of RFC-0020's bands the interval covers, `(2, 4]` (24–48 ps) and/or `> 4` (above 48 ps), and whether the SC-2 ceilings ([00 §7](../../../docs/spec/00-overview.md#7-success-criteria)) are crossed: by the cost model (`python3 -I research/tools/cost_model.py --set leaf_ps=…`, corrected profile, run before the measurement), `μ(16K) ≤ 45` needs `c_π ≤ 60 ps` (44.995 at 60.0 ps, 45.036 at 60.2) and `μ(1K) ≤ 12` needs `c_π ≤ 238 ps` (12.015 at 239) (E0, primary).
- `refuted`: `F` fails on any kernel (the counts do not explain the code they were calibrated on, so the projection method is void), or `lo ≤ 12 ps` (the counts admit the assumption).
- `inconclusive`: `F` passes and `lo ∈ (12, 24]` (the assumption is excluded but the trigger is not crossed at the floor), or a kernel of `F` keeps a coefficient of variation above 3% after one repeat.

Two further quantities are pre-registered as observations that the verdict does not depend on: (a) IDEA-0008's single-constant reading, `ns per leaf = κ · (Fp-multiplication equivalents per leaf)` with `κ` taken from the binding kernel (an `Fp2` multiplication counting 3, a `mul_by_base` 2), expected to hold within ±30% for `prove_forests`; (b) the ratio `c_π / c_ℓ` measured in one harness as a shape proxy, product tree per leaf against fraction tree per lookup (Method), expected in `[0.4, 0.8]`; it is a proxy because no LogUp prover exists (`crates/tessera-logup` is a stub) and the fraction-tree levels are built here from the engine's instances. The central value of `I` for the sensitivity run is the geometric mean of `lo` and `hi`, rounded to the nearest picosecond.

Nothing above this line is edited after pre-registration; amendments go below it with a date.

## Method

- **Harness.** One Rust program, `leafcost`, its own Cargo workspace in this directory, depending by path on `tessera-sumcheck`, `tessera-poly`, `tessera-field`, `tessera-hash` and `tessera-codec` for the kernels and on `tv-core`, `tv-field` and `tv-hash` for the independent checks (research rule 2: a research crate is a dependency of nothing, so DEP-2 is untouched). The kernels measured are the production ones, unmodified. There is no candidate implementation: the baseline is the specified kernel, and the "candidate" of the results table is the prediction from counts and unit costs, which the fit `F` judges.
- **Inputs.** Synthetic, from SplitMix64: leaves from seed `0x7E55_E7A0_0058_0001` (the seed of `crates/tessera-sumcheck/benches/throughput.rs` `forests`), every word reduced into `Fp` by `from_u64_reduced`, with one zero leaf per tree at a seeded position, as `SoftmaxRow` trees have (RFC-0009); engine tables, points and the fraction-tree inputs from seeds `0x7E55_E7A0_0058_0002` onward. The session context is the bench's (`tessera-protocol/1`, non-interactive, `test-small/1`, statement digest `0x5a` × 32, session id `7` × 16).
- **Measurements (E1).** Per leaf on one thread: `prove_forests` in the three 2^22-leaf shapes of the criterion, and 2^4 and 2^6 trees of 2^14 leaves (2^18 and 2^20 leaves) for the cache dependence; the `eq` tables of every level alone; the level build alone; `prove_batch` on the leaf level's instance (`eq` public `Fp2` × even × odd, 2^21 entries); `prove_batch` on three `Fp2` and on three `Fp` tables (2^21 entries); `bind_lowest` and binding every round (2^22 entries); the fraction-tree proxy. The unit microbenchmarks listed in the criterion, and a streaming copy of 2^22 `Fp2` elements for the memory-bandwidth context.
- **Fraction-tree proxy** ([RFC-0008 'Fraction tree'](../../../docs/rfcs/RFC-0008-logup-and-tables.md#fraction-tree), the Papini–Haböck level relation): for `N = 2^22` lookups, the leaf level with numerators 1 is the degree-3 batch of the instances `[eq, d_e, d_o]`, `[eq, d_e]`, `[eq, d_o]` on tables of `N/2` entries over `Fp2` (`d = α − key − β·value`, the lookup side of the combined tree), and a generic level is the degree-3 batch of `[eq, n_e, d_o]`, `[eq, n_o, d_e]`, `[eq, d_e, d_o]` on tables of `N/2` entries over `Fp2`; the engine's `γ` weights stand in for the level's combination challenge. The levels above the leaf level sum to one generic level of size `N/2`, so the fraction tree per lookup is `(T_build + T_leaf + T_generic) / N`, `T_build` the leaves `α − key − β·value` and the internal nodes `(n_L d_R + n_R d_L, d_L d_R)` built in `tessera-field` arithmetic by the harness (numerators 1 at the first layer); the product tree per leaf is `prove_forests` on `N` leaves divided by `N`.
- **Protocol** as RFC-0020's gate: 3 warm-up and 10 measured iterations per case; median, minimum, maximum and coefficient of variation; a coefficient above 3% is flagged and that case rerun once. The in-cache microbenchmarks repeat their kernel enough times per iteration that one iteration lasts at least 20 ms.
- **Counts (E0).** `count.py`: per leaf of the whole tree and per entry of each engine kernel, the Goldilocks multiplications, additions and subtractions, `Fp2` multiplications, `Fp2`-by-`Fp` multiplications and `Fp2` additions of the code as written, each rule a line naming the source it translates; the int32 translation per operation, low and high; bytes through HBM per leaf, fused and unfused; the interval `I`; IDEA-0008's multiplication-only count for comparison with its 29–46 and with SUR-0002's `c_ℓ` count (56–74 `Fp` multiplications per lookup, [SUR-0002](../../surveys/SUR-0002-lookups.md) `[E0-this]`).
- **Correctness checks**, all before timing, each aborting the run on failure:
  - field: the products `(P − 1)·(P − 1) = 1`, `2^63 · 2^63 = 0xfffffffec0000001` and `(2^32 − 1) · 2^32 = 0xffffffff00000000`, from `spec/vectors/field/goldilocks.json`;
  - forests: every proven batch is checked by a driver written around the verifier's `tv_core::tree::verify_product_level`, which squeezes the root challenges itself on a `tv-hash` transcript, forms each root claim `Σ_y v_y·eq(r, y)` from the trees' true products computed in `tv-field` arithmetic, carries each next claim to the next level, and checks every leaf claim against the stacked leaves' multilinear extension computed from its definition; the two transcripts must end in the same state; the hand case of a tree of four leaves `[2, 3, 5, 7]` (root 210) is accepted, and the same proof against the public root 211 is rejected, so the check can fail;
  - engine: every `prove_batch` is checked by `tv_core::sumcheck::verify_batch` against claims summed over the hypercube in `tv-field` arithmetic, the public `eq` factor evaluated by the verifier's own `eq`;
  - `eq` tables against the definition at sampled vertices; binding every round against `evaluate_base` at the same point; the level build against the trees' products.

## Reproduction

```sh
# from the repository root
cd research/experiments/EXP-0006-product-leaf-cost
cargo run --release --locked -- all results/run.json | tee results/run.txt   # first run without --locked writes Cargo.lock
python3 -I count.py | tee results/count.txt
python3 -I ../../tools/cost_model.py --set leaf_ps=<lo> | tee results/cost_model_lo.txt
python3 -I ../../tools/cost_model.py --set leaf_ps=<hi> | tee results/cost_model_hi.txt
python3 -I ../../tools/cost_model.py --sensitivity --set leaf_ps=<central> | tee results/cost_model_sensitivity.txt
```

Environment: Apple M4 Max, 16 cores (12 performance, 4 efficiency), 48 GiB; macOS 26.6.2 (Darwin 25.6.0); rustc 1.95.0 (59807616e 2026-04-14), cargo 1.95.0; Python 3.14.6; 2026-10-16. One process, no other load; the harness pins nothing, macOS schedules performance cores first.

## Results

Every correctness check passed before its kernel was timed: the three field vectors; the hand case (the tree `[2, 3, 5, 7]` accepted from its root 210 and rejected against 211 with `FinalMismatch { round: 0 }`); all five forest batches verified by the driver around `verify_product_level` from the trees' true products, transcripts ending in the same state, every leaf claim equal to the leaves' MLE; 14 `eq` tables against the definition at 17 vertices each; every built root equal to its tree's product; the five `prove_batch` cases (the leaf-level instance, three `Fp2` tables, three `Fp` tables, the two fraction-proxy levels) accepted by `verify_batch` from hypercube sums with the points and transcript states agreeing; binding every round equal to `evaluate_base`; the fraction tree's root `n/d = Σ 1/d_i` on 2^8 leaves. Raw output: [results/run.txt](results/run.txt), [results/run.json](results/run.json) (the pre-registered run), [results/run2.txt](results/run2.txt), [results/run2.json](results/run2.json) (the reproduction run), [results/count.txt](results/count.txt), [results/preregistration.txt](results/preregistration.txt) (the time and digest of the pre-registered report), the cost-model runs listed below.

**Unit costs** (E1, primary; one performance core, 2^12 elements in place by a constant, inside L1D; median of 10 after 3 warm-ups):

| Unit | ns | CV |
|---|---|---|
| Goldilocks multiplication | 0.598 | 4.11% after a rerun, flagged (minimum 0.586, maximum 0.647) |
| Goldilocks addition / subtraction | 0.342 / 0.500 | 0.79% / 0.92% |
| `Fp2` multiplication (Karatsuba, `mul_small(7)`) | 3.347 | 2.21% |
| `Fp2` by `Fp` (`mul_by_base`) | 0.878 | 0.86% |
| `Fp2` addition / subtraction | 0.442 / 0.267 | 0.77% / 0.49% |
| Streaming copy, 2^22 `Fp2`, one thread | 0.217 per element | 1.24%; 147.8 GB/s read plus write |

**The kernels and the fit `F`** (E1, primary, one performance core; nanoseconds per leaf, per table entry or per lookup as marked; predicted = counts of `count.py` × unit costs):

| Case | Baseline: measured (median; min; CV) | Candidate: predicted | Ratio |
|---|---|---|---|
| `prove_forests` 2^8 × 2^14 (2^22 leaves, RFC-0020's row shape), per leaf | 31.697 (31.284; 0.56%) | 31.108 | 0.981 |
| `prove_forests` 2^10 × 2^12 (2^22 leaves), per leaf | 31.236 (31.121; 0.52%) | 31.102 | 0.996 |
| `prove_forests` 1 × 2^22, per leaf | 31.403 (31.214; 1.35%) | 31.110 | 0.991 |
| `prove_forests` 2^4 × 2^14 (2^18 leaves), per leaf (context) | 31.469 (31.148; 0.35%) | 31.106 | 0.988 |
| `prove_forests` 2^6 × 2^14 (2^20 leaves), per leaf (context) | 31.683 (31.416; 1.48%) | 31.108 | 0.982 |
| `eq` tables of every level, per leaf | 3.273 (3.232; 1.17%) | 3.614 | 1.104 |
| level build (harness copy of `build_levels`), per leaf | 0.870 (0.837; **3.95%, flagged after a rerun**) | 0.598 | 0.688 |
| engine: `eq` public `Fp2` × even × odd (`Fp`), degree 3, 2^21, per entry | 27.184 (27.007; 0.56%) | 26.898 | 0.989 |
| engine: three `Fp2` tables, degree 3, 2^21, per entry | 37.348 (37.235; 0.29%) | 37.027 | 0.991 |
| engine: three `Fp` tables, degree 3, 2^21, per entry | 24.084 (23.976; 0.46%) | 25.176 | 1.045 |
| one `bind_lowest` pass, `Fp` into `Fp2`, 2^22, per entry | 0.871 (0.866; 1.74%) | 0.860 | 0.987 |
| binding every round, 2^22, per entry | 3.133 (3.111; 0.26%) | 2.888 | 0.922 |
| fraction proxy: build, per lookup | 8.914 (8.872; 0.43%) | 8.548 | 0.959 |
| fraction proxy: leaf level (numerators 1), 2^21, per entry | 82.361 (81.410; 0.72%) | 80.588 | 0.978 |
| fraction proxy: generic level, 2^21, per entry | 114.356 (113.525; 0.50%) | 111.082 | 0.971 |

`F` passes: every ratio lies in `[0.69, 1.10]`, the three forest shapes in `[0.98, 1.00]`. The leaf cost does not depend on the shape (31.2–31.7 ns per leaf over three shapes and three sizes from 2^18 to 2^22 leaves). The components of the predicted 31.1 ns per leaf: the `Fp2` rounds after the first 12.4 ns (40%), their folds 6.1 (20%), the first round 4.6 (15%), its folds 3.8 (12%), the `eq` tables 3.6 (12%), the level build 0.6 (2%).

**Pre-registered observations** (not part of the verdict): (a) `κ` from binding every round is 1.253 ns per `Fp`-multiplication equivalent (3.133 ns / 2.5); `κ × 25.5` predicts the forest within +0.8% to +2.3% on every shape and size, inside the idea's ±30%. (b) The `c_π / c_ℓ` shape proxy is 31.697 / 107.273 = **0.295** (the fraction tree per lookup: build 8.9 + (leaf level 82.4 + generic level 114.4) × 2^21 / 2^22), outside the expected `[0.4, 0.8]` and equal to the budget's assumed 12 / 40 = 0.30; by multiplication count the ratio is 25.5 / 77 = 0.33 against the proxy and 0.34–0.46 against SUR-0002's 56–74.

**Reproduction run** ([results/run2.txt](results/run2.txt)): the forest 32.099 ns per leaf (ratio 0.973), the `eq` tables 3.423 (1.051), the level build 0.928 with CV 3.71% (flagged again; ratio 0.648), the fraction proxy 107.1 ns per lookup (observation (b) 0.300); three cases of that run were disturbed by another process on this shared machine (three `Fp` tables median 48.6 ns with minimum 24.96, the first run's median, CV 33%; the two binding cases CV 42% and 49% with minima 0.904 and 3.244), so that run is context: the first run, in which every case but two has CV ≤ 1.74% and minima within 1.5% of medians, is the pre-registered run and decides.

**Counts and the H100 interval** (E0, primary, `count.py`; per leaf of the whole tree, 2^8 trees of 2^14 leaves):

| Quantity | Value |
|---|---|
| Operations in the harness's units | 2.5 `Fp` mul, 3.0 add, 2.0 sub; 6.0 `Fp2` mul, 2.5 `mul_by_base`, 9.0 `Fp2` add, 5.0 `Fp2` sub |
| By component (`Fp`-multiplication equivalents) | `eq` 3.0, round 1 4.5, its folds 3.5, `Fp2` rounds 9.0, `Fp2` folds 4.5, level build 1.0 = **25.5** (three nodes per round; 30.0 at four nodes, IDEA-0008's 29 being the four-node count without the level build; 31.5 with `mul_small(7)` charged as a multiplication) |
| Goldilocks operations | 25.5 multiplications, 6.0 `mul_small(7)`, 39.0 additions, 24.0 subtractions |
| int32 ops, low (lazy: 18 / 8 / 4 / 4) / high (as written: 26 / 13 / 8 / 5) | 759 / 1,173; multiplications only at 18: 459 |
| HBM bytes, fused / unfused | 40 / 168 |
| Compute, ps at the int32 peak / at 60% | 45.4 / 75.6 (low), 70.1 / 116.9 (high) |
| Memory, ps at the HBM peak / at 80% | 11.9 / 14.9 (fused), 50.1 / 62.7 (unfused) |
| Multiplications only, 18 int32 each (IDEA-0008's and SUR-0002's reading) | 27.4 ps at peak, 45.8 at 60% |

`I = [lo, hi] = [45.4, 116.9]` ps per leaf, `ρ_π ∈ [3.78, 9.74]`; memory never binds; the central value (geometric mean) is 73 ps.

**Cost model at the interval's ends** (E0, `cost_model.py`, corrected profile; [results/cost_model_lo.txt](results/cost_model_lo.txt), [results/cost_model_high_at_peak.txt](results/cost_model_high_at_peak.txt), [results/cost_model_sensitivity.txt](results/cost_model_sensitivity.txt), [results/cost_model_hi.txt](results/cost_model_hi.txt)):

| `c_π` (ps) | μ(1K) | μ(4K) | μ(16K) | μ(128K) | Against SC-2 (12, 45) |
|---|---|---|---|---|---|
| 12 (assumed) | 8.9 | 14.1 | 35.1 | 171.4 | holds |
| 45.4 (floor) | 9.3 (+5.2%) | 15.9 (+12.9%) | 42.0 (+19.5%) | 207.0 (+20.8%) | holds |
| 70.1 (high count at peak) | 9.7 (+9.0%) | 17.2 (+22.5%) | 47.1 (+33.9%) | 233.4 (+36.2%) | fails at 16K |
| 73 (central) | 9.7 (+9.5%) | 17.4 (+23.6%) | 47.7 (+35.6%) | 236.5 (+38.0%) | fails at 16K |
| 116.9 (ceiling) | 10.3 (+16.3%) | 19.8 (+40.6%) | 56.7 (+61.2%) | 283.3 (+65.3%) | fails at 16K |

Sensitivity at the central 73 ps: `c_π`'s share of `μ` is 10.4% / 22.9% / 31.4% / 32.9% at 1K / 4K / 16K / 128K (from 1.9% / 4.6% / 7.0% / 7.5% at 12 ps), the largest unit cost at 16K and 128K while `c_ℓ` stays at its assumed 40 ps; halving `c_π` is worth −5.2% / −11.4% / −15.7% / −16.5% (from −0.9% / −2.3% / −3.5% / −3.7%).

## Analysis

**Against the criterion.** `F` passes on every kernel, with the forest kernels predicted within 2% from in-cache unit costs and the counts of `count.py`: the operation count describes the production prover it prices. The floor of the interval, `lo = 45.4` ps, is 1.9× the 24 ps trigger of 08 §2 and 3.8× the assumption, so the substantive conditions of `validated` hold with margin. The verdict is nevertheless `inconclusive`, by the criterion's third clause: the level-build kernel, a kernel of `F`, kept a coefficient of variation of 3.95% after its rerun (3.71% in the reproduction run). That clause protects the fit against kernels too unstable to be fitted; here it fired on a protocol mismatch, not on instability: the kernel runs 3.6 ms per iteration and allocates 64 MiB of fresh pages each time (as the production `build_levels` does), so page zeroing, not arithmetic, sets both its noise and its 31% under-prediction, and its ratio stays in `[0.65, 0.71]` at either extreme of its range. The in-cache microbenchmarks avoid this by repeating their kernel until an iteration lasts 20 ms; the millisecond kernels did not, and the pre-registration did not say they should. The interval `I` does not depend on any measured kernel, and the fit is judged on forest kernels with CV ≤ 1.35%; nothing in the verdict word qualifies the numbers, which are reported as they are.

**What the count says.** Even a lazily reduced, fully fused kernel at 100% of the integer pipeline needs 45 ps per leaf, so RFC-0020's "proceed" band (`ρ_π ≤ 2`) is excluded by count; the interval covers the "re-plan" band at its floor (`ρ_π = 3.78`) and the "stop and redesign" band above 48 ps, which the as-written count reaches at 61% of peak and the lazy count at 95%. The SC-2 ceiling `μ(16K) ≤ 45` needs `c_π ≤ 60` ps: it holds at the floor and fails at the central value, at the as-written count even at peak (70.1 ps) and at the ceiling; `μ(1K) ≤ 12` holds at every point (≤ 10.3). The additions matter: 63 of them per leaf are 33–37% of the int32 work, so the multiplication-only reading of IDEA-0008 and SUR-0002 (27.4 ps at peak here) understates a kernel by about a third; SUR-0002's `c_ℓ` interval `[76, 169]` ps at peak and 60% is a multiplication-only count and should be read the same way.

**What the residuals mean.** The `eq` tables are over-predicted by 10%: `eq_table`'s doubling pass is a simpler loop than the in-place-by-constant unit loop and its early iterations are tiny. The three-`Fp`-table engine case is over-predicted by 4.5%: `zip3` keeps its products in registers, cheaper per operation than the unit loop. The level build is under-predicted by 31% for the allocation reason above. The forest kernels land within 2% because the `Fp2` rounds and folds, 60% of the time, are predicted almost exactly (the three-`Fp2`-table case: 0.991). The single-constant reading of the idea holds within 2.3%, far inside its ±30%, but the constant hides that the three `Fp2` units cost 5.6×, 1.5× and 0.7× an `Fp` multiplication rather than 3, 2 and 0: it fits this core and does not transfer to a GPU translation, which the per-operation count does.

**Observation (b).** The measured `c_π / c_ℓ` shape proxy, 0.295, equals the budget's assumed ratio 12 / 40 and falls below the idea's expected `[0.4, 0.8]`. The proxy over-counts the fraction tree relative to SUR-0002's analytic count because the engine binds the shared `eq` table once per instance (7 and 9 table folds per pair in the two levels instead of 5) and forms three instances' lines separately: 77 `Fp`-multiplication equivalents per lookup against 56–74. Either way the ratio is 0.30–0.46: the two unit costs are above their assumptions by similar factors (multiplications only at peak: `c_π` 27.4 against 12, 2.3×; `c_ℓ` 76–101 against 40, 1.9–2.5×), so the budget's relative weighting of the two trees is about right and its absolute level is not. The redundant `eq` folds are IDEA-0010's mechanism 1, quantified here at about 8 ns of the proxy's 107 ns per lookup (7.5%, E1).

Threats to validity:

- CPU against GPU: the per-operation int32 figures are hand counts from the Rust source with stated low and high translations, not measured SASS; the 64 int32 ops per clock per SM and the 1.98 GHz clock are documentation figures (secondary); the 60% utilization is [VYTA25]'s figure for a 377-bit-field multiplication kernel on an A40, applied to Goldilocks `Fp2` rounds it did not measure. These two factors are the whole width of the interval (1.55× and 1.67×).
- Operation counts against wall clock: the fit validates the arithmetic counts, which are hardware-independent, not their translation; the count charges nothing for the `γ` weights, the transcript's hashes, the de-interleaving copy and the scratch rows.
- Memory: the byte model is E0; on this core the leaf cost does not change between 2^18 and 2^22 leaves and the in-cache units predict it within 2%, consistent with HBM not binding on the GPU at either end (15 to 63 ps against 45 to 117 of compute); a GPU kernel with more passes than the as-written count would move the memory floor up.
- Protocol: two cases of the pre-registered run kept a coefficient of variation above 3% after their rerun (the `Fp` multiplication unit, 4.11%, whose share of any prediction is below 5%; the level build, 3.95%); the reproduction run was disturbed in three cases by another process on this shared machine and is reported as context only.
- Input distribution: uniform leaves with one zero per tree; control flow is data-independent (INV-ARCH-2), so the timings are distribution-free.
- The fraction-tree proxy is the engine's instances in the shape of RFC-0008's level relation, not a LogUp prover; its cost bounds the production fraction tree from above on this engine and says nothing about a fused GPU kernel.
- One machine, one day, one toolchain; the bench's figures of 2026-10-16 (31.9 ns per leaf, 3.25 for the `eq` tables, 3.05 for binding every round) agree within 3%.

## Verdict

`inconclusive`, by the letter of the pre-registered criterion (the coefficient-of-variation clause on the level-build kernel), with every substantive condition of `validated` met: `F` passes (predicted/measured 0.69–1.10, the forests 0.98–1.00), and the counted floor `lo = 45.4` ps (`ρ_π = 3.78`) excludes RFC-0020's "proceed" band and fires the 08 §2 trigger at every point of `I = [45.4, 116.9]` ps (E0 fitted to E1). IDEA-0008 moves to `inconclusive` with the same qualification: its deliverable, `c_π ∈ [45, 117]` ps on an H100 with a central 73 ps, stands on the count and the fit, neither of which the flagged kernel touches; its confidence rises from 0.6 to 0.7 (the arithmetic count predicts the production prover within 2%; the GPU translation remains the uncertainty), and its priority as a further E1 experiment falls to 0 (nothing E1 can settle remains but the CV clause, which a rerun with a repetition count for millisecond kernels would clear). The 12 ps assumption of [08 §1](../../../docs/spec/08-performance-budget.md#1-cost-model) is excluded at every point of the interval; `μ(16K) ≤ 45` holds only below 60 ps, that is for a lazily reduced, fused kernel above 75% of the integer pipeline's peak; `μ(1K) ≤ 12` holds throughout. Both are inputs to RFC-0020's gate and to the cost model's corrected profile, not changes to the design.

## Follow-ups

- **E2 (RFC-0020 v0.1 gate).** The `c_π` microbenchmark as specified (2^30 leaves in rows of 2^14), with Nsight's int32-pipe utilization and the kernel's fusion depth recorded: the two parameters that span the interval. The pre-registered prediction is `[45, 117]` ps; a measurement inside it validates the method, and the measured point names the branch ("re-plan" to 48 ps, "redesign" above; "proceed" is excluded by count).
- **Cost model and baseline.md §2.** Reprice `leaf_ps` in the corrected profile to 73 ps with the interval `[45, 117]` and restate the sensitivity table (`c_π` share 7% → 31% at 16K, halving it −15.7%); the source profile keeps 12 ps so that `--check` stays green. This is the consolidation step IDEA-0008 was for.
- **Protocol amendment for the next experiment of this kind.** Millisecond kernels need the repetition rule of the in-cache microbenchmarks (an iteration of at least 20 ms), or the CV clause fires on allocation noise; a rerun of this harness under that rule, pre-registered as an amendment, would convert this verdict without changing a number.
- **`c_ℓ`'s additions (SUR-0002, IDEA-0005).** Count the additions and subtractions of the fraction tree as this experiment does for the product tree; the multiplication-only count understates a kernel by about a third here, so SUR-0002's `[76, 169]` ps is a floor on its own terms.
- **IDEA-0010 (shared-point fusion), mechanism 1.** One `eq` table bound once per batch instead of once per instance: about 7.5% of the proxy's lookup cost (E1); an issue against `tessera-sumcheck` regardless of the idea's fate.
- **IDEA-0009 (four-ary levels).** The `Fp2` rounds after the first and their folds are 60% of the leaf cost (E1 split); the count here gives the per-level structure a four-ary or round-skipping variant must beat.
- **Observation (b) for the budget.** The measured ratio `c_π / c_ℓ` of 0.30 says the two trees are mispriced by the same factor, not relative to each other: the certificate-lens ideas gain weight because every leaf-stage cost does, not because the product tree is the odd one out.
