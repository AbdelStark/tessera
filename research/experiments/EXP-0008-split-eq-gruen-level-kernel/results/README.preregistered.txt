# EXP-0008: Is a split-`eq`, Gruen-corrected level kernel at least 20% faster per leaf than the engine's `eq`-table kernel, with bit-identical messages?

| | |
|---|---|
| Status | `planned` |
| Idea | [IDEA-0021](../../ideas/IDEA-0021-split-eq-gruen-level-kernel.md) (primary: the product-tree levels); [IDEA-0018](../../ideas/IDEA-0018-split-eq-fraction-level-kernel.md) as the second case (the fraction-tree proxy of EXP-0006), merged into it by the triage of 2026-10-16 |
| Evidence level | E1 for the time ratios (CPU wall clock here, one Apple M4 Max performance core); E0 for the operation counts and the cost-model translation; whether the H100 level kernel is compute-bound (IDEA-0021 kill criterion 2) and the shared-memory layout of the two `√N` tables in a streamed tile (kill criterion 3) stay E2 |
| Hardware | Apple M4 Max (12 performance cores, 128 KiB L1D and 16 MiB L2 per performance cluster), macOS 26.6.2 |
| Pre-registered on | 2026-10-16 |

## Hypothesis

In one harness on one M4 Max performance core, a prover for the product-tree levels of [RFC-0010 'Tree levels'](../../../docs/rfcs/RFC-0010-proof-schedule.md#tree-levels) that never materializes or binds an `eq` table (two `√N`-entry tables and one scalar per instance, [DT24]), evaluates two node sums per round instead of three and reconstructs the degree-3 round message from the instance's running claim ([Gru24] §3.2 in the transcript-preserving form of [JoltOpt26] and [Stwo26]) produces, for every level of each of EXP-0006's three 2^22-leaf forest shapes, round messages and final evaluations bit-identical to the production `prove_forests`', proofs the independent verifier accepts, and runs at least 20% faster per leaf than `prove_forests` (`T_base / T_cand ≥ 1.25`), as IDEA-0021's count predicts (25.5 → about 15 `Fp`-multiplication equivalents per leaf, E0).

## Success criterion

Written before any code. Two arms run in the same program on the same inputs for each forest shape, and `σ = 1 − T_cand / T_base` is the speed-up per shape, `T` the median time per leaf:

- `prove_forests` (baseline): the production forest prover of `crates/tessera-sumcheck` (`forest.rs`, `prover.rs`, `kernel.rs`), unmodified: one `eq` table per instance per level built by `eq_table`, bound with the two value tables after every round, three node sums (`p(0), p(2), p(3)`) per round.
- `split_forest` (candidate): the harness's forest prover: the same level build (the harness copy of `build_levels` from EXP-0006), then per level the split-`eq`, Gruen-corrected kernel of IDEA-0021's Mechanism parts 1–3 as described under Method, including the derivation of the root claims from the root values and the carry of each instance's claim from level to level, so that the candidate's timed region contains everything the baseline's does.

Shapes (stacked `SoftmaxRow`-like forests of one zero leaf per tree, as EXP-0006): the verdict's three 2^22-leaf shapes, 2^8 trees of 2^14 leaves (RFC-0020's `c_π` row shape), 2^10 trees of 2^12 leaves, and one tree of 2^22 leaves; and, as context, 2^4 and 2^6 trees of 2^14 leaves (2^18 and 2^20 leaves) for the size dependence.

Verdicts:

- `validated`: `σ ≥ 0.20` on each of the three 2^22-leaf shapes; every correctness check below passes (bit-identical proofs level by level, equal transcript states, verifier acceptance from the true roots, the degenerate-challenge cases); every arm the three ratios need has a coefficient of variation at most 3% after at most one rerun.
- `refuted`: `σ < 0.15` on each of the three shapes (IDEA-0021's kill criterion 1: the engine's nodes-by-addition rule already made the dropped node cheap, and only the `eq` part is real).
- `inconclusive`: anything else: `0.15 ≤ σ < 0.20` on some shape, shapes on both sides of a threshold, or a coefficient of variation above 3% after the rerun on an arm the verdict needs.

A candidate message that differs from the baseline's in any level, round or final evaluation is a defect in the harness, fixed and rerun before any number is read (the skill's rules of evidence); never a verdict.

**Second case, IDEA-0018 (the fraction-tree proxy).** EXP-0006's two proxy levels at 2^21 entries over `Fp2`, the leaf level with numerators 1 (`prove_batch` of degree 3 on `[eq, d_e, d_o]`, `[eq, d_e]`, `[eq, d_o]`) and the generic level (`[eq, n_e, d_o]`, `[eq, n_o, d_e]`, `[eq, d_e, d_o]`), each against the candidate's single-pass shared-`eq` kernel for the same three instances (IDEA-0018 mechanisms 1–4: no `eq` table, two nodes, the three instances' `γ` powers folded into one relation per pair, one `eq` structure for the level), on the same tables, point and transcript; `σ_leaf` and `σ_generic` are the per-entry speed-ups. This case sets IDEA-0018's status word, not the experiment's verdict: `validated` if `σ ≥ 0.30` on both levels (its note's hypothesis: at most 0.70× the time) with bit-identical messages; `refuted` if `σ < 0.10` on both (its kill criterion); `inconclusive` otherwise or on a flagged coefficient of variation.

**Observations, reported and not judged.**

- (a) The level kernel alone: EXP-0006's leaf-level instance (`eq` public over `Fp2`, the even and odd halves over `Fp`, 2^21 entries) proven by `prove_batch` against the candidate's level prover on the same tables, point and claim, per entry; IDEA-0021's kill criterion names this pair. The claim is an input to the candidate here (in a tree level it comes from the level above), computed outside the timed region.
- (b) Operation counts (E0, `count.py`): the candidate's and the baseline's `Fp`-multiplication equivalents and expanded Goldilocks operations per leaf, the prediction of both arms from EXP-0006's measured unit costs, the H100 interval `I` of the candidate computed as EXP-0006 computed the baseline's (int32 peak and 60% utilization, HBM peak and 80%), and the cost model under the `counted` profile with `leaf_ps = 73 × T_cand / T_base` (the E1 ratio applied to EXP-0006's E0-fitted centre), `leaf_ps` from the count ratio, and the fraction rows at `lookup_ps = 97` (the triage's honest `eq`-share row) and at `101 × T_cand / T_base` of the proxy, the latter flagged as IDEA-0018's defective input and not scored.
- (c) The `eq` tables of every level alone (EXP-0006's component, 3.27 ns per leaf), as the removed component's size.

**Degenerate-challenge fallback** (the completeness obligation queue item 6 asks for). Gruen's recovery divides by `ℓ_j(1) = s_i · r_{i,j}`, which is zero when the `eq` point's coordinate `r_{i,j}` is zero or when an earlier challenge `c_k` made `ℓ_k(c_k) = (1 − r_{i,k})(1 − c_k) + r_{i,k} c_k = 0` (then `s_i = 0` and every later message of the instance is zero); each has probability about 2^-128 per round over uniform `Fp2` challenges. The kernel then computes `v'_j(1)` from the pairs as a third node sum, and the message `ℓ_j(z)·v'_j(z)` is correct either way. Three deterministic checks exercise it, each requiring bit-identical messages with `prove_batch` on the same inputs and challenges: (i) a point with a zero coordinate (`ℓ_j(1) = 0` at that round); (ii) a point with a coordinate equal to one (`ℓ_j(0) = 0`: no fallback, the recovery divides by `s_i`); (iii) a scripted transcript that returns `c_k = (r_{i,k} − 1) / (2 r_{i,k} − 1)` at round `k`, so that `s_i = 0` from round `k + 1` on and the fallback runs in every later round, both provers driven by the same script. The branch depends on a public challenge only (INV-ARCH-2 constrains data-dependent control flow).

Nothing above this line is edited after pre-registration; amendments go below it with a date.

## Method

- **Harness.** One Rust program, `spliteq`, its own Cargo workspace in this directory (empty `[workspace]` table, `license = "Apache-2.0 OR MIT"`, `publish = false`, `#![forbid(unsafe_code)]`), depending by path on `tessera-sumcheck`, `tessera-poly`, `tessera-field`, `tessera-hash` and `tessera-codec` for the baseline kernels and the field, and on `tv-core`, `tv-field` and `tv-hash` for the independent checks, as EXP-0006 (research rule 2: a research crate is a dependency of nothing). The baseline kernels are the production ones, unmodified. The candidate is written in the harness; it binds its value tables with `tessera_poly::bind_lowest` and `bind_lowest_in_place`, the same arithmetic as the engine's crate-private `bind_base` and `bind_in_place`, and builds its small `eq` tables with `tessera_poly::eq_table`.
- **The candidate's level kernel**, per instance `i` on `n_i` variables with `eq` point `r_i` (lowest variable first), claim `c_i` and weight `γ^i`, in a batch of `n_max` rounds under the labels of `prove_batch` (`tree/{ℓ}/gamma`, `tree/{ℓ}/round/{j}`, `tree/{ℓ}/r/{j}`, `tree/{ℓ}/evals`, then `prove_forests`' `tree/{ℓ}/line`; root challenges under `tree/{ℓ}/root/{c}` as `prove_forests` squeezes them):
  1. Padding rounds `j < n_max − n_i` contribute `γ^i · 2^{n_max − n_i − j − 1} · c_i` at every node, as `prove_batch` contributes `2^{…}·S_i` with `S_i` computed from the tables (`c_i = S_i` for an honest prover).
  2. Own round `k` (from 0), with `m = n_i − k − 1` variables above the current one: `eq_lo = eq_table(r_i[k + 1 .. k + 1 + ⌈m/2⌉])` and `eq_hi = eq_table(r_i[k + 1 + ⌈m/2⌉ .. n_i])`, rebuilt each round (`2^{⌈m/2⌉} + 2^{⌊m/2⌋} − 2` multiplications, at most 2^11 entries in all at `n_i = 21`, inside L1D); the current variable's public line `ℓ(X) = s_i · ((1 − r_{i,k})(1 − X) + r_{i,k} X)`; the node sums `v(0)` and `v(2)` as the two-level sum `Σ_{x_hi} eq_hi[x_hi] · Σ_{x_lo} eq_lo[x_lo] · V_L(X, x)·V_R(X, x)` over the pairs `x = x_hi·2^{⌈m/2⌉} + x_lo`, each table's value at node 2 formed as `kernel.rs` forms it (`b + (b − a)`), the products in `Fp` and the scaling by `eq_lo` an `Fp2`-by-`Fp` product in the first own round, both in `Fp2` after; the recovery `v(1) = (c_i − ℓ(0)·v(0)) · ℓ(1)^{-1}`, or a third node sum at `X = 1` when `ℓ(1) = 0` (the fallback); the extrapolation `v(3) = v(0) + 3·(v(2) − v(1))` of the degree-2 `v`; the message `γ^i · ℓ(z)·v(z)` at `z = 0, 2, 3` added into the round's `Vec<Fp2>`. After the challenge `c`: `c_i ← ℓ(c)·v(c)` (Newton's form of the quadratic), `s_i ← ℓ(c)`, and the two value tables are bound (`bind_lowest` into `Fp2` on the first own round, `bind_lowest_in_place` after); the `eq` tables are never bound.
  3. The final evaluations `u_i, w_i` are the bound tables' single entries (the tables' first entries, lifted, for `n_i = 0`), absorbed in instance order under `tree/{ℓ}/evals`; then `ρ`, the next claim `u_i + ρ·(w_i − u_i)` on the point `(ρ ‖ r'_i)`, as RFC-0010 step 4.
  4. Root claims: at the level where a forest joins, the harness forms the public root values `v_y = even_1[y]·odd_1[y]` from the forest's level-1 halves (`2^s` products) and the claim `Σ_y v_y·eq(r, y)` with a `2^s`-entry `eq` table, inside the timed region (`SoftmaxRow` roots are zero in production and the claim is then zero at no cost).
- **The fraction-proxy kernel** (second case): one instance for the level's three, sharing the point, with the per-pair relation `d_e(X)·(d_o(X) + γ) + γ²·d_o(X)` on the leaf level (equal to `d_e d_o + γ d_e + γ² d_o`, the three instances' products with their `γ` powers) and `d_o(X)·(n_e(X) + γ²·d_e(X)) + γ·(n_o(X)·d_e(X))` on the generic level, the combined claim `c_0 + γ c_1 + γ² c_2` from the three instances' hypercube sums (computed outside the timed region, as a level above would supply them), the same recovery on the combined degree-2 polynomial, the value tables bound once each (two on the leaf level, four on the generic), and the final evaluations in the baseline's order (`d_e, d_o, d_e, d_o` and `n_e, d_o, n_o, d_e, d_e, d_o`).
- **Inputs.** Synthetic, SplitMix64: leaves from seed `0x7E55_E7A0_0058_0001` (EXP-0006's and the bench's), one zero leaf per tree at a seeded position; the level kernel's point and tables from `0x7E55_E7A0_0058_0002`; the fraction proxy from `0x7E55_E7A0_0058_0003`; the small-size equivalence sweep, the two-forest batch and the degenerate cases from `0x7E55_E7A0_0058_0005`. The session context is the bench's (`tessera-protocol/1`, non-interactive, `test-small/1`, statement digest `0x5a` × 32, session id `7` × 16).
- **Protocol** as RFC-0020's gate: 3 warm-ups and 10 measured iterations per arm; median, minimum, maximum and coefficient of variation; a coefficient above 3% reruns the arm once and the rerun is what is reported, flagged if still above. Every iteration lasts at least 20 ms by repetition (EXP-0006's protocol follow-up: the 2^18-leaf forests take about 8 ms once). Times per leaf of the whole forest in nanoseconds for the forest arms, per table entry for the kernel arms.
- **Correctness checks**, each aborting the run on failure, all before timing:
  - field: the three edge products of `spec/vectors/field/goldilocks.json`, as EXP-0006;
  - the hand case, one tree of the leaves `[2, 3, 5, 7]`: the candidate's proof accepted by the verifier driver from the root 210 and rejected against 211, and equal to the baseline's;
  - a small-size equivalence sweep: for every `n` from 1 to 12 at random points, and for two-instance batches with front-padding, the candidate's level proof equals `prove_batch`'s and the transcripts end in the same state (the split's edge cases `m = 0, 1, 2` and the padding path);
  - for every timed forest: the candidate's proof of every level equals the baseline's (`SumcheckProof` equality, every round and final evaluation), the transcripts end in the same state, the candidate's proofs are accepted by the driver written around `tv_core::tree::verify_product_level` from the trees' true products computed in `tv-field` arithmetic, and the leaf claims equal the baseline's and the stacked leaves' multilinear extension; the BLAKE3 digest (`tessera_hash::hash(Domain::ColumnMerge, ·)`) of the canonical little-endian bytes of every message is printed for both arms;
  - a two-forest batch of different heights and leaf counts (2^2 trees of 2^8 leaves with 2^5 trees of 2^6 leaves: the second joins at level 2 with one more variable, so the first is front-padded by one round): bit-identical to `prove_forests` and accepted;
  - the three degenerate-challenge cases above;
  - the level kernel and the two fraction levels: the candidate's proof equals `prove_batch`'s, the transcripts agree, and `tv_core::sumcheck::verify_batch` accepts it from claims summed over the hypercube in `tv-field`, the public `eq` evaluated by the verifier's own `eq`.
- **Counts (E0).** `count.py`: EXP-0006's per-pair rules for the baseline beside the candidate's (two products and two `eq_lo` scalings per pair per round, two folds, `√N` `eq` tables, the per-round recovery as a constant), per leaf of the whole tree; expanded Goldilocks operations; the int32 translation and the H100 interval as EXP-0006; the fraction proxy's two levels likewise.

## Reproduction

```sh
# from the repository root
cd research/experiments/EXP-0008-split-eq-gruen-level-kernel
cargo run --release --locked -- all results/run.json | tee results/run.txt   # first run without --locked writes Cargo.lock
python3 -I count.py | tee results/count.txt
python3 -I ../../tools/cost_model.py --profile counted --set leaf_ps=<73 x measured ratio> | tee results/cost_model_measured.txt
python3 -I ../../tools/cost_model.py --profile counted --set leaf_ps=<73 x count ratio> | tee results/cost_model_counted.txt
python3 -I ../../tools/cost_model.py --profile counted --set lookup_ps=97 | tee results/cost_model_fraction_97.txt
shasum -a 256 README.md                                   # the pre-registered text: results/preregistration.txt
```

Environment: Apple M4 Max, 16 cores (12 performance, 4 efficiency), 48 GiB; macOS 26.6.2 (Darwin 25.6.0); rustc 1.95.0 (59807616e 2026-04-14), cargo 1.95.0; Python 3.14.6; 2026-10-16. One process, one thread, no other load; macOS schedules performance cores first.

## Results

Pending the run.

| Case | Baseline | Candidate | Ratio |
|---|---|---|---|
| | | | |

Raw output: `results/` in this directory.

## Analysis

Pending the run.

## Verdict

Pending the run: one of `validated`, `refuted`, `inconclusive`, judged against the criterion above and nothing else.

## Follow-ups

Pending the run.
