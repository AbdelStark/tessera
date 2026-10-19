# EXP-0013: Does delaying the reduction of the terminal products in the split-`eq` node sums, one reduction per `x_hi` block instead of one per product, make the fraction-level kernel at least 15% / 8% faster per entry and the forest kernel at least 5% faster per leaf, with bit-identical messages?

| | |
|---|---|
| Status | `planned` |
| Idea | [IDEA-0026](../../ideas/IDEA-0026-delayed-reduction-fraction-node-sums.md) (primary: scale-first delayed reduction in the fraction-tree node sums); [IDEA-0030](../../ideas/IDEA-0030-deferred-reduction-level-kernel.md), folded in by the triage of 2026-10-17 as the product-forest arm and the relation-first ordering variant |
| Evidence level | E1 for the time ratios (CPU wall clock here, one Apple M4 Max performance core); E0 for the operation counts, the reduction counts and the cost-model translation; the int32 cost of a 128-bit accumulate on Hopper (IDEA-0030 kill criterion 2) and whether the post-Gruen kernels are compute-bound there (IDEA-0026 kill criterion 3) stay E2 |
| Hardware | Apple M4 Max (12 performance cores, 128 KiB L1D and 16 MiB L2 per performance cluster), macOS 26.6.2 |
| Pre-registered on | 2026-10-17 |

## Hypothesis

In one harness on one M4 Max performance core, the split-`eq`, Gruen-corrected level kernels validated by [EXP-0008](../EXP-0008-split-eq-gruen-level-kernel/README.md), rewritten so that every product whose result feeds only the node sum (a *terminal* product) is computed schoolbook as four unreduced `64 × 64 → 128`-bit partial products and accumulated into three-word (192-bit) accumulators that are reduced once per `x_hi` block, emit round messages and final evaluations bit-identical to EXP-0008's kernels' and to the production `prove_batch` / `prove_forests`, are accepted by the independent verifier, and are faster by at least the pre-registered margins: 15% per entry on the fraction proxy's leaf level and 8% on its generic level with the scale-first ordering of IDEA-0026 (own count: −20% to −22% and −11% to −13% of the kernels' int32 lane operations with the folds included, E0), and 5% per leaf on each 2^22-leaf forest shape with the deferred `eq_lo` scaling of IDEA-0030 (own count: −13% to −15% of the forest kernel's lane operations per leaf, E0).

## Success criterion

Written before any code. Every arm runs in one program on the same seeded inputs on the same machine. `σ = 1 − T_cand / T_base` is the speed-up of a pair, `T` the median time per table entry (fraction levels, the level kernel alone) or per leaf of the whole forest (forest arms). The baseline of every ratio is EXP-0008's split-`eq` kernel as it stands in that harness, copied here unchanged; the production engine (`prove_batch`, `prove_forests`) is timed as context and is never the denominator of a ratio, since the candidates differ from EXP-0008's kernel only in where the reductions happen.

Arms (the kernels are described under Method):

- `split` (baseline): EXP-0008's candidate kernels, unchanged: `fraction_leaf_sums`, `fraction_generic_sums`, `product_base_sums`, `product_ext_sums`, every product Karatsuba-reduced before it is added.
- `scale-first` (IDEA-0026, the verdict arm, fraction levels): the `eq_lo` scaling applied to one factor of the relation first, so that the relation's outer products are terminal: 4 of the 6 `Fp2` products per pair-round of the leaf level and 4 of the 10 of a generic level are deferred.
- `relation-first` (IDEA-0030's ordering, fraction levels): the relation computed reduced as today and only its `eq_lo` scaling deferred: 2 of 6 and 2 of 10 products.
- `deferred` (IDEA-0030, the product-forest arm): in `product_ext_sums`, `p = V_L·V_R` reduced and `eq_lo · p` deferred (1 of 2 products per node; the two orderings coincide in count on a two-factor product, so one arm); in `product_base_sums`, the first own round over `Fp` tables, `p = V_L·V_R` reduced in `Fp` and the `Fp2`-by-`Fp` scaling deferred as two unreduced products into two accumulators (relation-first is the cheaper ordering there: scale-first would reduce two products before the terminal pair).
- context: the production `prove_batch` (three instances on the fraction levels; `eq` as a public table on the level kernel) and `prove_forests`.

Verdicts (the experiment's verdict, which sets IDEA-0026's status word):

- `validated`: `σ_leaf ≥ 0.15` and `σ_generic ≥ 0.08` for `scale-first` against `split` at 2^21 entries; every correctness check below passes (bit-identical proofs against `split` and against the production engine, equal transcript states, verifier acceptance); every arm the two ratios need has a coefficient of variation at most 3% after at most one rerun.
- `refuted`: `σ_leaf < 0.10` (IDEA-0026's kill criterion 1: the reduction is not where the time goes; the lines, additions and folds are).
- `inconclusive`: anything else: `0.10 ≤ σ_leaf < 0.15`, or `σ_leaf ≥ 0.15` with `σ_generic < 0.08`, or a coefficient of variation above 3% after the rerun on an arm the verdict needs.

A candidate message that differs from `split`'s or the engine's in any level, round or final evaluation is a defect in the harness, fixed and rerun before any number is read (the skill's rules of evidence); never a verdict.

**Second case, IDEA-0030** (its own status word, not the experiment's verdict): `validated` iff `σ ≥ 0.05` for `deferred` against `split` per leaf on each of the three 2^22-leaf forest shapes (2^8 trees of 2^14 leaves, 2^10 of 2^12, one tree of 2^22) and `σ ≥ 0.03` for `relation-first` against `split` per entry on both fraction levels, with bit-identical messages and acceptance; `refuted` iff `σ < 0.02` on all five of those pairs; `inconclusive` otherwise or on a flagged coefficient of variation.

**Observations, reported and not judged.**

- (a) `relation-first` against `scale-first` on the fraction levels: which ordering is faster at equal correctness, and by how much against the count's prediction (scale-first defers twice as many products).
- (b) The level kernel alone (EXP-0006's leaf-level instance, `eq` public over `Fp2`, even and odd halves over `Fp`, 2^21 entries): `deferred` against `split`, both against `prove_batch`, per entry.
- (c) The dot-product microkernel: `Σ_i a_i·b_i` over 2^14 random `Fp2` pairs, Karatsuba-reduced products summed in `Fp2` against schoolbook unreduced products in three 192-bit accumulators reduced once per 2^10 products (the block cadence of the kernels); and the same for `Fp2`-by-`Fp` products; nanoseconds per product, results equal. This isolates the mechanism from the kernels' loop structure and is the empirical form of IDEA-0026's kill check 2 (whether the compiler already defers the reduction: then the two microkernels tie).
- (d) Counts (E0, `count.py`): per pair-round and per leaf, the Goldilocks operations of every arm with the full multiplications (each ending in `reduce128`), the unreduced wide products, the 128-bit accumulates and the per-block reductions counted as separate quantities; the int32 lane translation of [E0-field] (a Goldilocks multiplication 18 lane operations at the lazy end, 8 for the product and 10 for the reduction; 26 as written) with the accumulate at 4, 5 and 8 lane operations (IDEA-0026's and IDEA-0030's assumed range, and the kill point where IDEA-0030 says the saving halves); the H100 interval of the forest kernel recomputed as EXP-0008's `count.py` computes it; the prediction of every arm's time from EXP-0006's unit costs where the units exist.
- (e) Cost model (`counted` profile, `tools/cost_model.py`): IDEA-0026's scoring rule `c_ℓ = 101 × (1 − 0.755·r)`, `r` the equal-weighted saving of the two level kernels (folds included), once with `r` from the count and once with `r = (σ_leaf + σ_generic) / 2` from the clock, the latter flagged as a CPU ratio that overstates the GPU's; IDEA-0030's `c_π = 41.8 × (1 − σ)` with `σ` the forest arm's measured saving on RFC-0020's row shape and with the count's per-leaf lane-operation ratio, against the post-EXP-0008 reference `leaf_ps` 41.8; and the two together.
- (f) The forests of 2^20 and 2^21 leaves (2^6 and 2^7 trees of 2^14) for the size dependence.

**Overflow bound, pinned.** Every operand of a deferred product is canonical (`< P < 2^64`), so every partial product is below `P² < 2^128`. A block is one `x_hi` bucket of the split-`eq` kernel: `2^lo` pairs with `lo = ⌈m/2⌉` for `m` free variables above the current one, so `lo ≤ 10` on 2^21-entry tables (`n = 21`) and `lo ≤ 11` on `n = 22`. The accumulator that receives the most products is `C_1` (the cross terms `a_0 b_1 + a_1 b_0`) of the leaf level's scale-first kernel, two terminal `Fp2` products per node per pair: 4 products per pair, at most `4·2^lo` per block, so its value is below `2^(130 + lo) ≤ 2^141` at `lo = 11`, against the three-word accumulator's `2^192`: 51 bits of slack. The harness pins `MAX_BLOCK_LOG = 20` (a block of at most 2^20 pairs, `n ≤ 41`, more than any level Tessera proves), checks `130 + MAX_BLOCK_LOG ≤ 192` as a compile-time assertion, asserts `|eq_lo| ≤ 2^MAX_BLOCK_LOG` at every kernel entry, and tests the reduction at the maximal block with maximal operands (`4·2^20` products `(P − 1)²`, whose reduced sum is `2^22 mod P`) and on 2^16 random products against the sum of reduced products. The cadence depends on the level's variable count only, never on a value (INV-ARCH-2, RFC-0002 'Timing').

Nothing above this line is edited after pre-registration; amendments go below it with a date.

## Method

- **Harness.** One Rust program, `delayred`, its own Cargo workspace in this directory (empty `[workspace]` table, `license = "Apache-2.0 OR MIT"`, `publish = false`, `#![forbid(unsafe_code)]`), started as a copy of EXP-0008's `spliteq` harness (its measurement protocol, its split-`eq` kernels, its forest prover, its verifier drivers and its correctness checks, unchanged), depending by path on `tessera-sumcheck`, `tessera-poly`, `tessera-field`, `tessera-hash` and `tessera-codec` for the baseline and the field and on `tv-core`, `tv-field` and `tv-hash` for the independent checks (research rule 2: a research crate is a dependency of nothing). The accumulator is written in the harness over the field crate's public surface (`to_canonical_u64`, `from_u64_reduced`, `mul_small`), without `unsafe` and without any change to `tessera-field`.
- **The accumulator.** `Acc192 { lo, mid, hi: u64 }`: `add(p: u128)` is a 192-bit addition with carries (`u64::carrying_add`, stable on the pinned toolchain). `reduce()` uses `2^64 ≡ 2^32 − 1` and `2^96 ≡ −1 (mod P)`, so `2^128 ≡ −2^32`: `value = lo + (2^32 − 1)·mid − 2^32·hi`, each word canonicalized by `from_u64_reduced` (one conditional subtraction, valid for any `u64`), `mid` scaled by `mul_small(2^32 − 1)` and `hi` by one full multiplication, once per block per accumulator. `AccExt2 { c0, c1, c7: Acc192 }` accumulates a schoolbook `Fp2` product `(a_0 + a_1 X)(b_0 + b_1 X)`, `X² = 7`, as `c0 += a_0 b_0`, `c1 += a_0 b_1`, `c1 += a_1 b_0`, `c7 += a_1 b_1` (four wide products, four 192-bit adds, no Karatsuba pre-sums, so every operand stays below `2^64`) and an `Fp2`-by-`Fp` product as `c0 += a_0 b`, `c1 += a_1 b`; its `reduce()` is `(c0 + 7·c7, c1)` with one `mul_small(7)`.
- **The kernels.** Each is EXP-0008's two-level sum `Σ_{x_hi} eq_hi[x_hi] · Σ_{x_lo} eq_lo[x_lo] · R_z(x)` at the nodes `z ∈ {0, 2}` (and `1` under the degenerate-challenge fallback), with the inner sum's terminal products accumulated in one `AccExt2` per node and reduced where the baseline adds `h · acc[z]`, once per `x_hi` block. With `w = eq_lo[x_lo]`, per node and pair:
  - fraction leaf level, relation `e·(o + γ) + γ²·o` (the three instances `[eq, d_e, d_o]`, `[eq, d_e]`, `[eq, d_o]` with weights `1, γ, γ²`): scale-first computes `u = w·e` reduced, then `u·(o + γ)` and `(w γ²)·o` deferred, with `w γ²` read from a per-round table `eq_lo · γ²` (`2^lo` products per round, counted); relation-first computes `R = e·(o + γ) + γ²·o` reduced (two products) and `w·R` deferred.
  - fraction generic level, relation `d_o·(n_e + γ²·d_e) + γ·(n_o·d_e)`: scale-first computes `γ²·d_e`, `u = w·d_o` and `v = (w γ)·n_o` reduced (three products; `w γ` from a per-round table `eq_lo · γ`), then `u·(n_e + γ²·d_e)` and `v·d_e` deferred; relation-first computes `R` reduced (four products) and `w·R` deferred.
  - product level (`deferred`): `Fp2` rounds `p = l·r` reduced, `w·p` deferred; the first own round `p = l·r` in `Fp`, `w.mul_by_base(p)` deferred as `c0 += w_0 p`, `c1 += w_1 p`.
  - The node-2 values are formed as EXP-0008 forms them (`b + (b − a)`); the recovery of `v(1)` from the claim, the extrapolation of `v(3)`, the messages, the claim update and the folds (`bind_lowest`, `bind_lowest_in_place`, `bind_ext`) are EXP-0008's, shared by every arm, so an arm differs from `split` in the inner sum alone.
- **Inputs.** Synthetic, SplitMix64, EXP-0008's seeds so that the tables are the same ones: forest leaves from `0x7E55_E7A0_0058_0001` (one zero leaf per tree at a seeded position), the level kernel's point and tables from `0x7E55_E7A0_0058_0002`, the fraction proxy (keys, values, `α`, `β`, point, generic tables) from `0x7E55_E7A0_0058_0003`, the small-size sweeps, the two-forest batch and the degenerate cases from `0x7E55_E7A0_0058_0005`; new here, the microkernel's vectors from `0x7E55_E7A0_0058_0006` and the accumulator's random test from `0x7E55_E7A0_0058_0007`. The session context is the bench's (`tessera-protocol/1`, non-interactive, `test-small/1`, statement digest `0x5a` × 32, session id `7` × 16).
- **Sizes.** Fraction levels and the level kernel at 2^21 entries over `Fp2` (the leaf level with numerators 1 on `d_e, d_o`; the generic level on four random tables), as EXP-0008; forests of 2^22 leaves in the three verdict shapes and of 2^20 and 2^21 leaves (2^6 and 2^7 trees of 2^14) as context. Times per table entry for the kernel arms and per leaf of the whole forest for the forest arms, in nanoseconds.
- **Protocol** as RFC-0020's gate: 3 warm-ups and 10 measured iterations per arm; median, minimum, maximum and coefficient of variation; a coefficient above 3% reruns the arm once and the rerun is what is reported, flagged if still above; every iteration lasts at least 20 ms by repetition. One process, one thread.
- **Correctness checks**, each aborting the run on failure, all before timing:
  - field: the three edge products of `spec/vectors/field/goldilocks.json`, as EXP-0006 and EXP-0008;
  - the accumulator: the compile-time bound, 2^16 random products reduced once against the sum of reduced products, the maximal block `4·2^20 × (P − 1)²` reducing to `2^22`, and the `Fp2` forms against Karatsuba on random pairs (`cargo test --release`);
  - the hand case, one tree of the leaves `[2, 3, 5, 7]` on the `deferred` forest prover: equal to `prove_forests`' proof, accepted by the verifier driver from the root 210, rejected against 211;
  - small-size sweeps: for every `n` from 1 to 12 at random points, the `deferred` product kernel against `prove_batch` and against `split`, including six two-instance batches with front-padding; and, new here, the fraction relations at every `n` from 1 to 12 (`scale-first` and `relation-first` against `prove_batch`'s three instances and against `split`), so that the block cadence is exercised at `lo = 0, 1, 2, …` and with the per-round `γ` tables of every size;
  - the two-forest batch (2^2 trees of 2^8 leaves with 2^5 trees of 2^6) on `deferred`: bit-identical to `prove_forests`, accepted;
  - the three degenerate-challenge cases of EXP-0008 ((i) a zero coordinate, (ii) a coordinate one, (iii) a scripted `c_k` with `ℓ_k(c_k) = 0`) on the `deferred` kernel, whose fallback sums node 1 through a third accumulator set;
  - every timed forest: the `deferred` proofs of every level equal `split`'s and `prove_forests`', the transcripts end in the same state, the verifier driver around `tv_core::tree::verify_product_level` accepts from the trees' true products, the leaf claims equal the leaves' extension; the BLAKE3 digest of every message printed for the three arms;
  - the level kernel and the two fraction levels: each candidate's proof equals `prove_batch`'s and `split`'s, the transcripts agree, `tv_core::sumcheck::verify_batch` accepts from hypercube sums in `tv-field`; digests printed;
  - the microkernel: the deferred dot products equal the reduced ones.
- **Counts (E0).** `count.py`: EXP-0008's rules for `split` (copied) beside the candidates' rules read from this harness, per pair-round and per leaf; full multiplications, wide products, accumulates and per-block reductions as separate columns; the int32 translation and the H100 interval as EXP-0008; the `c_ℓ` and `c_π` rows of observation (e).

## Reproduction

```sh
# from the repository root
cd research/experiments/EXP-0013-delayed-reduction-node-sums
cargo test --release --locked                                            # the accumulator's checks; first run without --locked writes Cargo.lock
cargo run --release --locked -- all results/run.json | tee results/run.txt
python3 -I count.py | tee results/count.txt
python3 -I ../../tools/cost_model.py --profile counted --set leaf_ps=41.8 | tee results/cost_model_reference_41.8.txt   # the post-EXP-0008 reference
python3 -I ../../tools/cost_model.py --profile counted --set lookup_ps=<count row> | tee results/cost_model_lookup_count.txt
python3 -I ../../tools/cost_model.py --profile counted --set lookup_ps=<measured row> | tee results/cost_model_lookup_measured.txt
python3 -I ../../tools/cost_model.py --profile counted --set leaf_ps=<41.8 x (1 - sigma_forest)> | tee results/cost_model_leaf_measured.txt
python3 -I ../../tools/cost_model.py --profile counted --set leaf_ps=<...> --set lookup_ps=<...> | tee results/cost_model_both.txt
shasum -a 256 results/README.preregistered.txt           # the pre-registered text and its digest: results/preregistration.txt
```

Environment: Apple M4 Max, 16 cores (12 performance, 4 efficiency), 48 GiB; macOS 26.6.2 (Darwin 25.6.0); rustc 1.95.0 (59807616e 2026-04-14), cargo 1.95.0; Python 3.14.6; 2026-10-17. One process, one thread, no other load; macOS schedules performance cores first.

## Results

To be filled after the run; the raw output lands in `results/`.

| Case | Baseline `split` | Candidate | Ratio |
|---|---|---|---|
| | | | |

## Analysis

To be written after the run, against the success criterion and nothing else, with the threats to validity.

## Verdict

Pending the run.

## Follow-ups

Pending the run.
