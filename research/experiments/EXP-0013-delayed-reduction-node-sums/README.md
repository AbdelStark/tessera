# EXP-0013: Does delaying the reduction of the terminal products in the split-`eq` node sums, one reduction per `x_hi` block instead of one per product, make the fraction-level kernel at least 15% / 8% faster per entry and the forest kernel at least 5% faster per leaf, with bit-identical messages?

| | |
|---|---|
| Status | `done: validated` |
| Idea | [IDEA-0026](../../ideas/IDEA-0026-delayed-reduction-fraction-node-sums.md) (primary: scale-first delayed reduction in the fraction-tree node sums); [IDEA-0030](../../ideas/IDEA-0030-deferred-reduction-level-kernel.md), folded in by the triage of 2026-10-17 as the product-forest arm and the relation-first ordering variant |
| Evidence level | E1 for the time ratios (CPU wall clock here, one Apple M4 Max performance core); E0 for the operation counts, the reduction counts and the cost-model translation; the int32 cost of a 128-bit accumulate on Hopper (IDEA-0030 kill criterion 2) and whether the post-Gruen kernels are compute-bound there (IDEA-0026 kill criterion 3) stay E2 |
| Hardware | Apple M4 Max (12 performance cores, 128 KiB L1D and 16 MiB L2 per performance cluster), macOS 26.6.2 |
| Pre-registered on | 2026-10-19 |

## Hypothesis

In one harness on one M4 Max performance core, the split-`eq`, Gruen-corrected level kernels validated by [EXP-0008](../EXP-0008-split-eq-gruen-level-kernel/README.md), rewritten so that every product whose result feeds only the node sum (a *terminal* product) is computed schoolbook as four unreduced `64 × 64 → 128`-bit partial products and accumulated into three-word (192-bit) accumulators that are reduced once per `x_hi` block, emit round messages and final evaluations bit-identical to EXP-0008's kernels' and to the production `prove_batch` / `prove_forests`, are accepted by the independent verifier, and are faster by at least the pre-registered margins: 15% per entry on the fraction proxy's leaf level and 8% on its generic level with the scale-first ordering of IDEA-0026 (own count: −20% to −22% and −11% to −13% of the kernels' int32 lane operations with the folds included, E0), and 5% per leaf on each 2^22-leaf forest shape with the deferred `eq_lo` scaling of IDEA-0030 (own count: −13% to −15% of the forest kernel's lane operations per leaf, E0).

## Success criterion

Written before any code. Every arm runs in one program on the same seeded inputs on the same machine. `σ = 1 − T_cand / T_base` is the speed-up of a pair, `T` the median time per table entry (fraction levels, the level kernel alone) or per leaf of the whole forest (forest arms). The baseline of every ratio is EXP-0008's split-`eq` kernel as it stands in that harness, ported here without arithmetic changes to the current public APIs; the production engine (`prove_batch`, `prove_forests`) is timed as context and is never the denominator of a ratio, since the candidates differ from EXP-0008's kernel only in where the reductions happen.

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

Pre-registration clarification (2026-10-19, before code): the exact EXP-0008 kernel arithmetic is retained, while API signatures, typed transcript labels and session headers are adapted to current main. Every raw sample (including first runs rejected for CV) is saved; the repetition count is calibrated with margin and every recorded iteration is checked to last at least 20 ms. Correctness-only execution is available as `delayred check`. No cryptographic protocol or production code changes.


- **Harness.** One Rust program, `delayred`, its own Cargo workspace in this directory (empty `[workspace]` table, `license = "Apache-2.0 OR MIT"`, `publish = false`, `#![forbid(unsafe_code)]`), ported from EXP-0008's `spliteq` harness (its measurement protocol, its split-`eq` kernels, its forest prover, its verifier drivers and its correctness checks, with current-API adapters only), depending by path on `tessera-sumcheck`, `tessera-poly`, `tessera-field`, `tessera-hash` and `tessera-codec` for the baseline and the field and on `tv-core`, `tv-field` and `tv-hash` for the independent checks (research rule 2: a research crate is a dependency of nothing). The accumulator is written in the harness over the field crate's public surface (`to_canonical_u64`, `from_u64_reduced`, `mul_small`), without `unsafe` and without any change to `tessera-field`.
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

Environment: Apple M4 Max, 16 cores (12 performance, 4 efficiency), 48 GiB; macOS 26.6.2 (Darwin 25.6.0); rustc 1.95.0 (59807616e 2026-04-14), cargo 1.95.0; Python 3.14.6; 2026-10-19; production checkout 726502746cb5a6f29adcdbc7d95b8d2d98735272. One process, one thread, no other load; macOS scheduling is uncontrolled; no CPU affinity is claimed. Current production may already use split-eq kernels; its arm remains context, and EXP-0008 arithmetic remains the denominator.

## Results

Measured on 2026-10-19: **validated** for both IDEA-0026 and IDEA-0030 (E1, primary). The complete raw record is [run.json](results/run.json), [run.txt](results/run.txt) and [raw-batches.json](results/raw-batches.json). [summary.json](results/summary.json) is the mechanically recomputed verdict. All retained coefficients are at most 3%; the scale-first generic fraction arm reran once after its first CV was 3.4106%, and the retained CV is 0.8721%. The deferred RFC-0020 forest arm has CV **2.999517753%**, just below the pre-registered 3% cutoff; it is retained under the written rule. Every one of the 403 raw batches, warm-ups and discarded runs included, lasts at least 30.859125 ms. Reported per-execution times can be below 20 ms because the timed batch repeats executions.

Every candidate and split proof equals production, including final evaluations, claims and transcript states. Independent `tv-core` acceptance passes on all judged inputs. The large fraction digests (BLAKE3 of canonical message bytes) are `d230aeb17a2ec70d915a3b5d907c3c70f9e3f972c95398fccd122e7e9bb0a780` (leaf) and `e47ecc93d46cd2c8ab8c90cdac75d99088e4680b1d2a1d2384433eadc588e0b4` (generic), shared by split, both candidates and production. Small equivalence checks cover all n from 1 through 12, mixed front-padded product batches, a two-forest batch, zero/one equality coordinates, and scripted zero-factor challenges. The hand root 210 accepts and root 211 rejects. `accumulator_boundaries` passes for maximal operands at the pinned maximal block, random scalar/extension/base products and all three carry words against independent modular Horner reduction. [correctness.txt](results/correctness.txt) records correctness-only execution.

All table times below are E1 ns per leaf, entry or dot product, as labeled. Min/max and all ten retained samples are in the raw JSON; the table keeps the median and CV.

| Case | Median | CV |
|---|---:|---:|
| forest 2^8x2^14: production context | 18.630 ns/leaf | 0.8414% |
| forest 2^8x2^14: split | 18.024 ns/leaf | 1.8789% |
| forest 2^8x2^14: deferred | 13.543 ns/leaf | 2.9995% |
| forest 2^10x2^12: production context | 18.392 ns/leaf | 0.4894% |
| forest 2^10x2^12: split | 17.989 ns/leaf | 0.5206% |
| forest 2^10x2^12: deferred | 13.410 ns/leaf | 0.2989% |
| forest 2^0x2^22: production context | 18.465 ns/leaf | 0.6366% |
| forest 2^0x2^22: split | 18.158 ns/leaf | 1.1618% |
| forest 2^0x2^22: deferred | 13.336 ns/leaf | 0.3977% |
| forest 2^6x2^14: production context | 18.661 ns/leaf | 0.3879% |
| forest 2^6x2^14: split | 18.195 ns/leaf | 1.5487% |
| forest 2^6x2^14: deferred | 13.819 ns/leaf | 0.3720% |
| forest 2^7x2^14: production context | 18.724 ns/leaf | 0.7733% |
| forest 2^7x2^14: split | 18.084 ns/leaf | 0.6910% |
| forest 2^7x2^14: deferred | 13.524 ns/leaf | 0.7709% |
| product level 2^21: production context | 27.553 ns/entry | 1.0916% |
| product level 2^21: split | 17.028 ns/entry | 0.7126% |
| product level 2^21: deferred | 12.450 ns/entry | 0.3530% |
| fraction leaf 2^21 point0: production context | 82.992 ns/entry | 0.7783% |
| fraction leaf 2^21 point0: split | 38.891 ns/entry | 0.8712% |
| fraction leaf 2^21 point0: scale-first | 25.541 ns/entry | 0.8197% |
| fraction leaf 2^21 point0: relation-first | 34.469 ns/entry | 1.1388% |
| fraction generic 2^21 point0: production context | 115.097 ns/entry | 0.5032% |
| fraction generic 2^21 point0: split | 74.728 ns/entry | 1.2465% |
| fraction generic 2^21 point0: scale-first | 52.885 ns/entry | 0.8721% |
| fraction generic 2^21 point0: relation-first | 65.180 ns/entry | 0.7165% |
| dot ext reduced | 3.871 ns/product | 0.5117% |
| dot ext deferred | 0.956 ns/product | 2.9347% |
| dot base reduced | 3.576 ns/product | 1.9813% |
| dot base deferred | 0.536 ns/product | 0.8411% |

Verdict pairs (E1):

| Pair | Saving σ | Required |
|---|---:|---:|
| forest 2^8x2^14 | 24.8615% | ≥5% (IDEA-0030) |
| forest 2^10x2^12 | 25.4571% | ≥5% (IDEA-0030) |
| forest 2^0x2^22 | 26.5535% | ≥5% (IDEA-0030) |
| product level 2^21: deferred | 26.8894% | observation |
| fraction leaf 2^21 point0: scale-first | 34.3254% | ≥15% (leaf), ≥8% (generic) |
| fraction leaf 2^21 point0: relation-first | 11.3690% | ≥3% (IDEA-0030) |
| fraction generic 2^21 point0: scale-first | 29.2302% | ≥15% (leaf), ≥8% (generic) |
| fraction generic 2^21 point0: relation-first | 12.7768% | ≥3% (IDEA-0030) |

The E0 count is [count.py](count.py), reusing EXP-0008's committed denominator rules and asserting exact equality to its fraction and forest totals. [count.json](results/count.json) separates full reduced multiplications, wide products, 192-bit additions, block reductions and extension assembly. This avoids treating an unreduced schoolbook product as a cheaper Karatsuba product. Counts include folds, rebuilt split equality tables, per-round gamma-scaling tables and block-end work; transcript hashing, allocation/page-zeroing and spill traffic remain unpriced, as in EXP-0008.

| E0 case | Split int32 ops/unit | Candidate, 4 / 5 / 8 ops per accumulate |
|---|---:|---:|
| leaf scale-first, per entry | 768.598 | 601.197 / 617.197 / 665.197 |
| leaf relation-first, per entry | 768.598 | 685.037 / 693.037 / 717.037 |
| generic scale-first, per entry | 1324.598 | 1157.197 / 1173.197 / 1221.197 |
| generic relation-first, per entry | 1324.598 | 1241.037 / 1249.037 / 1273.037 |
| 2^8×2^14 forest, per leaf | 419.069 | 357.713 / 363.713 / 381.712 |

For the RFC-0020 forest shape, full per-product `reduce128` operations fall from 15.03775 to 10.03807 per leaf; the candidate adds 5.99961 wide products and 192-bit additions, 0.01304 block accumulator reductions and 0.00338 extension assemblies per leaf. The direct code count predicts 13.21% fewer lane operations at five operations per accumulate, larger than IDEA-0030's historical 5–7% estimate, which did not price the schoolbook replacement's complete removal of the terminal Karatsuba additions/subtractions. The E0 H100 forest interval is 21.39–52.88 ps against split's 25.05–64.03 ps, using EXP-0008's peak/utilization/HBM assumptions and no spill charge; it is not a GPU measurement. The lower bound uses four-operation accumulation and the upper eight with the as-written field costs.

Hopper has no native 192-bit integer add: a 64×64 product is assembled from 32-bit partial products, then added with a carry chain across the accumulator's 32-bit limbs. Four/five/eight operations per accumulator addition are sensitivity assumptions, not measured instruction counts. At the pinned bound the high word has fewer than 32 significant bits, but actual instruction selection, dependency chains and register spills still need PTX/SASS inspection and E2 timing.

Using EXP-0006's existing E1 units, the known reduced-operation component is 17.022 ns/leaf for split and 11.913 ns/leaf for deferred. The latter excludes the new wide/add/reduction units, for which that experiment gives no independent unit cost; it is a partial prediction. The E1 dot observations, 3.871→0.956 ns/product over Fp2 and 3.576→0.536 ns/product for extension-by-base, establish an isolated CPU sign rather than an H100 unit cost.

Cost-model projections (E0) below use the reconciled current `counted` profile: the committed `servable-c2-b32` report, P5 full-reencode planning charge, five regeneration equivalents and `leaf_ps=41.8` after the production split kernel. P4 combination and issue #329's remaining streaming work are unpriced, so these are partial projections, not complete prover bounds. Parameters and all outputs are saved under `results/cost_model_*.{json,txt}` and [projection-parameters.json](results/projection-parameters.json).

- Lane-count route, five-op accumulation: `lookup_ps=89.13157864262408`, `leaf_ps=36.278528089960616`. Accumulation sensitivity gives lookup 87.877–92.894 ps at four/eight ops.
- CPU-ratio route: `lookup_ps=76.76781842995113`, `leaf_ps=31.40790291521472`. These deliberately apply the CPU ratios to planning inputs and **are not GPU predictions**.

| E0 scenario | μ(1K) | μ(4K) | μ(16K) | μ(128K) |
|---|---:|---:|---:|---:|
| reference | 32.294 | 46.786 | 102.444 | 436.310 |
| lookup_count | 31.218 | 45.242 | 99.163 | 423.099 |
| leaf_count | 32.218 | 46.485 | 101.311 | 430.420 |
| both_count | 31.142 | 44.941 | 98.030 | 417.208 |
| both_cpu_ratio | 29.954 | 43.067 | 93.613 | 398.249 |

Concrete reproduction, from the repository root (the original pre-registration's placeholders above are preserved in its frozen copy):

```sh
cargo test --release --locked --manifest-path research/experiments/EXP-0013-delayed-reduction-node-sums/Cargo.toml
cargo run --release --locked --manifest-path research/experiments/EXP-0013-delayed-reduction-node-sums/Cargo.toml -- check
cargo run --release --locked --manifest-path research/experiments/EXP-0013-delayed-reduction-node-sums/Cargo.toml -- all research/experiments/EXP-0013-delayed-reduction-node-sums/results/run.json > research/experiments/EXP-0013-delayed-reduction-node-sums/results/run.txt
python3 -I research/experiments/EXP-0013-delayed-reduction-node-sums/count.py
python3 -I research/experiments/EXP-0013-delayed-reduction-node-sums/analyze.py
cargo fmt --manifest-path research/experiments/EXP-0013-delayed-reduction-node-sums/Cargo.toml --check
cargo clippy --all-targets --locked --manifest-path research/experiments/EXP-0013-delayed-reduction-node-sums/Cargo.toml -- -D warnings
```

The frozen README's SHA256 is `77b3f20ca34128314f715f0770543a602f362a3f9996d6e5b022e26605ad4c27`, committed before code in `8228a19`. The recorded environment is [environment.json](results/environment.json): Apple M4 Max, 48 GiB, macOS 26.6.2 / Darwin 25.6.0, rustc 1.95.0, cargo 1.95.0; production base `726502746cb5a6f29adcdbc7d95b8d2d98735272`. One process, one thread; sibling builds/tests paused during the timing slot. CPU affinity was not enforced.

## Analysis

Scale-first exceeds both written thresholds; relation-first exceeds 3% on both fraction levels and deferred exceeds 5% on all three forests. Every necessary arm meets the exact CV rule, so both cases validate. The dominant CPU improvement survives full proof-message comparison and independent checks, rather than being inferred from a microkernel. The scale-first ordering is materially better than relation-first on both fraction levels, consistent with deferring twice as many products. The kernel's cadence, branches selecting arms and work shapes depend on public dimensions and compile-time arms. Accumulator carries are arithmetic; no condition is selected by a product value in Rust.

The adversarial review is scoped to arithmetic-preserving research code. `Acc192::add/reduce` and `accumulator_boundaries` enforce the bound and canonical result; the independent modular Horner check exercises the high-word sign. `verify_level_arm`, `check_forests` and `check_degenerate` establish all round/final-message and transcript identities under RFC-0004/0010/0014, including degenerate recovery. `verify_views` recomputes public-factor relations in tv-field, and `verify_forests` also discharges the leaf claims against independently evaluated MLEs. There is no new accepting verifier, decoder, commitment, mask, gadget, challenge, witness or normative protocol surface. Inputs are public synthetic data, so local Vec buffers do not hold user secrets; a production port must retain SecretBuf ownership/clearing (OBS-6) and pass binary branchlessness checks (INV-ARCH-2). Research success does not certify binary constant-time behavior.

Threats to validity: M4 CPU results do not establish Hopper performance; a high-word carry chain or register spills can erase the E0 benefit. The int32 translation and utilization are assumptions, not instruction or hardware measurements. The single thread has no affinity guarantee, and the first forest candidate CV is extremely close to the threshold. Arm order is fixed, so thermal drift/cache residency could bias ratios; seeded full-field data and one zero per synthetic tree do not cover model-specific distributions. Production now contains its own split-eq optimization; its context arm is not EXP-0008's textbook baseline and is not the denominator. This is an unmasked level/forest proxy (E1), not an end-to-end zero-knowledge proof (E3). The arithmetic comparisons prove equality for the tested cases and support the exact modular derivation; they do not replace a production soundness/side-channel review.

## Verdict

`validated` (IDEA-0026) and `validated` (the IDEA-0030 second case), under the original numerical criteria. IDEA-0026's current score uses the E0 count-route lookup-only μ(16K) reduction of 3.20286%, confidence 0.8 and cost 1: priority **2.6** (was 2.3 at confidence 0.6). IDEA-0030's independent forest content gives 1.10580%, confidence 0.8, cost 1: priority **0.9** (was 0.4). Fraction ordering gains are alternatives and are not added to each other. The joint count-route reduction is 4.30866% at 16K (102.444→98.030, E0 partial projection), with all current-model omissions retained.

## Follow-ups

Promote the arithmetic-preserving implementation as a draft issue/RFC artifact against the current production split kernels, retaining secret buffers and adding the accumulator bounds and equivalence gates; no production code changed here. On H100, inspect PTX/SASS for carry instructions and spills, measure both fraction orderings and forest shapes under RFC-0020, and determine whether post-split kernels are compute- or memory-bound. Those E2 results settle the four/five/eight-operation accumulation assumption and true c_ℓ/c_π. A production port then needs F65537 calibration, binary branchlessness/clearing checks and full end-to-end masked-proof equivalence before any E3 or deployment claim.
