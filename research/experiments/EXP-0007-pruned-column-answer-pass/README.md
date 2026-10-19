# EXP-0007: Is an output-pruned DIF coset NTT at least 2× faster per row than a full one for the column-answer pass?

| | |
|---|---|
| Status | `done: validated` (run 2026-10-16) |
| Idea | [IDEA-0016](../../ideas/IDEA-0016-pruned-column-answer-pass.md) |
| Evidence level | E1 (CPU wall clock here, one Apple M4 Max performance core) for the ratios; E0 for the operation counts and the cost-model translation; the HBM behaviour of the full levels on the GPU stays E2 |
| Hardware | Apple M4 Max (12 performance cores, 128 KiB L1D and 16 MiB L2 per performance cluster), macOS 26.6.2 |
| Pre-registered on | 2026-10-16 |

## Hypothesis

In one harness on one M4 Max performance core, an output-pruned decimation-in-frequency coset NTT that computes only the `τ` queried symbols of a committed row (RFC-0011 'Opening protocol' step 7, the P5 column-answer pass of IDEA-0016) is at least 2× faster per row than a full coset NTT of size `n = 2^23` plus a gather in coefficient form, and at least 1.4× faster in today's evaluation form (both arms paying the size-`k` interpolation), at `k = 2^22`, `P = 200` and `τ ∈ {130, 163}` uniformly drawn distinct positions, while its `τ` symbols equal the production encoder's bit for bit.

## Success criterion

Written before any code. Six arms run in the same program on the same inputs; the two baselines are the full transforms, the two candidates the pruned ones, and the other two give context.

Coefficient form (IDEA-0002's basis: the row's `k` values are the coefficients `f_0, …, f_{k−1}`, and `g = f + X^k·m`):

- `full_coef` (baseline): the coefficient buffer `a_i = g_i·7^i` of size `n` is built, the harness's own radix-2 DIF kernel (the remainder tree of `crates/tessera-pcs/src/ntt.rs`, natural-order coefficients in, bit-reversed evaluations out) runs all `log2 n = 23` levels, and the `τ` symbols are gathered from the leaves `rev_n(j)`.
- `pruned_coef` (candidate): the same buffer and kernel, pruned by output: a group whose leaves hold no queried position is skipped; a group holding exactly one is descended one-sidedly (`f_lo ± c·f_hi` for the half that holds it, `m/2` multiply-adds per level, into a scratch buffer and then in place), down to the symbol; a group holding two or more runs its full butterfly level and recurses into both halves. The standard output-pruned DIF of [SB93], choosing its depth per branch.
- `fixed8_coef` (the idea note's accounting, secondary): `s = 8` full levels over every group, then one one-sided descent per query from its group of size `2^15`.
- `coset_ntt_coef` (context): the production `NttPlan::coset_ntt(g, 7)` (Cooley–Tukey kernel plus bit-reversal permutation) and a gather: today's kernel, and the correctness reference of the coefficient form.

Evaluation form (RFC-0011 'Encoding' today: the row's values are `f` on `H`, `g = f + Z_H·m`):

- `encode_row_eval` (baseline): the production `RowEncoder::encode_row` and a gather; the correctness reference of the evaluation form.
- `pruned_eval` (candidate): the production `NttPlan::intt` on the row (natural-order coefficients of `f`), the mask applied as RFC-0011 defines it (`−m_i` at `i < P`, `m_i` at `k + i`), the buffer `a_i = g_i·7^i`, then the pruned DIF of `pruned_coef`. The permutation inside `intt` (about 4 ms at `2^22`, EXP-0002's 0.886 ns per point) is charged to the candidate: a conservative choice, since a pruned pass on bit-reversed storage would avoid it.

The quantities: `R_coef(τ) = T(full_coef) / T(pruned_coef)` and `R_eval(τ) = T(encode_row_eval) / T(pruned_eval)`, medians over the measured iterations, on the uniform query sets.

Verdicts:

- `validated`: `R_coef ≥ 2.0` and `R_eval ≥ 1.4` at both `τ = 130` and `τ = 163`; every correctness check below passes; every arm's coefficient of variation at both `τ` is at most 3% after at most one rerun.
- `refuted`: `R_coef < 1.5` at either `τ` (IDEA-0016's kill criterion for the kernel).
- `inconclusive`: anything else: `1.5 ≤ R_coef < 2.0` at either `τ`; `R_coef ≥ 2.0` at both but `R_eval < 1.4` at either; or a coefficient of variation above 3% after the rerun on an arm a ratio needs.

A candidate symbol that differs from the reference is a defect in the harness, fixed and rerun before any number is read (the skill's rules of evidence), never a verdict.

Secondary quantities, reported but not judged: the harness's exact operation counts (butterflies, multiply-adds) per committed element for every arm, against the note's 9.3 butterfly-equivalents for the fixed-`s` variant and 23 for the full transform (E0); the same six arms on two adversarial query sets a live verifier could choose, `clustered` (every index in one residue class modulo 256) and `paired` (pairs `(j, j + n/2)`, which share a group down to size 2); the translation to the cost model under the `counted` profile, with the P5 encode term `73 ps × T(arm) / T(encode_row_eval)` added to `commit_ps = 94` and `regen_full_equivalents = 5`, against the honest baselines of the same form (`commit_ps` 167 in evaluation form, 143 in coefficient form, from the triage of IDEA-0016).

Nothing above this line is edited after pre-registration; amendments go below it with a date.

**Amendments, 2026-10-16 (after the first complete run, before the verdict was written; no criterion changes).** (1) The operation counters live inside the kernels (one 64-bit addition per level per group, nothing per butterfly), so the counts reported are those of each arm's pre-timing correctness run on the same inputs, and the counters also tick during timing; the pre-registration said "a separate counting run". (2) One context row was added, `build_coef` alone (the `k + P` multiplications and the zero fill that every coefficient-form arm but `coset_ntt_coef` pays inside its timed region), so that the analysis can separate the buffer from the transform. The six pre-registered arms and the verdict rule are unchanged; the first run's medians agreed with the reported second run within 1% on every arm (the first run's output was overwritten by the second in `results/run.txt`).

## Method

- **Harness.** One Rust program, `prune`, its own Cargo workspace in this directory (empty `[workspace]` table), `#![forbid(unsafe_code)]`, depending by path on `tessera-field` (the Goldilocks arithmetic), `tessera-pcs` (`NttPlan`, `RowEncoder`, `COSET_SHIFT`: the production kernels of the two baselines and the references) and `tessera-hash` (the record digest of each arm's symbols). The DIF kernel and its pruning are written in the harness, because `tessera-pcs` keeps its kernels crate-private; the full-transform arm of that kernel is checked against the production `coset_ntt` on all `2^23` symbols before anything is timed, so the baseline and the candidate share one kernel and the baseline reproduces the production kernel exactly.
- **Inputs.** Synthetic, SplitMix64: the row from seed `0x7E55_E7A0_0047_0401` (the seed of `crates/tessera-pcs/benches/throughput.rs` and EXP-0002), `P = 200` mask coefficients from `0x7E55_E7A0_0047_0402`, the queries from `0x7E55_E7A0_0047_0407`. Uniform set: `τ` distinct indices in `[0, n)`, repeats rejected, sorted. Clustered set: `r + 256·t` for `t < τ`, `r` the first draw modulo 256. Paired set: `⌊τ/2⌋` distinct draws `j_t < n/2`, each with `j_t + n/2`, and one more unpaired draw when `τ` is odd. `τ ∈ {130, 163}`.
- **Protocol** as RFC-0020's gate and EXP-0002: 3 warm-ups and 10 measured iterations per arm and query set; median, minimum, maximum and coefficient of variation; a coefficient above 3% reruns the arm once, and the rerun is what is reported, flagged if still above. The coefficient buffer is rebuilt inside the timed region of every arm that uses it (P5 builds it per row), and every iteration starts from the same inputs. Times are reported per row in milliseconds and per committed element in nanoseconds (divided by `k = 2^22`).
- **Correctness checks**, each aborting the run on failure:
  - the hand case of `crates/tessera-pcs/src/lib.rs` (`k = 8`, `P = 2`, row `1..=8`, masks 11, 12): the pruned pass in evaluation form equals `encode_row`'s codeword at all 16 positions and at the subsets `{0}`, `{3, 11}`, `{1, 5, 9, 13}`, `{2, 10}` (a pair `(j, j + n/2)`); in coefficient form it equals `coset_ntt(g, 7)` with `g = (f_0, …, f_7, 11, 12, 0, …, 0)` at the same subsets;
  - at full size, `full_coef`'s `2^23` leaves equal `coset_ntt(g, 7)` symbol for symbol, in the bit-reversed order the kernel leaves them;
  - at full size, for every `τ` and query set, `pruned_coef` and `fixed8_coef` equal `coset_ntt(g, 7)` at the queried positions and `pruned_eval` equals `encode_row` there, bit for bit; the record digest `tessera_hash::hash(Domain::ColumnMerge, ·)` over the canonical little-endian symbols is printed per arm.
- **Counts (E0).** The kernel counts its butterflies and multiply-adds as it runs (counters outside the timed iterations, from a separate counting run on the same inputs), and the report divides by `k`.

## Reproduction

```sh
# from the repository root
cd research/experiments/EXP-0007-pruned-column-answer-pass
cargo run --release --locked | tee results/run.txt        # first run without --locked writes Cargo.lock
# commit_ps = 94 + 73 × T(arm) / T(encode_row_eval), τ = 163, uniform: 167 and 143 are the triage's honest baselines
for c in 167 143 146 136 110; do
  python3 -I ../../tools/cost_model.py --profile counted --set regen_full_equivalents=5 --set commit_ps=$c
done                                                      # saved as results/cost_model_*.txt
shasum -a 256 README.md                                   # the pre-registered text: results/preregistration.txt
```

Environment: Apple M4 Max, 16 cores (12 performance, 4 efficiency), 48 GiB; macOS 26.6.2 (Darwin 25.6.0); rustc 1.95.0 (59807616e 2026-04-14), cargo 1.95.0; Python 3.14.6; 2026-10-16. One process, one thread, no other load; macOS schedules performance cores first.

## Results

Every correctness check passed before and after its arm was timed: the `k = 8` hand case in both forms at all five position sets; the harness kernel equal to `coset_ntt(g, 7)` on all `2^23` leaves (96,468,992 butterflies, 23.0 per committed element); every candidate equal to its reference at the queried positions for both `τ` and all three query sets (36 arm runs, same record digest as the reference's gather in every one: for example `737cb136…` for the coefficient form and `b5d53ce7…` for the evaluation form at `τ = 163`, uniform). No arm's coefficient of variation exceeded 1.8%, so no rerun was needed. Raw output: [results/run.txt](results/run.txt); the pre-registered text's digest and time: [results/preregistration.txt](results/preregistration.txt).

**The verdict's table** (E1, primary; one performance core; medians of 10 after 3 warm-ups, milliseconds per row with nanoseconds per committed element and the coefficient of variation; uniform queries):

| Case | Baseline | Candidate | Ratio |
|---|---|---|---|
| Coefficient form, `τ = 130` | `full_coef` 99.79 ms (23.79 ns/el; CV 0.20%) | `pruned_coef` 30.10 ms (7.18 ns/el; CV 0.30%) | `R_coef` = 3.32 |
| Coefficient form, `τ = 163` | `full_coef` 99.78 ms (23.79; CV 0.27%) | `pruned_coef` 31.29 ms (7.46; CV 0.51%) | `R_coef` = 3.19 |
| Evaluation form, `τ = 130` | `encode_row_eval` 139.56 ms (33.28; CV 0.07%) | `pruned_eval` 80.10 ms (19.10; CV 0.34%) | `R_eval` = 1.74 |
| Evaluation form, `τ = 163` | `encode_row_eval` 139.70 ms (33.31; CV 0.29%) | `pruned_eval` 80.89 ms (19.29; CV 0.28%) | `R_eval` = 1.73 |

**Secondary arms, uniform queries** (E1, primary): `fixed8_coef` 32.69 / 33.40 ms at `τ = 130 / 163` (7.79 / 7.96 ns/el; `full_coef / fixed8_coef` = 3.05 / 2.99); `coset_ntt_coef` 120.76 / 120.62 ms (28.79 / 28.76 ns/el; today's production kernel, `coset_ntt_coef / pruned_coef` = 4.01 / 3.86); `encode_row_eval / pruned_coef` = 4.64 / 4.47 (today's encoder against the coefficient-form pruned pass: what IDEA-0002 and this kernel give together). `build_coef` alone: 3.245 ms (0.774 ns/el; CV 1.14%).

**Operation counts** (E0, primary, the harness's counters; per committed element, `bf` butterflies and `ma` multiply-adds):

| Arm | `τ = 130` | `τ = 163` |
|---|---|---|
| `full_coef` | 23.000 bf | 23.000 bf |
| `pruned_coef`, uniform | 6.880 bf + 1.614 ma = 8.49 | 7.328 bf + 1.434 ma = 8.76 |
| `fixed8_coef` (any set) | 8.000 bf + 1.016 ma = 9.02 | 8.000 bf + 1.273 ma = 9.27 (the note's 9.3) |
| `pruned_coef`, clustered | 2.020 bf + 0.008 ma = 2.03 | 2.021 bf + 0.008 ma = 2.03 |
| `pruned_coef`, paired | 7.203 bf | 7.604 bf |

**Adversarial query sets** (E1, primary; milliseconds per row, medians):

| Set, `τ` | `pruned_coef` | `R_coef` | `fixed8_coef` | `pruned_eval` | `R_eval` |
|---|---|---|---|---|---|
| clustered, 130 | 8.72 (2.08 ns/el) | 11.44 | 32.59 | 57.99 | 2.41 |
| clustered, 163 | 8.72 (2.08) | 11.44 | 33.29 | 58.25 | 2.40 |
| paired, 130 | 26.36 (6.28) | 3.79 | 32.59 | 75.96 | 1.83 |
| paired, 163 | 28.10 (6.70) | 3.55 | 33.32 | 77.98 | 1.79 |

**Cost model** (E0, `tools/cost_model.py`, `counted` profile, `regen_full_equivalents = 5`; `commit_ps = 94 + 73 × T(arm) / T(encode_row_eval)` at `τ = 163`, uniform; files `results/cost_model_*.txt`):

| Scenario | `commit_ps` | μ(1K) | μ(4K) | μ(16K) | μ(128K) | Against the honest baseline of the same form |
|---|---|---|---|---|---|---|
| Honest, evaluation form (P5 priced, full re-encode) | 167 | 22.8 | 38.4 | 98.0 | 454.4 | — |
| Pruned, evaluation form | 136 (94 + 73 × 80.89/139.70) | 20.9 | 35.2 | 90.2 | 420.4 | −8.3% / −8.3% / −8.0% / −7.5% |
| Honest, coefficient form (full transform, measured ratio) | 146 (94 + 73 × 99.78/139.70) | 21.5 | 36.2 | 92.7 | 431.4 | — (the triage's count-based 143: 21.3 / 35.9 / 92.0 / 428.1) |
| Pruned, coefficient form | 110 (94 + 73 × 31.29/139.70) | 19.3 | 32.6 | 83.8 | 391.9 | −10.2% / −9.9% / −9.6% / −9.2% (−8.9% at 16K against 143) |

Against today's optimistic model (`regen` 4, `commit_ps` 94: 17.3 / 30.0 / 78.8 / 373.4) the priced-and-pruned P5 is +20.5% / +17.5% / +14.5% / +12.6% in evaluation form and +11.4% / +8.7% / +6.3% / +5.0% in coefficient form.

## Analysis

**Against the criterion.** `R_coef` = 3.32 and 3.19 at `τ = 130` and 163, both at least 2.0; `R_eval` = 1.74 and 1.73, both at least 1.4; every check passed; every coefficient of variation is below 3% without a rerun. The pre-registered rule returns `validated`, and the kill criterion (below 1.5× in coefficient form) is missed by a factor above two.

**Why the time ratio exceeds the count ratio.** The counts give 23 / 8.76 = 2.63× at `τ = 163`; the clock gives 3.19×. The full kernel pays 1.00 ns per butterfly over its 23 levels (96.5 ms without the 3.2 ms buffer build), of which twelve are leaf levels iterated over short in-cache chunks; the pruned kernel pays 0.76 ns per operation (28.0 ms for 8.76 operations per element), since its work is five to seven large-stride levels with long inner loops and descents that are long multiply-add loops with one twiddle per level. Both inherit the structure of the production kernel (`ct_group`/`ct_leaf` of `crates/tessera-pcs/src/ntt.rs`), and the harness's full kernel is 18% faster than the production `coset_ntt` (23.0 against 28.8 ns/el) for the two reasons EXP-0002 named (no bit-reversal permutation, no per-group shift multiplication), so the baseline is strong and the ratio against today's code is higher still (3.9–4.0×). Without the buffer build, which both coefficient-form arms pay, `R_coef` is 3.60 / 3.44.

**Adaptive against fixed depth.** The per-branch rule (skip, descend, or split) beats the note's fixed eight levels by 6–8% on uniform queries (30.1 against 32.7 ms; 31.3 against 33.4) and does 8.49 / 8.76 operations per element against 9.02 / 9.27, the latter matching the note's 9.3 exactly (`8 + 163 × 2^15 / 2^22 = 9.27`). The kill criterion's worry, that clustered residues modulo `2^s` force sub-transforms to be finished in full, does not materialize: for the fixed-depth variant clustering is neutral (eight full levels whatever the queries, then `τ` descents of `2^15 − 1` multiply-adds), and for the adaptive rule it is a gain, because a cluster in one residue class is descended one-sidedly for eight levels and then transformed inside a group of `2^15` (2.03 operations per element, 11.4×). A live verifier cannot slow P5 by its choice of indices: the adaptive cost is bounded by `⌈log2 τ⌉` full levels plus `τ` descents, and the worst set found, pairs `(j, j + n/2)` that share a group down to size 2, costs 7.6 butterflies per element and still runs at 3.55×.

**Evaluation form.** `pruned_eval` is the production `intt` (about 49.6 ms by subtraction, consistent with EXP-0002's 11.53 ns per point = 48.4 ms, of which about 3.7 ms is the bit-reversal permutation a chained kernel would avoid), the buffer and the pruned kernel; the interpolation is 61% of the candidate's time, which caps `R_eval` near 1.7 and is why the coefficient form of IDEA-0002 matters for P5 as much as for P1: today's encoder against the coefficient-form pruned pass is 4.5–4.6×.

**Translation to `μ`.** Under the `counted` profile the pruned pass makes P5's encode-only term 16.4 ps per committed element in coefficient form and 42.3 ps in evaluation form, against 52.1 and 73 ps for a full re-encode of the same form; μ(16K) falls 9.6% (coefficient form) and 8.0% (evaluation form) against the honest baselines, with proof size and verification unchanged (no message or check changes). The triage's estimates (−8.2% and −7.7%) were conservative by 1.4 and 0.3 points. The pricing finding of IDEA-0016 is untouched by this experiment and stands on the RFC texts: P5 exists and costs a regeneration plus the encode work above.

Threats to validity:

- **CPU against GPU, the main one.** On this CPU no level is memory-bound (one core streams at 140 GB/s, and the eight large-stride levels cost the same per butterfly as the in-cache ones), so the ratio measures operations and loop structure. On an H100 the pruned pass keeps exactly the levels that touch HBM (strides `2^22 … 2^15`) and removes the ones a fused kernel runs in shared memory. By E0 arithmetic the pruned pass moves about 32 B per committed element (one read-and-write pass over the 64 MB row buffer: about 10 ps at the HBM peak, 12 at 80%) and does about 8.8 operations per element (about 175 int32 operations at EXP-0002's translation, about 17.5 ps at 60% utilization), so it is near balance; the full transform does 23 operations per element, about 46 ps at the same rates. The GPU ratio therefore lies between about 1.1× (a kernel that is HBM-bound on the eight levels and in-cache otherwise) and about 2.9× (compute-bound), and only the RFC-0020 runner says where: E2.
- **Operation counts against wall clock.** The per-operation costs differ between arms (1.00 against 0.76 ns); the count ratio, 2.6×, is the part that carries to another machine, and the 3.2× includes the loop-structure effect.
- **Cache.** The 64 MiB row buffer exceeds the 16 MiB L2; the full levels stream, the descents' scratch (at most 128 KiB from a `2^15` group) and the paired set's deep full paths stay in L1 or L2. Both arms see the same buffer.
- **Input distribution.** Control flow is independent of the values (INV-ARCH-2), so the timings are distribution-free in the row and masks; the queries do shape the pruned kernel, and three sets bracket them (uniform, the protocol's draw; clustered, the best case; paired, the worst found).
- **The baseline is the harness's kernel**, not the production one, validated bit for bit on all `2^23` symbols; it is faster than the production `coset_ntt`, so the ratio against today's code would be higher, not lower.
- **One machine, one day, one toolchain.** The production kernels agree with EXP-0002 within 4% (`coset_ntt` 28.8 against 28.1 ns/el, `encode_row` 33.3 against 32.1).

## Verdict

`validated`. IDEA-0016 moves from `triaged` to `validated`. Its score changes to: gain −9.6% μ(16K) in coefficient form against the honest coefficient-form baseline (92.7 → 83.8, `counted`) and −8.0% in evaluation form (98.0 → 90.2), E0 from E1 ratios measured here, the kernel's own count 8.76 operations per committed element (E0, exact) against 23; confidence 0.6 → 0.7 (the CPU ratio is measured at 3.2× with exact counts and bit-exact symbols; what stays E2 is the HBM behaviour of the eight full levels, which bounds the GPU ratio between 1.1× and 2.9×); cost `s` unchanged; priority 9.6 × 0.7 / 1 = 6.7 (5.6 in evaluation form). Next action: `promote` (the pricing issue against RFC-0018, 08 §1 and RFC-0020, which stands on the texts, and a kernel note for RFC-0018 'Kernels' with the adaptive rule and its bound).

## Follow-ups

- **Promote IDEA-0016.** File the pricing issue (a P5 row in RFC-0018 'Pass structure', `n_regen` 4 → 5 in 08 §1, an encode-only P5 term in RFC-0020's executable cost model and `tools/cost_model.py`), and a kernel note under RFC-0018 'Kernels': the output-pruned DIF with the per-branch rule (skip, one-sided descent, split), its bound of `⌈log2 τ⌉` full levels plus `τ` descents, the self-check of RFC-0011 'Failure modes' as its correctness gate, and the measured 8.76 operations per committed element (E0) against 23. Re-read the ledger's P4 → 0 seed with P5 as the floor of opening-time regeneration.
- **E2 (RFC-0020 runner).** P5 per row on the H100: the eight large-stride levels against the full transform's HBM passes, with Nsight's DRAM throughput and int32 utilization recorded. The pre-registered prediction from this report is a ratio between 1.1× and 2.9×; a measurement inside it validates the E0 bound, and the point names whether P5's encode term is memory- or compute-priced.
- **Batching across rows.** P5 answers the same `τ` indices for every row, so the query routing and the descent twiddles are one plan shared by every row of a block; the harness ran one row at a time. A `B`-rows-per-block kernel (as P1's tiles) amortizes the plan and is the shape the E2 run should take.
- **Chained evaluation form.** A pruned pass on bit-reversed storage (the output order of the encoder's interpolation step) avoids `intt`'s permutation, about 3.7 ms per row here (E1, EXP-0002's unit cost): `R_eval` from 1.73 to about 1.82. Small; worth doing only if IDEA-0002's coefficient form is not adopted.
- **Input to IDEA-0002.** The coefficient form removes the interpolation from P5 as well as from P1: today's encoder against the coefficient-form pruned pass is 4.5–4.6× (E1), so IDEA-0002's gain in `μ` should be re-scored with P5 priced (two passes lose the size-`k` interpolation, not one).
