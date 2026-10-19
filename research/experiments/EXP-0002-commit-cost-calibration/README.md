# EXP-0002: Does an operation-and-byte count, calibrated on the CPU encoder and column chains, exclude the 30 ps assumption for `c_c`?

| | |
|---|---|
| Status | `done: inconclusive` (run 2026-10-16) |
| Idea | [IDEA-0004](../../ideas/IDEA-0004-calibrate-commit-cost.md) |
| Evidence level | E1 for the fit (CPU wall clock here), E0 for the counts and the H100 interval; the absolute `c_c` stays E2 until the RFC-0020 v0.1 gate |
| Hardware | Apple M4 Max (12 performance cores, 128 KiB L1D and 16 MiB L2 per performance cluster), macOS 26.6.2 |
| Pre-registered on | 2026-10-16 |

## Hypothesis

The operation and byte counts of the advice commitment as specified ([RFC-0011 'Encoding'](../../../docs/rfcs/RFC-0011-advice-commitment.md#encoding): inverse NTT of size `k = 2^22`, coset NTT of size `2k`; [RFC-0004 'Column digests'](../../../docs/rfcs/RFC-0004-hashing-and-transcript.md#column-digests): `B = 32` chained BLAKE3 digests), when they are checked against the CPU implementation of those kernels in one harness, place `c_c` on an H100 outside RFC-0020's "proceed" band at every point of the counted interval, that is `ρ_c = c_c / 30 ps > 2` even for the lowest count at the integer pipeline's peak, and the same counts explain the measured CPU kernels within a factor 2.

## Success criterion

Two quantities decide, both defined before any code:

- **The fit `F`.** For each of four kernels, the harness predicts the time per committed element from unit costs it measures itself on in-cache inputs, times the operation counts of `count.py`; `F` passes when every ratio predicted/measured lies in `[0.5, 2]`. The kernels: `NttPlan::intt` at 2^22 points, `NttPlan::coset_ntt` at 2^23 points, `RowEncoder::encode_row` at `k = 2^22` with `P = 200`, and `ColumnChains::absorb_block` over 2^23 columns and `B = 32` rows. The unit costs: nanoseconds per radix-2 butterfly and per point of bit reversal, from a least-squares fit over `NttPlan::ntt` at 2^10, 2^11, 2^12 and 2^13 points (64 KiB at most: inside the 128 KiB L1D); nanoseconds per BLAKE3 compression and per call, from a least-squares fit over `tessera_hash::hash(Domain::ColumnStep, ·)` on inputs of `64·c` bytes, `c ∈ {1, 2, 3, 4, 5, 8, 16}`; nanoseconds per Goldilocks multiplication, from an in-cache loop. The predicted encoder time uses the butterfly cost for all its butterflies and the multiplication cost for the `k` scalings of its `spread` step; the predicted chain time is `(call + 5 compressions) / 16` per committed element, 5 being the compressions of a 288-byte `column-step` input and 16 the committed elements per column per block (`B/2`).
- **The interval `I = [lo, hi]`**, in picoseconds per committed element, from `count.py` alone (E0): `lo = max(lowest operation count / 1.67×10^13 op/s, lowest byte count / 3.35×10^12 B/s)` and `hi = max(highest operation count / (0.6 × 1.67×10^13), highest byte count / (0.8 × 3.35×10^12))`. The rates are the H100 SXM int32 peak (132 SMs × 64 lanes × 1.98 GHz) and HBM3 peak, with the 60% integer utilization that [VYTA25] measured for field kernels and 80% of HBM, exactly as IDEA-0004 step 3 states them. The counts are the radix-2 counts of the code as written at the high end and the radix-4 / lazy-reduction translation at the low end; `count.py` documents each per-operation figure with the Rust source line it translates.

Verdicts:

- `validated`: `F` passes and `lo > 60 ps` (`ρ_c > 2` at the floor). The report then names which of RFC-0020's bands the interval covers, `(2, 4]` and/or `> 4`, and whether the SC-2 ceilings ([00 §7](../../../docs/spec/00-overview.md#7-success-criteria)) are crossed: by the cost model (`python3 -I research/tools/cost_model.py --set commit_ps=…`, corrected profile, run before the measurement), `μ(16K) ≤ 45` needs `c_c ≤ 70 ps` and `μ(1K) ≤ 12` needs `c_c ≤ 82 ps` (E0, primary).
- `refuted`: `F` fails on any kernel (the counts do not explain the code they were calibrated on, so the projection method is void: IDEA-0004's kill criterion), or `lo ≤ 30 ps` (the counts admit the assumption).
- `inconclusive`: `F` passes and `lo ∈ (30, 60]` (the assumption is excluded but the band cannot be named), or a kernel of `F` keeps a coefficient of variation above 3% after one repeat.

Nothing above this line is edited after pre-registration; amendments go below it with a date.

**Amendment, 2026-10-16 (while writing the harness, before any measurement).** The criterion identifies the per-butterfly and per-bit-reversal-point costs by one least-squares fit over `ntt` at 2^10–2^13 points. Over four sizes the two regressors, `(n/2)·log2 n` and `n`, are nearly proportional (their ratio runs only from 5 to 6.5), so that fit is ill-conditioned. The harness instead identifies the butterfly cost on `RowEncoder::encode_row` with `P = 0` at `k ∈ {2^9, …, 2^12}`, a kernel with the same two butterfly loops and no bit-reversal pass (its `k` spread multiplications are subtracted at the measured multiplication cost), and then the bit-reversal cost on `ntt` at 2^10–2^13 given it; the two-regressor fit is still printed as a cross-check. The tolerance, the kernels and the verdict rules are unchanged. Second, the chains' `commitment_id` is the 32 bytes `EXP-0002` zero-padded rather than a digest of that string; it is an opaque input either way.

## Method

- **Harness.** One Rust program, `calib`, its own Cargo workspace in this directory, depending by path on `tessera-field`, `tessera-pcs` and `tessera-hash`: the kernels measured are the production ones, unmodified. There is no candidate implementation: the baseline is the specified kernel, and the "candidate" of the results table is the prediction from counts and unit costs, which the fit `F` judges.
- **Inputs.** Synthetic, from SplitMix64: rows and NTT inputs from seed `0x7E55_E7A0_0047_0401` (the seed of `crates/tessera-pcs/benches/throughput.rs`), mask coefficients from `0x7E55_E7A0_0047_0402`, the hashed block from `0x7E55_E7A0_0047_0404`; every word reduced into `Fp` by `from_u64_reduced`. The `commitment_id` of the chains is the 32 bytes `EXP-0002`, zero-padded.
- **Measurements (E1).** Per committed element on one thread: `intt` 2^22, `coset_ntt` 2^23, `encode_row` (2^22, `P = 200`), `absorb_block` at 2^23 columns and `B = 32`; the unit microbenchmarks listed in the criterion; a streaming copy of 2^23 elements for the memory-bandwidth context. Thread scaling at `T ∈ {1, 2, 4, 8, 12}` for `encode_row` (each thread its own row copy and codeword) and for the chains (columns split into `T` nearly equal shards, one `ColumnChains` per shard). A sweep of `B ∈ {8, 16, 28, 32, 60, 64, 124, 128}` at 2^20 columns; `B ≡ 4 (mod 8)` makes `32 + 8B` a multiple of the 64-byte block, so the sweep also shows what the spec's `B = 32` pays in padding.
- **Protocol** as RFC-0020's gate: 3 warm-up and 10 measured iterations per case; median, minimum, maximum and coefficient of variation; a coefficient above 3% is flagged and that case rerun once. The in-cache microbenchmarks repeat their kernel enough times per iteration that one iteration lasts at least 20 ms.
- **Counts (E0).** `count.py`: butterflies, unit butterflies, multiplications, additions and subtractions per committed element, read from the structure of `crates/tessera-pcs/src/ntt.rs` and `encoding.rs`; compressions per committed element from the 288-byte `column-step` input; bytes through HBM per element from a fused to a four-pass layout; the int32 translation of `Goldilocks::{add, sub, mul}` (`crates/tessera-field/src/goldilocks.rs`), low and high, per source line; and the ARM64 instruction counts of `mul` and of one butterfly, read from the assembly rustc emits for `#[inline(never)]` anchors in the harness (`asm_count.sh`), as the measured input of the translation on the CPU side.
- **Correctness checks**, all before timing, each aborting the run on failure:
  - field: the products `(P − 1)·(P − 1) = 1`, `2^63 · 2^63 = 0xfffffffec0000001` and `(2^32 − 1) · 2^32 = 0xffffffff00000000`, from `spec/vectors/field/goldilocks.json`, through the assembly anchor;
  - encoder: the hand case of `crates/tessera-pcs/src/lib.rs` (`k = 8`, `P = 2`, row `1..=8`, masks 11 and 12: the top half of the codeword's `coset_intt` is `[11, 12, 0, 0, 0, 0, 0, 0]` and the fold modulo `X^8 − 1` evaluates back to the row); at full size, the codeword equals, symbol for symbol, the composition `NttPlan::intt` → mask → `NttPlan::coset_ntt`, an independent path through the bit-reversal permutation and the `Coset` shift; its `coset_intt` is zero from coefficient `k + P` upward; every thread's codeword equals the single-thread one;
  - chains: the case `commitment_id = 3d405d23…` of `spec/vectors/hash/hash.json` (5 rows, 3 columns, `B = 2`, a partial last block) reproduced by `ColumnChains`; for the measured 2^23-column block, 64 sampled columns re-derived from RFC-0004's formula with the incremental `Hasher` (the harness assembles the inputs itself); the same re-derivation inside every shard of the threaded runs; the digest of all 2^23 leaves printed for the record.

## Reproduction

```sh
# from the repository root
cd research/experiments/EXP-0002-commit-cost-calibration
cargo run --release --locked -- all | tee results/run.txt        # first run without --locked writes Cargo.lock
python3 -I count.py | tee results/count.txt
./asm_count.sh | tee results/asm.txt
python3 -I ../../tools/cost_model.py --set commit_ps=<lo> | tee results/cost_model_lo.txt
python3 -I ../../tools/cost_model.py --set commit_ps=<hi> | tee results/cost_model_hi.txt
```

Environment: Apple M4 Max, 16 cores (12 performance, 4 efficiency), 48 GiB; macOS 26.6.2 (Darwin 25.6.0); rustc 1.95.0 (59807616e 2026-04-14), cargo 1.95.0; Python 3.14.6; 2026-10-16. One process, no other load; the harness pins nothing, macOS schedules performance cores first.

## Results

Every correctness check passed before its kernel was timed: the three field vectors through `mul_anchor`; the `k = 8` hand case; `encode_row` equal to `intt` → mask → `coset_ntt` on all 2^23 symbols with degree below `k + P`; the `hash.json` chain case; 64 sampled leaves of the 2^23-column block after 13 absorptions equal to the reference derivation (record digest of all leaves `43080038…`); every thread's codeword equal to the reference and 4 sampled leaves per shard equal to the reference at every thread count. Raw output: [results/run.txt](results/run.txt), [results/run.json](results/run.json), [results/count.txt](results/count.txt), [results/asm.txt](results/asm.txt), [results/preregistration.txt](results/preregistration.txt) (the time and digest of the pre-registered report).

**Unit costs** (E1, primary; one performance core, inputs in L1D; median of 10 after 3 warm-ups, every coefficient of variation below 3%):

| Unit | ns | Detail |
|---|---|---|
| Radix-2 butterfly (`encode_row`, `P = 0`, `k = 2^9…2^12`) | 0.899 | 0.887, 0.899, 0.898, 0.913 at the four sizes |
| Point of bit reversal (`ntt` 2^10…2^13, given the butterfly) | 0.886 | 0.72, 0.81, 0.97, 1.04: grows with size, the tile pass is not free of cache effects even in L1 |
| Goldilocks multiplication, in place by a constant (throughput) | 0.719 | dependent chain: 2.856 ns per multiplication |
| `hash(column-step, ·)`, per call | 5.71 | plus 50.13 ns per compression; the fit's residual is at most 5 ns on 64 to 1,024 bytes; 288 bytes (5 compressions): 255 measured, 256.4 fitted |
| Streaming copy, 2^23 elements, one thread | 0.114 per point | 139.9 GB/s read plus write |

**The four kernels and the fit `F`** (E1, primary, one performance core; nanoseconds per committed element; predicted = counts of `count.py` × unit costs):

| Case | Baseline: measured (median; min; CV) | Candidate: predicted | Ratio |
|---|---|---|---|
| `intt` 2^22 | 11.533 (11.439; 1.62%) | 11 × 0.899 + 0.886 + 0.719 = 11.497 | 0.997 |
| `coset_ntt` 2^23 (2 points) | 28.068 (27.733; 1.14%) | 23 × 0.899 + 2 × 0.886 + 0.719 = 23.174 | 0.826 |
| `encode_row` 2^22, `P = 200` | 32.089 (31.931; 0.41%) | 34 × 0.899 + 0.719 = 31.294 | 0.975 |
| `absorb_block` 2^23 columns, `B = 32` | 19.220 (18.922; 1.05%) | (5.71 + 5 × 50.13) / 16 = 16.022 | 0.834 |

`F` passes: every ratio lies in `[0.83, 1.00]`. The encoder and the chain together cost 51.3 ns per committed element on one core (encoder 62.5%, hash 37.5%), reproducing the triage figures of IDEA-0004 (32.85 and 19.55 ns) within 3%.

**Counts and the H100 interval** (E0, primary, `count.py`; per committed element):

| Quantity | Low (radix-16, lazy reduction) | High (radix-2 as written) |
|---|---|---|
| Full multiplications / cheap multiplications by `2^(12j)` | 16.9 / 18.1 | 32.0 / 0 |
| Additions / subtractions | 34 / 34 | 34 / 34 |
| int32 ops per multiplication / addition / subtraction / cheap multiplication | 18 / 4 / 4 / 8 | 26 / 8 / 5 / – |
| NTT int32 ops | 721 | 1,274 |
| Hash int32 ops (0.3125 compressions × 792 or 960) | 248 | 300 |
| Total int32 ops | 969 | 1,574 |
| HBM bytes (fused / four passes per NTT) | 48 | 216 |
| Compute, ps at the int32 peak / at 60% | 57.9 / 96.5 | 94.1 / 156.8 |
| Memory, ps at the HBM peak / at 80% | 14.3 / 17.9 | 64.5 / 80.6 |

`I = [lo, hi] = [57.9, 156.8]` ps per committed element, `ρ_c ∈ [1.93, 5.23]`; memory never binds. The ARM64 anchors ([results/asm.txt](results/asm.txt)): `mul_anchor` 15 instructions plus `ret` (`mul`, `umulh`, five 64-bit add/sub/cmp, three `csel`, one shift, two constants), `ct_butterfly_anchor` 24 plus a move and `ret`.

**Cost model at the interval's ends** (E0, `cost_model.py`, corrected profile; [results/cost_model_lo.txt](results/cost_model_lo.txt), [results/cost_model_hi.txt](results/cost_model_hi.txt), [results/cost_model_high_at_peak.txt](results/cost_model_high_at_peak.txt)):

| `c_c` (ps) | μ(1K) | μ(16K) | Against SC-2 (12, 45) |
|---|---|---|---|
| 30 (assumed) | 8.9 | 35.1 | holds |
| 57.9 (floor) | 10.6 | 42.1 | holds |
| 94.1 (high count at peak) | 12.8 | 51.1 | fails both |
| 156.8 (ceiling) | 16.6 | 66.7 | fails both |

**`B` sweep** (E1, primary; 2^20 columns, one thread; nanoseconds per symbol):

| `B` | 8 | 16 | 28 | 32 | 60 | 64 | 124 | 128 |
|---|---|---|---|---|---|---|---|---|
| `column-step` bytes → compressions | 96 → 2 | 160 → 3 | 256 → 4 | 288 → 5 | 512 → 8 | 544 → 9 | 1024 → 16 | 1056 → 17 |
| Compressions per symbol | 0.250 | 0.188 | 0.143 | 0.156 | 0.133 | 0.141 | 0.129 | 0.133 |
| ns per symbol | 13.12 | 10.02 | 8.73 | 9.56 | 9.06 | 9.50 | 8.69 | 9.32 |

`B = 28` costs 8.7% less than the specified `B = 32` for 8.6% fewer compressions and a block buffer 12.5% smaller; every `B ≡ 4 (mod 8)` beats its neighbours because `32 + 8B` then fills whole 64-byte blocks.

**Thread scaling** (E1, primary; aggregate nanoseconds per committed element; the 12-thread rows kept a coefficient of variation above 3% after their rerun and are flagged):

| Threads | 1 | 2 | 4 | 8 | 12 |
|---|---|---|---|---|---|
| `encode_row` 2^22 | 32.39 | 16.43 (1.97×) | 8.39 (3.86×) | 4.37 (7.41×) | 3.20 (10.1×; CV 10.4%, flagged) |
| `absorb_block` 2^23 columns, sharded | 19.89 | 10.11 (1.97×) | 5.10 (3.90×) | 2.63 (7.57×) | 1.89 (10.5×; CV 23.2%, flagged) |

Across 12 performance cores the commitment costs about 5.1 ns per committed element (E1), 170× the 30 ps assumption; a CPU figure, context only (research rule 4).

## Analysis

**Against the criterion.** `F` passes on all four kernels, so the counts of `count.py` describe the code they were calibrated on, and IDEA-0004's kill clause ("the counts do not explain the ARM timings within 2×") did not fire. The floor of the interval, `lo = 57.9` ps, lies in `(30, 60]`: the pre-registered rule makes the verdict `inconclusive`, because the counts alone cannot say whether RFC-0020's "proceed" branch (`ρ_c ≤ 2`) is excluded. They do say three other things. The 30 ps assumption of [08 §1](../../../docs/spec/08-performance-budget.md#1-cost-model) is 1.9× below a floor that assumes a radix-16, lazily reduced kernel at 100% of the integer pipeline with the hash at its counted minimum. The SC-2 ceiling at 16K (`c_c ≤ 70` ps) holds only below the "high count at peak" point: a kernel of the CPU's radix-2 structure must run above 75% of peak to hold it, and at the 60% utilization of [VYTA25] both ends fail both ceilings (μ(16K) 52 to 67). And the interval spans RFC-0020's three bands, so the v0.1 measurement, not a tighter count, names the branch.

**What the residuals mean.** `intt` and `encode_row` are predicted within 3% from in-cache unit costs, so the out-of-cache levels of a 2^22- or 2^23-point transform cost almost nothing extra on this machine (the stream copy runs at 140 GB/s on one core, and the two or three levels above the 16 MiB L2 move 128 MiB each in about a millisecond of the 135 ms encode). The 17% the prediction misses on `coset_ntt` is what the standalone coset kernel pays and the encoder avoids: a shift multiplication per group, no unit butterflies, and a second bit-reversal point per element; a GPU kernel should fold the coset into the coefficient scaling as `RowEncoder` does. The 17% missed on the chain (3.2 ns per element) is the gather of 32 strided symbols per column and the assembly of the 288-byte input, not hashing: the sweep shows the time per symbol tracking compressions per symbol at about 61 ns per compression for `B ≤ 32` and 67 to 70 ns for `B ≥ 60`, against 50 ns in cache.

**What the CPU does and does not say about the GPU.** The fit validates the counts (butterflies, multiplications, compressions, bytes), which are hardware-independent; it says nothing about int32 ops per field operation or about utilization, and those two factors are the whole width of the interval (1.6× and 1.67×). The ARM anchors give a check on the translation's input: 15 instructions per multiplication, of which `mul` and `umulh` become four 32×32→64 products with carried adds and the five 64-bit adds, subtracts and compares become ten; a straight transliteration is about 34 int32 ops, above the high end's 26, so the high end is optimistic for a naive port (about 1,830 ops, 182 ps at 60%; the same band).

Threats to validity:

- CPU against GPU: the per-operation int32 figures are hand counts from the Rust source with stated low and high translations, not measured SASS; the 64 int32 ops per clock per SM and the 1.98 GHz clock are documentation figures (secondary). If wide products issue on the FMA pipe concurrently with integer adds, the floor drops to about 53 ps (the wide products are 85 of the 969 low ops); the verdict band does not change.
- Utilization: 60% is [VYTA25]'s figure for a 377-bit-field multiplication kernel on an A40, applied here to Goldilocks NTT levels and to BLAKE3 at 288-byte inputs, neither of which it measured; the ceiling of the interval is soft in both directions.
- Operation counts against wall clock: the counts assume the hash costs its ALU operations and nothing else (message loads and state moves are free), and that the radix-16 kernel's shared-memory passes are free; both favour the floor.
- Memory: the byte model (48 to 216 bytes per element) is E0; on this CPU memory never bound at 64 to 128 MiB working sets, consistent with HBM not binding on the GPU at any end (14 to 81 ps), but a GPU kernel with more passes than four per NTT would move the memory floor up.
- Input distribution: uniform rows and symbols; control flow is data-independent (INV-ARCH-2), so the timings are distribution-free, and the only data-dependent cost, canonical-form corrections, is counted at its worst case.
- The 12-thread rows are flagged (coefficients of variation 10% and 23% after a rerun): macOS schedules the 12 threads against its own work and 4 efficiency cores; the 1-to-8-thread rows are clean and scale at 93 to 99% efficiency.
- One machine, one day, one toolchain; the triage figures measured in another session agree within 3%.

## Verdict

`inconclusive`. IDEA-0004 moves to `inconclusive`: its method is validated (the counts explain the CPU kernels within 2×, the kill criterion did not fire), its deliverable is `c_c ∈ [58, 157]` ps on an H100 (E0, `ρ_c ∈ [1.9, 5.2]`), and that interval covers every band of RFC-0020's decision rule, so the branch stays with the v0.1 measurement. The idea's confidence falls from 0.6 to 0.5 (the interval cannot be narrowed from a CPU) and its priority as an experiment here to 0: nothing E1 can settle remains. The 30 ps assumption is excluded at every point of the interval, and the SC-2 ceilings hold only for a near-peak kernel; both are inputs to RFC-0020's gate, not changes to the design.

## Follow-ups

- **E2 (RFC-0020 v0.1 gate).** The `c_c` microbenchmark as specified, at 2^34 or more committed elements, with Nsight's int32-pipe utilization and the kernel's radix recorded: the two parameters that span the interval. The pre-registered prediction is `[58, 157]` ps; a measurement inside it validates the method, and the measured point names the branch (proceed below 60 ps, re-plan to 120, redesign above).
- **Survey (hashing and commitments).** Primary H100 figures for a batched Goldilocks NTT of 2^23 points (ICICLE, sppark) and for BLAKE3 at 288-byte inputs, to replace the borrowed 60% utilization; a figure at or below 30 ps per committed element is IDEA-0004's kill criterion.
- **Ideation: `B ≡ 4 (mod 8)`.** `B = 28` removes the padding compression of the `column-step` input: 8.6% fewer compressions (E0), 8.7% less chain time (E1), a block buffer 12.5% smaller, about −2% of `c_c` at the hash's 19–26% share (E0). `B` is a public commitment parameter of [RFC-0004](../../../docs/rfcs/RFC-0004-hashing-and-transcript.md#column-digests) and [RFC-0011](../../../docs/rfcs/RFC-0011-advice-commitment.md#encoding), so it changes through an RFC; small, free, and worth an idea note.
- **Kernel note for RFC-0011's coset NTT.** Fold the coset shift into the coefficient scaling as `RowEncoder` does; the standalone coset kernel costs 17% more per butterfly on this CPU (E1).
- **Input to IDEA-0003 (fused encode–hash tiles).** At the counted operation counts, HBM bytes do not bind at either end (14 to 81 ps against 58 to 157 ps of compute), so fusion's saving in bytes does not move `c_c` unless the kernel is far from the compute floor; its gain must be argued from the integer pipeline (E0).
- **IDEA-0008 (`c_π`).** The same harness pattern, with the product-tree leaf as the kernel.
