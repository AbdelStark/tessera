# IDEA-0006: Fold-sum round fusion sized by the HBM roofline, not deeper

| | |
|---|---|
| Status | `shelved` (H100) (2026-10-16): the decisive measurement is an unfused/fused kernel pair on an H100 (E2, the RFC-0020 v0.1 gate); the consolidation decision is recorded in [LEDGER.md](../LEDGER.md) |
| Origin | [SUR-0002](../surveys/SUR-0002-lookups.md) candidate ideas "Fused fraction-tree kernel: eight sumcheck rounds per kernel, level resident in shared memory" and "Univariate skip on the first `k` variables", and its finding [E0-this]; [SUR-0003](../surveys/SUR-0003-passes.md) finding [DMABKGR25] (sumcheck is bandwidth-bound, each round rewrites the table); RISK-10 ([00 §9](../../docs/spec/00-overview.md#9-risks)) |
| Lever | `c_ℓ` (its memory floor); `p_s` for the streamed leaf levels as a secondary effect |
| Spec surfaces | RFC-0018 'Pass structure' and 'Fusion' (the kernel shape of the fraction-tree level sumcheck; tier C), RFC-0020 'Measurement protocol' (the v0.1 unit-cost benchmark gains an unfused/fused kernel pair; tier C). No tier-A surface: the messages, their degree and the label schedule `tree/{ℓ}/round/{j}`, `tree/{ℓ}/r/{j}` of RFC-0014 are unchanged. The univariate-skip variant discussed below *would* change the tier-A message schema and is excluded from this idea, not proposed. |
| Priority | 0 (gain against the baseline is 0; −15.5% only against a hypothetical unfused reference) |

## Mechanism

Each fraction-tree level of RFC-0008 is proven by one degree-3 sumcheck (RFC-0008 'Fraction tree': "Each tree level is proven by a sumcheck of degree 3 per variable"). Round `i` of that sumcheck reads the level's `(n, d)` table, 32 B per entry in `Fp2` (two 16-byte elements), to form the round message, and after the verifier's challenge `r_i` is squeezed (`tree/{ℓ}/r/{i}`) the fold binds variable `i` and writes a table of half the size. The textbook kernel therefore makes three memory passes per round over the live table: read for the sums, read again for the fold, write the folded half.

The idea has three parts.

**(a) Adopt the two-round fold-sum fusion as the v1 kernel shape for every in-memory level.** One kernel performs the fold of round `i` (binding `r_i`) and, on the folded entries while they are still in registers, accumulates the two-point evaluation sums of round `i + 1`; the table is read once and written once per round instead of read twice and written once. This is the shape SP1's production CUDA ships (`two_round_fix_and_sum_*`, `logup_gkr_two_round_sum_*` in `lookahead.cuh`; [SP1GPU26], E0, primary, code read in SUR-0002). It needs no extra transcript message and no change of challenge order: round `i + 1`'s sums are computed after `r_i` is squeezed and sent only when complete, so INV-FS-1 holds exactly as in the unfused kernel. Bytes through HBM per lookup over the whole tree fall from 288 B to 192 B (SUR-0002 [E0-this], E0, primary, own count).

**(b) Fix the fusion depth from the roofline, `c_ℓ = max(compute floor, bytes / 3.35 TB/s)`, not from the byte count alone.** Fusion lowers only the memory floor. Once the memory floor is below the compute floor, deeper fusion buys nothing; the lever then moves to the `Fp2` multiplication (instruction count per Goldilocks multiplication, int32-pipe utilization), which is a different idea.

**(c) Treat the eight-round resident-level seed of SUR-0002 as refuted at the E0 level**, for two independent reasons. First, transcript legality: round `i + 1` cannot start before `r_i`, and `r_i` is derived from the *global* sum of round `i` over the whole level, so `k` fused rounds need `k` grid-wide synchronizations with the level resident on chip. H100 SXM aggregate shared memory is 228 KB × 132 SMs ≈ 30 MB (NVIDIA Hopper datasheet, secondary, vendor documentation; the figure is not in the corpus's [NVIDIA-H100] row, which gives HBM and INT8 only); the attention leaf level of a 405B, 16K-token, 16K-context unit holds 8064 × 16384 × 16384 ≈ 2.2×10^12 ≈ 2^41 entries, 69 TB at 32 B (E0, own arithmetic from the cost model's `A` and RFC-0018 'Leaf levels'); the linear-layer lookup leaves of the same unit hold 2.82×10^7 × 16384 ≈ 4.6×10^11 ≈ 2^38.7 entries, 15 TB (E0, own arithmetic, RISK-2 count). A level fits in aggregate shared memory only below about 9.4×10^5 ≈ 2^19.8 entries, so the resident variant applies to the last ~20 rounds of a 41-round leaf-level sumcheck, which hold 2^-21 of the work: it is feasible exactly where it is worthless. Second, the only transcript-legal way to bind `k` variables in one pass is Gruen's univariate skip (SUR-0002 seed; agenda `sumcheck` Question 1), which sends one degree-`3·(2^k − 1)` message in place of `k` degree-3 messages and so changes the tier-A message schema and the verifier's per-round check (RFC-0014 'Transcript schedule', 02 §1), and whose evaluation work per group of `2^k` entries grows with `2^k` (each of the sumcheck's multilinear factors must be evaluated at about `3·2^k` coset points from `2^k` known values, then multiplied), against `k` ordinary rounds whose per-entry work is linear in `k`; at the E0 level the skip saves bytes, not multiplications, so it cannot lower `max(compute, memory)` below the compute floor either (E0, own bound, order of magnitude; the exact count belongs to the `sumcheck` survey).

The streamed leaf levels of RFC-0018 (rounds that regenerate leaves tile by tile until the folded state fits) already fuse the first streamed round with the tree build; part (a) applies to every later round that runs over a resident folded state, so `p_s` falls by at most the same byte ratio (192/288) on those rounds. That effect is a bound, not an estimate, and is reported separately below.

## Why it could work

- SUR-0002 [E0-this] (E0, primary, own count): per lookup over the tree, 288 B through HBM with every sumcheck round through memory → 86 ps at 3.35 TB/s; 192 B with two-round fusion → 57 ps; 120 B with eight fused rounds → 36 ps. Compute floor: 18.5 `Fp2`-multiplication equivalents per lookup with a numerators-1 first layer → 76 ps at the int32 peak with 18 instructions per Goldilocks multiplication, 127 ps at the 60% utilization [VYTA25] measures for field-multiplication kernels (primary, measured on a 377-bit field on an A40; only the pipeline and utilization statements carry over), 56–75 ps at 12 instructions and 80% utilization. So `c_ℓ = max(compute, memory)` is 86 ps unfused and 57–76 ps with two-round fusion; the three byte figures fit `bytes/lookup = 96 + 192/k` for fusion depth `k` (E0, own fit, reproduces all three points), which gives 144 B → 43 ps at `k = 4`.
- [SP1GPU26] (E0, primary, code read 2026-10-10 in SUR-0002): the two-round kernels exist in production CUDA for a LogUp-GKR prover (KoalaBear, degree-4 extension), so the shape compiles, runs and is used; no timing is published.
- [DMABKGR25] (E0 modelling, primary, ISCA 2025, 255-bit field; read in SUR-0003): sumcheck rewrites every multilinear table each round with no inter-round reuse; arithmetic intensities 0.04–0.22 modmul/byte; bandwidth saturates first ("quickly saturate 512 GB/s"). The *byte-per-round* structure transfers to Goldilocks; the intensities do not (16× fewer bytes per element), which is exactly why the compute floor can be the binding one here.
- Hopper shared memory: 228 KB per SM, 132 SMs on the SXM part (secondary, vendor documentation). Against 69 TB at the attention leaf level: five orders of magnitude short, which is what kills the resident-level variant (E0, own arithmetic).
- The idea makes no claim that fusion is new: SUR-0002 Question 1 already states that round fusion "is standard in production code, so it is not Tessera's edge". What is new is the sizing rule (b) and the refutation (c).

## Expected gain

The cost model sees fusion only through `lookup_ps`. The baseline for this idea is the *unfused* kernel at its memory floor, `lookup_ps = 86`, not the corrected profile's 40 ps (which SUR-0002 found to be an assumption with no measurement behind it); the candidate is the two-round fused kernel at its memory floor, `lookup_ps = 57`. Both are E0, primary (own count), and both assume the kernel is memory-bound, i.e. a compute floor at or below 57 ps.

```sh
python3 -I research/tools/cost_model.py --set lookup_ps=86   # reference: unfused, memory-bound
python3 -I research/tools/cost_model.py --set lookup_ps=57   # candidate: two-round fold-sum fusion, memory-bound
```

```
lookup_ps=86  (Δ against the corrected profile, lookup_ps=40)
   S   linear  attention  weights  regen  streaming    total
   1K     5.47       1.77    0.021    4.0       0.42     11.7   Δ  +31.6%
   4K     5.39       6.97    0.021    4.0       2.32     18.7   Δ  +33.0%
  16K     5.08      26.26    0.020    4.0      11.24     46.6   Δ  +32.6%
 128K     3.30     136.56    0.013    4.0      77.93    221.8   Δ  +29.4%

lookup_ps=57
   S   linear  attention  weights  regen  streaming    total
   1K     4.10       1.37    0.021    4.0       0.42      9.9   Δ  +11.7%
   4K     4.04       5.39    0.021    4.0       2.32     15.8   Δ  +12.2%
  16K     3.81      20.31    0.020    4.0      11.24     39.4   Δ  +12.0%
 128K     2.48     105.62    0.013    4.0      77.93    190.0   Δ  +10.9%
```

Δμ of the candidate against the unfused reference (computed from the two runs above; E0):

| Context | Δμ | Δ proof size | Δ verification |
|---|---|---|---|
| 1K | 11.7 → 9.9, **−15.1%** (E0) | 0 (same messages) | 0 (same checks) |
| 4K | 18.7 → 15.8, **−15.6%** (E0) | 0 | 0 |
| 16K | 46.6 → 39.4, **−15.5%** (E0) | 0 | 0 |
| 128K | 221.8 → 190.0, **−14.3%** (E0) | 0 | 0 |

Roofline over fusion depth `k`, `c_ℓ = max(compute floor, (96 + 192/k) B / 3.35 TB/s)` (E0, own arithmetic from the SUR-0002 [E0-this] figures; compute floors are the four cases that survey lists):

| `k` | bytes/lookup | memory floor | `c_ℓ` at compute floor 56 ps | 76 ps | 101 ps | 127 ps |
|---|---|---|---|---|---|---|
| 1 (unfused) | 288 B | 86.0 ps | 86.0 | 86.0 | 101 | 127 |
| 2 (fold-sum, this idea) | 192 B | 57.3 ps | 57.3 | 76 | 101 | 127 |
| 4 | 144 B | 43.0 ps | 56 | 76 | 101 | 127 |
| 8 (SUR-0002 seed) | 120 B | 35.8 ps | 56 | 76 | 101 | 127 |

Reading: at a compute floor of 76 ps (18 instructions per multiplication, peak utilization), two-round fusion delivers `lookup_ps = 76`, μ(16K) 46.6 → 44.1, −5.3% (E0, `--set lookup_ps=76`), and `k = 4` or `k = 8` add nothing. At 101 or 127 ps fusion of any depth is worth zero. The −15.5% headline requires the compute floor to be at or below 57 ps, which the survey reaches only with 12 instructions per multiplication at 80% utilization. The eight-round seed beats two-round fusion only when the compute floor is below 57 ps, and then by at most 57 → 56 ps in the listed cases (`--set lookup_ps=36` gives −26.7% at 16K against the reference, but that point is memory-bound only at a compute floor below 36 ps, which no listed case reaches).

Scenario the model expresses only as a bound: if every in-memory round of the streamed leaf levels also saves the same byte fraction, `p_s` falls from 7 to at most 7 × 192/288 ≈ 4.67 ps per score per pass; `--set lookup_ps=57 --set stream_pass_ps_per_score=4.67` gives μ(16K) 46.6 → 35.6, −23.5%, and μ(128K) −26.0% (E0, upper bound: the first streamed round is already fused with the tree build per RFC-0018 and regenerates rather than reads its leaves, so its bytes are not the 288 B the ratio assumes). Not claimed as the gain; recorded so that the refuter can see what the secondary lever is worth at most.

All inputs are E0, primary (own counts) except the 3.35 TB/s HBM figure ([NVIDIA-H100], secondary, vendor documentation) and the 60% utilization ([VYTA25], primary, other field and GPU). No number here is measured on the target hardware; an RFC-0020 v0.1 measurement replaces every one of them.

## Security gate

| Property | Effect | Pass |
|---|---|---|
| Soundness bits (G-4, RFC-0014) | Unchanged. The fold-sum fusion sends the same degree-3 round messages in the same order under the same labels (`tree/{ℓ}/round/{j}` absorbed, then `tree/{ℓ}/r/{j}` squeezed); round `i + 1`'s sums are computed only after `r_i` exists and are sent only when complete over the whole level, so INV-FS-1 holds. A resident-level `k`-round variant that reused a challenge, or derived `r_{i+1}` from a partial sum before the round-`i` message was absorbed, would violate INV-FS-1; it is excluded by this idea, not proposed. The univariate skip changes the message degree and the verifier check and is outside this idea. | pass |
| Uniqueness upstream of the logits (SP-3, RFC-0009 Lemma 0) | Unchanged: no relation, table or witness changes; the kernel computes the same polynomial. | pass |
| Zero knowledge (G-5, RFC-0013) | Unchanged: masks and blinding rows are inputs to the same sumcheck; fusion changes when values are in registers, not what is sent. | pass |
| Verifier independence (SP-6) | Unchanged: a prover-side kernel; the verifier's checks and code are untouched. | pass |
| Post-quantum (hash-based commitments) | Unchanged. | pass |

## Prior rejection check

- [baseline.md §7](../baseline.md#7-already-rejected): no row covers round fusion or kernel shape; the closest rows (one-hot lookups, univariate lookups, base-field challenges, RFC-0008) concern the lookup argument, not its kernel.
- RFC-0008 Non-Goals, NG-6: "LogUp micro-architecture optimizations beyond the v1 kernels." This idea names a v1 kernel shape, which NG-6 leaves to RFC-0018, and NG-6 is a deferral, not a rejection.
- RFC-0018 Alternatives Considered 3 ("Multi-variable sumcheck rounds … Rejected for v1: message size and prover work grow as `(d + 1)^w` per round; reconsider if RISK-10 measurements exceed 50% overhead"): this is the same objection part (c) raises against the eight-round seed, now with the byte, shared-memory and evaluation-count arithmetic that separates the feasible two-round fusion (no multi-variable round, no message change) from the infeasible resident-level one. Nothing in this idea reopens Alternative 3.
- RFC-0002 fixes `Fp2` arithmetic and is untouched; the compute floor it implies is the quantity part (b) compares against, not something this idea changes.
- SUR-0002 Question 1 already notes that round fusion is standard in production code ([SP1GPU26], [Stwo26]). New here: the sizing rule `c_ℓ = max(compute, memory)` as the decision for fusion depth, and the E0 refutation of the eight-round resident-level seed.

## Kill criterion

An H100 measurement (RFC-0020 v0.1 protocol: same inputs, same GPU, digests matched against the CPU reference) in which the two-round fold-sum kernel takes at least as long per leaf as the unfused kernel, or a measured int32-pipe utilization that puts the compute floor at or above 86 ps (for example 18 instructions per Goldilocks multiplication at ≤ 60% utilization, the 127 ps case). Either shows the kernel is compute-bound at the measured utilization, makes fusion of any depth worth nothing, and sends the `c_ℓ` lever to the `Fp2` multiplication instead.

## Cheapest experiment

- **Hypothesis.** H1: the two-round fold-sum fusion lowers the lookup's memory floor from 86 to 57 ps and `c_ℓ` by `86 − max(57, compute floor)`. H2: no fusion depth `k > 2` lowers `c_ℓ` further at any compute floor ≥ 57 ps, and no resident-level variant is applicable to a leaf level above 2^20 entries.
- **E0, here, effort `s`.** The roofline table above over `k ∈ {1, 2, 4, 8}` and compute floors {56, 76, 101, 127} ps, the shared-memory residency check (2^19.8 entries fit; the leaf levels hold 2^38.7 and 2^41), and a univariate-skip evaluation count per group of `2^k` entries, as a script extending the `logup_opcount.py` of SUR-0002 with bytes per level. H2 is settled at this level; it is what this note already does, and the script should be committed under the experiment so the numbers are reproducible rather than quoted.
- **Why a CPU test cannot reach H1 (E1 insufficient).** The claim is about HBM bytes against the int32 pipe on one device; a CPU's L2/DRAM hierarchy and SIMD utilization are a different roofline, so a CPU fused/unfused ratio does not transfer (the same objection SUR-0002 raises against its own CPU calibration seed).
- **E2, effort `m`.** The RFC-0020 v0.1 unit-cost benchmark gains a kernel pair: unfused level sumcheck against the two-round fold-sum kernel, same inputs, same H100, Nsight Compute achieved DRAM bandwidth and int32-pipe utilization recorded per RFC-0020 'Report'. That run measures both the fusion gain and the compute floor, and so decides the kill criterion and the roofline cell this design is in.
- **Stays unknown until an H100 run.** The compute floor (instructions per `Fp2` multiplication as compiled, achieved utilization), which decides whether the −15.5% is real or zero; and the realized `p_s` reduction on the streamed levels, which the model can only bound.

## Refutations

Sweep triage of 2026-10-16: four refuter agents who did not write the idea, one per lens; each argument is condensed from that refuter's verdict.

| Lens | Verdict | Argument |
|---|---|---|
| soundness | survives | The two-round fold-sum fusion sends the same degree-3 messages under the same labels tree/{ℓ}/round/{j}, tree/{ℓ}/r/{j} (RFC-0014 schedule) and computes round i+1's sums only after r_i is squeezed, which is exactly INV-FS-1; the verifier's per-round check and the d/q' per-round bound are untouched. Part (c)'s refutation of the eight-round resident variant is right on transcript grounds: r_i is a function of the global round-i message, so binding k variables in one pass without k grid-wide synchronizations would mean squeezing a challenge before the message it depends on is absorbed, and the only legal alternative (univariate skip) changes the tier-A message degree, which the note excludes. No secret-class data is exposed; fusion changes register residency, not what is sent. |
| prior art | weakened | Part (a) is production practice the note itself credits to SP1's `two_round_fix_and_sum_*` kernels ([SP1GPU26]) and that SUR-0002 Question 1 already marks as 'not Tessera's edge'; the specification is silent (RFC-0018 'Kernels' names only 'bottom-up build fused with leaf generation', 'Fusion' only the P2/P3 passes) and RFC-0008 NG-6 defers kernel shape, so there is no rejection but nothing to promote either: the v1 kernel would adopt the SP1 shape with or without this note. RFC-0018 Alternatives Considered 3 (multi-variable rounds, (d + 1)^w) is correctly distinguished and not reopened. What is new is analysis, not a design: the c_ℓ = max(compute, bytes/3.35 TB/s) sizing rule and the E0 refutation of SUR-0002's eight-round resident-level seed (30 MB aggregate shared memory against 2^41 leaf entries). Those belong in SUR-0002's record and the RFC-0020 v0.1 benchmark plan rather than in an idea with its own Δμ, measured against a reference (`lookup_ps=86`) that is not the baseline. |
| cost model | weakened | The −15.5% headline is measured against a hypothetical unfused reference at `lookup_ps=86`, not against the baseline: reruns give 46.6 → 39.4 at 16K, and the candidate is +12.0% above the corrected profile. The gain exists only when the compute floor is at or below 57 ps, which SUR-0002 reaches solely with 12 instructions per multiplication at 80% utilization; at its central 76 ps the gain is −5.3% (`lookup_ps=76`: 44.1) and at 101 or 127 ps it is zero, so the compute floor's own 56–127 ps uncertainty spans the entire gain range. c_ℓ is the largest unit cost (25–28% of μ), so the term matters, and the roofline rule and the shared-memory refutation of the eight-round seed are correct and useful; but two-round fusion is the production baseline (SP1, Stwo) rather than an improvement over it, and the secondary p_s scenario (`lookup_ps=57 --set stream_pass_ps_per_score=4.67`: μ(16K) 35.6, +1.4% against the baseline) is an upper bound the note disclaims. Score it as a sizing rule for the v0.1 kernel pair with a conditional 0 to −15% gain, not as a −15% idea. |
| feasibility | weakened | The note's own H1 (memory floor 86 → 57 ps) is testable only at E2: it needs an unfused/fused CUDA kernel pair on an H100, and `tessera-gpu` has a `vector_add` template only (`crates/tessera-gpu/src/lib.rs`, `src/device.rs`) with the v0.1 unit-cost kernels (#55, #66) unwritten. The note states that a CPU fused/unfused ratio does not transfer, so nothing here moves it past E0, and the E0 part (roofline table, 2^19.8-entry residency bound, refutation of the eight-round seed) is already complete inside the note at zero cost. No tier-A surface and no RFC (RFC-0018 'Fusion' and RFC-0020 are tier C), which is in its favour, but as a research unit it is a sizing rule plus a shelve-until-H100 measurement, not an experiment this machine can run. |

## Score

| Gain | Confidence | Cost | Priority |
|---|---|---|---|
| 0 against the baseline: the fused kernel at `lookup_ps=57` is +12.0% above the corrected profile, and −15.5% μ(16K) only against a hypothetical unfused reference at 86 ps, conditional on a compute floor ≤ 57 ps (E0 roofline; floors from SUR-0002 [E0-this]) | 0.3: the compute floor's own 56–127 ps uncertainty spans the whole conditional gain, and the sizing rule rests on an unmeasured bytes-to-compute ratio | `l`: the decisive unfused/fused kernel pair needs an H100 before anything is known (E2, `m`); the E0 part is `s` and already done in the note; no RFC, no tier-A surface | 0 × 0.3 / 4 = 0 as a design idea; kept as the sizing rule for the RFC-0020 v0.1 kernel pair and as a SUR-0002 record line (the eight-round seed is refuted at E0) |

## Decision and links

Triage 2026-10-16: `triaged` at priority 0; three lenses find a sizing rule and an E0 refutation rather than a gain over the baseline. Recommendation for consolidation: `shelved` (H100) with two records carried out of it: SUR-0002's eight-round resident-level seed marked refuted at E0, and the c_ℓ = max(compute, bytes/3.35 TB/s) rule added to the RFC-0020 v0.1 benchmark plan. Experiments: none yet (candidate: an E0 roofline script, then the RFC-0020 v0.1 kernel pair). RFC or issue: none; the kernel shape belongs to RFC-0018 'Fusion' and the benchmark to RFC-0020 'Measurement protocol', both tier C. Ledger line updated on: pending consolidation.
