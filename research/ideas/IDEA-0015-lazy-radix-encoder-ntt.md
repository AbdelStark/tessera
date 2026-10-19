# IDEA-0015: Higher-radix, lazily reduced Goldilocks encoder NTT, calibrated on CPU to pin which end of EXP-0002's `c_c` interval a real kernel reaches

| | |
|---|---|
| Status | `triaged` (2026-10-16) |
| Origin | [EXP-0002](../experiments/EXP-0002-commit-cost-calibration/README.md)'s counted interval `c_c ∈ [57.9, 156.8]` ps, whose whole width is the kernel's radix and reduction strategy (969 int32 ops per committed element at radix-16 with lazy reduction against 1,574 at radix-2, E0); [SUR-0005](../surveys/SUR-0005-commitments.md) synthesis 1 ([Uulong26]: 10.6–17.3 ps per radix-2 butterfly on an RX 7900 XTX at a memory-bound plateau, E1 on another GPU; [VYTA25]: NTT "limited by data movement", `FF_add` at 40% of the INT32 pipe, E1); RISK-1 and RISK-8 ([00 §9](../../docs/spec/00-overview.md#9-risks)); new as a design idea: [IDEA-0004](IDEA-0004-calibrate-commit-cost.md) counted the lazy high-radix kernel as the floor of its interval, nothing implements it |
| Lever | `c_c` (the encoder's share, 74–81% of the committed element by operation count: 721 of 969 at the low end, 1,274 of 1,574 at the high end, E0 from EXP-0002): int32 operations per butterfly-equivalent through radix and deferred reduction, and HBM passes per transform through radix |
| Spec surfaces | None normative. A prover kernel under [RFC-0018 'Kernels'](../../docs/rfcs/RFC-0018-gpu-prover-runtime.md#kernels) (row "Coset NTT and RS encode with vanishing mask", whose note already demands canonical output). [RFC-0002 'Representation and reduction'](../../docs/rfcs/RFC-0002-field-arithmetic.md#representation-and-reduction) already permits GPU kernels to keep values in `[0, 2^64)` between operations and requires canonicalization before a buffer is hashed, encoded or copied, so lazy reduction is inside the specification. [RFC-0011 'Encoding'](../../docs/rfcs/RFC-0011-advice-commitment.md#encoding) (the code, the coset `7·L`, natural order) is unchanged. Not tier A ([02 §1](../../docs/spec/02-public-api.md#1-stability-tiers)); no RFC |
| Priority | 1.5 (−5.1% × 0.3 / 1: the middle of the honest range 0 to −10.1% at 16K under `counted`, since the centre 94 ps is EXP-0002's radix-2 count at the int32 peak and its lazy radix-16 count at 60% utilization is 96.5 ps; 0 to 3.0 across the range) |

## Mechanism

Today's encoder (`crates/tessera-pcs/src/ntt.rs` and `encoding.rs`, read 2026-10-16) is two radix-2 kernels, Cooley–Tukey (natural in, bit-reversed out) and Gentleman–Sande (bit-reversed in, natural out), over one bit-reversed twiddle table `Z[t] = ω^{rev(t)}`, recursing depth-first until a group of `LEAF = 2^12` entries (32 KiB) fits the first-level cache and then finishing it level by level. Every butterfly `(u, v) ↦ (u + c·v, u − c·v)` is fully reduced: `Goldilocks::{add, sub, mul}` each end in `canonicalize`, and EXP-0002's ARM64 anchors measured 24 instructions plus a move and `ret` per butterfly, 15 per multiplication (E1, primary). The change is prover-only, four parts, and leaves the codeword identical symbol for symbol.

1. **Radix-4 and radix-8 butterflies that fuse two and three levels.** In the remainder tree the kernels walk, a group of `4m` entries holding `f mod (X^{4m} − c^4)` splits in two radix-2 levels: one twiddle `c^2` for the first level, two twiddles `c` and `ω_4·c` for the second, that is 4 full multiplications per 4 points per 2 levels (0.5 per point per level). The fused radix-4 step applies the three twiddles `c, c^2, c^3` to the group's three non-leading quarters (3 full multiplications) and finishes with a 4-point transform whose only constant is `ω_4 = 2^48`, a shift-and-reduce in Goldilocks (EXP-0002's "cheap multiplication", 8–12 int32 ops against 18–26 for a full one, E0): 0.375 full multiplications per point per level, −25% (E0, operation count). Radix-8 applies 7 twiddles per 8 points per 3 levels with internal constants `ω_8^j = 2^{24j}`: 0.292 per point per level, −42% (E0). EXP-0002's low end is radix-16 (15 full and 17 cheap multiplications per 16 points per 4 levels: 0.234 full per point per level). The number of passes over a `2^22`- or `2^23`-point row that does not fit a cache level halves or thirds with the radix: `log_r n` passes instead of `log_2 n` for the levels above the resident group.
2. **Lazy reduction inside a butterfly.** Values between operations stay in `[0, 2^64)` (the range RFC-0002 grants GPU kernels): the addition keeps its carry fix (`+EPSILON` on carry, `2^64 ≡ 2^32 − 1 (mod P)`) and drops `canonicalize`, the subtraction is already free of a final correction, and `reduce128` drops its final conditional subtraction, so one full reduction happens per twiddle multiplication's output and one canonicalization per symbol at the end of the transform, as EXP-0002's low count assumes (4 int32 ops per addition instead of 8, 18 per multiplication instead of 26, E0). The hazard the implementation must settle: a sum of two values both above `2^64 − 2^32` can carry twice, so the kernel either bounds the operands by a range argument over the butterfly's data flow or spends a second two-instruction fix where the argument fails; the experiment's instruction count says what the fix costs. Control flow stays data-independent (RFC-0002 'Timing', INV-ARCH-2): every correction is a mask or a select.
3. **Coset shift folded into the coefficient scaling**, as `RowEncoder` already does (`spread`: the coset NTT of `g` is the NTT of `g(7X)`, coefficients `g_i·7^i`): EXP-0002 measured the standalone coset kernel at 17% more per butterfly than the encoder's fused form (E1), so the GPU kernel never multiplies by a shifted twiddle.
4. **A four-step layout** `n = n_1 × n_2`, sub-transforms sized for the first-level cache on the CPU and for shared memory on the GPU, which is the layout [IDEA-0003](IDEA-0003-fused-encode-hash-tiles.md) step 1 ("Batch the block") needs, and which makes a CPU-measured ratio transfer as an operation-count ratio rather than a cache artefact. It is the optional second part of the experiment; the kill criterion is decided by parts 1–3 inside the existing depth-first group structure, which keeps the baseline's memory behaviour.

Output: bit-identical to the radix-2 kernel after canonicalization (the TCE symbols that `ColumnChains` hashes, INV-TCE-1), checked against `spec/vectors/encoding/encoding.json` (the independent Python oracle of [RFC-0011 'Encoding'](../../docs/rfcs/RFC-0011-advice-commitment.md#encoding)) and by a differential test against `NttPlan` and `RowEncoder` on seeded random rows. No protocol, descriptor, parameter or verifier change.

## Why it could work

- E0, primary ([EXP-0002](../experiments/EXP-0002-commit-cost-calibration/README.md) `count.py`, counts fitted to E1 within 0.83–1.00 on four production kernels): the encoder is 721 int32 ops per committed element at radix-16 with lazy reduction and 1,274 at radix-2 as written, −43%; the hash is 248–300 and does not move. That −43% is the entire width of the `[57.9, 156.8]` ps interval apart from the utilization factor. At the `counted` profile's centre, `c_c = 94` ps with the encoder at 74–81% of the count, the encoder is about 73 ps and the hash about 21 ps; the full −43% on the encoder lands `c_c` near 62 ps (E0, assuming the GPU realizes the lazy high-radix count at the same 60% utilization [VYTA25] that the centre assumes), below the 70 ps that the SC-2 ceiling `μ(16K) ≤ 45` needs under the corrected profile (E0, EXP-0002's cost-model table).
- E1, primary on another GPU ([Uulong26] via SUR-0005: AMD RX 7900 XTX, HIP, code public, author-measured, unreproduced): a three-stage Goldilocks NTT (two radix-512 stages plus a power-of-two stage, fused transposes and twiddles) reaches 10.6 ps per radix-2-equivalent butterfly at its memory plateau (`2^27`) and 17.3 ps at Tessera's row size `2^23`, and is "memory-dominated" from `2^25` up. High radix is what cuts the passes that bound such a kernel, so on a GPU radix buys bytes as well as operations.
- E1, primary ([VYTA25], A40, 377-bit field, via SUR-0002 and SUR-0005): NTT "limited by data movement"; `FF_add` at 40% and `FF_mul` at 60% of the INT32 pipe's peak. The additions, whose canonicalization a lazy scheme removes, are the less efficient part of the butterfly on the measured hardware.
- E1, third party ([Rotko26] via SUR-0005, Rust Ligerito v0.8.0, Ryzen 9 7945HX): "the prover is memory-bandwidth bound" for a Ligero-style encoder on 16 cores: the same reading as [Uulong26], on a CPU.
- Secondary, unverified in this pass (`claim`): production CPU provers (Stwo's SIMD backend [Stwo26]; Plonky3's Goldilocks transforms) are reported to fuse levels and defer reduction; no numbers are published and no source was read for it here. Nothing in the corpus measures a high-radix lazy kernel over Goldilocks on a GPU, so a CPU realization ratio measured with the method EXP-0002 validated is the best E1 available before the v0.1 gate, and it narrows a 2.7× interval that currently decides whether SC-2 holds.

## Expected gain

The cost model has one knob for this idea, `commit_ps`. The scenario is derived from EXP-0002's counts (E0, primary): encoder share 74–81% of `c_c`, so at the `counted` centre of 94 ps the encoder is `0.775 × 94 ≈ 73` ps and the hash about 21 ps; the full counted −43% on the encoder gives `73 × 0.57 + 21 ≈ 62` ps, half of it realized gives about 78 ps. The baseline is the `counted` profile (`c_c` 94, `c_ℓ` 101, `c_π` 73 ps; [baseline.md §2](../baseline.md#2-the-cost-model)), since under the `corrected` profile `c_c = 30` ps is already below the interval's floor and the idea moves no number there.

Upper bound, the full counted −43% realized (E0, run 2026-10-16):

```sh
python3 -I research/tools/cost_model.py --profile counted --set commit_ps=62
```

```text
baseline: profile 'counted'
   S   linear  attention  weights  regen  streaming    total
   1K     9.20       3.70    0.021    4.0       0.42     17.3
   4K     9.06      14.59    0.021    4.0       2.32     30.0
  16K     8.54      54.98    0.020    4.0      11.24     78.8
 128K     5.55     285.91    0.013    4.0      77.93    373.4

scenario: commit_ps=62
   S   linear  attention  weights  regen  streaming    total
   1K     7.69       3.26    0.021    4.0       0.42     15.4   Δ  -11.3%
   4K     7.57      12.85    0.021    4.0       2.32     26.8   Δ  -10.8%
  16K     7.14      48.42    0.020    4.0      11.24     70.8   Δ  -10.1%
 128K     4.64     251.77    0.013    4.0      77.93    338.4   Δ   -9.4%
```

Half realized (E0, run 2026-10-16):

```sh
python3 -I research/tools/cost_model.py --profile counted --set commit_ps=78
```

```text
scenario: commit_ps=78
   S   linear  attention  weights  regen  streaming    total
   1K     8.45       3.48    0.021    4.0       0.42     16.4   Δ   -5.6%
   4K     8.32      13.72    0.021    4.0       2.32     28.4   Δ   -5.4%
  16K     7.84      51.70    0.020    4.0      11.24     74.8   Δ   -5.1%
 128K     5.09     268.84    0.013    4.0      77.93    355.9   Δ   -4.7%
```

| Context | Δμ | Δ proof size | Δ verification |
|---|---|---|---|
| 1K | −11.3% full, −5.6% half (E0, `counted` baseline) | 0 (E0: same codeword, same `τ_A`) | 0 (E0: verifier unchanged) |
| 4K | −10.8% full, −5.4% half (E0) | 0 | 0 |
| 16K | −10.1% full, −5.1% half (E0) | 0 | 0 |
| 128K | −9.4% full, −4.7% half (E0) | 0 | 0 |

Inputs: the −43% is E0 (EXP-0002 counts, fitted to E1 on one M4 Max core); the 94 ps centre and the 60% utilization are E0 built on [VYTA25]'s E1 figure for another field on another GPU (`secondary` for the H100 peak rates, documentation figures); the realized fraction is what the experiment measures (E1 here, E2 at the gate). Under the `corrected` profile (`c_c = 30` ps assumed) the idea moves nothing; what it buys there is the SC-2 ceiling itself, which EXP-0002 says holds only for a kernel near the integer pipeline's peak (`c_c ≤ 70` ps at 16K, E0). Under the `counted` profile for all three unit costs, `μ(16K) = 70.8` at 62 ps still exceeds 45: the idea removes `c_c`'s share of the overshoot, it cannot restore SC-2 alone.

## Security gate

| Property | Effect | Pass |
|---|---|---|
| Soundness bits (G-4, RFC-0014) | None. The codeword is the same `RS[n, k + P_A]` on `7·L`, symbol for symbol; RFC-0011's proximity bound, RFC-0014's budget and the verifier's column check see identical data. The only requirement is bit-exact canonical output: an unreduced symbol would be a non-canonical TCE encoding, which the verifier rejects (`ERR-CODEC`, INV-TCE-1), a completeness failure for the honest prover and never a soundness gain for a cheating one. Enforced by the encoding conformance vectors and the differential test against the radix-2 kernel | pass |
| Uniqueness upstream of the logits (SP-3, RFC-0009 Lemma 0) | None: no relation, table, witness or range changes; the encoder is downstream of the committed values and deterministic | pass |
| Zero knowledge (G-5, RFC-0013) | None: the mask coefficients `m_0 … m_{P_A−1}` enter the same coefficient positions above `k` and are drawn from the same stream in the same order; the kernel is deterministic and its control flow depends on lengths and indices only (INV-ARCH-2), so no timing channel is added that the row-at-a-time encoder lacks | pass |
| Verifier independence (SP-6) | None: the verifier's `tv-field` NTT re-encodes the 17 combined rows from [RFC-0011 'Encoding'](../../docs/rfcs/RFC-0011-advice-commitment.md#encoding) with its own algorithms and need not change (its work is negligible); nothing is copied in either direction; the research harness depends on prover crates only (research rule 2) | pass |
| Post-quantum (hash-based commitments) | None: BLAKE3 column chains and the Merkle root are untouched | pass |

Redaction (05 §3, INV-ARCH-6): the encoder takes plain `&mut [Fp]` and the `Secret<T>` boundary sits above it (`expose_secret` is named only in `crates/tessera-codec/src/secret.rs`, per `secret-access.toml`, checked 2026-10-16); the kernel logs nothing and its errors (`PcsError`) carry lengths only, so the boundary is unchanged. The experiment runs on seeded synthetic rows (research rule 6). Unsafe code: a SIMD variant, if any, lives in `tessera-field`'s permitted modules with `// SAFETY:` comments; the research workspace itself has none.

## Prior rejection check

- [RFC-0002 Alternatives Considered 4](../../docs/rfcs/RFC-0002-field-arithmetic.md#alternatives-considered) (and the [baseline.md §7](../baseline.md#7-already-rejected) row "Montgomery form for the prover") rejects Montgomery representation because "the special-form reduction above is faster". Lazy reduction is that special-form reduction applied once per fused level instead of once per operation; the item supports the idea rather than covering it. [RFC-0002 'Representation and reduction'](../../docs/rfcs/RFC-0002-field-arithmetic.md#representation-and-reduction) grants GPU kernels exactly the `[0, 2^64)` range this idea uses.
- [RFC-0011 Alternatives Considered 3](../../docs/rfcs/RFC-0011-advice-commitment.md#alternatives-considered) (baseline §7 row "Brakedown codes") rests on "NTT-based RS encoding is fast on GPUs", which SUR-0005 found unmeasured for Goldilocks ([Uulong26] is AMD-only and memory-bound); this idea tests the premise rather than re-proposing the rejected code.
- Adjacent ideas, neither of which implements a radix or reduction change: [IDEA-0003](IDEA-0003-fused-encode-hash-tiles.md) (fused encode-and-hash tiles, `shelved` (H100): the byte side of the same kernel; this idea supplies its part-1 layout) and [IDEA-0004](IDEA-0004-calibrate-commit-cost.md) / EXP-0002 (`inconclusive`: counted the kernel this idea would build and left the realized point to E2). SUR-0005's candidate "L2-resident fused encode-and-hash kernel" is the byte argument; this is the operation argument. baseline.md §7 has no row on the NTT kernel.

None rejects it, checked. What is new: a measured CPU realization ratio of the counted −43%, with the fitted method EXP-0002 validated, in place of an unmeasured interval.

## Kill criterion

On one M4 Max performance core, same inputs and harness as EXP-0002, the radix-4 or radix-8 lazy kernel is less than 15% faster per point than `NttPlan`'s radix-2 kernel at `2^22` and `2^23` (the butterfly is bound by the multiplication unit and the fully reduced form costs nothing measurable), or its ARM64 instruction count per butterfly-equivalent is within 10% of the radix-2 kernel's 24 (the double-carry fixes and twiddle handling give back what the fusion saves). Either says EXP-0002's radix-2 count is the honest ceiling and the interval should narrow upward, not downward. On the GPU side (E2, later): an Nsight profile showing the encode kernel bound by shared-memory bank conflicts or occupancy rather than int32 issue, where radix buys nothing.

## Cheapest experiment

Hypothesis (pre-registered here, before any code): the radix-8 lazy kernel needs at least 25% fewer nanoseconds per point than `NttPlan` at `2^23` on one M4 Max core (the multiplication saving alone), with an ARM64 instruction count per butterfly-equivalent within 10% of EXP-0002's lazy count translated to ARM64.

Method, effort `s` (days, no RFC), E1 reachable here: a scratch Cargo workspace (empty `[workspace]` table, research rule 2) under `research/experiments/`, depending by path on `tessera-field` and `tessera-pcs`; a lazy `u64` arithmetic written in the harness from RFC-0002 'Representation and reduction' and converted to `Goldilocks` at the boundary; radix-4 and radix-8 DIT and DIF butterflies with lazy reduction inside the existing depth-first group structure (parts 1–3 of the mechanism), the four-step layout (part 4) as a second step if the first survives. Measure ns per point at `2^20`, `2^22` and `2^23`, natural and coset, against `NttPlan::{ntt, intt, coset_ntt}` and `RowEncoder::encode_row` (`k = 2^22`, `P = 200`), RFC-0020 protocol as EXP-0002 ran it (3 warm-ups, 10 measured, median and coefficient of variation, one performance core, fastest-of-5 reported as the headline); count instructions per butterfly-equivalent from `#[inline(never)]` anchors as EXP-0002's `asm_count.sh` did; check bit-exact equality with the radix-2 output after canonicalization on seeded rows (SplitMix64, EXP-0002's seeds) and against every case of `spec/vectors/encoding/encoding.json`, plus the boundary values `0`, `1`, `P − 1`, `2^32`, `2^32 − 1`, `2^64 − 1` through the lazy arithmetic. Report the realized fraction of the counted −43% and the implied `c_c` interval (E0 translation as EXP-0002: int32 peak `secondary`, 60% utilization [VYTA25]).

Unknown until an H100 run (E2, the RFC-0020 v0.1 gate): the achieved int32 utilization of the lazy high-radix kernel, whether shared-memory passes are free as the low count assumes, and whether the encoder or the hasher binds first.

## Refutations

Sweep triage of 2026-10-16: four refuter agents who did not write the idea, one per lens; each argument is condensed from that refuter's verdict, with the cost-model reruns the scoring agent made to check disputed cells noted as such.

| Lens | Verdict | Argument |
|---|---|---|
| soundness | survives | Prover-only: the codeword is the same `RS[n, k + P_A]` symbol for symbol, and RFC-0002 'Representation and reduction' already grants GPU kernels values in `[0, 2^64)` between operations with canonicalization before any buffer is hashed, encoded or copied, so lazy reduction is inside the specification. As ADV-P nothing is gained: a non-canonical symbol is a non-canonical TCE encoding the verifier rejects (INV-TCE-1), and a symbol that differs from the committed polynomial's value fails the column check (RFC-0011 step 8.3), both completeness failures for the honest prover and never an acceptance. As ADV-V the only new surface is timing: the double-carry fix the note identifies must be a mask or select, never an `if` on data (RFC-0002 'Timing', INV-ARCH-2), which the note already requires; control flow depends on lengths and indices only, as today. No message, challenge, mask or verifier check changes. |
| prior art | survives | The specification already permits exactly this: RFC-0002 'Representation and reduction' lets GPU kernels keep values in `[0, 2^64)` and canonicalize before hashing, and RFC-0018's kernel row demands only canonical output, so lazy reduction and radix choice are prover implementation freedom, not a protocol change. RFC-0002 Alternative 4 rejects Montgomery form because the special-form reduction is faster, which supports rather than covers the idea; RFC-0011 Alternative 3 rests on the premise 'NTT-based RS encoding is fast on GPUs' that this idea tests. In the literature high-radix lazily-reduced NTTs are standard practice ([Uulong26]'s radix-512 Goldilocks kernel, SUR-0005; Plonky3/Stwo at claim level), so the technique itself is not new; what is new to the knowledge base is an E1 realization ratio for the lazy high-radix count that EXP-0002 only wrote down as the floor of its interval, the current crate being radix-2 with full reduction per butterfly (crates/tessera-pcs/src/ntt.rs:12, 56). Nothing in baseline.md §7 rejects it; SUR-0005's 'L2-resident fused encode-and-hash kernel' is the byte-side complement, not a duplicate. |
| cost model | weakened | Both scenarios reproduce (counted `commit_ps` 62 → Δμ(16K) −10.1%, 78 → −5.1%) and `c_c` is the second-largest unit cost at every context, so the term matters. The gain, however, is by construction the width of EXP-0002's interval rather than a move beyond it: the counted centre 94 ps is 'the high count at the pipeline's peak' (EXP-0002 'Counts' table, README lines 94 and 105, verified at triage: 94.1 ps = radix-2 count at 100% of the int32 pipe), and the same table puts the radix-16 lazy count at 96.5 ps at [VYTA25]'s 60% utilization. The note's derivation ('assuming the GPU realizes the lazy high-radix count at the same 60% utilization that the centre assumes') mis-reads the centre: a lazy radix-16 kernel at 60% lands at about 96 ps, zero gain over the centre, and 62 ps requires it near peak. Under the RFC-0020 reference profile the idea moves nothing. What survives is a calibration experiment that narrows the interval on CPU (E1), not a priced Δμ; the honest cost-model range is 0 to −10% at 16K depending on a utilization input the experiment cannot measure. |
| feasibility | weakened | Implementation is feasible and cheap: the encoder is two radix-2 kernels with `LEAF = 2^12` (ntt.rs:12, 56, 457-485), RFC-0002 explicitly grants `[0, 2^64)` lazy ranges on GPU kernels with canonicalization before hashing, nothing is tier A, and the differential test against `spec/vectors/encoding/encoding.json` is in place. What does not hold is the title's claim that a CPU run pins which end of EXP-0002's `c_c` interval a real kernel reaches: the interval's width is int32-instruction counts on the H100 integer pipe, while ARM64 does a 64×64→128 multiply in two instructions and EXP-0002 itself found the out-of-cache levels of a 2^23-point transform 'cost almost nothing extra on this machine', so the pass-count half of the radix argument (`log_r n` HBM passes) is untestable here and the measured ratio is an M4 realization ratio, not an H100 one (E1 for ARM64, still E0 translation for the GPU). The double-carry hazard of lazy addition (both operands above `2^64 − 2^32`) also needs a range argument or a second fix the note leaves to the implementation. Effort `s` and E1 are reachable, but the result narrows the interval only through the same fitted-count translation EXP-0002 already used, so the gain of −10% μ(16K) remains E0 after the experiment. |

## Score

| Gain | Confidence | Cost | Priority |
|---|---|---|---|
| −5.1% μ(16K), taken as the middle of the honest range 0 to −10.1% (E0 cost model under `counted`: `commit_ps` 94 → 78 and → 62; 0 at the centre's own utilization reading, since EXP-0002 prices the lazy radix-16 count at 96.5 ps at 60%; −5.6% / −11.3% at 1K); nothing under `corrected` | 0.3: the mechanism is standard and sound ([Uulong26], [VYTA25], E1 on other GPUs), but the headline mis-reads the counted centre (94.1 ps is the radix-2 count at the int32 peak, EXP-0002 README lines 94 and 105) and the CPU ratio reaches the GPU only through the fitted-count translation, so the number is extrapolated | `s`: days in a scratch workspace over `tessera-field` and `tessera-pcs`, prover-only, no RFC, E1 here | 5.1 × 0.3 / 1 = 1.5 (0 to 3.0 across the honest range) |

## Decision and links

Triage 2026-10-16: `triaged` at priority 1.5. The experiment is worth running as a calibration: it replaces an unmeasured radix-2-to-lazy-high-radix ratio by an M4 one with the method EXP-0002 validated and fixes the double-carry rule, but its headline must be restated before it runs: against the `counted` centre the gain is the utilization question, 0 if the GPU realizes the lazy count at 60% and about −10% near the int32 peak, and no CPU run answers it. Report the realized fraction of the counted −43% and the implied `c_c` interval, not a Δμ. Queued after IDEA-0016, IDEA-0021, IDEA-0019 and IDEA-0022, whose gains do not depend on a utilization input. Experiments: none yet (the E1 calibration above, pre-registered as EXP-NNNN when run). RFC or issue: none; the kernel is a prover implementation choice under RFC-0018 'Kernels'. Ledger line updated on: pending consolidation (2026-10-16).
