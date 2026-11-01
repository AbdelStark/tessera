# CPU row NTT measurements

This is the scalar baseline work for [#659](https://github.com/AbdelStark/tessera/issues/659),
not a completed acceleration or a real-model proof result. Production transforms are unchanged.
The real 1B proof and 8B calibration were left running throughout. Their changing load makes
these observations unsuitable for a throughput qualification or a regression threshold.

## Geometry and cost

The measured encoder uses `k = 2^22`, rate 1/2, `n = 2^23` and 200 mask coefficients,
as in [RFC-0011 Encoding](../rfcs/RFC-0011-advice-commitment.md#encoding).
It performs an inverse Cooley-Tukey transform of length `k`, a spread/scale, the serial
mask insertion, and a forward Gentleman-Sande transform of length `n`.
For the subgroup kernels, each first group skips its unit-twiddle products.

| Quantity | Count |
|---|---:|
| Inverse butterflies, `(k/2) log2(k)` | 46,137,344 |
| Forward butterflies, `(n/2) log2(n)` | 96,468,992 |
| Field products, including spread and masks | 134,218,330 |
| Field additions/subtractions | 285,212,872 |
| Logical butterfly value traffic, 32 bytes per butterfly | 4,563,402,752 bytes |
| Public encoder tables | 83,886,080 bytes (80 MiB) |
| Tables plus input row and output codeword | 184,549,376 bytes (176 MiB) |

The product count is the two butterfly counts minus `(k-1)+(n-1)` unit products,
plus `k` spread products and `3P` mask products. The add/subtract count is twice
the butterfly count plus `P`. This gives 0.091912 field operations per logical
butterfly byte. It is not a hardware floating-point intensity.

The host reports an Apple M1 Max, 10 physical CPU cores and 64 GiB RAM.
[Apple's specification](https://support.apple.com/en-ie/111900) lists a whole-chip
memory-bandwidth ceiling of 400 GB/s. Dividing all logical butterfly traffic by
that ceiling gives 11.4 ms, **only if every logical transfer reached DRAM at peak
bandwidth**. Cache reuse, twiddle accesses and the CPU's arithmetic throughput
prevent treating that number as an achieved bandwidth or a kernel lower bound.
No CPU issue-width or hardware-counter roofline was measured. Stack samples of
the live proof identify the NTT as a hotspot; they do not establish which hardware
resource limits it. H100 budgets in [08](../spec/08-performance-budget.md) remain
unmeasured here, and encoding alone omits column hashing from `c_c`.

## Observations on 2026-11-01

Baseline source: `7e2b722a99c0179321397cd4df36e1a2d3fa0468`.
The original optimized harness's fastest-of-five row was 491.14 ms (117.097 ns
per input element). A separate Criterion 0.5.1 harness used the public encoder,
observable inputs/output, 3 seconds of warm-up and 20 measured samples. Three
scalar runs had medians of 553.05, 524.96 and 896.18 ms, with coefficients of
variation 4.74%, 37.07% and 14.03%. All exceed RFC-0020's 3% noise threshold.
A serial before/after comparison on this host cannot establish a speedup.

An experimental safe Rust kernel fused the two smallest butterfly stages in
registers, preserving public group order and allocating no row storage. It passed
the scalar differential test, but an alternating comparison of both implementations
in one process showed a median candidate/scalar ratio of 1.002460 and ratio CV
8.20%. The complexity did not buy a measurable gain, so the candidate was reverted.
There is no acceleration claim. Both codewords were compared in full after each
pair. The first three pairs were warm-ups; the remaining 20 are below. Each pair
alternated which implementation ran first. Times are milliseconds.

| Pair | Scalar | Fused candidate | Candidate/scalar |
|---:|---:|---:|---:|
| 3 | 778.414791 | 726.027542 | 0.932700085 |
| 4 | 752.196958 | 750.357333 | 0.997554331 |
| 5 | 995.227667 | 754.685417 | 0.758304298 |
| 6 | 982.953917 | 1094.215625 | 1.113191174 |
| 7 | 932.108584 | 955.842583 | 1.025462698 |
| 8 | 948.183458 | 894.387042 | 0.943263705 |
| 9 | 773.142750 | 806.248666 | 1.042819927 |
| 10 | 761.201958 | 784.479334 | 1.030579764 |
| 11 | 749.852625 | 788.433833 | 1.051451721 |
| 12 | 748.185750 | 796.057000 | 1.063983108 |
| 13 | 768.894833 | 770.301417 | 1.001829358 |
| 14 | 755.802625 | 774.639542 | 1.024923064 |
| 15 | 834.952833 | 777.363625 | 0.931026993 |
| 16 | 743.551375 | 812.686708 | 1.092979901 |
| 17 | 796.038500 | 798.498375 | 1.003090146 |
| 18 | 748.657209 | 767.802208 | 1.025572450 |
| 19 | 785.045208 | 717.334416 | 0.913749181 |
| 20 | 801.551166 | 779.323125 | 0.972268719 |
| 21 | 755.016458 | 719.349083 | 0.952759473 |
| 22 | 877.330291 | 776.609667 | 0.885196459 |

## Repeatable baseline and guard

```sh
cargo bench -p tessera-pcs --bench throughput --locked
cargo test -p tessera-pcs --lib transforms_match_scalar_stages_at_boundaries_and_offsets --locked
```

The throughput harness now uses the existing `tessera-bench::Statistics`: three
warm-up iterations, ten measured iterations, median/min/max/CV and raw nanosecond
samples. It repeats a noisy run once and retains the rejected first samples. A
second noisy run stays explicitly flagged; rerun on an idle host before drawing
performance conclusions. Without `--bench` it retains the small smoke path.
Both transforms and row encoding make their completed output observable to the
optimizer. No third-party benchmark dependency was added to the project.

The updated release harness was run at the full geometry: its repeated row sample
had median 525.89 ms (125.382 ns per input), range 495.83 to 635.12 ms, and CV
6.95%. It remained flagged as noisy. These are scalar observations, not a gain.

The new scalar oracle independently spells out each radix-2 butterfly and checks
both directions with forward/inverse twiddle tables, subgroup and two coset shifts,
field boundary and seeded inputs, and four slice offsets from size 1 through 2^14
(across the recursive leaf boundary). It also checks surrounding canaries. A planted
addition-to-subtraction fault failed this test at size 2; the restored control passes.
Existing definition, independent verifier and conformance tests remain the external
oracles. No production buffer, worker, mask draw, parameter, transcript or planner
reservation changes in this baseline work. Public worker integration, complete proof
artifact comparisons and a measured acceleration remain owed by #659.
