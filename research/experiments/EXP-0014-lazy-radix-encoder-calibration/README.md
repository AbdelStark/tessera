# EXP-0014: Calibrate lazy radix-4/8/16 Goldilocks encoder arithmetic on CPU

| | |
|---|---|
| Status | `done: refuted` |
| Idea | [IDEA-0015](../../ideas/IDEA-0015-lazy-radix-encoder-ntt.md) |
| Evidence level | E1 CPU timing and ARM64 instruction observations; E0 exact operation counts and conditional H100 cost intervals |
| Hardware | Apple M4 Max, 48 GiB, macOS 26.6.2; one thread, no enforced affinity |
| Pre-registered on | 2026-10-19 |

## Hypothesis

A safe radix-16 encoder with lazy Goldilocks representatives, preserving the production depth-first traversal and mask/coset placement, produces exactly the production canonical codeword, realizes at least a 25% CPU time reduction at `k=2^22, P_mask=200`, and materially reduces the normalized instruction count. Radix-4/8 and fully reduced fused-radix variants identify which arithmetic and traversal changes explain the result. This calibrates the operation-count model of [EXP-0002](../EXP-0002-commit-cost-calibration/README.md); it does not validate a GPU throughput or a change in μ.

## Success criterion

The primary candidate is fixed as lazy radix-16, never selected after observing the fastest arm. Let `sigma=1−median(candidate)/median(production)` and `A=instruction_count(lazy_radix16_anchor)/instruction_count(radix2_anchor)` for equivalent blocks of sixteen points and four levels. Both anchors include arithmetic, loads, stores and spills; directives, labels and the final return are excluded. Counts are static compiler observations, not dynamic issue counts.

- `validated`: primary encoder `sigma>=0.25` at `k=2^22, P_mask=200`; primary natural forward transforms at both `2^22` and `2^23` have `sigma>=0.15`; `A<=0.90`; all correctness/oracle checks pass; every arm needed by these ratios retains CV at most 3% after at most one automatic rerun.
- `refuted`: necessary CVs are clean and either encoder `sigma<0.15`, either judged natural transform `sigma<0.15`, or `A>0.90`. This fires IDEA-0015's CPU/assembly kill criterion and narrows the CPU realization toward the conservative end, without asserting a GPU bound.
- `inconclusive`: all other outcomes, including encoder saving in `[0.15,0.25)`, missing/unusable assembly anchors, or necessary CV above 3% after the rerun.

A wrong candidate output is a harness defect: fix it and rerun correctness before measuring; it is never evidence of an optimization. A correction to the arithmetic derivation is recorded rather than hidden in a favourable count.

The E0 calibration outputs are required regardless of verdict: operation counts per transform and encoded element, normalized realized time ratios, separate hash cost from EXP-0002, and a conditional `c_c` interval using actual radix-16 counts at 100%/60% int32 utilization and the stated byte assumptions. Compare its arithmetic count to EXP-0002's radix-2 high and hypothetical radix-16 low. Report the realized fraction of that counted arithmetic gap and the analogous CPU time gap separately. No GPU band, SC-2 claim, or GPU Δμ is a success criterion.

## Method

Scope reaches only this experiment and IDEA-0015. Handover section 4.3 explicitly names lazy radix-16, so the primary arm implements radix-16 to test EXP-0002's actual lower-count mechanism; the idea's cheaper radix-4/8 probes remain controls, not substitutes. This revision is made before code and before the pre-registration commit. No production, verifier, protocol, mask generation or encoding-vector changes. The performance-optimization skill's baseline/oracle/roofline sequence is applied within the existing research harness: same-machine production timing is the baseline, while crates' Criterion and production gate changes belong to a later implementation. The optional four-step layout is omitted from this unit: the idea's kill criterion concerns the existing depth-first structure.

- **Arithmetic and corrected assumptions.** `P=2^64−2^32+1`, `EPSILON=2^32−1`. Lazy values are arbitrary words below `2^64`, always congruent to their field value. Addition uses the carry correction twice: after an initial overflow, adding EPSILON can overflow once more, and the last corrected value is below `2·EPSILON`. Subtraction analogously corrects an initial borrow with subtraction of EPSILON and corrects one possible second borrow; the second corrected value cannot borrow again. Full multiplication uses the RFC-0002 128-bit reduction with its final canonical subtraction omitted; its existing intermediate bounds prevent another carry/borrow. Canonicalize before conversion to Goldilocks, encoding, comparison or hashing. Maximum product is `(2^64−1)^2<2^128`.
- **Roots.** Derive roots from the production field's public `two_adic_generator`, and check independently with exact integer exponentiation. In particular `omega4=2^48` and production `omega8=−2^24=2^120 mod P, and omega16=−2^60=2^156 mod P`; IDEA-0015's historical `omega8=2^24` statement is corrected here before code. The inverse uses the matching inverse constants. Multiplication by a power `2^e` is implemented with bounded widened shifts and the Goldilocks congruences, including every `64<=e<96` case through `2^e=2^(e−32)−2^(e−64) mod P`, with sign negation for exponents from 96 to 191; no shift or intermediate exceeds 128 bits.
- **Arms.** Production `NttPlan` / `RowEncoder` (baseline); harness radix-4, radix-8 and radix-16 fused kernels with canonical arithmetic (controls); harness lazy radix-4/8 (observations) and lazy radix-16 (primary candidate). Fused kernels apply the non-leading input twiddle powers, then an internal four/eight/sixteen-point transform with cheap small-root multiplications. Cooley–Tukey and Gentleman–Sande maintain the production bit-reversal conventions. Recursion reaches cache-resident groups of `2^12` entries, then finishes fused levels locally; public tail levels use radix-2. Precompute public twiddles outside timing. No per-transform allocations in the kernel; record unavoidable wrapper copies separately or identically in both arms.
- **Encoding.** Preserve the production inverse transform, `7^i/k` spread, vanishing-mask placement and natural-order forward transform. Mask values are drawn in the same order and number as production. Report full `encode_row` at `k=2^20` and `2^22`, rate 1/2, `P_mask=200`. Timed transforms are natural `ntt`, `intt` and `coset_ntt` with shift 7 at `2^20`, `2^22`, `2^23`; natural ntt at the last two sizes is judged, remaining cases are observations. The fused encoder avoids separate bit reversal as production does.
- **Inputs.** Synthetic SplitMix64, rows/transforms seed `0x7E55_E7A0_0047_0401`, masks seed `0x7E55_E7A0_0047_0402`, matching EXP-0002. Arithmetic checks add seed `0x7E55_E7A0_0047_0414`. Every timing arm receives identical canonical inputs and masks. Correctness also feeds noncanonical boundary words through the lazy arithmetic directly.
- **Correctness before timing.** All field boundaries `0,1,P−1,P,2^32−1,2^32,2^64−1` and their pairwise products/sums/differences, random arithmetic against independent exact integer residues, and explicit double-carry/double-borrow cases. Every encoding/NTT conformance vector in `spec/vectors/encoding/encoding.json`, produced by the independent Python Horner oracle, must match. Small transforms through log size 12 and rows with zero, one, full and partial mask budgets test radix tails, inverse/forward and coset identities against production; full timed outputs compare symbol-for-symbol before timing, canonicality is checked, and digests are saved. The independent verifier's NTT/row oracle may be used as an additional check where public APIs support it; no verifier algorithm is copied into the candidate.
- **Measurement.** One process/thread in an exclusive timing slot, with builds/tests idle. Three warm-ups and ten measured samples, batches at least 20 ms by repetition; median/minimum/maximum/sample CV. Retain a second run only when initial CV exceeds 3%; save both raw runs and flag retained CV above 3%. No fastest-of-five selection. Plans and synthetic data creation are excluded; buffer reset policy is identical and documented. Capture CPU/OS/toolchains/UTC/base commit and whether affinity was enforced.
- **Counts and code generation.** `count.py` records full products, cheap-root operations, additions/subtractions including both possible corrections, canonicalizations, twiddle preparation and traversed bytes, with tail stages treated exactly. It checks the radix-2 denominator against EXP-0002. Assembly anchors use the same sixteen-point block and precomputed twiddle powers; record bodies, normalized count, spills and conditional branches. Candidate branches on secret values prohibit production promotion even if timings validate; no constant-time deployment claim follows from this experiment.
- **Hardware honesty.** Use EXP-0002's H100 peak 132×64×1.98 GHz int32 and 3.35 TB/s HBM as secondary assumptions, 60% int32 and 80% HBM utilization as borrowed assumptions. Include unchanged BLAKE3 cost separately. CPU ratios and ARM64 static instructions cannot measure Hopper utilization, shared-memory bank conflicts, occupancy or register pressure.

## Reproduction

```sh
cargo test --release --locked --manifest-path research/experiments/EXP-0014-lazy-radix-encoder-calibration/Cargo.toml
cargo run --release --locked --manifest-path research/experiments/EXP-0014-lazy-radix-encoder-calibration/Cargo.toml -- check
cargo run --release --locked --manifest-path research/experiments/EXP-0014-lazy-radix-encoder-calibration/Cargo.toml -- all
python3 -I research/experiments/EXP-0014-lazy-radix-encoder-calibration/count.py
python3 -I research/experiments/EXP-0014-lazy-radix-encoder-calibration/asm_count.py
```

The first dependency resolution creates the retained Cargo.lock; subsequent runs use `--locked`. The harness is its own workspace, `publish=false`, dual licensed Apache-2.0 OR MIT, and forbids unsafe code. All raw output is written under `results/`; no private data enters the experiment.

## Results

**Refuted**, under the frozen CPU/assembly criterion (E1, primary). The full encoder slows by 45.75%, the judged natural transforms slow by 33.95% and 37.41%, and the normalized static instruction ratio is 1.002710, exceeding the 0.90 cutoff. All necessary CVs are clean. No production optimization follows from this result.

All six candidates (canonical and lazy radix-4/8/16) reproduce the production output on every timed transform and encoder, and every independent Python-generated encoding/NTT vector. The small sweep covers log sizes 0–12, zero/one/half/full masks and both inverse/forward/coset directions. Arithmetic controls cover all pairwise boundary words, 4,096 seeded random word pairs, cheap powers through exponent 180, both second carry/borrow corrections and equivalent arbitrary-twiddle sixteen-point anchors. Every resulting symbol is canonical before comparison or hashing. Digests are printed in [run.txt](results/run.txt); exact output equality is asserted before any timing.

Raw evidence: [run.json](results/run.json), [run.txt](results/run.txt), [check.txt](results/check.txt), [asm.json](results/asm.json), [asm.txt](results/asm.txt), [count.json](results/count.json), [count.txt](results/count.txt), [summary.json](results/summary.json), [environment.json](results/environment.json). The recorded process finished with exit status 0. The original experiment remains unchanged.

All 77 arms completed, with six automatic reruns (first-attempt raw samples retained). All retained CVs are at most 3%, including observations. Every one of 1,079 raw batches lasts at least 20.082583 ms. [analyze.py](analyze.py) independently reconstructs normalized samples, medians and sample CVs from elapsed batch times/repetition counts, checks the 3+10 protocol and applies the original criterion.

| Judged pair (E1) | Production ns/element | Lazy16 ns/element | Saving σ | CV baseline / candidate |
|---|---:|---:|---:|---:|
| encode 2^22 | 34.941 | 50.927 | -45.75% | 0.464% / 2.026% |
| ntt 2^22 | 12.575 | 16.844 | -33.95% | 0.839% / 1.383% |
| ntt 2^23 | 13.160 | 18.083 | -37.41% | 1.219% / 1.021% |

The sixteen-point, four-level anchors have 738 instructions for radix-2 and 740 for lazy radix-16, excluding return/directives/labels. They normalize to 23.0625 and 23.125 instructions per radix-2-equivalent butterfly (E1 compiler observation). The bodies have zero branch/call instructions, but radix-16 has 12 stack accesses against 8 for radix-2. This is a static ARM64 block observation; it measures neither dynamic issue throughput nor whole-kernel spills. Both anchors are unrolled, use precomputed public twiddles and are checked for equal residues on synthetic inputs.

Full arm table (E1; input resets included identically in standalone transforms; encoder overwrites its output without a reset; plan construction excluded). Natural transforms use the production tiled bit-reversal traversal. Both encoders use the production in-place spread traversal and exact mask/coset placement. All public fused twiddle powers are precomputed before timing; their larger tables and preparation cost remain relevant limitations.

| Case | Median ns/element | CV | Rerun |
|---|---:|---:|---|
| ntt 2^20 production | 11.163 | 1.155% | yes |
| ntt 2^20 canonical4 | 13.947 | 1.019% | no |
| ntt 2^20 canonical8 | 13.511 | 1.745% | no |
| ntt 2^20 canonical16 | 17.241 | 1.437% | no |
| ntt 2^20 lazy4 | 12.978 | 1.686% | no |
| ntt 2^20 lazy8 | 13.054 | 1.429% | yes |
| ntt 2^20 lazy16 | 15.222 | 1.348% | no |
| intt 2^20 production | 10.687 | 1.242% | no |
| intt 2^20 canonical4 | 14.598 | 1.371% | no |
| intt 2^20 canonical8 | 13.237 | 1.211% | no |
| intt 2^20 canonical16 | 16.780 | 0.751% | no |
| intt 2^20 lazy4 | 14.494 | 1.727% | no |
| intt 2^20 lazy8 | 13.536 | 1.136% | no |
| intt 2^20 lazy16 | 14.746 | 1.833% | no |
| coset 2^20 production | 12.961 | 1.534% | no |
| coset 2^20 canonical4 | 15.001 | 1.461% | no |
| coset 2^20 canonical8 | 14.519 | 0.983% | no |
| coset 2^20 canonical16 | 18.147 | 2.194% | no |
| coset 2^20 lazy4 | 13.764 | 2.489% | no |
| coset 2^20 lazy8 | 14.043 | 1.445% | no |
| coset 2^20 lazy16 | 15.817 | 1.214% | yes |
| ntt 2^22 production | 12.575 | 0.839% | no |
| ntt 2^22 canonical4 | 15.832 | 0.723% | no |
| ntt 2^22 canonical8 | 15.925 | 1.524% | yes |
| ntt 2^22 canonical16 | 18.637 | 0.705% | no |
| ntt 2^22 lazy4 | 14.614 | 0.696% | no |
| ntt 2^22 lazy8 | 15.298 | 1.496% | no |
| ntt 2^22 lazy16 | 16.844 | 1.383% | no |
| intt 2^22 production | 12.169 | 1.863% | no |
| intt 2^22 canonical4 | 16.301 | 1.035% | no |
| intt 2^22 canonical8 | 15.616 | 2.470% | no |
| intt 2^22 canonical16 | 18.689 | 1.091% | no |
| intt 2^22 lazy4 | 15.843 | 2.049% | no |
| intt 2^22 lazy8 | 15.914 | 2.037% | no |
| intt 2^22 lazy16 | 16.494 | 1.406% | no |
| coset 2^22 production | 14.422 | 2.137% | no |
| coset 2^22 canonical4 | 16.505 | 0.478% | yes |
| coset 2^22 canonical8 | 16.525 | 1.765% | no |
| coset 2^22 canonical16 | 19.727 | 2.750% | no |
| coset 2^22 lazy4 | 15.534 | 1.362% | no |
| coset 2^22 lazy8 | 16.057 | 0.977% | no |
| coset 2^22 lazy16 | 17.391 | 1.109% | no |
| ntt 2^23 production | 13.160 | 1.219% | no |
| ntt 2^23 canonical4 | 17.176 | 0.633% | no |
| ntt 2^23 canonical8 | 16.343 | 0.995% | no |
| ntt 2^23 canonical16 | 20.787 | 0.307% | yes |
| ntt 2^23 lazy4 | 15.759 | 0.357% | no |
| ntt 2^23 lazy8 | 15.568 | 0.573% | no |
| ntt 2^23 lazy16 | 18.083 | 1.021% | no |
| intt 2^23 production | 12.697 | 0.439% | no |
| intt 2^23 canonical4 | 17.385 | 1.122% | no |
| intt 2^23 canonical8 | 15.781 | 0.434% | no |
| intt 2^23 canonical16 | 19.959 | 0.446% | no |
| intt 2^23 lazy4 | 17.212 | 0.750% | no |
| intt 2^23 lazy8 | 15.841 | 1.187% | no |
| intt 2^23 lazy16 | 18.204 | 0.505% | no |
| coset 2^23 production | 14.680 | 0.583% | no |
| coset 2^23 canonical4 | 17.798 | 0.761% | no |
| coset 2^23 canonical8 | 16.928 | 0.740% | no |
| coset 2^23 canonical16 | 21.601 | 2.101% | no |
| coset 2^23 lazy4 | 16.376 | 0.416% | no |
| coset 2^23 lazy8 | 16.083 | 0.555% | no |
| coset 2^23 lazy16 | 18.851 | 0.780% | no |
| encode 2^20 production | 32.010 | 1.176% | no |
| encode 2^20 canonical4 | 43.111 | 1.587% | no |
| encode 2^20 canonical8 | 40.246 | 0.679% | no |
| encode 2^20 canonical16 | 52.806 | 0.830% | no |
| encode 2^20 lazy4 | 40.945 | 0.747% | no |
| encode 2^20 lazy8 | 39.170 | 1.804% | no |
| encode 2^20 lazy16 | 45.655 | 0.769% | no |
| encode 2^22 production | 34.941 | 0.464% | no |
| encode 2^22 canonical4 | 47.371 | 0.253% | no |
| encode 2^22 canonical8 | 45.219 | 0.924% | no |
| encode 2^22 canonical16 | 55.825 | 0.419% | no |
| encode 2^22 lazy4 | 44.589 | 0.459% | no |
| encode 2^22 lazy8 | 43.376 | 0.916% | no |
| encode 2^22 lazy16 | 50.927 | 2.026% | no |

The best lazy encoder observation is radix-8 at 43.376 ns/element, still slower than production's 34.941. It is not substituted for the pre-registered primary radix-16.

E0 calibration: [count.py](count.py) reproduces EXP-0002's exact radix-2 full-product and butterfly denominators before counting the candidates. Counts retain tail stages and distinguish full products, cheap roots, negative cheap roots, widened shift differences and final canonicalization. The copied radix-16 floor is historical context, not a bound established by this CPU run.

| Encoder arithmetic, per committed element (E0) | Radix4 | Radix8 | Radix16 |
|---|---:|---:|---:|
| full | 23.7501 | 18.3751 | 14.5626 |
| add | 34.0000 | 34.0000 | 34.0000 |
| sub | 34.0000 | 34.0000 | 34.0000 |
| cheap | 8.2500 | 13.6250 | 17.4375 |
| cheap_negative | 2.7500 | 6.1250 | 8.5625 |
| cheap_wide_difference | 0.0000 | 2.6250 | 4.0000 |
| canonicalize | 2.0000 | 2.0000 | 2.0000 |
| traversed_value_bytes | 632.0015 | 456.0015 | 360.0015 |

The double-correction rule makes the lazy additions and subtractions cost an assumed six int32 operations, rather than the historical four. Negative internal roots add a subtraction; exponents at least 64 require a widened shift difference. The actual production roots are `omega8=−2^24` and `omega16=−2^60`, rather than simply using the note's convenient positive roots without compensating the permutation. Alternative primitive-root orderings are a possible future implementation, not evidence from this one.

At radix-16, the resulting assumed encoder count is 885.003–1,013.003 int32 operations per committed element, versus the historical 1,274.000 radix-2 high and 721.375 hypothetical radix-16 low. This realizes about 70.39% of that arithmetic gap at the optimistic translation. The CPU time-gap fraction is **−1.055**, because this implementation slows down: the operation reduction is not a realized ARM64 throughput gain. The static anchor result independently shows that reduced full-product counts do not automatically reduce total instructions after correction and register costs.

With unchanged BLAKE3 247.5–300 int32 operations per element, EXP-0002's H100 peaks and borrowed utilization assumptions, the conditional radix-16 commitment interval is approximately **67.7–140.3 ps** (E0). The exact interval, plus radix4/8 alternatives, is in count.json. It is a scenario for this arithmetic/memory accounting, not a narrowed confidence interval for all possible GPU implementations. It excludes bank conflicts, actual occupancy, twiddle-table/cache traffic beyond the stated value-buffer model, spills, and hash scheduling. `traversed_value_bytes` conservatively charges every fused value-buffer traversal as memory traffic; cache residency can lower HBM traffic, whereas the omitted traffic can raise it. Neither endpoint is a measured hardware bound.

No cost-model μ scenario is claimed: CPU ratios cannot select Hopper utilization or establish the existing 94 ps planning value. The calibrated mechanism is not suitable for a production port on this machine under the written criterion.

## Analysis

The candidate passes every arithmetic/conformance check but misses all three clock thresholds and the instruction threshold. The pre-registered kill criterion fires with clean necessary CVs, making the result refuted, rather than inconclusive. This refutes the pinned safe implementation's CPU realization hypothesis; it is not an impossibility claim about high-radix Goldilocks NTTs or GPU kernels.

The benchmark preserves the production tree and permutation/spread traversals, so the comparison does not intentionally hand the candidate a different transform or output order. Register pressure and the larger precomputed public-power tables are part of the implemented mechanism: the anchor shows additional stack accesses, while the lazy scalar range requires double corrections. These costs explain why counting fewer full multiplications alone is insufficient. Other root orderings, a different register schedule, architecture-specific vectorization or four-step layouts remain untested.

Arithmetic/security reading: `add/sub` bound both possible corrections for arbitrary u64 words; `reduce` bounds all u128 intermediates; `cheap` never shifts by 64 or above in a u128 product of an arbitrary u64 word, using the widened difference for exponents 64–95. `check_arithmetic` checks the residues with direct u128 modular arithmetic and public field powers, while the independent Python Horner vectors cover the complete encoding. All public outputs are canonical and all code forbids unsafe. ARM64 anchor branches are absent, but that observation does not prove whole-kernel/x86_64 constant-time behavior. Only synthetic buffers are used; a production port would require SecretBuf clearing and current secret-access rules.

Threats to validity: one machine/toolchain and no enforced core affinity; fixed arm order can carry thermal/cache effects; larger power tables alter cache pressure; static instruction counts are not dynamic issue counts; copies are included in standalone timing, and full encoder wrappers have different internal layouts with identical values. E0 int32 translations and memory/utilization are assumptions, not SASS or H100 measurements. No real model rows, masking simulator, verifier performance or E3 proof was measured.

## Verdict

`refuted` at E1 for the pre-registered lazy radix-16 CPU/assembly realization hypothesis. IDEA-0015 moves to `refuted` with explicit scope; its CPU implementation priority becomes **0** (was 1.5), confidence in this refutation **0.9** (was 0.3 in the original projected gain). No GPU speedup or Δμ is assigned, and the global c_c planning input is unchanged.

## Follow-ups

An H100 prototype with PTX/SASS and RFC-0020 timing is necessary to determine int32 utilization, carry scheduling, bank conflicts and occupancy. Before investing, compare an alternative primitive-root permutation and a register schedule to the same CPU anchors; a new hypothesis requires a new pre-registration. The optional four-step layout and encode/hash fusion remain separate experiments. No production change or promotion is recommended from this result.
