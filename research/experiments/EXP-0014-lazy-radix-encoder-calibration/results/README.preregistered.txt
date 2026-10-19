# EXP-0014: Calibrate lazy radix-4/8/16 Goldilocks encoder arithmetic on CPU

| | |
|---|---|
| Status | `planned` |
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

Pending committed pre-registration and implementation.

## Analysis

Pending. Explicit threats: CPU/GPU differences, operation-count translation, hash fraction, cache/traversal effects, compiler instruction scheduling, fixed arm order, no affinity, one machine/toolchain and synthetic input distribution.

## Verdict

Pending.

## Follow-ups

E2 H100 timing and PTX/SASS inspection settle actual utilization, carry instruction selection, memory traffic, shared-memory conflicts and occupancy. A production port requires SecretBuf ownership/clearing and binary branchlessness checks; this research code uses synthetic buffers only. Four-step layout and encode/hash fusion remain separate ideas, rather than additions to this calibration.
