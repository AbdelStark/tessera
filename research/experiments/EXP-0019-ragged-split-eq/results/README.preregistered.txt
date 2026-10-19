# EXP-0019: Ragged constant tails combined with the current split-eq forest prover

| | |
|---|---|
| Status | `planned` (preregistration only; no harness code or timing) |
| Idea | [IDEA-0019](../../ideas/IDEA-0019-analytic-padding-blocks.md) |
| Evidence level | E1 CPU timing and E0 exact checks/counts; no E2/E3 claim |
| Hardware | Apple M4 Max, one process/thread; macOS 26.6.2 (25G83); Rust/Cargo 1.95.0 |
| Pre-registered on | 2026-10-19; frozen copy/hash/UTC in `results/` |

## Hypothesis

On each original EXP-0009 judged shape S1–S4, combining ragged constant-tail storage and shared public boundary metadata with the **current production split-eq/Gruen algorithm** recovers the padding factor to within 10% and costs at most 5% more per real leaf than current production on the unpadded U23 reference, with exact production messages/claims/transcript and independent acceptance, under the paired noise gates below.

## Success criterion

For shape `c`, let `N_real(c)` include every original nonpadding leaf, including a zero leaf, and let `N_pad(c)` include the materialized padding of all its forests. Set `f_c=N_pad/N_real`. Count neither visited pairs nor a new packed-buffer size as “real leaves.”

In paired block `i`, `A_ci` is current production time per proof on `c`, `B_ci` is candidate time, and `U_ci` is a co-timed current-production proof of the **same fixed U23 input**. Define:

```text
R_c = median_i(A_ci / B_ci)
H_c = median_i((B_ci / N_real(c)) / (U_ci / N_real(U23)))
```

The original per-real-leaf meaning is retained: the entire proof, including level construction, allocations, public metadata/sorting, folds and transcript work, is divided by the original real input-leaf count. U23 is a full, unpadded production proof, not a candidate-only microkernel. Co-timing U23 separately for each case changes noise control, not its leaves or denominator.

Also compute aggregate arm time per proof as `sum(sample elapsed)/sum(sample repetitions)`, and aggregate ns/real as `sum(sample elapsed)/sum(sample repetitions*N_real)`. Apply the same factor interval and per-real ceiling to the resulting aggregate `R_agg,H_agg`; both aggregate and paired-median criteria must pass. Save ratios of arm medians and absolute ns/real and ns/padded figures as well. For every ratio, show its ten-block empirical min/max band and sample CV; these are observed spread, not confidence intervals. No paired subtraction, inferred common overhead, clipping or denominator correction is used. The U23 denominator's variability is explicitly included in the paired `H` CV rather than treated as exact timing.

- **`validated`:** every `c∈{S1,S2,S3,S4}` has `0.90≤R_c/f_c≤1.10`, `0.90≤R_agg/f_c≤1.10`, `H_c≤1.05` and `H_agg≤1.05`; all exact checks pass; all required per-arm and paired-ratio CVs are at most 3% after at most one complete block-cohort rerun; and the sham-noise control passes.
- **`refuted`:** exact checks and noise gates pass, but some judged shape has `R_c≤1+(f_c−1)/2` (at most half of its padding benefit recovered).
- **`inconclusive`:** every other performance result, including a per-real-leaf failure, an overshoot of the factor interval, failed CV/noise control, or implementation/correctness blockage. The previous experiment and its amendment remain inconclusive regardless of this result.

A mismatched proof is a harness defect, not a speedup. Fix it under these frozen inputs/criteria before judging, or record the blockage without a performance verdict. Never select a third timing cohort, discard a slow sample, change a shape or weaken a threshold to obtain a result. Mechanism/domain changes require a dated, reviewed and committed amendment before dependent code.

## Method

### Why this is an uncovered mechanism

At preregistration, `prove_forests` in `forest.rs` delegates to `prove_trees`. Forest levels are built densely by `leaf_level/build_levels` from a required power-of-two slice. `trees.rs` then calls `level.rs`, whose split-eq factorization builds two roughly square-root-sized weight tables, sums `v(0),v(2)`, recovers `v(1)` from the current claim and computes it directly when the public denominator is zero. It has no real-prefix/constant-subtree input for product forests.

Accepted [RFC-0034 “Constant leaves”](../../../docs/rfcs/RFC-0034-causal-blocks.md#constant-leaves) already requires closed-form constant subtrees with byte-identical messages. This is implementation evidence for that requirement, not a new protocol proposal. The [EXP-0009 amendment](../EXP-0009-analytic-padding-blocks/README.md#amendment-2026-10-19-amortized-ragged-loop) kept per-tree materialized eq and showed no useful gain over landed split-eq production. Neither that arm nor production implements the composition tested here. The source of split-eq is commit `2ac8e59`; current checkout and complete relevant file hashes are in [results/source-pins.json](results/source-pins.json).

### Baseline and candidate

- **A, `production`:** call the pinned `tessera_sumcheck::prove_forests` without modifying its crate, on the original fully materialized padded slices. This is the judged baseline, including #258.
- **B, `ragged-split-eq`:** an isolated research prover with the same outer forest/level schedule, original tree order, public constants and transcript labels. Build/store only each tree's necessary real-prefix levels and mixed boundary nodes; all-constant-one subtrees stay implicit. At a level, retain production's split-eq scalar/current-line/rest factors and two-node/Gruen recovery. Accumulate `v(0),v(2)` over real pairs plus exact closed-form dyadic tail weights; compute `v(1)` directly in the zero-denominator branch. Fold only necessary data/boundary entries. Restore original tree-index order before tree-coordinate folds and final evaluations. Sorting/equal-length runs and odd-boundary batching use public lengths only.

The split weights are indexed by the original padded hypercube coordinate, not packed-array offsets. A folded mixed boundary need not equal one and must remain explicit. The all-one tail contribution is present at each evaluation node even when it is never materialized. When position bits end, preserve tree-coordinate interpolation, padded trees and every forest's front-padding factors/join level. Do not re-group trees across forests or change their height to improve a number.

The candidate may reuse public field/poly/transcript primitives. It does not invoke production for its measured computation or copy verifier arithmetic. Its reconstructed claim is computed from its own levels/constant terms, not supplied from an untimed production proof or assumed from known zero roots. Use existing canonical field operations; new unchecked wide accumulators, custom SIMD, threading, lazy reductions or protocol-2 masks are outside this unit.

No contextual old-analytic arm enters the decision. If retained for context it must be labelled historical and cannot substitute for A. U23 candidate, U24 and 1K shapes are secondary controls only.

### Inputs and fixed shapes

Generate all eight timed inputs once, in the original EXP-0009 order below, using SplitMix64 seed `0x7E55_E7A0_0009_0001` and its pinned generator. A real tree has one seeded zero leaf, real leaves are reduced `Fp` words, and padded leaves/trees are one. The baseline and candidate read identical values with independent scratch. Correctness-only inputs use `0x7E55_E7A0_0009_0002` in the pinned check order. Do not regenerate timed inputs per arm or reorder RNG consumption.

| Shape | Definition | Padded leaves | Real leaves |
|---|---|---:|---:|
| S1 | 4,096 causal rows, `q+1` real leaves, one height 12 | 16,777,216 | 8,390,656 |
| S2 | Same rows grouped by existing heights 1–12, tallest first | 11,184,812 | 8,390,656 |
| S3 | 640 real full height-14 trees in a forest padded to 1,024 trees | 16,777,216 | 10,485,760 |
| S4 | 2,048 causal rows for five head-layer pairs, original grouped heights | 22,369,632 | 10,490,880 |
| U23 | 512 full height-14 trees | 8,388,608 | 8,388,608 |
| U24, secondary | 1,024 full height-14 trees | 16,777,216 | 16,777,216 |
| S1-1K, secondary | Original one-height shape, `T=1024` | 1,048,576 | 524,800 |
| S2-1K, secondary | Original grouped shape, `T=1024` | 699,052 | 524,800 |

These are synthetic product-forest kernel shapes. S1 versus S2 is an already-fixed schedule comparison, not a new grouping choice under RFC-0034. The RFC's causal advice-row permutation/`score_block` affects commitments; it does not change this tree's key-lowest leaf order. No complete causal-layout or proof-engine integration is claimed.

The context remains `tessera-protocol/1`, noninteractive, `test-small/1`, statement digest `[0x5a;32]`, session id `[7;16]`, with the existing tree labels. This is an unmasked kernel-equivalence test, not evidence that protocol/2's RFC-0032 masks can be skipped or that a real proof meets test-small parameters.

### Correctness and adversarial checks before timing

1. Reproduce all eight production transcript-state digests in pinned EXP-0009 `results/amendment-run.json`. Preserve the hand case `[2,3,1,1,5,7,11,1]`, true roots 6/385, all five original small shapes and three amendment unsorted/repeated-length vectors `[13,1,13,7,2,7,16]`, `[3;7]`, `[1,2,3,4,5,6,7,8]`. Preserve their seed order. New controls are appended after these generators.
2. Every candidate round message/final evaluation, forest index/join order, next/leaf claim and final transcript state must equal production exactly. Compare full canonical outputs, not just final state. Verify **both** proofs using `tv-core::tree::verify_product_level`, `tv-field` and `tv-hash`, with roots and padded-leaf MLEs computed independently in verifier arithmetic. No prover `eq`, fold or field implementation judges acceptance. The wrong-root hand case must retain exact independent rejection `level 0: FinalMismatch { round: 1 }`.
3. New checks cover public real-prefix lengths `0,1,2,3,4,7,8,15,16`, all-constant padded trees, height-one/no-own-variable levels, mixed heights, odd boundaries and non-lane-multiple runs. Zero-length prefixes mean public all-one trees and are correctness-only, never a zero real-leaf denominator. Include leaves 0,1,`P−1`, `2^32−1`, `2^32` and values near `P`; choosing a path must never inspect a leaf value.
4. Script public challenges to exercise eq endpoints `r_k=0,1`, the folded-line challenge `rho=0,1`, and `scale*r_k=0`: a zero current eq coordinate and a zero accumulated scale after an earlier line vanishes. Both implementations and the independent driver receive the same explicit public script. Assert exact `v(1)`/message agreement rather than relying on a random `2^-128` event. Product-only `prove_forests` squeezes no `lambda`; explicitly assert that schedule. Fraction trees and mixed product/fraction wrappers are outside this unit. These scripts test branches, not Fiat–Shamir security.
5. In an isolated scratch copy, a no-op control must pass. Mutate one behavior at a time: drop the constant-tail sum; use zero instead of one at an odd padded leaf; skip original tree-order restoration (or use an explicitly wrong original tree-index mapping if the implementation has no traversal permutation); index eq by packed coordinates; omit direct `v(1)` fallback. Each behavior-changing fault must fail the relevant full-message/independent check; a mutation that leaves behavior unchanged does not count as coverage. Preserve stdout, exact rejection/difference and source hash per mutation. Never mutate measured source or baseline crates.

### Paired CPU protocol and noise gate

No timing occurs until the parent grants an exclusive slot and every other heavy research process is paused. Compile/tests finish before the slot. Record OS/CPU/compiler/source hashes and any visible interference; do not claim an otherwise idle machine without evidence.

For each judged shape, measure **three warm-up blocks and ten measured blocks**. A block contains A(c), B(c), U(current U23). Rotate the six orders `ABU,BAU,UBA,UAB,AUB,BUA` by block index, continuing the counter from warmups; all ten measured orders are saved. Each arm sample repeats a complete fresh-transcript proof until total duration is at least 20 ms, reports time per proof and retains total elapsed time/repetitions. Input generation and independent verification are outside timing; complete prover scratch construction, metadata, allocations, level work, messages and drops are inside. Consume the complete output through `black_box`.

Before judged timings, run ten sham A/A production pairs on fixed S2, after three warm-up pairs, alternating order. Its median first/second ratio must be in `[0.97,1.03]`; both arm CVs and paired-ratio CV must be at most 3%. Record which invocation is “first,” order and all durations. This checks the paired measurement floor and is not a kernel arm.

CV means sample standard deviation divided by arithmetic mean of per-proof durations or paired ratios, as labelled. Compute CV for A, B, U and both `A/B` and `(B/N_real)/(U/N_U23)` in each primary cohort. Dividing a ratio by the fixed exact padding factor does not change its CV. These gates cover every deciding quantity, including the component durations and block ratios underlying the aggregate decisions. If any required CV exceeds 3%, rerun the **complete shape's three-arm cohort once**, retaining the first attempt; use only the prescribed second cohort for judgment. Every rerun starts with three fresh warm-up blocks, giving each arm three new warmups before its ten new samples. A failed sham control likewise gets at most one complete rerun with three fresh warm-up pairs. There is no third cohort, selective arm rerun or sample filtering. A persistent failure yields `inconclusive` even if point estimates look favourable.

Secondary paired A/B cohorts on U23, U24, S1-1K and S2-1K use the same three-warmup/ten-sample/20-ms policy. They report noise and potential unpadded overhead but do not choose the primary verdict or get additional retries. Save arm and paired medians/min/max/CVs, all sample orders/timestamps/repetitions, first attempts and exact output digests in separate raw JSON/text files. No timing result or confidence interval is inferred before measurement.

### Exact counts, integer and field bounds

Emit per-shape/level counts independently of clocks: original real/padded leaves, stored real/mixed nodes, visited pairs, constant blocks, boundary folds, eq-weight elements rebuilt, front-padding rounds, public inversions/fallbacks, allocation bytes and copied/visited field elements. Sorting/run construction and all residual work remain counted. Counts of bytes touched are not measured DRAM traffic or a GPU roofline result.

This registration uses 64-bit `usize` with checked shapes/products/shifts. Each forest has at most `2^24` padded leaves; each entire timed batch has fewer than `2^25`; all stored dense levels have fewer than `2^26` base elements. Height and tree-index bits are at most 24 combined, so `1usize<<bits`, byte sizes (8-byte Fp/16-byte Fp2), offsets and next-power-of-two calculations fit well below `isize::MAX`; still check before allocation. Metadata prefixes must lie in `[0,2^h]`, tree counts in `[0,2^s]`, and slice bounds must be checked. Do not subtract a prefix from a smaller domain or shift by the machine width.

Use normative Goldilocks `P=2^64−2^32+1` and `Fp2=Fp[i]/(i²−7)`; no Fp4 substitution from a stale performance note. Every raw base product is below `(P−1)^2<2^128`; no unreduced integer sum of products is introduced. Sums/tail weights, `2^t` coefficients and folded constants are field operations, not overflowing machine products. The existing field primitives retain their own reduction bounds. A zero public denominator takes the exact direct-sum branch, never division by zero. These bounds do not constitute a new field-implementation or protocol-soundness proof.

## Reproduction

**Before any code:** parent reviews and commits this README, its frozen byte copy, `results/source-pins.json` and `results/preregistration.txt`. Verify the frozen copy against `git show HEAD:<path>` and its registered SHA-256 before creating any `.rs`, `.py`, Cargo manifest or lockfile. If absent/different, stop at `planned`. This turn creates no harness/build/timing artifact.

After that commit, create one isolated safe Rust workspace under this directory, SPDX headers, dual licence, `publish=false`, no unsafe, path dependencies on current prover primitives and the independent verifier, its own Cargo.lock. Planned commands from the repository root:

```sh
cd research/experiments/EXP-0019-ragged-split-eq
cargo test --locked
cargo clippy --all-targets --locked -- -D warnings
cargo fmt --all --check
cargo run --release --locked -- check > results/check.txt
# only after the exclusive timing grant:
cargo run --release --locked -- paired results/run.json > results/run.txt
# planned exact counter/mutation commands, implemented only after commit:
python3 -I count.py > results/count.txt
python3 -I mutate.py > results/mutations.txt
```

The first post-registration dependency resolution creates the isolated lockfile before locked checks; record its hash and final measured source hash. The pins file protects baseline provenance; if relevant production or verifier code changes before measurement, stop and register the new baseline explicitly instead of silently updating it.

## Results

| Shape | Production | Combined candidate | `R_c/f_c` | `H_c` | Noise gate |
|---|---|---|---|---|---|
| S1–S4 | not run | not run | not run | not run | not run |

Raw results will be separate files in `results/`; no EXP-0009 result is overwritten.

## Analysis

Not run. The comparison is against current dense split-eq production, not the slower historical dense/materialized-eq harness. The original 5% normalization remains intentionally demanding and may still fail because small forests, tree-coordinate folds or bookkeeping cost more per real leaf. The paired design addresses the observed S1 drift but cannot guarantee a quiet host. CPU/synthetic shapes do not establish H100 occupancy, full RFC-0034 integration, fraction-tree savings or protocol-2 hiding.

## Verdict

`planned`; judgment awaits the committed preregistration, implementation and all registered checks. IDEA-0019 and EXP-0009 remain `inconclusive`; historical priorities/ratios are unchanged.

## Follow-ups

If validated, hand over the exact messages, counts and current-source comparison for a separate production-port review under accepted RFC-0034. H100 RFC-0020 follow-up must measure the composition's tile skipping, boundary occupancy and full memory traffic before pricing a GPU gain. Fraction trees, masked tree levels and complete proof integration need separate evidence. The current model already counts valid tree/lookup leaves; do not add a second padding saving or multiply the old dense ratio into current `leaf_ps=41.8`. No μ re-score or model change belongs to this preregistration.
