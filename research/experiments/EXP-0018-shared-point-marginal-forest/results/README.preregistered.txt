# EXP-0018: Marginal forest cost in a shared-point leaf-stage batch

| | |
|---|---|
| Status | `planned` |
| Idea | [IDEA-0010](../../ideas/IDEA-0010-shared-point-forest-fusion.md), hypothesis(a) only |
| Evidence level | Planned E1 CPU timings; E0 shape/accounting observations |
| Hardware | One Apple M4 Max CPU thread; no GPU measurement |
| Pre-registered on | 2026-10-19 |

## Hypothesis

With the current production split-eq tree prover, the marginal time of adding a 2^22-leaf product forest to its naturally shared-point fraction-tree batch is below90% of proving that forest alone, on every judged forest shape. The original exact-shared-point prove_batch proxy is measured separately to determine whether its older mechanism survives the current production kernel.

## Success criterion

Primary arms on exactly the same synthetic leaves and fraction denominators, in one harness:

- S: standalone current production prove_forests, including forest construction, all levels and leaf claims.
- F: production prove_trees with one unit-numerator fraction tree, including FractionTree construction, all levels and leaf claims.
- B: production prove_trees with the same fraction tree followed by the product forest; includes both constructions, every level and both leaf claims. The schedule order is fixed. Current production kernels are used unchanged; no private split-eq kernel is replaced by a slower generic reference.

Judged 2^22-leaf forest shapes: 2^8 trees of2^14 leaves, 2^10 trees of2^12 leaves, and one tree of2^22 leaves. The fraction tree has2^22 leaves in every case, height22. Context cases have2^20 and2^21 total forest/fraction leaves with forest height14; these do not decide the primary verdict.

For paired measured block i, normalized execution times are S_i,F_i,B_i; M_i=B_i−F_i and r_i=M_i/S_i. Report median M and median r, all raw triples, min/max and sample CV. Do not silently replace this paired statistic with a difference of independently selected best runs or independently selected medians. The empirical band[min r_i,max r_i] is a sensitivity band, not a confidence interval.

`validated` requires max(r_i)<0.90 on each judged shape, all M_i>0, and every correctness/noise check below passing. `refuted` requires min(r_i)>=0.90 on each judged shape and the same checks passing. Every other outcome is `inconclusive`: a band crossing0.90, shapes on different sides, unresolved/nonpositive subtraction, or a residual CV flag. This resolves the old note's overlapping success/kill boundary at exactly90% in favor of the kill. A noise failure is not evidence of a cheap marginal forest.

Each base arm's CV and the paired marginal M_i's CV must be<=3%. If any required value exceeds3%, rerun the entire10-block S/F/B comparison once, preserving the first run. Judge only the rerun, with any remaining flag forcing inconclusive. No selective arm/range/window retries. A proof/state/evaluation difference for the same workload is a harness defect to fix before reading timing results.

The primary verdict applies to hypothesis(a), not the entire idea: the original note says either hypothesis(a)'s kill or mechanism2's kill halves the idea, and both refute it. [EXP-0012](../EXP-0012-integer-leaf-round-window/README.md)'s v1 observation used a32-block cadence; it did not test mechanism2's specified2^16-term cadence. Neither that observation nor this experiment can globally refute that separate mechanism.

## Method

This file is preregistration only. No code, build or benchmark accompanies it until the parent reviews and commits the frozen text. A later implementation is a research-only Rust workspace with its own Cargo.lock, path dependencies on current prover/verifier crates, SPDX headers, Apache-2.0 OR MIT license and forbid(unsafe_code). No production edits or new dependencies.

### Current production and point alignment

The current public APIs are prove_batch, Forest/prove_forests and FractionTree/Tree/prove_trees. The production split-eq level kernel and LevelInstance are crate-private. Both its product and fraction relations already avoid a full eq table; the old 2026-10-16 full-eq baseline cannot stand in for current production.

The forest joins a taller fraction tree when their remaining heights meet. At that join its public root point is freshly squeezed and may differ from the fraction's current point. After a common level, equally sized instances receive the same line challenge and same sumcheck point. The harness must trace and assert the actual aligned/nested point pattern at every subsequent level, including the leaf level, and record the initial exception. It must not force equality by changing production challenges or ignore the forest's join work. The one-tree height22 case joins at the root. This measures the real schedule, rather than assuming that every level starts at the same point.

### Literal shared-point prove_batch proxy

For an explanatory second case, sum independently checked level-shaped workloads over the same forest's level sizes. Each level uses one seeded public eq point r and one full eq_table(r), built once in the timed region, shared by the product and fraction-shaped instances when batched:

- P: eq public over Fp2 times the two Fp forest halves.
- R: eq public over Fp2 times two Fp2 fraction-denominator halves (the original two-Fp2/two-Fp shape). This is a denominator-product proxy, explicitly not a complete LogUp fraction relation.
- T: one prove_batch containing R followed by P, using that same eq slice.

Include forest/fraction table construction consistently and subtract paired T−R against P as for production. Points are fixed input eq points, not sumcheck challenges reused across different workloads. The same90%/CV/marginal-positive rules give this proxy its own verdict; it cannot override the production verdict or reprice the production cost model. Record whether the implementation reuses only the borrowed eq allocation or also its folding work; pointer sharing alone does not prove computational sharing.

### Inputs and timing

SplitMix64 seeds: product leaves0x7E55_E7A0_0018_0001, fraction denominators/points0x7E55_E7A0_0018_0002, small/degenerate correctness0x7E55_E7A0_0018_0003. Forest leaves are22-bit synthetic integers with one seeded zero per tree. Fraction denominators are seeded Fp2 values alpha−key−beta*value, with synthetic22-bit keys and16-bit values; all constructors and clearing/drops included in S/F/B use the current production SecretBuf paths. Check zero denominator cases only as correctness controls, never as a faster benchmark input. No real weights/prompts/tokens.

One process/thread, exclusive CPU timing slot; record CPU/OS/toolchain, UTC, branch/main revisions and code hashes. Each arm has3 warmups and10 measured samples, each sample>=20ms by repetition, repetitions fixed before its paired series with a reserve and actual elapsed duration checked. Normalize by executed repetitions before subtraction. The ten blocks use arm orders SFB,FBS,BSF,BFS,FSB,SBF,SFB,FBS,BSF,BFS. Save order/timestamps/repetitions/raw durations as well as normalized times. This balances order effects without pretending samples are independent or stationary. Cache behavior, scheduler drift and subtracting two larger measurements remain threats even when CV passes.

### Correctness and comparison boundaries

S,F and B are different statements/batches: their transcript challenges and message bytes generally differ. Do not demand or claim byte equality across those three workloads. For each workload, require deterministic replay of the same inputs/context to match every message/final evaluation/claim/transcript state exactly; save full digests and verify the proof with the independent tv-core driver from roots/claims computed using tv-field. Verify the mixed tree levels with tv_core::tree::verify_level, carry both product/fraction claims through the actual join schedule, then independently evaluate the original leaf tables at their final points.

For the generic proxy, tv_core::sumcheck::verify_batch checks each P/R/T statement against independently computed hypercube claims and public eq factors. Same-workload replays must match; cross-workload differences are expected. If a harness optimization is introduced later, it must match the unmodified same-workload production reference bit for bit before timing.

Controls: a hand-sized product tree with root210 and exact wrong-root rejection, two mixed-height shapes/front padding, all-zero/one eq coordinates and a scripted prior-eq-zero recovery, unit fraction numerators, zero-variable cases, and a deliberately altered final claim/proof rejected at the intended independent check. Report each owed leaf claim rather than discarding it. No zero-knowledge mask or full LogUp root-identity claim is made by the fraction-shaped benchmark alone.

## Reproduction

After the preregistration commit and implementation, from the repository root:

```sh
cargo run --release --locked --manifest-path research/experiments/EXP-0018-shared-point-marginal-forest/Cargo.toml -- check
cargo run --release --locked --manifest-path research/experiments/EXP-0018-shared-point-marginal-forest/Cargo.toml -- all research/experiments/EXP-0018-shared-point-marginal-forest/results/run.json
```

The manifest/program do not exist in the planned phase. Freeze this file in results/README.preregistered.txt and record SHA-256 plus UTC in results/preregistration.txt, then verify the committed bytes before code.

## Results

Not run. Preregistration only.

## Analysis

This tests a CPU marginal cost in a particular joint workload. The difference includes batching/transcript overhead and cache interactions, not only removable eq arithmetic. Current split-eq production may leave almost none of the old full-eq saving. The ideal shared-point generic proxy may disagree with production because it does different work and deliberately fixes all input points. Paired subtraction can amplify noise; a small/negative marginal estimate is inconclusive rather than a negative per-leaf price. No E1 ratio establishes H100 HBM savings or the true c_pi, and no model unit cost is automatically replaced by this run.

## Verdict

Pending; the primary and proxy cases will be reported separately against the frozen rules.

## Follow-ups

The H100 gate must price joint tree kernels, actual memory traffic and the relevant same/nested-point layouts before changing c_pi. Mechanism2's2^16-term integer accumulator remains a separate preregistration. Full-stage mask/opening integration and streamed regeneration pass sharing are outside this CPU cost test.
