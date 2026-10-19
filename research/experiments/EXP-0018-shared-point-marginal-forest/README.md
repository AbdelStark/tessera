# EXP-0018: Marginal forest cost in a shared-point leaf-stage batch

| | |
|---|---|
| Status | `done: inconclusive` |
| Idea | [IDEA-0010](../../ideas/IDEA-0010-shared-point-forest-fusion.md), hypothesis(a) only |
| Evidence level | E1 CPU timings; E0 exact loop counts and point trace |
| Hardware | One Apple M4 Max CPU thread; no GPU measurement |
| Pre-registered on | 2026-10-19 |

## Hypothesis

With the current production split-eq tree prover, the marginal time of adding a 2^22-leaf product forest to its naturally shared-point fraction-tree batch is below 90% of proving that forest alone, on every judged forest shape. The original exact-shared-point prove_batch proxy is measured separately to determine whether its older mechanism survives the current production kernel.

## Success criterion

Primary arms on exactly the same synthetic leaves and fraction denominators, in one harness:

- S: standalone current production prove_forests, including forest construction, all levels and leaf claims.
- F: production prove_trees with one unit-numerator fraction tree, including FractionTree construction, all levels and leaf claims.
- B: production prove_trees with the same fraction tree followed by the product forest; includes both constructions, every level and both leaf claims. The schedule order is fixed. Current production kernels are used unchanged; no private split-eq kernel is replaced by a slower generic reference.

Judged 2^22-leaf forest shapes: 2^8 trees of2^14 leaves, 2^10 trees of2^12 leaves, and one tree of2^22 leaves. The fraction tree has 2^22 leaves in every case, height 22. Context cases have2^20 and2^21 total forest/fraction leaves with forest height14; these do not decide the primary verdict.

For paired measured block i, normalized execution times are S_i,F_i,B_i; M_i=B_i−F_i and r_i=M_i/S_i. Report median M and median r, all raw triples, min/max and sample CV. Do not silently replace this paired statistic with a difference of independently selected best runs or independently selected medians. The empirical band[min r_i,max r_i] is a sensitivity band, not a confidence interval.

`validated` requires max(r_i)<0.90 on each judged shape, all M_i>0, and every correctness/noise check below passing. `refuted` requires min(r_i)>=0.90 on each judged shape and the same checks passing. Every other outcome is `inconclusive`: a band crossing0.90, shapes on different sides, unresolved/nonpositive subtraction, or a residual CV flag. This resolves the old note's overlapping success/kill boundary at exactly90% in favor of the kill. A noise failure is not evidence of a cheap marginal forest.

Each base arm's CV and the paired marginal M_i's CV must be<=3%. If any required value exceeds3%, rerun the entire10-block S/F/B comparison once, preserving the first run. Judge only the rerun, with any remaining flag forcing inconclusive. No selective arm/range/window retries. A proof/state/evaluation difference for the same workload is a harness defect to fix before reading timing results.

The primary verdict applies to hypothesis(a), not the entire idea: the original note says either hypothesis(a)'s kill or mechanism 2's kill halves the idea, and both refute it. [EXP-0012](../EXP-0012-integer-leaf-round-window/README.md)'s v1 observation used a32-block cadence; it did not test mechanism 2's specified2^16-term cadence. Neither that observation nor this experiment can globally refute that separate mechanism.

## Method

The frozen plan was committed as 099f6ff before harness code. Its SHA-256 is `d48d27356b84efc4a7420e4543debde367bde0b3c9d965681a369d57dcf9599a`; [the frozen text](results/README.preregistered.txt) and [freeze record](results/preregistration.txt) remain intact. The research-only Rust workspace calls current production APIs unchanged, uses an independent lockfile and existing path dependencies, and forbids unsafe code. No production file changed.

### Current production and point alignment

The current public APIs are prove_batch, Forest/prove_forests and FractionTree/Tree/prove_trees. The production split-eq level kernel and LevelInstance are crate-private. Both its product and fraction relations already avoid a full eq table; the old 2026-10-16 full-eq baseline cannot stand in for current production.

The forest joins a taller fraction tree when their remaining heights meet. At that join its public root point is freshly squeezed and may differ from the fraction's current point. After a common level, equally sized instances receive the same line challenge and same sumcheck point. The harness must trace and assert the actual aligned/nested point pattern at every subsequent level, including the leaf level, and record the initial exception. It must not force equality by changing production challenges or ignore the forest's join work. The one-tree height 22 case joins at the root. This measures the real schedule, rather than assuming that every level starts at the same point.

### Literal shared-point prove_batch proxy

For an explanatory second case, sum independently checked level-shaped workloads over the same forest's level sizes. Each level uses one seeded public eq point r and one full eq_table(r), built once in the timed region, shared by the product and fraction-shaped instances when batched:

- P: eq public over Fp2 times the two Fp forest halves.
- R: eq public over Fp2 times two Fp2 fraction-denominator halves (the original two-Fp2/two-Fp shape). This is a denominator-product proxy, explicitly not a complete LogUp fraction relation.
- T: one prove_batch containing R followed by P, using that same eq slice.

Include forest/fraction table construction consistently and subtract paired T−R against P as for production. Points are fixed input eq points, not sumcheck challenges reused across different workloads. The same90%/CV/marginal-positive rules give this proxy its own verdict; it cannot override the production verdict or reprice the production cost model. Record whether the implementation reuses only the borrowed eq allocation or also its folding work; pointer sharing alone does not prove computational sharing.

### Inputs and timing

SplitMix64 seeds: product leaves0x7E55_E7A0_0018_0001, fraction denominators/points0x7E55_E7A0_0018_0002, small/degenerate correctness0x7E55_E7A0_0018_0003. Forest leaves are22-bit synthetic integers with one seeded zero per tree. Fraction denominators are seeded Fp2 values alpha−key−beta*value, with synthetic22-bit keys and16-bit values; all constructors and clearing/drops included in S/F/B use the current production SecretBuf paths. Check zero denominator cases only as correctness controls, never as a faster benchmark input. No real weights/prompts/tokens.

One process/thread, exclusive CPU timing slot; record CPU/OS/toolchain, UTC, branch/main revisions and code hashes. Each arm has 3 warmups and 10 measured samples, each sample >=20 ms by repetition, repetitions fixed before its paired series with a reserve and actual elapsed duration checked. Normalize by executed repetitions before subtraction. The ten blocks use arm orders SFB,FBS,BSF,BFS,FSB,SBF,SFB,FBS,BSF,BFS. Save order/timestamps/repetitions/raw durations as well as normalized times. This balances order effects without pretending samples are independent or stationary. Cache behavior, scheduler drift and subtracting two larger measurements remain threats even when CV passes.

### Correctness and comparison boundaries

S,F and B are different statements/batches: their transcript challenges and message bytes generally differ. Do not demand or claim byte equality across those three workloads. For each workload, require deterministic replay of the same inputs/context to match every message/final evaluation/claim/transcript state exactly; save full digests and verify the proof with the independent tv-core driver from roots/claims computed using tv-field. Verify the mixed tree levels with tv_core::tree::verify_level, carry both product/fraction claims through the actual join schedule, then independently evaluate the original leaf tables at their final points.

For the generic proxy, tv_core::sumcheck::verify_batch checks each P/R/T statement against independently computed hypercube claims and public eq factors. Same-workload replays must match; cross-workload differences are expected. If a harness optimization is introduced later, it must match the unmodified same-workload production reference bit for bit before timing.

Controls: a hand-sized product tree with root 210 and exact wrong-root rejection, two mixed-height shapes/front padding, all-zero/one eq coordinates and a scripted prior-eq-zero recovery, unit fraction numerators, zero-variable cases, and a deliberately altered final claim/proof rejected at the intended independent check. Report each owed leaf claim rather than discarding it. No zero-knowledge mask or full LogUp root-identity claim is made by the fraction-shaped benchmark alone.

## Reproduction

From the repository root (the timing command requires an exclusive slot):

```sh
cargo run --release --locked --manifest-path research/experiments/EXP-0018-shared-point-marginal-forest/Cargo.toml -- check research/experiments/EXP-0018-shared-point-marginal-forest/results/check.json
cargo run --release --locked --manifest-path research/experiments/EXP-0018-shared-point-marginal-forest/Cargo.toml -- check-large research/experiments/EXP-0018-shared-point-marginal-forest/results/check-large.json
cargo run --release --locked --manifest-path research/experiments/EXP-0018-shared-point-marginal-forest/Cargo.toml -- all research/experiments/EXP-0018-shared-point-marginal-forest/results/run.json
python3 -I research/experiments/EXP-0018-shared-point-marginal-forest/analyze.py
python3 -I research/experiments/EXP-0018-shared-point-marginal-forest/count.py
cargo fmt --check --manifest-path research/experiments/EXP-0018-shared-point-marginal-forest/Cargo.toml
cargo clippy --locked --manifest-path research/experiments/EXP-0018-shared-point-marginal-forest/Cargo.toml --all-targets -- -D warnings
```

Environment: Apple M4 Max, one CPU thread, macOS 26.6.2 (25G83), rustc 1.95.0/LLVM 22.1.2, 2026-10-19. [The pre-timing audit](results/audit-before-timing.json) pins HEAD 46d39e4, source/binary hashes and the exact toolchain. HEAD includes main de5a70c/#385; origin/main had advanced to 2a35a0c at capture but was not merged into this measured binary. The actual tree/prover/verifier source hashes define the tested baseline. No H100 was measured.

## Results

Both registered cases are **inconclusive**. Thirty large-shape arm checks passed deterministic same-workload replay and independent verification before timing; [check-large.json](results/check-large.json) contains every digest and the complete production point traces. [run.json](results/run.json) preserves all 16 attempts (ten initial comparisons, six required reruns), [the raw journal](results/run.json.samples.ndjson) records all 480 samples, and [analysis.json](results/analysis.json) independently recomputes every paired subtraction and verdict. Three warmups per arm preceded the first ten-block series of each comparison (90 warmup records total). A required rerun immediately repeated all ten paired blocks without three fresh warmups, retaining the same repetitions and arm orders. This follows the frozen instruction to rerun the ten-block comparison; it does not provide an independently rewarmed attempt, a method limitation. Repetitions were fixed from the original third warmup with a 60 ms reserve. Measured samples lasted 66.38–361.97 ms (minimum required 20 ms); repetitions were 1, 2 or 4. No extra timing or selective retry followed a residual flag.

E1 primary production, judged 2^22-leaf shapes. Times are medians in milliseconds; M and r are medians of paired block values, so M need not equal the difference between the displayed arm medians. Bands are empirical minima/maxima, not confidence intervals.

| Forest shape | S ms | F ms | B ms | Paired M ms | Median r | Empirical r band | Max arm CV | M CV | Attempts | Shape verdict |
|---|---|---|---|---|---|---|---|---|---|---|
| s8h14 | 77.786 | 264.956 | 342.200 | 77.690 | 100.51% | 97.86–104.41% | 0.55% | 2.13% | 2 | refuted |
| s10h12 | 78.200 | 265.807 | 343.719 | 77.932 | 100.11% | 90.47–110.92% | 1.45% | 4.75% | 2 | inconclusive |
| s0h22 | 78.089 | 265.883 | 343.296 | 76.787 | 98.51% | 95.16–102.84% | 0.60% | 1.99% | 2 | refuted |

Every final production ratio is at least 90%, but the s10h12 marginal CV remains 4.75% after its only permitted rerun. That unresolved subtraction noise forces the primary verdict to inconclusive; the two passing shapes alone do not satisfy the all three kill criterion. All marginal values are positive.

E1 secondary literal generic proxy, judged separately:

| Forest shape | P ms | R ms | T ms | Paired T−R ms | Median ratio | Empirical band | Max arm CV | Marginal CV | Attempts | Verdict |
|---|---|---|---|---|---|---|---|---|---|---|
| s8h14 | 134.138 | 191.042 | 311.513 | 120.429 | 89.68% | 88.38–91.68% | 0.45% | 1.04% | 1 | inconclusive |
| s10h12 | 134.683 | 191.625 | 312.703 | 121.329 | 89.99% | 87.94–92.93% | 0.73% | 1.54% | 2 | inconclusive |
| s0h22 | 134.011 | 190.981 | 310.658 | 119.494 | 89.20% | 87.86–91.34% | 0.58% | 0.83% | 1 | inconclusive |

All proxy noise checks pass on the retained attempt, but every band straddles 90%. Median saving near 10% cannot replace the registered every-block rule.

E1 context cases (not judged primary shapes):

| Leaves / forest | Production median ratio | Production band | M CV | Proxy median ratio | Proxy band | Proxy marginal CV |
|---|---|---|---|---|---|---|
| 2^20 / s6h14 | 100.19% | 97.77–102.72% | 2.23% | 89.15% | 88.30–90.41% | 0.76% |
| 2^21 / s7h14 | 98.36% | 96.04–101.18% | 1.62% | 89.24% | 88.02–90.33% | 0.96% |

### Empirical point alignment and exact accounting

In B, the s8h14 forest joins at global level 8 and s10h12 at level 10. The independently derived entering point digests differ from the fraction's point at each join. After that common level, their next points are equal and remain equal at every later level and at the leaves. The s0h22 case joins at root level 0 with the common empty point. Context s6h14/s7h14 join at 6/7 with the same initial exception. Complete traces, rather than an assumed shared initial point, are saved per workload.

Current production `level::State::add_round` builds its small split eq factors per instance per own round; it already avoids full eq tables/folds. The generic `prover::State::new` instead owns separate bound SecretBuf tables per instance, and `State::bind` folds each read table. T borrows the same initial eq slice for R/P but does not deduplicate their folds. [count.py](count.py) computes the following E0 loop counts; these are field multiplications in table construction, not complete kernel counts or measured times.

| Judged shape | Production split-eq builds S | Production split-eq builds F | Production B | Generic full-eq build saved by T vs P+R | Generic eq folding outputs saved |
|---|---|---|---|---|---|
| s8h14 | 23,938 | 23,992 | 47,930 | 4,194,034 | 0 |
| s10h12 | 23,822 | 23,992 | 47,814 | 4,193,268 | 0 |
| s0h22 | 23,992 | 23,992 | 47,984 | 4,194,281 | 0 |

For a generic n-variable level, the eq build requires 2^n−1 multiplications and subtractions; the bound eq table emits 2^n−1 fold outputs. Summing n=s,…,s+h−1 gives 2^(s+h)−2^s−h. T saves one build relative to P+R and saves zero fold outputs. For a production level, each own round's m remaining free coordinates builds low/high tables costing 2^ceil(m/2)+2^floor(m/2)−2 multiplications; sum m=0,…,n−1 and add the forest root pass. B's count is exactly S+F, even when the points align. This isolates a reason the older generic saving cannot be treated as a saving over the current split-eq baseline. It does not claim that batching overhead/cache interactions are additive.

### Subtraction noise audit

The independent analysis uses sample CV (ddof1) and the actual normalized block triples. It checks Var(B−F)=Var(B)+Var(F)−2Cov(B,F) numerically for every attempt and reconciles every sample with the append-only journal. It asserts nine warmup records per comparison (three in order for each arm), the ten registered block orders, fixed repetitions derived from each arm's third warmup, and the actual 20 ms sample floor. On retained judged production attempts, (mean B+mean F)/mean M is 7.78–7.91; the corresponding proxy factor is 4.16–4.19. These are subtraction scale indicators, not confidence intervals. A low base-arm CV therefore does not establish a low marginal CV: s10h12's retained base CVs are 1.45/1.22/1.26%, while M is 4.75%.

First attempts are not erased: production M CVs were 18.80%, 6.88%, 5.17% for the judged shapes and 5.18% for context s7h14, triggering full-series reruns. Proxy s10h12's first R/T CVs were 5.60/3.78%; context s6h14's P CV was 3.16%; both triggered full-series reruns. All final base-arm CVs pass 3%, all proxy marginal CVs pass 3%, and only the judged production s10h12 marginal retains a flag. No cause is assigned to outliers from this evidence alone; scheduler/frequency drift remains a threat despite serializing local benchmark writers.

### Correctness and adversarial checks

| Check | Evidence |
|---|---|
| Same-workload identity | All proof rounds, final evaluations, owed claims and final transcript state compare exactly on replay for each S/F/B and P/R/T workload; 30 large-shape arm digests retained. Cross-arm byte identity is neither required nor claimed. |
| Independent acceptance | tv-core verifies every tree level from tv-field-computed roots through the actual join schedule; every product leaf MLE and fraction unit numerator/denominator MLE is independently checked. Proxy claims, public eq factors, derived product levels and all final table evaluations are independently checked with tv-field/tv-core. |
| Known value and exact rejection | [2,3,5,7] root 210 passes; root 211 and an altered proof final evaluation each produce exact `level 0: FinalMismatch { round: 0 }`. An altered owed leaf claim is caught by the independently carried claim comparison. |
| Padding and degeneracy | Two different-N forest batches exercise genuine front padding; zero-variable generic levels and a height 0 fraction pass, while a height 0 forest gives exact `Height { forest: 0, height: 0, vars: 0 }`. Zero/one eq coordinates, unit numerators and zero-denominator tree arithmetic pass. |
| Prior-eq-zero recovery | Scripted production split-eq and generic full-eq proofs match exactly; after the forced zero public factor all later round messages are zero, and the independent final eq evaluation is zero. This is an arithmetic control, not a scripted Fiat–Shamir security theorem. |
| Scope and secrets | No verifier/production edit, no custom field arithmetic, synthetic resident inputs only. Current production constructors/scratch clear through SecretBuf in S/F/B; proxy-derived synthetic level Vecs do not claim those production clearing guarantees. No masked proof, LogUp root-balance or zero-knowledge theorem is claimed. |

Both ADV-P and ADV-V readings are scoped to the harness: roots/public factors are independently derived, all owed final claims are discharged, negative controls reject at their intended check, no real secret enters a log, and unchanged production/verifier code retains responsibility for canonical arithmetic, transcript sampling and secret clearing. The external tv-core APIs are the acceptance boundary; this experiment is not a replacement verifier or a full scheme soundness audit.

Verification: the committed preregistration audit, small and large correctness runs, `cargo fmt --check`, strict `cargo clippy` and `cargo xtask lint` pass.

## Analysis

The production data does not establish a cheap marginal forest: all retained pointwise ratios exceed 90%, and two judged shapes meet the refutation rule. The remaining shape fails the preregistered marginal-noise check, so the overall result remains inconclusive. The full-eq proxy shows a median saving around 10% in a different workload, with threshold-crossing empirical bands. It cannot override production or reprice a model term.

The experiment includes tree/forest construction, all levels, output conversion and drops; original synthetic leaves/denominators and the fixed proxy points are generated outside timing. FractionTree construction copies the supplied denominators and constructs its real numerator/denominator levels inside F/B. Alpha/key/value arithmetic that generated those supplied inputs is consistently outside both F and B; no streamed GEMM/key-generation/masking/opening or full LogUp root identity is measured. The proxy is explicitly a denominator product, not the full fraction numerator relation. Resident memory, unit numerators, seeded distributions, unfixed CPU frequency/scheduling and paired subtraction limit external validity.

The current counted leaf default 41.8 ps is unchanged. No E1 ratio here establishes H100 HBM traffic, H100 marginal cost, or a new c_π. The historic 73 ps and 32.2–32.5 ns figures in IDEA-0010 name earlier baselines and are not the comparison used here.

## Verdict

`done: inconclusive` for primary production hypothesis(a); the literal generic proxy is separately `inconclusive`. IDEA-0010 remains `triaged`, priority 0.87 and confidence 0.3 unchanged. Neither the registered half-score kill nor a global refutation fired. EXP-0012's 32-block v1 observation still does not judge mechanism 2's 2^16-term cadence.

## Follow-ups

A fresh preregistration may investigate the production marginal subtraction's noise or a different stable sampling unit; this run licenses no further unregistered retry. Mechanism 2's specified 2^16-term cadence remains a separate experiment. An H100 gate must price actual joint split-eq kernels, memory traffic, masks/openings and streamed regeneration before replacing c_π with a marginal figure. No RFC or production optimization is promoted from these inconclusive results.
