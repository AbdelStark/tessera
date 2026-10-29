# EXP-0012: Integer leaf round window

| | |
|---|---|
| Status | `done: refuted` (primary integer-forest criterion); hosted fraction case `inconclusive` |
| Idea | [IDEA-0022](../../ideas/IDEA-0022-integer-leaf-level-round-window.md), hosting IDEA-0017's in-memory window and IDEA-0010 mechanism 2 |
| Evidence level | E0 operation counts; E1 CPU wall clock; no E2 GPU evidence |
| Hardware | Apple M4 Max, macOS; exact environment saved with the run |
| Pre-registered on | 2026-10-19 |

## Hypothesis

A leaf-level integer grid over the first v rounds, v in {1,3,4,5}, preserves the production transcript and independent verifier acceptance, and its best fixed v in {3,4,5} saves at least 20% of whole-forest time against the EXP-0008 split-eq Gruen kernel on all three 2^22-leaf shapes.

## Success criterion

The judged baseline is EXP-0008's split-eq Gruen forest kernel ported to the current production transcript schedule; current production prove_forests is a separately timed context arm. For each v, sigma = 1 - median(candidate)/median(split). The judged shapes are 2^8 trees of 2^14 leaves, 2^10 trees of 2^12 leaves and one tree of 2^22 leaves. Context shapes have 2^20 and 2^21 leaves with height 14. Every arm includes level construction, root claims, all level proofs and leaf claims.

`validated` requires one fixed v in {3,4,5} with sigma >= 0.20 on every judged shape, bit-identical round messages, final evaluations, claims and transcript states to split and production, independent tv-core acceptance, and CV <= 3% for every arm used. `refuted` requires every v in {3,4,5} to have sigma < 0.15 on every judged shape and all correctness/CV checks passing. Everything else is `inconclusive`, including a remaining CV flag after one rerun. v=1 hosts IDEA-0010 mechanism 2 as an observation. The standalone full leaf-level kernel at 2^21 pairs is reported for all v as the note's kill-criterion diagnostic; it does not replace the whole-forest criterion. A differing digest is a harness bug to fix before judgment.

## Method

Port the EXP-0008 harness to current crates without production changes. Only the bottom integer leaf level uses a window; all higher levels use the same split kernel. Grid points {0,1,2}^v evaluate the two multilinear leaf halves as signed integers. Products are accumulated with decomposed eq weights over the remaining variables, then the grid collapses by quadratic Lagrange interpolation after each challenge. The current variable's eq factor and previous eq scale form the cubic message; all messages/challenges remain in production order. The original leaf tables bind once to the v challenges, and the split kernel finishes the remaining rounds. Window work includes grid formation, integer accumulation and the once-bound tables; no preprocessing is excluded.

Overflow cadence is fixed before code. Keys are at most K=2^22-1. For v<=5, abs(V)<=K*3^v, abs(product)<=K^2*9^v, and scaling by a canonical Goldilocks component is below 2^64*K^2*9^v. Accumulate at most B(v) blocks before signed-i128 reduction, with B(1)=32, B(3)=32, B(4)=32 and B(5)=8. The exact check B(v)*(2^64-1)*K^2*9^v < 2^127 is asserted independently before proving. (The note's coarse v=4 bound leaves no slack at 32 blocks; the exact integer inequality is authoritative.) Reduction handles signed sums exactly. A split-eq implementation may reduce at the end of each inner block group, never later than B(v); the actual cadence and extra reductions are counted. Fixed loops/cadence depend only on shapes, not key values.

Synthetic inputs use EXP-0008 SplitMix64 seeds: leaves 0x7E55_E7A0_0058_0001, isolated level 0x7E55_E7A0_0058_0002, correctness 0x7E55_E7A0_0058_0005, one seeded zero per tree, all other keys below 2^22. Check hand root 210 for [2,3,5,7], wrong-root rejection, small sizes/window boundaries, two forests with front padding, maximum integer leaves, and eq/challenge degeneracies (zero, one and prior eq scale zero). For every timed forest compare all proof bytes/digests, claims and transcript states to production and run the independent tv-core tree verifier from roots computed in tv-field. No verifier code changes.

RFC-0020 timing: one thread, 3 warmups, 10 measured samples, median/minimum/maximum/CV, each measured sample >=20 ms by repetition. Rerun each arm once if CV>3%, preserving both raw records. Save raw samples, run environment (CPU/OS/toolchain/date/git commit), and JSON. Time grid formation/accumulation separately as an observation so the i128 component's contribution is visible; isolating integer accumulation is a microkernel observation, not an additive attribution. count.py records exact structural counts and an explicitly assumed H100 int32 translation, never GPU measurements. Convert any leaf_ps scenario only using the existing counted cost-model tool.

## Reproduction

From the repository root:

```sh
cargo run --release --manifest-path research/experiments/EXP-0012-integer-leaf-round-window/Cargo.toml --locked -- check
cargo run --release --manifest-path research/experiments/EXP-0012-integer-leaf-round-window/Cargo.toml --locked -- all research/experiments/EXP-0012-integer-leaf-round-window/results/run.json
python3 -I research/experiments/EXP-0012-integer-leaf-round-window/count.py
```

The first build creates the independent Cargo.lock; no root workspace lockfile changes. Preregistration text is frozen in results/README.preregistered.txt and its SHA-256/time in results/preregistration.txt before harness code.

## Results

E1, Apple M4 Max, macOS 26.6.2, 48 GiB, rustc/cargo 1.95.0, one thread. Git revision `2d61eed410565563f4146220bbdda76267db05b9` (current production main merged at `7265027`); exact environment and harness SHA-256 are in [results/environment.json](results/environment.json). No CPU affinity or frequency lock was used. All E1 numbers below are medians of 10 samples after 3 warmups. Every measured sample exceeded 38.33 ms, including repeated executions where needed; minimum/maximum/CV and repetitions are in [results/run.json](results/run.json). All raw samples and the discarded first attempts are in [results/run.txt](results/run.txt).

Whole forests, nanoseconds per leaf (E1). The first three rows are the 2^22-leaf judged shapes; the final two have 2^20 and 2^21 leaves.

| Shape | Split v0 | v1 | v3 | v4 | v5 | Production |
|---|---|---|---|---|---|---|
| s8-h14 | 18.010 | 20.134 | 16.572 | 17.835 | 21.460 | 18.228 |
| s10-h12 | 17.996 | 19.792 | 16.758 | 17.591 | 21.597 | 18.127 |
| s0-h22 | 18.171 | 19.910 | 16.486 | 17.624 | 21.423 | 18.329 |
| s6-h14 | 18.090 | 20.030 | 16.779 | 17.803 | 21.634 | 18.387 |
| s7-h14 | 18.150 | 20.237 | 16.959 | 17.816 | 21.960 | 18.605 |

For v3, sigma is 7.98%, 6.88% and 9.27% respectively. v4 saves only 0.97–3.01%; v5 is slower. No judged width reaches 15% on any shape. Judged-arm CVs are at most 2.31%.

Standalone leaf level at 2^21 pairs and component observations (E1). Divide ns/pair by two for ns/original leaf.

| Arm | Full leaf level ns/pair | Grid formation/accumulation ns/pair | i128 microkernel ns/product (CV) |
|---|---|---|---|
| split v0 | 16.950 | — | — |
| v1 | 20.872 | 5.784 | 1.1965 (4.76%) |
| v3 | 14.284 | 8.091 | 1.1996 (3.16%) |
| v4 | 16.462 | 11.283 | 1.2375 (3.32%) |
| v5 | 23.741 | 18.450 | 1.5490 (0.76%) |

The standalone v3 level saves 15.73%; this clears the note's 15% isolated-level diagnostic but does not satisfy its whole-forest hypothesis. i128 microkernels v1/v3/v4 remain flagged after the prescribed rerun and support no precise component attribution. v5's stable microkernel gives an E0-over-E1 proxy of `1.5490 * (3/2)^5 / 18.4497 = 63.8%` of grid time; its cached precomputed operands differ from the complete grid, and this is not an additive decomposition or GPU prediction.

Every correctness check passed: signed reduction against Python-style exact modulo semantics on boundaries and 10,000 seeded full-range i128 words; exact overflow inequalities; small sizes/window boundaries n=0..12, front padding, maximum leaves, hand root 210 and exact wrong-root rejection `FinalMismatch { round: 0 }`, two mixed-height forests, and zero/one/prior-eq-zero challenges. Every timed forest width had every message/final evaluation/leaf claim/state bit-identical to current production, independent tv-core tree acceptance from roots computed in tv-field, and leaf claims equal to the independently evaluated leaf MLE. Timed fraction widths likewise equalled current production prove_batch's three proxy instances and passed independent tv-core verify_batch. Full fraction digests and abbreviated forest digests are in the raw output; full forest digests are compared inside the harness. [results/check.txt](results/check.txt) contains the correctness-only run.

E0 counts are reproducible with [count.py](count.py), saved in [results/count.json](results/count.json). At 2^21 pairs, integer subtotals for v1/v3/v4/v5 are 56/131/198.5/345.3125 assumed int32 operations per pair. These exclude separately reported once-binding, high-group field scaling, eq construction and grid collapse. The idea's 135-op v4 headline is not the implemented count: both accumulator components, the exact cadence and the distinct binding pass must be accounted for. These are structural counts and explicit instruction-cost assumptions, with no measured H100 throughput.

The v3 row-shape ratio applied to the reconciled post-split E0 planning value gives `leaf_ps = 41.8 * T_v3/T_split ≈ 38.46`. [results/cost-model-v3-scenario.txt](results/cost-model-v3-scenario.txt) runs the existing counted tool: μ(16K) 102.4→101.8 (−0.7%), μ(1K) −0.1%, with the current servable c2-b32 count/P5/table terms, unpriced P4 and remaining streamed-pass limitations. This illustrative E0 scenario is not a measured GPU cost and does not validate the original −5.2% μ projection.

## Analysis

The primary criterion is refuted: all three fixed widths stay below 15% whole-forest saving on every judged shape, with all judged CVs passing. v3 is a modest CPU improvement, while v4's extra grid points and v5's larger grid/reduction cost erase the intended 20% gain. Historical EXP-0008 absolute timings are not the baseline; current same-harness split timings decide.

The preregistered exact i128 bound passes at B(1,3,4)=32 and B(5)=8. At v4 the bound is 68133286889199490063175488963998436320; at v5 it is 153299895500698852642144850168996481720, both below 2^127. Signed products fit i64, and signed component sums fit i128. The grid is computed from fixed tables/public eq point (plus the already squeezed gamma in the fraction case); collapse only uses challenges after the corresponding message is absorbed. No messages, checks, masks, witness relations or verifier code change. Independent acceptance establishes algebraic compatibility, not a new soundness theorem.

Threats: one unpinned CPU thread and synthetic distributions; cache/resident-memory behavior; counts versus actual machine instructions; three noisy microkernels and the largest fraction v5 arm; and no GPU, GEMM regeneration, production SecretBuf clearing, masked end-to-end proof or streaming-layout measurement. The Vec fold port is for synthetic research only and cannot be transplanted into secret-handling production code. The v1 observation fixes a 32-block cadence; IDEA-0010's separate 2^16-term variant is not tested and is not globally refuted. Tests cover public-challenge degeneracies; binary constant-time behavior of this research kernel is not established.

## Verdict

`refuted` for EXP-0012 / IDEA-0022 under the frozen primary criterion. IDEA-0022 becomes `refuted` with priority 3.1→0 and confidence 0.6→0.9 in this CPU conclusion; E2 transfer remains unknown. IDEA-0010 and IDEA-0017 keep their independent remaining hypotheses and current status. The hosted fraction case is `inconclusive`, as judged below.

## Follow-ups

An H100 run must implement the same grid with int32 limb accumulation, retain byte-equivalence checks, and establish compute/HBM limits and tile layout before any E2 claim. No scheduler or production change is part of this experiment.

## Amendment 2026-10-19: hosted fraction-window case

This adds IDEA-0017's distinct in-memory fraction-leaf proxy before its code, without changing the primary integer-forest criterion. Its separate case compares a per-round split-eq fraction leaf kernel against the same kernel with a first window v in {2,3,4,5}, on 2^20, 2^21, 2^22, 2^23 and 2^24 synthetic leaves, with SplitMix64 seed 0x7E55_E7A0_0058_0003, key <2^32 and value <2^16. Denominators are alpha-key-beta*value and unit numerators; the three sumcheck instances are [eq,d_even,d_odd], [eq,d_even], [eq,d_odd]. A field-valued grid is permitted because denominators contain full-field challenges. Window messages and final evaluations must equal production prove_batch and the split kernel, and tv-core verify_batch must accept from independently computed hypercube sums. Current transcript messages, challenges and gamma weights remain unchanged. Existing zero/one/prior-eq-zero checks also cover this fraction window.

The separate IDEA-0017 case is `validated` if v=4 candidate time / split time <=1.5 on every size and all correctness checks and CV<=3% pass. It is `refuted` if the ratio exceeds v for every v>=2 on every size; otherwise `inconclusive`. Times per leaf per original round and maximum window widths are reported. Counting the input grid-read pass against v repeated original-input reads gives an E0 pass proxy of v, while the once-binding input read remains a separate pass. The harness keeps inputs resident and does not simulate GEMM regeneration, so this proxy establishes neither actual streamed-pass count nor regen latency. RFC-0020's same 3 warmups, 10 samples >=20ms, median/min/max/CV and one rerun apply. These observations cannot promote IDEA-0017's streamed claim to E2; its status/score remain conditional on planner and pass accounting.

### Results of the hosted fraction-window case

E1, ns per pair for the split reference and candidate/split ratios. Time per original leaf per sumcheck round is `ns/pair / (2*(log2(leaves)-1))`; for example at 2^24 leaves, split 0.838 ns/leaf/round and v4 2.127 ns/leaf/round. Inputs are resident and no regeneration work is simulated.

| Leaves | Split ns/pair | v2 ratio | v3 ratio | v4 ratio | v5 ratio |
|---|---|---|---|---|---|
| 2^20 | 38.125 | 1.539 | 1.880 | 2.519 | 3.552 |
| 2^21 | 38.169 | 1.539 | 1.864 | 2.525 | 3.565 |
| 2^22 | 38.152 | 1.543 | 1.876 | 2.544 | 3.553 |
| 2^23 | 38.524 | 1.540 | 1.867 | 2.498 | 3.537 |
| 2^24 | 38.530 | 1.552 | 1.877 | 2.539 | 3.584 |

v4 is 2.50–2.54× split, exceeding the registered 1.5× success ceiling on every size. v2 is 1.54–1.55×, below its factor-2 kill comparison, so the kill criterion does not hold. The case is therefore `inconclusive`; additionally 2^24/v5 retains CV 5.42% after a rerun (first 14.27%). All other fraction verdict arms pass CV. Neither that noisy arm nor a hypothetical pass factor can establish validation.

The grid reads each original input once, then the once-bind reads it again. count.py reports both passes; at v4 the counterfactual v regenerated reads versus two candidate reads is factor 2, not factor 4 once binding is included. The ordinary in-memory split reference reads shrinking folded tables rather than regenerating full input each round; this counterfactual is not its measured bandwidth. A future streamed implementation must determine whether binding reuses stored leaves or incurs another regeneration, and measure the GEMM/pass latency. This experiment checks a prove_batch fraction proxy, not the full current prove_trees/LogUp protocol.

### Review and validation

No production or verifier files changed. The harness has an independent lockfile, forbids unsafe code and passes cargo fmt and clippy with warnings denied. The soundness review checked integer bounds/sign reduction, challenge ordering, padding scale, final claims and degenerate recovery; the current independent verifier remains the acceptance oracle. [results/mutations.json](results/mutations.json) records the passing no-op control and 3/3 caught faults: sign correction, ternary extrapolation and quadratic collapse. Each mutant compiled and failed a correctness assertion. The runner initially rejected Rust's custom assertion text; its diagnostic was corrected, the existing control/sign/grid logs audited and the last mutant resumed, with the original runner output retained. This is a narrow arithmetic mutation campaign, not a full protocol audit.

Additional validation, from the repository root:

```sh
cargo fmt --manifest-path research/experiments/EXP-0012-integer-leaf-round-window/Cargo.toml --check
cargo clippy --manifest-path research/experiments/EXP-0012-integer-leaf-round-window/Cargo.toml --locked --all-targets -- -D warnings
python3 -I research/experiments/EXP-0012-integer-leaf-round-window/mutate.py
python3 -I research/tools/cost_model.py --profile counted --set leaf_ps=38.46
```

[results/preregistration-audit.json](results/preregistration-audit.json) verifies the primary frozen text against commit `afc1b14`, the unchanged hypothesis/criterion/method/reproduction, the preserved fraction amendment (commit `2d61eed`), and the recorded digests.


## Production stage-grid measurement, 2026-10-28 (#459)

This records RFC-0055's field-valued reduction-stage grid separately from the frozen integer
leaf experiment above. The source is the production `PreparedWindow` engine; neither runtime
workspace imports research code. The Criterion harness lives under
[`crates/tessera-sumcheck/benches/windowed`](../../../crates/tessera-sumcheck/benches/windowed/Cargo.toml).
Run it with:

```sh
cargo bench --manifest-path crates/tessera-sumcheck/benches/windowed/Cargo.toml --bench windowed -- --save-baseline before
cargo bench --manifest-path crates/tessera-sumcheck/benches/windowed/Cargo.toml --bench windowed -- --baseline before
```

One scalar CPU thread on Apple M1 Max, 64 GiB, macOS ARM64; two 65,536-entry synthetic `Fp2`
inputs, relation `eq · X · Y`, degree 3. Each timed iteration includes preparation, every round,
P3b contraction, final evaluations and clearing. Regeneration is a no-op over resident synthetic
inputs: this measures arithmetic and allocations, excluding witness/GEMM regeneration. Ten
samples, one second warmup, two second requested measurement per arm; Criterion increases the
collection time for longer arms. The initial grid used per-entry source calls and recomputed
public EQ products; the optimized grid uses chunked prefix reads and split public EQ weights.

| Own variables in first window | Initial grid (ms) | Optimized grid (ms) | Resident rounds (ms) |
|---|---:|---:|---:|
| 3 | 20.916 | 13.428 | 4.074 |
| 4 | 21.784 | 16.857 | 4.074 |
| 5 | 26.853 | 23.203 | 4.074 |
| 6 | 36.094 | 32.940 | 4.074 |
| 7 | 50.488 | 48.434 | 4.074 |
| 8 | 75.275 | 70.862 | 4.074 |

These are Criterion mean estimates. The initial resident measurement was 4.011 ms. Raw mean,
median and confidence estimates are in
[`results/windowed-cpu-2026-10-28.json`](results/windowed-cpu-2026-10-28.json).
The grid costs 3.3 to 17.4 times resident rounds in this full-extension fixture; it buys bounded
working storage, not a throughput improvement. The counting-allocator regression at 65,536
entries requires the windowed engine peak to be less than half the literal engine peak and
checks proof/transcript identity. A separate mixed-batch test checks the public allocation
model against every cut, including grid construction scratch and consecutive P3b passes.

A resident `Fp2` bind reads 32 bytes and writes 16 per pair, 24 bytes per original entry. The
M1 Max's [advertised 400 GB/s](https://support.apple.com/en-ie/111901) full-chip shared bandwidth gives a 0.06 ns/entry bandwidth floor,
not a one-core expectation. The grid instead performs `(3/2)^v` products and EQ scalings per
input entry, plus multilinear extension and contraction: its growth is dominated by arithmetic.
The arithmetic benchmark supplies no integer-grid, masked-protocol, GPU or GEMM timing result.


The separate [whole CPU budget fixture](results/windowed-cpu-budget-2026-10-28.json) uses the
bounded `tiny-2-servable` conformance model at three tokens. Its resident plan is 228,580,528
bytes. At a budget of 228,578,876 bytes, stage 1 takes a two-round window and one P3b pass;
its planned stage working set is 208,656 bytes. The measured whole-prover peak is 121,874,712
bytes. Both NI and a fresh Live session return independent-verifier `Accept`, and the seeded
NI proof bytes equal the resident proof's. An isolated child process checks that the manifest's
P3b counter equals the pass plan. `Engine::regenerate` reruns every graph node at P3 and P3b.
This checks the conservative CPU budget policy and regeneration path; leaf tables and the
cached witness dominate this small fixture, so it is not a whole-proof peak-memory speedup or
an out-of-core model qualification. The CPU planner retains its 64-byte-per-cell resident floor.
