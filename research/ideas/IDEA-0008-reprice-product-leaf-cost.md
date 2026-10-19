# IDEA-0008: Reprice the product-tree leaf: `c_π` by operation count and CPU calibration

| | |
|---|---|
| Status | `inconclusive` (2026-10-16, [EXP-0006](../experiments/EXP-0006-product-leaf-cost/README.md): the pre-registered CV clause fired on one kernel; every substantive condition of `validated` held) |
| Origin | [SUR-0004](../surveys/SUR-0004-attention.md) Q3 ("the 12 ps itself is an assumption until RFC-0020"); the method of [SUR-0002](../surveys/SUR-0002-lookups.md) `[E0-this]` applied to the product tree; [08 §2](../../docs/spec/08-performance-budget.md#2-unit-cost-targets) trigger row (`c_π` > 24 ps); [RFC-0020](../../docs/rfcs/RFC-0020-cost-model.md) `c_π` protocol row (product-tree GKR batched over rows of 2^14 leaves, 2^30 leaves) |
| Lever | `c_π` (its value and its uncertainty), and through it the sensitivity table of [baseline.md §2](../baseline.md#2-the-cost-model) that ranks every certificate-lens idea. No protocol term moves. |
| Spec surfaces | None normative. `research/tools/cost_model.py` (the corrected profile's `leaf_ps`), baseline.md §2 and §3 (the `c_π` rows), and after RFC-0020 v0.1 the restated projections of 00 and 08. Not tier A: no superseding RFC. |
| Priority | 13 on the decision-value reading (projection moved +22.2% × 0.6 / 1); 0 on the rubric's literal Δμ reading (see Score) |

## Mechanism

Nothing in the protocol changes. The idea replaces the assumed `c_π = 12 ps` (a figure the source states "to be measured", carried unchanged into the corrected profile) by a number with provenance, in three steps, each at the evidence level it can reach:

1. **E0, operation count.** Count Goldilocks (`Fp`) multiplications per leaf of a `SoftmaxRow` forest ([RFC-0009 'SoftmaxRow'](../../docs/rfcs/RFC-0009-gadgets.md#softmaxrow), `Π_{i ∈ V_q} k_qi = 0`, leaves over `Fp`) under the level protocol of [RFC-0010 'Tree levels'](../../docs/rfcs/RFC-0010-proof-schedule.md#tree-levels) and the kernel rule of [`crates/tessera-sumcheck/src/kernel.rs`](../../crates/tessera-sumcheck/src/kernel.rs): a degree-3 sumcheck of `eq(r_i, x)·Ṽ(0 ‖ x)·Ṽ(1 ‖ x)` per level, 4 nodes per pair, `k − 1` multiplications per node across `k` tables with the values along the line formed by addition, one `eq` table per instance per level built by doubling, every table folded after every round (`bind_base` on the first binding of an `Fp` table, two base multiplications per pair; `Fp2` folds after), levels halving geometrically so the whole tree costs twice its leaf level. The count is a small script that states each rule as a line (today in the session scratchpad; the experiment moves it under `research/experiments/`).
2. **E1, CPU calibration in one harness.** `cargo bench -p tessera-sumcheck --bench throughput -- --bench` already measures `prove_forests` in nanoseconds per leaf on one core, next to the `eq` tables alone and the engine's level-shaped degree-3 sumcheck ([`benches/throughput.rs`](../../crates/tessera-sumcheck/benches/throughput.rs) `forests`). Add a fraction-tree-shaped instance (numerators and denominators: two `Fp2` and two `Fp` tables, RFC-0008 'Fraction tree') to the same harness, so that `c_π / c_ℓ` is measured on the same machine, same inputs, same code (rule 4), and fit nanoseconds per leaf to the E0 count to get a nanoseconds-per-`Fp`-multiplication constant that both counts share.
3. **Translation to H100 (E0 on E1 inputs)**, exactly as SUR-0002 `[E0-this]` does for `c_ℓ`: 18 int32 instructions per Goldilocks multiplication, `1.67×10^13` int32 op/s (132 SMs × 64 lanes × 1.98 GHz) at peak, 60% utilization as the realistic figure ([VYTA25], secondary). The result goes into the corrected profile's `leaf_ps`; the source profile keeps 12 ps so that `cost_model.py --check` (the golden check against source §5 and §8) still passes.

The product tree and the fraction tree are both degree-3 GKR trees over the same `N = A·S` leaves per token; the product tree pays its own `eq` tables and folds. Nothing in the schedule shares them today (OQ-0010-1 says the two kinds run in the same levels but does not say they share tables), so the count is of a standalone product instance; the kill criterion below covers the case where a batched kernel makes the marginal cost smaller.

## Why it could work

- **E0, primary (own count, script above).** Per leaf of the whole tree, with Karatsuba `Fp2` multiplication (3 `Fp` multiplications): `eq` build 1.5, round-1 products 3.0 (`V0·V1` in `Fp`, then `eq·(V0V1)` as `Fp2 × Fp`), round-1 folds 1.75, `Fp2` rounds 6.0, `Fp2` folds 2.25 = 14.5 per leaf at the leaf level, **29 per leaf for the tree**; with schoolbook `Fp2` (4 `Fp` multiplications) **36**; the generation pass's upper count, which charges every round at `Fp2 × Fp2` cost, **46**. Against 12 ps assumed: 29 → 31 ps at the H100 int32 peak and 52 ps at 60% utilization; 36 → 39 / 65 ps; 46 → 50 / 83 ps. Every figure is above the 24 ps trigger of 08 §2, and the 60%-utilization figures are above the `ρ_π = 2` threshold of RFC-0020's decision rule (`ρ_π ∈ (2, 4]`: "proceed with re-plan"; 83 ps is `ρ_π = 6.9`: "stop and redesign" for the certificate).
- **E1, primary (Apple M4 Max, one core, cargo 1.95.0, release build of `benches/throughput.rs`, 2026-10-16; raw output in the session scratchpad, to be committed with the experiment).** `prove_forests`: 2^10 trees of 2^12 leaves 31.9 ns per leaf; one tree of 2^22 leaves 31.8; the mixed batch of 2^9×2^12, 2^11×2^9 and 1×2^20 leaves 32.7. The `eq` tables alone 3.25 ns per leaf (10%). The engine's level-shaped sumcheck (`Fp2` eq × 2 `Fp` tables, degree 3) over as many entries as leaves 27.9 ns per entry; three `Fp` tables 24.6. Binding every round, `Fp` into `Fp2`, 3.05 ns per entry, which is 2.5 `Fp`-multiplication equivalents per entry by the same rules, so about 1.2 ns per `Fp`-multiplication equivalent including memory traffic on this core; 31.9 ns per leaf is then about 26 equivalents per leaf, consistent with the Karatsuba count of 29 (E0 on E1).
- **E0 on E1, primary.** A crude throughput scaling of the E1 figure, one M4 core at about `1.8×10^10` int-op/s against the H100's `1.67×10^13` (ratio about 930, assumed, no measurement of either rate here), gives 34 ps at perfect scaling. Consistent with the count.
- **E0, primary.** SUR-0002 `[E0-this]` counts 56–74 `Fp` multiplications per lookup. By count, `c_π / c_ℓ` is 0.4–0.8 (29/74 to 46/56), not the 0.3 (12/40) the budget assumes: the two trees have the same shape and the same leaf count, and the product tree pays its own `eq` tables and folds.
- **Claim, direction only.** Expander's 111–122 ps per GKR gate on an H100 [Polyhedra24] (sponsored report, as SUR-0002 records it). Not Tessera's kernel; it says only that a GKR layer step on an H100 is tens of picoseconds, not 12.
- **Secondary.** [VYTA25] measures field-multiplication kernels at 60% of the int32 pipeline's theoretical peak on NVIDIA GPUs, with "per-SM INT32 performance roughly constant across generations" including H100; the utilization figure used in the translation.

Bibliography keys `[VYTA25]` and `[Polyhedra24]` are those of SUR-0002's source table; the consolidation agent resolves them in [bibliography.md](../bibliography.md).

## Expected gain

No direct gain: the idea reprices the lens. Central scenario (the generation pass's count at the int32 peak, and the E1-consistent counts at 60% utilization both land near it):

```sh
python3 -I research/tools/cost_model.py --set leaf_ps=50
```

```text
baseline: profile 'corrected'
   S   linear  attention  weights  regen  streaming    total
   1K     3.30       1.13    0.021    4.0       0.42      8.9
   4K     3.25       4.46    0.021    4.0       2.32     14.1
  16K     3.06      16.82    0.020    4.0      11.24     35.1
 128K     1.99      87.48    0.013    4.0      77.93    171.4

scenario: leaf_ps=50
   S   linear  attention  weights  regen  streaming    total
   1K     3.30       1.66    0.021    4.0       0.42      9.4   Δ   +5.9%
   4K     3.25       6.53    0.021    4.0       2.32     16.1   Δ  +14.7%
  16K     3.06      24.62    0.020    4.0      11.24     42.9   Δ  +22.2%
 128K     1.99     128.02    0.013    4.0      77.93    212.0   Δ  +23.7%
```

| Context | Δμ | Δ proof size | Δ verification |
|---|---|---|---|
| 1K | +5.9% (E0 on E1 inputs; 8.9 → 9.4) | 0 (E0: `c_π` is prover time only) | 0 (E0) |
| 4K | +14.7% (14.1 → 16.1) | 0 | 0 |
| 16K | +22.2% (35.1 → 42.9) | 0 | 0 |
| 128K | +23.7% (171.4 → 212.0) | 0 | 0 |

Bounds of the same scenario (`--set leaf_ps=32`, the Karatsuba count at peak and the crude M4 scaling; `--set leaf_ps=83`, the upper count at 60% utilization), E0 on E1 inputs:

| Scenario | Δμ 1K | Δμ 4K | Δμ 16K | Δμ 128K |
|---|---|---|---|---|
| `leaf_ps=32` | +3.1% (9.2) | +7.7% (15.1) | +11.7% (39.3) | +12.4% (192.7) |
| `leaf_ps=50` | +5.9% (9.4) | +14.7% (16.1) | +22.2% (42.9) | +23.7% (212.0) |
| `leaf_ps=83` | +11.1% (9.9) | +27.5% (17.9) | +41.4% (49.7) | +44.2% (247.2) |
| `leaf_ps=15` (the kill threshold) | +0.5% (8.9) | +1.2% (14.2) | +1.8% (35.8) | +1.9% (174.6) |

What the repricing does to the queue (`--sensitivity --set leaf_ps=50`, E0): the `c_π` share of `μ(16K)` goes from 7.0% to 23.9% (1K: 1.9% → 7.3%; 128K: 7.5% → 25.2%), and the gain from halving `c_π` goes from −3.5% to −11.9% at 16K (−0.9% → −3.7% at 1K, −3.7% → −12.6% at 128K). That promotes the certificate-lens ideas the ledger holds as second-order today: base-field or four-ary tree levels, forest fusion at a shared point, and certified block skipping, which removes leaves outright. At 83 ps the 16K projection (49.7) exceeds the SC-2 ceiling of 45 ([baseline.md §2](../baseline.md#2-the-cost-model)) on `c_π` alone, before `c_ℓ`'s own interval of `[40, 130]` ps (SUR-0002 Q1) is applied.

Evidence levels of the inputs: the multiplication counts E0 (primary, own count); the M4 nanoseconds E1 (primary); the H100 instruction count, peak rate and utilization E0 with a secondary source ([VYTA25]); every `μ` figure E0 (the cost model, a projection under 08 §1 and rule 7). The H100 absolute stays E0 until RFC-0020 v0.1 measures it (E2).

## Security gate

| Property | Effect | Pass |
|---|---|---|
| Soundness bits (G-4, RFC-0014) | None: no relation, challenge, label or parameter changes; the idea changes a planning constant and a benchmark. | pass |
| Uniqueness upstream of the logits (SP-3, RFC-0009 Lemma 0) | None: `SoftmaxRow` Lemma 5 and its relations are untouched. | pass |
| Zero knowledge (G-5, RFC-0013) | None: no mask, blinding row or opening changes. The bench uses seeded synthetic leaves (rule 6), no model data. | pass |
| Verifier independence (SP-6) | None: nothing under `verifier/` changes; the bench lives in a prover crate. | pass |
| Post-quantum (hash-based commitments) | None: no commitment changes. | pass |

## Prior rejection check

Checked against [baseline.md §7](../baseline.md#7-already-rejected) and the Alternatives Considered of RFC-0020, RFC-0009 and RFC-0010:

- RFC-0020 Alternatives Considered rejects "end-to-end measurement only", "analytical roofline only" and "measuring on B200 first". This idea is none of them: it is an operation count plus a CPU calibration before the v0.1 microkernel, which RFC-0020's own motivation calls for ("discovering then that a unit cost is 5× off would waste the GPU work in between"). RFC-0020 schedules the H100 measurement of `c_π`; nothing schedules a CPU-side calibration of it, which is exactly what SUR-0002 'Candidate ideas' seeded for `c_ℓ` ("Calibrate the op count on a CPU (E1)").
- baseline.md §7 row on RFC-0009's rejected certificates ("30 ps per one-hot entry against 12 ps per leaf") and RFC-0009 Alternatives Considered 3 rest on the 12 ps figure. If `c_π` is 30–80 ps, the one-hot comparison tightens from 2.5× to 0.4–1×; that row's "Reopens if" condition ("a certificate cheaper than one product leaf per score") becomes reachable at a higher bar. Not a rejection of this idea; a consequence of it.
- SUR-0004 'Candidate ideas' lists "base-field leaf levels of the `SoftmaxRow` product tree" with the killer "the 12 ps assumption may already be a base-field figure". The E0 count here already credits the base-field first round (3 of 14.5 multiplications per leaf at the leaf level) and still lands at 29–46 per leaf, so that killer does not fire against this idea.

New now: the E1 bench exists (`benches/throughput.rs` `forests`, measured 2026-10-16), and SUR-0002's counting method, applied to `c_ℓ`, has a result (`c_ℓ ∈ [40, 130]` ps) that this idea completes for the third unit cost. None: no prior rejection covers a CPU-first calibration of `c_π`.

## Kill criterion

RFC-0020 v0.1 measures `c_π ≤ 15 ps` per leaf on the H100 kernel (`ρ_π ≤ 1.25`); or the GPU leaf stage shares the `eq` tables and the folds between the product and fraction instances of one level so that the marginal cost of a product leaf, measured in the batch, is at most one third of the standalone count (then the standalone count overstates `c_π` and the marginal figure is the right model input). Either observation refutes the central scenario; the first refutes the idea outright.

## Cheapest experiment

Hypothesis (pre-registered here): the standalone product-tree leaf costs between 29 and 46 `Fp` multiplications (E0), the M4 bench fits that count within ±30% at a shared nanoseconds-per-multiplication constant (E1), and `c_π / c_ℓ` measured in one harness lies in `[0.4, 0.8]` (E1), so that `c_π ≥ 30 ps` on the H100 under SUR-0002's translation (E0).

Method: (1) commit the count script as `research/experiments/EXP-NNNN-product-leaf-cost/count.py` with each RFC-0010 and kernel rule as a line, SPDX header, `-I` clean; (2) add a fraction-tree-shaped instance (2 `Fp2` + 2 `Fp` tables, degree 3) to `benches/throughput.rs`'s `forests` cases, or an equivalent research-side harness that depends on `tessera-sumcheck` by path (rule 2), and run both on the same core with the same seeded leaves; (3) fit `ns per leaf = κ · (Fp multiplications per leaf)` over the forest, `eq`-only and engine cases; report `κ`, the residual, and the `c_π / c_ℓ` ratio; (4) translate as SUR-0002 does and open a cost-model change for `leaf_ps` in the corrected profile with the `--check` golden test still green.

Evidence reachable here: E0 and E1 (half of the E1 is already measured: the forest, `eq` and engine figures above). Effort `s` (up to 2 days; CONTRIBUTING's labels). Unknown until an H100 run: the absolute `c_π` in picoseconds (utilization and instruction count of the actual kernel), and whether the v0.1 kernel shares `eq` tables between instance kinds.

## Refutations

Sweep triage of 2026-10-16: four refuter agents who did not write the idea, one per lens; each argument is condensed from that refuter's verdict.

| Lens | Verdict | Argument |
|---|---|---|
| soundness | survives | No relation, challenge, label, parameter, mask or verifier check moves: the idea replaces a planning constant (c_π = 12 ps) by a counted and CPU-calibrated interval and adds a fraction-tree-shaped case to a prover-side bench. The bench runs on seeded synthetic leaves (research rule 6) inside `crates/tessera-sumcheck`, so no secret-class value enters a report and nothing under `verifier/` is touched (SP-6). There is no mechanism for ADV-P or ADV-V to exploit; the only soundness-relevant consequence is downstream, in that a repriced c_π reopens the RFC-0009 certificate row of baseline.md §7, and any idea it promotes gets its own review. |
| prior art | survives | The c_π = 12 ps figure is assumed (08 §2, trigger '> 24 ps'; RFC-0020) and RFC-0020 schedules only the H100 microbenchmark ('Product-tree GKR batched over rows of 2^14 leaves'); Alternatives Considered 1–3 reject end-to-end-only, roofline-only and B200-first, none of which a counted-and-CPU-calibrated figure is. No row of baseline.md §7 covers it; the RFC-0009 row ('30 ps per one-hot entry against 12 ps per leaf') depends on the 12 ps and is a downstream consequence, not a rejection. Prior art is the corpus's own method (SUR-0002 [E0-this] for c_ℓ) and SUR-0004 Q3's remark that 12 ps is unmeasured; new is the E1 `prove_forests` bench (31.9 ns per leaf, one M4 core) and the count of 29–46 Fp multiplications per leaf. Like IDEA-0004 it moves no protocol term, so it competes as a repricing that reorders the certificate-lens queue. |
| cost model | survives | Nothing is saved; the lens question is whether the repricing changes which ideas matter, and it does: `leaf_ps=50` reruns to μ(16K) 42.9 (+22.2%) and `--sensitivity` at that price moves c_π from 7.0% to 23.9% of μ(16K) and the gain from halving it from −3.5% to −11.9% (rerun confirmed), which promotes every certificate-lens idea the baseline ranks second-order. The inputs are the most honest in this batch: an E0 count of 29–46 Fp multiplications per leaf, an E1 M4 bench at 31.9 ns per leaf that fits the Karatsuba count (about 26 equivalents) within the stated ±30%, and the same translation SUR-0002 used for c_ℓ, so c_π / c_ℓ by count is 0.4–0.8 against the assumed 0.3, consistent with two degree-3 trees over the same leaves. Every projected value crosses the 24 ps trigger of 08 §2 and the 60%-utilization figures cross RFC-0020's ρ_π > 2 branch; at 83 ps the 16K projection (49.7) breaches SC-2 on c_π alone. Kill criterion (shared eq tables in a batched kernel) is the right one and is where IDEA-0010 lives. |
| feasibility | survives | The harness exists and half the E1 is measured: `crates/tessera-sumcheck/benches/throughput.rs` has the `forests`, 'eq tables alone' and engine cases, and `prove_batch` already mixes Fp2 and Fp tables through `product_sums` (`crates/tessera-sumcheck/src/kernel.rs`). No spec surface, no RFC, effort `s`, E0 and E1 reachable as claimed. One caveat: a 'fraction-tree-shaped instance' of 2 Fp2 + 2 Fp tables is a single degree-4 product, not the fraction tree's sum of two products (RFC-0008 'Fraction tree'), and no LogUp prover exists (`crates/tessera-logup` stub), so the c_π / c_ℓ ratio measured in one harness is a shape proxy until the logup crate lands; the c_π count and fit themselves are unaffected. |

## Score

| Gain | Confidence | Cost | Priority |
|---|---|---|---|
| Δμ of its own: 0. Scoreboard projection moved (EXP-0006, E0 count fitted at E1): `c_π` ∈ [45.4, 116.9] ps (`ρ_π` 3.8–9.7) against 12 assumed, central 73 ps; μ(16K) 35.1 → 42.0 at the floor (+19.5%), 47.7 at the central value (+35.6%), 56.7 at the ceiling (+61.2%); μ(1K) 8.9 → 9.3 / 9.7 / 10.3 (+5.2% / +9.5% / +16.3%); `c_π`'s share of μ(16K) 7% → 31% at 73 ps, the largest unit cost at 16K and 128K while `c_ℓ` stays at 40 ps; SC-2's μ(16K) ≤ 45 holds only below 60 ps | 0.7: the per-operation count (25.5 Fp multiplications, 6 `mul_small(7)`, 63 additions or subtractions per leaf) predicts the production prover within 2% on one M4 Max core (predicted/measured 0.98–1.00 on three forest shapes, 0.69–1.10 on every component), so the arithmetic is right; the GPU figure is the int32 translation and the 60% utilization, extrapolated as in EXP-0002, and the whole width of the interval | `s`, spent: E0 and E1 done in EXP-0006; the H100 absolute stays E2 until RFC-0020 v0.1 | Rubric Δμ reading: 0. Decision-value reading realized: 35.6 × 0.7 / 1 = 25 as a repricing of the lens; 0 as a further E1 experiment (nothing E1 can settle remains but the CV clause, which a rerun with a repetition count for millisecond kernels would clear) |

## Decision and links

Triage 2026-10-16: `triaged`; second-ranked experiment of the batch on the decision-value reading, after IDEA-0004 (its c_c twin). Experiment 2026-10-16: [EXP-0006](../experiments/EXP-0006-product-leaf-cost/README.md) `done: inconclusive` by the letter of its pre-registered criterion (the level-build kernel, 3.6 ms per iteration with 64 MiB of fresh allocations, kept a coefficient of variation of 3.95% after its rerun), while every substantive condition of `validated` held: the fit passed with predicted/measured 0.69–1.10 (forests 0.98–1.00; the idea's single-constant reading holds within 2.3%, not just ±30%), and the counted H100 floor 45.4 ps (`ρ_π` 3.78) excludes RFC-0020's "proceed" band and fires the 08 §2 trigger at every point of `[45.4, 116.9]` ps (E0 fitted to E1). The count: 25.5 Fp multiplications at three nodes per round (this note's 29 is the four-node count without the level build) plus 63 additions or subtractions, which are a third of the int32 work, so the multiplication-only reading ([27, 46] ps) understates; this applies to SUR-0002's `c_ℓ` count too. The `c_π / c_ℓ` shape proxy measured in one harness is 0.295, equal to the assumed 12 / 40 and below the expected 0.4–0.8: the two trees are mispriced by the same factor, not relative to each other. Next step: reprice `leaf_ps` in the corrected profile to 73 ps with the interval `[45, 117]` (baseline.md §2, the source profile unchanged so `--check` stays green) and rerun the sensitivity table (`c_π` share 7% → 31% at 16K, halving it −15.7%); re-score IDEA-0010 (mechanism 1, one `eq` table per batch, quantified at about 7.5% of the proxy's lookup cost, E1) and IDEA-0009 (the `Fp2` rounds after the first and their folds are 60% of the leaf time, E1); count `c_ℓ`'s additions the same way. Experiments: [EXP-0006](../experiments/EXP-0006-product-leaf-cost/README.md). RFC or issue: none; the cost-model and baseline.md change, and an input to RFC-0020's v0.1 measurement (pre-registered prediction `[45, 117]` ps; record the int32-pipe utilization and the kernel's fusion depth). Ledger line updated on: pending consolidation.
