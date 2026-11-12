# SUR-0018: The redesign queue against the MVP's measured profile

| | |
|---|---|
| Status | `current (2026-11-12)` |
| Agenda topic | `lookups`, `commitments` and `sumcheck`, specialized to [#745](https://github.com/AbdelStark/tessera/issues/745) (redesign the commitment, LogUp and tree levels against measured costs) |
| Questions | (1) Where does a real proof's time go, measured end to end (E3), and how far is each component's end-to-end cost from its microkernel's (E2)? (2) Ranked for each of the commitment, LogUp and the tree levels: which ideas in the ledger move the measured cost, with what evidence, and which RFC carries each? |
| Period covered | Tessera's own runs from 2026-11-10 to 2026-11-11 ([docs/perf/mvp/benchmark.md](../../docs/perf/mvp/benchmark.md)), the v0.1 microkernels (`docs/perf/v0.1/`) and the ledger's ideas up to IDEA-0044. No external sources: SUR-0002, SUR-0005 and SUR-0007 cover the literature and stay current. |
| Produced by | `/autoresearch survey` for #745, 2026-11-12 |

## Method

The baseline is the MVP's three accepted 8B proofs (Llama-3.1-8B-Instruct at the observatory's pinned revision, `tessera-protocol/1`, `ni-100`, `tessera prove --backend gpu`), read from `benchmark.md`, and the commitment report's counts (`docs/perf/mvp/commitment-report-8b.json`: 3,959,805 lookups per token, 10.1 G table rows). The 1,024 + 64 run's GPU utilization (1.2% mean) is the harness record's (`bench/inference-proofs/tessera/runs/2026-11-11-l31-8b-c1024-hosted-a100/run.json`). Every idea in the ledger's Ideas table was assigned to one of the three components or to none, and ranked by the share of the measured profile its term touches times its evidence. Derived numbers are this survey's arithmetic on those records ([E0-SUR18]), not new measurements.

## Findings

### The measured profile (E3)

| Phase | 8B, 8 + 8, H200 NVL (405d552) | 8B, 1,024 + 64, A100 80 GB (6d16160) | Component |
|---|---|---|---|
| P1 advice commitment | 177 s | 1,269 s (5.8%) | commitment |
| P2 leaf stage: lookup and table fraction trees, product forests | 1,311 s | **14,819 s (67.6%)** | LogUp, trees |
| P3 reduction stages | 512 s | 2,052 s (9.4%) | sumcheck glue |
| P4 openings | 558 s | 1,576 s (7.2%) | commitment |
| P5 column answers | 210 s | 1,241 s (5.7%) | commitment |
| Prove, total | 2,902 s | 21,907 s | |

- **The commitment** (P1, P4, P5) is 4,086 s, 18.7% of the full workload.
- **The leaf stage** (P2) is two thirds of it, and grows with the request: 1,311 s at 16 positions, 14,819 s at 1,088.

### End to end against the microkernels

- **Lookups.** At 1,088 positions, the linear layers make 4.31 G lookups (3.96 M per token), and attention makes about 14.2 G of the 18.5 G. P2 grows by 13,508 s from 16 to 1,088 positions, for about 18.4 G more lookups: **about 734 ns per lookup end to end** (E3, A100 host). The GPU fraction-tree microkernel measures 297 ps per lookup on an H100 (E2, [l-h100](../../docs/perf/v0.1/l-h100.md)). Even allowing the A100 half an H100's throughput, the end-to-end lookup is three orders of magnitude above its kernel.
- **Table side.** P2 at 16 positions, 1,311 s, is almost all table side: 10.1 G table rows against 63 M lookups. That is **at most 130 ns per table row** (E3, H200 host), against the same 297 ps kernel.
- **The GPU is idle.** Over the 1,024 + 64 proof, the device averaged 1.2% utilization. The leaf stage's time is the host's:
  - window grid preparation;
  - table fills and clears;
  - tree construction on one thread (`FractionTree::new`, before 9d1a5dd);
  - level uploads from pageable memory.
- **The commitment** is the nearest to its kernel. P1 encodes and hashes on the device, but its rows come from one host thread (`visit_rows`). P4 and P5 regenerate rows on one thread and copy them to the device from pageable memory.

## Synthesis

RFC-0020's v0.1 measurement found the unit costs 4.5–7.4 times its assumptions (`docs/perf/v0.1/decision.md`) and called for a protocol redesign. The MVP's end-to-end profile changes the order of work: the measured cost of a lookup in a real proof is about 2,500 times its microkernel's, so no protocol change to `c_ℓ`, `c_π` or `c_c` can show in the end-to-end numbers until the leaf stage's host work is gone. The ranking below puts implementation first in each component, then the protocol ideas by evidence. The protocol ideas still decide the floor once the implementation reaches the kernels, and the table side's fixed 10.1 G rows set every phase's size.

### Commitment (18.7% of the workload)

| Rank | Idea | Term | Evidence | Status | Carrier |
|---|---|---|---|---|---|
| 1 | Pinned, overlapped row pipeline for P1, P4 and P5: regeneration on every core, page-locked uploads, double-buffered against the kernels | `regen`, `stream` | E3 (profile: one host thread feeds the device; P4 + P5 are 2,817 s) | engineering | none needed (prover-only) |
| 2 | [IDEA-0016](../ideas/IDEA-0016-pruned-column-answer-pass.md) pruned column-answer pass | `regen` (P5) | E1, 3.32× on the pass (EXP-0007) | `promoted` | RFC-0018 'Pass structure' |
| 3 | [IDEA-0012](../ideas/IDEA-0012-johnson-regime-tau-linear-error.md) Johnson-regime `τ` | `τ`, `size`, P4/P5 per opened column | E0 exact: `τ_A` 155 → 130, `τ_W` 97 → 68 | `promoted` | [RFC-0021](../../docs/rfcs/RFC-0021-johnson-regime-proximity.md) (Draft) |
| 4 | [IDEA-0002](../ideas/IDEA-0002-coefficient-form-rows.md) coefficient-form rows | `c_c` | E0 + E1 split, −4.2% `μ(16K)` | `triaged` | superseding RFC to RFC-0011/0012/0013 |
| 5 | [IDEA-0025](../ideas/IDEA-0025-rate-two-thirds-live-fast.md) rate 2/3 live encoding | `c_c` against `τ` | E0, −4.2% to −5.7% `μ(16K)` | `triaged` | new parameter set |

Ruled out: IDEA-0013 (`refuted`), IDEA-0015 (`refuted` E1), IDEA-0024 (`rejected`), IDEA-0003 (`shelved` pending an H100 kernel).

### LogUp (P2's lookup trees, most of 67.6%)

| Rank | Idea | Term | Evidence | Status | Carrier |
|---|---|---|---|---|---|
| 1 | The leaf stage at device speed: every host step of P2 on every core or on the device (tree construction, window preparation, table fills and clears), pinned level uploads | `regen`, `stream` | E3 (734 ns per lookup against 297 ps; GPU 1.2% busy) | engineering, started (9d1a5dd: fraction trees and resident loads on every thread) | none needed (prover-only) |
| 2 | [IDEA-0034](../ideas/IDEA-0034-public-window-attention-shapes.md) public windowed attention shapes | `A`: attention's 14.2 G of 18.5 G lookups | E0 + finite validation (EXP-0021) | `triaged` | new RFC; changes `D` |
| 3 | [IDEA-0026](../ideas/IDEA-0026-delayed-reduction-fraction-node-sums.md) delayed reduction in fraction node sums | `c_ℓ` | E1, 29–34% on the CPU kernel | `promoted`, implemented on the CPU | RFC-0010 (prover-only) |
| 4 | [IDEA-0018](../ideas/IDEA-0018-split-eq-fraction-level-kernel.md) split-`eq` fraction-level kernel | `c_ℓ` | E0, about 4% of `c_ℓ` | `validated` | RFC-0010 (prover-only) |
| 5 | [IDEA-0005](../ideas/IDEA-0005-base-field-leaf-small-value.md) base-field leaves with integer rounds | `c_ℓ` | E1, noisy (−36% registered minima, one noise failure) | `inconclusive` | RFC-0010 |
| 6 | [IDEA-0019](../ideas/IDEA-0019-analytic-padding-blocks.md) analytic padding blocks | padding on `K_lin`, `A` | E1, mechanism confirmed bit for bit | `inconclusive` | RFC-0010 |

Table side, the fixed 1,311 s: IDEA-0007 (`rejected` on zero knowledge), IDEA-0023 (`refuted`) and IDEA-0027 (`killed` by RFC-0057) leave no protocol candidate. Its open routes are #691's structured table evaluation and #729's cheap table fraction trees, both engineering against the 130 ns per row measured here.

### Tree levels (product forests in P2)

| Rank | Idea | Term | Evidence | Status | Carrier |
|---|---|---|---|---|---|
| 1 | [IDEA-0021](../ideas/IDEA-0021-split-eq-gruen-level-kernel.md) Gruen-corrected split-`eq` level kernel | `c_π` | E1, 1.75× per leaf, messages bit-identical | `promoted`, implemented on the CPU | RFC-0010 (prover-only) |
| 2 | [IDEA-0030](../ideas/IDEA-0030-deferred-reduction-level-kernel.md) deferred reduction in the split-`eq` kernels | `c_π` | E1, 25–27% on forests | `promoted`, implemented on the CPU | RFC-0010 (prover-only) |
| 3 | Port 1 and 2 to the GPU level kernels (`crates/tessera-gpu/src/level.rs`) | `c_π` on the device | E2 needed | engineering | none needed |
| 4 | [IDEA-0009](../ideas/IDEA-0009-four-ary-tree-levels.md) four-ary levels | `stream`, glue, `size` | E0, bimodal (−5.7% or −31.2%) on an unresolved reading of RFC-0018 | `triaged` | RFC-0010 amendment |
| 5 | [IDEA-0010](../ideas/IDEA-0010-shared-point-forest-fusion.md) shared-point forest fusion | `c_π` | E1 inconclusive (98.5–100.5%) | `triaged` | RFC-0010 |
| 6 | [IDEA-0029](../ideas/IDEA-0029-packed-indicator-max-certificate.md) packed-indicator max certificate | `c_π` against range lookups | E0 unit validated; gain not current | `triaged` | RFC-0009 |

Ruled out: IDEA-0022 (`refuted`), IDEA-0020 (`rejected`).

## Candidate ideas

- **Measure the leaf stage per step, not per pass.** #744's manifest gives P2's time and device traffic; a split of P2 into tree construction, window preparation, level rounds and table work would rank rows 1 of LogUp and trees by measurement. Kill: if the sampled profile shows one step above 80%, the split is not needed.
- **Table side on the device.** The table fraction trees' leaves are public functions of the row index (`FractionTree::new`'s denominators): generating them inside the level kernels, as the `c_ℓ` microkernel already does, removes the host fills. Term: the fixed 1,311 s. Kill: a table whose rows the kernel cannot compute from public data.
- **Attention lookups by windowed shapes** (IDEA-0034) are the only protocol lever on the 77% of lookups that attention makes at 1,088 positions; the workload fixes the shape, so a windowed `D` changes the model the benchmark proves.

## Staleness triggers

- The next measured 8B profile after the leaf-stage engineering (the 9d1a5dd comparison is pending; vast.ai credit ran out on 2026-11-12).
- Any end-to-end lookup cost within 10× of the kernel's: protocol ideas then lead the ranking.
- RFC-0021 accepted or withdrawn.

## References

[E0-SUR18] This survey's arithmetic on the records named in Method. No new bibliography keys.
