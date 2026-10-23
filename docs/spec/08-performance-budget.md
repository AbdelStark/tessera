# 08 — Performance Budget

- Status: Accepted for v1 planning
- Governs: cost model, unit-cost targets, end-to-end targets, memory and proof-size budgets, profiling plan, regression gates
- Detailed in: [RFC-0020](../rfcs/RFC-0020-cost-model.md)

Every number in this document is a projection until RFC-0020's measurements replace it. Projections are labeled as such; measured values will be labeled with their run manifest.

## 1. Cost model

The proof multiplier for a model at context `S` (source §5):

```math
\mu(S) \approx \frac{1.15\,(C_{lin} c_{c} + K_{lin} c_{\ell}) + 1.15\cdot A\,\big(S\,(c_{c}+c_{\ell}+c_{\pi}) + \min(S, b)\,c_{c}\big) + W/T}{(F_{lin} + F_{att} S)/R} \;+\; n_{regen}
```

| Symbol | Meaning | 405B value |
|---|---|---|
| `C_lin`, `K_lin` | Committed elements and lookups per token in linear layers | 1.7×10^7 each (source); with RISK-2's decompositions, 3.6×10^7 committed elements and 3.8×10^7 lookups for the `minimal` profile, and 4.5×10^7 each for the default `servable` one |
| `A` | Scores per token per unit context: heads × layers / 2 | 8,064 |
| `b` | Score block: a token commits `A·(S + min(S, b))` scores, the valid half and the diagonal blocks' masked halves ([RFC-0034](../rfcs/RFC-0034-causal-blocks.md)). The source counts the valid half alone. | 512 |
| `c_c`, `c_ℓ`, `c_π` | Unit costs: committed element, lookup, product-tree leaf | 30, 40, 12 ps (assumed) |
| 1.15 | Sumcheck glue factor | assumed |
| `W/T` | Weight pass per token: 0.24 GPU-s per 16K-token batch. The weight column answers add no GPU time: they are read from the epoch's stored codewords ([RFC-0012](../rfcs/RFC-0012-weight-commitment.md#column-answers)) | — |
| `F_lin`, `F_att` | Native INT8 operations per token: linear, attention per unit context | 8.1×10^11, 4.1×10^6 |
| `R` | Native rate: 60% of 1,979 TOPS dense INT8 on H100 SXM | 1.19×10^15 op/s |
| `M` | Table rows per proof: one committed multiplicity and one fraction-tree leaf each ([RFC-0008](../rfcs/RFC-0008-logup-and-tables.md)) | 6.7×10^9 for `minimal` and 7.7×10^9 for `servable`, with two classes at `B_max = 32` and [RFC-0028](../rfcs/RFC-0028-balanced-split.md)'s balanced split ([`docs/perf/v0.2/`](../perf/v0.2/README.md)) |
| `n_regen` | Witness regenerations, each about one native forward pass: the passes P1 to P5 of [RFC-0018](../rfcs/RFC-0018-gpu-prover-runtime.md#pass-structure), which are the witness pass, the leaf stage, stage regeneration, the opening pass and the column answers. A leaf stage in token groups adds its merge's pass P2b ([RFC-0054](../rfcs/RFC-0054-token-groups.md#pass-structure)). Absent from the source model (RISK-9). | 5, and 6 when the leaf stage is grouped, as every 405B batch of 16K tokens is (estimate) |

`tessera bench model` implements this formula and four terms it leaves out: the leaf stage's memory plan of RISK-10 and [RFC-0054](../rfcs/RFC-0054-token-groups.md#the-executable-model) (the merge's folds, at `c_m` per merged element, and the table trees' streamed rounds), the column answers' evaluation of every committed element at the queried columns, at `c_a` per element (RFC-0018 P5), the opening pass's combination of every committed element into the combined rows, at `c_r` per element (RFC-0018 P4), and the table side of LogUp, `1.15 · M · (c_c + c_ℓ) / T`. It reads the counts from the compiler's commitment report, so projections update automatically when the compiler changes ([RFC-0020](../rfcs/RFC-0020-cost-model.md#executable-cost-model)).

## 2. Unit-cost targets

| Unit | Target (H100 SXM, one GPU) | Re-plan threshold | What dominates |
|---|---|---|---|
| `c_c` committed 64-bit element | 30 ps | > 60 ps | Rate-1/2 NTT encode, BLAKE3 column hashing, HBM traffic |
| `c_ℓ` LogUp-GKR lookup | 40 ps | > 80 ps | `Fp2` fraction-tree arithmetic |
| `c_π` product-tree leaf | 12 ps | > 24 ps | `Fp2` multiplications |
| Glue | ≤ 15% of commit + lookup time | > 30% | Sumcheck rounds, `eq` tables |
| Weight pass | ≤ 0.24 GPU-s per 16K-token batch (405B) | > 0.5 GPU-s | Limb-decomposed INT8 GEMMs at the fold point |

The re-plan thresholds are the go/no-go rule of [RFC-0020](../rfcs/RFC-0020-cost-model.md#decision-rule).

## 3. End-to-end targets (Llama-3.1-405B, one 8×H100 node)

Projected from the source (§8) with source unit costs and source commitment counts:

| Context | `μ` (projection) | Proof time per request | Sampling rate for 10% amortized overhead (`η = 0.3`) | Daily detection, 100 violating of 10^6 requests |
|---|---|---|---|---|
| 1K | 3.2 | 0.3 s | 10.4% | ≈100% |
| 4K | 6.5 | 2.3 s | 5.1% | 99.5% |
| 16K | 18.8 | 28.5 s | 1.8% | 83.3% |
| 128K (stretch) | 88.7 | 28 min | 0.38% | 31.4% |

These figures exclude witness regeneration. Corrected projections, computed with the executable model:

- RISK-2's decompositions raise the linear-layer term at 1K from 2.0 to 4.4 at the `minimal` profile's count, and to 5.3 at the default `servable` profile's.
- Five regeneration passes (RISK-9, RFC-0018 P1 to P5) add 5. The leaf stage's merge (RISK-10, [RFC-0054](../rfcs/RFC-0054-token-groups.md)) adds a sixth: a 16K-token batch's leaf stage is grouped at every context.
- The table side of LogUp commits one multiplicity and builds one fraction-tree leaf per table row, once per proof ([RFC-0008](../rfcs/RFC-0008-logup-and-tables.md)). With two classes at `B_max = 32` and [RFC-0028](../rfcs/RFC-0028-balanced-split.md)'s balanced split, the tables hold 6.7×10^9 rows for `minimal` and 7.7×10^9 for `servable`. Shared by the 16K-token batch, they add 0.048 at 1K and 0.044 at 16K for `minimal`, and 0.055 and 0.051 for `servable`.
- The score tensors skip their blocks above the diagonal, and the masked halves of the diagonal blocks stay committed: `A·b` more elements per token, `b = 512` at 405B ([RFC-0034](../rfcs/RFC-0034-causal-blocks.md)). They add 0.21 at 1K and 0.19 at 16K. That gives `μ(1K) ≈ 11.8` and `μ(16K) ≈ 27.1` for `minimal`, and `μ(1K) ≈ 12.7` and `μ(16K) ≈ 28.0` for `servable`.
- The column answers (P5) also evaluate every committed element at the queried columns, at a cost `c_a` per element that is not yet measured. Each picosecond of `c_a` adds 0.070 at 1K and 0.23 at 16K for `minimal`, and 0.084 and 0.25 for `servable`. Re-encoding a row costs at most what committing it does, `c_c = 30 ps` per element (assumed), which adds 2.1 and 7.0, or 2.5 and 7.4.
- The opening pass (P4) also combines every committed element into the combined rows, at a cost `c_r` per element that is not measured on H100 either. Each picosecond of `c_r` adds what one of `c_a` does. The CPU reference costs 0.55 ns per product on one Apple M4 Max core (`pcs/combine`). An element entering three combined rows then costs about 3% of encoding and hashing it on that core: three NTT points and two hashed symbols. If that share carried over to the GPU, about 1 ps at `c_c = 30 ps`, the combination would add about 0.07 at 1K and 0.23 at 16K for `minimal`.
- The leaf stage's memory (RISK-10, [RFC-0054](../rfcs/RFC-0054-token-groups.md)).
  - **Groups.** The 16K-token batch takes 256, 512 and 1,024 token groups at 1K, 4K and 16K over 0.2 TB of free HBM. Its merge's regeneration is counted above.
  - **The merge's folds.** At an estimated 2 to 4 ps per merged element (`c_m`), they add 0.16 to 0.32 at 1K, 0.30 to 0.60 at 4K and 0.82 to 1.63 at 16K for `minimal`, and 0.18 to 0.36 at 1K and 0.84 to 1.67 at 16K for `servable`.
  - **The table trees** stream one round, which adds under 0.005.
  - **One batch instead.** Streaming every level with shared challenges, as RFC-0018 first specified, would cost 36, 45 and 55 regenerations for `minimal`, adding 53 to 59, 85 to 99 and 190 to 235.
  - **Not priced.** The reduction stages' elementwise products exceed the memory too ([#452](https://github.com/AbdelStark/tessera/issues/452)).

v1.0 release ceilings ([SC-2](00-overview.md#7-success-criteria)) are `μ(1K) ≤ 12` and `μ(16K) ≤ 45`.
- **`minimal`.**
  - **1K.** The corrected projection, about 11.8, leaves 2% headroom. With `c_a = c_c` it is about 13.9, 16% above its ceiling. With the merge's folds and the table round at their upper estimates as well, it is about 14.2, 18% above.
  - **16K.** The same assumptions give about 35.7, 1.26× below its ceiling.
  - **What holds.** The 1K ceiling holds only with the merge's folds at their lower estimate and `c_a` below about 1 ps. The 16K ceiling holds while `c_a` stays below about 70 ps at the upper estimates.
- **The default `servable` profile.** At 1K it is about 12.7, 6% above its ceiling before the column answers. With `c_a = c_c` it is about 15.2, and with the upper estimates 15.6, 30% above. At 16K it is about 37.0, below.

Meeting SC-2 at 1K therefore needs fewer regenerations ([RFC-0054](../rfcs/RFC-0054-token-groups.md#open-questions) OQ-0054-2), `c_a` well below `c_c`, or fewer decompositions. Sampling rates for a 10% amortized overhead scale inversely with `μ`. At 1K, the corrected projection cuts the affordable rate from 10.4% to about 2.8% (2.4% at `c_a = c_c`) for `minimal` and 2.6% (2.2%) for `servable`. It cuts daily detection of 100 violations in 10^6 requests to about 94% (91%) and 93% (89%).

## 4. Memory budgets

| Resource | Budget | Rationale |
|---|---|---|
| GPU HBM per H100 (80 GB) | Weights ≤ 51 GB (405 GB / 8); working set ≤ 24 GB; reserve ≥ 5 GB | Weights stay resident for the witness pass and fold pass |
| Witness residency | ≤ one tile per stream (INV-ARCH-3) | 16K-context witnesses reach terabytes |
| Leaf stage | Its lookup trees and forests, or one token group's ([RFC-0054](../rfcs/RFC-0054-token-groups.md#when-groups-apply)); every layer's keys and values for the batch, 4.2 GB at 405B for 16K tokens; the merge's tables, about 1 GB at 16K, in HBM or host RAM | A 405B batch's leaf stage holds 9×10^11 to 5×10^12 leaves at 1K to 16K |
| Column-digest state | 32 bytes × codeword length (2^23 → 256 MiB) | Chained column digests ([RFC-0011](../rfcs/RFC-0011-advice-commitment.md#column-digests-and-root)) |
| LogUp multiplicity counters | ≤ 2^32 rows per table × 4 bytes, sharded across GPUs; 405B's tables hold 19 to 42 GiB in all ([`docs/perf/v0.2/`](../perf/v0.2/README.md)) | Table-side cost amortized over the batch, 0.04 to 0.08 of `μ(1K)` at `B_max = 32` with [RFC-0028](../rfcs/RFC-0028-balanced-split.md)'s balanced split, which cut it from 0.15 to 0.20 (§3) |
| Host RAM, prover node | ≤ 1 TiB | Weight staging and checkpoint streaming |
| Storage, prover node | About 15.2 TB per epoch at 405B, twice that during a refresh | The weights' stored codewords, from which the column answers are read ([RFC-0012](../rfcs/RFC-0012-weight-commitment.md#column-answers)) |
| Host RAM, verifier | ≤ 64 GiB | SC-3 |

## 5. Proof size and verification budgets

| Quantity | Budget (16K tokens, 16K context, `live-64`) | Projection |
|---|---|---|
| Opened advice columns | `τ_A = 155` columns × column length × 8 bytes | about 0.73 GB with message width 2^22 |
| Combined rows | at most 17 rows (one per claim group plus proximity) × 2^22 × 16 bytes | at most 1.14 GB |
| Sumcheck messages | about 6 stages × ≤ 50 rounds × degree ≤ 4 × 16 bytes × batch factors | < 10 MB |
| Weight openings | `τ_W = 97` columns at most per matrix × 10 matrices, each column holding every non-zero data row and the 4,096 blinding rows | about 0.35 GB |
| **Total proof** | **≤ 4 GiB (gate), 2 GiB (target)** | about 2 GB |
| Verification time | ≤ 300 s on 32 cores | seconds to tens of seconds: column hashing, column combinations, combined-row re-encoding, table MLEs |
| Dense table evaluation | ≤ 20 dense tables of ≤ 2^24 rows | about 2 s single-threaded |

## 6. Regression gates

- Microbenchmarks (`cargo bench -p tessera-bench`, [docs/perf](../perf/README.md)): field, NTT, BLAKE3 column hashing, sumcheck round, LogUp layer, product-forest levels, Merkle opening, each run with RFC-0020's measurement protocol (3 warm-up runs, 10 measured runs) and stored as a `tessera.microbenchmarks/1` baseline per runner class. Gate: median of 10 runs regresses by more than 5% against the stored baseline on the same runner class.
- GPU unit costs: nightly on the H100 runner with locked clocks (`nvidia-smi -lgc`). Gate: more than 5% regression in any unit cost.
- End-to-end: nightly 8B proof at 1K context on the GPU runner (from v0.5). Gate: `μ` regresses by more than 10%.
- A failing gate opens an issue labeled `type:perf` with the run manifests of baseline and candidate attached.

## 7. Profiling plan

| Milestone | What is profiled | Tools |
|---|---|---|
| v0.1 | Unit-cost microkernels: encode and hash, fraction tree, product tree | Nsight Compute (`ncu`) roofline; Nsight Systems (`nsys`) timelines |
| v0.3 | CPU reference prover per stage | `perf`, flame graphs, `tracing` spans |
| v0.5 | GPU prover per phase and stage; multi-GPU overlap; HBM bandwidth utilization | `nsys`, `ncu`, NVLink counters |
| v1.0 | 405B end to end at 1K, 4K, 16K | Same, plus run-manifest metrics |

Each profile is archived with its run manifest and referenced from the milestone's performance report.
