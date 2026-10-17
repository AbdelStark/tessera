# 08 — Performance Budget

- Status: Accepted for v1 planning
- Governs: cost model, unit-cost targets, end-to-end targets, memory and proof-size budgets, profiling plan, regression gates
- Detailed in: [RFC-0020](../rfcs/RFC-0020-cost-model.md)

Every number in this document is a projection until RFC-0020's measurements replace it. Projections are labeled as such; measured values will be labeled with their run manifest.

## 1. Cost model

The proof multiplier for a model at context `S` (source §5):

```math
\mu(S) \approx \frac{1.15\,(C_{lin} c_{c} + K_{lin} c_{\ell}) + 1.15\cdot A\, S\,(c_{c}+c_{\ell}+c_{\pi}) + W/T}{(F_{lin} + F_{att} S)/R} \;+\; n_{regen}
```

| Symbol | Meaning | 405B value |
|---|---|---|
| `C_lin`, `K_lin` | Committed elements and lookups per token in linear layers | 1.7×10^7 each (source); with RISK-2's decompositions, 3.6×10^7 committed elements and 3.8×10^7 lookups for the `minimal` profile, and 4.5×10^7 each for the default `servable` one |
| `A` | Scores per token per unit context: heads × layers / 2 | 8,064 |
| `c_c`, `c_ℓ`, `c_π` | Unit costs: committed element, lookup, product-tree leaf | 30, 40, 12 ps (assumed) |
| 1.15 | Sumcheck glue factor | assumed |
| `W/T` | Weight pass per token: 0.24 GPU-s per 16K-token batch. The weight column answers add no GPU time: they are read from the epoch's stored codewords ([RFC-0012](../rfcs/RFC-0012-weight-commitment.md#column-answers)) | — |
| `F_lin`, `F_att` | Native INT8 operations per token: linear, attention per unit context | 8.1×10^11, 4.1×10^6 |
| `R` | Native rate: 60% of 1,979 TOPS dense INT8 on H100 SXM | 1.19×10^15 op/s |
| `M` | Table rows per proof: one committed multiplicity and one fraction-tree leaf each ([RFC-0008](../rfcs/RFC-0008-logup-and-tables.md)) | 6.7×10^9 for `minimal` and 7.7×10^9 for `servable`, with two classes at `B_max = 32` and [RFC-0028](../rfcs/RFC-0028-balanced-split.md)'s balanced split ([`docs/perf/v0.2/`](../perf/v0.2/README.md)) |
| `n_regen` | Witness regenerations, each about one native forward pass: the passes P1 to P5 of [RFC-0018](../rfcs/RFC-0018-gpu-prover-runtime.md#pass-structure), which are the witness pass, the tree leaf levels, stage regeneration, the opening pass and the column answers. Absent from the source model (RISK-9). | 5 (estimate) |

`tessera bench model` implements this formula and three terms it leaves out: the streamed passes of RISK-10 over the attention trees' leaf level, the column answers' evaluation of every committed element at the queried columns, at `c_a` per element (RFC-0018 P5), and the table side of LogUp, `1.15 · M · (c_c + c_ℓ) / T`. It reads the counts from the compiler's commitment report, so projections update automatically when the compiler changes ([RFC-0020](../rfcs/RFC-0020-cost-model.md#executable-cost-model)).

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
- Five regeneration passes (RISK-9, RFC-0018 P1 to P5) add 5.
- The table side of LogUp commits one multiplicity and builds one fraction-tree leaf per table row, once per proof ([RFC-0008](../rfcs/RFC-0008-logup-and-tables.md)). With two classes at `B_max = 32` and [RFC-0028](../rfcs/RFC-0028-balanced-split.md)'s balanced split, the tables hold 6.7×10^9 rows for `minimal` and 7.7×10^9 for `servable`. Shared by the 16K-token batch, they add 0.048 at 1K and 0.044 at 16K for `minimal`, and 0.055 and 0.051 for `servable`. That gives `μ(1K) ≈ 10.6` and `μ(16K) ≈ 25.9` for `minimal`, and `μ(1K) ≈ 11.5` and `μ(16K) ≈ 26.8` for `servable`.
- The column answers (P5) also evaluate every committed element at the queried columns, at a cost `c_a` per element that is not yet measured. Each picosecond of `c_a` adds 0.064 at 1K and 0.23 at 16K for `minimal`, and 0.078 and 0.24 for `servable`. Re-encoding a row costs at most what committing it does, `c_c = 30 ps` per element (assumed), which adds 1.9 and 6.8, or 2.3 and 7.2.
- Streaming the attention trees' leaf levels (RISK-10) adds 0.36 to 0.48 at 1K, 2.0 to 2.7 at 4K and 9.6 to 12.8 at 16K, in both profiles. The 16K-token batch streams 5, 7 and 9 passes over 0.2 TB of free HBM, each at 6 to 8 ps per score ([RFC-0018](../rfcs/RFC-0018-gpu-prover-runtime.md#pass-structure) 'Leaf levels'). The rest of the leaf stage streams too, and is not priced yet ([#329](https://github.com/AbdelStark/tessera/issues/329)).

v1.0 release ceilings ([SC-2](00-overview.md#7-success-criteria)) are `μ(1K) ≤ 12` and `μ(16K) ≤ 45`. For `minimal`, with `c_a = c_c`, the 1K projection is about 12.5, 4% above its ceiling. The upper streaming estimate raises it to about 13.0, 8% above, and the 16K projection to about 45.6, 1.3% above. For the default `servable` profile the same assumptions give about 13.9 at 1K and 14.3 with streaming, 15% and 19% above its ceiling, and about 46.8 at 16K, 4% above. With streaming at its upper estimate, `minimal`'s ceilings hold while `c_a` stays below about 15 ps at 1K and 27 ps at 16K. `servable` reaches its 1K ceiling with streaming alone. Meeting SC-2 in either profile therefore needs `c_a` well below `c_c`, cheaper streaming, or fewer decompositions. Sampling rates for a 10% amortized overhead scale inversely with `μ`. At 1K, the corrected projection cuts the affordable rate from 10.4% to about 3.2% (2.7% at `c_a = c_c`) for `minimal` and 2.9% (2.4%) for `servable`, and daily detection of 100 violations in 10^6 requests to about 96% (93%) and 95% (91%).

## 4. Memory budgets

| Resource | Budget | Rationale |
|---|---|---|
| GPU HBM per H100 (80 GB) | Weights ≤ 51 GB (405 GB / 8); working set ≤ 24 GB; reserve ≥ 5 GB | Weights stay resident for the witness pass and fold pass |
| Witness residency | ≤ one tile per stream (INV-ARCH-3) | 16K-context witnesses reach terabytes |
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
