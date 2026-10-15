# 08 — Performance Budget

- Status: Accepted for v1 planning
- Governs: cost model, unit-cost targets, end-to-end targets, memory and proof-size budgets, profiling plan, regression gates
- Detailed in: [RFC-0020](../rfcs/RFC-0020-cost-model.md)

Every number in this document is a projection until RFC-0020's measurements replace it. Projections are labeled as such; measured values will be labeled with their run manifest.

## 1. Cost model

The proof multiplier for a model at context `S` (source §5):

```math
\mu(S) \approx \frac{1.15\,(C_{lin} c_{c} + K_{lin} c_{\ell}) + 1.15\cdot A\, S\,(c_{c}+c_{\ell}+c_{\pi}) + W/T}{(F_{lin} + F_{att} S)/R}
```

| Symbol | Meaning | 405B value |
|---|---|---|
| `C_lin`, `K_lin` | Committed elements and lookups per token in linear layers | 1.7×10^7 each (source); up to 2.8×10^7 (RISK-2) |
| `A` | Scores per token per unit context: heads × layers / 2 | 8,064 |
| `c_c`, `c_ℓ`, `c_π` | Unit costs: committed element, lookup, product-tree leaf | 30, 40, 12 ps (assumed) |
| 1.15 | Sumcheck glue factor | assumed |
| `W/T` | Weight pass per token: 0.24 GPU-s per 16K-token batch | — |
| `F_lin`, `F_att` | Native INT8 operations per token: linear, attention per unit context | 8.1×10^11, 4.1×10^6 |
| `R` | Native rate: 60% of 1,979 TOPS dense INT8 on H100 SXM | 1.19×10^15 op/s |

`tessera bench model` implements this formula with counts from the compiler's commitment report, so projections update automatically when the compiler changes ([RFC-0020](../rfcs/RFC-0020-cost-model.md#executable-cost-model)).

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

v1.0 release ceilings ([SC-2](00-overview.md#7-success-criteria)): `μ(1K) ≤ 8`, `μ(16K) ≤ 40`. At the RISK-2 upper bound the linear-layer term rises from about 2.1 to 3.3, so projected `μ(1K)` becomes about 4.5 and `μ(16K)` about 20; both stay within the ceilings.

## 4. Memory budgets

| Resource | Budget | Rationale |
|---|---|---|
| GPU HBM per H100 (80 GB) | Weights ≤ 51 GB (405 GB / 8); working set ≤ 24 GB; reserve ≥ 5 GB | Weights stay resident for the witness pass and fold pass |
| Witness residency | ≤ one tile per stream (INV-ARCH-3) | 16K-context witnesses reach terabytes |
| Column-digest state | 32 bytes × codeword length (2^23 → 256 MiB) | Chained column digests ([RFC-0011](../rfcs/RFC-0011-advice-commitment.md#column-digests)) |
| LogUp multiplicity counters | ≤ 2^32 rows per table × 4 bytes, sharded across GPUs | Table-side cost amortized over the batch |
| Host RAM, prover node | ≤ 1 TiB | Weight staging and checkpoint streaming |
| Host RAM, verifier | ≤ 64 GiB | SC-3 |

## 5. Proof size and verification budgets

| Quantity | Budget (16K tokens, 16K context, `live-64`) | Projection |
|---|---|---|
| Opened advice columns | `τ_A = 155` columns × column length × 8 bytes | about 0.73 GB with message width 2^22 |
| Combined rows | about 4 rows × 2^22 × 16 bytes | about 0.27 GB |
| Sumcheck messages | about 6 stages × ≤ 50 rounds × degree ≤ 4 × 16 bytes × batch factors | < 10 MB |
| Weight openings | `τ_W = 95` columns per weight type × 9 types | < 0.3 GB |
| **Total proof** | **≤ 4 GiB (gate), 1.5 GiB (target)** | about 1.3 GB |
| Verification time | ≤ 300 s on 32 cores | seconds to tens of seconds: column hashing, column combinations, combined-row re-encoding, table MLEs |
| Dense table evaluation | ≤ 20 dense tables of ≤ 2^24 rows | about 2 s single-threaded |

## 6. Regression gates

- Microbenchmarks (`criterion`): field, NTT, BLAKE3 column hashing, sumcheck round, LogUp layer, Merkle opening. Gate: median of 10 runs regresses by more than 5% against the stored baseline on the same runner class.
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
