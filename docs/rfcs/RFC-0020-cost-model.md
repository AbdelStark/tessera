# RFC-0020: Cost Model and Unit-Cost Measurement Gate

- Status: Accepted
- Authors: @AbdelStark
- Created: 2026-10-15
- Target milestone: v0.1
- Amended by: [RFC-0034](RFC-0034-causal-blocks.md) ('Executable cost model': the score block), [RFC-0054](RFC-0054-token-groups.md#the-executable-model) ('Executable cost model': the leaf stage's plan, the merge and the streaming term), [RFC-0055](RFC-0055-windowed-stages.md#the-executable-model) ('Executable cost model': the reduction stages' plan, their fold passes and folds; 'Testing Strategy')

## Summary

Every performance number in the source and in this corpus is a projection from a few unit costs and counts. This RFC turns the projection into code (`tessera bench model`), fixes how the unit costs are measured on H100 at milestone v0.1, and fixes the decision rule that follows the measurement: proceed, proceed with a re-plan, or stop and redesign. After v0.1, every production run measures the same unit costs as a side effect, and the model is re-fit against end-to-end measurements at each milestone.

## Motivation

"Every number here is a projection built on two unit costs, picoseconds per committed element and per lookup, that the first implementation milestone must measure on H100" (source §1). "Both comparisons rest on unmeasured unit costs; the first implementation milestone is to measure them" (source §8). The corpus adds terms the source omits: witness regeneration (RISK-9), streamed leaf levels (RISK-10), the column answers and the table side of LogUp. It also corrects the commitment counts (RISK-2). A model that cannot be run, or a measurement that cannot be reproduced, cannot carry these corrections.

## Goals

- An executable model that reproduces the source's tables from the source's inputs and accepts measured inputs.
- A reproducible measurement protocol for `c_c`, `c_ℓ`, `c_π`, glue, the weight pass and the native rate.
- A decision rule fixed before the measurement is taken.
- Continuous measurement and model re-fitting thereafter.

## Non-Goals

- Optimizing kernels (RFC-0018 and its issues).
- Measuring non-H100 hardware at v0.1 (OQ-0020-1).

## Proposed Design

### Executable cost model

`tessera-bench` implements:

```rust
pub struct UnitCosts { pub commit_ps: f64, pub lookup_ps: f64, pub product_leaf_ps: f64,
                       pub glue_fraction: f64, pub weight_pass_gpu_s: f64, pub native_ops_per_s: f64,
                       pub streaming_pass_ps: f64,                       // per table leaf per streamed round
                       pub merge_ps: f64,                                // per merged element, RFC-0054
                       pub window_ps: f64,                               // per windowed table entry, RFC-0055
                       pub column_answer_ps: f64,                        // per committed element, P5
                       pub row_combination_ps: f64 }                     // per committed element, P4
pub struct Counts {                       // from the compiler's CommitmentReport
    pub commits_per_token: f64, pub lookups_per_token: f64,          // linear layers
    pub scores_per_token_per_ctx: f64,                               // A = heads × layers / 2
    pub native_ops_lin: f64, pub native_ops_att_per_ctx: f64,
    pub regen_full_equivalents: f64,                                 // RISK-9
    pub table_rows: f64,                  // M: table rows per proof, the report's multiplicities
    pub score_block: f64,                 // b: RFC-0034's score block, 0 for the valid half alone
    pub reduction_stages: Vec<ReductionStage>,  // the report's stage tables, RFC-0055
}
pub struct ReductionStage { pub tables: Vec<StageTable> }
pub struct StageTable { pub per_token: bool, pub log2_entries: u32, pub count: f64 }
pub struct MemoryModel { pub free_bytes: f64, pub bytes_per_leaf: f64 }  // RISK-10, RFC-0054
pub struct LeafPlan { pub groups: u32, pub table_rounds: u32 }          // RFC-0054's plan
pub fn leaf_plan(counts: &Counts, mem: &MemoryModel, context: u32, batch_tokens: u32)
    -> Result<LeafPlan, ModelError>;
pub fn shared_streaming_passes(counts: &Counts, mem: &MemoryModel, context: u32, batch_tokens: u32)
    -> Result<u32, ModelError>;                                          // RFC-0018 as first written
pub struct StageWindow { pub bytes: f64, pub window_rounds: u32 }       // RFC-0055's plan
pub fn stage_plan(counts: &Counts, mem: &MemoryModel, batch_tokens: u32)
    -> Result<Vec<StageWindow>, ModelError>;
pub fn mu(counts: &Counts, u: &UnitCosts, mem: &MemoryModel, context: u32, batch_tokens: u32)
    -> Result<MuBreakdown, ModelError>;
pub struct MuBreakdown { pub linear: f64, pub attention: f64, pub weights: f64, pub regen: f64,
                         pub streaming: f64, pub column_answers: f64, pub row_combination: f64,
                         pub tables: f64,
                         pub total: f64 }
```

`μ` follows [08 §1](../spec/08-performance-budget.md#1-cost-model): the source's formula plus `n_regen` and four more terms. A token commits `A·S'` scores, `S' = S + min(S, score_block)`, and looks up and multiplies `A·S` of them ([RFC-0034](RFC-0034-causal-blocks.md#cost)):

- the leaf stage's plan ([RFC-0054](RFC-0054-token-groups.md#when-groups-apply)): its token groups `G` and its table trees' streamed rounds `r_t`, from the batch's `T·(K + 2·A·S)` lookup and forest leaves and `M` table leaves at `bytes_per_leaf` against `free_bytes`. A grouped stage adds its merge's pass to `n_regen` when `n_regen` is positive. Then a streaming term `(p_m · merge_ps · (K + 2·A·S) + r_t · streaming_pass_ps · M / batch_tokens) / native`, with `p_m` one when the stage is grouped: the merge's folds, an element per leaf, and the table trees' rounds. `shared_streaming_passes` counts what RFC-0018 first specified, every level of every tree streaming in one batch, `c(c + 1)/2` regenerations;
- the reduction stages' plan ([RFC-0055](RFC-0055-windowed-stages.md#the-window)): each stage's working set, its tables at 16 bytes per entry at the batch, and its window `c < n_max`, the fewest rounds after which its tables fit in `free_bytes`. Each windowable table is folded by `2^{max(0, c − p_i)}` with `p_i` its front padding; a matrix product's stays whole. Each windowed stage adds its fold pass to `n_regen` when `n_regen` is positive, and `window_ps · Σ_w E_w / batch_tokens / native` to the streaming term, `E_w` its windowed tables' entries. The native gadget relations are not counted yet ([#458](https://github.com/AbdelStark/tessera/issues/458));
- a column-answer term `(C + A · S') · column_answer_ps / native`, P5's evaluation of every committed element at the queried columns ([RFC-0018 'Column answers'](RFC-0018-gpu-prover-runtime.md#pass-structure)). A token commits `C` elements in its linear layers and one per score. P5 runs no sumcheck, so the term carries no glue.
- a row-combination term `(C + A · S') · row_combination_ps / native`: P4's combination of every committed element into the opening's combined rows ([RFC-0018 'Opening pass'](RFC-0018-gpu-prover-runtime.md#pass-structure)). It carries no glue either.
- a table term `(1 + glue) · M · (c_c + c_ℓ) / batch_tokens / native`. A proof commits one multiplicity per table row and builds the table's fraction tree over every row, however few lookups the table serves ([RFC-0008](RFC-0008-logup-and-tables.md)). Each of the `M` rows therefore costs one committed element and one tree leaf, priced as a lookup, once per proof. `M` is the commitment report's multiplicities ([#304](https://github.com/AbdelStark/tessera/issues/304)).

The weight column answers have no term. The prover reads them from its store of the epoch's codewords, overlapped with P5 ([RFC-0012 'Column answers'](RFC-0012-weight-commitment.md#column-answers)). A prover without the store evaluates them, about 1.3 GPU-s per 405B proof unit at `live-64`, which belongs in `weight_pass_gpu_s`.

Floating-point is acceptable here: the model is a planning tool, not part of the protocol. Inputs are checked: every count and cost finite and non-negative, the native rate, free memory and batch positive. An input the model cannot price is an error, never a NaN in a report.

`tessera bench model --plan plan.tep --context S` reads counts from the commitment report; with `--source-defaults` it uses the source's counts and unit costs. Outputs: `MuBreakdown` per context, proof time per request on `G` GPUs, sampling rate for a target amortized overhead, and daily detection probability for a given violation count (as in [RFC-0016](RFC-0016-audit-sampling.md#calculators)).

**Golden test.** With source defaults, `regen_full_equivalents = 0`, no merge or streaming cost, no column answers, no row combination, no table rows and no score block, the model reproduces the source's §5 chart data and §8 table: `μ` at 1K, 2K, 4K, 8K, 16K, 32K, 64K and 128K within ±0.05, and the sampling and detection columns within ±0.1 percentage point, computed from that table's `μ` column, which the source rounds to one decimal. [`spec/vectors/cost-model/`](../../spec/vectors/cost-model/README.md) records each figure with its line and each input with its provenance: `F_lin`, which the source writes as 8.1×10^11, is two operations per parameter, 2 × 405,853,388,800.

### Measurement protocol (v0.1 gate)

**Environment.** One H100 SXM5 80 GB; persistence mode on; ECC on; MIG off; exclusive process; application clocks locked to the maximum sustained SM clock and recorded; CUDA, driver and `tessera` versions recorded in the run manifest. All inputs are generated on the device by a counter-based generator so that host transfers are excluded.

| Unit | Microbenchmark | Size | Normalization |
|---|---|---|---|
| `c_c` | Rate-1/2 coset RS encoding with vanishing mask and BLAKE3 chained column hashing (`B = 32`) of rows of width 2^22 | ≥ 2^34 committed elements | wall time / committed elements |
| `c_ℓ` | LogUp-GKR prover, combined fraction tree, `Fp2` challenges, in-memory levels, including bottom-up build | 2^30 lookups into a 2^24-row table | wall time / lookups |
| `c_π` | Product-tree GKR batched over rows of 2^14 leaves | 2^30 leaves | wall time / leaves |
| Glue | Matmul and elementwise sumcheck rounds over 2^28-entry tables | — | percent of `c_c + c_ℓ` time for a representative mix |
| Weight pass | Limb-decomposed INT8 GEMM fold over one GPU's share of 405B weights (about 51 GB) | 51 GB | seconds |
| Native rate | INT8 GEMMs at 405B layer shapes with INT32 accumulation | all eight per-layer GEMM shapes | fraction of 1,979 TOPS achieved |
| Streaming pass | Regenerate scores and fold one attention tile's leaves | 2^30 scores | ps per score |
| Column answers `c_a` | Evaluate rows at `τ_A` distinct queried columns, hashing nothing: the `c_c` benchmark's encoder, and each query-only evaluation of [RFC-0018 'Column answers'](RFC-0018-gpu-prover-runtime.md#pass-structure) | the `c_c` benchmark's rows | wall time / committed elements, cheapest evaluation |
| Row combination `c_r` | Combine rows into a proximity row and two groups' combined rows, in blocks of `B = 32` rows on tensor cores ([RFC-0018 'Opening pass'](RFC-0018-gpu-prover-runtime.md#pass-structure)) | the `c_c` benchmark's rows | wall time / committed elements |

**Statistics.** Each benchmark runs 3 warm-up iterations and 10 measured iterations; the report gives median, minimum, maximum and coefficient of variation. A coefficient of variation above 3% is flagged and the run repeated. Every benchmark's output digest is compared with the CPU reference on a subsample, so a fast but wrong kernel cannot pass.

**Report.** `tessera bench unit-costs` writes `tessera.unit-costs/1` (JSON Schema in `spec/schemas/`): environment, each unit's statistics, Nsight Compute roofline data (achieved DRAM bandwidth, SM and tensor-pipe utilization) and the model's restated projections for 405B at 1K, 4K, 16K and 128K. The report and its run manifests are committed under `docs/perf/v0.1/`.

### Decision rule

Let `ρ_x = measured / assumed` for `x ∈ {c_c, c_ℓ, c_π}` (assumed 30, 40 and 12 ps).

| Outcome | Condition | Action |
|---|---|---|
| Proceed | every `ρ_x ≤ 2` | v0.5 GPU work proceeds as planned; projections in [00](../spec/00-overview.md) and [08](../spec/08-performance-budget.md) are restated with measured values |
| Proceed with re-plan | some `ρ_x ∈ (2, 4]` | Recompute `μ`; if the corrected 405B projection exceeds an SC-2 ceiling, amend SC-2 through an RFC and pull mitigations into scope (OQ-0004-1 hashing volume, small-value commitments NG-5, multi-variable rounds) |
| Stop and redesign | some `ρ_x > 4` | GPU prover work stops; a superseding RFC for the affected component is required before v0.5 work resumes |

The rule is recorded here, before measurement, so that the outcome cannot shape the criteria.

### Continuous measurement

- Every prover run derives unit costs from its own metrics (phase time divided by counts) and records them in the run manifest ([05 §4](../spec/05-observability.md#4-metrics)).
- Nightly GPU benchmarks gate regressions ([08 §6](../spec/08-performance-budget.md#6-regression-gates)).
- At each milestone from v0.5, the model is re-fit to measured end-to-end `μ` on 8B and 70B (v0.5) and 405B (v1.0). The report states the model's error per context length; the v1.0 target is within 25%.
- `μ` is always reported against the ideal native denominator (op counts at 60% of peak, the source's definition) and, separately, against the measured native INT8 forward on the same hardware.

## Alternatives Considered

1. **End-to-end measurement only.** Rejected: the first end-to-end GPU proof arrives at v0.5; discovering then that a unit cost is 5× off would waste the GPU work in between.
2. **Analytical roofline only.** Rejected: NTT, hashing and fraction-tree efficiency depend on implementation details a roofline cannot predict.
3. **Measuring on B200 first.** Rejected: the source's assumptions and its comparison with prior work are stated for H100 (OQ-0020-1 covers B200).

## Drawbacks

- Microbenchmarks can diverge from in-situ costs (cache effects, interference between phases); the continuous measurement corrects this later.
- The decision rule's thresholds are judgment calls, chosen to separate constant-factor slippage (up to 2×) from structural problems (over 4×).

## Migration / Rollout

v0.1 delivers the model, its golden test, the microbenchmarks, the report and the recorded decision. Later milestones extend the model with measured counts and passes.

## Testing Strategy

- Golden test against the source's tables, as above.
- Model unit tests: monotonicity in each unit cost; the streaming term is zero when leaves fit in memory and steps up by one pass per doubling of `N_level` beyond it; the column-answer and row-combination terms priced at `c_c`, with glue and the other unit costs at zero, each equal the linear and attention terms, since P4 and P5 handle exactly the elements P1 commits; each table row costs what a token's committed lookup costs, once per batch; a reduction stage takes the fewest window rounds after which its tables fit, front padding counted, and one fold pass; the corrected projections and streaming figures that [08 §3](../spec/08-performance-budget.md#3-end-to-end-targets-llama-31-405b-one-8h100-node), SC-2 and RISK-10 quote are the model's.
- Benchmark correctness: output digests match CPU references; a deliberately slowed kernel (sleep injection) changes the reported unit cost proportionally.
- Report validation: JSON Schema validation in CI for every committed report.

## Open Questions

- **OQ-0020-1.** Should the reference hardware move from H100 to B200 for v1.0 performance claims? Owner: @AbdelStark. Resolve at v0.5 after measuring B200 unit costs with the same protocol.

## References

- Source §1, §5 (cost model, unit costs, chart data), §8 (projections and sampling table).
- [08 Performance budget](../spec/08-performance-budget.md), [RFC-0006](RFC-0006-compiler-ir.md#outputs), [RFC-0018](RFC-0018-gpu-prover-runtime.md#pass-structure).
