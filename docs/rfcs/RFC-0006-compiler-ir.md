# RFC-0006: Compiler IR: Typed Integer Dataflow, Range Analysis, and Gadget Selection

- Status: Accepted
- Authors: @AbdelStark
- Created: 2026-10-15
- Target milestone: v0.2

## Summary

`tessera compile` lowers an `IntegerModel` into a typed integer dataflow graph in which every tensor carries a declared integer range and one of three kinds: virtual (never committed), committed advice, or lookup output. A sound range analysis proves that every lookup key and every gadget input is a true integer inside its table. Gadget selection maps each non-arithmetic node to a gadget from RFC-0009 and sizes its tables. Tiling cuts the graph into shallow independent tiles. The compiler emits the public claim list (RFC-0007), the private execution plan, and a commitment report that feeds the cost model. This RFC fixes the IR, the analysis rules and the selection rules.

## Motivation

The source's central cost lever is "commit only what is not arithmetic" (source §6, principle 2), constrained by "let the field absorb chains": linear arithmetic may wrap modulo `P`, but every lookup key must be a true integer in range (principle 3). Turning that into a buildable artifact needs a typed IR in which the constraint is checkable, an analysis that is sound for every input (not only calibration data), and a report that states the true commitment count rather than the projection (RISK-2).

## Goals

- A closed operation set covering the dense Llama-3.1 block, the scoring circuit and the anchor circuits.
- A range analysis that over-approximates every reachable value (INV-DM-2) and proves the true-integer property for every key.
- Deterministic output: identical bytes for identical `IntegerModel` and `ParamSet` (INV-DM-5).
- A commitment report precise enough to replace the source's projected counts.

## Non-Goals

- Searching over alternative requantization placements. Placement is part of `D`'s semantics and is fixed by the profile template at quantization time ([RFC-0005](RFC-0005-integer-model.md#profiles)).
- General model import (ONNX or framework graphs).
- Kernel scheduling inside a tile (RFC-0018).

## Proposed Design

### Operation set

```rust
pub enum Op {
    // Virtual, linear in their non-constant inputs
    MatMulWeight { weight: WeightRef },          // W x, W committed in the weight commitment
    Add, Sub, ConstMul { c: ConstRef },          // public integer constants, per channel allowed
    ConstAdd { c: ConstRef },
    Rotate { rope: TableId },                    // RoPE pair rotation by public constants
    Sum { axis: u8 }, Reshape, Slice, GatherPublic { index: ConstRef },
    // Virtual, degree 2
    MatMulAct,                                   // activation x activation: QK^T, PV
    Mul,                                         // elementwise product
    SumSquares { axis: u8 },
    // Gadget-backed: produce committed advice
    OneHot { vocab: u32 },                       // committed one-hot token vector
    Requant { shift: ShiftClassRef, lo: i64, hi: i64 },
    TableMap { table: TableId },                 // SiLU and other single-input nonlinearities
    SoftmaxRow { exp_table: TableId, sigma: u8 },// max certificate + reciprocal bracket
    RsqrtRow { kappa: u8, eps_int: u64 },        // multi-limb bracket
    Argmax, InverseCdf { prf: PrfRef },          // scoring certificates (RFC-0017)
    Bitwise { table: TableId },                  // AES and SHA-256 sub-operations (RFC-0015)
}

pub struct Node { pub id: NodeId, pub op: Op, pub inputs: Vec<TensorId>, pub output: TensorId }
pub struct Graph { pub nodes: Vec<Node>, pub tensors: Vec<TensorSpec> }  // TensorSpec: 03 §3
```

Node kinds follow from the operation: linear and degree-2 operations produce `Virtual` tensors; gadget-backed operations produce `Committed` or `LookupOutput` tensors plus the gadget's internal advice (limbs, multiplicities). Weights are `Committed { Weight }`; constants and shapes are `Public`.

### Range analysis

- Ranges are exact integer intervals computed with `i128` endpoints; an interval that would leave `i128` is marked unbounded.
- Inputs: token ids in `[0, vocab)`; positions in `[0, max_context)`; weights in `[−127, 127]`; activations of each type over their full type range. Ranges hold for every admissible input, never for calibration statistics.
- Transfer functions: interval arithmetic for `Add`, `Sub`, `ConstMul`, `ConstAdd`, `Sum` (scaled by axis length), `MatMulWeight` (using only the public bound `127 · fan_in · max|x|`, never the actual weight values: bounds derived from secret weights would leak them through the claim list and could not be re-derived by the verifier), `Rotate` (using the public RoPE constants), `Mul`, `SumSquares`, `MatMulAct`.
- Gadget outputs take their declared output type range (for example `[−128, 127]` after `Requant`).
- **Relational facts.** Gadgets may emit facts that tighten downstream bounds. v1 has one: `SoftmaxRow` emits `R_q · Σ_i e_qi ≤ 2^{2σ}`, so `R_q · Σ_i e_qi v_i` is bounded by `2^{2σ} · max|v|` rather than by the product of the individual bounds. Each fact has a proof obligation in RFC-0009 and is re-derived by the verifier.

**True-integer rule.** A tensor used as a lookup key or a gadget input that requires an integer must have a bounded range inside `(−(P−1)/2, (P−1)/2)` and, for keys, inside its table's key domain. Intermediate virtual tensors may exceed that range: they are computed modulo `P`, and because every key is an integer-linear or polynomial function of them whose own exact range is bounded, the key's field value equals its true integer value. The analysis therefore never needs to track wraps explicitly; it checks bounds only where integrality matters.

Violations: a key outside every feasible gadget's bound raises `ERR-COMPILE-RANGE`; an integrality-requiring input with an unbounded range raises `ERR-COMPILE-WRAP`.

### Gadget selection

For each `Requant` and `TableMap` node, with key bit width `b` (signed range width) and dropped bits `r`:

| Condition | Gadget | Cost per element |
|---|---|---|
| `b ≤ B_max` | `Requant1` / single-lookup `TableMap` | 1 commitment, 1 lookup |
| `B_max < b ≤ 2 · B_max − 2` | `RequantSplit` with split point `s = b − B_max` | 2 commitments, 2 lookups |
| otherwise | `ERR-COMPILE-RANGE` | — |

`B_max = 32` by default (`TableLimits.max_log2_rows`), the largest table whose prover-side cost is amortized over a 16K-token batch ([08 §4](../spec/08-performance-budget.md#4-memory-budgets)). The verifier's cost to evaluate a table does not depend on `b`: every v1 table is a piecewise function whose non-constant core has at most 2^24 entries (for a clamped requantization the core is the output range, 256 values for INT8), and its multilinear extension is evaluated in time proportional to the core ([RFC-0008](RFC-0008-logup-and-tables.md#structured-evaluation)). Tables with identical definitions are shared across layers and channels; a requantization table is identified by `(shift, lo, hi, b)`.

### Llama lowering

The lowering instantiates the profile's template from [RFC-0005](RFC-0005-integer-model.md#integer-semantics) per layer: embedding (`OneHot` then `MatMulWeight` on `Embed`), attention norm (`SumSquares`, `RsqrtRow`, `Mul`, `Requant`), Q/K/V (`MatMulWeight`), RoPE (`Rotate`, `Requant`), V (`Requant`), attention per KV group (`MatMulAct`, `SoftmaxRow`, `MatMulAct`, `Mul` by `R_q`), O projection, residual (`Requant`, `Add`), MLP norm, gate and up, `TableMap` (SiLU), product and down projection per profile, residual, final norm, LM head. Scoring nodes (RFC-0017) consume the logits; anchor circuits (RFC-0015) consume the token streams that also feed `OneHot`.

### Tiling and the execution plan

- Linear tiles are `(layer, operation)`; attention tiles are `(layer, kv_head, query_block)` with query blocks of 512 positions by default.
- Each tile's committed tensors occupy a contiguous range of advice rows (RFC-0011), assigned in a fixed traversal order: layer, then tile kind, then head, then query block.
- The `ExecutionPlan` (schema `0x0202`, private) lists tiles, their advice row ranges, the stage each relation joins (RFC-0010), and device assignments (RFC-0018). It contains no secret values.

### Outputs

```rust
pub fn compile(model: &IntegerModel, params: &ParamSet, limits: &TableLimits)
    -> Result<CompileOutput, CompileError>;

pub struct CompileOutput {
    pub claim_list: ClaimList,          // public, RFC-0007
    pub plan: ExecutionPlan,            // private
    pub tables: TableSetManifest,       // public, RFC-0008
    pub report: CommitmentReport,       // public JSON, tessera.commitment-report/1
}
```

`CommitmentReport` gives, per layer per token and in total for a reference batch: requantized elements by tensor; committed elements by class (lookup outputs, split limbs, per-row scalars, one-hot vectors, multiplicities, masks); lookups by table; product-tree leaves; table row counts; stage count; and the cost-model projection of `μ` at 1K, 4K, 16K and 128K using the current unit costs ([RFC-0020](RFC-0020-cost-model.md#executable-cost-model)).

### Invariants

- **INV-IR-1.** Every tensor's declared range over-approximates its reachable values.
- **INV-IR-2.** Every lookup key's range lies inside its table's key domain and inside the centered field range.
- **INV-IR-3.** Every committed tensor has exactly one producing gadget and its uniqueness argument (RFC-0009) covers it.
- **INV-IR-4.** `compile` is a pure function of its inputs.
- **INV-IR-5.** Range analysis and gadget selection read only public data (the claim list's contents), so the verifier re-derives them exactly (SP-7).

## Alternatives Considered

1. **Search over requantization placements.** The source describes a compiler that "places rescales to minimize commitments" (source §7). Considered. Rejected for v1 because moving a requantization changes `D`'s outputs, so placement must be co-optimized with fidelity at quantization time. The two profile templates cover the design points the source discusses; a placement search can be added later as a quantizer feature.
2. **MLIR or ONNX as the IR.** Rejected: large dependencies, floating-point-first semantics, and the verifier would still need its own independent representation of the claim list.
3. **A general constraint DSL (R1CS- or Plonkish-style).** Rejected: the prover's efficiency comes from tensor-structured batched sumchecks, which a flat constraint list discards.
4. **Calibration-based ranges.** Rejected: ranges must hold for every input, or an honest prover fails on rare inputs; soundness additionally requires the verifier to re-derive them from public data.
5. **Wrap tracking instead of bounds at keys.** Rejected as unnecessary: bounding keys' exact integer ranges already implies their field values are true integers.

## Drawbacks

- Worst-case ranges are loose; they force split gadgets on tensors whose typical values would fit a single lookup (RISK-2).
- Template-based placement leaves commitment reductions from smarter placement unexplored in v1.

## Migration / Rollout

v0.2 delivers IR, range analysis, gadget selection, Llama lowering and the commitment report for both profiles on 8B and 405B shapes (405B uses random weights of the right shape for counting). Tiling and the execution plan land in v0.3 with the first prover.

## Testing Strategy

- Range soundness: the integer engine runs 10^4 random and 10^3 adversarial inputs (maximal-magnitude activations, all-equal scores, extreme positions) on each conformance model; every observed value lies within its declared range.
- Gadget selection: synthetic nodes at `b = B_max`, `B_max + 1`, `2·B_max − 2`, `2·B_max − 1` select `Requant1`, `RequantSplit`, `RequantSplit`, `ERR-COMPILE-RANGE`.
- Relational fact: a test removes the softmax fact and checks that the O-projection key range grows to the product bound.
- Determinism: compiling twice yields identical claim-list, plan and table-manifest bytes.
- Report: for the `minimal` profile at 405B shapes, requantized elements per layer per token equal 137,216, matching the source; committed elements are reported separately with split limbs.
- Golden claim lists for the tiny conformance models in `spec/vectors/claims/`.

## Open Questions

- **OQ-0006-1.** Is `B_max = 32` the right default, against 30 or 34? Larger tables cost prover memory and table-side time per proof; smaller ones force more split gadgets. Owner: @AbdelStark. Resolve at the v0.2 gate using the 405B commitment report and the v0.1 unit costs.

## References

- Source §5, §6 principles 2–4 and 8, §7 ("Typed integer dataflow", "Integer model D").
- [RFC-0005](RFC-0005-integer-model.md), [RFC-0007](RFC-0007-claim-list.md), [RFC-0008](RFC-0008-logup-and-tables.md), [RFC-0009](RFC-0009-gadgets.md), [RFC-0010](RFC-0010-proof-schedule.md).
