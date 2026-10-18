# RFC-0006: Compiler IR: Typed Integer Dataflow, Range Analysis, and Gadget Selection

- Status: Accepted
- Authors: @AbdelStark
- Created: 2026-10-15
- Target milestone: v0.2
- Amended by: [RFC-0022](RFC-0022-saturating-residual.md) ('Range analysis': `RsqrtRow`'s range and domain; 'Llama lowering': the residual), [RFC-0024](RFC-0024-wide-requantization.md) ('Gadget selection': `RequantWide`), [RFC-0025](RFC-0025-affine-sites.md) ('Operation set': `Mul`'s broadcast; 'Llama lowering': affine sites), [RFC-0028](RFC-0028-balanced-split.md) ('Gadget selection': the split point), [RFC-0029](RFC-0029-table-set.md) ('Outputs': the claim list is the table set), [RFC-0030](RFC-0030-masked-exponentials.md) ('Range analysis': `SoftmaxRow`'s table width and the softmax fact's ground), [RFC-0034](RFC-0034-causal-blocks.md) ('Tiling and the execution plan': `e`'s order within an attention tile)

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
    Sum { axis: u8 }, Reshape, Slice { axis: u8, start: u32, len: u32 },
    GatherPublic { index: ConstRef },
    Transpose { perm: Vec<u8> }, Expand { axis: u8, len: u32 },   // head layouts, RFC-0037
    // Virtual, degree 2
    MatMulAct { transpose_rhs: bool },           // activation x activation: QK^T (transposed), PV
    Mul { broadcast: Broadcast },                // elementwise product: Trailing or Leading
    SumSquares { axis: u8 },
    // Gadget-backed: produce committed advice
    OneHot { vocab: u32 },                       // committed one-hot token vector
    Requant { shift: ShiftClassRef, lo: i64, hi: i64 },
    TableMap { table: TableId },                 // SiLU and other single-input nonlinearities
    SoftmaxRow { exp_table: TableId, sigma: u8 },// max certificate + reciprocal bracket
    RsqrtRow { kappa: u8, eps_int: u64, dim: u32 }, // multi-limb bracket over rows of dim
    Argmax, InverseCdf { prf: PrfRef },          // scoring certificates (RFC-0017)
    Bitwise { table: TableId },                  // AES and SHA-256 sub-operations (RFC-0015)
}

pub struct Node { pub id: NodeId, pub op: Op, pub inputs: Vec<TensorId>, pub outputs: Vec<TensorId> }
pub struct Graph {                               // TensorSpec: 03 §3
    pub nodes: Vec<Node>, pub tensors: Vec<TensorSpec>,
    pub constants: Vec<Vec<i64>>,                // by ConstRef: a scalar or one value per channel
    pub shift_classes: Vec<Vec<u8>>,             // by ShiftClassRef: r_c per class, 1 to 4 classes
    pub weights: Vec<(WeightRef, TensorId)>,     // WeightRef = (WeightType, layer)
}
```

Node kinds follow from the operation: linear and degree-2 operations produce `Virtual` tensors, except that one other than a matmul may produce a `Committed` tensor, which stage 1 proves equal to its relation by a zero-check ([RFC-0037](RFC-0037-stage-schedule.md#the-committed-first-checkpoint)); gadget-backed operations produce `Committed` or `LookupOutput` tensors plus the gadget's internal advice (limbs, multiplicities), which stays inside the gadget. Weights are `Committed { Weight }`; constants and shapes are `Public`.

A graph's nodes are in topological order. Every input is a graph input, a tensor no node produces, which is `Public` or `Committed`, or the output of an earlier node. No tensor has two producers. Ids are dense in creation order, so the graph is a pure function of the calls that build it (INV-DM-5). A node keeps the rules of its operation. Shapes have at least one axis, each of length at least 1. `[…, n]` is any shape whose last axis is `n`, and removing a tensor's only axis leaves `[1]`.

| Operation | Inputs → outputs | Shapes | Outputs and inputs |
|---|---|---|---|
| `MatMulWeight` | `x`, `W` → `y` | `W [out, in]`, `x […, in]`, `y […, out]` | `W` is the weight tensor its `WeightRef` declares |
| `Add`, `Sub` | `a`, `b` → `y` | all equal | |
| `ConstMul`, `ConstAdd` | `x` → `y` | `y = x`; the constant has 1 value or `x`'s last axis | |
| `Rotate` | `x`, `pos` → `y` | `x [n, …, 2m]`, `pos [n]`, `y = x` | `pos` is `Public`; the pair `(x_i, x_{i+m})` of each last-axis vector turns by its row's position at frequency `i`, Llama's `rotate_half` layout |
| `Sum`, `SumSquares` | `x` → `y` | `y` is `x` without the axis | |
| `Reshape` | `x` → `y` | equal element counts | |
| `Slice` | `x` → `y` | `start + len` within the axis, `len ≥ 1`; `y` is `x` with that axis of length `len` | |
| `GatherPublic` | `x` → `y` | each index a row of `x`'s outermost axis; `y` has one such row per index | |
| `Transpose` | `x` → `y` | `y`'s axis `i` is `x`'s axis `perm[i]`, `perm` a permutation of `x`'s axes | |
| `Expand` | `x` → `y` | `y` is `x` with an axis of length `len ≥ 1` inserted before axis `axis ≤ rank`, `x` repeated along it | |
| `MatMulAct` | `a`, `b` → `y` | equal leading axes; `a […, m, k]`, `b […, k, n]` or, transposed, `b […, n, k]`; `y […, m, n]` | |
| `Mul` | `a`, `b` → `y` | with `Trailing`, `b`'s shape is `a`'s or a prefix of it, broadcast over the trailing axes; with `Leading`, `a`'s last axis, broadcast over the leading axes; `y = a` | The broadcast is stated, not inferred: when `a`'s first and last axes have equal lengths, a `b` of that length fits both |
| `OneHot` | `t` → `h` | `h` is `t` with an axis of `vocab ≥ 1` added | `h` is `Committed { Advice }` |
| `Requant` | `x` (, `c`) → `y` | `y = x`; `c [x's last axis]`, the channel classes, when the set has more than one class | `c` is `Committed { Weight }`; `y` is a lookup output or advice; `lo ≤ hi` |
| `TableMap` | `x` (, `c`) → `y` | `y = x`; `c [x's last axis]`, the channel classes, when the table has several classes | `c` is `Committed { Weight }`; `y` is `LookupOutput { table }` |
| `SoftmaxRow` | `s` → `e`, `R` | `s […, n, n]`, its last two axes the query and key positions of one sequence; `e = s`; `R` is `s` without its last axis | `e` is `LookupOutput { exp_table }`, `R` advice; row `q` sees keys `i ≤ q`, and `e` is 0 at the others |
| `RsqrtRow` | `u` → `ρ` | `ρ = u` | `ρ` is advice; `dim ≥ 1`, and `κ ≤ 31`, so that `N = dim · 2^{2κ}` stays below `2^{94}` |
| `Argmax`, `InverseCdf` | `ℓ`, `h` → none | `h = ℓ`: one one-hot entry per logit | certificates: every value they commit is internal advice |
| `Bitwise` | `a`, `b` → `y` | all equal | `y` is `LookupOutput { table }` |

The IR is the prover's (tier C). The verifier re-derives what it checks from the claim list with a representation of its own (Alternative 2), so range analysis, gadget selection and the lowering may refine these rules before the protocol version is fixed.

### Range analysis

- Ranges are exact integer intervals computed with `i128` endpoints; an interval that would leave `i128` is marked unbounded.
- Inputs: token ids in `[0, vocab)`; positions in `[0, n)` for a sequence of `n` tokens ([RFC-0044](RFC-0044-lowering-nodes.md#graph-inputs)); weights in `[−127, 127]`; activations of each type over their full type range. Ranges hold for every admissible input, never for calibration statistics.
- Transfer functions: interval arithmetic for `Add`, `Sub`, `ConstMul` and `ConstAdd` (over the constant's least and greatest values), `Sum` (scaled by axis length), `MatMulWeight` (using only the public bound `max|W| · fan_in · max|x|`, `max|W| = 127`, never the actual weight values: bounds derived from secret weights would leak them through the claim list and could not be re-derived by the verifier), `Rotate` (each output is `a·C − b·S` or `a·S + b·C` with `|C|, |S| ≤ 2^bits` of its `Rope` table, so at most `2^{bits+1} · max|x|`), `Mul`, `SumSquares` (0 at least when the range holds 0), `MatMulAct` (the products' interval, scaled by the inner length). `Reshape`, `Slice` and `GatherPublic` keep their input's range.
- Gadget outputs take the range their gadget's lookups enforce (RFC-0009): `[lo, hi]` after `Requant`; a table's values after `TableMap` and `Bitwise` (`[−128, 127]` for `Silu`, `[0, 2^σ]` for `Exp`, `[0, 255]` for the byte tables); `[0, 1]` after `OneHot`; `e` in the `Exp` table's values and `R ∈ [0, 2^{σ+1})` after `SoftmaxRow`; `ρ ∈ [0, 2^w)` after `RsqrtRow`, with `w` the bit length of `⌊√(d · 2^{2κ} / e)⌋`, the largest root of any row ([RFC-0022](RFC-0022-saturating-residual.md#complete-norms)).
- **Relational facts.** Gadgets may emit facts that tighten downstream bounds. v1 has two. `SoftmaxRow` emits `R_q · Σ_i e_qi ≤ 2^{2σ}`, so `R_q · Σ_i e_qi v_i` is bounded by `2^{2σ} · max|v|` rather than by the product of the individual bounds. Each fact has a proof obligation in RFC-0009 and is re-derived by the verifier. The fact applies to a `Mul(MatMulAct(e, v), R)` node, untransposed, whose `e` and `R` are the two outputs of one `SoftmaxRow`, as the lowering builds the attention output. Its range is then `[2^{2σ} · min(v_lo, 0), 2^{2σ} · max(v_hi, 0)]`, since `e ≥ 0` and `R ≥ 0`: for INT8 values and `σ = 16`, RFC-0005's `[−2^39, 127 · 2^32]`. The fact sums every key position. It holds because the exponentials are 0 at masked positions, which `SoftmaxRow`'s lookups enforce: a masked position looks up its table's last key, which the table maps to 0 ([RFC-0030](RFC-0030-masked-exponentials.md#what-reads-e)). `OneHot` emits that its vectors hold exactly one 1 (RFC-0009 Lemma 8), so a `MatMulWeight` whose input is a `OneHot` output selects a column of its weight and takes the weight's range: the embedding rows stay INT8, `[−127, 127]`, where the public bound would be `127 · vocab`.

**True-integer rule.** A tensor used as a lookup key or a gadget input that requires an integer must have a bounded range inside `(−(P−1)/2, (P−1)/2)` and, for keys, inside its table's key domain. Intermediate virtual tensors may exceed that range: they are computed modulo `P`, and because every key is an integer-linear or polynomial function of them whose own exact range is bounded, the key's field value equals its true integer value. The analysis therefore never needs to track wraps explicitly; it checks bounds only where integrality matters.

The integrality-requiring inputs, each with the domain it must lie in (every domain lies inside the centered field range):

| Gadget | Input | Domain |
|---|---|---|
| `OneHot` | tokens | `[0, vocab)` |
| `Requant` | key; channel classes | the centered field range; `[0, classes)` |
| `TableMap` | key; channel classes | the table's keys: `[−2^{b−1}, 2^{b−1})` for `Silu`, `[0, 2^{key_bits})` for `Exp`; `[0, classes)`, given exactly when the table has several classes (otherwise `ERR-INTERNAL`) |
| `Bitwise` | both operands | `[0, 256)` |
| `SoftmaxRow` | scores | `[−2^21, 2^21]`, with a spread `x_hi − x_lo` below `2^{key_bits}` of its `Exp` table, since each key is the row maximum less a score. INT8 heads of width 128 reach `2^21` at `q = k = −128`. `key_bits` is the narrowest width from `bits(128 · 255 · d_head)` up whose last key maps to 0, the key every masked position looks up ([RFC-0030](RFC-0030-masked-exponentials.md#relation-1-every-position-looked-up)). |
| `RsqrtRow` | sums of squares | `u ≥ 0` and `u + e < 2^55`; the node needs `e ≥ 1`, and `w ≤ 32` for its range check to be one lookup |
| `Argmax`, `InverseCdf` | logits; one-hot entries | the centered field range; `[0, 1]` |

Violations: an integrality-requiring input with an unbounded range raises `ERR-COMPILE-WRAP`; one whose range leaves its domain raises `ERR-COMPILE-RANGE`, and so does a key outside every feasible gadget's bound (gadget selection).

### Gadget selection

Every gadget-backed node takes a gadget, numbered in node order (`GadgetId`). `B_max` is `GadgetPolicy.max_log2_rows`, from 1 to 32 and 32 by default ([RFC-0007](RFC-0007-claim-list.md#schema)).

A `Requant`'s gadget depends on its key's signed width `b`, from range analysis, and on its `n` shift classes `r_c`. The class index takes `⌈log2 n⌉` of a table's key bits ([RFC-0008](RFC-0008-logup-and-tables.md#table-kinds) 'Requant'):

| Condition | Gadget | Tables | Cost per element |
|---|---|---|---|
| `b + ⌈log2 n⌉ ≤ B_max` | `Requant1` | `Requant { b, r_c, lo, hi }` | 1 commitment, 1 lookup |
| otherwise, with `b ≤ 62` and `b + ⌈log2 n⌉ − B_max ≤ min(r_min − 1, B_max)`: `s = clamp(⌈(b + ⌈log2 n⌉)/2⌉, b + ⌈log2 n⌉ − B_max, min(r_min − 1, B_max))` ([RFC-0028](RFC-0028-balanced-split.md)) | `RequantSplit` with split point `s` | `Range { s }`, `Requant { b − s, r_c − s, lo, hi }` | 2 commitments, 2 lookups |
| otherwise, with `b ≤ 62`, `t + 2 + ⌈log2 n⌉ ≤ B_max` and `k + 1 ≤ B_max` | `RequantWide` with low part `s` and middle part `t` | `Range { s }` when `s ≥ 1`, `Range { t }`, `Range { k }`, `Requant { k + 1, [1], −2, 1 }`, `Requant { t + 2, r_c − s, lo, hi }` | 4 commitments, 5 lookups |
| otherwise | `ERR-COMPILE-RANGE` | — | — |

The split's conditions come from RFC-0009 Lemma 2:

- the high part's table fits: `s ≥ b + ⌈log2 n⌉ − B_max`;
- the low part's table fits: `s ≤ B_max`;
- the halves are true integers: `b ≤ 62`;
- the nested floor is the requantization: `s ≤ r_c − 1`, since `2^{r_c − 1}` must be a multiple of `2^s`.

Between these bounds the split takes the point that balances the two tables' rows, `2^s + 2^{b + ⌈log2 n⌉ − s}`, the extra bit in `Range { s }` on a tie ([RFC-0028](RFC-0028-balanced-split.md)).

A key the split cannot take, because the split would pass the least shift's rounding bit or its low part would not fit, takes `RequantWide` ([RFC-0024](RFC-0024-wide-requantization.md)). It has three parts:

- a low part of `s = min(r_min − 1, B_max)` bits, below every class's rounding bit;
- a middle part of `t = (r_max − s) + ⌈log2 M⌉` bits, `M = max(|lo|, |hi|, 1)`, past which every class saturates;
- a top part of `k = b − s − t` bits, which enters only through its saturation `clamp(h1, −2, 1)`.

Its conditions come from RFC-0009 Lemma 11: `b ≤ 62`, and its reduced and top tables fit, `t + 2 + ⌈log2 n⌉ ≤ B_max` and `k + 1 ≤ B_max`. The first gadget that applies is the cheapest.

With one class at `B_max = 32`, keys of up to 32 bits take `Requant1`. Keys of 33 to 62 bits split, or take `RequantWide` when their shift is too small for the split. A `TableMap`, `Bitwise` or `SoftmaxRow` reads its own table once, which must hold at most `2^{B_max}` rows: these lookups have no split.

`B_max = 32` is the largest table whose prover-side cost is amortized over a 16K-token batch ([08 §4](../spec/08-performance-budget.md#4-memory-budgets)). The verifier's cost to evaluate a table does not depend on `b`. Every v1 table is a piecewise function whose non-constant core has at most 2^24 entries; for a clamped requantization the core is the output range, 256 values for INT8. Its multilinear extension is evaluated in time proportional to the core ([RFC-0008](RFC-0008-logup-and-tables.md#structured-evaluation)).

Tables with identical definitions are shared across layers and channels: a requantization table is identified by its kind and `(b, r_c, lo, hi)`. New tables are numbered after the lowering's, in order of first use.

### Llama lowering

The lowering instantiates the profile's template from [RFC-0005](RFC-0005-integer-model.md#integer-semantics) per layer: embedding (`OneHot` then `MatMulWeight` on `Embed`), attention norm (`SumSquares`, `RsqrtRow`, `Mul`, `Requant`), Q/K/V (`MatMulWeight`), RoPE (`Rotate`, `Requant`), V (`Requant`), attention (`MatMulAct`, `SoftmaxRow`, `MatMulAct`, `Mul` by `R_q`), O projection, saturating residual (`Requant`), MLP norm, gate and up, `TableMap` (SiLU), product and down projection per profile, saturating residual, final norm, LM head. Scoring nodes (RFC-0017) consume the logits; anchor circuits (RFC-0015) consume the token streams that also feed `OneHot`.

It reads only the public part of `D`: the architecture, the profile, `PublicQuantSpec` and each layer's `Silu` input exponent. The graph is therefore what the verifier re-derives from the claim list (INV-IR-5). One graph covers one sequence of `n ≤ max_context` tokens at positions `0` to `n − 1`.

- **Inputs.** The tokens `[n]`, committed advice in `[0, vocab)`; the positions `[n]`, public; each weight matrix; for each site of several classes, the classes of the weight before it, `[rows]` in `[0, classes)`; and for each affine site, the multipliers of the weight before it, `[rows]` in `[0, 2^{16} − 1]` ([RFC-0025](RFC-0025-affine-sites.md)). Classes and multipliers are committed with the weights.
- **Layout.** Activations are position-major, `[n, channels]`. RoPE rotates Q and K as `[n, heads, d_head]`, pairing `x_i` with `x_{i + d_head/2}`.
  - Attention runs head-major. `Transpose` takes Q, K and V from `[n, heads, d_head]` to `[heads, n, d_head]`, and `Expand` then `Reshape` give K and V `[n_heads, n, d_head]`, with query head `h` reading KV head `h / (n_heads / n_kv_heads)` ([RFC-0037](RFC-0037-stage-schedule.md#head-layouts)).
  - One `MatMulAct` then forms every head's scores `[n_heads, n, n]`, one causal `SoftmaxRow` their exponentials and reciprocals, and `Mul(MatMulAct(e, v), R)` the attention output: the softmax fact's pattern.
  - A last `Transpose` and `Reshape` return the output to `[n, d_model]`.
- **Rescales.** Each `Requant` names its site's shift classes and takes the channel classes when there are several, as does the `Silu` lookup. An affine site's key is `Mul(x, m)`, with `x` what a `pot` site would rescale and `m` its multipliers, rescaled with its one shift; at a residual site `Mul(acc, m)` takes the place of `acc`. INT8 sites saturate to `[−128, 127]`. The embedding enters the residual by `ConstMul` with `2^{s_e}`, `s_e = e_Embed − residual_exponent` ([03 §2](../spec/03-data-model.md#quantization)), whose output, the first checkpoint, is committed advice ([RFC-0037](RFC-0037-stage-schedule.md#the-committed-first-checkpoint)).
- **The residual** saturates its sum within `[−2^{19}, 2^{19} − 1]` ([RFC-0022](RFC-0022-saturating-residual.md#saturating-residual)):
  - a site of one class, whose `2^r` fits a constant (`r ≤ 62`), rescales the fused key `Add(acc, ConstMul(res, 2^r))` with its shift;
  - a site of several classes rescales `acc` with its classes within `±(2^{20} − 1)`, then clamps `2(res + u)` with the shift 1.

  Either way, the residual after an update is a `Requant` output.
- **Tables.** Table 0 is RoPE's. Each layer then names an `Exp` table for its score scale and a `Silu` table, numbered in order of first use; a definition met before reuses its table.
  - Key widths come from public bounds. `key_bits` is the bit length of the scores' spread `128 · 255 · d_head`, 22 at `d_head = 128`.
  - `b` is the least width with `2^{b−1} > 127 · 128 · d_model`, the gate accumulator's bound: 27 at 8B.
- **Checks.** Range analysis must accept the graph. Then the graph must keep its profile's constraints, read on the forward pass with its ranges (`ERR-COMPILE-PROFILE`):
  - S1: every weight matmul's operands are INT8, except, in `minimal`, the O and down projections';
  - S2 (`servable`): every weight matmul's accumulator fits INT32;
  - S3: every `TableMap` applies a `Silu` table, and no scoring or anchor node is part of `D`.

  Every tensor then carries its analyzed range and the narrowest integer type holding it.
- **Validation** (`ERR-QUANT-ARCH`). The lowering refuses a public part that breaks RFC-0007's architecture rules (`d_model = n_heads · d_head`, `n_kv_heads` dividing `n_heads`, `vocab ≤ 2^18`, `max_context ≤ 2^17`). It also refuses:
  - an odd `d_head`;
  - quantization data that misses or repeats a point of the profile, or a site in both `shift_class_sets` and `affine_sites`;
  - a site after no weight with several classes;
  - an affine site that follows no Q, K, V, O, up or down projection;
  - a shift outside `[1, 63]`, or a score scale that is not positive;
  - an embedding shift that would push INT8 rows out of the residual;
  - `e = 0`, or `κ > 31`;
  - a table RFC-0008 refuses.

  The prover also checks the hidden part before lowering:
  - one weight per type and layer, in `(ty, layer)` order, with its architecture's shape and entries in `[−127, 127]`;
  - a `ChannelParams` for each weight a site of several classes or an affine site rescales and for no other;
  - at a site of several classes, one class per output channel, each inside the site's classes, and no multiplier;
  - at an affine site, one multiplier per output channel, each in `[1, 2^{16})`, and no class.

### Tiling and the execution plan

- Linear tiles are `(layer, operation)`; attention tiles are `(layer, kv_head, query_block)` with query blocks of 512 positions by default. [RFC-0041](RFC-0041-execution-plan.md#tiles) defines what each holds, and the tiles of the tokens, the prologue, the epilogue and the tables.
- Within an attention tile, `e`'s variables are ordered `(i_hi, h, q, i_lo)`, so that each advice row holds a block of keys against the tile's queries and heads, and the blocks above the diagonal are zero rows ([RFC-0034](RFC-0034-causal-blocks.md#causal-layout)).
- Each tile's committed tensors occupy a contiguous range of advice rows (RFC-0011), in [RFC-0041](RFC-0041-execution-plan.md#production-order)'s production order: node order, an attention node's tiles by KV head, then query block, and the tables' multiplicities last.
- The `ExecutionPlan` (schema `0x0202`, private) holds the claim list and one sequence's tiles ([RFC-0041](RFC-0041-execution-plan.md#the-plan)). Row ranges are the advice layout's (RFC-0011), the stage schedule is derived (RFC-0037), and devices are assigned at prove time (RFC-0018). It contains no secret values.

### Outputs

```rust
pub fn compile(model: &IntegerModel, params: &ParamSet, policy: &GadgetPolicy)
    -> Result<CompileOutput, CompileError>;

pub struct CompileOutput {
    pub claim_list: ClaimList,          // public, RFC-0007; its tables are the table set (RFC-0029)
    pub plan: ExecutionPlan,            // private
    pub report: CommitmentReport,       // public JSON, tessera.commitment-report/1
}
```

`CommitmentReport` gives, per layer per token and in total for a reference batch: requantized elements by tensor; committed elements by class (lookup outputs, split limbs, per-row scalars, one-hot vectors, multiplicities, masks); lookups by table; product-tree leaves; table row counts; stage count; and the cost-model projection of `μ` at 1K, 4K, 16K and 128K using the current unit costs ([RFC-0020](RFC-0020-cost-model.md#executable-cost-model)). The opening groups `G` and the masks follow from the claim points of the stage schedule, and join the report with it ([#113](https://github.com/AbdelStark/tessera/issues/113), [#149](https://github.com/AbdelStark/tessera/issues/149), [#154](https://github.com/AbdelStark/tessera/issues/154)).

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
- Lowering: seeded tiny models are lowered in both profiles, with grouped-query attention and sites of one to four classes. An exact interpreter of the IR runs each graph to the logits of an independent token-by-token implementation of RFC-0005's semantics, and every value stays inside its tensor's range. Llama-3.1 8B's and 405B's shapes lower from their public part alone.
- Determinism: compiling twice yields identical claim-list and plan bytes.
- Report: for the `minimal` profile at 405B shapes, requantized elements per layer per token equal 137,216, matching the source; committed elements are reported separately with split limbs.
- Golden claim lists for the tiny conformance models in `spec/vectors/claims/`.

## Open Questions

- **OQ-0006-1** (resolved). Is `B_max = 32` the right default, against 30 or 34? Larger tables cost prover memory and table-side time per proof; smaller ones force more split gadgets. Yes, by the 405B commitment reports ([`docs/perf/v0.2/`](../perf/v0.2/README.md), [#88](https://github.com/AbdelStark/tessera/issues/88)).
  - At 30, three of six class configurations do not compile, and the others cost 0.24 to 0.62 more of `μ(1K)` than at 32, against at most 0.13 of table side saved.
  - 34 lies outside RFC-0008's tables. It would gain about 0.2 of `μ(1K)`, and only with three or four classes at the down projection.

## References

- Source §5, §6 principles 2–4 and 8, §7 ("Typed integer dataflow", "Integer model D").
- [RFC-0005](RFC-0005-integer-model.md), [RFC-0007](RFC-0007-claim-list.md), [RFC-0008](RFC-0008-logup-and-tables.md), [RFC-0009](RFC-0009-gadgets.md), [RFC-0010](RFC-0010-proof-schedule.md).
