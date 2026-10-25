# RFC-0044: The Llama Lowering, Node by Node

- Status: Accepted
- Authors: @AbdelStark
- Created: 2026-10-18
- Target milestone: v0.3
- Amended by: [RFC-0071](RFC-0071-request-slots.md) ('Graph inputs' and 'A layer': a batch's token axis of `N` rows, its positions and its scores)
- Supersedes, on acceptance: [RFC-0006](RFC-0006-compiler-ir.md#llama-lowering) 'Llama lowering': the graph's nodes, inputs, constants, shift-class sets and tables are those this RFC lists, in its order.
- Origin: [#362](https://github.com/AbdelStark/tessera/issues/362)

## Summary

The verifier lowers the claim list into its own graph with code written from the specification alone (RFC-0001, RFC-0007 step 5), and that graph must equal the compiler's node for node:
- RFC-0041's production order follows node order, and the advice layout follows it;
- gadget ids are numbered in node order (RFC-0006 'Gadget selection');
- the derived tables must equal the claim list's, id for id (RFC-0007 step 5.3).

RFC-0006 describes the lowering in prose, and `tessera-compiler` was its only complete statement. This RFC states it node by node: every graph input, node, constant, shift-class set and table, in creation order, for both profiles and every form of site.

## Motivation

[RFC-0006](RFC-0006-compiler-ir.md#llama-lowering) 'Llama lowering' names each part's operations: the norm, the projections, RoPE, the attention, the residual's routes, the MLP. It does not say:
- the order of a layer's three projections and their rescales;
- the `Reshape`, `Transpose` and `Expand` chains of the head layouts;
- where an affine site's `Mul` and a classed site's class vector come in;
- which constants and shift-class sets each node creates;
- each output's shape and kind.

An independent implementation could not reproduce the graph's ids from it. A difference of one node moves every later tensor's rows and fails `layout_digest` (RFC-0031), so the verifier would reject every honest proof.

## Goals

- Every id of the lowered graph determined by the claim list and the sequence's length: tensors, nodes, constants, shift-class sets, tables and gadgets.
- A statement an independent implementation can follow without reading the compiler.

## Non-Goals

- Ranges and types. Every output's range is range analysis's (RFC-0006 'Range analysis', #117). This RFC gives each output's shape and kind; its type and range are the analysis's.
- `ConstraintSystemDigest`, which #120 defines on top of this graph.
- Gadget selection, which RFC-0006 and RFC-0024 define and which numbers its own tables after the lowering's.

## Proposed Design

### Notation

`n` is the sequence's length. `d = d_model`, `h = n_heads`, `kv = n_kv_heads`, `g = h / kv`, `dh = d_head`, `ff = d_ff`, `V = vocab`, `b = residual_bits`, and `σ = exp_sigma`.

Kinds:
- **A**: `Committed { Advice }`;
- **W**: `Committed { Weight }`;
- **P**: `Public`;
- **L(t)**: `LookupOutput { table: t }`;
- **v**: `Virtual`.

`p` is a site's `QuantPoint`. "The weight before `p`" is RFC-0005's: Q at `RopeQ`, K at `RopeK`, V at `V`, O at `AttnResidual`, Gate at `Silu`, Up at `Up`, Down at `MlpResidual`.

### Ids

- **Tensors and nodes.** Every tensor has a dense id, in creation order. A graph input is created when this RFC names it, and a node's outputs are created with the node, in output order. Nodes have dense ids of their own, in creation order.
- **Constants and shift-class sets** are dense lists in creation order. Each mention below creates a new entry, even one equal to an earlier entry: neither list shares entries.
- **Tables.** Table 0 is RoPE's: `Rope` with the architecture's `RopeSpec`, `max_context` and `d_head`. Each `Exp` and `Silu` table named below takes the next id when its definition is new, and the earlier table's id otherwise. Gadget selection's tables follow (RFC-0006 'Gadget selection').
- **Gadgets** are numbered in node order over the gadget-backed nodes (RFC-0006).

### Graph inputs

The lowering starts with two inputs:

1. `tokens`: `[n]`, A, values in `[0, V − 1]`;
2. `positions`: `[n]`, P, values in `[0, n − 1]`.

A batch of several requests ([RFC-0071](RFC-0071-request-slots.md#the-lowering)) has its token axis of `N` rows in place of `n`, here and below. Its positions are each real row's place in its request, and 0 at a padded row. Its scores are `[n_heads, N, n_max]`, each query reading the keys and values of its own request. With one request, `N = n_max = n`.

An input's type is the narrowest unsigned one that holds its values, of at least one bit: `u1` or `u2` for a class vector, `u16` for a multiplier vector. A weight's is `i8`. Every other input is created by a procedure below, when it is first named:
- weight `W_{T,ℓ}`: `[rows, cols]`, W, values in `[−127, 127]`, appended to the graph's weight list as `(T, ℓ)`, so that the list is in creation order;
- class vector: `[c]`, W, values in `[0, k − 1]` for a site of `k` classes;
- multiplier vector: `[c]`, W, values in `[0, 2^{16} − 1]`.

### Procedures

Each procedure lists what it creates, in order. Outputs are `[shape] kind`.

- **`matmul(T, ℓ, x, rows, cols)`**
  1. input `W_{T,ℓ}` `[rows, cols]`;
  2. `MatMulWeight { weight: (T, ℓ) }(x, W_{T,ℓ})` → `[n, rows]` v.
- **`key(x, p, c)`**: what the site `p` rescales of `x` `[n, c]`, with its shifts and its class vector.
  - **Affine site**, shift `r` ([RFC-0025](RFC-0025-affine-sites.md)):
    1. the multiplier vector `m` of the weight before `p`, `[c]`;
    2. `Mul { broadcast: Leading }(x, m)` → `[n, c]` v.

    It gives that output, the shifts `[r]`, and no class vector.
  - **`pot` site of one class `r`**: creates nothing, and gives `x`, `[r]`, and no class vector.
  - **`pot` site of `k ≥ 2` classes**: creates the class vector `κ` of the weight before `p`, `[c]`, and gives `x`, the site's shifts, and `κ`.
- **`requant(z, shifts, κ, c, lo, hi)`**
  1. a shift-class set `s = shifts`;
  2. `Requant { shift: s, lo, hi }(z)`, or `(z, κ)` when there is a class vector, → `[n, c]` A.
- **`rescale(x, p, c)`**: `key(x, p, c)`, then `requant` of its key, shifts and class vector over `c` channels with `lo = −128`, `hi = 127`. Its output is the site `p`'s.
- **`norm(res, p)`**
  1. `SumSquares { axis: 1 }(res)` → `[n]` v;
  2. `RsqrtRow { kappa: rsqrt_kappa, eps_int: rms_eps_int, dim: d }(·)` → `[n]` A;
  3. `Mul { broadcast: Trailing }(res, ρ)` → `[n, d]` v, `ρ` the `RsqrtRow`'s output;
  4. `rescale(·, p, d)`.
- **`rope(x, k)`**, for `k` heads:
  1. `Reshape(x)` → `[n, k, dh]` v;
  2. `Rotate { rope: 0 }(·, positions)` → `[n, k, dh]` v;
  3. `Reshape(·)` → `[n, k·dh]` v.
- **`heads(x, k)`**, for `k` heads:
  1. `Reshape(x)` → `[n, k, dh]` v;
  2. `Transpose { perm: [1, 0, 2] }(·)` → `[k, n, dh]` v. When `k = h`, this is the result.
  3. otherwise `Expand { axis: 1, len: g }(·)` → `[k, g, n, dh]` v;
  4. and `Reshape(·)` → `[h, n, dh]` v.
- **`residual(acc, res, p)`**: let `(z, shifts, κ) = key(acc, p, d)`. The site takes one route in every layer ([RFC-0038](RFC-0038-uniform-layers.md#the-residuals-route)):
  - **The fused key**, when in every layer the site has one shift `r ≤ 62`, of one class or affine:
    1. a constant `[2^r]`;
    2. `ConstMul { c }(res)` → `[n, d]` v;
    3. `Add(z, ·)` → `[n, d]` v;
    4. a shift-class set `[r]`;
    5. `Requant { shift, lo: −2^{b−1}, hi: 2^{b−1} − 1 }(·)` → `[n, d]` A.
  - **Two steps**, otherwise:
    1. `u = requant(z, shifts, κ, d, 1 − 2^b, 2^b − 1)`;
    2. `Add(res, u)` → `[n, d]` v;
    3. a constant `[2]`;
    4. `ConstMul { c }(·)` → `[n, d]` v;
    5. a shift-class set `[1]`;
    6. `Requant { shift, lo: −2^{b−1}, hi: 2^{b−1} − 1 }(·)` → `[n, d]` A.

  The last `Requant`'s output is the site `p`'s, and the residual after it.

### The prologue

1. `OneHot { vocab: V }(tokens)` → `[n, V]` A;
2. `matmul(Embed, 0, ·, d, V)`;
3. a constant `[2^{s_e}]`, where `s_e = e_Embed − residual_exponent` (03 §2);
4. `ConstMul { c }(·)` → `[n, d]` A, the first checkpoint `c_0` ([RFC-0037](RFC-0037-stage-schedule.md#the-committed-first-checkpoint)).

### A layer

Layer `ℓ`, from the residual `res`. `p(site)` is `QuantPoint::Layer { layer: ℓ, site }`.

1. `y = norm(res, p(AttnNorm))`.
2. The projections, all reading `y`:
   - `q = matmul(Q, ℓ, y, h·dh, d)`;
   - `k = matmul(K, ℓ, y, kv·dh, d)`;
   - `v = matmul(V, ℓ, y, kv·dh, d)`.
3. `q = rope(q, h)`, then `k = rope(k, kv)`.
4. The rescales:
   - `q = rescale(q, p(RopeQ), h·dh)`;
   - `k = rescale(k, p(RopeK), kv·dh)`;
   - `v = rescale(v, p(V), kv·dh)`.
5. The attention:
   1. `heads(q, h)`, then `heads(k, kv)`, then `heads(v, kv)`;
   2. `MatMulAct { transpose_rhs: true }(q', k')` → `[h, n, n]` v, the scores;
   3. the layer's `Exp` table (below);
   4. `SoftmaxRow { exp_table: t, sigma: σ }(scores)` → `[h, n, n]` L(t), the exponentials `e`, and `[h, n]` A, the reciprocals `R`;
   5. `MatMulAct { transpose_rhs: false }(e, v')` → `[h, n, dh]` v;
   6. `Mul { broadcast: Trailing }(·, R)` → `[h, n, dh]` v;
   7. `Transpose { perm: [1, 0, 2] }(·)` → `[n, h, dh]` v;
   8. `Reshape(·)` → `[n, h·dh]` v, the output `o`.
6. In `servable` only, `o = rescale(o, p(AttnOut), h·dh)`.
7. `acc = matmul(O, ℓ, o, d, h·dh)`, then `res = residual(acc, res, p(AttnResidual))`.
8. `y = norm(res, p(MlpNorm))`.
9. `gate = matmul(Gate, ℓ, y, ff, d)`, then `up = matmul(Up, ℓ, y, ff, d)`.
10. SiLU:
    1. the layer's `Silu` table `t` (below);
    2. the class vector of `Gate`, `[ff]`, when `p(Silu)` has several classes;
    3. `TableMap { table: t }(gate)`, or `(gate, κ)`, → `[n, ff]` L(t), the output `a`.
11. The product:
    - **`servable`**:
      1. `u = rescale(up, p(Up), ff)`;
      2. `Mul { broadcast: Trailing }(a, u)` → `[n, ff]` v;
      3. `rescale(·, p(Product), ff)`.
    - **`minimal`**: `Mul { broadcast: Trailing }(a, up)` → `[n, ff]` v.
12. `acc = matmul(Down, ℓ, ·, d, ff)`, then `res = residual(acc, res, p(MlpResidual))`. This is the layer's output, the checkpoint `c_{ℓ+1}`.

### The epilogue

1. `y = norm(res, FinalNorm)`;
2. `matmul(LmHead, 0, y, V, d)` → `[n, V]` v, the logits.

### The tables

- **`Exp`.** Let `w = bitlen(128 · 255 · dh)`, and `scale` the layer's score scale. For each `key_bits` from `w` up to 32, in order, take the definition `Exp { key_bits, sigma: σ, scale }`:
  - if an earlier table has that definition, the layer takes it, and the search ends;
  - otherwise, if that table maps its last key `2^{key_bits} − 1` to 0, it is a new table, and the search ends ([RFC-0030](RFC-0030-masked-exponentials.md#range-analysis-and-validation)).

  The search ends at the first width whose table maps its last key to 0. [RFC-0007](RFC-0007-claim-list.md#verifier-validation) step 2 guarantees that width exists, and that [RFC-0008](RFC-0008-logup-and-tables.md#table-specifications) allows its table.
- **`Silu`.** The definition is `Silu { b: bitlen(127 · 128 · d) + 1, r_by_class: shifts of p(Silu), in_exp: the layer's silu_in_exponent, out_exp: the exponent of p(Silu) }`. An earlier table of the same definition is reused ([RFC-0042](RFC-0042-silu-exponents.md)).

`bitlen(x)` is the number of bits of `x`, so `bitlen(1) = 1`.

## Alternatives Considered

1. **Leave the prose and compare digests in CI.** Divergence would be found, but each one by a failing build. The verifier's author would still have had to read the compiler, against RFC-0001.
2. **Ship the graph with the claim list.** RFC-0007 'Alternatives Considered' 1 rejects an explicit constraint system as the public artifact: gigabytes at 405B, and the verifier would still check it against the model.

## Drawbacks

- The lowering's structure is now a tier-A surface: reordering two nodes changes the advice layout and needs an RFC. That was already true in effect, since every honest proof depended on it.

## Migration / Rollout

v0.3, for #362:

1. this RFC;
2. an independent oracle, `spec/vectors/lowering/generate.py`, written from this RFC alone. It writes each conformance model's graph for several lengths: every input, node, constant, shift-class set and table, by id;
3. a `tessera-compiler` test that checks the lowering against those vectors;
4. the verifier's lowering (#107), held to the same vectors.

## Testing Strategy

- **The vectors.** For each conformance model and `n` in {1, 4, 5, `max_context`}, the compiler's graph equals the oracle's: the same nodes with their operations and parameters, the same inputs, output shapes and kinds, constants, shift-class sets, weights and tables, all by id.
- **Every form.** The conformance models cover both profiles, `pot` sites of one class and several, affine sites at `RopeQ`, `RopeK`, `AttnResidual` and `Up`, and both residual routes ([RFC-0038](RFC-0038-uniform-layers.md)). The oracle's own test adds any form they miss.

## Open Questions

None.

## References

- [RFC-0005](RFC-0005-integer-model.md), [RFC-0006](RFC-0006-compiler-ir.md), [RFC-0022](RFC-0022-saturating-residual.md), [RFC-0025](RFC-0025-affine-sites.md), [RFC-0030](RFC-0030-masked-exponentials.md), [RFC-0037](RFC-0037-stage-schedule.md), [RFC-0038](RFC-0038-uniform-layers.md), [RFC-0041](RFC-0041-execution-plan.md), [RFC-0042](RFC-0042-silu-exponents.md).
