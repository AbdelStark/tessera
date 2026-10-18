# The Llama lowering

The graph that [RFC-0044](../../../docs/rfcs/RFC-0044-lowering-nodes.md) lowers from each conformance model of [`../models/`](../models/README.md), and from two variants that take the forms the models miss, node by node, at the lengths `n` = 1, 4, 5 and `max_context`: every graph input, node, constant, shift-class set, weight and table, each by its id in creation order. The verifier lowers the claim list with code of its own, and its graph must equal the compiler's node for node (RFC-0044 'Summary'). These vectors hold both to one statement.

## How they are made

[`generate.py`](generate.py) is the independent oracle. It is written from RFC-0044 alone, with the definitions it cites: RFC-0005, RFC-0006, RFC-0008's table kinds, RFC-0022, RFC-0025, RFC-0030, RFC-0037, RFC-0038, RFC-0042 and [03 §2–3](../../../docs/spec/03-data-model.md). It shares no code with `crates/`, `verifier/` or `python/`.

- **What it reads.** Each model's public part only: the architecture, the profile, `PublicQuantSpec` and the SiLU input exponents ([RFC-0006](../../../docs/rfcs/RFC-0006-compiler-ir.md#llama-lowering) 'Llama lowering'). [`../models/generate.py`](../models/generate.py) rebuilds it, and [`../tables/generate.py`](../tables/generate.py) gives an `Exp` table's value at its last key ([RFC-0044](../../../docs/rfcs/RFC-0044-lowering-nodes.md#the-tables) 'The tables') and each table's row count.
- **Checks before writing.**
  - Each graph keeps [RFC-0006](../../../docs/rfcs/RFC-0006-compiler-ir.md#operation-set) 'Operation set': inputs before their readers, one producer per tensor, and each operation's shapes. Its weight list holds each weight once, read by the matmul its `WeightRef` names.
  - Every layer is one template ([RFC-0037](../../../docs/rfcs/RFC-0037-stage-schedule.md#layers-and-kinds) 'Layers and kinds').
  - The conformance models take the forms that RFC-0044 'Testing Strategy' says they cover, and with the variants below the cases take every form of its procedures.
  - The file reads back as the graphs.

`python3 spec/vectors/lowering/generate.py` rewrites `lowering.json` byte for byte, in about 3 seconds. `cargo xtask gen-vectors --check` runs it on a copy and fails when the file differs.

### Readings

RFC-0044 leaves two things to its reader, and the oracle reads them so:

- **The weight list is in creation order.** `matmul` records each weight "in the graph's weight list" as it creates it, so a layer's weights come as Q, K, V, O, Gate, Up, Down, after `Embed` and before `LmHead`.
- **Only a weight is `i8`.** A class vector and a multiplier vector are other inputs ('Graph inputs'), so their types are the narrowest unsigned ones that hold their values: `u1` or `u2` for classes, `u16` for multipliers.

## Variants

RFC-0044 'Testing Strategy': "The oracle's own test adds any form they miss." Two variants edit a conformance model's public part, every layer still rescaled alike ([RFC-0038](../../../docs/rfcs/RFC-0038-uniform-layers.md#the-rule)):

| Variant | Edits | Forms it adds |
|---|---|---|
| `tiny-2-servable variant` | `V` affine at shift 20; `MlpResidual` affine at shift 63; `AttnResidual` of one class, shift 5 in layer 0 and 63 in layer 1; `Silu` of one class, shift 6; layer 1 at layer 0's score scale | affine `V` and `MlpResidual`; two steps at an affine site, at a site of one class, and in a layer whose one shift is at most 62 ([RFC-0038](../../../docs/rfcs/RFC-0038-uniform-layers.md#the-residuals-route)); a `Silu` lookup with no class vector; a shared `Exp` table |
| `tiny-2-minimal variant` | 4 KV heads; `RopeK` of one class, shift 22; score scales 1/60000 and 1/43330 | heads with no `Expand`; an `Exp` table of 20 bits, since the 19-bit table's last key is 11; a 19-bit `Exp` table whose last key rounds to 0 before its zero tail starts ([RFC-0030](../../../docs/rfcs/RFC-0030-masked-exponentials.md#range-analysis-and-validation): the check reads the value) |

A variant is a public part and nothing more. The lowering reads nothing else (RFC-0006 'Llama lowering'), so `lowering_vectors.rs` makes the same edits to its model's public part and lowers that, with no artifact behind it.

## `lowering.json`

`cases` holds one case per model or variant and length.

| Field | Meaning |
|---|---|
| `model`, `file` | the conformance model, or its variant, and its artifact in `../models/` |
| `variant` | a variant's edits, each in every layer it names: `n_kv_heads`; `rescale`, each `[layer, site, "pot", shifts]` or `[layer, site, "affine", r]`, the site's entry in `shift_class_sets` or `affine_sites` at that layer, in place of its own; `score_scales`, each `[layer, num, den]` |
| `profile`, `n` | the profile, and the sequence's length |
| `routes` | each residual site's route, `fused key` or `two steps`, one for every layer |
| `forms` | the forms of RFC-0044's procedures that the graph takes |
| `tensors` | the number of tensors: graph inputs and node outputs |
| `checkpoints` | the tensors `c_0` to `c_L`: the residual after the embedding and after each layer (RFC-0037) |
| `inputs[]` | `[id, role, shape, kind, type, lo, hi]`, a graph input with its declared type and range `[lo, hi]`. Its role is `tokens`, `positions`, `weight T l`, `classes T l` or `multipliers T l`, for the weight type `T` and layer `l` |
| `nodes[]` | `[id, operation, parameters, inputs, outputs, at]`: the operation's name and parameters, the inputs' tensor ids, and each output as `[id, shape, kind]`. `at` names the step that created the node, for people |
| `constants` | the constants, by index: each a list of values |
| `shift_classes` | the shift-class sets, by index |
| `weights[]` | `[type, layer, tensor]`: the weight list, in creation order |
| `tables[]` | `[id, kind, log2_rows]`: the lowering's tables. A kind is `{"Rope": {...}}`, `{"Exp": {"key_bits", "sigma", "scale": [num, den]}}` or `{"Silu": {"b", "r_by_class", "in_exp", "out_exp"}}`, with [RFC-0008](../../../docs/rfcs/RFC-0008-logup-and-tables.md#table-specifications)'s field names, and `log2_rows` is `⌈log2 T⌉` for its `T` rows |

Kinds are written as RFC-0044 'Notation' writes them: `A` for `Committed { Advice }`, `W` for `Committed { Weight }`, `P` for `Public`, `v` for `Virtual` and `L(t)` for `LookupOutput { table: t }`. A type is `u` or `i` and its bits, 03 §1's `IntType`. A node's outputs carry no type or range: those are range analysis's, which RFC-0044 leaves to RFC-0006 (its 'Non-Goals').

Operations carry RFC-0006's names and parameters:

| Operation | Parameters |
|---|---|
| `OneHot` | `vocab` |
| `MatMulWeight` | `weight`: `[type, layer]` |
| `ConstMul` | `c`: a constant's index |
| `Add`, `Reshape` | none |
| `SumSquares` | `axis` |
| `RsqrtRow` | `kappa`, `eps_int`, `dim` |
| `Mul` | `broadcast`: `Leading` or `Trailing` |
| `Requant` | `shift`: a shift-class set's index; `lo`, `hi` |
| `Rotate` | `rope`: a table's id |
| `Transpose` | `perm` |
| `Expand` | `axis`, `len` |
| `MatMulAct` | `transpose_rhs` |
| `SoftmaxRow` | `exp_table`: a table's id; `sigma` |
| `TableMap` | `table`: a table's id |

## Who checks them

- `crates/tessera-compiler` (`tests/lowering_vectors.rs`) decodes each artifact, makes a variant's edits, lowers it at each length and compares the graph with the case, field by field. On a difference it reports the first item that differs: the case, the input, node or entry, and the field.
- The verifier's lowering ([#107](https://github.com/AbdelStark/tessera/issues/107)) will be held to the same vectors.
