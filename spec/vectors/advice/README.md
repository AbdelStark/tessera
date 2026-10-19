# Public model advice vectors

`generate.py` derives the full advice layout from the independent public
[lowering fixtures](../lowering/README.md) and Python integer range analysis.
It imports no implementation from either Rust workspace or the quantizer, reads
no weights or activations, and was written before the model-layout implementations.
The contract is [RFC-0011](../../../docs/rfcs/RFC-0011-advice-commitment.md), with
[RFC-0034](../../../docs/rfcs/RFC-0034-causal-blocks.md),
[RFC-0035](../../../docs/rfcs/RFC-0035-padded-rows.md),
[RFC-0041](../../../docs/rfcs/RFC-0041-execution-plan.md),
[RFC-0046](../../../docs/rfcs/RFC-0046-advice-segments.md),
[RFC-0047](../../../docs/rfcs/RFC-0047-advice-order.md) and
[RFC-0050](../../../docs/rfcs/RFC-0050-gadget-advice-columns.md).

Run `python3 -I spec/vectors/advice/generate.py`. `cargo xtask gen-vectors`
discovers this oracle automatically. The oracle expands every attention tile;
the verifier can use a different algorithm that counts repetitions without
allocating per-tile metadata.

`layout.json` version 1 contains 54 cases: every lowering fixture at row widths
`2^10` and `2^22`, plus each length-five graph with query blocks of one. Query
blocks 32 and 512 exercise causal rows and short blocks; segment counts 1, 3
and 8 include empty segments. Each case gives:

- The complete canonical table descriptors and gadget choices, auxiliary roles
  and repeated table requests, including virtual top limbs.
- Tiles in production order. Keys are `tokens`, `node/node_id`,
  `attention/node_id/kv_head/query_block` or `table/table_id`.
- Every allocated value as `[source, shape, permutation, base_row, rows, zero_at]`.
  Sources are `tensor/id`, `aux/node_id/role/component`, or `multiplicity/table_id`.
  Role names distinguish input/root limbs, each comparison array, Requant parts
  and Softmax maximum/slacks. Component indices are least significant first.
- `permutation[j]` names the natural local point coordinate used as layout
  coordinate `j`, least significant first. An attention share has fixed tile
  head/query coordinates and the full padded local `g × Q'` domain. The score
  permutation splits the key axis as RFC-0034 requires.
- `zero_at` is the first whole zero row relative to that value; `rows` means
  none. It concerns public padding/masking, never a zero secret value. Final
  carries and other real constant columns remain allocated. Two padded axes
  use RFC-0047's outermost-axis rule.
- Total allocated rows, `nu`, merged half-open zero-row intervals and segment
  intervals. Alignment gaps belong to the preceding segment. The `2^32` cap
  concerns allocated rows, including zero rows, rather than encoded rows.

The generator self-checks all root widths, the zero-maximum-root boundary,
complete product/carry lookup boxes, the 208 used/48 skipped MLP rows, the exact
row limit and index permutations by Cartesian enumeration. It retains whole
constants above Goldilocks. These checks are geometry/metadata evidence, not a
full-model proof, parameter-fit result or performance benchmark.
