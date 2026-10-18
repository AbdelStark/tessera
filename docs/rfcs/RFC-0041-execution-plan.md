# RFC-0041: Tiles and the Execution Plan

- Status: Accepted
- Authors: @AbdelStark
- Created: 2026-10-18
- Target milestone: v0.3
- Supersedes, on acceptance:
  - [RFC-0006](RFC-0006-compiler-ir.md#tiling-and-the-execution-plan) 'Tiling and the execution plan': what a tile holds, the production order, and the plan's contents. The plan carries the claim list and one sequence's tiles. It holds neither row ranges, nor stage membership, nor devices.
  - [RFC-0011](RFC-0011-advice-commitment.md#canonical-advice-layout) 'Canonical advice layout' rule 2: tensors are allocated tile by tile, in this RFC's production order;
  - [03 §5](../spec/03-data-model.md#5-artifact-catalog): `ExecutionPlan` is defined here.
- Origin: [#101](https://github.com/AbdelStark/tessera/issues/101)

## Summary

RFC-0006 gives tiles two shapes, `(layer, operation)` and `(layer, kv_head, query_block)`. It says the private `ExecutionPlan` (`0x0202`) "lists tiles, their advice row ranges, the stage each relation joins, and device assignments". RFC-0011 allocates the advice in "production order (layer, then tile kind, then head, then query block…)". None of these says:

- what a tile holds;
- where the tokens, the embedding, the final norm, the logits and the tables' multiplicities go;
- in what order a layer's tiles come;
- what the plan's bytes are.

This RFC defines the tiles of a sequence, which partition its committed values, and the production order that the advice layout follows. It defines the plan as the claim list it plans and the sequence's tiles. Row ranges, stage membership and devices are not in the plan: they depend on the batch or the machine, which compilation does not see, and the first two are derived by the verifier too.

## Motivation

- **An undefined tier-A schema** ([02 §1](../spec/02-public-api.md#1-stability-tiers)).
- **The layout depends on the order.** The verifier derives the advice layout itself, from the claim list and the batch descriptor (INV-PCS-1), and checks the prover's `layout_digest` against its own ([RFC-0031](RFC-0031-commitment-id.md#layout_digest)).
  - Two implementations of an informal order would place tensors at different offsets.
  - Every honest proof would then be rejected at RFC-0019 step 4.
  - The order must be stated exactly.
- **What compilation cannot know.**
  - `tessera compile` sees the model and the parameter set ([02 §2.2](../spec/02-public-api.md#22-tessera-rust-prover-side)). It does not see the batch, whose sequence lengths fix the row ranges.
  - Nor does it see the prover's devices and memory, which [RFC-0018](RFC-0018-gpu-prover-runtime.md#memory-scheduler)'s planner turns into assignments at prove time.
  - The stage schedule is derived from the lowering by the prover and the verifier alike ([RFC-0037](RFC-0037-stage-schedule.md)).
- **What the prover needs from the plan.**
  - `tessera prove` and `tessera commit-weights` take `--plan` and no claim list (02 §2.2), so the plan must carry the claim list.
  - [RFC-0029](RFC-0029-table-set.md#the-table-set) left to #101 whether it carries it or names it by digest.

## Goals

- Tiles that partition every committed value of a sequence's lowering, each value in exactly one tile.
- One production order, computable from the claim list and a sequence's length alone, for the prover's passes and the verifier's layout.
- A plan that is a pure function of the model's public part and the parameter set, deterministic to the byte, with no secret value.

## Non-Goals

- **A batch of several requests in one proof unit.** No document lowers several sequences together (RFC-0006: "one graph covers one sequence"), and none batches them in the schedule. Until #357 defines it, a proof unit of `tessera-protocol/1` covers one request, and this RFC orders one sequence's tiles.
- **Proof units of several layer ranges.** `tessera-protocol/1` sends one advice commitment (`Round1.advice`, [RFC-0036](RFC-0036-protocol-messages.md#round1)), so a proof is one unit.
- **The layout's offsets, alignment, zero rows and segments**, which are RFC-0011's and #106's.

## Proposed Design

### What is committed

A sequence's lowering commits, as advice:

- every graph tensor of kind `Committed { Advice }` or `LookupOutput` ([03 §3](../spec/03-data-model.md#3-circuit-types)): the tokens, which are a graph input, and node outputs;
- each gadget's committed values ([RFC-0009](RFC-0009-gadgets.md)), beyond the node's outputs: limbs, split parts, maxima, brackets;
- each lookup table's multiplicities ([RFC-0008](RFC-0008-logup-and-tables.md)), shared by every lookup into it.

Weights are the weight commitment's ([RFC-0040](RFC-0040-weight-commitment-artifacts.md)), and public and virtual tensors are not committed. Masking coefficients join with `tessera-protocol/2` ([RFC-0013](RFC-0013-zero-knowledge.md)).

### Tiles

The tiles of a sequence of `n` tokens are the following. `N = 2^{⌈log2 n⌉}` is its padded length.

- **`Tokens`.** The tokens.
- **`Node { node }`.** A node that commits a value, other than a layer's attention node: its committed outputs, then its gadget's committed values. These are the prologue's and epilogue's nodes and a layer's linear tiles `(layer, operation)` ([RFC-0037](RFC-0037-stage-schedule.md#layers-and-kinds) 'Layers and kinds' defines a layer's nodes).
- **`Attention { node, kv_head, query_block }`.** A layer's attention node is its `SoftmaxRow`. Its committed values are:
  - the exponentials `e[h, q, i]` and the reciprocals `R[h, q]`;
  - the gadget's maxima `c[h, q]` and the limbs of its two `NonNeg` instances.

  They are split into one tile per KV head `g` and query block `b`. The tile holds the query heads `h` with `⌊h / (n_heads / n_kv_heads)⌋ = g`, and the queries `q` from `b·Q'` to `(b + 1)·Q' − 1`, every key `i` with them. Here `Q' = min(Q, N)` with `Q = GadgetPolicy.query_block`, and `b` runs from 0 to `N / Q' − 1`.
- **`Table { table }`.** A lookup table's multiplicities.

Every committed value lies in exactly one tile. A node that commits nothing has no tile.

### Production order

The tiles of a sequence come in this order:

1. `Tokens`;
2. the `Node` and `Attention` tiles, in node order. The attention node's tiles take its place, by KV head, then query block. This orders the prologue's nodes, each layer's, then the epilogue's;
3. the `Table` tiles, by table id, for each table that a lookup reads.

Within a tile, values come in this order:
- the node's outputs, in output order;
- then the gadget's committed values, in the order RFC-0009 lists them for the gadget.

An attention tile's exponentials take the variable order `(i_hi, h, q, i_lo)` of [RFC-0034](RFC-0034-causal-blocks.md#causal-layout). This is the order in which RFC-0011 allocates the advice. RFC-0006's "layer, then tile kind, then head, then query block" is this order for one layer.

`Q` must be a power of two, since RFC-0034's layout takes its logarithm. A claim list with another `query_block` fails RFC-0007 step 6 ([RFC-0039](RFC-0039-statement-fields.md#the-checks): `Layout`).

### The plan

```rust
pub struct ExecutionPlan {          // schema 0x0202, version 1, private, .tep
    pub claims: ClaimList,          // the claim list it plans (RFC-0007, RFC-0026)
    pub tiles: Vec<Tile>,           // one sequence's tiles, in production order
}

pub enum Tile {
    Tokens,
    Node { node: u32 },                     // the node's index in the lowering's graph
    Attention { node: u32, kv_head: u32 },  // every query block of one KV head
    Table { table: u32 },                   // a TableId
}
```

- **Sequence length.** A lowering's node and tensor ids do not depend on the sequence's length; only shapes do. The plan therefore holds one sequence's tiles for every length. An `Attention` tile stands for its query blocks, which a length of `n` expands into `N / Q'` tiles in order.
- **Determinism.** `compile` writes the plan from the claim list it emits. Compiling twice gives the same bytes, as RFC-0006's determinism test requires, and the plan holds no secret value: the claim list is public, and the tiles follow from it.
- **Checking.** The plan's readers, `tessera prove` and `tessera commit-weights`, also take `D`. Each compiles `D`'s claim list for the plan's parameter set and derives its tiles. A plan whose `claims` or `tiles` differ is refused (`ERR-USAGE`): it is not `D`'s plan. The verifier never reads a plan.
- **What the plan does not hold.**
  - **Row ranges** are the advice layout's: a function of the claim list and the batch (RFC-0011).
  - **Stage membership** is the stage schedule's, which the prover derives from the lowering, as the verifier does (RFC-0037).
  - **Devices** are assigned at prove time. RFC-0018's planner splits the layers into contiguous ranges that match the advice segments, from `ProverConfig`.

## Alternatives Considered

1. **The plan as RFC-0006 lists it, with row ranges and devices.** Row ranges need the batch's lengths, which `compile` does not receive, and devices need the machine. A plan for one batch on one machine would be recompiled per batch, and its ranges would duplicate a layout that the verifier derives anyway.
2. **The claim list by digest.** It would leave `prove` needing a second file and a check that the two match. The claim list is a few kilobytes against a plan that already depends on it.
3. **Tiles of whole tensors only.** One tile per committed tensor would put a layer's exponentials, `n_heads · N²` values, in one tile. Streaming by KV head and query block is what bounds the prover's memory (RFC-0018) and keeps a score row's keys contiguous (RFC-0034).
4. **Linear tiles first, then attention tiles, in each layer**, reading "tile kind" as a grouping. It would separate the attention's values from the node order in which the prover computes them, for no gain in the layout.

## Drawbacks

- The plan repeats what it can be derived from. That is its purpose: a cache of the tiles, checked against the claim list, so that a plan and a claim list cannot disagree.
- A batch of one request until #357.

## Migration / Rollout

v0.3, for #101:

1. this RFC and its amendments;
2. `tessera-compiler` derives the tiles and the plan, encodes it, and checks a decoded plan against its claim list;
3. the advice layout (#106) and the witness regeneration by tile (#112) build on the tiles.

## Testing Strategy

- **Partition.** For each conformance model and several sequence lengths, `n` from 1 to `max_context` with powers of two and their neighbours, every committed value lies in exactly one tile once the attention tiles are expanded. The test enumerates every element of every committed tensor.
- **Order.** The expanded tiles follow node order, with attention tiles by KV head, then query block, and table tiles last, by id.
- **Independence of `n`.** The plan's tiles are the same for every length.
- **Determinism and secrecy.** Two compilations give the same bytes. A canary weight value is absent from the plan's bytes (#101).
- **Binding.** A plan whose tiles differ from its model's, by one tile changed, removed or added, is refused, and so is one whose claim list differs.

## Open Questions

None.

## References

- [RFC-0006](RFC-0006-compiler-ir.md), [RFC-0009](RFC-0009-gadgets.md), [RFC-0011](RFC-0011-advice-commitment.md), [RFC-0018](RFC-0018-gpu-prover-runtime.md), [RFC-0029](RFC-0029-table-set.md), [RFC-0034](RFC-0034-causal-blocks.md), [RFC-0037](RFC-0037-stage-schedule.md), [RFC-0039](RFC-0039-statement-fields.md), [RFC-0040](RFC-0040-weight-commitment-artifacts.md).
