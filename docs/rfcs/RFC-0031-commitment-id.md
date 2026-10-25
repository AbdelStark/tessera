# RFC-0031: `commitment_id`, Where Every Column Chain Starts

- Status: Accepted
- Authors: @AbdelStark
- Created: 2026-10-17
- Target milestone: v0.3
- Amended by: [RFC-0057](RFC-0057-one-proof-unit.md) ('The descriptor': a proof is one unit, so `unit` is always 0)
- Supersedes, on acceptance:
  - [RFC-0004](RFC-0004-hashing-and-transcript.md#column-digests) 'Hash function and domains': the `commitment-id` domain; 'Column digests': what `commitment_id` binds;
  - [RFC-0011](RFC-0011-advice-commitment.md#column-digests-and-root) 'Column digests and root': the advice commitment's `commitment_id` and `AdviceCommitment.layout_digest`;
  - [RFC-0012](RFC-0012-weight-commitment.md#layout) 'Layout': each weight matrix's `commitment_id` and row block;
  - [RFC-0019](RFC-0019-verifier.md#check-order) 'Check order', step 4: the advice commitment's `layout_digest`.
- Origin: [#219](https://github.com/AbdelStark/tessera/issues/219)

## Summary

Every column chain starts from `d_j[0] = H("column-init", commitment_id ‖ u64(j))` (RFC-0004), so every leaf and every root of a commitment depends on `commitment_id`. No document gave it a formula:

- RFC-0004 said it binds "shape, rate, `B`, purpose";
- RFC-0011 said "the layout parameters, `S`, `B`, `P_A` and the batch digest";
- RFC-0012 said nothing.

This RFC defines it as the hash of a canonical descriptor, under a new domain:

```text
commitment_id = H("commitment-id", TCE(CommitmentDescriptor))
```

The descriptor names the commitment's purpose, the context it belongs to, its code parameters, its row block and its row count. Both sides derive it from public data. The advice commitment's `layout_digest` is its `commitment_id`, which the verifier checks. The RFC also fixes the weight matrices' row block, which no document stated.

## Motivation

Two implementations that differ in `commitment_id` disagree on every column leaf, so no commitment can be computed independently until it is defined. The review of #52 found the gap: `tv-hash` computes `column_digest(commitment_id, j, B, symbols)` and takes `commitment_id` as given.

The descriptor carries domain separation, not soundness. A root binds its columns through the hash's collision resistance whatever the chains start from. What `commitment_id` adds:

- columns of different commitments start from different chains: advice of two proofs, ten weight matrices, two epochs;
- each chain is bound to the shape the verifier reads it with: `B`, the row count, the code;
- both sides start from the same 32 bytes, which they compute independently.

## Proposed Design

### The descriptor

```rust
pub enum CommitmentDescriptor {          // TCE (RFC-0003): a u16 tag, then the fields in order
    Advice {
        statement_digest: Digest,        // H("statement", TCE(PublicStatement)) (RFC-0007)
        unit: u32,                       // always 0: a proof is one unit (RFC-0057)
        layout: AdviceLayoutParams,      // the claim list's (RFC-0026)
        rows: u64,                       // the unit's non-zero rows, every segment's (RFC-0011)
    },
    WeightMatrix {
        layout: WeightLayoutParams,      // the claim list's (RFC-0026)
        matrix: WeightMatrix,
        log2_k: u8,                      // the matrix's message width, k_T = 2^log2_k (RFC-0012)
        row_block: u32,                  // B
        rows: u64,                       // its committed rows m_T, plus B_T from the version that adds blinding rows (RFC-0040, RFC-0065)
        epoch: u32,
    },
}

pub enum WeightMatrix { Embed, Q, K, V, O, Gate, Up, Down, LmHead, ChannelParams }  // RFC-0012's table order
```

`commitment_id = H("commitment-id", TCE(CommitmentDescriptor))`. Every field is public:

- **Advice.**
  - The statement digest binds the claim list, `R_W` and its epoch, the manifest and the batch. The layout, a function of the claim list and the batch descriptor (INV-PCS-1), follows from it.
  - `unit` is always 0 (RFC-0057). It keeps its place so that the descriptor's encoding, and every digest, is unchanged.
  - `layout` and `rows` state the code and the number of rows each column chain absorbs. The verifier derives the rows from the layout, never from the prover.
- **Weights.**
  - `layout` holds the parameter set's `k_W`, rate, `P_W` and `B_W`.
  - `matrix`, `log2_k` and `rows` are the matrix's own, from the architecture ([RFC-0012](RFC-0012-weight-commitment.md#layout) 'Layout').
  - `epoch` separates the commitments of a refresh. They hold the same weights under fresh masks.

One `commitment_id` serves every segment of a commitment. Each segment's chains start from it, and `column-merge` orders the segment digests (RFC-0011).

### The domain

RFC-0004's table gains a row, between `envelope` and `column-init`:

| Domain | Input | Used by |
|---|---|---|
| `commitment-id` | `TCE(CommitmentDescriptor)` | `commitment_id`, the start of every column chain (RFC-0031) |

Both workspaces' domain lists and the hash vectors follow RFC-0004's table order.

### The weight matrices' row block

RFC-0004 makes `B` a public parameter of each commitment, and RFC-0011 sets the advice's in `AdviceLayoutParams.row_block`. For weights, nothing did.

A weight matrix now chains its columns over blocks of the parameter set's row block ([RFC-0014](RFC-0014-interaction-and-soundness.md#parameter-sets) 'Parameter sets'): 32 for `live-64` and `ni-100`, 4 for `test-small`. It is the value `AdviceLayoutParams.row_block` carries in the claim list. The descriptor records it, so a matrix's chains are bound to the block they are read in.

### `layout_digest`

`AdviceCommitment.layout_digest`, which `Round1` sends ([RFC-0014](RFC-0014-interaction-and-soundness.md#messages)), is the advice commitment's `commitment_id`. The verifier derives its own from public data and rejects a `Round1` that differs, `Reject(StatementMismatch)`, at RFC-0019's step 4. A prover whose layout disagrees with the verifier's then fails before any work, with a reason, rather than at the columns.

### Unchanged

- The column chains, the leaves, the Merkle trees and `column-merge` (RFC-0004, RFC-0011).
- `R_W`'s preimage (RFC-0012).
- The transcript.

## Alternatives Considered

1. **A random `commitment_id`, sent by the prover.** Rejected: the verifier would read a value it can derive, contrary to RFC-0004's rule. It would bind nothing about the shape either.
2. **A hash of the layout alone, without the statement.** Rejected: two proofs of one claim list over batches of equal shape would start every chain alike. The statement digest costs nothing and separates them.
3. **`commitment_id` per segment.** Rejected: `column-merge` already orders segments, and one identifier per commitment keeps RFC-0004's formula as it is.
4. **The descriptor's fields concatenated without TCE.** Rejected: TCE is already the one canonical encoding of every hashed structure, and gives the descriptor one encoding per value (INV-DM-1).
5. **A weight row block of its own in `WeightLayoutParams`.** Rejected: that would change the claim list's schema for a value the parameter set already fixes.

## Drawbacks

- The advice descriptor depends on the statement, so the prover must fix the statement before the witness pass. It already does: the transcript starts from the statement digest, before `Round1`.
- A proof holds one unit (RFC-0057), so `Round1` carries one advice commitment and `unit` is 0.

## Migration / Rollout

v0.3, before the first commitment is computed (#102, #106, #109, #115, #116). The specification changes in place: no artifact or proof has been written. Both workspaces' `Domain` lists gain `commitment-id`, and `spec/vectors/hash/` gains its vectors.

## Testing Strategy

- **The domain.** `spec/vectors/hash/` lists `commitment-id` among RFC-0004's domains, with its digests. Both workspaces check it in table order.
- **The descriptor.** The oracle encodes four descriptors under TCE and hashes them: advice at `test-small/1`'s layout and at a production layout, a `test-small` `Q` matrix and a production `ChannelParams`.
  - Both workspaces recompute each `commitment_id` from the oracle's bytes.
  - The commitments' own encodings check against the same bytes when #102 and #106 define the type.
- **`layout_digest`.** A `Round1` whose `layout_digest` differs from the derived `commitment_id` is rejected with `Reject(StatementMismatch)` (#116, #128).
- **Separation.** Advice commitments of two proofs, and weight matrices of two epochs, start their chains from different digests (#106, #102).

## Open Questions

None.

## References

- [RFC-0003](RFC-0003-canonical-encoding.md), [RFC-0004](RFC-0004-hashing-and-transcript.md), [RFC-0007](RFC-0007-claim-list.md), [RFC-0011](RFC-0011-advice-commitment.md), [RFC-0012](RFC-0012-weight-commitment.md), [RFC-0014](RFC-0014-interaction-and-soundness.md), [RFC-0018](RFC-0018-gpu-prover-runtime.md), [RFC-0019](RFC-0019-verifier.md), [RFC-0026](RFC-0026-claim-list-layout.md).
- [#219](https://github.com/AbdelStark/tessera/issues/219), [#52](https://github.com/AbdelStark/tessera/issues/52).
