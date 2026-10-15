# RFC-0011: Streaming Ligero-Family Commitment for Per-Proof Advice

- Status: Accepted
- Authors: @AbdelStark
- Created: 2026-10-15
- Target milestone: v0.1 (scheme), v0.3 (layout and openings)

## Summary

All per-proof advice (lookup outputs, per-row scalars, limbs, one-hot vectors, multiplicities, masking coefficients) is committed in one Ligero-family tensor commitment. The advice polynomial is laid out as a matrix of rows of width `k = 2^22`. Each row is Reed–Solomon encoded at rate 1/2 as soon as the witness pass produces it, folded into chained column digests, and freed. Rows that are structurally zero are never encoded or hashed. Openings recompute rows in a second streaming pass to form a few combined rows, after which the verifier queries `τ` columns. This RFC fixes the layout, encoding, streaming procedure, opening protocol and soundness argument.

## Motivation

The source states: "Advice is RS-encoded (rate 1/2) row by row during the witness pass, folded into column hashes and freed; long-context witnesses reach terabytes and are never resident. The opening recomputes rows in a second streaming pass. Proof size is dominated by τ opened columns" (source §7, "Per-proof commitments stream"). VerInf's fixed Ligero row width made proof size and verification linear, 93.6 GB and 17.7 h (source §4 table, "Change"). A 16K-token, 16K-context batch for 405B commits about 2.4×10^12 elements, so neither the witness nor its encoding can ever be resident.

## Goals

- Commit 10^12+ elements with memory proportional to one row block.
- Openings at any number of claim points with a handful of combined rows.
- Proof size about 2 GB and verifier work of seconds for a 16K-token batch at 16K context.
- Byte-identical commitments from the CPU and GPU backends.

## Non-Goals

- Hiding (blinding rows and masked evaluations) beyond the encoding mask defined here; see RFC-0013.
- The persistent weight commitment (RFC-0012), which reuses this scheme with different parameters.
- Recursive compression of openings (NG-3).

## Proposed Design

### Canonical advice layout

The verifier must locate every committed tensor from public data, so the layout is a deterministic function of the claim list and the batch descriptor:

1. Every committed tensor `X` has a multilinear domain of `n_X` variables obtained by padding each dimension to a power of two, with padded dimensions ordered most significant so that padding forms whole rows.
2. Tensors are allocated in production order (layer, then tile kind, then head, then query block, as in [RFC-0006](RFC-0006-compiler-ir.md#tiling-and-the-execution-plan)), each at the next offset aligned to `2^{max(n_X, log2 k)}`. Every tensor therefore occupies whole rows; tensors smaller than a row get one row.
3. The advice polynomial `Ã` has `ν` variables; its matrix view has row index = high `ν − log2 k` bits and column index = low `log2 k` bits. A claim `X̃(r) = v` is the claim `Ã(r ‖ base_X) = v`, where `base_X` is the tensor's fixed row-offset bits.
4. **Zero rows.** Rows that hold only padding or alignment gaps are public and are skipped: not encoded, not hashed, not opened. They cost nothing.
5. The rows are partitioned into `S` contiguous segments (public, default 8), aligned to layer boundaries, so that each GPU can commit its own segment ([RFC-0018](RFC-0018-gpu-prover-runtime.md#partitioning)).

### Encoding

- A row's `k` values are evaluations of a polynomial `f` of degree `< k` on the subgroup `H` of order `k`.
- The encoding adds a vanishing mask: `g(X) = f(X) + Z_H(X) · m(X)` with `Z_H(X) = X^k − 1` and `m` a random polynomial of degree `< P_A` (`P_A` mask coefficients, at least the parameter set's `τ_A`: 160 for `live-64`, 200 for `ni-100`). On `H` the mask vanishes, so the message is unchanged.
- The codeword is `g` evaluated on the coset `7·L`, where `L` is the subgroup of order `n = 2k`; `7·L` is disjoint from `H`. Rate `ρ = (k + P_A)/n ≈ 1/2`.
- Encoding cost per row: an inverse NTT of size `k` and a coset NTT of size `2k`.

### Column digests and root

- Within each segment, column `j` is chained over row blocks of `B = 32` non-zero rows ([RFC-0004](RFC-0004-hashing-and-transcript.md#column-digests)), giving one segment digest per column.
- `leaf_j = H("column-merge", d_j^{(1)} ‖ … ‖ d_j^{(S)})`.
- The root is the Merkle root over the `n` leaves. `commitment_id` binds the layout parameters, `S`, `B`, `P_A` and the batch digest.

```rust
pub struct AdviceLayoutParams { pub log2_k: u8, pub log2_inv_rate: u8, pub mask_coeffs: u32,
                                pub row_block: u32, pub segments: u32 }
pub struct AdviceCommitment { pub root: Digest, pub layout_digest: Digest }

pub struct StreamingCommitter { /* chain state: 32 bytes per column per segment */ }
impl StreamingCommitter {
    pub fn new(params: &AdviceLayoutParams, layout: &AdviceLayout, segment: u32, mask_rng: &mut MaskRng) -> Self;
    pub fn push_row(&mut self, row_index: u64, values: &[Fp]) -> Result<(), PcsError>; // must follow layout order
    pub fn finish(self) -> SegmentDigests;
}
pub fn merge_segments(parts: &[SegmentDigests]) -> AdviceCommitment;
```

### Streaming procedure (witness pass)

For each non-zero row in layout order within the segment: regenerate the row's values from the witness engine; draw the row's mask coefficients from the per-proof mask stream (a PRG keyed by the per-proof secret seed, so a restart reproduces them, [RFC-0018](RFC-0018-gpu-prover-runtime.md#crash-consistency)); encode; append the codeword to the block buffer; every `B` rows, advance all column chains and clear the buffer. Memory: one block buffer, `B · n · 8` bytes (2 GiB at `k = 2^22`, sharded by column across a GPU's memory), plus `32 · n` bytes of chain state per segment.

### Opening protocol

After the last stage (RFC-0010):

1. **Grouping.** Claims on `Ã` are grouped by their column sub-point (the low `log2 k` coordinates). Claims from one stage on same-shaped tensors share a column sub-point, so the number of groups `G` is about the number of stages; the compiler's report lists `G` (target `≤ 16`).
2. **Row coefficients.** For group `g`, `c_g(row) = Σ_{claims in g} γ_claim · eq(row-part of point, row)`, with batching challenges `γ` from the transcript. Both sides compute `c_g` from public data; it is non-zero only on rows of the claimed tensors.
3. **Proximity coefficients.** `ξ_row`, drawn from the transcript after Round 1 (label `pcs/advice/xi`), defined as `eq(ξ*, row)` for a random point `ξ*`.
4. **Opening pass.** The prover regenerates every non-zero row once more and accumulates `u_g = Σ_row c_g(row) · A_row` for all groups and `u_ξ = Σ_row ξ_row · A_row` (message space, over `Fp2`), plus the combined mask polynomials `M_g = Σ_row c_g(row) m_row` and `M_ξ`.
5. **Send.** `u_g`, `M_g` for every group, and `u_ξ`, `M_ξ` (label `pcs/advice/combined`).
6. **Evaluation check.** For each group, the verifier checks `Σ_{claims in g} γ_claim · v_claim = ⟨u_g, eq(column sub-point)⟩`.
7. **Column queries.** The verifier draws `τ_A` column indices (live: verifier's choice after receiving step 5; non-interactive: transcript with grinding, RFC-0014). The prover answers with every non-zero row's symbol in each queried column, in segment order, plus a canonical Merkle multiproof.
8. **Column check.** For each queried `j`: recompute `leaf_j` from the column; verify the multiproof; check `Σ_row c_g(row) · col_j[row] = enc(u_g)(x_j) + Z_H(x_j) · M_g(x_j)` for every `g`, and the same for `ξ`.

Soundness: if the committed rows are `δ`-far from the code for `δ < (1 − ρ)/2`, the proximity check passes with probability at most `(1 − δ)^{τ_A} + n/|Fp2|`; otherwise each row decodes uniquely and the evaluation checks bind every claim to the decoded rows (Ligero, Brakedown). Conservatively each query gives `−log2((1 + ρ)/2) ≈ 0.415` bits ([RFC-0014](RFC-0014-interaction-and-soundness.md#parameter-sets)).

### Sizes at the default parameters (16K tokens, 16K context, 405B)

| Quantity | Value |
|---|---|
| Committed elements | about 2.4×10^12 |
| Non-zero rows | about 5.8×10^5 |
| Opened columns (`τ_A = 155`) | 155 × 5.8×10^5 × 8 bytes ≈ 0.72 GB |
| Combined rows (`G + 1 ≤ 17`) | ≤ 17 × 2^22 × 16 bytes ≈ 1.14 GB |
| Merkle multiproof | < 0.2 MB |
| Verifier work | hash 0.72 GB of columns; `(G + 1) · τ_A` inner products of length 5.8×10^5; `G + 1` coset NTTs of size 2^23 over `Fp2` |

### Invariants

- **INV-PCS-1.** Layout is a function of the claim list and batch descriptor only.
- **INV-PCS-2.** Rows are pushed in layout order; out-of-order pushes are rejected by `StreamingCommitter`.
- **INV-PCS-3.** Column digests are chained per segment and merged per leaf; a column's leaf depends on every non-zero row's symbol in that column and on nothing else.
- **INV-PCS-4.** Mask coefficients are fresh per proof and never reused across commitments.

### Failure modes

| Failure | Response |
|---|---|
| Honest prover's opening pass regenerates a different row than the witness pass | `ERR-INTERNAL` caught by a self-check: the prover recomputes one random column's leaf before sending |
| Combined row not consistent with columns | `Reject(Proximity)` |
| Group evaluation mismatch | `Reject(Evaluation)` |
| Multiproof invalid | `Reject(MerklePath)` |

## Alternatives Considered

1. **Storing the encoded witness.** Rejected: terabytes per proof at 16K context.
2. **FRI-based multilinear commitments (BaseFold, WHIR).** Considered: smaller proofs (polylogarithmic) and faster verification. Rejected for v1: the prover must run a folding protocol over the entire committed vector after commitment, which needs either the vector resident or many regeneration passes; the tensor scheme needs one opening pass. BaseFold/WHIR remain candidates for recursive compression (NG-3).
3. **Brakedown's linear-time codes.** Considered for encoding speed. Rejected: NTT-based RS encoding is fast on GPUs, RS proximity analysis is the best understood, and the source assumes RS at rates 1/2 and 1/4.
4. **Systematic encoding.** Rejected: opened systematic symbols are witness entries, which would need a separate hiding layer.
5. **Per-tensor commitments.** Rejected: thousands of roots, openings and Merkle trees instead of one.
6. **Padding tensors in place to powers of two.** Rejected: up to 23% wasted commitments for 53,248-wide MLP tensors; ordering padded dimensions most significant turns all padding into skipped zero rows.

## Drawbacks

- Every opening needs a full regeneration pass of the witness, about one native forward pass of extra compute (RISK-9 in [00 §9](../spec/00-overview.md#9-risks)).
- Combined rows dominate proof size when many column sub-points exist; the compiler must keep `G` small.
- Proof size is linear in `τ_A` times the number of non-zero rows.

## Migration / Rollout

v0.1 delivers the encoding, chained digests, Merkle tree and a non-streaming reference commit and open, plus the GPU encode-and-hash microkernel for the unit-cost measurement. v0.3 delivers the canonical layout, streaming committer and opening protocol on CPU. v0.5 moves both to the GPU.

## Testing Strategy

- Encoding: codewords of random rows lie in the code (degree check by interpolation on random subsets); the mask vanishes on `H`; CPU and GPU codewords are identical.
- Streaming equals batch: the streaming committer and a reference that holds the whole matrix produce the same root for random layouts, including zero rows, partial final blocks and `S ∈ {1, 2, 8}`.
- Layout: prover and verifier derive identical layouts for conformance models; every tensor's claims map to the right row offsets.
- Openings: random claims at 1–40 points in up to 16 groups verify; one altered element of `u_g`, `M_g`, a column symbol or a multiproof sibling is rejected with the expected reason.
- Proximity: a matrix with one row replaced by a non-codeword at relative distance `δ ∈ {0.05, 0.25, 0.5}` is rejected at an empirical rate at least `1 − (1 − δ)^τ` over 10^4 trials at small `τ`.
- Self-check: a fault-injected opening pass that perturbs one row triggers `ERR-INTERNAL` before any message is sent.

## Open Questions

None.

## References

- Source §4 table (fixed Ligero row width), §6 principle 9, §7 ("Per-proof commitments stream"), §8.
- Ames, Hazay, Ishai, Venkitasubramaniam, "Ligero", CCS 2017.
- Golovnev, Lee, Setty, Thaler, Wahby, "Brakedown", CRYPTO 2023.
- Ben-Sasson, Carmon, Ishai, Kopparty, Saraf, "Proximity gaps for Reed–Solomon codes", FOCS 2020.
