# RFC-0011: Streaming Ligero-Family Commitment for Per-Proof Advice

- Status: Accepted
- Authors: @AbdelStark
- Created: 2026-10-15
- Target milestone: v0.1 (scheme), v0.3 (layout and openings)
- Amended by: [RFC-0031](RFC-0031-commitment-id.md) ('Column digests and root': `commitment_id` and `layout_digest`), [RFC-0034](RFC-0034-causal-blocks.md) ('Canonical advice layout': the score tensors' layout and their masked blocks), [RFC-0035](RFC-0035-padded-rows.md) ('Canonical advice layout': what padded rows hold), [RFC-0068](RFC-0068-leaf-stage-zero-knowledge.md) ('Opening protocol' step 1 and 'Sizes': in `tessera-protocol/2`, the column masks' group, `G ≤ 17`), [RFC-0013](RFC-0013-zero-knowledge.md) ('Opening protocol' step 1 and 'Sizes', in `tessera-protocol/2`: a group per reduction stage for its masks' folds, #538), [RFC-0070](RFC-0070-mask-batching.md) ('Opening protocol' step 1 and 'Sizes', in `tessera-protocol/2`: the masks' batching group, `G ≤ 20`), [RFC-0071](RFC-0071-request-slots.md) ('Canonical advice layout' rule 4: a batch's padded rows between slots), [RFC-0013](RFC-0013-zero-knowledge.md#masked-extensions) ('Canonical advice layout' rule 1, in `tessera-protocol/2`: a token axis pads to four rows at least, #488), [RFC-0013](RFC-0013-zero-knowledge.md#masked-extensions) ('Opening protocol' step 2, in `tessera-protocol/2`: a committed claim's mask row joins its group, [#554](https://github.com/AbdelStark/tessera/issues/554)), draft [RFC-0074](RFC-0074-anchor-binding.md) ('Canonical advice layout' and INV-PCS-1: the anchor's inputs from the log, and rule 4 for a tensor non-zero at padding; in effect when it is accepted)

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

The verifier must locate every committed tensor from public data, so the layout is a deterministic function of the claim list and the batch descriptor, and of the anchor's inputs from the logs the statement names (draft [RFC-0074](RFC-0074-anchor-binding.md#the-advice-layout-and-the-openings)):

1. Every committed tensor `X` has a multilinear domain of `n_X` variables obtained by padding each dimension to a power of two, a token axis to four rows at least in `tessera-protocol/2` (RFC-0013 'Masked extensions'), with padded dimensions ordered most significant so that padding forms whole rows. [RFC-0047](RFC-0047-advice-order.md#the-order-of-a-tensors-variables) fixes this layout order: the padded axes, then the others, each in axis order. A score tensor `e` orders each attention tile's variables `(i_hi, h, q, i_lo)` instead, so that its rows are blocks of keys ([RFC-0034](RFC-0034-causal-blocks.md#causal-layout)). A padded token, position, layer or head holds the witness its gadgets compute on zero inputs: zero, except for a few table values and constants ([RFC-0035](RFC-0035-padded-rows.md#what-a-padded-row-holds)).
2. Tensors are allocated tile by tile, in the production order of [RFC-0041](RFC-0041-execution-plan.md#production-order) (node order, an attention node's tiles by KV head, then query block, the tables' multiplicities last), each at the next offset aligned to `2^{max(n_X, log2 k)}`. An attention tile's share of a tensor is allocated as a tensor of its own. Every tensor therefore occupies whole rows; tensors smaller than a row get one row.
3. The advice polynomial `Ã` has `ν = log2 k + ⌈log2 R⌉` variables, `R` the rows the layout allocates ([RFC-0047](RFC-0047-advice-order.md#ν)); its matrix view has row index = high `ν − log2 k` bits and column index = low `log2 k` bits. A claim `X̃(r) = v` is the claim `Ã(r ‖ base_X) = v`, with `r`'s coordinates in the tensor's layout order, where `base_X` is the tensor's fixed row-offset bits. For a score tensor, `r` is taken in the order of rule 1.
4. **Zero rows.** Rows that hold only zero padding or alignment gaps are public and are skipped, the padded rows between a batch's slots among them where they fill whole rows ([RFC-0071](RFC-0071-request-slots.md)): not encoded, not hashed, not opened. They cost nothing. So are a score tensor's rows that hold only masked positions, which are zero ([RFC-0034](RFC-0034-causal-blocks.md#zero-rows)). Every row of a tensor non-zero at padding ([RFC-0072](RFC-0072-anchor-relations.md#padded-positions)) is committed: its padded positions hold computed values (draft [RFC-0074](RFC-0074-anchor-binding.md#the-advice-layout-and-the-openings)).
5. The rows are partitioned into `S` contiguous segments (public, default 8), aligned to layer boundaries, so that each GPU can commit its own segment ([RFC-0018](RFC-0018-gpu-prover-runtime.md#partitioning)). [RFC-0046](RFC-0046-advice-segments.md#segments) fixes the layers and rows each segment holds.

### Encoding

- A row's `k` values are evaluations of a polynomial `f` of degree `< k` on the subgroup `H` of order `k`.
- The encoding adds a vanishing mask: `g(X) = f(X) + Z_H(X) · m(X)` with `Z_H(X) = X^k − 1` and `m` a random polynomial of degree `< P_A` (`P_A` mask coefficients, at least the parameter set's `τ_A`: 160 for `live-64`, 200 for `ni-100`). On `H` the mask vanishes, so the message is unchanged.
- The codeword is `g` evaluated on the coset `7·L`, where `L` is the subgroup of order `n = 2k`; `7·L` is disjoint from `H`. Rate `ρ = (k + P_A)/n ≈ 1/2`.
- Evaluations are in natural order. With `ω_m = 7^((P − 1)/m)`, the primitive `m`-th root of unity of [RFC-0002](RFC-0002-field-arithmetic.md#constants), position `i` of a row is `f(ω_k^i)` and symbol `j` of the codeword is `g(7·ω_n^j)`. Column `j` of the commitment holds symbol `j` of every row, so a queried column `j` is checked at `x_j = 7·ω_n^j`.
- The mask is `m(X) = Σ_{i < P_A} m_i·X^i`, its coefficients drawn in the order `m_0, m_1, …` from the row's mask stream ('Streaming procedure'). `P_A ≤ k`, so `g` has degree below `n` and the codeword determines it. That bound is about decoding only. Soundness uses the rate `ρ = (k + P_A)/n` as it is, and at rate 1/2 with `P_A = k` the code is all of `Fp^n`, where the proximity check proves nothing. The parameter sets keep `P_A` far below `k`, with `ρ ≤ 0.51` ([RFC-0014](RFC-0014-interaction-and-soundness.md#parameter-sets)), and the per-query bits there use this `ρ`.
- Encoding cost per row: an inverse NTT of size `k` and a coset NTT of size `2k`.

### Column digests and root

- Within each segment, column `j` is chained over row blocks of `B = 32` non-zero rows ([RFC-0004](RFC-0004-hashing-and-transcript.md#column-digests)), giving one segment digest per column.
- `leaf_j = H("column-merge", d_j^{(1)} ‖ … ‖ d_j^{(S)})`.
- The root is the Merkle root over the `n` leaves.
- Every segment's chains start from one `commitment_id`, the hash of the advice descriptor ([RFC-0031](RFC-0031-commitment-id.md#the-descriptor)): the statement digest, the proof unit, `AdviceLayoutParams` and the unit's non-zero rows. `AdviceCommitment.layout_digest` is that `commitment_id`; the verifier derives its own and rejects another (`Reject(StatementMismatch)`).

```rust
pub struct AdviceLayoutParams { pub log2_k: u8, pub log2_inv_rate: u8, pub mask_coeffs: u32,
                                pub row_block: u32, pub segments: u32 }
pub struct AdviceCommitment { pub root: Digest, pub layout_digest: Digest }  // layout_digest: the commitment_id

pub struct StreamingCommitter<'layout, 'work, M: MaskSource + ?Sized> { /* one block and chain state */ }
impl<'layout, 'work, M: MaskSource + ?Sized> StreamingCommitter<'layout, 'work, M> {
    pub fn new(params: CommitmentParams, layout: &'layout AdviceLayout, segment: u32,
               encoder: &'work RowEncoder, mask_rng: &'work mut M) -> Result<Self, PcsError>;
    pub fn push_row(&mut self, row_index: u64, values: &[Fp]) -> Result<(), PcsError>; // must follow layout order
    pub fn finish(self) -> Result<SegmentDigests<'layout>, PcsError>;
}
pub fn merge_segments(parts: &[SegmentDigests<'_>]) -> Result<AdviceCommitment, PcsError>;
```

The CPU implementation borrows the canonical layout's row runs and the encoder's public
precomputed tables. `CommitmentParams.commitment_id` is the descriptor hash defined by
RFC-0031; the caller derives it from the statement and proof unit. `MaskSource` supplies the
caller's per-proof PRG stream. A refused push draws nothing, and finishing a segment releases
that borrow so the next segment can consume its prescribed portion. Only public digests and
layout identity survive `finish`. A merge refuses missing, repeated, reordered or mixed-context
parts before hashing; a stack frontier produces the root without retaining a Merkle tree. The
[CPU allocation report](../perf/streaming-advice-commitment.md) measures the one-block bound
and accounts separately for the borrowed encoder tables and retained segment digests.

### Streaming procedure (witness pass)

For each non-zero row in layout order within the segment: regenerate the row's values from the witness engine; draw the row's mask coefficients from the per-proof mask stream (a PRG keyed by the per-proof secret seed, so a restart reproduces them, [RFC-0018](RFC-0018-gpu-prover-runtime.md#crash-consistency)); encode; append the codeword to the block buffer; every `B` rows, advance all column chains and clear the buffer. Memory: one block buffer, `B · n · 8` bytes (2 GiB at `k = 2^22`, sharded by column across a GPU's memory), plus `32 · n` bytes of chain state per segment.

For a segment of fewer than `B` committed rows the CPU block holds just those rows, and an
empty segment has no block. The encoder tables, input row and mask source are shared or
caller-owned baseline state. [The allocator measurements](../perf/streaming-advice-commitment.md)
separate that baseline, the per-worker heap, retained segment chains and the merge frontier.

### Opening protocol

After the last stage (RFC-0010):

1. **Grouping.** Claims on `Ã` are grouped by their column sub-point (the low `log2 k` coordinates). In `tessera-protocol/2` RFC-0070's mask claims, every masking polynomial's `G(r)`, the tree levels' masks and the `SoftmaxRow` leaf mask, are not among them: [RFC-0070](RFC-0070-mask-batching.md)'s batching opens them, and its claim `y` joins the pool, the last group. A claim may be a combination `Σ_t a_t·X̃_t(r)` of committed tensors at one point ([RFC-0037](RFC-0037-stage-schedule.md#the-claim-pool)); its pieces share `r`'s column sub-point, and step 2 adds `a_t·eq(row part of (r ‖ base_t), row)` for each. Claims from one stage on same-shaped tensors share a column sub-point, so the number of groups `G` is about the number of stages; the compiler's report lists `G` (target `≤ 16`; in `tessera-protocol/2`, [RFC-0068](RFC-0068-leaf-stage-zero-knowledge.md) adds a group for the column masks per distinct mask width `c_I`, one at 405B, [RFC-0013](RFC-0013-zero-knowledge.md#masked-extensions) 'Virtual tensors' at most one per reduction stage for its masks' folds, two at 405B, and [RFC-0070](RFC-0070-mask-batching.md) one for the masks' batching, whose claim `y` opens every masking polynomial's `G(r)`, every tree-level mask claim and the `SoftmaxRow` leaf mask's, where `G ≤ 20`).
2. **Row coefficients.** For group `g`, `c_g(row) = Σ_{claims in g} γ_claim · eq(row-part of point, row)`. Claim `t` of the group, counted from 0 in the order the group lists its claims, takes `γ_claim = γ^{t+1}`, with `γ` squeezed under `pcs/advice/gamma` (RFC-0014). No claim takes the coefficient 1, which the blinding claim of [RFC-0013](RFC-0013-zero-knowledge.md#hiding-openings) takes. In `tessera-protocol/2` a claim on a committed advice tensor that RFC-0013 masks is a claim on its masked extension at point `p`: a graph tensor, a gadget's auxiliary column (RFC-0050), a table's multiplicities (RFC-0032) or the greedy maximum `M` (RFC-0053). Claims on mask rows, the masks' pad `S` and blinding rows are not: a fold's `u` (RFC-0013 'Virtual tensors'), RFC-0068's `u_i` and RFC-0070's `y` read their rows as they are. For each piece of such a claim, its mask row `R_V`'s two committed rows take `γ_claim·a·Z(p)` and `γ_claim·a·Z(p)·X`, `a` the piece's coefficient, once per piece whatever its layer and tile images. `p` is the piece's point on its kind: `(r ‖ ρ)` for a stacked piece of layer weights `eq(ρ, ·)`, and `r` for a kind without layers or for a `Layers::One` piece, which takes its layer tensor's own row ([RFC-0037](RFC-0037-stage-schedule.md#the-claim-pool), [RFC-0069](RFC-0069-one-layer-claims.md)). `Z` is never taken at a layer image's point `(r ‖ ℓ)`, where it vanishes at the Boolean layer. An implementation that splits a stacked piece into per-layer sources weighted `eq(ρ, ℓ)`, as the prover's `tessera-protocol/1` native openings do, splits its data rows only, and adds the mask term once, at `(r ‖ ρ)`. A tiled piece's images likewise each add their data rows, and the mask term enters once. A piece whose layer weights carry per-layer constants has no mask defined yet ([#556](https://github.com/AbdelStark/tessera/issues/556)). The mask's column functional is the claim's own, so it joins the data's group and adds none ([RFC-0013](RFC-0013-zero-knowledge.md#masked-extensions) 'Committed tensors', [#554](https://github.com/AbdelStark/tessera/issues/554)). Both sides compute `c_g` from public data; it is non-zero only on rows of the claimed tensors and their mask rows.
3. **Proximity coefficients.** `ξ_row`, drawn from the transcript after Round 1 (label `pcs/advice/xi`), defined as `eq(ξ*, row)` for a random point `ξ*`.
4. **Opening pass.** The prover regenerates every non-zero row once more and accumulates `u_g = Σ_row c_g(row) · A_row` for all groups and `u_ξ = Σ_row ξ_row · A_row` (message space, over `Fp2`), plus the combined mask polynomials `M_g = Σ_row c_g(row) m_row` and `M_ξ`.
5. **Send.** `u_g`, `M_g` for every group in group order, then `u_ξ`, `M_ξ` (label `pcs/advice/combined`): `G + 1` combined rows, each a `u` of `k` values in `Fp2` with an `M` of `P_A` coefficients.
6. **Evaluation check.** The verifier checks that the combined rows have that shape. Then, for each group in order, it checks `Σ_{claims in g} γ_claim · v_claim = ⟨u_g, eq(column sub-point)⟩`. Either failure is `Reject(Evaluation)`.
7. **Column queries.** The verifier draws `τ_A` column indices uniformly and independently, with replacement (live: verifier's choice after receiving step 5, sent in draw order; non-interactive: transcript with grinding, RFC-0014), so an index can repeat. The prover answers each distinct drawn index once, in increasing order. No codeword outlives the witness pass, so the prover regenerates every non-zero row once more and evaluates its codeword at those indices ([RFC-0018](RFC-0018-gpu-prover-runtime.md#pass-structure) P5). The answer does not carry the indices, which the verifier knows from its draw:
   - for each distinct index, every non-zero row's symbol in that column, in segment order;
   - the canonical Merkle multiproof of that set ([RFC-0004](RFC-0004-hashing-and-transcript.md#merkle-trees)).
8. **Column check.** In this order:
   1. The answer holds exactly one column per distinct queried index, each with one symbol per non-zero row.
   2. The verifier recomputes `leaf_j` from each column and verifies the multiproof of the distinct indices against the root.
   3. For each distinct queried `j`, in increasing order, it checks `Σ_row c_g(row) · col_j[row] = enc(u_g)(x_j) + Z_H(x_j) · M_g(x_j)` for every `g`, and the same for `ξ`. Here `enc(u)(x)` is the polynomial of degree `< k` that takes the values `u` on `H`, evaluated at `x`, and the sum runs over the non-zero rows.

   A failure of 1 or 2 is `Reject(MerklePath)` at the smallest queried index: a multiproof fails as a whole, never at one column. A failure of 3 is `Reject(Proximity)` at the first `j` that fails.

Soundness: if the committed rows are `δ`-far from the code for `δ < (1 − ρ)/2`, the proximity check passes with probability at most `(1 − δ)^{τ_A} + ϑ·n/|Fp2|`, where `ϑ = ν − log2 k` is the row index's bits. The coefficients of step 3 are a tensor, `eq(ξ*, row)`, whose `ϑ` coordinates each cost one line proximity gap (Diamond–Posen; Diamond–Gruen, Theorems 2 and 3; [RFC-0014](RFC-0014-interaction-and-soundness.md#soundness-budget)); otherwise each row decodes uniquely and the evaluation checks bind every claim to the decoded rows (Ligero, Brakedown). The bound takes the `τ_A` draws to be independent, which drawing with replacement makes them. Each draw lands outside the agreeing columns with probability at least `δ`, whether or not it repeats an earlier one, so checking a repeated column once loses nothing. Conservatively each query gives `−log2((1 + ρ)/2) ≈ 0.415` bits ([RFC-0014](RFC-0014-interaction-and-soundness.md#parameter-sets)).

### Sizes at the default parameters (16K tokens, 16K context, 405B)

Canonical `servable`, two shift classes, Q=512, including RFC-0050's complete columns. These are layout counts and byte sizes; timings require separate measurements.

| Quantity | Value |
|---|---|
| Committed row entries, padding included | 953,580 × 2^22 ≈ 4.0×10^12 |
| Non-zero rows | 953,580 |
| Allocated rows, gaps included | 2,328,576 (`ϑ = 22`) |
| Opened columns (`τ_A = 155`) | 155 × 953,580 × 8 bytes ≈ 1.18 GB |
| Combined rows (`G + 1 ≤ 17`; `≤ 21` in `tessera-protocol/2`, with RFC-0068's group, RFC-0013's fold groups of the two reduction stages and RFC-0070's batching group) | ≤ 17 × 2^22 × 16 bytes ≈ 1.14 GB; ≤ 1.41 GB |
| Merkle multiproof | < 0.2 MB |
| Verifier work | hash 1.18 GB of columns; `(G + 1) · τ_A` inner products of length 953,580; `G + 1` coset NTTs of size 2^23 over `Fp2` |

### Invariants

- **INV-PCS-1.** Layout is a function of the claim list, the batch descriptor and the anchor's inputs from the logs the statement's manifest names (draft [RFC-0074](RFC-0074-anchor-binding.md#the-advice-layout-and-the-openings)) only.
- **INV-PCS-2.** Rows are pushed in layout order; out-of-order pushes are rejected by `StreamingCommitter`.
- **INV-PCS-3.** Column digests are chained per segment and merged per leaf; a column's leaf depends on every non-zero row's symbol in that column and on nothing else.
- **INV-PCS-4.** Mask coefficients are fresh per proof and never reused across commitments.

### Failure modes

| Failure | Response |
|---|---|
| Honest prover's opening pass regenerates a different row than the witness pass | `ERR-INTERNAL` caught by a self-check: the prover recomputes one random column's leaf before sending |
| Combined rows of another count or length | `Reject(Evaluation)` |
| Group evaluation mismatch | `Reject(Evaluation)` |
| Answer with another number of columns, or a column of another length | `Reject(MerklePath)` |
| Multiproof invalid | `Reject(MerklePath)` |
| Combined row not consistent with columns | `Reject(Proximity)` |

## Alternatives Considered

1. **Storing the encoded witness.** Rejected: terabytes per proof at 16K context.
2. **FRI-based multilinear commitments (BaseFold, WHIR).** Considered: smaller proofs (polylogarithmic) and faster verification. Rejected for v1: the prover must run a folding protocol over the entire committed vector after commitment, which needs either the vector resident or many regeneration passes; the tensor scheme needs two, one to combine rows and one to answer the column queries. BaseFold/WHIR remain candidates for recursive compression (NG-3).
3. **Brakedown's linear-time codes.** Considered for encoding speed. Rejected: NTT-based RS encoding is fast on GPUs, RS proximity analysis is the best understood, and the source assumes RS at rates 1/2 and 1/4.
4. **Systematic encoding.** Rejected: opened systematic symbols are witness entries, which would need a separate hiding layer.
5. **Per-tensor commitments.** Rejected: thousands of roots, openings and Merkle trees instead of one.
6. **Padding tensors in place to powers of two.** Rejected: up to 23% wasted commitments for 53,248-wide MLP tensors; ordering padded dimensions most significant turns all padding into skipped zero rows.

## Drawbacks

- Every opening needs two full regeneration passes of the witness, each about one native forward pass of extra compute (RISK-9 in [00 §9](../spec/00-overview.md#9-risks)): step 4 forms the combined rows, and step 7 evaluates every row's codeword at the queried columns, which are drawn only after the combined rows are sent ([RFC-0018](RFC-0018-gpu-prover-runtime.md#pass-structure) P5).
- Combined rows dominate proof size when many column sub-points exist; the compiler must keep `G` small.
- Proof size is linear in `τ_A` times the number of non-zero rows.

## Migration / Rollout

v0.1 delivers the encoding, chained digests, Merkle tree and a non-streaming reference commit and open, plus the GPU encode-and-hash microkernel for the unit-cost measurement. v0.3 delivers the canonical layout, streaming committer and opening protocol on CPU. v0.5 moves both to the GPU.

## Testing Strategy

- Encoding: codewords of random rows lie in the code (degree check by interpolation on random subsets); the mask vanishes on `H`; CPU and GPU codewords are identical.
- Streaming equals batch: the streaming committer and a reference that holds the whole matrix produce the same root for random layouts, including zero rows, partial final blocks and `S ∈ {1, 2, 8}`.
- Layout: prover and verifier derive identical layouts for conformance models; every tensor's claims map to the right row offsets.
- Openings: random claims at 1–40 points in up to 16 groups verify; one altered element of `u_g`, `M_g`, a column symbol or a multiproof sibling is rejected with the expected reason.
- Repeated queries: a draw that repeats an index is answered with each distinct column once, in increasing order, and verifies. Two other answers are rejected with `Reject(MerklePath)`: the distinct columns in draw order, from a draw that is not already increasing, and the drawn columns with the repeated one sent twice.
- Proximity: a matrix with one row replaced by a non-codeword at relative distance `δ ∈ {0.05, 0.25, 0.5}` is rejected at an empirical rate at least `1 − (1 − δ)^τ` over 10^4 trials at small `τ`.
- Self-check: a fault-injected opening pass that perturbs one row triggers `ERR-INTERNAL` before any message is sent.

## Open Questions

None.

## References

- Source §4 table (fixed Ligero row width), §6 principle 9, §7 ("Per-proof commitments stream"), §8.
- Ames, Hazay, Ishai, Venkitasubramaniam, "Ligero", CCS 2017.
- Golovnev, Lee, Setty, Thaler, Wahby, "Brakedown", CRYPTO 2023.
- Ben-Sasson, Carmon, Ishai, Kopparty, Saraf, "Proximity gaps for Reed–Solomon codes", FOCS 2020.
- Diamond, Posen, "Proximity Testing with Logarithmic Randomness", ePrint 2023/630, 2024.
- Diamond, Gruen, "Proximity Gaps in Interleaved Codes", ePrint 2024/1351, 2025.
