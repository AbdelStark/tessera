# RFC-0012: Persistent Weight Commitment Opened by the Folding Pass

- Status: Accepted
- Authors: @AbdelStark
- Created: 2026-10-15
- Target milestone: v0.3 (binding), v0.4 (hiding budget and refresh)
- Amended by: [RFC-0026](RFC-0026-claim-list-layout.md) ('Layout': the fields of `WeightLayoutParams`), [RFC-0031](RFC-0031-commitment-id.md) ('Layout': each matrix's `commitment_id` and row block), [RFC-0040](RFC-0040-weight-commitment-artifacts.md) ('Layout': wide inputs, positions, committed rows and leaves; 'Budget state machine': the state's fields), [RFC-0049](RFC-0049-weight-proximity.md) ('Opening by the folding pass': the proximity row's coefficients and when the fold computes it), [RFC-0052](RFC-0052-channel-params-claims.md) ('Opening by the folding pass': opening several claims on `ChannelParams` in one pass), [RFC-0065](RFC-0065-masked-channel-params.md) ('Layout' and 'Column answers': `ChannelParams`' own blinding rows `B_CP`, and the six rows' order `β_T`, `R_W`, proximity pair)
- Superseded in part by: [RFC-0064](RFC-0064-weight-linking-proof.md) ('Refresh and linking proof': the linking proof opens the difference of the two commitments; 'Failure modes': the link's four rows; 'Budget state machine': epoch `e + 1` starts with the link's rows and columns spent), [RFC-0065](RFC-0065-masked-channel-params.md) ('Budget state machine': `reserve` takes `2·e_T + 4` rows of a matrix with `e_T` claims, 18 for `ChannelParams` at `e = 7`)

## Summary

The weights of `D` are committed once per epoch in ten Ligero-family matrices (one per weight type plus one for hidden channel parameters) at rate 1/4, under a single root `R_W`. Rows are (layer, high output bits) and columns are (low output bits, input), so the combined row an opening reveals is a partial fold of the weights that the prover already computes for the matmul sumcheck. One pass over the weights per proof serves both. Hiding comes from vanishing masks with `P_W = 2^16` random coefficients per row and one fresh blinding row per revealed combination. The prover answers column queries from a store of the epoch's codewords. A budget state machine tracks opened columns and blinding rows; when the budget runs out, the prover refreshes the commitment and publishes a linking proof.

## Motivation

The source keeps VerInf's "persistent weight commitment, opened-column budget, refresh with a linking proof" but restructures it so that "its opening reuses the prover's own weight pass" (source §4 table, "Keep, restructure"; §6 principle 7). For 405B, one pass over 405 GB of INT8 weights takes about 15 ms of HBM time on 8×H100 and is independent of token count (source §5, "Weights"). The commitment must be binding for soundness and hiding for the treaty setting, in which the verifier sees hundreds of openings of the same weights.

## Goals

- One root `R_W` per epoch binding every weight and channel parameter.
- One weight pass per proof that yields both the sumcheck fold and the opening's combined rows.
- Perfect hiding of weights across all openings within an epoch, enforced by a persistent budget.
- Refresh that preserves the verifier's trust through a linking proof.

## Non-Goals

- Hiding the architecture (NG-2).
- Mixture-of-experts weight layouts (NG-1).
- Commitment cost: committing is offline, once per epoch.

## Proposed Design

### Layout

| Matrix | Logical shape per layer | Rows | Columns |
|---|---|---|---|
| `Embed` | `d_model × vocab` (transposed embedding) | `(o_hi)` | `(o_lo, v)` |
| `Q`, `K`, `V`, `O`, `Gate`, `Up`, `Down` | `out × in` | `(layer, o_hi)` | `(o_lo, i)` |
| `LmHead` | `vocab × d_model` | `(o_hi)` | `(o_lo, i)` |
| `ChannelParams` | per (type, layer): one value per output channel, its class index or affine multiplier | `(type, layer_hi)` | `(layer_lo, channel)` |

- Dimensions are padded to powers of two; padded output rows and a per-layer matrix's padded layers are structurally zero and skipped, as in RFC-0011. [RFC-0040](RFC-0040-weight-commitment-artifacts.md#the-matrices) gives every entry's position and the committed rows.
- `o_lo` has `a_T` bits, chosen so the message width `k_T = 2^{a_T} · in_pad` is `2^20` for 405B shapes (`a = 6` for 16,384-wide inputs, `a = 4` for the 53,248-wide down projection padded to 65,536, `a = 3` for the 2^17-wide embedding). Smaller models use `k_T = max(in_pad, min(2^20, out_pad · in_pad))`, so that a row holds at least one input row ([RFC-0040](RFC-0040-weight-commitment-artifacts.md#the-matrices)).
- `ChannelParams` holds the tensor `[types, L_pad, C]`, `L_pad` the next power of two of `n_layers`, flattened row-major ([RFC-0010](RFC-0010-proof-schedule.md#multilinear-extensions)):
  - the types in this table's order, `Embed` first, 9 padded to 16;
  - `C`, the largest padded output width among the weights that have a `ChannelParams` ([03 §2](../spec/03-data-model.md#quantization)): `2^16` at 405B, Gate's and Up's, and 1 when no weight has one;
  - entry `(T, ℓ, c)`: channel `c`'s class index if `T`'s outputs reach a site of several classes, its multiplier `M_c` if they reach an affine site ([RFC-0025](RFC-0025-affine-sites.md)), and 0 for padding, for a channel past `T`'s width and for a weight with no `ChannelParams`.

  Its message width is `k_W`, as every 405B matrix's is: the low `log2 k_W` bits of the flattened index are its columns, the rest its row. A narrower message would raise its rate `(k + P_W)/n_T` above the one `τ_W` is computed for. At `k = 2^16` with `P_W = 2^16` the rate is 1/2, and `live-64`'s 97 queries would give about 40 bits instead of 64. At 405B a row holds 16 layers' values of one type. Each type that has a `ChannelParams` fills 8 rows, and the other types' rows are zero and skipped. At 8B, with `C = 2^{12}` when only `O` and `Down` have a `ChannelParams`, the tensor holds `2^{21}` entries, and a row holds all 32 layers of 8 types. A model whose tensor is smaller than `k_W` takes `k = min(k_W, types · layers · C)` after padding, as the other matrices take `min(2^20, out_pad · in_pad)`.
- Rate 1/4: codewords of length `n_T = 4 k_T` on the coset `7·L_T`.
- Each row's encoding carries a vanishing mask with `P_W = 2^16` random coefficients (RFC-0011 encoding), so the effective rate is `ρ' = (k_T + P_W)/n_T ≈ 0.266`.
- Each matrix has `B_W = 4096` blinding rows: random message rows appended after the data rows, committed with them, each used at most once. RFC-0013's masks are over `Fp2`, so each blinding row it uses is two of these. A proof consumes six per matrix: two for the evaluation combined row, two for the proximity combined row, and two as the mask row of the masked extension ([RFC-0013](RFC-0013-zero-knowledge.md#masked-extensions)), reserved in the order evaluation pair, mask row, proximity pair (RFC-0065 'The reservation'). `ChannelParams`, with `e` claims, consumes `2e + 4`: one pair for its evaluation rows, a mask row per claim and the proximity pair. It has `B_CP` blinding rows ([RFC-0065](RFC-0065-masked-channel-params.md)).
- Each matrix chains its columns over blocks of the parameter set's row block `B` ([RFC-0014](RFC-0014-interaction-and-soundness.md#parameter-sets)), from its own `commitment_id`: the hash of its `WeightMatrix` descriptor, which holds `WeightLayoutParams`, the matrix, its message width, `B`, its rows and the epoch ([RFC-0031](RFC-0031-commitment-id.md#the-descriptor)).
- Each column is one chain over the matrix's committed rows, `leaf_j = d_j[⌈m/B⌉]` ([RFC-0004](RFC-0004-hashing-and-transcript.md#column-digests)): a matrix has no segments and no `column-merge`.
- `R_W = H("weights-root", TCE(WeightLayoutParams) ‖ root_Embed ‖ … ‖ root_ChannelParams ‖ u32(epoch))`, the hash of `TCE(WeightCommitmentPublic)` ([RFC-0040](RFC-0040-weight-commitment-artifacts.md#weightcommitmentpublic)). The first commitment is epoch 0. `WeightLayoutParams` holds `log2 k_W`, the inverse rate's `log2`, `P_W` and `B_W`, the parameter set's values; the shapes and padding follow from the architecture ([RFC-0026](RFC-0026-claim-list-layout.md)).

### Opening by the folding pass

After the `MatMulWeight` stage of RFC-0010, weight type `T` has one claim `W̃_T(ρ_ℓ, r_o, r_i)`, where the layer index is a sumcheck variable ([RFC-0010](RFC-0010-proof-schedule.md#batching-within-a-stage)):

1. **Fold.** For each layer, the prover computes the partial fold over high output bits, `F_ℓ(o_lo, i) = Σ_{o_hi} eq(r_o^{hi}, o_hi) · W_ℓ[(o_hi, o_lo), i]`. This is the single pass over the weights. It runs as limb-decomposed INT8 GEMMs on tensor cores: each `Fp2` coefficient of `eq` splits into sixteen 8-bit limbs per base coordinate, and the products are recombined exactly.
2. **Sumcheck input.** The matmul sumcheck over `(ℓ, i)` needs `W̃(ℓ, r_o, i) = Σ_{o_lo} eq(r_o^{lo}, o_lo) F_ℓ(o_lo, i)`, which costs `2^{a} · in` per layer from `F_ℓ`.
3. **Combined row.** Let `β_T = β_T^{(0)} + X·β_T^{(1)}` be the next unused `Fp2` blinding row, two committed rows ([RFC-0013](RFC-0013-zero-knowledge.md#hiding-openings)).
   - **Blinding claim.** The prover sends `q_T = ⟨β_T, eq(r_o^{lo}) ⊗ eq(r_i)⟩` under `pcs/weights/{T}/blinding`, and `θ_T` is then squeezed under `pcs/weights/{T}/gamma` (RFC-0014, row 7).
   - **Check.** The opening reveals `u_T = θ_T·Σ_ℓ eq(ρ_ℓ, ℓ) F_ℓ + β_T`. The verifier checks `θ_T·v_T + q_T = ⟨u_T, eq(r_o^{lo}) ⊗ eq(r_i)⟩`, and at each queried column it gives the data rows their coefficients scaled by `θ_T`, and the blinding rows `1` and `X`.
   - **Soundness.** `v_T` and `q_T` are fixed before `θ_T`, so a false `v_T` passes with probability at most `1/|Fp2|`: `θ_T·(v_T − f_T) + (q_T − Q̃_T)` vanishes for at most one `θ_T`. With no challenge between them, any split `(v_T + Δ, q_T − Δ)` would pass ([RFC-0013](RFC-0013-zero-knowledge.md#hiding-openings)).
4. The fold `Σ_ℓ eq(ρ_ℓ, ℓ) F_ℓ` does not depend on `r_i`. It is formed as soon as the layer rounds fix `ρ_ℓ`, from the per-layer folds `F_ℓ` retained in device memory (`n_layers · 2^20` elements of `Fp2` per type, about 2 GiB at 405B), so one pass over the weights per proof suffices. Scaling it by `θ_T` and adding `β_T` once `θ_T` is drawn costs one pass over its `k_T` values. For `ChannelParams`, which has `m ≥ 1` claims in the pool (one per reading site), the single pass over its committed rows accumulates all `m` evaluation folds `F_i` and the proximity row together, and `WeightCombinedRows` carries one evaluation row per claim in pool order ([RFC-0052](RFC-0052-channel-params-claims.md)).

Proximity uses a second combined row and its own `Fp2` blinding row. In `tessera-protocol/1` its point `ξ*_T` is squeezed right after the seed, its coefficients index the committed rows by position, and the fold pass computes it beside the folds ([RFC-0049](RFC-0049-weight-proximity.md#the-proximity-row)). Its coefficients are the tensor of [RFC-0011](RFC-0011-advice-commitment.md#opening-protocol) step 3, `eq(ξ*_T, row)` for a random point `ξ*_T` with one coordinate per bit of the matrix's row index, data and blinding rows together (label `pcs/weights/{T}/xi`). Its field term is `ϑ_T·n_T/|Fp2|`, with `ϑ_T` those bits ([RFC-0014](RFC-0014-interaction-and-soundness.md#soundness-budget)). Column queries (`τ_W` per matrix) are drawn and answered as [RFC-0011](RFC-0011-advice-commitment.md#opening-protocol) step 7 says: with replacement, each distinct queried column once and in increasing index order. Each answered column is the full column, all non-zero data rows and all blinding rows, followed by the canonical multiproof of the same set. 'Column answers' says where the prover reads them.

### Column answers

The weights' codewords stay fixed for an epoch: their masks and blinding rows come from `masks_seed`, not from the proof. So a proof can read its column answers from a store. An advice row cannot be answered that way, because its codewords exist only during the witness pass ([RFC-0018 'Column answers'](RFC-0018-gpu-prover-runtime.md#pass-structure)). Both methods below send the same bytes, so the verifier is unaffected.

- **The codeword store, the default.** When the prover commits an epoch, it writes each matrix's codewords column-major. Each column's symbols are contiguous: the non-zero data rows in layout order, then the blinding rows. The matrix's Merkle tree is written beside them. A proof reads its distinct queried columns and their multiproof, and checks them against the matrix's root before sending them. A part that fails is recomputed, a column by evaluation, so a damaged store costs time, not a rejected proof.
  - **Size.** At 405B the store holds 453,126 rows of `2^22` symbols, about 15.2 TB per epoch: the nine weight matrices' non-zero rows, `ChannelParams`' at most 882 at the common width, and 40,960 blinding rows. In `tessera-protocol/2`, [RFC-0065](RFC-0065-masked-channel-params.md)'s `B_CP` adds 8,184 rows for `ChannelParams`' claims: 461,310 rows, 15.48 TB. `ChannelParams`' data rows are 56 at 405B, 8 for each of its seven types ('Layout'). 882 overcounts them: the layout allows at most 72, 8 for each of nine types.
  - **Reads.** A proof reads 0.35 GB at `live-64` (`τ_W = 97`) and 0.44 GB at `ni-100` (122), and 0.36 GB and 0.45 GB with RFC-0065's rows. The reads start with the queries and overlap P5, which regenerates the advice witness after the same queries, so they take no GPU time.
  - **Secrecy.** The store is secret-class: its symbols determine the weights, the masks and the blinding rows. It is written with mode 0600, as `WeightCommitmentState` is ([06 §7](../spec/06-security.md#7-secrets-handling)). It leaves the prover only as answered columns.
  - **Refresh.** A refresh writes the next epoch's store while the old one serves the linking proof, and deletes the old one afterwards, so a refresh needs twice the space.
- **Evaluation of the queried symbols, the fallback** for a node without the store. Symbol `j` of row `r` is `⟨A_r, w_j⟩ + Z_H(x_j) · m_r(x_j)`, the same evaluation RFC-0018 'Column answers' applies to advice rows. It is one product of the rows with the `τ'` vectors of barycentric weights `w_j`, limb-decomposed on tensor cores:
  - 8 INT8 products per element and column for a weight row;
  - 64 for a blinding row's elements and for every row's `P_W` mask coefficients.

  At 405B those products come to about 1.3 GPU-s at `τ_W = 97` and 1.7 at 122, as operation counts. Unless the prover keeps the masks and blinding rows, it also regenerates them from `masks_seed`: 581 GB of PRG output per proof, and 654 GB with RFC-0065's rows.

### Budget state machine

```rust
pub struct WeightCommitmentState {          // schema 0x0302, secret, persisted with mode 0600 (RFC-0040)
    pub public: WeightCommitmentPublic,     // layout, the ten roots, epoch
    pub masks_seed: [u8; 32],               // secret-class; regenerates masks and blinding rows
    pub budget: [TypeBudget; 10],           // in WeightMatrix order
}

pub struct TypeBudget {                      // its limits are the layout's P_W and the matrix's B_T (RFC-0065)
    pub opened_columns: Vec<u32>,           // distinct column indices revealed this epoch, increasing
    pub next_blinding_row: u32,             // rows below this index are burned
}
```

Transitions, each persisted (fsync, atomic rename) before the corresponding message is sent:

| Event | Precondition | Effect |
|---|---|---|
| `reserve` (before `Round1`) | For every matrix: `next_blinding_row + r_T ≤ B_T − 4` and `\|opened_columns\| + τ_W ≤ P_W − τ_link`, with `r_T = 2·e_T + 4` for the matrix's `e_T` claims and `B_T` its blinding rows ([RFC-0065](RFC-0065-masked-channel-params.md)). Blinding rows `B_T − 4` to `B_T − 1` and `τ_link` columns of headroom are reserved exclusively for the linking proof | Burns `r_T` blinding rows per matrix, six for a single-claim matrix; records them in `Round1` |
| `record_queries` (before `Round3Response`) | Queries are those of the current session | Adds the distinct query indices to `opened_columns` |
| `abort` | Any time | Nothing is released: burned rows and opened columns stay burned |
| `refresh` | Any time, once per epoch; required when `reserve` fails | Produces epoch `e + 1` and a `WeightLinkProof`, through `link` |
| `link` | Epoch `e` has no link; see [RFC-0064](RFC-0064-weight-linking-proof.md#one-link-per-epoch) | Burns epoch `e`'s last four rows, records the link's columns, closes epoch `e`, and writes epoch `e + 1`'s state from row 4 |

`reserve` failing yields `ERR-WITNESS-BUDGET`. At the defaults, an epoch supports about 670 live proofs: `(B_W − 8)/6 = 681` by blinding rows, since the linking proofs take four rows at each end of an epoch, and `(P_W − τ_link)/τ_W ≈ 674` by columns in epoch 0. A later epoch also spends `τ_link` columns at its start on its link, which leaves 673 ([RFC-0064](RFC-0064-weight-linking-proof.md#the-budget)). The verifier tracks the blinding-row indices declared per epoch and rejects reuse with `Reject(Budget)`; reuse would only hurt the prover, but rejecting it makes prover bugs visible.

### Refresh and linking proof

[RFC-0064](RFC-0064-weight-linking-proof.md) supersedes this section. Its linking proof opens the difference of the two commitments against the public claim 0, with a blinding pair from each epoch, and keeps each commitment's proximity row. It also defines the transcript, `τ_link`, the message and the verifier's checks. The text below, which opened each commitment at `ρ_T` and revealed `W̃_T(ρ_T)`, is kept for the record.

1. Commit the same weights with a fresh `masks_seed` as epoch `e + 1`, root `R_W'`.
2. Derive a random point `ρ_T` per matrix from a transcript over `(R_W, R_W', epoch)`.
3. Open both commitments at `ρ_T` and show equal values. Each opening is one evaluation per matrix, with an evaluation and a proximity combined row, each blinded by an `Fp2` row: the last four blinding rows of epoch `e` and the first four of epoch `e + 1`, whose proofs then start at row 4.
4. The verifier accepts `R_W'` as the successor of `R_W` if every pair is equal and every opening verifies. Two multilinear polynomials over `ν` variables that differ agree at a random point with probability at most `ν/|Fp2|`.

```rust
pub struct WeightLinkProof { pub old_root: Digest, pub new_root: Digest, pub epoch: u32,
                             pub openings_old: Vec<WeightOpening>, pub openings_new: Vec<WeightOpening> }
```

### Invariants

- **INV-WC-1.** Within an epoch, no blinding row is revealed twice and at most `P_W` distinct columns per matrix are revealed.
- **INV-WC-2.** Budget state is persisted before any message that consumes budget is sent.
- **INV-WC-3.** `R_W` binds the layout, every matrix root and the epoch.
- **INV-WC-4.** One weight pass per proof produces both the sumcheck folds and the combined rows.

### Failure modes

| Failure | Response |
|---|---|
| Budget exhausted | `ERR-WITNESS-BUDGET`; operator refreshes |
| State file lost or corrupted | The prover refreshes before the next proof, from the model's refresh key `K`, which gives every epoch's `masks_seed`; without it, it starts a new model entry ([RFC-0064](RFC-0064-weight-linking-proof.md#one-link-per-epoch)). Remade, the link is the same bytes. The linking proof uses only the four blinding rows and the column headroom reserved for linking at each end of the epoch ([RFC-0064](RFC-0064-weight-linking-proof.md#the-budget)), so it stays within the hiding budget even though the lost state's exact contents are unknown |
| Combined row inconsistent with columns | `Reject(Proximity)` |
| The linking proof fails ([RFC-0064](RFC-0064-weight-linking-proof.md#the-verifiers-checks)) | Verifier rejects the refresh; the old root stays current |

## Alternatives Considered

1. **One matrix for all weight types.** Rejected: types differ in input width, so a common row width wastes columns, and separate matrices let each type open at its own common point.
2. **Rows (layer, input), columns (output).** Considered: matches a column-major weight layout. Rejected: the combined row would then be a fold over inputs, which the matmul sumcheck does not use, so the opening would need a second weight pass.
3. **Hiding by committing weights with random padding columns only (no blinding rows).** Rejected: combined rows reveal linear combinations of weight rows, which padding alone does not hide.
4. **A fresh weight commitment per proof.** Rejected: committing 405 GB at rate 1/4 per proof costs far more than the proof's weight pass and loses the persistent anchor parties agree on.
5. **Hiding through a ZK-STARK wrapper over the openings.** Rejected for v1 (source §10, zero-knowledge alternative): larger trusted base and recursion.
6. **Answer the column queries by re-encoding every row per proof.** Rejected: each row takes an inverse NTT of size `2^20` and a coset NTT of size `2^22`. At the advice's 30 ps per element, that is about 14 GPU-s per 405B proof unit, +1.3 on `μ(1K)` for a 16K-token batch. Evaluating only the queried symbols costs about a tenth of that and is the fallback; reading the store costs no GPU time ('Column answers').

## Drawbacks

- The epoch length (about 670 proofs) forces periodic refreshes, each requiring a full recommitment.
- Rate 1/4 doubles commitment work against rate 1/2; it is paid once per epoch, not per proof.
- State persistence is security-critical: losing the budget file forfeits the epoch.
- The default column answers need a store of the epoch's codewords: about 15.2 TB at 405B, and twice that during a refresh.

## Migration / Rollout

v0.3: binding commitment, layout and fold-pass opening on CPU (no masks, no blinding rows), with column answers from codewords held in memory. v0.4: masks, blinding rows, budget state machine, refresh and linking proof. v0.5: GPU fold pass with limb-decomposed GEMMs, and the codeword store with evaluation as its fallback.

## Testing Strategy

- Layout: for random small `D`, every weight is recoverable from the committed rows at the documented position; padded rows are skipped.
- Fold equivalence: `Σ_{o_lo} eq(r_o^{lo}, o_lo) F_ℓ(o_lo, i)` equals the direct `W̃_ℓ(r_o, i)` for random points; limb-decomposed GEMM folds equal field-arithmetic folds bit for bit.
- Opening: claims at the common point verify; altering one element of `u_T`, one column symbol or one weight after commitment is rejected.
- Budget: a sequence of proofs exhausts the budget exactly at the predicted count; the next `reserve` fails; an aborted session burns its rows; a crash between `reserve` and `Round1` never leads to row reuse (fault-injection test).
- Refresh: a linking proof between equal weights verifies; between weights differing in one entry it is rejected.
- Column answers: stored and evaluated columns equal the re-encoded rows' symbols at every queried index, repeated and boundary indices included. A store with one corrupted symbol fails the check against the root, and that column is evaluated instead.
- Hiding smoke test: the revealed columns and combined rows of 50 proofs on a tiny model pass the distinguisher suite of [07 §7](../spec/07-testing-strategy.md#7-zero-knowledge-tests).

## Open Questions

- **OQ-0012-1.** Should `B_W` and `P_W` be raised to lengthen epochs, at the cost of larger rows and slower commitment? Owner: @AbdelStark. Resolve at v1.0 from the refresh cost measured on 405B and the proof rate the parties expect.

## References

- Source §4 table (persistent weight commitment), §5 ("Weights"), §6 principles 7 and 10, §7 ("Weights opened by the folding pass").
- VerInf (blinding rows, column budget, refresh with linking proof), as summarized in the source.
- [RFC-0010](RFC-0010-proof-schedule.md), [RFC-0011](RFC-0011-advice-commitment.md), [RFC-0013](RFC-0013-zero-knowledge.md).
