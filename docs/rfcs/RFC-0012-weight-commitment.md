# RFC-0012: Persistent Weight Commitment Opened by the Folding Pass

- Status: Accepted
- Authors: @AbdelStark
- Created: 2026-10-15
- Target milestone: v0.3 (binding), v0.4 (hiding budget and refresh)

## Summary

The weights of `D` are committed once per epoch in ten Ligero-family matrices (one per weight type plus one for hidden channel parameters) at rate 1/4, under a single root `R_W`. Rows are (layer, high output bits) and columns are (low output bits, input), so the combined row an opening reveals is a partial fold of the weights that the prover already computes for the matmul sumcheck. One pass over the weights per proof serves both. Hiding comes from vanishing masks with `P_W = 2^16` random coefficients per row and one fresh blinding row per revealed combination. A budget state machine tracks opened columns and blinding rows; when the budget runs out, the prover refreshes the commitment and publishes a linking proof.

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
| `ChannelParams` | per (type, layer): class index and affine multiplier per output channel | `(type, layer)` | `(channel)` |

- Dimensions are padded to powers of two; padded output rows are structurally zero and skipped, as in RFC-0011.
- `o_lo` has `a_T` bits, chosen so the message width `k_T = 2^{a_T} · in_pad` is `2^20` for 405B shapes (`a = 6` for 16,384-wide inputs, `a = 4` for the 53,248-wide down projection padded to 65,536, `a = 3` for the 2^17-wide embedding). Smaller models use `k_T = min(2^20, out_pad · in_pad)`.
- Rate 1/4: codewords of length `n_T = 4 k_T` on the coset `7·L_T`.
- Each row's encoding carries a vanishing mask with `P_W = 2^16` random coefficients (RFC-0011 encoding), so the effective rate is `ρ' = (k_T + P_W)/n_T ≈ 0.266`.
- Each matrix has `B_W = 4096` blinding rows: random message rows appended after the data rows, committed with them, each used at most once. RFC-0013's masks are over `Fp2`, so each blinding row it uses is two of these. A proof consumes six per matrix: two for the evaluation combined row, two for the proximity combined row, and two as the mask row of the masked extension ([RFC-0013](RFC-0013-zero-knowledge.md#masked-extensions)).
- `R_W = H("weights-root", TCE(WeightLayoutParams) ‖ root_Embed ‖ … ‖ root_ChannelParams ‖ u32(epoch))`.

### Opening by the folding pass

After the `MatMulWeight` stage of RFC-0010, weight type `T` has one claim `W̃_T(ρ_ℓ, r_o, r_i)`, where the layer index is a sumcheck variable ([RFC-0010](RFC-0010-proof-schedule.md#batching-within-a-stage)):

1. **Fold.** For each layer, the prover computes the partial fold over high output bits, `F_ℓ(o_lo, i) = Σ_{o_hi} eq(r_o^{hi}, o_hi) · W_ℓ[(o_hi, o_lo), i]`. This is the single pass over the weights. It runs as limb-decomposed INT8 GEMMs on tensor cores: each `Fp2` coefficient of `eq` splits into sixteen 8-bit limbs per base coordinate, and the products are recombined exactly.
2. **Sumcheck input.** The matmul sumcheck over `(ℓ, i)` needs `W̃(ℓ, r_o, i) = Σ_{o_lo} eq(r_o^{lo}, o_lo) F_ℓ(o_lo, i)`, which costs `2^{a} · in` per layer from `F_ℓ`.
3. **Combined row.** Let `β_T = β_T^{(0)} + X·β_T^{(1)}` be the next unused `Fp2` blinding row, two committed rows ([RFC-0013](RFC-0013-zero-knowledge.md#hiding-openings)).
   - **Blinding claim.** The prover sends `q_T = ⟨β_T, eq(r_o^{lo}) ⊗ eq(r_i)⟩` under `pcs/weights/{T}/blinding`, and `θ_T` is then squeezed under `pcs/weights/{T}/gamma` (RFC-0014, row 7).
   - **Check.** The opening reveals `u_T = θ_T·Σ_ℓ eq(ρ_ℓ, ℓ) F_ℓ + β_T`. The verifier checks `θ_T·v_T + q_T = ⟨u_T, eq(r_o^{lo}) ⊗ eq(r_i)⟩`, and at each queried column it gives the data rows their coefficients scaled by `θ_T`, and the blinding rows `1` and `X`.
   - **Soundness.** `v_T` and `q_T` are fixed before `θ_T`, so a false `v_T` passes with probability at most `1/|Fp2|`: `θ_T·(v_T − f_T) + (q_T − Q̃_T)` vanishes for at most one `θ_T`. With no challenge between them, any split `(v_T + Δ, q_T − Δ)` would pass ([RFC-0013](RFC-0013-zero-knowledge.md#hiding-openings)).
4. The fold `Σ_ℓ eq(ρ_ℓ, ℓ) F_ℓ` does not depend on `r_i`. It is formed as soon as the layer rounds fix `ρ_ℓ`, from the per-layer folds `F_ℓ` retained in device memory (`n_layers · 2^20` elements of `Fp2` per type, about 2 GiB at 405B), so one pass over the weights per proof suffices. Scaling it by `θ_T` and adding `β_T` once `θ_T` is drawn costs one pass over its `k_T` values.

Proximity uses a second combined row with random coefficients and its own `Fp2` blinding row. Column queries (`τ_W` per matrix) are drawn and answered as [RFC-0011](RFC-0011-advice-commitment.md#opening-protocol) step 7 says: with replacement, each distinct queried column once and in increasing index order. Each answered column is the full column, all non-zero data rows and all blinding rows, followed by the canonical multiproof of the same set.

### Budget state machine

```rust
pub struct WeightCommitmentState {          // schema 0x0302, secret, persisted with mode 0600
    pub public: WeightCommitmentPublic,     // R_W, epoch, layout
    pub masks_seed: Secret<[u8; 32]>,       // regenerates masks and blinding rows
    pub budget: [TypeBudget; 10],
}

pub struct TypeBudget {
    pub opened_columns: BitSet,             // distinct column indices revealed this epoch
    pub next_blinding_row: u32,             // rows below this index are burned
    pub max_opened_columns: u32,            // P_W
    pub blinding_rows: u32,                 // B_W
}
```

Transitions, each persisted (fsync, atomic rename) before the corresponding message is sent:

| Event | Precondition | Effect |
|---|---|---|
| `reserve` (before `Round1`) | For every matrix: `next_blinding_row + 6 ≤ B_W − 4` and `\|opened_columns\| + τ_W ≤ P_W − τ_link`. Blinding rows `B_W − 4` to `B_W − 1` and `τ_link` columns of headroom are reserved exclusively for the linking proof | Burns six blinding rows per matrix; records them in `Round1` |
| `record_queries` (before `Round3Response`) | Queries are those of the current session | Adds the distinct query indices to `opened_columns` |
| `abort` | Any time | Nothing is released: burned rows and opened columns stay burned |
| `refresh` | Any time; required when `reserve` fails | Produces epoch `e + 1` and a `WeightLinkProof` |

`reserve` failing yields `ERR-WITNESS-BUDGET`. At the defaults, an epoch supports about 670 live proofs: `(B_W − 8)/6 = 681` by blinding rows, since the linking proofs take four rows at each end of an epoch, and `(P_W − τ_link)/τ_W ≈ 674` by columns. The verifier tracks the blinding-row indices declared per epoch and rejects reuse with `Reject(Budget)`; reuse would only hurt the prover, but rejecting it makes prover bugs visible.

### Refresh and linking proof

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
| State file lost or corrupted | The prover refreshes before the next proof. The linking proof uses only the two blinding rows and the column headroom reserved for linking, so it stays within the hiding budget even though the lost state's exact contents are unknown |
| Combined row inconsistent with columns | `Reject(Proximity)` |
| Linking evaluations differ | Verifier rejects the refresh; the old root stays current |

## Alternatives Considered

1. **One matrix for all weight types.** Rejected: types differ in input width, so a common row width wastes columns, and separate matrices let each type open at its own common point.
2. **Rows (layer, input), columns (output).** Considered: matches a column-major weight layout. Rejected: the combined row would then be a fold over inputs, which the matmul sumcheck does not use, so the opening would need a second weight pass.
3. **Hiding by committing weights with random padding columns only (no blinding rows).** Rejected: combined rows reveal linear combinations of weight rows, which padding alone does not hide.
4. **A fresh weight commitment per proof.** Rejected: committing 405 GB at rate 1/4 per proof costs far more than the proof's weight pass and loses the persistent anchor parties agree on.
5. **Hiding through a ZK-STARK wrapper over the openings.** Rejected for v1 (source §10, zero-knowledge alternative): larger trusted base and recursion.

## Drawbacks

- The epoch length (about 670 proofs) forces periodic refreshes, each requiring a full recommitment.
- Rate 1/4 doubles commitment work against rate 1/2; it is paid once per epoch, not per proof.
- State persistence is security-critical: losing the budget file forfeits the epoch.

## Migration / Rollout

v0.3: binding commitment, layout and fold-pass opening on CPU (no masks, no blinding rows). v0.4: masks, blinding rows, budget state machine, refresh and linking proof. v0.5: GPU fold pass with limb-decomposed GEMMs.

## Testing Strategy

- Layout: for random small `D`, every weight is recoverable from the committed rows at the documented position; padded rows are skipped.
- Fold equivalence: `Σ_{o_lo} eq(r_o^{lo}, o_lo) F_ℓ(o_lo, i)` equals the direct `W̃_ℓ(r_o, i)` for random points; limb-decomposed GEMM folds equal field-arithmetic folds bit for bit.
- Opening: claims at the common point verify; altering one element of `u_T`, one column symbol or one weight after commitment is rejected.
- Budget: a sequence of proofs exhausts the budget exactly at the predicted count; the next `reserve` fails; an aborted session burns its rows; a crash between `reserve` and `Round1` never leads to row reuse (fault-injection test).
- Refresh: a linking proof between equal weights verifies; between weights differing in one entry it is rejected.
- Hiding smoke test: the revealed columns and combined rows of 50 proofs on a tiny model pass the distinguisher suite of [07 §7](../spec/07-testing-strategy.md#7-zero-knowledge-tests).

## Open Questions

- **OQ-0012-1.** Should `B_W` and `P_W` be raised to lengthen epochs, at the cost of larger rows and slower commitment? Owner: @AbdelStark. Resolve at v1.0 from the refresh cost measured on 405B and the proof rate the parties expect.

## References

- Source §4 table (persistent weight commitment), §5 ("Weights"), §6 principles 7 and 10, §7 ("Weights opened by the folding pass").
- VerInf (blinding rows, column budget, refresh with linking proof), as summarized in the source.
- [RFC-0010](RFC-0010-proof-schedule.md), [RFC-0011](RFC-0011-advice-commitment.md), [RFC-0013](RFC-0013-zero-knowledge.md).
