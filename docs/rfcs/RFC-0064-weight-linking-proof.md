# RFC-0064: The Weight Linking Proof Opens the Difference

- Status: Accepted
- Authors: @AbdelStark
- Created: 2026-10-24
- Target milestone: v0.4
- Supersedes:
  - [RFC-0012](RFC-0012-weight-commitment.md#refresh-and-linking-proof) 'Refresh and linking proof': the openings, the message and the transcript;
  - RFC-0012 'Failure modes': the linking proof's rows, four at each end of an epoch, not two;
  - [RFC-0014](RFC-0014-interaction-and-soundness.md#soundness-budget) 'Soundness budget': the weight linking row.
- Amends:
  - RFC-0012 'Budget state machine': a `link` transition, one per epoch; epoch `e + 1` starts with the link's rows burned and its columns opened;
  - [RFC-0014](RFC-0014-interaction-and-soundness.md#grinding) 'Grinding' and 'Transcript schedule': the link's proof of work and labels;
  - [RFC-0043](RFC-0043-rejection-payloads.md): `WeightLinkProof` among the prover's artifacts, and its nonce under `PowPhase::Queries`;
  - [RFC-0013](RFC-0013-zero-knowledge.md) 'Cost': the column budget after epoch 0;
  - [RFC-0004](RFC-0004-hashing-and-transcript.md#hash-function-and-domains)'s domain table: `weights-link` and `weights-refresh`;
  - [RFC-0040](RFC-0040-weight-commitment-artifacts.md) 'masks_seed': every epoch's seed is derived from the model's refresh key, and the seed's expansion is normative;
  - [RFC-0019](RFC-0019-verifier.md#ledger) 'Ledger': one successor per root, and `weights link`'s claim list;
  - [03 §5](../spec/03-data-model.md#5-artifact-catalog): `WeightLinkProof` is defined here.
- Amended by: [RFC-0065](RFC-0065-masked-channel-params.md) (each matrix's own blinding rows `B_T` where this RFC writes `B_W`: `ChannelParams`' are `B_CP`)
- Origin: [#485](https://github.com/AbdelStark/tessera/issues/485), from the simulator argument ([`docs/zk/simulator.md`](../zk/simulator.md) §6.4, requirement R3)

## Summary

A refresh recommits the same weights under fresh masks as epoch `e + 1`, and a linking proof shows the two commitments hold the same weights. RFC-0012 opened each commitment at a random point `ρ_T` and compared the two values. In `tessera-protocol/2` that reveals `W̃_T(ρ_T)`, a random linear functional of the secret weights, per matrix and per refresh, which no simulator can produce (§6.4).

This RFC opens the difference. Per matrix there is one combined row over both commitments: the old rows' coefficients are `eq(ρ_T, ·)`, the new rows' are their negatives, and each epoch contributes a fresh blinding pair. It is checked against the public claim 0 at the same columns of both commitments. Each commitment keeps its own proximity row. The proof reveals nothing about the weights, its soundness is unchanged, and it is smaller than two openings.

The RFC also gives what RFC-0012 left undefined: the transcript, the query count `τ_link` and its proof of work, the message, the verifier's checks, and the rows and columns each epoch spends.

## Motivation

- **The leak.** [#485](https://github.com/AbdelStark/tessera/issues/485); `docs/zk/simulator.md` §6.4, confirmed by an independent reading. Every other value a `tessera-protocol/2` proof reveals is public, masked or determined by values already sent (INV-ZK-1); RFC-0012's linking proof was the exception.
- **The gaps.** RFC-0012 names `τ_link` but never sets it, derives `ρ_T` "from a transcript over `(R_W, R_W', epoch)`" with no labels, and gives `WeightLinkProof` fields (`openings_old`, `openings_new`) whose type no document defines. #153 cannot implement it, nor #158 verify it, without them.
- **A row count.** RFC-0012 'Refresh and linking proof' and 'Budget state machine' reserve four blinding rows at each end of an epoch, and 'Failure modes' says two.

## Goals

- The linking proof is zero-knowledge in the sense of the simulator argument: its view is simulatable from public data (R3).
- Its soundness is RFC-0012's: two different weight sets agree at a random point with probability at most `ν_T/|Fp2|` per matrix.
- Each commitment is still proven close to the code within the linking proof.
- Every message, label, check and budget transition is defined, so that #153 and #158 implement it from this text.

## Non-Goals

- **The proximity row's generator** in `tessera-protocol/2`: its coefficients on the blinding rows, R1 and R2 of the simulator argument. [#394](https://github.com/AbdelStark/tessera/issues/394) settles them for proofs, and the linking proof's proximity rows follow whatever it settles (below).
- **`tessera-protocol/1`**, which commits no masks or blinding rows for the weights and has no refresh ([RFC-0040](RFC-0040-weight-commitment-artifacts.md)).
- **Losing `K`.** The linking proof regenerates epoch `e`'s rows from its seed, which `K` gives. A prover that has lost `K` cannot link and starts a new model entry.

## Proposed Design

### The inputs

- **Two states.** Epoch `e`'s `WeightCommitmentState` and epoch `e + 1`'s, committed from the same weights with equal `WeightLayoutParams` (RFC-0040). Their public parts are `old` and `new`, with roots `R_W = H("weights-root", TCE(old))` and `R_W'` (RFC-0012 'Layout').
- **The seeds are derived.** Each epoch's `masks_seed_e = H("weights-refresh", K ‖ u32(e))`, a new domain of RFC-0004 ('One link per epoch' says why).
  - **`K`** is the model's refresh key, 32 bytes drawn from the operating system's CSPRNG when epoch 0 is committed.
  - **Kept apart.** `K` is secret-class and kept apart from every epoch's state file, in the operator's key store, so that a state file alone gives its own epoch's seed and no other.
  - **One model, one key.** `K` is never shared between model entries. Two models under one `K` would take the same masks in each epoch, and at a column both open, the difference of their symbols would reveal the difference of their weights.
- **The expansion is fixed.** The map from `masks_seed` to an epoch's masks and blinding rows is RFC-0040's, versioned and with a conformance vector. #147, which commits the masks, specifies it. Until it does, a link may be remade only by the build that made it.
- **The claim list.** The matrices' shapes, which matrices have committed rows, and each matrix's `commitment_id_T` (its row block `B` included, RFC-0031) come from a claim list whose weights the commitment holds. The prover has it, and the verifier takes it as an input.
- **The matrices.** Every `WeightMatrix` `T` with committed rows, `m_T > 0`, in RFC-0040's order. A matrix with no committed rows is not linked, as it has no proximity row in a proof either ([RFC-0049](RFC-0049-weight-proximity.md#the-schedule)).
- **Positions.** Committed row `j < m_T` is the `j`-th committed data row in RFC-0040's order, the index RFC-0049's proximity row uses. Both epochs share the layout, so row `j` holds the same weights in both. The blinding rows follow the data rows, `m_T + i` for blinding row `i < B_W`.

### The transcript

The linking proof is non-interactive. Its transcript is RFC-0004's, initialized with:

```rust
SessionContext {
    protocol: "tessera-protocol/2",
    mode: Mode::NonInteractive,
    params: "weights-link/1",
    statement_digest: H("weights-link", R_W ‖ R_W'),
    session_id: [0; 16],
}
```

`weights-link` is a new domain of RFC-0004, over `old_root: Digest ‖ new_root: Digest`, 64 bytes.

- **No parameter set.** The link is bound to the weight commitment, not to a claim list. `live-64` and `ni-100` share their weights' layout (RFC-0014 'Parameter sets'), and the commitment holds no parameter set, so one commitment can serve claim lists of both. A claim list's parameter set in the transcript would give it two links. `weights-link/1` is a `ParamSetId` that names this transcript and no parameter set.
- **Its parameters** come from the whole `WeightLayoutParams`:

  | `log2_k`, `log2_inv_rate`, `P_W`, `B_W` | `τ_link` | `g_link` |
  |---|---|---|
  | 20, 2, 65,536, 4096 (`live-64`, `ni-100`) | 122 | 20 |
  | 8, 2, 64, 32 (`test-small`) | 8 | 0 |

  A layout with no row here has no linking proof: both rows have `B_W ≥ 8`, so an epoch's four rows at each end never overlap. `τ_link` is also the headroom `reserve` keeps (RFC-0012).

For each linked matrix `T` in order, the transcript runs:

| Order | Operation | Label | Value |
|---|---|---|---|
| 1 | squeeze | `link/{T}/rho` | `ρ_T ∈ Fp2^{ν_T}`, `ν_T = log2 k_T + ⌈log2 m_T⌉`, in RFC-0010's variable order: the `log2 k_T` column coordinates `ρ_T^{col}` first, the low bits, then the `⌈log2 m_T⌉` row coordinates `ρ_T^{row}` |
| 2 | absorb | `link/{T}/blinding` | `q_T ∈ Fp2` |
| 3 | squeeze | `link/{T}/theta` | `θ_T ∈ Fp2` |
| 4 | squeeze | `link/{T}/xi/old`, then `link/{T}/xi/new` | each commitment's proximity point, `⌈log2(m_T + B_W)⌉` coordinates |
| 5 | absorb | `link/{T}/combined` | `TCE((difference, proximity_old, proximity_new))`: the three `CombinedRow`s' encodings concatenated, in that order, in one absorb |

Then, once:

| Order | Operation | Label | Value |
|---|---|---|---|
| 6 | grind | `link/query-pow` | a `u64` nonce whose output has `g_link` leading zero bits ([RFC-0014](RFC-0014-interaction-and-soundness.md#grinding)) |
| 7 | squeeze, per linked matrix in order | `link/{T}/queries` | `τ_link` column indices in `[0, n_T)`, uniform and with replacement, as `pcs/weights/{T}/queries` draws them |

`{T}` is the matrix's name, as RFC-0036 spells it. The labels join RFC-0014's tier-A patterns, and `link/query-pow` is a proof of work beside `seed-pow` and `query-pow` (RFC-0014 'Grinding').

### The difference row

For matrix `T`, let `β^{old}_T` be epoch `e`'s blinding rows `B_W − 4` and `B_W − 3`, and `β^{new}_T` epoch `e + 1`'s rows 0 and 1. Each pair is one `Fp2` row, its two committed rows with `1` and `X`, as RFC-0013 'Hiding openings' makes every blinding pair.

- **The blinding claim.** `q_T = ⟨β^{old}_T + β^{new}_T, eq(ρ_T^{col})⟩`, absorbed before `θ_T` is squeezed.
- **The row.** The prover sends

  ```text
  u_T = θ_T·Σ_{r < m_T} eq(ρ_T^{row}, r)·(A^{old}_r − A^{new}_r) + β^{old}_T + β^{new}_T
  ```

  with its combined mask polynomial `M_T`. That is the same combination of the two commitments' mask polynomials, the pairs' masks included. The sum runs over the data rows `r < m_T` alone: `ρ_T^{row}` has `⌈log2 m_T⌉` coordinates, and a row index from `m_T` to `2^{⌈log2 m_T⌉} − 1` names no row, though `eq(ρ_T^{row}, ·)` is defined there.
- **For an honest prover** both epochs hold the same data rows, so the data part of `u_T` is zero and `u_T = β^{old}_T + β^{new}_T`.
- **The evaluation check.** `q_T = ⟨u_T, eq(ρ_T^{col})⟩`: RFC-0013's group check with the public claim 0, `θ_T·0 + q_T`.
- **The column check.** At each distinct queried column `c` of matrix `T`, in increasing order:

  ```text
  θ_T·Σ_{r < m_T} eq(ρ_T^{row}, r)·(old_c[r] − new_c[r]) + β^{old}_T(c) + β^{new}_T(c) = enc(u_T)(x_c) + Z_H(x_c)·M_T(x_c)
  ```

  Here `old_c` and `new_c` are the two commitments' answers at `c`, each verified against its own matrix root, with `old_c[r]` the symbol of data row `r`. `β(c) = β^{(0)}(c) + X·β^{(1)}(c)` combines a pair's two symbols. Both commitments share the code and the coset, so position `c` is one point `x_c` of both.

### The proximity rows

Each commitment keeps its own proximity row, so that each is proven close to the code within the linking proof.

- **The point.** `ξ*` is `link/{T}/xi/old` for epoch `e` and `link/{T}/xi/new` for epoch `e + 1`.
- **The designated pair.** Epoch `e`'s rows `B_W − 2` and `B_W − 1`, and epoch `e + 1`'s rows 2 and 3.
- **The coefficients.** The row is formed as a `tessera-protocol/2` proof's proximity row on that commitment is. That is RFC-0049 'The proximity row' over its data and blinding rows, as #394 settles its coefficients on the blinding rows, with these designated pairs as its blinding pair.
- **The check.** At each queried column, RFC-0011 step 8.3's check against that commitment's answers.

### The message

```rust
pub struct WeightLinkProof {                 // schema 0x0303, version 1, .twl
    pub old: WeightCommitmentPublic,         // epoch e (RFC-0040)
    pub new: WeightCommitmentPublic,         // epoch e + 1, the same layout
    pub query_nonce: u64,                    // link/query-pow
    pub matrices: Vec<MatrixLink>,           // one per linked matrix, ids strictly increasing
}
pub struct MatrixLink {
    pub matrix: u8,                          // WeightMatrix id (RFC-0036 'Matrix ids')
    pub blinding: Fp2,                       // q_T
    pub difference: CombinedRow,             // u_T, M_T
    pub proximity_old: CombinedRow,
    pub proximity_new: CombinedRow,
    pub columns_old: Vec<Vec<Fp>>, pub proof_old: MultiProof,   // each distinct queried column, increasing:
    pub columns_new: Vec<Vec<Fp>>, pub proof_new: MultiProof,   // every committed row, data then blinding
}
```

- **TCE.** The fields in order.
- **The types.** `CombinedRow` and `MultiProof` are RFC-0036's wire types, and a combined row is `k_T` values of `Fp2` with `P_W` mask coefficients.
- **Matrix ids.** The decoder refuses ids that are not strictly increasing, or that name no `WeightMatrix`, with `ERR-CODEC-NONCANONICAL` at the entry's first byte, as RFC-0036 'Matrix ids' does.
- **Columns.** Each answered column is the full column, every committed data row and blinding row (RFC-0012 'Opening by the folding pass'), followed by the canonical multiproof of the set ([RFC-0004](RFC-0004-hashing-and-transcript.md#merkle-trees)).
- **The roots.** The message carries both public commitments because the verifier's ledger holds each epoch's `R_W` alone ([RFC-0019](RFC-0019-verifier.md#ledger)). The matrix roots it needs come from `old` and `new`, which `R_W` and `R_W'` bind.

### The verifier's checks

`tessera-verify weights link` takes the ledger, the `WeightLinkProof` and the claim list of the ledger entry it extends, which the verifier has validated as for a proof (RFC-0007). It checks, in this order, and the first failure is the verdict:
1. **Decoding.** The proof decodes under `ResourceLimits`, else `Reject(Malformed { artifact: WeightLinkProof, code })` with the decoder's code.
2. **The statement.**
   - `H("weights-root", TCE(old))` is the current root of the claim list's ledger entry, or the ledger records `(R_W, R_W')` as a successor in that entry's chain, a link sent again; else `StatementMismatch { field: WeightsRoot }`.
   - The ledger records no successor of `R_W`, or records this proof's `R_W'`, else `{ field: WeightsRoot }`: a root takes one successor.
   - `old.epoch < 2^32 − 1` and `new.epoch = old.epoch + 1`, else `{ field: WeightsEpoch }`.
   - `new.layout = old.layout`, the layout has a row in 'The transcript''s table, and `matrices` lists exactly the matrices the claim list commits rows of, else `{ field: Layout }`. Each matrix's shape and `commitment_id_T`, for both epochs, are the claim list's (RFC-0040, RFC-0031).
3. **The transcript.** The verifier rebuilds it, absorbing each matrix's `q_T` and combined rows at their places.
4. **Shapes and evaluation.** Each combined row has `k_T` values and `P_W` mask coefficients, and each `q_T = ⟨u_T, eq(ρ_T^{col})⟩`. A failure is `Reject(Evaluation { commitment: Weight })`, at the first matrix.
5. **The proof of work.** `query_nonce` meets `g_link`, else `Reject(ProofOfWork { phase: Queries })`.
6. **The queries.** The verifier squeezes each matrix's `τ_link` indices.
7. **Columns,** per matrix in order, epoch `e`'s answer before `e + 1`'s:
   - one column per distinct index, with `m_T + B_W` symbols each, and the multiproof against that commitment's matrix root, else `Reject(MerklePath { commitment: Weight, column })` at the smallest queried index;
   - then, at each distinct index in increasing order, the difference row's column check, then the old proximity row's, then the new one's, else `Reject(Proximity { commitment: Weight, column })` at the first failing `c`.
8. **Accept.** The ledger records `R_W'` as `R_W`'s successor, appends `(epoch e + 1, R_W')` to the claim list's entry, and records the link's rows in each epoch's used blinding rows ([RFC-0019](RFC-0019-verifier.md#ledger)).
   - **Other claim lists.** Another entry at `R_W`, of a claim list with the same commitment, is extended by checking the same link with that claim list. The successor is the one recorded, so the check accepts.
   - **The same link again** is accepted and extends nothing new. The prover may resend it.

### The budget

| Epoch | Blinding rows | Columns |
|---|---|---|
| `e`, at its end | `B_W − 4`, `B_W − 3`: the difference row's pair; `B_W − 2`, `B_W − 1`: the proximity row's | the link's distinct queried columns join `opened_columns`, within the `τ_link` of headroom `reserve` keeps |
| `e + 1`, at its start | 0, 1: the difference row's pair; 2, 3: the proximity row's | its state starts with the link's distinct queried columns in `opened_columns` |

- **The state.** Epoch `e + 1`'s state starts with `next_blinding_row = 4`, and its first proof reserves from row 4. Epoch 0, which no link starts, reserves from row 0.
- **Columns for proofs.** Epoch `e + 1`'s proofs share `P_W − 2τ_link` columns: the link's at its start and the next link's headroom at its end. At `live-64` that is 673 proofs (`(2^16 − 244)/97`), against RFC-0012's 674, and the rows' bound stays 681.
- **One count.** These are RFC-0012's four rows at each end of an epoch, used once each. RFC-0012 'Failure modes' should read "the four blinding rows", and the `reserve` precondition stays `next_blinding_row + 6 ≤ B_W − 4`: `next_blinding_row + r_T ≤ B_T − 4` with [RFC-0065](RFC-0065-masked-channel-params.md#the-reservation)'s counts.
- **The parameters.** 'The transcript' gives `τ_link` and `g_link` per layout. The production layout's are `ni-100`'s `τ_W` and `g_query`, since the link is non-interactive, and `reserve`'s headroom is `τ_link`.

### One link per epoch

A second link from epoch `e` would reveal what one hides:
- **Columns.** It could open `τ_link` more columns of epoch `e`, past `P_W`: up to `(P_W − τ_link) + 2τ_link = 65,658` at the production layout. Past `P_W`, a row's symbols determine linear functionals of its weights that no mask covers (Lemma 3).
- **The proximity pair.** Both links' old proximity rows would carry the same designated pair. Their difference would reveal a combination of the weights alone, if #394 keeps the blinding rows out of the proximity combination.

So every link from epoch `e` is one link: the same bytes, however many times it is made.
- **A function of `K`, the epoch and the weights.** Both epochs' seeds come from `K` ('The inputs'). With RFC-0040's expansion they fix both commitments, their rows and both roots. The transcript is then a function of the roots and the rows, and `query_nonce` is the least nonce that meets `g_link`.
- **What it closes.** A remade link opens the same columns with the same pairs, and reveals nothing a first one did not. That holds for each way a second link could otherwise arise:
  - a resend after a crash, before or after the first was sent;
  - a lost or a restored budget record;
  - two nodes that make the link from one epoch;
  - a verifier that claims to hold no link.
- **Not proofs.** Determinism covers the link alone. Two nodes proving from copies of one epoch's state, or a state restored from a stale backup, still reuse blinding rows: RFC-0012's `reserve` assumes one writer per state. A restored record older than the state's last transition must not `reserve`.
- **Forward secrecy among epochs.** A leaked state file gives its own epoch's seed only, as with fresh seeds, since each seed is derived from `K` and not from the one before. A leaked `K` gives every epoch's.

RFC-0012's state machine gains a transition, for the budget's bookkeeping:

| Event | Precondition | Effect |
|---|---|---|
| `link` (before the `WeightLinkProof` is sent) | Every reservation of epoch `e` has recorded its queries or aborted | Writes epoch `e + 1`'s state, from row 4 with the link's columns opened, and the `WeightLinkProof` beside it. Then burns rows `B_W − 4` to `B_W − 1` of epoch `e`, records the link's columns there, and closes epoch `e` to `reserve` |

- **The order.** Epoch `e + 1`'s state and the link are persisted first (fsync, atomic rename), and closing epoch `e` is the commit point.
  - **Not yet usable.** Epoch `e + 1` takes no `reserve` until epoch `e` is closed.
  - **A crash before the commit point** leaves epoch `e` open with nothing sent. On restart, an existing epoch `e + 1` state and link are kept, never rewritten, and the commit is completed.
  - **A crash after it** leaves both epochs and the link in place.
- **Its errors.** A `link` whose precondition fails is `ERR-WITNESS-BUDGET`, as a failed `reserve` is.
- **The column budget.** `reserve` keeps `τ_link` columns free in every epoch, so the link fits whatever the epoch's record held, a lost record included: the deterministic link opens the same columns however often it is made.
- **The verifier** records one successor of each root (check 2). The rule that protects the prover's secret is the link's determinism, and the verifier's record is a second line.

### Soundness

- **Evaluation.** Suppose both commitments are close to the code, with decoded data rows `D^{old}` and `D^{new}`.
  - The column checks bind `u_T` to them, so `⟨u_T, eq(ρ_T^{col})⟩ = θ_T·(D̃^{old} − D̃^{new})(ρ_T) + Q̃_T`, with `Q̃_T` the true blinding value.
  - The check passes when `θ_T·(D̃^{old} − D̃^{new})(ρ_T) + (Q̃_T − q_T) = 0`. `q_T` is absorbed before `θ_T`, so for `D̃^{old}(ρ_T) ≠ D̃^{new}(ρ_T)` that holds for at most one `θ_T`.
  - Two different multilinear polynomials over `ν_T` variables agree at `ρ_T` with probability at most `ν_T/|Fp2|` (Schwartz–Zippel). The roots fix both commitments before `ρ_T` is squeezed.
  - Per matrix, then: `(ν_T + 1)/|Fp2|`. At 405B the largest `ν_T` is 37 (`Gate`, `Up` and `Down`), which gives 122.75 bits per matrix. The ten together, `Σ (ν_T + 1) = 345`, give 119.57 bits.
  - RFC-0014's calculator counts this `(ν_T + 1)/q` in its weight-linking row, for the matrix of most variables ([#516](https://github.com/AbdelStark/tessera/issues/516)). This RFC's `ν_T` is `log2 k_T + ⌈log2 m_T⌉` over the committed rows: 30 for `ChannelParams`, whose 882 rows the calculator's input counts otherwise.
- **Why `θ_T`.** With the claim 0 and no challenge, any `q_T` would match `u_T` for any data. `θ_T` is drawn after `q_T` and multiplies the difference, as RFC-0013 'Hiding openings' orders it for proofs.
- **Proximity, per commitment.** `((1 + ρ')/2)^{τ_link} + ϑ_T·n_T/|Fp2|`, RFC-0014's bound.
  - At the production layout, `ρ' = 0.2656` and `b(ρ') = 0.6602`, so 122 queries give 80.54 bits, and the proof of work brings them to 100.54.
  - The field term is at most `18 · 2^22/2^128`, 101.8 bits.
  - The points `ξ*` are squeezed after both roots and every blinding claim are fixed. The queries follow the combined rows and the proof of work.
- **Grinding.** A prover can re-draw `θ_T` and `ξ*` by sending another `q_T`, and a later matrix's `ρ_T` by changing an earlier matrix's messages, one hash each. The bounds above are per hash, under RFC-0014's convention for `λ` bits. The first `ρ_T` changes only with a new `R_W'`, and the queries only with the proof of work.
- **What the link does not need.** The new epoch's proofs test its commitment's proximity on their own queries, so a successor far from the code proves nothing false later either. The linking proof's own test makes the link's evaluation meaningful: the commitments it binds decode uniquely.

### Zero knowledge

This is requirement R3 of the simulator argument, met:
- **The scalar.** `q_T` is one functional of `β^{old}_T + β^{new}_T`, two fresh pairs used nowhere else. `eq(ρ_T^{col}) ≠ 0`, so `q_T` is uniform.
- **The row.** Given `q_T`, `u_T` is uniform subject to the evaluation check, and `M_T` is uniform, since it includes the pairs' masks (RFC-0013 'Masks of combined masks'). That is Lemma 5's view.
- **What is not revealed.** No evaluation of the weights. The only claim is the public 0.
- **The proximity rows** are proofs' proximity rows on their commitments, with designated pairs of their own. Lemma 5 and §6.2 apply to them as they apply to a proof's.
- **The columns** are at most `τ_link` distinct per matrix and commitment, inside each epoch's `P_W` (§6.1, Lemma 3), and only one link draws them ('One link per epoch').
- **Two epochs.** The difference row's column check relates the two epochs only at a column opened in both: the link's own, and any column a proof of either epoch also opens. Given the designated pairs' symbols there, the combined polynomial is uniform but on those columns, so nothing about the weights follows.
  - The simulator composes the two epochs' simulators, §6's for each. It samples each epoch's view, then solves the link's pairs at the shared columns, as Lemma 6 solves a proof's pairs at earlier columns.
  - Theorem 2 of `docs/zk/simulator.md` covers one epoch. Its two-epoch form is this composition.
- **The seeds.** Masks come from seeds by a PRG, and seeds from `K` by `weights-refresh`. The simulator's PRG hybrid replaces each epoch's seed with uniform masks in turn, one step per epoch, so `ε_PRG` counts every epoch the view spans. Soundness does not depend on it: the masks are the prover's own choice.
- **The test** (R3's): two weight sets that differ at `ρ_T`, with the random oracle programmed to give both the same `ρ_T`, give identically distributed linking views. RFC-0012's openings reveal the two different `W̃_T(ρ_T)`, which a distinguisher tells apart: the negative control.

### Sizes at 405B

The production layout links ten matrices, each with `k_T = 2^20` and `n_T = 2^22`. The epoch holds 453,126 committed rows, 40,960 of them blinding rows (RFC-0012 'Column answers').

| Part | Bytes |
|---|---|
| Combined rows: 3 per matrix, each `(2^20 + 2^16)` `Fp2` values | 534.8 MB |
| Columns: at most 122 per matrix and commitment, each a committed row's symbol, for both commitments | 884.5 MB |
| Multiproofs: at most `122·(22 − log2 122)` digests per matrix and commitment | 1.2 MB |
| Total, at most | 1.42 GB |

RFC-0012's two openings carried four combined rows per matrix, 713.0 MB: the difference saves 178.3 MB. A link is made once per epoch, which serves about 670 live proofs (535 non-interactive ones, `(2^16 − 244)/122`).

## Alternatives Considered

1. **Mask the two evaluations.** Each opening would reveal a masked value `W'(ρ_T)` and the verifier would compare them. Two epochs' independent masks make the two values differ, so they cannot be compared. Correlating the masks across epochs would reveal their difference to anyone holding both, and each side would need a mask row of its own. The difference needs no mask row and reveals nothing.
2. **One blinding pair, from the new epoch only.** `u_T` would be as uniform. Then the old epoch's last rows go unused, and the difference row would carry no designated pair of epoch `e`, which the epoch simulator expects of every revealed combination over epoch `e`'s rows (§6.2). Using one pair from each epoch keeps every revealed combination with a designated pair in each commitment it reads.
3. **No proximity rows in the link.** Later proofs would test the new commitment's proximity anyway. But the link's evaluation binds the weights only if both commitments decode uniquely, and the link has to stand alone for `tessera-verify weights link`.
4. **Interactive linking.** A verifier could draw `ρ_T` and the queries itself. The link is published once and checked by every verifier of the model, so it is non-interactive, with the proof of work in place of a verifier's draw.

## Drawbacks

- The link depends on #394 for its proximity rows' coefficients on the blinding rows, as every `tessera-protocol/2` proof does.
- The model's refresh key `K` is one more secret to keep: it gives every epoch's seed ('One link per epoch').
- A link is about 1.4 GB at 405B, against about 0.35 GB of column answers for a live proof. It is made once per epoch.

## Migration / Rollout

- **#153** implements the refresh and the linking proof, with R3's test.
- **#158** implements `tessera-verify weights link`, with the claim list as an input, and the ledger's epochs and successors.
- **`tessera-protocol/1`** has no refresh and is unaffected.
- **`docs/zk/simulator.md`** marks R3 met by this RFC.

## Testing Strategy

- **One link.** A link made twice from one epoch's state, before and after a crash, from a restored record and on a second node, is the same bytes each time. A verifier that holds a successor of `R_W` refuses another `R_W'` from it, and accepts the same link again.
- **Soundness.**
  - Equal weights link.
  - Weights that differ in one entry, of any matrix, are refused with `Reject(Proximity)` or `Reject(Evaluation)`, the commitments being honest.
  - A link whose `q_T` is changed after `θ_T` is drawn is refused with `Reject(Evaluation)`.
- **Each check alone.** Each of the verifier's checks has a test that breaks exactly it and asserts its verdict:
  - another `R_W`, another epoch or another layout;
  - a missing or extra matrix;
  - a combined row of another shape;
  - a nonce short of `g_link`;
  - a column missing or extra, and a multiproof sibling changed;
  - one symbol of either commitment changed, which the difference row's or a proximity row's column check refuses.
- **Zero knowledge.** Two weight sets that differ at `ρ_T`, over `F65537` with the test layout and the random oracle programmed to give both the same `ρ_T`, give linking views no distinguisher of #162 tells apart. RFC-0012's openings, which reveal the two different `W̃_T(ρ_T)`, are the negative control.
- **The budget.**
  - The link uses rows `B_W − 4` to `B_W − 1` of epoch `e` and 0 to 3 of `e + 1`.
  - Epoch `e + 1`'s first `reserve` starts at row 4, with the link's columns opened.
  - A link made after the budget record is deleted, from the seed alone, uses the same rows and stays within `P_W`.
- **Vectors.** A link between two test-layout epochs, from an independent oracle: the transcript's challenges, the rows, the columns and the verdict.

## Open Questions

- **OQ-0064-1.** The proximity rows' coefficients on the blinding rows are #394's. If #394 removes the blinding rows from the proximity combination, the link's proximity rows follow it, with the soundness argument #394 gives.

## References

- [RFC-0011](RFC-0011-advice-commitment.md) 'Opening protocol'; [RFC-0012](RFC-0012-weight-commitment.md); [RFC-0013](RFC-0013-zero-knowledge.md) 'Hiding openings'; [RFC-0014](RFC-0014-interaction-and-soundness.md) 'Soundness budget', 'Grinding', 'Parameter sets'; [RFC-0036](RFC-0036-protocol-messages.md); [RFC-0040](RFC-0040-weight-commitment-artifacts.md); [RFC-0049](RFC-0049-weight-proximity.md).
- `docs/zk/simulator.md` §3 (Lemmas 3, 5, 6), §6 (one epoch), §9 (R3).
- Ligero (Ames, Hazay, Ishai, Venkitasubramaniam, 2017); Diamond and Posen, 'Proximity Testing with Logarithmic Randomness' (2024).
