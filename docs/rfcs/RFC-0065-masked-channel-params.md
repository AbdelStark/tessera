# RFC-0065: Masked `ChannelParams` Claims in `tessera-protocol/2`

- Status: Accepted
- Authors: @AbdelStark
- Created: 2026-10-25
- Target milestone: v0.4
- Supersedes:
  - [RFC-0052](RFC-0052-channel-params-claims.md#non-goals) 'Non-Goals': `tessera-protocol/2`'s masks and blinding rows for `ChannelParams`, which this RFC defines;
  - [RFC-0012](RFC-0012-weight-commitment.md#budget-state-machine) 'Budget state machine': `reserve`'s six rows per matrix, now `4·m_T + 2`;
  - [RFC-0013](RFC-0013-zero-knowledge.md#cost) 'Cost': the weight budget per proof.
- Amends:
  - RFC-0052 'Opening by the single-pass fold': in `tessera-protocol/2`, each evaluation row carries its claim's mask row and its own blinding pair;
  - [RFC-0040](RFC-0040-weight-commitment-artifacts.md#the-matrices) 'The matrices' and 'Encoding, leaves and roots': `ChannelParams`' blinding rows `B_CP`;
  - [RFC-0064](RFC-0064-weight-linking-proof.md) 'The inputs', 'The difference row', 'The proximity rows', 'The budget': each matrix's own blinding rows `B_T`;
  - [RFC-0014](RFC-0014-interaction-and-soundness.md#soundness-calculator) 'Soundness calculator': the field term's `ϑ_T` counts `B_T`;
  - [RFC-0026](RFC-0026-claim-list-layout.md) `WeightLayoutParams.blinding_rows`: a single-claim matrix's `B_W`.
- Origin: [#487](https://github.com/AbdelStark/tessera/issues/487), from the simulator argument ([`docs/zk/simulator.md`](../zk/simulator.md) §4.5, requirement R9)

## Summary

In `tessera-protocol/2`, a weight matrix's claim is a masked evaluation `W'(p) = W̃(p) + Z(p)·R̃_W(p_col)`, opened by one evaluation row that carries the mask row and its own blinding pair (RFC-0013). `ChannelParams` has `m` claims, one per reading site (RFC-0052). Each sits at a point of `Ã_CP` whose padding and weight-type coordinates are Boolean, so a mask whose `Z` runs over `Ã_CP`'s variables vanishes there. One mask row or one blinding pair shared by the `m` claims leaks too.

This RFC gives each claim its own mask row, under a `Z` over the type tensor's own coordinates, all of them challenges, and its own blinding pair. A proof then reserves `4m + 2` rows of `ChannelParams`: 30 at `m = 7`. `ChannelParams` gets enough blinding rows, `B_CP`, that it never ends an epoch before the other matrices: 20,464 at 405B against `B_W = 4096`.

## Motivation

[#487](https://github.com/AbdelStark/tessera/issues/487), from `docs/zk/simulator.md` §4.5, confirmed by an independent reading:
- **The points are Boolean.** Claim `i` evaluates `Ã_CP` at `point_i = r_{c,i} ‖ 0^{γ−c_i} ‖ ρ_i ‖ vertex_4(t_i)` (RFC-0052). RFC-0013's `Z(x) = Π_{i≥2} x_i(1 − x_i)` over `Ã_CP`'s variables is 0 there, so the masked value is `Ã_CP(point_i)` itself.
- **Two claims share a column point.** At 405B, `k_CP = 2^{20} < C·L_pad = 2^{23}`, so the type bits fall in the row part. Two sites of one width read at one `(r_c, ρ_1)` then have one `column_point`. A mask row shared between them enters both values with the same functional, and their difference reveals `Ã_CP(point_1) − Ã_CP(point_2)`.
- **Several evaluation rows share a pair.** With one blinding pair `β` for the `m` rows, `u_1 − u_2` loses `β`, and three rows also eliminate a shared mask row. What remains is a linear combination of the secret folds `F_i`.

RFC-0052 left `tessera-protocol/2` to #147, and RFC-0012's budget counts six rows per matrix and proof, enough for one claim only.

## Goals

- Every `ChannelParams` claim and evaluation row is simulatable from public data, in one proof and across an epoch (R9).
- An epoch lasts as many proofs as RFC-0012's budget gives the other matrices: `ChannelParams` never ends it first.
- One rule covers every matrix: `Embed`..`LmHead` are its `m = 1` case, unchanged.

## Non-Goals

- **The proximity row's generator** on the blinding rows, R1 and R2 of the simulator argument, which [#394](https://github.com/AbdelStark/tessera/issues/394) settles. This RFC adds blinding rows and does not change how the proximity row combines them.
- **How the stages produce the claims.** The leaf stage's per-layer split values, which precede the stacking at `ρ_1`, are [#486](https://github.com/AbdelStark/tessera/issues/486)'s (R8). This RFC needs only the stacked claim's point (below).
- **`tessera-protocol/1`**, which commits no masks or blinding rows for the weights (RFC-0040).

## Proposed Design

### The claims

As RFC-0052 defines them: `ChannelParams` has `C = 2^γ`, `L_pad = 2^λ` and `ν_CP = γ + λ + 4` variables. Claim `i ∈ 0..m`, in pool order, reads weight type `t_i` of padded output width `2^{c_i}`:
- the **type tensor** `P_{t_i}`, of `c_i + λ` variables, is `Ã_CP` with its type coordinates fixed to `vertex_4(t_i)` and its high `γ − c_i` channel coordinates to 0;
- the claim is on `P_{t_i}` at `x_i = r_{c,i} ‖ ρ_i`, which is `Ã_CP` at `point_i`;
- `column_point_i` and `row_point_i` split `point_i` at `log2 k_CP`.

**The count.** `m` is the number of weight types with a `ChannelParams`, which the claim list makes public (RFC-0040 'The matrices'): each such type is followed by one site, which leaves one claim (RFC-0052, INV-SCH-5). It is `m_CP`, the same for every claim list of the model.

### The masks

Each claim `i` has its own mask row `R_i`, one `Fp2` row of `k_CP` values: two blinding rows of `ChannelParams`, with `1` and `X` ('The reservation'). In `tessera-protocol/2` the claim's value is the masked extension of the type tensor:

```math
P'_{t_i}(x) = \tilde P_{t_i}(x) + Z(x) \cdot S_i(x), \qquad Z(x) = \prod_{j \ge 2} x_j (1 - x_j), \qquad S_i(x) = \langle R_i, \mathrm{eq}(\mathrm{col}(x), \cdot) \rangle
```

- **`Z` runs over the type tensor's `c_i + λ` variables**, the lowest spared, as RFC-0013 'Masked extensions' writes it. The padding and type coordinates are not among them.
- **`col(x)`** is the column part of `x`'s embedding into `Ã_CP`: its first `log2 k_CP` coordinates, padding zeros included. `S_i` is multilinear in `x`, so `P'` has degree at most 3 per variable, as every masked extension.
- **On the hypercube** `Z = 0` and `P' = P̃`. The stage relation that reads `P_{t_i}` reads `P'_{t_i}` in its place, as RFC-0013 'Committed tensors' says, and its sums are unchanged.
- **At the claim.** `v_i = P'_{t_i}(x_i) = Ã_CP(point_i) + Z(x_i)·⟨R_i, eq(column_point_i)⟩`.
- **`Z(x_i) ≠ 0` requires** every coordinate of `x_i` but the lowest to be off `{0, 1}`. For a claim from a `Mul(Leading)` instance, `(r_c, ρ)` is the instance's final point, all challenges. For a claim from the leaf stage, `r_c = π_c(r_s)` is a projection of the leaf point and `ρ_1` is RFC-0037's layer point, both challenges. A producing stage that fixed a coordinate would violate R8, which #486 enforces for the stages. Each `Z(x_i)` vanishes with probability at most `2(c_i + λ − 1)/|Fp2|`.

`Embed`..`LmHead` keep RFC-0013's rule: one mask row `R_W`, with `Z` over the matrix's variables, whose claim point has no fixed coordinate.

### The evaluation rows

For each claim `i` in pool order, with `β_i` its designated blinding pair:

1. **Blinding claims.** The prover sends `q_i = ⟨β_i, eq(column_point_i)⟩`. Under `pcs/weights/ChannelParams/blinding` (RFC-0014's transcript schedule, row 7), the transcript absorbs the `m` values as one `TCE(Vec<Fp2>)` in pool order, before `θ_CP` is squeezed under `pcs/weights/ChannelParams/gamma`. A single-claim matrix absorbs its one `q_T` as today.
2. **The row.** After `θ_CP`, the prover reveals

   ```text
   u_i = θ_CP · (F_i + Z(x_i)·R_i) + β_i,
   ```

   `F_i` RFC-0052's fold. The single pass over the committed rows forms the `F_i` and the proximity row as RFC-0052 says, and the `R_i` and `β_i` are added once `θ_CP` is drawn.
3. **Evaluation check (RFC-0019 step 9).** `θ_CP · v_i + q_i = ⟨u_i, eq(column_point_i)⟩`, for each `i`, else `Reject(Evaluation { commitment: Weight(ChannelParams) })`.
4. **Column check (RFC-0019 step 11).** At each queried column `q`, with `c[·]` the column's symbols, `R_i^{(0)}, R_i^{(1)}` and `β_i^{(0)}, β_i^{(1)}` the pairs' rows, and `w_i` the codeword of `evaluations[i]` with its mask (RFC-0011 encoding):

   ```text
   w_i[q] = θ_CP · ( Σ_{j < m_CP} eq(row_point_i, r_j)·c[j] + Z(x_i)·(c[R_i^{(0)}] + X·c[R_i^{(1)}]) ) + c[β_i^{(0)}] + X·c[β_i^{(1)}],
   ```

   else `Reject(Proximity { commitment: Weight(ChannelParams), column: q, row: i })`, as RFC-0052 numbers the rows.

For `m = 1` these are RFC-0012's step 3 with RFC-0013's mask term, the rule every other matrix follows.

### The proximity row

Unchanged: one combined row, its own designated pair, and the generator's coefficients on the data and blinding rows. Its row is `m` in `Reject(Proximity)`, after the evaluation rows. [#394](https://github.com/AbdelStark/tessera/issues/394) settles the generator's coefficients on the blinding rows (R1, R2), the new rows included.

### The reservation

A proof reserves, per opened matrix `T` with `m_T` claims, `r_T = 4·m_T + 2` consecutive blinding rows from `b = next_blinding_row`:

| Rows | Use |
|---|---|
| `b + 4i`, `b + 4i + 1`, for `i ∈ 0..m_T` | `β_i`, claim `i`'s designated pair |
| `b + 4i + 2`, `b + 4i + 3` | `R_i`, claim `i`'s mask row |
| `b + 4m_T`, `b + 4m_T + 1` | the proximity row's designated pair |

- In each pair, the first row enters with `1` and the second with `X`.
- `Embed`..`LmHead` reserve 6 rows: `β`, `R_W`, the proximity pair. `ChannelParams` reserves `4m + 2`: 30 at `m = 7`.
- The `Round1` `WeightReservation` declares each matrix's `b` ([RFC-0036](RFC-0036-protocol-messages.md#round1) adds it in version 2), so every slot is fixed before the seed.
- A matrix with no claim is not opened and reserves nothing.

RFC-0012's `reserve` precondition becomes, per matrix, `next_blinding_row + r_T ≤ B_T − 4`, and its effect burns `r_T` rows. The four rows at each end of an epoch stay the link's (RFC-0064).

### `ChannelParams`' blinding rows

Each matrix `T` has `B_T` blinding rows after its data rows:

```text
B_T  = B_W                                for Embed..LmHead,
B_CP = B_W + 4·(m_CP − 1)·⌊(B_W − 4)/6⌋   when m_CP ≥ 1, and B_W when m_CP = 0.
```

- **`N_0 = ⌊(B_W − 4)/6⌋`** is the number of proofs epoch 0 gives a single-claim matrix by rows: it reserves from row 0, and the link keeps the last four (RFC-0064 'The budget'). A later epoch reserves from row 4 and gives `N_1 = ⌊(B_W − 8)/6⌋`. At `B_W = 4096`, `N_0 = 682` and `N_1 = 681`.
- **`ChannelParams` never ends an epoch first.** `B_W − 4 ≥ 6·N_0`, so `B_CP − 4 ≥ (4·m_CP + 2)·N_0` in epoch 0, and likewise `B_CP − 8 ≥ (4·m_CP + 2)·N_1` after it, since `N_0 ≥ N_1`.
- **At 405B** every per-layer weight's site has two classes: RFC-0012's 882 `ChannelParams` rows are seven types over 126 layers, so `m_CP = 7` and `B_CP = 4096 + 24·682 = 20,464`. The test parameter set's `B_W = 32` gives `N_0 = N_1 = 4` and, at `m_CP = 7`, `B_CP = 128`.
- **The layout.** `WeightLayoutParams.blinding_rows` stays `B_W`, the blinding rows of a single-claim matrix, and `B_CP` follows from it and the claim list. RFC-0040's `rows`, in `commitment_id_T`, is `m_T + B_T`. Blinding row `i < B_T` is committed row `m_T + i`.
- **The link.** RFC-0064 reads `B_T` for `B_W`: the last four rows of epoch `e` are `B_T − 4` to `B_T − 1`, and a column has `m_T + B_T` symbols.
- **The calculator.** RFC-0014's field term is `ϑ_T = ⌈log2(m_T + B_T)⌉` in `tessera-protocol/2`, and its `matrices` input gives each matrix's `B_T`.

### Cost at 405B

| Item | Before | With this RFC |
|---|---|---|
| `ChannelParams` rows per proof | 6 | 30 |
| `ChannelParams` blinding rows | 4,096 | 20,464 |
| Codeword store | 453,126 rows, 15.2 TB | 469,494 rows, 15.75 TB |
| Column answers per proof | 3.63 MB per column: 0.35 GB at `τ_W = 97`, 0.44 GB at 122 | 3.76 MB per column: 0.36 GB and 0.46 GB, 12.7 MB and 16.0 MB more |
| Blinding claims, `ChannelParams` | 1 | `m = 7`, 192 bytes more |
| `ϑ_CP` | `⌈log2(882 + 4096)⌉ = 13`: 102.30 bits | `⌈log2(882 + 20,464)⌉ = 15`: 102.09 bits |
| Proofs per epoch, `ChannelParams` | 136 at `B_W` once the leak is fixed | 682 by rows in epoch 0 and 681 after, as every matrix; 674 and 673 by columns (RFC-0012, RFC-0064) |

A column answer carries every row of its matrix (RFC-0012 'Column answers'), hence the 16,368 more symbols per queried column, 8 bytes each. The store's figure is RFC-0012's 453,126 rows plus those 16,368, at `2^{22}` symbols of 8 bytes.

### Soundness

- **The evaluation rows.** For a false `v_i`, `θ_CP·(v_i − f_i) + (q_i − Q̃_i)` vanishes for at most one `θ_CP`, `f_i` the true masked value and `Q̃_i = ⟨β_i, eq(column_point_i)⟩`. Every `q_i` is absorbed before `θ_CP`, so a false claim passes with probability at most `1/|Fp2|`: the calculator's openings row, unchanged.
- **The masks are fixed in advance.** Each `R_i` is committed with the epoch, and the reservation names it in `Round1`, before the seed. `P'_{t_i}` is then a fixed polynomial before any challenge of the stage that reads it, and it equals `P̃_{t_i}` on the hypercube, so the stage proves the same statement about it.
- **Proximity.** `ChannelParams`' field term grows with its rows, to `15·n_CP/|Fp2|`, 102.09 bits. Its queries row is unchanged.

### Zero knowledge

**One proof** (`docs/zk/simulator.md` §4.5, with Lemmas 2 and 5):
- **The values.** `v_i` is `Ã_CP(point_i) + Z(x_i)·R̃_i(column_point_i)`, with `Z(x_i) ≠ 0` and `eq(column_point_i) ≠ 0`. Each `R_i` enters one value and no other, so the `m` values are jointly uniform, whether or not two claims share a column point.
- **The blinding claims.** `q_i` is one functional of the fresh pair `β_i`, uniform.
- **The rows.** `u_i` takes `β_i` with `(1, X)`, and no other evaluation row does, so the `m` rows are triangular. Each is uniform on the hyperplane its check fixes.
- **The proximity row** is Lemma 5's, with the `R_i` among the free rows: each is read at one point, so it keeps `k_CP − 1` free directions.

**One epoch** (Lemma 7): `N` proofs reveal `m` evaluation rows and one proximity row each. Clearing the `m·N` designated pairs `β_i` with the evaluation rows leaves the proximity rows' block `I + R` on the fixed words, and a uniform block on the mask rows, now `2m·N ≥ 2N` columns. The count is Lemma 7's, `C(N, 2)·p^{−4}` per matrix.

R9 is met.

### The verifier

- **Step 5** checks each matrix's `r_T` declared rows against the ledger, which records them in a set of `B_T` bits per matrix ([RFC-0019](RFC-0019-verifier.md#ledger)).
- **Step 9** checks the `m` evaluation rows with their blinding claims, and **step 11** the column checks above.
- **Its inputs.** `m_CP` and `B_CP` come from the claim list and the layout, never from the proof.

## Alternatives Considered

1. **One mask row for `ChannelParams`, `Z` over `Ã_CP`.** Rejected: `Z` vanishes at every claim's point, and a shared mask row cancels between two claims at one column point.
2. **One blinding pair for the `m` evaluation rows.** Rejected: differences of rows lose the pair, and three rows also eliminate a shared mask.
3. **A larger `B_W` for every matrix.** Rejected: 16,368 more rows in each of ten matrices, 163,680 in all, 36% more store and column answers, for one matrix's need.
4. **`B_W` unchanged for `ChannelParams`.** Rejected: 136 proofs per epoch at `m = 7`, five times as many refreshes, each a new commitment of 15 TB.
5. **Reduce the `m` claims to one point first**, with a masked sumcheck over `Ã_CP` (RFC-0052's alternative 2). Rejected: the `m` values are stage outputs the verifier sees, so they keep their masks. It saves `m − 1` pairs at the cost of a stage and its masking polynomial.
6. **The masks in the advice.** Rejected: an advice claim's value is revealed in its group, so `R̃_i(column_point_i)` would be public and `v_i` would give `Ã_CP(point_i)`.

## Drawbacks

- `ChannelParams` holds about five times as many blinding rows as the other matrices, 3.6% of the store and of every column answer.
- `B_CP` depends on the claim list as well as on the layout, so a model's `ChannelParams` row count follows from both.

## Migration / Rollout

- **#147** commits the masks and blinding rows, `B_CP` included, and implements the reservation and the budget.
- **#154** builds the masked extensions of the type tensors in the stages that read them, and **#159** verifies them.
- **#162** distinguishes two witnesses that differ in `ChannelParams`. The current text, with no per-claim mask, is the negative control.
- **The calculator's `B_T`** is [#521](https://github.com/AbdelStark/tessera/issues/521)'s: both calculators and the soundness oracle.
- `docs/zk/simulator.md` marks R9 met by this RFC.

## Testing Strategy

- **Hiding (#162).** Two `ChannelParams` contents agreeing on everything public give identically distributed `v_i`, `q_i` and `u_i`, two claims at one column point included. Under the old text they differ: the negative control.
- **Adversarial (#147, #159).**
  - A false `v_i` gives `Reject(Evaluation)`.
  - A row that swaps two slots of the reservation, `β_i` for `R_i`, gives `Reject(Proximity)` at row `i`.
  - A reservation that reuses a burned row gives `Reject(Budget)`.
- **Budget.** At `test-small` (`B_W = 32`, `m_CP = 7`), `ChannelParams` serves the same number of proofs as the other matrices, 4, and the fifth `reserve` fails with `ERR-WITNESS-BUDGET`.

## Open Questions

None.

## References

- [RFC-0012](RFC-0012-weight-commitment.md), [RFC-0013](RFC-0013-zero-knowledge.md), [RFC-0014](RFC-0014-interaction-and-soundness.md), [RFC-0040](RFC-0040-weight-commitment-artifacts.md), [RFC-0052](RFC-0052-channel-params-claims.md), [RFC-0064](RFC-0064-weight-linking-proof.md).
- [`docs/zk/simulator.md`](../zk/simulator.md) §4.5, §6.2 and §9.
- [#487](https://github.com/AbdelStark/tessera/issues/487), [#394](https://github.com/AbdelStark/tessera/issues/394), [#486](https://github.com/AbdelStark/tessera/issues/486).
