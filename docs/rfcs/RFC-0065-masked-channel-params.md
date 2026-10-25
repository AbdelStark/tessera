# RFC-0065: Masked `ChannelParams` Claims in `tessera-protocol/2`

- Status: Accepted
- Authors: @AbdelStark
- Created: 2026-10-25
- Target milestone: v0.4
- Supersedes:
  - [RFC-0052](RFC-0052-channel-params-claims.md#non-goals) 'Non-Goals': `tessera-protocol/2`'s masks and blinding rows for `ChannelParams`, which this RFC defines;
  - [RFC-0012](RFC-0012-weight-commitment.md#budget-state-machine) 'Budget state machine': `reserve`'s six rows per matrix, now `2·e_T + 4`;
  - [RFC-0013](RFC-0013-zero-knowledge.md#cost) 'Cost': the weight budget per proof.
- Amends:
  - RFC-0052 'Opening by the single-pass fold': in `tessera-protocol/2`, each evaluation row carries its claim's mask row and the matrix's blinding pair;
  - RFC-0012 'Layout' and 'Column answers': `ChannelParams`' blinding rows `B_CP`, and the store's figures;
  - RFC-0013 'Masked extensions', 'Hiding openings', 'Simulator argument' and INV-ZK-2: each `ChannelParams` claim's mask, over its type tensor's coordinates, and the pair its matrix's evaluation rows share;
  - [RFC-0014](RFC-0014-interaction-and-soundness.md) 'Messages' and 'Soundness calculator': each matrix's blinding claims, and the field term's `B_T`;
  - [RFC-0019](RFC-0019-verifier.md#check-order) 'Check order' step 5: the reservation's range;
  - [RFC-0036](RFC-0036-protocol-messages.md) 'Versions and envelopes': schema version 2's `WeightReservation` and `weight_blinding`;
  - [RFC-0007](RFC-0007-claim-list.md#verifier-validation) 'Verifier validation' and [RFC-0039](RFC-0039-statement-fields.md): in `tessera-protocol/2`, a site with a `ChannelParams` is affine until #486;
  - [RFC-0026](RFC-0026-claim-list-layout.md) and [RFC-0031](RFC-0031-commitment-id.md): `blinding_rows` is a single-claim matrix's `B_W`, and a matrix's `rows` count its `B_T`;
  - [RFC-0040](RFC-0040-weight-commitment-artifacts.md#the-matrices) 'The matrices' and 'Encoding, leaves and roots': `B_CP`, and the leaves over the blinding rows;
  - [RFC-0049](RFC-0049-weight-proximity.md#the-calculator) 'The calculator' and [RFC-0064](RFC-0064-weight-linking-proof.md): each matrix's own `B_T`.
- Amended by: [RFC-0068](RFC-0068-leaf-stage-zero-knowledge.md) ('What the stages owe': a revealed value may also lie inside a column value that carries its own column mask; the leaf stage meets it, and `tessera-protocol/2` admits sites of two classes or more)
- Origin: [#487](https://github.com/AbdelStark/tessera/issues/487), from the simulator argument ([`docs/zk/simulator.md`](../zk/simulator.md) §4.5, requirement R9)

## Summary

In `tessera-protocol/2` a weight matrix's claim is a masked evaluation `W'(p) = W̃(p) + Z(p)·R̃_W(p_col)`, opened by an evaluation row that carries the mask row and a blinding pair (RFC-0013). `ChannelParams` has `e` claims, one per reading site (RFC-0052). Each sits at a point of `Ã_CP` whose padding and weight-type coordinates are Boolean, so a mask whose `Z` runs over `Ã_CP`'s variables vanishes there. A mask row shared by the `e` claims leaks too.

This RFC gives each claim its own mask row, under a `Z` over the type tensor's own coordinates, and makes that row the designated block of the claim's evaluation row. The matrix's one blinding pair carries the blinding claims. A proof then reserves `2e + 4` rows of each opened matrix: six for a single-claim matrix, as today, and 18 for `ChannelParams` at `e = 7`. `ChannelParams` gets enough blinding rows, `B_CP`, that it never ends an epoch before the other matrices: 12,280 at 405B against `B_W = 4096`.

The claims of the reduction stages are then hidden. A site of two classes or more reads its class tensor in the leaf stage, which sends it once per layer at Boolean layer points, so its claims are not. Until [#486](https://github.com/AbdelStark/tessera/issues/486)'s leaf layout reads the type tensors only at points whose coordinates other than the lowest are challenges (R8), `tessera-protocol/2` refuses such sites. In the two-class reference quantization of 405B, every site after a weight has two classes, so `tessera-protocol/2` waits for #486 there.

## Motivation

[#487](https://github.com/AbdelStark/tessera/issues/487), from `docs/zk/simulator.md` §4.5, confirmed by an independent reading:
- **The points are Boolean.** Claim `i` evaluates `Ã_CP` at `point_i = r_{c,i} ‖ 0^{γ−c_i} ‖ ρ_i ‖ vertex_4(t_i)` (RFC-0052). RFC-0013's `Z(x) = Π_{i≥2} x_i(1 − x_i)` over `Ã_CP`'s variables is 0 there, so the masked value is `Ã_CP(point_i)` itself.
- **Two claims share a column point.** At 405B, `k_CP = 2^{20} < C·L_pad = 2^{23}`, so the type bits fall in the row part. Two sites of one width read at one `(r_c, ρ)` then have one `column_point`. A mask row shared between them enters both values with the same functional, and their difference reveals `Ã_CP(point_1) − Ã_CP(point_2)`.
- **Several evaluation rows share a mask.** With one mask row and one pair for the `e` rows, differences of rows cancel both, and what remains is a linear combination of the secret folds `F_i`.

RFC-0052 left `tessera-protocol/2` to #147, and RFC-0012's budget counts six rows per matrix and proof, enough for one claim.

## Goals

- Every `ChannelParams` claim, evaluation row and blinding claim is simulatable from public data, in one proof and across an epoch (R9), given the stages' reads of the type tensors.
- An epoch lasts as many proofs as RFC-0012's budget gives the other matrices: `ChannelParams` never ends it first.
- One rule covers every matrix: `Embed`..`LmHead` are its one-claim case, with six rows as today.

## Non-Goals

- **The proximity row's generator** on the blinding rows, R1 and R2 of the simulator argument, which [#394](https://github.com/AbdelStark/tessera/issues/394) settles. This RFC adds blinding rows and does not change how the proximity row combines them.
- **The leaf stage's layout.** How the leaf stage reads each tensor, the type tensors included, is [#486](https://github.com/AbdelStark/tessera/issues/486)'s. This RFC states what it needs from it ('What the stages owe').
- **The rank of the masks' revealed points** on the compiled schedule, R10 ([#488](https://github.com/AbdelStark/tessera/issues/488)), which covers the type tensors' masks with every other.
- **`tessera-protocol/1`**, which commits no masks or blinding rows for the weights (RFC-0040).

## Proposed Design

### The claims

As RFC-0052 defines them: `ChannelParams` has `C = 2^γ`, `L_pad = 2^λ` and `ν_CP = γ + λ + 4` variables. Claim `i ∈ 0..e`, in pool order, reads weight type `t_i` of padded output width `2^{c_i}`:
- the **type tensor** `P_{t_i}`, of `c_i + λ` variables, is `Ã_CP` with its type coordinates fixed to `vertex_4(t_i)` and its high `γ − c_i` channel coordinates to 0;
- the claim is on `P_{t_i}` at `x_i = r_{c,i} ‖ ρ_i`, which is `Ã_CP` at `point_i`;
- `column_point_i` and `row_point_i` split `point_i` at `log2 k_CP`.

**The count.** `e_T` is the number of claims matrix `T` has in the pool: 1 for `Embed`..`LmHead`, and `e_CP` for `ChannelParams`.
- **What `e_CP` is.** In the Llama templates it is the number of weight types with a `ChannelParams`, the types whose site is affine or has two classes or more (03 §2, RFC-0025). Each such type is followed by one site, which leaves one claim (RFC-0052, INV-SCH-5). The claim list makes these types public (RFC-0040 'The matrices').
- **How the verifier gets it.** It computes `e_CP` from `PublicQuantSpec` and never reads it from the proof. A schedule whose pool holds another number of `ChannelParams` claims is the verifier's own inconsistency: `Inconclusive(ERR-INTERNAL)`.
- **The variables.** Every type tensor has at least two variables, `c_i + λ ≥ 2`, as a masked tensor needs (RFC-0013). A weight without layers, which RFC-0052 would read at `ρ_i = 0^λ`, cannot have one: neither `Embed` nor `LmHead` feeds a site that rescales per channel (RFC-0025).

**The data rows.** `ChannelParams`' committed data rows are `m_CP`, as every matrix's are `m_T`. At 405B a type fills 8 rows of `k_CP = 2^{20}` (RFC-0012 'Layout'), so `m_CP = 56` for seven types. RFC-0012 'Column answers' counts it as 882, an overcount: the layout allows at most 8 rows per type, 72 for nine types.

### The masks

Each claim `i` has its own mask row `R_i`, one `Fp2` row of `k_CP` values: two blinding rows of `ChannelParams`, with `1` and `X` ('The reservation'). In `tessera-protocol/2` the type tensor is read through its masked extension:

```math
P'_{t_i}(x) = \tilde P_{t_i}(x) + Z(x) \cdot S_i(x), \qquad Z(x) = \prod_{j \ge 2} x_j (1 - x_j), \qquad S_i(x) = \langle R_i, \mathrm{eq}(\mathrm{col}(x), \cdot) \rangle
```

- **`Z` runs over the type tensor's `c_i + λ` variables**, the lowest spared, as RFC-0013 'Masked extensions' writes it. The padding and type coordinates are not among them.
- **`col(x)`** is the column part of `x`'s embedding into `Ã_CP`: its first `log2 k_CP` coordinates, padding zeros included. `S_i` is multilinear in `x`, so `P'` has degree at most 3 per variable, as every masked extension.
- **On the hypercube** `Z = 0` and `P' = P̃`. Every stage relation that reads `P_{t_i}` reads `P'_{t_i}` in its place (RFC-0013 'Committed tensors'), and its sums are unchanged.
- **At the claim.** `v_i = P'_{t_i}(x_i) = Ã_CP(point_i) + Z(x_i)·⟨R_i, eq(column_point_i)⟩`.

`Embed`..`LmHead` keep RFC-0013's rule: one mask row `R_W`, with `Z` over the matrix's variables, whose claim point has no fixed coordinate.

### What the stages owe

A masked extension hides its tensor only where it is revealed at points whose coordinates other than the lowest are off `{0, 1}` (R8). For the type tensors:
- **Every revealed value of `P'_{t_i}`** sits at such a point. That covers the claim `v_i`, which is a direct evaluation `P'_{t_i}(x_i)`, and any value a stage reveals on the way, such as a lookup's key forms at the leaf level's pair-bit points (`docs/zk/simulator.md` §4.3).
- **No stage reveals** `P̃_{t_i}`, a value of `P'_{t_i}` at a point with a Boolean coordinate other than the lowest, or a combination all of whose terms are such values, as RFC-0037's stack over layers is (R8's second clause). Such values may lie inside a column value that carries its own column mask ([RFC-0068](RFC-0068-leaf-stage-zero-knowledge.md#column-masks)), which masks them as one.
- **The values' functionals.** Each `Z` at those points vanishes with probability at most `2(c_i + λ − 1)/|Fp2|`. The values' functionals of the matrix's pair are independent ('Zero knowledge'): R10, which #488 checks on the compiled schedule for each matrix, `β_T` included.
- **The mask is never dropped.** No `ChannelParams` evaluation row is opened without its `Z(x_i)·R_i` term. If `Z(x_i) = 0`, the prover aborts before it sends `v_i`, and its reservation stays burned (RFC-0012 `abort`). The pair is shared, so one unmasked row would reveal `β_T`, and with it a fold through every other claim's `q_j`. Two unmasked rows would reveal the difference of their folds.
- **The reduction stages meet this.** A claim from a `Mul(Leading)` instance, an affine site's multiplier tensor, sits at the instance's final point, all challenges.
- **The leaf stage did not, as first specified.** RFC-0068 replaces it in `tessera-protocol/2`, and the leaf stage then meets this section ([RFC-0068](RFC-0068-leaf-stage-zero-knowledge.md#zero-knowledge) 'Zero knowledge', R9). RFC-0045 'The split' sends one value of each class tensor per real layer, at Boolean layer coordinates, and RFC-0037 stacks them at `ρ_1`. Each per-layer value has `Z = 0`, so it is the secret's own, and their stack is `Ã_CP(point_i)` with no mask term.
  - 126 such values per site and proof reveal `⟨eq(r_c), class_{t,ℓ}⟩` for each layer. Classes survive a refresh, so the proofs of every epoch add up.
  - `tessera-protocol/2`'s leaf layout ([#486](https://github.com/AbdelStark/tessera/issues/486), R8) must read each class tensor's `P'` only at points whose coordinates other than the lowest are challenges, with no per-layer value. One example is a layer point per site in place of the shared `ρ_1`.
  - **Until RFC-0068, such sites were refused.** That refusal is lifted. It read: in `tessera-protocol/2`, every site with a `ChannelParams` must be affine. A site of two classes or more is refused. The compiler refuses to emit such a claim list, with `ERR-USAGE`. The verifier rejects one at claim-list validation, `Reject(StatementMismatch { field: Rescaling { site } })` (RFC-0007 step 2.4, RFC-0039 step 2.4). The honest prover's `v_i`, the stack, would fail step 9 anyway.
  - **At 405B** all seven claims come from two-class sites, so its `tessera-protocol/2` proofs needed RFC-0068's leaf stage.

### The evaluation rows

Each opened matrix `T` reserves one blinding pair `β_T` for its evaluation rows ('The reservation'). For each claim `i` of `T`, in pool order:

1. **Blinding claims.** The prover sends `q_i = ⟨β_T, eq(column_point_i)⟩`.
   - **The message.** `Round2Response.weight_blinding` holds every opened matrix's claims' `q_i`: the matrices in id order, each matrix's claims in pool order. That is one value for `Embed`..`LmHead`, and `e_CP` for `ChannelParams`. Any other length than `Σ_T e_T` is `Reject(Malformed)`, as RFC-0036 'Shapes' rules lengths.
   - **The transcript.** Under `pcs/weights/{T}/blinding` (RFC-0014's transcript schedule, row 7), it absorbs matrix `T`'s values as one `TCE(Vec<Fp2>)`, of length `e_T`, before `θ_T` is squeezed under `pcs/weights/{T}/gamma`.
2. **The row.** After `θ_T`, the prover reveals

   ```text
   u_i = θ_T · (F_i + Z(x_i)·R_i) + β_T,
   ```

   `F_i` RFC-0052's fold. The single pass over the committed rows forms the `F_i` and the proximity row as RFC-0052 says, and the `R_i` and `β_T` are added once `θ_T` is drawn.
3. **Evaluation check (RFC-0019 step 9).** `θ_T · v_i + q_i = ⟨u_i, eq(column_point_i)⟩`, for each `i`, else `Reject(Evaluation { commitment: Weight })`.
4. **Column check (RFC-0019 step 11).** The symbols are as follows: `c[·]` is the queried column's symbols; `R_i^{(0)}, R_i^{(1)}` and `β^{(0)}, β^{(1)}` are the rows of the claim's mask and the matrix's pair; and `w_i` is the codeword of `evaluations[i]` with its mask (RFC-0011 encoding). Then at each queried column `q`:

   ```text
   w_i[q] = θ_T · ( Σ_{j < m_T} eq(row_point_i, r_j)·c[j] + Z(x_i)·(c[R_i^{(0)}] + X·c[R_i^{(1)}]) ) + c[β^{(0)}] + X·c[β^{(1)}],
   ```

   else `Reject(Proximity { commitment: Weight, column: q })`.

For `Embed`..`LmHead` this is RFC-0012's step 3 with RFC-0013's mask term: one claim, `R_W` and `β_T`.

### The proximity row

Unchanged: one combined row, its own designated pair, and the generator's coefficients on the data and blinding rows. [#394](https://github.com/AbdelStark/tessera/issues/394) settles the generator's coefficients on the blinding rows (R1, R2), the new rows included.

### The reservation

A proof reserves, for each opened matrix `T`, `r_T = 2·e_T + 4` consecutive blinding rows from `b_T`:

| Rows | Use |
|---|---|
| `b_T`, `b_T + 1` | `β_T`, the evaluation rows' pair |
| `b_T + 2 + 2i`, `b_T + 3 + 2i`, for `i ∈ 0..e_T` | `R_i`, claim `i`'s mask row |
| `b_T + 2 + 2e_T`, `b_T + 3 + 2e_T` | the proximity row's designated pair |

- **The pairs.** In each pair, the first row enters with `1` and the second with `X`.
- **The counts.** `Embed`..`LmHead` reserve 6 rows: `β_T`, `R_W`, the proximity pair. `ChannelParams` reserves `2·e_CP + 4`: 18 at `e_CP = 7`.
- **The declaration.** `Round1`'s `WeightReservation` (RFC-0014 'Messages', in RFC-0036's schema version 2) holds one `(matrix, b_T)` per opened matrix, the matrices in strictly increasing id order. So every slot is fixed before the seed.
  - **Its epoch** is `statement.weights_epoch`, not repeated.
  - **Ids out of order** are `ERR-CODEC-NONCANONICAL` at decode, as RFC-0036's keyed sequences are, so the proof gives `Reject(Malformed)`.
  - A matrix with no claim is not opened and reserves nothing.
- **The prover's precondition.** RFC-0012's `reserve` precondition becomes, per matrix, `next_blinding_row + r_T ≤ B_T − 4`, and its effect burns `r_T` rows. The four rows at each end of an epoch stay the link's (RFC-0064).
- **The verifier's check (RFC-0019 step 5).** It derives each `r_T` from the claim list. It refuses with `Reject(Budget { epoch })` a declaration whose matrices are not the opened ones, or whose ranges do not satisfy all of:
  - `b_T ≥ 4` after epoch 0;
  - `b_T + r_T ≤ B_T − 4`;
  - no overlap with the rows the ledger has burned in the epoch.

  So no column check reads past a column's `m_T + B_T` symbols (SP-8).

### `ChannelParams`' blinding rows

Each matrix `T` has `B_T` blinding rows after its data rows:

```text
B_T  = B_W                                for Embed..LmHead,
B_CP = B_W + 2·(e_CP − 1)·⌊max(B_W − 4, 0)/6⌋   when e_CP ≥ 1, and 0 when e_CP = 0.
```

- **`N_0 = ⌊(B_W − 4)/6⌋`** is the number of proofs epoch 0 gives a single-claim matrix by rows: it reserves from row 0, and the link keeps the last four (RFC-0064 'The budget'). A later epoch reserves from row 4 and gives `N_1 = ⌊(B_W − 8)/6⌋`. At `B_W = 4096`, `N_0 = 682` and `N_1 = 681`.
- **`ChannelParams` never ends an epoch first.** `B_W − 4 ≥ 6·N_0`, so `B_CP − 4 ≥ (2·e_CP + 4)·N_0` in epoch 0. Likewise `B_CP − 8 ≥ (2·e_CP + 4)·N_1` after it, since `N_0 ≥ N_1`.
- **No `ChannelParams`.** With `e_CP = 0`, the matrix commits no row (RFC-0040), is never opened or linked, and has no blinding rows.
- **At 405B**, `e_CP = 7`: `B_CP = 4096 + 12·682 = 12,280`. The test parameter set's `B_W = 32` gives `N_0 = N_1 = 4` and, at `e_CP = 7`, `B_CP = 80`.
- **The layout.** `WeightLayoutParams.blinding_rows` stays `B_W`, the blinding rows of a single-claim matrix, and `B_CP` follows from it and the claim list. Blinding row `i < B_T` is committed row `m_T + i`, and RFC-0040's `rows`, in `commitment_id_T`, is `m_T + B_T`.
- **The leaves.** In `tessera-protocol/2` a column's chain runs over all `m_T + B_T` committed rows, so its leaf is `d_j[⌈(m_T + B_T)/B⌉]` (RFC-0040 'Encoding, leaves and roots').
- **The link.** RFC-0064 reads `B_T` where it wrote `B_W` for a matrix's rows:
  - the last four rows of epoch `e` are `B_T − 4` to `B_T − 1`;
  - a column has `m_T + B_T` symbols;
  - each commitment's proximity point has `⌈log2(m_T + B_T)⌉` coordinates.

  Its parameter table keeps the layout's `B_W`.
- **The calculator.** RFC-0014's field term is `ϑ_T = ⌈log2(m_T + B_T)⌉` in `tessera-protocol/2`. Its `matrices` input gives each matrix's claims `e_T`, from which it derives `B_T` ([#521](https://github.com/AbdelStark/tessera/issues/521)). `max(B_W − 4, 0)` keeps `N_0` a count for a parameter set of fewer than four blinding rows, which no set of RFC-0014 has.

### Cost at 405B

| Item | Before | With this RFC |
|---|---|---|
| `ChannelParams` rows per proof | 6 | 18 |
| `ChannelParams` blinding rows | 4,096 | 12,280 |
| Codeword store | 453,126 rows, 15.20 TB | 461,310 rows, 15.48 TB |
| Column answers per proof | 3.63 MB per column: 0.35 GB at `τ_W = 97`, 0.44 GB at 122 | 3.69 MB per column: 0.36 GB and 0.45 GB, 6.4 MB and 8.0 MB more |
| Masks and blinding rows regenerated by the fallback | 581 GB per proof | 654 GB |
| `Round2Response.weight_blinding` | 10 values | 16 values: 96 bytes more |
| `ϑ_CP` | `⌈log2(m_CP + 4096)⌉ = 13`: 102.30 bits | `⌈log2(m_CP + 12,280)⌉ = 14`: 102.19 bits |
| Proofs per epoch | 682 by rows in epoch 0, 681 after | the same, `ChannelParams` included; 674 and 673 by columns (RFC-0012, RFC-0064) |

- **The answers.** A column answer carries every row of its matrix (RFC-0012 'Column answers'), hence 8,184 more symbols per queried column, 8 bytes each.
- **The store.** It is RFC-0012's 453,126 rows plus those 8,184, each of `2^{22}` symbols of 8 bytes. That figure counts `ChannelParams`' data rows by RFC-0012's bound of 882. With the 56 rows the layout gives, every row count above is 826 lower.
- **`ϑ_CP`** is the same with either count of data rows. Down's `ϑ = 18` still binds the weights' field terms.

### Soundness

- **The evaluation rows.** For a false `v_i`, `θ_T·(v_i − f_i) + (q_i − Q̃_i)` vanishes for at most one `θ_T`, with `f_i` the true masked value and `Q̃_i = ⟨β_T, eq(column_point_i)⟩`. Every `q_i` is absorbed before `θ_T`, so a false claim passes with probability at most `1/|Fp2|`: the calculator's openings row, unchanged.
- **The masks are fixed in advance.** Each `R_i` is committed with the epoch, and the reservation names it in `Round1`, before the seed. `P'_{t_i}` is then a fixed polynomial before any challenge of a stage that reads it. It equals `P̃_{t_i}` on the hypercube, so the stage proves the same statement about it.
- **The slots.** The column check gives each reserved row its role. A prover that swaps `β_T` and some `R_i` passes the evaluation check and fails the column check.
- **Proximity.** `ChannelParams`' field term grows with its rows, to `14·n_CP/|Fp2|`, 102.19 bits. Its queries row is unchanged.

### Zero knowledge

**One proof** (`docs/zk/simulator.md` §4.5, with Lemmas 2 and 5), given the stages' reads ('What the stages owe'):
- **The rows.** `u_i` takes `R_i` with the coefficient `θ_T·Z(x_i)`, invertible unless `θ_T = 0` or `Z(x_i) = 0`, and with `(1, X)`; no other evaluation row takes it. So the `e_T` rows are triangular in their masks, and jointly uniform.
- **The scalars.** `q_i` is a functional of `β_T`, and `v_i` follows from `u_i` and `q_i` by the evaluation check. The map from `(R_0, …, R_{e−1}, β_T)` to `(u_i, q_i)_i` is onto the values the checks allow:
  - choose `β_T` with `⟨β_T, eq(column_point_i)⟩ = q_i`;
  - then solve for each `R_i` from `u_i`.

  The view is therefore uniform subject to the checks.
- **Shared column points.** Two claims at one column point give the public relation `q_i = q_j`, which says nothing about the weights.
- **Values revealed in the stages.** Let `e_y = eq(col(y), ·)` for a point `y`. Once `u_i` is known, a value `w = P'_{t_i}(y)` at another point `y` reads

  ```text
  w = public + S_{i,y} − (Z(y) / (θ_T·Z(x_i)))·⟨β_T, e_y⟩,    S_{i,y} = P̃_{t_i}(y) − (Z(y)/Z(x_i))·⟨F_i, e_y⟩.
  ```

  - **When `S_{i,y} = 0`** the value carries no secret. That happens at the leaf level's pair-bit points around `x_i`, which differ from it in the lowest coordinate only. Its functional of `β_T` still joins the span the other values must avoid. #488 decides `S ≡ 0` symbolically: it holds exactly when `y` has `x_i`'s row point and `Z(y) ≡ Z(x_i)`.
  - **Otherwise** `e_y` must lie outside the span of every other functional of `β_T` the matrix reveals, the `q_j` and every other claim's values included, apart from public relations.

  This is R10 per matrix, with `β_T`'s functionals together (#488).
- **One matrix, one fate.** The claims of a matrix share `β_T`, so each is hidden only if all of them are masked. 'What the stages owe' applies to every claim of the matrix, or none is hidden.
- **The proximity row** is Lemma 5's, with `β_T`'s spare directions among its free rows.

**One epoch** (Lemma 7):
- **What is revealed.** `N` proofs reveal `e_T` evaluation rows and one proximity row each, per matrix.
- **The elimination.** Clearing the evaluation rows with their designated `R_i` leaves the proximity rows' block `I + R` on the fixed words. A uniform block sits on the `β_T` columns, `2N` of them.
- **The bound.** The count is Lemma 7's, `C(N, 2)·p^{−4}` per matrix.
- **The singular case.** `β_T` carries `e_T` functionals, where `R_W` carried one. A scalar falls in the vectors' row span only if `(w_1/w_0)^2 = 7` for two of its coefficients. That is impossible, since 7 is a non-residue modulo `p`, given that `{eq(c), eq(σ(c))}` are independent over the distinct column points `c` of every functional of `β_T`, the stages' included, with `σ` the conjugation of `Fp2`.
- **What that costs.** The independence fails with probability about `2e_T·log2 k/p ≈ 2^{−55}`, within the singular event's `1/p`.
- **For `ChannelParams` only.** The one-claim matrices keep §6.2's reading, with `β` designated and `U` on `R_W`; it is equally valid for them.

R9 is met for every matrix all of whose claims meet 'What the stages owe'. With RFC-0068's leaf stage, that is every model's.

### The verifier

- **Step 5** derives each `r_T` and checks the declared ranges ('The reservation'). The ledger records the burned rows in a set of `B_T` bits per matrix ([RFC-0019](RFC-0019-verifier.md#ledger)).
- **Steps 9 and 11.** Step 9 checks the `e_T` evaluation rows with their blinding claims, and step 11 runs the column checks above.
- **Its inputs.** `e_CP` and `B_CP` come from the claim list and the layout, never from the proof.

## Alternatives Considered

1. **One mask row for `ChannelParams`, `Z` over `Ã_CP`.** Rejected: `Z` vanishes at every claim's point, and a shared mask row cancels between two claims at one column point.
2. **A blinding pair per claim, with its own mask row** (`4e + 2` rows, 30 at `e = 7`). It hides as well, with `β_i` designated for row `i`. Rejected for its cost: `B_CP` of 20,464 and 3.6% more store against 1.8%. With a mask row per claim, the mask rows can serve as the evaluation rows' designated blocks, and one pair per matrix carries the scalars.
3. **A larger `B_W` for every matrix.** Rejected: 8,184 more rows in each of ten matrices, 81,840 in all, 18% more store and column answers, for one matrix's need.
4. **`B_W` unchanged for `ChannelParams`.** Rejected: 227 proofs per epoch at `e = 7`, three times as many refreshes, each a new commitment of 15 TB.
5. **Reduce the `e` claims to one point first**, with a masked sumcheck over `Ã_CP` (RFC-0052's alternative 2). Rejected: the `e` values are stage outputs the verifier sees, so they keep their masks. It would save only `2(e − 1)` rows, at the cost of a stage and its masking polynomial.
6. **The masks in the advice.** Rejected: an advice claim's value is revealed in its group, so `R̃_i(column_point_i)` would be public and `v_i` would give `Ã_CP(point_i)`.
7. **A mask over the channel coordinates only**, one mask row per type and row block, which would survive per-layer values. Rejected: 8 rows per type at 405B, about 116 rows per proof with one pair per matrix, and per-layer values reveal the data the stacking binds in any case. #486 removes them instead.

## Drawbacks

- `ChannelParams` holds three times as many blinding rows as the other matrices. Its 8,184 extra rows add 1.8% to the store and to every column answer.
- `B_CP` depends on the claim list as well as the layout, so a model's `ChannelParams` row count follows from both.
- Until RFC-0068, `tessera-protocol/2` admitted only affine `ChannelParams` sites. RFC-0068 lifts the refusal, which reached further:
  - The Silu site can never be affine: an affine site follows the Q, K, V, O, up or down projection (RFC-0007 step 2.2, RFC-0025). So every `tessera-protocol/2` model needs a one-class Silu, one scale for all of Gate's channels.
  - The `minimal` profile's O and Down refuse `affine` (RFC-0025 'The quantizer'), so those two need one class too.
  - `tessera-quant` will need a `tessera-protocol/2` mode.

## Migration / Rollout

- **#147** commits the masks and blinding rows, `B_CP` included, and implements the reservation and the budget.
- **#486** specifies the `tessera-protocol/2` leaf stage, reading the class tensors as 'What the stages owe' requires, and lifts the refusal of sites of two classes or more: RFC-0068.
- **#154** builds the masked extensions of the type tensors in the stages that read them, and **#159** verifies them.
- **#162** distinguishes two witnesses that differ in `ChannelParams`, at the openings and at the leaf stage's split values. The current text is the negative control.
- **#521** makes both calculators and the soundness oracle count `B_T`.
- **`docs/zk/simulator.md`** marks R9 met for models whose `ChannelParams` sites are all affine, and for the others with RFC-0068.

## Testing Strategy

- **Hiding (#162).** Two `ChannelParams` contents that agree on everything public give identically distributed `v_i`, `q_i` and `u_i`, two claims at one column point included. Under the old text they differ: the negative control.
- **Adversarial (#147, #159).**
  - A false `v_i` gives `Reject(Evaluation { commitment: Weight })`.
  - A row that swaps `β_T` for some `R_i` gives `Reject(Proximity { commitment: Weight, column })`.
  - A `tessera-protocol/2` claim list with a site of two classes or more is accepted at claim-list validation (RFC-0068).
  - The reservation gives `Reject(Budget)` when:
    - it reuses a burned row;
    - it reaches past `B_T − 4`;
    - it starts below row 4 after epoch 0;
    - it names another set of matrices.
- **Budget.** At `test-small` (`B_W = 32`), on a model with seven `ChannelParams` sites, `B_CP = 80`. `ChannelParams` then serves the same 4 proofs as the other matrices, and the fifth `reserve` fails with `ERR-WITNESS-BUDGET`.

## Open Questions

None.

## References

- [RFC-0012](RFC-0012-weight-commitment.md), [RFC-0013](RFC-0013-zero-knowledge.md), [RFC-0014](RFC-0014-interaction-and-soundness.md), [RFC-0040](RFC-0040-weight-commitment-artifacts.md), [RFC-0045](RFC-0045-leaf-layout.md), [RFC-0052](RFC-0052-channel-params-claims.md), [RFC-0064](RFC-0064-weight-linking-proof.md).
- [`docs/zk/simulator.md`](../zk/simulator.md) §4.3, §4.5, §6.2 and §9.
- [#487](https://github.com/AbdelStark/tessera/issues/487), [#394](https://github.com/AbdelStark/tessera/issues/394), [#486](https://github.com/AbdelStark/tessera/issues/486), [#488](https://github.com/AbdelStark/tessera/issues/488).
