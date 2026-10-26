# RFC-0068: The Leaf Stage in `tessera-protocol/2`: Column Masks and a Layer Sumcheck

- Status: Accepted
- Authors: @AbdelStark
- Created: 2026-10-25
- Target milestone: v0.4
- Amended by: [RFC-0069](RFC-0069-one-layer-claims.md) ('The layer sumcheck' step 1 and 'Zero knowledge': an outside value at one layer; 'Non-Goals': the epilogue's claim), [RFC-0070](RFC-0070-mask-batching.md) ('The layer sumcheck' step 6: `G(ρ')` is sent last under `stage/0/layer/evals` and opened by RFC-0070's batching; the proof-size figures), [RFC-0013](RFC-0013-zero-knowledge.md#masked-extensions) ('The layer sumcheck' step 4, 'Zero knowledge' and 'Testing Strategy': a committed tensor's mask takes its data claim's column sub-point and no layer variable, so the degree-4 case is a virtual or type tensor's, [#554](https://github.com/AbdelStark/tessera/issues/554))
- Amends, in `tessera-protocol/2`:
  - [RFC-0032](RFC-0032-zero-knowledge-tree-levels.md#leaf-checks) 'Leaf checks': each LogUp instance's key and value columns take a mask row of their own;
  - [RFC-0045](RFC-0045-leaf-layout.md#the-leaf-point) 'The leaf point' and 'The split': `P` without the tree index's coordinate, and the values of sites in the layers sent once per site and tensor, at a random layer point, after a layer sumcheck;
  - [RFC-0037](RFC-0037-stage-schedule.md#the-layer-point) 'The layer point', 'Splits' and 'Messages and transcript': stage 1's layer point is the layer sumcheck's, the leaf stage's atom splits are taken there, and their labels;
  - [RFC-0014](RFC-0014-interaction-and-soundness.md#transcript-schedule) 'Transcript schedule' rows 5 and 6, and 'Soundness calculator': the layer sumcheck's labels and terms;
  - [RFC-0036](RFC-0036-protocol-messages.md) schema version 2: the leaf stage's proof carries the layer sumcheck;
  - [RFC-0032](RFC-0032-zero-knowledge-tree-levels.md#leaf-checks) 'Leaf checks', 'Degree' and 'Where the claims go': the tree index takes a forest's last round, so the round of raised degree is the one before it, and the key and value claims enter the layer sumcheck;
  - [RFC-0013](RFC-0013-zero-knowledge.md#masked-extensions) 'Masked extensions' and 'Masked sumcheck': `Z` vanishes until the round that binds the last of its variables, which a LogUp leaf level reaches before its last round;
  - [RFC-0011](RFC-0011-advice-commitment.md#opening-protocol) 'Opening protocol' and the figures that follow from its `G ≤ 16` (RFC-0011, RFC-0013, RFC-0019, RFC-0057, 08 §5): one more group, `G ≤ 17`;
  - [RFC-0065](RFC-0065-masked-channel-params.md#what-the-stages-owe) 'What the stages owe', [RFC-0007](RFC-0007-claim-list.md) 'Verifier validation' step 2.4 and [RFC-0039](RFC-0039-statement-fields.md) step 2.4: `ChannelParams` sites of two classes or more are admitted.
- Origin: [#486](https://github.com/AbdelStark/tessera/issues/486)

## Summary

`tessera-protocol/2` as specified reveals unmasked activation evaluations at the leaf stage (`docs/zk/simulator.md` §4.3, requirement R8). RFC-0045 evaluates every looked-up tensor of a site in the layers once per real layer, at a point whose layer coordinates are Boolean. There a tensor's masked extension equals the tensor, since RFC-0013's `Z` vanishes at any Boolean coordinate but the lowest. These evaluations leave the prover in two places:
- **the split values,** one per site, tensor and real layer: 53,489 at 405B;
- **the LogUp columns.** Each instance's key and value columns are concatenations of the same per-layer evaluations (RFC-0045 'The leaf point'), and RFC-0032 gives them no mask of their own. The last tree level reveals them at its pair points, as public-weight sums of unmasked terms, which R8's second clause forbids.

This RFC changes three things:
- **Column masks.** Each instance's key column, and its value column when its table has values, takes one `Fp2` mask row: `κ'_I = κ̃_I + Z·R̃_{κ,I}`. The revealed column values are then masked by the instance's own rows, whatever tensors and layers the instance holds.
- **A layer sumcheck.** In place of the per-layer split values, one masked sumcheck reduces every instance's leaf claims to one value per site and tensor at a random layer point `ρ'`, where the tensors' own masks act. The column masks' terms are folded into it, so they are revealed only at its final point. Stage 1 stacks at `ρ_1 = ρ'_y`, the point's layer coordinates.
- **Admission.** `ChannelParams` sites of two classes or more, refused in `tessera-protocol/2` until this RFC, are admitted. Their class tensor is no longer revealed per layer.

At 405B this costs:
- 36 column mask rows, 72 committed rows, against 953,580 committed advice rows;
- one masked sumcheck of 22 rounds;
- one more advice opening group, about 64 MiB. The proof, about 2.67 GB in `tessera-protocol/2`, which has no token groups, becomes about 2.74 GB, before #489 opens the masking polynomials. RFC-0013's fold groups (#538) add two more, about 2.87 GB, and [RFC-0070](RFC-0070-mask-batching.md)'s batching of the mask claims one more, about 2.94 GB in all.

The split shrinks from 53,489 lookup-tensor values to 614. `tessera-protocol/1` is unchanged.

## Motivation

- **The leak.**
  - RFC-0045 sends `X̃_t(π_t(r_s) ‖ ℓ)` for each site `s` in the layers, tensor `t` and real layer `ℓ`.
  - The same terms make up each instance's columns, `κ̃_I(P[0..h_I]) = Σ_b eq(P[k_b..h_I], o_b/2^{k_b})·form_b + pad`.
  - RFC-0032 reads the columns through the tensors' masked extensions, which vanish at Boolean layers. So the column values that the last tree level reveals at `(b ‖ r')` are unmasked combinations.
  - `simulator.md` §4.3's "masked by those tensors' own masks once R8 is met" holds only once the per-layer terms are masked, and as specified they are not.
- **What #486 weighed, and a third option.**
  - **One mask row per tensor and layer (R1).** It masks every per-layer evaluation. At 405B that is 53,298 `Fp2` rows, 106,596 committed rows, 11.2% more than the 953,580 committed advice rows. It keeps 53,489 split values.
  - **A layout whose leaf point gives random layer coordinates (R2).** It needs blocks for padded layers, 81,746,788,352 more lookups at 405B (1.59%). Its layer point per site, `P[k_s..k_s+λ]`, depends on the site's block size, so stage 1 would stack sites at different layer points, against RFC-0037's one.
  - **Masking tensors.** Masking the tensors more cleverly is fragile. Tables differ by layer (requantization tables by shift, `Silu` tables by input exponent), so an instance can hold a site's later layers but not its first. A mask keyed to one layer then misses that instance, as an adversarial read of this RFC's first draft showed with valid claim lists.
  - **Masking the columns.** The columns are what is revealed, so masking them directly does not depend on how instances and layers interleave.

## Goals

- No leaf-stage value revealed in `tessera-protocol/2` without a uniform mask (R8).
- No change to RFC-0045's layout, to RFC-0037's single layer point, or to `tessera-protocol/1`.
- Costs computed from the compiler's 405B layout.

## Non-Goals

- **The masking polynomials' openings** (#489), which [RFC-0070](RFC-0070-mask-batching.md) specifies. The layer sumcheck adds one more claim to its batching.
- **Grouped leaf stages** (RFC-0054). `tessera-protocol/2` accepts none.
- **Claims at Boolean layers outside the leaf stage.** The epilogue's claim on the last checkpoint, `Layers::One(L − 1)` (RFC-0037), sits at a Boolean layer where the stacked kind's mask vanishes. [RFC-0069](RFC-0069-one-layer-claims.md) masks it (#535).

## Proposed Design

**Notation.**
- `L` is the number of real layers, and `λ = ⌈log2 L⌉`.
- A tensor of a site in the layers is stacked over `2^λ` layer slots, the layer variables `y` highest (RFC-0037). Its other variables are `x`.
- `r_s = P[0..k_s]` is site `s`'s point (RFC-0045 'The leaf point'), and `π_t(r_s)` is tensor `t`'s coordinates in it.
- Instance `I` has a column of `2^{h_I}` entries, its leaf claims at `P[0..h_I]`.
- **`P` in `tessera-protocol/2`.** RFC-0045 'The leaf point' defines `P` for `tessera-protocol/1`.
  - In `tessera-protocol/2` every LogUp tree is a forest with a padding tree (RFC-0032 'Forests of two trees at least'). The tree index is the forest leaf level's highest variable, so it takes the last tree level's last round challenge `r_{N−1}`, with `N` that level's round count and `H` the tallest forest's height.
  - A tree of height `h` then has its leaves at `(ρ_{H−1}, r_{N−h}, …, r_{N−2})`, and its columns' claims are RFC-0032's at `p_low`.
  - So `P = (ρ_{H−1}, r_{N−2}, r_{N−3}, …, r_0)`: RFC-0045's point without `r_{N−1}`. RFC-0045's identities hold at it as written, and `N` and `H` appear nowhere else in this RFC.

### Column masks

For each LogUp instance `I`:
- **The rows.** The key column `κ_I` takes a mask row `R_{κ,I}`, and the value column `e_I`, when its table has values, a mask row `R_{e,I}`. Each is an `Fp2` advice mask row, two committed rows, over the column's low `c_I = min(h_I, log2 k)` variables, drawn from the per-proof seed (INV-ZK-3).
- **The extension** of the column becomes
  ```text
  κ'_I(z) = κ̃_I(z) + Z(z)·R̃_{κ,I}(z_col),     Z(z) = Π_{i ≥ 2} z_i(1 − z_i)  over the column's h_I variables
  ```
  `κ̃_I` is RFC-0045's composition of the blocks' forms, read through the tensors' masked extensions. `e'_I` is built the same way.
- **On the hypercube** `Z = 0`, so the leaf level's polynomials and every identity of RFC-0008 and RFC-0032 hold unchanged.
- **Degree.** `κ'_I` has degree at most 3 in each variable, as a masked tensor has. RFC-0032's bound of 7 is unchanged, in the round it holds for, the one before the last.
- **The multiplicities** keep their own masked extension (RFC-0032).

### The layer sumcheck

In `tessera-protocol/2` the leaf stage ends as follows, after `logup/values`.

1. **The values outside the layers.** For each site outside the layers, in node order, the prover sends its tensors' values at their points, as RFC-0045 says, under `stage/0/split`. An outside claim that RFC-0037 splits into atoms has its atom values follow, as RFC-0037 'Splits' orders them. Their points' coordinates are all challenges, so the tensors' masks act there. A value at one layer, such as a read of the last checkpoint, has its layer coordinates Boolean, and RFC-0069's mask of the layer's tensor acts instead.
2. **The claims.** Each instance has its key claim `κ'_I(P[0..h_I])`, and its value claim when its table has values: that of RFC-0032, derived from the last level and `logup/values`. By RFC-0045's identity and the column mask,
   ```text
   κ'_I(P[0..h_I]) = Σ_{(s,t) layered} Σ_ℓ C^κ_{I,s,t}(ℓ)·X'_t(π_t(r_s) ‖ ℓ)  +  out^κ_I  +  pub^κ_I  +  Z(P[0..h_I])·R̃_{κ,I}(P[0..c_I])
   C^κ_{I,s,t}(ℓ) = Σ_{b ∈ I, site s, layer ℓ} eq(P[k_b..h_I], o_b/2^{k_b})·a^κ_{b,t}
   ```
   - `a^κ_{b,t}` is `t`'s coefficient in block `b`'s key form.
   - `out^κ_I` is the outside sites' blocks, from step 1's values.
   - `pub^κ_I` collects the forms' constants, the padding term `(1 − LT(P[0..h_I], m_I))·t_0`, and RFC-0037's native targets' public terms.
   - `C^κ_{I,s,t}(ℓ) = 0` at a padded layer, which has no blocks.
   - Value claims are the same with `e` for `κ`.
   - The verifier computes every `C`, `out` and `pub` from public data and `P`.
   - `X'_t` is the tensor's masked extension, which the columns read. At a Boolean layer coordinate it equals `X̃_t`, so these terms are unmasked. With one layer, `λ = 0`, there is no layer coordinate and `X'_t`'s point is all challenges.
3. **Batching.** The prover sends `Γ` under `stage/0/layer/zk-sum`, and the transcript yields `ρ_zk` under `stage/0/layer/zk-rho`. The verifier then draws `γ` under `stage/0/layer/gamma`. The claims are indexed by `i`: instances in table order, each instance's key claim, then its value claim. The verifier sets
   ```text
   W_{s,t}(ℓ) = Σ_i γ^i·C_{i,s,t}(ℓ),     H = Σ_i γ^i·(claim_i − out_i − pub_i)
   ```
4. **The sumcheck** proves `H` as the sum of two kinds of instances over `n = max(λ, max_I c_I)` variables. Each instance takes the last `n_i` of the `n` challenges, as RFC-0010 'Batching within a stage' front-pads.
   - **Scaling.** RFC-0010's batching enters each instance's claim scaled by `2^{n − n_i}`. That needs each instance's own claim, which the verifier does not know here: the folds' sums are secret.
   - So each instance's summand is multiplied instead by `2^{−(n − n_i)}`, and the sumcheck starts from `H` itself. Summing a summand that ignores its first `n − n_i` variables over all `n` counts it `2^{n − n_i}` times, so the scaled sum is the instance's own.
   - The same factor multiplies each instance's final value in the last check.
   - **The layer instance,** over the `λ` layer variables:
     ```text
     Σ_{y ∈ {0,1}^λ} Σ_{(s,t) layered} W̃_{s,t}(y)·X'_t(π_t(r_s) ‖ y)
     ```
     `X'_t` is the tensor's masked extension, equal to `X̃_t` on Boolean `y`.
   - **One fold per column mask,** over the mask's `c_i` variables, with `p_i = P[0..h_{I(i)}]`:
     ```text
     Σ_{w ∈ {0,1}^{c_i}} γ^i·Z(p_i)·eq(p_i[0..c_i], w)·R_{·,I(i)}[w]
     ```
     The verifier never learns `R̃(p_i[0..c_i])`, which would unmask the column; this is RFC-0013's fold for a virtual tensor's mask.
   - **Masking.** RFC-0013's masked sumcheck masks it, its masking polynomial `G` committed in Round 1.
   - **Rounds.** Round `j` sends its polynomial under `stage/0/layer/round/{j}`, as RFC-0010 'Sumcheck messages' does, then draws `ρ'_j` under `stage/0/layer/r/{j}`.
   - **The stage's degree** `d`, RFC-0010's one degree for every round, is 4 when `λ ≥ 1`, and 2 when `λ = 0`. The verifier derives it from `λ` alone, never from the masks' variables.
     - A fold has degree 2 in each variable.
     - The layer instance has degree 2 in every round but its last: until then its remaining layer variables are Boolean, so `Z`'s layer factors vanish.
     - In its last round, `W̃·X'_t` has degree 3: `Z`'s factor `y(1 − y)` times `W̃`. It has degree 4 when the tensor's mask depends on `y_{λ−1}`:
       - a virtual tensor of `n_V ≤ log2 k − 1` variables, whose folded mask takes all of them, its layer variables included (RFC-0013 'Virtual tensors');
       - a type tensor whose mask's column point holds `y_{λ−1}` (RFC-0065).

       A committed advice tensor's mask never does: its columns are its data claim's column sub-point, which holds no layer variable (RFC-0013 'Committed tensors', [#554](https://github.com/AbdelStark/tessera/issues/554)).
     - So 4 bounds every round. On the compiler's schedule, only virtual tensors reach it:
       - **405B at 16K positions.** The two `RsqrtRow` sites' root inputs, read through the `SumSquares` outputs, virtual tensors of 14 non-layer variables: `n_V = 21 = log2 k − 1`. The sites' 374 committed columns of 14 variables do not reach it, nor do the seven class tensors, of 10, 14 or 16 non-layer variables, whose masks' column point holds the layer variables `y_0..y_3` only.
       - **405B past 16,384 positions.** The root inputs have `n_V ≥ 22`, so their masks stop below `y_6`, and no pair reaches 4. An honest last round then has degree 3, and `d` stays 4.
       - **The conformance claim lists,** at 1 to 5 positions, laid out at 3 to 5, and at their `max_context`, 64, with `λ = 1` or 2. The `RsqrtRow` root inputs reach it in every case. At 1 to 5 positions, in the lists of one layer variable, some `Requant` inputs reach it too: virtual matmul, `Rotate` and elementwise-product outputs of 7 or 8 non-layer variables, read directly or through linear maps (`ConstMul`, `Reshape`). `tiny-4-servable`, of two, has the root inputs alone.
     - At 405B, `n = 22` rounds of degree 4. An honest round's polynomial has degree 2 in every round but the last.
     - A round message of another length than `d` values is `Reject(Sumcheck { stage: 0, round: j })` (RFC-0010).
   - **The masking polynomial** `G` has each `g_j` of degree `d`: `2(n(d + 1) + 1)` committed elements, 222 at 405B.
5. **The final values.** The prover sends, under `stage/0/layer/evals`, in this order:
   1. `v_{s,t} = X'_t(π_t(r_s) ‖ ρ'_y)` for every layered site and tensor, in RFC-0045's order of sites and tensors. `ρ'_y` is the layer instance's point, the last `λ` challenges.
   2. `u_i = R̃_{·,I(i)}(ρ'_{w_i})` for every column mask, in claim order. `ρ'_{w_i}` is fold `i`'s point, the last `c_i` challenges.
   3. `G(ρ')`, the masking polynomial's value, which [RFC-0070](RFC-0070-mask-batching.md)'s batching opens.
6. **The check.** The verifier computes each `W̃_{s,t}(ρ'_y)`, in `O(L)` per pair, and each fold's public factor. It then checks the last round against the batched final value plus `ρ_zk·G(ρ')`. `G(ρ')` is sent last under `stage/0/layer/evals`, and [RFC-0070](RFC-0070-mask-batching.md)'s batching sumcheck opens it with every other mask claim.
   - A final mismatch is `Reject(Sumcheck { stage: 0, round: n })`. RFC-0010's format derives `p_j(1)` from the claim, so no earlier round can mismatch.
   - The outside claims' atom splits of step 1 are checked before the sumcheck, a mismatch being `Reject(Sumcheck { stage: 0, round: 0 })`.
7. **The result.**
   - **Each `v_{s,t}`** is the stacked claim `X'_t(π_t(r_s) ‖ ρ'_y)` of RFC-0037 'The layer point', and stage 1's layer point is `ρ_1 = ρ'_y`. `tessera-protocol/2` draws no `stage/1/layer/{c}`.
   - **A stacked claim that RFC-0037 splits into atoms** is split at that point. Its atom values follow under `stage/0/layer/split`: by site and tensor in RFC-0045's order, then in the claim's atom order. A mismatch is `Reject(Sumcheck { stage: 0, round: n })`, as RFC-0037 gives a stage's splits its last round.
   - **Each `u_i`** is a claim on its mask row at `ρ'_{w_i}`, which joins the advice pool. Its column functional is `eq(ρ'_{w_i}, ·)`. The `u_i` of one `c_i` share a column sub-point, and form one more opening group of RFC-0011 per distinct `c_i`. At 405B every column has `h_I ≥ 22`, table 3's the shortest at 22, so every `c_I = 22` and they form one group, at `ρ'`.
   - **Stage 1** is otherwise unchanged.

**Soundness.**
- A false `claim_i` survives batching with probability at most `n_claims/|Fp2|` over `γ`.
- The sumcheck adds at most `n·d/|Fp2|` over `ρ'`.
- The masked sumcheck's `ρ_zk` adds `1/|Fp2|` (RFC-0013).
- These replace `tessera-protocol/1`'s stacking error of `λ/|Fp2|` over `ρ_1`. At 405B, with `n = 22`, `d = 4` and 36 claims, the total is `(36 + 88 + 1)/|Fp2| < 2^{-121}`.
- **The calculator** (RFC-0014) counts the layer sumcheck as one more reduction stage of `tessera-protocol/2`:
  - its `n` rounds of degree `d`, in `rounds`;
  - its `m = n_claims`, the claims `γ` batches, in `stages`, which brings its `ρ` term. The sumcheck has `n_claims + 1` instances, the layer instance and the folds, but `γ` batches the claims.
  - The layer point it replaces has no term in RFC-0014's calculator. #537 adds one for `tessera-protocol/1`.

### What is sent, in `tessera-protocol/2`

| Order | Item | Label |
|---|---|---|
| after `logup/values` | the outside sites' values, then their atom values | `stage/0/split` |
| | `Γ`, then `ρ_zk` | `stage/0/layer/zk-sum`, `stage/0/layer/zk-rho` |
| | `γ` | `stage/0/layer/gamma` |
| per round `j < n` | its polynomial, then `ρ'_j` | `stage/0/layer/round/{j}`, `stage/0/layer/r/{j}` |
| | the `v_{s,t}`, then the `u_i`, then `G(ρ')` (RFC-0070) | `stage/0/layer/evals` |
| | the atom values of split stacked claims | `stage/0/layer/split` |
| stage 1 | as RFC-0014 row 6, with no `stage/1/layer/{c}` | |

Round 2's leaf proof carries the outside values as `splits`, plus the layer sumcheck with its masked sum, final values and atom splits. Their TCE layout, and what each `stage/0/layer/*` label absorbs, are schema version 2's, which RFC-0036 leaves to #148 and #154. Every message is absorbed before the challenge that follows it (INV-FS-1). The column masks, like every mask row, are committed in Round 1.

### Zero knowledge

This section covers what the verifier sees of the leaf stage (`simulator.md` §4.3).

- **The column values.**
  - The last tree level reveals each column at the two pair points `(b ‖ r')`, `b ∈ {0, 1}`, and `logup/values` sends the value column at the line point, which is in their span (RFC-0032).
  - Each value is `κ̃_I(z_b) + Z(z_b)·R̃_{κ,I}(z_{b,col})`. Since `z_b`'s coordinates other than the lowest are challenges, `Z(z_b) ≠ 0` except with probability at most `2(h_I − 1)/|Fp2|`.
  - The two points differ in the lowest coordinate, which the mask's variables include, so they give `R_{κ,I}` two functionals.
- **The folds.** The verifier sees `u_i = R̃(ρ'_{w_i})`, a third functional of the same row, and the round messages, masked by `G` (Lemma 4).
- **Per column mask** there are three functionals, at `(0 ‖ r')`, `(1 ‖ r')` and `ρ'_{w_i}` on the mask's variables. Their `eq` vectors are independent except with probability at most `9/|Fp2|` (Lemma 2), since `r'` and `ρ'` are independent challenges.
  - The column values are then uniform, and so are the `u_i`, except where the final equation ties them. That equation `ρ_zk·G(ρ')` absorbs (Lemma 4).
  - No structural rank condition arises. Each instance's columns are masked by rows only that instance uses, wherever its blocks and layers lie.
- **The tensors.**
  - Their per-layer evaluations enter only the column values, already masked, and the layer instance's round sums, masked by `G`.
  - The `v_{s,t}` sit at `(π_t(r_s) ‖ ρ'_y)`, every coordinate but the lowest a challenge, so each tensor's own mask `R_t` acts there (RFC-0013).
  - A committed tensor's `R_t` is read at `π_t(r_s)`'s column sub-point alone, `ρ'_y` lying in the claim's row part (RFC-0013 'Committed tensors', #554). In the compiler's leaf layouts one site reads each tensor, at one point (`crates/tessera-prover/tests/leaf_read_points.rs`), so the leaf stage reveals one value of it.
  - `R_t` is then revealed at that point and at stage 1's final point, as in every stacked claim. R10's check (#488) covers those points for the gadgets' inputs: a committed input as a pool atom of the leaf stage, a virtual one as a virtual value that stage 1 proves. It does not cover the value forms' `v_{s,t}` yet, gadget outputs and auxiliary columns, nor the other items of RFC-0013 'Committed tensors' list of R10's gap, which blocks #154 and #159.
- **The outside values** sit at points of challenges and are masked by their tensors' own masks, or, at one layer, by the layer's tensor's (RFC-0069).
- **R8.** No tensor or column of the leaf stage is revealed alone at a point where its mask vanishes, and every revealed sum of such terms carries a column mask. R8 is met for the leaf stage.
- **R9.** `ChannelParams`' class tensors are read at the leaf per layer, but those evaluations are now revealed only inside masked columns.
  - Their stacked claim sits at RFC-0065's `x_i = π(r_s) ‖ ρ'_y`, with random coordinates, where the per-claim masks act.
  - Their Boolean-layer terms carry no `R_i` and no `β_T` functional.
  - So sites of two classes or more meet RFC-0065 'What the stages owe', amended to read: "every revealed value of a type tensor sits at a point whose coordinates other than the lowest are challenges, or inside a column value that carries its own column mask (RFC-0068)".
  - RFC-0065's rule that the prover aborts when `Z(x_i) = 0` applies to these claims before `stage/0/layer/evals`.
  - `tessera-protocol/2` admits them, and RFC-0007's and RFC-0039's step 2.4 refusal is lifted.

### Cost at 405B

The figures are computed from the compiler's layout: `LeafLayout::model` on the servable lowering with two classes, at 16K positions.

| Item | `tessera-protocol/1` | `tessera-protocol/2` with this RFC |
|---|---|---|
| Sites, of them in the layers; instances | 18, 16; 25 | the same |
| Split values, lookup tensors only | 53,489: 423 layered pairs × 126 layers, plus 191 outside | 614: 423 at `ρ'` and 191 outside, plus 36 `u_i` |
| Split bytes | 856 KB | 10.4 KB |
| Column mask rows | — | 36 `Fp2` rows (25 key and 11 value columns): 72 committed rows on 953,580 |
| Layer sumcheck | — | 22 rounds of degree 4, 88 `Fp2` values; masking polynomial `2(22·5 + 1) = 222` committed elements |
| Advice opening groups | `G ≤ 16`, `G + 1 ≤ 17` combined rows (RFC-0011) | one more, at `ρ'`: a blinding pair, a blinding claim and a combined row of `k = 2^{22}` `Fp2` values, about 64 MiB. `G ≤ 17` |
| Proof | about 2.8 GB with the token groups' 120 MB (08 §5) | about 2.67 GB without token groups, and 2.74 GB with this group, before #489 |

Atom split values come in addition, in both protocols.

**For comparison:**
- R1 would add 53,298 `Fp2` rows, 106,596 committed (11.2%), with no new opening group, and keep 856 KB of split values;
- R2 would add 81,746,788,352 lookups (1.59%) and give each site its own layer point.

## Alternatives Considered

1. **One mask row per tensor and layer (R1).** It is sound and simple to argue, but costs about 1,500 times this RFC's mask rows and keeps the per-layer values.
2. **A layout whose leaf point supplies random layer coordinates (R2).** Stage 1 would stack sites at different layer points. Aligning every site's layer bits would mean padding blocks to the largest size.
3. **A leaf mask per tensor, weighted to layer 0.** This RFC's first draft. An instance holding a site's later layers but not its layer 0 gets no mask, which valid claim lists produce. A mask per (site, instance) pair repairs it, but then needs a rank condition on the compiled schedule over every revealed combination. Its mask terms also persist into later stages, which complicates them.
4. **The layer sumcheck without column masks.** It removes the per-layer split values but leaves the columns unmasked.
5. **Folding the layer weights into stage 1.** Stage 1's stacked instances already sum over the layer variables with public weights, so no rounds would be added. But committed leaf tensors have no stage-1 instance, each pair's weighted sum is secret, which raises step 4's scaling problem there, and the column masks' folds would still need a masked sumcheck.

## Drawbacks

- One more masked sumcheck, of 22 rounds at 405B, with one more masking polynomial, opened by [RFC-0070](RFC-0070-mask-batching.md)'s batching.
- One more advice opening group, about 64 MiB of proof, and a blinding pair. Reducing the `u_i` once more, inside stage 1 to a column sub-point stage 1 already opens, would remove it. That is left to the implementation's cost measurements.
- #489's openings of the masking polynomials decided whether the proof stays inside 08's gate in `tessera-protocol/2`: a group per mask claim would have added 166.7 GiB. [RFC-0070](RFC-0070-mask-batching.md)'s batching adds one group, and the proof stays inside the gate, at about 2.94 GB.
- `tessera-protocol/2`'s leaf stage differs from `tessera-protocol/1`'s, so both workspaces implement both.

## Migration / Rollout

1. **This RFC.** It amends RFC-0032, RFC-0045, RFC-0037, RFC-0014, RFC-0036, RFC-0065, RFC-0013, RFC-0011, RFC-0007 and RFC-0039, and the figures of RFC-0019, RFC-0057 and 08 §5 that count the opening groups. It updates `simulator.md` §4.3, and marks R8 met for the leaf stage and R9 met for every `ChannelParams` site.
2. **The prover:** column masks, the layer sumcheck, `ρ_1 = ρ'_y` (#148, #154), and `ChannelParams`' multi-class sites (#147).
3. **The verifier** (#159): the same, written independently.
4. **The distinguisher suite** (#162). As the negative control, it tells two witnesses apart through `tessera-protocol/1`'s per-layer split values, and through its column values with the column masks set to zero. It tells neither apart through this RFC's messages.

## Testing Strategy

- **Soundness.**
  - Each of these is rejected at the stated round:
    - a false key or value claim;
    - a false outside value;
    - a false `v_{s,t}`;
    - a false `u_i`, at the opening;
    - a round polynomial off by one.
  - Soundness calibration over F65537 measures the layer sumcheck against its bound.
- **Zero knowledge.**
  - With a column mask set to zero, the column values of two witnesses that differ in one layer's activations are told apart, as the negative control. With the mask in place they are not.
  - The same holds for a claim list whose `Silu` exponent changes at layer 63, the case that broke the first draft.
- **The schedule.**
  - `ρ_1 = ρ'_y` in `tessera-protocol/2`, and no per-layer split value is sent.
  - `stage/0/split` holds exactly the outside values.
  - A `ChannelParams` site of two classes is accepted.
- **Costs.** A test counts, on the 405B layout:
  - the layered pairs and outside values, 423 and 191;
  - the instances and their columns, 25 key and 11 value, the shortest of height 22;
  - `n = 22` and `d = 4`, with the two `RsqrtRow` root inputs the only pairs whose masks reach degree 4 at 16K positions, and none past 16,384 positions;
  - that no committed tensor's mask depends on a layer variable (RFC-0013 'Committed tensors', #554).

## Open Questions

None.

## References

- [RFC-0013](RFC-0013-zero-knowledge.md), [RFC-0032](RFC-0032-zero-knowledge-tree-levels.md), [RFC-0037](RFC-0037-stage-schedule.md), [RFC-0045](RFC-0045-leaf-layout.md), [RFC-0065](RFC-0065-masked-channel-params.md).
- `docs/zk/simulator.md` §3 and §4.3.
- Chiesa, Forbes, Spooner, *A Zero Knowledge Sumcheck and its Applications*, 2017; Xie et al., *Libra*, 2019.
