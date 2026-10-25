# RFC-0068: The Leaf Stage in `tessera-protocol/2`: Column Masks and a Layer Sumcheck

- Status: Accepted
- Authors: @AbdelStark
- Created: 2026-10-25
- Target milestone: v0.4
- Amends, in `tessera-protocol/2`:
  - [RFC-0032](RFC-0032-zero-knowledge-tree-levels.md#leaf-checks) 'Leaf checks': each LogUp instance's key and value columns take a mask row of their own;
  - [RFC-0045](RFC-0045-leaf-layout.md#the-split) 'The split': the values of sites in the layers are sent once per site and tensor, at a random layer point, after a layer sumcheck;
  - [RFC-0037](RFC-0037-stage-schedule.md#the-layer-point) 'The layer point' and 'Splits': stage 1's layer point is the layer sumcheck's, and the leaf stage's atom splits are taken there;
  - [RFC-0014](RFC-0014-interaction-and-soundness.md#transcript-schedule) 'Transcript schedule' rows 5 and 6, and 'Soundness calculator': the layer sumcheck's labels and terms;
  - [RFC-0065](RFC-0065-masked-channel-params.md#what-the-stages-owe) 'What the stages owe', [RFC-0007](RFC-0007-claim-list.md) 'Verifier validation' step 4 and [RFC-0039](RFC-0039-statement-fields.md) step 2.4: `ChannelParams` sites of two classes or more are admitted.
- Origin: [#486](https://github.com/AbdelStark/tessera/issues/486)

## Summary

`tessera-protocol/2` as specified reveals unmasked activation evaluations at the leaf stage (`docs/zk/simulator.md` §4.3, requirement R8). RFC-0045 evaluates every looked-up tensor of a site in the layers once per real layer, at a point whose layer coordinates are Boolean. There a tensor's masked extension equals the tensor, since RFC-0013's `Z` vanishes at any Boolean coordinate but the lowest. These evaluations leave the prover in two places:
- **the split values,** one per site, tensor and real layer: 53,489 at 405B;
- **the LogUp columns.** Each instance's key and value columns are concatenations of the same per-layer evaluations (RFC-0045 'The leaf point'), and RFC-0032 gives them no mask of their own. The last tree level reveals them at its pair points, as public-weight sums of unmasked terms, which R8's second clause forbids.

This RFC changes three things:
- **Column masks.** Each instance's key column, and its value column when its table has values, takes one `Fp2` mask row: `κ'_I = κ̃_I + Z·R̃_{κ,I}`. The revealed column values are then masked by the instance's own rows, whatever tensors and layers the instance holds.
- **A layer sumcheck.** In place of the per-layer split values, one masked sumcheck reduces every instance's leaf claims to one value per site and tensor at a random layer point `ρ'`, where the tensors' own masks act. The column masks' terms are folded into it, so they are revealed only at its final point. Stage 1 stacks at `ρ_1 = ρ'`.
- **Admission.** `ChannelParams` sites of two classes or more, refused in `tessera-protocol/2` until this RFC, are admitted. Their class tensor is no longer revealed per layer.

At 405B this costs 36 column mask rows, 72 committed rows on 2,328,576, and one masked sumcheck of 22 rounds. The split shrinks from 53,489 values to 614. `tessera-protocol/1` is unchanged.

## Motivation

- **The leak.**
  - RFC-0045 sends `X̃_t(π_t(r_s) ‖ ℓ)` for each site `s` in the layers, tensor `t` and real layer `ℓ`.
  - The same terms make up each instance's columns, `κ̃_I(P[0..h_I]) = Σ_b eq(P[k_b..h_I], o_b/2^{k_b})·form_b + pad`.
  - RFC-0032 reads the columns through the tensors' masked extensions, which vanish at Boolean layers. So the column values that the last tree level reveals at `(b ‖ r')` are unmasked combinations.
  - `simulator.md` §4.3's "masked by those tensors' own masks once R8 is met" holds only once the per-layer terms are masked, and as specified they are not.
- **What #486 weighed, and a third option.**
  - **One mask row per tensor and layer (R1).** It masks every per-layer evaluation. At 405B that is 53,298 `Fp2` rows, 106,596 committed rows, 4.6% of the advice. It keeps 53,489 split values.
  - **A layout whose leaf point gives random layer coordinates (R2).** It needs blocks for padded layers, 81,746,788,352 more lookups at 405B (1.59%). Its layer point per site, `P[k_s..k_s+λ]`, depends on the site's block size, so stage 1 would stack sites at different layer points, against RFC-0037's one.
  - **Masking tensors.** Masking the tensors more cleverly is fragile. Tables differ by layer (requantization tables by shift, `Silu` tables by input exponent), so an instance can hold a site's later layers but not its first. A mask keyed to one layer then misses that instance, as an adversarial read of this RFC's first draft showed with valid claim lists.
  - **Masking the columns.** The columns are what is revealed, so masking them directly does not depend on how instances and layers interleave.

## Goals

- No leaf-stage value revealed in `tessera-protocol/2` without a uniform mask (R8).
- No change to RFC-0045's layout, to RFC-0037's single layer point, or to `tessera-protocol/1`.
- Costs computed from the compiler's 405B layout.

## Non-Goals

- **The masking polynomials' openings** (#489). The layer sumcheck adds one.
- **Grouped leaf stages** (RFC-0054). `tessera-protocol/2` accepts none.
- **Claims at Boolean layers outside the leaf stage.** The epilogue's claim on the last checkpoint, `Layers::One(L − 1)` (RFC-0037), sits at a Boolean layer where no mask acts. That is filed with this RFC as its own issue.

## Proposed Design

**Notation.**
- `L` is the number of real layers, and `λ = ⌈log2 L⌉`.
- A tensor of a site in the layers is stacked over `2^λ` layer slots, the layer variables `y` highest (RFC-0037). Its other variables are `x`.
- `r_s = P[0..k_s]` is site `s`'s point (RFC-0045 'The leaf point'), and `π_t(r_s)` is tensor `t`'s coordinates in it.
- Instance `I` has a column of `2^{h_I}` entries, its leaf claims at `P[0..h_I]`.

### Column masks

For each LogUp instance `I`:
- **The rows.** The key column `κ_I` takes a mask row `R_{κ,I}`, and the value column `e_I`, when its table has values, a mask row `R_{e,I}`. Each is an `Fp2` advice mask row, two committed rows, over the column's low `c_I = min(h_I, log2 k)` variables, drawn from the per-proof seed (INV-ZK-3).
- **The extension** of the column becomes
  ```text
  κ'_I(z) = κ̃_I(z) + Z(z)·R̃_{κ,I}(z_col),     Z(z) = Π_{i ≥ 2} z_i(1 − z_i)  over the column's h_I variables
  ```
  `κ̃_I` is RFC-0045's composition of the blocks' forms, read through the tensors' masked extensions. `e'_I` is built the same way.
- **On the hypercube** `Z = 0`, so the leaf level's polynomials and every identity of RFC-0008 and RFC-0032 hold unchanged.
- **Degree.** `κ'_I` has degree at most 3 in each variable, as a masked tensor has. RFC-0032's last-round bound of 7 is unchanged.
- **The multiplicities** keep their own masked extension (RFC-0032).

### The layer sumcheck

In `tessera-protocol/2` the leaf stage ends as follows, after `logup/values`.

1. **The values outside the layers.** For each site outside the layers, in node order, the prover sends its tensors' values at their points, as RFC-0045 says, under `stage/0/split`. Their points' coordinates are all challenges, so the tensors' masks act there.
2. **The claims.** Each instance has its key claim `κ'_I(P[0..h_I])`, and its value claim when its table has values: that of RFC-0032, derived from the last level and `logup/values`. By RFC-0045's identity and the column mask,
   ```text
   κ'_I(P[0..h_I]) = Σ_{(s,t) layered} Σ_ℓ C^κ_{I,s,t}(ℓ)·X̃_t(π_t(r_s) ‖ ℓ)  +  out^κ_I  +  pub^κ_I  +  Z(P[0..h_I])·R̃_{κ,I}(P[0..c_I])
   C^κ_{I,s,t}(ℓ) = Σ_{b ∈ I, site s, layer ℓ} eq(P[k_b..h_I], o_b/2^{k_b})·a^κ_{b,t}
   ```
   - `a^κ_{b,t}` is `t`'s coefficient in block `b`'s key form.
   - `out^κ_I` is the outside sites' blocks, from step 1's values.
   - `pub^κ_I` collects the forms' constants, the padding term `(1 − LT(P[0..h_I], m_I))·t_0`, and RFC-0037's native targets' public terms.
   - `C^κ_{I,s,t}(ℓ) = 0` at a padded layer, which has no blocks.
   - Value claims are the same with `e` for `κ`.
   - The verifier computes every `C`, `out` and `pub` from public data and `P`. At a Boolean layer each tensor's masked extension equals `X̃_t`, which is why `X̃_t` appears.
3. **Batching.** The prover sends `Γ` under `stage/0/layer/zk-sum`, and the transcript yields `ρ_zk` under `stage/0/layer/zk-rho`. The verifier then draws `γ` under `stage/0/layer/gamma`. The claims are indexed by `i`: instances in table order, each instance's key claim, then its value claim. The verifier sets
   ```text
   W_{s,t}(ℓ) = Σ_i γ^i·C_{i,s,t}(ℓ),     H = Σ_i γ^i·(claim_i − out_i − pub_i)
   ```
4. **The sumcheck** proves `H` as the sum of two kinds of instances, batched by RFC-0010 'Batching within a stage' over `n = max(λ, max_I c_I)` variables. Each instance takes the last of the `n` challenges its variables need, scaled by `2^{n − n_i}`.
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
   - **Rounds.** Round `j` sends its polynomial under `stage/0/layer/round/{j}`, then draws `ρ'_j` under `stage/0/layer/r/{j}`.
   - **Degrees.** A fold has degree 2 in each variable. The layer instance has degree 2 except in its last round, which has degree 3: until then its remaining layer variables are Boolean, so `Z`'s layer factors vanish. At 405B, `n = 22`: 21 rounds of degree 2, then 1 of degree 3.
5. **The final values.** The prover sends, under `stage/0/layer/evals`, in this order:
   1. `v_{s,t} = X'_t(π_t(r_s) ‖ ρ'_y)` for every layered site and tensor, in RFC-0045's order of sites and tensors. `ρ'_y` is the layer instance's point, the last `λ` challenges.
   2. `u_i = R̃_{·,I(i)}(ρ'_{w_i})` for every column mask, in claim order. `ρ'_{w_i}` is fold `i`'s point, the last `c_i` challenges.
6. **The check.** The verifier computes each `W̃_{s,t}(ρ'_y)`, in `O(L)` per pair, and each fold's public factor. It then checks the last round against the batched final value plus `ρ_zk·G(ρ')`. `G(ρ')` is opened as RFC-0013 opens masking polynomials (#489).
   - A round whose polynomial does not match the previous claim is `Reject(Sumcheck { stage: 0, round: j })`.
   - A final mismatch is `Reject(Sumcheck { stage: 0, round: n })`.
7. **The result.**
   - **Each `v_{s,t}`** is the stacked claim `X'_t(π_t(r_s) ‖ ρ'_y)` of RFC-0037 'The layer point', and stage 1's layer point is `ρ_1 = ρ'_y`. `tessera-protocol/2` draws no `stage/1/layer/{c}`.
   - **A stacked claim that RFC-0037 splits into atoms** is split at that point. Its atom values follow, under `stage/0/layer/split`, in RFC-0037's order.
   - **Each `u_i`** is a claim on its mask row at `ρ'_{w_i}`, which joins the advice pool. Its column functional is `eq(ρ'_{w_i}, ·)`.
   - **Stage 1** is otherwise unchanged.

**Soundness.**
- A false `claim_i` survives batching with probability at most `n_claims/|Fp2|` over `γ`.
- The sumcheck adds at most `Σ_j d_j/|Fp2|` over `ρ'`, where `Σ_j d_j = 2(n − 1) + 3`.
- The masked sumcheck's `ρ_zk` adds `1/|Fp2|` (RFC-0013).
- These replace `tessera-protocol/1`'s stacking error of `λ/|Fp2|` over `ρ_1`. At 405B, with `n = 22` and 36 claims, the total is `(36 + 45 + 1)/|Fp2| < 2^{-121}`.
- **The calculator** (RFC-0014) counts, in `tessera-protocol/2`:
  - `n − 1` rounds of degree 2 and one of degree 3;
  - one more stage of `m = n_claims` instances, with its `ρ` term;
  - no stacking term at stage 1.

### What is sent, in `tessera-protocol/2`

| Order | Item | Label |
|---|---|---|
| after `logup/values` | the outside sites' values | `stage/0/split` |
| | `Γ`, then `ρ_zk` | `stage/0/layer/zk-sum`, `stage/0/layer/zk-rho` |
| | `γ` | `stage/0/layer/gamma` |
| per round `j < n` | its polynomial, then `ρ'_j` | `stage/0/layer/round/{j}`, `stage/0/layer/r/{j}` |
| | the `v_{s,t}`, then the `u_i` | `stage/0/layer/evals` |
| | the atom values of split stacked claims | `stage/0/layer/split` |
| stage 1 | as RFC-0014 row 6, with no `stage/1/layer/{c}` | |

Round 2's leaf proof carries the outside values as `splits`, plus the layer sumcheck with its masked sum, final values and atom splits. Their TCE layout is schema version 2's (RFC-0036, #148, #154). Every message is absorbed before the challenge that follows it (INV-FS-1). The column masks, like every mask row, are committed in Round 1.

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
  - `R_t` is then revealed at that point and at stage 1's final point, as in every stacked claim. R10's check (#488) covers those points as before.
- **The outside values** sit at points of challenges and are masked by their tensors' own masks.
- **R8.** No tensor or column of the leaf stage is revealed alone at a point where its mask vanishes, and every revealed sum of such terms carries a column mask. R8 is met for the leaf stage.
- **R9.** `ChannelParams`' class tensors are read at the leaf per layer, but those evaluations are now revealed only inside masked columns. Their stacked claim sits at `(π(r_s) ‖ ρ'_y)`, with random coordinates, where RFC-0065's per-claim masks act. So sites of two classes or more meet RFC-0065 'What the stages owe', and `tessera-protocol/2` admits them.

### Cost at 405B

The figures are computed from the compiler's layout: `LeafLayout::model` on the servable lowering with two classes, at 16K positions.

| Item | `tessera-protocol/1` | `tessera-protocol/2` with this RFC |
|---|---|---|
| Sites, of them in the layers; instances | 18, 16; 25 | the same |
| Split values | 53,489: 423 layered pairs × 126 layers, plus 191 outside | 614: 423 at `ρ'` and 191 outside, plus 36 `u_i` |
| Split bytes | 856 KB | 10.4 KB |
| Column mask rows | — | 36 `Fp2` rows (25 key and 11 value columns): 72 committed rows on 2,328,576 |
| Layer sumcheck | — | 22 rounds, 21 of degree 2 and 1 of degree 3; masking polynomial `2(21·3 + 4 + 1) = 136` committed elements |

**For comparison:**
- R1 would add 53,298 `Fp2` rows (106,596 committed, 4.6%) and keep 856 KB of split values;
- R2 would add 81,746,788,352 lookups (1.59%) and give each site its own layer point.

## Alternatives Considered

1. **One mask row per tensor and layer (R1).** It is sound and simple to argue, but costs about 1,500 times this RFC's mask rows and keeps the per-layer values.
2. **A layout whose leaf point supplies random layer coordinates (R2).** Stage 1 would stack sites at different layer points. Aligning every site's layer bits would mean padding blocks to the largest size.
3. **A leaf mask per tensor, weighted to layer 0.** This RFC's first draft. An instance holding a site's later layers but not its layer 0 gets no mask, which valid claim lists produce. A mask per (site, instance) pair repairs it, but then needs a rank condition on the compiled schedule over every revealed combination. Its mask terms also persist into later stages, which complicates them.
4. **The layer sumcheck without column masks.** It removes the per-layer split values but leaves the columns unmasked.
5. **Folding the layer weights into stage 1.** Stage 1 fixes the layer coordinates at `ρ_1` rather than summing over them, so this would add `λ` rounds to every one of its instances.

## Drawbacks

- One more masked sumcheck, of 22 rounds at 405B, with one more masking polynomial whose opening #489 must cover.
- `tessera-protocol/2`'s leaf stage differs from `tessera-protocol/1`'s, so both workspaces implement both.

## Migration / Rollout

1. **This RFC.** It amends RFC-0032, RFC-0045, RFC-0037, RFC-0014, RFC-0065, RFC-0007 and RFC-0039. It updates `simulator.md` §4.3, and marks R8 met for the leaf stage and R9 met for every `ChannelParams` site.
2. **The prover:** column masks, the layer sumcheck, `ρ_1 = ρ'_y` (#148, #154), and `ChannelParams`' multi-class sites (#147).
3. **The verifier** (#159): the same, written independently.
4. **The distinguisher suite** (#162). As the negative control, it tells two witnesses apart through `tessera-protocol/1`'s per-layer split values, and through its column values with the column masks set to zero. It tells neither apart through this RFC's messages.

## Testing Strategy

- **Soundness.**
  - Each of these is rejected at the stated round:
    - a false key or value claim;
    - a false outside value;
    - a false `v_{s,t}`;
    - a false `u_i`;
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
  - the instances and their columns, 25 key and 11 value;
  - `n = 22`.

## Open Questions

None.

## References

- [RFC-0013](RFC-0013-zero-knowledge.md), [RFC-0032](RFC-0032-zero-knowledge-tree-levels.md), [RFC-0037](RFC-0037-stage-schedule.md), [RFC-0045](RFC-0045-leaf-layout.md), [RFC-0065](RFC-0065-masked-channel-params.md).
- `docs/zk/simulator.md` §3 and §4.3.
- Chiesa, Forbes, Spooner, *A Zero Knowledge Sumcheck and its Applications*, 2017; Xie et al., *Libra*, 2019.
