# RFC-0068: The Leaf Stage in `tessera-protocol/2`: Leaf Masks and a Layer Sumcheck

- Status: Accepted
- Authors: @AbdelStark
- Created: 2026-10-25
- Target milestone: v0.4
- Amends:
  - [RFC-0013](RFC-0013-zero-knowledge.md#masked-extensions) 'Masked extensions': a layered tensor that the leaf stage reads takes a second mask row, its leaf mask;
  - [RFC-0045](RFC-0045-leaf-layout.md#the-split) 'The split', in `tessera-protocol/2`: the values of sites in the layers are sent once per site and tensor, at a random layer point, after a layer sumcheck;
  - [RFC-0037](RFC-0037-stage-schedule.md#the-layer-point) 'The layer point', in `tessera-protocol/2`: stage 1's layer point is the layer sumcheck's point;
  - [RFC-0014](RFC-0014-interaction-and-soundness.md#transcript-schedule) 'Transcript schedule' row 6 and 'Soundness calculator', in `tessera-protocol/2`: the layer sumcheck's labels and rows.
- Origin: [#486](https://github.com/AbdelStark/tessera/issues/486)

## Summary

`tessera-protocol/2` as specified reveals unmasked activation evaluations at the leaf stage (`docs/zk/simulator.md` §4.3, requirement R8). RFC-0045 evaluates every looked-up tensor once per real layer, at a point whose layer coordinates are Boolean. There the masked extension `V' = Ṽ + Z·R̃_V` equals the tensor, since RFC-0013's `Z` vanishes at any Boolean coordinate but the lowest. These evaluations leave the prover in two places:
- **the split values,** one per site, tensor and real layer: 53,489 at 405B;
- **the LogUp column values.** Each instance's key and value columns are concatenations of these per-layer evaluations (RFC-0045 'The leaf point'). The last tree level reveals the columns at the leaf point (RFC-0032 'Leaf checks'), each a public-weight sum of unmasked terms, which R8's second clause forbids.

This RFC masks both:
- **A leaf mask.** Each tensor in the layers that the leaf stage reads takes one more mask row, `R_{t,0}`. It enters the extension as `Z_x(x)·eq(y, 0)·R̃_{t,0}(x_col)`, with `y` the layer variables and `Z_x` RFC-0013's `Z` over the other variables. It vanishes on the hypercube, so every identity holds unchanged. It masks every revealed combination of layer values that weights layer 0, and RFC-0045's columns weight every real layer.
- **A layer sumcheck.** In place of the per-layer split values, a masked sumcheck over the `λ` layer variables reduces the instances' leaf claims to one value per site and tensor, at a random layer point `ρ'`, where the masked extensions are uniform. Stage 1 stacks at `ρ_1 = ρ'`.

At 405B this costs:
- 423 leaf mask rows, 846 committed rows on 2,328,576 (0.04%);
- one masked sumcheck of 7 rounds.

The split shrinks from 53,489 values to 614. `tessera-protocol/1` is unchanged.

## Motivation

- **The leak.** RFC-0045 sends `X̃_t(π_t(r_s) ‖ ℓ)` for each site `s`, tensor `t` and real layer `ℓ`. The same terms make up each instance's columns, `κ̃_I(P) = Σ_b eq(P[k_b..h], o_b/2^{k_b})·form_b + pad`. In `tessera-protocol/2` the columns are read through the tensors' masked extensions, which have no mask rows of their own (RFC-0032). Every term sits at Boolean layer coordinates, so the column values the last tree level reveals are unmasked too. `simulator.md` §4.3's "masked by those tensors' own masks once R8 is met" holds only once the per-layer evaluations are masked. As specified, they are not.
- **What #486 weighed.**
  - **One mask row per tensor and layer** (R1). It masks every per-layer evaluation. At 405B that is 53,298 `Fp2` mask rows, 106,596 committed rows, 4.6% of the advice. It keeps 53,489 split values.
  - **A layout whose leaf point gives random layer coordinates** (R2). It needs blocks for padded layers (1.59% more lookups at 405B) and a layer point per site, `P[k_s..k_s+λ]`. Stage 1 would then stack sites at different layer points, against RFC-0037's one point, so a tensor reached from two sites would take claims at two layer points.
- **The observation.** Every revealed combination of a tensor's layers weights layer 0, since layer 0 is real for every site. One mask row weighted to layer 0 therefore masks all of them, which 'Zero knowledge' argues. A layer sumcheck then removes the per-layer values themselves.

## Goals

- No leaf-stage value revealed in `tessera-protocol/2` without a uniform mask (R8).
- No change to RFC-0045's layout, to RFC-0037's single layer point, or to `tessera-protocol/1`.
- Costs computed from the compiler's 405B layout.

## Non-Goals

- **`ChannelParams` sites of two classes or more.** Their class tensor is a committed weight tensor, read at the leaf per layer, and its leaf mask would be a `ChannelParams` mask row. That changes RFC-0065's reservation, so `tessera-protocol/2` keeps refusing such sites (RFC-0007 'Verifier validation' step 4, RFC-0039 step 2.4) until a follow-up specifies it.
- **The masking polynomials' openings** (#489). The layer sumcheck adds one.
- **Grouped leaf stages** (RFC-0054). `tessera-protocol/2` accepts none.

## Proposed Design

**Notation.**
- `L` is the number of real layers, and `λ = ⌈log2 L⌉`.
- A layered tensor `X_t` is stacked over `2^λ` layer slots, with the layer variables `y` highest (RFC-0037). Its other variables are `x`, the lowest of all being `x`'s lowest.
- A padded layer's slice is 0.
- `r_s = P[0..k_s]` is site `s`'s point at the leaf (RFC-0045 'The leaf point'), and `π_t(r_s)` is tensor `t`'s coordinates in it.

### The leaf mask

For each tensor `X_t` read by a site in the layers, the masked extension of RFC-0013 gains one term:

```text
X'_t(x ‖ y) = X̃_t(x ‖ y) + Z(x ‖ y)·R̃_t(x_col) + Z_x(x)·eq(y, 0)·R̃_{t,0}(x_col)
Z_x(x) = Π_{i ≥ 2} x_i(1 − x_i)   over x's variables only
```

- **`R_{t,0}`** is the tensor's leaf mask: an `Fp2` advice mask row, two committed rows, over `x`'s column variables, like `R_t`. Masks come from the per-proof seed (INV-ZK-3).
- **On the hypercube** `Z_x(x) = 0` for every Boolean `x` of two variables or more, so `X' = X̃` there, and every sumcheck identity holds unchanged. A tensor of one non-layer variable has `Z_x = 1`, so the schedule refuses one in `tessera-protocol/2` (`ERR-INTERNAL`). At 405B the leaf's 423 layered reads have 10 to 35 non-layer variables, 379 of them 14.
- **At a Boolean layer `ℓ`**, `X'_t(x ‖ ℓ) = X̃_t(x ‖ ℓ) + [ℓ = 0]·Z_x(x)·R̃_{t,0}(x_col)`. Layer 0's evaluation is masked, the others are not.
- **At a point whose coordinates other than the lowest are random**, both mask terms are present.
- **Degree.** `X'` has degree at most 2 in each variable, as before.
- **Where the claims go.**
  - A committed tensor's claim is an advice functional over its data positions, `R_t`'s row and `R_{t,0}`'s row, all with column part `eq(p_col)`, so it stays in one opening group (RFC-0011).
  - A virtual tensor's producing relation folds both mask terms into its stage's sumcheck, as RFC-0013 folds `R_t`. So `R̃_t` and `R̃_{t,0}` are revealed only at that stage's final point.

### The layer sumcheck

In `tessera-protocol/2` the leaf stage ends as follows, after `logup/values`.

1. **The values outside the layers.** For each site outside the layers, in node order, the prover sends its tensors' values at their points, as RFC-0045 says, under `stage/0/split`. Their points' coordinates are all challenges, so `R_t` masks them.
2. **The claims.** Each instance `I` has its key claim, and its value claim when its table has values, at `P[0..h_I]` (RFC-0045). By RFC-0045's identity, each such claim is
   ```text
   claim_I = Σ_{(s,t) layered} Σ_ℓ C_{I,s,t}(ℓ)·X'_t(π_t(r_s) ‖ ℓ) + out_I + pub_I
   C_{I,s,t}(ℓ) = Σ_{b ∈ I, site s, layer ℓ} eq(P[k_b..h_I], o_b/2^{k_b})·a_{b,t}
   ```
   - `a_{b,t}` is `t`'s coefficient in block `b`'s key or value form.
   - `out_I` is the outside sites' blocks, computed from step 1's values.
   - `pub_I` collects the forms' constants `a_{b,0}`, the padding term `(1 − LT(P[0..h_I], m_I))·t_0`, and RFC-0037's native targets' public terms.
   - `C_{I,s,t}(ℓ) = 0` for a padded layer, which has no blocks.
   - The verifier computes every `C`, `out_I` and `pub_I` from public data and `P`.
3. **Batching.** The verifier draws `γ` under `stage/0/layer/gamma`, after the masked sum below, and sets
   ```text
   W_{s,t}(ℓ) = Σ_I γ^{i(I)}·C_{I,s,t}(ℓ),     H = Σ_I γ^{i(I)}·(claim_I − out_I − pub_I)
   ```
   with `i(I)` the claim's index: instances in table order, each instance's key claim, then its value claim.
4. **The sumcheck.** It proves
   ```text
   H = Σ_{y ∈ {0,1}^λ} Σ_{(s,t) layered} W̃_{s,t}(y)·X'_t(π_t(r_s) ‖ y)
   ```
   over the `λ` layer variables, masked by RFC-0013's masked sumcheck.
   - Its masking polynomial `G` is committed in Round 1 with the others.
   - The prover sends `Γ = Σ_y G(y)` under `stage/0/layer/zk-sum`, and the transcript yields `ρ_zk` under `stage/0/layer/zk-rho`, both before `γ`.
   - Round `j` sends its polynomial under `stage/0/layer/round/{j}` and draws `ρ'_j` under `stage/0/layer/r/{j}`.
   - **Degree.** In rounds before the last, the variables still summed are Boolean, so `Z`'s layer factors vanish and each round has degree 2. The last round has degree 3.
5. **The values.** The prover sends `v_{s,t} = X'_t(π_t(r_s) ‖ ρ')` for every layered site and tensor, in RFC-0045's order of sites and tensors, under `stage/0/layer/evals`.
6. **The check.** The verifier computes each `W̃_{s,t}(ρ')`, in `O(L)` per pair, and checks the last round against `Σ_{(s,t)} W̃_{s,t}(ρ')·v_{s,t} + ρ_zk·G(ρ')`. `G(ρ')` is opened as RFC-0013 opens masking polynomials (#489). A mismatch is `Reject(Sumcheck { stage: 0, round: λ })`.
7. **The result.** Each `v_{s,t}` is the stacked claim `X'_t(π_t(r_s) ‖ ρ')` of RFC-0037 'The layer point'.
   - **Stage 1's layer point** is `ρ_1 = ρ'`. `tessera-protocol/2` draws no `stage/1/layer/{c}`.
   - Stage 1 is otherwise unchanged. A virtual tensor's claim is a stage-1 instance, and a committed tensor's goes to the pool.

**Soundness.**
- A false `claim_I` survives batching with probability at most `n_claims/|Fp2|` over `γ`.
- It survives the sumcheck with probability at most `(2(λ − 1) + 3)/|Fp2|` over `ρ'`, by Schwartz–Zippel on each round.
- The masked sumcheck's `ρ_zk` adds `1/|Fp2|` (RFC-0013).
- These replace `tessera-protocol/1`'s stacking error `λ/|Fp2|` over `ρ_1`. At 405B, with `λ = 7` and 50 claims at most, the total is below `2^{-120}`.

### What is sent, in `tessera-protocol/2`

| Order | Item | Label |
|---|---|---|
| after `logup/values` | the outside sites' values | `stage/0/split` |
| | `Γ`, then `ρ_zk` | `stage/0/layer/zk-sum`, `stage/0/layer/zk-rho` |
| | `γ` | `stage/0/layer/gamma` |
| per round `j < λ` | its polynomial, then `ρ'_j` | `stage/0/layer/round/{j}`, `stage/0/layer/r/{j}` |
| | the layered values `v_{s,t}` | `stage/0/layer/evals` |
| stage 1 | as RFC-0014 row 6, with no `stage/1/layer/{c}` | |

Round 2's leaf proof carries the outside values as `splits`, and the layer sumcheck as one more `SumcheckProof` with its masked sum. Their TCE layout is schema version 2's (RFC-0036, #148, #154).

### Zero knowledge

This section covers what the verifier sees of each layered tensor `t` (`simulator.md` §4.3).

- **The column values.** The last tree level reveals each instance's columns at the two pair points `(b ‖ r')`, `b ∈ {0, 1}` (RFC-0032). `t` enters each with weight `Σ_ℓ C(ℓ)·X'_t(x_b ‖ ℓ)`, whose mask part is `C(0)·Z_x(x_b)·R̃_{t,0}(x_{b,col})`.
  - `C(0) ≠ 0`, since layer 0 has a block for every lookup of a layered site. It vanishes only if `eq(P[k..h], o/2^k) = 0`, with probability at most `h/|Fp2|`.
  - `Z_x(x_b) ≠ 0`, since `x_b`'s coordinates other than the lowest are challenges.
  - The two pair points differ in the lowest coordinate, which `x_col` includes for a tensor read over the block's lowest axis. They then give `R_{t,0}` two functionals with independent `eq` vectors.
  - A tensor read over other axes alone, such as a class over the channel axis, gives one functional at both points. That functional is still masked.
- **The layer sumcheck.** Its round messages are masked by `G` (Lemma 4). Its final values `v_{s,t}` sit at `(π_t(r_s) ‖ ρ')`, every coordinate but the lowest a challenge, so `Z ≠ 0` and `R_t` masks them. `R_{t,0}` enters them too, with weight `eq(ρ', 0)·Z_x`.
- **Stage 1** reveals `R_t` and `R_{t,0}` folded at its final point, as RFC-0013 says of every virtual tensor.
- **Jointly.** Each mask row is revealed at a few points:
  - `R_{t,0}`: the pair points of the instances reading `t`, `ρ'`'s point, and stage 1's final point;
  - `R_t`: `ρ'`'s point and stage 1's final point.

  The values are jointly uniform, apart from the public relations the verifier checks (the leaf identities and the sumcheck's equations), when the revealed points' `eq` vectors are independent (Lemma 2). For points with independent uniform coordinates that fails with probability at most `e^2/|Fp2|` per row. The compiled schedule's structured points are R10's to check (#488), and this RFC adds its leaf masks to that check.
- **R8** is met for the leaf stage. No value is revealed at a Boolean layer coordinate, and every combination of layer values is masked by `R_{t,0}`.

### Cost at 405B

The figures are computed from the compiler's layout: `LeafLayout::model` on the servable lowering with two classes, at 16K positions.

| Item | `tessera-protocol/1` | `tessera-protocol/2` with this RFC |
|---|---|---|
| Sites, of them layered | 18, 16 | the same |
| Split values | 53,489: 423 layered pairs × 126 layers, plus 191 outside | 614: 423 layered values and 191 outside |
| Split bytes | 856 KB | 10 KB |
| Leaf mask rows | — | 423 `Fp2` rows, 846 committed rows on 2,328,576 (0.04%) |
| Layer sumcheck | — | 7 rounds, the last of degree 3; one masking polynomial of `2(6·3 + 4 + 1) = 46` committed elements |

**For comparison:**
- R1's per-layer masks would add 53,298 `Fp2` rows (106,596 committed rows, 4.6%) and keep 856 KB of split values;
- R2's layout would add 81,746,788,352 lookups (1.59%) and break RFC-0037's one layer point.

## Alternatives Considered

1. **One mask row per tensor and layer (R1).** It is sound and simple to argue. It costs 126 times the mask rows of this design and keeps the per-layer values.
2. **A layout whose leaf point supplies random layer coordinates (R2).** The column values and split values would then be masked by `R_t` alone. But a site's layer point would be `P[k_s..k_s+λ]`, which depends on the site's block size, and stage 1 would stack sites at different layer points. Aligning every site's layer bits would mean padding blocks to the largest size.
3. **The layer sumcheck alone.** It removes the per-layer split values but leaves the column values unmasked. The first draft of this design made that mistake.
4. **Folding the layer weights into stage 1.** Stage 1 fixes the layer coordinates at `ρ_1` instead of summing over them, so this would add `λ` rounds to every one of its instances.

## Drawbacks

- One more masked sumcheck, and with it one more masking polynomial whose opening #489 must cover.
- `tessera-protocol/2`'s leaf stage differs from `tessera-protocol/1`'s, so both workspaces implement both.
- `ChannelParams` sites of two classes or more stay refused in `tessera-protocol/2`.

## Migration / Rollout

1. **This RFC.** It amends RFC-0013, RFC-0045, RFC-0037 and RFC-0014, and updates `simulator.md` §4.3 and R8.
2. **The prover** (#148, #154): leaf masks, the layer sumcheck, and `ρ_1 = ρ'`.
3. **The verifier** (#159): the same, written independently.
4. **The distinguisher suite** (#162). As the negative control, it tells apart two witnesses through the per-layer split values and through the column values of `tessera-protocol/1`'s leaf stage. It tells neither apart through this RFC's messages.

## Testing Strategy

- **Soundness.**
  - A false per-layer leaf claim, a false outside value and a false layered value are each rejected at their checks: the leaf identity, the layer sumcheck's rounds and its final equation.
  - Soundness calibration over F65537 measures the layer sumcheck against its bound.
- **The masks' effect.**
  - With a leaf mask set to zero, the column values of two witnesses that differ in one layer's activations are told apart, as the negative control.
  - With the leaf mask in place they are not.
- **The schedule.** `ρ_1 = ρ'` in `tessera-protocol/2`. No per-layer split value is sent. `stage/0/split` holds exactly the outside values.
- **Costs.** A test counts the 405B layout's layered pairs and outside values (423 and 191) and the leaf mask rows.

## Open Questions

None. The `ChannelParams` sites of two classes or more are a follow-up, filed with this RFC.

## References

- [RFC-0013](RFC-0013-zero-knowledge.md), [RFC-0032](RFC-0032-zero-knowledge-tree-levels.md), [RFC-0037](RFC-0037-stage-schedule.md), [RFC-0045](RFC-0045-leaf-layout.md), [RFC-0065](RFC-0065-masked-channel-params.md), `docs/zk/simulator.md` §3 and §4.3.
- Chiesa, Forbes, Spooner, *A Zero Knowledge Sumcheck and its Applications*, 2017; Xie et al., *Libra*, 2019.
