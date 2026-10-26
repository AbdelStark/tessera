# RFC-0013: Zero Knowledge: Masked Sumchecks, Masked Extensions, and Hiding Openings

- Status: Accepted
- Authors: @AbdelStark
- Created: 2026-10-15
- Target milestone: v0.4
- Amended by: [RFC-0032](RFC-0032-zero-knowledge-tree-levels.md) ('What the verifier sees': tree levels and LogUp roots; 'Masked extensions': `Z` spares the lowest variable; 'LogUp and product-tree roots'; OQ-0013-2), [RFC-0065](RFC-0065-masked-channel-params.md) ('Masked extensions': each `ChannelParams` claim's own mask row, with `Z` over its type tensor's coordinates; 'Hiding openings', 'Simulator argument' step 4 and INV-ZK-2: the pair a matrix's evaluation rows share), [RFC-0068](RFC-0068-leaf-stage-zero-knowledge.md) ('Masked extensions': `Z` vanishes until the round that binds the last of its variables; 'Masked sumcheck': the raised round's `g_j`; 'Cost': the column masks' opening group), [RFC-0069](RFC-0069-one-layer-claims.md) ('Masked extensions' and 'What the verifier sees': a claim at one layer of a stacked committed kind is on the layer's tensor, with its own mask row), [RFC-0070](RFC-0070-mask-batching.md) ('Masked sumcheck' step 4: no `G_k(r)` is opened alone, the masks' batching opens every one; 'Cost': its group and its pad; INV-ZK-2: the pad's use)
- Superseded in part by: [RFC-0065](RFC-0065-masked-channel-params.md) ('Cost': the weight budget per proof, `2e + 4` rows for a matrix of `e` claims)

## Summary

Tessera is made zero-knowledge with four mechanisms, each tied to one kind of revealed data: (1) every sumcheck is masked with a committed random polynomial so its round messages are simulatable; (2) every tensor whose evaluation crosses a stage boundary is replaced by a masked extension `V' = Ṽ + Z·R_V`, which agrees with `Ṽ` on the hypercube but has uniformly random evaluations off it; (3) every commitment uses vanishing-polynomial encoding masks so that opened columns are uniform, and every revealed combined row includes a fresh blinding row; (4) every tree level is a masked sumcheck over masked extensions that spare its pair bit, and LogUp roots are never sent: a masked sumcheck proves the LogUp identity at them ([RFC-0032](RFC-0032-zero-knowledge-tree-levels.md)). Challenges derive from a coin-tossed seed, so the construction holds against a malicious verifier. Layers are sumcheck variables, so each proof reveals one masked evaluation per tensor kind rather than one per layer, which keeps the weight-commitment budget affordable.

## Motivation

The verifier is a counterparty state; weights, prompts, outputs and activations must stay hidden from it even if it deviates (source §2). The source's default is "Libra-style masked sumchecks with small committed masking polynomials, plus hiding commitments" (source §7, "Zero knowledge"; §10). A non-ZK proof leaks, per stage, partial sums of witness polynomials, evaluations of every tensor at random points, combined rows that are linear combinations of committed rows, and LogUp roots that are functions of every key. Each needs a mask.

## Goals

- Statistical zero knowledge against a malicious verifier, with simulation error below `2^{-100}` for 405B-sized proofs.
- Prover overhead below 2% of committed elements and negligible sumcheck work.
- A written simulator argument reviewed before implementation (RISK-6).
- Budget accounting for every use of persistent randomness (weight blinding rows).

## Non-Goals

- Hiding the architecture, sequence lengths or `Û` (public by statement design).
- A ZK-STARK wrapper (source §10 alternative; rejected below).
- Side channels outside the protocol messages; message timing is handled by INV-ARCH-2 and [RFC-0018 'Release schedule'](RFC-0018-gpu-prover-runtime.md#release-schedule).

## Proposed Design

### What the verifier sees and how each item is hidden

| Revealed item | Mechanism | Randomness source |
|---|---|---|
| Commitment roots | Hash of columns whose symbols are uniformly masked | Encoding masks |
| Seed | Coin toss (live) or hash with proof-of-work (non-interactive) | Both parties |
| Sumcheck round messages | Masked sumcheck | Committed masking polynomial `G_k` over `Fp2` per stage |
| Final evaluations of each stage | Masked extensions | One `Fp2` mask row `R_V` per stacked tensor kind, and in `tessera-protocol/2` one per kind and layer claimed at one layer ([RFC-0069](RFC-0069-one-layer-claims.md)) |
| Tree-level and LogUp-identity round messages | Masked sumcheck ([RFC-0032](RFC-0032-zero-knowledge-tree-levels.md)) | `G_ℓ` per tree level, and the identity's |
| Tree-level evaluations at the pair bit, and their mask values | Masked extensions that spare the pair bit ([RFC-0032](RFC-0032-zero-knowledge-tree-levels.md)) | One `Fp2` mask per level and polynomial, a block of [RFC-0070](RFC-0070-mask-batching.md)'s masks' block |
| LogUp roots | Not sent: the identity sumcheck's final evaluations, masked ([RFC-0032](RFC-0032-zero-knowledge-tree-levels.md)) | The root levels' masks |
| Combined rows `u_g` | Blinding rows | One fresh `Fp2` blinding row per revealed combination |
| Opened columns | Vanishing encoding masks | `P_A` (advice) or `P_W` (weights) coefficients per row |
| `Û`, lengths, digests | Public by design | — |

**Masks over `Fp2`.** Committed values are in `Fp`, but every value in the table above but the columns is in `Fp2` (`Fp[X]/(X^2 − 7)`). A mask drawn from `Fp` and scaled by any public coefficient covers one `Fp`-dimension of an `Fp2` value and leaves the other one unmasked:
- an `Fp` blinding row leaves `Σ_row Im(c_g(row))·A_row` in the clear, `k` linear combinations of the witness, or of the weights, per combined row;
- `Fp` coefficients of a masking polynomial leave each round-message coefficient's component off the line through `ρ_k`.

Every mask in this RFC is therefore uniform over `Fp2`. An `Fp2` mask value is two committed `Fp` values, and an `Fp2` row is two committed `Fp` rows, `B = B^{(0)} + X·B^{(1)}`, which enter a combination with the coefficients `1` and `X`.

### Stacked layers

Each stage treats per-layer instances of the same tensor kind as one stacked tensor with `⌈log2 n_layers⌉` layer variables (padded layers are zero). A stage therefore ends with one claim per stacked tensor at a point `(ρ_ℓ, r)` that includes a layer coordinate. This is equivalent to the batching of [RFC-0010](RFC-0010-proof-schedule.md#batching-within-a-stage) with `eq(ρ_ℓ, ℓ)` in place of powers of `γ`. It matters here because every revealed evaluation consumes mask randomness: one per tensor kind per proof instead of 126.

### Masked sumcheck

For a stage proving `H = Σ_x f(x)` over `n` variables with round degree `d`:

1. In Round 1, the prover commits `G_k(x) = a_0 + Σ_{j=1}^{n} g_j(x_j)` with each `g_j` a random univariate polynomial over `Fp2` of the degree `d` that the sumcheck's round messages hold in every round (RFC-0010 'Sumcheck messages'), at least every round's degree, the round whose degree masked extensions raise included: up to 7 at a tree level, and up to 10 at a reduction stage ('Masked extensions' 'Degree'; [RFC-0070](RFC-0070-mask-batching.md)). Each coefficient, and `a_0`, is two values of the advice mask block. A lower degree would leave a coefficient of that round's message unmasked ([`docs/zk/simulator.md`](../zk/simulator.md) Lemma 4, R4).
2. In the stage, the prover sends `Γ = Σ_x G_k(x)`; the transcript yields `ρ_k ∈ Fp2` (label `stage/{k}/zk-rho`).
3. The parties run the sumcheck on `f + ρ_k G_k` with claim `H + ρ_k Γ`.
4. At the end, the prover sends `G_k(r)` last among the stage's final evaluations, and the final check reads it. In `tessera-protocol/2` it is a claim on the masks' block, which [RFC-0070](RFC-0070-mask-batching.md)'s batching sumcheck opens with every other mask claim: `G_k(r)`'s column functional is not an `eq` vector, so no RFC-0011 group opens it alone.

The round messages are then simulatable from `H`, `Γ` and the final evaluation (Chiesa, Forbes, Spooner 2017; Libra, Xie et al. 2019), because each of their non-constant coefficients is masked by a uniform `Fp2` value. Cost: `2(n(d + 1) + 1)` committed elements per stage.

### Masked extensions

For every stacked tensor `V` whose evaluation is revealed (as a stage output claim or an opening claim), define

```math
V'(x) = \tilde V(x) + Z(x) \cdot \tilde R_V(x_{col}), \qquad Z(x) = \prod_{i \ge 2} x_i (1 - x_i)
```

where `R_V` is a random `Fp2` advice row, two committed rows, over `V`'s `c` column variables `x_col`:
- **a committed tensor's** are the coordinates of its data claim's column sub-point, `c = min(n_P, log2 k)` with `n_P` the variables of the tensor as the advice layout places it: one layer's tensor for a stacked kind, one attention tile's share for a tiled tensor ('Committed tensors', [#554](https://github.com/AbdelStark/tessera/issues/554));
- **a virtual tensor's** are the low `c` of its `n_V` variables, `c = min(n_V, log2 k)`, and `c = min(n_V, log2 k − 1)` when a reduction stage folds its mask ('Virtual tensors', [#488](https://github.com/AbdelStark/tessera/issues/488)). A stacked tensor's layer variables are its highest, so they are columns only as far as `c` reaches: all of them when `c = n_V`.

Properties:

- **`Z` spares the lowest variable `x_1`** ([RFC-0032](RFC-0032-zero-knowledge-tree-levels.md)). It therefore also masks a point whose lowest coordinate is Boolean: a LogUp leaf level reads its keys, values and multiplicities there, at its pair bit. A masked tensor has two variables at least. So in `tessera-protocol/2` the token axis pads to at least four rows, `2^{max(⌈log2 n⌉, 2)}`, and so does a batch's slot ([RFC-0071](RFC-0071-request-slots.md#the-slots)). An axis padded to four rows is laid out as a padded axis ([RFC-0047](RFC-0047-advice-order.md#the-order-of-a-tensors-variables)), so `n = 1`, 2 and 3 share one layout. Every tensor indexed by token, such as the tokens and the per-position sums of squares at `n ≤ 2`, then has two variables, and RFC-0035's padded rows fill those past `n`. R10's check refuses a schedule that still reveals a tensor of fewer ([#488](https://github.com/AbdelStark/tessera/issues/488)).
- `V' = Ṽ` on the Boolean hypercube, so every sumcheck identity over the hypercube holds unchanged with `V'` in place of `Ṽ`.
- `Z` vanishes at every summation point of every round before the one that binds the last of its variables, so the prover's round computations change only from that round on. In a sumcheck over `V`'s variables that is the last round. A LogUp leaf level's last round binds the tree index, which no `Z` of its keys, values or multiplicities holds, so there it is the round before the last ([RFC-0068](RFC-0068-leaf-stage-zero-knowledge.md), RFC-0032).
- `V'(p) = Ṽ(p) + Z(p) · R̃_V(p_col)`. For `p` whose coordinates other than the lowest are off `{0, 1}`, `Z(p) ≠ 0`, and the values `R̃_V(p_col^{(i)})` at the at most `e_V` revealed points are jointly uniform whenever the vectors `eq(p_col^{(i)}, ·)` are linearly independent over `Fp2`, which fails with probability at most `e_V^2 / |Fp2|` for points whose column coordinates are independent challenges. Points that share column coordinates, and in particular values that differ only in layer or tile bits, which are dependent with certainty, are R10's ('Committed tensors' 'Hiding'). An `Fp` row would need the `2e_V` real and imaginary parts of those vectors independent over `Fp`, which Schwartz–Zippel bounds only by about `2e_V·log2(k)/|Fp|`, near `2^-55`: hence the `Fp2` row.
- **Committed tensors.** `V'(p)` is a linear functional of the advice vector (data positions plus `R_V`'s row) whose column part is `eq(p_col ‖ 0^{log2 k − c})` in both terms, so it falls in the same opening group as the data claim ([RFC-0011](RFC-0011-advice-commitment.md#opening-protocol)). That fixes the mask's columns ([#554](https://github.com/AbdelStark/tessera/issues/554)):
  - **Its columns.** `p_col` is the part of `p` that RFC-0011 rule 3 puts in the claim's column sub-point, in that order: the low `c = min(n_P, log2 k)` variables, in layout order ([RFC-0047](RFC-0047-advice-order.md#the-order-of-a-tensors-variables), [RFC-0034](RFC-0034-causal-blocks.md#causal-layout)), of the tensor the layout places, of `n_P` variables. `R̃_V` is the `c`-variable extension of `R_V`'s first `2^c` entries. No claim reads the others, but the row holds fresh mask-stream values in all `k` entries, its entries past `2^c` included, as a virtual tensor's does (INV-ZK-3). So every tensor's mask row, committed or virtual, is uniform in all its entries: the proximity row's argument draws on that (`docs/zk/simulator.md` §4.4, Lemma 5), and the row's contents, which the commitment hashes, are fixed by the mask stream alone. Other mask rows follow their own RFCs and are not covered here: RFC-0070's `M` and `S` hold public zeros at their public positions, and RFC-0068's column masks below a row leave their entries past `2^{c_I}` unread and unspecified. §4.4 counts none of them as a free row.
  - **What it leaves out.** Every other coordinate of `p` is in the claim's row part, and `R̃_V` does not depend on it:
    - a stacked kind's `λ` layer variables, since the layout places the kind as `L` tensors, one per layer (RFC-0011 rule 2);
    - an attention tile's index bits, its KV head's and its query block's, since each tile's share is a tensor of its own (RFC-0011 rule 2, [RFC-0034](RFC-0034-causal-blocks.md#causal-layout));
    - the placed tensor's own variables past the low `log2 k`.

    `Z` still runs over all of `V`'s `n_V` variables, these included.
  - **Every case.** `n_P ≥ log2 k`: the mask takes the low `log2 k` and the rest are row bits. `n_P < log2 k`: it takes all `n_P`, and the layer and tile bits are row bits even where a row could hold them. `λ = 0` (one layer, or a kind outside the layers) and a tensor of one tile leave out nothing but the variables past `log2 k`. A tile of `log2 k` variables or more is the first case. A mask of `2^c` entries holds at most `2^c` independent functionals, so a kind reveals at most `2^c` values with independent functionals per proof, whatever its layers and tiles. Values at one point are one functional: RFC-0069's 33 claims at 2 points are two. Below a row that is `2^{n_P}` across all the kind's layers and tiles, 4 for a per-position kind at `n ≤ 4`. R10's check holds the values it covers, on the claim lists it runs, to that bound. A dependency found later among the uncovered values needs more mask capacity, a change to this rule, not to the gate.
  - **Degree.** Under the rule a committed stacked factor has degree 2 in its top layer variable, its mask holding no layer variable. It had 3 only where the rejected reading's `c` reached that variable. The stage's degree in `tessera-protocol/2` is #565's ([#565](https://github.com/AbdelStark/tessera/issues/565)).
  - **The group.** `R_V`'s two committed rows take `γ_claim·a·Z(p)` and `γ_claim·a·Z(p)·X` in the claim's group, once per piece. Beside them the data rows take `γ_claim·a·w(ℓ)·eq(row part of (r ‖ base_ℓ), row)` per layer image, with RFC-0034's tile weights for a tiled piece ([RFC-0037](RFC-0037-stage-schedule.md#the-claim-pool) 'The claim pool', `a` the piece's coefficient, RFC-0011 step 2). The mask adds no group.
  - **The layer input.** `X_in` (RFC-0037 'Claims and atoms') is one stacked kind with a mask row of its own, distinct from `c_0`'s and `X_last`'s, whose columns are its slices' common column sub-point. A claim on it is one piece at `(r ‖ ρ)`, its slices `c_0` and `X_last[ℓ − 1]` its data rows. It is never a `c_0` piece plus an `X_last` piece with shifted weights (`docs/zk/simulator.md` §4.3, R8).
  - **Hiding.** The values at `V`'s `e_V` revealed points are jointly uniform when their vectors `eq(p_col^{(i)} ‖ 0)` are independent (Lemma 2). Two values whose points differ only in coordinates the mask leaves out, two layer points or two tiles at one column sub-point, share one vector: a combination of them cancels `R̃_V` and reveals `Z(p')·Ṽ(p) − Z(p)·Ṽ(p')`. R10's check refuses such a pair on the compiled schedule (`cargo xtask check-mask-rank`, #488), and finds none among the values it covers.
    - **What the check covers:** the compiled schedule's pool atoms on committed advice tensors, the leaf stage's claims on gadget inputs and RFC-0069's one-layer claims among them; virtual tensors' stage values and their folds; and every weight matrix's claims, with `β_T`.
    - **What it does not cover yet.** This is the one list of R10's gap; `tessera_compiler::schedule::masks` documents the same ten items, and RFC-0068 and `docs/zk/simulator.md` R10 refer to it:
      1. the leaf stage's values of gadget outputs and auxiliary columns at their leaf read points, RFC-0068's `v_{s,t}` on value forms;
      2. the leaf reads of a tensor over some of its block's axes (`π_t`, RFC-0045 'The split');
      3. RFC-0068's column masks, their values at the last tree level's pair points and at `ρ'_{w_i}`, and the layer sumcheck's folds;
      4. `SoftmaxRow`'s reciprocal at its row point, its maximum, `Exp` at its score-leaf point and its row sums at key coordinates `½` (RFC-0030, RFC-0037 'The claim pool');
      5. the native relations' raw evaluations, on graph tensors and on auxiliary columns at the natives' own points: the root's bytes and input limbs in each coefficient product, `SoftmaxRow`'s maximum in its key, forest and padding relations, the greedy certificate's reads of the one-hot tokens, and the greedy maximum `M`, a certificate source, in its key and selected relations (RFC-0050, RFC-0030, RFC-0053);
      6. the multiplicities' claims (RFC-0032);
      7. the tree levels' masks, the identity sumcheck's root levels and the `SoftmaxRow` leaf mask (RFC-0032, RFC-0070);
      8. a committed group whose layer weights carry per-layer constants, whose mask no document defines yet ([#556](https://github.com/AbdelStark/tessera/issues/556));
      9. batches of several requests (RFC-0071);
      10. the anchor's circuits (RFC-0066, RFC-0072), whose schedules the gate does not lower: their committed bits and bytes, which RFC-0072 leaves to these masks and to R10 once they join a batch's graph ([#146](https://github.com/AbdelStark/tessera/issues/146), [#526](https://github.com/AbdelStark/tessera/issues/526)).

      Items 1, 2, 4, 5, 6 and 8 to 10 add functionals on committed tensors' masks, which under this rule every layer and tile of a kind shares. The tightest case is `n ≤ 4`: a per-position kind's mask has 4 entries for all its layers, and the covered values already use one. So #154 covers items 1 and 5 at `n = 3` and 4 first.
    - **This blocks #154 and #159.** Neither may ship `tessera-protocol/2`'s masks until R10's check covers every item of the list, or a written argument shows each item's values independent of the others on its row.
    - **The prover runs R10.** The CI gate runs the conformance claim lists and 405B's alone. R10 protects the prover's secrets, so in `tessera-protocol/2` the prover runs R10's check on the composed schedule of the claim list, length and request it proves, and refuses with `ERR-INTERNAL` a schedule with a finding, before it commits anything (#154). The verifier need not: a schedule that fails R10 harms only the prover that proves it. The check costs seconds per proof. The conformance claim lists' leaf layouts read every tensor at one site and one point (`crates/tessera-prover/tests/leaf_read_points.rs`). Once R10's check covers item 1, #154's run must also verify that property on the claim list it proves.
  - **Why not the layer variables too.** A mask over the stacked tensor's low `min(n_P + λ, log2 k)` variables would put its term at the column sub-point `p_col ‖ ρ`, where no data claim lies. `V'(p)` would then leave its group, and opening its two terms in two groups reveals `Ṽ(p)` (`docs/zk/simulator.md` §7). It would still leave the layer variables past `log2 k` out whenever `n_P + λ > log2 k`, so R10's check would still be needed.
  - **One layer.** In `tessera-protocol/2`, when the kind has a layer coordinate (`λ ≥ 1`), a claim at one layer `ℓ` of a stacked committed kind is on the layer's tensor in its own variables, `V = X_κ[ℓ]`, which takes its own row `R_V`: the stacked kind's `Z` vanishes at the Boolean layer ([RFC-0069](RFC-0069-one-layer-claims.md)). Its columns follow the same rule, the layer's tensor being the placed one.
  - **Weights** read their mask rows the same way, at the claim's column point on its matrix: a matrix's `R_W` enters the evaluation row its claim's column point reads ([RFC-0012](RFC-0012-weight-commitment.md#opening-by-the-folding-pass) 'Opening by the folding pass'), and each `ChannelParams` claim's `R_i` is read as `⟨R_i, eq(col(x), ·)⟩`, a column point that can hold layer coordinates ([RFC-0065](RFC-0065-masked-channel-params.md#the-masks)).
- **Virtual tensors.** The producing relation proves `V'(p) = Σ_x rel(p, x)·inputs(x) + Z(p)·R̃_V(p_col)`. The second term never becomes a claim the verifier sees: it joins the stage's sumcheck as a fold, so the verifier learns `R̃_V` only at the fold's final point (as in Libra's treatment of layer masks). In a reduction stage of RFC-0010 the fold is as follows ([#538](https://github.com/AbdelStark/tessera/issues/538)); RFC-0032's tree levels and RFC-0030's `SoftmaxRow` leaf fix their own folds, inside their instances' summands.
  - **Its variables.** The fold runs over `log2 k` variables whatever `V`'s `c = min(n_V, log2 k − 1)`: `Σ_{w ∈ {0,1}^{log2 k}} Z(p)·eq(p_col ‖ 0^{log2 k − c}, w)·R_V[w]`. A mask row is a row of `k` entries, of which `V`'s extension reads the first `2^c`, so the sum is `Z(p)·R̃_V(p_col)`, unchanged. Every fold of a stage then ends at one point, the stage's last `log2 k` challenges `r''`.
  - **The whole row is random.** A virtual tensor's mask row holds fresh mask-stream values in all `k` entries, its entries past `2^c` included (INV-ZK-3). The fold's final value reads them all, so it is independent of `R_V`'s values at points of `c` coordinates. With zeros there, `u` would be `Π_{j > c}(1 − r''_j)·R̃_V(r''_low)`: a same-stage reveal of `V'` at `q_col = r''_low` would then give `Ṽ(q)`, a dependence a rank check over full-width `eq` vectors does not see.
  - **An instance of its own.** The relation's instance has `n_i` variables, which need not be `R_V`'s: `MatMulWeight`'s, `MatMulAct`'s and `SumSquares`' run over an axis the output does not have. So the fold is an instance of its own, never part of the relation's summand.
    - It takes its relation's batching coefficient and adds no power of `γ`.
    - Its summand is scaled by `2^{log2 k − n_i}`. Front-padded to the stage's `n` rounds ([RFC-0010](RFC-0010-proof-schedule.md#batching-within-a-stage)), it then contributes `2^{n − n_i}·Z(p)·R̃_V(p_col)`.
    - With the relation's `2^{n − n_i}·Ṽ(p)`, the two instances contribute RFC-0010's `2^{n − n_i}·V'(p)`, the scaled claim of a value the verifier knows. The batching needs neither instance's own sum, and both stay secret.
  - **The stage's rounds** are `n = max(max_i n_i, log2 k)` when the stage has a fold: a fold's variables count among its instances'.
  - **Its final value** `u = R̃_row(r'')`, the `log2 k`-variable extension of `R_V`'s whole row, is sent after every instance's evaluations, one per fold in instance order, under `stage/{k}/evals`, and counts among the evaluations RFC-0010's length check expects. The final check takes `2^{log2 k − n_i}·Z(p)·eq(p_col ‖ 0^{log2 k − c}, r'')·u`.
  - **`u` is a claim on `R_V`'s row** at `r''`, its column functional `eq(r'', ·)`, and joins the advice pool (INV-SCH-3). It is what binds the fold: a prover that ran the fold on another row would pass every round and the final check, and fails only at the opening, `Reject(Evaluation { commitment: Advice })`.
  - **The opening.** A stage's `u` share the column sub-point `r''`, so a reduction stage with folds adds at most one opening group of RFC-0011. At 405B that is two, stages 1 and 2: their 16 and 96 folds in the model's schedule with no request, 15 and 98 with a generation request (RFC-0053), of masks of 14 and 21 column variables at 16K tokens.
  - **Revealed points.** `r''` is one of `R_V`'s revealed points (R10). A virtual tensor's `c` stays below `log2 k`, so `u` also reads the row's entries past `2^c`, which `V'` never reads, and stays independent of every `R̃_V(p_col)`.
    - **Why.** With `c = log2 k`, `u = R̃_V(r'')`. A stage that also revealed `V'` at a point `q` with `q_col = r''` would then reveal `Ṽ(q) = V'(q) − Z(q)·u`.
    - **It happens.** At 405B (`max_context` 131,072) it happens for every `n` in `(16384, 32768]`. The `SumSquares` outputs at layer positions 0 and 38 have `n_V = 15 + 7 = 22 = log2 k`. Stage 1 sends their `V'` at its last 22 challenges, the point of their own folds in the same stage.
    - **The cost** is none in rows: the row has `k` entries either way. The narrower `c` can make two values of `V` that differ only in `V`'s `log2 k`-th variable share a column point, which `c = log2 k` kept apart. R10's check refuses such a pair, and finds none in the compiled schedules.
    - **The check.** R10's check (`cargo xtask check-mask-rank`, #488) runs on the compiled schedule and refuses such a reveal, which it finds under `c = min(n_V, log2 k)`.
  - **Folds with no relation instance,** such as [RFC-0068](RFC-0068-leaf-stage-zero-knowledge.md#the-layer-sumcheck)'s column masks, scale the summand by `2^{−(n − c)}`, and the sumcheck starts from the public sum.
- **Degree.** `V'` has degree 1 in the tensor's lowest variable, which `Z` spares. In each other variable `Z` has degree 2, and `R̃_V` adds 1 when the variable is one of its columns: `V'` has degree 3 there, or 2. In `tessera-protocol/2` a reduction stage's degree `d`, RFC-0010's largest degree among its instances ('Sumcheck messages'), takes each instance's degree from its relation alone: never from a mask's columns, nor from the variable a round binds ([#565](https://github.com/AbdelStark/tessera/issues/565)).
  - **Terms and factors.** An instance's summand is its relation's, as the RFC that defines the relation writes it: a sum of product terms. A factor of a term is `eq`, a public table, or a read of a tensor. Public tables include selectors, kernels, a constant per channel or per layer, `W_b` and RFC-0053's coefficients. A scalar coefficient is no factor. A linear combination of one tensor's values at several points is one read, as RFC-0053's `Hplus` is, not one read per point. One tensor read twice in one term, as in `X̃²` or `h̃·(1 − h̃)`, is two reads.
  - **The rule.** For each term `t`, `d_1(t)` counts its factors, and `s(t)` counts its reads of a committed or virtual tensor, a weight included, whatever their points. A read of a public tensor counts in `d_1(t)` alone. The instance's degree is the largest `d_1(t) + 2s(t)` over its terms: each masked read counts 3 in place of 1. A fold ('Virtual tensors') has degree 2.
  - **Why the factor count of the RFC's form.** RFC-0010 sometimes gives a degree per variable, which can be lower: `Recomposition`'s is 2. An implementation may also split a term differently: the CPU prover writes RFC-0053's `Hplus` as `nt + 1` terms of four factors. The factor count of the defining RFC's form is decidable from the schedule alone, and both parties compute it alike. It bounds every round's degree, because a masked read adds at most 2 in any variable. It overcounts when a round binds a masked read's lowest variable or a variable outside its mask's columns, or when a public selector vanishes. Both implementations still use it: `d` is the round messages' length, not the polynomial's exact degree.
  - **The table.** It lists each relation with every operand that can be committed or virtual being so. A public operand is counted by the rule, term by term, as the rows marked "public" show. "Stacked" means with stacking variables, layers or heads, which the relation sums with a public `eq` weight (RFC-0010 'Batching within a stage').

    | Relation | `d_1` | `s` | `d` |
    |---|---|---|---|
    | `MatMulWeight`, `MatMulAct` (RFC-0010) | 2, or 3 stacked | 2 | 6, or 7 stacked |
    | `Mul` | 3 | 2 | 7 |
    | `Mul`, one operand public | 3 | 1 | 5 |
    | `SumSquares` | 3 | 2 | 7 |
    | `Rotate` (RFC-0037), `Selection` (RFC-0072) | 2 | 1 | 4 |
    | `ScaleChannels` (RFC-0072) | 3 | 1 | 5 |
    | `BitFn` (RFC-0066): `Xor`, `And`, `Ch` | 3 | 2 | 7 |
    | `Xor` or `And`, one input public; `Ch(e, f, g)` with `e`, or any two inputs, public | 3 | 1 | 5 |
    | `Ch(e, f, g)` with `f` or `g` alone public: `e·g` or `e·f` keeps two reads | 3 | 2 | 7 |
    | `BitFn`: `Xor3`, `Maj` | 4 | 3 | 10 |
    | `Xor3` or `Maj`, one input public | 4 | 2 | 8 |
    | `Xor3` or `Maj`, two inputs public | 4 | 1 | 6 |
    | `Equality` of a committed output (RFC-0037): `ConstMul` by one constant in every layer, `ConstAdd`, `Add`, `Sub` | 2 | 1 | 4 |
    | `Equality` of a `ConstMul` by a constant per channel or per layer, a public factor | 3 | 1 | 5 |
    | `Equality` of a `Mul` | 3 | 2 | 7 |
    | `Equality` of a `Mul`, one input public | 3 | 1 | 5 |
    | `Equality` of a `Mul`, both inputs public: `eq·Y'` alone | 2 | 1 | 4 |
    | `Booleanity` | 3 | 2 | 7 |
    | `OneHotVocabulary`, `OneHotIndex`, `InputPadding`, `BitPadding` (RFC-0035, RFC-0072) | 3 | 1 | 5 |
    | `AssertEqual`, one side public or none: `eq·ψ·a` | 3 | 1 | 5 |
    | `AssertEqual`, both sides public | 3 | 0 | 3 |
    | `OneHotCount` | 2 | 1 | 4 |
    | `Recomposition` (RFC-0072), `x` public or not: `eq·W_b·b̂` | 3 | 1 | 5 |
    | A root's coefficient product (RFC-0050) | 4 | 3 | 10 |
    | A root's padding checks (RFC-0050); `SoftmaxRow`'s key, forest and padding checks (RFC-0030) | 3 | 1 | 5 |
    | `SoftmaxRow`'s reciprocal (RFC-0030) | 3 | 2 | 7 |
    | Provisional: the greedy certificate's selected logits (RFC-0053), `L·Hplus` with its coefficients | 3 | 2 | 7 |
    | Provisional: a request's tokens and greedy keys (RFC-0053) | 2 | 1 | 4 |
    | Provisional: an observed request's logits (RFC-0053) | 1 | 1 | 3 |
    | A fold | — | — | 2 |

  - **RFC-0053's rows are provisional.** RFC-0053 holds for `tessera-protocol/1` only: it says `tessera-protocol/2` is not supported by its contract. Its final message sends `H` at Boolean time coordinates, where `H'`'s mask vanishes (`docs/zk/simulator.md` R8), an open gap of its `tessera-protocol/2` form. The rows count its relations as written, `Hplus` one read. They change with that form.
  - **The invariant.** No row of the table exceeds 10, so no reduction stage built from these relations does. A relation that would exceed 10 needs a row and a new bound first. The compiler and the verifier each check `d ≤ 10` when they derive a stage's degree, and stop with `ERR-INTERNAL` otherwise; no input reaches that check today. The sumcheck engines refuse a degree of 11 themselves ('Testing Strategy').
  - **A round's true degree** can be lower than the table's, as above. The honest prover sends the round polynomial at `0, 2, …, d` all the same.
  - **Every Llama schedule.** Stage 1 holds RFC-0050's root coefficient products, `Σ R_i·R_h·U_k` times `eq`: `d = 10`. Its other instances reach 7 at most. The later reduction stages hold matmuls, elementwise products and sums of squares: `d = 7`. This holds at 405B, two reduction stages of 42 and 35 rounds at 16K positions, and in the conformance claim lists, two or three.
  - **Not only a count.** At 405B's contexts, up to `2^17` positions, the final norm's products reach degree 10 in their last round whatever the masks' columns. Their per-row digit columns have at most 17 variables, below `log2 k = 22`, so each mask holds every row variable. A batch past `2^22` positions would take the last row variable out of the columns. `Xor3` and `Maj` reach 10 on SHA-256's schedules (RFC-0066), whose tensors' variables are all columns.
  - **The bound** for `tessera-protocol/2`'s reduction stages is therefore 10, not 7: RFC-0014's `d ≤ 10` and step 1 of 'Masked sumcheck'. Tree levels stay at 7 (RFC-0032), and `tessera-protocol/1` at 7.

Weights follow the same rule with the mask row stored as blinding rows of the weight matrix ([RFC-0012](RFC-0012-weight-commitment.md#budget-state-machine)); each proof consumes one `Fp2` mask row, two blinding rows, per weight matrix. `ChannelParams` takes one per claim, with `Z` over the claim's type tensor, since its claims' points have Boolean coordinates ([RFC-0065](RFC-0065-masked-channel-params.md)).

### Hiding openings

- **Columns.** Every committed row carries a vanishing mask of `P` random coefficients (RFC-0011 encoding). Any set of at most `P` codeword positions outside `H` is uniformly distributed and independent of the message. Advice commitments are opened once (`τ_A ≤ P_A`); weight matrices are opened up to `P_W` distinct positions per epoch.
- **Combined rows.** Each revealed combination, each claim group `g` and the proximity row, includes a fresh `Fp2` blinding row `Q_g = Q_g^{(0)} + X·Q_g^{(1)}`, two committed rows, each with its own vanishing mask. `Q_g` enters with coefficient 1, its two rows with `1` and `X`. The prover reveals `u'_g = Σ_row c_g(row) A_row + Q_g`. `u'_g` is uniform on `Fp2^k` because `Q_g` is, and is used exactly once.
- **Blinding claims.** For a claim group, the prover sends the blinding claim `q_g = ⟨Q_g, eq(col_g)⟩` before the claims' coefficients exist. Every `q_g` is absorbed, in group order, under `pcs/advice/blinding`, before `γ` is squeezed (RFC-0014, row 7). The group check then reads `Σ_t γ^{t+1}·v_t + q_g = ⟨u'_g, eq(col_g)⟩` ([RFC-0011](RFC-0011-advice-commitment.md#opening-protocol) steps 2 and 6).
  - **Soundness.** The column checks bind `u'_g` to the committed rows, so `⟨u'_g, eq(col_g)⟩ = Σ_t γ^{t+1}·f_t + Q̃_g`, with `f_t` the true evaluations and `Q̃_g = ⟨Q_g, eq(col_g)⟩`. The check holds exactly when `(q_g − Q̃_g) + Σ_t γ^{t+1}·(v_t − f_t) = 0`. Everything in it but `γ` is fixed before `γ` is drawn, so a false claim makes it a non-zero polynomial of degree at most `T_g`, the group's number of claims, which vanishes with probability at most `T_g/|Fp2|`.
  - **Both conditions are needed.** If `q_g` were sent after `γ`, the prover could choose it to cancel any false claim. If a claim also had the coefficient 1, `q_g` could cancel that claim's error whatever the order.
  - **Hiding.** `q_g` reveals one linear functional of the fresh row `Q_g`, which is uniform since `eq(col_g) ≠ 0`. Given `q_g`, `u'_g` is uniform subject to its group check.
- **The proximity row** carries no claim. Its blinding row enters with coefficient 1, and only the column checks see it.
- **Masks of combined masks.** The combined vanishing-mask polynomials `M_g` include `Q_g`'s mask, so they are uniform as well.
- **A weight matrix of several claims** (RFC-0065). `ChannelParams`' `e` evaluation rows share one pair `β_T`, which carries their `e` blinding claims. Each row takes its claim's mask row `R_i` as its designated block, so the rows stay triangular. Claims at one column point give equal blinding claims, a public relation.

### LogUp and product-tree roots

[RFC-0032](RFC-0032-zero-knowledge-tree-levels.md) supersedes this section's masking pair:
- **It leaked.** It put `(μ, δ)` in the lookup tree and `(−μ, δ)` in the table tree. With one `δ` in both, the verifier learns `d_a/d_t`, and with the public table, `Π_i(α − a_i)`, a fingerprint of the secret lookups.
- **Its accounting was open.** Its leaf checks would reveal `μ` and `δ`, which enter them with public coefficients.

In `tessera-protocol/2`:
- LogUp roots are not sent: a masked sumcheck over the two forests' root levels proves the LogUp identity.
- Product-tree roots are public: zero by the max certificate, and one for padding trees.
- Every level below a root is a masked extension that spares its pair bit, in a forest of two trees at least.

### Malicious verifier

- The seed is coin-tossed: the verifier commits to its share before seeing the prover's, so the seed is uniform even if the verifier is malicious; every sumcheck challenge derives from it by Fiat–Shamir. Masked-sumcheck simulation needs only that challenges are uniform.
- Live column queries are the verifier's free choice; hiding of columns holds for every choice within the budget.
- Selective aborts cannot produce two transcripts over one `Round1`: the prover answers at most one seed per `Round1` (INV-PROTO-1, [RFC-0014](RFC-0014-interaction-and-soundness.md#invariants)).

### Simulator argument

Before any v0.4 implementation work merges, [`docs/zk/simulator.md`](../zk/simulator.md) must give a complete simulator for one proof and its proof of statistical indistinguishability, reviewed by a second maintainer. Its draft shows that the protocol as specified is not yet zero-knowledge, and lists the requirements its theorems need (its §9). Outline:

1. The simulator plays the coin toss, programming the random oracle at `H("coin-toss", ·)` to a uniformly chosen seed.
2. It commits to uniformly random rows in place of advice; roots are distributed identically because opened positions are uniform in both worlds and unopened leaves are random-oracle outputs of high-entropy inputs.
3. For the LogUp identity sumcheck and each tree level, in order, it samples the masked final evaluations and mask values uniformly subject to each one's final equation, then simulates its rounds ([RFC-0032](RFC-0032-zero-knowledge-tree-levels.md) 'For the simulator argument'). For each stage in order, it does the same.
4. It samples each blinding claim `q_g` uniformly before `γ`, then each combined row `u'_g` uniformly subject to its group check, and each weight matrix's `u_T` likewise after `θ_T`. A matrix's several blinding claims are sampled uniformly subject to their public relations: equal at a shared column point (RFC-0065).
5. It answers column queries with uniform symbols consistent with the sampled combined rows (solvable because each queried column adds one linear constraint per row on independent mask coefficients).
6. Indistinguishability error is bounded by the sum of degenerate events: `Z(p) = 0`, linearly dependent `eq` vectors, and the masked-sumcheck failure terms, each at most a small polynomial over `|Fp2|`.

### Cost

| Item | 405B, 16K tokens |
|---|---|
| Mask rows `R_V` (about 2,000 stacked tensors, two rows each, a projection; the tree levels' 2,621 masks and the `SoftmaxRow` leaf's are packed in RFC-0070's masks' block, 780 `Fp2` rows with the masking polynomials) | about 1.6×10^10 elements, 0.6% of advice |
| Stage masking polynomials | < 2×10^4 elements |
| Blinding rows for advice combined rows (`G + 1 ≤ 17`, two rows each; in `tessera-protocol/2`, `G + 1 ≤ 21` with RFC-0068's column masks' group, the two reduction stages' fold groups and RFC-0070's batching group) | 34 rows; 42 |
| The masks' batching ([RFC-0070](RFC-0070-mask-batching.md)) | the pad `S`: 780 `Fp2` rows, 1,560 committed, 0.164% of the advice's non-zero rows; one sumcheck of 32 rounds of degree 2; one opening group. It opens the 46 masked sumchecks' `G(r)`, the 2,621 tree-level mask claims and the `SoftmaxRow` leaf mask's |
| Weight budget per proof | 6 blinding rows per weight matrix (two each for evaluation, proximity and mask), and `2e + 4` for `ChannelParams`' `e` claims, 18 at `e = 7`: one evaluation pair for the matrix and a mask row per claim ([RFC-0065](RFC-0065-masked-channel-params.md)) |
| Extra sumcheck work | from the round that binds the last of `Z`'s variables: the last round, or a LogUp leaf level's round before it |

The weight budget per proof is six rows, and the linking proofs take four at each end of an epoch, so `B_W = 4096` gives `(4096 − 8)/6 = 681` proofs per epoch, and `ChannelParams`' `B_CP` gives as many ([RFC-0065](RFC-0065-masked-channel-params.md)), matching the column budget of about 674 in epoch 0 and 673 after it ([RFC-0012](RFC-0012-weight-commitment.md#budget-state-machine), [RFC-0064](RFC-0064-weight-linking-proof.md#the-budget)).

### Invariants

- **INV-ZK-1.** Every value sent to the verifier is public, uniformly masked, or determined by previously sent values and public data.
- **INV-ZK-2.** Every mask row, blinding row and masking polynomial is used in exactly one revealed combination or evaluation set, within its degrees of freedom. A matrix's shared pair `β_T` (RFC-0065) is one evaluation set: it enters all of the matrix's evaluation rows, each with its own designated mask row, the rank reading of R5 (`docs/zk/simulator.md`). The masks' pad `S` ([RFC-0070](RFC-0070-mask-batching.md)) is the masks' batching's: the masking polynomials, the tree levels' masks and the `SoftmaxRow` leaf mask enter that batching only through `M + ρ·S`, which `S` covers (Lemma 4a), and every later use is under a block of its own.
- **INV-ZK-3.** Masking randomness is drawn from the per-proof secret seed's PRG (advice) or from the epoch's mask seed (weights), never from the transcript.

## Alternatives Considered

1. **ZK-STARK wrapper over the GKR verifier** (source §10 alternative). Considered: gives compact, transferable proofs as a side effect. Rejected for v1: proving the GKR verifier inside a STARK requires recursion (NG-3) and enlarges the trusted base with a second proof system, while masking keeps the trusted base to one protocol.
2. **Commit every round message and prove consistency in zero knowledge.** Rejected: one commitment per round message across thousands of rounds, plus a second argument to check them.
3. **Per-layer masked weight evaluations.** Rejected: 126 mask rows per weight type per proof, which would cut the epoch length by about 40×.
4. **Doubling every committed tensor with a random half (extra variable).** Rejected: doubles commitment cost.
5. **Honest-verifier ZK only.** Rejected: the verifier is an adversarial state (source §2).

## Drawbacks

- The masked extension raises round degrees, which complicates the sumcheck engine with a last-round special case.
- Correctness of zero knowledge rests on a written argument and statistical tests, not on a mechanized proof.
- Every revealed evaluation has a randomness cost; adding a new revealed claim requires adding its mask.

## Migration / Rollout

v0.3 ships a sound, non-hiding protocol. v0.4 adds the four mechanisms behind the protocol version bump that introduces them; the non-hiding mode remains available only under the `test-params` feature for debugging.

## Testing Strategy

- Masked sumcheck: completeness for degrees 1–10; the simulator's round messages and real round messages pass a two-sample distribution test over 10^5 runs on a toy instance.
- Stage degrees ([#565](https://github.com/AbdelStark/tessera/issues/565)):
  - **The vector.** `spec/vectors/schedule/degrees.json` gives, per instance, its stage and index, its relation with what fixes its row (native kind, `BoolFn`, broadcast, stacking, which operands are public), `d_1`, `s` and `d`, and each stage's `d`.
    - **Its schedules** are the composed schedules `tessera-protocol/2` proves: `derive_composed_schedule` with `model_natives` for the request, not `derive_schedule`'s graph-only schedule, which lacks the natives and records 7 for stage 1.
    - **Its cases** are each conformance claim list of `spec/vectors/claims/`, at `schedule.json`'s lengths 1, 4, 5 and `max_context` laid out as `tessera-protocol/2` lays them out, for each of the three requests: none, a prefill and a generation. They add `tessera_compiler::anchor`'s schedules from `derive_graph_schedule`: SHA-256 over a block of public bytes and over committed bytes, and AES-128-CTR. SHA-256 holds `Xor3` and `Maj` and public operands of `Ch`, `Maj`, `Xor`, `AssertEqual`, `Recomposition` and `Mul`.
    - **Its makers.** `cargo xtask gen-vectors` writes it from the compiler's schedules and this RFC's table. It is a differential vector, as `schedule.json` is. The verifier, `tv-core`, derives each schedule and each instance's degree with code of its own and compares every field. `gen-vectors --check` fails when the file differs. No independent oracle computes schedules.
    - **What it records:** 10 for stage 1 of every Llama schedule, 7 for the later stages, and 10 for SHA-256's stage of `Xor3` and `Maj`.
  - in a stage of `d = 10`, a round message of 7 values is `Reject(Sumcheck { stage, round })`;
  - **the bound, at the sumcheck level**: `tessera-sumcheck`'s `prove_batch` and `tv-core`'s sumcheck checks each prove and verify a batch of degree 10, and each refuses a batch of degree 11 with its degree error.
- Masked extensions: `V' = Ṽ` on the hypercube (exhaustive for small `n`); revealed `V'(p)` values over 10^5 runs of a fixed witness are uniform (chi-square on low bits) and uncorrelated with the witness (two witnesses give indistinguishable distributions).
- Committed tensors' columns ([#554](https://github.com/AbdelStark/tessera/issues/554)):
  - a claim on a stacked committed kind of `n_P < log2 k` with `λ ≥ 1`, and one on a tile smaller than a row, open in their data claims' groups with `R_V`'s row at `eq(p_col ‖ 0)`, and the group checks hold;
  - a prover that puts the mask term at `p_col ‖ ρ` instead fails its group check, `Reject(Evaluation { commitment: Advice })`;
  - as the negative control, two values of such a kind at one column sub-point and two layer points, or of such a tile at two tiles, tell two witnesses apart, and R10's check refuses the schedule that reveals them;
  - a committed mask row's entries past `2^c` are mask-stream values, and two witnesses' proximity rows are indistinguishable on a layout where no committed tensor fills a row, as 405B's at 16 positions;
  - a stacked piece's mask term enters its group once, with `Z` at `(r ‖ ρ)` (RFC-0011 step 2): a prover that adds it per layer image instead, where `Z` vanishes at the Boolean layer, is rejected at the opening, `Reject(Evaluation { commitment: Advice })` or `Reject(Proximity)`;
  - the prover refuses, with `ERR-INTERNAL`, a claim list whose composed schedule fails R10's check at the length and request it is asked to prove.
- Hiding openings: opened columns and blinding-row combinations for two different witnesses are indistinguishable by the distinguisher suite of [07 §7](../spec/07-testing-strategy.md#7-zero-knowledge-tests).
- Virtual tensors' folds ([#538](https://github.com/AbdelStark/tessera/issues/538)):
  - completeness with `n_i` below, equal to and above `log2 k`, the last giving a factor below 1; with `c < log2 k − 1` and `c = log2 k − 1`; and in a stage whose relations all have fewer than `log2 k` variables, so that `n = log2 k`;
  - a prover that omits the factor `2^{log2 k − n_i}`, with `n_i ≠ log2 k`, is rejected at the final check, `Reject(Sumcheck { round: n })`;
  - a prover whose `u` is not `R_V`'s row's value is rejected at the opening, `Reject(Evaluation { commitment: Advice })`;
  - as the negative control of the whole-row rule, a mask row with zeros past `2^c` and a same-stage reveal of `V'` at `q_col = r''_low` tell two witnesses apart.
- Blinding claims, adversarially. Each of these provers is rejected with `Reject(Evaluation)`:
  - one that compensates a false claim through `q_g`, sent after `γ`;
  - one that sends `q_g` before `γ` but cancels the error of a claim it gives the coefficient 1;
  - one that splits a weight claim as `(v_T + Δ, q_T − Δ)`.
  Over the test field, a false claim with an honest `q_g` is accepted at a rate within `T_g/|F|`.
- LogUp roots: not sent; the identity sumcheck's final evaluations and every tree level's evaluations are independent of the key multiset ([RFC-0032](RFC-0032-zero-knowledge-tree-levels.md)).
- End-to-end simulator test: simulated and real transcripts of the tiny model are indistinguishable at significance 0.001 over 10^4 runs.
- Budget: each proof consumes exactly six weight blinding rows per matrix and fails cleanly when they run out.
- **Masks over `Fp2`.** Each of these distinguishers tells two witnesses apart with probability close to 1 when its mask is drawn from `Fp`, and is at chance with the masks of this RFC:
  - one that reads `u'_g` off the line of an `Fp` blinding row;
  - one that reads a round-message coefficient off the line through `ρ_k`.

## Open Questions

- **OQ-0013-1.** Can mask rows for small tensors (tree levels below 2^12 entries) be packed into shared rows without breaking the independence condition, to cut the mask-row count? Owner: @AbdelStark. Resolve during v0.4 with the simulator argument; adopt only if the argument covers it. Answered for tree levels' masks: [RFC-0070](RFC-0070-mask-batching.md) packs them in aligned blocks of its masks' block, which its batching opens.
- **OQ-0013-2** (resolved by [RFC-0032](RFC-0032-zero-knowledge-tree-levels.md): masked tree levels, masked extensions that spare the pair bit in forests of two trees at least, and an identity sumcheck in place of revealed LogUp roots). A tree level reveals `Ṽ_{ℓ+1}(0 ‖ r'_i)` and `Ṽ_{ℓ+1}(1 ‖ r'_i)` ([RFC-0010](RFC-0010-proof-schedule.md#tree-levels) 'Tree levels'), and fraction-tree levels reveal their numerators and denominators at the same points. These points have a Boolean coordinate, where `Z` vanishes, so masked extensions do not hide them. At the level where a single tree joins, they are exactly the two entries of its level 1: for a SoftmaxRow tree, the products of each half of the row's keys. A forest reveals its trees' level-1 entries combined at the level's random point. The round messages of tree levels are not masked either: RFC-0014's schedule has no `zk-sum` or `zk-rho` for them. How are tree levels made zero-knowledge, or why is what they reveal acceptable? Owner: @AbdelStark. Resolve before the label table freezes at `tessera-protocol/1` (v0.3), since the answer may add labels.

## References

- Source §2 (malicious verifier), §6 principle 11, §7 ("Zero knowledge"), §10.
- Chiesa, Forbes, Spooner, "A zero knowledge sumcheck and its applications", 2017.
- Xie, Zhang, Zhang, Papamanthou, Song, "Libra", CRYPTO 2019.
- Ames, Hazay, Ishai, Venkitasubramaniam, "Ligero", CCS 2017 (hiding via randomized encodings).
