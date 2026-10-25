# RFC-0070: Opening Every Mask Claim in One Batching Sumcheck in `tessera-protocol/2`

- Status: Accepted
- Authors: @AbdelStark
- Created: 2026-10-25
- Target milestone: v0.4
- Amends, in `tessera-protocol/2` only:
  - [RFC-0013](RFC-0013-zero-knowledge.md#masked-sumcheck) 'Masked sumcheck' step 4, 'Cost' and INV-ZK-2: no masking polynomial's value is opened alone;
  - [RFC-0010](RFC-0010-proof-schedule.md#sumcheck-messages) 'Sumcheck messages': a masked sumcheck's final evaluations end with its `G(r)`, and a tree level's with its mask values and then `G(r)`;
  - [RFC-0032](RFC-0032-zero-knowledge-tree-levels.md#masked-tree-levels) 'Masked tree levels', 'Masked extensions that spare the pair bit', 'Labels' and 'Cost': `G_ℓ(r)` is sent with the level's evaluations, the mask values' order, and the mask rows `R_t` as blocks of the masks' block;
  - [RFC-0068](RFC-0068-leaf-stage-zero-knowledge.md#the-layer-sumcheck) 'The layer sumcheck' steps 5 and 6: `G(ρ')` likewise;
  - [RFC-0011](RFC-0011-advice-commitment.md#opening-protocol) 'Opening protocol' step 1 and 'Sizes': one more group;
  - [RFC-0014](RFC-0014-interaction-and-soundness.md#transcript-schedule) 'Transcript schedule' and 'Soundness calculator': the batching's labels, after the last stage, and its sizes;
  - [RFC-0036](RFC-0036-protocol-messages.md#versions-and-envelopes) 'Versions and envelopes': what version 2's `Round2Response` carries;
  - [RFC-0019](RFC-0019-verifier.md#check-order) 'Check order', and [04 §2](../spec/04-error-model.md#2-verdicts): the batching's checks and `Reject(MaskBatch)`;
  - INV-SCH-3, in [RFC-0010](RFC-0010-proof-schedule.md#invariants) and [RFC-0037](RFC-0037-stage-schedule.md#invariants): the mask claims reach the batching, and only its claim `y` the opening pool;
  - the opening-group figures of RFC-0011, RFC-0013, RFC-0019, RFC-0057 and [08 §5](../spec/08-performance-budget.md);
  - [`docs/zk/simulator.md`](../zk/simulator.md) §1, §4.2, §4.3a, §4.4, §7, §8, §9 and §10: Lemma 4a, R5, R10 and R11.
- Origin: [#489](https://github.com/AbdelStark/tessera/issues/489); research: [IDEA-0040](../../research/ideas/IDEA-0040-batched-mask-opening.md), [EXP-0027](../../research/experiments/EXP-0027-batched-mask-opening/README.md)

## Summary

`tessera-protocol/2` reveals two kinds of claim on the advice that no opening covers (`docs/zk/simulator.md` R11):
- each masked sumcheck's `G(r) = a_0 + Σ_j g_j(r_j)`, whose column functional, `(r_j^m)` over the coefficients, is not an `eq` vector, so RFC-0011 cannot open it;
- each tree level's mask value `R̃_t(r'_t)`, at its own level's point, so that claims of different levels never share a column sub-point.

One RFC-0011 group per claim would cost a combined row of 64 MiB each. The compiler's 405B schedule has 2,668 such claims, the `SoftmaxRow` forest's leaf mask among them. Even sharing a group among the claims at one column point, EXP-0027's 2,667 would take 612 groups: 38.25 GiB.

This RFC opens them all with one masked sumcheck. The masking polynomials' coefficients and the tree levels' and `SoftmaxRow` leaf's masks form one block `M`. Every claim is a linear functional `⟨c_i, M⟩`, and a challenge `α` combines them into `⟨C, M⟩`. A pad `S`, laid out like `M` and committed in Round 1, masks the sumcheck that reduces `⟨C, M⟩` to one value of `M + ρ·S` at one point, which one more opening group opens. At 405B this adds 65.85 MiB to the proof, `S`'s 1,560 committed rows and 32 rounds of degree 2. Zero knowledge needs `S` to cover `M`'s layout, not a rank condition (Lemma 4a).

## Motivation

- **The claims.** Before this RFC, RFC-0013 step 4 said `G(r)` "is opened from the advice commitment as an ordinary claim", and RFC-0032 that the mask value `R̃_t` "is a claim on the advice". Neither can be opened as RFC-0011 opens claims:
  - **`G(r)`.** RFC-0011 groups claims by column sub-point and opens `eq` functionals. `G(r)`'s functional puts `r_j^m` on the coefficient of `X^m` in round `j`.
  - **`R̃_t(r'_t)`.** Each level's final point is its own, so the claims of different levels would take groups of their own.
- **The count at 405B**, from the compiler's schedule ([EXP-0027](../../research/experiments/EXP-0027-batched-mask-opening/README.md) (d), `servable` with two classes, 16K tokens and context):
  - **Masked sumchecks: 46.** The two reduction stages (42 and 35 rounds), RFC-0068's layer sumcheck (22), the LogUp identity (2), and 42 tree levels: 1,004 rounds, at most 8,006 coefficients, recounted from EXP-0027's rounds under this RFC's one degree per sumcheck: 7 for the stages (RFC-0013's bound) and the tree levels, 4 for the layer and identity sumchecks. EXP-0027's per-round degrees gave 4,562; both fit the same block.
  - **Tree-level mask claims: 2,621.** Two per level of each fraction forest, its root level's included, and one per level of the `SoftmaxRow` forest below its root. 697 are full width, `2^{22}` entries.
  - **The leaf mask.** The `SoftmaxRow` forest's leaf mask `R_h` is one more claim, at a point no committed claim shares.
  - **One group per claim**, on EXP-0027's 2,667, costs 2,667 groups, 166.7 GiB. **One group per column point** costs 612: the tree levels' claims fall on 566 column points, and each `G(r)` takes a group of its own. That is 38.25 GiB. Either is far beyond 08 §5's 4 GiB gate.
- **The issue's projection** was about 200 groups. The tree levels' claims alone are 2,621.

## Goals

- Every `G(r)` and every tree-level mask claim opened, within the proof-size gate.
- Zero knowledge for the opening, with a written lemma (`docs/zk/simulator.md` R11).
- A soundness error the calculator counts with the sizes it already reads.
- No change to `tessera-protocol/1`.

## Non-Goals

- **The other masks.** RFC-0068's column masks and the reduction stages' fold masks keep their own groups. They could join the batching only if `S` covered their whole rows ('Alternatives Considered' 3).
- **A smaller pad.** EXP-0027's `S` on the rows whose index has Hamming weight at most 2 never leaked on protocol-like claims, which would cut `S` to 56 `Fp2` rows, 112 committed. It needs a rank lemma of its own ('Open Questions').

## Proposed Design

Everything below is `tessera-protocol/2`. Every value is in `Fp2` unless it says otherwise, as RFC-0013 requires of every mask.

### The claims

The **mask claims** are:
- each masked sumcheck's `G(r)` at its final point `r`: the LogUp identity's (RFC-0032), each tree level's, RFC-0068's layer sumcheck's and each reduction stage's. The prover sends `G(r)` as the sumcheck's last final evaluation ('Labels'). The sumcheck's final check reads it as before, `p_n(r_n) = f(r) + ρ·G(r)`, so the sumcheck keeps its own rejection.
- each tree level's mask values `R̃_t(p_t)`, the ones `tree/{ℓ}/evals` carries after the instances' evaluations (RFC-0032 'Masked extensions that spare the pair bit'). Here `p_t` is the first `c_t = min(n_i, log2 k)` coordinates of the instance's own point `r'_i`, the level's last `n_i` challenges (RFC-0010 'Tree levels' step 3). They are the masks of every fraction forest's levels `t = 0, …, h − 1` and of every product forest's levels `t = 1, …, h − 1`.

- the `SoftmaxRow` forest's leaf mask value `R̃_h(p_h)`. Its leaf level `h` takes a mask row `R_h` of its own (RFC-0032 'Leaf checks'), which RFC-0030's reduction folds inside its instance's summand, in a reduction stage. `R_h` spans the instance's `c_h = min(n_h, log2 k)` column variables, all `k` at 405B. The stage sends `R̃_h(p_h)` last among that instance's final evaluations, with `p_h` the first `c_h` coordinates of the instance's point.

**Their order** is the transcript's, and numbers them `i = 0, …, N − 1`:
1. the LogUp identity's `G(r)`;
2. then, for each tree level `ℓ` in order, its mask values, then its `G_ℓ(r)`. A level's mask values come in the order of its instances (RFC-0010 'Tree levels'), each fraction forest's numerators' before its denominators';
3. then the layer sumcheck's `G(ρ')`;
4. then, for each reduction stage in order, the leaf mask value `R̃_h(p_h)` when it holds the `SoftmaxRow` leaf instance, then its `G_k(r_k)`.

`N = 2,668` at 405B: EXP-0027's 2,667 and the leaf mask's one. `tessera-protocol/2` always has the LogUp identity, so `N ≥ 1`.

### The masks' block `M`

`M` is a logical `Fp2` tensor of `R_M` rows of `k` entries. Its cube has `m = log2 k + ⌈log2 R_M⌉` variables, the column variables lowest. It holds every masking polynomial's coefficients and the masks `R_t` and `R_h` whose values are mask claims, in blocks of power-of-two size:
- **The coefficient block.** The masking polynomials, in the claims' order. A sumcheck of `n` rounds of degree `d` takes `1 + n·(d + 1)` consecutive slots: `a_0`, then round `j`'s `g_{j,0}, …, g_{j,d}` for `j = 1, …, n`, constant first. `d` is the degree its round messages hold, one for every round (RFC-0010 'Sumcheck messages'), so that every `g_j` has degree `d` (R4): 7 at a tree level in `tessera-protocol/2`, whose last round RFC-0032 raises; 4 for the LogUp identity (RFC-0032 'Soundness'); RFC-0068's `d` for the layer sumcheck; and each reduction stage's own. The block is padded with public zeros to `2^{⌈log2 Σ_s (1 + n_s·(d_s + 1))⌉}` slots, over the sumchecks `s`: 8,192 at 405B, for at most 8,006 coefficients.
- **A tree-level block** per mask `R_t` or `R_h` whose value is a mask claim, of `2^{c_t}` entries: the mask's values, in the order of its `c_t` column variables (RFC-0032).
- **The packing.** The blocks are placed in order of decreasing size, ties in the claims' order with the coefficient block first, each at the next free position of the cube. Every size is a power of two, so every block is aligned to its size. A block of at most `k` entries lies within one row; the coefficient block, if larger, takes whole rows. The tail of the last row is public zeros, and so are the cube's rows `R_M` to `2^{⌈log2 R_M⌉} − 1`, which are not committed. Packing tree levels' masks is OQ-0013-1's admissible case (`docs/zk/simulator.md` §7): each block's functionals lie on its own block.
- **Nothing else.** `M` holds no other mask. A position of the cube that holds a mask value is **non-public**, and every other position is a public zero. A mask with no claim in `C` could sit in `M` as long as `S` covers it (Lemma 4a), but nothing calls for one.

A block's place is a function of the claim list and the batch descriptor, so `M`'s layout is public before any challenge (INV-PCS-1). At 405B `R_M = 780`, so `m = 22 + 10 = 32`. Each `Fp2` row of `M` is two committed `Fp` rows, `M_w = M_w^{(0)} + X·M_w^{(1)}`, and these are the rows RFC-0013 and RFC-0032 commit for these masks.

**In the advice.** `M`'s rows and then `S`'s are advice rows, each `Fp2` row as its two `Fp` rows, the real part first, in their rows' order. Where that sequence sits among the advice's rows is `tessera-protocol/2`'s advice layout's, which places every mask row (RFC-0050 leaves it to [#154](https://github.com/AbdelStark/tessera/issues/154)). The batching needs only each of these rows' index, which the layout gives both parties.

### The pad `S`

- **Its layout is `M`'s.** `S` has `R_M` `Fp2` rows, two committed `Fp` rows each. It is uniform over `Fp2` at every non-public position of `M`, claimed or not, and zero at every public one.
- **Its randomness.** It is drawn from the per-proof PRG (INV-ZK-3) and committed in Round 1 with the rest of the advice. Its support is `M`'s layout, fixed before any challenge, and never `C`'s, which depends on challenges.
- **Where it is used.** `S`, like `M` after the batching, enters nothing the verifier sees except through `z = M + ρ·S`, or under a block of fresh randomness of its own: `y`'s group's blinding pair, the proximity row's, and each row's encoding mask in the columns ('Zero knowledge').

### The batching sumcheck

After the last reduction stage's messages, and before the advice groups' blinding claims (RFC-0014 row 7):

1. **The combination.** The verifier squeezes `α` under `masks/alpha`. Claim `i` is `v_i = ⟨c_i, M⟩` for a public, sparse `c_i` over `M`'s cube:
   - for a masking polynomial's `G(r)`: `1` at `a_0`'s slot and `r_j^e` at `g_{j,e}`'s, for `e ≤ d`, and zero elsewhere;
   - for a mask value `R̃_t(p_t)`: `eq(p_t, x)` at position `x` of `R_t`'s block, and zero elsewhere.

   With `C = Σ_i α^i·c_i`, the claims combine to `v = Σ_i α^i·v_i = ⟨C, M⟩`.
2. **The masked sum.** The prover sends `Γ_S = ⟨C, S⟩` under `masks/zk-sum`, and the verifier squeezes `ρ` under `masks/zk-rho`.
3. **The rounds.** A sumcheck of degree 2 over the `m` variables, the lowest first, proves `Σ_x C̃(x)·z̃(x) = v + ρ·Γ_S`, with `z = M + ρ·S`. Round `j`, for `j = 1, …, m`, binds variable `j`: its polynomial goes under `masks/round/{j − 1}` and its challenge `r*_j` under `masks/r/{j − 1}`, in RFC-0010's format, which numbers a stage's rounds from 0 ('Sumcheck messages').
4. **The end.** The prover sends `y = z̃(r*)` under `masks/evals`, as a TCE `Vec<Fp2>` of one value, as RFC-0010 sends a stage's evaluations. The verifier computes `C̃(r*)` itself and checks the last round against `C̃(r*)·y`.
5. **`y` is one advice claim.** It is a claim at column sub-point `r*_{1..log2 k}` on the pieces `M^{(0)}, M^{(1)}, S^{(0)}, S^{(1)}`, with coefficients `(1, X, ρ, ρX)` times `eq(r*_row, w)` on row `w`. It joins the advice pool (INV-SCH-3) and forms one more RFC-0011 group, the last in group order: its column sub-point is the batching's own.
6. **What is never sent.** `M̃(r*)` and `S̃(r*)` are never sent apart. No `G(r)` or `R̃_t(p_t)` reaches the opening pool: INV-SCH-3 sends the mask claims to the batching, whose `y` alone joins the pool.

**The verifier's `C̃(r*)`.** `C̃(r*) = Σ_i α^i·⟨c_i, eq(r*, ·)⟩`, from public data:
- a tree-level block of `2^c` entries in row `w_t`, the `j_t`-th block of its size in that row, gives `α^i·eq(p_t, r*_{1..c})·eq(r*_{c+1..log2 k}, j_t)·eq(r*_row, w_t)`, in `O(m)`;
- the coefficient block's claims read `eq(r*, ·)` at its slots, `2^{13}` values at 405B computed once, with `eq` over the row bits of the rows it takes.

At 405B that is about `9.2·10^4` terms, each a few multiplications.

**Rejections.**
- The batching's messages are checked as RFC-0010 checks a stage's, and stop at the first failure: `Reject(MaskBatch { round })`, its `round` counted from 0 as the labels are ([04 §2](../spec/04-error-model.md#2-verdicts)).
  - **A round.** A missing round, or one of another length than 2 values, `p_j(0)` and `p_j(2)`, fails at that round, `j − 1`.
  - **The end.** A further round after the `m` rounds, evaluations other than one value, or a false final check fails at round `m`. RFC-0010's format derives each round's value at 1 from the running claim, so a false claim, a wrong `Γ_S` or an altered coefficient fails there.
- `y`'s group fails as every group does: `Reject(Evaluation { commitment: Advice })`.
- A batching message that does not decode, `Γ_S` among them, is `Reject(Malformed)` with the message's artifact ([RFC-0043](RFC-0043-rejection-payloads.md#malformed)).
- The batching runs after the stages' checks and before the opening groups' ([RFC-0019](RFC-0019-verifier.md#check-order) 'Check order').

### Labels

`Round2Response` (schema version 2) carries the batching after the stages: `zk_sum`, `Γ_S`, one `Fp2`; `rounds`, `m` rounds of 2 values; and `evals`, `[y]` ([RFC-0036](RFC-0036-protocol-messages.md#versions-and-envelopes)).

`tessera-protocol/2` adds, after row 6 and before row 7 of RFC-0014's schedule:

| Operation | Label |
|---|---|
| squeeze `α` | `masks/alpha` |
| absorb `Γ_S`, squeeze `ρ` | `masks/zk-sum`, `masks/zk-rho` |
| per round `j = 1, …, m`: absorb the round polynomial, squeeze `r*_j` | `masks/round/{j − 1}`, `masks/r/{j − 1}` |
| absorb `[y]` | `masks/evals` |

Each masked sumcheck's evaluations label carries, last, its `G(r)`:
- `logup/identity/evals`;
- `tree/{ℓ}/evals`, after the level's mask values, which follow its instances' evaluations;
- `stage/0/layer/evals`;
- `stage/{k}/evals`.

So every `v_i` is absorbed before `α`, and `Γ_S` before `ρ`. RFC-0010's length check counts `G(r)` among the final evaluations, and a tree level's mask values too ('Sumcheck messages', as amended).

### Zero knowledge

**Lemma 4a (the masks' batching).** Let `T` be `M`'s non-public positions. Let `S` be uniform over `Fp2^T`, zero off `T`, and independent of everything the verifier saw before `Γ_S`. Then, given that earlier view and `ρ ≠ 0`, the batching's view `(Γ_S, the rounds, y)` depends on the secrets only through `v`, which the earlier view fixes.
- **Proof.**
  - `C_T ≠ 0`: claim 0 puts `α^0 = 1` on a non-public position of its own block, and the blocks are disjoint. So `Γ_S` is uniform, and given `Γ_S`, `S_T` is uniform on the hyperplane `⟨C_T, ·⟩ = Γ_S`.
  - So `z_T = M_T + ρ·S_T` is uniform on `{⟨C_T, z_T⟩ = v + ρ·Γ_S}`, a set public values fix. Off `T`, `z` is zero.
  - Every later message of the batching is a function of `z`, `C` and the challenges.
  - So a simulator samples `Γ_S` uniformly, samples `z_T` on that hyperplane, and runs the honest prover.
  - The argument holds for every `C` and every adaptive choice of the challenges.
- **No rank condition.** A dependence among the functionals the rounds reveal is shared by `M`'s side, so it is a public relation. What the lemma needs is coverage: `S` must be uniform wherever `M` holds a mask value, claimed or not.
- **The degenerate event.** `ρ = 0`, `1/q`.

**The rules it rests on.** EXP-0027 shows the first three necessary, exactly, on a toy protocol over F65537. The other two are design rules.
- **The pad is required.** Without `S`, the batching leaks on every draw. It also leaks on a protocol-like layout, through a fully pinned two-entry row, which every LogUp forest's root level has.
- **`S` covers `M`'s layout, not `C`'s support.**
  - **The leak.** Laid out by `C`'s support, `S` leaks on every draw of a layout with a block no claim reads. Laid out by `M`'s layout, unclaimed blocks included, it does not.
  - **Where `C` differs from the layout.** A challenge `r_j = 0` empties round `j`'s slots of positive degree, and a Boolean coordinate empties half a row. `M`'s layout does not depend on either.
- **`M̃(r*)` is never sent alone.**
  - **The leak.** Sent alone, it leaks on a layout with no spare direction.
  - **At 405B.** `M` has many spare directions, so it would not leak there. The rule keeps the lemma free of any rank condition.
- **`S` over `Fp2`, a design rule.** The proof as written needs `S`'s two `Fp` blocks independent, as RFC-0013 requires of every mask. EXP-0027 does not test it: its prime fields stand for `Fp2`.
- **`M` holds only claimed masks, a design rule.** Lemma 4a does not need it, since `S` covers every position of `M`. It keeps `S` no larger than the claims need.
- **After the batching.** `S_T = ρ^{-1}·(z_T − M_T)`. A later message revealing a functional of `M` alone or of `S` alone, within the batching's span, would reveal it. The later messages that read them are each covered by a block of their own:
  - `y`'s group by its blinding pair `Q_y`;
  - the proximity row by `Q_ξ`, and under the tensor generator by Lemma 5's free rows, which are activation mask rows, not `S`'s;
  - the columns by each row's encoding mask (Lemma 6).

**R10** ([#488](https://github.com/AbdelStark/tessera/issues/488)) does not count `r*` as a revealed point of any tree-level mask row: `y` reveals `z̃(r*)`, which `S` masks.

**The independent reading.** EXP-0027's adversarial reading of the lemma, of its composition with `y`'s group, the proximity row and the columns, and of the ordering, found no break. This RFC settles the two gaps it raised:
- what `S` and `M` may enter after the batching;
- how `G(r)` is carried, under `…/evals`.

It also counts the two terms the reading found missing ('Soundness').

### Soundness

A false mask claim survives the batching with probability at most `(N − 1 + 2m + 1)/q`, and its opening adds `y`'s `1/q`, which the openings row counts:
- **`α`.** Every `v_i` is absorbed before `α`. A false claim makes `E(α) = Σ_i α^i·(v_i − ⟨c_i, M⟩)` a non-zero polynomial of degree at most `N − 1`.
- **`ρ`.** `Γ_S` is absorbed before `ρ`. A false `v` makes `(v − ⟨C, M⟩) + ρ·(Γ_S − ⟨C, S⟩)` a non-zero polynomial of degree 1 in `ρ`.
- **The rounds.** `m` rounds of degree 2 give `2m/q`.
- **`y`.** It is bound by its group's opening: `1/q` for its one claim.

Further points:
- **Order.** Both orders matter. EXP-0027's controls accepted a cheater in 9,999 of 10,000 trials when `Γ_S` came after `ρ`, and in every trial when `α` came before the claims.
- **Public zeros.** `C` vanishes on `M`'s public zeros, so values a prover puts there change neither `v` nor the sum the rounds prove. They change the honest messages only.
- **The measurement.** Over F65537, a cheater built to hit every term was accepted at its exact predicted rate:
  - 366 of 10⁶ trials, against 381 ± 78 (4σ);
  - 421 of 2·10⁶ trials, against 427 ± 83 (4σ).
- **At 405B.** `(2,666 + 64 + 1)/q ≈ 2^{−116.6}`.

**The calculator** ([RFC-0014](RFC-0014-interaction-and-soundness.md#soundness-calculator)) counts the batching with the sizes it already reads:
- **`stages`.** The batching is one more stage, batching its `N` claims as instances, which bounds `α`'s term.
- **`rounds`.** Its rounds, one per variable of `M`'s cube, of degree 2.
- **`claim_groups`.** One group of one claim.
- **`advice_rows`.** `S`'s rows, with `M`'s and every other mask row of `tessera-protocol/2`. At 405B they add a few thousand rows to the 2,328,576 of `tessera-protocol/1`'s allocation, still under `2^{22}`, so `ϑ_A` stays 22.
- **`ρ`.** Its term is the one per stage that `tessera-protocol/2` already adds.

The calculator's stage term is `(2m + N)/q`, plus `ρ`'s `1/q`: one more than the batching's own bound, which is conservative.

### Cost at 405B

From EXP-0027's count, at `live-64` (`τ_A = 155`, `P_A = 160`):

| Item | Cost |
|---|---|
| One opening group | a combined row of `2^{22}` `Fp2` values, 64 MiB; its mask and blinding claim; a blinding pair, two `Fp` rows |
| `S` | 780 `Fp2` rows, 1,560 committed `Fp` rows: 0.164% of the advice's 953,580 non-zero rows, 6.54·10^9 elements; 1.93 MB of opened symbols (2.41 MB at `ni-100`) |
| The batching's messages | `(2m + 2)` `Fp2` values, 1,056 bytes at `m = 32` |
| The prover | at most 2.6·10^{10} `Fp2` multiplications for the rounds, over `C`'s 3.27·10^9 non-zero entries |
| The verifier | `C̃(r*)` in about 9.2·10^4 terms; one more group's `τ_A` inner products and coset NTT |
| In all | +65.85 MiB of proof (66.30 MiB at `ni-100`). One group per column point would have cost 38.25 GiB |

The packing also saves mask rows. Unpacked, the tree levels' masks take 2,621 `Fp2` rows, and packed with the leaf mask and the coefficient block `M` takes 780.

**The figures that count groups.**
- **Groups.** `tessera-protocol/2` has `G ≤ 20` opening groups at 405B, `G + 1 ≤ 21` combined rows, `≤ 1.41 GB`.
- **Blinding rows.** RFC-0013's cost table reserves 42 of them for the combined rows.
- **The proof.** It was about 2.87 GB with RFC-0068's group and the fold groups, and becomes about 2.94 GB: within 08 §5's gate, above its target.

## Alternatives Considered

1. **One group per claim, or per column point.**
   - **The cost.** 166.7 GiB, or 38.25 GiB, at 405B.
   - **What it would need.** It needs no new lemma, since Lemma 5 covers each group.
   - **Why rejected.** It exceeds the gate by an order of magnitude.
2. **Put `G`'s coefficients where an `eq` functional reads them.**
   - **The idea.** Write round `j`'s coefficients so that `g_j(r_j)` is an `eq` evaluation.
   - **Why it fails.** The values at points of one block still need a group per sumcheck. A transposed layout costs a free column in about `4·10^4` rows, or `w` groups (#489's design notes).
   - **Libra's mask.** It is size-optimal but recurses: its final `G(r*)` is again a claim of one point per block.
3. **Batch every mask claim, the column masks and the fold masks included.**
   - **What it would save.** Three more groups, 192 MiB.
   - **Why not now.** `S` would have to cover their whole rows, and a fold reads its row's every entry (RFC-0013 'Virtual tensors'). Left for a measurement of the trade.
4. **No pad, with a rank argument on the compiled schedule.**
   - **Why rejected.** EXP-0027's bare batching leaks on the protocol's own layout, through the root levels' pinned rows. So a rank condition would fail where it matters.
5. **`S` on `C`'s support.**
   - **Why rejected.** `C`'s support depends on challenges, and EXP-0027 shows the leak through a block no claim reads.

## Drawbacks

- **Prover cost.** `S`'s rows are committed and opened.
- **Prover work.** The batching costs at most `2.6·10^{10}` `Fp2` multiplications at 405B, a measurement for v0.5.
- **Proof size.** The proof grows by one group, about 64 MiB.
- **A rule to keep.** `M` holds only claimed masks. Another mask put there without a claim in `C` and coverage by `S` would be exposed.
- **Lemma 5 under the tensor generator.** Its argument with `S`'s rows in the proximity row is written in `docs/zk/simulator.md` §4.4, not tested. R1 makes it exact.

## Migration / Rollout

1. **This RFC.** It amends RFC-0010, RFC-0011, RFC-0013, RFC-0014, RFC-0019, RFC-0032, RFC-0036, RFC-0037, RFC-0057, RFC-0068, 04 §2, 08 §5 and `docs/zk/simulator.md`, and marks R11 met in the specification. `tessera-protocol/1` is unchanged.
2. **The verifier's `RejectReason`** gains `MaskBatch { round }`, `REJ-MASK-BATCH`, with the verdict schema. That code change is the verifier's own.
3. **Implementation:**
   - [#148](https://github.com/AbdelStark/tessera/issues/148) commits `S`, sends `G(r)` with each masked sumcheck's evaluations, and runs the batching;
   - [#149](https://github.com/AbdelStark/tessera/issues/149) opens `y`'s group;
   - [#159](https://github.com/AbdelStark/tessera/issues/159) checks both.
4. **Tests.** [#162](https://github.com/AbdelStark/tessera/issues/162)'s distinguisher suite carries EXP-0027's controls ('Testing Strategy').

## Testing Strategy

- **Conformance.** A vector of the batching on a toy layout: `M`, `S`, the claims, `α`, `ρ`, the rounds, `y` and `C̃(r*)`, from an independent oracle. EXP-0027's `batch.py` is the starting point.
- **Adversarial, one check each,** on the batching run from given claims, so that no earlier check catches the fault first:
  - a false claim, a `Γ_S` inconsistent with `S`, or a round coefficient changed: `Reject(MaskBatch { round: m })`;
  - a round missing a value: `Reject(MaskBatch { round: j − 1 })`;
  - a `y` off `z̃(r*)` with the last round refit to it, so that the final check holds: `Reject(Evaluation { commitment: Advice })` at its group.
- **In a whole proof,** a false `G(r)` fails at its own sumcheck's final check, unless the prover shifts one of the sumcheck's final evaluations to make up for it. The batching then fails, `Reject(MaskBatch { round: m })`, which RFC-0019 checks before the openings, except with probability `(N + 2m)/q`.
- **Soundness calibration over F65537.** EXP-0027 (c)'s composite cheater, against `(N + 2m + 1)/q`. Its ordering controls are `Γ_S` after `ρ`, and `α` before the claims.
- **Zero knowledge (#162).**
  - **The candidate.** The batching's view is identical for two witnesses whose earlier views agree.
  - **The negative controls.** Each must be told apart: no `S`; `M̃(r*)` sent alone on a layout with no spare direction; `S` on `C`'s support with an unclaimed block; and `S` drawn over `Fp`, on a layout with fewer than `4m` non-public positions. With more, the `2m` revealed `Fp2` functionals, `4m` over `Fp`, leave an `Fp` pad no uncovered direction, so the control would not separate.

## Open Questions

- **OQ-0070-1. A smaller pad.** `S` on the rows whose index has Hamming weight at most 2 had leak dimension 0 for every protocol-like `C` in EXP-0027. At 405B it would take 56 `Fp2` rows instead of 780. It leaks for sparse, sub-cube and single-entry `C`, so it needs a rank lemma, and the protocol's `C` must be shown to satisfy it.
- **OQ-0070-2. The other mask-only groups.** RFC-0068's column masks and the two reduction stages' folds could join `M`, if `S` covers their rows. That would save three groups, 192 MiB, for their rows in `S`.

## References

- [#489](https://github.com/AbdelStark/tessera/issues/489); [IDEA-0040](../../research/ideas/IDEA-0040-batched-mask-opening.md); [EXP-0027](../../research/experiments/EXP-0027-batched-mask-opening/README.md).
- RFC-0011, RFC-0013, RFC-0014, RFC-0032, RFC-0068; [`docs/zk/simulator.md`](../zk/simulator.md).
- Chiesa, Forbes, Spooner, "A Zero Knowledge Sumcheck and its Applications" (2017); Xie et al., "Libra" (CRYPTO 2019): masked sumchecks and their masks' openings.
- The fold-with-a-random-instance shape of BlindFold and Vega, there with homomorphic commitments ([research/bibliography.md](../../research/bibliography.md) `[VegaDocs26]`, `[KS25]`).
