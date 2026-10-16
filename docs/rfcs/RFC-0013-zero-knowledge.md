# RFC-0013: Zero Knowledge: Masked Sumchecks, Masked Extensions, and Hiding Openings

- Status: Accepted
- Authors: @AbdelStark
- Created: 2026-10-15
- Target milestone: v0.4

## Summary

Tessera is made zero-knowledge with four mechanisms, each tied to one kind of revealed data: (1) every sumcheck is masked with a committed random polynomial so its round messages are simulatable; (2) every tensor whose evaluation crosses a stage boundary is replaced by a masked extension `V' = Ṽ + Z·R_V`, which agrees with `Ṽ` on the hypercube but has uniformly random evaluations off it; (3) every commitment uses vanishing-polynomial encoding masks so that opened columns are uniform, and every revealed combined row includes a fresh blinding row; (4) every LogUp instance's two fraction trees include a pair of masking leaves whose fractions cancel, so that their roots reveal nothing. Challenges derive from a coin-tossed seed, so the construction holds against a malicious verifier. Layers are sumcheck variables, so each proof reveals one masked evaluation per tensor kind rather than one per layer, which keeps the weight-commitment budget affordable.

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
- Side channels outside the protocol messages; timing is handled by INV-ARCH-2.

## Proposed Design

### What the verifier sees and how each item is hidden

| Revealed item | Mechanism | Randomness source |
|---|---|---|
| Commitment roots | Hash of columns whose symbols are uniformly masked | Encoding masks |
| Seed | Coin toss (live) or hash with proof-of-work (non-interactive) | Both parties |
| Sumcheck round messages | Masked sumcheck | Committed masking polynomial `G_k` over `Fp2` per stage |
| Final evaluations of each stage | Masked extensions | One `Fp2` mask row `R_V` per stacked tensor kind |
| LogUp roots `(n_a, d_a)`, `(n_t, d_t)` | A pair of masking leaves `(μ, δ)`, `(−μ, δ)` | One committed `μ` and `δ` in `Fp2` per instance |
| Combined rows `u_g` | Blinding rows | One fresh `Fp2` blinding row per revealed combination |
| Opened columns | Vanishing encoding masks | `P_A` (advice) or `P_W` (weights) coefficients per row |
| `Û`, lengths, digests | Public by design | — |

**Masks over `Fp2`.** Committed values are in `Fp`, but every value in the table above but the columns is in `Fp2` (`Fp[X]/(X^2 − 7)`). A mask drawn from `Fp` and scaled by any public coefficient covers one `Fp`-dimension of an `Fp2` value and leaves the other one unmasked:
- an `Fp` blinding row leaves `Σ_row Im(c_g(row))·A_row` in the clear, `k` linear combinations of the witness, or of the weights, per combined row;
- `Fp` coefficients of a masking polynomial leave each round-message coefficient's component off the line through `ρ_k`;
- `Fp` masking leaves leave the root fraction's component off `Fp`.

Every mask in this RFC is therefore uniform over `Fp2`. An `Fp2` mask value is two committed `Fp` values, and an `Fp2` row is two committed `Fp` rows, `B = B^{(0)} + X·B^{(1)}`, which enter a combination with the coefficients `1` and `X`.

### Stacked layers

Each stage treats per-layer instances of the same tensor kind as one stacked tensor with `⌈log2 n_layers⌉` layer variables (padded layers are zero). A stage therefore ends with one claim per stacked tensor at a point `(ρ_ℓ, r)` that includes a layer coordinate. This is equivalent to the batching of [RFC-0010](RFC-0010-proof-schedule.md#batching-within-a-stage) with `eq(ρ_ℓ, ℓ)` in place of powers of `γ`. It matters here because every revealed evaluation consumes mask randomness: one per tensor kind per proof instead of 126.

### Masked sumcheck

For a stage proving `H = Σ_x f(x)` over `n` variables with round degree `d`:

1. In Round 1, the prover commits `G_k(x) = a_0 + Σ_{j=1}^{n} g_j(x_j)` with each `g_j` a random univariate polynomial of degree `d` over `Fp2` (each coefficient, and `a_0`, two values of the advice mask block).
2. In the stage, the prover sends `Γ = Σ_x G_k(x)`; the transcript yields `ρ_k ∈ Fp2` (label `stage/{k}/zk-rho`).
3. The parties run the sumcheck on `f + ρ_k G_k` with claim `H + ρ_k Γ`.
4. At the end, `G_k(r)` is opened from the advice commitment as an ordinary claim.

The round messages are then simulatable from `H`, `Γ` and the final evaluation (Chiesa, Forbes, Spooner 2017; Libra, Xie et al. 2019), because each of their non-constant coefficients is masked by a uniform `Fp2` value. Cost: `2(n(d + 1) + 1)` committed elements per stage.

### Masked extensions

For every stacked tensor `V` whose evaluation is revealed (as a stage output claim or an opening claim), define

```math
V'(x) = \tilde V(x) + Z(x) \cdot \tilde R_V(x_{col}), \qquad Z(x) = \prod_i x_i (1 - x_i)
```

where `R_V` is a random `Fp2` advice row, two committed rows, over `V`'s column variables (its low `min(n_V, log2 k)` variables). Properties:

- `V' = Ṽ` on the Boolean hypercube, so every sumcheck identity over the hypercube holds unchanged with `V'` in place of `Ṽ`.
- `Z` vanishes at every summation point of every round except the last, so the prover's round computations change only in the last round.
- `V'(p) = Ṽ(p) + Z(p) · R̃_V(p_col)`. For `p` off the hypercube, `Z(p) ≠ 0`, and the values `R̃_V(p_col^{(i)})` at the at most `e_V` revealed points are jointly uniform whenever the vectors `eq(p_col^{(i)}, ·)` are linearly independent over `Fp2`, which fails with probability at most `e_V^2 / |Fp2|`. An `Fp` row would need the `2e_V` real and imaginary parts of those vectors independent over `Fp`, which Schwartz–Zippel bounds only by about `2e_V·log2(k)/|Fp|`, near `2^-55`: hence the `Fp2` row.
- **Committed tensors.** `V'(p)` is a linear functional of the advice vector (data positions plus `R_V`'s row) whose column part is `eq(p_col)` in both terms, so it falls in the same opening group as the data claim ([RFC-0011](RFC-0011-advice-commitment.md#opening-protocol)).
- **Virtual tensors.** The producing relation proves `V'(p) = Σ_x rel(p, x)·inputs(x) + Z(p) Σ_{x_col} eq(p_col, x_col) R_V[x_col]`; the second sum is added to the stage's sumcheck over the column variables, so the verifier never learns `R̃_V(p_col)` itself, only `R̃_V` at the stage's final point (as in Libra's treatment of layer masks).
- Degree: `V'` has degree at most 3 in each variable; round messages grow by a few coefficients.

Weights follow the same rule with the mask row stored as blinding rows of the weight matrix ([RFC-0012](RFC-0012-weight-commitment.md#budget-state-machine)); each proof consumes one `Fp2` mask row, two blinding rows, per weight matrix.

### Hiding openings

- **Columns.** Every committed row carries a vanishing mask of `P` random coefficients (RFC-0011 encoding). Any set of at most `P` codeword positions outside `H` is uniformly distributed and independent of the message. Advice commitments are opened once (`τ_A ≤ P_A`); weight matrices are opened up to `P_W` distinct positions per epoch.
- **Combined rows.** Each revealed combination, each claim group `g` and the proximity row, includes a fresh `Fp2` blinding row `Q_g = Q_g^{(0)} + X·Q_g^{(1)}`, two committed rows, each with its own vanishing mask. `Q_g` enters with coefficient 1, its two rows with `1` and `X`. The prover reveals `u'_g = Σ_row c_g(row) A_row + Q_g`. `u'_g` is uniform on `Fp2^k` because `Q_g` is, and is used exactly once.
- **Blinding claims.** For a claim group, the prover sends the blinding claim `q_g = ⟨Q_g, eq(col_g)⟩` before the claims' coefficients exist. Every `q_g` is absorbed, in group order, under `pcs/advice/blinding`, before `γ` is squeezed (RFC-0014, row 7). The group check then reads `Σ_t γ^{t+1}·v_t + q_g = ⟨u'_g, eq(col_g)⟩` ([RFC-0011](RFC-0011-advice-commitment.md#opening-protocol) steps 2 and 6).
  - **Soundness.** The column checks bind `u'_g` to the committed rows, so `⟨u'_g, eq(col_g)⟩ = Σ_t γ^{t+1}·f_t + Q̃_g`, with `f_t` the true evaluations and `Q̃_g = ⟨Q_g, eq(col_g)⟩`. The check holds exactly when `(q_g − Q̃_g) + Σ_t γ^{t+1}·(v_t − f_t) = 0`. Everything in it but `γ` is fixed before `γ` is drawn, so a false claim makes it a non-zero polynomial of degree at most `T_g`, the group's number of claims, which vanishes with probability at most `T_g/|Fp2|`.
  - **Both conditions are needed.** If `q_g` were sent after `γ`, the prover could choose it to cancel any false claim. If a claim also had the coefficient 1, `q_g` could cancel that claim's error whatever the order.
  - **Hiding.** `q_g` reveals one linear functional of the fresh row `Q_g`, which is uniform since `eq(col_g) ≠ 0`. Given `q_g`, `u'_g` is uniform subject to its group check.
- **The proximity row** carries no claim. Its blinding row enters with coefficient 1, and only the column checks see it.
- **Masks of combined masks.** The combined vanishing-mask polynomials `M_g` include `Q_g`'s mask, so they are uniform as well.

### LogUp and product-tree roots

- Each LogUp instance's two trees ([RFC-0008](RFC-0008-logup-and-tables.md#fraction-tree)) include a pair of masking leaves: `(μ, δ)` in the lookup tree and `(−μ, δ)` in the table tree, with `μ` and `δ ≠ 0` uniform in `Fp2`, each committed as two `Fp` values. With `μ` and `δ` in `Fp`, the root fraction's component off `Fp`, and the class of the denominators' product in `Fp2^*/Fp^*`, would show.
  - The two fractions cancel, so the instance's sum is unchanged.
  - Each root `(n, d)` becomes uniform: `d` is multiplied by `δ`, and `n/d` is shifted by `μ/δ`.
  - Without the pair, a lookup root would reveal its fraction `Σ_i 1/(α − a_i)`, a function of the secret lookup multiset. A zero-numerator leaf alone rescales `n` and `d` but keeps that fraction.
  - The masking leaves change both trees' leaf claims. The lookup tree's numerators are 1 except at its masking leaf, and the table tree's denominators are `α − t` except at its masking leaf. How both leaf checks account for the committed `μ` and `δ` belongs to the tree levels' zero knowledge (OQ-0013-2).
- Product-tree roots are zero by the statement (the max certificate), so they reveal nothing. Intermediate levels are masked extensions like any virtual tensor, except at the evaluations a level reveals, which OQ-0013-2 leaves open.

### Malicious verifier

- The seed is coin-tossed: the verifier commits to its share before seeing the prover's, so the seed is uniform even if the verifier is malicious; every sumcheck challenge derives from it by Fiat–Shamir. Masked-sumcheck simulation needs only that challenges are uniform.
- Live column queries are the verifier's free choice; hiding of columns holds for every choice within the budget.
- Selective aborts cannot produce two transcripts over one `Round1`: the prover answers at most one seed per `Round1` (INV-PROTO-1, [RFC-0014](RFC-0014-interaction-and-soundness.md#invariants)).

### Simulator argument

Before any v0.4 implementation work merges, `docs/zk/simulator.md` must give a complete simulator for one proof and its proof of statistical indistinguishability, reviewed by a second maintainer. Outline:

1. The simulator plays the coin toss, programming the random oracle at `H("coin-toss", ·)` to a uniformly chosen seed.
2. It commits to uniformly random rows in place of advice; roots are distributed identically because opened positions are uniform in both worlds and unopened leaves are random-oracle outputs of high-entropy inputs.
3. For each stage in order, it samples the masked final evaluations uniformly subject to the stage's final-check equation, then simulates round messages with the masked-sumcheck simulator.
4. It samples each blinding claim `q_g` uniformly before `γ`, then each combined row `u'_g` uniformly subject to its group check, and each weight matrix's `u_T` likewise after `θ_T`.
5. It answers column queries with uniform symbols consistent with the sampled combined rows (solvable because each queried column adds one linear constraint per row on independent mask coefficients).
6. Indistinguishability error is bounded by the sum of degenerate events: `Z(p) = 0`, linearly dependent `eq` vectors, and the masked-sumcheck failure terms, each at most a small polynomial over `|Fp2|`.

### Cost

| Item | 405B, 16K tokens |
|---|---|
| Mask rows `R_V` (about 2,000 stacked tensors and tree levels, two rows each) | about 1.6×10^10 elements, 0.6% of advice |
| Stage masking polynomials | < 2×10^4 elements |
| Blinding rows for advice combined rows (`G + 1 ≤ 17`, two rows each) | 34 rows |
| Weight budget per proof | 6 blinding rows per weight matrix (two each for evaluation, proximity and mask) |
| Extra sumcheck work | last round only |

The weight budget per proof is six rows, and the linking proofs take four at each end of an epoch, so `B_W = 4096` gives `(4096 − 8)/6 = 681` proofs per epoch, matching the column budget of about 674 ([RFC-0012](RFC-0012-weight-commitment.md#budget-state-machine)).

### Invariants

- **INV-ZK-1.** Every value sent to the verifier is public, uniformly masked, or determined by previously sent values and public data.
- **INV-ZK-2.** Every mask row, blinding row and masking polynomial is used in exactly one revealed combination or evaluation set, within its degrees of freedom.
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

- Masked sumcheck: completeness for degrees 1–7; the simulator's round messages and real round messages pass a two-sample distribution test over 10^5 runs on a toy instance.
- Masked extensions: `V' = Ṽ` on the hypercube (exhaustive for small `n`); revealed `V'(p)` values over 10^5 runs of a fixed witness are uniform (chi-square on low bits) and uncorrelated with the witness (two witnesses give indistinguishable distributions).
- Hiding openings: opened columns and blinding-row combinations for two different witnesses are indistinguishable by the distinguisher suite of [07 §7](../spec/07-testing-strategy.md#7-zero-knowledge-tests).
- Blinding claims, adversarially. Each of these provers is rejected with `Reject(Evaluation)`:
  - one that compensates a false claim through `q_g`, sent after `γ`;
  - one that sends `q_g` before `γ` but cancels the error of a claim it gives the coefficient 1;
  - one that splits a weight claim as `(v_T + Δ, q_T − Δ)`.
  Over the test field, a false claim with an honest `q_g` is accepted at a rate within `T_g/|F|`.
- LogUp roots: the distribution of each instance's roots is independent of the key multiset.
- End-to-end simulator test: simulated and real transcripts of the tiny model are indistinguishable at significance 0.001 over 10^4 runs.
- Budget: each proof consumes exactly six weight blinding rows per matrix and fails cleanly when they run out.
- **Masks over `Fp2`.** Each of these distinguishers tells two witnesses apart with probability close to 1 when its mask is drawn from `Fp`, and is at chance with the masks of this RFC:
  - one that reads `u'_g` off the line of an `Fp` blinding row;
  - one that reads a round-message coefficient off the line through `ρ_k`;
  - one that reads the root fraction's component off `Fp`.

## Open Questions

- **OQ-0013-1.** Can mask rows for small tensors (tree levels below 2^12 entries) be packed into shared rows without breaking the independence condition, to cut the mask-row count? Owner: @AbdelStark. Resolve during v0.4 with the simulator argument; adopt only if the argument covers it.
- **OQ-0013-2.** A tree level reveals `Ṽ_{ℓ+1}(0 ‖ r'_i)` and `Ṽ_{ℓ+1}(1 ‖ r'_i)` ([RFC-0010](RFC-0010-proof-schedule.md#tree-levels) 'Tree levels'), and fraction-tree levels reveal their numerators and denominators at the same points. These points have a Boolean coordinate, where `Z` vanishes, so masked extensions do not hide them. At the level where a single tree joins, they are exactly the two entries of its level 1: for a SoftmaxRow tree, the products of each half of the row's keys. A forest reveals its trees' level-1 entries combined at the level's random point. The round messages of tree levels are not masked either: RFC-0014's schedule has no `zk-sum` or `zk-rho` for them. How are tree levels made zero-knowledge, or why is what they reveal acceptable? Owner: @AbdelStark. Resolve before the label table freezes at `tessera-protocol/1` (v0.3), since the answer may add labels.

## References

- Source §2 (malicious verifier), §6 principle 11, §7 ("Zero knowledge"), §10.
- Chiesa, Forbes, Spooner, "A zero knowledge sumcheck and its applications", 2017.
- Xie, Zhang, Zhang, Papamanthou, Song, "Libra", CRYPTO 2019.
- Ames, Hazay, Ishai, Venkitasubramaniam, "Ligero", CCS 2017 (hiding via randomized encodings).
