# RFC-0013: Zero Knowledge: Masked Sumchecks, Masked Extensions, and Hiding Openings

- Status: Accepted
- Authors: @AbdelStark
- Created: 2026-10-15
- Target milestone: v0.4

## Summary

Tessera is made zero-knowledge with four mechanisms, each tied to one kind of revealed data: (1) every sumcheck is masked with a committed random polynomial so its round messages are simulatable; (2) every tensor whose evaluation crosses a stage boundary is replaced by a masked extension `V' = Ṽ + Z·R_V`, which agrees with `Ṽ` on the hypercube but has uniformly random evaluations off it; (3) every commitment uses vanishing-polynomial encoding masks so that opened columns are uniform, and every revealed combined row includes a fresh blinding row; (4) every LogUp fraction tree includes a zero-numerator leaf with a random denominator so its root reveals nothing. Challenges derive from a coin-tossed seed, so the construction holds against a malicious verifier. Layers are sumcheck variables, so each proof reveals one masked evaluation per tensor kind rather than one per layer, which keeps the weight-commitment budget affordable.

## Motivation

The verifier is a counterparty state; weights, prompts, outputs and activations must stay hidden from it even if it deviates (source §2). The source's default is "Libra-style masked sumchecks with small committed masking polynomials, plus hiding commitments" (source §7, "Zero knowledge"; §10). A non-ZK proof leaks, per stage, partial sums of witness polynomials, evaluations of every tensor at random points, combined rows that are linear combinations of committed rows, and LogUp roots that are products over all keys. Each needs a mask.

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
| Sumcheck round messages | Masked sumcheck | Committed masking polynomial `G_k` per stage |
| Final evaluations of each stage | Masked extensions | One mask row `R_V` per stacked tensor kind |
| LogUp root `(0, d_root)` | Random-denominator leaf | One committed `δ` per tree |
| Combined rows `u_g` | Blinding rows | One fresh blinding row per revealed combination |
| Opened columns | Vanishing encoding masks | `P_A` (advice) or `P_W` (weights) coefficients per row |
| `Û`, lengths, digests | Public by design | — |

### Stacked layers

Each stage treats per-layer instances of the same tensor kind as one stacked tensor with `⌈log2 n_layers⌉` layer variables (padded layers are zero). A stage therefore ends with one claim per stacked tensor at a point `(ρ_ℓ, r)` that includes a layer coordinate. This is equivalent to the batching of [RFC-0010](RFC-0010-proof-schedule.md#batching-within-a-stage) with `eq(ρ_ℓ, ℓ)` in place of powers of `γ`. It matters here because every revealed evaluation consumes mask randomness: one per tensor kind per proof instead of 126.

### Masked sumcheck

For a stage proving `H = Σ_x f(x)` over `n` variables with round degree `d`:

1. In Round 1, the prover commits `G_k(x) = a_0 + Σ_{j=1}^{n} g_j(x_j)` with each `g_j` a random univariate polynomial of degree `d` (coefficients in the advice mask block).
2. In the stage, the prover sends `Γ = Σ_x G_k(x)`; the transcript yields `ρ_k ∈ Fp2` (label `stage/{k}/zk-rho`).
3. The parties run the sumcheck on `f + ρ_k G_k` with claim `H + ρ_k Γ`.
4. At the end, `G_k(r)` is opened from the advice commitment as an ordinary claim.

The round messages are then simulatable from `H`, `Γ` and the final evaluation (Chiesa, Forbes, Spooner 2017; Libra, Xie et al. 2019). Cost: `n(d + 1) + 1` committed elements per stage.

### Masked extensions

For every stacked tensor `V` whose evaluation is revealed (as a stage output claim or an opening claim), define

```math
V'(x) = \tilde V(x) + Z(x) \cdot \tilde R_V(x_{col}), \qquad Z(x) = \prod_i x_i (1 - x_i)
```

where `R_V` is a random advice row over `V`'s column variables (its low `min(n_V, log2 k)` variables). Properties:

- `V' = Ṽ` on the Boolean hypercube, so every sumcheck identity over the hypercube holds unchanged with `V'` in place of `Ṽ`.
- `Z` vanishes at every summation point of every round except the last, so the prover's round computations change only in the last round.
- `V'(p) = Ṽ(p) + Z(p) · R̃_V(p_col)`. For `p` off the hypercube, `Z(p) ≠ 0`, and the values `R̃_V(p_col^{(i)})` at the at most `e_V` revealed points are jointly uniform whenever the vectors `eq(p_col^{(i)}, ·)` are linearly independent, which fails with probability at most `e_V^2 / |Fp2|`.
- **Committed tensors.** `V'(p)` is a linear functional of the advice vector (data positions plus `R_V`'s row) whose column part is `eq(p_col)` in both terms, so it falls in the same opening group as the data claim ([RFC-0011](RFC-0011-advice-commitment.md#opening-protocol)).
- **Virtual tensors.** The producing relation proves `V'(p) = Σ_x rel(p, x)·inputs(x) + Z(p) Σ_{x_col} eq(p_col, x_col) R_V[x_col]`; the second sum is added to the stage's sumcheck over the column variables, so the verifier never learns `R̃_V(p_col)` itself, only `R̃_V` at the stage's final point (as in Libra's treatment of layer masks).
- Degree: `V'` has degree at most 3 in each variable; round messages grow by a few coefficients.

Weights follow the same rule with the mask row stored as a blinding row of the weight matrix ([RFC-0012](RFC-0012-weight-commitment.md#budget-state-machine)); each proof consumes one mask row per weight matrix.

### Hiding openings

- **Columns.** Every committed row carries a vanishing mask of `P` random coefficients (RFC-0011 encoding). Any set of at most `P` codeword positions outside `H` is uniformly distributed and independent of the message. Advice commitments are opened once (`τ_A ≤ P_A`); weight matrices are opened up to `P_W` distinct positions per epoch.
- **Combined rows.** For each revealed combination (each claim group `g` and the proximity row), a fresh blinding row `Q_g` with its own vanishing mask is included with coefficient 1: the prover reveals `u'_g = Σ_row c_g(row) A_row + Q_g` and the claimed value `q_g = ⟨Q_g, eq(col_g)⟩`, which enters the group check as one more claim. Soundness is unchanged: `Q_g` is a committed row like any other. `u'_g` is uniform because `Q_g` is used exactly once; `q_g` is determined by `u'_g` and the other revealed values, so it adds no information.
- **Masks of combined masks.** The combined vanishing-mask polynomials `M_g` include `Q_g`'s mask, so they are uniform as well.

### LogUp and product-tree roots

- Each LogUp fraction tree includes one extra leaf `(0, δ)` with `δ` random and committed. The sum is unchanged; `d_root` is multiplied by a uniform non-zero factor.
- Product-tree roots are zero by the statement (the max certificate), so they reveal nothing. Intermediate levels are masked extensions like any virtual tensor.

### Malicious verifier

- The seed is coin-tossed: the verifier commits to its share before seeing the prover's, so the seed is uniform even if the verifier is malicious; every sumcheck challenge derives from it by Fiat–Shamir. Masked-sumcheck simulation needs only that challenges are uniform.
- Live column queries are the verifier's free choice; hiding of columns holds for every choice within the budget.
- Selective aborts cannot produce two transcripts over one `Round1`: the prover answers at most one seed per `Round1` (INV-PROTO-1, [RFC-0014](RFC-0014-interaction-and-soundness.md#invariants)).

### Simulator argument

Before any v0.4 implementation work merges, `docs/zk/simulator.md` must give a complete simulator for one proof and its proof of statistical indistinguishability, reviewed by a second maintainer. Outline:

1. The simulator plays the coin toss, programming the random oracle at `H("coin-toss", ·)` to a uniformly chosen seed.
2. It commits to uniformly random rows in place of advice; roots are distributed identically because opened positions are uniform in both worlds and unopened leaves are random-oracle outputs of high-entropy inputs.
3. For each stage in order, it samples the masked final evaluations uniformly subject to the stage's final-check equation, then simulates round messages with the masked-sumcheck simulator.
4. It samples combined rows `u'_g` uniformly and sets each `q_g` to satisfy its group check.
5. It answers column queries with uniform symbols consistent with the sampled combined rows (solvable because each queried column adds one linear constraint per row on independent mask coefficients).
6. Indistinguishability error is bounded by the sum of degenerate events: `Z(p) = 0`, linearly dependent `eq` vectors, and the masked-sumcheck failure terms, each at most a small polynomial over `|Fp2|`.

### Cost

| Item | 405B, 16K tokens |
|---|---|
| Mask rows `R_V` (about 2,000 stacked tensors and tree levels) | about 8×10^9 elements, 0.3% of advice |
| Stage masking polynomials | < 10^4 elements |
| Blinding rows for advice combined rows (`G + 1 ≤ 17`) | 17 rows |
| Weight budget per proof | 3 blinding rows per weight matrix (evaluation, proximity, mask) |
| Extra sumcheck work | last round only |

The weight budget per proof is three rows, so `B_W = 2048` gives `(2048 − 2)/3 = 682` proofs per epoch, matching the column budget of about 674 ([RFC-0012](RFC-0012-weight-commitment.md#budget-state-machine)).

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
- LogUp roots: `d_root` distribution independent of the key multiset.
- End-to-end simulator test: simulated and real transcripts of the tiny model are indistinguishable at significance 0.001 over 10^4 runs.
- Budget: each proof consumes exactly three weight blinding rows per matrix and fails cleanly when they run out.

## Open Questions

- **OQ-0013-1.** Can mask rows for small tensors (tree levels below 2^12 entries) be packed into shared rows without breaking the independence condition, to cut the mask-row count? Owner: @AbdelStark. Resolve during v0.4 with the simulator argument; adopt only if the argument covers it.

## References

- Source §2 (malicious verifier), §6 principle 11, §7 ("Zero knowledge"), §10.
- Chiesa, Forbes, Spooner, "A zero knowledge sumcheck and its applications", 2017.
- Xie, Zhang, Zhang, Papamanthou, Song, "Libra", CRYPTO 2019.
- Ames, Hazay, Ishai, Venkitasubramaniam, "Ligero", CCS 2017 (hiding via randomized encodings).
