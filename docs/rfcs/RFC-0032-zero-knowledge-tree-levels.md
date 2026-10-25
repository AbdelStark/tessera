# RFC-0032: Zero Knowledge of the Tree Levels and the LogUp Roots

- Status: Accepted
- Authors: @AbdelStark
- Created: 2026-10-17
- Target milestone: v0.4 (`tessera-protocol/2`); the design and its labels settled in v0.3
- Supersedes, on acceptance, for `tessera-protocol/2`:
  - [RFC-0013](RFC-0013-zero-knowledge.md#logup-and-product-tree-roots) 'LogUp and product-tree roots': the masking pair; the rows of 'What the verifier sees' for LogUp roots and tree levels; OQ-0013-2;
  - [RFC-0008](RFC-0008-logup-and-tables.md#fraction-tree) 'Fraction tree': the roots and the leaf checks;
  - [RFC-0010](RFC-0010-proof-schedule.md#tree-levels) 'Tree levels': forests of one tree, and the levels' sumchecks;
  - [RFC-0014](RFC-0014-interaction-and-soundness.md#transcript-schedule) 'Transcript schedule', rows 4 and 5.
- Amended by: [RFC-0068](RFC-0068-leaf-stage-zero-knowledge.md) ('Leaf checks': each LogUp instance's key and value columns take a mask row of their own, and the leaf level's raised degree falls in the round before the last), [RFC-0070](RFC-0070-mask-batching.md) ('Masked tree levels', 'Masked extensions that spare the pair bit' and 'Labels': `G_ℓ(r)` is sent with the level's evaluations, and it and the mask values `R̃_t` are claims on the masks' block, which RFC-0070's batching opens)
- Origin: [#224](https://github.com/AbdelStark/tessera/issues/224)

## Summary

The leaf stage leaks in `tessera-protocol/2` as RFC-0013 designs it.

- **The levels.** A tree level sends its round messages unmasked. Its evaluations sit at a point whose pair bit is Boolean, where RFC-0013's `Z = Π x_i(1 − x_i)` vanishes, so masked extensions leave them as they are. A tree that joins alone reveals its level 1 exactly. For a `SoftmaxRow` tree, that is the products of each half of a row's keys (OQ-0013-2).
- **The roots.** RFC-0013's masking pair, `(μ, δ)` in the lookup tree and `(−μ, δ)` in the table tree, shares one `δ`. The verifier therefore learns `d_a/d_t`, and with the public table, `Π_i(α − a_i)`: a fingerprint of the secret lookup multiset. Its leaf checks would also reveal `μ` and `δ`, which enter them at a public position.

This RFC makes every value of the leaf stage simulatable:

1. Every tree level runs RFC-0013's masked sumcheck.
2. Every forest holds at least two trees, so every level below the root has a random coordinate besides the pair bit.
3. Every level below the root, the leaves of a LogUp forest excepted, is a masked extension whose `Z` spares the pair bit, folded into the next level's sumcheck.
4. LogUp roots are not sent. A masked sumcheck over two variables proves the LogUp identity at the roots, and its final evaluations, masked, are the forests' root claims. The masking pair goes.
5. A LogUp forest's leaf level is RFC-0008's formula of the lookup keys, the lookup values and the multiplicities, read through their masked extensions. RFC-0013's masked extensions spare their lowest variable, which is the leaf level's pair bit, so the leaf checks keep RFC-0008's form. A product forest's leaf level takes a mask, which RFC-0030's reduction folds.

`tessera-protocol/1` (v0.3), which hides nothing, is unchanged.

## Motivation

RFC-0013 makes `tessera-protocol/2` zero-knowledge against a malicious verifier, item by item. The review of #59 found that the tree levels escape it (OQ-0013-2).

- **Round messages.** RFC-0014's schedule gives stages `zk-sum` and `zk-rho`, and gives tree levels nothing.
- **Evaluations.** A level reduces its claims to `Ṽ_{ℓ+1}(0 ‖ r')` and `Ṽ_{ℓ+1}(1 ‖ r')`, and a fraction level to its numerators and denominators at the same points ([RFC-0010](RFC-0010-proof-schedule.md#tree-levels)). RFC-0013's masked extension equals `Ṽ` at any point with a Boolean coordinate, and the pair bit is one.
- **A tree that joins alone.** At its first level, `r'` is empty, and the two evaluations are its level 1. Every LogUp tree joins alone ([RFC-0010](RFC-0010-proof-schedule.md#tree-levels)), so both halves of its fraction show.
- **The roots.**
  - **The leak.** The lookup root's denominator is `δ·Π_i(α − a_i)` and the table root's is `δ·Π_j(α − t_j)`, padding included. The verifier knows `α`, the table and the padding counts, so the ratio reveals `Π_i(α − a_i)`. That value is an evaluation, at a random point, of the polynomial whose roots are the secret lookups, enough to test a guessed witness.
  - **The accounting.** RFC-0013 left open how the leaf checks account for `μ` and `δ`. A committed scalar that enters a final check with a public coefficient is revealed by it.

## Proposed Design

Everything below is `tessera-protocol/2`. A level, a forest and a pair bit are RFC-0010's. `V` stands for a product forest's level, and for each of a fraction forest's numerators `N` and denominators `D`.

### Masked tree levels

Each batched tree level `ℓ` runs RFC-0013's masked sumcheck ([RFC-0013](RFC-0013-zero-knowledge.md#masked-sumcheck) 'Masked sumcheck'):

- the prover commits a masking polynomial `G_ℓ` in Round 1;
- at the level, it sends `Γ_ℓ = Σ_x G_ℓ(x)` under `tree/{ℓ}/zk-sum`, and `ρ_ℓ` is squeezed under `tree/{ℓ}/zk-rho`, before `γ_ℓ`;
- the level's sumcheck runs on its combination plus `ρ_ℓ·G_ℓ`; the prover sends `G_ℓ(r)` at the level's final point, last under `tree/{ℓ}/evals`, and [RFC-0070](RFC-0070-mask-batching.md)'s batching sumcheck opens it with every other mask claim.

The LogUp identity sumcheck below is masked the same way.

### Forests of two trees at least

A forest of one tree takes a padding tree of its height. That holds for every LogUp tree, and for a product forest of one tree:

| Forest | Padding tree's leaves | Its root |
|---|---|---|
| Product | 1 | 1 |
| Fraction | `(0, 1)` | `(0, 1)` |

The forest's root level then has a variable, the tree index, and every level `t` below it has at least one variable besides its pair bit. The padding tree is public and constant, so a prover computes its levels in closed form. The verifier pays one more round per level.

### Masked extensions that spare the pair bit

Level `t ≥ 1` of a forest has the pair bit `b`, its lowest variable, and the others `x`. Each of its polynomials is replaced by

```text
V'_t(b ‖ x) = Ṽ_t(b ‖ x) + Z_t(x)·R̃_t((b ‖ x)_col),        Z_t(x) = Π_j x_j·(1 − x_j)
```

- **The mask.** `R_t` is a fresh `Fp2` mask row over the level's column variables, the pair bit among them, as in [RFC-0013](RFC-0013-zero-knowledge.md#masked-extensions) 'Masked extensions'.
- **Agreement.** `Z_t` vanishes when `x` is Boolean, whatever `b`, so `V'_t = Ṽ_t` on the level's hypercube, and every level identity holds with `V'` in place of `Ṽ`.
- **What a level reveals.** At its final point `r'`, it sends `V'_{t+1}(0 ‖ r')` and `V'_{t+1}(1 ‖ r')`. `r'` holds at least one random coordinate, so `Z_{t+1}(r') ≠ 0`, and `R̃_{t+1}` at the two points masks them, independently, since the pair bit is a variable of the row.
- **Folding.** A level that proves the claim `V'_t(p) = v`, at `p = ρ ‖ q` with `q` its coordinates besides the pair bit's, proves

  ```text
  v = Σ_x eq(p, x)·( g(x) + Z_t(q)·R_t[x_col] )
  ```

  - `g(x)` is the level's combination of the next level, `V'_{t+1}(0 ‖ x)·V'_{t+1}(1 ‖ x)`, or the fraction combination of RFC-0010.
  - The prover sends `R̃_t` at the level's final point `r'` beside the next level's evaluations. That value is a claim on the masks' block, which [RFC-0070](RFC-0070-mask-batching.md)'s batching sumcheck opens: each level's point is its own, so no RFC-0011 group could share it.
  - The verifier never sees `R̃_t(p)` (RFC-0013 'Masked extensions', virtual tensors).
- **Degree.** `Z_{t+1}` vanishes at every summation point but the last round's, so only the last round's degree rises, to 7 for a product or fraction combination. RFC-0014's bound already allows `d ≤ 7`.
- **Masks per level.** One mask row per level of a product forest, and two per level of a fraction forest, for `N` and `D`, its leaf level excepted ('Leaf checks'). The leaf level's key and value columns take column masks of their own ([RFC-0068](RFC-0068-leaf-stage-zero-knowledge.md#column-masks)).
- **The root level** of a LogUp forest has one variable, the tree index `y`, and no pair bit. Its polynomials take `Z_0(y) = y·(1 − y)` and their own mask rows. The first level folds them like any other, at the identity's point `p = r_L` or `r_T`, with the factor `Z_0(p)`. A product forest's root values are public, and its root claim is computed by the verifier, as before.

### The LogUp identity sumcheck

An instance has two forests, `L` (its lookup tree and a padding tree) and `T` (its table tree and a padding tree). Their root levels hold:

- `N_L(0) = n_a`, `D_L(0) = d_a`, `N_L(1) = 0`, `D_L(1) = 1`;
- `N_T(0) = n_t`, `D_T(0) = d_t`, `N_T(1) = 0`, `D_T(1) = 1`.

The identity `n_a·d_t + n_t·d_a = 0` is `F(0, 0) = 0` for

```text
F(y_L, y_T) = N'_L(y_L)·D'_T(y_T) + N'_T(y_T)·D'_L(y_L)
```

- **The sum.** `Σ_{y ∈ {0,1}^2} eq((0, 0), y)·F(y) = 0` holds exactly when `F(0, 0) = 0`, since the masked root levels agree with the true ones on `{0, 1}`.
- **One sumcheck.** All instances' sums, combined by `γ`, run as one masked sumcheck over `(y_L, y_T)`, after `logup/beta`.
- **Its result.** The prover sends every instance's `N'_L(r_L)`, `D'_L(r_L)`, `N'_T(r_T)` and `D'_T(r_T)` at the final point `(r_L, r_T)`. They are uniform, masked by the root levels' rows, and the verifier checks the final equation. They are the forests' root claims: every lookup forest's root point is `r_L`, and every table forest's is `r_T`.

No root is sent, and the verifier checks no root denominator. A leaf denominator is `α − κ − β·e` for a lookup, with the key and the value fixed before `α` and `β`; `α − t_j` for a table row; and 1 for a padding tree. Each is zero with probability at most `1/|Fp2|`, so all are non-zero except with probability `(N + T)/|Fp2|`. When they are, the root denominators are non-zero, and the identity is RFC-0008's.

### Leaf checks

A forest's leaf level `h` has its leaf claims at `p = ρ ‖ q`.

**Masked extensions spare the lowest variable.** In `tessera-protocol/2`, RFC-0013's masked extension of a tensor takes

```text
Z(x) = Π_{i ≥ 2} x_i·(1 − x_i)
```

over every variable but the lowest.

- It still vanishes on the hypercube, and at every summation point but the last round's, for a tensor of two variables at least.
- It also masks a point whose lowest coordinate is Boolean.
- A LogUp leaf level's pair bit is the lowest bit of the lookup index, or of the table row's, and so the lowest variable of the keys, the values and the multiplicities. A LogUp tree has height 2 at least, padded with neutral leaves, so that they have two variables.

**Fraction forests.** The leaf level has no mask row of its own. Each instance's `κ'` and `e'` carry its column mask rows ([RFC-0068](RFC-0068-leaf-stage-zero-knowledge.md#column-masks) 'Column masks'). With `pt` the padding tree's coordinate and `x_low` the leaf index within a tree, its polynomials are RFC-0008's leaves read through the masked extensions `κ'`, `e'` and `m'`:

| Polynomial | Leaf level |
|---|---|
| Lookup numerators | `1 − pt`: public |
| Lookup denominators | `(1 − pt)·(α − κ'(x_low) − β·e'(x_low)) + pt` |
| Table numerators | `−(1 − pt)·m'(x_low)` |
| Table denominators | `(1 − pt)·(α − t̃(x_low)) + pt`: public |

- **What the last level reveals.** The last level's evaluations at `(b ‖ r')` are masked through `κ'`, `e'` and `m'`, or public.
- **The leaf checks keep RFC-0008's form**, with `p_top`, `p`'s padding coordinate:
  - the verifier checks the lookup numerators' claim against `1 − p_top`;
  - it checks the table denominators' claim against `(1 − p_top)·(α − t̃(p_low)) + p_top`;
  - the prover sends `e'(p_low)` under `logup/values`, from which the verifier derives the claim on `κ'(p_low)`;
  - the table numerators' claim gives `m'(p_low)`.
- **Where the claims go.** The claims on `κ'` and `e'` enter the leaf stage's layer sumcheck, which reduces them to claims on the tensors the columns read and on the column masks ([RFC-0068](RFC-0068-leaf-stage-zero-knowledge.md#the-layer-sumcheck)). Those on `m'` go to the openings, as RFC-0013 says.
- **Degree.** The tree index is the leaf level's highest variable, so it takes the last level's last round. The `Z` of `κ'`, `e'` and `m'` spares it, and does not vanish at that level's summation points from the round before the last on. The fraction forests' leaves therefore raise that round's degree, to at most 7, and not the last round's (RFC-0068).

**Product forests.** `C̃·k'` would raise the leaf level's degree past 7, so a `SoftmaxRow` forest's leaf level takes its own mask row, like a level above it. RFC-0030's reduction folds it: `Σ_x eq(p, x)·(C(x)·k(x) + Z_h(q)·R_h[x_col]) = L'_h(p) − 1 + C̃(p)`. The instance exists already, so no stage is added.

### What the verifier sees

These rows replace RFC-0013's row for LogUp roots and add the tree levels:

| Revealed item | Mechanism | Randomness |
|---|---|---|
| Tree-level and identity round messages | Masked sumcheck | `G_ℓ`, and the identity's `G`, per level |
| Tree-level evaluations `V'_{t+1}(b ‖ r')` and the mask values `R̃_t` | Masked extensions that spare the pair bit; at a LogUp leaf level, those of its keys, values and multiplicities | One `Fp2` mask row per level and polynomial, LogUp leaf levels excepted; there, each instance's key and value columns' own mask rows ([RFC-0068](RFC-0068-leaf-stage-zero-knowledge.md#column-masks)) |
| LogUp roots | Not sent: the identity sumcheck's final evaluations, masked | The root levels' mask rows |

### Labels

`tessera-protocol/2` replaces RFC-0014's rows 4 and 5:

| Order | Operation | Label |
|---|---|---|
| 4 | squeeze `α`, `β`; then the identity sumcheck: absorb masked sum, squeeze `ρ`, squeeze `γ`, rounds, final evaluations | `logup/alpha`, `logup/beta`, `logup/identity/zk-sum`, `logup/identity/zk-rho`, `logup/identity/gamma`, `logup/identity/round/{j}`, `logup/identity/r/{j}`, `logup/identity/evals` |
| 5 | per tree level `ℓ`: squeeze the root point of the product forests that join at `ℓ`, absorb masked sum, squeeze `ρ_ℓ`, `γ_ℓ`, and `λ_ℓ` at a level with fraction forests, rounds, evaluations, line | `tree/{ℓ}/root/{c}`, `tree/{ℓ}/zk-sum`, `tree/{ℓ}/zk-rho`, `tree/{ℓ}/gamma`, `tree/{ℓ}/lambda`, `tree/{ℓ}/round/{j}`, `tree/{ℓ}/r/{j}`, `tree/{ℓ}/evals`, `tree/{ℓ}/line` |

`logup/roots` has no use in `tessera-protocol/2`. `logup/values` stays, after the last level, as in row 5 of `tessera-protocol/1`. `tree/{ℓ}/evals` carries each instance's evaluations, then the mask values `R̃_t`, then `G_ℓ(r)` ([RFC-0070](RFC-0070-mask-batching.md)).

### Soundness

- **The identity sumcheck.** Two rounds of degree at most 4, `8/|Fp2|`.
- **The LogUp identity.** Its error is `2(N + T)/|Fp2|` per instance: RFC-0008's `(N + T)/|Fp2|`, plus the zero-denominator event above.
- **The masks.** They change no identity on the hypercube, and the masked sumchecks keep their degree bounds.
- **The calculator** ([RFC-0014](RFC-0014-interaction-and-soundness.md#soundness-budget)) counts each of these.

### Cost

| Item | Cost |
|---|---|
| Masked sumcheck | `2(n(d + 1) + 1)` committed elements per level: about 700 at `n = 45`, `d = 7`, so a few times 10^4 per proof |
| Mask rows | One `Fp2` row per level of a product forest, two per level of a fraction forest. RFC-0013's cost table counts tree levels among its mask rows; OQ-0013-1's packing applies. |
| Padding trees | Constant, computed in closed form; one more round per level |
| Identity sumcheck | Two rounds |

### For the simulator argument

RFC-0013's simulator ([#145](https://github.com/AbdelStark/tessera/issues/145)) covers every row of 'What the verifier sees', these included. Its steps extend as follows:

- **The identity sumcheck.** The simulator samples its final evaluations uniformly, subject to its final equation, and simulates its rounds with the masked-sumcheck simulator.
- **Each tree level, in order.** It samples the evaluations `V'_{t+1}(b ‖ r')` and the mask value `R̃_t` uniformly, subject to the level's final equation, then simulates its rounds. They are uniform when the `eq` vectors of the points at which each mask row is revealed are linearly independent: `(0 ‖ r')`, `(1 ‖ r')` and the next level's final point. That fails with probability at most `3^2/|Fp2|` per row.
- **The value messages** `e'(p_low)` are evaluations of the value columns, which their column masks mask ([RFC-0068](RFC-0068-leaf-stage-zero-knowledge.md#zero-knowledge)). They are in the span of the last level's pair-point values.

## Alternatives Considered

1. **Accept what the levels reveal.**
   - Rejected: a level reveals linear functionals of secret activations, and the roots a fingerprint of the lookups. The verifier is a malicious party (06), so either lets it test a guessed witness.
2. **Masking leaves in both halves of every tree**, to hide a lone tree's level 1.
   - Rejected: each needs a slot. Its committed values enter the leaf checks with public coefficients, which reveal them.
3. **Independent masking pairs**, `μ_a/δ_a + μ_t/δ_t = 0`, which make the roots uniform.
   - Rejected: the leaf checks still reveal every masking scalar.
4. **One fraction tree per instance, lookup and table as its halves.**
   - Rejected: RFC-0008 'Alternatives Considered' 4 still holds; it would pad the shorter side to the longer.
   - Not needed: the identity sumcheck combines two forests of different heights.
5. **A one-hot zero test for `SoftmaxRow`** in place of its product tree, to avoid revealing which half holds a row's maximum.
   - Rejected: it commits a vector of a row's length per row, as many elements as the exponentials.
6. **Commit the top levels and prove them separately.**
   - Rejected: more commitments, and a second argument for what the masks already do.

## Drawbacks

- `tessera-protocol/2`'s leaf stage differs from `tessera-protocol/1`'s: no roots, an identity sumcheck, padding trees, masked levels. The non-hiding mode under `test-params` keeps `tessera-protocol/1`'s.
- Every LogUp tree doubles its leaf count with a constant padding tree. A prover that materializes it pays twice.
- Fraction forests need two mask rows per level.

## Migration / Rollout

The design and its labels are settled now, before `tessera-protocol/1` freezes the label table in v0.3. The masking itself lands in v0.4 with `tessera-protocol/2`:

- the masked sumchecks and the masked extensions (#148, #154);
- the identity sumcheck and the LogUp side (#149);
- the verifier's checks (#159);
- the simulator argument (#145), which gates them.

## Testing Strategy

- **Agreement.** `V'_t = Ṽ_t` on every level's hypercube, exhaustively for small trees. A forest of one tree and its padded form prove the same leaf claims.
- **Hiding**, over 10^4 runs of two witnesses with the same public data, each indistinguishable in the distinguisher suite of [07 §7](../spec/07-testing-strategy.md#7-zero-knowledge-tests):
  - the evaluations of every level;
  - the identity's final evaluations;
  - a lone LogUp tree's level 1.
- **The root leak.** A distinguisher computing `Π_i(α − a_i)` from `d_a/d_t` tells two lookup multisets apart against RFC-0013's masking pair, and is at chance here.
- **Soundness, each rejected:**
  - an invalid multiset, at the identity sumcheck: `Reject(LogUpIdentity)` ([RFC-0033](RFC-0033-tree-level-rejections.md));
  - a level's evaluations shifted by a mask value: `Reject(TreeLevel)`;
  - a value evaluation `e'(p_low)` off by one: `Reject(Evaluation)` at the opening.

  Over `F65537`, the identity's false-accept rate stays within `2(N + T)/|F|`.

## Open Questions

None. The mask-row packing of OQ-0013-1 applies to tree levels unchanged.

## References

- [RFC-0008](RFC-0008-logup-and-tables.md), [RFC-0010](RFC-0010-proof-schedule.md), [RFC-0013](RFC-0013-zero-knowledge.md), [RFC-0014](RFC-0014-interaction-and-soundness.md), [RFC-0030](RFC-0030-masked-exponentials.md).
- Xie, Zhang, Zhang, Papamanthou, Song, "Libra", CRYPTO 2019 (masked layer polynomials).
- Chiesa, Forbes, Spooner, "A zero knowledge sumcheck and its applications", 2017.
- [#224](https://github.com/AbdelStark/tessera/issues/224), [#59](https://github.com/AbdelStark/tessera/issues/59), [#145](https://github.com/AbdelStark/tessera/issues/145).
