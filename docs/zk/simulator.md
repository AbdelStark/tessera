# Zero-knowledge simulator argument

- **Status.** Draft for review. RISK-6 and [06 §9](../spec/06-security.md#9-security-gates) gate any `tessera-protocol/2` implementation on this argument, reviewed by a second maintainer ([RFC-0013](../rfcs/RFC-0013-zero-knowledge.md#simulator-argument) 'Simulator argument'). An independent adversarial reading has corrected this revision (§11). No maintainer has reviewed it.
- **Scope.** One `tessera-protocol/2` proof, live or non-interactive, against a malicious verifier. Then every proof and linking proof of one weight epoch, against verifiers that collude.
- **Issue.** [#145](https://github.com/AbdelStark/tessera/issues/145).
- **Related.** [#394](https://github.com/AbdelStark/tessera/issues/394) (mask use and the proximity combination), [#224](https://github.com/AbdelStark/tessera/issues/224) (tree levels), [#162](https://github.com/AbdelStark/tessera/issues/162) (distinguisher suite).

## 1. Result

**As first specified, `tessera-protocol/2` was not zero-knowledge.** Four of its messages revealed secret values outright, and the specification now masks each:
- **The leaf stage's per-layer split values** sat at Boolean layer coordinates, where the masked extension equals the tensor (§4.3, R8). [RFC-0068](../rfcs/RFC-0068-leaf-stage-zero-knowledge.md) sends none, and gives each LogUp instance's columns mask rows of their own.
- **`ChannelParams`' claims** sat at Boolean coordinates too (§4.5, R9). [RFC-0065](../rfcs/RFC-0065-masked-channel-params.md) masks them. With RFC-0068's leaf stage, `tessera-protocol/2` admits every site, those of two classes or more included.
- **The weight linking proof** revealed `W̃_T(ρ_T)` (§6.4, R3). [RFC-0064](../rfcs/RFC-0064-weight-linking-proof.md) opens the difference instead.
- **The epilogue's claim on the last checkpoint** sat at the Boolean layer `L − 1` (§4.3, R8). [RFC-0069](../rfcs/RFC-0069-one-layer-claims.md) masks a claim at one layer with the layer tensor's own mask row.

The protocol is still not shown zero-knowledge: the theorems below need the requirements of §9 that are open, R2 and R10 among them.

Further gaps are missing pieces of the protocol, not leaks:
- **Masked weights across an epoch.** Under the tensor proximity generator, they have no usable joint bound (§6.3, R2).
- **The masking polynomials' openings.** Their `G(r)`, and the tree levels' mask claims, had no opening scheme the budget counts (§4.2, R11). [RFC-0070](../rfcs/RFC-0070-mask-batching.md) opens them all in one masked batching sumcheck, which Lemma 4a covers (§4.3a).
- **Structured points.** Some masked points need a rank check on the compiled schedule (§3, R10).

The theorems below hold once the requirements of §9 are met.

**Theorem 1 (one proof).**
- **Assumptions.** R4, R7, R8, R9, R10 and R11 hold, along with the per-proof PRG of INV-ZK-3 and the random-oracle model for the transcript and commitment hashes.
- **Claim.** A simulator exists that outputs a view whose distance from `V*`'s view of a real proof is at most `ε_1 + ε_PRG + ε_H` (§8). It is given the public statement, the claim list, the parameter set, the public weight commitment, and black-box access to any verifier `V*`.
- **The terms.** `ε_1` is statistical, and at the 405B default `ε_1 < 2^-105`. `ε_PRG` is the PRG's computational distance, and `ε_H` the hash terms.

**Theorem 2 (one epoch).**
- **Assumptions.** Theorem 1's, plus R2, R3 and R6.
- **Claim.** The joint view of every verifier of one weight epoch is simulatable from public data and its proofs' statements. Its distance is the proofs' summed distances plus `ε_W` (§6.2), and `ε_W ≤ 2^-230` at the default `B_W`.

## 2. Model

- **Fields.**
  - `Fp` with `p = 2^64 − 2^32 + 1`, `Fp2 = Fp[X]/(X^2 − 7)`, `q = |Fp2| = p^2`.
  - A vector over `Fp2` is a pair of vectors over `Fp`.
  - "Uniform" means uniform over the stated set.
- **The verifier.**
  - **Its strategy.** `V*` follows any strategy: its coin-toss share, its column queries, and whether and when it aborts are arbitrary functions of what it has seen. In non-interactive mode, it queries the random oracle at will.
  - **Colluding verifiers.** Several verifiers of one epoch pool their views, and choose which sessions complete.
- **Secrets.** The witness: the tokens, activations and every other advice data row. The weights. All masking randomness: the advice masks, from the per-proof secret seed; and the weights' masks and blinding rows, from the epoch's `masks_seed`.
- **Pseudorandomness.** INV-ZK-3 draws masks from a PRG. A first hybrid replaces each PRG output by uniform values, at a computational distance `ε_PRG`. The rest of the argument is statistical, so the theorems are computational overall.
- **Hashes.**
  - Commitment roots, column leaves, the transcript and the non-interactive seed are random-oracle outputs.
  - A root over columns that are never opened is then independent of their contents, up to collisions.
  - The simulator programs the oracle at the seed and at the opened leaves.
  - `ε_H` collects collisions, `≤ Q_H^2/2^256` for `Q_H` queries, and programming failures (§4.1).

## 3. Tools

**Lemma 1 (linear views).**
- **The setting.** A view `y = A·s + B·w` over `Fp`, with `s` uniform over `Fp^m` and `w` fixed. Let `P·y = c` be every linear relation the verifier can check, with `c` computable from public data.
- **The claim.** If the image of `A` is the whole subspace `ker P`, then `y` is uniform on `{y : P·y = c}`, and a simulator that samples that set uniformly is exact.
- **Proof.** `A·s` is uniform on `im A = ker P`, and `P·B·w = c`.

In practice `A` is triangular: each revealed combination gets a **designated** block of randomness, which enters it with an invertible coefficient block and no earlier combination. Each new combination is then uniform given the earlier ones, except for the relations `P` names.

**Lemma 2 (`Fp2` masks).**
1. **`Fp2` vectors.** A vector `R = R^{(0)} + X·R^{(1)}` that enters every revealed value `Fp2`-linearly, as `c·⟨R, e⟩`, gives jointly uniform values when the vectors `e` are `Fp2`-linearly independent.
   - For the `eq` vectors of `t` points with independent uniform coordinates, independence fails with probability `≤ t^2/q`, by Schwartz–Zippel over `Fp2`.
   - **Structured points.** The premise is independent uniform coordinates, and some revealed points lack it. Softmax row sums sit at key coordinates equal to `1/2` (RFC-0037's pool order), and some claims share a column sub-point. For such rows the condition is a rank condition on the compiled schedule (R10).
2. **`Fp` rows.** `Fp` rows that enter one revealed `Fp2` value with coefficients `c_i ∈ Fp2` cover all of `Fp2` unless every `c_i` lies on one `Fp`-line.
   - **One pair, `(1, X)`.** This always covers.
   - **One pair of challenge-derived coefficients.** It fails with probability about `1/p`. For `eq(ξ*, b_0), eq(ξ*, b_1)` with indices differing in bit `i`, the ratio is `ξ*_i/(1 − ξ*_i)`, in `Fp` exactly when `ξ*_i ∈ Fp`.
   - **Several such rows.** Their `eq` coefficients share `ξ*`'s coordinates, so further rows do not keep multiplying the failure by `1/p`.
     - Two `Fp` rows give one ratio, so the failure is about `1/p`.
     - Three `Fp` rows whose indices differ in two different bits `i` and `j` give two ratios. They force `ξ*_i` and `ξ*_j` into `Fp`, probability about `p^{-2}`.
     - Nothing below `p^{-2}` is claimed here.

**Lemma 3 (encoding masks; Ligero).**
- **The setting.** A row `A` encoded as `g = enc(A) + Z_H·m`, with `m` uniform of degree `< P`, and evaluated on a coset disjoint from `H`.
- **The claim.** At most `P` distinct positions of `g` are jointly uniform and independent of `A`.
- **Why.** `Z_H(x_j) ≠ 0` off `H`, and the evaluations of `m` at `P` distinct points are an invertible Vandermonde image of its coefficients.

## 4. One proof

In transcript order (RFC-0014 'Transcript schedule', with RFC-0032's rows 4 and 5): each item's distribution, what the simulator samples, and the degenerate events.

### 4.1 Session, `Round1`, seed

- **Public items.** `SessionContext`, the statement and `Round1`'s public fields are public, among them the weight reservation's declared blinding-row indices (RFC-0012 'Budget state machine'). `leaf_groups` is zero in `tessera-protocol/2` (RFC-0054 OQ-0054-1).
- **The advice root.**
  - **What the simulator does.** It commits to uniformly random columns.
  - **Why that matches.** Every column the verifier opens is uniform in both worlds: data and other non-designated rows by Lemma 3, designated rows by §4.6. Every other leaf is an oracle output on a high-entropy input.
  - **Programming.** Once the queries are known, the simulator programs the opened leaves. It fails only if `V*` queried a leaf input first.
- **The seed.**
  - **Live.** The verifier commits to its share before it sees the prover's (INV-PROTO-2), so the seed is uniform from `V*`'s view.
  - **Non-interactive.** The simulator programs the oracle at the seed's input, after the proof of work.
- **Degenerate events.** The programming failures, in `ε_H`.

### 4.2 The LogUp identity sumcheck and the tree levels (RFC-0032)

**Lemma 4 (masked sumchecks).**
- **The setting.** A stage or level proves `H = Σ_x f(x)` over `n` variables, its round messages of RFC-0010's one degree `d`.
  - The prover commits `G(x) = a_0 + Σ_j g_j(x_j)`, each `g_j` uniform of degree `d` over `Fp2`.
  - It sends `Γ = Σ_x G(x)`, and `ρ` is squeezed.
  - The rounds run on `f + ρ·G` from the claim `H + ρ·Γ`, and the prover sends `G(r)`, which RFC-0070's batching opens (§4.3a).
- **Claim.** Given `ρ ≠ 0`, the view `(Γ, p_1, …, p_n, G(r))` is simulatable from `H` and the final value `f(r)`.
- **Proof.**
  - Round `j`'s polynomial is `F_j(X) + ρ·2^{n−j}·g_j(X) + c_j`, with `F_j` the unmasked round polynomial and `c_j` constant in `X`. Its non-constant coefficients are masked by `g_j`'s, which enter no other round, so they are uniform. Its constant is fixed by the round's check.
  - `Γ = 2^n·(a_0 + Σ_j g_{j,0}) + 2^{n−1}·Σ_{j, m ≥ 1} g_{j,m}` is uniform through `a_0`.
  - `G(r)` is fixed by the final check, `p_n(r_n) = f(r) + ρ·G(r)`.
  - The pair reveals nothing beyond the checks. `G(r) − Γ/2^n = Σ_{j, m ≥ 1} g_{j,m}·(r_j^m − 1/2)`. Substituting the revealed coefficients, this equals `ρ^{-1}·(public − (f(r) − H/2^n))`, since `Σ_j (F_j(r_j) − F_{j−1}(r_{j−1})/2)/2^{n−j}` telescopes, with `F_0(r_0) = H`.
- **Other inputs to `f`.** Batched instances, zero-checks and layer points change only `f`'s public factors.
- **Opening `G(r)` (R11).** `G(r)` is also an advice claim, but its column functional is `(r_j^m)` over the coefficient positions, not an `eq` vector, so RFC-0011 cannot open it alone. Each tree level's mask claim `R̃_t(r'_t)` sits at its own level's point.
  - **The cost of one group each.** A blinding pair, a blinding claim and a combined row of `k` `Fp2` values, about 64 MB at `k = 2^{22}`, per claim. At 405B that is 2,668 claims, and EXP-0027's 2,667 would take 612 groups even one per column point: 38.25 GiB.
  - **What [RFC-0070](../rfcs/RFC-0070-mask-batching.md) does.** The prover sends `G(r)` with the sumcheck's evaluations, so the final check above keeps its rejection. One masked batching sumcheck then opens every `G(r)`, every tree-level mask claim and the `SoftmaxRow` leaf mask's, with one more group (§4.3a, Lemma 4a).
- **Requirement R4.** Each `g_j` has the degree `d` that the sumcheck's round messages hold in every round (RFC-0010), which is at least every round's degree, the round whose degree the masked extensions raise included, up to 7: the last round, or at a LogUp leaf level the round before it (RFC-0032, RFC-0068, RFC-0070).
- **Degenerate event.** `ρ = 0`: `1/q` per masked sumcheck.

**The identity sumcheck.** Its final evaluations `N'_L(r_L), D'_L(r_L), N'_T(r_T), D'_T(r_T)` are values of masked root levels, `Ṽ(y) + y(1 − y)·R̃(y)`.
- **Why they are uniform.** They are uniform when `y(1 − y) ≠ 0` at `r_L` and `r_T`.
- **What the simulator does.** It samples them uniformly. `ρ·G(r)` then absorbs the final equation, and Lemma 4 gives the rounds.

**Each tree level `t`, in order.**
- **What it reveals.** `V'_{t+1}(0 ‖ r')` and `V'_{t+1}(1 ‖ r')`, and `R̃_t(r'_col)`.
- **Why they are uniform.** Mask row `R_t` is revealed at three points: `(0 ‖ r'_{t−1})` and `(1 ‖ r'_{t−1})` through level `t`'s values, and `r'_t` directly. By Lemma 2 the values are uniform when the three `eq` vectors are independent and `Z_t ≠ 0` at the first two.
- **What the simulator does.** It samples the values, and Lemma 4 gives the rounds.
- **Product-forest roots** are public.
- **Degenerate events, per level:**
  - `ρ_ℓ = 0`;
  - `Z_t = 0` at a revealed point, a coordinate other than the pair bit's falling in `{0, 1}`: `≤ 2(n_t − 1)/q` per point;
  - dependent `eq` vectors: `9/q` per mask row.

### 4.3 Stages, splits and masked extensions

Each stage is Lemma 4's masked sumcheck. Its final evaluations and its split values are evaluations `V'(p) = Ṽ(p) + Z(p)·R̃_V(p_col)`, with `Z` over every variable but the lowest (RFC-0032 'Leaf checks').

**Requirement R8.** No masked tensor is revealed at a point where its own `Z` vanishes, nor alone in a weighted sum all of whose terms are such points. Such a point is where the masked extension equals the tensor.
- **Today's `Z`.** It runs over every variable but the lowest, so a Boolean coordinate other than the lowest is such a point.
- **A weighted sum.** RFC-0037's shifted layer weights `eq(ρ, ℓ + 1)` would violate R8 if revealed alone. They are revealed only inside a committed group, with `w(0)·c'_0(r)`, which masks them.
- **What R8 does not cover.** Advice claims on the committed vector `Ã(r ‖ base_X)` have Boolean base bits, but they are not evaluations of a masked tensor in its own variables. R8 does not apply to them.
- **Claims at one layer.** The epilogue reads the last checkpoint at layer `L − 1` alone (RFC-0037 `Layers::One`), a Boolean layer coordinate. [RFC-0069](../rfcs/RFC-0069-one-layer-claims.md) makes such a claim one on the layer's tensor in its own variables, with a mask row of its own. At 405B one row masks the 33 claims of the final norm on the last checkpoint, in stages 1 and 2, at two points.
- **The leaf stage broke it as first specified.** RFC-0045's leaf stage, with RFC-0037's splits, sends one value per tensor, site and real layer, and stacks the values at the layer point `ρ_1` afterwards (RFC-0045 'The split', RFC-0037 'Splits').
  - **Why that leaks.** Each per-layer value sits at a Boolean layer coordinate, so it is unmasked. That is up to 126 evaluations of each looked-up activation tensor, per proof.
  - **The spec gap.** RFC-0045 states its layout for `tessera-protocol/1` only, and `tessera-protocol/2`'s leaf layout is unspecified.
- **Two remedies.**
  - **One mask row per tensor and layer.** The stacked extension becomes `Ṽ + Z(x)·Σ_ℓ eq(y, ℓ)·R̃_ℓ(x_col)`, with `Z` over the non-layer variables.
    - **Soundness.** It holds.
    - **The cost.** Up to 126 mask rows for each tensor kind the lookups read, not for every tensor.
  - **A layout whose leaf point gives a random layer coordinate.** RFC-0045 has no blocks for padded layers, 126 is not a power of two, and a site's lookups interleave by layer, so this needs three changes:
    - explicit blocks for all `2^λ` layer slots;
    - the layers innermost for each site and lookup, aligned to `2^{k+λ}`;
    - a layer point per site in place of a shared `ρ_1`, which changes stage 1's stacking.

  RFC-0054's OQ-0054-1 names the same failure for grouped splits.
- **What `tessera-protocol/2` does** ([RFC-0068](../rfcs/RFC-0068-leaf-stage-zero-knowledge.md)). Neither remedy: the columns take mask rows of their own, and a masked layer sumcheck replaces the per-layer values.
  - It reduces every instance's leaf claims to one value per site and tensor at a random layer point `ρ'_y`, where the tensors' own masks act. Stage 1 stacks there.
  - The column masks' terms are folded into it, as a virtual tensor's are, so each column mask is revealed only at the sumcheck's final point besides the column values.
  - The two remedies above are its alternatives 1 and 2. At 405B it costs 36 mask rows, where the first would cost 53,298.

**The LogUp leaf level and `logup/values`** (RFC-0032 'Leaf checks').
- **What is revealed.** The last level reveals the leaf polynomials at `(0 ‖ r')` and `(1 ‖ r')`, through `κ'`, `e'` and `m'`. `logup/values` sends `e'(p_low)` at the level's line point `(λ ‖ r')`, which is in the span of the first two: a public, deterministic dependence.
- **Why they are uniform.**
  - **What masks them.** The key and value columns are concatenations of their blocks' tensors (RFC-0045). The tensors' own masks do not mask them: a block of a site in the layers sits at a Boolean layer coordinate. Each instance's key column, and its value column when its table has values, carries a mask row of its own instead (RFC-0068 'Column masks').
  - **The keys and values.** At each of the two points, the lookup denominators read the key and value columns through their column masks: four mask values for an instance with values, covering three revealed combinations, the two denominators and `e'(λ ‖ r')`. The values are then uniform subject to that one public relation.
  - **The column masks' third point.** Each column mask is revealed once more, at the layer sumcheck's final point. Its three `eq` vectors are independent except with probability `9/q` (Lemma 2). The masks belong to one instance each, so no rank condition over shared tensors arises.
  - **The multiplicities.** The table numerators reveal `m'` at the same two points, and `m'(p_low)` follows from them. The same argument applies through `m`'s mask.

**Committed tensors.** Mask row `R_V` is revealed at `e_V` points, one per claim on `V`. The values are uniform when `Z(p) ≠ 0` at each point and the `e_V` vectors `eq(p_col)` are independent (Lemma 2).

**Virtual tensors.** `R̃_V(p_col)` is folded into the producing stage, and the fold reveals instead the extension of `R_V`'s whole row at the stage's last `log2 k` challenges, `u = R̃_row(r'')`: one more functional of `R_V`, counted in `e_V`. R10's rank therefore runs over vectors of the row's full width, `V`'s values padded with zeros past its `c` column variables (RFC-0013 'Virtual tensors'). A linear node's output carries its atoms' masks, not one of its own, since RFC-0037's split identities relate it to them.

**What the simulator does.** It samples each tensor's revealed values uniformly, subject to the split identities, which are linear and public. The stages' final equations do not constrain the values, since `ρ·G(r)` absorbs them (Lemma 4).

**Degenerate events.**
- `Z(p) = 0`: `≤ 2(n_V − 1)/q` per point (this R8 rules out structurally, beyond the probabilistic case);
- dependent `eq` vectors: `e_V^2/q` per mask row.

### 4.3a The masks' batching (RFC-0070)

After the last stage, [RFC-0070](../rfcs/RFC-0070-mask-batching.md) opens the mask claims, every masked sumcheck's `G(r)`, every tree level's `R̃_t(p_t)` and the `SoftmaxRow` leaf's `R̃_h(p_h)`, in one sumcheck.
- **The block.** The claims are linear functionals `⟨c_i, M⟩` of the masks' block `M`, which holds every masking polynomial's coefficients, the tree levels' masks whose values `tree/{ℓ}/evals` carries, and the `SoftmaxRow` forest's leaf mask, packed in aligned blocks, and nothing else.
- **The combination.** A challenge `α` combines them into `v = ⟨C, M⟩`, `C = Σ_i α^i·c_i`.
- **The pad.** `S`, laid out like `M` and committed in Round 1, masks the sumcheck: the prover sends `Γ_S = ⟨C, S⟩`, `ρ` is squeezed, and the rounds prove `Σ_x C̃·z̃ = v + ρ·Γ_S` with `z = M + ρ·S`.
- **The end.** `y = z̃(r*)` is one advice claim, in one more group.

**Lemma 4a (the masks' batching).** Let `T` be `M`'s non-public positions, `S` uniform over `Fp2^T` and zero off `T`, independent of everything sent before `Γ_S`. Given the earlier view and `ρ ≠ 0`, the batching's view `(Γ_S, the rounds, y)` depends on the secrets only through `v`, which the earlier view fixes.
- **Proof.**
  - `C_T ≠ 0`, since claim 0, the LogUp identity's `G(r)`, puts 1 on its `a_0` slot. So `Γ_S` is uniform, and given it, `S_T` is uniform on the hyperplane `⟨C_T, ·⟩ = Γ_S`.
  - So `z_T` is uniform on `{⟨C_T, z_T⟩ = v + ρ·Γ_S}`, which public values fix, and `z` is zero off `T`.
  - Every later message of the batching is a function of `z`, `C` and the challenges.
- **What the simulator does.** It samples `Γ_S` uniformly, then `z_T` on that hyperplane, then runs the honest prover. That is exact for every `C` and every adaptive choice of the challenges.
- **No rank condition.** Coverage replaces it: `S` must be uniform wherever `M` holds a mask value, claimed or not. A dependence among the rounds' functionals is shared by `M`'s side, a public relation. EXP-0027 measured the revealed functionals at `2m`, the most the sumcheck's shape allows, on protocol-like `C`.
- **What the lemma needs, each shown necessary by EXP-0027 on a toy protocol over F65537:**
  - a pad, since the bare batching leaks through a root level's pinned two-entry row;
  - `S` on `M`'s layout, not on `C`'s support, which depends on challenges, and leaks through a block no claim reads;
  - `M̃(r*)` and `S̃(r*)` never sent apart.
- **Two design rules beside it.** `S` is drawn over `Fp2`, as every mask is: the proof as written needs its two `Fp` blocks independent, and EXP-0027's prime fields do not test it. `M` holds only claimed masks, which keeps `S` no larger than needed: the lemma does not need it.
- **After the batching.** `S_T = ρ^{-1}·(z_T − M_T)`, so `M` and `S` enter nothing revealed later except under fresh randomness of its own:
  - `y`'s group, under its blinding pair (§4.4);
  - the proximity row, under `Q_ξ` and the free rows Lemma 5 names, which are not `S`'s;
  - the columns, under each row's encoding mask (Lemma 6).
- **Degenerate event.** `ρ = 0`: `1/q`.

### 4.4 Advice openings

**What is revealed** (RFC-0011 'Opening protocol', RFC-0013 'Hiding openings'):
- **Per group `g`:** the blinding claim `q_g = ⟨Q_g, eq(col_g)⟩`, then the combined row `u'_g = Σ_row c_g(row)·A_row + Q_g` and its mask `M_g`.
- **Once:** the proximity row `u_ξ` and its mask `M_ξ`.
- **The advice's rows.** Data rows; every mask row `R_V` and `R_t`; every masking polynomial's coefficient rows; the masks' pad `S` (RFC-0070); and the blinding pairs `Q_g` and `Q_ξ`.
- **The masks' batching group.** `y`'s group is one group like the others, with its own blinding pair `Q_y`: Lemma 5 covers it.

**Lemma 5 (advice combinations).** The joint view `(q_g, u'_g, M_g)_g`, with `u_ξ` and `M_ξ`, is uniform subject to the group checks `Σ_t γ^{t+1}·v_t + q_g = ⟨u'_g, eq(col_g)⟩`.
- **Under RFC-0011's tensor generator**, this holds except with probability about `p^{-2}`.
  - **The groups.** Each `u'_g` takes `(1, X)` on `Q_g` and no other blinding pair, so the groups' rows are triangular.
  - **The proximity row.** `u_ξ` takes generator coefficients on every non-zero row. After the groups' rows are eliminated, the rows still free for it are:
    - `Q_ξ`;
    - each tensor's mask row `R_V` on its support, minus its `e_V` scalar functionals. A row read at as many points as its support is fully used. The tree levels' masks are `M`'s, which the batching leaves no free direction (below).

    The groups' use of the mask rows is absorbed by the `Q_g`.
  - **Mostly not free: the masking polynomials' rows.** Each non-constant coefficient `g_{j,m}` is pinned by its revealed round coefficient `F_{j,m} + ρ·2^{n−j}·g_{j,m}`. Of the constants, only `a_0 + Σ_j g_{j,0}` is pinned, so each masking polynomial keeps `n` spare directions.
  - **Not counted free: `M`'s and `S`'s rows.** After the batching, `S_T` is a function of `z_T` and `M_T` (§4.3a), so neither block is free for `u_ξ` beyond the batching's span. The argument below needs neither: its second free row is a large activation mask row.
  - **When it fails.** It fails only if every free row's coefficient lies on one `Fp`-line (Lemma 2.2).
    - **`Q_ξ`'s pair.** It is in no group, so the elimination leaves its coefficients unchanged. Its ratio forces some `ξ*_i` into `Fp`: about `1/p`.
    - **One more row.** Take any free row whose index differs from the pair's in a bit `j ≠ i`. Its coefficient, after the `Q_g` are eliminated, lies on the pair's line only when a non-constant Möbius function of `ξ*_j` lands in `Fp`: about `1/p`, independently.
    - **Together** that gives about `p^{-2}`. The large activation mask rows (`n_V ≥ 22`) supply such a row at every position.
  - **The scalars.** The scalar values `R̃_V(p_col)` stay uniform by Lemma 7's singular-case argument (§6.2).
- **Under R1**, it is exact: `Q_ξ` enters only as the fixed word, with `(1, X)`.
- **`q_g`.** It is one functional of `Q_g`, uniform since `eq(col_g) ≠ 0`. Given `q_g`, `u'_g` is uniform on the hyperplane its group check fixes.

**R1 is a soundness requirement, not a zero-knowledge one.**
- **What it is.** Draft RFC-0021's generator gives `Q_ξ` the fixed coefficients `1` and `X` and no generator coefficient, so that the combination is the affine-space generator its proximity bounds are stated for.
- **The text conflict.** RFC-0013 'The proximity row' gives `Q_ξ` coefficient 1, while RFC-0011 step 3, RFC-0012 and RFC-0049 give every row its generator coefficient. The texts disagree, and the generator's soundness argument decides between them (#394).

**What the simulator does.** It samples each `q_g` before `γ`, each `u'_g` on its hyperplane, and `u_ξ` and the `M`s, all uniformly.

### 4.5 Weight openings, one proof

**What is revealed, per matrix `T`** (RFC-0012 'Opening by the folding pass'):
- the blinding claim `q_T`;
- the evaluation row `u_T = θ_T·(Σ_ℓ eq(ρ_ℓ, ℓ)·F_ℓ + c·R_W) + β_T`, and its mask;
- the proximity row and its mask;
- the masked evaluation `W'(p) = W̃(p) + Z(p)·R̃_W(p_col)`.

**Why it is simulatable.** Lemma 5's argument applies with `β_T` designated for the evaluation row. `R_W` enters the scalars only through `W'(p)`, which is uniform once `Z(p) ≠ 0`. Its vector appearances are covered by `β_T` and the proximity row's free rows.

**Requirement R9.** `ChannelParams` needs masks of its own.
- **Why.** Every `ChannelParams` claim sits at `r_c ‖ 0^{γ−c_i} ‖ ρ_i ‖ vertex_4(t_i)` (RFC-0052): coordinates that are Boolean, so `Z = 0` and `W'(point_i) = W̃(point_i)`.
- **Shared points.** At 405B the type bits fall in the row part (`k = 2^20 < C·L_pad`), so two sites of one width share a column point. One mask row would then be read twice at one `eq` vector.
- **Shared masks.** The proof carries `m` evaluation rows (RFC-0052). If they share one `β` and one mask row, three of them eliminate both and reveal combinations of the folds.
- **The requirement.**
  - A mask per claim, with `Z` over the type tensor's own random coordinates `(r_c, ρ)`.
  - One designated block per evaluation row, and the blinding claims and the proximity row covered too.
  - The budget of RFC-0012, and Lemma 7, recounted for them.

  A first count gave each evaluation row its own pair as well: `4m + 2` rows per proof, 30 at `m = 7`, and 136 proofs per epoch at `B_W = 4096`. The design below needs `2m + 4`.

**Met by RFC-0065, with RFC-0068's leaf stage for sites of two classes or more.** [RFC-0065](../rfcs/RFC-0065-masked-channel-params.md) gives each claim its own mask row `R_i`, with `Z` over the type tensor's `c_i + λ` coordinates, and each matrix one pair `β_T` for its evaluation rows.
- **The cost.** A proof reserves `2e + 4` rows of a matrix with `e` claims: six for `Embed`..`LmHead`, as before, and 18 for `ChannelParams` at `e = 7`.
- **The epoch.** `ChannelParams` has `B_CP = B_W + 2·(e_CP − 1)·⌊(B_W − 4)/6⌋` blinding rows, 12,280 at 405B, so it serves as many proofs per epoch as the other matrices.
- **The argument.** It is Lemma 5's, with `R_i` designated for evaluation row `i`.
  - The map from the mask rows and `β_T` to the rows and blinding claims is onto what the checks allow. Linearly dependent column points give public relations among the `q_i`.
  - A value of `P'_{t_i}` revealed in a stage at a point `y` carries `β_T`'s functional at `y`. When it also carries a secret term, that functional must be independent of `β_T`'s others (R10, per matrix).
  - Lemma 7 covers the epoch with the `β_T` columns as its uniform block. Its singular case needs `{eq(col_i), eq(σ(col_i))}` independent, `σ` the conjugation, which fails with probability about `2^{−55}` within the singular event's `1/p`.
- **One matrix, one fate.** The claims share `β_T`, so one unmasked row would reveal `β_T`, and with it the folds through the other claims' `q_j`. No evaluation row is opened without its mask term. A prover whose `Z(x_i)` vanishes aborts.
- **The leaf stage.** As first specified, it revealed each class tensor once per real layer, at Boolean layer coordinates, before stacking at `ρ_1` (RFC-0045, RFC-0037). Those values were unmasked whatever the openings do, so `tessera-protocol/2` admitted only affine `ChannelParams` sites.
  - RFC-0068 reveals the class tensors' per-layer evaluations only inside masked column values, and their stacked claim at `(π(r_s) ‖ ρ'_y)`, every coordinate but the lowest a challenge, where RFC-0065's per-claim masks act.
  - So every site meets RFC-0065 'What the stages owe', and `tessera-protocol/2` admits sites of two classes or more.

### 4.6 Column answers

**Lemma 6 (columns).** For `τ ≤ P` distinct queried columns:
- **Every non-designated row.** Its symbols are uniform and independent of everything else revealed (Lemma 3). These rows are the data rows, the mask rows, the masking polynomials' rows and other blinding rows. A row's encoding mask enters the combined masks, but each combined mask is made uniform by its designated pair's own mask.
- **The designated pairs' symbols** are fixed by the column checks, `Σ_row c(row)·col_j[row] = enc(u)(x_j) + Z_H(x_j)·M(x_j)`.

- **`ChannelParams` under RFC-0065.** Its evaluation rows' designated blocks are the mask rows `R_i`, and the pair `β_T` they share is non-designated. Each of its combined masks is covered by its `R_i`'s mask.

**What the simulator does.** It samples the non-designated symbols. It then solves for the designated pairs' symbols in triangular order, the groups first and the proximity row last, and for `ChannelParams` each `R_i`'s symbols for its evaluation row. Each solve is a 2×2 system over `Fp` in the pair's two rows, with an invertible matrix. Last, it programs the leaves.

**The bound on queries.** INV-PROTO-3 and the budget bound the distinct columns: `τ_A ≤ P_A` per advice commitment, and `≤ P_W` per weight matrix and epoch.

### 4.7 Scoring

**What is public.** The scoring claims of row 9 carry `Û` and the lengths, public by design (RFC-0013's table).

**Requirement R7.** A scoring message that carries an evaluation of a secret tensor needs a masked extension, as a stage output does. RFC-0017's masked messages must be checked against this when they land.

## 5. Aborted sessions

- **Advice and masking polynomials** are fresh per `Round1`. INV-PROTO-1 discards them on abort, so two sessions' views are independent, and their distances add.
- **Weights.** The reserved blinding rows and opened columns stay burned (RFC-0012: `abort` releases nothing). An aborted session reveals a prefix of a complete proof's items, so §6 counts it as one proof.
- **Requirement R6.** The budget never rolls back.
  - **Why.** A reused reservation leaks.
    - **`β` alone reused.** It stays masked by the fresh `R_W` rows.
    - **`β` and `R_W` both reused.** `u_T(i) − u_T(i') = θ_i·F_i − θ_{i'}·F_{i'} + (θ_i·c_i − θ_{i'}·c_{i'})·R_W` is still masked by `R_W`. But the verifier can solve it for `R_W` and substitute into `W'(p_i) = W̃(p_i) + Z(p_i)·R̃_W(p_col)`. That gives a public linear relation between `W̃(p_i)` and `⟨F_{i'}, e_i⟩`, both secret.
  - **What enforces it.** RFC-0012 persists each transition before the message it allows. A crash-recovery path ([#156](https://github.com/AbdelStark/tessera/issues/156)) must resume from the persisted state.

## 6. One epoch

The weights, their encoding masks and their blinding rows persist for an epoch: `B_W` per matrix, and `ChannelParams`' `B_CP` ([RFC-0065](../rfcs/RFC-0065-masked-channel-params.md)). Verifiers may pool their views and choose which sessions complete.

### 6.1 Columns

At most `P_W` distinct columns per matrix are opened per epoch, linking proofs included, so every non-designated row's symbols at all of them are jointly uniform (Lemma 3).

A weight column answer carries every blinding row (RFC-0012 'Opening by the folding pass'). So a column opened by an earlier proof also fixes the symbols of later proofs' designated pairs, and every later combined row must satisfy its column check at every column `j ∈ J` opened before. The verifier can check that check whether or not the later proof queries `j`.

The epoch simulator therefore samples each later combined row and mask subject to these relations, at most `P_W` of them. It solves for the designated pairs' symbols at new columns as in Lemma 6.

### 6.2 Combined rows across proofs

**What each proof reveals, per matrix.**
- **Its rows.** Its evaluation row and its proximity row, with their masks. `ChannelParams` has one evaluation row per claim (RFC-0052).
- **Their designated blocks.** `β(i)` enters only its evaluation row, with `(1, X)`. For `ChannelParams`, each evaluation row's designated block is its claim's mask row `R_j(i)`, and `β(i)`, shared by them, carries the blinding claims (RFC-0065). The fixed word `F(i)` enters its own proximity row with `(1, X)`.
- **Every other blinding row**, other proofs' pairs included, enters proximity row `j` with the generator's coefficient.

**Requirement R2.** The generator's coefficients on blinding rows are independent and uniform over `Fp2`, as Draft RFC-0021's free-row stream draws them. The alternative is to exclude those rows from the proximity combination, with its own soundness argument (RFC-0021, #394).

**Lemma 7 (epoch rank).** Under R2, the joint view of an epoch's combined rows and masks is uniform subject to their checks. That includes §6.1's relations at earlier columns. The exception has probability `≤ C(N, 2)·p^{-4}` per matrix, about `2^{-238}` at `N = 681`.
- **The basis.**
  - Write each combined row and mask as an element of `Fp2[X]_{<k+P}`: message and mask together, as the codeword's polynomial.
  - The opened columns `J` grow adaptively with the queries. So the argument runs sequentially: each message is uniform given everything before it, in a basis that splits off the evaluations at the columns opened so far.
  - A combination acts on every coordinate of the remaining part with the same coefficient matrix, so one rank event covers them all.
- **Clearing the evaluation pairs.** Eliminate the `β` columns with the evaluation rows, or for `ChannelParams` the `R_j` columns. The proximity rows keep, over their `2N` `Fp` components, a block `I + R` on the fixed-word columns:
  - `I` is the identity of each row's own `(1, X)`;
  - `R` has zero diagonal blocks and uniform off-diagonal 2×2 blocks;
  - a uniform block `U` sits on the mask rows' columns `R_W`, at least `2N` of them, or for `ChannelParams` on the `β` columns, `2N` of them.
- **Counting dependencies.** A non-zero `v` with `v^T·(I + R) = 0` and `v^T·U = 0` is needed for a failure.
  - If `v` has one non-zero block, its own block is not zero.
  - Otherwise every column block holds a uniform term and vanishes with probability `p^{-2N}`, and `v^T·U = 0` with probability `p^{-2N}`.
  - Over the `p^{2N}` vectors, that gives `p^{-2N}`.
- **Adaptivity.** The verifier knows the coefficients before a session's later messages, so colluding verifiers may choose which sessions complete. A union over the completed subsets gives `Σ_{n ≥ 2} C(N, n)·p^{-2n}`, dominated by `C(N, 2)·p^{-4}`.
- **The scalars.** `W'(p_i)` and `q_T(i)` stay uniform.
  - **The common case.** When `I + R` is invertible, the designated pairs cover the vector views, and the scalars are covered as in §4.5.
  - **When `I + R` is singular** (probability about `1/p`, for `N = 2` when `det(I − BA) = 0`), `U` covers the vectors, and `R_W` also carries `W'(p_i)`. The joint claim still holds. A scalar functional `⟨R_W, e_i⟩` lies in the vectors' row span only if `e_i[pos']/e_i[pos] ∈ Fp` for every pair of positions in the support of its `eq` vector `e_i`. That has probability about `p^{-(column bits)}`, independently of the rank event.

So `ε_W ≤ 10·C(N, 2)·p^{-4} ≤ 2^{-230}` over the ten matrices at `N = 681`.

### 6.3 The tensor generator

**Why Lemma 7 fails without R2.** With RFC-0012's and RFC-0049's generator, `eq(ξ*_T, row)` over data and blinding rows together, the blocks of `R` are not uniform. They are values of degree-`ϑ_T` polynomials in the `Fp`-coordinates of each `ξ*_T`, with `ϑ_T ≤ 18` (RFC-0014). So Lemma 7's count fails.

**The general bound.** Schwartz–Zippel must run over `Fp`, since each proximity row and its conjugate depend on those coordinates. It gives `≤ 2Nϑ_T/p`: about `2^{-49}` at `N = 681` and `ϑ_T = 18`, far above `2^{-100}`.

**What is not known.** A sharper argument for tensor-structured coefficients may exist; none is known here. Within one proof the free advice rows save the tensor generator (§4.4). Across an epoch, the weights' only free rows are the blinding rows, which every proof's combination shares. Until such an argument exists, masked weights need R2.

### 6.4 Linking proofs

**The leak.** RFC-0012 'Refresh and linking proof' opens both epochs' commitments at a random `ρ_T` with blinded combined rows, no mask row, and the verifier comparing the values. In `tessera-protocol/2` that reveals `W̃_T(ρ_T)`, a random linear functional of the secret weights, per matrix and per refresh.

**Requirement R3.** Open the difference.
- **The rows.** The evaluation row is `Σ_row eq(ρ_row, row)·(A^{old}_row − A^{new}_row)`, with a blinding pair from each epoch's reserved rows. It is checked at the same columns of both commitments against the public claim `0`.
- **Each commitment's proximity row.** Each keeps its own, with its own fixed word, so that each commitment is still proven close to the code. The blinding claims are absorbed before `θ`.
- **Simulation.** The view is then Lemma 5's, and §6.1's relations apply to it.
- **Soundness.** It is unchanged: two different multilinear polynomials over `ν` variables agree at `ρ_T` with probability `≤ ν/q`.

**A text inconsistency.** RFC-0012's linking proof reserves four blinding rows at each end of an epoch, but its 'Failure modes' says two. R3's rows must be counted once.

**Met.** [RFC-0064](../rfcs/RFC-0064-weight-linking-proof.md) specifies the difference opening as above. Its seeds come from a model key by `weights-refresh`, so the PRG hybrid replaces each epoch's seed in turn and `ε_PRG` counts every epoch a view spans. The difference row takes a blinding pair from each epoch: rows `B_W − 4` and `B_W − 3` of epoch `e`, and 0 and 1 of `e + 1`. The proximity rows take the next pair at each end. Those are the four rows at each end, counted once, and RFC-0012 'Failure modes' now says four.

## 7. OQ-0013-1: packing small mask rows

Packing several masks in one row, each in its own aligned sub-block at a public offset `o`, behaves differently by mask kind:
- **Tree levels' masks: admissible.** These are the virtual masks OQ-0013-1 asks about. Their only scalar claim is `R̃_t(r'_t)`, which stands alone.
  - **Independence.** Each block's functionals are supported on its own block. Within a block, Lemma 2 applies to its own points.
  - **The cost.** The opening functional is `eq(offset ‖ p_col)`, a column sub-point of its own, which no RFC-0011 group would share. [RFC-0070](../rfcs/RFC-0070-mask-batching.md) packs them in the masks' block `M` and opens their claims with the masking polynomials' in one batching sumcheck (§4.3a), so packing costs no group. It saves mask rows: at 405B, 2,621 tree-level masks, the leaf mask and the coefficients fit in 780 rows.
- **The `SoftmaxRow` leaf mask: admissible.** RFC-0030's fold reads only `R_h[x_col]` over its `c_h` column variables (RFC-0013 'Virtual tensors' exempts it from #538's rule), so its functionals lie on its own block. At 405B `c_h = 22`, a whole row of `M`.
- **Reduction stages' #538 fold masks: not admissible.** Such a fold reads its mask's whole row (RFC-0013 'Virtual tensors', #538), every block of a packed row included, and the row must be random in all its entries, so these masks are not packed.
- **Committed tensors' masks: not admissible, unless the tensor's data sits at the mask's column offset.**
  - **Why.** `V'(p)` can be opened as one claim only because both terms have column part `eq(p_col)` (RFC-0013 'Masked extensions', committed tensors; RFC-0011 step 1 requires a combination's pieces to share one column sub-point). A mask packed at `o ≠ 0` has column part `eq(o ‖ p_col)`.
  - **The leak.** The claim then splits across two groups, and the two revealed values give `Ṽ(p)`.
- **#162's distinguisher suite** must include a packed row of each kind, the committed one as the negative control.

## 8. Error bound

`ε_1` sums, in units of `1/q`:
- **Masked sumchecks.** One term per stage, tree level, identity sumcheck and the masks' batching: `ρ = 0`.
- **Revealed masked points.** `2(n − 1)` per point: `Z = 0`.
- **Mask rows.** `e_V^2` per tensor mask row, and 9 per tree-level mask: dependent `eq` vectors.
- **Proximity rows.** About `p^{-2}` each, under the tensor generator (Lemma 5).

**At the 405B default.** EXP-0027 counts the tree levels and the masked sumchecks from the compiler's schedule; the tensors' figures are still RFC-0013's projections.
- **The counts.**
  - 2,622 masks of `M`, the 2,621 tree levels' and the `SoftmaxRow` leaf's, each revealed at its masked points and by its claim;
  - 46 masked sumchecks, and the masks' batching;
  - about 5,000 masked points of the tensors, and about 2,000 tensor mask rows with `e_V ≤ 4`.
- **The `Z` terms.** These masks are revealed at 5,144 masked points: two per level below a root and two for the leaf mask, at most `2(n − 1) = 82` each, and one per root level, 2 each. They sum to 149,236. With the tensors' `≈ 450,000`, that gives `≈ 6.0·10^5/q ≈ 2^{-108.8}`.
- **The `eq` terms.** `9 × 2,622` and the tensors' `≈ 32,000` give `≈ 5.6·10^4/q ≈ 2^{-112.2}`.
- **The masked sumchecks.** 47 terms, negligible beside these.

So `ε_1 ≈ 6.5·10^5/q ≈ 2^{-108.7} < 2^{-105}`, given R10's rank checks. The tensors' counts must still be recomputed from the compiler's schedule once `tessera-protocol/2`'s layout is compiled, RFC-0068's column masks and layer sumcheck included.

**Outside `ε_1`.** `ε_PRG` and `ε_H` are separate. Per epoch, `ε_W ≤ 2^{-230}` adds to the proofs' terms.

## 9. Requirements on the specification

| | Requirement | Where | Status |
|---|---|---|---|
| R1 | The proximity row's blinding pair is the fixed word, with coefficients `1` and `X` and no generator coefficient. This is soundness's choice, not zero knowledge's, which holds either way (§4.4) | RFC-0011 step 3, RFC-0012, RFC-0049 for `tessera-protocol/2`, against RFC-0013 'The proximity row' | Conflicting text; #394 |
| R2 | For masked weights, the proximity generator's coefficients on blinding rows are independent uniform `Fp2`, or those rows are excluded with their own soundness argument | RFC-0012, RFC-0049; Draft RFC-0021's free-row stream meets it | Not met by the tensor generator; #394 |
| R3 | The linking proof opens the difference of the two commitments, with the public claim 0, each keeping its own proximity row | [RFC-0064](../rfcs/RFC-0064-weight-linking-proof.md), superseding RFC-0012 'Refresh and linking proof' | Met in the specification by RFC-0064 ([#485](https://github.com/AbdelStark/tessera/issues/485)); [#153](https://github.com/AbdelStark/tessera/issues/153) implements it, with R3's test |
| R4 | Each masking polynomial's `g_j` has its sumcheck's message degree `d`, at least every round's, a raised round's included | RFC-0013 'Masked sumcheck' | Stated by this change |
| R5 | INV-ZK-2 reads as a rank condition. Each mask, blinding row and masking polynomial has a designated revealed combination where it enters with an invertible block, and the joint coefficient matrix over the masks has full rank. Further uses are allowed where Lemmas 4a, 5 and 7 cover them. The masks' batching is such a use: the masking polynomials, the tree levels' masks and the `SoftmaxRow` leaf mask enter it only through `M + ρ·S`, which the pad `S` covers, so its functionals need not have full rank (Lemma 4a) | RFC-0013 'Invariants' | Literal one-use conflicts with the proximity row; #394 |
| R6 | The weight budget never rolls back, and an aborted session burns what it reserved | RFC-0012 'Budget state machine' | Met by RFC-0012; #156 must preserve it |
| R7 | A scoring message that carries a secret evaluation has a masked extension | RFC-0017 | To check when its masked messages land |
| R8 | No masked tensor is revealed at a point where its own `Z` vanishes, nor alone in a sum all of whose terms are such points. RFC-0045's per-layer split values violated it, and so did the LogUp columns, sums of such terms | RFC-0045, RFC-0037 'Splits' and 'The claim pool', RFC-0013 'Masked extensions', [RFC-0068](../rfcs/RFC-0068-leaf-stage-zero-knowledge.md), [RFC-0069](../rfcs/RFC-0069-one-layer-claims.md) | Met in the specification for the leaf stage by RFC-0068 ([#486](https://github.com/AbdelStark/tessera/issues/486)): column masks and a layer sumcheck. Met for claims at one layer, the epilogue's on the last checkpoint, by [RFC-0069](../rfcs/RFC-0069-one-layer-claims.md) ([#535](https://github.com/AbdelStark/tessera/issues/535)). [#148](https://github.com/AbdelStark/tessera/issues/148) and [#154](https://github.com/AbdelStark/tessera/issues/154) implement it, and [#162](https://github.com/AbdelStark/tessera/issues/162) tests it |
| R9 | `ChannelParams` claims take masks over the type tensor's own coordinates, a mask row per claim, which is its evaluation row's designated block, and one pair per matrix for the blinding claims: `2e + 4` rows per proof. `ChannelParams`' own blinding rows `B_CP` keep its epoch as long as the others' | [RFC-0065](../rfcs/RFC-0065-masked-channel-params.md), superseding RFC-0052's `tessera-protocol/2` gap and RFC-0012's six-row `reserve` | Met in the specification by RFC-0065 ([#487](https://github.com/AbdelStark/tessera/issues/487)), for sites of two classes or more with RFC-0068's leaf stage ([#486](https://github.com/AbdelStark/tessera/issues/486)). [#147](https://github.com/AbdelStark/tessera/issues/147) implements it, and [#162](https://github.com/AbdelStark/tessera/issues/162) tests it |
| R10 | Every mask row's revealed points, as the compiled schedule places them, have linearly independent `eq` vectors, apart from dependencies implied by public relations among the revealed values, such as `e'(λ ‖ r')`'s. For a weight matrix whose evaluation rows share a pair `β_T` ([RFC-0065](../rfcs/RFC-0065-masked-channel-params.md)), the check runs per matrix, over every functional of `β_T` together. A value that carries a secret term (`S ≢ 0`) must lie outside the span of the others. One that carries none (`S ≡ 0`, decided symbolically) adds its functional to the span but is not refused: two claims at one column point give equal blinding claims, such as `q_O = q_Down` for O's and Down's at the leaf stage's shared point at 405B, or Q's and O's multipliers at stage 1's final point in `tiny-2-affine`, and are valid. The points depend on challenges, so the compiler checks the generic (symbolic) rank and Schwartz–Zippel bounds the rest. Softmax row sums at `1/2` and shared column sub-points are included. The masks' batching's final point `r*` is not a revealed point of any mask in `M`: `y` reveals `z̃(r*)`, which `S` masks ([RFC-0070](../rfcs/RFC-0070-mask-batching.md)) | the compiler's schedule (RFC-0006, RFC-0037) and the verifier's derivation | Checked in part: `cargo xtask check-mask-rank` checks the compiled schedule's pool atoms, virtual tensors' values and folds, and every matrix's `β_T`. It runs each claim list of `spec/vectors/` and the 405B claim list at every `⌈log2 n⌉`, at `tessera-protocol/2`'s padded token axis, under both readings of [#554](https://github.com/AbdelStark/tessera/issues/554) ([#488](https://github.com/AbdelStark/tessera/issues/488)). No stage reveals a weight's value today, so the `β_T` pass guards future schedules. Not yet checked: the leaf stage's `tessera-protocol/2` points (RFC-0068), gadget outputs' values at their leaf points, `SoftmaxRow`'s reciprocal and `Exp` reads and its row sums at `1/2`, gadget auxiliary columns, multiplicities, tree-level and batching masks, native raw openings (RFC-0053's greedy reads, which disclaim `tessera-protocol/2`), batches (RFC-0071), and committed groups whose layer weights carry per-layer constants, whose mask no document defines ([#556](https://github.com/AbdelStark/tessera/issues/556)). #148 and #154 bring most of these points |
| R11 | Each masked sumcheck's `G(r)`, each tree level's mask claim and the `SoftmaxRow` leaf mask's is opened. Either way, the budget counts it: through its own group, with a blinding pair and a combined row, or through a batching sumcheck that reduces them to shared column points | RFC-0013 'Masked sumcheck', RFC-0011 'Opening protocol', RFC-0032 | Met in the specification by [RFC-0070](../rfcs/RFC-0070-mask-batching.md) ([#489](https://github.com/AbdelStark/tessera/issues/489)): one masked batching sumcheck, Lemma 4a, one group. [#148](https://github.com/AbdelStark/tessera/issues/148), [#149](https://github.com/AbdelStark/tessera/issues/149) and [#159](https://github.com/AbdelStark/tessera/issues/159) implement it, and [#162](https://github.com/AbdelStark/tessera/issues/162) tests it |

## 10. Testing

These tests check the implementation of the masks, not this argument (07 §7). Each mechanism maps to one in #162's distinguisher suite:
- **Lemma 4.** Round messages' non-constant coefficients, and `Γ`, are uniform for two witnesses.
- **Lemma 4a.** The masks' batching's view is identical for two witnesses whose earlier views agree. Its negative controls, each told apart: no pad; `M̃(r*)` sent alone on a layout with no spare direction; `S` on `C`'s support with a block no claim reads; and `S` drawn over `Fp`, on a layout of fewer than `4m` non-public positions, beyond which an `Fp` pad leaves no direction uncovered.
- **Lemma 2.** Each revealed masked value is uniform for two witnesses. With an `Fp` mask, a distinguisher on the value's line wins.
- **R8 and R9.** For two witnesses, every revealed evaluation has the same distribution. As the negative control, `tessera-protocol/1`'s per-layer split values and `ChannelParams` claims are told apart, and so are RFC-0068's column values with the column masks set to zero, and RFC-0069's claims at one layer with their rows set to zero.
- **Lemma 5.** Combined rows are uniform for two witnesses, under the tensor generator and under the fixed word.
- **Lemma 6.** Opened symbols are uniform for two witnesses, up to `P` columns. At `P + 1` columns of one row they are not, which is the negative control.
- **Lemma 7 and R2.** Over a small field, the joint combined rows of many proofs of one epoch are uniform, the relations at earlier columns included, with free coefficients. The tensor generator's rank failures are measured against them.
- **R3.** A linking proof's view is identical for two weight sets that differ at `ρ_T`, with the random oracle programmed to give both the same `ρ_T` ([RFC-0064](../rfcs/RFC-0064-weight-linking-proof.md#zero-knowledge)). RFC-0012's openings reveal the two different `W̃_T(ρ_T)`, which is the negative control.

## 11. Review record

An independent adversarial reading of the first revision found:
- **Two outright leaks.** The leaf stage's per-layer values, R8, and `ChannelParams`' claims, R9.
- **Errors in the first revision:**
  - the epoch simulator ignored the relations at earlier columns (§6.1);
  - the packing answer was wrong for committed tensors (§7);
  - Lemma 7 lacked its scalar and adaptivity steps;
  - R1 was misclassified (§4.4);
  - R6's formula was incomplete;
  - several constants were off: `1/ρ` in Lemma 4, and `ϑ_T` and the `2^{-49}` of §6.3.

It confirmed Lemmas 2, 4 and 5, the union arithmetic of Lemma 7, and R2 and R3.

A second reading of the rewrite confirmed the fixes to the epoch columns, packing, Lemma 7, R1's classification, R2 and R3, and the small corrections. It found:
- **Errors in the rewrite:**
  - R6's leak formula was wrong;
  - §4.4 counted the masking polynomials' rows as free;
  - R8 was stated too literally;
  - R9's row count was wrong, from the first reading's own figure.
- **Gaps it added to the requirements:**
  - the LogUp leaf level and `logup/values` (§4.3);
  - Lemma 2's premise on structured points (R10);
  - the opening of `G(r)` and of the tree levels' mask claims (R11).

A third, focused check of the edited passages found no leak. It found nine statements to correct:
- Lemma 2.2's count of rows;
- §4.4's failure argument, which now uses `Q_ξ`'s unmodified pair;
- the masking polynomials' spare constants;
- what hiding needs under R11's batching option;
- R10's allowance for public dependencies, and its symbolic check;
- the LogUp leaf level's masks, and `m'`;
- R8's wording;
- a leftover sentence in Lemma 7;
- the table's order.

This revision applies all of it. A maintainer's review is still required.

## References

- Chiesa, Forbes, Spooner, "A zero knowledge sumcheck and its applications", 2017.
- Xie, Zhang, Zhang, Papamanthou, Song, "Libra", CRYPTO 2019.
- Ames, Hazay, Ishai, Venkitasubramaniam, "Ligero", CCS 2017.
- [RFC-0013](../rfcs/RFC-0013-zero-knowledge.md), [RFC-0032](../rfcs/RFC-0032-zero-knowledge-tree-levels.md), [RFC-0011](../rfcs/RFC-0011-advice-commitment.md), [RFC-0012](../rfcs/RFC-0012-weight-commitment.md), [RFC-0014](../rfcs/RFC-0014-interaction-and-soundness.md), [RFC-0037](../rfcs/RFC-0037-stage-schedule.md), [RFC-0045](../rfcs/RFC-0045-leaf-layout.md), [RFC-0052](../rfcs/RFC-0052-channel-params-claims.md), [Draft RFC-0021](../rfcs/RFC-0021-johnson-regime-proximity.md).
