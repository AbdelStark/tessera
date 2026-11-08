# IDEA-0044: Keep the tensor proximity generator for masked weights by proving its epoch rank directly

| | |
|---|---|
| Status | `validated` |
| Origin | [#394](https://github.com/AbdelStark/tessera/issues/394); [simulator argument](../../docs/zk/simulator.md) §6.3 and requirement R2; [#145](https://github.com/AbdelStark/tessera/issues/145) |
| Lever | zero knowledge across a weight epoch (G-5); `regen` (the proximity row's PRG draws, [RFC-0075](../../docs/rfcs/RFC-0075-weight-mask-cost.md)); no Δμ credited |
| Spec surfaces | None if it holds: RFC-0011 step 3, RFC-0012 and RFC-0049 keep `eq(ξ*_T, row)` over data and blinding rows, and the simulator's Lemma 7 gains a proof for that generator. R2 then becomes one sufficient condition, not a requirement |
| Priority | correctness prerequisite: Theorem 2 of the simulator argument, and IDEA-0043's context commitments, need an epoch argument under some generator |

## Mechanism

Nothing in the protocol changes. The idea is an argument.

The simulator argument's Lemma 7 shows that an epoch's combined weight rows hide the weights when the proximity generator gives the blinding rows independent uniform `Fp2` coefficients (R2). Under the generator the specification has, `eq(ξ*_T, row)` over every committed row, its only bound is Schwartz–Zippel over `Fp` on one determinant: `2Nϑ_T/p`, about `2^{-49}` at `N = 681` and `ϑ_T = 18`. That is far above the `2^{-100}` target. §6.3 says that a sharper argument for tensor coefficients may exist.

The sharper argument would use more of the epoch than §6.3 does.
- **Spare blinding rows.** §6.3's determinant uses only the proximity pairs `F(i)`. But each proximity row reaches every blinding row of the epoch: the mask rows `R_W`, the unreserved rows and the link rows. An epoch has `B_W = 6N + 8` of them against `4N` revealed `Fp` vectors.
- **Rank instead of one determinant.** The view leaks only if the revealed vectors' blinding coefficients lose rank: some non-zero `Fp`-combination of them has no blinding term. With `2N + 8` spare columns, the event needs many polynomial conditions at once. For `eq` vectors, all of them come down to ratios falling into `Fp`.
- **Sequential independence.** Each `ξ*_i` is a random-oracle output drawn after the earlier sessions' views are fixed. The seed is uniform from `V*`'s view in Live sessions too (§4.1). So the event can be bounded one session at a time, against the span of the earlier sessions' vectors, with no union over which sessions complete.

If the event has probability `p^{-e}` per session with `e ≥ 2`, then the epoch term is at most `N·p^{-e}`. That is `2^{-118}` at `N = 681` and `e = 2`, and the tensor generator then needs no R2.

## Why it could work

- Lemma 2.2 of the simulator argument already finds the pattern for one row: a pair of `eq` coefficients falls on one `Fp`-line with probability about `1/p`, but three rows that differ in two bits need two coordinates in `Fp`, about `p^{-2}` (E0, primary: the argument's own lemma).
- The obvious structured failure, every coordinate of `ξ*` in `Fp`, makes each `eq` vector real and its pair rank one. Over the `a` row bits that vary across the blinding block, it has probability about `p^{-a}` (E0).
- §6.3's `2^{-49}` comes from one `2N × 2N` determinant, which ignores the `2N + 8` spare columns. A rank bound over a wider matrix is usually far smaller than any one of its minors' (E0).

None of this is a proof. The cheapest test is to measure the event's rate over small primes and read off its exponent.

## Expected gain

There is no Δμ. Two things change if it holds:
- **Theorem 2 is proved without R2.** The simulator argument's epoch theorem would hold for the generator that RFC-0011, RFC-0012 and RFC-0049 specify.
- **The R2 decision gets one fewer reason.** #394's choice between R2 and RFC-0021's free-row stream would no longer be forced by zero knowledge. [SUR-0001](../surveys/SUR-0001-proximity.md) records an independent soundness reason for uniform coefficients: the Johnson-regime bounds.

On the `regen` side, RFC-0075 already accounts for full-support regeneration, with no blinder-exclusion credit, so no cost change is credited here.

## Security gate

| Property | Effect | Pass |
|---|---|---|
| Soundness bits (G-4, RFC-0014) | None: the generator is unchanged | pass |
| Uniqueness upstream of the logits (SP-3, RFC-0009 Lemma 0) | None | pass |
| Zero knowledge (G-5, RFC-0013) | The idea is a ZK argument. A proof of `P[D_N] ≤ 6Nϑ/p^2`, about `2^{-111.8}` at 405B, is written below; until a reviewer confirms it, the tensor generator has only §6.3's bound | unknown |
| Verifier independence (SP-6) | None | pass |
| Post-quantum (hash-based commitments) | None | pass |

## Prior rejection check

- **Simulator argument §6.3.** It states "A sharper argument for tensor-structured coefficients may exist; none is known here", so the question is open, not rejected.
- **Draft RFC-0021.** Its free-row stream meets R2 by changing the generator. This idea keeps the generator.
- **RFC-0075.** It takes no blinder-exclusion credit, so the question has no effect on cost there.

## Kill criterion

A structured failure at rate about `1/p`: a layout where the tensor generator's revealed blinding coefficients lose rank with probability that scales as `p^{-1}`. That would show §6.3's bound is essentially tight, and R2 (or exclusion with its own soundness argument) necessary.

## Cheapest experiment

[EXP-0033](../experiments/EXP-0033-tensor-generator-epoch-rank/README.md): exact rank counts over `Fp2 = Fp[X]/(X^2 + 1)` for eleven primes from 7 to 127, on RFC-0012's and RFC-0065's blinding-row layout. Each run compares the tensor generator with and without R1 against uniform coefficients (R2) in the same harness, and fits the failure exponent. E0 (exact arithmetic), effort `s`. A measured exponent is not a proof, and toy sizes are not 405B's. If the data supports the idea, the next unit is the lemma itself.

## Result

[EXP-0033](../experiments/EXP-0033-tensor-generator-epoch-rank/README.md) is `done: validated` (E0). Over 264 cells of `10^6` exact trials:
- **G-T.** The tensor generator's rank loss falls as `p^{-2}` or faster in every layout, with fitted exponents 2.03 to 4.53.
- **Where `p^{-2}` comes from.** The one `p^{-2}` row is the exact `1/q` event of a row coordinate equal to 1 where the blinding block straddles a power of two. At 405B that is about `2^{-114}` per matrix and epoch.
- **G-T1.** Under R1 there were 2 failures in `8.8×10^7` trials, both at `p = 7`.
- **G-U.** Uniform coefficients had none.

So §6.3's `2Nϑ/p` does not describe the event. The zero-knowledge gate stays `unknown` until the lemma is written and reviewed.

**Amended by [EXP-0035](../experiments/EXP-0035-epoch-rank-slope/README.md)** (`done: inconclusive`, E0). At `p = 7` the epoch rate grows faster than linearly in the number of sessions: slope 1.65, 95% interval `[1.36, 2.18]`. A fixed per-session rate summed over the epoch is therefore not supported. The lemma needs a term that grows with the span of earlier sessions, and whether that term scales as `1/q` or `1/p` decides between 'near target at 405B' and 'R2 needed'. Until that is measured, R2 is the safe choice.

**Further amended by [EXP-0036](../experiments/EXP-0036-epoch-rank-p-scaling/README.md)** (`done: inconclusive` as registered, E0). The span term falls far faster than `1/q`. From `p = 7` to `11` the rate drops 13 to 22 times. At `p = 19` there are no failures in any cell, against 54 to 87 expected under `1/q`. So the faster-than-linear growth is a small-field effect of high order in `1/p`, a post-hoc reading. A per-session bound remains plausible, but only an analytic lemma settles it. R2 stays the safe choice until then.

## Partial argument: one event for the whole epoch

This is proved here, as an E0 derivation, but not reviewed.

**Setting.** It is the vector view of [EXP-0033](../experiments/EXP-0033-tensor-generator-epoch-rank/README.md)'s model.
- **The rows `Q*`.** These are the blinding rows that are neither an evaluation pair `β_i` nor a proximity pair `F_i`: the mask rows `R_W`, the unreserved rows and the link rows. There are `|Q*| = B − 4N = 2N + 8` of them.
- **The vectors `e(x)`.** Write `e(x) = (eq(x, Dn + r))_{r ∈ Q*} ∈ Fp2^{Q*}`, and let `x̄` be the conjugate of `x`, coordinatewise. `eq` has `Fp` coefficients, so `conj(e(x)) = e(x̄)`.

**Claim.** Let `bad_j` be the event that session `j`'s two revealed `Fp` vectors are dependent on the earlier sessions' vectors, including the evaluation rows. `bad_j` is EXP-0033's event E first occurring at session `j`. Then, with or without R1,

`bad_j ⊆ D_N := { the 2N vectors e(ξ_1), e(ξ̄_1), …, e(ξ_N), e(ξ̄_N) are Fp2-linearly dependent }`, read as a list counted with multiplicity, so a real `ξ_j` or a repeated point counts as dependent.

So `P[E] ≤ P[D_N]`, with no union over sessions and no dependence on which sessions complete.

**Proof.**
1. **Restrict to `Q*`.** A dependency that involves session `j` gives `u ≠ 0` in `Fp^2`, and a vector `v` in the earlier sessions' span, with `v = φ_u ∘ c_j`, where `c_j` is session `j`'s proximity coefficients and `φ_u(z) = u_1·Re z + u_2·Im z`. Restrict both sides to `Q*`. The evaluation rows vanish there. No `F` column is in `Q*`, so neither R1's `(1, X)` nor any other session's own pair appears.
2. **Write both sides as real parts.** Every non-zero `Fp`-functional on `Fp2` is `Re(θ·)` for a unique `θ ≠ 0`. The left side is `Re(θ·e(ξ_j))` with `θ ≠ 0`. The right side is `Re(Σ_{i<j} θ_i·e(ξ_i))`.
3. **Remove the real part.** `Re z = (z + z̄)/2` for odd `p`. So the equality becomes `θ·e(ξ_j) + θ̄·e(ξ̄_j) − Σ_{i<j} (θ_i·e(ξ_i) + θ̄_i·e(ξ̄_i)) = 0`. That is an `Fp2`-linear relation with non-zero coefficients on `e(ξ_j)` and `e(ξ̄_j)`. ∎

**A numerical check.** It is an uncommitted harness built on EXP-0033's code, recorded here only as a check of the algebra. Over six configurations it ran `1.2×10^6` trials: G-T and G-T1, `N` from 1 to 16, `p ∈ {7, 11}`. Every one of the 6,498 trials in E was also in `D_N`. `D_N` occurred 7,504 times in all, so for G-T the bound is close to the measured event.

**What remains.** Bound `P[D_N]`: the chance that `2N` conjugate-paired `eq` vectors at uniform points are dependent on `2N + 8` fixed Boolean rows. This statement no longer mentions sessions, generators or adaptivity. The next section proves `P[D_N] ≤ 6Nϑ/p^2` (unreviewed); the items below record why a direct determinant argument fell short.
- **What it contains.** `D_N` contains EXP-0033's exact `1/q` straddle event, so the lemma must carry that term.
- **Why a direct proof falls short.** A sequential proof needs each pair `(ξ_i, ξ̄_i)` to impose two independent conditions on the multilinear functions supported on `Q*` that vanish at the earlier points. That space has dimension at least 10. A single 2×2 determinant gives only Schwartz–Zippel's `2ϑ/p` over `Fp`, because `ξ̄` is not independent of `ξ`.
- **What would finish it.** A codimension-2 argument for the conjugate pair over that space. [EXP-0036](../experiments/EXP-0036-epoch-rank-p-scaling/README.md)'s rapid decay with `p` suggests that one exists.

## Correction: the evaluation rows touch `Q*`

Step 1 of the partial argument says that the evaluation rows vanish on `Q*`. That holds in EXP-0033's model, whose evaluation row is `(1, X)` on `β` alone. It is false for the protocol. The weight evaluation row is `u_T = θ_T·(F + c·R_W) + β_T` (simulator argument §5; RFC-0065 step 3). It puts the coefficient `θ_T·c` on the mask rows `R_W`, which are in `Q*`. `ChannelParams`' rows put `θ_T·Z(x_i)` on each claim's `R_i`. The reviewer had marked this as unchecked. The inclusion `bad_j ⊆ D_N` is therefore unproved as stated, and EXP-0033 to EXP-0036 measured the simplified model.

**The corrected reduction.**
1. **A global quotient replaces the restriction.**
   - **The evaluation vectors.** Write `c'_i = θ_i·c_i`, or `θ_i·Z(x_i)` per claim for `ChannelParams`. Session `i`'s evaluation row has blinding vector `w_i = (1, X)` on `β(i)` plus `c'_i·(1, X)` on `R(i)`.
   - **The map `Ψ`.** On each session's block `(β(i), R(i))`, take the quotient by `span{w_i, w̄_i}`, and drop every `F` pair. The span is conjugation-stable, so the quotient map is real.
     - For a weight matrix, `(r_0, r_1) − M_i·(b_0, b_1)` does it, with `M_i ∈ Fp^{2×2}` the unique matrix sending `(1, X)` to `c'_i·(1, X)`. It is real because it equals its conjugate.
     - For `ChannelParams`, the block has `2 + 2e` rows and `e` evaluation vectors, and the quotient still leaves 2 coordinates.
   - **The target.** The spare rows (unreserved and link) map to themselves, so `Ψ` maps onto `Fp^{2N + s}`, with `s ≥ 8` spare rows at full budget (RFC-0012 `B_T − 8`, RFC-0065 `B_CP − 8`).
2. **The inclusion.**
   - **Apply `Ψ`.** It kills every `w_i` and `w̄_i`, and the dropped `F` columns remove R1's designated entries. So a dependency among all the revealed vectors becomes `Σ_i (θ_i·G(ξ_i) + θ̄_i·conj(G(ξ_i))) = 0`, with `G = Ψ ∘ e`.
   - **It is non-trivial.** If every `θ_i = 0`, the evaluation rows alone are dependent. But they sit on disjoint `β` blocks, with `(1, X)` and `(1, −X)` independent, so that is impossible.
   - So `E ⊆ D_N^Ψ := { G(ξ_1), conj(G(ξ_1)), …, G(ξ_N), conj(G(ξ_N)) are Fp2-dependent }`.
3. **Why `Ψ` may depend on challenges drawn later.** `θ_i` is a random-oracle output drawn after `ξ_i`. But each `ξ_i`, and each `θ_i`, is uniform given everything drawn before it. `Z(x_i) ≠ 0` is enforced, since the prover aborts otherwise and the reservation stays burned (RFC-0065). So the challenges are jointly independent and uniform, adaptive choices in between included.
   - **Condition on the `θ`s first.** `Ψ` is then fixed, and the `ξ_i` stay independent and uniform.
   - **A slot whose `θ_i` is never drawn.** That slot gets a lazily sampled `θ_i` all the same, which is never revealed.
4. **The theorem below applies with `G` in place of `e`, unchanged.** Its proof used only three facts. `G` is multilinear with real coefficients, as `Σ_r eq(ξ, r)·Ψ(1_r)`. `G`'s coordinates are `Fp`-independent, since `Ψ` is onto and the `eq(·, r)` are independent. And the target has dimension `2N + s ≥ 2N + 3`. `Q*` now means this quotient, and the bound covers `ChannelParams` as well.

**A numerical check of the corrected inclusion.** This is an uncommitted harness. It used the protocol's row shapes: `β` pair, `R` pair carrying `c'·(1, X)`, `F` pair, 8 spare rows, `ϑ = 6`, `Dn = 3`, and `w = −1` at `p ≡ 3 (mod 4)`.
- **The inclusion held.** In 140,000 trials over `p ∈ {3, 7, 11}`, `N ∈ {1, …, 4}`, with and without R1, every trial in E was in `D_N^Ψ`.
- **For example.** At `p = 7`, `N = 2` and no R1, E occurred 847 times and `D_N^Ψ` 879 times.
- **What a looser map misses.** A first attempt with per-stage maps, each dropping the session's own block, missed 27 trials at `p = 3`. That is why the argument conditions on every `θ` first.

## Proof of the bound on `P[D_N]`

This is an E0 derivation. It is proved here but not reviewed. It replaces the proof-route sketch of [#709](https://github.com/AbdelStark/tessera/pull/709), and needs neither Bezout nor fibre dimensions.

**Theorem.** Let `Fp2 = Fp(√w)` with `w` a non-residue and `p` odd; Goldilocks has `w = 7` ([RFC-0002](../../docs/rfcs/RFC-0002-field-arithmetic.md)). Let `ϑ` be the number of row variables, and let `ξ_1, …, ξ_N` be independent and uniform in `Fp2^ϑ`. Then

`P[D_N] ≤ N·ε(ϑ, p)`, with `ε(ϑ, p) = (2ϑ − 1)/p^2 + 4(ϑ − 1)(p + 1)/p^3 ≤ 6ϑ/p^2` for `p ≥ ϑ`.

At `N = 681` and `ϑ_T = 18` that is `73,548/p^2`, about `2^{-111.8}`. It holds with or without R1, by the reduction above, and it contains EXP-0033's exact `1/q` straddle event.

**Notation.**
- **Conjugation.** `z̄` is conjugation (`√w ↦ −√w`), applied coordinatewise to vectors.
- **The real cone.** `R_m = Fp2*·Fp^m ∪ {0}` is the set of vectors that are real up to a scalar.
- **Dot product.** `u·v` is the bilinear dot product, never the Hermitian one.

**Lemma A (a pair is dependent only on the real cone).** `u` and `ū` are `Fp2`-dependent if and only if `u ∈ R_m`.

*Proof.* The "if" direction is clear. For "only if", let `u ≠ 0` and `ū = c·u`.
- If `c ≠ −1`, then `u + ū = (1 + c)·u` is real, so `u = (u + ū)/(1 + c)`.
- If `c = −1`, then `u = √w·(u/√w)`, and `u/√w` is real. ∎

**Lemma B (one session).** Let `g: Fp2^ϑ → Fp2^3` be multilinear with coefficients in `Fp^3`, and let its three coordinate polynomials be `Fp`-linearly independent. For uniform `ξ`, `P[g(ξ) ∈ R_3] ≤ ε(ϑ, p)`.

*Proof.*
1. **Choose a coordinate.** For a coordinate `t`, write `g = g_0 + ξ_t·h`, with `g_0` and `h` multilinear in the other coordinates `ξ'`. Let `n = g × h = g_0 × h`, a function of `ξ'`. Some `t` has `n ≢ 0`. Suppose instead that `g × ∂_t g ≡ 0` for every `t`. Change one coordinate at a time from `ξ` to an independent `ζ`. Each step keeps `g` parallel over `Fp2(ξ, ζ)`, every intermediate vector is non-zero, because independence makes each coordinate `g_k ≢ 0`, and parallelism between non-zero vectors is transitive. So `g(ξ) ∥ g(ζ)`, and each ratio `g_j/g_k` is a constant. It lies in `Fp` because the coefficients are real, so the coordinates are `Fp`-dependent. That contradicts independence. Fix such a `t`, and condition on `ξ'`. The remaining randomness is `s = ξ_t`, uniform in `Fp2`.
2. **`n(ξ') = 0`.** Some `n_j` is a non-zero polynomial of total degree at most `2(ϑ − 1)`. Schwartz–Zippel over `Fp2` gives probability at most `2(ϑ − 1)/p^2`. Count every `s` as bad.
3. **`n(ξ') ≠ 0`.** Then `g_0` and `h` are independent, and `s ↦ [g_0 + s·h]` is injective onto the projective line `ℓ = {u : n·u = 0}`. Since `g_0 + s·h ≠ 0`, `s` is bad exactly when `[g_0 + s·h]` is a real point of `ℓ`.
   - **If `n(ξ') ∉ R_3`.** Then `ℓ̄ = {u : n̄·u = 0} ≠ ℓ`. Two real points of `ℓ` would make `ℓ` conjugation-stable, so `ℓ` has at most one real point. At most one `s` is bad: probability `1/p^2`.
   - **If `n(ξ') ∈ R_3`.** `ℓ` has `p + 1` real points, so the conditional probability is at most `(p + 1)/p^2`.
4. **The event `n(ξ') ∈ R_3` is rare.** Write `ξ' = x + √w·y` with `x, y ∈ Fp^{ϑ−1}`, and let `Δ_jk(x, y) = n_j(x + √w·y)·n_k(x − √w·y) − n_k(x + √w·y)·n_j(x − √w·y)`. Its total degree is at most `4(ϑ − 1)`. Because `n` has `Fp` coefficients, `n̄(ξ') = n(ξ̄')`, so `n(ξ') ∈ R_3` forces every `Δ_jk = 0`.
   - **If some `Δ_jk` is non-zero.** Schwartz–Zippel over the grid `Fp^{2(ϑ−1)}` bounds the probability by `4(ϑ − 1)/p`.
   - **If every `Δ_jk` is zero.** The substitution `ξ = x + √w·y`, `ζ = x − √w·y` is invertible because `p` is odd. It turns the identities into `n_j(ξ)·n_k(ζ) = n_k(ξ)·n_j(ζ)` in independent variables. So `n = f·c` for one polynomial `f ≢ 0` and a constant vector `c`, which is real because `n`'s coefficients are. Now `g_0·n = h·n = 0`, which gives `c·g_0 ≡ c·h ≡ 0`, hence `c·g ≡ 0`. That is an `Fp`-dependence among `g`'s coordinates, a contradiction.
5. **Sum.** `2(ϑ − 1)/p^2 + 1/p^2 + 4(ϑ − 1)/p · (p + 1)/p^2 = ε(ϑ, p)`. ∎

**Hypotheses, as the specification meets them.**
1. **Every challenge uses up a slot.** `N` counts every `ξ*_T` drawn in the epoch, aborted sessions included. RFC-0012's `reserve` burns a proof's rows before `Round1`, and so before `ξ*_T` is derived. Its `abort` releases nothing, and the simulator argument (§6) counts an aborted session as one proof (R6). So a verifier that aborts and retries draws no extra points. Row reuse by a rolled-back or copied state is outside the theorem, as the operator's duty (draft RFC-0073).
2. **The bound is per matrix.** Summed over the ten weight matrices it is about `2^{-108.5}`, still under `2^{-100}`. `ChannelParams` is covered by the corrected reduction above.
5. **The evaluation challenge is fresh.** Each `θ_i` is uniform given everything before it, and `Z(x_i) ≠ 0` (RFC-0065 'The mask is never dropped'). Then `Ψ` can be fixed before the `ξ`s are drawn (correction, step 3).
3. **The evaluation points are uniform.** Each `ξ*_T` is a random-oracle output, uniform in `Fp2^{ϑ_T}` by rejection sampling (RFC-0004).
4. **The rows are distinct Boolean points.** The `Q*` rows `Dn + r` are distinct points of `{0,1}^ϑ`.

**Proof of the theorem.**
1. **First dependence.** `D_N` is the union over `j` of "session `j`'s pair is the first to fall in the span `U` of the earlier pairs". By the independence of `ξ_j`, it suffices to bound that event given `ξ_1, …, ξ_{j−1}` with the earlier vectors independent.
2. **Quotient by the earlier span.** `U` is conjugation-stable, so `U = U_0 ⊗ Fp2` with `U_0` real of dimension `2(j − 1)`. Let `π: Fp^{Q*} → Fp^m` be a real surjection with kernel `U_0`. Then `m = 2N + 8 − 2(j − 1) ≥ 10`.
3. **Apply Lemma A.** The event says that `π(e(ξ_j))` and its conjugate are dependent, that is, `π(e(ξ_j)) ∈ R_m`.
4. **The coordinates are independent.** `G(ξ) = π(e(ξ)) = Σ_{r ∈ Q*} eq(ξ, r)·π(1_r)` is multilinear with real coefficients. Its coordinates are `Fp`-independent: the `eq(·, r)` are linearly independent, as the Lagrange basis of the Boolean cube, and `π` is onto.
5. **Project to three coordinates.** Compose with any real surjection `P: Fp^m → Fp^3`. This keeps the coordinates independent, and maps `R_m` into `R_3`.
6. **Sum over sessions.** Lemma B bounds each session by `ε(ϑ, p)`. Summing over the `N` sessions gives the theorem. ∎

**Review.** An independent reading, a fresh agent given only this text, found every step sound and no counterexample. It also ran an exhaustive search over all `3^12` maps `g` at `p = 3`, `ϑ = 2`, plus hill-climbing at `p = 7` and `11`; the worst case was about `2.7–3.9/p^2`. It asked for the wording fixes above and for hypotheses 1 and 2 to be stated. Agent review is not the second human reading that adoption needs.

**Second review: what is still open.** A second independent agent read the correction against the specification. It found the quotient algebra sound, and the challenge-independence step sound with interleaved sessions too. It found four gaps in how the theorem is applied:
1. **Linking proofs are revealed vectors.** RFC-0064 gives each commitment a tensor proximity row `eq(ξ_link, ·)` over the epoch's blinding rows, with its designated pair in the link rows. A difference row adds another pair. The 8 link rows are therefore not spare. Each link proximity row must join the vector list and `Ψ`, at a cost of 2 target dimensions. The effective spare count is then the unreserved rows `U` alone.
2. **Epoch 0 can have no spare row.**
   - **The worst case.** `U = 0` at `N_0 = 682` proofs, for both weights (`B = 4096`) and `ChannelParams` (`B = 12,280`). The last session then has `m = 2`, so the event is a point of `P^1` being real, about `1/p`.
   - **Later epochs pass.** They have `U = 2` and `14`.
   - **Whether 682 is reachable.** Under accepted RFC-0012, repeated column queries can let rows bind first at 682. Under draft RFC-0073's count, `N_0 ≤ 674` and `U = 48`.
   - **The fixes.** Cap `N_0` at 681, adopt RFC-0073's count, or require `U ≥ 1` after the links. Each is a specification decision.
3. **`θ_T = 0` for `ChannelParams`.** All `e` evaluation rows then equal `β_T`. This dependency carries no data, so it is no leak, but a pure rank event `E` is not contained in `D_N^Ψ`. Define `E` by a non-zero data part, or add `N/q`. The text's `(r − M·b)` formula is the weights' map. For `ChannelParams` the functionals are `Φ_k = −Φ_β/c'_k`, which needs every `c'_k ≠ 0`.
4. **The scalar clause is not covered.** It covers `q_i`, `W'(p_i)`, and `ChannelParams`' stage values on `β_T` and `R_k`. These are fixed rank-one functionals. They leak if their directions fall in the real span of the `G(ξ_j)` and their conjugates, which is a different event from `D_N^Ψ`. Lemma 7's scalar step relies on R2's uniform block. The theorem's proof may extend by starting `U_0` from those directions, but a naive union costs a factor of about `N`.

So what is proved, pending human review, is the **vector-rank part** of Lemma 7 for the tensor generator. It holds at about `2^{-111.8}` per matrix in epochs after the first, once item 1 is added; that costs no dimension at `U ≥ 1`. Items 2 to 4 remain. The rest of this section states what the full result would settle.

**What it would settle, once items 1 to 4 are closed and review confirms it.**
- **Lemma 7 for the tensor generator.** The simulator argument's Lemma 7 holds for the tensor generator, with or without R1, at `2^{-111.8}` per matrix and epoch. That clears the `2^{-100}` target, so #394's choice between R2 and RFC-0021's free-row stream is no longer forced by zero knowledge.
- **The other reason for R2.** SUR-0001's soundness reason for uniform coefficients, the Johnson-regime bounds, is untouched.
- **The data.** EXP-0035's faster-than-linear growth at `p = 7` sits where `N·ε ≥ 1`, so the bound is vacuous there, and it does not contradict the data.

**A numerical check of Lemma B.** This is an uncommitted harness, recorded only as a check, like the one above.
- **The setting.** Exhaustive enumeration of every `ξ ∈ Fp2^ϑ`, with `w = −1`, over 1,020 random `g` that have sparse Boolean supports and independent coordinates.
- **The result.** The worst rate was `0.0554`, `0.0233` and `0.0080` at `ϑ = 2` and `p = 7, 11, 19`, against bounds of `0.155`, `0.061` and `0.020`. That is about `2.8/p^2` throughout, so the `p^{-2}` order is tight. At `ϑ = 3` the worst rate was `0.082` at `p = 7`, against a bound of `0.289`. No `g` exceeded the bound.

## Refutations

None yet.
