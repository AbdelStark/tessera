# Zero-knowledge simulation and secret-isolation audit

Reference for Step 4 of `soundness-audit`.

## 1. Revealed-value accounting ([RFC-0013](../../../docs/rfcs/RFC-0013-zero-knowledge.md), INV-ZK-1..3)

Trace every byte that crosses `TB-1` (prover ↔ verifier) and confirm it falls into one of three buckets:
1. **Public by statement**: `ArchSpec`, sequence lengths, `LogManifest` digests, `Û` claims.
2. **Uniformly masked with fresh randomness**:
   - Sumcheck round messages masked by `ρ_k G_k(x)` where `G_k(x) = a_0 + Σ_{j=1}^n g_j(x_j)` has fresh random coefficients per stage.
   - Cross-stage tensor evaluations masked by `V'(x) = Ṽ(x) + Z(x) · R̃_V(x_col)` with `Z(x) = Π_i x_i(1 − x_i)` and a fresh advice row `R_V` (or weight blinding row).
   - LogUp fraction-tree roots `(n_a, d_a)` and `(n_t, d_t)` masked by the canceling leaf pair `(μ, δ)` in the lookup tree and `(−μ, δ)` in the table tree (`δ ≠ 0`), which both shifts `n/d` by `μ/δ` and scales `d` by `δ`.
   - Combined rows `u'_g = Σ_row c_g(row) A_row + Q_g` and combined vanishing masks `M_g`, each blinded by a fresh row `Q_g` used in at most one linear combination.
   - Opened columns masked by the `P_A` (advice) or `P_W` (weight) vanishing-polynomial coefficients per row, with the number of distinct opened columns bounded by `τ_A ≤ P_A` (advice, single-use) and `≤ P_W` across the weight epoch.
3. **Deterministic function of public data and values already in bucket 2**: for example, the final evaluation of a sumcheck once the round polynomials and challenges are fixed, or `q_g = ⟨Q_g, eq(col_g)⟩` once `u'_g` and the data claims are fixed.

## 2. Degenerate evaluation points and degree-of-freedom exhaustion

- **Boolean coordinates (`Z(p) = 0`)**: `Z(x) = Π_i x_i(1 − x_i)` vanishes whenever any coordinate `x_i ∈ {0, 1}`. Any protocol step that evaluates a tensor at a point with a Boolean coordinate—such as GKR tree level reductions evaluating `Ṽ_{ℓ+1}(0 ‖ r')` and `Ṽ_{ℓ+1}(1 ‖ r')` on the pair bit (`OQ-0013-2`, #224)—receives zero masking from `Z · R̃_V` and requires dedicated masking of its round messages and revealed child evaluations.
- **Linear independence of `eq(p_col^{(i)}, ·)`**: if a stacked tensor `V` is evaluated at `e_V` points in one proof, check that `e_V` does not exceed the degrees of freedom of `R_V`, and that no two revealed claims evaluate `R_V` at the same column sub-point `p_col` with different layer points, or different attention-tile index bits (KV head, query block), unless each evaluation has its own mask row. A committed tensor's `R_V` takes its data claim's column sub-point alone, so layer bits, tile bits and the placed tensor's variables past `log2 k` never separate its functionals ([RFC-0013](../../../docs/rfcs/RFC-0013-zero-knowledge.md#masked-extensions) 'Committed tensors', #554).
- **Weight epoch budget ([RFC-0012](../../../docs/rfcs/RFC-0012-weight-commitment.md#budget-state-machine))**: each proof consumes 3 blinding rows per weight matrix (evaluation, proximity, extension mask) and up to `τ_W` distinct columns. Verify that the budget state machine persists usage to disk before `Round3Response` and refuses to sign or prove once remaining columns `< τ_W` or remaining blinding rows `< 3` (`ERR-WITNESS-BUDGET`).
- **Abort and state restoration (`INV-PROTO-1`)**: if a live session aborts after `Round1` / `ProverSeedShare`, the prover never answers a second `SeedReveal` on the same `Round1` (`ERR-PROTO-REUSE`).

## 3. Secret isolation and side-channel checks ([05 §3](../../../docs/spec/05-observability.md#3-data-classification-and-redaction), [06 §7](../../../docs/spec/06-security.md#7-secrets-handling))

- **Redaction (`OBS-1`)**: `cargo xtask lint` verifies that `expose_secret` and `expose_secret_mut` appear only in `secret-access.toml` modules. Audit each listed module manually to confirm no reference, slice or derived secret escapes through a return value, mutable out-parameter, closure or trait implementation.
- **Heap residency and zeroization (`OBS-2`)**: secrets that grow or move across stack frames are allocated inside `Secret<Box<T>>` or `Secret<Vec<T>>` with capacity reserved up front so ` realloc` never leaves an unzeroized copy in freed heap pages.
- **Constant-time execution (`INV-ARCH-2`)**: inspect every loop bound, `if`/`match`, early return and division in prover witness, field, polynomial and encoding code that touches secret-class data. Control flow must depend only on public dimensions (`ArchSpec`, token counts, parameter set).
