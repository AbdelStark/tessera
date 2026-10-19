# Tessera offensive attack catalog

Concrete attack classes against Tessera's scheme and implementation, with the exact algebraic or mechanical lever each exploits and the real kills already found in this repository. Use as the playbook for Step 1 of `redteam-scheme`.

## 1. Fiat–Shamir & algebraic batching (`ADV-P`)

- **Late-bound / unbound claims (Kill `#237`)**: If a prover claim `q` enters a linear combination weighted by a challenge `γ` or `ξ` that was squeezed *before* `q` was absorbed into the transcript, the prover computes the false data combination first, squeezes `γ`, and solves a single linear equation for `q` so the combined check passes with probability 1.
  - *Target seam*: `pcs/advice/gamma` vs `q_g` in `pcs/advice/combined`; `pcs/weights/{T}/xi` vs `q_T` in `pcs/weights/{T}/combined`; `stage/{k}/zk-rho` vs `stage/{k}/zk-sum`.
  - *Check*: every claim in a linear combination must be absorbed before the challenge that weights it, and weighted by a coefficient no other free claim shares.
- **Challenge-reuse cross-cancellation (Kill `#62`)**: If one challenge `γ` både weights batch instances `i = 0..m−1` and combines two claims `(p_i, q_i)` inside each instance as `Σ_i γ^i (p_i + γ q_i)`, then `γ^{i+1}` multiplies both `q_i` and `p_{i+1}`. A cheating prover sets the error `Δq_i = −Δp_{i+1}` so the two errors cancel identically for every `γ`.
  - *Target seam*: LogUp fraction-tree levels (`p_i` numerator claim, `q_i` denominator claim) mixed with product forests.
  - *Check*: inner combinations must use an independent squeeze (`tree/{ℓ}/lambda`) so the polynomial is `Σ_i γ^i (p_i + λ q_i)` of degree `m − 1` in `γ` and `1` in `λ`.
- **Batching-size soundness erosion (`#58`, `#103`)**: Combining `M` instances with powers `1, γ, …, γ^{M−1}` has soundness error `(M − 1)/|Fp2|`. At 405B and 16K context, `SoftmaxRow` has `R ≈ 2^28` rows; batching them as `2^28` separate instances erodes soundness by 28 bits (down to `≈ 2^-100`, below the 112-bit round union target).
  - *Target seam*: any per-row or per-token tree or sumcheck instance.
  - *Check*: rows of the same shape must be stacked into one multilinear table over `s = log2 R` high-order row variables and reduced from a single random evaluation point `tree/{ℓ}/root/{c}` (`s/|Fp2|` error).
- **Front-padding coordinate shift (`#46`, `#53`, `#59`)**: Under LSB-first variable binding (`x_1` = index bit 0, bound in round 0), an instance with `n_i < n_max` variables padded in the low `d = n_max − n_i` variables must have constant rounds `p(0) = p(1) = claim / 2` for the first `d` rounds and evaluate at the *last* `n_i` challenges. If the prover or verifier slices the *first* `n_i` challenges instead, the evaluation point disagrees with the stacked layer coordinate `ρ_ℓ`.
- **Owed-claim drop (`#59`)**: A sumcheck or tree-level verifier function that checks internal round consistency (`p(0) + p(1) = claim`) returns an owed leaf/final claim. If a caller ignores that return value (or links it to the wrong next-stage tensor), any false initial claim passes as long as the prover runs a self-consistent sumcheck on a polynomial of its choice.

## 2. LogUp-GKR & lookup tables (`ADV-P`)

- **Phantom padding rows (Kill `#62`)**: When a table of `T` rows is padded to `2^{h_t}` leaves for the GKR fraction tree, what sits in the padding leaves `j ∈ [T, 2^{h_t})`? If the padding leaf has denominator `α − 0` (i.e. synthetic table entry `t_j = 0`, encoding `key = 0, value = 0`) and uses the prover's committed multiplicity `m_j`, then setting `m_j > 0` on a padding slot admits `(key = 0, value = 0)` as a valid lookup even when `(0, 0)` is not in the table!
  - *Target seam*: table tree leaf padding, lookup tree leaf padding, structured MLE `t̃(ρ)` evaluation for `T ≠ 2^k`.
  - *Check*: padding leaves must repeat an admissible entry (row 0: `(−m_j, α − t_0)` in the table tree and `(1, α − t_0)` in the lookup tree), and the verifier's `t̃(ρ)` evaluation must include `t_0 · (1 − LT(ρ, T))`.
- **Zero root or internal denominator**: In projective fraction trees `(N, D)`, `N_0 D_1 + N_1 D_0` with `D_0 = 0` or `D_1 = 0` collapses the sum to `0/0`. If the verifier checks `n_a d_t + n_t d_a == 0` without rejecting `d_a == 0` and `d_t == 0`, the prover sets `n_a = d_a = 0` at the root and proves any lookup multiset.
- **Multiplicity wrap modulo `P`**: If a single table entry is looked up `P` times (or the prover sets `m_i = P − 1 ≡ −1 (mod P)`), the sum of `1/(α − a_j)` wraps modulo `P`. Check that total lookups `N < P` and committed multiplicities are range-checked or bounded so no cancellation mod `P` occurs.
- **Tuple-compression collision**: `a = key + β · value` compresses pairs into `Fp2`. If `β` is squeezed *before* both the lookup keys and lookup values are committed in `Round1`, the prover chooses `(key', value')` after seeing `β` such that `key' + β · value' = key + β · value`.

## 3. Uniqueness gadgets & Goldilocks integer bounds (`ADV-P`)

- **Lemma 0 modular wrap (`a ≡ b (mod P)` with `|a − b| = P`)**: Goldilocks `P = 2^64 − 2^32 + 1 ≈ 1.84 × 10^19`. Whenever a gadget checks an integer equality or inequality via a linear relation in `Fp` plus range lookups, attempt to shift a witness limb or slack by `± P`:
  - In `RequantSplit`: `2^s z_hi = z_off − z_lo (mod P)`. If `b > 62` or `s > r − 1`, construct `z_lo' ∈ [0, 2^s)` where `(z_off − z_lo') · 2^{−s} mod P` lands inside `[0, 2^{b−s})`.
  - In `BigIntCompare`: `y_j − x_j − β_{j−1} = d_j − 2^L β_j`. If unnormalized limb products exceed `2^62`, add `P` to one limb equation to flip `β_last` from `1` to `0` and prove `X ≤ Y` when `X > Y`.
- **Causal-mask zero injection in `SoftmaxRow`**: The max certificate proves `c_q ≥ x_qi` (via `k_qi = c_q − x_qi ∈ Exp`) and `∃ i ∈ V_q : k_qi = 0` (via `Π_{i ∈ V_q} k_qi = 0`). If the product tree includes masked positions `i > q` with prover-chosen `k_qi` instead of forcing masked leaves to `1` via the public selector `C(q, i) k_qi + 1 − C(q, i)`, the prover sets `c_q > max_{i ≤ q} x_qi` and puts `k_qi = 0` at a masked position `i > q`, corrupting the softmax row while passing both checks!
- **Argmax tie-breaking shift**: `Argmax` must select the *smallest* vocabulary index `y` achieving the maximum logit `m`: `g_v = m − ℓ_v − lt_v ∈ [0, 2^29)` where `lt_v = 1` iff `v < y`. Test whether a prover can pick a later tied index `y' > y` (should fail because `v = y < y'` has `lt_y = 1` and `ℓ_y = m`, giving `g_y = −1 ≡ P − 1 ∉ [0, 2^29)`).

## 4. Polynomial commitments & Merkle multiproofs (`ADV-P`)

- **Combined-row / column-query desynchronization**: In `RFC-0011`, the verifier checks `Σ_r c_g(r) col_j[r] + q_col_j[g] == enc(u'_g, M_g)[j]` at each queried column `j` with domain point `x_j = 7 · ω_n^j`. Attack vectors:
  - Supplying `u'_g` of length `≠ k` or `M_g` of length `≠ P_A`, or omitting the proximity row (`G + 1` rows required).
  - Permuting the order of opened columns in `Round3Response` (must be strictly increasing deduplicated indices, `#220`).
  - Mutating one symbol in a queried column and checking that the failing index reported is the first failing column in query order (`#220`).
- **Non-canonical Merkle multiproofs (`#52`)**:
  - Extra unused sibling hashes appended to the proof.
  - Duplicate or unsorted leaf indices passed to multiproof verification.
  - Leaf index `≥ tree_size` or `tree_size` not matching the public commitment layout.

## 5. Zero knowledge & side channels (`ADV-V`)

- **Boolean-coordinate evaluation leak (Kill `#59` / `OQ-0013-2`)**: The masked extension `V'(x) = Ṽ(x) + Z(x) R̃_V(x_col)` uses `Z(x) = Π_i x_i(1 − x_i)`. At any evaluation point where even one coordinate is `0` or `1`, `Z(x) = 0` and `V'(x) = Ṽ(x)` is revealed completely unmasked!
  - *Target seam*: GKR tree reductions (`tree/{ℓ}/evals`) which reveal child evaluations at `0 ‖ r'` and `1 ‖ r'`; any boundary/padding check at hypercube coordinates.
- **Mask-row reuse across claims**: If two opening claims in different groups touch the same mask row `R_V` at the same column point, or if a blinding row `Q_g` is reused across two combined rows, taking their difference cancels the mask and leaks a linear combination of the secret witness rows.
- **Selective abort / seed-share grinding (`INV-PROTO-1`)**: `ADV-V` commits `c_V`, receives `Round1` + `s_P`, computes `seed = H("coin-toss", s_V ‖ s_P)`, and aborts if the resulting challenges don't land on a target structure, then reconnects asking the prover to reuse `Round1` with a new `s_V'`.

## 6. Decoders, resource limits & repository gates (`ADV-P` / `ADV-S`)

- **Pre-allocation length bomb (`#41`, `#44`)**: Craft a 12-byte TCE payload with `vec_len = 2^62` followed by EOF. If the decoder calls `Vec::with_capacity(len)` before checking `len <= limits.max_vec_len` and `len * MIN_SIZE <= remaining_bytes`, the verifier panics or OOMs under `ResourceLimits`.
- **Deep-nesting stack overflow**: Craft 257 nested `Option<Option<...>>` or `Vec<Vec<...>>` containers to test `DEPTH_CEILING = 256` before recursion.
- **Isolation & secret-access gate bypasses (Kills `#208`, `#32`)**:
  - Hiding a forbidden `tessera-*` dependency behind an optional Cargo feature not enabled by default (`--all-features` required).
  - Pulling prover files into `verifier/` via `[lib] path = "../../crates/..."`, inline `mod m { #[path = "..."] mod inner; }`, `#[cfg_attr(all(), path = "...")]`, or a macro expanding to `include!`.
  - Calling `r#expose_secret` (raw identifier) or dropping a nested `.clippy.toml` in a subdirectory to shadow the root `disallowed-methods` policy.
