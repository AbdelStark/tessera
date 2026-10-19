# Uniqueness, integer bounds and padding audit

Reference for Step 3 of `soundness-audit`. Cite the exact inequality and `file:line` for each item checked.

## 1. Lemma 0 and centered-range discipline ([RFC-0002](../../../docs/rfcs/RFC-0002-field-arithmetic.md), [RFC-0006](../../../docs/rfcs/RFC-0006-compiler-ir.md#range-analysis), [RFC-0009](../../../docs/rfcs/RFC-0009-gadgets.md))

Goldilocks has `P = 2^64 − 2^32 + 1`. Lemma 0 states that if integers `a, b` satisfy `a ≡ b (mod P)` and `|a|, |b| < (P − 1)/2`, then `a = b` over `ℤ`.

- **Boundary gate**: every integer enters `Fp` through `from_i64` and leaves through `to_centered_i128`, with worst-case static range `[lo, hi]` satisfying `max(|lo|, |hi|) < 2^62 < (P − 1)/2` (INV-IR-1).
- **Verifier re-derivation**: the verifier recomputes worst-case ranges of every virtual expression from public `ArchSpec` and `GadgetPolicy` data alone, before checking the proof, and rejects any claim list where a lookup key or compared quantity can exceed its table or `2^62` (SP-7, [RFC-0007](../../../docs/rfcs/RFC-0007-claim-list.md)).
- **Accumulator overflow**: INT8 × INT8 GEMM accumulators stay below `2^31` (S2 of [RFC-0005](../../../docs/rfcs/RFC-0005-integer-model.md#integer-semantics)); witness-side multi-limb products (`ρ^2 u` up to `2^89` in `RsqrtRow`, `u · Z` in `InverseCdf`) use widened `i128` or explicit `BigIntCompare` carry chains, never wrapping `i64`.

## 2. Gadget uniqueness checklist ([RFC-0009](../../../docs/rfcs/RFC-0009-gadgets.md))

Every committed value upstream of the logits must have exactly one passing witness (SP-3). Check each gadget in scope against its lemma and its required alternative-witness test:

| Gadget | Lemma & precondition | Required soundness checks | Required alternative-witness tests |
|---|---|---|---|
| `Requant1` | Lemma 1; `b ≤ B_max` (`32`), `z` true integer in `[−2^{b−1}, 2^{b−1})` | `pack(c, z) = (z + 2^{b−1}) + 2^b · c` cannot collide across classes `c` because `z + 2^{b−1} ∈ [0, 2^b)` | `y ± 1`; wrong class `c'`; boundary `z = −2^{b−1}` and `2^{b−1} − 1` |
| `RequantSplit` | Lemma 2; `B_max < b ≤ 2·B_max − 2`, `s = b − B_max`, **precondition `s ≤ r − 1`**, `n · 2^b < P` | Verifier checks `s ≤ r − 1` and counts class term `2^b · c` (`n · 2^b`, #118) from public claim list; `(z_off − z_lo) · 2^{−s}` is range-checked in `[0, 2^{b−s})` | `z_lo' = z_lo ± 1`; `z_lo' = z_lo + 2^s`; wrap at `n = 4, b = 62` (#118); over `F65537`, demonstrate wrap when `s > r − 1` |
| `NonNeg` | Lemma 3; `k ≤ 62`, `n = ⌈k / L⌉` limbs (`L = 16`) | Low `n − 1` limbs in `[0, 2^L)`, top virtual limb in `[0, 2^{k − (n−1)L})` | Negative `v = −1`; `v = 2^k`; limb carry transfer `v_0 + 2^L, v_1 − 1` |
| `BigIntCompare` | Lemma 4; base-`2^L` limbs, borrow chain `β_j ∈ {0, 1}`, `β_{-1} = 0`, `β_last = 0` | Every difference limb `d_j ∈ [0, 2^L)`, every borrow `β_j ∈ {0, 1}`, every limb equation `< 2^41` so Lemma 0 applies per limb | `X = Y + 1`; forged borrow `β_j = 2`; unnormalized sum-of-products limb |
| `SoftmaxRow` | Lemmas 5 & 6; `k_qi = c_q − x_qi`, `(k_qi, e_qi) ∈ Exp`, `Π_{i ∈ V_q} k_qi = 0`, `NonNeg` brackets on `R_q` | Causal mask `C(q, i)` forces masked leaves (`i > q`) to `C(q, i) k_qi + 1 − C(q, i) = 1` so a zero product can only come from a valid position `i ≤ q`; `s_q ≥ 2^σ > 0` | `c_q` = second-largest score; `c_q = max + 1`; `k_qi = 0` placed at a masked position `i > q`; `R_q ± 1` |
| `RsqrtRow` | Lemma 7; `u = Σ res_i^2 + e < 2^55`, `N = d · 2^{2κ}`, two `BigIntCompare` brackets | Both `ρ^2 u ≤ N` and `N + 1 ≤ (ρ + 1)^2 u` checked over exact limb chains; `ρ ∈ [0, 2^{κ+1})` | `ρ − 1` and `ρ + 1`; corrupted `u` limb recomposition |
| `OneHot` | Lemma 8; degree-3 booleanity `h(1 − h) = 0`, `Σ h_v = 1`, `Σ v h_v = t` | Zero-check point drawn after `h` is committed; error `3 log2(vocab)/\|Fp2\|` counted in budget | Two ones with cancelling index sum; all zeros; single one at `t' ≠ t` |
| `Argmax` | Lemma 9; `lt_v = Σ_{w > v} h_w`, `g_v = m − ℓ_v − lt_v ∈ [0, 2^29)`, `Σ h_v ℓ_v = m` | Strict inequality `ℓ_v ≤ m − 1` for `v < y` enforces first-index tie-breaking (`min{v : ℓ_v = max ℓ}`) | Second tied index `y' > y` with `ℓ_{y'} = m`; second-highest logit |
| `InverseCdf` | Lemma 10; `C_< · 2^32 ≤ u · Z` and `u · Z + 1 ≤ C_≤ · 2^32` | Both comparators exact; `ĥ` proven one-hot; `Z = Σ e_v < 2^{σ+18}` | `ŷ − 1` and `ŷ + 1`; boundary `u = 0` and `u = 2^32 − 1` |

## 3. Padding and boundary-condition audit ([RFC-0008](../../../docs/rfcs/RFC-0008-logup-and-tables.md), [RFC-0010](../../../docs/rfcs/RFC-0010-proof-schedule.md))

- **Table and lookup padding**: when a lookup count `N` or table size `T` is not a power of two, padding must never introduce a new synthetic `(key, value)` pair into the table. RFC-0008 pads the lookup tree by repeating entry 0 (and adding the pad count to `m_0`) and pads the table tree by repeating row 0 (`t_0` with committed multiplicity `m_j`), with the verifier's structured table evaluation adding `t_0 · (1 − LT(ρ, T))`. Check that no padding slot uses `(0, α)` or `(0, 0)` unless `(0, 0)` is an authenticated table entry (#62).
- **Front-padding in batched sumcheck**: under LSB-first variable binding ([RFC-0010](../../../docs/rfcs/RFC-0010-proof-schedule.md#multilinear-extensions)), an instance of `n_i < n_max` variables is constant in the first `d = n_max − n_i` bound variables, its initial claim is scaled by `2^d`, each dummy round satisfies `p(0) = p(1) = claim / 2`, and its evaluation point is the suffix of the last `n_i` challenges (so stacked tensors share their high-order layer coordinate).
