# RFC-0017: Scoring and the Unexplained-Information Bound Û

- Status: Accepted
- Authors: @AbdelStark
- Created: 2026-10-15
- Target milestone: v0.3 (exact, greedy), v0.4 (sampling, bounded mode)
- Amended by: [RFC-0035](RFC-0035-padded-rows.md) ('In-circuit scoring', constraints 3 and 4: the row selector `ψ`)
- Amended by: [RFC-0061](RFC-0061-sampling-randomness.md) ('Summary', 'Sampler PRF' and 'Testing Strategy': a keyed stream class of `b` public bits in place of `request_id`, 0 by default, under a key the verifier pins; 'Semantics': what `Û` bounds), [RFC-0071](RFC-0071-request-slots.md) ('In-circuit scoring': in a batch, a request's tokens are its slot's rows)

## Summary

The third conjunct of the statement scores the recorded outputs against `D`'s logits and publishes `Û`, an upper bound in bits on the information in the outputs that `D`, the prompt and the committed sampling randomness do not explain. `Û` is the code length of the outputs under a committed predictor `q`, so it is valid for any predictor by Gibbs' inequality, and a poor predictor only inflates it. Two predictors are defined: **exact** (`q` is a point mass on `D`'s own decoded token; `Û = 0` iff the outputs equal `D`'s outputs) and **mixture** (`q = (1 − ε)·δ_ŷ + ε·p̂`, for bounded mode with an unmodified floating-point deployment). Decoding is greedy or inverse-CDF temperature sampling from an AES-based PRF of a keyed stream class and the output position, under a sampler key whose commitment the verifier pins ahead of the window it serves ([RFC-0061](RFC-0061-sampling-randomness.md)).

## Motivation

The source keeps VerInf's public output: "a bound `Û` on unexplained information, valid for any predictor by Gibbs' inequality … works with an unmodified floating-point deployment; exact replay is the `Û = 0` limit; a poor predictor only hurts the prover" (source §4 table). It requires "unique witness upstream of the logits, one-sided freedom downstream" (source §4, §6 principle 5), and in exact mode "decoding is greedy or inverse-CDF sampling from a committed PRF stream, and `Û = 0` certifies token-for-token equality" (source §7). Bounded-mode fidelity is an open problem with a target of `Û ≤ 0.1` bits per token against VerInf's measured 0.88 (source §9, problem 4). The source does not fix a predictor family; this RFC does.

## Goals

- A precise meaning for `Û` and a proof that the circuit's `Û` is a valid upper bound.
- In-circuit scoring with downstream freedom that can only raise `Û`.
- A sampler PRF that serving engines can implement at negligible cost.
- Per-request `Û` disclosure without per-token disclosure.

## Non-Goals

- Top-k, top-p, repetition penalties and logit biases. Each needs in-circuit sorting or extra state; future work.
- Choosing the acceptance bound on `Û`; that is policy.

## Proposed Design

### Semantics

For request outputs `y_1 … y_n` and committed predictors `q_t` (distributions over the vocabulary that depend only on `D`, the prompt, previous outputs and the committed sampling randomness), define `Û = Σ_t b_t` with `b_t ≥ −log2 q_t(y_t)`. Arithmetic coding with the `q_t` encodes `y_1 … y_n` in at most `Û + 2` bits given everything the statement fixes. So the outputs carry at most `Û + 2` bits beyond what is determined by `(D, prompt, sampler key)`. That bounds a covert channel in the outputs given the recorded prompt and lengths. What the serving stack puts in the prompt, and where it ends a request, are outside it ([RFC-0061 'What remains the prover's'](RFC-0061-sampling-randomness.md#what-remains-the-provers)). Validity needs only that each `q_t` sums to at most 1 and is fixed before `y_t` is chosen, which holds because `q_t` is a function of committed data and the sampler stream. (Gibbs' inequality gives the expected-value form; the code-length form holds per sequence.)

### Decoding and the sampler PRF

```rust
pub enum Decoding { Greedy, Temperature { t: Rational } }   // 0 < t ≤ 4, at most 4 distinct values per batch
pub enum PredictorSpec {
    Exact { decoding: Decoding },
    Mixture { decoding: Decoding, eps_log2: u8 },            // ε = 2^{-k}, k in 0..=16, default 6
}
```

### Sampler PRF

For temperature decoding at output position `t` of a request whose prompt is `x_0, …, x_{n−1}`, under the sampler key `K` with `b` class bits ([RFC-0061](RFC-0061-sampling-randomness.md#the-stream)):

```text
h     = SHA-256( "tessera-stream-class/v1" ‖ K ‖ 0^25 ‖ u64_le(n) ‖ u32_le(x_0) ‖ … ‖ u32_le(x_{n−1}) )
class = u64_le(h[0..8]) >> (64 − b)      for 1 ≤ b ≤ 64;   class = 0 for b = 0
u_t   = u32_le( AES-128_K( u64_le(class) ‖ u32_le(t) ‖ 0^4 )[0..4] )
```

The class takes the slot of `request_id`. However the serving stack varies a prompt, it reaches at most `2^b` streams. At the reference `b = 0`, the verifier's default, it reaches one. `K` is derived from the prover's committed base key, a nonce that the verifier draws when it pins the commitment, `b`, and the start of the pin's fixed-length window, so neither the prover nor its verifier chooses it ([RFC-0061 'The sampler key and its pins'](RFC-0061-sampling-randomness.md#the-sampler-key-and-its-pins)). The serving engine's sampler must draw exactly this stream and select the inverse-CDF token under its own distribution. The reference implementation lives in `tessera-scoring` so engines can link or port it.

### `D`'s decoded token

- Vocabulary softmax at temperature `T`: `e_v = Exp_T[c − ℓ_v]` with a max certificate over the vocabulary (as `SoftmaxRow`, [RFC-0009](RFC-0009-gadgets.md#softmaxrow), without a reciprocal; `VocabMax`), `Exp_T` a vocabulary exponential table with `sigma = exp_sigma`, the scale `σ_ℓ / T` folded in (`σ_ℓ = 2^{logits_exponent}`, [RFC-0027](RFC-0027-logits-exponent.md)), and the narrowest `key_bits ∈ [min_key_bits, max_log2_rows]` (`min_key_bits = max(1, bit_length(hi − lo))` for the lowered LM-head logit interval `[lo, hi]`) reaching `Exp_T[2^{key_bits} − 1] = 0` ([RFC-0030](RFC-0030-masked-exponentials.md)), `Z = Σ_v e_v`. For greedy mixture decoding (`PredictorSpec::Mixture { decoding: Decoding::Greedy, .. }`), `p̂_t` uses `T = 1` (`Exp_1`).
- `ŷ_t`: `Argmax` over the narrowest physical `Range { bits }` (`1 ≤ bits ≤ 29`) covering the public logit span ([RFC-0053](RFC-0053-greedy-runtime-boundary.md)) for greedy; `InverseCdf` with `u_t` for temperature ([RFC-0009](RFC-0009-gadgets.md#inversecdf-temperature-sampling), [RFC-0058](RFC-0058-inverse-cdf-columns.md)). In mixture mode, `h_{ŷ_t}` is a separate committed `OneHot` column (`SampledIndicator`) rather than the recorded token's `h_{y_t}`.
- Smoothed distribution: `p̂_t(v) = (e_v + 1)/(Z + V)` with `V = vocab`, which is positive everywhere and sums to 1.

### Predictors

- **Exact:** `q_t = δ_{ŷ_t}`. Constraint: `y_t = ŷ_t` (the certificate is built on `y_t`'s own one-hot vector). `b_t = 0`. If the deployment deviated, no witness exists and the prover reports `ERR-WITNESS-EXACT-MISMATCH`.
- **Mixture:** `q_t = (1 − ε) δ_{ŷ_t} + ε p̂_t` with `ε = 2^{−k}`, `k = eps_log2 ∈ 0..=16`.

### In-circuit scoring (mixture)

Per output token `t`, `OneSidedLog2` checks the one-sided mixture bound over vocabulary `V = vocab ∈ [1, 2^{18}]`, exponential output precision `σ = exp_sigma ∈ [0, 32]`, and `k = eps_log2 ∈ 0..=16`, with partition width `W_Z = bit_length(V · 2^σ)` (`k_Z = ⌈W_Z / 16⌉`), scaled-numerator width `W_Q = bit_length((2^σ + 1) · (2^{17} − 1))` (`k_Q = ⌈W_Q / 16⌉`), and radix-`2^{16}` comparison limb count `m = ⌈bit_length(max((2^{W_Z} − 1 + V) · 2^{16}, (2^{W_Q} − 1) · (2^{63} − 1))) / 16⌉` (`= ⌈(W_Q + 63) / 16⌉`).

1. When `1 ≤ k ≤ 16`, `m_t(1 − m_t) = 0`; when `k = 0` (`ε = 1`, pure surprisal `q_t = p̂_t`, where `1 − ε = 0`), `m_t = 0`. If `m_t = 1` then `h_{y_t} = h_{ŷ_t}` (vocabulary zero-check `m_t (h_{y_t,v} − h_{ŷ_t,v}) = 0`).
2. If `m_t = 1` (`1 ≤ k ≤ 16`): `B_t = c_match(k) = ⌈−log2(1 − 2^{−k}) · 2^8⌉` (`[0, 256, 107, 50, 24, 12, 6, 3, 2, 1, 1, 1, 1, 1, 1, 1, 1]` for `k = 0..=16`, a public integer constant in `FixedBits` units of `2^{−8}` bits). Valid because `q_t(y_t) ≥ 1 − ε`.
3. If `m_t = 0`: `B_t = 256 · k + 256 · I_t + f_t` (representing `b_t = k + I_t + f_t / 2^8` bits) with `Y_t = Σ_v h_{y_t,v} · e_{t,v}`, `Z_t = Σ_v C_v · e_{t,v}`, the degree-2 equality `q_t − (Y_t + ψ_t) · E_t = 0`, and the radix-`2^{16}` `BigIntCompare` constraint `ψ_t · (Z_t + V) · 2^{16} ≤ q_t · P_t`, `ψ_t` the row selector, 1 on a real token and 0 on a padded one ([RFC-0035](RFC-0035-padded-rows.md#the-relations)), where `E_t = E2[f_t] = ⌊2^{f_t/2^8} · 2^{16}⌋` comes from `Exp2Frac { frac_bits: 8, out_bits: 16 }` and `P_t = 2^{I_t}` from `Pow2 { max_exp: 63 }` ([RFC-0008](RFC-0008-logup-and-tables.md#table-specifications), as `2^{63}` exceeds Goldilocks' centered range `(P − 1)/2`). Because `E2` rounds down, the check implies `2^{−b_t} ≤ ε p̂_t(y_t) ≤ q_t(y_t)`. Factoring `q_t` keeps `BigIntCompare`'s right-hand limb products `Q_a · P_b` quadratic and strictly below `2^{34} ≪ 2^{40}` ([RFC-0009](RFC-0009-gadgets.md#bigintcompare)).
4. **Comparison coefficients and limb boxes** ([RFC-0050 'Complete comparison schema'](RFC-0050-gadget-advice-columns.md#complete-comparison-schema)). Write `V = V_0 + 2^{16} · V_1` (`V_0 = V mod 2^{16}`, `V_1 = ⌊V / 2^{16}⌋`, `V_h = 0` for `h ≥ 2`), `Z_t = Σ_{0 ≤ h < k_Z} Z_h · 2^{16h}` (`Z_h = 0` for `h ≥ k_Z`), `q_t = Σ_{0 ≤ a < k_Q} Q_a · 2^{16a}`, and `P_t = Σ_{0 ≤ b < 4} P_b · 2^{16b}`. For `0 ≤ j < m`:
   - `X_0 = 0`, and `X_j = ψ_t · (Z_{j−1} + V_{j−1})` for `1 ≤ j < m`;
   - `Y_j = Σ_{a + b = j, 0 ≤ a < k_Q, 0 ≤ b < 4} Q_a · P_b`.
   Their public input bounds `(A_left[j], A_right[j])` are derived from the full lookup boxes (`box(Z_h) = 2^{16} − 1` for `0 ≤ h < k_Z − 1`, `box(Z_{k_Z−1}) = 2^{W_Z − 16(k_Z − 1)} − 1`, `box(Z_h) = 0` for `h ≥ k_Z`; `box(Q_a) = 2^{16} − 1` for `0 ≤ a < k_Q − 1`, `box(Q_{k_Q−1}) = 2^{W_Q − 16(k_Q − 1)} − 1`; `box(P_b) = 2^{16} − 1` for `b ∈ {0, 1, 2}`, `box(P_3) = 2^{15} − 1`): `A_left[0] = 0`, `A_left[j] = box(Z_{j−1}) + V_{j−1}` (`1 ≤ j < m`), and `A_right[j] = Σ_{a+b=j} box(Q_a) · box(P_b)`. Carry widths, equation boxes and Lemma 0 checks follow from `(A_left, A_right)` per [RFC-0050](RFC-0050-gadget-advice-columns.md#complete-comparison-schema).
5. **Committed columns, lookup requests and row equalities.** Per row, `OneSidedLog2` commits `6 + (k_Z − 1) + (k_Q − 1) + 3 + 6m` values in order: (1) `m_t`, (2) `I_t`, (3) `P_t`, (4) `f_t`, (5) `E_t`, (6) `q_t`, (7) `Z_0 .. Z_{k_Z−2}`, (8) `Q_0 .. Q_{k_Q−2}`, (9) `P_0, P_1, P_2`, (10) `BigIntCompare`'s six arrays (`LeftNorm` `x'_j`, `LeftCarry` `c_{L,j}`, `RightNorm` `y'_j`, `RightCarry` `c_{R,j}`, `Diff` `d_j`, `Borrow` `b_j`) in [RFC-0050](RFC-0050-gadget-advice-columns.md) order. Its `2 + k_Z + k_Q + 4 + 6m` lookup requests follow the same order, with each virtual top limb (`Z_{k_Z−1} = (Z_t − Σ_{j < k_Z−1} Z_j · 2^{16j}) / 2^{16(k_Z−1)} ∈ Range(W_Z − 16(k_Z − 1))`, `Q_{k_Q−1} = (q_t − Σ_{j < k_Q−1} Q_j · 2^{16j}) / 2^{16(k_Q−1)} ∈ Range(W_Q − 16(k_Q − 1))`, and `P_3 = (P_t − P_0 − 2^{16} P_1 − 2^{32} P_2) / 2^{48} ∈ Range(15)`, unique in `[0, 2^{63})` since `P > 2^{63}`) emitted immediately after its low `Range(16)` limbs: `(I_t, P_t) ∈ Pow2(63)`, `(f_t, E_t) ∈ Exp2Frac(8, 16)`, `Z_0 .. Z_{k_Z−1}`, `Q_0 .. Q_{k_Q−1}`, `P_0 .. P_3`, then `BigIntCompare`'s `6m` `Range` requests in [RFC-0050](RFC-0050-gadget-advice-columns.md) order. Its `3m + 6` row equalities come in order: (1) `m_t = 0` (`k = 0`) or `m_t − m_t^2 = 0` (`1 ≤ k ≤ 16`), (2) `q_t − (Y_t + ψ_t) · E_t = 0`, (3) `BigIntCompare`'s `3m + 3` equalities (`X_j + c_{L,j−1} − x'_j − 2^{16} c_{L,j} = 0` for `0 ≤ j < m` with `c_{L,−1} = 0`, then `c_{L,m−1} = 0`; `Y_j + c_{R,j−1} − y'_j − 2^{16} c_{R,j} = 0` for `0 ≤ j < m` with `c_{R,−1} = 0`, then `c_{R,m−1} = 0`; `y'_j − x'_j − d_j + 2^{16} b_j − b_{j−1} = 0` for `0 ≤ j < m` with `b_{−1} = 0`, then `b_{m−1} = 0`), and (4) `(ψ_t − 1) · Z_t = 0`.
6. **Runtime table interning.** Any runtime table not already in `ClaimList.tables` is appended at `TableId(ClaimList.tables.len()..)` and deduplicated by `TableKind` (first occurrence wins): first each distinct effective temperature's `Exp_T` table in first-seen output-bearing request order (`T = 1` for greedy mixture requests), then for each output-bearing request in batch order: `Argmax`'s physical `Range { bits: min_key_bits }` (`min_key_bits = max(1, bit_length(hi − lo))` for the public logit interval `[lo, hi]`, [RFC-0053](RFC-0053-greedy-runtime-boundary.md#public-range-and-catalog)) for greedy or bounded `InverseCdf`'s `Range` requests ([RFC-0058](RFC-0058-inverse-cdf-columns.md)) for temperature, followed by `OneSidedLog2`'s lookup requests in the order of item 5.
7. `B_t = ψ_t · (m_t · c_match(k) + (1 − m_t)(256 · k + 256 · I_t + f_t))`, a virtual integer expression in `2^{−8}`-bit units of degree 3 whose public factor `ψ_t` makes a padded row's `B_t` 0.
8. Per-request `Û_r = FixedBits(Σ_{t ∈ r} B_t)` (fixed point, units of `2^{−8}` bits) is a public claim; per-token `B_t` values stay hidden. In a batch, `t ∈ r` runs over request `r`'s slot, its positions `prompt_len_r − 1` to `n_r − 2` ([RFC-0071](RFC-0071-request-slots.md#the-greedy-certificate-and-scoring)).

**One-sided freedom.** The prover chooses `m_t`, `I_t`, `f_t`. Claiming a mismatch when tokens match, or a larger `I_t` or `f_t` than needed, raises `Û`; no choice lowers it below a valid bound. This is the rule of source §6, principle 5.

### Expected behavior

With mismatch rate `r` (the fraction of positions where the deployment's token differs from `ŷ`), `E[Û]/token ≈ (1 − r) · ε/ln 2 + r · (k + s̄)`, with `s̄` the mean smoothed surprisal of mismatched tokens. For `k = 6` and `r = 1%`, about `0.02 + 0.01 · (6 + s̄)` bits per token. The deployment's sampler must use the committed PRF stream for bounded mode to be useful; without it `r` approaches the sampling entropy and `Û` grows accordingly. The v0.4 fidelity study measures `r` and `Û` for an FP8 deployment proxy against 8B and 70B `D` (SC-6).

### Verifier

The verifier recompiles the scoring circuit from `ClaimList`'s public quantization, architecture and gadget-policy fields ([RFC-0026](RFC-0026-claim-list-layout.md)) and `PublicStatement.predictor`, checks it like any other gadget, reads `Û_r` per request, and returns `Reject(UBoundExceeded)` if any `Û_r` or the batch total exceeds the bound passed with `--bound`.

### Invariants

- **INV-SC-1.** Every `q_t` is a function of committed data and the sampler stream only, and sums to at most 1.
- **INV-SC-2.** Downstream freedom can only increase `Û`.
- **INV-SC-3.** Only per-request sums of `b_t` are revealed.

## Alternatives Considered

1. **Pure surprisal, `q_t = p̂_t`** (the mixture with `k = 0`). Kept as a setting, rejected as the default: it counts the sampling entropy itself as unexplained, about 1–3 bits per token at temperature 1.
2. **A CDF-window predictor** that spreads mass around `u_t` in `D`'s cumulative distribution, so near-misses cost fewer bits than under the mixture. Considered as potentially tighter for FP8 deployments whose distribution is close to `D`'s. Deferred to the v0.4 fidelity study (OQ-0017-1): it needs prefix-sum comparisons at several offsets per token.
3. **ChaCha20 or a field-native PRF for sampling.** Rejected: AES is already in the anchor circuits, and serving hardware has AES instructions.
4. **Revealing per-token `b_t`.** Rejected: it would reveal which positions match `D`, which leaks output information.

## Drawbacks

- Bounded mode needs the deployment's sampler to adopt the committed PRF stream, a serving-stack change, though a small one.
- Temperature values are limited to four per batch because each needs its own vocabulary exponential table.
- `Û` is loose by up to 2 bits per request plus the 2^-8 rounding per token.

## Migration / Rollout

v0.3: exact predictor with greedy decoding. v0.4: sampler PRF, temperature decoding, mixture predictor, fidelity study.

## Testing Strategy

- Sampler PRF: test vectors (key, class bits, prompt, position → class, `u_t`) shared by `tessera-scoring`, the verifier and the circuit; the reference sampler and the `InverseCdf` certificate agree on 10^5 random distributions.
- Exact mode: honest proofs give `Û = 0`; one altered output token yields `ERR-WITNESS-EXACT-MISMATCH`.
- Mixture: for random logits and outputs, the circuit's `Û_r` is at least the exact `Σ −log2 q_t(y_t)` computed in 256-bit arithmetic and at most that plus `n · 2^{-8} + n · c_match`-level rounding slack; a prover claiming `m_t = 1` on a mismatched token is rejected; a prover claiming an `I_t` one smaller than needed is rejected.
- Gibbs validity smoke test: for synthetic `D` and outputs sampled from `q`, the average `Û` per token is at least the empirical entropy rate.
- Fidelity study (v0.4): `r` and `Û` per token with 95% bootstrap intervals for an FP8 proxy deployment of 8B and 70B, reported against SC-6.

## Open Questions

- **OQ-0017-1.** Does a CDF-window predictor lower bounded-mode `Û` enough to justify its circuit cost? Owner: @AbdelStark. Resolve in the v0.4 fidelity study; adopt by a superseding RFC if it reaches the source's 0.1 bits per token target where the mixture does not.

## References

- Source §2 (statement conjunct 3), §4 table (`Û` and Gibbs' inequality), §6 principle 5, §7 (exact mode decoding), §9 problem 4.
- Cover and Thomas, "Elements of Information Theory", 2nd ed. (Gibbs' inequality, arithmetic coding).
- [RFC-0009](RFC-0009-gadgets.md), [RFC-0015](RFC-0015-transcript-anchor.md).
