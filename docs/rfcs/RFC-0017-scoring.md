# RFC-0017: Scoring and the Unexplained-Information Bound Û

- Status: Accepted
- Authors: @AbdelStark
- Created: 2026-10-15
- Target milestone: v0.3 (exact, greedy), v0.4 (sampling, bounded mode)
- Amended by: [RFC-0035](RFC-0035-padded-rows.md) ('In-circuit scoring', constraints 3 and 4: the row selector `ψ`)
- Amended by: [RFC-0061](RFC-0061-sampling-randomness.md) ('Summary', 'Sampler PRF' and 'Testing Strategy': a keyed stream class of `b` public bits in place of `request_id`, 0 by default, under a key the verifier pins; 'Semantics': what `Û` bounds)

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

- Vocabulary softmax at temperature `T`: `e_v = Exp_T[c − ℓ_v]` with a max certificate over the vocabulary (as `SoftmaxRow`, [RFC-0009](RFC-0009-gadgets.md#softmaxrow)), `Exp_T` a vocabulary exponential table with the scale `σ_ℓ / T` folded in, `Z = Σ_v e_v`.
- `ŷ_t`: `Argmax` for greedy; `InverseCdf` with `u_t` for temperature ([RFC-0009](RFC-0009-gadgets.md#inversecdf-temperature-sampling)).
- Smoothed distribution: `p̂_t(v) = (e_v + 1)/(Z + V)` with `V = vocab`, which is positive everywhere and sums to 1.

### Predictors

- **Exact:** `q_t = δ_{ŷ_t}`. Constraint: `y_t = ŷ_t` (the certificate is built on `y_t`'s own one-hot vector). `b_t = 0`. If the deployment deviated, no witness exists and the prover reports `ERR-WITNESS-EXACT-MISMATCH`.
- **Mixture:** `q_t = (1 − ε) δ_{ŷ_t} + ε p̂_t`.

### In-circuit scoring (mixture)

Per output token, committed: match bit `m_t`; integer part `I_t ∈ [0, 64)`; fraction `f_t ∈ [0, 2^8)`.

1. `m_t ∈ {0, 1}`; if `m_t = 1` then `h_{y_t} = h_{ŷ_t}` (zero-check over the vocabulary of `m_t (h_y − h_ŷ)`).
2. If `m_t = 1`: `b_t = c_match = ⌈−log2(1 − ε)⌉_8` (a public constant in units of 2^-8 bits). Valid because `q_t(y_t) ≥ 1 − ε`.
3. If `m_t = 0`: `b_t = k + I_t + f_t / 2^8` with the constraint `(e_{y_t} + ψ_t) · 2^{I_t} · E2[f_t] ≥ ψ_t·(Z + V) · 2^{16}`, `ψ_t` the row selector, 1 on a real token and 0 on a padded one ([RFC-0035](RFC-0035-padded-rows.md#the-relations)), where `E2[f] = ⌊2^{f/2^8} · 2^{16}⌋` is a 256-row public table and `2^{I_t}` comes from a 64-row `Pow2` table. Because `E2` rounds down, the check implies `2^{-(b_t)} ≤ ε p̂_t(y_t) ≤ q_t(y_t)`. The products exceed `P`, so the comparison uses `BigIntCompare` ([RFC-0009](RFC-0009-gadgets.md#bigintcompare)).
4. `b_t = ψ_t · (m_t · c_match + (1 − m_t)(k + I_t + f_t/2^8))`, a virtual expression of degree 3 whose public factor `ψ_t` makes a padded row's `b_t` 0.
5. Per-request `Û_r = Σ_{t ∈ r} b_t` (fixed point, units of 2^-8 bits) is a public claim; per-token `b_t` values stay hidden.

**One-sided freedom.** The prover chooses `m_t`, `I_t`, `f_t`. Claiming a mismatch when tokens match, or a larger `I_t` or `f_t` than needed, raises `Û`; no choice lowers it below a valid bound. This is the rule of source §6, principle 5.

### Expected behavior

With mismatch rate `r` (the fraction of positions where the deployment's token differs from `ŷ`), `E[Û]/token ≈ (1 − r) · ε/ln 2 + r · (k + s̄)`, with `s̄` the mean smoothed surprisal of mismatched tokens. For `k = 6` and `r = 1%`, about `0.02 + 0.01 · (6 + s̄)` bits per token. The deployment's sampler must use the committed PRF stream for bounded mode to be useful; without it `r` approaches the sampling entropy and `Û` grows accordingly. The v0.4 fidelity study measures `r` and `Û` for an FP8 deployment proxy against 8B and 70B `D` (SC-6).

### Verifier

The verifier recompiles the scoring circuit from `ScoringTemplate` (claim list) and `PredictorSpec` (statement), checks it like any other gadget, reads `Û_r` per request, and returns `Reject(UBoundExceeded)` if any `Û_r` or the batch total exceeds the bound passed with `--bound`.

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
