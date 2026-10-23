# IDEA-0039: Score instances without the score tensor

| | |
|---|---|
| Status | `validated` ([EXP-0026](../experiments/EXP-0026-causal-score-rounds/README.md)) |
| Origin | [#458](https://github.com/AbdelStark/tessera/issues/458): RFC-0055's review found `SoftmaxRow`'s key and forest relations in stage 1 over the score tensor, 8.8 TB at 1K context and 141 TB at 16K for a 405B batch of 16K tokens |
| Lever | `regen`, `stream`, memory (RISK-10); the stage-1 sumcheck share of glue |
| Spec surfaces | None in the messages: a prover algorithm for RFC-0009's relations, which RFC-0010 batches in stage 1. RFC-0055 'Native relations' and RFC-0018 'Pass structure' describe the prover's memory and passes |
| Priority | Triage below |

## Mechanism

Stage 1 proves `SoftmaxRow`'s two claims on the score tensor as sums over it ([RFC-0009](../../docs/rfcs/RFC-0009-gadgets.md#softmaxrow)):

```text
Σ_x eq(r, x) · M(x) · (c(row(x)) − s(x)) = L̃(r) − 1 + M̃(r)          (forest)
Σ_x eq(ρ, x) · M(x) · (c(row(x)) − s(x)) = κ̃(ρ) − K_0·(1 − M̃(ρ))    (key)
```

- **The terms.** `x = (ℓ, h, q, i)` ranges over the padded `[layers, heads, queries, keys]` tensor, the key index `i` lowest. `M` is the public mask: causal `i ≤ q < n`, real heads and real layers. `c` is the row maximum, and `s = Q·Kᵀ` the score, bilinear in the INT8 queries and keys of each head, a key head shared by its query heads.
- **The prover today.** It materializes `s` and the broadcast `c` at the score tensor's size.
- **The candidate.** It never materializes the score tensor.

**Rounds over the key bits** (`j = 1, …, log2 n_pad`). Split `i` and `q` at bit `j`: `i = (i_lo, i_hi)`, `q = (q_lo, q_hi)`, the low parts the first `j` bits.
- **The mask.** With the low key bits at the bound point `(r_{<j}, X)`, `M̃` is `1` when `i_hi < q_hi`, `LT((r_{<j}, X), q_lo + 1)` when `i_hi = q_hi`, and `0` when `i_hi > q_hi`, for a real row, head and layer.
- **The score.** `s̃` is `Q_q · K̃((r_{<j}, X), i_hi)`, with `K̃` the keys folded over their bound bits.
- **The round.** So round `j`'s share of `s` is `Σ_{ℓ,h,q} eq_{ℓ,h,q} · Q_q · [P_X(q_hi) + eq_hi(q_hi)·LT(·, q_lo + 1)·K̃(·, q_hi)]`, `P_X` a prefix sum over the key blocks `i_hi < q_hi` of `eq_hi(i_hi)·K̃((r_{<j}, X), i_hi)`.
- **Its cost.** `O(H_kv·n/2^j·d)` for the prefix sums, and `O(H·n·d)` for the dot products, per layer and per evaluation point.
- **The row maximum.** `c` is constant over `i`. Its share is a row table times `Σ_{i_hi} eq_hi·M̃`, which has the same closed form.

**After the key bits.** `s̃(r_i, q) = Q_q · K̃(r_i)` and `c` are row tables over `(ℓ, h, q)`, and `M̃(r_i, q) = LT(r_i, q + 1)` for a real row. The remaining rounds run on tables of `H·n·L` entries.

**Memory.** `Q` and `K` as INT8, `K̃`'s folds, and the row tables: `T·(H·d·L + 2·H·L·16 + …)` bytes for a batch of `T` tokens, against `T·H·S·L·32` for the two score-sized tables. It is linear in the tokens and independent of the context.

**Work.** `O(T·H·d·L·log S)` multiplications by `Fp2` weights, against `O(T·H·S·L)` table entries per round for the resident prover.

## Why it could work

- **Bilinear scores.** The scores are the `MatMulAct` output `Q·Kᵀ`, with no requantization before `SoftmaxRow` (`tessera-compiler/src/lower.rs`, `attention`).
- **A causal mask.** Its multilinear extension at a point whose low key bits are bound has the three-case form above. That is the standard decomposition of `LT` ([RFC-0010](../../docs/rfcs/RFC-0010-proof-schedule.md#multilinear-extensions)).
- **Linear scores in the key.** A sumcheck prover only needs each round polynomial. A prover that never stores a table but evaluates each round from folds of the factors it is bilinear in, as RFC-0010's `MatMulAct` instances already do, sends the same polynomial (E0).

## Expected gain

- **At 405B, 16K tokens.** The score instances' working set drops from 8.8 TB (1K context) and 141 TB (16K) to about 60 GB at any context. Their rounds over the key bits cost about `3·log2 S` dot products of length `d` per query row and head, against a score-sized table per round.
- **No pass, no protocol change.** Stage 1 keeps one pass for these instances, its grid pass, and the windows of RFC-0055 serve its elementwise instances as before.

| Context | Δμ | Δ proof size | Δ verification |
|---|---|---|---|
| 1K–128K | removes a working set that no projection counted, so 0 on the corrected `μ` | 0 | 0 |

## Security gate

| Property | Effect | Pass |
|---|---|---|
| Soundness bits (G-4, RFC-0014) | none: the messages are RFC-0010's | yes |
| Zero knowledge (RFC-0013) | none: the unmasked part of the same polynomials | yes |
| Verifier independence | none: prover only | yes |
| Secrets | `Q`, `K`, their folds and the row tables live in `SecretBuf` | yes, by construction |

## Triage

- **Priority.** 0.8: it removes a context-dependent working set from stage 1 that blocks 405B proofs at any context.
- **First killer.** The candidate's round polynomials differ from the literal prover's on some shape: padding, grouped query heads, or a key block straddling the row. [EXP-0026](../experiments/EXP-0026-causal-score-rounds/README.md) checks it exactly.
