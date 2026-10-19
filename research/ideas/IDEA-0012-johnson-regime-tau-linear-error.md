# IDEA-0012: Johnson-regime proximity accounting under the linear-error bounds of [BCHKS25] and [DKT26], with uniform `Fp2` combination coefficients

| | |
|---|---|
| Status | `promoted`: Draft [RFC-0021](../../docs/rfcs/RFC-0021-johnson-regime-proximity.md) (2026-10-16), after `validated` by [EXP-0003](../experiments/EXP-0003-johnson-linear-tau/README.md) (the parameters) and [EXP-0005](../experiments/EXP-0005-list-decoding-extractor/README.md) (the extractor argument, awaiting a second reader: OQ-0021-1) |
| Origin | [SUR-0001](../surveys/SUR-0001-proximity.md) candidate ideas 1 and 2; the reopening condition of [IDEA-0001](IDEA-0001-list-decoding-proximity.md) (a conjecture-free bound with an error term linear in `n/q'`), met by [BCHKS25] and [DKT26]; RFC-0014 Alternatives Considered 3; RISK-4 |
| Lever | `τ`; proof size (opened columns 0.72 → 0.60 GB live, 0.90 → 0.75 GB non-interactive); verification (hashing of opened columns and the `(G + 1) · τ_A` inner products fall by 16%) |
| Spec surfaces | RFC-0011 'Opening protocol' step 3 (proximity coefficients `eq(ξ*, row)` → independent uniform `Fp2` values) and the `pcs/advice/xi` entry of RFC-0014 'Transcript schedule'; RFC-0014 'Soundness budget' (proximity row) and 'Parameter sets' (`τ_A`, `τ_W`); the soundness calculators of RFC-0014 'Soundness calculator' must evaluate `E₀`. Parameter sets and the transcript schedule are tier A (02 §1): a superseding RFC |
| Priority | 0.9 on proof size (−6% × 0.6 / 4); 0 on `Δμ`, which the lever does not touch |

## Mechanism

1. **Generator.** Draw the proximity coefficients `ξ_row` as independent uniform `Fp2` values from a transcript-seeded stream after Round 1 (the verifier already evaluates `5.8×10^5` coefficients per group for `c_g`, so deriving as many for `ξ` is within its budget), instead of `eq(ξ*, row)`. The combined row `u_ξ = Σ_row ξ_row · A_row` is then a uniform element of the affine space spanned by the rows, which is the object the Johnson-regime theorems speak about.
2. **Accounting.** Replace the proximity row of RFC-0014 'Soundness budget', `((1 + ρ')/2)^τ + n/|Fp2|`, by `(1 − γ)^τ + E₀/(|Fp2| − 1)` with `γ = 1 − √ρ' − η` and `E₀ = johnsonExceptionCount(n, D, ⌈(√ρ' + η) n⌉, η)` of [ArkLib26] (`D = k + P − 1`), choosing `η` per parameter set (0.001 to 0.003 for the advice, 0.005 for the weights).
3. **Parameters.** `live-64`: `τ_A` 155 → 129 (77.4 bits one-shot at `η = 0.001`) or 130 (80.4 bits at `η = 0.002`), `τ_W` 97 → 68 (86.0 bits). `ni-100`: `τ_A` 193 → 162 (82.2 bits before grinding), `τ_W` 122 → 85 (86.0 bits). The bits per query rise from 0.415 to 0.496 (advice) and from 0.660 to 0.942 (weights).

## Why it could work

- [BCHKS25] Theorem 1.5 (primary, E0; ECCC TR25-169 page 9, read 2026-10-16): correlated agreement for lines up to the Johnson radius with `a = ((2t^5 + 3tγρ)/(3ρ^{3/2})) n + t/√ρ` bad coefficients, linear in `n`; Theorem 4.6 (page 28) is the list form with the same `a` for lines. At `live-64` advice and `η = 0.003`: 69.6 bits, `τ_A = 130`.
- [DKT26] as formalized in [ArkLib26] (primary, E0; Lean statements read at commit `4dae3acff846125521b7b3998010885071746f16`): `mcaError_affineLine_johnson_le` bounds the affine-line mutual-correlated-agreement error at radius `1 − √ρ − η` by `min(1, E₀/|F|)` with no characteristic hypothesis; `mcaError_affineSpace_le_of_exactAgreement` carries it to affine spaces of every dimension at `E₀/(|F| − 1)`; `johnsonExceptionCount_lt_closed` gives `E₀ < (8/3) n t³/ρ`. At `live-64` advice and `η = 0.001`: 77.4 bits, `τ_A = 129`.
- [GG25] Lemma 2.15 (primary, E0; ECCC TR25-166 page 8): a lossless line correlated agreement with list size below `q'` and `δ < Δ(C)` transfers to affine spaces at `err · q'/(q' − 1)`: the row count (5.8×10^5) enters nowhere.
- The Johnson radius is the ceiling: at rate 1/2 the bits per query tend to 0.5, so `τ_A ≥ 128` at 64 bits; the idea takes 129 to 130 of a possible 128. Capacity (`τ_A = 64`) is closed for RS codes over smooth cosets of prime fields ([CS25], [KKH26], [Kam26]).

## Expected gain

```sh
python3 -I research/tools/proximity.py --bchks25 --arklib
```

| Context | Δμ | Δ proof size | Δ verification |
|---|---|---|---|
| all | 0 | −6% live (opened columns 0.72 → 0.60 GB of 2 GB), −7% non-interactive (0.90 → 0.75 GB) | −16% of the column hashing and of the `(G + 1) · τ_A` inner products; the `G + 1` coset NTTs unchanged |

Inputs: E0 from the primary statements, evaluated exactly in [EXP-0003](../experiments/EXP-0003-johnson-linear-tau/README.md).

## Security gate

| Property | Effect | Pass |
|---|---|---|
| Soundness bits (G-4, RFC-0014) | field term 77.4 bits (advice) and 86.0 bits (weights) one-shot at the chosen `η`, above the 64-bit live target; 82.2 and 86.0 bits before the 20 grinding bits of `ni-100`. The bound is the mutual form, machine-checked. The extractor argument from RFC-0011's evaluation and column checks to one decoded matrix is written in [EXP-0005](../experiments/EXP-0005-list-decoding-extractor/README.md): list binding plus per-proof consistency, extra terms below 2^-107; it awaits an adversarial second reading, and it changes the wording of SP-3 for this commitment (one decoded matrix per accepted proof, not one message per commitment) | pass (pending soundness review of EXP-0005) |
| Uniqueness upstream of the logits (SP-3) | unchanged | pass |
| Zero knowledge (G-5, RFC-0013) | fewer columns opened (130 ≤ `P_A = 160`, 68 ≤ `P_W`), so the vanishing masks hide at least as much; the combined row `u_ξ` is masked by `M_ξ` as before; the coefficient stream is public | pass |
| Verifier independence (SP-6) | the verifier evaluates `E₀` from the Lean definitions in its own calculator and derives the coefficient stream from the transcript; no prover code | pass |
| Post-quantum | unchanged (hash-based) | pass |

## Prior rejection check

RFC-0014 Alternatives Considered 3 rejected Johnson-bound analysis because "provable bounds in that regime carry field-size terms that erode the margin, and conjectured bounds are not acceptable for treaty use". EXP-0001 confirmed the first clause for [BCIKS20] (57 bits at best). New now: [BCHKS25] (November 2025) and [DKT26] (September 2026, Lean-checked) replace the `n²/(η^7 q')` term by `n/(η^5 q')` and `n/(η^3 q')` terms, which clear 64 bits with 13 bits to spare; the second clause (no conjectures) is kept, since the Johnson regime needs none. [BCGM26] Theorem 5 covers the eq generator RFC-0011 uses today, but at about `2^-54` (SUR-0001), which is why the mechanism switches the generator.

## Kill criterion

The list-decoding extractor for RFC-0011's opening cannot bind every claim to one decoded matrix without an additional error term that drops the proximity row below 64 one-shot bits; or RFC-0011 must keep the eq generator and no Johnson-regime bound for it reaches 64 bits.

## Cheapest experiment

Done: [EXP-0003](../experiments/EXP-0003-johnson-linear-tau/README.md) (the numbers) and [EXP-0005](../experiments/EXP-0005-list-decoding-extractor/README.md) (the extractor argument), both E0, no hardware. Next: `promote`, with a second reader's `soundness-review` of EXP-0005 before the RFC leaves Draft.

## Refutations

Self-triage by the authoring agent (one agent, four lenses, flagged as such).

| Lens | Verdict | Argument |
|---|---|---|
| soundness | survives (pending review) | The field term is settled by machine-checked statements; the extractor argument is written in EXP-0005 from the exact-agreement form of mutual correlated agreement, with the `Fp`/`Fp2` coefficient lemma and the list bound, and adds no term above 2^-107; a second reader must check the two steps read from the Lean documentation rather than the proofs. |
| prior art | survives | RFC-0014's reason was true of [BCIKS20] and is false of [BCHKS25] and [DKT26]; the rule it protects (no conjectures) is respected. Deployed systems report 4.6–11.1% proof-size savings from the same bounds ([DKT26], `claim`). |
| cost model | weakened | `τ` moves proof size and verifier time, not `μ`; the gain is 6–7% of the proof and 16% of the verifier's column work. |
| feasibility | survives | E0 complete; the RFC is tier-A work (generator, budget row, parameter sets) with the soundness calculators to extend. |

## Score

| Gain | Confidence | Cost | Priority |
|---|---|---|---|
| −6% proof size, −16% verifier column work (E0) | 0.6 (the bound is proven; the extractor argument is unwritten) | `l` | 0.9 on size; 0 on `Δμ` |

## Decision and links

Experiments: [EXP-0003](../experiments/EXP-0003-johnson-linear-tau/README.md), `done: validated`; [EXP-0005](../experiments/EXP-0005-list-decoding-extractor/README.md), `done: validated` (the extractor argument, for review). Promoted on 2026-10-16: Draft [RFC-0021](../../docs/rfcs/RFC-0021-johnson-regime-proximity.md) supersedes RFC-0011 'Opening protocol' step 3 and its soundness paragraph and RFC-0014 'Soundness budget' and 'Parameter sets' (`live-64/2`, `ni-100/2`), with `E₀` to be implemented in both soundness calculators; acceptance waits on the second reading of EXP-0005 and the maintainer's pull request (`issue-workflow`). Spec note filed with EXP-0003: the current proximity row's `n/|Fp2|` is the affine-space bound, while the eq generator's known bound is `20 · n/|Fp2| = 2^-100.7` ([DG24] Theorems 2 and 3); targets still met; an issue. Ledger line updated on 2026-10-16.
