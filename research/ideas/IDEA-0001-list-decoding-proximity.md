# IDEA-0001: List-decoding proximity analysis to cut the opened-column count `τ_A`

| | |
|---|---|
| Status | `refuted` by [EXP-0001](../experiments/EXP-0001-list-decoding-tau/README.md) under the provable bound of [BCIKS20]; the reopening condition in "Decision and links" fired on 2026-10-16 and the successor is [IDEA-0012](IDEA-0012-johnson-regime-tau-linear-error.md) |
| Origin | [baseline.md §4](../baseline.md#4-parameters-proof-size-verification-e0) and the "Reopens if" column of [§7](../baseline.md#7-already-rejected) (RFC-0014 Alternatives Considered 3); RISK-4 |
| Lever | `τ`; proof size (the opened columns are 0.72 GB of the 2 GB proof); verification (hashing the opened columns) |
| Spec surfaces | RFC-0014 'Soundness budget' and 'Parameter sets'. Parameter sets are tier A (02 §1): a superseding RFC |
| Priority | 0 after EXP-0001: gain at most 4% of the opened columns, confidence 0.9 that the bound is unusable |

## Mechanism

Keep the Ligero-family test of RFC-0011 as it is (uniform `Fp2` combination coefficients over the rows, `τ` column queries drawn after the combined rows are fixed), but account its soundness with the Johnson-regime case of [BCIKS20] Theorem 1.2 instead of the unique-decoding case. A combined row that is `δ`-far from the code is caught by one query with probability `δ`; the Johnson regime allows any `δ < 1 − √ρ' − η` instead of `δ ≤ (1 − ρ')/2`, so each query yields up to `−log2(√ρ' + η)`, about 0.5 bits at rate 1/2, instead of 0.415 (the figure RFC-0014 Alternatives Considered 3 quotes). The theorem's error term `ε_J` joins the soundness budget as a one-shot event.

## Why it could work

- [BCIKS20] Theorem 1.2 (primary, E0): the collection of affine spaces displays a proximity gap for every `δ ∈ (0, 1 − √ρ)`, with `ε_J = (k + 1)^2 / ((2 · min(η, √ρ/20))^7 · q)` in the Johnson regime, `η := 1 − √ρ − δ`.
- [BCIKS20] Remark 1.5 (primary, E0): coefficients sampled from an extension field `F_q'` replace `q` by `q'` in the denominators. Tessera combines rows with `Fp2` coefficients (RFC-0011 'Opening protocol'), so `q' ≈ 2^128`.
- [BCIKS20] Theorem 1.7 (primary, E0): for a uniformly random element of the affine space spanned by the rows, the error carries no factor in the number of rows (5.8×10^5 for the advice at 16K tokens); only the parametrized-curve version (Theorem 1.6) pays a factor `l`.

## Expected gain

The cost model does not see `τ`: opened columns cost the prover nothing in `μ`. The metrics moved are proof size and verifier work.

```sh
python3 -I research/tools/proximity.py --johnson
```

| Context | Δμ | Δ proof size | Δ verification |
|---|---|---|---|
| all | 0 | at most −1.5% (`τ_A` 155 → 149, opened columns 0.72 → 0.69 GB); −8% if the error term could be ignored (`τ_A` → 129) | about −2% (hashing of opened columns) |

Inputs: E0 from the theorem's statement, read from the primary source on 2026-10-16.

## Security gate

| Property | Effect | Pass |
|---|---|---|
| Soundness bits (G-4, RFC-0014) | the Johnson-regime error `ε_J` is at least 2^-57.2 for the advice commitment and 2^-57.9 for the weight commitment at every `η` (the `min(η, √ρ'/20)` cap stops it improving), below the 64-bit live target and far below the 100-bit non-interactive target | fail |
| Uniqueness upstream of the logits (SP-3) | unchanged | pass |
| Zero knowledge (G-5) | unchanged: the vanishing masks and blinding rows do not depend on the proximity analysis | pass |
| Verifier independence (SP-6) | unchanged | pass |
| Post-quantum | unchanged | pass |

## Prior rejection check

RFC-0014 Alternatives Considered 3: "Johnson-bound (list-decoding) proximity analysis. Considered: about 0.5 bits per query at rate 1/2 instead of 0.415. Rejected: provable bounds in that regime carry field-size terms that erode the margin, and conjectured bounds are not acceptable for treaty use." New now: the field-size term is quantified (EXP-0001) instead of asserted, and three 2025–2026 results change what is provable: Crites and Stewart's negative results on the conjectures up to capacity (ePrint 2025/2046), the "optimal proximity gaps" preprint with a list bound independent of the field size (arXiv 2609.05870), and Arnon's open-problems survey (ePrint 2026/680). The `proximity` survey reads them; this idea does not depend on them.

## Kill criterion

The error term of the best provable list-decoding proximity gap exceeds 2^-64 for `n = 2^23`, `k' ≈ 2^22`, `q' = 2^128` at every `η` for which the Johnson radius exceeds the unique-decoding radius.

## Cheapest experiment

E0: evaluate Theorem 1.2 exactly over `η` with `research/tools/proximity.py --johnson`, with the unique-decoding case as the in-harness baseline that must reproduce RFC-0014's `τ` values. Effort `s`. An H100 settles nothing here; the question is mathematical.

## Refutations

Self-triage by the authoring agent (one agent, four lenses, flagged as such; a sweep uses separate refuters).

| Lens | Verdict | Argument |
|---|---|---|
| soundness | refuted | The error term caps at 57–58 bits because `min(η, √ρ'/20)` stops improving at `η = √ρ'/20`; smaller `η` buys distance, never field bits. A one-shot 2^-57 event is below the live target the parameter set must meet (RFC-0014 'Parameter sets'). |
| prior art | weakened | RFC-0014 Alternatives Considered 3 anticipated the conclusion qualitatively. The quantification, and the dependence on the cap `√ρ'/20`, are new. |
| cost model | weakened | `τ` moves proof size and verifier time, not `μ`; the largest admissible gain is 4% of the 0.72 GB of opened columns. |
| feasibility | survives | Pure analysis; done in EXP-0001. |

## Score

| Gain | Confidence | Cost | Priority |
|---|---|---|---|
| at most −1.5% proof size (E0) | 0.9 that the bound is unusable | `s` | 0 |

## Decision and links

Experiments: [EXP-0001](../experiments/EXP-0001-list-decoding-tau/README.md), `done: refuted`. RFC or issue: none; RFC-0014 Alternatives Considered 3 stands, now with numbers. Reopens if the `proximity` survey finds a conjecture-free proximity gap for Reed–Solomon codes whose error term is independent of, or at most linear in, `n / q'` at `δ` near `1 − ρ'` (the claim to read in arXiv 2609.05870); then a new experiment evaluates that statement with the same tool. Ledger line updated on 2026-10-16. Reopened the same day: the `proximity` survey ([SUR-0001](../surveys/SUR-0001-proximity.md)) found [BCHKS25] and [DKT26], and the successor [IDEA-0012](IDEA-0012-johnson-regime-tau-linear-error.md) is `validated` by [EXP-0003](../experiments/EXP-0003-johnson-linear-tau/README.md) (`τ_A` 155 → 129–130 at 77–80 one-shot bits, E0).
