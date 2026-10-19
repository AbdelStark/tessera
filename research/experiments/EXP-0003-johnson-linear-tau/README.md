# EXP-0003: Does a conjecture-free Johnson-regime bound with error linear in `n` reduce `τ_A` below 155?

| | |
|---|---|
| Status | `done: validated` |
| Idea | [IDEA-0012](../../ideas/IDEA-0012-johnson-regime-tau-linear-error.md) |
| Evidence level | E0 (analytic) |
| Hardware | none |
| Pre-registered on | 2026-10-16 |

## Hypothesis

Accounting the proximity test of RFC-0011 with the Johnson-regime correlated-agreement theorem of [BCHKS25] (Theorem 1.5: the number of bad combination coefficients is linear in `n`) or with the machine-checked Johnson mutual-correlated-agreement bound of [DKT26] as stated in [ArkLib26] lowers `τ_A` below 155 at 64 live bits while the bound's field-size term stays at or above 64 bits, and lowers `τ_W` and the non-interactive counts in the same way.

## Success criterion

- `validated`: some admissible `η` gives `τ_A ≤ 140` (at least 10% fewer opened columns) with a field-size term of at least 64 bits for the `live-64` advice commitment, and the weight commitment keeps at least 64 bits at its own reduced `τ_W`; for `ni-100`, some `η` gives `τ_A ≤ 175` with at least 80 one-shot bits (RFC-0014 counts 20 grinding bits on top).
- `refuted`: at every admissible `η` the field-size term is below the target, or `τ_A` stays at or above 150.
- `inconclusive`: a constant of either statement cannot be read from a primary source.

Pre-registration disclosure: the `proximity` survey ([SUR-0001](../../surveys/SUR-0001-proximity.md)) reports scratch evaluations of both bounds (`τ_A = 130` at 66.7 or 80.2 bits) made by the survey agent from its own transcription. This experiment was pre-registered after reading that survey, so the criterion is not blind to the expected outcome; what it adds is an evaluation from the primary statements (the ECCC report's Theorem 1.5 and the Lean definitions of `johnsonExceptionCount`, both read on 2026-10-16), in the same tool as the unique-decoding baseline, with the exact exception count rather than the closed asymptotic bound, and a check of the two transfer lemmas (lines to affine spaces; lines to the eq generator) that decide whether the bounds apply to Tessera's test.

## Method

- Code: [research/tools/proximity.py](../../tools/proximity.py), modes `--bchks25` and `--arklib`, which evaluate:
  - [BCHKS25] Theorem 1.5 as stated in ECCC TR25-169 (7 November 2025), page 9: for `γ ∈ (0, 1 − √ρ)`, `η = 1 − √ρ − γ`, `m = max(⌈√ρ/(2η)⌉, 3)`, `t = m + 1/2`, correlated agreement holds for every pair of words once more than `a = ((2t^5 + 3tγρ)/(3ρ^{3/2})) · n + t/√ρ` coefficients `z` give a `γ`-close combination; the soundness error is `a/q'`. Theorem 4.6 (page 28) gives the list (mutual) form for degree-`M` curves with `|E| ≤ M · a`; `M = 1` is the line. Here `ρ = D/n` with `D` the maximum degree (the paper's `k`; the code's dimension is `D + 1`).
  - [ArkLib26] `johnsonExceptionCount n D A η` (`ArkLib/Data/CodingTheory/HiddenDerivative/Parameters/Johnson/FiniteBounds.lean`): `m = max(⌈√ρ₋/(2η)⌉, 3)`, `t = m + 1/2`, `μ = ⌈t/√ρ₋⌉ − 1`, `h = ⌈t²/(3ρ₋)⌉ − 1`, `θ = (n − D)/(A − D)`, `E₀ = (2μ − 1)h + θ(h + μ + 4Dμh) + (n − D − 1)μ`, with `A = ⌈(√ρ₋ + η) n⌉` and `ρ₋ = D/n`; its closed bound `E₀ < (8/3) n t³/ρ₋` (`johnsonExceptionCount_lt_closed`, hypotheses `1 ≤ D ≤ n − 2`, `η > 0`, `(√ρ₋ + η) n ≤ A`). The affine-line MCA error at radius `1 − √ρ₋ − η` is `min(1, E₀/|F|)` (`mcaError_affineLine_johnson_le`, `ArkLib/Data/CodingTheory/ReedSolomon/MutualCorrelatedAgreement/Johnson/Probability.lean`, no characteristic hypothesis), and the affine-space MCA error in every dimension is `E₀/(|F| − 1)` (`mcaError_affineSpace_le_of_exactAgreement`, `LineToAffine.lean`, applied to the `LineExactAgreementBound` that the line theorem's proof constructs). Read at commit `4dae3acff846125521b7b3998010885071746f16` of `Verified-zkEVM/ArkLib`.
- Transfer lemmas checked in the primary sources: [GG25] Lemma 2.13 and Lemma 2.15 (ECCC TR25-166 rev. 1, page 8: a line proximity gap or a lossless line correlated agreement with list size below `q` and `δ < Δ(C)` gives the affine-space form at error `err · q/(q − 1)`) and Remark 2.16 (no such reduction was known to its authors for mutual correlated agreement; [ArkLib26] `LineToAffine.lean` now supplies one); [DG24] Theorem 2 and Theorem 3 (CiC 1(4), pages 7 and 10: within the unique-decoding radius, a code with affine-line gaps at false-witness bound `ε ≥ e + 1` has tensor-style gaps for `ϑ` random values at `ϑ · ε/q`).
- Inputs: the `live-64/1` and `ni-100/1` parameter sets of RFC-0014 (advice `log2 k = 22`, rate 1/2, `P_A` 160 or 200; weights `k_W = 2^20`, rate 1/4, `P_W = 2^16`), `D = k + P − 1`, `|F| = q' = |Fp2| ≈ 2^128`, `η` swept over `{0.05, 0.02, 0.01, 0.005, 0.003, 0.002, 0.001, 0.0005}`.
- Baseline in the same harness: the unique-decoding case, which must reproduce RFC-0014's `τ` values (155, 97, 193, 122); a transcription error in the shared code would move them.
- Per `η`: `γ = 1 − √ρ − η`, bits per query `−log2(1 − γ)`, `τ = ⌈needed bits / bits per query⌉`, the field-size term in bits for each bound (`−log2(a/q')`, `−log2(E₀/(q' − 1))`, and the closed bound), and the opened-column bytes at that `τ`.
- Output: one table per parameter set; the best admissible `τ` per set (field term at or above the target) is the headline.

## Reproduction

```sh
python3 -I research/tools/proximity.py --bchks25 --arklib > research/experiments/EXP-0003-johnson-linear-tau/results/johnson-linear.txt
```

Environment: Python 3.14.6, macOS (Darwin 25.6.0), Apple M4 Max, 2026-10-16. Exact arithmetic on the theorems' bounds (`E₀` as a rational number); no timing.

## Results

Correctness check (unique decoding): `τ_A = 155` (64.3 bits), `τ_W = 97` (64.0 bits), `ni-100` 193 (80.1 bits) and 122 (80.5 bits), equal to RFC-0014's table.

`live-64` advice (`ρ = 0.500019`, 64 one-shot bits needed; unique decoding: `τ_A = 155`, 0.719 GB of opened columns):

| `η` | `γ` | bits per query | `τ_A` | opened GB | [BCHKS25] `a/q'` (bits) | [ArkLib26] `E₀/(q'−1)` (bits) | closed bound (bits) |
|---|---|---|---|---|---|---|---|
| 0.05 | below the unique-decoding radius: Corollary 1.4 / unique decoding applies | | | | | | |
| 0.02 | 0.2729 | 0.4597 | 140 | 0.650 | 83.0 | 90.3 | 90.0 |
| 0.01 | 0.2829 | 0.4797 | 134 | 0.622 | 78.1 | 87.3 | 87.0 |
| 0.005 | 0.2879 | 0.4898 | 131 | 0.608 | 73.3 | 84.4 | 84.1 |
| 0.003 | 0.2899 | 0.4939 | 130 | 0.603 | 69.6 | 82.2 | 81.9 |
| 0.002 | 0.2909 | 0.4959 | 130 | 0.603 | 66.7 | 80.4 | 80.2 |
| 0.001 | 0.2919 | 0.4979 | 129 | 0.599 | 61.7 | 77.4 | 77.2 |
| 0.0005 | 0.2924 | 0.4990 | 129 | 0.599 | 56.7 | 74.4 | 74.2 |

Best admissible: [BCHKS25] `τ_A = 130` (`η = 0.003`, 69.6 bits), [ArkLib26] `τ_A = 129` (`η = 0.001`, 77.4 bits): −16.1% and −16.8% opened columns, 0.72 → 0.60 GB.

`live-64` weights (`ρ = 0.265625`, unique decoding `τ_W = 97`): `τ_W = 68` for every `η ≤ 0.005` (−29.9%), with 75.1 bits ([BCHKS25], `η = 0.005`) or 86.0 bits ([ArkLib26]); at `η = 0.05` already 78 (90.2 / 95.3 bits).

`ni-100` advice (80 one-shot bits before 20 grinding bits; unique decoding 193, 0.896 GB): [BCHKS25] best `τ_A = 175` (`η = 0.02`, 83.0 bits, −9.3%, 0.812 GB); [ArkLib26] best `τ_A = 162` (`η = 0.003`, 82.2 bits, −16.1%, 0.752 GB). `ni-100` weights (unique decoding 122): [BCHKS25] 87 (`η = 0.01`, 80.1 bits), [ArkLib26] 85 (`η ≤ 0.005`, 86.0 bits).

Raw output: [results/johnson-linear.txt](results/johnson-linear.txt).

## Analysis

The field-size terms are linear in `n`, as claimed: `a ≈ (2t^5/(3ρ^{3/2})) · n` with `t ≈ √ρ/(2η)` falls 5 bits per halving of `η` (83.0 → 78.1 → 73.3), and `E₀ ≈ θ · 4Dμh ∝ n t³/ρ` falls 3 bits per halving (90.3 → 87.3 → 84.4); the exact `E₀` is 0.3 bits better than the closed bound. Against the [BCIKS20] term that EXP-0001 evaluated, which capped at 57.2 bits because of `min(η, √ρ/20)^7`, the new terms are 20 to 30 bits better at the same `η`, and that alone is the difference: the field term no longer binds before `τ` can move. The validated clause of the criterion holds for both statements on every parameter set; the mutual form ([ArkLib26]) is both the stronger property and the tighter bound.

The prize is bounded by the Johnson radius itself. Bits per query are `−log2(1 − γ)` with `γ < 1 − √ρ`, so at rate 1/2 the limit is `−log2(√(1/2)) = 0.5` bits per query and `τ_A ≥ 128` at 64 bits; `τ_A = 129` is within one query of it. Capacity (1 bit per query, `τ_A = 64`) is closed for RS codes over smooth multiplicative cosets of prime fields ([CS25], [KKH26], [Kam26], SUR-0001 synthesis 4). Everything the Johnson regime can give at rate 1/2 is 155 → 128, and this experiment gets 129 to 130 of it.

What moves: opened columns 0.72 → 0.60 GB (−6% of the 2 GB proof), 0.90 → 0.75 GB non-interactive; the verifier's hashing of opened columns and its `(G + 1) · τ_A` inner products of length 5.8×10^5 fall by the same 16%; its `G + 1` coset NTTs do not. `μ` does not move (the cost model does not see `τ`).

Conditions on which the result stands, each a soundness-review item before an RFC:

1. **The generator.** Both statements are for lines and, through [GG25] Lemma 2.15 (correlated agreement, lossless, list size below `q'`, `δ < Δ(C)`) or [ArkLib26] `mcaError_affineSpace_le_of_exactAgreement`, for affine spaces with independent uniform coefficients, at error `a/(q' − 1)` or `E₀/(q' − 1)` in every dimension. RFC-0011 step 3 draws `ξ_row = eq(ξ*, row)`: a tensor generator with `ϑ = ⌈log2 5.8×10^5⌉ = 20` random values, for which the known bound ([DG24] Theorems 2 and 3) holds only within the unique-decoding radius, at `ϑ · n/q' = 2^-100.7`, and the only Johnson-regime statement is [BCGM26] Theorem 5 at about `2^-54` (per SUR-0001, not re-derived here). The result therefore applies to Tessera after the opening protocol switches the proximity coefficients to independent uniform `Fp2` values (a transcript-seeded stream of `5.8×10^5` coefficients, the same count the verifier already evaluates for each `c_g`). Step 3 and the transcript schedule are tier A: a superseding RFC.
2. **Binding beyond unique decoding.** The Ligero argument RFC-0011 cites decodes each row uniquely. In the Johnson regime the committed rows may be close to several codeword matrices, and the evaluation checks must bind every claim to one of them; that is what mutual correlated agreement supplies ([ArkLib26] for affine spaces, machine-checked, no characteristic hypothesis; [BCHKS25] Theorem 4.6 for lines), but the extractor argument itself, from the column checks and the evaluation checks of RFC-0011 step 6 and 8 to one decoded matrix, is not written in any Tessera document. It is the gate before promotion; the numbers above are conditional on it.
3. **Base field against extension field.** Rows are `Fp`-valued, coefficients are in `Fp2`. The statements apply to `RS[Fp2, D, k']` with `D ⊆ Fp`; the close codewords they produce are `Fp2`-polynomials, and the one-line lemma that an `Fp2`-polynomial of degree below `k'` agreeing with an `Fp`-valued row on at least `k'` points of `D` has `Fp` coefficients (unique interpolation over `Fp`) brings them back to the committed rows. To be written in the RFC.
4. **Versions.** [BCHKS25] was read from the ECCC text of 7 November 2025; the ePrint 2025/2055 and STOC 2026 versions were not. [DKT26] was read from the Lean statements, not from the paper. A constant that differs in either changes the table; the proof-size conclusion survives any constant-factor change (each factor 2 is one bit against margins of 6 to 13 bits at the chosen `η`).
5. **Non-interactive mode.** The field terms of 82.2 bits (advice) and 86.0 bits (weights) are one-shot before grinding; with RFC-0014's 20 grinding bits on every retryable bound they clear 100.

Threats to validity beyond those: the mask is counted (`D = k + P − 1`, `ρ = 0.500019`, not 0.5); the opened-column bytes use RFC-0011's 5.8×10^5 rows at 16K tokens; the weight-matrix row count (4096) is the tool's assumption and affects only the (negligible) weight-opening bytes.

A spec note that falls out of condition 1, independent of this idea: RFC-0014's proximity row writes `n/|Fp2|` for the field term, which is the affine-space bound of [BCIKS20]; for the eq generator RFC-0011 actually draws, the known unique-decoding bound is `20 · n/|Fp2| = 2^-100.7` ([DG24] Theorems 2 and 3, read on 2026-10-16). Both parameter sets still meet their targets (100.7 one-shot bits, 120.7 with grinding), so no parameter changes; the budget row should name the tensor bound. An issue, not an RFC.

## Verdict

`validated` (E0). IDEA-0012 moves to `validated`; its promotion waits on the extractor argument (condition 2), which goes to `soundness-review` before an RFC that changes RFC-0011 step 3 (generator), RFC-0014 'Soundness budget' (the Johnson-regime row, with the exact `E₀`) and 'Parameter sets' (`τ_A` 155 → 129 or 130, `τ_W` 97 → 68, `ni-100` 193 → 162 and 122 → 85). RFC-0014 Alternatives Considered 3 was right about [BCIKS20] and about conjectures, and is now wrong as a matter of fact about "provable bounds in that regime"; its rule (no conjectures) is kept.

## Follow-ups

- Soundness review: write the list-decoding extractor for RFC-0011's opening from the affine-space MCA statement (every claim bound to one decoded matrix across groups), with the `Fp`/`Fp2` coefficient lemma; this is the gate.
- Superseding RFC draft (`promote IDEA-0012`): uniform `Fp2` proximity coefficients, the Johnson-regime budget row with `E₀` evaluated by `tessera params check` and `tessera-verify params` from the Lean definitions, new `τ` values, and the issue for the `ϑ · n/|Fp2|` correction of the current row.
- Read the ePrint and STOC versions of [BCHKS25] and the [DKT26] paper when accessible; pin the ArkLib commit in the RFC (the tool records `4dae3acff846125521b7b3998010885071746f16`).
- Rate 1/4 under the same bounds (EXP-0004): `τ_A` near 67 at twice the encoding volume.
