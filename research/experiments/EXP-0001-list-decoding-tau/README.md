# EXP-0001: Does a provable list-decoding bound reduce `τ_A` below 155?

| | |
|---|---|
| Status | `done: refuted` |
| Idea | [IDEA-0001](../../ideas/IDEA-0001-list-decoding-proximity.md) |
| Evidence level | E0 (analytic) |
| Hardware | none |
| Pre-registered on | 2026-10-16 |

## Hypothesis

Accounting the Ligero-family proximity test of RFC-0011 with the Johnson-regime case of [BCIKS20] Theorem 1.2 lowers `τ_A` below 155 at 64 live bits while the theorem's error term stays at or below 2^-64.

## Success criterion

- `validated`: some admissible `η` gives `τ_A ≤ 140` (at least 10% fewer opened columns) with a field-size term of at least 64 bits for the advice commitment, and the weight commitment keeps at least 64 bits at its own `τ_W`.
- `refuted`: at every admissible `η` the field-size term is below 64 bits, or `τ_A` stays at or above 150.
- `inconclusive`: the theorem's constants cannot be read from a primary source.

Pre-registration caveat, recorded for honesty: before this criterion was written, the author had run an earlier version of the calculator whose Johnson formula was recalled from memory (parametrized by an integer `m`, with a row factor); it reported field terms between 16 and 58 bits. The criterion above was fixed before the exact statement was transcribed from the paper and run. The recalled formula is gone from the tool.

## Method

- Code: [research/tools/proximity.py](../../tools/proximity.py), function `johnson`, which evaluates Theorem 1.2 (Johnson regime), Theorem 1.7 (affine space: no factor in the number of rows) and Remark 1.5 (`q' = |Fp2|`) of [BCIKS20], transcribed from the ECCC report TR20-083 dated May 30, 2020, pages 2 to 4, read on 2026-10-16.
- Inputs: the `live-64/1` and `ni-100/1` parameter sets of RFC-0014 (advice: `log2 k = 22`, rate 1/2, `P_A` = 160 or 200; weights: `k_W = 2^20`, rate 1/4, `P_W = 2^16`); the degree bound `k' = k + P − 1`; `q' = 2^128`; `η ∈ {0.1, 0.05, √ρ'/20, 0.02, 0.01, 0.005, 0.001}`.
- Baseline in the same harness: the unique-decoding case of the same theorem, which must reproduce RFC-0014's `τ` values (155, 97, 193, 122). This is the correctness check: a transcription error in the shared code would move them.
- Output: `τ` and the field-size term in bits for each `η`, and the `τ` the capacity conjecture would give (reported for scale only; conjectures are inadmissible under RFC-0014 and were reported false in general in 2025).

## Reproduction

```sh
python3 -I research/tools/proximity.py --johnson > research/experiments/EXP-0001-list-decoding-tau/results/johnson.txt
```

Environment: Python 3.14.6, macOS (Darwin 25.6.0), Apple M4 Max, 2026-10-16. No measurement of time; the result is exact arithmetic on the theorem's bound.

## Results

Correctness check (unique decoding): `τ_A = 155` (64.3 bits), `τ_W = 97` (64.0 bits), `ni-100` 193 (80.1 bits) and 122 (80.5 bits), all equal to RFC-0014's table.

Johnson regime, `live-64` advice (`ρ' = 0.500019`, unique decoding gives 155):

| `η` | `δ` | bits per query | `τ_A` | field-size term (bits) |
|---|---|---|---|---|
| 0.1, 0.05 | below the unique-decoding radius: the first case of the theorem applies | | | |
| 0.0354 (= `√ρ'/20`) | 0.2575 | 0.4296 | 149 | 57.2 |
| 0.02 | 0.2729 | 0.4597 | 140 | 51.5 |
| 0.01 | 0.2829 | 0.4797 | 134 | 44.5 |
| 0.005 | 0.2879 | 0.4898 | 131 | 37.5 |
| 0.001 | 0.2919 | 0.4979 | 129 | 21.2 |

Johnson regime, `live-64` weights (`ρ' = 0.265625`, unique decoding gives 97): `τ_W` falls from 92 (`η = 0.1`) to 68 (`η = 0.001`), the field-size term from 57.9 to 25.1 bits, 57.9 bits being its maximum at every `η ≥ √ρ'/20 = 0.0258`.

Capacity conjecture at `η = 0.01` (scale only, inadmissible): `τ_A = 66`, `τ_W = 35`.

Raw output: [results/johnson.txt](results/johnson.txt).

## Analysis

The error term `ε_J = (k'+1)^2 / ((2 · min(η, √ρ'/20))^7 · q')` is smallest when `η = √ρ'/20`, and there it equals about 2^-57.2 for the advice commitment and 2^-57.9 for the weights: below the 64-bit live target at every `η`, so the criterion's `refuted` clause holds before `τ` is even considered. At that `η` the saving is 155 → 149 opened columns (4%); pushing `η` down to reach `τ_A = 129` collapses the field-size term to 21 bits. The non-interactive sets need 100 bits and are further out of reach. The rate-1/4 weight commitment is in the same position.

Threats to validity:

- The ECCC May 2020 statement was read; the JACM 2023 version may carry different constants. The margin (57 against 64 bits, and the 7th-power dependence on the cap) is large enough that a constant-factor change does not alter the verdict; a change in the exponent of `n` or in the cap `√ρ/20` would, and the `proximity` survey checks for one.
- Tessera's rows are codewords over `Fp` with vanishing-mask coefficients; the degree bound `k + P − 1` was used, and the masks only enlarge it.
- Theorem 1.7 (affine spaces) was applied because RFC-0011 combines rows with independent uniform `Fp2` coefficients. An opening that used powers of a single challenge would be the parametrized-curve case of Theorem 1.6 and pay a factor `l = 5.8×10^5` (about 19 bits), which would be worse.
- Newer results (2025–2026) are out of scope here by design; they are queue item 1.

## Verdict

`refuted`. IDEA-0001 moves to `refuted`; RFC-0014 Alternatives Considered 3 is confirmed with numbers rather than asserted. The reopening condition is recorded in the idea note.

## Follow-ups

- The `proximity` survey reads arXiv 2609.05870 (a list bound independent of the field size and a mutual correlated agreement result are claimed), ePrint 2025/2046, ePrint 2026/680, ePrint 2025/870 and arXiv 2605.07595. If one of them states a conjecture-free proximity gap for Reed–Solomon codes with an error term independent of, or linear in, `n/q'` near `δ = 1 − ρ'`, a new experiment evaluates it with the same tool: the prize is `τ_A` near 66 and about 0.4 GB off the proof.
- The rate trade-off (queue): rate 1/4 for the advice gives `τ_A = 95` under unique decoding, but doubles the encoded and hashed volume; the cost model prices the `c_c` increase against the 0.28 GB saved.
