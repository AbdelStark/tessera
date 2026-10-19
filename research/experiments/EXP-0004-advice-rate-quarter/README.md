# EXP-0004: Does rate 1/4 for the advice commitment buy proof size cheaply enough in `μ`?

| | |
|---|---|
| Status | `done: refuted` |
| Idea | [IDEA-0013](../../ideas/IDEA-0013-advice-rate-quarter.md) |
| Evidence level | E0 (analytic) |
| Hardware | none |
| Pre-registered on | 2026-10-16 |

## Hypothesis

Encoding the advice at rate 1/4 instead of 1/2 cuts the proof by at least 10% (fewer opened columns) at a cost in `μ` below 10% at 16K tokens under the corrected cost-model profile.

## Success criterion

- `validated`: proof size falls by at least 10% and `Δμ` at 16K stays at or below +10% for the `c_c` the volume implies.
- `refuted`: `Δμ` at 16K exceeds +10% for a proof-size gain below 15%.
- `inconclusive`: the effect of the rate on `c_c` cannot be bounded from the tools and IDEA-0004's counts.

Pre-registration disclosure: the `τ` and `Δμ` figures of this experiment were computed while scoping the ledger's queue item 5 (bootstrap run and the sweep day), before this criterion was written; the criterion was then fixed with those numbers in view. The experiment is recorded so that the queue item closes with its numbers on file rather than in a session transcript.

## Method

- Code: [research/tools/proximity.py](../../tools/proximity.py) `--sweep-rates` (unique decoding, rates 1/2 to 1/16) and `--log2-k 22 --rate 1/4 --mask 160 --rows 580000 --bits 64 --bchks25 --arklib` (Johnson regime at rate 1/4); [research/tools/cost_model.py](../../tools/cost_model.py) `--set commit_ps=60` and `--set commit_ps=45` against the corrected profile.
- The `c_c` bracket: at rate 1/4 the coset NTT doubles in size (`4k` against `2k`) while the inverse NTT of size `k` is unchanged, and the hashed symbols per committed element double. With IDEA-0004's measured encoder-to-hash split of 63:37 (E1, CPU) and its butterfly counts (11 for the inverse NTT, 23 for the coset NTT per element at rate 1/2; 11 and 46 at rate 1/4), the factor is `0.63 × 57/34 + 0.37 × 2 ≈ 1.8`, so `c_c` 30 → about 54 ps; 45 ps (hashing-only growth) and 60 ps (everything doubles) bracket it.
- Baseline in the same harness: rate 1/2 must give `τ_A = 155` (RFC-0014), and the corrected profile must print its baseline `μ` (8.9, 14.1, 35.1, 171.4).

## Reproduction

```sh
python3 -I research/tools/proximity.py --sweep-rates > research/experiments/EXP-0004-advice-rate-quarter/results/rate-sweep.txt
python3 -I research/tools/proximity.py --log2-k 22 --rate 1/4 --mask 160 --rows 580000 --bits 64 --bchks25 --arklib > research/experiments/EXP-0004-advice-rate-quarter/results/rate-quarter-johnson.txt
python3 -I research/tools/cost_model.py --set commit_ps=60 > research/experiments/EXP-0004-advice-rate-quarter/results/cost-commit-60.txt
python3 -I research/tools/cost_model.py --set commit_ps=45 > research/experiments/EXP-0004-advice-rate-quarter/results/cost-commit-45.txt
```

Environment: Python 3.14.6, macOS (Darwin 25.6.0), Apple M4 Max, 2026-10-16. Exact arithmetic; no timing.

## Results

Correctness checks: rate 1/2 gives `τ_A = 155` (64.3 bits); the corrected profile prints `μ` = 8.9 / 14.1 / 35.1 / 171.4 at 1K / 4K / 16K / 128K.

Rate against `τ_A` and opened columns (`live-64` advice, unique decoding):

| Rate | `n` | `ρ'` | bits per query | `τ_A` | opened GB | encode-and-hash volume |
|---|---|---|---|---|---|---|
| 1/2 | 2^23 | 0.500019 | 0.4150 | 155 | 0.719 | 2× |
| 1/4 | 2^24 | 0.250010 | 0.6781 | 95 | 0.441 | 4× |
| 1/8 | 2^25 | 0.125005 | 0.8301 | 78 | 0.362 | 8× |
| 1/16 | 2^26 | 0.062502 | 0.9125 | 71 | 0.329 | 16× |

Johnson regime at rate 1/4 ([ArkLib26] bound): `τ_A = 65` at `η = 0.005` (84.0 bits), 0.302 GB; [BCHKS25]: 65 at 73.2 bits.

`μ` under the corrected profile:

| Context | baseline | `c_c = 45 ps` | `c_c = 60 ps` |
|---|---|---|---|
| 1K | 8.9 | 9.8 (+10.3%) | 10.7 (+20.6%) |
| 4K | 14.1 | 15.6 (+10.8%) | 17.1 (+21.5%) |
| 16K | 35.1 | 38.9 (+10.6%) | 42.6 (+21.2%) |
| 128K | 171.4 | 187.8 (+9.6%) | 204.3 (+19.2%) |

Raw output: [results/rate-sweep.txt](results/rate-sweep.txt), [results/rate-quarter-johnson.txt](results/rate-quarter-johnson.txt), [results/cost-commit-60.txt](results/cost-commit-60.txt), [results/cost-commit-45.txt](results/cost-commit-45.txt).

## Analysis

The proof shrinks by 0.28 GB (14% of 2 GB; 21% under the Johnson bounds, where rate 1/4 reaches `τ_A = 65`), and `μ` grows by 10.6% to 21.2% at 16K, with the operation-count estimate (`c_c ≈ 54 ps`) near +17%. The criterion's `refuted` clause holds: more than +10% of `μ` for less than 15% of the proof under unique decoding. The rate is a knob that trades the primary metric for the secondary one at a poor exchange rate, because `c_c` is a fifth of `μ` at every context while the opened columns are a third of the proof. RISK-8's hash budget, about 530 GB/s per H100 at rate 1/2, doubles to about 1,060 GB/s, which no measured BLAKE3 kernel approaches on H100 (IDEA-0004 and SUR-0002 count the kernel as it is specified at 2–4× the 30 ps assumption already).

Threats to validity: the `c_c` factor is an operation count anchored on a CPU split (E1), not a GPU measurement; the cost model's streaming and regeneration terms are unaffected by the rate, which is right for the advice (the block buffer doubles but stays resident); diminishing returns past rate 1/4 (78 and 71 columns at 1/8 and 1/16) make the steeper rates strictly worse. A deployment that pays for verification rather than proving reads the same numbers the other way; that is why the idea is shelved rather than closed.

## Verdict

`refuted` as a default. IDEA-0013 moves to `refuted` with the `shelved` reading for size-bound deployments; queue item 5 closes. [IDEA-0012](../../ideas/IDEA-0012-johnson-regime-tau-linear-error.md) delivers 0.12 GB of the same saving at no prover cost and comes first.

## Follow-ups

- If the v0.1 gate (RFC-0020) measures `c_c` well below 30 ps, re-run: the exchange rate improves in proportion.
- A `live-64-small`-style parameter set (rate 1/4, `τ_A = 95` or 65) is the way to offer the knob without changing the default; RFC-0014 'Parameter sets' is tier A.
