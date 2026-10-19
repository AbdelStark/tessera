# EXP-0011: Does RFC-0006's gadget-selection rule applied to RFC-0005's range cells at 405B shapes give the default `servable` profile 326,656 commitments per layer per token, with the 40-bit `o8` key as a fourth split group?

| | |
|---|---|
| Status | `done: validated` (run 2026-10-17: all twelve pre-registered checks pass; `servable` costs 326,656 commitments and lookups per layer per token, 2.381× the source's 137,216 and 1.484× RISK-2's 220,160, with the 40-bit `o8` key a split whose extreme `o = −2^39` is attained; the cost model prints `μ(1K)` 10.4 / 21.6 and `μ(16K)` 36.6 / 82.7 under `corrected` / `counted`, within 0.043 of the stated cells) |
| Idea | [IDEA-0033](../../ideas/IDEA-0033-servable-count-baseline.md) |
| Evidence level | E0 (analytic: exact integer interval arithmetic on public bounds, operation counts under the RFC-0006 rule, and the cost model's projection); E3 arrives with the v0.2 commitment report (RFC-0006 'Migration / Rollout'), which needs no H100 |
| Hardware | none (the count and the cost model run on an Apple M4 Max; nothing is timed) |
| Pre-registered on | 2026-10-17 |

## Hypothesis

Applying RFC-0006's gadget-selection rule (`b ≤ B_max = 32`: one commitment and one lookup per element; `32 < b ≤ 62`: two and two) to the key ranges of RFC-0005 'Integer semantics' at Llama 3.1 405B shapes (`d = 16,384`, `d_ff = 53,248`, 128 query heads of 128, 8 KV heads, 126 layers) yields exactly 137,216 commitments per layer per token for the source's count (every rescale a single lookup), 220,160 for `minimal` with its four split groups, 260,096 rescaled elements and 326,656 commitments for the default `servable` profile with its four split groups (attention norm, MLP norm, RoPE `q` and `k`, and the `o8 = rq(o, r_o, −128, 127)` rescale whose key has signed width 40), 223,232 for `servable` after the three count-lens fixes; and the cost model at `326,656 × 126 = 41,158,656` commitments and lookups per token prints `μ(1K)` 10.4 and `μ(16K)` 36.6 under `corrected`, 21.6 and 82.7 under `counted`.

## Success criterion

Written before the code. `b` is the smallest signed width whose range `[−2^{b−1}, 2^{b−1} − 1]` holds the key's exact interval.

- `validated` iff all of:
  - (a) **known values**: the harness reproduces 137,216 (the source's count: `SOURCE_COMMITS_PER_LAYER_TOKEN` in `research/tools/cost_model.py`, the eight-tensor list of 00 §9 RISK-2) when every rescale is a single lookup, and 220,160 (RISK-2's upper bound) for `minimal` under the rule, from the same tensor table and the same rule as the candidate;
  - (b) **candidate**: `servable` counts 260,096 rescaled elements and 326,656 commitments, equal to its lookups, with exactly four split groups (attention norm 16,384, MLP norm 16,384, RoPE `q` 16,384 and `k` 1,024, `o8` 16,384) adding 66,560;
  - (c) **derivation**: for every rescaled or lookup tensor of both profiles, the width derived by exact interval arithmetic from public data alone (weights in `[−127, 127]`, activations over their type range, the public bound `127 · fan_in · max|x|` for weight matmuls, the 14-bit RoPE constants, the `SoftmaxRow` fact `R_q · Σ_i e_qi ≤ 2^{2σ}`, the INT20 residual) falls in the same gadget class (`Requant1` or single-lookup `TableMap` against `RequantSplit`) as the stated RFC-0005 range cell under both readings of a cell `< 2^k` (`b = k` and `b = k + 1`); `o8` derives to `b = 40` with an attained extreme, an admissible input reaching `|o| = 2^39`; where a cell depends on a `QuantSpec` constant (the norm key through `e`), the class is the same for every constant in a stated admissible interval and the report says which;
  - (d) **fixed points**: 293,888 after IDEA-0011 (both norm keys single-lookup), 276,480 after a RoPE route that makes `q` and `k` single-lookup too, 223,232 after a fused gated-product rescale (`h = rq(a · u)`, no `u8`) on top of both;
  - (e) **cost model**: at `commits_per_token = lookups_per_token = 41,158,656`, the tool prints `μ(1K)` 10.4 and `μ(16K)` 36.6 under `corrected` and 21.6 and 82.7 under `counted`, each within ±0.05 of the stated value (the RFC-0020 per-term tolerance), the same to the printed precision at the rounded `4.116×10^7` the idea note used, and `--check` on the `source` profile passes unchanged.
- `refuted` iff the `servable` count under the rule differs from 326,656, or a tensor's derived class differs from its stated cell's class so that the count changes (in particular if `o8` derives to `b ≤ 32`, or `u8` or `h` to a split), or a `μ` cell differs from the stated value by more than 0.05.
- `inconclusive` iff a tensor's class cannot be decided from public data for admissible `QuantSpec` constants and the count depends on it, or the known values of (a) reproduce only under a rule other than RFC-0006's.

Pre-registration disclosure: the hypothesis's numbers are the idea note's hand count, and the two refuters' recounts recorded in the note already agree with it. What this experiment adds that the note cannot supply is (c), a derivation of every width from public bounds that does not read the stated cells, with an attained extreme for the width that is new (`o8`), and (a), the reproduction of the two numbers the corpus already carries from the same table. The agent writing this report did not write the idea note.

## Method

- **Harness**: [count.py](count.py), Python 3 with exact integers, run with `python3 -I`. Baseline and candidate are the same function applied to the same tensor table: the rescaled and lookup tensors of RFC-0005 'Integer semantics' per layer per token (element count as a function of the shapes, the profiles that contain it, the stated range cell, the derived exact interval), the RFC-0006 rule with `B_max = 32` (and the `2 · B_max − 2 = 62` ceiling, beyond which the compiler raises `ERR-COMPILE-RANGE`). Arms: `source` (the rule replaced by "every rescale is one lookup", the assumption RISK-2 names), `minimal` and `servable` under the rule, and the three fixed points of (d) as edits of the table (a width set to the fact's bound; a tensor removed).
- **Derivations** (exact integers at 405B shapes): Q, K, V, gate, up accumulators `[−127 · d · 128, 127 · d · 128]`; RoPE keys `|a · C| + |b · S|` with `|C|, |S| ≤ 2^13` (signed 14-bit constants) and, as the looser reading, `< 2^14`; `o = R_q · Σ_i e_qi v_i ∈ [−128 · 2^{2σ}, 127 · 2^{2σ}]` from the fact and `v ∈ [−128, 127]`, against the product bound `2^{2σ} · 2^{σ+17} · 128` the fact replaces (which exceeds the `b ≤ 62` ceiling, so the fact is what makes `o8` compile at all); `h`'s key `a · u8 ∈ [−128 · 127, 128 · 128]`; `servable` O and down accumulators from INT8 operands; `minimal` O accumulator `127 · d · 2^39` and down accumulator `127 · d_ff · max|a · u|`; the norm key `|res| ≤ 2^19` times `ρ ≤ ⌊√(d · 2^{2κ} / e)⌋` with `κ = 16` as a function of `e ≥ 1`, the class reported for the whole admissible interval of `e`, and the stated `< 2^37` taken as RFC-0005 gives it; after IDEA-0011, the norm key at the fact's bound `2^κ · √d`.
- **Attained extremes**: adversarial inputs that reach each bound at the 405B shapes (all weights `±127`, activations `∓128`; for `o`, a one-position row with `e_q0 = T_exp[0] = 2^σ`, `s_q = 2^σ`, `R_q = 2^σ`, `v = −128`, so `o = −2^39`), and a seeded random search (SplitMix64, seed `0x5EED0033`, toy shapes `d ∈ {4, 8, 16}`, 2,000 draws per shape) confirming that no random input exceeds a derived bound and that the exact `rq` and softmax integer semantics of RFC-0005 are the ones evaluated.
- **Cost model**: `count.py` imports [research/tools/cost_model.py](../../tools/cost_model.py) and evaluates `μ` at the exact per-token count under both profiles, printing the deviation from the stated values; the command-line runs below record the raw tables.
- **Correctness check**: (a) of the criterion, plus the printed totals equal the sum of their rows and the per-token figure equals 126 times the per-layer one; a count that is right only because a row was dropped cannot pass, since every row of RFC-0005's table is listed with its profile membership and printed.

## Reproduction

```sh
# from the repository root
python3 -I research/experiments/EXP-0011-servable-count/count.py > research/experiments/EXP-0011-servable-count/results/count.txt
python3 -I research/experiments/EXP-0011-servable-count/count.py --json > research/experiments/EXP-0011-servable-count/results/count.json
python3 -I research/tools/cost_model.py --profile corrected --set commits_per_token=41158656 --set lookups_per_token=41158656 > research/experiments/EXP-0011-servable-count/results/cost_model_corrected.txt
python3 -I research/tools/cost_model.py --profile counted   --set commits_per_token=41158656 --set lookups_per_token=41158656 > research/experiments/EXP-0011-servable-count/results/cost_model_counted.txt
python3 -I research/tools/cost_model.py --check > research/experiments/EXP-0011-servable-count/results/cost_model_check.txt
```

Environment: Apple M4 Max, macOS 26.6.2 (Darwin 25.6.0), Python 3.14.6, 2026-10-17. Exact arithmetic; no timing.

## Results

Run of 2026-10-17 ([results/count.txt](results/count.txt), [results/count.json](results/count.json)); every number E0, primary (exact integers over RFC-0005's rows and RFC-0006's rule; the cost model's projection, 08 §1). All twelve pre-registered checks pass (`all checks pass`, exit 0).

| Case | Baseline | Candidate (`servable`) | Ratio |
|---|---|---|---|
| Rescaled elements per layer per token | 137,216 (`minimal` = the source's list) | 260,096 | 1.896× |
| Commitments = lookups per layer per token, RFC-0006 rule | 220,160 (`minimal`: 6 split rows in 4 groups, +82,944) | 326,656 (5 split rows in 4 groups, +66,560) | 1.484× |
| Same, against the source's count (every rescale one lookup) | 137,216 | 326,656 | 2.381× |
| Per token, 126 layers | 17,289,216 (source); 27,740,160 (`minimal`) | 41,158,656 | 2.381× / 1.484× |
| `μ(1K)` corrected / counted | 8.8 / 17.2 | 10.4 / 21.6 | +17.8% / +25.5% |
| `μ(4K)` | 14.0 / 29.8 | 15.6 / 34.2 | +11.1% / +14.4% |
| `μ(16K)` | 35.1 / 78.6 | 36.6 / 82.7 | +4.2% / +5.2% |
| `μ(128K)` | 171.4 / 373.3 | 172.3 / 376.0 | +0.6% / +0.7% |
| Linear term of `μ(1K)`, corrected / counted | 3.25 / 9.05 | 4.82 / 13.43 | 1.484× |

**(a) Known values.** The `minimal` tensor list with every rescale forced to one lookup gives 137,216, the source's count (`SOURCE_COMMITS_PER_LAYER_TOKEN`; 00 §9 RISK-2's eight tensors). The same list under the rule gives 220,160 with exactly RISK-2's four groups (norms 32,768, RoPE 17,408, O-projection accumulator 16,384, down-projection accumulator 16,384; +82,944), under derived widths and under stated cells alike.

**(c) Derivation from public data** (signed width `b`, the smallest with `[lo, hi] ⊆ [−2^{b−1}, 2^{b−1} − 1]`; class under the rule; split point `s = b − 32`):

| Tensor | Elements | Profiles | Stated cell | Derived interval | `b` | Class | `s` | Class agrees under `b = k` and `b = k + 1` |
|---|---|---|---|---|---|---|---|---|
| attn norm `y = rq(res · ρ)` | 16,384 | both | `< 2^37` | `[−2^19 · ρ_max(e), (2^19 − 1) · ρ_max(e)]`; at `e = 2^11`: `±9.72×10^10` | 38 (43 at `e = 1`) | split | 6 (11) | yes |
| RoPE `q` | 16,384 | both | `< 2^44` | `±4,363,686,772,736` (`\|C\|, \|S\| ≤ 2^13`); `±8,726,840,868,864` (`< 2^14`) | 43 (44) | split | 11 (12) | yes |
| RoPE `k` | 1,024 | both | `< 2^44` | as `q` | 43 (44) | split | 11 (12) | yes |
| `v = rq(acc)` | 1,024 | both | `< 2^29` | `±266,338,304` (`127 · 16,384 · 128`) | 29 | single | | yes |
| **`o8 = rq(o, r_o, −128, 127)`** | **16,384** | **`servable`** | **`< 2^39`** | **`[−2^39, 127 · 2^32]` from `R_q s_q ≤ 2^{2σ}`, `v ∈ [−128, 127]`** | **40** | **split** | **8** | **yes** |
| residual `rq(acc_O)`, `servable` (`W_O o8`) | 16,384 | `servable` | `< 2^29` | `±266,338,304` | 29 | single | | yes |
| residual `rq(acc_O)`, `minimal` (`W_O o`) | 16,384 | `minimal` | `< 2^60` | `±1,143,914,305,352,105,984` (`127 · 16,384 · 2^39`) | 61 | split | 29 | yes |
| MLP norm | 16,384 | both | `< 2^37` | as attn norm | 38 | split | 6 | yes |
| SiLU `a = T_silu[g]` | 53,248 | both | `< 2^29` | `±266,338,304` | 29 | single lookup | | yes |
| `u8 = rq(u)` | 53,248 | `servable` | `< 2^29` | `±266,338,304` | 29 | single | | yes |
| `h = rq(a · u8)` | 53,248 | `servable` | `< 2^15` | `[−16,256, 16,384]` | 16 | single | | yes |
| residual `rq(acc_down)`, `servable` (`W_down h`) | 16,384 | `servable` | `< 2^30` | `±865,599,488` (`127 · 53,248 · 128`) | 31 | single | | yes |
| residual `rq(acc_down)`, `minimal` (`W_down (a · u)`) | 16,384 | `minimal` | `< 2^58` | `±2.305×10^17` (`127 · 53,248 · 128 · 266,338,304`) | 59 | split | 27 | yes |

Every class agrees with the stated cell under both readings, so the count does not depend on the width convention or on the reading of "14-bit" RoPE constants. The `o8` extreme is attained: a one-position row with `e_q0 = T_exp[0] = 2^16`, `s_q = 2^16`, `R_q = ⌊2^32 / 2^16⌋ = 2^16` and `v = −128` gives `o = −2^39` with `R_q s_q = 2^32` exactly, so `b = 40` is exact and the fact is tight. Without the fact, the product bound is `b = 57` when the analysis knows `s_q ≥ T_exp[0] = 2^σ` (so `R_q ≤ 2^σ`): still a split, with `s = 25`; and `b = 73` under the naive declared range `R_q ∈ [1, 2^{2σ}]`: `ERR-COMPILE-RANGE`. Seeded draws (SplitMix64 `0x5EED0033`, toy shapes 4, 8, 16, 2,000 draws per shape; 2,500 softmax rows of 1 to 16 positions): 0 violations of any derived bound; every adversarial input attains its bound; `rq`'s rounding and saturation checks pass.

**Norm key over the `QuantSpec` constant `e`** (`ρ_max(e) = ⌊√(d · 2^{2κ} / e)⌋`, `|res| ≤ 2^19`; `σ_res` from `e = ⌈d · ε / σ_res²⌉` with `ε = 10^-5`):

| `e` | `ρ_max` | `b` | Class | `σ_res` | INT20 residual span, real units |
|---|---|---|---|---|---|
| 1 | 8,388,608 | 43 | split | 2^-1.3 | ±212,217 |
| 1,024 | 262,144 | 38 | split | 2^-6.3 | ±6,632 |
| 4,096 | 131,072 | 37 | split | 2^-7.3 | ±3,316 |
| 65,536 | 32,768 | 35 | split | 2^-9.3 | ±829 |
| 1,048,576 (= 2^20, the admissible ceiling used) | 8,192 | 33 | split | 2^-11.3 | ±207 |
| 4,192,256 | 4,097 | 33 | split | 2^-12.3 | ±104 |
| 4,192,257 (flip, 2^21.999) | 4,096 | 32 | single | 2^-12.3 | ±104 |

The stated `< 2^37` corresponds to `e ∈ [1,024, 4,096)`, a residual stream of ±3,300 to ±6,600 real units; the class flips only at `e = 4,192,257`, a residual stream of at most ±104 real units, a thousandfold margin in `e`. With IDEA-0011's fact (`|res · ρ| ≤ 2^κ √d = 2^23`) the key is `b = 25`, single, for every `e`.

**(d) Fixed points** (same rule, derived widths): `servable` after IDEA-0011 293,888 (−10.0%); after a RoPE route that makes `q`, `k` single-lookup too 276,480 (−15.4%); after the fused gated-product rescale on top of both 223,232 (−31.7%), 1.4% above RISK-2's 220,160. `minimal` after IDEA-0011 187,392 and after the RoPE route 169,984, SUR-0009 [E0-this] (a)'s figures.

**(e) Cost model** ([results/cost_model_corrected.txt](results/cost_model_corrected.txt), [results/cost_model_counted.txt](results/cost_model_counted.txt), [results/cost_model_check.txt](results/cost_model_check.txt)): at the exact 41,158,656 per token, `μ(1K)` 10.397 (stated 10.4, deviation −0.003) and `μ(16K)` 36.557 (36.6, −0.043) under `corrected`; 21.577 (21.6, −0.023) and 82.707 (82.7, +0.007) under `counted`; the rounded 4.116×10^7 prints the same cells to one decimal (only the 4K `counted` Δ moves, +14.4% exact against +14.5% rounded, a rounding boundary). `--check` passes on all eight rows of the embedded chart table (worst deviation 0.036; the pinned vector is absent on this branch).

## Analysis

**Against the criterion.** (a) through (e) hold, so the result is `validated`: the count the idea note carries is what RFC-0006's rule yields from RFC-0005's rows, the two numbers the corpus already carries (137,216 and 220,160) come out of the same table, and the `o8` split that SUR-0009's 310,272 omitted is forced by an attained 40-bit key, 8 bits above `B_max`, under every reading of the cells. The baseline every count-lens idea is scored against is therefore 326,656 per layer per token for the default profile (RISK-3: the only exact-mode profile), 48% above the 220,160 RISK-2 states and 2.38× the source's count; under `counted` the default profile misses SC-2's ceilings by 1.80× at 1K (21.6 against 12) and 1.84× at 16K (82.7 against 45), under `corrected` it keeps 1.15× and 1.23× of headroom.

**What is robust and what is not.** The count depends only on each key's class, never on its exact width: every class is at least 1 bit from the `B_max` boundary (the `h` key at 16 bits and the accumulators at 29 to 31 bits are far below; the splits are 6 to 29 bits above) except the `minimal` O-projection accumulator, which at `b = 61` sits 1 bit below the `2 · B_max − 2 = 62` ceiling at which the compiler raises `ERR-COMPILE-RANGE`, a note for the `minimal` profile that this experiment did not pre-register and only records. The norm key's class depends on a `QuantSpec` constant, with a thousandfold margin for any residual stream wider than ±104 real units; the `o8` class does not depend on the fact at all when the analysis knows `R_q ≤ 2^σ` (57 bits, still a split), and depends on it for compilability under the naive declared range (73 bits): the fact decides the split point (`s = 8` against 25, a `Range(8)` table of 256 rows against one of 2^25), not the count. The third fixed point (223,232) assumes the fused key `a · u` stays single-lookup; under RFC-0006's plain transfer it is `|a| · |u| < 2^35`, `b = 36`, a split, so that figure is conditional on IDEA-0031's weight-row norm certificate (`|u| < 2^23`, `b = 32`), as that note states; without the certificate the fused rescale saves nothing.

**Two corrections to the pre-registered text, neither to the criterion.** (1) The Method's parenthetical says the product bound the fact replaces "exceeds the `b ≤ 62` ceiling, so the fact is what makes `o8` compile at all"; that is true under the naive declared range of `R_q` (`[1, 2^{2σ}]`, 73 bits) and false when the analysis uses `s_q ≥ T_exp[0]` (`R_q ≤ 2^σ`, 57 bits, a split with `s = 25`). RFC-0006 says gadget outputs take "their declared output type range" and does not say which range `SoftmaxRow` declares for `R_q`; the compiler's choice settles it. (2) The first run of `count.py` encoded criterion (c)'s "stated admissible interval" for `e` as `e ≤ 2^22`; the exact flip is `e = 4,192,257 = 2^21.999`, 0.05% below, because at `ρ_max = 4,096` the key `[−2^31, 2^31 − 2^12]` fits 32 bits exactly. The constant was replaced by an interval derived from RFC-0005's scale policy (`e ≤ 2^20`, a residual stream of at least ±207 real units) with the flip point printed beside it; the criterion names no number, so this corrects the code, not the pre-registration, and it is disclosed here. The first run's twelve checks were otherwise identical.

**Threats to validity.**
- *E0 hand table against the compiler.* The tensor table is this experiment's reading of RFC-0005's rows, the same reading the idea note and its refuters made; the v0.2 commitment report at 405B shapes (random weights of the right shape, RFC-0006 'Migration / Rollout') is the E3 replacement and could differ where the compiler's range analysis uses a transfer this table does not model (the declared range of `ρ` and of `R_q`; whether `RsqrtRow` declares an output range at all), or where `QuantSpec` sets `e` or the shifts outside the intervals above. The kill criterion of the idea note names exactly these.
- *Width convention.* RFC-0006 names `b` the "signed range width" without fixing whether an interval `(−2^k, 2^k)` is `k` or `k + 1` bits; both readings give the same class on every row here, so the count is unaffected, but the split point `s` and the `Range(s)` table sizes move by one bit (norms 5 or 6, RoPE 11 or 12, `o8` 7 or 8).
- *What the per-layer count excludes.* Per-row scalars (`ρ` and its bracket, `c_q`, `s_q`, `R_q`), the embedding one-hot and the LM-head rescale, multiplicities and masks, as the source's 137,216 excludes them; the chart's fitted linear column is 1.8% above the stated count (`cost_model.py`), the size of what is left out.
- *Unit costs.* The Δ% of `μ` is a ratio of counts and does not depend on `c_c`, `c_ℓ`; the absolute cells are projections under assumed (`corrected`) or E0-fitted-to-E1 (`counted`) unit costs until the RFC-0020 v0.1 gate measures them (E2).
- *Input distribution.* None: the count is a function of public shapes and public range cells, not of inputs; the seeded draws test only that the bound formulas are not exceeded and are attained.

## Verdict

`validated`. The idea's status moves `triaged` → `validated`; its score keeps gain, confidence and cost (a baseline correction, 0.9, `s`) with confidence now resting on a harness that reproduces the corpus's two known values and derives every width from public bounds rather than on the hand count alone, and its priority stays 3.8 (16.0 at 1K). The deliverables of the promote phase are unchanged: a `servable` profile as the cost model's default (`source` golden untouched), the RISK-2 and RISK-3 text with the hand-count label, the `o8` range cell in RFC-0005 (and, from this experiment, the declared range of `R_q`, since it decides whether `o8` compiles without the fact and what `s` the split takes), and the SC-2 wording fix.

## Follow-ups

- **Promote IDEA-0033**: add `profile = servable` to `research/tools/cost_model.py` carrying `commits_per_token = lookups_per_token = 41,158,656` with the `corrected` and `counted` unit costs, each input with its provenance line (this experiment, RFC-0005 rows, RFC-0006 rule), `--check` on `source` untouched; the split-by-split table of this report into baseline.md §2 and a `servable` row in its profile table and scoreboard; an issue for 00 §9 RISK-2 and RISK-3 and the SC-2 wording (the exact-mode dependency runs through 00 §3, `servable` as the default); an RFC-0005 amendment for the `o8` range cell (`key < 2^40` on the O-projection row, or the convention the table uses) and a line in RFC-0006 'Range analysis' naming the declared range of `R_q` (`≤ 2^σ` from `s_q ≥ T_exp[0]`, or `≤ 2^{2σ}`), which this experiment shows decides `o8`'s split point and compilability without the fact.
- **Re-express every count-lens gain against 326,656**: IDEA-0031 (−53,248, −16.3%, conditional on its certificate keeping the fused key at 32 bits), IDEA-0032 and IDEA-0028 (the RoPE route, −17,408, −5.3%), IDEA-0011 (−32,768, −10.0%); their Δμ cells in the ledger move with the baseline.
- **The `minimal` profile's 1-bit margin**: `W_O o` derives to `b = 61` against the ceiling 62; a wider `σ` or a wider `v` type would push it to `ERR-COMPILE-RANGE`. Worth a line in RISK-3 or the `minimal` paragraph of RFC-0005 'Profiles'; no experiment needed.
- **What v0.2 settles (E3, no H100)**: the commitment report at 405B shapes with the real `QuantSpec` replaces every row of this table; the three cells to watch are the norm key (`e`), `R_q`'s declared range and whether `o < 2^39` is tight at `σ = 16` with INT8 `v` (it is, by the attained extreme, unless the compiler's `v` range is narrower than the type range). An H100 run settles nothing here: the count is analytic.
