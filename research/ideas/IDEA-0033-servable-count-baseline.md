# IDEA-0033: Cost the `servable` profile with every split (326,656 per layer per token, including the 40-bit attention-output rescale) as the cost model's default and RISK-2's figure

| | |
|---|---|
| Status | `validated` (2026-10-17, [EXP-0011](../experiments/EXP-0011-servable-count/README.md) `done: validated`: the harness reproduces 137,216 and 220,160 from the same table and rule and gives `servable` 326,656 per layer per token, every width derived from public bounds in the same class as the stated cell, `o8` at an attained 40 bits; `μ(1K)` 10.4 / 21.6 and `μ(16K)` 36.6 / 82.7 within 0.043 of the stated cells, E0; priority 3.8 unchanged; next: promote) |
| Origin | [SUR-0009](../surveys/SUR-0009-compiler.md) seed "Cost the `servable` profile in the cost model and in RISK-2" ([E0-this] (b): 260,096 per layer per token before splits, 310,272 with the norm and RoPE splits); RISK-2 ([00 §9](../../docs/spec/00-overview.md#9-risks): "220,160 ... +60%"); RISK-3 (`servable` is the default and the only exact-mode profile); SC-2 ([00 §7](../../docs/spec/00-overview.md#7-success-criteria): `μ(1K) ≤ 12`, `μ(16K) ≤ 45`, stated for exact mode). New here: the attention-output rescale `o8` is a `RequantSplit` the survey's count omits (+16,384 per layer per token) |
| Lever | `C_lin` and `K_lin` as cost-model *inputs*, not as design terms: the model carries the `minimal` profile's 220,160 (RISK-2, corrected) while [RFC-0005 'Profiles'](../../docs/rfcs/RFC-0005-integer-model.md#profiles) makes `servable` the default and RISK-3 makes it mandatory for exact mode; the scoreboard's μ cells ([baseline.md §5](../baseline.md#5-scoreboard)) and SC-2's headroom move with it. No μ is saved; every count-lens idea is scored against this baseline |
| Spec surfaces | 00 §9 RISK-2 and RISK-3 (risk text, tier C); RFC-0005 'Integer semantics' range table (one added range cell for `o8`; the row's value `o < 2^39` is unchanged); [baseline.md §2](../baseline.md#2-the-cost-model) and `research/tools/cost_model.py` (non-normative, research/README.md). No table definition, claim-list schema, parameter set or TCE format changes, so no tier-A surface ([02 §1](../../docs/spec/02-public-api.md#1-stability-tiers)) and no superseding RFC; the `o8` gadget selection itself is already what RFC-0006's rule yields |
| Priority | 3.8 (+4.2% × 0.9 / 1 at 16K under `corrected`, a baseline shift rather than a saving; +17.8% at 1K would give 16.0; +5.2% / +25.6% under `counted`) |

## Mechanism

**1. The count** (E0, primary: [RFC-0005 'Integer semantics'](../../docs/rfcs/RFC-0005-integer-model.md#integer-semantics) rows and range cells, [RFC-0006 'Gadget selection'](../../docs/rfcs/RFC-0006-compiler-ir.md#gadget-selection) with `B_max = 32`, [RFC-0006 'Llama lowering'](../../docs/rfcs/RFC-0006-compiler-ir.md#llama-lowering); Llama 3.1 405B shapes `d = 16,384`, `d_ff = 53,248`, 128 query heads of 128 and 8 KV heads, 126 layers, as 00 §9 and SUR-0009 use them; all read on 2026-10-17). Rescaled tensors of the `servable` profile per layer per token, each element one commitment and one lookup (`Requant1`, single-lookup `TableMap`) or two and two (`RequantSplit`):

| Tensor | Elements | Key range (RFC-0005) | Signed width `b` | Gadget (RFC-0006, `B_max = 32`) | Commitments = lookups |
|---|---|---|---|---|---|
| Attn norm `y = rq(res · ρ)` | 16,384 | key `< 2^37` | 37 | `RequantSplit`, `s = 5` | 32,768 |
| RoPE `q = rq(·)` | 16,384 | key `< 2^44` | 44 | `RequantSplit`, `s = 12` | 32,768 |
| RoPE `k = rq(·)` | 1,024 | key `< 2^44` | 44 | `RequantSplit`, `s = 12` | 2,048 |
| `v = rq(acc)` | 1,024 | key `< 2^29` | 29 | `Requant1` | 1,024 |
| **`o8 = rq(o, r_o, −128, 127)`** | **16,384** | **`o < 2^39` using `R_q s_q ≤ 2^{2σ}`** | **40** (39 if `< 2^39` names the width; both `> 32`) | **`RequantSplit`, `s = 8`** (`z_lo ∈ [0, 2^8)` plus a reduced `Requant { 32, 1 class, r_o − 8, −128, 127 }` table of `2^32` rows) | **32,768** |
| Residual `rq(acc, r_res, −2^19, 2^19 − 1)` | 16,384 | `< 2^29` | 29 | `Requant1` (INT20 output) | 16,384 |
| MLP norm | 16,384 | key `< 2^37` | 37 | `RequantSplit`, `s = 5` | 32,768 |
| `u8 = rq(u)` | 53,248 | `u < 2^29` | 29 | `Requant1` | 53,248 |
| SiLU `a = T_silu[g]` | 53,248 | key `< 2^29` | 29 | single-lookup `TableMap` | 53,248 |
| `h = rq(a · u8)` | 53,248 | `|a · u8| ≤ 128 · 128 < 2^15` | 15 | `Requant1` | 53,248 |
| Residual | 16,384 | `< 2^30` | 30 | `Requant1` | 16,384 |
| **Total** | **260,096** | | | four split groups: norms 32,768 + RoPE 17,408 + `o8` 16,384 = **66,560** extra | **326,656** |

Per token over 126 layers: `326,656 × 126 = 4.116×10^7` commitments and as many lookups (E0, primary: arithmetic). The `minimal` profile at the same rule: 137,216 elements, 220,160 with its four split groups (norms, RoPE, O-projection accumulator `< 2^60`, down-projection accumulator `< 2^58`), the RISK-2 figure (E0, primary: 00 §9; SUR-0009 [E0-this] (a) reconstructs it). Ratio `servable`/`source`: 326,656 / 137,216 = 2.38×; `servable`/RISK-2: 1.48×.

Fixed points after the count-lens ideas, same rule (E0, primary: arithmetic on the table): after [IDEA-0011](IDEA-0011-rsqrt-fact-norm-key.md) (both norms `Requant1`) 293,888; after a RoPE route that brings `q`, `k` to one lookup too (SUR-0009 Question 1: `B_max = 34` for the RoPE table alone, or pre-RoPE requantization) 276,480; after a fused gated-product rescale (`h = rq(a · u)` directly, dropping the `u8` rescale, 53,248 elements; a `D` change through RFC-0005) 223,232 with all three. The `o8` split stays in every one of them: the `< 2^39` bound is already the relational-fact bound (`SoftmaxRow`'s `R_q s_q ≤ 2^{2σ}`), and no idea in the knowledge base touches it.

**2. Cost model** (`research/tools/cost_model.py`, non-normative): add `profile = servable` carrying `commits_per_token = lookups_per_token = 326,656 × 126` with the unit costs of `corrected` (and the `counted` prices as a second row of the same table), each input with its provenance line (the RFC-0005 row, the RFC-0006 rule, this note); until then the documented scenario is `--set commits_per_token=4.116e7 --set lookups_per_token=4.116e7`. The golden check on the `source` profile (`--check`, RFC-0020 criterion ±0.05 per term) stays untouched, since `source` keeps 137,216. [baseline.md §2](../baseline.md#2-the-cost-model) gains the split-by-split table above for both profiles as the decomposition of the `C_lin`, `K_lin` row, and its profile table a `servable` row.

**3. Specification**: 00 §9 RISK-2 states that 220,160 is the `minimal`-profile figure and that the default `servable` profile costs 326,656 commitments per layer per token with its four splits (2.38× the source's 137,216), raising the 405B linear-layer multiplier to about 4.8 at 1K under the corrected unit costs; RISK-3 cross-references that figure ("both profiles are costed" names the number the corpus has). RFC-0005 'Integer semantics' gains the range cell "key `< 2^40`" (or the width convention the table uses for the norm's `< 2^37`) on the O-projection row for `o8`, today stated only as `o < 2^39` on the attention-out row, so that gadget selection on `o8` is explicit in the row that creates the key. Nothing in `D`'s semantics changes.

**4. Replacement**: the v0.2 commitment report ([RFC-0006 'Outputs'](../../docs/rfcs/RFC-0006-compiler-ir.md#outputs), ['Migration / Rollout'](../../docs/rfcs/RFC-0006-compiler-ir.md#migration--rollout): both profiles at 405B shapes) replaces the hand count; until then every count-lens idea note quotes its gain against both baselines (`minimal` 220,160 and `servable` 326,656).

## Why it could work

- Every input is primary text, read on 2026-10-17. RFC-0005 'Integer semantics' gives the `servable` rows (`o8 = rq(o, r_o, −128, 127)` with `o < 2^39`; `u8`, `h`; `u < 2^29`), 'Profiles' makes `servable` the default and the only exact-mode profile (RISK-3), and RFC-0006 'Gadget selection' sends every key with `b > B_max = 32` to `RequantSplit` (E0, primary).
- SUR-0009 [E0-this] (b) counted 260,096 elements and 310,272 with the norm and RoPE splits and ran the model (`μ(1K)` corrected 8.8 → 10.2, counted 17.2 → 20.9; E0, primary); with the `o8` split the figures are 10.4 and 21.6 (E0, primary: `cost_model.py` run 2026-10-17, table below). The survey's own text already calls the `servable` count "contested" as the model's input and says RISK-2's 220,160 "is a `minimal`-profile figure and should say so".
- The `o8` omission is checkable from one row: `o < 2^39` is wider than `2^32` by 7 bits, so under the rule that routes the `2^37` norm key to a split, `o8` splits too; the only way out is a tighter bound, and `< 2^39` is already the relational-fact bound, not the interval one (RFC-0006 'Range analysis', "v1 has one"), so no range fact remains to apply (E0, primary).
- Consequence for SC-2 (E0, primary: arithmetic on the table below): under `corrected` the 1K headroom falls from 12 / 8.8 = 1.36× to 12 / 10.4 = 1.15× and the 16K headroom from 45 / 35.1 = 1.28× to 45 / 36.6 = 1.23×; under `counted` exact mode misses the 1K ceiling by 21.6 / 12 = 1.8× and the 16K ceiling by 82.7 / 45 = 1.84×. Every projected Tessera cell of baseline.md §5 is today for a profile the specification does not ship as default and cannot run in exact mode.
- Method precedent: research/README.md rule 7 labels Tessera's numbers as projections and demands an evidence level on every number; RFC-0006 'Outputs' makes the commitment report the input that "feeds the cost model" and RFC-0020 'Executable cost model' the consumer, so the profile a count belongs to is part of the number. A projection that names the wrong profile is a defect by the knowledge base's own rule.

## Expected gain

No Δμ is saved; the idea shifts the baseline every count-lens idea is scored against. The scenario (E0, primary: run 2026-10-17 on this machine):

```sh
python3 -I research/tools/cost_model.py --profile corrected --set commits_per_token=4.116e7 --set lookups_per_token=4.116e7
```

```
baseline: profile 'corrected'
   S   linear  attention  weights  regen  streaming    total
   1K     3.25       1.13    0.021    4.0       0.42      8.8
   4K     3.20       4.46    0.021    4.0       2.32     14.0
  16K     3.02      16.82    0.020    4.0      11.24     35.1
 128K     1.96      87.48    0.013    4.0      77.93    171.4

scenario: commits_per_token=4.116e7, lookups_per_token=4.116e7
   S   linear  attention  weights  regen  streaming    total
   1K     4.82       1.13    0.021    4.0       0.42     10.4   Δ  +17.8%
   4K     4.75       4.46    0.021    4.0       2.32     15.6   Δ  +11.1%
  16K     4.47      16.82    0.020    4.0      11.24     36.6   Δ   +4.2%
 128K     2.91      87.48    0.013    4.0      77.93    172.3   Δ   +0.6%
```

```sh
python3 -I research/tools/cost_model.py --profile counted --set commits_per_token=4.116e7 --set lookups_per_token=4.116e7
```

```
baseline: profile 'counted'
   S   linear  attention  weights  regen  streaming    total
   1K     9.05       3.70    0.021    4.0       0.42     17.2
   4K     8.91      14.59    0.021    4.0       2.32     29.8
  16K     8.40      54.98    0.020    4.0      11.24     78.6
 128K     5.46     285.91    0.013    4.0      77.93    373.3

scenario: commits_per_token=4.116e7, lookups_per_token=4.116e7
   S   linear  attention  weights  regen  streaming    total
   1K    13.43       3.70    0.021    4.0       0.42     21.6   Δ  +25.5%
   4K    13.23      14.59    0.021    4.0       2.32     34.2   Δ  +14.5%
  16K    12.46      54.98    0.020    4.0      11.24     82.7   Δ   +5.2%
 128K    8.10     285.91    0.013    4.0      77.93    376.0   Δ   +0.7%
```

For reference, the survey's 310,272 (`--set commits_per_token=3.909e7 --set lookups_per_token=3.909e7`, no `o8` split) gives 10.2 / 15.3 / 36.3 / 172.2 under `corrected` (+15.1% / +9.3% / +3.5% / +0.5%) and 20.9 / 33.5 / 82.1 / 375.5 under `counted` (+21.5% / +12.2% / +4.4% / +0.6%) (E0, primary, same run); the `o8` split alone is therefore +2.7 points of μ(1K) under `corrected` and +4.0 under `counted`.

| Context | Δμ | Δ proof size | Δ verification |
|---|---|---|---|
| 1K | +17.8% (8.8 → 10.4) corrected; +25.5% (17.2 → 21.6) counted | opened advice columns about +36% (committed linear elements per token 2.82×10^7 → 4.12×10^7 against 8.3×10^6 attention commitments at 1K); proof total about +13% (the combined rows and weight openings of [baseline.md §4](../baseline.md#4-parameters-proof-size-verification-e0) do not move) | column hashing and the `(G+1) τ_A` inner products about +36%; table MLEs unchanged (every table stays a piecewise function with a ≤ 2^24 core, RFC-0006) |
| 4K | +11.1% (14.0 → 15.6); +14.5% (29.8 → 34.2) | opened columns about +21%; proof total about +8% | about +21% on the same parts |
| 16K | +4.2% (35.1 → 36.6); +5.2% (78.6 → 82.7) | opened columns about +8% (0.72 → about 0.78 GB hashed); proof total about +3% | about +8% on the same parts; the 300 s target unchanged |
| 128K | +0.6% (171.4 → 172.3); +0.7% (373.3 → 376.0) | opened columns about +1.2% | about +1.2% |

Evidence levels of the inputs: the 326,656 count is E0, primary (RFC-0005 rows, RFC-0006 rule), **claim-level in the sense of RISK-2**: it is a hand count the v0.2 commitment report replaces, exactly as 220,160 is. The unit costs are the profile's (`corrected`: assumed, E0; `counted`: E0 counts fitted to E1 on an Apple M4 Max, baseline.md §2). The Δ proof size and Δ verification columns are E0, derived: with attention commitments taken as `A · S = 8,064 · S` per token (cost model), the opened columns' length and the hashing and inner-product work scale with the committed-element count (RFC-0011; baseline.md §4). All projections (08 §1).

Not in the cost model (E0, primary: RFC-0006 'Gadget selection', RFC-0009 `RequantSplit`, 08 §4): the `o8` split adds one `2^32`-row reduced `Requant` table (identified by `(r_o − 8, −128, 127, 32)`, shared across layers if `r_o` is) and one `Range(8)` table to the table set, about `2^32 ≈ 4.3×10^9` table-side leaves and `2^32 × 4 B = 16 GB` of multiplicity counters per table replica, on top of the ten `2^32` tables SUR-0002 already attributes to the splits. This is a cost of the specified design, not of the idea.

## Security gate

| Property | Effect | Pass |
|---|---|---|
| Soundness bits (G-4, RFC-0014) | Unchanged. The idea changes a non-normative tool, two risk paragraphs and one range cell that restates a bound the table already carries (`o < 2^39`); no relation, table definition, challenge, label or absorption changes (INV-FS-1..4 untouched). Indirectly it protects soundness: the explicit `o8` cell closes a reading in which an implementation takes `o < 2^39` as fitting a `2^32` table with a per-tensor shift and routes `o8` to `Requant1`, which would violate the true-integer rule (RFC-0006: the key must lie inside its table's key domain) and Lemma 1's precondition in RFC-0009. The verifier re-derives the range from public data either way (SP-7, INV-IR-5). | pass |
| Uniqueness upstream of the logits (SP-3, RFC-0009 Lemma 0) | Unchanged: `o8` is already a `RequantSplit` under RFC-0006's rule, whose Lemma 2 gives exactly one `(z_lo, y)`; the note counts it, it does not introduce it. The split precondition `s ≤ r − 1` holds with margin (`s = 8`; `r_o` rescales a `2^39` value to INT8, so `r_o ≈ 31`). | pass |
| Zero knowledge (G-5, RFC-0013) | Unchanged: no new committed value or opening; the count is a function of public shapes and public range cells, and the commitment report is already a public output (RFC-0006 'Outputs'). | pass |
| Verifier independence (SP-6) | Unchanged: nothing under `verifier/` is touched; the cost model lives under `research/` and the RFC text is specification the verifier reads, not prover code. | pass |
| Post-quantum (hash-based commitments) | Unchanged. | pass |

## Prior rejection check

- [baseline.md §7 "Already rejected"](../baseline.md#7-already-rejected): no row concerns the cost model's count input or the profile; the only rescale-count row (M31/QM31 fields, "about twice the linear-layer rescales") is about the field, not the profile. [baseline.md §6](../baseline.md#6-levers-which-technique-class-moves-which-term) lists "range analysis, rescale placement" as a lever on `C_lin`, `K_lin` (RISK-2) without a rejection.
- [RFC-0005 Alternatives Considered](../../docs/rfcs/RFC-0005-integer-model.md#alternatives-considered): none rejects costing `servable`; 'Profiles' explicitly defers both counts to the v0.2 commitment report, and the `minimal` paragraph names 137,216 "before RISK-2 corrections" for that profile only.
- [RFC-0006 Alternatives Considered](../../docs/rfcs/RFC-0006-compiler-ir.md#alternatives-considered): about range analysis rules (calibration ranges, wrap tracking), not about which profile the cost model costs. Its Goals name "a commitment report precise enough to replace the source's projected counts", which is the same intent.
- 08 §1 and 00 §9 RISK-2 use the `minimal` count by inheritance from the source (the source's `D` is the `minimal` dataflow); SC-2 derives its "1.4× and 1.9× headroom" from that count while being stated for exact mode, which only `servable` offers (RISK-3). This is an inconsistency between two spec paragraphs, not a decision.
- SUR-0009's seed is the same action without the `o8` split (310,272) and without the SC-2 consequence. What is new: the `o8` split, the per-tensor decomposition for both profiles, the fixed points after IDEA-0011, the RoPE route and the fused rescale, and the headroom arithmetic. "None" after the check.

## Kill criterion

The v0.2 commitment report at 405B shapes produces a `servable` count that differs from 326,656 per layer per token, for example because the compiler's range analysis yields `b ≤ 32` for `o8` through a fact this reading missed (there is none in RFC-0006 today: `< 2^39` is already the relational-fact bound), because RFC-0005's `o8` row is amended (a two-stage normalization of `o`, or `R_q` applied after the O projection), or because the maintainer rules that the cost model tracks `minimal` until the report lands. Each outcome is a one-line correction of the count in the tool and the risk text, not a loss; the idea is refuted only if the report shows the `servable` count is *not* above the `minimal` one, which would need `o8`, `u8` and `h` (122,880 elements) to cost nothing, contrary to RFC-0005's rows.

## Cheapest experiment

Hypothesis: applying RFC-0006's selection rule with `B_max = 32` to RFC-0005's range cells at 405B shapes yields exactly 137,216 (`source`), 220,160 (`minimal` with splits), 260,096 (`servable` elements), 326,656 (`servable` with splits) and 223,232 (`servable` after IDEA-0011, the RoPE route and the fused rescale), and the cost model at `4.116×10^7` per token gives μ(1K) 10.4 / 21.6 and μ(16K) 36.6 / 82.7 under `corrected` / `counted`.

Method (E0, effort `s`, hours): a count script (Python big integers, seeded, under `research/experiments/EXP-NNNN-servable-count/`) that derives every rescaled tensor's signed key width from RFC-0005's range cells (interval arithmetic with the public bound `127 · fan_in · max|x|`, the RoPE constants, and the `SoftmaxRow` fact), applies the selection rule for both profiles and the three fixed-point improvements, and prints the table above; then the two cost-model runs above, and a `servable` profile added to `cost_model.py` with `--check` still passing on `source`. The pre-registered criterion is the five counts and the four μ values to the printed precision. Reaches E0 on this machine; E3 arrives with the v0.2 commitment report (random weights of the right shape, RFC-0006 'Migration / Rollout'), which needs no H100 either.

What stays unknown until v0.2: whether the compiler's interval analysis at the real `QuantSpec` (the `e`, the shifts, `max_context`) reproduces the stated range cells exactly, in particular the `2^37` norm key (IDEA-0011's kill criterion) and whether `o < 2^39` is tight at `σ = 16` with INT8 `v`.

## Refutations

Sweep triage of 2026-10-17: four refuter agents who did not write the idea, one per lens; each argument is condensed from that refuter's verdict, with the cost-model reruns the scoring agent made to check disputed cells noted as such.

| Lens | Verdict | Argument |
|---|---|---|
| soundness | survives | No protocol change: the idea edits a non-normative tool, two risk paragraphs and one range cell restating a bound the table already carries. The count's load-bearing claim is verified: RFC-0005 gives `o < 2^39` using `R_q s_q ≤ 2^{2σ}` and `servable: o8 = rq(o, r_o, −128, 127)` (docs/rfcs/RFC-0005-integer-model.md:73-74), and RFC-0006 routes any signed key width b > B_max = 32 to `RequantSplit` (RFC-0006:75-83), so `o8` splits under either width convention (39 or 40) and `2^39` is already the relational-fact bound (RFC-0009:85, the only fact RFC-0006 has). Making the `o8` cell explicit removes a reading in which an implementation routes a 40-bit key to a `2^32` table, which would break INV-IR-2 and Lemma 1's precondition, so the change can only tighten the spec. Lemma 2's precondition s = 8 ≤ r_o − 1 holds. Nothing is revealed, drawn early, or trusted from the prover. |
| prior art | survives | Verified against primary text: RFC-0005:73 bounds the attention output `o < 2^39 using R_q s_q <= 2^{2 sigma}`, RFC-0005:74 makes `o8 = rq(o, r_o, -128, 127)` a `servable` rescale, and RFC-0006:79-80 route any signed width above B_max = 32 to `RequantSplit`, so `o8` splits (b = 40) and SUR-0009's 310,272 ([E0-this] (b), SUR-0009:28: 'with the norm and RoPE splits') indeed omits it; 00 §9 RISK-2 (docs/spec/00-overview.md:103) states 220,160 from the `minimal` tensor list while RFC-0005:95 makes `servable` the default and RISK-3 (00-overview.md:104) requires it for exact mode, and `cost_model.py` has no `servable` profile (PROFILES at research/tools/cost_model.py:87, verified). No RFC Alternatives entry and no baseline §7 row concerns which profile the cost model counts; RFC-0005:95 defers both counts to the v0.2 report, which is the note's own replacement clause. The action is SUR-0009's fourth candidate idea (SUR-0009:64) made concrete; new is the `o8` split (+16,384 per layer per token, +2.7 / +4.0 points of μ(1K)), the per-tensor decomposition and the SC-2 headroom arithmetic. It saves no μ and is a baseline correction rather than a design idea, but nothing prior excludes it, and IDEA-0031 and IDEA-0032 already disagree on this very number, which is the reason to settle it. |
| cost model | survives | The count is verified from primary text: RFC-0005:73 bounds `o < 2^39` by the `SoftmaxRow` fact and :74 makes `o8 = rq(o, r_o, −128, 127)` a `servable` rescale, RFC-0006:79-80 route any signed width in (32, 62] to `RequantSplit`, and RFC-0006:67 says v1 has exactly one relational fact (already used), so no tighter bound exists; summing RFC-0005's rows gives 260,096 elements and 66,560 split-added, 326,656 per layer per token, and the rerun gives μ(1K) 10.4 / 21.6 and μ(16K) 36.6 / 82.7 (`corrected` / `counted`), matching the note. The term it moves is the baseline every count-lens idea is scored against, and under `counted` the default profile misses SC-2 by 1.8× at both contexts, so it matters more than any single saving in this batch; no Δμ is claimed and every number is labelled E0 hand count pending the v0.2 commitment report, which is honest. One wording correction: SC-2 (00 §7:85) does not state exact mode; the dependency runs through 00 §3:47 (`servable` is the default so exact mode is always available), which still makes `servable` the profile the ceilings are judged on. Adopting it reprices IDEA-0031 and IDEA-0032 (whose gains exist only against this baseline) and slightly dilutes every other count-lens gain in percentage terms; the `--check` golden test on `source` is untouched, as the note says. |
| feasibility | survives | Entirely feasible here: the count is a few hours of exact-integer scripting with no hardware, no tier-A surface (risk text, one RFC-0005 range cell, the non-normative `research/tools/cost_model.py`), and the cost-model runs reproduce the note's figures exactly (μ(1K) 10.4 / 21.6, μ(16K) 36.6 / 82.7 under `corrected` / `counted`; E0). The `o8` observation holds from primary text: RFC-0005 gives `o < 2^39` and `o8 = rq(o, r_o, −128, 127)` (docs/rfcs/RFC-0005-integer-model.md:73-74), RFC-0006's rule sends `b > B_max = 32` to `RequantSplit` (docs/rfcs/RFC-0006-compiler-ir.md:79-83), and the ten-row element tally sums to 260,096 with 66,560 split extras (own recount, E0). `--check` on the `source` profile is unaffected because `source` keeps 137,216 (cost_model.py:87-108). The only caveat is the one the note already carries: the figure is a hand count the v0.2 commitment report replaces, so the deliverable should be labelled as such in RISK-2. |

## Score

| Gain | Confidence | Cost | Priority |
|---|---|---|---|
| A baseline correction, not a saving: μ(1K) 8.8 → 10.4 (+17.8%) and μ(16K) 35.1 → 36.6 (+4.2%) under `corrected`, 17.2 → 21.6 (+25.5%) and 78.6 → 82.7 (+5.2%) under `counted`, from 326,656 commitments and lookups per layer per token (260,096 elements plus 66,560 split-added, including the omitted 40-bit `o8` split, +16,384 per layer per token); under `counted` the default profile misses SC-2 by 1.80× at 1K and 1.84× at 16K. E0 from primary text (RFC-0005:67-80, :73-74; RFC-0006:79-80), reproduced by three refuters, the scoring rerun and [EXP-0011](../experiments/EXP-0011-servable-count/README.md) (exact 41,158,656 per token; the four μ cells within 0.043 of the stated values), pending the v0.2 commitment report that replaces it (RFC-0005:95) | 0.9, now resting on EXP-0011 rather than the hand count alone: the harness reproduces the corpus's 137,216 and 220,160 from the same table and rule, derives every key width from public bounds (`127 · fan_in · max|x|`, the RoPE constants, the `SoftmaxRow` fact) in the same gadget class as the stated cell under both width conventions, and attains `o = −2^39` (so `b = 40` is exact and the fact is tight); the norm key's class holds for every `QuantSpec` `e ≤ 4,192,256` (a residual stream wider than ±104 real units), a thousandfold margin over the stated cell's `e ∈ [1,024, 4,096)`. Residual uncertainty is the compiler's own transfers at v0.2 (the declared ranges of `ρ` and `R_q`), which cannot move the count unless they change a class; `source` keeps 137,216 so `--check` is untouched; the wording correction stands (SC-2 does not state exact mode; the dependency runs through 00 §3:47) | `s`: no tier-A surface (risk text, one RFC-0005 range cell, the non-normative cost model); hours | 4.2 × 0.9 / 1 = 3.8 at the 16K default (16.0 at 1K) |

## Decision and links

Triage 2026-10-17: `triaged` at priority 3.8, the first of this batch to act on: all four lenses survive, the count is verified from primary text (the `o8` split is forced by `o < 2^39` and RFC-0006's `b > B_max` rule), and two ideas of this sweep (IDEA-0031, IDEA-0032) already disagree on the number it settles. The deliverable is the count script and a `servable` profile as the cost model's default (`source` golden untouched), the RISK-2 / RISK-3 text with the hand-count label the v0.2 report replaces, the `o8` range cell in RFC-0005, and the SC-2 wording fix (cite 00 §3:47, not SC-2, for the exact-mode dependency). Every count-lens gain in the ledger is then re-expressed against 326,656; IDEA-0031's and IDEA-0032's gains exist only against this baseline.

Experiment 2026-10-17: [EXP-0011](../experiments/EXP-0011-servable-count/README.md) `done: validated`, all twelve pre-registered checks passing (E0, primary): the counts 137,216 / 220,160 / 260,096 / 326,656 / 223,232 and the four μ cells reproduce; every key width derived from public bounds falls in the stated cell's class; `o8` derives to an attained 40 bits (`o = −2^39` at a one-position row with `v = −128`, `R_q s_q = 2^32` exactly). Two findings beyond the hypothesis: without the `SoftmaxRow` fact `o8` is a 57-bit split (`s = 25`) when the analysis knows `R_q ≤ 2^σ` and `ERR-COMPILE-RANGE` (73 bits) under the naive `R_q ≤ 2^{2σ}`, so the fact decides the split point and compilability, not the count, and RFC-0006 should name the declared range of `R_q`; the `minimal` profile's `W_O o` accumulator derives to 61 bits, one below the 62-bit ceiling. The 223,232 fixed point is conditional on IDEA-0031's certificate holding the fused key at 32 bits (RFC-0006's plain transfer gives 36). Status `triaged` → `validated`; next: promote (the cost model's `servable` profile, the RISK-2 / RISK-3 and SC-2 text, the `o8` range cell and the `R_q` declared range in the RFCs).

Experiments: [EXP-0011](../experiments/EXP-0011-servable-count/README.md) (`done: validated`). RFC or issue: none yet (candidate issue: RISK-2 names the `minimal` count while SC-2 is stated for exact mode; candidate RFC-0005 amendment: the `o8` range cell; candidate RFC-0006 line: the declared range of `R_q`). Ledger line updated on: pending consolidation.
