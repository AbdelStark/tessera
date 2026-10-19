# IDEA-0011: RMSNorm relational bound and the current split-count kill criterion

| | |
|---|---|
| Status | `refuted` for the current commitment/lookup count-saving claim (EXP-0017, 2026-10-19); relational bound survives |
| Origin | RISK-2 ([00 §9](../../docs/spec/00-overview.md#9-risks): "RMSNorm products near 2^36"); the relational-fact mechanism of [RFC-0006 'Range analysis'](../../docs/rfcs/RFC-0006-compiler-ir.md#range-analysis), which v1 uses once (the `SoftmaxRow` fact); [SUR-0002](../surveys/SUR-0002-lookups.md) Question 3 synthesis (the split gadget is the remaining count lever) and its "wider `B_max`" seed / OQ-0006-1, which this idea dominates for the norm case; new |
| Lever | Current `C_lin`/`K_lin` gain is zero at κ30; exact joint width38 remains split at 405B. Logical table rows fall by 134,217,728 after sharing. The κ16 commitment/lookup claim and earlier table estimates are historical below. |
| Spec surfaces | RFC-0006 'Range analysis' (a second relational fact) and the range cell of RFC-0005 'Integer semantics' (attn norm: "key `< 2^37`" tightens); RFC-0009 `RsqrtRow` (the fact and its proof obligation, one sentence; no new committed value). No change to `D`, to the claim-list schema or to any TCE format. Table definitions are tier A ([02 §1](../../docs/spec/02-public-api.md#1-stability-tiers)) and the norm's `Requant` table changes `(b, s)`, so promotion goes through an RFC amendment; before 1.0 no protocol version is frozen |
| Priority | 0 (EXP-0017: current count gain is zero; confidence 0.95 in this scoped refutation; historical scores retained below) |

## Current result: EXP-0017 (2026-10-19)

[EXP-0017](../experiments/EXP-0017-rmsnorm-relational-bound/README.md) refutes the current count-saving hypothesis at RFC-0023's κ30. The exact ordinary 405B key interval is `[-137438952750,137438952750]`, width 38: narrower than the compiler's interval width52, but still `RequantSplit`, with two commitments and two lookups per element. Baseline and candidate therefore both count `C=45,566,660`, `K=45,440,678` per token; per-layer saving is zero. The first embedding-derived norm and final norm are explicitly included. At 8B the current exact width is37, also split.

The algebraic implication remains valid with exact integer brackets, epsilon>=1 and the padding zero pin. Complete coordinate-magnitude enumeration and 93,600 toy rows with 954,440 candidate roots pass; 160,877 production-table key outputs match independent integer rounding/saturation. The historical κ16 controls do become single (405B width24,8B width23), so their count saving cannot be carried into current κ30.

There is a separate exact table-row reduction: `M=7,693,467,648` to `7,559,249,920` (−134,217,728), after deduplicating all table definitions and retaining the old first norm's Range21 where another site uses it. The current conditional `counted` model prices this at only −0.0082784% μ(1K) and−0.0024214% μ(16K), with its existing omissions; no GPU performance or memory measurement is claimed. This is a possible separately scoped follow-up, not validation of this idea's count hypothesis.

The mechanism, security table, four-lens reviews and scores below are historical κ16 notes. They are retained as provenance and are not fresh approval of current production changes or the full protocol. RFC-0022/0023/0028/0035 already address the old residual, root-range, split and padding assumptions. Priority moves 0.9→0, confidence 0.6→0.95 for the scoped count refutation; no production promotion is recommended. Raw records, independent checks, mutations and limitations are linked in the experiment report.

## Mechanism (historical κ16 proposal)

Setting ([RFC-0005 'Integer semantics'](../../docs/rfcs/RFC-0005-integer-model.md#integer-semantics), [RFC-0009 `RsqrtRow`](../../docs/rfcs/RFC-0009-gadgets.md#rsqrtrow)): for a residual row `res ∈ ℤ^d`, `u = Σ_i res_i² + e` with `e ≥ 1` the exact integer form of `d·ε/σ_res²` rounded up, `N = d · 2^{2κ}` with `κ = 16`, and `ρ = max{ρ ∈ ℕ : ρ² u ≤ N}`. `RsqrtRow` commits `ρ` and the limbs of `u`, proves `ρ² u ≤ N` and `N + 1 ≤ (ρ + 1)² u` with `BigIntCompare` (exact over the integers, Lemma 4), and Lemma 7 makes `ρ` unique. The norm's requantization key is `z_i = res_i · ρ`, today bounded by interval arithmetic as `|res_i| < 2^19` times `ρ < 2^{κ+1}`, so `|z_i| < 2^36`, signed width `b = 37` (RFC-0005 states "key `< 2^37`"), above `B_max = 32`: [RFC-0006 'Gadget selection'](../../docs/rfcs/RFC-0006-compiler-ir.md#gadget-selection) picks `RequantSplit` with split point `s = 5`, two commitments and two lookups per element.

The fact. Every term of `u` is a non-negative integer, so `res_i² ≤ u` for every `i`. Multiply by `ρ² ≥ 0` and use the proven bracket: `(ρ · res_i)² = ρ² res_i² ≤ ρ² u ≤ N = d · 2^{2κ}`, hence

```
|ρ · res_i| ≤ 2^κ · ⌈√d⌉      for every i, for every input, for every ρ that passes RsqrtRow,
```

and strictly `< 2^κ · √d` when `e ≥ 1`. Numbers: `2^23` for `d = 16,384` (Llama 3.1 405B, `d = 2^14`, `√d = 128`), `2^22.5` for `d = 8,192`, `2^22` for `d = 4,096`.

What changes:

1. **RFC-0009 `RsqrtRow`** gains the line "Relational fact emitted: `max_i |ρ · res_i| ≤ 2^κ · ⌈√d⌉`, implied by `ρ² u ≤ N` and `res_i² ≤ u`", mirroring `SoftmaxRow`'s "Relational fact emitted: `R_q · s_q ≤ 2^{2σ}`". Its proof obligation is the first `BigIntCompare` inequality the gadget already proves; no new committed value, lookup or relation.
2. **RFC-0006 'Range analysis'** lists the fact as the second relational fact ("v1 has two"): the range of `Mul(res, ρ)` after `RsqrtRow` is `[−2^κ⌈√d⌉, 2^κ⌈√d⌉]` instead of the interval product. The derivation reads only `d` and `κ` from the public `QuantSpec`, so the verifier re-derives it exactly (INV-IR-5, SP-7).
3. **Gadget selection** then sees `b = 24` for `d = 16,384` (`25` with one bit of margin; both `≤ B_max = 32`) and picks `Requant1` for the attention norm, the MLP norm and the final norm. The norm's table is `Requant { b = 24 or 25, classes = 1, r_n, −128, 127 }`, identified by `(shift, lo, hi, b)`: `2^24`–`2^25` rows with one class, since the norm shift `r_n` is per tensor and the RMSNorm gains are folded into the next weights (RFC-0005 'Fold'), against a `2^32`-row reduced table plus a `Range(5)` table and a committed `z_lo` per element today.
4. **RFC-0005** tightens the attn-norm range cell from "key `< 2^37`" to "key `≤ 2^κ⌈√d⌉` (`2^23` at `d = 16,384`)". `D`'s semantics are untouched: `y_i = rq(res_i · ρ, r_n, −128, 127)` computes the same value.

## Why it could work

- The bracket `ρ² u ≤ N` is already a proof obligation of RFC-0009 `RsqrtRow` (Lemma 7, via Lemma 4), so the fact costs no commitment, lookup or relation (E0, primary: RFC-0009 text, read 2026-10-16).
- The relational-fact mechanism exists and is used once (`SoftmaxRow`, RFC-0006 'Range analysis'; the attention output's `< 2^39` bound in RFC-0005 depends on it), so the compiler and verifier already have the code path a second fact needs (E0, primary).
- The bound holds for every admissible input, as RFC-0006 Alternative 4 demands: it is an algebraic consequence of the bracket, not a calibration statistic. Checked exactly over the integers on 7,590 synthetic rows (`d ∈ {4096, 8192, 16384}`, `κ = 16`, `e ∈ {1, 2^10, 2^21, 2^22, 2^25}`; maximal-magnitude, single-spike, all-zero, all-equal, uniform and sparse rows; `ρ` by integer square root) with no violation; the maximum observed `|ρ · res_i|` reaches the bound to within 2^−0.03 on spike rows, so the bound is tight (E0, primary: scratchpad script, this note).
- The count: RISK-2's 220,160 − 137,216 = 82,944 extra commitments per layer per token; SUR-0002 Q3 counts the eight rescaled tensors of the `minimal` profile from RFC-0006's lowering, of which the two norms are 2 × 16,384 = 32,768, i.e. 39.5% of the RISK-2 excess (E0, primary: arithmetic on 00 §9 and SUR-0002 Q3).
- Cost model (E0, primary: `research/tools/cost_model.py`, corrected profile): `C_lin = K_lin` per token `2.819×10^7 × (220,160 − 32,768) / 220,160 = 2.3995×10^7`.

## Expected gain

```sh
python3 -I research/tools/cost_model.py --set commits_per_token=2.3995e7 --set lookups_per_token=2.3995e7
```

```
baseline: profile 'corrected'
   S   linear  attention  weights  regen  streaming    total
   1K     3.30       1.13    0.021    4.0       0.42      8.9
   4K     3.25       4.46    0.021    4.0       2.32     14.1
  16K     3.06      16.82    0.020    4.0      11.24     35.1
 128K     1.99      87.48    0.013    4.0      77.93    171.4

scenario: commits_per_token=2.3995e7, lookups_per_token=2.3995e7
   S   linear  attention  weights  regen  streaming    total
   1K     2.81       1.13    0.021    4.0       0.42      8.4   Δ   -5.5%
   4K     2.77       4.46    0.021    4.0       2.32     13.6   Δ   -3.4%
  16K     2.61      16.82    0.020    4.0      11.24     34.7   Δ   -1.3%
 128K     1.70      87.48    0.013    4.0      77.93    171.1   Δ   -0.2%
```

| Context | Δμ | Δ proof size | Δ verification |
|---|---|---|---|
| 1K | −5.5% (8.9 → 8.4) | opened advice columns about −11% (the removed 4.2×10^6 committed elements per token against 2.82×10^7 linear + 8.3×10^6 attention), proof total about −4% | column hashing and the `(G+1) τ_A` inner products about −11%; table MLEs unchanged |
| 4K | −3.4% (14.1 → 13.6) | opened columns about −7%, proof total about −2.5% | about −7% on the same parts |
| 16K | −1.3% (35.1 → 34.7) | opened columns about −2.8% (0.72 → 0.70 GB), proof total about −1% | about −3% on the same parts |
| 128K | −0.2% (171.4 → 171.1) | opened columns about −0.4% | about −0.4% |

Evidence levels of the inputs: the counts are E0, primary, but **claim-level in the sense of RISK-2**: 220,160 is the corpus's upper bound, replaced by the v0.2 commitment report; if the compiler's range for the norm key is already `≤ 2^32` wide the count gain is zero (see "Kill criterion"). The Δ proof size and Δ verification columns are E0, derived: the advice matrix has `5.8×10^5` rows for `2^22`-element columns at 16K/16K (baseline §4, RFC-0011), so the opened columns' length and the hashing and inner-product work scale with the committed-element count; the combined rows (1.14 GB) and weight openings do not move. Attention commitments per token taken as `A · S = 8064 · S` (cost model). All projections (08 §1).

Not in the cost model (E0, primary: 08 §4 and SUR-0002 Q3): the norm's `2^32`-row reduced table and its `Range(5)` table leave the table set, so the table-side leaves per proof fall by about `2 × 2^32 ≈ 8.6×10^9` of the `~4×10^10` SUR-0002 attributes to ten `2^32` tables (about 2% of the `4.6×10^11` linear-layer lookups per 16K-token proof), and the multiplicity counters by `2 × 2^32 × 4 B = 32 GB` per table replica, replaced by a `2^25 × 4 B = 128 MB` table.

## Security gate

| Property | Effect | Pass |
|---|---|---|
| Soundness bits (G-4, RFC-0014) | Unchanged. The fact is implied by `ρ² u ≤ N`, which `RsqrtRow` already proves exactly over the integers (Lemma 4); a `ρ'` that fails the bracket is rejected today, and one that passes equals `ρ` (Lemma 7), so every accepted witness satisfies the fact with probability 1. The `Requant1` key `z_i = res_i · ρ` is a true integer by the existing range (`\|z_i\| < 2^36 < (P−1)/2`, Lemma 0), so its field value equals its integer value whatever table width the compiler picks; a `2^24`–`2^25`-row table can only reject a key the honest prover never produces. No challenge, label or absorption changes (INV-FS-1..4 untouched). | pass |
| Uniqueness upstream of the logits (SP-3, RFC-0009 Lemma 0) | `Requant1` Lemma 1 ("the table is a function of its key") replaces `RequantSplit` Lemma 2 for the norm output; both give exactly one `y`. One fewer committed value (`z_lo`) per norm element, so one fewer value for INV-IR-3 to cover. | pass |
| Zero knowledge (G-5, RFC-0013) | One fewer committed limb per norm element; no new committed value, no new opening. The table width `b` and the fact are derived from the public `d`, `κ`; the compiler reads no weight or activation (RFC-0006 range analysis uses only public bounds). | pass |
| Verifier independence (SP-6) | The verifier re-derives the fact from the RFC text and the claim list's `d`, `κ` (INV-IR-5, SP-7): the inequality `(ρ res_i)² ≤ ρ² u ≤ d 2^{2κ}` needs only that `u`'s terms are squares plus a non-negative `e`, all public. No prover code or data is trusted. | pass |
| Post-quantum (hash-based commitments) | Unchanged. | pass |

Spec observations made while checking, all pre-existing and independent of the idea (E0, primary, from the RFC texts; to be raised as issues by the triage or consolidation agent, not fixed here):

- (a) RFC-0005 declares the residual INT20, but the row "Residual: `res += rq(acc, r_res, −2^19, 2^19 − 1)`" clamps the increment, not `res`; summing 2 × 126 + 1 clamped INT20 terms gives a worst-case `|res|` near `2^27` and `u = Σ res_i² + e` near `2^68 > P`, against `RsqrtRow`'s "`u < 2^55`" and the table's "`s < 2^54`". Either `res` is re-clamped (then the text should say so) or the `u` recomposition is not exact. The fact here needs `u` exact, exactly as `RsqrtRow` already does, so the question belongs to RFC-0005/RFC-0006, not to this idea.
- (b) RFC-0006 sets `s = b − B_max` without the class bits of `pack(c, z) = (z + 2^{b−1}) + 2^b · c`, so a reduced table with 4 classes has `2^{B_max + 2}` rows; the norm table has one class and is unaffected.
- (c) `RsqrtRow` range-checks `ρ ∈ [0, 2^{κ+1})`, but the honest `ρ = ⌊√(N/u)⌋` exceeds `2^{κ+1} − 1` whenever `u ≤ d/4`, i.e. for an all-zero (or near-zero) residual row when `e ≤ d/4 = 4,096`: the honest prover would fail (`REJ`), not a soundness bug but a completeness one. The scratchpad check found 1–2 such rows per setting at `e ∈ {1, 2^10}` and none at `e ≥ 2^21`. Whether `e ≥ 2^12` always holds at the 405B `QuantSpec` is unknown here (depends on `σ_res`). Unrelated to the fact, which holds for every `ρ` that passes.

## Prior rejection check

- [RFC-0006 Alternatives Considered](../../docs/rfcs/RFC-0006-compiler-ir.md#alternatives-considered) 4, "Calibration-based ranges": rejected because ranges must hold for every input and be verifier-derivable. Not this idea: the bound is a proven worst-case inequality from public constants, satisfied by every passing witness.
- RFC-0006 Alternative 5, "Wrap tracking instead of bounds at keys": unrelated (the key stays a bounded true integer).
- RFC-0009 Alternatives 4 and 5 (rsqrt by table, Newton iteration) and the matching row of [baseline.md §7](../baseline.md#7-already-rejected): unrelated; the bracket stays as it is.
- baseline §7 has no row on range facts; [baseline §6](../baseline.md#6-levers-which-technique-class-moves-which-term) lists "range analysis, rescale placement" as a lever on `C_lin`, `K_lin` (RISK-2) without a rejection.
- The competing fix for the same tensor, SUR-0002's "wider `B_max` (34 or 36)" seed and OQ-0006-1: a `2^37` key needs `b ≤ B_max`, i.e. `B_max = 37` and a `2^37 × 4 B = 512 GB` counter array, or still a split at 34 or 36; the fact reaches `Requant1` at `B_max = 32` for free. It does not dominate the seed for the other RISK-2 tensors (Q/K after RoPE near `2^44`, O-projection chain near `2^60`, down projection near `2^58`), which have no bracket to exploit.
- What is new: the observation that the `RsqrtRow` bracket, already paid for, bounds the product `res_i · ρ` jointly where interval arithmetic bounds the factors separately. "None" after the check.

## Kill criterion

The v0.2 commitment report (or a replication of RFC-0006's interval analysis at the 405B `QuantSpec`) shows the compiler's norm key already has `b ≤ 32`, so `Requant1` is already selected and the count gain is zero. Analytically (E0, this note): if the compiler bounds `ρ` by `⌊√(N/e)⌋` (its value at `res = 0`) rather than by the range check `2^{κ+1}`, the interval key is `⌊√(N/e)⌋ · 2^19`, which is `2^31.5` at `e = 2^21` (`b = 33`, still a split) and reaches `b ≤ 32` only for `e ≥ 2^22`, i.e. `σ_res ≤ 2^−12.3` at `ε = 10^−5` and `d = 16,384` (Llama 3.1's `rms_norm_eps`; secondary: public model configuration). The kill fires only in that regime, and then reduces the idea to a table-size and completeness gain (`2^24`–`2^25` rows instead of `2^32`–`2^33`, and a key bound independent of `e`) rather than a count gain.

## Cheapest experiment

Hypothesis: for every `d ∈ {4096, 8192, 16384}`, `κ = 16`, `e ≥ 1` and every integer row with `|res_i| < 2^19`, the honest `ρ` satisfies `max_i |ρ · res_i| ≤ 2^κ ⌈√d⌉`, and the compiler's interval bound at the same inputs exceeds `2^32` unless `e ≥ 2^22`.

Method (E0, effort `s`, one day): the Python integer script already run for this note, promoted into `research/experiments/EXP-NNNN-rsqrt-fact/` with its pre-registered criterion (zero violations over the row families above, plus an exhaustive sweep at toy `d ∈ {4, 16}` and 8-bit `res`), the cost-model scenario above, and a diff of RFC-0006 'Range analysis', RFC-0009 `RsqrtRow` and RFC-0005's range cell. An adversarial check belongs with it: a `ρ' ≠ ρ` that satisfies `|ρ' res_i| ≤ 2^κ⌈√d⌉` for all `i` must still fail one of the two `BigIntCompare` inequalities (it does, by Lemma 7; the test states the exact `REJ`).

What stays unknown until v0.2 (needs no H100): the compiler's actual `b` for the norm key at the 405B `QuantSpec`, which decides count gain against table-size-only gain; and `e`'s value, which decides observation (c).

## Refutations

Sweep triage of 2026-10-16: four refuter agents who did not write the idea, one per lens; each argument is condensed from that refuter's verdict.

| Lens | Verdict | Argument |
|---|---|---|
| soundness | survives | The fact is a theorem about every accepted witness: RsqrtRow proves ρ²u ≤ N exactly over the integers (RFC-0009 'RsqrtRow', Lemma 4) and u = Σ res_i² + e with e ≥ 1 is a true integer below 2^55, so (ρ·res_i)² ≤ ρ²u ≤ d·2^{2κ} gives abs(ρ·res_i) < 2^κ·√d = 2^23 at d = 2^14, strict because equality needs e = 0 (E0, primary, refuter's derivation); it is derived from the public d and κ, so the verifier re-derives it (RFC-0006 INV-IR-5, SP-7) and never trusts a prover-declared range. It is soundness-neutral by construction: y is pinned by the Requant1 lookup itself (Lemma 1), whose key z = res_i·ρ is a true integer by Lemma 0 (abs(z) < 2^36), so a narrower table can only reject a key the honest prover never produces; one fewer committed limb means one fewer value for INV-IR-3 to cover. No challenge, label, mask or opening changes, and ADV-V learns nothing from a table width that is a function of public constants. The fact inherits RsqrtRow's existing dependence on the residual being a true INT20 (RFC-0005 clamps the increment, not res), which the note's observation (a) correctly flags as a pre-existing RFC-0005/0006 consistency issue rather than a weakness of the idea; b = 24 is tight (abs(z) ≤ 2^23 − 1), so the one-bit margin to b = 25 is the safe table. |
| prior art | survives | Not in the specification: RFC-0006 'Range analysis' says 'v1 has one' relational fact (SoftmaxRow), RFC-0005 bounds the norm key by interval arithmetic ('key < 2^37'), and RFC-0006 therefore routes it to RequantSplit above B_max = 32. The fact is a correct algebraic consequence of what RsqrtRow already proves (u = Σ res_i² + e, N = d·2^{2κ}, ρ²u ≤ N exact by Lemma 4), since res_i² ≤ u gives (ρ·res_i)² ≤ N; the key is res_i·ρ with gains folded into the next weights (RFC-0005), so no other factor widens it. Not rejected: RFC-0006 Alternatives Considered 4 (calibration ranges) demands exactly the every-input, verifier-derivable property this bound has, and RFC-0009 Alternatives 4–5 concern the rsqrt mechanism, not its range; [Jo26] names key width as the open lever. Standard interval refinement by a proven joint inequality, but no prior art in the corpus applies it here; new is the observation itself and that it dominates the 'wider B_max' seed / OQ-0006-1 for the norm tensors at zero protocol cost. The three spec observations (a)–(c) in the note are independent findings worth issues. |
| cost model | survives | The scenario reruns exactly (`commits/lookups_per_token=2.3995e7`: −5.5 / −3.4 / −1.3 / −0.2%) and its one derived input checks: 2.819e7 × (220,160 − 32,768)/220,160 = 2.3994e7, and the norm key is 37 bits in the spec (RFC-0005 'key < 2^37'), above B_max = 32, so RequantSplit is selected today (RFC-0006) and the two norms are 2 × 16,384 of the 82,944-element RISK-2 excess (00 §9). The term matters at the context the idea targets (1K: C_lin/K_lin halved is −8.0 / −10.6%, baseline.md §2) and the gain is a count reduction, so it is independent of the unit-cost uncertainty that dominates every other idea here; it only grows if regen falls below 4. At 16K it is −1.3%, inside noise, so the rubric scores it at 1K, not 16K. The sole input risk is the kill criterion the note states (the compiler may already bound ρ tightly enough when e ≥ 2^22), which a one-day E0 experiment settles; the claim-level label on 220,160 is honest and the saving costs no new committed value. |
| feasibility | survives | The cheapest experiment is a Python integer script at E0, effort `s`, already run; it needs no hardware and no code in the workspace. The spec change lands at the cheapest possible moment: range analysis, gadget selection and the verifier's re-derivation are unimplemented (`crates/tessera-compiler/src/lib.rs` is a stub; #77, #80, #117 open), so the second relational fact is a sentence in RFC-0006 'Range analysis' (today 'v1 has one'), RFC-0009 RsqrtRow and the RFC-0005 range cell ('key < 2^37', verified). The norm table's (b, s) is a tier-A table definition, but no protocol version is frozen before v0.3 and the compiler derives b from the spec anyway, so the RFC amendment is small. The one dependency is observation (a): the fact needs u exact, which RFC-0005's residual bound must guarantee either way. |

## Score

| Gain | Confidence | Cost | Priority |
|---|---|---|---|
| −5.5% μ(1K), −3.4% at 4K, −1.3% at 16K, −0.2% at 128K (E0 cost model at `commits/lookups_per_token=2.3995e7`; the bound abs(ρ·res_i) < 2^23 is an E0 theorem, primary; the RISK-2 count 220,160 is claim-level). Scored at 1K: RISK-2 is a short-context lever (rubric: 16K unless the idea targets short context) | 0.6: the mechanism is a proven theorem that all four lenses verify, and the saving is a count reduction independent of unit-cost uncertainty; but the count input is a claim-level upper bound and the kill criterion (the compiler may already bound ρ below 2^32 when e ≥ 2^22) is unsettled | `l` by the rubric's letter: the norm table's (b, s) is a tier-A table definition, so promotion is an RFC amendment to RFC-0005/0006/0009; the feasibility lens finds the amendment small (a sentence in each, before any protocol version is frozen) and the experiment is `s`, E0 here, already run | 5.5 × 0.6 / 4 = 0.83 (1.65 if the small amendment is weighted `m`); highest-ranked design idea of the batch and testable here now |
| Re-score 2026-10-16 under `counted` (`c_c` 94, `c_ℓ` 101 ps): −7.1% μ(1K) (17.2 → 16.0), −4.0% at 4K, −1.4% at 16K, −0.2% at 128K; with the fifth pass priced (regen 5, `c_c` 167 ps, EXP-0007) −7.5% at 1K (22.6 → 20.9) and −1.6% at 16K (97.8 → 96.2); count inputs unchanged (E0, claim-level) | 0.6 (unchanged: the kill criterion on the compiler's range for `ρ` is still unsettled) | unchanged (`l` by the rubric's letter) | 7.1 × 0.6 / 4 = 1.1 |

## Decision and links

Triage 2026-10-16: `triaged` at priority 0.83, the highest-ranked design idea of the batch. Next step: promote the integer script into a pre-registered E0 experiment with the kill criterion (does the compiler's range for ρ already put the norm key at or below 2^32 when e ≥ 2^22?), then an RFC amendment adding the second relational fact; observations (a)–(c) go out as issues against RFC-0005/0006/0009 regardless. Experiments: none yet. RFC or issue: none yet (observations (a)–(c) above are candidate issues against RFC-0005/RFC-0006/RFC-0009). Ledger line updated on: pending consolidation. Re-scored 2026-10-16 (queue item 9) under the `counted` profile: priority 0.83 → 1.1 (−7.1% at 1K; −7.5% with [EXP-0007](../experiments/EXP-0007-pruned-column-answer-pass/README.md)'s fifth pass priced), confidence unchanged; still ahead of IDEA-0002 (1.0) among the design ideas, both behind the prover-only experiments of queue items 6–8.


## Re-score 2026-10-19 (E0, conditional)

The historical tables above use their pinned model. This calculation uses the current formula, with P5 kept separate from commit_ps; [baseline §2](../baseline.md#2-the-cost-model) states its omissions. Count-only reductions are 32768 per layer per token, times 126. Certificate overhead and fidelity remain unresolved; this re-score does not validate the mechanism.

| Baseline | Δμ(1K) | Δμ(4K) | Δμ(16K) | Δμ(128K) |
|---|---|---|---|---|
| historical 326656-current-formula | -6.32% | -4.20% | -1.76% | -0.27% |
| current-c2 | -5.92% | -4.02% | -1.73% | -0.26% |

Exact inputs and outputs: [main-sync-2026-10-19.json](../experiments/EXP-0011-servable-count/results/main-sync-2026-10-19.json).
