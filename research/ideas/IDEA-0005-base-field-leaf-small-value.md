# IDEA-0005: Base-field leaf level with small-value first rounds, calibrated on CPU against Stwo

| | |
|---|---|
| Status | `triaged` (2026-10-16) |
| Origin | [SUR-0002](../surveys/SUR-0002-lookups.md#candidate-ideas) seeds "Small-value first layer" and "Calibrate the op count on a CPU (E1) against Stwo's `prove_batch`"; [SUR-0003](../surveys/SUR-0003-passes.md#candidate-ideas) seed "Small-value first rounds" ([DDBDT26]); RISK-1 ([00 §9](../../docs/spec/00-overview.md#9-risks), unit costs unmeasured); source §9 bet 1a ([Source26]) |
| Lever | `c_ℓ`: its compute floor, and above all its uncertainty, the `[40, 130]` ps interval SUR-0002 leaves open |
| Spec surfaces | None that change. RFC-0008 'Fraction tree' and 'Batching' fix the relation, the tree and the challenges, all unchanged; the idea is a prover evaluation order inside the v1 kernel, which RFC-0008 NG-6 scopes out of the RFC and RFC-0020 v0.1 measures. No tier-A surface ([02 §1](../../docs/spec/02-public-api.md#1-stability-tiers)); no superseding RFC |
| Priority | 0.32 (−2.1% × 0.3 / 2 at 16K; 0.36 at 1K), unchanged by the re-score of 2026-10-16 under the `counted` profile (`c_ℓ` 101 ps: the saving is a fraction of `c_ℓ`, whose share of `μ` is 32% there against 28%); up to 0.55 if the leaf-level share EXP-0006's proxy measured holds; its c_ℓ calibration deliverable is the SUR-0002 seed restated |

## Mechanism

Today's reading of RFC-0008 'Fraction tree' materializes every leaf as a pair `(n, d) ∈ Fp2 × Fp2`: lookups contribute `(1, α − a_i)`, table rows `(−m_j, α − t_j)`, with `a = key + β·value` and `t = key_t + β·value_t`. The idea keeps the relation, the tree, the batching and the challenges `α, β ∈ Fp2` exactly as specified and changes only how the prover evaluates the leaf level and the first sumcheck.

1. **Never materialize `Fp2` leaves.** Each LogUp instance keeps its leaf level as base-field tables: `key` (virtual; the witness pass materializes it as an integer `< 2^32`, RFC-0006 range analysis), `value` (committed advice, `≤ 16` bits for every table kind of RFC-0008 'Table kinds'), `multiplicity` (committed advice, `< N < 2^48` on the table side). The denominator of a leaf is `d = α − key − β·value`, a linear form in the challenge monomials `{1, α, β}` with integer coefficients `(−key, −value)`; it is computed only when a sumcheck round needs it.
2. **Specialized first layer.** On the lookup side the numerators are all 1, so the first internal node is `n_1 = d_L + d_R`, `d_1 = d_L d_R` (Stwo's `LogUpSingles`, [Stwo26]); on the table side the numerators are base-field `−m_j`, so `n_1 = −m_L d_R − m_R d_L` costs two `Fp × Fp2` products instead of two `Fp2 × Fp2` (Stwo's `LogUpMultiplicities`, SP1's `felt_t` numerators [SP1GPU26]). This part is standard and is the baseline the idea must beat, not the idea.
3. **Small-value first rounds of the level-1 sumcheck.** The level-1 sumcheck proves the claim on the level-1 values `(n_1, d_1)` as a function of the leaves: per pair, the degree-3 polynomial `λ·d_L d_R + d_L + d_R` (lookup side; with `−m` weights on the table side), with Gruen's `eq` correction and two evaluation points per round (Stwo's `correct_sum_as_poly_in_first_variable`, [Stwo26]). Expand the summand in the challenge monomials `{1, α, α², β, β², αβ}` (and `λ` times them): its coefficients are integer polynomials in `key_L, key_R, value_L, value_R` (products `key·key`, `key·value`, `value·value`, `key`, `value`, 1), all `< 2^64` for one pair. Rounds `1 … k` (`k = 3 … 5`) follow the small-value prover of [DDBDT26]: accumulate each coefficient over the pairs of the round in integer multiply-accumulate (MAC) accumulators, `2^{j−1}` accumulators per coefficient in round `j` (one per Boolean assignment of the already-bound variables, the challenge-dependent weights factored out), with 32×16-bit products, 64-bit accumulators and no modular reduction; then combine the six accumulated coefficients with `1, α, α², β, β², αβ` in `Fp2` once per round per accumulator, and apply the `eq` correction as today. After round `k`, fold the leaf tables with the `k` bound challenges once into `Fp2` arrays (the only `Fp2` pass over the leaf level) and continue with the standard `Fp2` algorithm for the remaining rounds.
4. **Levels ≥ 2 unchanged.** `(n, d) ∈ Fp2`, three `Fp2` products per node pair per evaluation point, as RFC-0008 and SUR-0002 [E0-this] count them.
5. **Overflow rule.** The integer accumulators hold `Σ key·value`-type products: `2^32 · 2^16 · 2^k · N_round` must stay `< 2^64`, so for `N_round > 2^11` pairs per accumulator the kernel reduces per block of `2^11` pairs (one Goldilocks reduction per block per accumulator) or accumulates in 128 bits. The combined tree's table side obeys the same rule with `m_j < 2^48` by splitting `m_j` into 16-bit limbs or by reducing per block of one pair.

The same kernel serves both sides of the combined tree, since the table side's leaves `(−m_j, α − key_j − β·value_j)` are also base-field tables. The messages sent to the verifier are the same polynomials as today, coefficient for coefficient.

## Why it could work

- **Own E0 count** (`logup_leaf_count.py`, session scratchpad; primary; assumptions: 1 `Fp2` mul = 3 `Fp` mul, two evaluation points per round, `N/2` pair-rounds per level, generic pair-round 8 + 2 fold `Fp2` mul, numerators-1 pair-round 4 + 1, small-value pair-round in round `j` = 6 MAC accumulators × `2^{j−1}` at 0.1 `Fp2`-mul equivalents each, one `Fp2` fold of the leaves after round `k`, `eq` tables 1 per leaf): textbook 14.7 `Fp2`-mul equivalents per lookup; numerators-1 first layer 11.2; plus small-value rounds `k = 2`: 10.6 (−5.1%), `k = 3`: 10.4 (−6.6%), `k = 4`: 10.4 (−6.7%), `k = 5`: 10.5 (−6.0%). The level-1 sumcheck falls from 2.50 to 1.76 (−30%) at `k = 3`; the optimum is `k = 3–4` because the accumulator count doubles per round while the pairs halve. E0, primary.
- **The counts disagree, which is itself evidence.** SUR-0002 [E0-this] counts 24.5 textbook and 18.5 with a numerators-1 first layer; this count gives 14.7 and 11.2 with the same structure and a different pair-round convention (SUR-0002's geometric sum over rounds against this note's `N/2` pair-rounds per level). A factor 1.65 between two analytic counts of the same algorithm is exactly the uncertainty an E1 calibration removes; the first-level share is 22% (level-1 sumcheck) to 33% (leaf, level-1 nodes and sumcheck) in this count and 13–16% in SUR-0002's, so the small-value saving is bounded by that share in both. E0, primary.
- **[DDBDT26]** (E1, primary, CPU in Jolt; abstract read, PDF blocked per SUR-0003): the small-value sumcheck prover gives "over an order of magnitude" on the Spartan sub-protocol and 1.7–2.2× on a high-degree Shout sumcheck, in the regime the idea targets: summands that are 32/64-bit integers for the first rounds, falling back to field arithmetic once challenges are bound.
- **[Stwo26]** (E0, primary, code read in SUR-0002): `LogUpSingles` (numerators implicitly 1), `LogUpMultiplicities` (base-field numerators, extension denominators), `prove_batch` with one batched sumcheck per level, two evaluation points with the `eq` correction. **[SP1GPU26]** (E0, primary, code read in SUR-0002): first-layer numerators stored as base-field `felt_t`, denominators as `ext_t`, two rounds fused per kernel. Both ship step 2 of the mechanism, so the layout is proven in production and step 2 alone is not Tessera's edge; step 3 is in neither.
- **[VYTA25]** (primary, A40, 377-bit field; read in SUR-0002): field multiplication runs on the int32 pipe at 60% of peak with "Math Pipe Throttle" the dominant stall, and per-SM int32 throughput is roughly constant across generations including H100. Under SUR-0002's 18 int32 instructions per Goldilocks multiplication, an integer MAC that replaces one is worth about 9× per operation (E0, derived; the 0.1 `Fp2`-equivalent per MAC above is 3× more conservative than this). What transfers to Goldilocks is the pipeline statement, not the cycle counts.
- **Bound on the gain.** The first level is the only level with base-field data, so the saving is at most the level-1 sumcheck's share: 22% of this count, 13–16% of SUR-0002's (E0, primary). The gain in `μ` is modest; the calibration is the point.

## Expected gain

Two scenarios, because the baseline constant is itself the open question.

Scenario A, the idea's saving applied to the design's 40 ps assumption: −6.6% of the count (`k = 3`, E0, own count) gives `lookup_ps = 37.4`, rounded to 37.

```sh
python3 -I research/tools/cost_model.py --set lookup_ps=37
```

```
baseline: profile 'corrected'
   S   linear  attention  weights  regen  streaming    total
   1K     3.30       1.13    0.021    4.0       0.42      8.9
   4K     3.25       4.46    0.021    4.0       2.32     14.1
  16K     3.06      16.82    0.020    4.0      11.24     35.1
 128K     1.99      87.48    0.013    4.0      77.93    171.4

scenario: lookup_ps=37
   S   linear  attention  weights  regen  streaming    total
   1K     3.16       1.09    0.021    4.0       0.42      8.7   Δ   -2.1%
   4K     3.11       4.30    0.021    4.0       2.32     13.8   Δ   -2.2%
  16K     2.93      16.21    0.020    4.0      11.24     34.4   Δ   -2.1%
 128K     1.91      84.28    0.013    4.0      77.93    168.1   Δ   -1.9%
```

| Context | Δμ | Δ proof size | Δ verification |
|---|---|---|---|
| 1K | −2.1% (E0) | 0 (E0: same messages, same openings) | 0 (E0: same checks) |
| 4K | −2.2% (E0) | 0 (E0) | 0 (E0) |
| 16K | −2.1% (E0) | 0 (E0) | 0 (E0) |
| 128K | −1.9% (E0) | 0 (E0) | 0 (E0) |

Scenario B, the same saving applied to SUR-0002's compute floor (18.5 `Fp2`-mul equivalents, 76 ps at the int32 peak, E0, primary): 76 → 71 ps. `python3 -I research/tools/cost_model.py --set lookup_ps=71` gives `μ(16K) = 42.9` (+22.0% against the 40 ps baseline) in place of 44.1 (+25.5%) at 76 ps: the idea is worth −2.7% of `μ(16K)` at the pessimistic end of the interval (E0).

Context the model expresses directly: the calibration's value. `c_ℓ ∈ [40, 130]` ps (SUR-0002, E0) is a swing of +0% to +63.7% in `μ(16K)` (`--set lookup_ps=130`, E0); `lookup_ps = 80` is +28.3% and already reaches the SC-2 ceiling of 45 (45.1). An E1 fit with ±20% error replaces that interval by `c_fit ± 0.2 c_fit`, that is at most a ±12% band on `μ(16K)` at `c_fit = 76` (E0). This is the gain the experiment delivers whatever the sign of the small-value result.

Inputs and their evidence: the operation counts and shares are E0 (own count, primary); the 9× MAC-to-multiplication ratio is E0 derived from [VYTA25]'s pipeline statement (primary, other field and GPU) and SUR-0002's instruction count (E0); the Jolt speedups are E1 on another prover and CPU ([DDBDT26], primary); the 40 ps baseline is the design's own assumption (`claim`, [Source26]); the 76 ps floor is SUR-0002's E0 count. Every Δμ above is E0 until the experiment replaces the constant.

### Re-scored under the `counted` profile (2026-10-16)

The `counted` profile of `tools/cost_model.py` carries `c_ℓ` 101 ps (SUR-0002's textbook count at the int32 peak; interval 76–169), `c_c` 94 and `c_π` 73. Scenario A's −6.6% of the count, applied to 101 ps:

```sh
python3 -I research/tools/cost_model.py --profile counted --set lookup_ps=94.3   # scenario A: −6.6% of 101 ps
python3 -I research/tools/cost_model.py --profile counted --set lookup_ps=89.4   # upside: −11.5%, the E1 proxy's leaf-level share × the mechanism's −30%
python3 -I research/tools/cost_model.py --profile counted --set lookup_ps=76     # the interval's ends: the calibration's value
python3 -I research/tools/cost_model.py --profile counted --set lookup_ps=169
```

| Context | `counted` baseline μ | Δμ, scenario A (101 → 94.3) | Δμ, upside (101 → 89.4) | `c_ℓ` 76 / 169 (the calibration's swing) |
|---|---|---|---|---|
| 1K | 17.3 | −2.4% (16.9) | −4.1% | −8.8% / +23.9% |
| 4K | 30.0 | −2.3% (29.3) | −3.9% | −8.4% / +22.9% |
| 16K | 78.8 | −2.1% (77.1) | −3.7% | −7.9% / +21.5% |
| 128K | 373.4 | −2.0% (366.1) | −3.4% | −7.3% / +19.9% |

All E0 (the cost model). The scored gain is unchanged at −2.1% of `μ(16K)`: the saving is a fraction of `c_ℓ`, and `c_ℓ`'s share rose only from 28% to 32% when every unit cost rose together. A ±20% fit band around 101 ps is ∓6.4% of `μ(16K)` (73.8–83.8), the calibration's value restated. Three things did change, all from [EXP-0006](../experiments/EXP-0006-product-leaf-cost/README.md):

1. **The memory-floor objection lifts.** The cost-model refuter's argument was that the 86 ps unfused memory floor (SUR-0002, 288 B per lookup) hides a compute saving below the 76 ps compute floor. At 101 ps compute exceeds that floor; at the 60% utilization of [VYTA25] (127–169 ps) it does for every count; and EXP-0006 found additions a third of the int32 work on the product tree, so SUR-0002's multiplication-only interval is itself a floor. The compute-bound regime is now the likelier one, and the small-value saving is visible in it (E0).
2. **The count disagreement is three-way, and this note's count is the low outlier.** EXP-0006's fraction-tree proxy (RFC-0008's level relation built from the engine's instances, numerators 1 at the leaf level) runs at 107 ns per lookup on one M4 Max core, 77 `Fp`-multiplication equivalents by a count fitted within 5% (E1, primary; about 7.5% of it the redundant per-instance `eq` folds), that is about 25.7 `Fp2`-multiplication equivalents against SUR-0002's 18.5–24.5 and this note's 11.2–14.7. Hypothesis H2's fit has an E1 anchor on this repository's own engine before the experiment runs; the kill criterion's second clause (the fitted numerators-1 count within 10% of 18.5) is not yet decided, because the proxy over-counts and the production kernel will be fused.
3. **The leaf level's share may be larger than either count says.** In the proxy the leaf-level sumcheck is 38% of the time per lookup (82.4 ns per entry over `N/2` entries against the build's 8.9 and the upper levels' 57.2 ns per lookup; E1), above the 22% (this note) and 13–16% (SUR-0002) that bound the mechanism's reach. If a fused kernel keeps that share, the mechanism's −30% of level 1 is −11.5% of the tree (101 → 89.4 ps), the upside column: −3.7% of `μ(16K)`, priority 0.55. The experiment's count should include additions and subtractions as EXP-0006's `count.py` does (its follow-up for `c_ℓ`).

## Security gate

| Property | Effect | Pass |
|---|---|---|
| Soundness bits (G-4, RFC-0014) | Unchanged. The relation, the combined tree, the batching, the challenges `α, β, λ ∈ Fp2` and their transcript positions (`logup/alpha`, `logup/beta`, after Round 1) are those of RFC-0008 and RFC-0014; the prover's round messages are the same polynomials computed in a different order, so the error stays `(N + T)/|Fp2|` plus the sumcheck terms. The integer accumulators are an honest-prover concern only: an overflow (rule 5 of the mechanism) makes an honest prover's message wrong and the proof rejected; it can never make a cheating prover's message pass, because the verifier's checks do not depend on how the prover computed the message. | pass |
| Uniqueness upstream of the logits (SP-3, RFC-0009 Lemma 0) | Unchanged: no committed value, range or gadget changes; the leaf tables are the same `key`, `value`, `multiplicity` polynomials the verifier already receives claims on. | pass |
| Zero knowledge (G-5, RFC-0013) | Unchanged: RFC-0013 masks the sumcheck messages with the committed `G_k` and the tree root with a zero-numerator leaf; neither depends on the prover's arithmetic. The leaf tables are the same secret-class data (values, multiplicities, keys) as today and stay in `Secret<T>` as [05 §3](../../docs/spec/05-observability.md#3-data-classification-and-redaction) requires; the research prototype uses synthetic seeded inputs (research rule 6). | pass |
| Verifier independence (SP-6) | Unchanged: the verifier sees the same messages and runs the same checks; nothing is added to `verifier/`. | pass |
| Post-quantum (hash-based commitments) | Unchanged: no commitment, hash or challenge changes. | pass |

No row is `unknown`: the idea changes the prover's evaluation order only.

## Prior rejection check

- RFC-0008 'Alternatives Considered' 1–6 cover other lookup arguments (one-hot, univariate), base-field challenges, separate trees, shipped tables and floating-point generators; none is the prover's first-level arithmetic. [baseline.md §7](../baseline.md#7-already-rejected) has the same rows for RFC-0008 and nothing about sumcheck micro-architecture.
- RFC-0008 'Non-Goals' excludes "LogUp micro-architecture optimizations beyond the v1 kernels (NG-6)". This idea is therefore scoped as a choice *inside* the v1 kernel, settled by the RFC-0020 v0.1 unit-cost measurement, not as a design change; it changes no RFC text.
- RFC-0014 Alternatives Considered ("`Fp4` LogUp", 2× lookup cost) is about the challenge field, unrelated.
- New since those decisions: [DDBDT26] (March 2026) gives the small-value prover with E1 numbers; SUR-0002 (2026-10-16) found no GPU per-lookup figure anywhere and two analytic counts (SUR-0002's and this note's) disagree by 1.65×, so the constant is an open interval that an E1 experiment here can narrow.

Verdict: none, after the check above.

## Kill criterion

In the same CPU harness, the small-value first rounds (`k` at its best value in `{2, 3, 4, 5}`) cut the level-1 sumcheck time by less than 30% (so the whole fraction tree by less than 5%), or the fitted `Fp2`-multiplication count per leaf of the numerators-1 variant lands within 10% of SUR-0002's 18.5. The first refutes the mechanism; the second says the compute floor is 76 ps at peak and a `c_ℓ ≤ 40` ps needs a faster `Fp2` multiplication, not a cheaper tree.

## Cheapest experiment

Hypothesis: (H1) the small-value first rounds cut the level-1 sumcheck time by at least 30% and the whole tree by at least 5% at `2^20`–`2^24` leaves; (H2) nanoseconds per leaf of the three variants fit a linear model in `Fp2`-multiplication equivalents with one constant, and that constant, applied to the numerators-1 count, lands outside ±10% of 18.5.

Method: a research Cargo workspace under `research/experiments/EXP-NNNN-.../` (own `[workspace]`, `tessera-field` by path for Goldilocks `Fp` and `Fp2 = Fp[X]/(X² − 7)`, research rule 2) implementing the combined fraction tree and its per-level sumchecks over `2^20`–`2^24` seeded synthetic leaves in three variants: (a) generic `Fp2` leaves, (b) numerators-1 lookup side and base-field-numerator table side, (c) variant (b) plus small-value rounds `k ∈ {2, 3, 4, 5}`; single thread, `criterion`, nanoseconds per leaf and a counted `Fp2` multiplication total per variant; the prover's messages of (c) compared bit for bit with (a) on the same inputs and challenges (the correctness check that the evaluation order changes nothing). External anchor: Stwo's `benches/lookups.rs` (`SimdBackend`, `LogUpSingles`, `LogUpGeneric`, `2^16` rows, [Stwo26]) run on the same machine, as a sanity bound on nanoseconds per leaf, never as the baseline (research rule 4). Fit: ns per leaf against `Fp2`-multiplication equivalents across (a), (b), (c) gives the calibrated constant and the level-1 share.

Evidence level reachable here: E1 (CPU, machine named in the report). Effort: `s`–`m` (days: the tree and the three variants are a few hundred lines over `tessera-field`; the Stwo bench is a checkout and a `cargo bench`).

What stays unknown until an H100: the int32-pipe utilization of the MAC loop against the `Fp2` loop, the real MAC-to-Goldilocks-multiplication ratio (assumed 0.1 `Fp2`-equivalents here, 1/27 by [VYTA25]'s derivation), whether the kernel is compute- or memory-bound (SUR-0002: 288 B per lookup unfused), and whether the CPU fit transfers (SUR-0002's own killer for the calibration seed).

## Refutations

Sweep triage of 2026-10-16: four refuter agents who did not write the idea, one per lens; each argument is condensed from that refuter's verdict.

| Lens | Verdict | Argument |
|---|---|---|
| soundness | survives | Only the prover's evaluation order changes: the relation, the combined tree, α, β, λ and their transcript positions (RFC-0008 'Batching'; RFC-0014 schedule) are untouched and the round messages are the same polynomials coefficient for coefficient, so the verifier's checks and the (N + T)/q' plus sumcheck bounds are unchanged. An integer-accumulator overflow (mechanism rule 5) can only make an honest message wrong and the proof rejected at p(0) + p(1) = claim; it cannot make a cheating message pass, because acceptance never depends on how the prover computed it. The per-block reductions are data-independent, so INV-ARCH-2 holds; the leaf tables are the same secret-class data already in Secret<T>. The bit-for-bit comparison of variant (c) against (a) is the right adversarial test and must stay pre-registered. |
| prior art | survives | RFC-0008 'Fraction tree' fixes the leaves as (1, α − a_i) and (−m_j, α − t_j) without a representation, and NG-6 defers 'LogUp micro-architecture optimizations beyond the v1 kernels', a deferral rather than a rejection; Alternatives 1–6 concern other lookup arguments, base-field challenges (3) and generators, none the prover's leaf arithmetic. The separation principle [Jo26] forbids exploiting table values per lookup; this exploits the integer bit width of keys and values (RFC-0006 range analysis), which [Jo26] names as the open dial. Prior art covers most of the mechanism and the note admits it: step 2 ships in Stwo and SP1 ([Stwo26], [SP1GPU26]) and step 3 is [DDBDT26]'s small-value prover; new is its application to denominators that carry Fp2 challenges via the monomial expansion in {1, α, α², β, β², αβ} and the E1 calibration. If [DDBDT26]'s body (unread, PDF blocked per SUR-0003) already treats challenge-linear factors, the novelty shrinks to the calibration. |
| cost model | weakened | Scenario A reruns to −2.1 / −2.2 / −2.1 / −1.9% (`lookup_ps=37`), far below the [40, 130] ps uncertainty of c_ℓ, which the note itself quantifies as +0% to +63.7% on μ(16K) (rerun `lookup_ps=130`: +63.7%). The mechanism's saving is bounded by the level-1 sumcheck's share, 13–33% depending on which of two E0 counts that disagree by 1.65× is right, and it is a compute saving only: SUR-0002's own byte count puts the unfused kernel at 288 B per lookup, an 86 ps memory floor above the 76 ps compute floor, so until IDEA-0006's fusion brings the memory floor below compute the small-value rounds are invisible. What survives is the calibration deliverable (narrowing c_ℓ to c_fit ± 20%), the SUR-0002 seed restated and the c_ℓ twin of IDEA-0004, not a gain of the mechanism; its E1 result must be read jointly with IDEA-0006's roofline. |
| feasibility | weakened | No fraction tree or LogUp level sumcheck exists to modify: `crates/tessera-logup/src/lib.rs` is a stub, and `prove_batch` proves one product of at most `degree` tables per instance (`crates/tessera-sumcheck/src/prover.rs`), whereas a fraction-tree level is a sum of two products (RFC-0008 'Fraction tree') with Gruen two-point messages. The experiment therefore writes its own level prover for all three variants, including an integer-MAC sumcheck with 6 × 2^(j−1) accumulators, per-block reductions and the eq correction, plus a bit-for-bit comparison against a field reference it also writes: effort `m`, not the `s`–`m` claimed. The Stwo anchor is a separate checkout with its own toolchain outside the repository's 1.95.0 pin. E1 is reachable and no tier-A surface moves (RFC-0008 NG-6 scopes it inside the v1 kernel); the real deliverable is the c_ℓ calibration, which IDEA-0004's method covers for c_c. |

## Score

| Gain | Confidence | Cost | Priority |
|---|---|---|---|
| −2.1% μ(16K), −1.9% to −2.2% elsewhere (E0 cost model at `lookup_ps=37`; mechanism count E0 with two counts 1.65× apart; floors from SUR-0002 [E0-this]) | 0.3: mechanism sound and plausible, but the saving is compute-only under a memory floor (86 ps) that hides it until fusion, the two E0 counts disagree, and [DDBDT26]'s body is unread | `m`: about a week for a research workspace that writes its own level prover and integer-MAC sumcheck; no RFC, no tier-A surface, E1 here | 2.1 × 0.3 / 2 = 0.32 |
| Re-scored 2026-10-16 under `counted` (`c_ℓ` 101 ps): −2.1% μ(16K), −2.4% at 1K (E0 cost model at `lookup_ps=94.3`); −3.7% μ(16K) if the leaf-level share of EXP-0006's E1 proxy (38%) holds for a fused kernel | 0.3: unchanged; the memory-floor objection lifts at the counted price (E0), but the counts now disagree three ways with this note's the lowest (E1 proxy: 25.7 `Fp2`-equivalents per lookup), and [DDBDT26]'s body is unread | `m`: unchanged | 2.1 × 0.3 / 2 = 0.32 (0.36 at 1K; 3.7 × 0.3 / 2 = 0.55 at the upside) |

## Decision and links

Triage 2026-10-16: `triaged` at priority 0.32. The mechanism's gain is inside the c_ℓ uncertainty and invisible under the memory floor; the c_ℓ calibration it carries is the valuable part and should be run with IDEA-0004's method after IDEA-0006's roofline sizes the kernel, not as a standalone experiment. Experiments: none yet (the queue's next `experiment` action if triage ranks it). RFC or issue: none; RFC-0008 unchanged. Ledger line updated on: pending consolidation (2026-10-16).

Re-score 2026-10-16 (`ideate (score)` under the `counted` profile): `triaged` at 0.32, unchanged in score, changed in standing: the memory-floor objection no longer hides the mechanism at the counted price, EXP-0006's proxy gives H2 an E1 anchor (25.7 `Fp2`-equivalents per lookup on the engine as written) and puts the leaf level at 38% of the lookup's time, so the experiment, when it runs, should count additions as EXP-0006 does and report the leaf-level share as its own quantity. Still after IDEA-0010's and IDEA-0011's experiments in the queue. Ledger line updated on: 2026-10-16.
