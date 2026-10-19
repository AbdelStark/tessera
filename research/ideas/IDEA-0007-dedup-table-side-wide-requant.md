# IDEA-0007: Deduplicated table side for wide Requant tables: commit the touched rows, prove membership arithmetically

| | |
|---|---|
| Status | `rejected` (2026-10-16): security-gate fail on zero knowledge (G-5, RFC-0013); the only ZK-admissible cap has a negative gain, and RFC-0006 Alternatives Considered 4 already rejects input-dependent completeness |
| Origin | new (an alternative lookup-argument structure). Answers [OQ-0006-1](../../docs/rfcs/RFC-0006-compiler-ir.md#open-questions) (`B_max` 30 / 32 / 34) from the argument side; [RISK-2](../../docs/spec/00-overview.md#9-risks); RFC-0008 Drawbacks ("a small table used rarely still pays its full row count per proof"); [SUR-0002](../surveys/SUR-0002-lookups.md) Question 3 and its seed "Wider `B_max`", whose cost (`2^b` table-side leaves and `2^b × 4 B` counters per table) is what this removes |
| Lever | `K_lin` and `C_lin` (the RISK-2 excess: `RequantSplit` → `Requant1` for every wide key); table-side leaves and multiplicity memory ([08 §4](../../docs/spec/08-performance-budget.md#4-memory-budgets)), hence `free` HBM and `stream` |
| Spec surfaces | RFC-0008 'Relation', 'Fraction tree', 'Table kinds' (a sparse table side and a `Clamp` kind, or `Requant` with shift 0); RFC-0009 (a per-slot membership gadget beside `RequantSplit`); RFC-0006 'Gadget selection' and `TableLimits`; RFC-0011 (three more advice columns per wide table); RFC-0019 (the verifier evaluates the slot columns instead of the dense table MLE). Table definitions and the claim-list schema are tier A ([02 §1](../../docs/spec/02-public-api.md#1-stability-tiers)): a superseding RFC is required |
| Priority | 0 (hard gate: security-gate fail) |

## Mechanism

Today every table instance has a dense side: `2^b` rows `t_j = key_j + β·value_j`, each with a committed multiplicity `m_j` and one leaf `(−m_j, α − t_j)` in the combined fraction tree (RFC-0008 'Relation', 'Fraction tree'). The compiler therefore caps keys at `B_max = 32` bits (`TableLimits.max_log2_rows`, RFC-0006) and lowers every wider rescale to `RequantSplit` (RFC-0009): two committed values and two lookups per element instead of one (RISK-2).

The change, for a `Requant { b, classes, r, lo, hi }` instance whose key width `b` exceeds `B_max`:

1. **Sparse table side.** Drop the dense `2^b` rows. The prover commits a list `S` of `T'_max` slots `(key'_j, value'_j, m'_j)`, `j < T'_max`: the distinct keys the witness touched, with their multiplicities, in any order, padded with `(0, f(0), 0)`. `T'_max` is a public, input-independent parameter of the table instance (it is a tree shape and a claim-list field). The LogUp identity becomes

   ```math
   \sum_{i<N} \frac{1}{\alpha - key_i - \beta\,value_i} \;=\; \sum_{j<T'_{max}} \frac{m'_j}{\alpha - key'_j - \beta\,value'_j}
   ```

   with `key'`, `value'`, `m'` as three committed advice columns (RFC-0011) where today only `m` is committed. The fraction tree keeps its shape: lookups contribute `(1, α − a_i)`, slots contribute `(−m'_j, α − t'_j)`, the `(0, δ)` masking leaf of RFC-0013 stays.

2. **Per-slot membership.** The verifier can no longer evaluate the table side as a closed-form MLE, because the rows are prover-chosen. Each slot proves it is a row of the table by the arithmetic of `RequantSplit`, applied once per slot instead of once per lookup. With `r` the shift and `z'_j = key'_j − 2^{b−1}` the signed key (offset encoding, RFC-0008 'Table kinds'):
   - commit `rem'_j` and prove `rem'_j ∈ Range { r }` (one lookup; when `r > B_max`, `NonNeg(rem'_j, r)` with `⌈r / B_max⌉` limbs, RFC-0009 Lemma 3);
   - define the virtual quotient `q'_j = (z'_j + 2^{r−1} − rem'_j) · 2^{−r}` (a linear relation, no commitment);
   - prove `(pack(class, q'_j), value'_j) ∈ Clamp { b − r, classes, lo, hi }`, a table with `|classes| · 2^{b−r}` rows whose value is `clamp(q, lo, hi)`: constant below `lo`, identity on a core of `hi − lo + 1` values (256 for INT8 output), constant above. It is the existing `Requant` kind with shift 0, or a new kind; either is a tier-A table definition. Its own dense side has `2^{b−r}` rows, below `2^{B_max}` whenever `b − r ≤ 32`, which holds for every INT8-output rescale (the pre-clamp quotient is a few bits wider than the output).

   A slot with `m'_j = 0` contributes nothing to the sum and may hold anything that passes its two lookups (the padding does). A slot with `m'_j ≠ 0` must equal some lookup element `a_i` as an `Fp2` value, else the rational identity fails except with probability `(N + T'_max)/|Fp2|`; the β-Schwartz–Zippel step then gives `key'_j = key_i` and `value'_j = value_i` as field elements, and `key_i` is a verifier-ranged true integer (SP-7, RFC-0006 range analysis), so `key'_j` is one too. Lemma 2's Euclidean-division argument with split point `r` (all quantities below `2^62`, Lemma 0) then gives a unique `(rem'_j, q'_j)`, and the `Clamp` lookup a unique `value'_j = clamp(⌊(z'_j + 2^{r−1}) / 2^r⌋, lo, hi)`, which is the table's value. Membership is therefore proven for every slot that matters.

3. **Compiler.** Range analysis selects `Requant1` for every `b ≤ 62` (Lemma 0's bound) and marks the table instance sparse when `b > B_max`; `TableLimits.max_log2_rows` stops being a cost limit and becomes the dense/sparse switch.

4. **Prover.** Building `S` costs the same `N` random-access increments the dense counters cost today, into a hash histogram keyed by `key` instead of a `2^b` counter array (`2^b × 4 B` per table, 08 §4); the histogram holds at most `T'_max` entries. If the witness touches more than `T'_max` distinct keys, the prover cannot complete the proof at this `T'_max` (see the zero-knowledge row).

Per-slot cost at the model's unit costs (`commit_ps = 30`, `lookup_ps = 40`, `leaf_ps = 12`; E0, primary, [tools/cost_model.py](../tools/cost_model.py)): 4 committed elements (`key'`, `value'`, `m'`, `rem'`) = 120 ps and 3 lookups (one combined-tree leaf, one `Range`, one `Clamp`) = 120 ps, about 240 ps; with two `NonNeg` limbs (`r > 32`) 5 committed and 4 lookups, about 310 ps. A dense row today costs one committed multiplicity and one leaf, 70 ps. Break-even against the dense side it replaces: `T'_max ≈ 0.29 · 2^b` (0.23 with limbs), E0.

## Why it could work

- RISK-2 ([00 §9](../../docs/spec/00-overview.md#9-risks); E0, primary): worst-case ranges (Q/K after RoPE near `2^44`, the O-projection chain near `2^60`, the down-projection accumulator near `2^58`, RMSNorm products near `2^36`) force `RequantSplit` on tensors whose typical values fit one lookup: up to 220,160 against 137,216 commitments and lookups per layer per token (+60%); the corrected profile of the cost model carries `2.82×10^7` against `1.757×10^7` per token.
- RFC-0009 Lemma 2 (E0, primary): the split's uniqueness proof is the Euclidean-division argument this idea applies per slot; Lemma 0 bounds every compared quantity below `2^62`, so no new lemma is needed, only a restatement with split point `r`. [RKM25] ([SUR-0002](../surveys/SUR-0002-lookups.md) findings table; E1, primary) proves rounding by the same arithmetic plus range proofs per element and fails there on count; per distinct row the same arithmetic is paid `T'_max` times instead of `N`.
- RFC-0008 'Structured evaluation' (E0, primary): the verifier's table work is already `O(n · core)` and independent of `2^b`, so replacing the dense side by committed slots costs the verifier only the `Clamp` and `Range` MLEs it already knows how to evaluate, plus three more opened columns.
- 08 §4 (E0, primary): the dense design budgets `≤ 2^32 × 4 B` of counters per table sharded across GPUs, about 2 GB per GPU per table, inside the 24 GB working set that bounds the 0.2 TB `free` HBM of the streaming term; ten tables take most of it. A hash histogram of `T'_max ≤ 2^30` entries at 16 B is 16 GB per table across 8 GPUs, 2 GB per GPU, no worse; at `2^28` it is 4× smaller (E0).
- Own E0 (primary, derivation in "Expected gain"): the dense table side of ten `2^32` tables costs `10 × 2^32 × 70 ps = 3.0 s` per proof; at `T'_max = 2^30` the slots cost `10 × 2^30 × 240 ps = 2.6 s`; the RISK-2 excess they remove costs 14.0 s of the 426 GPU-s of a 16K-token 405B proof at `μ = 35.1`, and 0.87 s of the 6.25 GPU-s of a 1K-token proof (14%).
- SUR-0002 Question 3 (E0, primary): every other count lever fails on the key-width wall ([QSL+25] merging, [RKM25] range arithmetic) because it works per element; the split's second lookup is the only remaining `K_lin` lever, and this is the only mechanism found that pays its cost per distinct key rather than per element.

## Expected gain

The cost model carries no table-side term (dense rows, counters or slots are per proof, amortized over the 16K-token batch), so the scenario expresses only the RISK-2 excess removed; the per-proof terms follow in words.

```sh
python3 -I research/tools/cost_model.py --set commits_per_token=1.757e7 --set lookups_per_token=1.757e7
```

```
baseline: profile 'corrected'
   S   linear  attention  weights  regen  streaming    total
   1K     3.30       1.13    0.021    4.0       0.42      8.9
   4K     3.25       4.46    0.021    4.0       2.32     14.1
  16K     3.06      16.82    0.020    4.0      11.24     35.1
 128K     1.99      87.48    0.013    4.0      77.93    171.4

scenario: commits_per_token=1.757e7, lookups_per_token=1.757e7
   S   linear  attention  weights  regen  streaming    total
   1K     2.06       1.13    0.021    4.0       0.42      7.6   Δ  -14.0%
   4K     2.03       4.46    0.021    4.0       2.32     12.8   Δ   -8.7%
  16K     1.91      16.82    0.020    4.0      11.24     34.0   Δ   -3.3%
 128K     1.24      87.48    0.013    4.0      77.93    170.7   Δ   -0.4%
```

Upper bound with the freed counter memory added to the streaming term's `free` HBM (`0.2 TB → 0.36 TB`: ten tables × 2 GB per GPU × 8 GPUs of counters, if the histogram is held off-GPU or is much smaller):

```sh
python3 -I research/tools/cost_model.py --set commits_per_token=1.757e7 --set lookups_per_token=1.757e7 --set free_bytes=0.36e12
```

```
scenario: commits_per_token=1.757e7, lookups_per_token=1.757e7, free_bytes=0.36e12
   S   linear  attention  weights  regen  streaming    total
   1K     2.06       1.13    0.021    4.0       0.34      7.5   Δ  -15.0%
   4K     2.03       4.46    0.021    4.0       1.99     12.5   Δ  -11.1%
  16K     1.91      16.82    0.020    4.0       9.99     32.7   Δ   -6.8%
 128K     1.24      87.48    0.013    4.0      71.43    164.2   Δ   -4.2%
```

| Context | Δμ (RISK-2 excess removed, model) | Δμ (plus freed HBM, upper bound) | Δ proof size | Δ verification |
|---|---|---|---|---|
| 1K | −14.0% (E0) | −15.0% (E0) | +3 opened advice columns per wide table: 10 × 3 × `τ_A` (155) × 16 B ≈ +75 kB of the 2 GB proof (E0); the advice shrinks by the split's second committed value, −1.06×10^7 elements per token, −37% of `C_lin` (E0) | +10 `Clamp` and +10 `Range` MLE evaluations, `O(core)` each (E0); the dense wide-table MLE evaluations disappear |
| 4K | −8.7% (E0) | −11.1% (E0) | as above | as above |
| 16K | −3.3% (E0) | −6.8% (E0) | as above | as above |
| 128K | −0.4% (E0) | −4.2% (E0) | as above | as above |

Per-proof terms the model cannot express (E0, own derivation: `native(16K) × 16384 = 12.1 s` and `μ = 35.1` give 426 GPU-s per 16K-token proof, 6.25 GPU-s per 1K-token proof; ten wide tables assumed, RFC-0008 'Table kinds' and the RISK-2 tensors give between four and ten):

| `T'_max` per table | Slots added per proof | Dense side removed | Net at 16K (excess 14.0 s) | Net at 1K (excess 0.87 s) |
|---|---|---|---|---|
| 2^28 | 0.64 s | 3.0 s | −16.4 s, −3.8% | −3.2 s (the dense side, which the baseline does not model, dominates) |
| 2^30 | 2.6 s | 3.0 s | −14.4 s, −3.4% | −1.3 s |
| 2^32 (the dense alternative's row count) | 10.3 s | 3.0 s | −6.7 s, −1.6% | +6.4 s: a loss |
| provable `min(N_table, 2^b)` | `N_table` summed over wide tables is 82,944 split elements per layer per token × 126 layers × tokens: `1.07×10^10` at 1K (2.6 s), `1.71×10^11` at 16K (41 s) | 3.0 s | +24 s at 16K: a loss | −1.3 s, only because the dense side goes |

The provable bound never wins where the key width is the problem: a slot costs 240 ps and the split it replaces 70 ps per element, so `T'_max = N_table` loses whenever `N_table < 2^b`, and `T'_max = 2^b` wins only for `2^b < 0.29 · N_table` (`b ≤ 34` at 16K), below every RISK-2 width (`b ≥ 36`). The gain therefore rests on a statistical cap `T'_max ≪ min(N_table, 2^b)`, which is what the zero-knowledge row and kill criterion (1) are about. Inputs: unit costs and counts E0 (primary: RFC-0009 'Summary of costs', 00 §9, the cost model); distinct-key counts unknown at every evidence level.

## Security gate

| Property | Effect | Pass |
|---|---|---|
| Soundness bits (G-4, RFC-0014) | The rational identity's error is `(N + T'_max)/|Fp2|` with `T'_max ≤ 2^b`, no worse than today's `(N + T)/|Fp2|` with `T = 2^32` (RFC-0008 'Relation'). The table side is now prover-chosen, so the β-collision argument changes: a bad slot `(key', value')` with `m' ≠ 0` must collide as `key' + β·value'` with some honest lookup element, probability at most `N/|Fp2|` per slot over `β`, and `T'_max` slots give `N · T'_max / |Fp2| ≤ 2^36 · 2^32 / 2^128 = 2^−60` per table (SUR-0002 Question 3 counts `N ≈ 4.6×10^10 < 2^36` per table at 16K), `2^−57` over ten tables, inside the 64-bit live target once re-added to the LogUp line of the budget. The soundness-review skill must check this line before an RFC. | pass (E0, pending the adversarial reading) |
| Uniqueness upstream of the logits (SP-3, RFC-0009 Lemma 0) | Per slot: Lemma 2 restated with split point `r` gives one `(rem', q', value')` for each `key'` in the verifier-derived range (all quantities below `2^62`). Per element: Lemma 1 as today, since the slot set with `m' ≠ 0` is a subset of the table's rows and the lookup `(key, value)` must match one of them. Padding slots (`m' = 0`) carry no witness value upstream of the logits. | pass (E0) |
| Zero knowledge (G-5, RFC-0013) | `T'_max` must be public and input-independent. The only provable bound is `min(N_table, 2^b)`, which gives no saving (above). A statistical cap makes completeness depend on the secret activations: a prompt whose wide-table activations touch more than `T'_max` distinct keys cannot be proven at this shape, and the abort or shape change leaks at least one bit about the prompt to the counterparty verifier of RFC-0013 'Malicious verifier'. `S` itself is committed advice under RFC-0011 masks and the `(0, δ)` leaf stays, so the opened columns leak nothing new. A fallback with the same public shape (for example, padding to the dense side when the cap is exceeded) was not found: the shape differs. Whether RFC-0013's threat model accepts an input-dependent completeness failure is the first thing to settle, as a precise statement to the RFC's owner. | `fail` (set at triage 2026-10-16 from `unknown`: see Refutations, soundness) |
| Verifier independence (SP-6) | The verifier gains a `Clamp` table kind (or `Requant` with shift 0), three advice columns per wide table and the per-slot linear relation, and loses the dense wide-table MLE; all written from the specification under the `verifier-independence` skill. No prover code is shared. | pass |
| Post-quantum (hash-based commitments) | Unchanged: the slot columns live in the RFC-0011 hash-based tensor commitment. | pass |

A `fail` anywhere rejects the idea. The `unknown` is the first thing the experiment or the survey must settle.

## Prior rejection check

Checked against [baseline.md §7](../baseline.md#7-already-rejected) and the RFCs' Alternatives Considered:

- RFC-0008 Alternatives Considered 1 rejects one-hot and decomposable arguments (Lasso, Shout) because each lookup commits several chunks and the separation principle [Jo26] bounds per-lookup cost. Does not apply: this commits per distinct row, not per lookup, and per-lookup cost is unchanged at one leaf.
- RFC-0008 Alternatives Considered 2 rejects univariate two-level lookups (Baloo, cq, Caulk) for their pairing-based commitments. Does not apply: this is multilinear LogUp-GKR with an arithmetic membership gadget; the "commit the touched subtable" idea of Caulk and Baloo is the resemblance, the mechanism of proving membership differs.
- RFC-0008 Alternatives Considered 5 (shipping table contents) is unrelated: the verifier regenerates the `Clamp` and `Range` tables as today.
- RFC-0005 Alternatives Considered 3 (affine scales everywhere, rejected because every key then needs the split gadget) would be reopened by this mechanism if it survives: with no split, affine scales cost one lookup like power-of-two scales.
- OQ-0006-1 (`B_max` 30 / 32 / 34) becomes moot for wide tables; it remains for the dense tables.
- RFC-0009 'Summary of costs' lists `RequantSplit` at two commitments and two lookups and names RISK-2 as its drawback; no alternative to the split is recorded there.

New: the observation that the split's membership arithmetic can be paid per distinct key instead of per element, and the quantified break-even `T'_max ≈ 0.29 · 2^b`. None of the rejections above covers it.

## Kill criterion

Any one of: (1) the measured number of distinct keys per wide `Requant` table per 16K-token proof exceeds `0.29 · 2^b` of the dense alternative (for `b = 32`, `1.25×10^9`), that is, wide-key activations are spread rather than concentrated, so no `T'_max` below the dense cost is ever safe; (2) RFC-0013's threat model rejects any input-dependent completeness condition and no provable per-table bound below `min(N_table, 2^b)` exists (then the only admissible variant is the provable one, which loses, above); (3) the v0.2 commitment report measures a split fraction small enough that RISK-2 does not bite (then the lever is worth less than 3% at 16K).

## Cheapest experiment

Hypothesis: on a seeded synthetic Llama-3.1-8B-shaped integer model, the distinct accumulator keys per wide `Requant` table per 2K–16K-token proof number below `0.1 · 2^b` for `b ∈ {36, 44, 58, 60}` under the compiler's worst-case ranges, and a two-level LogUp over `F65537` with per-slot membership rejects every adversarial slot with an empirical false-accept rate below `(N + T'_max)/|F|`.

Method, in order:
1. E0 (`s`): the cost-model scenarios above, a per-slot operation count in a table beside RFC-0009 'Summary of costs', and the zero-knowledge question written as a precise statement (an input-dependent completeness failure of probability `p` per proof, the bit it leaks, the shape alternatives) for RFC-0013's owner and its open-question list.
2. E1 (`s`–`m`): seeded random INT8 weights and prompts (rule 6, no checkpoints); run the integer forward pass, histogram the distinct keys of every wide `Requant` instance at 2K, 4K, 8K, 16K tokens and report the ratio to `2^b` and the growth with context; then prototype the two-level LogUp over the test field `F65537` with adversarial tests (a slot with the wrong `value'`, a slot with `key'` outside the verifier-derived range, a wrong `m'`, a lookup matching no slot, a slot with `m' = 0` and garbage) and measure the false-accept rate against `(N + T'_max)/|F|` over 10^5 seeds. The machine is named in the report.

Effort: `s` for step 1, `m` for step 2 (a Rust experiment workspace depending by path on `tessera-field`'s test field). Unknown until an H100: the GPU hash-histogram cost against the counter array (both are `N` random-access increments, but a hash table has worse locality). Unknown until a v0.2 commitment-report run on a real checkpoint: the distinct-key count on real activations, which synthetic weights understate because random weights spread accumulators more uniformly than trained ones.

## Refutations

Sweep triage of 2026-10-16: four refuter agents who did not write the idea, one per lens; each argument is condensed from that refuter's verdict.

| Lens | Verdict | Argument |
|---|---|---|
| soundness | refuted | G-4 and SP-3 hold, but not by the note's argument: a slot with m' ≠ 0 need not equal any lookup element, since two slots with the same bogus t' and m'_1 + m'_2 ≡ 0 (mod P) cancel their residue and pass the identity of RFC-0008 'Relation'; what rescues membership is that every slot, whatever its m', carries the Range and Clamp lookups, which bound rem' and pack(c', q') as true integers so that key' = 2^r·pack + rem' + const is a genuine row by Lemma 2's Euclidean argument (RFC-0009, Lemma 0), and the lookup side then follows as today with error (N + T'_max)/q' plus the N·T'_max/q' β-collision term the note computes (2^−60 per table, E0, primary). Zero knowledge is where it dies: RFC-0013 promises statistical ZK with a witness-free simulator and indistinguishable transcripts for two witnesses; an honest prover that aborts or changes the public claim-list shape when a prompt's activations touch more than T'_max distinct keys is distinguished with probability 1, leaking at least one bit about secret-class activations (05 §3), and RFC-0006 Alternatives Considered 4 already rejects exactly this shape of input-dependent completeness. The only ZK-admissible cap is the provable min(N_table, 2^b), which the note's own table shows loses (240 ps per slot against 70 ps per split element; wins only for b ≤ 34, below every RISK-2 width ≥ 36; E0, primary). Under research rule 1 a G-5 weakening is rejected whatever it saves; the one salvage is a reopened issue if RFC-0013's owner amends the threat model, which nothing in the RFC text supports. |
| prior art | weakened | Not rejected and not excluded: RFC-0008 Alternatives Considered 1 and 2 reject one-hot arguments (per-lookup committed chunks) and univariate two-level lookups (pairing commitments), neither of which this is, and the separation principle [Jo26] explicitly leaves open the once-per-proof table term and the key width, which are exactly the terms this moves. The mechanism is, however, the committed-subtable idea of Caulk/Baloo/cq (acknowledged) transplanted into LogUp-GKR with `RequantSplit`'s Euclidean-division arithmetic (RFC-0009 Lemma 2) moved from the lookup side to the slot side; the only new element is paying that arithmetic per distinct key. Prior art also supplies the known answer to the public-size problem: those schemes pad the subtable to the lookup count, and the note's own table shows that sizing (T'_max = min(N_table, 2^b)) loses (+24 s at 16K), so the surviving variant rests entirely on a statistical cap with the ZK unknown that the soundness lens settles against it. RFC-0005 Alternative 3 (affine scales) would be reopened as a consequence. |
| cost model | weakened | The scenario removes the entire RISK-2 excess (`commits/lookups_per_token=1.757e7`), rerun −14.0 / −8.7 / −3.3 / −0.4%; the term it moves matters only at short context (K_lin halved is −10.6% at 1K and −2.5% at 16K, baseline.md §2), and at the rubric's default 16K the −3.3% is inside the 1.757e7-vs-2.82e7 uncertainty of C_lin itself (00 §9, an upper bound the v0.2 report replaces). The realized gain rests on two unmeasured inputs: the split fraction actually chosen by the compiler and the distinct-key count per wide table, which must fall below 0.29·2^b for any T'_max to beat the dense side, and the note states the count is unknown at every evidence level. Its per-proof table also mixes units: the '1K' column uses a 1K-token proof (6.25 GPU-s) while the model amortizes per-proof terms over a 16K-token batch (`batch_tokens`, `stream_tokens` = 16384), so in the model's own units the dense-side saving at 1K is about 3% of μ, not the −3.2 s of 6.25 s the row implies. A real short-context lever with an honest kill criterion, but the −14% is an upper bound on an upper bound until the E1 histogram exists. |
| feasibility | weakened | The decisive input, distinct keys per wide `Requant` table, cannot be obtained here: the integer model, compiler, range analysis, lowering and witness engine are stubs (`tessera-compiler`, `tessera-ir`, `tessera-gadgets`, `tessera-witness`; issues #77, #78, #80 open, #78 sized `l`), no Python `tessera_quant` package exists in the tree, and research rule 6 forbids the real checkpoints the note admits are needed because synthetic weights understate concentration. Kill criterion (1) therefore waits for the v0.2 commitment report (#88), outside `research/`. The soundness half is feasible at E1: `F65537` is exported (`crates/tessera-field/src/lib.rs`) for the two-level LogUp prototype. Adoption touches five RFCs (0006 gadget selection, 0008 table kinds, 0009, 0011, 0019) on tier-A table and claim-list definitions with the zero-knowledge row failing: effort `l` with the gain unmeasurable until v0.2. |

## Score

| Gain | Confidence | Cost | Priority |
|---|---|---|---|
| −14.0% μ(1K), −3.3% μ(16K) (E0 cost model on the RISK-2 worst-case count, an upper bound on an upper bound) for the statistical-cap variant only; the ZK-admissible cap min(N_table, 2^b) has a negative gain (+24 s at 16K; E0, the note's own table) | 0.1: the gainful variant fails the zero-knowledge gate and the admissible variant loses; the distinct-key histogram is unknown at every evidence level | `l`: five RFCs on tier-A table and claim-list definitions; the decisive input is unmeasurable before the v0.2 report (#88); the soundness half is E1 on F65537 at `m` | 0: hard gate. Security-gate fail on zero knowledge (G-5), and RFC-0006 Alternatives Considered 4 covers input-dependent completeness with nothing new; research rule 1 rejects a G-5 weakening whatever it saves |

## Decision and links

Triage 2026-10-16: `rejected`. The soundness lens resolved the zero-knowledge `unknown` to `fail` (an honest prover whose claim-list shape or abort depends on how many distinct keys the secret activations touch distinguishes two witnesses with probability 1, against RFC-0013's witness-free simulator and 'Malicious verifier'), and the only ZK-admissible cap loses by the note's own table. Reopens if RFC-0013's owner amends the threat model to accept input-dependent completeness (nothing in the RFC text supports it), or if a same-public-shape fallback for the over-cap case is found; the distinct-key histogram from the v0.2 commitment report (#88) would then size T'_max. The soundness-lens observation that slot membership rests on the per-slot Range and Clamp lookups, not on m' ≠ 0, is worth keeping for any future sparse-table design. Experiments: none yet. RFC or issue: none. Ledger line updated on: pending (the consolidation agent merges this note).
