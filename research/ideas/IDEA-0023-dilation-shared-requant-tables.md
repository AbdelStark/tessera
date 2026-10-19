# IDEA-0023: Dilation-shared requantization tables: one `2^32`-row table per `(lo, hi)` by scaling keys with public powers of two, and a table-side term in the cost model

| | |
|---|---|
| Status | `triaged` (2026-10-16) |
| Origin | [SUR-0002](../surveys/SUR-0002-lookups.md) Question 3 synthesis ("the once-per-proof table term is where RFC-0008 spends structure"; its wider-`B_max` seed priced ten `2^32` tables at about 9% of linear-layer lookups per proof) and ledger queue item 31; [RFC-0006 'Gadget selection'](../../docs/rfcs/RFC-0006-compiler-ir.md#gadget-selection) ("a requantization table is identified by `(shift, lo, hi, b)`"); [RFC-0008 Drawbacks](../../docs/rfcs/RFC-0008-logup-and-tables.md#drawbacks) ("a small table used rarely still pays its full row count per proof"); the cost model has no table-side term ([tools/cost_model.py](../tools/cost_model.py), read 2026-10-16: no parameter prices table rows; E0, primary). New |
| Lever | `C_lin` and `K_lin` per token through the once-per-proof table side: LogUp-GKR commits `2^b` multiplicities and adds `2^b` fraction-tree leaves per distinct table definition per proof ([RFC-0008 'Relation', 'Fraction tree', 'Batching'](../../docs/rfcs/RFC-0008-logup-and-tables.md#relation)), amortized over the 16K-token batch; also multiplicity-counter memory ([08 §4](../../docs/spec/08-performance-budget.md#4-memory-budgets): `≤ 2^32` rows × 4 B per table) and the verifier's dense-table count ([08 §4](../../docs/spec/08-performance-budget.md#4-memory-budgets): `≤ 20` dense cores) |
| Spec surfaces | RFC-0006 'Gadget selection' (the table identity rule `(shift, lo, hi, b)` → `(lo, hi, class-set shape)`, a public per-tensor dilation `j_t`); RFC-0008 'Table kinds' and 'Relation' (the lookup key carries a public scale `2^{j_t}`; `TableSetManifest` lists shared instances); RFC-0007 (nothing new is hidden: `j_t` is derived from `act_exponents` and `shift_class_sets`, which are public); RFC-0009 `RequantSplit` (split point `s + j_t`); RFC-0019 `tv-tables` (re-derives `j_t`, evaluates one MLE per shared table). Table definitions are tier A ([02 §1](../../docs/spec/02-public-api.md#1-stability-tiers)): promotion is an RFC amendment to RFC-0006 and RFC-0008. The cost-model term (4) is non-normative |
| Priority | 0.13 (−1.7% × 0.3 / 4 at 1K, the short-context lever the note targets, at 10 distinct wide tables today; 1.0 at 50 tables; a tier-A amendment to RFC-0006 and RFC-0008); decided by the v0.2 `TableSetManifest`'s distinct-table count |

## Mechanism

Setting ([RFC-0005 'Integer semantics'](../../docs/rfcs/RFC-0005-integer-model.md#integer-semantics), [RFC-0008 'Table kinds'](../../docs/rfcs/RFC-0008-logup-and-tables.md#table-kinds)): every requantization is `rq(k, r, lo, hi) = clamp(floor((k + 2^{r−1}) / 2^r), lo, hi)` with a power-of-two shift `r` in `pot` mode; the table `Requant { b, classes, r_by_class, lo, hi }` has `|classes| · 2^b` rows and is shared only between tensors with identical definitions, i.e. identical `(shift, lo, hi, b)` (RFC-0006 'Gadget selection'). In the 405B `minimal` profile the eight requantized tensor kinds of 126 layers carry per-layer, per-class activation exponents, so the shift differs from layer to layer and the sharing rule produces many wide tables whose definitions differ only in `r`.

**The identity (E0, primary: the `rq` definition).** For every integer `k`, every `r ≥ 1` and every `j ≥ 0`,

```
rq(k · 2^j, r + j, lo, hi) = rq(k, r, lo, hi),
```

because `(k · 2^j + 2^{r+j−1}) / 2^{r+j} = (k + 2^{r−1}) / 2^r` as rationals, so the floors and clamps agree. The same holds for `Silu` (`h = (k · 2^j) >> (r + j) = k >> r`, and `σ_h`, `σ_out` are properties of `h` and of the output, not of `k`) and for `Exp` (`k · 2^j` with `scale / 2^j` gives the same real argument `k · scale`). Checked exactly with Python integers on 2026-10-16: 346,304 cases exhaustively for `b ∈ {6, 9, 12}`, every `r ∈ [1, b)` and `j ∈ [0, 6]`, and 10^6 sampled `(b, r, j, k, lo, hi)` at production widths `b ∈ {22, 24, 25, 29, 30}` with `j ≤ 30 − b`; zero violations of the `Requant` and `Silu` forms and of the width bound `|k · 2^j| < 2^{b+j−1}` (E0, primary: scratchpad script of this note, to be promoted into the experiment).

Hence one table with shift `r_T` serves every tensor with shift `r_t ≤ r_T` through the dilated virtual key `k · 2^{j_t}`, `j_t = r_T − r_t`: a public linear map on a virtual key, which costs no commitment and no lookup (the key polynomial is virtual, RFC-0008 'Fraction tree'; its scale enters the claim handed to the producing stage, RFC-0010).

What changes:

1. **Table identity (RFC-0006 'Gadget selection').** A requantization table is identified by `(lo, hi, class-set shape)` instead of `(shift, lo, hi, b)`. The class window stays a 4-consecutive-shift tuple as today (RFC-0005 `pot` mode: "at most 4 consecutive weight-exponent classes"), so one dilation `j_t` aligns the whole window: the table's `r_by_class` is `r_{T,c} = r_{t,c} + j_t` for every class `c`. For each tensor `t` the compiler records `j_t` (public: per-tensor activation exponents and class sets are `act_exponents` and `shift_class_sets` of the public claim list, [RFC-0007 'Public and hidden quantization data'](../../docs/rfcs/RFC-0007-claim-list.md#public-and-hidden-quantization-data)) and emits `key' = pack(c, acc · 2^{j_t})`. The shared table's width is `b_T = r_T + max_t (b_t − r_t)`: tensors with the same output range and the same pre-clamp headroom `b_t − r_t` dilate to the same width, which is why the table collapses by output shape rather than by layer.
2. **Split tensors (RFC-0009 `RequantSplit`).** The split point becomes `s_t + j_t`, so the reduced key `z_hi` keeps width `b − s = 30` (the `B_max = 32` budget less two class bits; the compressed key must stay below `2^{B_max}`, RFC-0008 'Table kinds'). The low limb's `Range(s_t + j_t)` is one commitment and one lookup as before, and `Range` tables are tiny and already shared, so split tensors share at zero cost. The precondition `s ≤ r − 1` is invariant under `(s, r) ↦ (s + j, r + j)`. `Requant1` tensors share only within their slack `30 − b_t`: V at `b = 29` has 1 bit (RFC-0005: key `< 2^29`); the norm after [IDEA-0011](IDEA-0011-rsqrt-fact-norm-key.md) at `b = 24–25` has 5–6 bits; `Exp` at `key_bits = 22` has up to 8–10 bits (E0, primary: RFC-0005 range cells).
3. **Extension (optional, larger surface).** Replace the 4-class tuple by a hidden per-channel factor `f_c = 2^{r_T − r_c}`, committed as a channel-parameter vector exactly where the hidden class index lives today ([RFC-0012 'Layout'](../../docs/rfcs/RFC-0012-weight-commitment.md#layout), `ChannelParams`; RFC-0007: "the key is the virtual linear combination `acc + 2^b · class_c`"), with `key = acc · f_c` (a degree-2 virtual key) and one lookup per channel per proof into a 4-row table proving `f_c` lies in the public set `{2^{r_T − r} : r ∈ class set}`. This removes the class dimension (`b` up to 32) and the 4-class fidelity limit (RFC-0007 Drawbacks). Not needed for the gain below; recorded so the triage can judge it separately.
4. **Cost model.** [tools/cost_model.py](../tools/cost_model.py) gains a term `table_rows_per_proof × (commit_ps + leaf-or-lookup cost) / (native(S) × batch_tokens)` behind a parameter defaulting to 0, so the golden check still holds and the `TableSetManifest` row count becomes a priced input.

Verifier: `tv-tables` evaluates the shared table's MLE exactly as today (structured evaluation is per table definition, RFC-0008 'Structured evaluation'); only the public scale `2^{j_t}` of each key enters the claim list the leaf stage hands to the producing stage.

## Why it could work

- The table side is real and uncounted. LogUp-GKR commits one multiplicity column per table instance and the combined tree has `T` table leaves per instance ([PH23], E0, primary; RFC-0008 'Relation' and 'Fraction tree', E0, primary); the cost model prices neither (E0, primary: no `table` parameter in `cost_model.py`), and RFC-0008's own Drawbacks name the cost.
- The separation principle leaves exactly three levers, the lookup count, the key width and the once-per-proof table term ([Jo26], E0, primary; SUR-0002 Question 3), and this idea moves the third without touching per-lookup cost.
- One `2^32`-row table costs `2^32 × (c_c + c_ℓ) = 2^32 × 70 ps = 0.30 s` per proof when its leaf is priced at the lookup cost, `2^32 × 42 ps = 0.18 s` at the model's `leaf_ps = 12` (E0, primary: the model's unit costs, themselves assumptions until RFC-0020). Against `native × batch_tokens = 11.3 s` (1K) and `12.1 s` (16K) per 16K-token batch (E0, primary: `native_s_per_token` from `--json`), with the model's glue factor 1.15, that is `+0.031 μ` (1K) and `+0.029 μ` (16K) per table at 70 ps, `+0.018 / +0.017 μ` at 42 ps (E0).
- How many distinct `(shift, lo, hi, b)` tuples the 405B `minimal` profile produces is unknown until the v0.2 `TableSetManifest` (RFC-0006 'Rollout': v0.2 delivers gadget selection and the commitment report). SUR-0002 assumed ten (claim level); 8 requantized tensor kinds × 126 layers with `pot`-mode exponents that vary by a few bits per layer make 10–50 plausible (claim level, this note). Amortized per token, `T` tables of `2^32` rows add `T × 2^32 / 16,384 = T × 2.62×10^5` committed multiplicities and as many leaves: `2.6×10^6` (10) to `1.3×10^7` (50) against `C_lin = 2.82×10^7` (E0, corrected profile), i.e. 9–46% of the linear-layer count.
- Dilation sharing collapses the wide tables to one per `(lo, hi)` shape: about five (INT8 `Requant`, INT20 residual `Requant`, `Silu`, `Exp`, `Range`) regardless of layer count (E0, by construction from RFC-0005's table of rescaled tensors). Stwo batches GKR instances across heights in one sumcheck per level, which the shared instances rely on and which RFC-0008 'Batching' already assumes ([Stwo26], E0, primary).
- Side effects in the same direction (E0, primary: 08 §4): multiplicity counters fall from `T × 2^32 × 4 B` (160 GB at `T = 10`, 800 GB at 50) to about 80 GB across the GPUs; the verifier's dense-table count stays inside the `≤ 20` cores budget, which 50 distinct tables would exceed.

## Expected gain

The cost model has no table-side term, so the scenario adds the amortized table side to both counts, `commits_per_token = lookups_per_token = 2.819×10^7 + T × 2^32 / 16,384`, pricing a table leaf at `lookup_ps` (upper end; at `leaf_ps` the term is 60% of this). The idea's gain is the difference between the with-table-side baseline (`T` = 10 or 50 distinct wide tables under today's identity rule) and the shared case (`T` = 5). Numbers measured on 2026-10-16 (E0, primary).

```sh
python3 -I research/tools/cost_model.py --set commits_per_token=3.081e7 --set lookups_per_token=3.081e7   # today, 10 tables
python3 -I research/tools/cost_model.py --set commits_per_token=4.130e7 --set lookups_per_token=4.130e7   # today, 50 tables
python3 -I research/tools/cost_model.py --set commits_per_token=2.950e7 --set lookups_per_token=2.950e7   # shared, 5 tables
```

```
baseline: profile 'corrected'
   S   linear  attention  weights  regen  streaming    total
   1K     3.30       1.13    0.021    4.0       0.42      8.9
   4K     3.25       4.46    0.021    4.0       2.32     14.1
  16K     3.06      16.82    0.020    4.0      11.24     35.1
 128K     1.99      87.48    0.013    4.0      77.93    171.4

scenario: commits_per_token=3.081e7, lookups_per_token=3.081e7          (today, 10 tables)
   S   linear  attention  weights  regen  streaming    total
   1K     3.61       1.13    0.021    4.0       0.42      9.2   Δ   +3.5%
   4K     3.55       4.46    0.021    4.0       2.32     14.4   Δ   +2.1%
  16K     3.35      16.82    0.020    4.0      11.24     35.4   Δ   +0.8%
 128K     2.18      87.48    0.013    4.0      77.93    171.6   Δ   +0.1%

scenario: commits_per_token=4.13e7, lookups_per_token=4.13e7            (today, 50 tables)
   S   linear  attention  weights  regen  streaming    total
   1K     4.84       1.13    0.021    4.0       0.42     10.4   Δ  +17.3%
   4K     4.76       4.46    0.021    4.0       2.32     15.6   Δ  +10.8%
  16K     4.49      16.82    0.020    4.0      11.24     36.6   Δ   +4.1%
 128K     2.92      87.48    0.013    4.0      77.93    172.3   Δ   +0.5%

scenario: commits_per_token=2.950e7, lookups_per_token=2.950e7          (shared, 5 tables)
   S   linear  attention  weights  regen  streaming    total
   1K     3.46       1.13    0.021    4.0       0.42      9.0   Δ   +1.7%
   4K     3.40       4.46    0.021    4.0       2.32     14.2   Δ   +1.1%
  16K     3.21      16.82    0.020    4.0      11.24     35.3   Δ   +0.4%
 128K     2.08      87.48    0.013    4.0      77.93    171.5   Δ   +0.1%
```

The printed `Δ` is against the model's baseline, which omits the table side; the idea's Δμ is shared-against-today, computed from the `--json` totals (E0):

| Context | Δμ, 10 → 5 tables | Δμ, 50 → 5 tables | Δ proof size | Δ verification |
|---|---|---|---|---|
| 1K | −1.7% (9.18 → 9.03) | −13.3% (10.41 → 9.03) | opened advice columns shrink by the removed multiplicities: 5 × 2^32 = 2.1×10^10 elements per proof, about −0.8% of the committed elements of a 16K/16K proof unit (`4.6×10^11` linear + `2.2×10^12` attention, E0); −7% for 45 tables removed; the combined rows and weight openings do not move | 5 (or 45) fewer `t̃(ρ)` evaluations, `O(core)` each (RFC-0008 'Structured evaluation'; 08 §4 budgets about 2 s for 20 cores), and the table count stays inside the `≤ 20` dense-core budget that 50 tables would break |
| 4K | −1.1% (14.36 → 14.21) | −8.7% (15.57 → 14.21) | as above | as above |
| 16K | −0.4% (35.43 → 35.29) | −3.5% (36.57 → 35.29) | as above | as above |
| 128K | −0.05% (171.6 → 171.5) | −0.5% (172.3 → 171.5) | as above | as above |

Under the `counted` profile (`c_c` 94, `c_ℓ` 101 ps) the same scenarios give −2.4 / −1.4 / −0.5 / −0.07% (10 → 5) and −17.8 / −11.1 / −4.3 / −0.6% (50 → 5) (E0, primary, same runs with `--profile counted`). Pricing the table leaf at `leaf_ps` instead of `lookup_ps` scales every Δ by 0.6. Scored at 1K as a short-context lever (rubric: 16K unless the idea targets short context).

Evidence levels of the inputs: unit costs E0 primary (the model's assumptions, 08 §2; RFC-0020 replaces them); `C_lin = 2.82×10^7` E0 primary (corrected profile, itself the RISK-2 upper bound); `2^32` rows per wide table E0 primary (`TableLimits.max_log2_rows`, RFC-0006); the number of distinct wide tables today, 10–50, **claim level** (SUR-0002 assumed 10; nothing has counted it); five shared tables E0 by construction. The table count is the uncertainty, and it sets the gain from −1.7% to −13.3% at 1K.

## Security gate

| Property | Effect | Pass |
|---|---|---|
| Soundness bits (G-4, RFC-0014) | The dilation is an exact integer identity. The dilated key `acc · 2^{j_t}` is a true integer whenever `acc` is (range analysis bounds `acc`, and the compiler checks `|acc| · 2^{j_t} < 2^{b_T − 1} ≤ 2^31 < (P − 1)/2`, INV-IR-2, Lemma 0), so its field value equals its integer value. The shared table is still a function of its key, so Lemma 1 (`Requant1`) holds unchanged, and Lemma 2 (`RequantSplit`) holds with `(s + j_t, r + j_t)`, whose precondition `s + j_t ≤ r + j_t − 1` is the old one. The LogUp error `(N + T)/|Fp2|` per instance (RFC-0014 budget line "`N + T ≤ 2^38` per instance") grows only through `N` (more lookups share one instance; `N` summed over the shared tensors is the same total as before, now in fewer instances, each still far below `2^38` at 16K: SUR-0002 counts `N ≈ 4.6×10^10 < 2^36` per table); `T = 2^32` per table is unchanged and there are fewer tables, so the union bound over instances shrinks. No challenge, label, absorption or opening changes (INV-FS-1..4 untouched). | pass (E0) |
| Uniqueness upstream of the logits (SP-3, RFC-0009 Lemma 0) | Every committed output is still the unique table value of a determined key: `Requant1` by Lemma 1 on `(c, acc · 2^{j_t})`; `RequantSplit` by Lemma 2 with the shifted split point. No committed value is added or removed per element. | pass (E0) |
| Zero knowledge (G-5, RFC-0013) | `j_t` is public but a function of data RFC-0007 already publishes (`act_exponents`, `shift_class_sets`); the table set and every key scale are functions of the claim list, so the claim-list shape is input-independent and the RFC-0013 simulator is unchanged; multiplicity columns stay committed advice under RFC-0011 masks with the `(0, δ)` leaf per tree. The optional hidden factor `f_c` (extension 3) reveals exactly what the hidden class index reveals today (membership in a public 4-element set) and is committed with the weights (RFC-0012 `ChannelParams`). | pass (E0) |
| Verifier independence (SP-6) | The identity rule and the key scale are specification text: `tv-*` re-derives `j_t` from the public `QuantSpec` (SP-7, INV-IR-5) and evaluates one MLE per shared table definition; nothing prover-declared is trusted. | pass |
| Post-quantum (hash-based commitments) | Unchanged. | pass |

A `fail` anywhere rejects the idea. No `unknown` in this gate: the open input (the distinct-table count) is a cost input, not a security one.

## Prior rejection check

Checked against [baseline.md §7](../baseline.md#7-already-rejected) and the RFCs' Alternatives Considered on 2026-10-16:

- [RFC-0008 Alternatives Considered](../../docs/rfcs/RFC-0008-logup-and-tables.md#alternatives-considered) 1–6 (one-hot lookups, univariate lookups, base-field challenges, separate trees, shipping table contents, floating-point generators): none concerns table sharing by key dilation; 5 ("shipping table contents") is the opposite direction, and the verifier still regenerates every shared table.
- [RFC-0006 Alternatives Considered](../../docs/rfcs/RFC-0006-compiler-ir.md#alternatives-considered) 1–5: 1 (search over requantization placements) moves rescales and changes `D`; this idea leaves `D` bit-identical. 4 (calibration ranges) does not apply: `j_t` and `b_T` come from public exponents and hold for every input.
- RFC-0006 already shares tables "with identical definitions"; this makes definitions identical where today only the shift differs. It is the natural completion of that rule, not a reversal.
- [IDEA-0007](IDEA-0007-dedup-table-side-wide-requant.md) (deduplicated table side) was rejected on zero knowledge because its public shape depended on the secret activations; this idea keeps the full `2^b` row enumeration and an input-independent shape, so that objection does not apply.
- SUR-0002's wider-`B_max` seed and OQ-0006-1 are orthogonal and become cheaper under this idea (fewer tables to widen); [IDEA-0011](IDEA-0011-rsqrt-fact-norm-key.md) narrows the norm key, which enlarges the slack `30 − b_t` this idea uses.
- baseline §7 has no row on table sharing; baseline §6 lists "structured tables, batching" as a lever on `verify` and "tuple tables" on `c_ℓ`, `K_lin`, without a rejection.

New: the observation that a public power-of-two dilation of a virtual key makes requantization, SiLU and exponential tables with different shifts identical, and the pricing of the table side as a per-token term. "None" after the check.

## Kill criterion

Any one of: (1) the v0.2 `TableSetManifest` for 405B shapes lists at most 6 distinct tables of `2^30` rows or more under today's `(shift, lo, hi, b)` rule, leaving nothing to share (the gain is then below the 10-table scenario's −1.7% at 1K, inside the unit-cost uncertainty); (2) the RFC-0020 gate measures the table-side cost per row below 10 ps (zero-multiplicity blocks skipped and table leaves far cheaper than lookup leaves), making the term under 1% of `μ(1K)` even at 50 tables; (3) the slack `30 − b_t` is zero for the tensors that dominate the distinct-table count (V and `Silu` at `b = 29–30`) and the split point cannot absorb `j_t` because `s + j_t > r − 1` for some class, so those tensors cannot share.

## Cheapest experiment

Hypothesis: (a) `Requant_{r+j}(k · 2^j) = Requant_r(k)`, `Silu_{r+j}(k · 2^j) = Silu_r(k)` and `Exp_{scale/2^j}(k · 2^j) = Exp_{scale}(k)` over the exact integer generators for every key in range at test parameters and for 10^6 sampled keys at production parameters; (b) a synthetic `pot`-mode `QuantSpec` for 405B shapes, with per-layer exponents drawn from a stated distribution (a few bits of spread per layer, as RFC-0005's "unusually small magnitudes" drawback suggests), yields between 10 and 50 distinct `(shift, lo, hi, b)` tuples under today's rule and at most 6 under the `(lo, hi, shape)` rule.

Method, in order:
1. E0 (`s`, one day): promote this note's scratchpad script into `research/experiments/EXP-NNNN-dilation-shared-tables/` with the pre-registered criterion (zero violations; the Python reference generators for `Silu` and `Exp` are written for the experiment with 128-bit fixed point as RFC-0008 'Generators and digests' specifies, since no `tessera_quant` package exists in the tree yet), the distinct-tuple count of (b) with its interval, and the table-side term added to `cost_model.py` behind a parameter defaulting to 0 (golden check unchanged). E1 on this CPU is reachable for (a) but adds nothing: the identity is exact.
2. The adversarial test belongs with it: a key `k' ≠ k · 2^{j_t}` (a non-multiple of `2^{j_t}`) that passes the shared table must produce the wrong `y` for the original tensor only if the compiler's width check `|acc| · 2^{j_t} < 2^{b_T−1}` is violated, which the test demonstrates over `F65537` with the precondition broken and absent with it kept (as RFC-0009's testing strategy does for `s > r − 1`).

Effort: `s` for the experiment; the RFC amendment to RFC-0006 and RFC-0008 is `m` in size but touches tier-A table definitions (`l` by the rubric's letter; no protocol version is frozen before v0.3). What stays unknown until v0.2 (no H100 needed): the actual number of distinct wide tables, which decides the gain between −1.7% and −13.3% at 1K. What stays unknown until the RFC-0020 gate (E2): the table-side cost per row, between the model's 42 and 70 ps.

## Refutations

Sweep triage of 2026-10-16: four refuter agents who did not write the idea, one per lens; each argument is condensed from that refuter's verdict, with the cost-model reruns the scoring agent made to check disputed cells noted as such.

| Lens | Verdict | Argument |
|---|---|---|
| soundness | survives | The identity `rq(k·2^j, r + j) = rq(k, r)` follows directly from RFC-0005's definition `rq(x, r, lo, hi) = clamp(floor((x + 2^{r−1}) / 2^r), lo, hi)` (RFC-0005:62) as an equality of rationals, and the table's value depends only on the key, so Lemma 1 holds on `(c, acc·2^{j_t})` and Lemma 2 holds with `(s + j_t, r + j_t)`, whose precondition is the old `s ≤ r − 1` (RFC-0009:38-50). As ADV-P the refuter looked for a second witness through the non-multiples of `2^{j_t}` the shared table also contains: the dilated key is a virtual true integer pinned upstream by range analysis (SP-7, re-derived by `tv-tables`), `abs(acc)·2^{j_t} < 2^{b_T−1} ≤ 2^31` keeps the hidden-class pack injective and Lemma 0 applicable, and a non-multiple row can only be reached if `acc` is not a true integer, which the upstream argument excludes; for `Exp`, a dilated key in `[0, 2^{key_bits+j})` still forces `k ∈ [0, 2^{key_bits})` by Lemma 0, so Lemma 5 is intact. As ADV-V nothing leaks: per-tensor activation exponents and shift class sets are public in RFC-0007's table (RFC-0007:77-78), so `j_t` is a function of already public data, and the class assignment stays hidden in the pack. One budget to watch rather than a kill: merging every INT8 requant of 126 layers into one instance puts `N` near 2^36.6–2^37.3 at 16K tokens, inside RFC-0014's `N + T ≤ 2^38` per-instance line (RFC-0014:99) but with only 1.6–2.5× margin, so the RFC amendment must have the planner check that line per shared instance. |
| prior art | survives | The specification already shares one table across shifts, but only within a tensor: `pot` mode limits each tensor to 4 consecutive shift classes packed into the key as `(class_c, acc)` (RFC-0005:58; RFC-0007:82), tables are shared only 'with identical definitions', identified by `(shift, lo, hi, b)` (RFC-0006:83), and RFC-0008's Drawbacks concede that 'a small table used rarely still pays its full row count per proof' (RFC-0008:129), so cross-tensor sharing by a public power-of-two dilation of the virtual key is not in the spec and is the natural completion of its own rule. No Alternatives Considered row rejects it: RFC-0008's 1–6 (RFC-0008:119-124) and RFC-0006's 1–5 (RFC-0006:121-125) are on other subjects, RFC-0007 Alternative 2 rejects public per-channel classes (RFC-0007:121) whereas `j_t` is per tensor and derived from the already-public `act_exponents` (RFC-0007:50), and IDEA-0007's zero-knowledge failure (ledger) does not transfer because the row enumeration and shape stay input-independent. [Jo26] supports rather than excludes it: the once-per-proof table term is exactly the lever the separation principle leaves open (SUR-0002:39, 57). One claim of novelty is overstated: SUR-0002 Question 3 already priced the table side (ten `2^32` tables as 9% of linear-layer lookups, SUR-0002:59, 69), so the cost-model term formalizes a known gap, while the dilation identity itself is new. The unknown table count (claim level) is a cost-model matter. |
| cost model | weakened | The scenarios reproduce (corrected: 10 tables 9.2 / 50 tables 10.4 / shared 9.0 at 1K; native 11.26 s per 16K batch at 1K; `T × 2^32 / 16,384 = 2.62M` per token per 10 tables), and the identity is exact, so the mechanism is not in doubt. The gain is not larger than its inputs: it is proportional to `(T − 5) × rows × price`, where `T` is claim-level (10–50, 'nothing has counted it'), every table is assumed at the `2^32`-row maximum (RFC-0006 `B_max`; `Exp` at `2^22` rows would contribute 1/1,000 of that), and the per-row price spans 42–70 ps (×0.6), so the 1K gain runs from −1.0% to −13.3% and at the rubric's 16K it is at most −3.5% (−0.4% at `T = 10`), inside the unit-cost uncertainty at every point. The idea's own kill criterion (1) fires on `T ≤ 6` from the v0.2 `TableSetManifest`, which is the first thing to learn; until then the robust benefits are the verifier-side ones (≤ 20 dense cores, multiplicity memory) rather than Δμ, and scoring it at 1K puts it where `regen` (23–45% of μ) dominates anyway. |
| feasibility | weakened | The dilation identity is exact (checked against RFC-0008:61-63: `rq` depends on `(k + 2^{r−1}) / 2^r` as a rational, `Silu` on `h = k >> r`, `Exp` on `k · scale`), the key is virtual so it costs no commitment, and the width arithmetic is consistent with RFC-0006:79-83 (`B_max = 32`, `RequantSplit` above it) and RFC-0005:58 (4-class windows). Feasibility is the problem: table definitions are tier A (02 §1), so this is an RFC amendment to RFC-0006 and RFC-0008 plus `tv-tables` re-deriving `j_t` (effort `l` by the letter); the gain's only uncertain input, the number of distinct wide tables (10–50, claim level), cannot be measured here because `tessera-compiler` and `tessera-tables` are 5–6-line stubs (crates/tessera-compiler/src/lib.rs, crates/tessera-tables/src/lib.rs) and the v0.2 `TableSetManifest` does not exist, so the pre-registered count (b) is over a synthetic `QuantSpec` the note itself designs; and the tensors most likely to multiply the table count (V at `b = 29`, `Silu` at 29–30) have 0–1 bits of slack under `B_max = 32`, so they share only by becoming `RequantSplit` (2 commitments + 2 lookups per element, far more than the table side saves) or not at all. The cheapest experiment reaches E0 only; nothing here is or can be E1, and the Δμ range (−1.7% to −13.3% at 1K) is set entirely by a claim-level count. |

## Score

| Gain | Confidence | Cost | Priority |
|---|---|---|---|
| −1.7% μ(1K) at 10 distinct wide tables today (−13.3% at 50; −0.4% / −3.5% at 16K; E0 cost model under `corrected` with the amortized table side added to both counts; ×0.6 if a table leaf is priced at `leaf_ps`). The dilation identity is E0 exact (346,304 exhaustive and 10^6 sampled cases, zero violations, primary); the distinct-table count 10–50 is claim level. Scored at 1K as a short-context lever | 0.3: the identity is exact and every gate row passes, but the gain's only driver is a claim-level count that no artefact here can produce before the v0.2 `TableSetManifest`, every table is priced at the `2^32`-row maximum, and the tensors that would multiply the count have 0–1 bits of dilation slack | `l`: a tier-A amendment to RFC-0006 'Gadget selection' and RFC-0008 'Table kinds' plus `tv-tables`; the E0 experiment itself is one day (`s`) | 1.7 × 0.3 / 4 = 0.13 (1.0 at 50 tables) |

## Decision and links

Triage 2026-10-16: `triaged` at priority 0.13. The identity check and the table-side term in `cost_model.py` (behind a parameter defaulting to 0) are worth a day and can be filed now as a research deliverable; the idea's standing is decided by the v0.2 `TableSetManifest`'s distinct-table count (kill criterion 1), so no experiment beyond that E0 check is queued until the count exists, and the RFC amendment waits for it. The verifier-side benefits (dense-core count, multiplicity memory) are the robust part and belong in the amendment's motivation if it is written; the planner check of RFC-0014's `N + T ≤ 2^38` line per shared instance goes with it. Experiments: none yet (the one-day E0 check, as EXP-NNNN when run). RFC or issue: none yet (an amendment to RFC-0006 and RFC-0008 if the v0.2 count justifies it). Ledger line updated on: pending consolidation (2026-10-16).
