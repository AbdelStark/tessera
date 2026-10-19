# IDEA-0032: Certified RoPE ranges with a per-table width limit

| | |
|---|---|
| Status | `shelved` (2026-10-19: public-cap/domain/ZK policy, wider-table specification and E2 fidelity/serving evidence needed) |
| Origin | Merge of [IDEA-0028](IDEA-0028-per-table-bmax-rope-calibrated.md) into this route; [SUR-0009](../surveys/SUR-0009-compiler.md), current RFC-0023/0028/0047/0048 and the current c2/B32 compiler report |
| Lever | Conditional removal of one RoPE split limb and one lookup per Q/K output; shared weight-row norm argument also supports the distinct [IDEA-0031](IDEA-0031-fused-gated-product-rescale.md) fusion proposal |
| Spec surfaces | RFC-0005/0006/0007/0008/0009/0012/0013/0014/0019/0031/0040/0047/0048: model semantics, cap policy, certificate commitment/opening, table limits, soundness accounting and independent verifier derivation; no such production change is made here |
| Priority | 0 active while shelved; historical conditional 0.2 retained below, with no current net performance claim |

## Decision and scope of the merge

This is the canonical merged note. IDEA-0028 is shelved as absorbed here: the repository vocabulary has no `merged` status. This note incorporates its per-table break-even rule, exact table-sharing test, eight-byte-counter accounting and proposed table-side calibration. No new idea ID, experiment, harness, production edit or promotion is created. IDEA-0031 remains separately shelved for real-weight/fidelity and serving evidence; the [shared certificate](#shared-weight-row-norm-certificate) below is its reference too. Its own model-placement change is not merged into the RoPE route.

This 2026-10-19 revision is E0 analytic reasoning from the current specification and existing public compiler records, including [EXP-0017's raw probe](../experiments/EXP-0017-rmsnorm-relational-bound/results/compiler.json). It is not a new experiment. The original proposal and its four-lens reviews are preserved under [historical proposal](#historical-proposal-and-reviews). Their `pass` labels are historical, not current security approval. The present complete protocol/cap-disclosure/ZK gate remains unresolved. No model weights were inspected and no fidelity result is inferred from synthetic data or local rounding bounds.

## Current baseline and route

The [405B servable c2/B32 report](../../docs/perf/v0.2/commitment-report-405b-servable-c2-b32.json) counts `C=45,566,660`, `K=45,440,678`, `M=7,693,467,648` per token/proof as appropriate (E0 compiler counts; RsqrtRow's row costs remain estimated). Q/K RoPE requantizations cover 16,384+1,024=17,408 outputs per layer/token, or 2,193,408 per token across 126 layers. A 16,384-token batch therefore has `N_split=35,936,796,672` candidate split removals (E0 shape arithmetic), conditional on replacing every such site. Their existing full key has signed width 44, shift classes 22/23 and two classes. Under RFC-0028's rounding-bit clamp its split is 21, its shared low table is `Range{21}` and its high table is `Requant{b23,r[1,2],lo−128,hi127}`, with `2^24` padded rows. It is not a `2^32` high table.

These are the current compiler's conservative endpoints. Weights are in `[-127,127]` under RFC-0048's model-domain assumption, norm outputs in `[-128,127]`, so a length 16,384 accumulator is symmetrically bounded by 266,338,304. `Rotate` uses inclusive `abs(C),abs(S)<=2^bits`: with current bits 14 its declared symmetric key endpoint is 8,727,373,545,472 and signed width 44. Treating `2^bits` as excluded, omitting the negative INT8 endpoint, or dropping the signed bit understates the width.

The proposed route combines a public certified row-norm cap, the current norm-output bound, lower-precision public RoPE constants and, only where necessary, a table limit above 32 bits. None is already a production range rule. All current ordinary RMSNorm keys remain split at κ30: [EXP-0017](../experiments/EXP-0017-rmsnorm-relational-bound/README.md) refutes IDEA-0011's current count-saving claim. This route gets no norm-removal credit and does not create a split-free profile. Changing RoPE precision or the quantized weight policy changes `D` and requires real-model fidelity gates and bit-exact serving evidence.

## Shared weight-row norm certificate

This section is shared by this Q/K route and [IDEA-0031](IDEA-0031-fused-gated-product-rescale.md). It states an integer implication and the protocol obligations a future RFC would have to discharge; it is not an implemented or fully reviewed certificate protocol.

### Integer statement and the whole-sum bound

For each true output row `w` of a named tensor, of true fan-in `d`, fix a public integer `k` and `B=2^k`. Under the domain premise `w_i in[-127,127]`, define `W=sum_i w_i^2`, so `0<=W<=16129*d`. Commit four auxiliary integers and range-check each `s_j in[0,2^(k+1))`. The proposed row relation is

```text
sum_i w_i^2 + s_0^2 + s_1^2 + s_2^2 + s_3^2 = 2^(2k).
```

All sums here are integers before interpreting the relation in Goldilocks. With `U=2^(k+1)-1`, the complete left side is at most `L=16129*d+4*U^2`; the right side is `B^2`. A sufficient centered-integer policy is `max(L,B^2)<(P-1)/2`, `P=2^64-2^32+1`, with each intermediate sum/product bounded by those nonnegative totals. Lemma 0 then lifts the field congruence to equality. Bounding one square alone does not establish this. At `d=16384,0<=k<=13`, the largest permitted left side is 1,337,868,292, below 2^31, and `B^2<=67,108,864` (E0 exact arithmetic). The accepted equality implies `W<=B^2`; a row above the cap has no integer slack witness. Conversely, every nonnegative integer `B^2-W` is a sum of four integer squares; each root is at mostB and fits the chosen range. No existence claim is made when the difference is negative.

A future policy must check the displayed inequality for every supported shape/cap. It must reject unsupported large caps or use a separately proved limb comparison; it cannot extrapolate the k<=13 argument to arbitrary k. Every root needs its range lookup, including the high bits of its box, and every true row needs a zero-check of its own relation. Proving only the total sum over rows permits an over-cap row to cancel an under-cap row. Padded output rows use a public selector ψ and the relation `ψ*(sum_i w_i^2-B^2)+sum_j s_j^2=0`, with explicit zero pins for all four auxiliaries; padded input weights must satisfy the publisher's zero-domain premise or an independent domain proof. Do not require the nonzero slackB on a row whose committed padding is prescribed to be zero.

### RFC-0048 trust boundary and a concrete cancellation attack

[RFC-0048](../../docs/rfcs/RFC-0048-weight-trust.md) separates arbitrary-commitment uniqueness from integer-model fidelity/totality. Its publisher-validated `R_W` supplies weights in `[-127,127]`, valid channel classes/multipliers and zero padding. Conditional on that premise, the whole-sum argument above is an integer norm proof over the bound weights. The certificate does not itself prove those domains.

For an untrusted arbitrary-field commitment, take two row entries `w_0=1`, `w_1=2^48` in Goldilocks. Since `(2^48)^2=-1 mod P`, their squares sum to0. Set `s_0=B` and the other auxiliaries0: every auxiliary range check and the field relation passes, while the centered integer interpretation of the second weight is far outside INT8 and violates the proposed small row norm (E0 algebraic counterexample). An integer-norm guarantee independent of the RFC-0048 domain premise therefore needs independent domain binding/range proofs for the committed weights and padding. The same issue applies to classes and affine multipliers where their domains enter a value/range argument. It is not repaired by an additional range check only on the slack.

This attack does not refute RFC-0048's uniqueness guarantee: a malformed root can name a different or partial field-valued function. It refutes using this bare sum-of-squares relation as an unconditional integer-weight-domain proof. A future narrower-table design must retain the arbitrary-commitment uniqueness argument separately. Any new table's packed-key box must remain injective below P; widening a table alone does not establish the integer value half for malformed classes or keys.

### Public policy, binding, zero knowledge and cost obligations

The cap must be a public policy fixed independently of secret weights/inputs, for example an architecture/tensor-type policy that the quantizer must meet or reject. Publishing the smallest successful k, a per-row measured norm, an adaptive fallback or a secret-dependent subset discloses a weight predicate; public shift classes do not establish that such additional disclosure is free. A fixed cap still defines a public acceptance predicate, and any public success/failure or fallback policy must be part of the stated leakage model. SP-7 requires the verifier to derive the permitted cap, fan-in, root widths, integer bounds and table choice from authenticated public policy; a prover cannot merely assert a narrower range.

A policy ID/version, cap assignment, tensor/type/layer/row coverage, true/padded dimensions and domain-assurance mode must be bound to the claim list and `R_W`/epoch. The four auxiliary columns need a defined commitment descriptor and openings linked to those same weight rows; the current `ChannelParams` schema and ten-root weight descriptor do not silently contain them. A separate epoch certificate root is a possible design option, not a specification decision made here. Its commitment and all claims must be fixed before their zero-check/batching/opening challenges. Every row must be covered, including all layers and affected types. A cached verification result must key on the exact weight root, epoch, claim-list digest, cap-policy/domain-assurance mode, proof parameters and verifier/certificate version; it is not transferable to another root, cap or epoch.

The four-square decompositions are intentionally nonunique auxiliary witnesses. They do not enter the computation of `D` or select an output, but they are still secret-class data that must be hidden. A future certificate needs masked zero-checks, hiding commitments/openings, a simulator and a disclosure policy; it cannot expose row norms, slack roots or unmasked evaluations. The extra relation/lookups, sumcheck degrees, padding selectors, LogUp populations, proof mode/grinding and openings need a fresh RFC-0014 budget. An epoch proof consumes its own RFC-0012/0013 mask/blinding/opened-column budget; reusing it with fresh challenges is not free, and the present epoch cache is not a new universally reusable ZK proof. Protocol/2 and its fresh budget obligations cannot be inferred from a protocol/1 arithmetic demonstration.

For Q/K only, shapes give 2,193,408 true rows and 8,773,632 auxiliary scalars plus as many range lookups. W_up alone for IDEA-0031 gives 6,709,248 rows and 26,836,992 scalars/lookups. The combined scope is 8,902,656 rows and 35,610,624 scalars/lookups (E0 shape counts, before padding, masks, blinding and row-relation work). Q/K's square sums read 35,936,796,672 weight coefficients; W_up's read 109,924,319,232. Thus “four scalars per row” is not a complete epoch cost, and the historical 7.9e7-scalar all-tensor estimate is not the Q/K-only count. No certificate cost is credited as zero or already amortized in the current model.

### Current norm-output bound and both consumers

On a real row let `z_i=rho*res_i`, `u=sum res_i^2+e`, `e>=1`, and use the exact root bracket `rho^2*u<=d*2^(2κ)`. Then `sum z_i^2<=d*2^(2κ)`; on a padded row the rho zero pin makes z and the norm output zero. Round-half-up by shift `r_n`, then clamp to INT8: componentwise rounding error is at most 1/2, and clamping to `[-128,127]` does not increase magnitude. Consequently

```text
||y||_2 <= sqrt(d) * (2^(κ-r_n) + 1/2),
|w dot y| <= 2^k * sqrt(d) * (2^(κ-r_n) + 1/2).
```

At current 405B, RFC-0023/fixture values κ30, `r_n=27`, `d=16384` give `||y||_2<=1088`, hence the conservative integer accumulator bound `A_k=1088*2^k` (E0). The analogous 8B bound uses `sqrt(4096)=64`, giving 544. The general derivation must use integer/rational square-root ceilings where dimensions are not perfect squares; no floating-point estimate may set a table width. Clamp can only improve this bound, but no further improvement is credited here.

For this route, retain the current compiler's conservative `Rotate` transfer `abs(key)<=A_k*2^(bits+1)`. Its signed width covers the inclusive symmetric interval `[-K,K]`, so it is `b=1+bitlen(K)` for K>0. There is one extra bit even if a power-of-two endpoint is attained. The current powers give:

| k / maximum row RMS at 405B | A_k | RoPE bits 9: signed b | RoPE bits 10: signed b | IDEA-0031 key `a*u`: signed b |
|---|---|---|---|---|
| 10 / 8 | 1,114,112 | 32 | 33 | 29 |
| 11 / 16 | 2,228,224 | 33 | 34 | 30 |
| 12 / 32 | 4,456,448 | 34 | 35 | 31 |
| 13 / 64 | 8,912,896 | 35 | 36 | 32 |

The final column uses `a in[-128,127]`, `u in[-A_k,A_k]`, hence the inclusive symmetric product bound `128*A_k`. A fused single table under current `B_max=32` needs `b+ceil(log2 n)<=32`: k13 permits one class, k12 two, k11 four in these conservative power-of-two examples. This is an E0 sufficient range condition for IDEA-0031, not evidence its new `D`, quantizer or serving kernel passes fidelity. The old strict `abs(u)<2^23` target was stronger than necessary for a signed 32-bit `a*u` key. Using the precise product endpoints is what decides the gadget.

For RoPE, the same class-bit rule applies: with k11 and bits 9 the conservative signed width is 33, so one/two/four classes need 33/34/35 table bits; k12 and bits 9 need 34/35/36. The old claim that the route inherently requires a single class is not current. A stronger cap k10 gives b32 at bits 9, fitting one class under the existing 32-bit cap, but its real-model feasibility is unknown. Coefficient correlations in the public sine/cosine generator could tighten the generic Rotate interval; that is a separate prospective refinement and receives no extra width or removal credit in this note.

RoPE precision also changes its scale. If a fixture changes bits 14 to 9 while preserving its intended physical output scale, the old RoPE shifts 22/23 become 17/18, subject to the quantizer's public scale policy; leaving them unchanged is a different `D`. Multiple classes are allowed by current lowering. Pair/head alignment may be a fidelity policy, but neither rotation nor the schema proves “one class only.” Affine multipliers can widen these keys again and are excluded from the displayed pot examples.

## Per-table break-even and exact sharing

Let `S_old,S_new` be the complete sets of distinct table definitions, keyed by kind and all parameters, not by site. Charge padded tree rows. Define `Added=S_new-S_old`, `Removed=S_old-S_new`; a table is removed only when no site still uses it. The correct incremental time test is

```text
sum_Added T_j*p_row(j) - sum_Removed T_j*p_row(j)
  + certificate_amortization + extra_pass/reset/communication/opening_cost
  < N_split*s_saved.
```

Here `p_row(j)` includes the table-side commitment, fraction-tree work and counter effects actually measured for that size/layout; `s_saved` is the incremental saved split commitment+lookup cost, with whichever glue/opening terms the comparison consistently includes. In a uniform-price simplification with positive `DeltaM`, `p_row < N_split*s_saved/DeltaM`. The current planning model uses `1.15*(c_c+c_l)` for its table term; it already charges `M` once per proof, so do not add amortized rows to per-token C/K a second time. Its omitted or bounded P4/P5/streaming work and unpriced certificate/scheduling costs remain limitations.

With zero removal credit, the normalized maximum table prices `p_row/s_saved` for this 405B batch are 8.3671875,4.18359375,2.091796875,1.0458984375,0.52294921875 for total table bits 32,33,34,35,36 respectively (E0 exact ratios). These are total `b+class_bits`, not magnitude bits. They are conditional break-even ratios, not benchmark prices. Separate Q/K shifts/caps, per-layer/head variants or affine sites can create multiple new definitions; then sum all added rows and assign only the lookups each actually removes. The canonical union and actual use count, not the largest single table, decides the trade.

The current saved baseline has the old `2^24` high table used only by all 252 Q/K sites, but its `Range{21}` is also used by the first norm. A complete uniform replacement could remove that high definition if no other candidate site needs it, while Range21 remains. This makes an eligible conditional credit of `2^24`, not `2^32`; no complete candidate manifest has been derived here, so the default ratio above books zero removal credit. If those exact assumptions hold, a 2^34 replacement has ratio 2.0938416422 instead of 2.091796875. The old blanket `2^32` credit and the assumed loss of a shared Range table are invalid for today's baseline.

A future `p_row` calibration must pre-register first. EXP-0006's fraction proxy can help separate table-sized arithmetic from lookups, but it does not measure a full table-side LogUp proof or a GPU/HBM price. Test relevant size/layout/cache regimes, include reset/count/encode/tree and any fixed scheduling cost, retain raw samples/CVs and avoid extrapolating CPU p_row to H100. No such calibration is run in this merge unit.

## Eight-byte counters, peak memory and public layout

The shared RoPE instance has `N_split=35,936,796,672` real lookups; all keys/classes may concentrate in one admissible row. Padding its lookup forest gives `2^36=68,719,476,736` total counted leaves, hence that generic public upper bound on an individual multiplicity (E0). This is not an attainability claim for table row 0 under the refined model: padding goes there, while a zero activation can concentrate its real lookups at another table row. The real population alone already exceeds u32. A u32 counter is insufficient even before padding; use exact u64 counts or an explicitly proved/checked equivalent. The public bound must remain below P and be derived from actual batch/shape/padding/table-use totals. Counter overflow cannot be excused as a performance artifact.

At 8 bytes per padded row, a 2^34 table needs 128 GiB across the node, 16 GiB per GPU if perfectly sharded over 8 GPUs. In a uniform-u64 design, current baseline M counters use 57.3208008 GiB across the node; retaining them with the new table means 185.3208008 GiB total, 23.1651001 GiB per GPU, or 24,873,336,832 bytes. This already exceeds the decimal 24 GB working-set budget before other scratch, communication, witness and tree state. Even the conditional 2^24 high-table credit leaves 24,856,559,616 bytes per GPU (E0). This is not relying on u64 for every small/unused table: allowing implicit-zero counters for the publicly unused Rope table and u32 for the first norm’s 2^20-row table reduces the baseline counter bytes to 61,476,438,016 across the node. The remaining active table populations all permit concentration beyond u32 in the displayed 16K batch. Adding the 2^34 table still costs 24,864,423,936 bytes per GPU, or 24,847,646,720 with the conditional 2^24 high-table credit (E0 from the current report’s table-use counts). These remain above 24 GB before other scratch; alternative compression/scheduling would require its own exact bounds and public memory plan. The old 36 GiB/GPU co-residency estimate assumed ten full 2^32 tables and is superseded by current balanced/shared counts; its conclusion that the original P1 schedule is not yet feasible remains.

For total table bits 32/33/34/35/36, eight-byte counter arrays alone need 32/64/128/256/512 GiB across the node, or 4/8/16/32/64 GiB per GPU under ideal eight-way sharding. These figures exclude padded lookup/table-tree states, masks, column-hash state and network buffers. A table-by-table P1 schedule can reduce counter residency toward the maximum individual table rather than the sum, but requires a concrete pass/regeneration/caching/communication plan. It does not create free HBM or preserve the old pass count automatically. The counter reset and worst-case memory-access/timing envelope are part of that plan; a sparse/adaptive allocation would disclose secret-dependent information.

[RFC-0047](../../docs/rfcs/RFC-0047-advice-order.md) fixes the canonical padded-axis layout, claim-coordinate permutation, row limit and `n=prompt_len+output_len`. A future sequential-counter implementation must emit the same canonical tensor/row/segment order and bind new auxiliary descriptors; it cannot change the committed layout to match whichever table is counted first. Layout at max_context must fit 2^32 advice rows, column-answer limits and segment rules. Table width above 32 is also a distinct tier-A change: current TableSpec/Range/Requant generators, table row indices, limits, codec, prover and independent verifier all need consistent definitions and conformance vectors. A per-use limit must key by authenticated use/definition, not an unauthenticated string “rope” or a global change to every Requant table.

## Current security and feasibility gate

| Property | Current assessment |
|---|---|
| Integer norm implication | Conditional E0 argument above, including the whole-sum bound; not valid for arbitrary-field weights without independent domain binding |
| Arbitrary-root uniqueness and value semantics | Must be re-established for the complete modified range/table/packing protocol; RFC-0048 does not turn a slack relation into a domain proof |
| SP-7/public range derivation | Public authenticated policy and independent verifier derivation required; current schema has no such certificate/cap map |
| Zero knowledge and disclosure | `unknown`: fixed-cap policy, auxiliary hiding, simulator, epoch budgets and timing envelope still need a specification decision/proof |
| Soundness/Fiat–Shamir | `unknown` for a complete certificate/widened-table protocol: commit-before-challenge, full row coverage, packed-box injectivity, padded N/T and fresh RFC-0014 aggregation/grinding bounds required |
| Fidelity/serving | `unknown`: constrained real weights, changed RoPE constants/scales/classes and bit-exact serving must pass the current fidelity gates; local error or synthetic rows cannot settle it |
| Memory/cost | Original co-resident 2^34-counter schedule fails the stated 24 GB scratch budget; sequential scheduling and measured p_row are unresolved |
| Verifier independence | Specification-first independent implementation required; no prover code is imported into the verifier by this note |

This is a shelved conditional route, not a rejected algebraic norm bound and not a validated full security claim. No net Δμ is newly assigned. The earlier current-formula rescore assumed away these certificate/policy/scheduling costs and the current width/class changes; it remains a historical sensitivity scenario, not a promotion input. The numerical fidelity gates remain those in RFC-0005 and the real-weight work tracked by #93/#170/#183, as IDEA-0031's current shelving decision records.

## Reopening conditions

Reopen only with an explicit cap/domain/disclosure policy and proposed certificate/commitment layout; exact public range/class/shift derivation and full shared-table manifest; measured or conservatively bounded memory/pass/reset cost; a fresh soundness/ZK accounting; and real-weight fidelity plus bit-exact serving evidence. The E0 certificate argument is available to a later pre-registered test, but this unit launches none. IDEA-0028 has no separate queue item after absorption, and IDEA-0031 keeps its distinct E2 gate.

## Historical proposal and reviews

The text below is the prior2026-10-17 proposal plus its conditional2026-10-19 rescore. Its narrower-class assumption, old splits/table credits, four-square term bound, blanket ZK approval and norm-removal premise are superseded above. These dated reviews are retained as evidence of what was considered, not as approval of this revision.

## Mechanism

Setting ([RFC-0005 'Integer semantics'](../../docs/rfcs/RFC-0005-integer-model.md#integer-semantics), [RFC-0006 'Range analysis'](../../docs/rfcs/RFC-0006-compiler-ir.md#range-analysis)): `acc = W_q y` and `acc = W_k y` are INT32 accumulators bounded by interval arithmetic at `< 2^29` (the public bound `127 · fan_in · max|x|` with `fan_in = d = 16,384`); RoPE rotates pairs `(a, b) ↦ (a·C − b·S, a·S + b·C)` with 14-bit constants from the `Rope` table, and the `Rotate` transfer gives `|a·C − b·S| < 2 · 2^28 · 2^14 = 2^43`, a signed key width of 44 bits; `q = rq(·, r_q, −128, 127)` then selects `RequantSplit` (`B_max = 32 < 44 ≤ 62`), 2 commitments and 2 lookups per element (RFC-0006 'Gadget selection', [RFC-0009 `RequantSplit`](../../docs/rfcs/RFC-0009-gadgets.md#requantsplit)). The idea replaces the split by a single lookup through three ingredients, none sufficient alone (E0, primary: SUR-0009 Synthesis Question 1 and the arithmetic below, re-derived 2026-10-17 in this note's scratchpad script):

1. **Two certificates tighten the accumulator to `|acc| < 2^23`.** (a) A per-tensor weight-row norm: `tessera-quant` enforces `‖w_row‖₂ ≤ 2^k` for every output row of `W_q` and `W_k`, with `k` a public per-tensor constant in the claim list (SUR-0009 seed "Weight-row norm certificate"; [CPPU24], [CFGZS25] for the accumulator-aware constraint, [LNP22] for the certificate form). The certificate is a four-squares relation per row at the weight commitment, `2^{2k} − Σ_i w_i² = a² + b² + c² + d²` with four committed scalars range-checked once, proven once per epoch with the weights and cached by verifiers: `7.9×10^7` committed scalars over the `1.99×10^7` rows of a 405B model, 0.02% of one 16K-token batch's `4.6×10^11` (E0, primary: SUR-0009 Question 2). (b) A relational fact from `RsqrtRow` ([RFC-0009](../../docs/rfcs/RFC-0009-gadgets.md#rsqrtrow)): `Σ_i (ρ·res_i)² ≤ d · 2^{2κ}` gives `‖y‖₂ ≤ 2^{κ−r_n} √d + √d/2` for the norm output at zero proof cost (SUR-0009 seed "`‖y‖₂` relational fact", IDEA-0011's sibling). Cauchy–Schwarz in the `MatMulWeight` transfer: `|acc| ≤ ‖w_row‖₂ ‖y‖₂ ≤ 2^k · (2^{κ−r_n} √d + √d/2)`, which is 23.0 bits at row RMS 16 and 24.0 bits at row RMS 32 for `d = 16,384` (E0, primary: SUR-0009 [E0-this] (d)). The route below needs the 23-bit case, i.e. `W_q`, `W_k` rows at RMS ≤ 16 after SmoothQuant migration and QK equalization (RFC-0005 'Migrate').
2. **9-bit RoPE constants.** The `Rope` table of [RFC-0008 'Table kinds'](../../docs/rfcs/RFC-0008-logup-and-tables.md#table-kinds) emits `(C, S) = (round(2^bits cos θ), round(2^bits sin θ))` at `bits = 9` instead of 14, a public `QuantSpec` constant; the generator is the definition (RFC-0008 'Generators and digests'), so `D` changes and the fidelity gates of RFC-0005 decide. The `Rotate` transfer then gives `|a·C − b·S| ≤ (|a| + |b|) · 2^9 < 2 · 2^23 · 2^9 = 2^33`, signed width `b = 34`. The neighbouring cells do not reach 34: 10-bit constants with `|acc| < 2^23`, or 9-bit with `|acc| < 2^24`, both give `2^34` and width 35; 10-bit constants need `|acc| < 2^22` (row RMS 8). The title's "9–10-bit" is therefore 9 bits at RMS 16, or 10 bits at RMS 8 (E0, own arithmetic).
3. **Per-table limit.** `TableLimits.max_log2_rows` becomes a map by table kind: `Requant(rope) → 34`, everything else 32. RFC-0006 'Gadget selection' compares `b` against the table's own limit, so the RoPE rescale selects `Requant1` (1 commitment, 1 lookup) while every other table keeps OQ-0006-1's default. The verifier re-derives `b = 34` from public `k`, `κ`, `r_n`, `d` and `bits` (INV-IR-5, SP-7).
4. **Single class.** RFC-0005 writes the RoPE rescale as `rq(·, r_q, −128, 127)` with one shift per tensor and no channel index (against `r_v,c` for V), and a rotation pair `(a, b)` combines two output channels that must share one shift, so the RoPE table is `Requant { b: 34, classes: 1, r_by_class: [r_q], lo: −128, hi: 127 }` with `2^34` rows and a plain key `z + 2^33` (no `pack(c, ·)`, no hidden class). `W_q` and `W_k` therefore get a per-tensor power-of-two scale. With 4 classes the table would be `2^36` rows, `6.9×10^10` leaves per batch, more than the `3.6×10^10` the split costs; with 2 classes `2^35 = 3.4×10^10`, a wash (E0, own arithmetic): the route exists only single-class. A per-head class with the pair-aligned shift folded into the table definition per head multiplies table instances and is rejected by the same arithmetic.
5. **Memory.** `2^34 × 4 B = 64 GiB` of multiplicity counters for the one table, sharded 8 GiB per GPU over the 8-GPU unit; 08 §4's per-table line ("`≤ 2^32` rows per table × 4 bytes, sharded across GPUs") is amended for this one kind, and the 8 GiB come out of the `≤ 24 GB` working set per H100 (08 §4).
6. **Verifier.** `tv-tables` evaluates the RoPE `Requant` MLE by the structured rule: the core is the 256 INT8 outputs, cost `O(n · core)` independent of `b` ([RFC-0008 'Structured evaluation'](../../docs/rfcs/RFC-0008-logup-and-tables.md#structured-evaluation)); it re-derives the range bound and the limit from public data and checks the weight-row certificates with the weight opening (RFC-0012).

Together with IDEA-0011 (norms) and SUR-0009's fused-rescale seed, this completes a split-free `servable` profile at 260,096 rescaled elements per layer per token (E0, primary: SUR-0009 [E0-this] (b)).

## Why it could work

- **The count is real and named.** RoPE Q+K are 17,408 of the 82,944 RISK-2 excess per layer per token in `minimal` and of the 50,176 split-added elements in `servable` (310,272 − 260,096); `17,408 × 126 × 16,384 = 3.59×10^10` lookups and as many commitments per 16K-token batch (E0, primary: 00 §9, RFC-0005, SUR-0009 [E0-this] (a) and (b)).
- **Width arithmetic: no single ingredient suffices.** Worst case `2^44` (28 + 14 + 1 + sign); certificates alone reach 39–40 bits of signed width with 14-bit constants, 34 with 9-bit constants at `|acc| < 2^23` (E0, primary: SUR-0009 Synthesis Question 1 and this note's check). The pre-RoPE INT16 requantization alternative (key `2 · 2^15 · 2^14 = 2^30`) costs one extra rescale per element, 2 + 2 like the split (E0, primary: SUR-0009 Question 1).
- **Table side priced.** A `2^34`-row table costs `2^34` committed multiplicities and `2^34` fraction-tree leaves per proof ([PH23], E0, primary; RFC-0008 'Relation', 'Fraction tree'; [IDEA-0023](IDEA-0023-dilation-shared-requant-tables.md)'s pricing at `c_c + c_ℓ` per row, upper end, 0.6× at `leaf_ps`): `1.05×10^6` per token amortized over 16,384 tokens against `2.19×10^6` saved per token, net positive only single-class (E0, primary).
- **Fidelity of 9-bit constants, the unmeasured input.** A 9-bit cosine or sine has absolute error `≤ 2^-10` per constant, so a rotated pair's error is `≤ (|a| + |b|) · 2^-10 ≤ 2^-9.5 ‖(a, b)‖₂`, about 0.14% of the pair norm; the INT8 step that follows is `1/127 ≈ 0.79%` of full scale, so at full scale the perturbation is 0.18 LSB and flips the rounding of roughly that fraction of elements by one LSB (E0, own argument: uniform rounding offsets; for 14-bit constants the figure is 0.005 LSB). Whether that moves perplexity is measured nowhere: SUR-0009 found no source on quantized RoPE constant width (one search; FPGA quarter-wave tables without error analysis). Llama 3's choice to keep attention unquantized in FP8 and to cap FP8 scales for the 405B outliers ([Grattafiori24], E3 primary for the throughput figure) is the warning that Q/K are the fidelity-sensitive tensors.
- **Key width is the lever the separation principle leaves open** ([Jo26], E0, primary): per-lookup cost cannot fall through table structure, but the key width sets which gadget the compiler selects and, in LogUp-GKR, the once-per-proof table side (SUR-0002 Question 3).
- **Accumulator-aware quantization has a track record.** A2Q+ and AXE hold a 16-bit accumulator at W4A4 on Llama3 1B–8B with a 0.3–0.6 perplexity cost ([CPPU24], [CFGZS25], E1 primary on their GPUs); Tessera asks for 3–5 bits off a 28-bit worst case at W8A8, a far weaker constraint (E0, SUR-0009 Question 1).

## Expected gain

The cost model has no table-side term, so the scenario adds the amortized RoPE table side to both counts, pricing a table leaf at `c_c + c_ℓ` per row (upper end). Baseline: the `servable` profile with the norm and RoPE splits, `310,272 × 126 = 3.909×10^7` commitments and lookups per token (E0, primary: SUR-0009 [E0-this] (b); the generation pass quoted `4.116×10^7`, which matches nothing in the corpus and is replaced). Gross: `−17,408 × 126 = 3.690×10^7`. Net: `+ 2^34 / 16,384 = 3.795×10^7`. Run 2026-10-17 (E0, primary):

```sh
python3 -I research/tools/cost_model.py --set commits_per_token=3.909e7 --set lookups_per_token=3.909e7   # servable with splits (baseline of this note)
python3 -I research/tools/cost_model.py --set commits_per_token=3.690e7 --set lookups_per_token=3.690e7   # gross: RoPE rescale as Requant1
python3 -I research/tools/cost_model.py --set commits_per_token=3.795e7 --set lookups_per_token=3.795e7   # net of the 2^34 table side at c_c + c_l per row
```

```
baseline: profile 'corrected'
   S   linear  attention  weights  regen  streaming    total
   1K     3.25       1.13    0.021    4.0       0.42      8.8
   4K     3.20       4.46    0.021    4.0       2.32     14.0
  16K     3.02      16.82    0.020    4.0      11.24     35.1
 128K     1.96      87.48    0.013    4.0      77.93    171.4

scenario: commits_per_token=3.909e7, lookups_per_token=3.909e7          (servable with splits)
   S   linear  attention  weights  regen  streaming    total
   1K     4.58       1.13    0.021    4.0       0.42     10.2   Δ  +15.1%
   4K     4.51       4.46    0.021    4.0       2.32     15.3   Δ   +9.3%
  16K     4.25      16.82    0.020    4.0      11.24     36.3   Δ   +3.5%
 128K     2.76      87.48    0.013    4.0      77.93    172.2   Δ   +0.5%

scenario: commits_per_token=3.690e7, lookups_per_token=3.690e7          (gross)
   S   linear  attention  weights  regen  streaming    total
   1K     4.32       1.13    0.021    4.0       0.42      9.9   Δ  +12.2%
   4K     4.26       4.46    0.021    4.0       2.32     15.1   Δ   +7.5%
  16K     4.01      16.82    0.020    4.0      11.24     36.1   Δ   +2.8%
 128K     2.61      87.48    0.013    4.0      77.93    172.0   Δ   +0.4%

scenario: commits_per_token=3.795e7, lookups_per_token=3.795e7          (net of the table side)
   S   linear  attention  weights  regen  streaming    total
   1K     4.45       1.13    0.021    4.0       0.42     10.0   Δ  +13.6%
   4K     4.38       4.46    0.021    4.0       2.32     15.2   Δ   +8.4%
  16K     4.13      16.82    0.020    4.0      11.24     36.2   Δ   +3.2%
 128K     2.68      87.48    0.013    4.0      77.93    172.1   Δ   +0.4%
```

The printed `Δ` is against the model's `minimal`-count baseline; the idea's Δμ is scenario-against-servable-baseline from the `--json` totals (E0, primary):

| Context | Δμ, gross (`corrected` / `counted`) | Δμ, net of the `2^34` table side (`corrected` / `counted`) | Δ proof size | Δ verification |
|---|---|---|---|---|
| 1K | −2.5% (10.15 → 9.90) / −3.4% (20.90 → 20.19) | −1.3% (10.15 → 10.02) / −1.8% (20.90 → 20.53) | opened advice columns: −3.6×10^10 committed elements per 16K-token batch (the split's second commitment per element) and +1.7×10^10 multiplicities, net −1.9×10^10, about −0.7% of a 16K/16K proof unit's committed elements (`4.6×10^11` linear + `2.2×10^12` attention, E0); the weight opening gains `7.9×10^7` certificate scalars once per epoch (E0) | one `t̃(ρ)` evaluation of a 256-value core replaces the split's two tables (a `2^32` `Requant` and a `Range(12)`), `O(n · core)` each (RFC-0008 'Structured evaluation'); the dense-core count stays within `≤ 20` (08 §4); plus one four-squares check per weight row at the epoch's weight opening, cached (E0) |
| 4K | −1.7% / −2.1% | −0.9% / −1.1% | as above | as above |
| 16K | −0.7% / −0.8% | −0.3% / −0.4% | as above | as above |
| 128K | −0.1% / −0.1% | −0.05% / −0.06% | as above | as above |

On the `minimal` profile the gross saving is SUR-0009's 187,392 → 169,984 per layer per token after IDEA-0011, −3.1 points of `μ(1K)` corrected (E0, primary: SUR-0009 seed "Per-table `B_max`"). Pricing the table leaf at `leaf_ps` instead of `c_c + c_ℓ` moves the net toward the gross (the table side shrinks by 0.4×). A short-context lever: its value is completing the split-free `servable` profile together with IDEA-0011 and the fused-rescale seed (50,176 split-added elements removed per layer per token), not its own Δμ at 16K.

Evidence levels of the inputs: unit costs E0 primary (the model's assumptions, 08 §2, `counted` from EXP-0002/EXP-0006; RFC-0020 replaces them); counts E0 primary (RFC-0005's tensor list); the `2^23` accumulator bound E0 primary conditional on row RMS ≤ 16 after migration, which is **unmeasured** (SUR-0009: no source measures 405B weight-row norms after SmoothQuant); the table-side price per row E0 from the model's unit costs (42–70 ps); the fidelity of 9-bit constants **unmeasured anywhere** (claim level until the experiment).

## Security gate

| Property | Effect | Pass |
|---|---|---|
| Soundness bits (G-4, RFC-0014) | `Requant1` Lemma 1 holds unchanged (the table is a function of its key). The key is a true integer by INV-IR-2 only under the certified bound `\|acc\| < 2^23` and the public constant width, both re-derived by the verifier from public data (SP-7, INV-IR-5): `\|z\| < 2^33 ≪ 2^62`, so Lemma 0 applies and a wrapped key is impossible. The weight-row certificate is a checked relation at the weight opening: a malicious prover whose rows exceed `2^k` fails that check and never reaches the lookup with an out-of-range key; the four committed squares are range-checked once so the relation holds over the integers (sum `< 2^{2k+1} < 2^58 < P`, as SUR-0009 Question 2 argues for the norm form). The LogUp error per instance grows with `T`: `N = 2^35.1` RoPE lookups per 16K batch plus `T = 2^34` gives `N + T = 2^35.6 ≤ 2^38` (RFC-0014's budget line), error `2^-92.4` in `Fp2`, inside RISK-5's about `2^-90`, and the union bound over instances gains one term; RFC-0014's table must restate it. No challenge, label, absorption or opening order changes (INV-FS-1..4 untouched). | pass (E0) |
| Uniqueness upstream of the logits (SP-3, RFC-0009 Lemma 0) | Each `q`, `k` element is the unique table value of a determined key (Lemma 1 on `z + 2^33`); the split's `z_lo`, `z_hi` pair disappears, removing a committed value per element rather than adding one. The certificate adds four committed scalars per weight row, each pinned by a range lookup and the degree-2 relation (a second decomposition of the same difference into four squares passes too, but the squares are not upstream of the logits and the relation only certifies the bound, so no second witness for any logit-bearing value exists). | pass (E0) |
| Zero knowledge (G-5, RFC-0013) | The public `k` per tensor and `bits` are design constants the honest prover always meets (the quantizer enforced the norm), not functions of the input or of the actual norm, so the claim list carries one more public number per tensor and the RFC-0013 simulator is unchanged; the certificate's scalars are committed with the weights under the RFC-0012 hiding budget. The multiplicity column of a `2^34` table is committed advice under RFC-0011 masks as today. The timing side channel of secret-indexed multiplicity counting (issue #257, [EXP-0010](../experiments/EXP-0010-multiplicity-timing-leak/README.md): a live verifier learns at most 7.2 / 3.9 / 2.5 bits per proof at a 0.1 / 1 / 3% jitter floor) scales with the counter footprint: the RoPE table quadruples one table's footprint, so EXP-0010's envelope must be re-stated with it; a constant-duration P1 envelope closes it as before. | pass (E0), with the EXP-0010 envelope to re-derive |
| Verifier independence (SP-6) | The `Rope` generator at 9 bits, the `Requant { b: 34, classes: 1 }` definition and the per-kind limit are specification data (`spec/vectors/tables/`, `TableLimits` in the parameter set), reimplemented in `tv-tables` from the text; `tv-*` re-derives `k`, `κ`, `r_n`, `d`, `bits` → `b = 34` and trusts nothing prover-declared. | pass |
| Post-quantum (hash-based commitments) | Unchanged: the certificate is a sumcheck relation over hash-committed data. | pass |

A `fail` anywhere rejects the idea. No row is `unknown`: the open inputs (row RMS after migration, the fidelity of 9-bit constants) are completeness and cost inputs, not security ones; the zero-knowledge row carries one re-derivation (the EXP-0010 envelope at a 4× footprint).

## Prior rejection check

Checked against [baseline.md §7](../baseline.md#7-already-rejected) and the RFCs' Alternatives Considered on 2026-10-17:

- [RFC-0008 Alternatives Considered](../../docs/rfcs/RFC-0008-logup-and-tables.md#alternatives-considered) 1–6 (one-hot lookups, univariate lookups, base-field challenges, separate trees, shipping table contents, floating-point generators): none concerns table width or RoPE constant width.
- [RFC-0009 Alternatives Considered](../../docs/rfcs/RFC-0009-gadgets.md#alternatives-considered): softmax brackets, one-hot maximum, rsqrt by table, Newton iteration; no entry on RoPE or on `RequantSplit` against a wider table.
- [RFC-0006 Alternatives Considered](../../docs/rfcs/RFC-0006-compiler-ir.md#alternatives-considered) 4 (calibration ranges) rejects data-dependent bounds the verifier cannot re-derive; the certified norm is public, input-independent and re-derived (SP-7), the position [CFGZS25] also takes. RFC-0006 Alternative 1 (search over requantization placements) would move the rescale before RoPE, which SUR-0009 prices at the split's cost.
- [RFC-0005 Alternatives Considered](../../docs/rfcs/RFC-0005-integer-model.md#alternatives-considered) 3 (affine scales everywhere) and 6 (percentile clipping) do not touch RoPE constant width; RFC-0005 fixes 14 bits in its summary without an Alternatives entry.
- OQ-0006-1 is open (`B_max` 30/32/34 as a global default, "resolve at the v0.2 gate"); this idea answers a narrower question (one table at 34, the rest at 32) with the per-kind map.
- [SUR-0002](../surveys/SUR-0002-lookups.md)'s "wider `B_max`" seed priced ten `2^32` tables widened together at 37% of linear lookups in table side (negative); widening one table is a different arithmetic (SUR-0009 Synthesis Question 1).
- [IDEA-0007](IDEA-0007-dedup-table-side-wide-requant.md) (`rejected`, G-5 fail) attacked the same keys from the table side with a data-dependent sparse table; this idea keeps the full input-independent enumeration, so that rejection does not transfer.
- [IDEA-0023](IDEA-0023-dilation-shared-requant-tables.md) prices the table side and shares tables by dilation; a single-class RoPE table at `b = 34` has no slack to share and is one instance either way.
- baseline §7 has no row on `B_max`, RoPE or accumulator certificates; baseline §6 lists "range analysis, rescale placement, wider accumulators, quantization co-design" as the RISK-2 lever without a rejection.

New: the single-class condition and its break-even (`2^34` wins, `2^35` is a wash, `2^36` loses), the explicit `TableLimits` map, the `N + T` and counter-memory accounting at `T = 2^34`, the RMS ≤ 16 condition that the width arithmetic imposes on `W_q`/`W_k`, and the synthetic constant-width test. "None" after the check.

## Kill criterion

Any one of: (a) `W_q` or `W_k` needs more than one shift class to pass the RFC-0005 gates (the table becomes `classes · 2^34` rows and the table side exceeds the saving at 4 classes, breaks even at 2); (b) the certified bound cannot reach `|acc| < 2^23` for Q and K (row RMS above 16 after migration and QK equalization), leaving `b ≥ 35` even with 9-bit constants, where `2^35` table leaves (`3.4×10^10`) match the `3.6×10^10` saved and 128 GiB of counters exceed the working set; (c) 9-bit RoPE constants move the 70B/405B perplexity ratio or top-1 agreement past the RFC-0005 gates (unmeasured anywhere); (d) 08 §4's per-table counter budget (`≤ 2^32 × 4 B`) is firm or the `≤ 24 GB` working set cannot absorb 8 GiB per GPU of counters (RISK-10 worsens); (e) the v0.1 unit costs put a table-side leaf at or above `c_ℓ` with the gross gain under 1.5% of `μ(1K)`, so the net gain falls under 1%, below the cost model's input uncertainty.

## Cheapest experiment

Hypothesis: (i) with seeded Gaussian `W_q`, `W_k` rows at Llama-like INT8 statistics (`d = 16,384`, `d_head = 128`, 128 query heads and 8 KV heads) and norm outputs `y` bounded by the `RsqrtRow` fact, exact integer RoPE at 9-bit constants followed by INT8 requantization changes at most a fraction of `q`, `k` elements by one LSB against the 14-bit reference that is below the INT8 requantization's own rounding noise, measured as the deviation of the scores `Σ_j q_j k_ij` and of the softmax weights across positions up to 16K (pre-registered criterion: the 9-bit-induced score deviation is smaller than the deviation between the 14-bit integer pipeline and its real-valued reference); (ii) the width arithmetic (interval `2^44`, certified `2^33` at RMS 16, `2^34` at RMS 32) and the table-side break-even at `classes ∈ {1, 2, 4}` hold as stated.

Method, in order:
1. E0 (`s`, half a day): the arithmetic of (ii) as a script with exact Python integers, promoted from this note's scratchpad; the cost-model scenarios above re-run with IDEA-0023's table-side parameter once it exists.
2. E1 (`s`, two days, this CPU): the synthetic pipeline of (i), measuring the fraction of elements flipped by one LSB at 14-, 10- and 9-bit constants, the score and softmax-weight deviations at positions up to 16K (RoPE angles grow with position, so the constant error is position-dependent), and the per-row RMS distribution that the `2^23` bound requires; the honest-prover completeness check (no key outside `[−2^33, 2^33)` across 10^6 sampled rows at RMS ≤ 16) and the adversarial check over `F65537` at toy widths (`b ≤ 12`: a key outside the table is rejected; a row violating the norm fails the four-squares relation).
3. E2 needed, not here: the perplexity and top-1 gates on real 70B/405B weights (RFC-0005 fidelity harness), and the measured row RMS of `W_q`, `W_k` after SmoothQuant migration; both decide kill criteria (b) and (c).

Effort: `s` for the E1 experiment; the design change is `l` by the rubric's letter (a tier-A amendment to RFC-0005, RFC-0006 and RFC-0008 plus a new weight-row gadget in RFC-0009 and its `tv-*` reimplementation), and it depends on IDEA-0011's sibling fact and the weight-norm certificate, which are shared with SUR-0009's fused-rescale seed. What stays unknown until the RFC-0020 gate (E2): the table-side cost per row, between the model's 42 and 70 ps, which sets the net gain between the gross and the net rows above.

## Refutations

Sweep triage of 2026-10-17: four refuter agents who did not write the idea, one per lens; each argument is condensed from that refuter's verdict, with the cost-model reruns the scoring agent made to check disputed cells noted as such.

| Lens | Verdict | Argument |
|---|---|---|
| soundness | survives | The soundness argument is IDEA-0028's with the constant width fixed: `Requant1` Lemma 1 (RFC-0009:40), a key < 2·2^23·2^9 = 2^33 (strictly, since \|acc\| < 2^23; checked) inside Lemma 0's margin, the bound re-derived by the verifier from public data (SP-7, INV-IR-2 at RFC-0006:114), the ‖y‖₂ fact exact via RsqrtRow Lemma 7, and N + T = 2^35.6 ≤ 2^38 on the RFC-0014:99 LogUp line; α, β still follow `round1` (RFC-0014:78-80). The 9-bit `Rope` table is a public generator in the claim list (RFC-0008 'Generators'), so the verifier regenerates it and trusts nothing prover-declared; its fidelity is a completeness and policy input, not a soundness one. The note's ZK row is accurate: the set of shift classes per tensor is already public (RFC-0007:78) while the assignment is hidden (RFC-0007:79), so one public k per tensor and a single-class W_q / W_k leak nothing new. Two inherited caveats: a 4-byte counter can wrap at N > 2^32 lookups into one row (completeness; IDEA-0028 mechanism 3 says so, this note does not), and the EXP-0010 timing envelope must be re-derived at the 4× counter footprint, as the note already requires. |
| prior art | survives | Nothing rejects it: OQ-0006-1 (RFC-0006:147, SPEC.md:105) keeps B_max 30 / 32 / 34 open; RFC-0005:10 and :69 fix '14-bit constants' in the summary and the semantics table with no Alternatives entry on constant width (RFC-0005:137-144 verified); RFC-0006 / RFC-0008 / RFC-0009 Alternatives contain nothing on table width, RoPE or accumulator certificates, and baseline §7 has no such row. The route is SUR-0009's three seeds taken together (SUR-0009:61-63) and its Synthesis Question 1 (SUR-0009:51), which the note cites as origin; a survey seed is not a prior rejection, and the note adds the single-class condition and break-even (2^34 wins, 2^35 a wash, 2^36 loses), the TableLimits map and the N + T / counter accounting. IDEA-0007's rejection (data-dependent sparse table, G-5 fail; LEDGER.md:108) does not transfer to an input-independent 2^34 enumeration, and SUR-0002's negative 'wider B_max for all tables' seed is a different arithmetic. Two consolidation notes: IDEA-0028 of this sweep is the same route and should be merged into this one; and the note's baseline 310,272 per layer per token, with its dismissal of 4.116e7 as 'matches nothing in the corpus' (IDEA-0032:35), omits the `o8` split that RFC-0005:73 (`o < 2^39`) and RFC-0006:79-80 force, which IDEA-0033 counts; the gross RoPE saving is unaffected. |
| cost model | weakened | Rerun confirms the net gain against the `servable`-with-splits baseline: −1.3% / −1.8% at 1K (`corrected` 10.15 → 10.02, `counted` 20.90 → 20.53) and −0.3% / −0.4% at 16K; the gross saving is only −2.5% / −3.4% at 1K and the 2^34 table side at `c_c + c_ℓ` per row takes half of it back. The note's own kill criterion (e) names 1% as the cost model's input uncertainty, and the 16K figure is below it while the 1K figure sits within a factor of two of it on a per-row price spanning 42-195 ps; pricing the leaf at `leaf_ps` moves the net toward the gross but the direction of the actual table-side cost (non-unit numerators from the first layer, 8-byte counters forced by N > 2^32, 64-128 GiB of counter memory against a 24 GB working set per H100, 08 §4:59) is the other way. The baseline 310,272 omits the `o8` split that IDEA-0033 finds (326,656 per layer per token), which shrinks the percentages further, and the 2^34 pricing does not credit the removed 2^32 reduced table and Range(12) table (IDEA-0028 does, with (2^34 − 2^32) rows). It is the same idea as IDEA-0028 (ledger item 26's seed) with a slightly different baseline and pricing convention; the honest reading of both is that the single RoPE-table widening is worth under 3% of μ(1K) and nothing at 16K until two certificate ideas, a fidelity measurement and a calibrated p_row exist, so one merged note with IDEA-0028's calibration experiment is what should survive. |
| feasibility | weakened | Same dependency chain as IDEA-0028 and IDEA-0031 (weight-row norm certificate, ‖y‖₂ fact, quantizer at one shift class), and the same blocked decisive input: the synthetic E1 test measures LSB flips and score deviations, not the perplexity and top-1 gates that kill criterion (c) names, which need `tessera-quant` (open issue #67) and real weights (research rule 6). The memory accounting is inconsistent with the sibling note: this instance receives 3.59×10^10 lookups per 16K batch, above 2^32, so 4-byte counters can overflow on an honest concentrated key distribution (IDEA-0028 mechanism 3 says so), making the footprint 128 GiB, 16 GiB per GPU, not the 64 GiB / 8 GiB stated here; with P1 counting every table in one pass (docs/rfcs/RFC-0018-gpu-prover-runtime.md:80) and ten 2^32 tables already near 20 GiB per GPU, the ≤ 24 GB working set (08 §4) is exceeded by E0 arithmetic. Table definitions, `TableLimits` and `QuantSpec` are tier A, so promotion is `l` for −1.3% to −1.8% of μ(1K) net. This note and IDEA-0028 are the same route; merge them and keep IDEA-0028's `p_row` experiment plus this note's width arithmetic. |

## Score

| Gain | Confidence | Cost | Priority |
|---|---|---|---|
| net −1.8% μ(1K) under `counted` (20.90 → 20.53) and −1.3% under `corrected` (10.15 → 10.02) against the `servable`-with-splits baseline, −0.4% / −0.3% at 16K; gross −3.4% / −2.5% at 1K with the 2^34 table side at `c_c + c_ℓ` per row taking half back; E0. The percentages shrink further against IDEA-0033's 326,656 (the note's 310,272 omits the `o8` split), the 2^34 pricing does not credit the removed 2^32 reduced table and Range(12) table (IDEA-0028 does), and the table-side price spans 42-195 ps per row | 0.3: Lemma 1 and the width arithmetic (key < 2^33 at RMS ≤ 16 with 9-bit constants, N + T = 2^35.6 ≤ 2^38) are exact, but the route rests on two unwritten certificates, an unmeasured fidelity gate (c) that needs real weights, a single shift class on `W_q` / `W_k`, and a claim-level `p_row`; kill (d) fires by E0 arithmetic (N = 3.59×10^10 > 2^32 forces 8-byte counters, 128 GiB in all, 16 GiB per GPU beside about 20 GiB of ten 2^32 tables under P1's one-pass counting, RFC-0018:80, against ≤ 24 GB) unless counting is re-sequenced per table | `l`: table definitions, `TableLimits` and `QuantSpec` are tier A; the E1 synthetic constant-width test is `s` (two CPU days here) but does not reach kill (c) | 1.8 × 0.3 / 4 = 0.1 (0.2 with IDEA-0028's credit for the removed tables) |

## Decision and links

Triage 2026-10-17: `triaged` at priority 0.1. No hard gate (OQ-0006-1 keeps B_max open; IDEA-0007's G-5 rejection does not transfer to an input-independent enumeration). Primary of the merge with IDEA-0028 (same route, same seed SUR-0009:61-63): this note keeps the full route, the single-class condition and the width arithmetic; from IDEA-0028 come the per-table break-even rule, the removed-table credit, the 8-byte-counter observation and the E1 `p_row` calibration at T ≈ N, the one experiment worth running now. The merged note must correct the memory accounting (128 GiB, 16 GiB per GPU) and price the per-table re-sequencing of P1 that kill (d) forces, replace 310,272 by IDEA-0033's 326,656, and wait for the certificate note it shares with IDEA-0031 and for a fidelity measurement (E2) before any RFC. Not queued on its own.

Experiments: none yet (the E1 synthetic constant-width test when queued). RFC or issue: none yet (an amendment to RFC-0005, RFC-0006 and RFC-0008 if validated). Ledger line updated on: pending consolidation (2026-10-17).


## Re-score 2026-10-19 (E0, conditional)

The historical tables above use their pinned model. This calculation uses the current formula, with P5 kept separate from commit_ps; [baseline §2](../baseline.md#2-the-cost-model) states its omissions. Count-only reductions are 17408 per layer per token, times 126. Certificate overhead and fidelity remain unresolved; this re-score does not validate the mechanism.

| Baseline | Δμ(1K) | Δμ(4K) | Δμ(16K) | Δμ(128K) |
|---|---|---|---|---|
| historical 326656-current-formula | -2.23% | -1.48% | -0.62% | -0.09% |
| current-c2 | -2.09% | -1.42% | -0.61% | -0.09% |
| current-c2 with IDEA-0028 hypothetical 2^32 removed-table credit | -2.35% | -1.60% | -0.69% | -0.10% |

The new 2^34-row table is charged through table_rows, once per proof. Do not add its amortized rows to both per-token counts as well. The removed-table credit depends on the actual balanced split tables and their other users; 2^32 is a sensitivity scenario, not an established saving. The norm certificates,9-bit RoPE fidelity, class restriction and counter-memory problem are unchanged.

Exact inputs and outputs: [main-sync-2026-10-19.json](../experiments/EXP-0011-servable-count/results/main-sync-2026-10-19.json).
