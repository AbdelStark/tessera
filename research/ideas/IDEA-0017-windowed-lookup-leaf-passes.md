# IDEA-0017: Windowed lookup leaf passes — current accounting and feasibility

| | |
|---|---|
| Status | `shelved` (2026-10-19: operational pass/memory schedule and full-leaf completeness remain unresolved under #329; hosted fraction proxy inconclusive) |
| Origin | SUR-0007/SUR-0003 streamed-round windows; current RFC-0018/0020/0032/0045 and the completed fraction amendment of EXP-0012 |
| Lever | Potential incremental regeneration beyond the current priced baseline, if a concrete schedule demonstrates it; no new count or cost-model option is introduced |
| Spec surfaces | A byte-identical prover algorithm may leave protocol messages unchanged, but full tree/leaf/mask and pass accounting must be demonstrated before that claim; no production change here |
| Priority | 0 active while shelved; historical conditional 0.41 retained below, with no current validated Δμ |

## Current decision and evidence boundary

This bounded unit is a feasibility/accounting clarification, not an experiment. The current normative model already prices regeneration and attention leaf streaming; it does not price the whole leaf stage. The missing work is exactly [#329](https://github.com/AbdelStark/tessera/issues/329), checked open on 2026-10-19, last updated 2026-10-17. Its large whole-stage figures are first-order issue analysis, not a newly validated operational schedule or current model output.

The completed [EXP-0012 fraction case](../experiments/EXP-0012-integer-leaf-round-window/README.md#amendment-2026-10-19-hosted-fraction-window-case) is `inconclusive`: resident v4 costs 2.50–2.54 times split, missing the registered<=1.5 criterion, while v2 costs 1.54–1.55 times split and does not satisfy the registered factor-v kill criterion. The 2^24/v5 arm also retains CV 5.42%. This does not refute every streamed window, but establishes no reduction in regeneration, full LogUp time or GPU cost. The parent integer-product-forest criterion is independently refuted and cannot stand in for this fraction case.

The archived proposal below contains obsolete n_regen=4 counts, a single aggregate Requant instance, universal six-byte leaves, a u64 grid and an assumed one-input-read window. They are superseded by this note. No `lookup_stream` option, μ rescore, new experiment/harness or protocol-security endorsement is added. Reopening requires the operational schedule and completeness conditions below, then a new preregistration for any test.

## What the current model already prices

[RFC-0018 pass structure](../../docs/rfcs/RFC-0018-gpu-prover-runtime.md#pass-structure) names P1 witness/commit, P2 leaf work, P3 stages, P4 opening and P5 column answers. It explicitly uses `n_regen=5` forward-equivalent estimates for P1–P5, with P2 streaming, P4 combination, P5 evaluation and the weight fold in separate terms. P3 may fuse into P2/P4 when its point is already known; P2's first streamed round may fuse with bottom-up construction. This is a coarse planning decomposition, not a manifest that assigns every instruction to a distinct full forward.

[RFC-0020 executable model](../../docs/rfcs/RFC-0020-cost-model.md#executable-cost-model), the [normative Rust model](../../crates/tessera-bench/src/model.rs) and [research mirror](../tools/cost_model.py) currently use:

```text
attention_leaves = 8064 * context * 16384
c(S) = max(0, ceil(log2(attention_leaves*32 / 200000000000)))
streaming = c(S) * streaming_pass_ps * 8064 * context / native_time
```

At contexts 1K/4K/16K/128K, this yields 5/7/9/12 streamed rounds (E0 exact model arithmetic), with streaming_pass_ps=7 in the current corrected/counted research scenarios. The six-to-eight ps estimate already includes score regeneration plus folding for those attention passes. Charging the same work again as extra regen would double count it. The linear lookup term already pays an in-memory fraction proof, including bottom-up build; the table term pays one table-side tree and multiplicity commitment per row, once per proof. A window's replacement arithmetic cannot be added again as entirely new streaming work without subtracting the replaced work consistently.

Additional levels, linear-layer key regeneration, joint batching/memory constraints and their scheduling overhead remain outside the attention-only streaming line, as RFC-0018/0020 now explicitly state and #329 tracks. Their existence is not proof of a specific number of extra full forwards. A proposed correction must distinguish the base P2 regeneration already represented in five, incremental passes after fusion, rereads of cached leaves, GEMMs regenerated from checkpoints, and table-side generation that needs no model GEMM. Blindly changing regen=5 to5+c or reusing the old 7→5 scenario does not establish any of those distinctions. P4/P5's own costs and their present unpriced/bounded terms are unrelated savings unless the new schedule actually removes them; a pre-query window cannot remove P5.

## Exact conditional capacity table at current counts

Use the [current 405B servable c2/B32 report](../../docs/perf/v0.2/commitment-report-405b-servable-c2-b32.json), not historical RISK-2 totals. For one lookup table definition with real population `n_j = lookups_per_token_j*T`, its lookup tree has `N_j=2^ceil(log2 n_j)` leaves. Distinct Requant definitions have distinct trees; shared definitions combine all of their actual users. The report omits the row gadgets' range-table allocations (#118/#122), so it is not a complete leaf-stage manifest.

Define the capacity arithmetic `c_b(N)=max(0,ceil(log2(b*N/M)))`, with `M=200000000000` bytes. The table below is exact E0 arithmetic for a SINGLE instance given all of M, excluding other trees, masks, eq/grid/output buffers and implementation temporaries. Entries under c16/c32 are triples for token populations T=1024/4096/16384. They are capacity-derived round counts, NOT a proved whole-stage regeneration schedule.

| Current table IDs / use class | Lookups per token | log2 padded N at 1K/4K/16K tokens | c16 | c32 |
|---|---|---|---|---|
| 4, first norm high table | 16,384 | 24/26/28 | 0/0/0 | 0/0/0 |
| 7/8/9/14, each distinct definition | 2,064,384 | 31/33/35 | 0/0/2 | 0/1/3 |
| 5, RoPE Q/K shared high table | 2,193,408 | 32/34/36 | 0/1/3 | 0/2/4 |
| 3, shared Range21 | 2,209,792 | 32/34/36 | 0/1/3 | 0/2/4 |
| 10/11/12, each distinct definition | 4,128,768 | 32/34/36 | 0/1/3 | 0/2/4 |
| 2/13, each distinct definition | 6,709,248 | 33/35/37 | 0/2/4 | 1/3/5 |
| 6, shared projection high table | 6,838,272 | 33/35/37 | 0/2/4 | 1/3/5 |

These stable IDs refer only to the linked pinned report. For example, Q/K at 16K tokens has 35,936,796,672 real lookups, hence 2^36 padded leaves; its c32 is 4, not the old aggregate-Requant 5. Silu has 109,924,319,232 real lookups, padded to 2^37; its c32 is 5 and c16 is 4. No single “Requant instance” contains all the old 83,968 sites. Different instances sharing the batched challenge do not thereby share their leaf buffers, key-producing GEMMs or cached folded state.

At 32 bytes uniformly, a tree with c32=c has conditional oversize-level sum `c+(c-1)+...+1=c(c+1)/2`. For the 16K groups with c32=3/4/5, those sums are 6/10/15. If ONLY the bottom level uses 16 bytes and every level above uses 32, the sums instead are `c16+c32*(c32-1)/2`, namely 5/9/14. It is incorrect to replace every internal level by the bottom leaf's storage width. These sums are round opportunities under the specified per-level memory model, not distinct full-model passes: shared regeneration, retention, recomputation and per-level construction must still be scheduled. Summing all per-table triangles and pricing each as a full forward double counts shared layer work; giving every simultaneously live table all of M understates peak memory.

Each current table-side instance has at most 2^32 rows, so its unmasked 32-byte array alone fits M; all current M=7,693,467,648 table rows at 32 bytes total 246,190,964,736 bytes, which does not. That aggregate includes the structurally unused Rope table counted by the report; excluding it still does not fit. A table tree regenerates public row tuples and reads multiplicities, not the model's full GEMMs. Its residency, batching and mask state remain part of #329, not another automatic forward-equivalent. Shrinking token units does not shrink every table definition and adds commitments/openings; the old unit-count/proof-size estimates are not current planner results.

### Fp/Fp2 and compact-leaf qualifications

- A small product leaf can initially be one Fp value, 8 bytes. Once challenges bind it, the folded values generally live in Fp2, 16 bytes. For an initially oversized 8-byte leaf array, fitting the folded state after j>=1 rounds requires `16*N/2^j<=M`, rather than naively halving the original 8-byte footprint forever. The integer-product path of EXP-0012 is only an initial small-leaf window.
- An unmasked lookup fraction leaf is `(1,d)`, `d=alpha-key-beta*value in Fp2`: its implicit numerator needs no array, while denominators need16 bytes/leaf. It is not four small Fp tables. A table-side leaf has base-field numerator `-m` (8 bytes if stored canonically) and an Fp2 denominator (16), hence 24 initial bytes; folded nonconstant numerators generally becomeFp2, giving 32 bytes/remaining leaf. Internal fraction nodes have both components in Fp2. The generic RFC-0018 rule uses 32 bytes; smaller specialized representations need their own transition and peak-buffer plan.
- The old compact `u32 key + u16 value =6B` is neither universal nor implemented by this note. A current Exp table with sigma=16 has value 2^16 at key 0, which needs 17 unsigned bits; generic base-field values/multiplicities are wider still. Range tables have no value column, INT8 outputs can use signed byte encodings, and alignment/metadata/padding add costs. Small compressed raw leaves becoming full-field folded arrays can increase bytes per surviving entry. No six-byte table is supplied as an operational schedule or used to price a pass reduction.
For the displayed 16K groups with N=2^35/2^36/2^37, the raw eight-byte capacity arithmetic alone gives 1/2/3, while the ordinary once-bound Fp2 state requires 2/3/4 rounds to fit when the initial array is oversized. Lookup denominators already use the latter 16-byte model. This is why a raw Fp leaf or compact key/value count cannot be substituted for a persistent folded-state schedule. Unmasked product nodes remain in Fp before binding, but above the small-leaf level their canonical values are full-width field elements, not the 22-bit integers of EXP-0012.

- Current protocol/2 adds neutral padding trees and masked extensions; public constants may be computed in closed form, while mask state is additional. Logical padding coordinates and their sumcheck rounds remain even when no constant array is allocated. Neither doubling every tree's memory blindly nor deleting its contributions is correct.

## Grid bounds and leaf completeness

For a width v ternary grid on `{0,1,2}^v`, binary multilinear extrapolation can amplify an input bound K to `K*3^v`, with signed values. A product of two such inputs has magnitude at most `K^2*9^v`. If a canonical eq-weight component w is multiplied before reduction, a sufficient batch-cadence B check is

```text
B*(P-1)*K^2*9^v < 2^127       for signed i128,
B*(P-1)*K^2*9^v < 2^128       for unsigned magnitude storage.
```

The sign still needs exact handling, and every other summed term needs its own bound. “Use u128” alone does not prove arbitrary accumulation safe. EXP-0012's KEY_MAX=2^22-1 product experiment checks its exact signed bound at B=32 for v1/v3/v4 and B=8 for v5; that is not a bound for 32-bit lookup keys. With K=2^32-1, the simple weighted K^2 term already exceeds 2^127 at v0/B1; for v>=1 it exceeds 2^128 as well. There is no positive signed-i128 cadence for this unreduced formulation under that worst-case bound. One unsigned single product may fit at v0, but adding products or extrapolation terms needs reduction or wider limbs.

At an unmasked lookup leaf, write `d=alpha-k-beta*e`. In `d_L*d_R`, the integer coefficient bounds are 1 for alpha², `2*K*3^v` for alpha, `2*E*3^v` for alpha*beta, `K^2*9^v` for the constant, `2*K*E*9^v` for beta, and `E^2*9^v` for beta². These six signed coefficient streams must be handled before combining with full-field alpha/beta/eq/lambda factors. The actual current value bound E and all sum lengths must be public. Table-side cross terms add multiplicities: a numerator bound M0 produces terms as large as `2*M0*K*9^v` or `2*M0*E*9^v` before the full-field eq weight; a historical 48-bit multiplicity allowance cannot inherit the small-product grid bound. Limb decomposition, an earlier modular reduction or a full Fp2 grid can make an algorithm feasible, but no such general integer kernel or cost is established here. EXP-0012's fraction arm deliberately uses a full Fp2 grid instead.

The old triage's exceptional `(0,delta)` leaf, with full-field delta, cannot be cast to the small-integer path. Any legacy/custom representation containing it must retain its block on a general Fp2 path or prove a separate symbolic expansion including delta; replacing it with a small denominator or dropping it changes the polynomial. This is a completeness requirement even when its numerator is zero: it still multiplies denominators and scales other fractions. The archived mechanism has not implemented that fallback.

For CURRENT protocol/2, RFC-0013 explicitly retires the old masking pair; [RFC-0032](../../docs/rfcs/RFC-0032-zero-knowledge-tree-levels.md#forests-of-two-trees-at-least) uses a separate public neutral padding tree with leaves `(0,1)`. Its [leaf formulas](../../docs/rfcs/RFC-0032-zero-knowledge-tree-levels.md#leaf-checks) use lookup numerator 1-pt, masked key/value/multiplicity extensions, public padding coordinates and neutral denominators. Their off-cube values and high-degree mask corrections are not all small integers, even when Boolean real leaves are. A full implementation must include them, the minimum-height/padding rules, table-side numerators, non-power-of-two inputs, front padding, final leaf checks and degenerate challenges while preserving messages/masks. EXP-0012's three-instance unit-numerator `prove_batch` proxy supplies none of those full `prove_trees`/LogUp/ZK coverage claims. Historical “same simulator” or “all four tables base-field” statements below are not current approval.

## Missing operational schedule and reopening conditions

A 3^v grid can compute v sequential messages without sending any future challenge early; this alone does not reduce regeneration to ceil(c/v). The completed proxy reads original input once to form the grid and AGAIN after all v challenges to bind it. If leaves are not retained, that second read is another regeneration. If retained, their storage/promotion and lifetime must fit. If the once-bound state remains too large, the next window needs an explicit materialization/host-stream/recompute plan. Rebuilding internal nodes from original leaves after prior challenges costs construction and binding work, not just another cheap ternary grid. Any build/first-round/P3 fusion must trace when root claims, level points, gamma/lambda and masks are actually known.

At v4 the proxy therefore offers only a counterfactual 4/2 original-input read ratio, not a measured 4x or 2x regeneration speedup; its resident per-round baseline reads shrinking folded arrays, not four full regenerated witnesses. The v4 wall-time penalty and missing table/mask paths also remain. #329 may choose different groups/challenges/units, which can change openings, proof size and the protocol rather than merely its prover arithmetic. That decision is outside this note.

Reopen with: a complete table/forest/leaf-layout manifest under RFC-0045, including row-gadget tables and protocol mode; per-level working sets and peak lifetimes; a per-tile/window scan list distinguishing regeneration from rereads and construction; legal challenge/claim/mask timing and final binding; explicit GEMM dependency sharing/fusion; and an incremental cost comparison against the CURRENT five-regeneration plus priced attention-streaming baseline. Only then pre-register a full-fraction/current-protocol test and any E2 pass timing. Until then, no speculative cost-model option or added μ saving is justified.

## Historical proposal and reviews

The original2026-10-16 proposal, scores and later hosted-case summary follow as provenance. The current clarification above supersedes its n_regen4/aggregate-instance/six-byte/u64/pass-factor claims. Historical external benchmarks and `pass` security rows are not new evidence or review of a complete current streaming implementation.

## Mechanism

Today's reading of [RFC-0018 'Leaf levels'](../../docs/rfcs/RFC-0018-gpu-prover-runtime.md#pass-structure): a tree level whose working set (two `Fp2` tables over its nodes, 32 B per node) exceeds the memory budget `M` is proven by streaming, one regeneration pass per early round, until the folded state fits. The text, RISK-10 and the cost model count those passes for the attention trees only (`stream(S) = c(S) · p_s · A · S / native(S)`, `p_s = 7` ps per score, "cheap on tensor cores"). The P2 row of the pass table says the same passes run over "affected linear tiles" and regenerate "lookup keys and values", but no term prices them. This idea does two things: it prices them (step 1) and it cuts them by a factor `v` without touching the protocol (steps 2–4).

1. **Accounting.** For each table instance ([RFC-0008 'Batching'](../../docs/rfcs/RFC-0008-logup-and-tables.md#batching): one combined tree per table definition across every layer, head, token and channel) compute the leaf-level working set of a proof unit and the streamed-round count under RFC-0018's rule, in two layouts: the specified `Fp2` node pairs (32 B per leaf) and base-field leaves as [IDEA-0005](IDEA-0005-base-field-leaf-small-value.md) keeps them (`key` as `u32`, `value` as `u16`, 6 B per leaf; the denominator `α − key − β·value` is formed in registers when a round needs it). Attribute to each streamed round the regeneration it needs, read off [RFC-0005 'Integer semantics'](../../docs/rfcs/RFC-0005-integer-model.md#integer-semantics): the `Requant` keys of the Q, K, V, O-residual and down-residual rescales are GEMM accumulators, and the O and down accumulators depend on the whole layer before them (attention output `o`, SiLU and `up`), so regenerating the `Requant` instance's leaves is one full layer forward per layer; the norm keys `res_i · ρ` are elementwise from the checkpointed residual; the `Silu` keys `g = W_gate y` need the gate GEMM alone (27.4% of a layer's GEMM multiply-accumulates, E0, own arithmetic below). Sum the forward-equivalent fraction per streamed round; the table is in 'Why it could work'.
2. **Windows.** Keep RFC-0010's messages, labels and verifier. For the sumcheck over a streamed level (degree 3: `eq` times the two products of the fraction relation `λ(n_L d_R + n_R d_L) + d_L d_R`, [RFC-0008 'Fraction tree'](../../docs/rfcs/RFC-0008-logup-and-tables.md#fraction-tree)), one regeneration pass over the level's tiles accumulates, per aligned block of `2^v` consecutive pairs (pair bit lowest, [RFC-0010 'Multilinear extensions'](../../docs/rfcs/RFC-0010-proof-schedule.md#multilinear-extensions) and 'Tree levels'), a `3^v`-point grid of the degree-2 inner polynomial's partial sums for the three products, weighted by the suffix `eq` weights of the block; the `eq` factor's own variables leave the grid by decomposability ([DDBDT26] Lemma 7). The `v` round messages are inner products of the grid with the Lagrange weights of the challenges as they arrive: message `j` is formed only after `r_{j−1}` is squeezed, exactly as in the per-round prover, each multiplied by Gruen's linear factor of the current `eq` variable ([Gru24] §3.2, the form RFC-0010 'Sumcheck messages' already sends: `d` values, `p_j(1)` derived from the claim). On the leaf level the four tables are base-field: the denominators are integer-coefficient linear forms in `{1, α, β}`, so the grid entries are integer polynomials in `(α, β)` of degree ≤ 2 whose coefficients are accumulated as `u64` with one Goldilocks reduction per block ([DDBDT26] §3; IDEA-0005's monomial expansion), then combined in `Fp2` once per block.
3. **Fold once per window.** After the window, the regenerated tile is folded by the `v` challenges once (the only `Fp2` pass over the tile) and the next window starts from the folded state; `c` streamed rounds become `⌈c / v⌉` regeneration passes, `v = 4–6`. Every level whose working set exceeds `M` is treated this way, from the level below the first in-memory level down to the leaves; an oversize internal level's nodes are rebuilt from the regenerated leaves inside the same pass (RFC-0018 'Kernels': "bottom-up build fused with leaf generation").
4. **Expose the count.** The run manifest (RFC-0018 'Pass structure') records the pass count per level and per table instance, so that v0.3's pass counter checks the E0 table of this note and v0.5 measures the cost per pass (RISK-9 and RISK-10 resolutions).

## Why it could work

- **Own E0, the pass table** (primary; arithmetic from RFC-0005's per-layer counts × 126 layers × tokens in the unit against `M = 0.2` TB of free HBM, RFC-0018's rule; scratchpad script `passes.py`, reproduced by hand): lookups per layer per token are 137,216 in the `minimal` set (RISK-2's breakdown of the source count) and 220,160 in the RISK-2 upper bound; the `Requant` definition's instance holds 83,968 of them (Q 16,384, K 1,024, V 1,024, O-residual 16,384, down-residual 16,384, two norms 16,384 each) and the `Silu` instance 53,248. `c_leaf` is the leaf-level count under RFC-0018's formula; `Σ` is the total over every oversize level of the tree (the level `k` above the leaves holds `N / 2^k` nodes at 32 B, so it streams `c_32 − k` rounds), which is the count RFC-0018's text implies and the cost model's leaf-only `c(S)` omits.

| Tokens in the unit | Instance | Leaves `N` | `c_leaf` at 32 B / 6 B | `Σ` over levels at 32 B / 6-B leaves |
|---|---|---|---|---|
| 16K | `minimal` set | 2.83×10^11 | 6 / 4 | 21 / 19 |
| 16K | RISK-2 set | 4.54×10^11 | 7 / 4 | 28 / 25 |
| 16K | `Requant` instance | 1.73×10^11 | 5 / 3 | 15 / 13 |
| 16K | `Silu` instance | 1.10×10^11 | 5 / 2 | 15 / 12 |
| 4K | `minimal` set | 7.08×10^10 | 4 / 2 | 10 / 8 |
| 4K | `Requant` instance | 4.33×10^10 | 3 / 1 | 6 / 4 |
| 1K | `minimal` set | 1.77×10^10 | 2 / 0 | 3 / 1 |
| 1K | `Requant` instance | 1.08×10^10 | 1 / 0 | 1 / 0 |

  All E0, primary. The instances of one unit are batched per level (RFC-0008 'Batching'), so one streamed round of the batch regenerates every instance's leaves; the `Requant` instance alone needs the whole layer forward, so each streamed round costs about one forward-equivalent of GEMM work (0.92 of `native(16K)`, 0.995 of `native(1K)`: the linear share of the cost model's `native(S)`, E0). The unpriced term is therefore 4–7 forward-equivalents at 16K under the leaf-only reading (6 B / 32 B layouts) and 19–28 under the every-level reading, against the cost model's `n_regen = 4` for everything else (RFC-0018 P2 row and RISK-9, primary; the attribution to GEMMs is this note's derivation, E0).
- **Proof units cannot absorb it.** For the `minimal` set's leaf level to fit in 0.2 TB at 32 B, a unit may hold 361 tokens (1,928 at 6 B): 46 (9) units for a 16K-token batch, each with its own advice commitment and up to 17 combined rows of 2^22 × 16 B (RFC-0011 'Sizes', primary), that is up to +51 GB (+9 GB) of proof against the 4 GiB gate of SC-3 (E0, own arithmetic). So the planner streams these levels or the proof-size budget breaks; kill criterion (a) is unlikely on the numbers, and the term is real.
- **[DDBDT26]** (E1, primary; Apple M4 Max, BN254, Jolt; full text read in SUR-0007): round batching gives 4.10–6.01× on Spartan's outer sumcheck in steady state, 10.9× end to end and 57.9× when the baseline is memory-bound (121 s → 2.1 s), peak RSS 50.1 → 11.5 GB; the streaming variant runs 1.62–2.70× slower than the linear-space prover (Table 10); Lemma 6 gives `Θ((d log d) M (k + log log M))` time at `O(M^{1/k})` space. The field is 254-bit, so only the pass structure and the ratio to the linear-space prover transfer.
- **[BCFFMMZ25]** (E1, primary; Xeon 8259CL, 128-bit field, `n = 30`): `k = 2` passes at 1.8–2× the linear-space time and 0.008× its memory (503 s / 219 MiB against 271 s / 27.3 GiB); the pass count `log_η(n/k) + d k`. Same transfer caveat.
- **SUR-0007 own count** (E0, primary): a `v = 4` window costs 15 `Fp` multiplications per pair, `v = 5` 23, `v = 6` 34, against about 33–38 for the per-round prover over the same rounds, plus `2 · 1.5^v` `Fp2` additions per pair and about 4 multiplications per pair to bind a tile to the previous window's challenges. The window is not more multiplications; what it adds is additions.
- **[DMABKGR25]** (E0 modelling, primary; 255-bit field) and **[MDG+26]** (E0 modelling, primary): a streamed round is paid in bytes, "two full passes of reading and writing `N` elements" per round; for Tessera's lookup levels the bytes are the GEMM regeneration, so the passes, not the multiplications, are the cost.
- **[EXP-0006]** (E1, primary; M4 Max, same harness): the leaf-level sumcheck is 82.4 ns × 2^21 / 2^22 = 41.2 of the fraction proxy's 107.3 ns per lookup, 38%, the level where the base-field grid acts (`c_ℓ` lever).
- **Production shape.** [Stwo26] and [SP1GPU26] (E0, primary; code read in SUR-0002) keep the first layer base-field (`LogUpSingles`, `felt_t` numerators) and fuse two rounds per kernel; neither streams or batches rounds, so steps 2–3 are not in production code.

## Expected gain

The cost model has no term for the linear-layer streamed rounds, so the scenario prices them through `regen_full_equivalents` (one forward-equivalent per streamed round of the batched leaf level, E0 above). Reference: today's P2 lookup passes under the leaf-only reading at the base-field layout, `c_leaf = 3` for the `Requant` instance (`regen_full_equivalents = 4 + 3 = 7`). Candidate: one `v = 4` window pass plus the residual rebinding and fold, about one forward-equivalent (`regen_full_equivalents = 5`). Both E0, primary (own derivation); the pass cost per round is the cost model's one-forward convention for P1 and P4 (RFC-0018, primary).

```sh
python3 -I research/tools/cost_model.py --set regen_full_equivalents=7   # reference: today's P2 lookup passes priced
python3 -I research/tools/cost_model.py --set regen_full_equivalents=5   # candidate: one v = 4 window plus residual
```

```
baseline: profile 'corrected'
   S   linear  attention  weights  regen  streaming    total
   1K     3.30       1.13    0.021    4.0       0.42      8.9
   4K     3.25       4.46    0.021    4.0       2.32     14.1
  16K     3.06      16.82    0.020    4.0      11.24     35.1
 128K     1.99      87.48    0.013    4.0      77.93    171.4

scenario: regen_full_equivalents=7
   S   linear  attention  weights  regen  streaming    total
   1K     3.30       1.13    0.021    7.0       0.42     11.9   Δ  +33.8%
   4K     3.25       4.46    0.021    7.0       2.32     17.1   Δ  +21.3%
  16K     3.06      16.82    0.020    7.0      11.24     38.1   Δ   +8.5%
 128K     1.99      87.48    0.013    7.0      77.93    174.4   Δ   +1.8%

scenario: regen_full_equivalents=5
   S   linear  attention  weights  regen  streaming    total
   1K     3.30       1.13    0.021    5.0       0.42      9.9   Δ  +11.3%
   4K     3.25       4.46    0.021    5.0       2.32     15.1   Δ   +7.1%
  16K     3.06      16.82    0.020    5.0      11.24     36.1   Δ   +2.8%
 128K     1.99      87.48    0.013    5.0      77.93    172.4   Δ   +0.6%
```

The gain is the candidate against the reference, both above today's baseline because the baseline omits the term. `regen` is an additive forward-equivalent count and the model's unit holds `stream_tokens = 16384` tokens at every context (sets of short requests, RFC-0018 'Proof units'), so the saving is −2.0 on `μ` at every context and its percentage is largest where `μ` is smallest:

| Context | Δμ | Δ proof size | Δ verification |
|---|---|---|---|
| 1K | 11.9 → 9.9, −16.8% (E0); 0 if a unit holds one 1K request, where the base-field leaf level fits (`c = 0`) and the term is instead +16 units × 17 combined rows of proof | 0 (E0: same messages, same openings) | 0 (E0: same checks) |
| 4K | 17.1 → 15.1, −11.7% (E0) | 0 | 0 |
| 16K | 38.1 → 36.1, −5.2% (E0) | 0 | 0 |
| 128K | 174.4 → 172.4, −1.1% (E0; the unit still holds 16K tokens, so the lookup term is unchanged in absolute terms) | 0 | 0 |

Under the `counted` profile (`c_c` 94, `c_ℓ` 101, `c_π` 73 ps, E0 fitted to E1): 81.8 → 79.8 at 16K, −2.4%; 20.3 → 18.3 at 1K, −9.9% (`--profile counted --set regen_full_equivalents=7`, then `=5`). At the specified 32 B layout the leaf-only stake is 5–7 forward-equivalents, so the reference is `regen_full_equivalents = 9–11` and a `v = 4–6` window saves 3–5: −8% to −12% of `μ(16K)` corrected (E0, same scenario shape; `regen_full_equivalents=10` gives 41.1 at 16K). Under the every-level reading of RFC-0018 (`Σ = 13–15` rounds for the `Requant` instance at 16K) the stake is a `μ(16K)` of about 48–50 and the window's saving 9–11 forward-equivalents, about −20% (E0, own arithmetic, outside what the model's single `regen` scalar expresses without the `lookup_stream` line proposed in 'Cheapest experiment'). Alternatively the planner avoids the passes by shrinking proof units, in which case the gain is proof size (each avoided unit saves up to 17 combined rows, 1.14 GB at `k = 2^22`, RFC-0011 'Sizes') rather than Δμ. Every input is E0 and conditional on RFC-0018's planner streaming these levels; the pass count is settled by v0.3's pass counter (E1 count) and v0.5's measurement (E2–E3).

## Security gate

| Property | Effect | Pass |
|---|---|---|
| Soundness bits (G-4, RFC-0014) | Prover-only. Every message is the polynomial RFC-0010 'Sumcheck messages' specifies, computed from a grid instead of a fold, coefficient for coefficient; message `j` is formed after `r_{j−1}` is squeezed under `tree/{ℓ}/r/{j − 1}`, so INV-FS-1 holds as today; the LogUp error `(N + T)/\|Fp2\|` (RFC-0008 'Relation') and the per-round sumcheck terms are unchanged. An honest prover's `u64` accumulator overflow would change a message and fail `p_j(0) + p_j(1) = claim_j`, never pass. | pass |
| Uniqueness upstream of the logits (SP-3, RFC-0009 Lemma 0) | No committed value, gadget, table or range changes; the witness is the same. | pass |
| Zero knowledge (G-5, RFC-0013) | The grid, the regenerated keys and the folded tiles are prover-internal secret-class state ([05 §3](../../docs/spec/05-observability.md#3-data-classification-and-redaction), `Secret<T>` per [06](../../docs/spec/06-security.md)); the masked sumcheck of RFC-0013 adds the same committed mask `G_k` to the same messages, so the simulator is unchanged. | pass |
| Verifier independence (SP-6) | Nothing under `verifier/` changes; the verifier's per-round check and labels are RFC-0010's and RFC-0014's. | pass |
| Post-quantum (hash-based commitments) | Unchanged: no commitment, hash or opening changes. | pass |

No `fail`, no `unknown`.

## Prior rejection check

- [RFC-0018 Alternatives Considered 3](../../docs/rfcs/RFC-0018-gpu-prover-runtime.md#alternatives-considered) rejects multi-variable sumcheck rounds for `(d + 1)^w` message growth. Not applicable: the window sends RFC-0010's messages unchanged, one degree-3 polynomial per variable; the `3^v` grid is prover bookkeeping ([SUR-0007](../surveys/SUR-0007-sumcheck.md#synthesis) synthesis 3 and [SUR-0003](../surveys/SUR-0003-passes.md#synthesis) synthesis 3 make the same point). The alternative's own reopening clause ("reconsider if RISK-10 measurements exceed 50% overhead") is not needed, since no protocol change is proposed.
- RFC-0018 Alternatives Considered 1 (NVMe instead of regeneration) is "kept as a fallback for linear-layer leaves, whose regeneration needs full GEMMs": the RFC knows the linear-layer passes are GEMM-shaped and prices neither option. This note prices the regeneration; the NVMe fallback at tens of GB/s against 1.7–9 TB of leaves per pass (E0) is 100–1,000 s per pass, which the RFC already judged slower.
- [baseline.md §7](../baseline.md#7-already-rejected) has no row for streamed-round batching or for the pass count of the lookup levels.
- SUR-0003's seed "Multi-round streamed passes" (ledger queue item 23) and SUR-0007's "Round-batched streamed windows" target the attention `stream` term (`p_s`, `c ≈ 8–9` at 16K); [IDEA-0006](IDEA-0006-fold-sum-fusion-roofline.md) (`shelved`) refuted deep fusion of in-memory levels on transcript legality (round `i + 1` needs `r_i`), which the grid answers without a protocol change, as SUR-0007 recorded; [IDEA-0005](IDEA-0005-base-field-leaf-small-value.md) (`triaged`) holds the base-field leaf layout and small-value first rounds for the in-memory case.
- **New:** the linear-layer lookup leaf levels, whose streamed rounds regenerate GEMM-shaped keys, are absent from the cost model, from RISK-9's `n_regen = 4` and from RISK-10's attention-only estimate, and their count couples to proof-unit sizing (OQ-0018-1) through the proof-size gate; the every-level reading of RFC-0018's 'Leaf levels' multiplies the count by about `(c + 1)/2`.

## Kill criterion

Any one of: (a) RFC-0018's planner never streams a lookup level at 16K, with units sized so every level fits inside the SC-3 proof-size gate (the E0 arithmetic above says 46 units at 32 B or 9 at 6 B, +9 to +51 GB of combined rows, so this needs a different unit design); then the term is proof-unit count and the idea moves proof size only. (b) In the E1 proxy the `3^v` grid accumulations and `2 · 1.5^v` `Fp2` additions per pair exceed the per-round prover's time per pair at `v ≥ 4`, capping windows at `v = 2–3` and the pass saving at 2× (SUR-0007 counts the multiplications as 15 against 33–38 at `v = 4`, so the additions must cost more than 20 multiplications per pair for this to fire). (c) The leaf stage's tile-to-index map cannot deliver aligned blocks of `2^v` consecutive pairs for the lookup leaves (a layout question for the leaf stage and OQ-0010-1, not a protocol one).

## Cheapest experiment

- **E0** (`s`, this machine): the pass table above per table instance at 1K / 4K / 16K tokens per unit in both layouts and both readings of RFC-0018 'Leaf levels', with the forward-equivalent fraction per streamed round read off RFC-0005's step table and the GEMM shapes of 405B; proposed as a `lookup_stream` line beside `regen` in [tools/cost_model.py](../tools/cost_model.py) (parameters: lookups per layer-token, bytes per leaf, tokens per unit, forward-equivalents per pass, window `v`), so that the term is scored like the others. Deliverable: the table and the model change, reviewed against RFC-0018's text by a reader who did not write this note (the every-level reading is this note's interpretation).
- **E1** (`s`–`m`, reachable here): a research workspace over `tessera-field` (own `[workspace]`, README rule 2) with a leaf-level fraction prover for 2^20–2^24 seeded `(key < 2^32, value < 2^16)` leaves: the per-round prover (regenerate, sum, fold, one round per pass) against the windowed prover at `v ∈ {2, 3, 4, 5}` (one pass per window), same harness, same inputs, Apple M4 Max, RFC-0020's protocol (3 warm-ups, 10 iterations, median and CV). Hypothesis: the window's messages equal the per-round prover's bit for bit on the same challenges, and its time per leaf per round is at most 1.5× the per-round prover's at `v = 4` while its bytes touched per round (the pass proxy) fall by `v`. Success: equality holds and the time ratio is below 1.5 at `v = 4`; kill (b) fires if the ratio exceeds the pass ratio `v` at every `v ≥ 2`.
- **Unknown until an H100 run** (E2, RFC-0020 v0.1–v0.5): the cost of one GEMM regeneration pass relative to the window's additions on the GPU, the pass counter's actual count (v0.3, E1 count on CPU), and OQ-0018-1's answer for the planner's default.
- Effort: `s` for the E0 table and the model line; `m` for the E1 workspace; no RFC (tier-C text of RFC-0018 and RFC-0020 only).

## Refutations

Sweep triage of 2026-10-16: four refuter agents who did not write the idea, one per lens; each argument is condensed from that refuter's verdict, with the cost-model reruns the scoring agent made to check disputed cells noted as such.

| Lens | Verdict | Argument |
|---|---|---|
| soundness | survives | Every message is still RFC-0010's degree-3 round polynomial, formed only after `r_{j−1}` is squeezed under `tree/{ℓ}/r/{j−1}` (RFC-0010:95-99, 108-112), and the grid is a function of the level's tables and `eq` point fixed before round 1, so INV-FS-1 holds and no RFC-0014 bound moves; an accumulator error produces a message that fails `p_j(0) + p_j(1) = claim_j` at the verifier, never one that passes. Two completeness details in the mechanism are wrong as written but harmless to soundness: (1) the leaf-level integer coefficients cannot be `u64`, since `key_L · key_R` alone is below 2^64 per pair and the line rule at z = 2 and the sum over 2^v pairs add about 5 + v bits, so the grid needs `u128` or a reduction per pair; (2) every fraction tree holds the zero-knowledge leaf `(0, δ)` with `δ` a random `Fp2` element (RFC-0013:86), so the block containing it is not base-field and must take the `Fp2` path. ADV-V sees the same transcript; the regenerated keys and the grid stay prover-internal secret-class state. The pricing claim is consistent with RFC-0018:81 (P2 'affected linear tiles, once per streamed round'), which indeed has no cost-model term. |
| prior art | survives | The pricing gap is verified against the primary text: RFC-0018's P2 row regenerates 'Attention and affected linear tiles, once per streamed round' (RFC-0018:81), Alternative 1 concedes linear-layer leaf regeneration 'needs full GEMMs' (RFC-0018:112), and the executable cost model's `stream` term counts attention scores only (research/tools/cost_model.py:140-144), so no term prices the lookup-level passes. RFC-0018 Alternative 3 (RFC-0018:114) rejects multi-variable rounds for `(d + 1)^w` message growth and does not cover a prover-side window that sends RFC-0010's messages unchanged, as SUR-0007 synthesis 3 already argued. The window mechanism itself is not new: it is SUR-0007's 'Round-batched streamed windows' seed and SUR-0003's 'Multi-round streamed passes' (ledger item 23), both aimed at the attention `stream` term, and IDEA-0005 mechanism 3 already proposes integer-accumulated first rounds for the fraction leaf level. What is new is the target (the batched `Requant`/`Silu` instances of RFC-0008 'Batching') and the forward-equivalent accounting per streamed round; the 'every-level' reading is this note's interpretation of RFC-0018:88, the same unresolved ambiguity IDEA-0009 carries. Nothing in baseline.md §7 rejects it. |
| cost model | weakened | The scenarios reproduce (corrected `regen` 7 → 5: μ(16K) 38.1 → 36.1, −5.2%; 1K 11.9 → 9.9, −16.8%; counted 81.8 → 79.8, −2.4% at 16K), and the saving is an additive 2.0 on μ. But the claim that these passes are 'absent from the cost model' overstates the gap: RISK-9 defines the four regenerations as witness, per-stage and opening passes 'plus the streamed passes of RISK-10' (00 §9) and 08 §1's `n_regen` row reads 'witness pass, streamed tree leaf levels, opening pass' (verified at triage), so about one forward-equivalent of P2 is already priced and the reference is 6, not 7: `regen` 6 → 5 gives 37.1 → 36.1 (−2.7%) at 16K and 10.9 → 9.9 (−9.2%) at 1K (rerun at triage). The stake itself ranges 2–11 forward-equivalents depending on the leaf layout (6 B against 32 B) and on an 'every-level' reading the note labels its own interpretation, and it vanishes if the planner shrinks proof units instead (RFC-0018:90, kill (a)), turning the gain into proof size; the pass price of one forward-equivalent per streamed round is the note's own attribution (E0). At 16K the gain is of the size of the regen term's own uncertainty, so the sign is robust and the magnitude is not. |
| feasibility | weakened | The accounting half is well founded: RFC-0018:81 says P2 passes cover 'affected linear tiles, once per streamed round' and the cost model's `streaming(S)` prices attention scores only (cost_model.py:15, 140-158), so the linear-layer streamed rounds are indeed unpriced beyond the one forward-equivalent of 08 §1; but the 'every-level reading' (Σ = 13–28 rounds) is the note's own interpretation of RFC-0018:88, and the pass table is unreviewed own arithmetic. The windowed mechanism is implementable ([DDBDT26] §5–6, SUR-0007's own count), prover-only, messages unchanged, so no RFC; but its gain is contingent on two open questions: OQ-0018-1 (RFC-0018:140: the planner may split proof units instead of streaming, making the term proof size, not μ) and OQ-0010-1 (RFC-0010:170: the leaf stage's layout is undefined and `crates/tessera-logup/src/lib.rs` is a 5-line stub), so kill criterion (c) (aligned `2^v`-pair blocks) cannot be checked against any code today. The E1 experiment is reachable here only as a proxy over `tessera-field` (effort `m`), and the pass cost it stands in for (one GEMM regeneration per round) is E2. |

## Score

| Gain | Confidence | Cost | Priority |
|---|---|---|---|
| −2.7% μ(16K) (E0 cost model, `corrected`, `regen` 6 → 5, 37.1 → 36.1: the reference is 6 because 08 §1's `n_regen = 4` already books 'streamed tree leaf levels'); −9.2% at 1K under the model's 16K-token unit (0 if a unit holds one 1K request); −5.2% / −16.8% at the note's `regen` 7 reference; up to about −20% under the every-level reading (E0, own arithmetic). The pass table and the one-forward-equivalent price per streamed round are the note's own E0 | 0.3: the pass count, the per-round price and the every-level reading are the note's own interpretations of RFC-0018, OQ-0018-1 may turn the term into proof size, the stake spans 2–11 forward-equivalents, and the E1 proxy cannot price a GEMM regeneration (E2) | `m`: about a week for a leaf-level windowed fraction prover over `tessera-field` (own workspace) plus the E0 pass table and a `lookup_stream` cost-model line (`s`); no RFC (tier-C text of RFC-0018 and RFC-0020 only) | 2.7 × 0.3 / 2 = 0.41 (0.78 at the note's reference; about 3 under the every-level reading) |

## Decision and links

Triage 2026-10-16: `triaged` at priority 0.41. The valuable part is the accounting: the E0 pass table per table instance and the `lookup_stream` line in `cost_model.py` (effort `s`), reviewed against RFC-0018 'Pass structure' and 'Leaf levels' by a reader who did not write the note and tied to OQ-0018-1, should be done before any kernel work, and the two completeness fixes the soundness lens found (`u128` or a per-pair reduction for the leaf grid; the block holding the `(0, δ)` leaf on the `Fp2` path) go into the mechanism when the note is next edited. The windowed prover shares its `3^v` grid with IDEA-0022 and should be a case of that experiment rather than one of its own. Experiments: none of its own (a case of IDEA-0022's, if that runs). RFC or issue: none; the pass table and model line as a research deliverable. Ledger line updated on: pending consolidation (2026-10-16).


Experiment triage 2026-10-19: the committed fraction amendment of [EXP-0012](../experiments/EXP-0012-integer-leaf-round-window/README.md) supplies the hosted in-memory fraction-window case at 2^20–2^24 leaves and v=2..5. All messages/final evaluations/states equal current production prove_batch's three proxy instances, accepted by independent tv-core from true hypercube claims. v4 costs 2.50–2.54× split (E1), failing the registered 1.5× ceiling; v2 costs 1.54–1.55×, below its factor-2 kill comparison, so the case is inconclusive rather than refuted. The largest v5 arm also remains CV 5.42% after one rerun. Count the grid's original-input pass and the once-bind pass separately: v4 has a factor-2 counterfactual pass reduction after binding, not a measured factor-4 regeneration win. No GEMM regeneration, full LogUp proof, planner or H100 is measured. Keep the streamed hypothesis triaged with its existing conditional score; the E0 pass accounting and E2 layout/regeneration questions still decide it.
