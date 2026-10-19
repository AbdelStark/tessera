# IDEA-0034: Public windowed attention shapes with exact masked-position constraints

| | |
|---|---|
| Status | `triaged` (2026-10-19, independent four-lens review completed; unknown gates; no promotion) |
| Origin | Ledger seed33; [SUR-0004](../surveys/SUR-0004-attention.md), public static attention patterns; attention/counts/wild lenses |
| Lever | `A`, score commitments/lookups/max leaves, P4/P5 and streaming counts; changes `D` and its native denominator, with fidelity cost unknown |
| Spec surfaces | RFC-0005 integer semantics/fidelity, RFC-0006 lowering/tiling, RFC-0007/0026 claim-list schema/validation, RFC-0009/0030/0035 softmax/padding, RFC-0010/0037/0038 schedule/template, RFC-0011/0034/0047 layout, RFC-0013 masking, RFC-0014 transcript/budget, RFC-0017 scoring, RFC-0041/0044 plan/lowering. Tier A: superseding RFC and versioned descriptor required. |
| Priority | 0: no credited gain while counts, statement/fidelity policy and security gates remain unresolved |

## Mechanism

Define a new integer model `D_pattern`, rather than claiming to accelerate the existing full-causal Llama-3.1 `D` unchanged. A public descriptor fixes a model-wide window `W >= 1` and, optionally, one public global/local flag `g_l` per layer. The narrow first candidate uses all-local layers and `W=1024` (a design parameter, not a calibrated fidelity choice). A later periodic global/local policy is the same mechanism, not a second idea. Bind the exact descriptor, its canonical version and the lowered layout into `IntegerModel`, ClaimList/statement digests and commitment descriptors **before** seed or LogUp challenges. Serving, scoring, witness generation and the independent verifier must all implement that same bound model.

For real layer/head/query/key coordinates, define the public Boolean-domain selector

```text
P(l,h,q,i) = [l<L][h<H][q<n][0<=i<=q]
             * (g_l + (1-g_l)*[q-i<W]).
```

The diagonal is retained, so each real row is nonempty. Padding has `P=0`. Derive the exact multilinear extension of this Boolean function; multiplying independently extended comparison predicates is not its extension. An interval-comparison/carry automaton is a candidate evaluation algorithm, not a proved closed form supplied here. Public per-layer parameters must preserve a single generic layer template or account explicitly for a different schedule: RFC-0038's rejection of freely different templates cannot be bypassed by calling them flags.

Keep a logical padded score cube and the existing max-shift semantics **over the selected set**. Replace the causal selector by `P` in the virtual lookup key and max leaves, retaining all current pins:

```text
kappa = P*(c_q-x_qi) + (1-P)*K0,     Exp[K0]=0,
product_leaf = P*(c_q-x_qi) + (1-P),
s_q = sum_i e_qi,                    PV_q = sum_i e_qi*v_i.
```

These are Boolean-domain identities reduced by sumcheck, not products of extensions at the final point. Every masked position in a committed row is pinned to `(K0,0)` by the lookup; a skipped physical row is public zero and cannot contain prover-chosen advice. Max trees still use constant-one masked leaves, including both the old causal suffix and the new old-key prefix; their dyadic decomposition, tree multiplicities and forest masks remain obligations. Retain current padded-row `c=R=0` pins and reciprocal/exp_sigma nonwrap preconditions. Prove the max/reciprocal uniqueness argument for nonempty selected rows; apply the padded branch only where the public selector says the row is padded.

Use the current public key-block layout to omit only rows entirely outside this public window. Mixed boundary rows remain committed and constrained. Layout mapping, `nu`, allocated `R`, physically committed `M`, zero gaps, diagonal/halo blocks and short-query blocks must be rederived together under RFC-0047. Keep the full virtual score relation `x=QK` on the logical cube: an excluded score itself is not forced to zero. Only its exponential `e` is public zero outside the selected set. The selector/key reduction must transport its resulting score claim to the ordinary dense MatMulAct relation, and `PV` must still sum over the full logical key cube with the omitted e entries defined as zero. No masked-score kernel or product-of-extensions shortcut is assumed. Keeping dense intermediate arithmetic for proof generation might erase the expected time saving, so it must be counted. No secret top-k decision, skip bit, sorting witness or routing certificate is introduced.

## Why it could work

- E0, primary specification: RFC-0030 pins masked exponentials and RFC-0034 already skips public all-zero causal rows without deleting their logical constraints. A public window extends that form of structure, but the existing selector/layout is insufficient as written.
- [Gemma25], as read in SUR-0004, supplies a primary model-design precedent for interleaved local/global layers; it supplies **no** fidelity result for retrofitting this integer Llama model. [Nawrot25] cautions that sparse-attention quality depends on workload and configuration. Those historical readings motivate a candidate, not an adopted model or new measurement.
- E0, primary arithmetic: for one real row at position `q`, an all-local pattern retains `min(q+1,W)` keys. Over a length-`n` sequence with `n>=W`, the logical pair count is `W*n-W*(W-1)/2`, versus `n*(n+1)/2` dense causal pairs. This does not count padded/committed boundary rows, sumcheck work, masks or native execution.

## Expected gain

Unevaluated; **zero credited gain**. The current model's scalar `A*S` cannot describe layer-dependent windows, partial tiles, unchanged linear work and a changed native denominator at once. Merely reducing `scores_per_token_per_ctx` would overstate evidence. A future comparison must report both absolute proof time/work relative to the original dense deployment and `mu_pattern = proof_work(D_pattern)/native_work(D_pattern)`, plus the fidelity/scoring bound `U-hat` against the named deployment. Neither denominator nor fidelity is measured here.

The exact current E0 control is:

```sh
python3 -I research/tools/cost_model.py --profile counted \
  --contexts 1024,4096,16384,131072 \
  --set commits_per_token=45566660 --set lookups_per_token=45440678 \
  --set table_rows=7693467648 --set leaf_ps=41.8 \
  --set column_answer_ps=94 --set row_combination_ps=0 \
  --set regen_full_equivalents=5 --json
```

This unchanged control reproduces μ = 32.294103 / 46.785504 / 102.443917 / 436.310457 at 1K / 4K / 16K / 128K (E0, current compiler counts and model assumptions, primary repository evidence). It is **not** the candidate's prediction. P4 remains unpriced, P5 is a full-re-encode planning bound, the rest of leaf-stage streaming (#329) and unallocated gadget range tables remain outside these partial totals. No H100 or end-to-end evidence is created. The inspected model SHA256 is `e9d460183078f38e0267d7aae4554c4293fef15253202227504754f5db88c0f6`; the c2/B32 compiler-report SHA256 is `829c098d79f5eab062d1f485d02afca07cf6c02b2f69f4ef7a432c2b2f952b6c`.

| Context | Δμ | Δ proof size | Δ verification |
|---|---|---|---|
| 1K | unevaluated; 0 credited | unevaluated; 0 credited | unevaluated; 0 credited |
| 4K | unevaluated; 0 credited | unevaluated; 0 credited | unevaluated; 0 credited |
| 16K | unevaluated; 0 credited | unevaluated; 0 credited | unevaluated; 0 credited |
| 128K | unevaluated; 0 credited | unevaluated; 0 credited | unevaluated; 0 credited |

## Security gate

| Property | Effect | Pass |
|---|---|---|
| Soundness bits (G-4, RFC-0014) | Unknown until exact public selector, complete relation/lookup/tree counts, descriptor binding and both-mode schedule are derived and independently checked. No unbudgeted selector compression or smaller query count is assumed. | unknown |
| Uniqueness upstream of the logits (SP-3, RFC-0009 Lemma 0) | The selected-set maximum is unique in value when every real row includes its diagonal, all selected keys are true integers, masked e values are pinned and current reciprocal bounds hold. Complete proof/constraints for D_pattern and boundary/padded rows are not yet supplied. This does not prove equality to the old D. | unknown |
| Zero knowledge (G-5, RFC-0013) | The pattern and lengths are public by policy; no witness-dependent sparse route is exposed. Existing tree masking, full conditioned view and #145/#394 joint-simulator obligations remain; a different physical layout needs its own inventory. | unknown |
| Verifier independence (SP-6) | No shared code is proposed. The descriptor, selector MLE, layout, relations and counts need an independent implementation from the superseding RFC and independent oracle data. | unknown (future implementation) |
| Post-quantum (hash-based commitments) | Retains Goldilocks/Fp2 and the current hash-based commitments; no pairing or lattice assumption is added. | pass (unchanged assumption only) |

ADV-P must not choose a window after seeing challenges, drop a selected coordinate by relabelling it padding, choose max from an unselected coordinate or alter an unconstrained e in a mixed boundary row. ADV-V must learn no data-dependent sparsity from layout/length or an extra opening. Public-policy disclosure is an explicit statement choice, not a substitute for #145's simulator. Any security fail rejects the idea before performance.

## Prior rejection check

Checked current baseline §7 and RFC-0005/0010/0011/0013/0018/0034/0038 alternatives on 2026-10-19. RFC-0034 Alternative2 rejects proving only a nonrectangular set of valid positions: this proposal retains the logical cube and masked-position pins. RFC-0038 Alternative1 rejects freely different layer templates: the generic pattern-node schedule must be specified, not assumed. RFC-0005's non-Llama-family non-goal means Gemma is only motivation, not authorization to claim its model support. No existing IDEA-0001..0033 is this public-shape mechanism; dynamic certified zero skipping and top-B routing have different statements and security/cost obligations. This is a new research note from an existing untriaged seed, not a reopening based on a fresh measurement.

## Kill criterion

One complete passing witness for the proposed selected-set relation whose output differs from a direct exact-integer `D_pattern` reference on the same public descriptor refutes the mechanism (E0 counterexample). In particular, an accepted nonzero excluded exponential, a false selected-set maximum or a nonzero padded row is sufficient. Passing this gate alone earns no performance or fidelity claim; the route remains inactive until a named policy meets the RFC-0005 fidelity/scoring gate and lowers fully counted cost.

## Cheapest experiment

First pre-register an E0 selector/layout/uniqueness audit, effort `s` for the bounded mathematical unit: enumerate tiny windows, non-power-of-two lengths/layers/heads and tile boundaries; derive the selector extension from its Boolean table; compare proposed factor evaluation against direct MLE; derive and check score-claim transport to full `x=QK` and the full-domain PV relation with zero e outside the pattern; check every real row has its diagonal and every excluded/padded entry is zero; exhibit old-full-causal and new-window model outputs separately. Include one-check-at-a-time malformed-descriptor, masked-e, max-membership and padding attacks. Then derive physical row/lookup/tree/mask counts and a denominator-correct planning scenario. No experiment ID, code or execution is created here. Real-weight fidelity and H100 native/prover performance remain E2 gates; synthetic success cannot establish them. Adoption effort is `l` because tier-A semantics/schema/schedule change.

## Refutations

These four rows were independently confirmed on2026-10-19; [the review record](results/seed-review-2026-10-19.json) pins the reviewed drafts and specification evidence. This is adversarial triage, not protocol approval. The [2026-10-19 exact preflight control](results/seed-preflight-2026-10-19.json) gives a concrete selector trap: for two positions and W=1 at q=i=1/2, multiplying the causal/lower-bound MLEs gives 9/16 whereas the joint Boolean-mask MLE is 1/2 (E0, primary repository-oracle arithmetic, independently executed by the reviewer). This refutes the shortcut, not the exact-selector mechanism, and is not a full-domain exhaustion or protocol proof.

| Lens | Verdict | Argument |
|---|---|---|
| soundness | weakened | Known masked-e and padded-row attacks return if the window is treated as an unchecked skip set. The logical-cube/key-zero/constant-one design addresses the attack shape, but a full selector reduction, uniqueness proof and joint ZK inventory are missing. |
| prior art | survives | Public windows are established model structure, not a novel sparse-attention algorithm. The distinct Tessera mechanism is binding that structure into D/ClaimList and the public-zero-row proof layout; current causal-only RFCs and existing ideas do not implement it. |
| cost model | weakened | The dominant attention/counts sensitivity makes this worth studying, but logical pair savings do not determine committed rows, native time or U-hat. Zero gain is credited rather than inheriting old sparsity projections. |
| feasibility | weakened | The exact small-domain audit is CPU-feasible. Tier-A model/schema/lowering/scoring changes, uniform-template treatment, real-weight fidelity and H100 validation prevent a local adoption result. |

## Score

| Gain | Confidence | Cost | Priority |
|---|---|---|---|
| 0 credited; all candidate scoreboard deltas unevaluated | 0.1: plausible mechanism, no complete protocol/count/fidelity result | `l` (weight4): tier-A RFC and E2 fidelity/performance; preliminary E0 audit is `s` | 0*0.1/4 = 0 |

## Decision and links

`triaged`, with unknown gates and zero active priority; not promoted. Experiments: none. RFC/issue: none created. Independent four-lens review completed; ledger consolidated on2026-10-19. Next action: pre-register a bounded E0 selector/layout/uniqueness audit, with the changed-model scope and fidelity boundary explicit.
