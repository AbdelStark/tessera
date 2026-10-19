# IDEA-0035: A second Ligero commitment over the combined opening rows

| | |
|---|---|
| Status | `triaged` (2026-10-19, independent four-lens review completed; unknown gates; no promotion) |
| Origin | Ledger seed37; [SUR-0005](../surveys/SUR-0005-commitments.md) Ligerito-style level2; commitment/parameters/wild lenses |
| Lever | `size`, `verify`; additional prover commitment/sumcheck work; no reduction of original opened columns or credited Δμ |
| Spec surfaces | RFC-0011 opening steps5–8 and descriptor/layout, RFC-0013 combined-row masking, RFC-0014 transcript/soundness/parameter sets, RFC-0019 check order, RFC-0031/0036/0037/0047. Depends on a specified coefficient-basis route from IDEA-0002 for cheap public weights; tier-A superseding RFC/fresh version required. |
| Priority | 0: no credited gain until the full relation, extractor/soundness budget, current group counts and joint hiding are established |

## Mechanism

Replace the **transmission**, not the computation or original binding obligations, of RFC-0011's combined vectors with one new hash-based Ligero commitment `C_U`. The original advice commitment and every distinct queried original column remain. No circuit proves a hash or a verifier: this is an additional algebraic PCS level, ending with an ordinary small-row opening, not recursive verifier compression.

The target uses the coefficient-basis proposal [IDEA-0002](IDEA-0002-coefficient-form-rows.md), which is still triaged and unaccepted. Its message coordinates are polynomial coefficients; column weights are monomials. Keeping current evaluation-basis rows instead would require dense Lagrange weights and a separate explicit verifier-cost analysis. No cost improvement from IDEA-0002 is counted again here.

Let `m=G+1` be the actual number of opening vectors, including proximity. For unmasked protocol/1 call them `U_g=u_g`; for a future hiding version use exactly the already blinded `U_g=u'_g` and include the original blinding rows in column coefficients. Commit both Fp components of every U_g using one base-field tensor `V[g,a,i]`, with component lane `a=0,1` fixed publicly and `U_g[i]=V[g,0,i]+X*V[g,1,i]`. Its raw payload is `2*m*k1` base-field elements (E0 shape count), not `m*k1` base-field elements. At the final U point, the two component claims use the same group/coefficient point and the fixed Boolean lane 0 or 1. This is not a silently new Fp2-native PCS. Pad only to public zero coordinates. Keep the short combined encoding-mask polynomials `M_g` transmitted and bound as before. An illustrative second-level row width is `k2=2^14`, rate near1/2 with explicit mask degree; these are E0 design parameters, not approved security parameters. Derive level2 allocated rows R2, nonzero committed rows M2 and nu2 separately under RFC-0047; `ceil(2*m*k1/k2)` is only a packed base-row lower bound before mask/blinding reservations and alignment, not the approved M2. Tensor alignment can enlarge M2 beyond that count. Bind C_U's dimensions, group order, basis and both-level identities in a new descriptor/protocol version.

Preserve and link **both** classes of original equations. For original column index j with point x_j, retain its canonical column-chain/Merkle check and compute its left side locally. In the coefficient-basis variant:

```text
sum_i U_g[i] * x_j^i
  = sum_r c_g(r)*original_column_j[r] - Phi_basis(x_j)*M_g(x_j).
```

For an evaluation group at column subpoint p_g, retain:

```text
sum_i U_g[i]*eq(p_g,i) = sum_t gamma^(t+1)*v_t [+ q_g].
```

The bracketed term is present only in the future hiding schedule and stays absorbed before gamma. A proximity vector has column equations but no fabricated evaluation claim. Weight openings are outside the initial scope. For the chosen IDEA-0002 coefficient route, `Phi_basis(x)=x^k1`; the current evaluation-basis route instead uses `Phi_basis(x)=Z_H(x)=x^k1-1` and Lagrange rather than monomial column weights. These are different formats; the eventual accepted equation, natural semantic columns and descriptor discriminator must match the chosen basis. No old root is reinterpreted.

After C_U, every M_g, all original response columns and the target values above are fixed, draw fresh batching coefficients for the complete canonical equation list. Prove their one batched inner product `<U,W>=Y` by an ordinary sumcheck over the public padded group/coefficient cube. Public W is a sum of group indicators times eq vectors or monomial vectors; monomials factor as `product_b (1-r_b+r_b*x_j^(2^b))`. Do not replace an extension of a product by a product of extensions. At the final point, independently open U's two Fp component claims under C_U and check their Fp2 reconstruction. Their truth is an outstanding obligation until the level2 opening accepts; a must-use sumcheck terminal claim alone is not acceptance.

Use a standard nonrecursive level2 opening: compute its small combined and proximity rows, send them explicitly, query and check its columns against C_U, and check the final U evaluation with the level2 combined row. The second-level proximity coefficients, claim batching, combined-row absorption and query ordering need their own complete schedule. The new root must precede every dependent query/challenge; the first-level column queries must follow C_U (the replacement for receiving U) and M. Do not reuse first-level challenges, claim powers or labels by default. Live and NI modes, grinding coverage, selective aborts and replay each need explicit accounting.

## Why it could work

- [NA25], as read in SUR-0005, supplies the primary Ligerito matrix-vector/partial-sumcheck mechanism. Its different field and parameter theorem are motivation, not an instantiated Goldilocks/Fp2 soundness proof here.
- E0, primary specification: current RFC-0011 sends every U in full, although later checks consume only linear functionals of it. A binding commitment plus a linked inner-product proof can in principle check those functionals without transmitting U. Merely sending C_U with no reduction would leave original claim/column consistency unchecked and is rejected.
- E0, primary algebra: coefficient-basis monomial and MLE eq weights have factored evaluation formulas. Current evaluation-basis RS weights do not gain that property by renaming the message. That is why the IDEA-0002 dependency is explicit.
- The second commitment is over data already materialized by P4, not the multi-terabyte original advice. Original P5 is still required; no full-forward pass is claimed removed.

## Expected gain

Unevaluated; **zero credited gain** in μ, total bytes and verifier time. The historical 17-vector/1.14GB/48% figures in SUR-0005 are not a current estimate: the full joined stage/gadget claim pool and its group budget (review item R21) have not been established. Current allocated R and committed M differ; using either as the other changes proof bytes and soundness dimension.

E0 dimensional bookkeeping only: original transmitted U payload is `16*m*k1` bytes, with `k1=2^22` today. Candidate payload includes **unchanged original columns and multiproof**, the new root/descriptor, all M polynomials, batched sumcheck messages, all level2 combined rows/masks, `8*M2*tau2_distinct` base-field column-symbol bytes, and level2 Merkle paths. Level2 mask/row reservations and padding count too. No tau2 is approved until the complete two-level budget meets G-4. Current μ lacks this protocol's prover/verification/size terms, so a fabricated commit_ps reduction would be misleading.

The unchanged E0 control is:

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
| Soundness bits (G-4, RFC-0014) | Additional commitment proximity, equation batching, sumcheck and final-opening errors need a joint extractor/reduction and both-mode budget. Bind every original evaluation and column equation to the same U, including proximity and all groups. No automatic reuse of the Ligerito theorem or marginal 2^-120-style term is accepted. | unknown |
| Uniqueness upstream of the logits (SP-3, RFC-0009 Lemma 0) | Does not change D, gadget relations, lookup tables, integer bounds or canonical witnesses. If opening binding fails the whole proof is unsound; that remains the soundness gate rather than a claimed uniqueness repair. | pass (unchanged semantic surface only) |
| Zero knowledge (G-5, RFC-0013) | Replacing explicit blinded rows by commitments/linear proofs is not automatically a subset-view theorem: new roots/challenges, adaptive queries, mask reuse and conditioning require simulation. Existing #145/#394 joint-mask obligations remain unknown, including both Fp components, level2 masks and original epoch views. No hiding adoption before that argument. | unknown |
| Verifier independence (SP-6) | Requires a new independent descriptor/PCS/relation implementation and oracle vectors in tv-* from a superseding RFC, with explicit terminal-claim consumption. No prover code may be shared. | unknown (future implementation) |
| Post-quantum (hash-based commitments) | Uses the current hash-based RS commitment family over Goldilocks/Fp2, without pairing or lattice additions. This is not a new quantum security proof. | pass (unchanged assumption only) |

ADV-P attacks include fixing different U values for different original equations, dropping the proximity vector, changing a component/group order, fabricating final PCS values, and choosing U after first-level queries. ADV-V attacks include adaptive queries or aborts that create an extra conditioned view of existing blinders. A complete transcript simulation/extraction obligation includes these; tests alone will not discharge it.

## Prior rejection check

Checked baseline §7 and RFC-0011/0013/0014/0018 alternatives on 2026-10-19. RFC-0011 Alternative2 rejects folding the original nonresident witness; this candidate works only on its materialized combined rows and retains original column checks. Alternative5 rejects thousands of per-tensor roots; this adds one aggregate root, not one per tensor. NG-3 rejects recursive compression of the verifier; no verifier/hash circuit is proposed, but the explicit no-recursion scope still requires maintainer interpretation in an RFC. IDEA-0002 changes the row basis, not the transmission of combined rows; composition is a dependency, not duplicate credited gain. IDEA-0013/0025 change rates/query parameters rather than add this commitment/reduction level. No earlier IDEA-0001..0033 specifies this separate mechanism. There is no newly read theorem/measurement that closes its gates.

## Kill criterion

At the same complete G-4 budget and the same current workload/claim pool, if the minimum fully accounted two-level proof payload is **not smaller** than the original opening payload, refute this size-oriented mechanism. This comparison includes both levels' columns, paths, masks, final rows, sumcheck and descriptors and never deletes original columns. Any false original equation accepted by the new relation or any established hiding failure rejects it earlier by research rule1, regardless of size.

## Cheapest experiment

Pre-register an E0 relation/extraction preflight, effort `m`: specify the complete two-level transcript and prove that acceptance binds the original evaluation and column equations to one U except for explicitly budgeted events; independently check tiny-field transcripts with mutations dropping each equation, changing one component/group, moving C_U after queries and dropping the final opening. Include an unchanged control and retain a malicious balanced cross-group case, not only single-value corruption. Derive current R/M/nu, complete joined G/T_g and exact serialized payload counts before claiming a size result. No experiment ID, harness or execution is created here. Only after proof review should a same-harness E1 prototype compare both opening variants and acceptance by independently written verification. H100 prover costs and full end-to-end verification remain unknown; adoption is `l` with tier-A version/schedule/PCS work.

## Refutations

The independent reviewer confirmed all four verdicts on2026-10-19; [the review record](results/seed-review-2026-10-19.json) pins the reviewed drafts and specification evidence. This is adversarial triage, not protocol approval.

| Lens | Verdict | Argument |
|---|---|---|
| soundness | weakened | The explicit sumcheck/final opening supplies the missing coupling a bare C_U would lack, but no complete two-level extractor, grinding budget or joint simulator is supplied. Unknown is not approval. |
| prior art | survives | Ligerito already provides the general approach. The new research mechanism is its bounded application to Tessera's materialized combined rows, with original column consistency retained and an explicit coefficient-basis dependency. It is distinct from rate changes and row-basis changes. |
| cost model | weakened | The size/verifier deficit is structural, but old group counts and timing projections are stale and the new proof/cost terms do not exist in the current model. No net reduction, 17-vector assumption or elimination of original columns is credited. |
| feasibility | weakened | A finite relation/budget/byte-count preflight is CPU-feasible. Full proof review, basis/version dependency and independent PCS integration are substantial; unavailable H100 evidence is not needed to falsify the relation but is needed for a prover-performance claim. |

## Score

| Gain | Confidence | Cost | Priority |
|---|---|---|---|
| 0 credited; candidate size/verification/μ deltas unevaluated | 0.1: plausible composition, no complete binding/budget/ZK/count result | `l` (weight4): tier-A PCS/schedule plus IDEA-0002 dependency; initial E0 preflight is `m` | 0*0.1/4 = 0 |

## Decision and links

`triaged`, zero active priority and unknown gates; not promoted. Experiments: none. RFC/issue: none created. Independent four-lens review completed; ledger consolidated on2026-10-19. Next action: pre-register a bounded E0 two-level relation/extractor, transcript and current-count preflight, treating the proposed basis as an explicit experimental assumption. Adoption still needs the superseding RFC.
