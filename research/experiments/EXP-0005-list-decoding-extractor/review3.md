# EXP-0005: fresh review of the current RFC-0021 Draft, 2026-10-19

This is a new adversarial reading under `soundness-review`, not a renewal of the 2026-10-16 approvals. The reviewer did not author RFC-0021. Scope: the Draft at SHA-256 `57e24256f7985008878f54b40721169e883f016af3918a49aa1d14cb5d41b1bb`, in checkout `46d39e45e83561b2139ace790e48cf27cf6e4753`, against main `de5a70ca534a940e230c6f30a1518581bc117a26` through RFC-0047/0048, the original EXP-0005 report, `terms.py` and its recorded output. No RFC, historical experiment section, production implementation or ledger was changed by this review.

**Verdict: keep Draft; no full soundness/ZK approval.** The pinned exact-agreement/list statements and default arithmetic support the proposed non-hiding argument, subject to the explicit bridge and edge-case findings below. The masked protocol has an unresolved joint-freshness/simulation obligation and a literal invariant conflict. These are acceptance blockers, not an established secret-recovery attack. OQ-0021-4 and OQ-0021-6 remain maintainer decisions; this review supplies no invented answer to either.

## Primary-source check and its limits

The reviewer freshly fetched eight files from ArkLib commit `4dae3acff846125521b7b3998010885071746f16`, read the theorem statements, their glue and relevant proofs/definitions, and recorded SHA-256 values in [review3-summary.json](review3-summary.json). These are the actual pinned sources, not citations inferred from prior reviews. This is source-level E0 review; Lean and its transitive dependencies were **not rebuilt**, and no mechanized Tessera protocol reduction was constructed.

- [LineToAffine.lean, theorem at lines 193–207](https://raw.githubusercontent.com/Verified-zkEVM/ArkLib/4dae3acff846125521b7b3998010885071746f16/ArkLib/Data/CodingTheory/ReedSolomon/MutualCorrelatedAgreement/LineToAffine.lean): the exceptional set is fixed for the received family and radius, before the seed/candidate. Outside it, the polynomial decomposition has equality of full agreement sets, at density at most `B/(q−1)` in every affine dimension. It requires the integer threshold inequality and dimension `k′ ≤ n(1−radius)`.
- [Johnson/Agreement.lean, lines 104–117](https://raw.githubusercontent.com/Verified-zkEVM/ArkLib/4dae3acff846125521b7b3998010885071746f16/ArkLib/Data/CodingTheory/ReedSolomon/MutualCorrelatedAgreement/Johnson/Agreement.lean) and [Johnson/Probability.lean, lines 50–62](https://raw.githubusercontent.com/Verified-zkEVM/ArkLib/4dae3acff846125521b7b3998010885071746f16/ArkLib/Data/CodingTheory/ReedSolomon/MutualCorrelatedAgreement/Johnson/Probability.lean): the rounded Johnson theorem supplies the exact line-agreement premise. Required: `1 ≤ D ≤ n−2`, positive slack and `sqrt(D/n)+η ≤ 1`; no characteristic restriction appears in this instantiation.
- [FiniteBounds.lean, lines 66–108](https://raw.githubusercontent.com/Verified-zkEVM/ArkLib/4dae3acff846125521b7b3998010885071746f16/ArkLib/Data/CodingTheory/HiddenDerivative/Parameters/Johnson/FiniteBounds.lean): the transcribed `m,t,μ,h,θ,E₀` definitions match, including natural subtraction. The relevant subtractions are positive at the checked profiles.
- [PairwiseJohnson.lean, lines 75–81](https://raw.githubusercontent.com/Verified-zkEVM/ArkLib/4dae3acff846125521b7b3998010885071746f16/ArkLib/Data/CodingTheory/ReedSolomon/ListDecodability/PairwiseJohnson.lean): the word-list bound requires `D+1 ≤ A` and `nD < A²`. The matrix reduction is an additional argument: if 50 close matrices existed, a separating functional exists because at most `binom(50,2)=1225 < q` proper kernels must be avoided. Their 50 images would contradict the word bound 49. This avoids assuming that an arbitrarily large list already satisfies `L² < q`.
- [ProximityGenerator/Basic.lean, lines 341–342](https://raw.githubusercontent.com/Verified-zkEVM/ArkLib/4dae3acff846125521b7b3998010885071746f16/ArkLib/Data/CodingTheory/ProximityGenerator/Basic.lean) defines `(1,x)`; `AffineGenerator.lean` and [FullAgreement.lean](https://raw.githubusercontent.com/Verified-zkEVM/ArkLib/4dae3acff846125521b7b3998010885071746f16/ArkLib/Data/CodingTheory/ReedSolomon/MutualCorrelatedAgreement/FullAgreement.lean) provide the affine transfer and interpolation argument. They do not themselves prove Tessera's mask freshness, transcript security or artifact binding.

The paper statements/negative capacity results mentioned in the Draft were not re-proved or independently re-audited here. The security conclusions below rely on the freshly inspected pinned Lean statements. No capacity-regime claim or whole mathematical validation follows from this review.

## Findings

### R3-1: the unmasked zero-row weight matrix has no fixed first row

**Concrete completeness/count defect.** [RFC-0040](../../../docs/rfcs/RFC-0040-weight-commitment-artifacts.md#the-matrices) permits a model with no `ChannelParams`; that matrix commits no data rows. In the unmasked protocol there are no epoch blinding rows, so `M=0`. Its leaves are explicitly `d_j[0]`. RFC-0021's first nonzero row and `C=M−1` are undefined here.

Specify an explicit empty-matrix branch, including the zero challenge count, zero codeword/claim semantics, transcript position, shape and verifier checks. Do not silently choose a nonexistent fixed row or use natural subtraction without specifying what is checked. Add an independent no-ChannelParams vector. This branch needs no affine-generator theorem; its matrix is public. The parent must choose its normative message treatment.

### R3-2: bridge the fixed physical pair to the affine theorem

**Proof exposition gap with a direct E0 derivation.** In masked mode there are `M−2` free coefficients and a fixed pair with coefficients `1,i`. Applying the theorem literally to `M` physical rows is not its `(1,x)` generator. Apply it instead to `M−1` virtual words, with fixed word `V₀=U_a+iU_b`, and all other physical rows unchanged. The free dimension is then exactly `M−2`. In unmasked mode with `M≥1`, use the first physical word and dimension `M−1`.

For the masked fixed word the theorem gives a polynomial `P₀∈Fp2[X]`. Split its coefficients uniquely as `P_a+iP_b`, with `P_a,P_b∈Fp[X]` and degrees at most `D`. Since the coset points are in `Fp`, equality to `U_a+iU_b` at a point is equivalent to both physical component equalities. Thus the common agreement set and linear combination lift back to all physical rows. For the remaining rows, the Draft's Frobenius/interpolation argument gives `Fp` coefficients. This justifies physical-row extraction without applying the false premise that `V₀` itself is `Fp`-valued.

Write this bridge into the RFC. It does not prove ZK. Do not repair R3-3 by excluding other blinding rows without redoing the received-family/list argument.

### R3-3: one-use wording conflicts with the full free-row stream; joint ZK remains unknown

**Acceptance blocker/unknown, not a demonstrated leak.** The Draft explicitly gives every other nonzero physical row a free coefficient, including the groups' blinding pairs and all other epoch blinding rows. A group's pair therefore contributes both to its designated revealed group combination and to the proximity combination. An epoch's future/previous mask rows can also contribute to proximity combinations of other proofs.

This contradicts the literal “used in exactly one revealed combination” of Draft INV-PCS-8 and main INV-ZK-2. The tension is partly inherited: main's tensor generator already combines data and blinding rows together (RFC-0012), while its one-use wording and marginal-hiding argument do not supply the missing joint analysis. The new early uniform stream makes those coefficient uses explicit; moving it early does not by itself create or repair the joint-mask argument.

Two `Fp` components with fixed coefficients `1,i` do give a uniform **unconditioned single-row** `Fp2` mask. Reservation/burning of six rows and choosing designated roles before the seed do not establish conditional freshness after other combinations, extension-mask evaluations, columns, earlier proofs or linking openings have been revealed. One must write the joint linear system, prove its rank under the actual adaptive transcript and persistent budgets, and charge all bad-rank events to the stated statistical target. Include masks of combined mask polynomials, not only the message vectors.

The complete `docs/zk/simulator.md` required by RFC-0013 is absent. Reconcile the invariants and supply/review the joint simulator before approving the masked protocol. Renaming a pair's “role” alone is not that proof. No rank-failure probability or leakage attack is invented here, and the unmasked MCA theorem is not invalidated by this unknown.

### R3-4: carry current layout bounds and public grouping into the final allocation

**Required clarification/vector gate.** The Draft correctly distinguishes physical nonzero count `M` from allocated span `R`, and correctly preserves RFC-0047's `ν` and claim permutation. Segments use RFC-0046 order, zero gaps consume no coefficient, and an empty segment still has a chain-initial digest. The fixed advice pair is in the last segment; weights have none.

Make explicit that the final allocated span and the `2^32` preallocation maximum include appended advice blinding pairs. At a power-of-two boundary, appending them can change `ν`; at the maximum it can make a previously fitting data layout fail. Pin deterministic public slots/group assignments before the seed; do not let accidental equality of stage-derived column points choose or move committed mask rows. A grouping implementation must preserve this public layout rule.

Test zero/one/two padded axes and relation-to-layout claim permutations; `R` immediately below/at a power-of-two and at the maximum; empty segments; gaps and post-table blinding rows; the fixed pair's exact two skipped calls; weight reservation/linking roles; RFC-0047 GQA and `n=prompt_len+output_len≤max_context`. Alternatives 1's shorthand `log2 M` is not the current tensor randomness cost when `R≠M`: it is `ceil(log2 R)`.

### R3-5: the one-claim weight bound needs its current claim-pool premise

**Inherited integration dependency, not settled here.** The affine `θ_T v_T+q_T` proof is valid for one already-bound claim. RFC-0045's open Classes question explicitly routes one `ChannelParams` claim per reading site and leaves #105 to decide batching versus one combined claim; RFC-0037 INV-SCH-5 assumes one claim per weight type. The Draft's `L/q` weight term cannot silently close that open mapping. Verify the actual compiler-derived claim/group counts and any preceding reduction. Otherwise specify the multi-claim check/binding error and corresponding budgets/masks. The default numerical claims below are conditional on their stated `G,T` bounds, not proof of those bounds for every admitted claim list.

### R3-6: arithmetic evidence is sound at the defaults, but the tool is not the promised exact calculator

`terms.py` reproduces recorded `results/terms.txt` byte for byte. Its docstring still says the RFC pins independent per-claim coefficients; the current Draft pins powers `γ^{t+1}`. Its powers column multiplies by 257 for 256 claims, while the current polynomial degree is at most 256. This is conservative, not a security failure. Correct the provenance/shape wording, and do not call its `2^128` denominator exact Goldilocks arithmetic.

`research/tools/proximity.py` uses floating-point square roots/ceilings and `FIELD_BITS=128`; it is a planning tool, not an implementation of the Draft's required exact rational/integer checks. No normative Rust calculator or pinned protocol/conformance vector was validated by this review. Future acceptance must not treat reproduction of this Python output as those gates passing.

## Independent arithmetic (E0)

A separate scratch oracle imported no research/prover/verifier arithmetic. It used `Fraction`, integer comparisons for every radical ceiling, and the actual field order `q=(2^64−2^32+1)^2`; displayed logarithms used Decimal precision 90. All quoted theorem hypotheses above passed at all four defaults, including `k′≤n(1−γ)`. Full rational `E₀`, source hashes and checks are in [review3-summary.json](review3-summary.json).

| Profile | `A_thr` | `L` | Field bits | Query bits |
|---|---:|---:|---:|---:|
| live advice, `η=3/1000`, `τ=130` | 5,956,920 | 49 | 82.172989 | 64.202435 |
| NI advice, `η=3/1000`, `τ=163` | 5,956,949 | 49 | 82.172980 | 80.498859 |
| live weights, `η=1/200`, `τ=68` | 2,182,666 | 49 | 85.971145 | 64.079153 |
| NI weights, `η=1/200`, `τ=85` | 2,182,666 | 49 | 85.971145 | 80.098941 |

These are one-shot components before any permitted PoW credit, not a whole-proof verdict. With `G=16,T=256,L=49`, the exact degree-256 claim term has 110.385290 bits; the historical degree-257 bound has 110.379666 bits. Their shared 110.4 rounding is correct. The list bound is not `johnsonMu`; the reviewed pairwise theorem gives 49.

## ADV-P: timing, binding and retries

The seed fixes all coefficient streams before any stage message; transcript-global counter/ratchet and rejection sampling are inherited from RFC-0004. The received matrix, row roles and radius must already be bound by Round1/descriptors/reservations. Later stage slack cannot reroll these coefficients. Live session state permits one seed; noninteractive rerolls of the early stream require a new valid seed-PoW state. Independent live queries follow the fixed combined rows; NI query rerolls pay query PoW. Batching/stage challenges remain cheaply rerollable in both modes and receive no seed-PoW credit. This matches the new timing argument under the repository's random-oracle assumptions.

For each list element fixed before the batching challenge, false group claims produce a nonzero degree-at-most-`T_g` polynomial; the union over the at-most-49 list elements covers adaptive selection after batching. Hash binding and canonical shape checks remain separate premises. A wrong group polynomial intersects the exact proximity agreement set in at most `D` columns; shared queries cover any inconsistent group without multiplying that consistency term by `G`. Repeated draws are independent draws even when their answer is transmitted once.

Qualify the Draft's “every accepted proof ... exactly one” by the stated exceptional/binding/query events; existence/uniqueness is not an unconditional deterministic guarantee of finite queries. Its cross-proof `2^-89.8` calculation is a conditional single-pair query calculation, not a replacement for field/hash/batching events, reroll accounting or an epoch-wide union. RFC-0048's distinction remains: extracted `Fp` membership is not a proof of weight-domain fidelity, and it does not establish unrelated gadget integer bounds.

## Decisions and implementation gates retained

- **OQ-0021-4:** leave per-commitment versus per-proof SP-1 interpretation open. Printing an eleven-commitment union does not choose the target. Do not adopt 130/68 under a per-proof claim by inference.
- **OQ-0021-6:** fresh protocol identifiers and exact schema/parameter mapping remain unassigned. `/1` and accepted ZK `/2` cannot acquire a new schedule implicitly. The Draft's later rollout paragraph must be made consistent after that decision; parameter-set `/2` is not a transcript version.
- **ADV-V:** no complete joint simulator was supplied or approved; mask/reservation/linking roles must be included. This review does not grant SP-2.
- **SP-6/SP-8, secrets and tests:** no production/verifier implementation was changed or approved. Independent vectors, exact rejection tests, resource-limit checks, mask clearing, no-op-controlled mutations and the fresh protocol mapping remain implementation gates. No new sumcheck/gadget/unsafe/test-feature change was reviewed.

The original EXP-0005 result stays `inconclusive`. This new review supports scoped E0 theorem/number checks and identifies the remaining obligations; it does not promote the idea, approve the revised RFC or establish an H100/performance result.

## Follow-up on the parent's textual repairs

Lightweight diff review during the EXP-0018 exclusive timing slot; no builds, timings or arithmetic reruns. Reviewed revised Draft SHA-256 `c22f39f13eb114bc2f607621eeb15b311d02ed70f50cca21776a1d0fe66fcde9` and `terms.py` SHA-256 `7563e3a9b161a70001fbf3e162a7dcf34550f278519f3595baed74a1a3e5aedd`.

- **R3-1:** the new empty branch explicitly has `C=0`, public `d_j[0]` leaves, empty symbol vectors for the scheduled nonempty query set, zero combined rows and direct zero-evaluation checks. This is a coherent superseding rule with existing message shapes: `u` retains its required length and is all zero; the unmasked encoding mask remains empty. It adds no theorem hypothesis and needs no new open question merely to use these shapes. Clarify that “no claim-binding challenge is needed” does not omit any scheduled non-xi challenge: retain all gamma/theta calls to avoid shifting later transcript state. Only the xi stream has zero calls. Independent vectors and exact rejection checks remain required.
- **R3-2:** the explicit virtual fixed word and component-polynomial split correctly supply the reviewed bridge; no extra characteristic, dimension or distribution assumption was introduced. The masked fixed word need not be Frobenius-fixed. Textual proof gap addressed.
- **R3-3:** OQ-0021-7 and revised INV-PCS-8 now explicitly preserve the unknown joint simulator/rank obligation and inherited one-use tension. They correctly avoid claiming a leak or silently excluding rows. This finding is **recorded, not resolved**; it still blocks masked acceptance/implementation.
- **R3-4:** final `R`, appended pairs, maximum-layout check and public pre-seed slots are now explicit. Textual ambiguity addressed; conformance/boundary tests remain open.
- **R3-6:** the docstring now identifies the degree-257 table as a legacy conservative envelope and the current degree as 256. No numerical code changed. Also label the `1/q` zero-first-coefficient column historical: a fixed coefficient has no zero-first-draw event. The planning tool's floating arithmetic and absent implementation validation remain limitations.

R3-5's current ChannelParams claim-pool dependency, OQ-0021-4/6 decisions, the qualified-security wording and joint-ZK unknown remain. The revised document remains Draft; this follow-up is not full protocol approval.

## RFC-0049 delta review and preserved oracle, 2026-10-19

Reviewed accepted [RFC-0049](../../../docs/rfcs/RFC-0049-weight-proximity.md), SHA-256 `bf0c7a8c4c6de02d990331f937f82fe2274744543314bb27f34d60e592750043`, through main `2a35a0c1cd023e4794bb6fa8b4dd17d7df293b23`, against Draft SHA-256 `41585a8ce96e8b77b68dd91aa47cfa69ded051dc84b118c1c39d70c8e9d2ff02`. This delta supersedes the earlier follow-up's approval of retaining empty query/zero-proximity placeholders: that rule would now conflict with accepted main. No Draft file was edited by this reviewer.

### Exact corrections needed in the Draft

1. **Timing/generator distinction, including Summary and schedule.** Replace the universal “tensor squeezed after the stages” comparison with:

   > RFC-0049 already squeezes the weights' tensor proximity points immediately after the seed in `tessera-protocol/1`. This proposal replaces that early tensor generator, indexed by committed-row positions, with an early independent free-row coefficient stream. The advice's coefficients move from row 7 to the early stream. RFC-0049 leaves the masked protocol's timing to its separate design; this Draft's masked early stream remains a new schedule subject to OQ-0021-6/7. Existing protocol identifiers retain their accepted meanings.

   The same distinction must appear in Motivation, Transcript schedule, Grinding, Alternatives 6, Drawbacks and Migration. Pin the new expanded advice/weight stream order and matrix-id set in conformance vectors; do not infer it from “at the same place.” The accepted early weights already have the seed-PoW retry property; it is not a newly obtained protocol-1 timing gain. Main's residual generic row-7/grinding prose does not override RFC-0049's specific protocol-1 rule.

2. **Empty matrix branch and test.** Replace serialized empty-column/zero-proximity placeholders with:

   > For an unmasked weight matrix with `M=0`, retain RFC-0049's omission rules: no proximity row, no xi calls and no queries row. Do not serialize a placeholder zero proximity row or a queried empty-column answer component. Its public root still uses RFC-0040's `d_j[0]` leaves. Other message components, active matrix-id sets, derived zero claims and batching calls follow the protocol's explicit RFC-0036 mapping; this omission does not invent an additional evaluation claim or challenge. OQ-0021-6 must pin that mapping. A masked matrix with reserved committed blinding rows is not the `M=0` case.

   RFC-0036 already derives `weight_combined` entries from matrices with claims. Do not silently make its canonical opening/query/answer sets disagree or choose a new enum/schema layout here. Add empty/nonempty `ChannelParams`, `M=1` (zero free calls but a real proximity row), next-label and malformed-extra-component vectors. If an explicit field-layout decision is needed beyond these omission rules, keep it within OQ-0021-6 rather than treating the earlier placeholder branch as established main.

3. **`R` versus committed positions.** Qualify the tensor-factor paragraph:

   > The advice tensor uses allocated span `R` and `ceil(log2 R)` row coordinates under RFC-0047. In unmasked weights RFC-0049 instead uses `ceil(log2 m_T)`, with position `j` naming the `j`-th committed physical row, not its structural index. The new free stream traverses that committed order while skipping only its designated fixed row; it does not revive structural-index gaps. The masked physical-row count separately includes its committed epoch blinders.

4. **Scope/reference and numbers.** Add RFC-0049 to the affected sections and references, explicitly superseding its protocol-1 tensor-generator rule only for the newly assigned schedule. Preserve its empty omission and committed-position semantics. Label the existing `P_W=65536,B_W=4096` weight rows/results as masked planning profiles, not exact protocol-1 figures. RFC-0049's operational protocol-1 weight mask is zero and it commits no `B_W` rows even though those parameter values exist in its bound layout. An exact calculator must derive operational degree/row count by protocol. The historical values are unchanged, not newly measured/recalibrated protocol-1 results. Calling the masked degree an upper-degree planning envelope does not replace a proof/vector for the actual unmasked protocol.

RFC-0049 explicitly leaves multiple claims on one weight matrix to [#386](https://github.com/AbdelStark/tessera/issues/386); therefore R3-5 remains open. Its unmasked early timing does not supply the missing joint masked simulator in OQ-0021-7. OQ-0021-4/6 remain decisions. No new theorem assumption is needed merely to adopt RFC-0049's early timing and omission rules.

### Reproducible independent arithmetic and OQ-0021-4 correction

The former scratch oracle is now preserved as [review3_exact.py](review3_exact.py), with SPDX header, no Tessera imports, exact ceiling checks and CLI/self-tests. It reproduces every field of the four-profile `arithmetic` object in [review3-summary.json](review3-summary.json). It labels the weight profiles as masked planning inputs and does not change the historical EXP-0005 criteria, code or results.

```sh
python3 -I research/experiments/EXP-0005-list-decoding-extractor/review3_exact.py --self-test --check-summary research/experiments/EXP-0005-list-decoding-extractor/review3-summary.json
python3 -I research/experiments/EXP-0005-list-decoding-extractor/review3_exact.py --query-union
```

For a **query-only** per-proof interpretation, the relevant joint sum is `alpha_A^tau_A + 10 alpha_W^tau_W`, not a separate approximate bit increment to each matrix. With the same live masked-planning inputs as the historical table:

| Advice / weight queries | Joint query bits | Versus `2^-64` |
|---|---:|---|
| 130 / 68 | 60.630503 | fails |
| 138 / 71 | 63.524715 | fails |
| 138 / 72 | 64.414282 | passes |

The old OQ-0021-4 phrase “about eight advice columns and three weight columns” is insufficient: 138/71 fails even before adding other errors. These pass/fail statements are proved in the oracle by exact Fraction powers of rational square-root enclosures at 160 bits; Decimal precision 90 is used only to display bits. The upper bound for 138/72 is about `0.750393 × 2^-64`, leaving query-only margin. This is an example, **not** a chosen SP-1 interpretation, minimal allocation, unmasked calibration or full-error budget. Replace the approximate increment with an explicit joint-sum requirement; OQ-0021-4 remains open and all additional terms must be included for any adopted total target.

Self-tests and comparison with the persisted default arithmetic passed. No performance claim or production implementation validation was added.
