# Blueprint for a separate verifier model

Status: proposed research plan for [#392](https://github.com/AbdelStark/tessera/issues/392),
checked against source revision `7f08aed075d118d1cf0dc53dab5f810d34f07e21`.
This document contains theorem **statements and open obligations**, not Lean proofs.
No statement below is kernel checked, no Rust-to-Lean refinement is proved, and no
protocol-level soundness or zero-knowledge theorem follows from the listed tests.
Maintainer review of the scope and proof-status wording is pending.

The proposed first route is a separate mathematical Lean model, independently written
from the normative specification, with conformance tests against the verifier. This
prepares [RFC-0019 OQ-0019-1](../rfcs/RFC-0019-verifier.md#open-questions); it does not
resolve that question. The owner still decides after [#186](https://github.com/AbdelStark/tessera/issues/186)
provides `docs/verifier/spec.md`. This blueprint neither replaces that predicate-per-check
specification nor adds a release gate under NG-7. All code correspondences below are
**provisional**, including the ones whose function names have been checked.

## Scope and proof-status discipline

The first model covers three surfaces: interleaved commitments, the batched LogUp/tree
reduction, and the acceptance/transcript schedule. It starts with Protocol One. Protocol
Two gets separate statement signatures and explicit unproved obligations, rather than
an assumption that its masks preserve the Protocol One proof.

Use these statuses in a future proof inventory:

| Status | Required evidence |
|---|---|
| Statement | Quantifiers, definitions, dependencies and imported assumptions are written down; no proof credit. All `BP-*` entries here have this status. |
| Checked lemma | Pinned Lean/toolchain/library revisions; kernel-checked term; audited transitive axioms, including `#print axioms`; no `sorryAx` or unlisted project axiom. |
| Refinement evidence | Saved independent vectors or differential runs, with source/model revisions and counterexamples. Finite tests do not prove refinement. |
| Refinement theorem | A checked relation between all supported Rust inputs/results and the model, including failure behavior, with translation semantics and trusted components explicit. |
| Protocol theorem | Composition of the checked reductions, commitment assumptions and challenge model, with all remaining claims discharged and a reviewed failure budget. A lemma about a round is insufficient. |

A future file that uses `sorry`, `admit` or an assumed soundness axiom remains a statement
at that boundary. Checking that a conditional theorem elaborates does not discharge its
assumptions. Lean's [axioms documentation](https://lean-lang.org/theorem_proving_in_lean4/Axioms-and-Computation/)
explains why the axiom inventory is part of the result.

## Shared definitions and boundaries

Define from [RFC-0002](../rfcs/RFC-0002-field-arithmetic.md),
[RFC-0003](../rfcs/RFC-0003-canonical-encoding.md),
[RFC-0004](../rfcs/RFC-0004-hashing-and-transcript.md) and
[RFC-0010](../rfcs/RFC-0010-proof-schedule.md):

- `Fp = ZMod p`, `p = 2^64 − 2^32 + 1`, and the specified quadratic extension `Fp2`.
  Field laws require primality and irreducibility lemmas; mapping Rust limbs to these
  objects requires canonical-value and operation-refinement obligations.
- A finite table's multilinear extension, low coordinate first, and `eq`. Keep row-major
  tensor coordinates, stacked high axes, front padding and per-axis padding distinct.
- Canonical TCE bytes and typed decoding, including exact lengths and resource limits.
  Use bytes at the transcript boundary, not a serialization-free abstract record.
- A public session descriptor containing protocol and parameter-set identifiers, claim
  list and statement, roots/epochs, derived layouts and ordered instances. Derive geometry
  and ranges independently (SP-7); do not accept a constraint system supplied by the prover.
- A pending claim with source identity, point, value and public coordinate map. A multiset
  of these debts is part of the verifier state. Equality of values alone cannot merge debts.
- An ordered trace of `Absorb(label, bytes)`, challenge draws and verdicts. Model rejection,
  inconclusive resource/internal failures and acceptance separately (SP-8, RFC-0019).

The abstract field, serialization, hash and challenge definitions are not implementation
correctness proofs. The executable oracle must return the same first error and remaining
debts, not merely the same Boolean result.

## Surface 1: interleaved commitment checks

Normative inputs: [RFC-0011](../rfcs/RFC-0011-advice-commitment.md),
[RFC-0012](../rfcs/RFC-0012-weight-commitment.md),
[RFC-0031](../rfcs/RFC-0031-commitment-id.md),
[RFC-0047](../rfcs/RFC-0047-advice-order.md) and
[RFC-0049](../rfcs/RFC-0049-weight-proximity.md).
Read each with its accepted amendments. A Draft RFC is not silently imported as an
accepted implementation contract. In particular, coefficient-form experiments and a
different generator require their own format binding and protocol identifiers.

Define a descriptor-derived ordered row set, subgroup `H`, evaluation domain, encoded
row `enc(u) + Z_H M`, canonical column digest/Merkle leaf and a group of original claims.
For group `g`, define the coefficients and the combined row exactly as RFC-0011's opening
protocol specifies, including powers beginning at `gamma^1`. Define the proximity vector
as the tensor `eq(xi*, row)`, not independent randomness for every row. Weight layouts,
matrix order and channel-coordinate maps are separate definitions from advice layouts.

| Statement | Proposed conclusion and dependencies | Provisional predicate correspondence |
|---|---|---|
| `BP-PCS-DESCRIPTOR` (INV-PCS-1, INV-PCS-3, INV-TCE-1) | Equal validated public descriptors produce equal ordered rows, coefficient maps and leaf bytes. A root authenticated for another commitment ID/layout cannot serve this one except via the stated hash-binding assumption. Depends on canonical encoding, exact row count, skipped-zero-row semantics and domain separation. | [advice/opening.rs](../../verifier/crates/tv-core/src/advice/opening.rs) `Commitment::bind`; [weights.rs](../../verifier/crates/tv-core/src/weights.rs) `derive`, `check_public`; [opening.rs](../../verifier/crates/tv-core/src/opening.rs) `Evaluated::check_columns`; `tv-hash` Merkle predicates. |
| `BP-PCS-EVALUATION` (RFC-0011 opening steps 1–6, INV-FS-1) | If the original claims are fixed before `gamma`, a false group equality survives its random combination only under the appropriate nonzero-polynomial root bound. Prove the coefficient polynomial is nonzero before using that bound. The equality leaves a debt to authenticate the combined rows. Depends on fixed claim list/order, exact shape and independently derived points. | [advice/opening.rs](../../verifier/crates/tv-core/src/advice/opening.rs) `Commitment::prepare`, `Opening::check_evaluations`; [weights.rs](../../verifier/crates/tv-core/src/weights.rs) `check_evaluations`; [opening.rs](../../verifier/crates/tv-core/src/opening.rs) `check_evaluations`. |
| `BP-PCS-COLUMNS` (RFC-0011 opening steps 7–8, RFC-0012 opening) | On every authenticated queried column, all advertised combinations equal the re-encoding of their combined rows plus their mask polynomials. Duplicate draws are checked once as data, while the probability space retains every independent draw with replacement. Depends on `BP-PCS-DESCRIPTOR`, exact sorted-distinct answer indices, canonical multiproofs and fixed combined rows before queries. | [opening.rs](../../verifier/crates/tv-core/src/opening.rs), [advice/opening.rs](../../verifier/crates/tv-core/src/advice/opening.rs), [weights.rs](../../verifier/crates/tv-core/src/weights.rs): the respective `Evaluated::check_columns` methods. |
| `BP-PCS-BINDING` (RFC-0011 soundness, RFC-0014 budget) | Except with the reviewed proximity/list and batching error, accepted groups bind to the same decoded committed rows. Depends on the tensor-generator proximity theorem, its line-gap terms and parameter restrictions; unique decoding alone does not prove interleaved proximity or group binding. | The preceding checks composed with [model_openings.rs](../../verifier/crates/tv-core/src/model_openings.rs) `Claims::derive`, `advice_claims`, `weight_claims`, and the top-level driver. This is a mathematical obligation, not a theorem implemented by one Rust function. |

Imported assumptions to state explicitly: collision/binding security of the specified hash
construction; the RS/interleaved proximity theorem **for the tensor generator actually
used**, at the specified rate and distance; the field root bound; fixed commitments and
claims before their batching challenges; fresh query randomness or the separately justified
Fiat–Shamir model. Audit the exact theorem hypotheses and their RFC-0014 calculator terms
before importing a numerical bound. No new bound is supplied here.

Protocol Two additionally needs the exact blinding and mask-row coefficient maps of
RFC-0013, including once-per-piece mask contributions at the claim's own point. Weight
epoch hiding under adaptive verifiers, aborts, budget persistence and refresh/linking are
not consequences of `BP-PCS-BINDING`. The joint hiding obligation in
[#394](https://github.com/AbdelStark/tessera/issues/394) and the requirements of
[#145](https://github.com/AbdelStark/tessera/issues/145) remain open. No theorem about a
single commitment proves an epoch budget or makes a restored state file safe.

## Surface 2: batched LogUp and tree reduction

Normative inputs: [RFC-0008](../rfcs/RFC-0008-logup-and-tables.md),
[RFC-0010](../rfcs/RFC-0010-proof-schedule.md),
[RFC-0037](../rfcs/RFC-0037-stage-schedule.md),
[RFC-0045](../rfcs/RFC-0045-leaf-layout.md) and, for grouped proofs,
[RFC-0054](../rfcs/RFC-0054-token-groups.md).

Define table tuples and compression, integer multiplicities below the characteristic,
the padded lookup/table leaves and every tree's height and join level. Table padding
copies row 0; it does not introduce a freely weighted new tuple. A fraction node is
`(nL*dR + nR*dL, dL*dR)`, with pair bit lowest. Product forests and both sides of every
lookup instance join the ordered batch; public forest padding roots are included.

| Statement | Proposed conclusion and dependencies | Provisional predicate correspondence |
|---|---|---|
| `BP-LU-IDENTITY` (RFC-0008 relation, RFC-0009 Lemma 0) | For roots computed from the canonical padded leaves by the fraction-node rule, nonzero root denominators and `na*dt + nt*da = 0` imply the padded rational identity. Authenticating arbitrary advertised roots to those leaves is a separate `BP-TREE-STEP`/`BP-LEAF-COVERAGE` obligation. With true integer keys/values, bounded multiplicities and tuples/commitments fixed before `alpha,beta`, a false membership identity has the RFC-0014-budgeted error. Separate compression collisions from the rational-identity root bound. | [logup.rs](../../verifier/crates/tv-core/src/logup.rs) `check_roots`, `check_leaves`; independent range/claim validation supplies the integer premises. These functions alone do not prove the premises. |
| `BP-TREE-STEP` (RFC-0010 tree levels, INV-FS-1) | Each accepted level reduces all incoming claims to the stated next-level claims except for the sumcheck/batching/line error. Denominator and numerator combination uses `lambda`, across-instance combination uses `gamma`. All smaller instances contribute `gamma^i * 2^(nmax-ni) * ci`; final relation evaluations use the suffix point and no extra padding factor. Depends on degree bounds, exact instance order/count, fixed entering claims and the final relation equality. | [sumcheck.rs](../../verifier/crates/tv-core/src/sumcheck.rs) `reduce_in`, `verify_batch_in`; [tree.rs](../../verifier/crates/tv-core/src/tree.rs) `verify_level`, `verify_product_level`. |
| `BP-LEAF-COVERAGE` (RFC-0045, RFC-0010 termination, SP-3) | Reducing every scheduled tree and checking its leaves produces exactly the required lookup-key, value, multiplicity and forest-leaf debts. Consuming all these debts in reductions/openings binds the rational identity to the intended model constraints. Depends on a schedule enumeration theorem, correct public padding and `BP-TREE-STEP`, not just an honest test of one level. | [leaf.rs](../../verifier/crates/tv-core/src/leaf.rs) `Schedule::verify`; [stages.rs](../../verifier/crates/tv-core/src/stages.rs) `verify_composed_bound`; [model_openings.rs](../../verifier/crates/tv-core/src/model_openings.rs) `Claims::derive`. |
| `BP-GROUP-MERGE` (RFC-0054, INV-FS-1) | Every group's leaf debt and the shared table-tree debts enter the canonical merge, at their original source points, before the merged claims are consumed. A merge that forgets a group or a forest leaf is insufficient. Depends on `BP-LEAF-COVERAGE`, grouping fixed by Round1, the merge polynomial and its separate soundness terms. | [driver.rs](../../verifier/crates/tv-core/src/driver.rs) grouped leaf path and [stages.rs](../../verifier/crates/tv-core/src/stages.rs); refine individual group/merge predicates after #186 enumerates them. |

The sumcheck round model reconstructs omitted `p(1)` from the running claim. Therefore
an algebraically corrupted, well-shaped round need not fail at that round; the terminal
relation and outstanding opening debts are essential. A statement that concludes from
`reduce_in` alone that the original relation holds is false. Its `Reduction`, like tree,
stage and opening results, carries work still owed.

Protocol Two has a **different identity statement**: RFC-0032's masked LogUp identity,
masked tree levels and pair-bit-spared extensions, RFC-0068's layer sumcheck, RFC-0013's
virtual folds and RFC-0070's mask-claim batching. Public roots from `check_roots` cannot
stand in for that proof. The top-level driver at the pinned revision is Protocol One;
`protocol2_sumcheck` tests degree support, not a complete masked driver. Full Protocol
Two correspondence, fresh/independent mask functionals, degenerate points and the
simulator are explicit open obligations under #145, #148, #149, #154 and #159.

## Surface 3: session schedule and acceptance

Normative inputs: [RFC-0014](../rfcs/RFC-0014-interaction-and-soundness.md),
[RFC-0019](../rfcs/RFC-0019-verifier.md),
[RFC-0036](../rfcs/RFC-0036-protocol-messages.md) and
[RFC-0004](../rfcs/RFC-0004-hashing-and-transcript.md).

Define a state machine whose state contains validated public inputs, transcript bytes,
ordered remaining instances, outstanding claims and resource/session state. Its `Accept`
state requires no outstanding debt, including external anchor/scoring bindings. Model
live verifier entropy separately from NI transcript output, rejection sampling, grinding
and retry limits. A successful low-level check cannot erase a debt it has merely moved.

| Statement | Proposed conclusion and dependencies | Provisional predicate correspondence |
|---|---|---|
| `BP-FS-TRACE` (INV-FS-1..4, INV-TCE-1) | All fixed public context and each dependent prover message are canonically absorbed, under the specified unique labels, before the relevant challenge. In particular, Round1 fixes the statement/roots before LogUp challenges and combined rows precede column queries. Trace equality requires byte equality and order, not just label presence. | [driver.rs](../../verifier/crates/tv-core/src/driver.rs) `verify_non_interactive`, `verify_non_interactive_bytes`, `open_live`; [tv-hash transcript.rs](../../verifier/crates/tv-hash/src/transcript.rs) `Transcript::new`, `absorb`, `challenge_fp2`, `challenge_indices`, `check_grind`. |
| `BP-ACCEPT-DEBTS` (SP-3, SP-7, RFC-0010 termination) | Acceptance implies the public schedule was derived independently and every scheduled relation, final claim and opening/external debt was checked exactly where required. Combine `BP-LEAF-COVERAGE`, `BP-GROUP-MERGE` when applicable, the reduction-stage analogue of `BP-TREE-STEP`, and `BP-PCS-BINDING`. A `must_use` annotation is helpful but is not a proof of whole-program consumption. | [schedule.rs](../../verifier/crates/tv-core/src/schedule.rs), [stages.rs](../../verifier/crates/tv-core/src/stages.rs) `verify_composed_bound`, [model_openings.rs](../../verifier/crates/tv-core/src/model_openings.rs), [driver.rs](../../verifier/crates/tv-core/src/driver.rs). |
| `BP-VERDICT-REFINEMENT` (SP-8, RFC-0019 check order) | For related bytes, public configuration and session state, Rust and the executable model produce the same first typed verdict, report and live state transition, or a specified inconclusive resource result. Include malformed inputs, terminal session failures, bounds before allocation and the distinction between prover faults and verifier faults. | [driver.rs](../../verifier/crates/tv-core/src/driver.rs) byte/typed entry points, [driver/live.rs](../../verifier/crates/tv-core/src/driver/live.rs), [tv-hash transcript.rs](../../verifier/crates/tv-hash/src/transcript.rs). |
| `BP-COMPOSITION` (RFC-0014 soundness budget, SP-3) | Under the explicitly stated commitment and challenge assumptions, a false statement reaches Accept only within a reviewed composed failure bound. Depends on every preceding statement, integer/gadget uniqueness and external binding lemmas outside this initial scope. Interactive soundness does not establish NI Fiat–Shamir security. | Whole driver plus the independent soundness calculator; #186 must supply the complete predicate/code and lemma inventory before this claim can be closed. |

Record a symbolic error ledger per event: tuple compression, LogUp identities, root
claims, sumcheck rounds/batching/line challenges, split/merge claims, group evaluation,
tensor proximity and queries, and hash/Fiat–Shamir assumptions. Its sizes come from the
validated schedule and parameter bundle, with grouping and protocol-specific additions.
Reconcile it with RFC-0014's calculator, including distinct grinding/retry budgets and
one-shot versus union rows. Do not multiply or union incompatible models, use a
headline security target as a lemma, or turn an empirical cheating frequency into a proof.

## Negative cases and present evidence

These are **sensitivity evidence and future proof obligations**. A named existing test
does not establish any universal statement above. No Lean differential harness exists
yet. The [testing strategy](../spec/07-testing-strategy.md#4-soundness-and-uniqueness-tests)
and [recorded driver corpus](../../verifier/crates/tv-core/tests/fixtures/driver/README.md)
are the starting corpus; keep their expected first verdicts when adding model runs.

| Missing premise or wrong reading | Existing evidence at the pinned revision | Remaining case / obligation |
|---|---|---|
| Statement digest/roots changed before LogUp challenges (`BP-FS-TRACE`) | [driver.rs tests](../../verifier/crates/tv-core/tests/driver.rs) `each_statement_field_and_version_preserve_their_exact_first_error`; [driver/corpus.rs](../../verifier/crates/tv-core/src/driver/corpus.rs) `every_value_of_the_recorded_live_session_is_bound`. | A model/Rust trace comparison on a valid, re-bound altered public statement must locate its absorption before `logup/alpha,beta`. An early statement-mismatch rejection alone proves no challenge dependency. Lower-level transcript sensitivity is insufficient. |
| Dropped/repeated/converted tree instance (`BP-LEAF-COVERAGE`) | [logup_robustness.rs](../../verifier/crates/tv-core/tests/logup_robustness.rs) `instances_present_or_absent_at_a_level_get_the_predicted_error`, also seeded. | Prove top-level enumeration includes every required instance. A caller omitting an instance from both its schedule and proof is not covered by changing only a supplied level list. |
| False final relation or unconsumed debt (`BP-TREE-STEP`, `BP-ACCEPT-DEBTS`) | [soundness.rs](../../verifier/crates/tv-core/tests/soundness.rs) `a_lie_in_any_round_is_caught_unless_the_final_evaluations_are_forged`; [mutations.rs](../../verifier/crates/tv-core/tests/mutations.rs) stage/tree mutations. | The test name's exception matters: forge final values too, then require rejection by their original-source openings. Prove conservation/consumption of every debt across the complete pipeline. |
| False claim matched by a forged combined row (`BP-PCS-COLUMNS`) | [opening_checks.rs](../../verifier/crates/tv-core/tests/opening_checks.rs) `a_false_claim_with_a_combined_row_to_match_fails_the_column_check`; [opening_mutations.rs](../../verifier/crates/tv-core/tests/opening_mutations.rs) element mutations. | Prove re-encoding/column binding and tensor-generator proximity; finite column cases cannot establish the probability bound. |
| Another descriptor, row count or missing queried column (`BP-PCS-DESCRIPTOR`) | [weight_openings.rs](../../verifier/crates/tv-core/tests/weight_openings.rs) `public_artifacts_bind_root_epoch_and_every_layout_field_in_order`, `an_authenticated_proper_subset_does_not_answer_every_query`, `authenticated_and_consistent_columns_still_have_the_derived_row_count`. | Model all canonical layouts, matrix/channel coordinate maps, and skipped public rows. |
| Illegal lookup padding / false integer premise (`BP-LU-IDENTITY`) | [padding.rs](../../verifier/crates/tv-core/tests/padding.rs) `a_nonzero_value_on_a_padded_row_layer_or_head_is_rejected` checks the graph zero-check; it does **not** establish lookup row-0 padding. | Add a model/Rust case for a fresh `(0,0)` lookup padding row with nonzero multiplicity, and boundary multiplicity/packed-key wrap cases. Prove lookup leaf layout and RFC-0009 range/uniqueness premises independently of graph padding. |
| Group leaf claims consumed at the wrong point (`BP-GROUP-MERGE`) | [driver_grouped.rs](../../verifier/crates/tv-core/tests/driver_grouped.rs) `the_stages_accept_the_merged_claims_only_at_their_merged_points`, `each_grouped_check_rejects_with_its_own_reason`. | Model every group and shared table debt; prove no group/forest is omitted by public enumeration. |
| Reused/degenerate mask (`BP-PCS-BINDING`, Protocol Two) | [protocol2_sumcheck.rs](../../verifier/crates/tv-core/tests/protocol2_sumcheck.rs) covers degrees, lengths and protocol separation only. The R10 gate supplies bounded schedule rank evidence (#488). | No complete Protocol Two driver/refinement test here. Require reused-row and dependent-functional negative cases, Boolean/pair-bit degeneracies, full revealed-point coverage, epoch/abort analysis and the #145 simulator obligations before any ZK conclusion. |
| Late public binding, altered label or biased draw (`BP-FS-TRACE`) | [tv-hash tests](../../verifier/crates/tv-hash/tests) and [driver/corpus.rs](../../verifier/crates/tv-core/src/driver/corpus.rs) provide transcript/corpus evidence. | Prove the exact rejection-sampling distribution and retry behavior, event ordering and protocol separation, then state the NI transform assumptions independently. A hash test is no random-oracle theorem. |

## Refinement work plan and independence

1. Write the above definitions and pure predicates in a separate, pinned Lean project
   outside both production dependency graphs. Begin with `eq`, front padding, fraction
   nodes and one sumcheck terminal check. Prove these leaf lemmas before composition.
2. Specify canonical input/result/trace fixtures from RFCs. Use independent mathematical
   oracles and preserve the existing expected rejection reasons. Feed identical fixtures
   to Lean's executable predicates and `tv-*` from an external conformance harness.
   Store source revisions, toolchain, parameters, seed, supported modes and failures.
3. Retain a manifest connecting each `BP-*` statement to its definitions, Lean theorem
   name, transitive axioms, Rust predicate, tests and unresolved obligations. Validate
   each negative case with a targeted faulty model/implementation and a passing control.
   A finite manifest/corpus cannot promote itself to a refinement theorem.
4. Once #186 exists, replace every provisional correspondence with its predicate ID
   and exact code location. Review all public `tv-core` checks there; this selected
   three-surface inventory is deliberately not that exhaustive specification.
5. Evaluate extraction only for a leaf function after its semantics are fixed. Pin
   [Aeneas](https://github.com/AeneasVerif/aeneas) or
   [hax](https://github.com/pereq-dev/hax), frontend and Lean backend revisions; record
   the supported Rust subset and any opaque primitives. Audit integer widths/overflow,
   panics, allocation/errors, slices/indices, loops/termination, byte order and hash/FFI
   models. A verified theorem about extracted code still needs translation fidelity and
   deployment correspondence; a tool's ability to emit Lean proves neither.

SP-6 and DEP-1..3 remain the boundary: no prover/research code, generated implementation
or dependency enters `verifier/`. The separate model follows the specification rather
than copying prover algorithms. Data fixtures may be shared through `spec/vectors/`;
an external harness may compare outputs without making either runtime depend on the
other. No production dependency, proof format, accepted parameter or verdict changes
are proposed here. Changed tier-A semantics would require a superseding RFC.

## Review and exit conditions

The blueprint is ready for maintainer scope review when each statement has its premises,
dependencies, provisional predicate and negative/open case, and the links resolve. That
review is distinct from #145's simulator approval, #186's written specification review
and #187's external security review. Until maintainer review confirms the route and
wording, #392 remains open and OQ-0019-1 remains unresolved. Later checked lemmas,
refinement evidence and composition proofs must be reported with their own status and
revisions; they must not be inferred from approval of this plan.

Background: [SUR-0012](../../research/surveys/SUR-0012-verifier.md) is a dated literature
survey, not evidence that Tessera is verified. The Lean, Aeneas and hax primary links
above were checked on 2026-10-28 only for the relevant proof/translation boundaries.
This blueprint imports no claim about their current coverage of Tessera.
