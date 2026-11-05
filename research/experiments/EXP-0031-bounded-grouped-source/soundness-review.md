# EXP-0031 and RFC-0077 review

Two adversarial readings by the implementing agent, not independent human review.
Scope: finite synthetic prototype and protocol-preserving runtime contract.
No production verifier/acceptance, parameter, source-pin or admission change.

## Preconditions and evidence

| Requirement | Evidence and decision |
|---|---|
| Source identity includes site, layer, role and full padded index | `fixture.rs::datum`, `adapter.rs::group`; current resident constructors and independent `audit.py::audit` compare all 43,372 finite index records and ordered value digests. The abstract contract proves projection/insertion under a canonical callback; a production callback remains unimplemented. |
| Public-only geometry, exceptional groups and slot gaps | Frozen fixtures and independent full-coordinate enumeration cover 42 cases, six impossible splits, held roles, prologue/epilogue, value/no-value columns and forests. No observed secret sparsity chooses capacity or mode. |
| True integers and canonical arithmetic | Synthetic term values are integers 0..3; key forms stay in 0..15 and values in 3..33. `from_i64` and existing field kernels are used. SplitMix wrapping is the explicitly public fixture generator, not witness/resource arithmetic. Size helpers use checked addition/multiplication and addressability guards. |
| Complete caller geometry for component bounds | Message dimension now includes every group tree height, public table height and merged polynomial dimension. Independent audit rejects under-declaration even when the observed allocation still fits. Tensor/forest counts include the complete merged lists. |
| Restricted split and original point transport | `merge_probe.rs` independently projects original read coordinates, evaluates selected blocks, then compares against materialized original-polynomial evaluations outside the candidate counter. Both current merge kernels agree at all 285 legal cuts. |
| Group pass cannot hide unrestricted original regeneration | Explicit factory call counter and eager-materialization mutant. A selected block with an empty group-variable set is legitimately whole and remains charged. The counter does not prove arbitrary future callbacks contain no hidden work. |
| Every original P2b polynomial is regenerated once | Per-polynomial callback counts and unrestricted-builder delta are exact at every cut; original source/copy work is added beside current MergeMemory. |
| Retained/overlapping owners are not confused | Source index records, earlier records, group layout/input work, kept tables, caller blocks, original P2b source/copies, engine state and outputs have explicit prototype bills and allocation checks. The phase inventory marks factory/cache/replay/native/opening/publication work still owed; unknown is never production zero. |
| Root relation, transcript and verifier independence stay unchanged | Production base is archived from `646f7ad`; reference overlay exposes only grouped constructors/module visibility with a doc comment. `run.py::verify_registration` rejects production/verifier source drift. No verifier dependencies or code are added. |
| No new Fiat-Shamir or cryptographic claim | The recording transcript is an allocation-free deterministic fixture, including reduced hash words. It is not production TCE or exactly uniform production randomness. Equality here is not NI/Live acceptance; the RFC requires unchanged production schedule/budget checks. |
| Clearing and redaction | No real model, token, seed or statistics input is read. Derived prototype buffers use SecretBuf; the existing merge API receives Secret-owned copies. Copy-only field slices are immediately wrapped without an intervening user callback; both the source and returned copy are cleared by their owners. Logs contain public counts, identities and digests, with no payload dumps. |
| Production refusal and robustness | Both grouped physical guards are unchanged. No experimental assertion/panic parser is exposed as a production verifier. The proposed route must reject unknown owners, overflow and insufficient complete budgets before P1; full-route controls remain required. |

## Findings resolved before promotion

1. Synthetic sites were initially not in canonical node order. The current
   constructor refused; ordering was corrected and the failure log retained.
2. Initial group split preparation materialized each original polynomial. It now
   evaluates selected blocks; a new compiled fault catches eager unrestricted
   materialization even with equal values. Original polynomials remain available
   only to the reference oracle and P2b's unrestricted builder.
3. A blanket prohibition on whole sources would break empty-group-variable roles.
   The draft explicitly retains and charges those canonical whole selected blocks.
4. One message-bound call used only merged polynomial dimensions and missed a
   larger lookup-tree height. Public geometry derivation and an independent
   under-declaration control now enforce the component's complete precondition.
5. Resource grouping can affect existing soundness accounting. The draft requires
   the existing budget checks for the chosen grouping and freezes counts,
   parameters, identities and grouping before P1; no mid-proof memory retry.

All twelve registered compiled faults and both added review regressions fail their
relevant Rust/Python assertions. Original/restored controls pass; compile errors
are never counted as detection. Four faults need the independent Python audit,
including shared-factory, schema and identity errors. These are finite source and
accounting controls, not fourteen proof-forgery demonstrations.

## Decision

Pass for the finite experiment and abstract contract scope. Production factory
conformance, complete phase admission, real-model fidelity, independent acceptance
and whole-process time/memory are unproven and remain explicit implementation or
qualification gates. No global soundness, hiding, performance or frontier-model
approval is claimed by this review.
