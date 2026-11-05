# RFC-0077: Bounded Grouped Checkpoint Sources

- Status: Accepted
- Authors: @AbdelStark
- Created: 2026-11-05
- Target milestone: v1.0, after complete implementation and qualification
- Proposes to amend: [RFC-0018](RFC-0018-gpu-prover-runtime.md#memory-scheduler), checked checkpoint fallback and its grouped P2/P2b ownership; [RFC-0054](RFC-0054-token-groups.md#pass-structure), the CPU source interface and lifetime accounting
- Scope: protocol/1 CPU planning/source interfaces; no tier-A schema, label, parameter, relation or verdict change; protocol/2 grouping remains excluded
- Acceptance: recorded by the #687 merge under explicit maintainer authorization for autonomous project work; implementing-agent review, no independent human approval claimed
- Origin: [#687](https://github.com/AbdelStark/tessera/issues/687), [IDEA-0042](../../research/ideas/IDEA-0042-bounded-grouped-source.md)

## Summary

Define the canonical source and complete lifetime contract a grouped checked
checkpoint route must satisfy. It may not retain the complete model leaf witness
merely because its proof trees are grouped. It must regenerate selected group
sources, keep all source/kernel/output overlap priced, and regenerate original
polynomials at P2b under the existing merge schedule.

This RFC specifies the conditions for a future grouped extension to the checked
fallback. Both production grouped admission guards stay in place until a
complete implementation passes the gates below. Finite synthetic evidence is
not a complete factory, a production memory bound or a real-model result.

## Motivation

The current CPU builds `ModelWitness` before its grouped branch. Its padded source
columns and forest leaves remain live while group columns/shares are constructed.
Checkpoint-backed tensor access can also retain complete nonweight tensors.
Reducing tree sizes or pricing only returned messages cannot remove these owners.
RFC-0018's checked physical fallback is explicitly ungrouped-only, so admitting
grouped proofs requires an explicit amendment and complete composition.

## Goals

- Preserve exactly the existing canonical integer statement, grouped columns,
  multiplicities, forest leaves, splits, merged points and original openings.
- Bound every group/source callback and all phase overlap from public geometry
  before P1, refusing unsupported or incomplete candidates.
- Preserve the resident route as first choice and oracle, one proof unit per
  proof, and the existing P2/table/P2b challenge schedule.
- Separate source/kernel evidence, full production admission and real-model
  qualification instead of treating one as proof of another.

## Non-Goals

- Protocol/2 grouping, a new hiding/security claim, new PCS or changed parameters.
- GPU throughput, longer-sequence chunk linking, or a claim every model/context
  fits. A full original polynomial required by the current merge may still be
  infeasible; it must refuse before P1.
- Treating the finite prototype as a production checkpoint/gadget adapter or
  assuming a checkpoint cache is bounded without pricing it.

## Proposed Design

### Canonical source identity

Each public source descriptor identifies its site, real layer, tensor role,
original padded shape, term projection and token/layer membership. Request slots
are [RFC-0071](RFC-0071-request-slots.md)'s: high token bits name the request and
low bits its position. No source identity may collapse to a position alone.
Padding includes slot gaps, masked queries/keys, padded channels/layers and all
node-specific virtual/public roles.

Descriptors must be derived from checked public lowering, layer/leaf/native and
advice layouts. The source factory is an internal canonical implementation,
not an unverified caller declaration of geometry or bytes. Inconsistent read
agreement, identities or unsupported shapes refuse before P1.

### Scoped source factory

The future CPU factory must provide two scoped operations:

1. Lend the current group's selected canonical tensor/lookup/forest sources,
   with public construction, retained and callback/replay peak bounds.
2. After the merge's first table rounds, regenerate one original polynomial in
   its full natural order for P2b, with its public source/copy/replay peak bound.

The interface must make the phase and source identity explicit. A lease cannot
outlive the owner it borrows; every derived secret buffer clears on release and
unwind. Callbacks cannot hide full tensor caches, gadget temporaries, recursive
replay, backing capacity or borrowed model copies outside the declared bill.
An indexed synthetic function is not evidence that a real callback needs no work.

The concrete Rust interface is tier B/C and remains implementation work. No
new secret-access exemption, unsafe code or dependency on prover code from the
verifier is authorized by this RFC.

### Group pass

Process groups in the existing order. The token index's low bits select its
interleaved subset; real layers select their layer subset. Prologue/epilogue
belong to layer subset zero; no-token blocks belong to token subset zero, whole.
Public channel tensors read by token-bearing blocks can have an empty group
variable set and intentional repeated claims.

Keep the required current sources or bounded tile leases through tree proving,
their split evaluations and merge-table accumulation. Price their overlap with
columns, forest shares, kernels, prior outputs and kept tables. Evaluate each
split from its restricted group source; do not quietly materialize a complete
stage source or original polynomial during this phase. Release a source only
after every consumer, including native preparation if applicable, has transferred
or completed its obligation.

An empty group-variable set is an explicit exception to any shrink expectation:
its selected block is the whole canonical tensor. Keep and price that whole
source when required. The prohibition above concerns an additional unrestricted
materialization, not this required held source. Naming a full allocation a
"block" never removes it from the bill.

Use the unchanged affine forms, canonical leaf permutation and row-zero padding.
P1 counts the selected grouping's complete padding multiplicities. Preserve
RFC-0054's roots/levels/values/splits/gamma order and the separate table batch.

### P2b and points

Keep current MergeTables and use its legal public cut. Regenerate every original
polynomial once at P2b; never substitute a group-local reordered table. Retained
folded states, current original source, temporary owned copy/permutation,
weights/equality, callback work and outputs are separate simultaneous owners.

The merge binds free variables first and group variables second. Scatter its
point back to the original polynomial axes as RFC-0054 specifies. Native and
reduction consumers and every original advice/weight opening keep those claims;
zero coefficients or coincident field points cannot delete an obligation.

### Complete admission

Use the [full phase inventory](../../research/experiments/EXP-0031-bounded-grouped-source/phase-inventory.md)
as the implementation checklist. Every preparation/planning, P1, current group,
table batch, P2b, native/stage, target transfer, P4/P5 and packet phase must have
a checked bound, including shared backing owners and caller capacities.

Unknown differs from known zero. Checked sums price coexisting owners and a
checked maximum prices alternative phases. Overflow, unaddressable arrays,
unknown owners, unsupported callbacks and a complete peak exceeding the host
allowance return the existing typed memory refusal before P1. Unknown geometry
may retain its existing typed unsupported-shape refusal; never admit it with a
guessed zero. The allocator must not decide the plan after Round1.

Select only complete candidates under the unchanged public grouping rules and
positive-window requirements. Preserve both guards until full implementation
and controls pass; a component calculator or measured small peak cannot remove
one guard independently. The resident logical-cell floor is unchanged.
Caller model, stack, allocator/runtime and external transport overhead retain
their existing separate reporting and whole-process measurement obligations.

Recompute and enforce the existing soundness-budget checks for the selected
public grouping; a smaller memory estimate cannot waive them. Freeze the chosen
grouping, canonical count plan, source identities and all parameters before P1.
No mid-proof memory retry may change those choices after multiplicities or
Round1 have been fixed.

### Pass work

Retain RFC-0054's group pass followed by P2b. Charge checkpoint/KV storage and
ensure real group tile scheduling, not a per-value replay that repeats entire
forwards. Report actual regeneration work; this design carries no measured
forward-equivalent count or throughput claim. If the factory cannot satisfy
the required schedule and bounds, it is not an admitted implementation.

## Security Argument

The [abstract argument](../../research/experiments/EXP-0031-bounded-grouped-source/contract.md)
proves insertion/deletion are inverse on declared group domains; source-axis
projection preserves every canonical value; unchanged forms/leaf permutation and
padding preserve entire columns and counts; and point scattering preserves
restricted and original multilinear evaluations. Equal messages preserve the
existing challenge schedule by induction. This is conditional on a canonical
factory and checked descriptors, not proof a future adapter meets them.

The accepted relation and independent verifier do not change. No new soundness
estimate, witness-uniqueness relaxation or formal zero-knowledge claim is made.
Production factory conformance, clearing, exact bounds and independent complete
acceptance are mandatory implementation gates, not assumptions that can be
replaced by the finite experiment.

## Evidence

[EXP-0031](../../research/experiments/EXP-0031-bounded-grouped-source/README.md)
records the committed registration, actual runs, independent recount and fault
controls. Its case counts are E0 finite coverage; allocator observations are E1
on the named CPU. The production admission, whole-model fidelity/memory/time and
GPU evidence remain absent. No improvement to the frontier cost-model scoreboard
is credited.

## Alternatives Considered

- Keep complete ModelWitness and change only message/tree bills: fails to remove
  the source floor, though it can be conservatively priced and honestly refused.
- Fetch a whole original tensor for each group split: repeats a large original
  source before P2b; use the restricted source and keep its lifetime explicit.
- Assume every group shrinks uniformly: loses held roles and exceptional groups.
- Drop original opening targets or use favorable field collisions: changes the
  statement or underprices it; rejected.
- Reduce the existing resident floor, limits or parameters: unnecessary and not
  authorized by this design.

## Drawbacks

The factory and scheduler must coordinate source leases, gadget tiles and native
consumers. The retained cache or largest original P2b polynomial may still exceed
budget. Conservative public bounds can refuse feasible layouts. Repeated source
reads can harm performance; finite stateless synthetic callbacks do not settle
that cost. This design adds no immediate production admission.

## Migration / Rollout

Accept this RFC before extending the checked fallback. Implement and validate
the factory and complete phase composition in a separate issue/PR. Retain the
resident path, both grouped guards and protocol/2 exclusion through that work.
Only remove the guards together after complete allocation/refusal, determinism,
NI/Live independent acceptance and exact tamper controls pass. Qualify actual
1B/8B and later frontier models separately; existing immutable attempts are not
restarted, altered or counted as acceptance because this Draft exists.

## Testing Strategy

- Reproduce the committed finite experiment and its independent audit; buildable
  source, identity and accounting faults must fail with restored controls.
- Production conformance: every canonical gadget/role, request-slot gap, padded
  axis/layer, no-token block, forest, original polynomial and target matches the
  resident oracle; test exceptional and boundary group counts and dimensions.
- Count complete allocation lifetimes, including replay/cache/native handoffs,
  prior outputs, original/permuted/copy buffers and publication. Unknown owners,
  overflow and one-byte-under complete budgets refuse before P1.
- Preserve complete artifacts/transcripts for identical public plans, grouping,
  source inputs and challenges, and exact independent NI/Live verdicts
  and tamper errors. Keep verifier resource limits, pins and parameters fixed.
- Record real-model fidelity and whole-process time/memory before claiming model
  qualification; use the target GPU protocol before any GPU claim.

## Open Questions

- OQ-0077-1: Which canonical factory/lease implementation satisfies every gadget
  and native consumer with bounded checkpoint replay and the required pass work?
- OQ-0077-2: Which supported real-model shapes fit the complete conservative bound,
  including the largest original P2b polynomial and cache construction?

These are implementation/qualification obligations. Until resolved for a proposed
admitted route, it must refuse; they are not permission to weaken the contract.

## References

- [IDEA-0042](../../research/ideas/IDEA-0042-bounded-grouped-source.md)
- [EXP-0031](../../research/experiments/EXP-0031-bounded-grouped-source/README.md)
- [RFC-0018](RFC-0018-gpu-prover-runtime.md), [RFC-0054](RFC-0054-token-groups.md), [RFC-0057](RFC-0057-one-proof-unit.md), [RFC-0071](RFC-0071-request-slots.md)
