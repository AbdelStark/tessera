# IDEA-0042: Bounded canonical sources for grouped checkpoint proving

| | |
|---|---|
| Status | `promoted` |
| Origin | #687; EXP-0030; SUR-0015; current CPU source ownership |
| Lever | source memory and `regen`; correctness prerequisite, no credited gain |
| Spec surfaces | RFC-0018 Memory scheduler and Pass structure; RFC-0054 Groups and merge ownership; tier-B CPU interface/internal source adapter; no tier-A change |
| Priority | correctness prerequisite before the historical performance queue; numerical score zero |

## Mechanism

The CPU currently constructs a complete `ModelWitness` before choosing the leaf
proof's grouped branch [TesseraGrouped26]. Grouping its trees does not release
that source. Replace that ownership, in a future implementation, with a public
descriptor plus canonical regeneration of the current group's held columns and
forest leaves. Evaluate its splits while its source is live, accumulate its
merge tables, then clear the group payload before the next group. P2b regenerates
each original polynomial once after the first merge rounds. An original table,
permutation/fold state, equality weights and callback buffers are separate owners.

Derive geometry from the public graph, layer map, request slots and leaf layout.
Allocation and iteration domains cannot depend on nonzero counts, challenges or
witness values. Non-token roles belong to token subset zero; prologue/epilogue
roles belong to layer subset zero. Preserve padding tuples, multiplicities,
coordinate order, merged points, original openings and all transcript bytes.
No production admission is proposed until every phase has a checked complete bill.

For public owners `B` retained across phases and phase-specific owners `A_p`, the
eventual check is `max_p(B + A_p) <= host`, with checked arithmetic. Unknown owners
are infeasible, not zero. At the group phase, `A_p` includes the current source,
layout/columns/shares, kernel work, earlier outputs and kept merge tables. P2b
includes all still-retained group/table messages beside original regenerated
values, permutation, weights, folded state and merge output. Later proof phases
retain their own existing obligations. This is a contract, not a validated bill.

## Seeds and lenses

The scoped seed set is exhausted by these ten implementation obligations, each
drawn from RFC-0018/0054 or current ownership [TesseraGrouped26]: complete resident
source, token-bit restriction, layer-bit restriction, exceptional subset-zero
roles, request-slot gaps, lookup padding/counts, forest query/layer coordinates,
split coordinates, kept merge/P2b buffers, and cross-phase retained messages.

| Lens | Candidate or reason for no separate candidate |
|---|---|
| Commitment | Preserve advice rows, commitment roots and opening targets; no new PCS |
| Lookup | Regenerate exactly the current canonical columns and counts |
| Max certificate | Preserve all forest leaves and public roots; no new certificate |
| Counts/compiler | Public descriptors select held roles; never delete constraints |
| Attention | Keep all required earlier KV/checkpoint data charged; no causal shortcut |
| Passes | One group payload at a time, then registered P2b regeneration |
| Parameters | No parameter, bound, query or group-limit relaxation |
| Zero knowledge | Protocol/1 only; protocol/2 grouping remains excluded; clearing remains required |
| Hardware | CPU correctness/allocation evidence only; no GPU extrapolation |
| Other paradigm | Recursion/new commitment families change the task's protocol and are not needed for this ownership prerequisite |

## Why it could work

RFC-0054 already specifies regeneration from checkpoints and a merge pass.
EXP-0030 establishes why a resident logical floor cannot certify a bounded route;
it does not validate this replacement. SUR-0015's compact counters address a
different owner. Current component bounds price group layouts, lookup trees,
merge work and returned messages, but explicitly exclude their surrounding
source/callback lifetimes [TesseraGrouped26]. All statements here are E0 source
analysis, not measured memory or an all-input source-equivalence proof.

## Expected gain

| Context | Credited delta mu | Delta proof size | Delta verification |
|---|---|---|---|
| 1K | 0, E0 bookkeeping | 0, intended invariant | 0, intended invariant |
| 4K | 0, E0 bookkeeping | 0, intended invariant | 0, intended invariant |
| 16K | 0, E0 bookkeeping | 0, intended invariant | 0, intended invariant |
| 128K | 0, E0 bookkeeping | 0, intended invariant | 0, intended invariant |

No cost-model parameter is changed. The baseline remains the current model;
source payload/allocation observations in EXP-0031 cannot be translated into
whole-proof speed, RSS, fidelity, GPU or real-model acceptance credit.

## Security gate

| Property | Effect | Pass |
|---|---|---|
| Soundness bits, G-4 | Same accepted relation/checks; E0 contract preserves canonical messages, no changed parameter or claimed new bound | pass for design preservation under the stated factory contract; production conformance remains mandatory |
| Uniqueness, SP-3 | Same integer `D` and every original obligation; identity/projection bijection removes none | pass for the abstract contract; no proof a future checkpoint adapter implements it |
| Zero knowledge, G-5 | No new protocol/1 hiding claim; protocol/2 excluded; OBS-6 clearing required | pass for scope exclusion; clearing still an implementation gate |
| Verifier independence, SP-6 | No verifier code/dependency changes | pass by design; recheck final diff |
| Hash-based commitments | No cryptographic primitive change | pass by design |

Unknown in-scope contract gates prevent RFC promotion until the all-input argument
settles them. Finite success alone is insufficient. A contract-level argument may
support a Draft RFC; production adapter correctness remains a separate obligation,
and neither the RFC nor this experiment is a ready implementation.

## Prior rejection check

RFC-0018's former independent-tile-point alternative is answered by RFC-0054's
existing merge, which is retained. RFC-0054's per-level regeneration and multiple
proof-unit alternatives are not reintroduced. The baseline's rejected commitment,
recursion and soundness shortcuts are unrelated and unused. New here is bounded
CPU source ownership and its full lifetime contract, not a new lookup protocol.

## Kill criterion

Any canonical value/role/coordinate mismatch, missing owner, secret-dependent
allocation decision, uncaught registered fault or underpriced measured finite
allocation refutes the frozen candidate. Unsupported geometry must refuse rather
than use a favorable field coincidence or omit an obligation.

## Cheapest experiment

[EXP-0031](../experiments/EXP-0031-bounded-grouped-source/README.md) compares a
stateless synthetic source adapter against current resident grouped constructors,
then compares both current merge modes on the resulting polynomial data. E0
geometry and E1 CPU allocation counts; no timing measurement while real jobs run.
Effort is `m`: prototype, lifecycle accounting and an RFC before production work.
An H100 run would settle throughput and device lifetimes, neither claimed here.

## Refutations

| Lens | Verdict | Argument |
|---|---|---|
| Soundness | weakened | Row equality is necessary; finite fixtures do not establish all-input adapter correctness or a production security approval |
| Prior art | survives | Uses the accepted grouped merge; addresses a current CPU source owner rather than claiming a new protocol |
| Cost model | weakened | Existing frontier projections do not establish this CPU bottleneck's cost; zero gain credited |
| Feasibility | survives | Finite public/synthetic geometry and allocation checks need no real weights, GPU or changes to live jobs |

## Score

| Gain | Confidence | Cost | Priority |
|---|---|---|---|
| 0 credited | 0.6 for finite feasibility; not a soundness probability | m, runtime RFC and adapter work | 0 numerical; authorized correctness prerequisite |

## Decision and links

Self-triaged and reviewed by the implementing agent, not an independent human.
[EXP-0031](../experiments/EXP-0031-bounded-grouped-source/README.md) is finite
E0/E1 validated with all twelve registered faults, an extra phase control and
restored controls. Accepted [RFC-0077](../../docs/rfcs/RFC-0077-bounded-grouped-checkpoint-sources.md)
promotes the contract and its explicit production gates; no production source,
guard, schema, secret access or admission changes. Issue:
[#687](https://github.com/AbdelStark/tessera/issues/687). Ledger updated 2026-11-05.
