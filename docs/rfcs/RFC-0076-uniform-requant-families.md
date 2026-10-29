# RFC-0076: One Requantization Family per Layer Site

- Status: Draft. No rule below takes effect before acceptance.
- Authors: @AbdelStark
- Created: 2026-10-29
- Target milestone: v0.3
- Origin: [#619](https://github.com/AbdelStark/tessera/issues/619), parent [#617](https://github.com/AbdelStark/tessera/issues/617)
- Supersedes on acceptance, only in the scope below:
  - [RFC-0006](RFC-0006-compiler-ir.md#gadget-selection), [RFC-0024](RFC-0024-wide-requantization.md#selection): canonical model Requant selection;
  - [RFC-0007](RFC-0007-claim-list.md#verifier-validation): step 5's public re-derivation of that selection;
  - [RFC-0009](RFC-0009-gadgets.md#requantsplit): Split/Wide applicability, not their lookup relations or uniqueness lemmas;
  - [RFC-0038](RFC-0038-uniform-layers.md#the-rule): uniform graph nodes also require uniform selected Requant read roles;
  - [02 §1](../spec/02-public-api.md#1-stability-tiers) and [09 §3](../spec/09-release-and-versioning.md#3-protocol-and-schema-versions): the one-time, unreleased Protocol One compatibility exception in this RFC.

## Summary

In Protocol One's canonical Llama model lowering, a Requant template position
must select a common family over its real layers. Keep the old selection when
its ordered tensor roles and projections already agree. Otherwise promote the
site to Split if every layer can take it, then Wide if every layer can take it.
Refuse a site neither family can represent within the public bounds.

The key's integer, channel classes, shifts, saturation and model `D` do not
change. A promoted gadget may use a wider public signed envelope for that key.
The verifier derives this envelope and the complete selection independently.
RFC-0045's read-agreement guard remains mandatory.

## Motivation

RFC-0038 makes layer graph nodes uniform, but RFC-0006 chooses each node's
gadget separately. RFC-0050 then gives different auxiliaries to One, Split and
Wide. A model can pass compilation and persistent weight commitment yet fail
RFC-0045 before its first proof.

The fixed exported 1B candidate in #617 has this failure at its MLP residual:
with `B=31` or `32`, one of its sixteen real layers takes Wide and fifteen take
Split; at `B=30`, two take Wide. These are E0 public metadata observations, not
successful proofs. [EXP-0029](../../research/experiments/EXP-0029-uniform-requant/README.md)
reproduces the refusal on this public lowering and a small synthetic model.
Its candidate passes the unchanged leaf constructor in both cases. Changing
the table policy alone did not resolve the fixed real model's incompatibility.

## Goals

- Compatible ordered Requant source roles at every admitted layer position.
- The exact same integer computation and one passing auxiliary witness.
- Complete deterministic public selection, tables, lookup requests and advice.
- Preserve old canonical artifacts when no site needs promotion.
- An explicit pre-release migration decision and independent implementation gates.

## Non-Goals

- A new gadget relation, table function, message schema, transcript label,
  parameter set, challenge field, trust assumption or target soundness.
- Removing a range, padding, lookup, read-agreement, opening or budget check.
- Changing quantization to avoid the failing scales, or splitting layers into
  separately committed models.
- Protocol Two promotion or a hiding claim. Its existing canonical selection
  remains unchanged, including any refusal of this model.
- A full real-model proof, fidelity qualification, GPU implementation or
  performance conclusion from the finite experiment.

## Proposed Design

### Scope and public grouping

This rule applies only to the canonical Llama model lowering of
`tessera-protocol/1`, for both existing profiles. Fragment/anchor graphs and
`tessera-protocol/2` retain their current per-node selection. Dispatch uses the
validated claim list's protocol version; a proof supplies no selection hint.

Derive the layer template from the public architecture, quantization and
canonical checkpoints, as RFC-0037/0038 require. Each Requant position is one
group of its real-layer nodes. Prologue and epilogue Requants are singleton
groups. Padded layers are not members. Invalid templates or RFC-0038's mixed
one-class/several-class sites remain refused by their existing checks.

For layer `i`, public range analysis gives signed width `b_i`, with key
`z_i` in `[-2^(b_i-1),2^(b_i-1))`. The public shifts are `r_i,c` in `[1,63]`,
their count `n_i` is in `[1,4]`, and `c_i=ceil(log2 n_i)`. The public table
limit is `B` in `[1,32]`. Signed model saturation has
`-2^31 <= lo_i < 0 <= hi_i < 2^31`; set
`M_i=max(abs(lo_i),abs(hi_i),1)` and `m_i=ceil(log2 M_i)`.
These are integers derived from the statement, never hidden channel values.

### Preserve the old case

First compute the old per-node plans in graph order under RFC-0006/0024/0028,
without allocating auxiliary table ids. Return the first old refusal before
processing any promotion. This RFC
repairs incompatible otherwise-valid plans; it does not waive another refusal.

Derive each plan's ordered semantic tensor roles and axis projections from its
lookup forms, using RFC-0045's first-read order. If they agree over every real
layer, retain every old plan exactly, including its widths and split points.
Do not promote merely because table ids, shifts or coefficients differ.

For canonical Requant nodes, compatibility requires the same family, the same
one-class/several-class presence and, for Wide, the same presence of `z_lo`.
The graph template fixes the tensors' domains and class projections. This
characterization is not permission to omit the full leaf read-agreement check.

### Common Split

For each layer set `b'_i=max(b_i,2)` and define

```text
a_i = max(1, b'_i + c_i - B)
d_i = min(r_i,min - 1, B, b'_i - 1)
s_i = clamp(ceil((b'_i + c_i)/2), a_i, d_i)
```

Split is feasible for the group only if every layer has:

- `b'_i <= 62` and `n_i * 2^b'_i < P`, with `P` Goldilocks;
- `a_i <= d_i`, so `1 <= s_i < b'_i` and `s_i < r_i,c` for every class;
- its low table `Range(s_i)` and high table
  `Requant(b'_i-s_i, [r_i,c-s_i], lo_i, hi_i)` fit `2^B` rows;
- every selected table regenerates under the existing table-definition limits,
  including its core-size bounds. Row capacity alone is not sufficient.

When feasible, select Split in every layer, retaining each layer's own `s_i`.
Its committed auxiliaries and lookup order are precisely RFC-0009/0050's.
Otherwise try common Wide; never install a partially selected group.

### Common Wide

Initially set `s_i=min(r_i,min-1,B)`. If any layer has `s_i=0`, set `s_i=0`
for every layer. This fixes low-column presence, not the hidden data.
Then set

```text
t_i  = r_i,max - s_i + m_i
b'_i = max(b_i, s_i + t_i + 1)
k_i  = b'_i - s_i - t_i
```

Wide is feasible only if every layer has:

- `0 <= s_i < r_i,min`, `t_i >= 1`, `k_i >= 1` and `b'_i <= 62`;
- `s_i <= B`, `t_i + 2 + c_i <= B`, and `k_i + 1 <= B`;
- regenerable existing table specifications and all RFC-0009 Lemma 11 input,
  class and true-integer preconditions.

Use RFC-0024's existing lookups, in this exact order:

1. `Range(s_i)` of `z_lo` when `s_i>0`;
2. `Range(t_i)` of `h0`;
3. `Range(k_i)` of the virtual `h1_off`;
4. `Requant(k_i+1,[1],-2,1)` of `2*h1_off`, returning `g`;
5. `Requant(t_i+2,[r_i,c-s_i],lo_i,hi_i)` of the existing packed middle/sign key.

The original `s=min(r_min-1,B)` and “Split must fail first” restrictions are
selection preferences, not premises of the lookup lemma. In a promoted group,
`s=0` and `k=1` are allowed under the explicit conditions above. No zero-width
Range table or omitted virtual top range is allowed.

### Finalization, order and refusal

If common Wide also fails, the compiler returns `ERR-COMPILE-RANGE`, naming the
first layer in graph order whose Wide precondition fails. It must not retry
with a changed `D`, exceed `B`, or leave an incomplete selected object.
The verifier rejects an unsupported or noncanonical public selection with
`Reject(StatementMismatch)` at its public claim-list validation step.

Process groups by their first real node in graph order and layers in increasing
layer order. Use checked or widened public arithmetic, including the packed
bound `n*2^b'` (which may equal `2^64` before refusal); never wrap or narrow it
before comparison. Resolve all groups' abstract plans before assigning auxiliary
table ids. Keep
the lowering's base tables. Walk final gadgets in node order; use the lowest
id of an equal table kind, or append at first use (RFC-0050). Discard temporary
plans rather than retaining their unused auxiliary tables. Number gadgets in
the existing node order, including non-Requant gadgets.

Build the complete ordered lookup-request cache with the final gadgets,
including repeated ids and virtual range requests. Advice allocation, witness
generation, leaf layout, native reductions, openings and reports must consume
that same canonical selection. A public mutable cache or the experiment's
scratch visibility overlay is not a production implementation.

### Security argument

All statements below are E0 deductions from the cited gadget lemmas. They are
the draft's self-review, not independent approval or measured global security.

**N1: A wider envelope does not change the key.** Since `b'_i>=b_i`, the original
true integer `z_i` lies in the widened signed interval. Neither its graph nor
its field value changes. `b'_i<=62` gives `0<=z_off<2^b'_i<P`. Channel values
keep RFC-0048's existing committed-class-domain assumption; the selector reads
only their public class count and shifts.

**N2: Split uniqueness and value.** The two lookup boxes give
`z_lo + 2^s*j = z_off + 2^b'*c (mod P)`. Both sides lie in
`[0,n*2^b')`, strictly below `P`, so they are equal as integers. Thus `z_lo`
and the reduced key are unique. Since `s<r_c`, nested floor division gives the
original rounded/clamped output (RFC-0009 Lemma 2). This proof requires neither
`b'>B` nor that One be unavailable. The strict packed bound remains essential:
the known four-class `b'=62` wrap attack remains refused.

**N3: Wide uniqueness and value.** The range lookups pin the mixed-radix digits
of `z_off` in a box of exactly `2^(s+t+k)=2^b'` possibilities. Congruence is
integer equality because `b'<=62`; the digits, then `g` and `y`, are unique
(RFC-0009 Lemma 11). This uses the virtual top range, even though it has no
committed auxiliary column. With `k>=1`, `2^(b'-1)` is divisible by `2^(s+t)`,
so `floor(z/2^s)=2^t*h1+h0`. With `s<r_c`, the original rounding reduces to
this integer's rounding at `r_c-s`. The chosen `t` ensures saturation when the
top lies outside `[-2,1]`, exactly as RFC-0024 proves. These deductions also
hold at `s=0` or `k=1`; neither needs the old incidental `k>=2` or `3`.

**N4: Padding and compatible reads.** For a padded zero key and class zero,
Split has low digit zero because `s<b'`. Wide has low and middle digits zero
and `h1=0` because `k>=1`. Thus every committed auxiliary and output is zero;
`lo<=0<=hi` is required. The original padding selectors and their checks remain.
The common family, class presence and low presence give identical ordered
semantic roles, over the unchanged template domains and projections. RFC-0045
still checks this fact; it is not inferred from a table's dimensions alone.

**N5: Binding and error accounting.** Each actual lookup, advice column and
opening remains an existing relation and obligation. No new challenge, label,
unchecked value or polynomial degree is introduced. Some quantities and tree
heights change, so all actual round/union/forest counts, per-instance `N+T`,
opening groups and claims must be re-derived from the final runtime geometry.
Keep the root zero-denominator/fraction checks, multiplicities below `P`,
within-column and cross-column batching, and every final-evaluation/opening debt.
No relation may be skipped because its new auxiliary is absent at another site.
The existing complete RFC-0014 budget must pass before acceptance. Old model
presets, selected-site subtotals and the finite experiment cannot establish
that budget. A failure must refuse; no target or proof-of-work requirement is
relaxed by this RFC.

**N6: Scope, secrets and independent derivation.** Family choice depends only
on validated public data. This introduces no new secret read, randomness,
logging surface or persistent state. Public grouping has at most the graph's
node count; checked dimensions and the verifier's metadata/resource limits still
apply before allocation. No secret arithmetic or constant-time rule changes.
Actual witness auxiliaries remain in the
existing clearing secret buffers. Both workspaces implement the public rule
from this document, with their own representation and traversal; no prover
code is copied into the verifier. Protocol Two receives no new selection,
mask, revealed evaluation or hiding assertion.

### Review matrix

| Checklist surface | Evidence / obligation |
|---|---|
| Field/integer overflow, class packing, rounding | N1–N3, explicit 62-bit/packed/table bounds; checked public arithmetic in both implementations |
| Canonical encoding and verifier derivation | Finalization and migration; exact independently re-derived tables/requests, no proof-supplied plan |
| Fiat–Shamir and batching | N5; existing absorption/order retained, complete changed round counts and degrees tested at integration |
| Gadget uniqueness and alternative witnesses | N2–N4; EXP-0029 integer/field, wrap and perturbation evidence; production adversarial tests still required |
| Zero padding and source projections | N4; retained leaf/padding guards; explicit `s=0`, `k=1`, class and singleton-axis tests |
| Commitments and opening debts | N5; new auxiliaries allocated/opened by existing canonical machinery, no dropped claims |
| Secrets and robustness | N6 and typed refusal; bounded public grouping and checked arithmetic, clearing/redaction tests at integration |
| Zero knowledge | Not claimed; Protocol Two behavior unchanged |
| Anchors, audit coins, ledgers, refresh and scoring | Unchanged; no new identities, decisions or messages on those surfaces |
| Whole implementation and non-author review | Not completed by this draft; mandatory before production adoption |

### Compatibility decision

This RFC proposes a **one-time exception** to 02 §1 / 09 §3 solely for this
unreleased Protocol One canonical-selection refinement. It does not grant a
general pre-release version waiver. No GitHub release exists as checked on
2026-10-29; implementation must recheck the release state before landing. This GitHub
inventory is not a waiver for publication through another channel; verify the
release history before using the exception.

If accepted before the first release supporting Protocol One, keep its current
protocol/schema/parameter identifiers. For a complete model whose sites are all
already compatible, preserve selections, table ids, advice placement and artifact
bytes. If any site is promoted, auxiliary ids/placements at other sites may move
under the canonical first-use order. Previously compiled incompatible
claim lists/plans must be regenerated from the same `D`. Their old table lists
are noncanonical under the new rule and get `StatementMismatch`; do not guess,
silently translate, relabel or bypass validation. Revalidate any existing weight
artifact against its actual new plan; unchanged weight-root reuse is not assumed.
Record this refinement and pinned implementation revision in the first release.

If a supporting release exists first, this exception is unavailable: normal
02/09 versioning applies and a superseding version decision is required before
implementation. A merged draft or passing experiment does not authorize an
implementation against either policy. Protocol Two remains unchanged throughout.

## Alternatives Considered

1. Remove the leaf guard: rejected because it leaves incompatible source claims.
2. Group layers by selected family: RFC-0038's extra weight claims/masks and
   routing work; unnecessary before testing a common existing family.
3. Change the model's scales or clip its computation: does not prove the fixed `D`.
4. Widen one site envelope only: its different shifts can still choose different
   families. This rule resolves the family and low-column presence explicitly.
5. Force Wide everywhere: needless work, and some shift/table combinations
   cannot take it. Preserve compatible sites and reject infeasible promotion.
6. Dummy auxiliary columns or a new bitwise requant gadget: new routing/padding
   obligations or relations, where the existing lemmas already suffice.
7. Keep incompatible lookup-cache metadata: the initial experiment adapter
   correctly refused it. Production finalization builds all metadata together.
8. Assign a new protocol version immediately: valid under normal policy; the
   explicit narrow exception above avoids reinterpreting any released format
   while preserving every previously compatible plan. Review must decide this
   exception before adoption; it is not inferred from an empty release list.

## Drawbacks

Promoted layers can add committed columns/lookups and larger envelopes. A forced
zero low part may make Wide infeasible. Table ids and advice placement can move
when any site is promoted. No performance saving is credited; actual counts and
resource budgets must be measured after implementation. The rule must be
implemented twice and tested across protocols, lengths and degenerate axes.

## Migration / Rollout

1. Review the mathematical argument and compatibility exception through the RFC
   process, including a genuine non-author maintainer review under 09 §6.
2. Accept the decision before production adoption; conditional references alone
   change no accepted behavior.
3. Implement complete compiler selection/finalization without a public mutable
   lookup cache. Keep fragment and Protocol Two paths unchanged.
4. Independently implement verifier grouping/derivation from this specification,
   with independent data oracles and no prover dependency or copied code. The
   oracle writer follows the normative document without reading the implementation
   being checked; EXP-0029 is not a substitute for that independent evidence.
5. Regenerate affected public vectors, check both workspaces and all metadata,
   padding, clearing, opening and actual soundness-budget invariants.
6. Recompile the same real 1B candidate, verify numerical equality and derive its
   complete budget/resources. Produce a certifying-parameter proof, independently
   accept it and reject targeted tampering. Repeat the model ramp at 8B. A new
   concrete failure remains work; a public-layout pass is not this completion.

## Testing Strategy

The frozen [EXP-0029](../../research/experiments/EXP-0029-uniform-requant/README.md)
is finite E0 evidence: 154,900 sites, 4,590,008 integer/field cases, 9,881,680
rejected auxiliary perturbations, ten caught physical source faults and
unchanged/restored controls. Candidate layouts pass the unchanged tiny and
real-public constructors at `B=30,31,32`. These are not global soundness,
performance, independent-verifier or full-proof results.

Production tests must additionally cover:

- Exact old bytes/ids/layouts for every unchanged conformance model and Protocol
  Two path; no dependency on request length, slot packing or prover hints.
- One/Split, One/Wide, Split/Wide, different positive low widths and mixed
  zero/nonzero low presence, including `b'=2`, `s=0`, `k=1`, `L=1` and axis length 1.
- Every public precondition violated alone, especially four classes at 62 bits,
  split at a rounding bit, missing virtual top range, wrong table widths and
  a wider envelope outside 62 bits, with exact errors/rejection reasons.
- Independent whole table/gadget/request-cache/advice sequences and opening
  identities, including repeated ids and no unused provisional auxiliary tables.
- Honest zero and nonzero padded values to pin the existing padding predicates,
  source binding, clearing and all promised opening debts.
- F65537 soundness calibration and Goldilocks boundary attacks; no use of
  `test-small` as certification of the real model.
- Exact numerical output equality, actual complete budget evaluation and
  independently verified/tampered certifying-parameter real-model proofs.

## Open Questions

No selection formula is left to implementer choice. The decision gates are
non-author review of N1–N6 and the one-time compatibility exception. Production
counts, resource feasibility, fidelity and full proofs remain integration
deliverables; this draft does not assert their success.

## References

- [IDEA-0041](../../research/ideas/IDEA-0041-uniform-requant.md),
  [EXP-0029](../../research/experiments/EXP-0029-uniform-requant/README.md).
- [RFC-0001](RFC-0001-workspace-and-verifier-independence.md),
  [RFC-0006](RFC-0006-compiler-ir.md), [RFC-0007](RFC-0007-claim-list.md),
  [RFC-0009](RFC-0009-gadgets.md), [RFC-0014](RFC-0014-interaction-and-soundness.md),
  [RFC-0024](RFC-0024-wide-requantization.md), [RFC-0028](RFC-0028-balanced-split.md),
  [RFC-0038](RFC-0038-uniform-layers.md), [RFC-0045](RFC-0045-leaf-layout.md),
  [RFC-0048](RFC-0048-weight-trust.md), [RFC-0050](RFC-0050-gadget-advice-columns.md).
