# RFC-0053: Exact-Greedy Runtime Boundary

- Status: Accepted
- Authors: @AbdelStark
- Created: 2026-10-21
- Target milestone: v0.3
- Amended by: [RFC-0071](RFC-0071-request-slots.md) ('Public request and sources': in a batch, each request's certificate, key column and time coefficient on its own slot)
- Origin: #128 and #130; IDEA-0038 / EXP-0025
- Supersedes: RFC-0009 Argmax's physical range/maximum representation; RFC-0037's native boundary and source schedule; RFC-0045's runtime lookup columns; RFC-0047/0050's complete advice order and RFC-0041's private tile enum, extended below. It adds no protocol-message field or parameter preset and preserves RFC-0011's per-group evaluation checks.

## Summary

Bind exact-greedy certificates to the actual model logits, next-token OneHot and
token input. One maximum array and publicly derived range keys join the runtime
circuit. Native final messages expose only evaluations that remain owed to the
original graph or canonical openings. No external scalar substitutes for a
model output. This contract defines the protocol/1 exact-greedy runtime boundary.

## Motivation

RFC-0026 explicitly needs no new claim-list field for greedy decoding. RFC-0050
fixes model auxiliaries, but leaves scoring roles for their integration. The
current model reducer retains token/logit debts; the standalone Argmax retains
maximum and output-view debts. A session driver must discharge them.

## Goals

- Complete source identities, public domains, range definitions and native inputs.
- Keep the existing single mixed stage loop and canonical PCS equations.
- Preserve every true-integer, padding, transcript and source-binding obligation.

## Non-Goals

- Temperature/mixture, anchors, ledger or protocol/2 masking.
- New commitment primitives, parameter values or a performance claim.
- Treating the finite E0 experiment as a full-protocol proof or budget measurement.

## Proposed Design

### Public request and sources

Let `n=prompt+outputs`, `V=vocab`, with positive prompt and at least one output.
Zero-output requests omit maximum/key/selected certificates, retaining canonical
token binding and every model check. Tensor variables are LSB-first
vocabulary coordinates followed by model-time coordinates. Logical tensors use
their independent per-axis power-of-two padding.

`L[t,v]` is the actual final LM-head output; `H[t,v]` the model's actual OneHot.
The output token at `prompt+u` uses logits at `prompt-1+u`. Certificate ids are
the graph node count plus the verifier-derived request ordinal, reserving every
model node id. Arithmetic is checked. A maximum is a
certificate source, never an invented model node or model auxiliary.

Allocate `M[t]` on the model's padded time domain. Real output-logit positions
are `prompt-1 <= t < n-1`; every other entry is zero. In a proof unit, place this
maximum after its model tensors/auxiliaries and before table multiplicities.
Multiple requests retain their public batch order and each its own key column and key relation, named by its ordinal. In a batch ([RFC-0071](RFC-0071-request-slots.md#the-greedy-certificate-and-scoring)), request `r`'s time is its slot's rows. One maximum `M` serves the batch, a single source in one `GreedyMaximum` tile, real on every request's window, positions `prompt_r − 1 ≤ p < n_r − 1` of slot `r`. Each request keeps its own key column, in batch order, and key relation, whose time coefficient is `Σ_{u<outputs_r} eq(P_row, u)·eq(r_time, r·S + prompt_r − 1 + u)`. The selected and token-binding relations run once over the whole token axis. A maximum per request would let a prover hide a compensating entry outside a window (RFC-0071 'Alternatives Considered' 5).
Apply the existing whole-row public-zero rule and segment/maximum-size checks to
the complete runtime layout; secret-observed zeros never remove rows.
Real inactive maximum positions remain committed and are proved zero by the
selected relation; model-time padding follows the existing implicit-zero rule.
Append `GreedyMaximum { gadget: u32 }` as private `Tile` tag 4. Existing tags
0 through 3 retain their encodings. The compiler's model-only execution plan
remains unchanged; the runtime expansion inserts this separate tile, one per batch
(RFC-0071), before all used table tiles. This tile has no graph shares or query block.

### Public range and catalog

The verifier re-derives an inclusive logit interval `[lo,hi]` from public model
data. Both endpoints must satisfy RFC-0009's strict 28-bit logit bound. Use
`k=max(1,bit_length(hi-lo))`, at most 29 bits (E0). The selected equality makes
the maximum an actual logit. Thus `m-l-lt` lies in `[-(hi-lo)-1,hi-lo]` as a
true integer; membership in Range(k) is exactly the needed nonnegativity and
implies the original logical 29-bit predicate. Negative field encodings cannot
enter this small range. No observed witness narrows the interval.

Model table ids and the claim list's bytes remain unchanged. Reuse an exact
Range(k) model table if present. Otherwise append the independently generated
implicit Range(k) to the runtime catalog after the model catalog. For several
required widths, append unique missing widths in ascending order. This rule
defines proof-local ids; it does not change RFC-0029's model catalog order.
Multiplicity counts include model and certificate uses of a shared table.
The physical width must satisfy `GadgetPolicy.max_log2_rows`; refuse its gadget
domain otherwise. No policy or range is weakened to make the table fit.

Append each request's logical key column after its model lookup columns, with
canonical first-read/table packing as RFC-0045 prescribes. Its domain is padded
`(outputs,V)`, and both padding axes select key zero. Tree roots, table trees,
splits and multiplicities use the complete runtime catalog/counts. Implicit
table definitions/digests and physical locations are verifier-derived.

### Original-point increment

Write `Hplus[t,v]=H[(t+1) mod 2^b,v]` on the padded model-time cube of `b` bits.
For a time point `r`, partition increment carries by their first zero. For each
`j<b`, the coefficient is `prod(i<j,r_i)*(1-r_j)` and the original time point is
`(0^j,1,r[j+1:])`. The wrap coefficient is `prod(r_i)`, at the all-zero time
point. Vocabulary coordinates stay unchanged. This is an exact linear identity
(E0); active output positions never use the cyclic wrap.

Native final messages carry each original `H` evaluation, in increasing carry
length and then wrap order. The verifier computes `Hplus` from these values and
retains each as a separate original-source opening claim. No value is an
unchecked per-image split. The complete native final vector is absorbed before
PCS Gamma. Existing per-group Gamma powers and evaluation checks remain intact.

The key and selected instances have the same variable count and therefore project
their LM-head claims at the same stage point. Retain both original obligations.
For a weight type, repeated claims with identical column point, row point and
gamma share its existing single evaluation row; check that row's evaluation
against **each** claimed value. Distinct verifier-derived points on that weight
type remain an internal geometry error. `ChannelParams` retains RFC-0052's one
evaluation row per original claim. The last combined row always uses the
independent proximity point, never an alias's row point. No wire field, row or
challenge is added, and this introduces no batching error term: equal-point
values are checked individually against the same row.

### Native relations

All three relations participate in the existing stage-one batch. Let `N` be the
model vocabulary-plus-time variable count. The public coefficients below are
multilinear extensions of their specified Boolean coefficient vectors, not an
arbitrary product of public MLEs on overlapping axes.

1. **Key**, degree two, `N` variables (E0). Reduce the earlier Range key claim at
   output/vocabulary point `P` to a linear functional of `L`, `Hplus` and `M`.
   Its time coefficient is
   `sum(u<outputs) eq(P_row,u)*eq(r_time,prompt-1+u)`.
   Vocabulary coefficients are the truncated equality for logits, the strict
   bounded suffix kernel for `Hplus`, and `LT(P_vocab,V)*eq(0,r_vocab)` for the
   maximum. Signs are maximum minus logit minus suffix.
2. **Selected**, degree three, `N` variables (E0). At the earlier stage zero-point
   `Z_time`, the cube sum is
   `sum(t) eq(Z_time,t) * (active(t)*sum(v<V,Hplus[t,v]*L[t,v])-M[t]) = 0`.
   The maximum term covers every padded time, so inactive/padded maxima are
   constrained to zero. Its source factors are `L`, `Hplus`, `M`.
3. **Token binding**, degree two, `N` variables (E0). The zero-check is
   `sum(t) eq(Z_time,t)*(sum(v) v*H[t,v]-Tokens[t]) = 0`.
   Keep the original OneHot Booleanity, vocabulary and mass checks. This native
   replaces only the weighted-input boundary, and only when the verifier's
   derived schedule contains that exact replacement. No token scalar is supplied
   from outside the proof's canonical token commitment.

For zero-output requests, append a degree-one, zero-valued observation instance
whose raw final vector contains `L(r)` at the model vocabulary/time point. Its
local relation is identically zero; it asserts no greedy predicate. It sends the
actual LM-head observation into the ordinary GKR reduction, which still owes
all graph checks and canonical source openings. Supplying an arbitrary logit
scalar is not permitted. Token binding remains mandatory even when `n=1`.

Key/selected raw final order is `L(r), M(r_time)`, followed by all carry/wrap
original `H` evaluations. Token-binding order is `H(r), Tokens(r_time)`.
Retain original points and values separately even when they repeat. Route `L(r)`
into the actual LM-head graph reduction; open maxima, OneHot and token inputs at
their canonical locations. A local native check never discharges those debts.

### Schedule, transcript and budget

The verifier derives native tags, owners, projections, lookup counts and field
counts. Preserve the model's existing native slots in their original order, then
append token binding, then key and selected for nonempty outputs, or the logit
observation for empty outputs. Use the actual OneHot node as token-binding owner
and the actual LM-head node as the greedy/observation owner. Native tags follow
this registration order; a source id never changes the graph's node count. The graph's LM-head boundary and weighted OneHot input boundary may be
deferred only to these replacements; absence/mismatch is an internal error.
Use the existing stage labels, one mixed batching challenge and final-vector
absorption. No proof supplies an alternative schedule or source map.

Before acceptance, derive the complete field/proximity budget from the actual
runtime schedule, degree maxima, variable counts, original source claims,
opening groups, table counts and matrix rows. Enforce the accepted preset's
targets without weakening it. The preset name, complete commitment layout and
session mode must all match: NI grinding credits cannot apply to a Live session.
An E0 primitive test does not establish that a
model/preset passes. Protocol/2 is not supported by this contract; its joint
mask/opening budget remains a separate prerequisite.

Count each shared tree level once, with its actual maximum degree/variables and
all fraction/product instances. Count every graph and native zero-check, and
every raw native final even when it leaves fewer graph projections. For the
packet geometry a degree-`d` round sends `d` fields, since the sumcheck omits
`p(1)` and recovers it from the current claim (RFC-0010 'Sumcheck messages'). For the
LogUp bound use the complete padded lookup/table tree leaf populations. Advice
groups count original combinations, including repeated source obligations, not
only their images or distinct points. Their counts come from the canonical
opening plan after the original claims are fixed. Advice proximity spans the
padded allocated row index, including gaps; weight proximity uses actual
committed rows. A failed target is `Reject(Params { name })`, not a weight-epoch
budget rejection. No proof supplies a count or a substitute parameter set.

## Security Argument

Conditioned on every original source opening and graph reduction passing,
increment's carry partition gives the actual next-token OneHot at every Boolean
time/vocabulary point. The key is RFC-0009's masked `m-l-lt`; the selected
identity fixes `m` to the same token's actual logit and pins inactive maxima to
zero. OneHot and token binding use the same committed input, with its existing
integer/vocabulary checks. Lemma 9 then proves the lowest-index maximum.

The key and token polynomials have per-variable degree at most two; selected
has degree at most three. Existing sumcheck bounds apply with the actual
batch metadata. Each additional raw source value is fixed before PCS Gamma and
checked individually under the existing commitment equations. A hypothetical
global PCS sum with per-group-reset powers could cancel unrelated false claims;
it is not part of this construction. The runtime budget predicate is mandatory.
The analytic argument is conditional on enforcing every source and budget predicate.
The finite experiment alone does not establish implementation soundness.

## Alternatives Considered

- Direct physical Range(29): valid but unnecessarily large for small proved
  logit intervals. NonNeg limbs add auxiliary columns; neither changes the
  original logical predicate, but they are outside this contract.
- Another committed shifted OneHot: adds a vocabulary-sized witness column.
- Separate row-view/suffix stages: needs dependent native scheduling machinery.
- A tensor-product time/vocabulary proof: expands domains and can exceed the
  existing variable cap. Original-point evaluations avoid that expansion.
- A new cross-group PCS rule: unnecessary; retain all original source values.

## Drawbacks

More original OneHot evaluations/opening groups must be counted. A full runtime
catalog/layout and clearing producer source must be implemented independently.
The finite experiment does not settle whole-proof performance or protocol/2.

## Migration / Rollout

No release has assigned tessera-protocol/1. Follow 09 section 3 before emitting
new complete proofs; do not silently read a released old artifact as this
runtime schedule. PR439 accepts this contract with complete bounded CPU proofs in
NI and fresh Live sessions, original-source opening checks, the complete runtime
budget, deterministic repeated reports and actual driver fault injection. The
frozen E0 experiment remains E0; implementation tests are separate evidence.
No current claim-list field, wire schema version, parameter value or protocol/2
decoder changes. The current driver proof unit contains one exact-greedy request;
pass planning and batching remain their roadmap issues. Test-small acceptances
certify nothing, and no production-scale performance or zero-knowledge claim
follows from the bounded CPU cases.

## Testing Strategy

[EXP-0025](../../research/experiments/EXP-0025-greedy-boundary/README.md) validates
finite E0 identities, degrees, integer predicates and source-point mappings.
Implementation must also test actual mixed CPU proofs, every original source
opening, both modes, exact first rejections and repeated report bytes. Include
ragged domains, all-negative/tied logits, one output, inactive maxima, public
span boundaries and compensating changes to two carry evaluations that preserve
their weighted sum but must fail original-source openings. Run both workspaces,
isolation and actual source mutations before review.

## Open Questions

- Complete runtime/model preset budget and proof performance require actual
  implementation evidence; no positive result is assumed.
- Protocol/2 masks and multiple proof-unit batching remain their roadmap work.

## References

- [IDEA-0038](../../research/ideas/IDEA-0038-greedy-boundary-bindings.md).
- [EXP-0025](../../research/experiments/EXP-0025-greedy-boundary/README.md).
- [RFC-0009](RFC-0009-gadgets.md), [RFC-0011](RFC-0011-advice-commitment.md),
  [RFC-0014](RFC-0014-interaction-and-soundness.md), [RFC-0017](RFC-0017-scoring.md),
  [RFC-0026](RFC-0026-claim-list-layout.md), [RFC-0037](RFC-0037-stage-schedule.md),
  [RFC-0045](RFC-0045-leaf-layout.md), [RFC-0047](RFC-0047-advice-order.md),
  [RFC-0050](RFC-0050-gadget-advice-columns.md).
