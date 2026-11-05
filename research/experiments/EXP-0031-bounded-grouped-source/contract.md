# EXP-0031 abstract source contract

E0 argument, conditional on canonical descriptors and the source callback below.
This is not a proof that a future checkpoint/gadget adapter implements that
callback, a complete memory bound, or a production admission decision. Finite
implementation checks remain separately required by the registration.

## Assumptions and refusals

Let `A(s,l,t,i)` be the unique integer or field value of canonical source tensor
`t` at site `s`, real layer `l`, and its full natural padded index `i`. Its
definition includes node-specific padding, masks, slot gaps and public roles.
An adapter must return this value; a token's position alone is not its identity.
The contract does not replace any gadget equation or integer range rule.

Public graph-derived layouts supply each lookup's domain and token axis, its
forms, each term's tensor axes, block offsets, and the site's layer membership.
The current layout constructor enforces agreement of a tensor's reads. Caller
claims about shapes or memory are not authenticated by a size calculator.
Inconsistent identities, axes, domains, group bits or read agreement must refuse
before admission. A source callback's replay, temporaries and returned owners
must have their own public bound; an unknown bound is infeasible.

The following argument concerns supported complete descriptors. For a shape an
implementation cannot derive or bound, including any exceptional forest it does
not support, it must refuse before P1. It cannot assume uniform shrinkage or
silently omit a role. Protocol/2 grouping remains excluded.

## Index restriction is a bijection

For padded axis widths `b_0,...,b_(k-1)`, the last axis occupies the lowest bits
of a natural row-major index. If the token axis is `a`, its lowest bit is
`p = sum_(j>a) b_j`. Group token subset `s` fixes positions
`S = {p,...,p+g_T-1}` to the bits of `s`.

Define `I_s(y)` by visiting full index positions in increasing order: insert
`s` at `S`, and consume successive bits of `y` at the other positions. Deleting
`S` from `I_s(y)` returns `y`. Conversely, inserting into the deletion of any
full index whose fixed bits are `s` returns that index. Hence this is a bijection
between the group's natural domain and exactly its full-domain restriction.
This includes an empty fixed set; no division or field collision is involved.

A term's index is the projection of that full index onto its declared axes,
with the same lowest-first variable ordering. The resident constructor reads
the stored array at that projection. The regenerating constructor requests
`A(s,l,t,projection(I_s(y)))`. They therefore obtain the same value under the
callback assumption, including every padded coordinate and slot gap.

## Lookup columns and counts

Evaluate the unchanged affine key/value forms on those equal term values.
Ring operations preserve equality; no bound or integer representation changes.
Place each value at the same block offset and canonical leaf permutation. That
permutation keeps the low bit and reverses the remaining bits; it is an
involution, so distinct column positions remain distinct leaves.

The group layout's aligned disjoint blocks cover their declared entries.
Initialize every other column entry to that table's row-zero tuple in both
constructors. The entire ordered key/value columns are equal, not merely their
digests or sampled entries. Counting each key therefore gives equal
multiplicities, including each group's additional row-zero padding. P1 must use
the chosen public grouping's counts; ungrouped counts cannot be substituted.

Public membership selects one layer subset by the real layer index, with
prologue/epilogue in subset zero. A token-bearing block selects one token subset
by its token bits. A block without a token axis belongs to token subset zero,
whole. Thus blocks are partitioned without losing constraints. A channel tensor
read by token-bearing blocks can instead be read repeatedly with an empty group
variable set; those are intentional claims on one polynomial, not permission to
duplicate a no-token lookup block.

## Forests and split points

For a supported stacked forest, fix query-index bits at
`q..q+g_T` and layer-index bits at `l..l+g_L`. The public conditions
`q+g_T <= l` and `l+g_L <= vars` make them disjoint. The same insertion/deletion
proof gives exactly the current forest share, in its original order. Leaves of
padded queries/layers remain whatever canonical `A` specifies; they are not
dropped. Nonstacked forests need explicit membership and their own supported
descriptor, rather than an assumed layer reduction.

For a tensor, let `S_t` be its fixed coordinates. Assign a group's free point
to the complement of `S_t`, in increasing order, and its subset bits to `S_t`.
Expansion of the multilinear extension over the Boolean complement is exactly
the extension of the restricted table. A term's free coordinates in the group
point are its original coordinates with removed fixed positions deleted and
higher coordinates shifted down. A factory cannot reinterpret a repeated
dimension as a token axis; axes come from the public graph.

The existing merge binds the complement first, then the group variables. The
inverse scatter of its final point therefore names the same original tensor
evaluation. P2b requests that same full `A`, not a group-local permutation or a
newly padded tensor. The original native/stage/opening claims and forest leaf
claims remain obligations at those points.

## Messages and security boundary

With equal columns, forest leaves and split evaluations, the unchanged grouped
prover kernels compute the same roots, levels and messages, in the same group
order. For equal prior transcript bytes they draw equal challenges; induction
through groups, table batch and merge preserves their schedule. The two-pass
merge still regenerates original polynomials only after its first table rounds.
No challenge moves earlier, and no field coincidence is an accounting premise.

This is semantic preservation under the callback/layout assumptions. It neither
changes the verifier's accepted relation nor supplies a new cryptographic
soundness estimate. SP-3, integer ranges, Fiat-Shamir binding and the independent
verifier remain as specified. A production factory must establish the assumptions
for every canonical gadget, request slot, checkpoint and opening target. Current
finite synthetic equality cannot establish that implementation fact.

## Ownership and resource boundary

For each live allocation owner, charge requested capacity and any overlapping
old buffer during growth. Sum owners that coexist, then take a checked maximum
over phases. A phase that retains earlier outputs cannot replace their sum by
the latest group's size. An original regenerated P2b table and its permutation,
weights/equality, folded state, kept tables and callback scratch are distinct
owners whenever they coexist. Unknown owners, arithmetic overflow and
unaddressable arrays make the plan infeasible before P1.

The present `SourceBill` concerns current source columns, one forest share and
index records, including retained earlier index records. Its allocation-free
integer arithmetic and independent recount do not price checkpoint replay,
gadget temporaries, lookup proofs, group messages, complete merge work or P1-P5.
The source prototype's oracle arrays, borrowed public layouts and serialized
reports also remain explicit exclusions. A complete physical route must compose
all those owners; this component cannot authorize it. Clearing is required on
every release/unwind path, regardless of whether formal zero knowledge is claimed.
