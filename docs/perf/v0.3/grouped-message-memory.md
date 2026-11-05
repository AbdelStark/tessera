# Grouped leaf message ownership

`GroupedMessageMemory::new` prices the requested heap storage of grouped leaf
messages from a `GroupedMessageShape`, under
[RFC-0018](../../rfcs/RFC-0018-gpu-prover-runtime.md#memory-scheduler) and
[RFC-0054](../../rfcs/RFC-0054-token-groups.md#the-stages-outputs).
It allocates nothing and reads no witness value or transcript challenge.
Overflow or an unaddressable array returns `LeafError::Memory`; unsupported group
counts and dimensions return an exact `LeafError::Internal` rule.

The caller must derive the counts from its public layout and schedule before P1.
They bound every group, including token-zero and layer-zero exceptions, rather
than assuming every tree shrinks with the number of groups. `max_vars` bounds
all tree heights, forest leaf dimensions and merged polynomial dimensions.
`tensors` bounds both each group's split values and the total tensor polynomials.
`forests` also bounds the total forest polynomials, including disjoint groups.
The calculator does not authenticate a caller's pairing of shape and inputs.

## Owners and overlap

| Bound | Included owners |
|---|---|
| `retained_bytes` | CPU proof Box, group descriptors, roots, complete levels and their points/claims, key/value and forest claims, split and leaf-point arrays, table roots/levels/multiplicity claims, merge rounds/point/variable and evaluation-range metadata, merged claims, original CPU atom splits |
| `wire_bytes` | Additional allocations of the production `ProvenGroupedLeaf::to_wire`: group roots/levels/values/splits, table roots/levels, merge proof and atom splits |
| `driver_bytes` | Merge polynomial descriptors and group-coordinate vectors, tensor-read triples, grouped merge claims and their points, gammas and current free/full split coordinates |
| `overlap_bytes` | Checked sum of all three components, conservatively allowing them to coexist |

The bound includes nested vectors and spare capacity. It uses the pinned Rust
1.95 vector minimum capacities and a conservative doubling-growth envelope,
including overlap with an old allocation during growth. Tree levels use a
uniform maximum dimension and four final evaluations per participating tree;
merge output uses one per polynomial. This deliberately overprices smaller
levels and absent value claims. It does not inspect their field values.

`to_wire` is the CPU driver's existing clone operation, extracted into one method
so allocation tests measure the production conversion. It retains field order,
group order, every level proof and every split. Original point/claim owners stay
in the returned grouped proof while the wire copy is built.

## Scope and validation

Counting-allocation controls retain actual grouped proof output after proving and
measure the separate production wire-copy peak. They exercise token, layer and
combined groupings, padded token counts, mixed tree heights, lookups with and
without values, forests, and in-memory and two-pass merge modes. Borrowed inputs
and the transcript predate counting; the proving peak is not compared to a
message-only bound. A separate control retains actual `Merge` input metadata,
including long claim points, without allocating exponential field tables.
Sizing tests cover empty counts, boundary group/dimension counts and exact
arithmetic/addressability refusals.

Borrowed layouts, witness/tables and their capacities remain additional owners.
So do group construction, split evaluation, merge payload tables and kernel
scratch, table caches, transcript/callback/encoding work, native stages, original
opening targets, PCS work and packet publication. Stack, allocator bookkeeping,
runtime overhead and caller model storage need separate accounting and process
measurement. See the [CPU lifetime inventory](cpu-memory-lifetimes.md).

This component alone admits no physical grouped route. The
[scoped grouped composition](scoped-grouped-memory.md) supplies the additional
source and complete phase ownership required for admission. This component establishes no real-model
acceptance, model-fidelity qualification or whole-proof time/memory result;
[#641](https://github.com/AbdelStark/tessera/issues/641),
[#629](https://github.com/AbdelStark/tessera/issues/629) and
[#617](https://github.com/AbdelStark/tessera/issues/617) retain those obligations.
