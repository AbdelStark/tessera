# Lookup-group memory

`LookupGroupMemory::new` prices complete `prove_lookup_group` execution from
public lookup row counts/value modes, forest heights/tree-index widths and the
ordered `TreeId` schedule. It reads no field value, witness or transcript challenge.
Repeated shapes remain separate owners. Empty groups and zero-height lookup
trees are supported; each listed lookup has at least one row.

This implements the resident lookup-group component of
[RFC-0054](../../rfcs/RFC-0054-token-groups.md#the-leaf-stage-with-groups) under
[RFC-0018's memory scheduler](../../rfcs/RFC-0018-gpu-prover-runtime.md#memory-scheduler).
It composes the [resident tree bound](resident-tree-memory.md) and adds its wrapper
owners. It does not enable grouped physical admission under
[#641](https://github.com/AbdelStark/tessera/issues/641).

## Owners and overlap

The public planner checks positive lookup lengths and uses the original group's
schedule checks: one lookup/forest entry each, no table entry, duplicate or unknown
index. `ResidentTreeMemory` checks forest height/index geometry, addressability and
aggregate overflow. Planning allocates only two position maps and a tree-shape
vector; `planning_bytes` bounds their requested peak. No field table is allocated.

`peak_bytes` starts before fraction construction and includes every canonical tree
owner and resident proof buffer. It adds the optional tree-construction vector,
scheduled input vector, lookup/forest position maps and roots. The scheduled input
capacity remains charged when the tree engine can reuse it for returned leaves.

After proving, the wrapper keeps complete levels and final tree leaves while it
builds lookup key/value claims and forest claims, each with cloned points. The
bound includes the by-position sort vector, optional value results, value-message
growth and a temporary cloned fraction point. Value evaluation of a non-power-of-two
column pads a complete base-field table. Its blocked evaluator simultaneously
holds a low equality table and a high partial table; their capacities remain
charged until evaluation returns. Lookup values are never used to reduce a bound.

`output_bytes` prices complete returned levels, roots, value messages and all
key/value/forest claims and points. The tree model's `level_bytes` separates those
retained levels from final tree leaves the wrapper releases. Bounds include the
pinned Rust 1.95 vector growth/minimum capacities and public group labels.

## Additional owners

Borrowed lookup columns, forest leaves and input descriptors/capacities remain
additional. So do transcript/encoding/callback work, stack, allocator bookkeeping,
runtime overhead and caller model storage. The caller must derive the supplied
geometry from its actual public inputs; the calculator authenticates nothing.

The complete grouped leaf stage also owns its layouts/construction, columns,
forest shares, splits and their evaluation/regeneration, merge blocks/claims,
[kept merge tables and two-pass work](merge-memory-bound.md), the generated table
batch and all retained groups. CPU wire conversion, native stages, original
opening targets, weight work and packet publication add further overlapping owners.
The [complete scoped grouped composition](scoped-grouped-memory.md) includes
these additional phases and their admission checks.
This component establishes neither actual-model acceptance nor whole-proof
time/memory qualification; #641, #629 and #617 remain open.

## Validation

Counting starts before owned group construction and continues through returned
output, so releasing an untracked input allocation cannot hide a peak. Borrowed
synthetic sources are created before counting; an allocation-free transcript
isolates the engine. Separate measurements exercise the actual padded value
evaluator at and around its 1,024-entry block boundary.

The same fixtures run separately through a real transcript and independent
`tv-core` tree checks, with the low-level driver's tree prefix renamed on the prover.
Every remaining key/value/forest leaf claim is checked against direct evaluations
in `tv-field`, and final transcript states agree. This tests the component; full
grouped NI/Live acceptance and tamper controls remain part of the CPU driver suites.
