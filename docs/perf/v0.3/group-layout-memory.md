# Grouped layout construction

`GroupedLayout::construction_memory` and `group_memory` bound requested
heap storage for the public layouts of [RFC-0054](../../rfcs/RFC-0054-token-groups.md#groups)
under [RFC-0018](../../rfcs/RFC-0018-gpu-prover-runtime.md#memory-scheduler).
Both calculators allocate nothing and read no tensor value or transcript challenge.
They return `GroupLayoutMemory`: `output_bytes` prices retained capacities and
`peak_bytes` adds temporary construction owners. Checked overflow
or an unaddressable array request returns `LeafError::Memory`.

## Grouping construction

The constructor bound includes all nested fixed-axis arrays, polynomial descriptors,
group-variable vectors, coordinate buffers, shares, blocks and both read maps.
It also includes held-read flags, the first coordinate pair retained while another
lookup's trial pair is checked, and coordinate-vector reallocation overlap.
Coordinate buffers retain their original capacity after group variables are removed.

Shares and the two read maps reserve their exact public counts before filling.
The grouping caches the stage's sorted table identifiers. That small index is
included in both the construction bound and `heap_bytes`; per-group sizing can
then visit known tables without allocating a temporary index. The constructor's
rules and ordered layout contents are unchanged.

This bound sizes a supplied stage/grouping; it does not validate the grouping or
authenticate the caller's pairing of inputs. `GroupedLayout::new` retains its
existing validation. The borrowed stage and layer map remain additional.
The compiler's term types have allocation-free equality. A caller supplying a
custom `T::eq` must price any allocation or other callback work it introduces;
the metadata calculator does not bound arbitrary user code.

## One group's construction

The per-group bound derives the actual held blocks for each cached table and the
reads of the selected token/layer subsets. It includes read and coordinate clones,
per-table block-vector capacities, instance descriptors, table-index nodes,
insertion/iteration overlap, stable-sort scratch and vector growth overlap.
Blocks move from the table index into instances without duplicating their buffers.
An invalid group has the same error as `group`.

Vector minimum capacities/doubling and B-tree node sizes follow the pinned Rust
1.95 implementation. The B-tree bound permits eleven keys/values, twelve edges,
parent/length metadata, both node alignments and transient splits. It assumes no
favorable allocation reuse. These are requested bytes; allocator bookkeeping,
stack, inline objects and runtime overhead remain additional.

## Scope and controls

Counting starts before each constructor and continues while its output is live.
Inputs predate counting and are not released inside it. Planning is checked for
zero allocations. Scalar reads isolate held flags without coordinate scratch or
share growth. Other controls cover long coordinates, repeated lookup trials,
many tables, unsorted block sizes, empty/absent groups, token/layer grouping and
unequal layer subsets. Actual standard-library vector capacities independently
check the growth assumptions; arithmetic overflow controls allocate nothing.

Twenty-one separately compiled faults are assertion-caught, with two passing
eight-case controls: cached table ids; all three fixed-axis nesting levels;
share blocks/read maps/descriptors; held flags; retained coordinates/group variables,
trial pairs and polynomial/coordinate descriptors; group read descriptors/points,
block buffers and instance descriptors; block work, table-index nodes, old vector
capacity and invalid-group refusal. Each run starts from the same frozen source,
which is restored after the campaign. These controls concern metadata allocation,
not actual-model acceptance or a whole-proof budget.

This prices public layout construction only. Borrowed grouping/stage data,
lookup columns, forest shares, split evaluation, retained group proofs,
[lookup-group proving](lookup-group-memory.md), merge tables/work, table batches,
CPU wire/native/opening phases and packets still require full composition.
Both grouped-physical guards remain in place. This establishes no actual-model
Accept, full proof memory/time qualification, fidelity, formal hiding or GPU claim;
[#641](https://github.com/AbdelStark/tessera/issues/641),
[#629](https://github.com/AbdelStark/tessera/issues/629) and
[#617](https://github.com/AbdelStark/tessera/issues/617) remain open.
