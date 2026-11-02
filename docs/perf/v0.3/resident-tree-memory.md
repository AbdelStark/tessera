# Resident tree-batch memory

`ResidentTreeMemory::new` prices canonical `FractionTree::new` construction and
`prove_trees_in` from the ordered public `ResidentTreeShape` list. A shape gives
a fraction height and leaf numerator mode, or a forest height and tree-index
variables. Planning allocates nothing and reads no field value, witness or
transcript challenge. Repeated shapes remain separate owners.

This component implements [RFC-0010's tree levels](../../rfcs/RFC-0010-proof-schedule.md#tree-levels)
and [RFC-0018's memory scheduler](../../rfcs/RFC-0018-gpu-prover-runtime.md#memory-scheduler)
for the resident group kernel of [RFC-0054](../../rfcs/RFC-0054-token-groups.md#the-leaf-stage-with-groups).
It is a prerequisite for the full grouped physical route under
[#641](https://github.com/AbdelStark/tessera/issues/641); it does not enable that route.

## Owners and overlap

`fraction_bytes` includes every canonical fraction-level descriptor, the full
extension-field denominator leaves, explicit base numerator leaves when present,
and both extension numerator and denominator arrays of all interior levels.
Height-zero fractions have an inline root and allocate no levels. The sum also
bounds sequential construction: each constructor fills its final owned arrays
without an extra full-table temporary. Borrowed source data remains additional.

`peak_bytes` includes that fraction storage beside all forest level descriptors
and owned base-field arrays. Forest construction copies borrowed leaves into the
owned even/odd layout. The bound conservatively retains every level through the
proof rather than relying on the engine's earlier level releases.

Each batch level is derived from the actual join rule. Its kernel adds all active
split states, descriptors, folded extension operands at their original capacities,
index/point metadata, messages and final evaluations. A fraction is charged four
operands even when its unit numerators save two in the final level. Equality
scratch includes the low/high tables and the fraction kernel's scaled low table;
it also covers a joining forest's root evaluation.

All returned levels coexist in `output_bytes`: round vectors and messages, roots'
challenges, level points, tree indices, carried claims and their points, plus final
leaf claims. Iterator collection can retain a larger state allocation when mapping
to claims; the output charge allows that capacity. The proving peak includes this
complete output beside current kernel output and transition scratch. Dynamic
group/level/round labels include old/new String growth under pinned Rust 1.95.

## Additional owners

The following remain additional throughout every phase in which they are live:

- Borrowed forest leaves, base numerator data and denominator-source/callback work.
- The consumed input tree vector and its spare capacity, including any inherited
  capacity retained by iterator collection in output.
- Transcript/encoding callbacks, caller group layout/lookup columns/forest shares,
  table/count sources, group splits, merge blocks and retained merge tables.
- Stack, allocator bookkeeping, runtime/process overhead and caller model storage.

The [two-pass merge model](merge-memory-bound.md) prices a later component. A
complete driver must compose both models with all surrounding owners and actual
lifetimes before admitting grouping. These counts authenticate no caller geometry
and establish neither a complete phase bound nor real-model resource qualification.

## Validation

Counting-allocation controls separately measure canonical fraction construction,
complete construction/proving and retained output. The proving measurement starts
before constructing its owned trees, so freeing an input allocation cannot hide a
peak behind a negative counter. An allocation-free transcript isolates kernel
storage; the same synthetic sources separately pass the independent `tv-core`
tree-level verifier and transcript driver.

Cases cover empty batches, zero-height roots, both numerator modes, height-one and
stacked forests, unequal heights and interleaved/reversed schedules. A large
height-one forest keeps its owned base arrays live beside folded extension buffers,
so freed parent levels cannot hide an omitted folded-buffer charge. Typed invalid
and unaddressable cases allocate nothing. On 64-bit targets, 512 public shapes of
height 50 force aggregate overflow without constructing a field table.
