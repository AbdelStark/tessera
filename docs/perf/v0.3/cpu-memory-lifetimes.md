# CPU proof memory lifetimes

This is the implementation inventory for [#629](https://github.com/AbdelStark/tessera/issues/629),
under [RFC-0018's memory scheduler](../../rfcs/RFC-0018-gpu-prover-runtime.md#memory-scheduler).
It is not a replacement allocation bound or a real-model proof qualification.

The CPU driver still keeps the complete `Witness`. Its planner still charges
64 bytes per logical padded cell, plus the separate weight reservation and
additional stage/table working sets. A smaller measured RSS does not justify
reducing that reservation. Caller-owned model storage is outside `MemoryBudget.host`
and must also appear in any whole-process measurement.

| Phase | Storage that must be accounted for |
|---|---|
| Input and preparation | Caller-owned decoded INT8 model and input read/decode overlap; public execution metadata; regenerated weight commitment trees and construction scratch |
| Planning | Lowering, range analysis, final selection, runtime catalog, layer/leaf/native plans, advice tile placement and stage/table plans |
| P1 witness | All graph inputs, including every weight widened to `i64`, and all intermediate tensors in the current resident engine |
| P1 commitment | Retained witness, one source tile, encoder tables, codeword block, column chains, segment digests and owned opening layout/tree |
| P2 source construction | Resident witness, padded lookup source columns and Softmax forest leaves in `ModelWitness`; every used generator core and canonical count source |
| P2 proof | The complete [table batch](table-batch-memory.md), resident lookup/forest operands, retained group proofs and merge state; table sources are freed before P3 |
| Native construction | Resident witness and `ModelWitness` coexist with the native prover's owned tables until construction finishes |
| P3/P3b | Resident witness, native operands, stage grids/folds and retained proof messages; window regeneration currently creates another complete witness |
| Original opening targets | Every original source, coefficient, point and weight obligation is copied into owned `ModelOpenings` before stage/native working sets are freed |
| P4 advice | Resident witness/source tile, encoder, target/group metadata, row block, sampled column, `(G+1)*(k+P)` clearing extension-field accumulators, and all public combined-row outputs during the final copy |
| P4 weights | Retained advice outputs and targets, one matrix's regenerated fold, committed weight trees and accumulated weight messages |
| P5 | Resident witness/source tile, encoder, retained commitment trees, distinct queried columns times all committed rows, public output copies, weight answers and multiproofs |
| Packet assembly | Retained proof messages and their temporary encodings, including metrics' proof-size encodings |

`ModelWitness` is freed after native construction: native tables own the leaf
inputs they need, and reduction stages read the original witness. Once
`ModelOpenings::derive` has transferred every opening obligation, the stage and
native working sets are freed before P4 begins. Neither release removes an
opening target, changes a challenge, or changes a committed source row.

## Tensor regeneration

`Regenerator::tensor` supplies one real, row-major integer tensor from the
checkpoint source. Computed tensors transfer their clearing allocation out of
the replay; inputs and checkpoints come from the same readers as tile replay.
A weight request widens only that matrix. No replayed tensor remains cached in
the source after the call.

This interface requests a whole tensor. Requesting scores can materialize their
full quadratic domain; requesting a weight can allocate a full `i64` matrix.
The result size is not its replay cone's peak. Reduction consumers still need
bounded tensor leases or chunked reads and a public bound on simultaneous
working sets before this interface can replace the resident driver.

`CheckpointTensors` supplies the common integer tensor contract used by tile and
reduction consumers. It borrows model matrices in their original INT8 allocations
and owns every nonweight tensor in clearing storage, including complete score
tensors. Representation availability and selection use public graph/source metadata,
never secret values (INV-ARCH-2). Its `storage_bytes` reports retained cache storage only: construction
scratch, the checkpoint store, canonical tokens, engine metadata and the borrowed
model remain separate allocations. This component does not change CPU admission.

`Engine::canonical_batch_tokens` owns a clearing copy of the canonical packed
token sequence, including zero slot gaps. A future checkpoint driver can retain
that copy after the caller releases its batch. Construction temporarily holds
both the engine's `i64` token input and the new `u32` copy; both belong in its
phase bound.

## Public opening geometry

`OpeningGeometry::derive` counts the current Protocol One CPU producer's original
advice combinations, source pieces, expanded images and point coordinates, and
every matrix's original claims and coordinates. It also bounds the distinct
advice column points before challenges are sampled. Inputs are the canonical
public lowering, runtime table catalog and leaf/native/advice/weight layouts for
one proof unit; the computation reads no witness or model weights.

The geometry and actual `ModelOpenings` share the original-source traversal.
Coordinate maps accept opaque `Copy` values: they copy and reorder coordinates,
insert Boolean constants, and retain all attention-share images. They cannot do
field arithmetic on a coordinate label. The actual producer separately computes
the original coefficients, including fixed-share weights. Neither numerical zero
coefficients nor repeated ChannelParams reads remove an opening obligation.

Ordinary leaf sites share their common leaf-point coordinates. Grouped sites use
the suffix of the global merge point in RFC-0054's original-variable order.
Equal-height table instances share their global table-batch point; different
heights keep separate origins, which may overcount. Two distinct origins can
evaluate equally in the field, so the symbolic column-group count is an upper
bound. A memory plan must use that upper bound rather than assume a collision.

For message length `k`, mask length `P` and at most `G` column groups, the
private accumulator and public combined-row payloads coexist during the final
copy and require `32*(G+1)*(k+P)` bytes of extension-field payload alone.
Encoder tables, row blocks, self-check scratch, targets/groups, container
capacities, retained messages and every other phase remain separate charges.
These counts are preparation metadata, not a complete allocation or soundness
bound. The CPU driver and its conservative resident-cell floor are unchanged.

## Generated table cores

`GeneratedTable::core_storage_len` derives the number of materialized values from
public table parameters without allocating a core. `Exp` retains the prefix before
its zero tail, `Exp2Frac` retains its value column, and `Silu` retains each class's
reduced integer domain in class order. Arithmetic kinds store no dense values even
when the digest logically enumerates a value column (RFC-0008).

Generation reserves the complete core before writing values and fills it in canonical
order without speculative buffer growth. A core that cannot fit the target address
space or be reserved returns `TableError::Memory`. Compiler, gadget, witness, leaf,
padded-stage, native and CPU error paths report that resource failure as
`ERR-RES-MEMORY`; malformed parameters and digests keep their existing classifications.
Resource failure does not choose a different canonical model family.

These counts are not a complete table or proof reservation. Descriptor and class
capacities, cloned specifications, digest work, engine registries, count backing,
constructor overlap, allocator bookkeeping and every other phase remain separate.
Allocation tests measure requested bytes, not whole-process RSS. The CPU resident
floor and admission rule remain unchanged; real-model qualification is still owed.

## Evidence required before changing admission

The tensor tests compare every graph tensor with the existing resident engine,
including graph inputs, checkpoints and native intermediates. Current-thread
allocation counters run the engine on one thread and check that only the returned
allocation survives; a weight request must allocate only one matrix. Invalid
tensor identifiers fail before allocation.

Whole-proof comparisons must preserve complete deterministic artifacts, independent
NI/Live acceptance and exact tamper rejection under ordinary, grouped and windowed
schedules. Component measurements cannot stand in for a complete phase bound:
public group/query bounds, all retained metadata, constructor scratch, output-copy
peaks and clearing must also be included. The real NI-100 1B proof, full runtime
soundness budget and the remaining #617 requirements are still owed.
