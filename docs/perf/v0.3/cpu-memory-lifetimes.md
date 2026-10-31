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
| Packet assembly | Retained proof messages, encoder-owned scratch and the bounded envelope writer; manifest metadata remains separate |

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

## Multiplicity backing

`MultiplicityPlan` chooses dense counts or fixed weighted updates from public table
rows and an update bound. It uses `min(8*padded_rows, 16*max_updates)` bytes of backing
payload; a tie uses dense counts. The actual constructor uses this same plan and
checks target addressability before reserving storage. Weighted entries remain
fully initialized, including unused entries; secret sparsity never changes storage
or traversal length (INV-ARCH-2, OBS-6).

`Engine::runtime_multiplicity_plans` derives the used tables, row counts and update
bounds through the existing canonical gadget/runtime builders. Bounds include every
lookup occurrence, per-dimension padding contribution, active request's padded
runtime keys and the final row-0 tail (RFC-0008/0045/0071). Grouping changes the tail's
weight, not its number of stored contributions. This method reads no weights,
tokens or witness values. Its architecture, engine and runtime must describe the
same canonical proof unit.

The method builds temporary public metadata and may prepare gadget cores. Those
allocations, retained engine cores, descriptors and constructor overlap remain
separate charges. A backing plan authenticates neither arbitrary supplied metadata
nor a host budget. CPU admission and the resident-cell floor remain unchanged.

## Proof-byte metrics

The CPU driver counts existing TCE encoders' writes into a private size sink for
Round1, the same sumcheck/opening parts of Round2, and Round3. It retains no encoded
output buffer for those measurements. Labels and part membership stay the same.
The metrics-only sum saturates at `u64::MAX` and cannot change session state or
serve as an allocation bound. Allocations inside an encoder, retained proof
messages and final packet publication remain separate costs.

## Generated table read chunks

The P2 planner first selects the existing 256-row reference plan. It retains that
plan's window cuts and uses spare public allowance for a larger power-of-two read
chunk, up to 65,536 rows. Each candidate charges the full source scratch and must
fit beside the retained maximum window working set. If none fits, the original
chunk remains feasible. Repeated selection can shrink a previous large chunk.
The charged nonzero chunk is passed to the table engine for ordinary and grouped
batches. The conservative resident-cell floor and all source backing stay charged.

For each generated table, scratch uses
`128*min(chunk,table_rows) + 32*(height+1) + 512` bytes: paired extension chunks,
three base chunks, clearing count scratch, partner outputs, carry stack and
container descriptors. The count reader's public fixed-capacity traversal is
unchanged; no secret sparsity or sorted keys affect chunk selection. The standalone
[count-reader benchmark](../../../crates/tessera-prover/benches/count-reads/README.md)
records configuration measurements separately from whole-prover performance.

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

## Packet publication

The NI CLI proof and producer Live messages use the existing streaming TCE envelope
writer. Its first pass sizes the header, and its second pass writes and hashes the
same encoding through a clearing buffer of at most 64 KiB. The message remains live;
there is no complete encoded-envelope allocation beside it. Any allocation an
encoder itself owns remains separate from this output-buffer bound.

Directory publication still creates temporary files exclusively with mode 0600,
finishes and syncs the file, and hard-links it to a fresh final path. Existing files
and symlinks cause the same usage error. Failure removes the unpublished temporary.
The envelope writer's rename-based file helper is not used because its replacement
semantics would violate this no-overwrite contract. Manifest artifact metadata uses
the existing bounded 32-byte trailer read and preserves secret-digest redaction.

Retained proof objects, filesystem and manifest bookkeeping, input preparation and
all earlier phases still require their own charges. This buffer removal changes
neither CPU admission nor the unresolved complete cached-route phase bound.

## Advice opening allocation plan

`AdviceOpeningMemory` derives the requested live heap storage owned by grouping
and the complete P4 pass from public layout, commitment parameters and upper
claim/image/group counts. It retains every image, including zero-weight ones.
Counts must cover the actual canonical obligations; this helper does not
authenticate arbitrary supplied metadata or admit a CPU source route.

The grouping peak charges the original-claim-sized descriptor and power arrays,
column points, all image descriptors/row points, and the proximity point. Repeated
claims can grow one group's descriptor vector; its old and new arrays coexist
during reallocation. A full extra image-descriptor array bounds that overlap.

P4 charges private accumulators and their containers, the row/mask block, codeword,
samples, row numbers and self-check sums. Flush scratch includes coefficients, their
raw word copy, clearing wide accumulators, and equality factors/partial products.
Self-check charges both inversion vectors; publication charges retained weights
and every public combined row while private accumulators remain live. The maximum
uses these verified lifetimes rather than assuming unrelated scratch coexists.

Physical allocation tests cover short and larger rows, repeated/grouped/zero-weight
images and output-copy overlap. A separate public synthetic layout matching the
1B diagnostic's row width/arity gives 8,830,068,048 bytes for these PCS-owned
allocations without allocating large payloads. This is not whole-model memory
evidence. Caller claims, layout/P1 tree, encoder, source row, mask generator,
allocator overhead and other proof phases remain separately charged.

## Advice column allocation plan

`AdviceColumnMemory` bounds PCS-owned P5 storage before column queries are sampled.
It charges the original draw-sized query array, private column containers and values,
one codeword, both Merkle index frontiers, proof siblings, and simultaneous private
and public column copies. For N columns, height h and Q draws, it uses at most
`min(N - 1, min(Q, N) * h)` siblings. A bound based on `N - min(Q, N)` would fail
when draws repeat: fewer actual distinct leaves can require more siblings.

After indices exist, Merkle opening reserves the exact canonical sibling count from
adjacent sorted distinct indices. The unchanged level traversal determines digest
order. Contiguous intervals use their dyadic prefix/suffix complements; validated
coverage of every leaf returns the canonical empty proof. No query is drawn earlier
and no favorable collision reduces the preflight charge. The allocation benchmark's
[cardinality argument and samples](../../../crates/tessera-hash/benches/openings/README.md)
record the proof and the measured effect.

Physical tests cover singleton/boundary, repeated, unsorted, sparse and all-leaf
queries at several row widths, including private/public output overlap. The public
synthetic 1B-width fixture with 193 draws plans 94,306,424 bytes for P5-owned storage
without allocating large buffers. Caller P1/layout, encoder, mask/source row, retained
Round2, allocator overhead and other phases remain additional. Complete CPU admission
and real-model proof qualification remain pending.

## Public layout retention

`AdviceLayout::heap_bytes` counts the capacities of its placement, tile and segment
arrays, slot lengths, dense axes/order and all nonzero-row runs. The cloned layout
retained by P1 allocates each vector at its length, so the original footprint also
bounds that copy. Count both while they coexist. Physical allocation tests cover
200 generated layouts plus slotted batches, originals and clones, including spare
capacity. The helper allocates nothing and returns `None` on checked-count overflow.

This is retained requested heap only. Inline structs, caller tile shapes, constructor
scratch and allocator overhead remain separate; the helper does not admit a CPU route.

## Sequential advice commitment allocation plan

`AdviceCommitmentMemory` covers encoder construction and PCS storage for a
sequential P1 whose caller exactly reserves its segment-descriptor array, as the
CPU driver now does. It charges all segment chains (including empty segments),
the largest active codeword block, the merged tree while every chain remains
live, and the simultaneous owned layout clone. The encoder owns inverse/forward
NTT tables and spread factors; its constructor also holds a temporary powers
array beside the table being built. Phase maxima follow those lifetimes.

Physical tests cover multiple segments, different row counts, empty segments,
partial blocks, private-buffer clearing and publication. The synthetic 1B-width
fixture plans 2,499,805,280 requested bytes for these allocations without creating
large buffers. Caller original layout/model-advice metadata, source row, masks,
checkpoint/tile generation, allocator overhead and other phases remain separate.
Neither this helper nor exact descriptor reservation changes CPU admission.

## Checkpoint and tensor-cache constructor buffers

`Engine::checkpoint_memory` plans canonical u32 token conversion, INT32 checkpoint
storage, and every nonweight i64 tensor in `CheckpointTensors`, including full scores.
It uses the engine's validated public lowering without reading a model/token or
allocating. Graph-input INT8 weights stay borrowed. Counts include both cache
metadata arrays, checkpoint ids/containers, repeated leaf-input copies, INT64/INT32
checkpoint overlap, and all public reader/operand/reachability metadata. Cached
outputs move from `Live`; their union is charged once. The temporary public
`positions_values` vector coexists with its clearing copy and has its own charge.

Reader edges, attention cores and operand descriptors reserve their public counts
exactly. Replay closes a requested cone in reverse node order, using RFC-0006's
validated topological order and stopping at checkpoints (INV-ARCH-3); it needs no
DFS worklist. Independent set-frontier tests compare all singleton, repeated and
combined requests, including checkpoints. Physical tests cover normal and slotted
construction, exact retained storage, intermediate overlap and zero-allocation
refusals. Address and multiplication boundaries are isolated in allocator-free tests.

These are requested buffers owned by these constructors. Borrowed engine/core tables,
graph/model data, thread-runtime/stack storage, tile/gadget/native work, allocator
and instrumentation overhead, input preparation and other proof phases still need
separate charges. This helper does not change CPU admission or qualify a real-model
whole-process budget. Physical measurements run on one engine thread so the counting
allocator observes every buffer allocation under test.

## Engine table loading

`Engine::table_memory` plans the sequential table loader's requested buffers
without building a core or reading model data. It charges the full id-indexed
registry, cloned class vectors, dense lookup cores and RoPE's `d_head/2` stored
`u128` frequencies. RoPE computes position pairs on demand; its logical pair
count is not retained storage. The registry and frequencies reserve exact capacities.
The table digest's one `u64` Exp zero-tail piece coexists with the loaded cores.

All supplied payloads are summed, including duplicate ids: replacement constructs
the new table while the previous table remains live. This conservatively bounds
both retained storage and replacement overlap. Graph validation, thread runtime,
allocator bookkeeping, borrowed specifications and every other proof phase remain
separate. This helper authenticates neither table parameters nor their digests and
does not change CPU admission.


## Padded model lookup sources

`ModelWitness::memory` plans its owned padded lookup columns, public domain arrays,
Softmax maximum columns, product-forest leaves and optional packed greedy maximum.
Each ordered read selects its axes from the canonical lookup request. Per-axis
padding is charged in full; source values and secret sparsity never change the
plan. A Softmax forest charges its key column times the padded layer count, with
no forest at key height zero. Runtime maximum and keys remain distinct sources.

Nested site/layer/column descriptors, forests, greedy column indices and borrowed
request descriptors reserve exact capacities. Own constructor scratch includes
the public ranges used during scatter, greedy indices and request descriptors.
Every product, sum and requested array checks overflow and target addressability.
The planner allocates nothing and authenticates none of its supplied metadata.

The canonical source's simultaneously live tile, gadget cores and construction
work are additional. Borrowed engine/model/graph, lent view arrays, subsequent
P2/native/window state, runtime and allocator overhead also remain separate.
Physical tests assert exact retained bytes and compare the observed constructor
peak with this owned bound plus a separately measured fixture source peak. That
fixture measurement is not a public tile bound and cannot admit the real CPU route.

## Retained source and gadget templates

`Requant::heap_bytes`, `RsqrtRow::heap_bytes` and `SoftmaxRow::heap_bytes`
count their requested owned template heap without allocating or reading witness
values. Inline descriptors are excluded. Wide requantization includes its box and
every owned table role, even when two roles have the same table id. Root templates
include both comparisons' bound/carry vectors and table capacities. Softmax includes
its dense Exp core. Cloned templates are measured separately; these counts do not
predict constructor scratch or scalar witness generation.

`Resident::metadata_bytes` and `Regenerator::metadata_bytes` count retained
structure arrays, templates, boxed auxiliary descriptors, nested constraint forms,
request/read/row arrays, runtime key columns and grouped-column capacity. The
regenerator borrows its tokens and checkpoints; the resident source borrows its
tensors. Neither owns another copy of those buffers. Physical allocation tests
compare the counts with constructor retention for every conformance model, ordinary
and slotted runtime sources, and an empty grouping vector with spare capacity.

These are retained requested bytes, measured after construction. Borrowed inputs,
constructor temporaries, each returned tile, replay/gadget witness work, stack and
allocator/runtime overhead still require separate charges. The complete phase bound,
CPU admission change and real-model proof remain pending.

## Computed score inputs

`ModelNativePlan::score_input_memory` plans the requested clearing buffers used
by the default native constructor's RFC-0056 score inputs without allocating or
reading witness values. Each family includes raw INT8 queries/compact GQA keys,
per-axis padded copies, and the copied base-field maximum. A short context does
not cap the query/key domain: its head dimension may exceed the position count.
Slotted batches retain their existing literal route and have no computed inputs.
The explicit literal oracle is separate from this plan.

The constructor reserves each raw buffer exactly and clears it immediately after
its padded copy is built. For raw buffers rq/rk, padded buffers pq/pk and maximum
m (all in bytes), the family peak is
`pq + max(rq + rk, rk + pk, pk + m)` and retention is `pq + pk + m`.
Sequential families retain earlier inputs while constructing the next. No input
is deduplicated by its values. Packing tests independently count requested bytes,
compare every real coordinate with the engine and check zero padded coordinates.

Arc/container metadata, original maximum tables, all other native operands,
equality/runtime factors, constructor scratch outside this packing step, stage
rounds/windows, borrowed data and runtime/allocator overhead remain additional.
This component does not qualify native construction as a whole, change CPU
admission or demonstrate a real-model proof.

## Source metadata construction

`TemplateMemory` bounds requested allocation bytes for public gadget construction
and full Goldilocks constraints. Each associated `memory` method allocates nothing.
These bounds sum requests, including replaced allocations, so they also cover peak
and retained heap without relying on deallocation timing. They exclude borrowed
specs, inline template descriptors, table-list extraction and secret witnesses.

Root geometry has at most 32 root bits, four root bytes and 15 product limbs.
Its largest schema has 187 committed columns, 189 lookups and 97 relations.
Only one normalization side per bracket expands the product: every `(a,b,k)`
appears at one coefficient, giving at most `2*4*4*4 = 128` product terms across
all coefficients. Charges include temporary NonNeg constraints used to recover
high affine digits, intermediate comparison metadata, and both the product result
vectors and the final relation vectors. Tests cover every root width.

Requant charges its wide box, owned classes and temporary expected-sign class.
The widest constraint form owns at most 24 affine pairs and requests 18 more
construction pairs. Softmax derives its Exp core without materializing it, includes
the digest's zero-tail piece, and covers all 49 possible NonNeg limbs, including
99 committed columns, 100 lookups and seven relations.

Growing-vector charges use the pinned Rust policy: capacity doubles or satisfies
the larger requested length, with minimum capacities 8/4/1 by element size. Summed
requests are bounded by four times the larger of that minimum and the maximum
length reached, including lengths before deduplication. Literal/exact arrays have
their own charges. Allocation controls test the element-size regimes and isolated
address/product overflow.

`Engine::source_metadata_memory` composes structure arrays, reader edges, source
containers, ordered reads/requests and all gadget construction. It includes table
list/row extraction, boxed auxiliary descriptors and lookup-id validation arrays.
Every Requant-named table payload is additionally charged in lookup-role order:
a malformed role can generate a large core before its constructor rejects the kind
or digest. Repeated ids do not remove an owned-role charge. A public malformed
Exp fixture pins this charge at 128 MiB without generating the core.

This is resident source construction metadata. Runtime range validation/key columns,
grouped-column transfers, regenerator input validation, scalar/tile generation,
borrowed inputs and allocator/runtime overhead remain separate. Current-thread
allocation tests cover resident and cached tensor sources. No component here
changes CPU admission or qualifies a real-model proof.

## Source runtime and input validation

`Graph::validation_memory_bytes` includes producer ids, per-node borrowed-view
vectors and the operation checks' temporary shape vectors. `analysis_memory_bytes`
adds ranges, analysis producers, per-node output ranges, committed-input coverage
headers and every allocating reverse move. A `BitDecompose` reader follows its own
move chain; repeated readers of one chain each allocate their own bitsets. Full
transpose/expand moves avoid a bitset only while coverage is still `All`. Partial
coverage counts each input element as a byte, plus both coordinate-index vectors.
The sums include replaced allocations and the returned ranges. They are public,
allocation-free size calculations for valid graphs, not graph or range validators.

`Engine::runtime_source_metadata_memory` adds that analysis, possible read-vector
growth and the copied greedy key-column descriptors to source construction.
A runtime without a certificate adds no runtime allocation. Range/input validation
still runs in the original order and retains its exact errors.
`source_replay_validation_memory` separately charges the regenerator's input
checks: clearing token/channel copies and both the public and clearing position
vectors, once per input occurrence, plus the checkpoint-shape check's temporary
id and residual-buffer descriptor vectors. INT8 weights remain borrowed. The bound is
for canonical lowering shapes and excludes actual replay execution.

`SourceMetadataMemory::with_grouped_columns` includes one caller-owned column
vector transfer at its supplied capacity and its temporary sorted table-id copy
at initialized length. Spare capacity counts; every repeated transfer requires
another charge. These constructor/request sums exclude borrowed input storage,
checkpoint/cache payloads, returned tile/scalar-gadget work, stack/runtime and
allocator overhead, and the remaining proof phases. They do not admit the cached
CPU route or replace the resident-cell floor.

## Resident and cached tile work

`Requant::witness_memory`, `RsqrtRow::witness_memory` and
`SoftmaxRow::witness_memory` bound requested bytes of honest witness construction,
including returned clearing columns and public descriptors. The root bound also
covers comparison constraints, temporary product/constant limb buffers, coefficient
polynomials and the later `into_tensors` transfer. Its polynomial work is a subset
of the two bracket expansions and comparisons already bounded for full constraints.
`SoftmaxRow::evaluation_memory` includes its complete Goldilocks constraints,
product leaves, original lookup columns and grouped concatenation. Reciprocal and slack
lookups can share a Range table; growing their shared scalar key buffer is charged.
Only the Exp lookup has position keys/values; its group moves both buffers.
The bounds exclude borrowed templates/inputs, counting, stack/runtime and allocator
overhead, and use the pinned compiler's allocation-request growth bound.

`Engine::source_tile_work_memory` sums public work over all graph nodes to bound
any one resident/cached tile. A table tile can visit every contributing node;
attention recounts run one row at a time. The sum includes output copies,
input/output/share descriptors and coordinate ranges, returned lookup/auxiliary
columns, scalar gadget work, copied class vectors and Requant padding witnesses.
Softmax rows charge their temporary padded row and complete evaluation, including
outer tile descriptors for row callbacks and empty attention cuts.
`runtime_source_tile_work_memory` additionally charges the maximum and all padded
runtime key columns and descriptors.

A table tile's canonical `MultiplicityPlan::payload_bytes` is a separate charge;
this payload is not hidden inside the work sum. Both helpers allocate nothing and
read no witness values. They assume valid canonical lowering/gadget metadata and
canonical resident/cache borrowers; arbitrary TensorSource callback allocations,
replay execution, source construction, retained input/template data, encoding,
stack/runtime and allocator overhead remain outside their scope. The helpers do
not admit the cached CPU route or replace the resident-cell floor.

## Owned native construction and preparation

`ModelNativePlan::memory(lowering, literal_scores)` derives three allocation-free
bounds from matching public metadata. `construction_peak_bytes` covers retained
native storage and constructor overlap; `prepared_peak_bytes` includes one initial
prepare; `instance_bytes` adds the metadata requests of one instance per tag.
The methods cover the ordinary computed-score representation and the literal
oracle, and slotted batches always select the existing literal representation.

Construction includes every owned raw field table, padding/causal mask, table/data
container, coefficient/factor vector, fixed/point vector and Arc descriptor. Roots
charge their original root/input columns, byte/top columns and residual table;
Softmax retains its original maximum, reciprocal and average, plus either literal
score/broadcast tables or the existing computed input peak. Integer/evaluation
scratch uses equality halves or blocked evaluation storage, rather than charging
a full quadratic table for the computed path. Canonical grouped point maps and
token-axis analysis are included; arbitrary callback allocations are excluded.

Preparation charges equality vectors and cache growth without assuming coincident
Fiat–Shamir points. Runtime relations include time/vocabulary/output equality
scratch, public coefficient vectors and term construction. Observe creates and
clears two coefficient arrays even though it retains no public factors. The
pinned hash-table estimate includes bucket/control/alignment requests throughout
growth; allocation tests check it with distinct public keys. Instance metadata
borrows tables and clones their term/factor descriptions; computed instances
create no such borrowed-table vectors.

These bounds exclude the already-owned public plan/lowering, leaf/model witnesses,
sources, sumcheck/computed-round state, windows, claims/openings, stack/runtime and
allocator overhead. They authenticate no input and do not admit a cached CPU
route. The remaining phase/lifetime bound and real-model execution are still owed.


## Computed score round state

`ModelNativePlan::score_round_memory_bytes` charges both key and forest states
for every Softmax family without assuming they run at different times. It uses
matching public lowering geometry and allocates nothing. Slotted batches retain
the literal representation and contribute zero computed round storage.

One state charges its box and the growing point across all own variables. A key
round additionally needs a cloned/extended low point, at most `Np` equality
weights and two dimension-sized clearing folds. The transition after the last
key binding allocates four full padded `[Lp, Hp, Np]` extension-field row arrays,
the key equality weights and one dimension-sized fold. Row binding truncates
these arrays without reducing their capacity; the bound retains their full size.
The larger scratch phase is added to each state's persistent storage, and every
state's bound is summed.

Allocator assertions wrap start, each round and each bind in the existing
literal/computed message differential, including padded layers/heads/positions,
single-position contexts and dimensions large enough for key scratch to dominate.
Retained native inputs, generic sumcheck wrappers/messages, output vectors,
windows, stack/runtime and allocator overhead remain separate. No CPU route is
admitted by this helper.


## Regenerated weight work

`WeightLayout::regenerated_work_memory(row_block, draws, channel_claims)` prices
protocol/1 work without allocating or reading weights. The commitment retains ten
matrix-type Merkle trees, a cloned layout and its tree descriptors. Constructor
work includes nested canonical-model validation, row-reader metadata, encoder
tables/construction powers, one row and encoded block, and column chains. Tree
growth retains the old leaves while requesting the final node array.

The model-reference count includes the two global tensors and seven per-layer
types, plus class/multiplier references. It borrows their payloads. Fold work
includes committed-row indices, all evaluation/proximity coefficient vectors,
public equality factors/partials, clearing folds/proximity/row buffers, the
sumcheck input and copied combined output. Every ChannelParams claim is counted,
including repeated points and zero combinations. Row readers are canonical;
arbitrary MatrixRows callback allocations remain separate.

Column answers charge the full draw-vector capacity, at most `min(draws, N)`
candidate columns, one regenerated matrix tree, encoding work, canonical sibling
storage, proof-index/self-check pairs and public output copies. The sibling bound
is `min(N - 1, min(draws, N) * log2(N))`; it assumes no favorable query collisions.
The returned fold/answer peaks cover one matrix at a time. Previously retained
combined rows or answers are added separately using the all-matrix output bounds,
as are the persistent commitment and caller-owned model/layout payloads.

This component excludes input/layout/state decoding, original opening targets,
generic proof/message wrappers, stack/runtime and allocator overhead. The complete
CPU phase/lifetime bound and cached-route admission remain separate work.


## Original opening targets

`OpeningGeometry::target_memory(lowering, leaf, native)` derives allocation-free
bounds for the original `ModelOpenings` producer. `retained_bytes` includes all
advice-claim descriptors, mapped pieces and point capacities, and all ten weight
point lists. Each separately grown vector gets its own minimum-capacity slack;
repeated points, zero coefficients and every ChannelParams obligation remain.

`construction_peak_bytes` adds a conservative superset of the complete native
original list and its point/piece capacities, fixed-site and read lists, stable
sort buffers, Softmax row copies, native round points, pool point/coefficient/layer
scratch, and advice-image projection/axis/attention-shape buffers. The public point
length bound follows graph shapes, pool/atom/coefficient coordinates, leaf reads,
global stage rounds and weight-coordinate counts. It assumes no favorable sampled
coordinate or coefficient value.

These bounds cover canonical leaf/stage callbacks. They exclude borrowed
lowering/layout/leaf/native/stage state, the public geometry planner's own
derivation, PCS grouping/accumulators, generic messages, stack/runtime and allocator
overhead. The geometry planner requires the complete runtime selection's tables,
including the greedy-range table when present; the lowering's initial table list
is insufficient for a runtime proof. Full CPU admission remains separate work.


## Ordinary batched sumcheck storage

`BatchMemory::new::<F>(degree, shapes)` bounds `prove_batch` without reading source
values or allocating. `BatchShape` declares each own-variable count, every literal
table, the sum-of-products term count, sent evaluation count and whether rounds
are delegated. Counts precede relation, table or term-weight deduplication.

The peak includes each half-table binding (or singleton), unread-table evaluation
scratch, state/binding containers, both identity-map types and per-bucket vectors,
compiled term-plan construction, generic kernel and term scratch, transient bound
views, labels, round/evaluation messages and returned point/index metadata. Shared
scratch growth is charged through its maximum public geometry. The pinned B-tree
node bound follows Rust 1.95's key/value/edge/header layout and alignment; tests
measure both actual batch map types. No numeric weight coincidence or pointer
alias reduces the public charge.

`output_bytes` includes the returned proof and `ProvenBatch` metadata. Borrowed
instances/tables and their constructors, transcript callbacks/encodings, delegated
`ComputedState` payloads, stack/runtime and allocator overhead remain additional.
Native score payloads already have their separate public bound. Window-engine and
stage-source/wrapper storage are separate; this helper admits no CPU route.


## Window-engine descriptor and output storage

`WindowMetadataMemory::new::<F>(shapes, terms, degree)` is an allocation-free
additional charge beside the unchanged `WindowMemory::working_bytes`. It counts
the consumed canonical specification array overlapping the prepared states,
per-instance tables/terms/factor/EQ vectors, folded-column containers and private
true sums. The output charge includes every round message, point, final evaluation
and instance/index vector. It assumes no public-column or term cancellation.

The peak adds canonical preparation containers and temporary EQ-point copies,
message lines and Lagrange bases, labels, and a split suffix-EQ pair independent
of preparation order. Each independently grown vector receives its own capacity
slack. The original grid/fold model and its margins remain intact. Tests measure
canonical specification cloning, preparation and proof output together, including
singleton relations whose term/unused-column descriptors dominate their payload.

Caller-owned original specifications and arbitrary spare capacities, source
callbacks and retained/regenerated witness, transcript encodings, delegated stage
wrappers, stack/runtime and allocator overhead remain separate. This helper does
not change proof construction, transcript ordering, memory admission or CPU routing.


## Persistent stage-driver state

`StageStateMemory::new(graph, schedule)` counts the canonical `StageProver`
constructor and persistent state without reading a field value or allocating.
The constructor charge includes publicness/producer derivation and the pool.
The retained charge adds all nested claim/part vectors for every declared layer,
plus every leaf/layer/zero/round coordinate inserted by the complete schedule.
The coordinate-map bound follows the pinned std/hashbrown bucket growth and
includes control/alignment bytes. No repeated point or coefficient reduces it.

These are state charges: `construction_peak_bytes` covers `new`/`for_graph` before
leaf evaluation; `retained_bytes` covers persistent canonical state through the
schedule. Pending window preparations, relation arenas and source payloads,
temporary evaluations/claims, returned messages, borrowed inputs, callbacks and
stack/runtime/allocator overhead remain additional. The helper assumes a compiled
valid graph/schedule and does not authenticate metadata or change CPU admission.


## Canonical relation sources

`StageProver::relation_source_memory(stage, native_tables)` builds public operand
geometry without loading witness data or drawing challenges. The caller supplies
every native table before sharing; native payload/instance callback work stays
separate. This public planner allocates, and its construction work is additional.

`StageSourceMemory::metadata_bytes` charges canonical arena/plan/point/index/factor
storage, independently grown outer vectors and each nested allocation. Generated
column/factor functions record their box size and owned captured-vector bytes.
Nested Broadcast/Spread boxes, tensor shape/layer lists, selection rows, EQ points,
RoPE frequencies and slot lengths are counted. Shared Arcs are conservatively
charged for every owning recipe. Counting-allocator fixtures cover spare capacity
and captured allocations; the original field functions remain the value oracle.

`resident_input_bytes` additionally bounds the original prime-field columns and
canonical Vec growth requests. Streaming sources retain recipes instead of these
arrays. Relation contractions/public EQ tables, prepared-window and sumcheck state,
source replay, construction temporaries, returned messages, borrowed inputs and
runtime/allocator overhead remain additional. These component charges do not
admit the cached route or replace the existing resident floor.


## Relation-table producers

`RelationMemory` bounds canonical table builders from public geometry without
allocating or reading any value. Its matmul, elementwise, rotate, selection and
recomposition methods return constructor peak, retained table heap and additional
borrowed-instance metadata charges. Actual constructors and instances are measured
separately, including both matrix layouts, no-contraction axes, broadcast/partner
inputs, public operands, ragged bit widths and zero-variable cases.

The bills include contracted/expanded fields, public equality/kernel arrays,
saved point fragments and canonical descriptor/term growth. Elementwise clones
and pre-EQ overlap, selection low contractions and EQ scratch, and public-x
recomposition's clearing clone overlap remain charged. Matmul conservatively
counts both possible contractions even when one operand is borrowed. Every
selection member and term remains before specialization or cancellation.

Original borrowed operands/points/kernel, original relation objects retained
separately, outer relation/instance containers, returned claims and subsequent
sumcheck/wrapper work, callbacks and runtime/allocator overhead remain additional.
These are producer charges and leave CPU admission unchanged.


## Stage-driver temporary work

`StageProver::work_memory(stage)` sizes canonical driver containers and temporary
work without allocating or reading field values. `StageWorkMemory` separates
bookkeeping, evaluation, returned output and leaf padding-validation charges.
`peak_bytes` conservatively adds them beside the other component reservations.

The ordinary sharing map counts every graph instance's cloned public key and
nested coordinates/layer descriptors before deduplication. Built/Instance and
index containers are additional. Evaluation includes actual tensor split/blocked
EQ geometry, padded public tables, canonical claim points and their outer arrays,
layer/slot lists, expected/coefficient point temporaries and stage labels. Outputs
include splits, instance targets and leaf per-layer values plus flattening storage.
Leaf validation charges producer/pinned/reached arrays, the table cache and the
established table-constructor bound. `committed_source` walks without allocating.

These driver charges exclude persistent stage state, source arenas/plans, relation
fields and instance payloads, sumcheck engines, pending next-stage windows,
transcript/codec callbacks and runtime/allocator overhead. They assume compiled
valid geometry and canonical native claims; arbitrary callback ownership remains
additional. CPU admission is unchanged.

### Retained graph metadata

`Graph::heap_bytes` counts the current requested heap storage of the public graph:
top-level node, tensor, constant, shift-class and weight-binding capacities;
every node's input/output and transpose-permutation capacity; tensor shapes;
each constant and class vector; and optional request-slot lengths. Empty vectors
with spare capacity still count. It reads no constant or tensor values and does
not allocate or validate. Exhaustive graph, node, tensor and operation patterns
make new owning fields require an accounting decision at compile time.

The inline graph, construction and validation scratch, tensor payloads, other
compiler/layout metadata and allocator overhead remain separate. This getter
does not alter CPU admission or enable the cached driver (RFC-0006/0018, #641).

`LayerMap::heap_bytes` counts its nested input-kind arrays, checkpoint ids and
uniformity flags. `StageSchedule::heap_bytes` adds its own layer map, sites,
stage/instance/claim/pool arrays, all symbolic point and layer coordinates,
coefficient products and LT-coordinate vectors, combined-selection coefficients,
public terms and nested pool-part atom indices. These measure retained capacity
only, with the same exclusions as graph accounting. Tests compare cloned real
conformance schedules with allocator measurements after adding spare storage
and less common nested coefficient/layer ownership (RFC-0037/0018).

### Retained public layout and native-plan metadata

`Lowering::heap_bytes`, `LeafLayout::heap_bytes`, `GroupedLayout::heap_bytes`,
`GroupLayout::heap_bytes` and `ModelNativePlan::heap_bytes` count owned retained
vector capacities. Lowering includes graph, table class metadata and binding lists;
leaf layouts include every affine-form term, domain/axis/read coordinate, site,
layer and instance block; grouped layouts include polynomial coordinates, fixed
variables and per-share block/read lists. The native plan includes cloned request
slots, family windows/nodes/shapes, native output lists and its complete schedule.

`Selection::heap_upper_bound`, `RuntimePlan::heap_upper_bound` and
`ModelAdviceLayout::heap_upper_bound` also cover the pinned Rust1.95 B-tree nodes.
The conservative node bound covers all occupied nodes and empty/transient roots;
it is checked against the actual key/value types. Advice layout ownership adds
the PCS geometry, tiles and nested share ranges, column sources and bindings.
Borrowed lowerings are counted by their owner.

These are allocation-free retained-storage queries over public metadata, without
value reads or validation. Construction/analysis scratch, inline objects, witness
and proof payloads, driver state and allocator/runtime overhead remain separate.
They do not change CPU admission or enable the cached route (RFC-0018, #641).
