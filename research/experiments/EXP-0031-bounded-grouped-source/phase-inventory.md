# Grouped physical phase composition obligations

Source: main `646f7addc7bc737f2fbb8970c865df8333877093`, RFC-0018 Memory
scheduler, RFC-0054 and the [CPU lifetime inventory](../../../docs/perf/v0.3/cpu-memory-lifetimes.md).
E0 ownership analysis. EXP-0031 supplies finite E1 requested-allocation controls
for the source and merge subset only. No complete production phase is admitted.

`B` denotes every allocation retained across a phase: caller execution/weight/batch
capacities, session and selected-plan metadata, public engine/table cores,
canonical tokens/checkpoints/cache, commitments and all still-live outputs.
It is a sum of distinct owners, not a choice between them. Borrowing a value does
not eliminate its backing owner. Caller model storage and documented process
overhead retain RFC-0018's separate measurement scope.

| Phase | Owners to compose beside B | Existing evidence | Still owed before grouped admission |
|---|---|---|---|
| Validation and session preparation | Model/plan validation, maximum-context preflight, source/grouping descriptors, regenerated weight layout/trees, worker controls | Existing session and compiler bounds; GroupedLayout construction bound | Source-factory construction and every callback/replay owner; simultaneous caller inputs and spare capacities |
| Public planning | Group candidate layouts, table/count plans, merge shapes/claim maps, window candidates, report construction, construction/output overlap | Layout and physical-planner component controls | Full grouped candidate construction and public-only selection; unsupported/unknown owners must refuse |
| P1 | Canonical tokens and gaps, checkpoints/KV, replay scratch, native/nonweight caches, one tile, every grouped padding multiplicity, advice row encoders/codeword/chains/trees | Existing checked ungrouped physical components | Canonical grouped counts and a bounded factory/cache schedule; a checkpoint source can still retain complete score/activation tensors |
| P2 current group | Group layout; raw selected tensor/tile leases if retained; key/value columns; forest shares; descriptor/schedule metadata; lookup/forest kernel operands; split coordinate/evaluation scratch | Group-layout, LookupGroupMemory, ResidentTreeMemory; EXP-0031 columns/forest/index and restricted split controls | Canonical gadget/checkpoint factory integration and its working tiles; complete overlap with earlier proofs, kept tables, native consumers and current raw source leases |
| P2 group completion | All prior group messages/points/splits/claims/gammas; current source/blocks until its split and merge-table accumulation finish; kept Q/tables | GroupedMessageMemory and MergeMemory; EXP-0031 finite input/construction controls | Concrete lease release and clearing boundaries; preserve each role's canonical identity and prove the actual callback work fits |
| Table batch | Every group output and kept merge table; generator cores, canonical counts/backing, chosen caches/chunks, fraction regeneration/levels, table proof/claims | TableBatchBudget and table component controls | Grouped lifetime composition, not a separate allowance that spends the same spare bytes twice |
| P2b | Merge metadata and prior group/table proofs; kept tables; current original polynomial; temporary owned source copy/permutation; weights/equality; folded states from prior polynomials; merge rounds/output | MergeMemory plus caller-source/block additions measured by EXP-0031 at every legal cut | Production callback bounds and final phase composition; original source is additional even when returned folded state is small |
| Native construction | Original source access/leases, native operands/owned tables, grouped leaf claims and graph atom splits | Existing native components and opening geometry | Replace reliance on a complete ModelWitness with canonical bounded inputs; cannot clear a lease still borrowed by a native consumer |
| P3/P3b | Native/reduction operands, retained grouped messages, stage grids/windows/folds, checkpoint source and replay temporaries | Existing stage/window/native memory components | Grouped merged-point integration and exact overlap/release evidence for the bounded factory |
| Target transfer | Every original advice/weight source, coefficient, image and point; ModelOpenings construction/copy; previous native/stage buffers until transfer completes | OpeningGeometry and target bounds | Prove no original target is omitted or deduplicated through a favorable zero/coincidence; validate grouped whole-route transfer |
| P4 advice and weights | Source tile/row encoder, all opening accumulators and final public copies; retained proof/targets; per-matrix weight work/trees | Existing commitment/opening/weight component bounds | Recompose with grouped target/message counts and all factory/cache owners; original witness obligations unchanged |
| P5 | Queried columns, encoder/source, retained commitments/outputs, weight answers and multiproofs | Existing column/weight/response bounds | Full grouped simultaneous ownership and unchanged verifier/query limits |
| Round2/wire and publication | Original grouped/table/merge/native/stage messages, production wire clone, query containers, protocol transcript/encoding, packet writer, retained commitments and manifest work | GroupedMessageMemory; DriverWireMemory/response/packet components | Checked full grouped packet phase and whole-process measurements; no ownership disappears merely because it is serialized |

## What the experiment measures

- Source stage: current ordered lookup columns and one forest share, index record
  allocations and earlier retained index records. Borrowed public layouts and
  materialized reference/lent views are outside its counter. Serialized report
  storage is outside the counter and explicitly not a production proof output.
- Merge input stage: current group layout construction, public driver metadata,
  restricted group-block split evaluation and retained Merge inputs. The
  unrestricted original-table builder is used only by the reference oracle and
  P2b; the candidate split uses its selected block. A polynomial with an empty
  group-variable set legitimately has a whole-tensor selected block, which remains
  fully charged. Factory descriptors are borrowed inputs.
- Merge construction/proving: current MergeMemory totals plus caller-owned group
  blocks and original-source/copy peaks, including earlier folded states already
  covered by that component. Transcript hashing is allocation-free in this test.
- The Python audit independently reconstructs fixture/index/value/claim/cut and
  composition arithmetic. Production MergeMemory/GroupLayout component totals,
  Rust ABI sizes and allocator observations remain pinned producer inputs, not
  independent proofs of those components' internals.

## Refusal and qualification

A complete candidate must carry every phase, every source constructor and callback,
caller capacities and packet work as known checked bounds. `Unknown` is different
from a proven empty owner of zero bytes. The checked maximum of complete phases,
plus caller batch ownership, must fit before P1. Missing owners, unsupported
geometry, overflow, unaddressable arrays and an allowance one byte below that
complete maximum refuse before P1. Component source bills cannot fill unknown
production fields with zero.

The current two physical admission guards remain unchanged. Neither the experiment
nor RFC acceptance alone permits bypassing them. The resident 64-byte logical-cell floor
remains the first-choice oracle. Independent NI/Live whole-route acceptance,
artifact equivalence, exact tamper errors, production clearing/refusal controls
and actual real-model fidelity/time/memory qualification remain separate gates.

Factory work must also follow the registered pass schedule. A synthetic constant
time datum callback does not establish a bounded checkpoint replay or one-forward
P2. Repeated per-entry GEMMs cannot be relabelled one regeneration. The production
implementation must retain or tile the necessary current-group sources with their
complete charged lifetimes and report actual pass work.
