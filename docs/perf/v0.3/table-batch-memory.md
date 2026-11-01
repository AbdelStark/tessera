# Bounded table batch: allocation model (#637)

This records the allocation/lifetime model and implementation evidence for accepted
RFC-0054 table streaming. The original resident-cell floor remains, with excess P2
working storage charged separately. It demonstrates no real-model proof and does
not discharge #629's other phases. Geometry bounds below are distinct from measured
heap peaks and timings.

## Resident baseline allocations and lifetimes

For table `i`, let `T_i = 2^h_i` include row-0 table padding, `b = size_of(Base)`
and `e = size_of(Ext)`. Both scalar fields are exact, with `b=8`, `e=16` over
Goldilocks/Fp2. Generator cores and multiplicity backing storage are separate.

| Buffer | Payload bound | Lifetime / owner |
| --- | --- | --- |
| Public table columns | `b*T_i` keys, plus `b*T_i` for valued tables | `TableColumns`, retained through P2 and the enclosing Round2 scope in the resident baseline |
| Dense field multiplicities | `b*T_i` | leaf/group adapter, alongside the table batch |
| Temporary negated counts | `b*T_i` | `table_tree` construction, overlaps the new tree and all previous trees |
| Fraction leaf level | `(b+e)*T_i` | `FractionTree::new`, survives until its final level proof |
| Fraction parent levels, h>0 | `2*e*(T_i-2)` | all levels retained together before proving |
| Root and level metadata | root `2*e`, plus container capacity for `h_i+1` levels | tree/batch owner |
| Current level sumcheck | bound numerator/denominator halves, split EQ weights, messages/evaluations | overlaps all trees and source columns; level buffers release after use |
| Lookup/forest and merge state | existing retained group data, proofs and merge tables | coexists with the table batch; not removed by streaming tables |

The parent-level formula excludes the root and equals zero at height 1. Height 0
has a root only. Vector/container capacity, sources and sumcheck scratch must be
charged beside payloads; payload formulas alone are not an allowance.

`prove_table_batch` builds all trees before absorbing roots. Therefore an
instance-at-a-time constructor still retains the sum of preceding tree payloads.
Changing that loop alone does not bound the batch. `g=0` also needs streamed table
trees within the mixed lookup/table/forest batch (RFC-0054 'The table batch').

## Bounded ownership

1. A source descriptor identifies the immutable public generated rows and the
   already committed multiplicities. Multiplicity physical capacity remains
   selected by public table/update bounds, as in `MultiplicityColumn`; it is not
   selected by secret sparsity. Existing caller-owned backing storage is explicit.
2. A fraction regenerator reads paired numerator/denominator leaves in chunks of
   at most `C` rows. A clearing binary-carry stack combines canonical adjacent
   leaves. To regenerate a level's node, it visits that node's contiguous leaf
   subcube, without a full level or table-sized field conversion.
3. Regeneration scratch is at most `2*e*(min(C,T_i)+h_i+1)` bytes per active
   regenerator, plus fixed metadata. It is reused across consecutive nodes, never
   cached across all table nodes. Caller-provided output chunks are charged too.
4. Roots (`2*e` each) and small public shape/state metadata remain for all trees.
   They are required before the first level and retain original tree order.
5. An oversized level streams one round per halving until its folded tables fit.
   Root/table regeneration performs no model forward. The existing factored-EQ
   sumcheck machinery can be reused only with identical gamma/lambda/round/eval
   labels and final evaluation order. Every active tree and resident lookup/forest
   operand participates in the single level allowance.
6. Current grids/folds, public split-EQ weights, source chunks, messages and retained
   roots/merge/source data form the level's maximum simultaneous working set.
   Fold/grid replacement must release its old allocation before allocating its
   successor. No full multiplicity or negated-count copy may hide outside this sum.

All derived arrays live in clearing buffers. Source geometry/availability and
allocation errors stop the computation; an errored pass is discarded. Root zero
denominators, identity checks, row-0 padding, multiplicity claims and the exact
number of levels/rounds are unchanged. No value-dependent row omission is allowed.

## Acceptance checklist recorded before implementation

- Implement and independently compare every regenerated node/level, not roots alone:
  the fraction-root operation is commutative, so root equality misses a leaf reorder.
- Complete fault injection and broader gates for the integrated generated table/count
  sources, public P2 plan and both CPU routes.
- Charge the complete working set and check it using counting allocators, including
  source/merge coexistence and metadata. Retain the literal engine as the oracle.
- Show transcript/proof/claims identity for fixed schedules and independent tiny
  NI/Live acceptance plus targeted tamper rejection. Tiny test parameters certify
  nothing about an actual model proof.
- Record actual regeneration work/time. Bounded space does not establish acceptable
  real-model time. Do not infer RSS or admission from logical domain counts.
- Keep #629/#617 open for the complete witness, advice, native/weight/opening
  lifetimes, actual soundness budget, canonical-selection decision and real proof.

## Bounded component implementation

`FractionTreeRows` regenerates every ordered level node, not only the commutative
root, with a clearing binary-carry stack. `PreparedTreeBatch` keeps the complete
canonical mixed or table-only schedule: lookup trees and forests remain resident,
and table trees stream one round per halving until their folds fit. The table
path supplies the already drawn gamma to the window engine, preserving gamma
before lambda without drawing gamma again. The original resident entry points
remain the comparison oracle.

`TableSource` supplies padded public keys/values and the original committed
counts in paired bounded chunks. Its fraction adapter uses three clearing base
chunks and does not create a complete negated-count column. The caller must
provide exact immutable rows and counts, including row-0 padding; the adapter
does not infer or replace committed multiplicities. `prove_logup_bounded` keeps
tables in the ordinary mixed batch; `prove_table_batch_bounded` uses the original
`tables/tree` scope for grouped stages. Both return the original outstanding
claims, with all verifier root, identity, table-evaluation and opening checks
still owed.

The batch bound currently charges:

- Caller-declared coexisting source/merge capacity and metadata.
- All resident fraction allocations, forest levels and the input vector's actual
  capacity, even while states are being constructed.
- Conservative retained metadata/output: `65536 + I*(v+1)*(v+8)*(256+8*e)`,
  with `I` the total tree count and `v` the largest tree/forest variable count.
- The largest regenerator scratch, its declared source-read scratch, and the
  level adapter's paired `4*C*e` output scratch.
- Constructor wrapper/input capacity, charged separately from caller-owned
  source descriptors at 4,096 bytes per tree.
- Every level's simultaneous grid/fold and construction scratch, using
  `WindowMemory::with_window_vars(..., 1)`. All levels are planned before table
  root reads; no allowance is applied separately to each table.

This is a conservative component bound, not a replacement for the CPU admission
floor. Native source storage and the public CPU reservation are described below.
These component measurements do not qualify a real-model proof.

Construction has its own allocation-free geometry check before schedule/source
vectors, lookup trees, source reads or challenges. It charges caller-owned storage,
resident unit-numerator lookup trees (at most `3*e*N` plus level descriptors),
forest levels, all batch metadata, wrapper vectors and the largest generated
read/partner scratch. The detailed batch planner then checks every level before
reading generated roots. A check after constructing the resident trees is too
late to enforce an allowance.

Two refusal regressions exposed that ordering bug during review. With only
131,072 B available beyond the declared sources, the old mixed helper allocated
6,293,856 B for 131,072 lookups; the old table-only helper allocated 2,490,368 B
for 16,384 descriptors sharing one source. Both returned an allocation error
eventually, after exceeding their allowance. The revised helpers reject both
cases within the allowance, before source reads or transcript changes. This was
a generic-helper construction bug; the CPU's separate before-P1 reservation was
not shown to be bypassed.

A subsequent isolated campaign caught all three additional planted faults:
removing the construction refusal, ignoring the mixed batch's configured chunk
limit, and ignoring the table-only batch's configured chunk limit. Initial and
restored controls each passed all five library unit tests. These cases supplement
the eight earlier library faults and eleven native-source/admission faults below.

### First complete-batch allocation regression

The public synthetic test uses four table domains: 1, 3, 5 and 16,381 real rows,
with mixed valued/signed and Range tuples. The last table pads to 16,384 rows.
Each lookup column has five real entries and row-0 padding to eight. The test
compares every proof level and root with the unchanged resident implementation,
under a 1,700,000-byte allowance and 17-row source chunks. The counting scope
includes preparation, every tree level, messages and returned claims; borrowed
input capacities are added separately.

Observed on the development M1 Max while unrelated long jobs were active:

| Quantity | Resident | Bounded |
| --- | ---: | ---: |
| Borrowed source capacity and metadata | 394,760 B | 394,760 B |
| Measured additional live heap peak | 1,055,120 B | 160,304 B |
| Measured heap peak plus charged sources | 1,449,880 B | 555,064 B |
| Single observed elapsed time | 5.986 ms | 155.498 ms |

The regression requires the combined bounded peak to fit the declared allowance
and the additional live heap peak to be less than half the resident peak. This
run shows a space improvement and a substantial regeneration-time cost; it is
not a controlled throughput benchmark or a real-model memory measurement.
Separate differential cases compare roots, complete round/evaluation messages,
final claims and the next challenge for both NI and Live transcript contexts,
ragged chunks, signed values, Range tables, forests and zero-height fraction
trees. The mixed and table-only cases also reuse the same source in separate
instances and check each configured chunk bound (1, 3, 17 and 256), in both
transcript modes. Those tests use non-certifying test parameters.

The first isolated fault campaign caught wrong fraction-parent arithmetic and
swapped child parity, but a partner-buffer capacity fault survived the broad
peak regression. A focused allocator case was then added: 32,768 destination
elements with 13-row source chunks must use at most 2,936 bytes of additional
scratch. The planted fault used 2,097,152 bytes and failed that assertion.
The revised isolated campaign caught 7/7 compiled faults: parent arithmetic,
child parity, partner-buffer capacity, count sign, value compression, omitted
root absorption, and omitted early fixed-allowance refusal. Both unchanged and
restored controls passed. These library checks complement the native-source/CPU
campaigns below.
A supplemental malformed-column case also caught omission of the lookup-value
length check, with unchanged/restored controls. The combined campaigns cover
eight planted faults; the complete component suites passed 194 tests (one long
calibration test ignored), and the focused allocation regression passed after
being imported into the worktree. Clippy passed for all targets of both crates.

### CPU integration and admission

The CPU backend now builds `GeneratedCounts` from a generated table and the same
runtime-configured canonical table tile used by P1. The constructor checks table
ID and padded count domain. The source reads counts through bounded clearing
`u64` chunks; it retains no dense field conversion or public table columns.
All lookup keys are checked against the actual table row count, after visiting
every key, with a redacted `INV-IR-2` error. The source uses the claim list's
regenerated table specification, including its digest, rather than caller-supplied
row contents.

The ordinary route constructs lookup columns only and calls
`prove_logup_bounded`. The grouped route uses explicit `LookupInstance` inputs
for each group's resident lookup trees, then the generated table batch. It keeps
the canonical grouped row-0 counts, including every group's padding. No dummy
table or count column stands in for an actual obligation. The resident entry
points remain available as comparison oracles. Generated source cores/counts
are dropped after P2; their construction is included in the P2 stage timer.

`cpu::table_plan` derives a P2 reservation from public leaf domains, table sizes
and group metadata before P1. It charges generated cores and count capacity
(each at most `8*T`), lookup-source columns, forest leaves, resident lookup/tree
levels, complete output/metadata capacity, bounded source/partner scratch and
every level's simultaneous window-engine state. Grouped reservations include
all retained group proofs, split/merge claim points and merge tables. With `P`
polynomials, `G` groups and the existing public `j*`, the retained merge payload
bound is `16*G*P*2^(j*+2)`; class/point/vector metadata and every group's tree
proofs are additional terms in the code.

Native descriptor/container capacity has its own 4,096-byte reservation per
tree, separate from the generic constructor's 4,096-byte charge. A compile-time
size bound covers four native source slots, one public generator descriptor,
four class bytes and six vector headers. Runtime source accounting includes
unused native-vector capacity as well as each used descriptor and its backing
storage. Selecting a budget again does not accumulate either reservation.

An integration regression exercises the projection and the actual bounded
constructor together, using one and 64 singleton sources with no level-working
set slack. With the two reservations sharing one bucket, the constructor refused
projected allowances of 73,392 B and 525,984 B. Reserving its wrappers separately
raises those projections to 77,488 B and 788,128 B and permits both batches.
These synthetic geometries expose a projection/constructor consistency gap;
they are not an observed whole-CPU heap overrun or a real-model proof.
The permanent regression failed with `Window(Allocation)` before the fix and
passed with the separate charge. It also caught a compiled fault that added
the charge into stored state on each selection; the restored control passed.

The original 64-byte resident-cell floor remains. Let `base` be that floor plus
the weight reservation, and `reserve = 64*(table_rows + lookup_cells)`. The P2
component uses this named part of the existing floor; excess over it is charged
explicitly. The whole plan charges
`base + max(reduction_working_set, max(0, P2_working_set - reserve))`.
A smaller host budget selects bounded table rounds/reduction windows or groups,
or refuses before P1. The table engine may use fewer streamed rounds than the
public reservation predicts when its actual capacities are smaller; its entire
batch still has to fit the reserved allowance. This does not replace #629's
remaining whole-model resource qualification or lower the legacy floor.

The ordinary public API NI/fresh-Live suite passes for all three bounded tiny
model variants at lengths 3 and 5. The whole-proof allocator/budget regression
passes all three variants at lengths 3, 5 and 8, including proofs under their
own planned allowance, smaller-budget paths and before-P1 refusal. Forced
one-, two- and three-bit grouped NI/Live proofs pass the independent verifier;
the measured grouped NI peak is bounded by its plan. These are non-certifying
test-parameter demonstrations.

One observed ordinary case used 51,010,763 B against a 228,896,048 B plan. Its
old minimum-utilization assertion (at least a quarter of the plan) was removed:
the upper-bound and before-P1 refusal checks remain, as does the full cell floor.
This is a conservative reservation, not a measurement that a real model fits.

The seeded ordinary CPU fixture remains byte-for-byte identical to the saved
resident reference: 900,574 B, SHA-256
`b22e602d8d69b774e855d156fef7f30252e79c83f94be6b4d4069d0ff2b0bf27`.
The seeded test also reproduces those bytes across one and two prover threads.
This is the same non-certifying tiny fixture, not a real-model proof.

The native reader's boundary test uses a canonical table tile with
`2^30 = 1,073,741,824` logical rows. Reading its final 13 rows allocates only the
104-byte count chunk, which is released after the call; generated keys/values
and canonical counts match their source. It also checks a real nonzero-count
row and rejects mismatched table IDs/domains, invalid keys and chunk shapes.
That is a large-domain row-access test, not a complete proof over that domain.

The native-source/admission campaign caught 11 compiled faults: omitted table
identity or count-domain checks, an admitted one-past key, wrong count sign,
dropped generated values, omitted value/count chunk-length checks, omitted read
range checks, omitted count-read scratch, streaming a resident source, and
omitted fixed reservation. Unchanged and restored controls passed in both
campaigns. These are separate from the eight library faults above. The count
reader tests include an overflowing starting index and require malformed reads
to leave the destination unchanged.

### Adjacent subtree reads and the CLI time gate

The initial workspace run and isolated canary CLI runs failed the unchanged
30-second Live Round2 receive limit with `Inconclusive(ERR-PROTO-TIMEOUT)`.
Captured peer reports confirm this is an operational timeout, not an Accept or
a detected invalid proof. One completed NI run spent 64.679 seconds between the
P1 completion and leaf-stage completion; a subsequent profiling run spent
101.342 seconds there and completed NI proving in 108.145 seconds. Other model
jobs and repository checks were active, so these are observations under contention,
not controlled benchmark results or an independently accepted new proof.

A five-second sample in the latter run captured 3,621 main-thread stacks, all
inside tree-level `State::prepare`; 3,620 were under `LevelSource::read` and
`FractionTreeRows::read_level`. The old reader visited adjacent subtrees separately,
shortening each source read to one subtree even when more rows fit in its chunk.
An earlier diagnostic attempt separately hit the 180-second CLI build watchdog;
prebuilding the binaries outside that watchdog reproduced the actual Round2
timeout.

The revised reader scans the requested contiguous leaf interval in full bounded
chunks and emits each original subtree at its public boundary. It retains the
same carry stack and the declared maximum chunk storage. For `K` nodes with
`2^b` leaves each, source calls fall from `K*ceil(2^b/C)` to `ceil(K*2^b/C)`.
The same leaves and fraction-parent operations are evaluated in the same order
within each subtree. No value-dependent skipping or retained level is introduced.

The new regression first failed with 13 calls where two 17-row chunks suffice.
It now checks read counts and scratch allocation across ragged chunk/subtree
boundaries, compares every result with the unchanged resident `FractionTree`,
and requires all scratch allocations to be released. Both component suites
then passed 198 tests, with one long calibration test ignored.
An isolated campaign caught seven compiled faults in the revised reader:
wrong parent arithmetic, a shifted carry bit, an omitted read offset, missing
subtree emissions, shortened chunks, oversized chunks and an omitted final
subtree. Unchanged and restored controls passed. Clippy passed for both component
crates and the isolated benchmark workspace.

The complete-batch Criterion fixture uses one generated valued table, 256-row
chunks and all roots/levels, including buffer clearing. It records each proof's
digest and raw samples in
[table-batch.json](../../../crates/tessera-sumcheck/benches/windowed/measurements/table-batch.json).
Observed means on the M1 Max under the active model jobs:

| Table rows | Before mean | After mean | Source calls before | Source calls after |
| --- | ---: | ---: | ---: | ---: |
| 16,384 | 120.790 ms | 69.477 ms | 66,880 | 3,648 |
| 65,536 | 544.049 ms | 310.340 ms | 269,568 | 16,640 |

Criterion reports reductions of 42.5% and 43.0%, with unchanged proof digests.
These are synthetic component measurements, not model throughput. To compare two
reader revisions, run the first command before the reader change and the second
after it. The measurement file records both source hashes.

```sh
CARGO_TARGET_DIR=/tmp/tessera-table-bench cargo bench --locked \
  --manifest-path crates/tessera-sumcheck/benches/windowed/Cargo.toml \
  --bench table-batch -- --save-baseline before
# After applying the reader change, reuse that target directory:
CARGO_TARGET_DIR=/tmp/tessera-table-bench cargo bench --locked \
  --manifest-path crates/tessera-sumcheck/benches/windowed/Cargo.toml \
  --bench table-batch -- --baseline before
```

The first post-optimization canary CLI run passed in 70.26 seconds overall: both NI/Live
proofs in both log formats, independent verifier acceptance, expected rejection
controls, and the complete redaction scan. The JSON runs spent 11.133 and
18.385 seconds between P1 and leaf-stage completion. The 30-second message
deadline and every acceptance/redaction assertion remain unchanged. After the
reservation correction, the freshly built CLI passed the same suite in 51.86
seconds, with four independent accepting reports. These tests still use
non-certifying parameters. Workspace and CI gate results are recorded in the
pull request.

### Remaining real-model work

The table/count source still uses the existing multiplicity reader. Its weighted
representation scans its public update capacity per chunk, so bounded memory
alone does not establish acceptable real-model time. Count-index construction
and table-regeneration work need separate measured qualification. No witness
row, group or table leaf may be omitted to reduce that cost.

The actual 1B/8B proof blockers remain separate: the leaf-layout/canonical-family
decision, complete tensor/advice/weight/opening lifetime accounting, real
soundness calibration and a named-parameter independent verifier verdict.
Neither these component tests nor their tiny CPU proofs discharge those gates.

### Diff-level adversarial review

This records the implementation review, not an independent protocol audit or
production qualification. Repository and CI gate results belong in the pull
request's validation record.

| Obligation | Evidence and result |
| --- | --- |
| Canonical counts bind P1 to P2; no substituted table | `cpu/backend.rs` builds each `GeneratedCounts` from `GeneratedTable::from_spec` and the same runtime source's advice table tile. `leaf/table_source.rs::new` checks table ID and padded domain. The native-source tests and ID/domain mutants cover those guards. |
| RFC-0008 signs, ordered subtrees and row-0 padding | `streaming.rs::Fractions::read` supplies `(-m, alpha-compress(key,value))`; `fraction_rows.rs::Scratch::subtree` combines adjacent canonical subcubes. Full level/claim/challenge differential tests cover ragged domains, signed values, Range tables, zero-height trees, forests and repeated sources. Count-sign, value-compression and child-parity faults fail. |
| INV-FS-1 and RFC-0010 degree/batching schedule | `PreparedTreeBatch::prove` keeps joins, gamma, lambda, original level relation, line challenge and next claims; `PreparedWindow::prove_with_gamma` consumes that gamma once. Complete messages match the resident oracle, including the next challenge. The seeded CPU proof matches the saved resident proof bytes. No additional challenge or acceptance check is introduced. |
| Leaf/opening obligations are retained | Both bounded LogUp routes return the original `ProvenLogUp`/`ProvenTables` outstanding claims. Grouped proving retains every group's split and forest claims in the existing merge; multiplicity claims reach the existing pool/openings. Independent tiny NI/Live acceptance and verifier tamper suites passed. Verifier root-denominator, identity, table-evaluation and opening checks are unchanged. |
| INV-IR-2 and integer multiplicities | The native reader visits every lookup key and checks it against actual table rows. Canonical `MultiplicityColumn` counters enforce their original total bound below the field modulus before conversion. One-past and negative-key tests, final-row/count comparisons and the one-past mutant cover the new reader. No integer gadget or range rule changes. |
| OBS-6 and redacted errors | New count chunks, fraction carry stacks, partner chunks and window folds use `SecretBuf`; generated descriptors contain public row-zero values and geometry. Source/allocation errors propagate without substituting values and carry static errors only. Borrowers are registered in `secret-access.toml`; no new accessor is exposed. The unchanged canary CLI redaction scan passes for both log formats, with NI/Live proofs and injected-leak controls. |
| RFC-0018 complete component allowance | Public CPU planning retains the cell floor and all sources/group merge state. Batch preparation plans every level before generated roots. The additional construction preflight addresses the two measured over-allocation refusal cases above. Native source metadata and generic constructor wrappers have separate reservations, checked together in the singleton/repeated-source regression. Complete-batch and whole-CPU/grouped counting regressions passed. |
| SP-6 and protocol scope | No verifier source/dependency, schema, label, parameter set, table definition or verdict semantics change. Verifier default and test-parameter suites, formatting and Clippy passed. Protocol/1 remains non-hiding; this change establishes no formal zero-knowledge property. |


## Aligned fraction chunks (#663)

The real 1B NI-100 run completed P1 with 8,758 rows in eight segments. A later live
sample placed all 82 main-thread samples in Round 2's generated fraction-root
construction. This identifies work to measure; it proves neither whole-run cost nor
independent acceptance. The original model run keeps its frozen executable.

`FractionTreeRows` now compacts aligned power-of-two subtrees inside its existing
numerator/denominator buffers, then carries their roots at the original subtree
height. It restores every root's original leaf offset before emitting ordered nodes.
The source-read sequence and the original tree boundaries stay fixed. Buffers shorter
than 32 elements or with a ragged capacity use the unchanged scalar traversal.
An initial candidate regressed one-row chunks, which is why this fallback remains.

For `N` leaves and `R` output nodes, both traversals compute exactly `N-R` parents,
each with three extension-field products and one extension-field sum. There is no
algebraic shortcut, denominator test or secret-dependent choice. A logical parent
reads four and writes two field values: 96 bytes at 16 bytes per `Fp2`, or 1/24
extension-field operations per logical byte. At `N=2^20`, a root has 1,048,575 parents
and 100,663,200 logical bytes. A hypothetical 400 GB/s stream would move those bytes
in 0.252 ms; cache reuse, arithmetic and control flow prevent treating that division
as a measured hardware limit. No issue-width or hardware-counter roofline is claimed.

The allocation inventory is unchanged: two `chunk`-element clearing field buffers,
one `2*(height+1)`-element clearing stack, their descriptors and the original
source-read scratch. The in-place destinations precede unread pairs; unused tails
remain in their clearing owners. Allocation controls retain exactly three owned
allocations, no retained scratch and the original `scratch_bytes()` bound. No plan
charge or host allowance was reduced.

The isolated benchmark workspace adds a public synthetic root fixture with `2^20`
leaves, both extension coefficients, and chunks 1, 17, 256 and 4,096:

```sh
cargo bench --manifest-path crates/tessera-sumcheck/benches/windowed/Cargo.toml \
  --bench fraction-rows --locked -- --save-baseline before
# After changing the kernel:
cargo bench --manifest-path crates/tessera-sumcheck/benches/windowed/Cargo.toml \
  --bench fraction-rows --locked -- --baseline before
```

Criterion 0.5.1 uses 3 seconds of warm-up and 20 samples. Completed roots are consumed
and their public fixture digest printed. Both revisions produce
`0be4ad72fd531bd8d6bb6fb3d6d17aed2f7cb309d513ab930551a54d9ca63240`.
The host is the same M1 Max/64 GiB used for the row measurements, with the 1B proof
and 8B calibration co-running. Raw samples and noisy initial candidates are retained;
these are kernel observations, not H100 or complete proof qualification.

The controls compare every ordered node and exact source request for all valid
subranges through height 7, nine chunk capacities, field boundaries and both
extension coefficients. Nonzero denominators keep upper nodes informative; separate
cases preserve zeros. The unchanged resident fraction/LogUp engines check full
levels, mixed and table-only messages, challenges and outstanding claims. Four
compiled faults—crossing output-node boundaries, wrong leaf stride, lost root offset
and wrong carry height—fail the node oracle. Source errors and pre-read public
validation remain fatal with their original typed errors.


### Measurements

The [raw record](../../../crates/tessera-sumcheck/benches/windowed/measurements/fraction-rows.json)
retains both compiler profiles and every corrected paired sample. In a same-process
production-profile comparison (thin LTO, one codegen unit), three warm-up pairs preceded
20 alternating measured pairs, each timing four complete roots per implementation:

| Chunk | Scalar median, ms | Compacted/fallback median, ms | Median paired ratio | Scalar/candidate CV |
|---:|---:|---:|---:|---:|
| 1 | 60.39 | 59.22 | 0.969 | 7.21% / 4.82% |
| 3 | 61.01 | 61.22 | 0.982 | 7.09% / 6.05% |
| 17 | 62.84 | 56.71 | 0.891 | 4.58% / 4.47% |
| 256 | 62.23 | 43.41 | 0.708 | 3.07% / 5.46% |
| 4,096 | 64.80 | 44.52 | 0.669 | 6.46% / 9.59% |

Both larger chunks won all 20 pairs in that run. A second comparison with the isolated
benchmark's default profile (no LTO, 16 codegen units) gave median ratios 0.804 and 0.767,
with 19/20 wins for each. The large-chunk benefit repeats across these profiles. Small
cases remain inconclusive: the default-profile paired ratios were 1.032, 0.991 and 1.070
for chunks 1, 3 and 17, unlike the production-profile observations above.

The corrected serial Criterion comparison reported lower large-chunk cost, but still
reported regressions in the small cases against its earlier saved baseline. All cases
had variation above 3%; those observations are retained rather than presented as a clean
regression threshold. Changing co-runners and compiler layout limit attribution. The
scalar fallback is kept, and an idle-host run is still required for throughput qualification.
No cost-model or end-to-end proof multiplier was re-fit from these kernel timings.
