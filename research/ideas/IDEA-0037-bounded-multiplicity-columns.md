# IDEA-0037: Publicly bounded physical multiplicity columns

| | |
|---|---|
| Status | `validated` |
| Origin | [SUR-0015](../surveys/SUR-0015-witness-multiplicity-storage.md), remaining #112 |
| Lever | CPU working memory and feasibility of `regen`; correctness prerequisite |
| Spec surfaces | Internal witness storage/API (tier C, 02 §1); RFC-0041/0045/0050 logical columns remain unchanged |
| Priority | 0 performance credit (E0); blocks complete #112 acceptance |

## Mechanism

Keep a table's exact canonical logical column. Public metadata gives actual rows
`T`, padded table rows `T_pad`, and a checked upper bound `M` on weighted updates,
including per-axis padding contributions and the final row-0 tail. An update is
`(u32 row, u64 amount)`, in clearing storage. Repeated table requests are separate
contributions, not deduplicated lookups.

Compare three physical implementations: dense counters (`8*T_pad` bytes);
fixed-capacity weighted entries (at most `M*sizeof(Update)` bytes), sorted in
place and read by exact run sums; and fixed row-window recounting. Select dense
versus sorted using public bounds only. Neither the number of distinct keys nor
their distribution chooses a capacity, mode, public length or debug output.
Refuse a key outside actual `T`, a public capacity overrun, integer overflow or a
count reaching Goldilocks. Missing positions and table padding read as zero.

Expose fixed logical row access/streaming; do not expose a secret-sized map as a
public layout. Sparse physical zeros must never become secret-dependent skipped
commitment rows. The committer still owes every public canonical position.

## Seeds and lens pass

The scoped source set is SUR-0015, RFC-0018/0045/0050 and the baseline's storage
rejections. Ten alternatives exhaust that bounded question:

| Lens / seed | Decision |
|---|---|
| Lookup: dense counters | baseline, retain when its public bound is smaller |
| Lookup: fixed-capacity weighted sorted updates | candidate, exact run sums |
| Passes: row-window recounting | comparison arm; public repeated-replay cost |
| Hardware: GPU atomics | defer to #165; no target GPU here |
| ZK: secret-sized hash/tree map | reject unpadded cardinality allocation and uncleared nodes |
| ZK: sampled sparsity mode selector | reject secret-dependent physical shape |
| Commitment: skip secret-zero multiplicity rows | reject; changes the public commitment obligation |
| Counts: smaller root/table domains from observed values | reject; public worst-case range contract remains |
| Parameters: cap tables or change a preset to fit RAM | reject; unrelated protocol/domain change |
| Max certificate / attention: drop auxiliary lookups | reject; incomplete witness/counts |
| Wild paradigm: external disk histogram | defer; secret disk lifecycle and I/O scheduling exceed this unit |

No new commitment, max certificate, attention shape, parameter, compiler count or
cryptographic paradigm is needed for this physical-storage question.

## Why it could work

[RustSlice26] specifies an allocation-free sort; [CubReduce26] specifies exact
run aggregation. [Henriksen20] explains why histogram distribution affects work,
so this proposal refuses its secret-data inspector as a mode selector.
[TesseraWitness26] supplies the canonical padding/clearing obligations (all E0).

## Expected gain

No parameter of `research/tools/cost_model.py` changes. The baseline remains the
reference (`python3 -I research/tools/cost_model.py --help`). Credited delta mu,
proof size and verification gain are 0 (E0) at 1K, 4K, 16K and 128K. The question
is whether complete CPU counters can fit; no GPU speed, pass reduction or
production peak-memory bound is inferred from an isolated counter.

## Security gate

| Property | Effect | Pass |
|---|---|---|
| G-4 soundness | Identical integer counts at all canonical positions; no challenge change | pass as a required exact-equivalence gate; experiment must reject every malformed update |
| SP-3 uniqueness | Count each required lookup/padding contribution once | pass conditional on complete integration and exact field-bound checks |
| G-5 ZK | Public mode/capacity/length; values and storage clear | pass for allocation metadata; data-dependent execution still needs RFC-0018's existing public release schedule before protocol/2 delivery |
| SP-6 independence | No verifier dependency or algorithm change | pass |
| Post-quantum | No primitive/assumption change | pass |

## Prior rejection check

RFC-0011/RFC-0018 reject storing the encoded/full witness; baseline §7 preserves
that rejection. Storing one requested table's bounded counter representation is
not storing all witness tensors/codewords. A windowed arm must report its extra
replay visits and cannot claim the existing forward-pass count unchanged.

## Kill criterion

Reject the candidate if any canonical count differs, a secret distribution
changes its mode/capacity/public logical length, its public capacity is exceeded,
count arithmetic wraps or reaches the field modulus, or the measured storage
cannot satisfy its stated public bound. A correct compact counter does not by
itself establish #112's checkpoint-plus-working-tile memory criterion.

## Cheapest experiment

[EXP-0024](../experiments/EXP-0024-bounded-multiplicities/README.md), E0 exact
equivalence plus E1 CPU allocation measurements, against the same dense baseline
and a row-window arm. Synthetic inputs only; no production integration before
the result. No H100 is needed for the representation test. GPU implementation,
release-budget calibration and full witness-source memory remain later gates.

## Refutations

| Lens | Verdict | Argument |
|---|---|---|
| Soundness / ADV-P / ADV-V | survives conditionally | Test duplicate requests, actual-vs-padded domain, weighted row-0 tail, modulus boundaries, public capacities and clearing. Do not infer full ZK from a fixed allocation. |
| Prior art | survives | These are established counter representations; value is exact canonical integration and bounded secret storage, not algorithm novelty. |
| Cost model | weakened | Zero credited mu gain. Sorting and window replay may be slower; CPU feasibility is the only proposed benefit. |
| Feasibility | survives | Standard library and existing SecretBuf; internal API only. Pre-registration and allocator evidence precede production code. |

## Score

| Gain | Confidence | Cost | Priority |
|---|---|---|---|
| 0 credited (E0) | 0.6 for established mechanism, unmeasured integration | s for experiment; full #112 separately | 0; authorized correctness prerequisite |

## Decision and links

EXP-0024 validates the registered finite exact/public-allocation/counter-memory criteria on Rust 1.95 and 1.88, including 14 physical fault detections and zero retained allocations. This is not complete model witness integration, a full memory bound or timing ZK. No production/GPU gain credited. Ledger updated 2026-10-20.

### Production integration boundary

The #112 CPU implementation uses the experiment's public fixed-capacity weighted
representation but does **not** sort secret keys: standard comparison sorting and
binary/run searches have secret-dependent control flow, forbidden by INV-ARCH-2.
Each requested public chunk is cleared and receives a fixed-length scatter over
all `M` entries, using `select_unpredictable` for the index and count. Keyed
counting addresses retain RFC-0018's release-schedule obligation. This exchanges
query work for public loop bounds; EXP-0024's sorting measurements are not a
production timing or allocation claim. Production chunk/count/clearing tests
validate that implementation separately.
