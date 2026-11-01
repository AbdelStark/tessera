# Bounded fraction-level cache (#667)

P2 repeatedly regenerates fraction-tree levels from the original table leaves.
A batch may retain one intermediate level in two clearing `SecretBuf` arrays.
Ancestor reads fold those cached nodes in their original order. Descendant reads
still use the original leaf source and full original offsets. Cache construction
reads every original leaf, including canonical row-zero padding, before installing
the cache. A source error discards the construction and stops the proof.

## Selection and lifetime bound

For original height `h`, a cache at depth `d` from the root retains `2^d`
numerators and denominators: `2 * size_of(Field) * 2^d` bytes. Cache descriptors
are included in the regenerator's existing scratch/metadata charge. The complete
batch charges the **sum** of all cache payloads before any preparation allocation
or source read. Source backing, construction scratch, partner outputs, resident
lookup/forest trees, returned messages and window work remain separate charges
(RFC-0018/0054, OBS-6).

The CPU physical route first selects the unchanged feasible cuts and bounded read
chunks, then spends remaining allowance on caches. It tries public row caps from
`2^20` down to `2`, with `d = min(h - 4, log2(cap))` when `h > 4`. Thus each node
still represents at least sixteen original leaves, and each cache is at most
32 MiB over Goldilocks/Fp2. All table caches must fit together beside the same
window peak. Otherwise the route keeps the uncached plan. Resident-route
selection and the 65,536-row chunk cap are unchanged. Re-selection may shrink
caches again; no model value or count distribution enters the decision.

Both arrays survive through their batch and clear on normal return, source error
or drop. The cache changes no tree height, original leaf claim, root denominator
check, fraction identity, round polynomial, transcript label or PCS opening.

## Validation

Ordered-node differential tests compare every level against the unchanged scalar
fold, including ragged chunks, boundary field values, zero denominators and all
four supported fields. A separate installed-cache test checks descendant offsets
near the end of a `2^32`-leaf domain without scanning that domain in a unit test.
Mixed-height LogUp/table batches compare complete messages, claims and the next
transcript challenge in Live and NI modes. Counting-allocator tests cover shared
refusal before allocation/read, construction/error cleanup and complete-batch
peak storage within the original allowance. CPU policy tests preserve feasibility,
cuts and chunks, including exact-fit fallback and repeated selection.

The full physical-route fixture compares encoded NI proofs against
the resident route and uses independent NI/Live verification with tamper controls.
These synthetic fixtures use non-certifying test parameters. Actual 1B/8B proof
acceptance, fidelity and process resources remain separate requirements of
[#629](https://github.com/AbdelStark/tessera/issues/629) and
[#617](https://github.com/AbdelStark/tessera/issues/617).

## Measurement

The reproducible Criterion cases are in
[`table_batch.rs`](../../crates/tessera-sumcheck/benches/windowed/src/table_batch.rs).
They time a complete batch including cache construction and internal clearing;
returned proof destruction also occurs inside Criterion's iteration. The
uncached and cached cases check an identical deterministic artifact digest before
timing. The cases use heights 14 and 18, 65,536-row maximum chunks and the CPU
policy's `h - 4` cache depth. They use one synthetic public valued table and
`test-small/1`; they measure no model proof or hardware throughput qualification.

```sh
CARGO_PROFILE_BENCH_LTO=thin CARGO_PROFILE_BENCH_CODEGEN_UNITS=1 \
  cargo bench --manifest-path crates/tessera-sumcheck/benches/windowed/Cargo.toml \
  --bench table-batch --locked -- bounded-table-cache
```

A separate operator probe uses the original generated table and canonical
weighted multiplicity reader from the public 64-position conformance model:
6,291,456 actual rows padded to 8,388,608, 24,963 fixed weighted updates and
65,536-row chunks. Its depth-19 cache retains 16,777,216 bytes. Timing includes
construction and internal clearing, with returned proof hashing/destruction
outside the timer. It uses the standalone release profile, distinct from the
Criterion thin-LTO build. All cases preserve the original source and proof work.

The raw samples and confidence intervals are retained in
[`fraction-cache.json`](../../crates/tessera-sumcheck/benches/windowed/measurements/fraction-cache.json).
The host was an M1 Max with 64 GiB RAM, macOS 15.6.1 and Rust 1.95.0. Real-model
jobs and other repository/host work remained active. Criterion ran twice because
the first run exceeded RFC-0020's 3% coefficient-of-variation threshold.

| Criterion case | Uncached median | Cached median | Uncached CV | Cached CV |
| --- | ---: | ---: | ---: | ---: |
| First, height 14 | 92.29 ms | 37.38 ms | 20.76% | 15.00% |
| First, height 18 | 1360.91 ms | 571.62 ms | 6.28% | 5.67% |
| Repeat, height 14 | 81.20 ms | 38.58 ms | 26.57% | 43.83% |
| Repeat, height 18 | 1487.61 ms | 564.79 ms | 8.64% | 2.67% |

Every case is noisy except the repeated height-18 cached case. On both runs and
both geometries, the mean confidence intervals separate, but these timings still
need a quiet-host throughput qualification.

The canonical-count probe alternated uncached/cached order for six measured
pairs after one warm-up pair. Median times were 88.94 s uncached and 30.42 s
cached. Source calls fell from 11,904 to 2,176. The mean paired cached/uncached
ratio was 0.322, with a 95% bootstrap interval of 0.244 to 0.404. Variability was
high: CV 29.88% uncached, 14.92% cached and 34.18% for paired ratios. Each pair
improved; their ratios ranged from 0.196 to 0.487. All fourteen runs produced
artifact digest `472a45eaf2d14387308d8af48f46a80a3187de416aa380ab51e95164c89b248d`.

The optional cache is retained because the observed effect exceeds the measured
variation in both experiments. No speedup for a whole-model proof is inferred.
