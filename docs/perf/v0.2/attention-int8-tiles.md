# Exact integer attention score tiles

Issue [#621](https://github.com/AbdelStark/tessera/issues/621) extends the private CPU integer
reference's [bounded weight primitive](reference-int8-tiles.md) to narrow attention scores.
RFC-0005 integer semantics, RFC-0030 score masking and SC-7 remain unchanged. Both model
profiles may accelerate QK products; full traces retain the int64 oracle. Wide Exp/value
products retain their existing int64 route. Model artifacts, proof generation, verification
and transcript bytes are outside this change.

## Exactness and resource bounds

Each operand must be CPU int64 and its values must lie in [-128,127] before conversion.
For each RHS value, `hi = w >> 1`, `lo = w & 1`, and `w = 2*hi + lo`, including -128.
The two bounded integer products avoid saturating signed-int16 pairs, even on an
unsigned-offset backend: `2*255*64 = 32640 < 32768`. At an inner tile of 1024, high and low
products have absolute bounds `2^23` and `2^17`. Both INT32 products widen before doubling
or accumulation. Every full or intermediate sum is bounded by `128*128*K < 2^63`; other
inner dimensions decline acceleration. No floating-point conversion is used.

The shared primitive uses at most 128 rows, 1024 columns and 1024 inner entries per tile.
Batch broadcasting creates views. A lazy ordinal index visits one logical head at a time,
using O(rank) metadata; it never reshapes an expanded batch or snapshots an index pool.
An empty result returns before batch traversal, including a billion logical empty heads.
Each RHS tile is sliced and transposed before conversion; neither a whole head nor a whole
batch is converted just to supply a tile. The required int64 output is allocated once,
separately from bounded packing and accumulation scratch. There is no persistent activation
cache. This does not bound the model's required quadratic attention output.

A missing or unsupported optional `torch._int_mm` kernel declines the entire result,
including completed heads, so the caller can use its original int64 product. Typed OOM and
ordinary allocator errors propagate; allocation failure does not retry through a larger
representation. Inputs remain unchanged. This Python reference does not establish
constant-time execution or cleared allocator storage; no new secret values reach reports,
logs or errors (05 section 3).

## Differential and adversarial evidence

All 421 Python tests pass on Apple M1 Max with no skips, including the existing golden
conformance traces and native integer backend. Strict types, lint and formatting pass.
The first concurrent full run passed 419 cases but failed the unchanged 30-second Exp-core
timing gate at 37.2 seconds. With only the two own model jobs temporarily paused, the entire
suite passed in 107.14 seconds. Both jobs resumed in a `finally` block; fuzz workers remained
active and no threshold changed.

New cases cover 2D/3D/4D broadcasts, strided GQA head mapping against a literal Python-integer
oracle, signed extremes, all tile tails, odd totals above `2^32`, pre-conversion wide refusal,
partial-backend failure, allocator errors, bounded source copies and empty logical batches.
Int64 outputs are poisoned before the tail checks, preventing allocator reuse from hiding
an unwritten element. The test for a huge empty batch intercepts eager index pooling before
it can allocate a billion Python integers.

No-op and restored controls pass. All 14 meaningful planted source faults fail assertions:
wrong query head; wrong key head; materialized broadcast; omitted left range check; omitted
right range check; INT32 accumulation; wrong high digit; missing low digit; incomplete inner
tail; incomplete column tail; partial-result success; allocator fallback; empty-batch
traversal; and routing the full trace through the optimized path.

## Actual exported 1B candidate

The same [provisional 1B D](requested-silu.md#provisional-real-1b-diagnostic) was read from its
real artifact and evaluated on six positions. Every logit in the 6 x 128256 output matrix
matches the int64-score route byte for byte. All 16 layer score products accelerated, with
zero fallbacks. Both runs retain the previously validated bounded weight products.

The concurrent paired run observed 8.316 seconds with int64 scores and 8.572 seconds with
tiled scores; combined peak RSS was 4,074,520,576 bytes. These short products do not establish
a whole-model speedup. They establish only this numerical comparison, not complete held-out
fidelity, alpha selection, an accepted proof or a hardware qualification.

The real `ni-100` proof attempt still produced no proof: the unchanged RFC-0045 leaf-layout
guard refused inconsistent tensor reads across layers. The proposed compiler remedy remains
[Draft RFC-0076 in PR #620](https://github.com/AbdelStark/tessera/pull/620), awaiting genuine
non-author maintainer review. These tiles neither adopt that proposal nor bypass the guard.

## Work and measurement scope

With four heads and dimension 64, the 512-position case computes 67,108,864 dot-product
terms. Original int64 operands plus required output occupy at least 10 MiB under ideal
reuse, giving 12.8 integer multiply/add operations per byte. At 1024 positions these counts
are 268,435,456 terms and 36 MiB, or about 14.22 operations per byte. The baseline targets
integer compute with reused operands; the tiled path trades two narrow dot products and
bounded packing for a faster backend. These lower traffic bounds exclude rereads, scratch
and library packing, and imply no hardware bandwidth or GPU throughput qualification.

Measured 2026-10-29 on Apple M1 Max/64 GiB, Python 3.13.18, PyTorch 2.14.1 and NumPy 2.5.4.
The first cohort ran beside both real-model jobs and two fuzz workers. Its median process
medians were 90.508 versus 80.129 ms at 512 positions and 406.855 versus 367.435 ms at 1024.
Within-process coefficients of variation reached 108%; some pairs regressed. That cohort
does not establish an improvement under full concurrent load and remains part of the evidence.

For the final cohort, the two own model jobs were temporarily paused and resumed in a
`finally` block after 141.77 seconds. Fuzz workers stayed active. Vector generation and type
checking also overlapped parts of the measurement, so this is not an isolated hardware run.
Both modes import the same helper before timing, avoiding mode-dependent library startup
in the process RSS comparison. Five alternating pairs each use three warmups and ten samples:

| Positions | Mode | Median of process medians | Process medians, ms | Median peak RSS |
|---|---|---|---|---|
| 512 | Int64 | 52.227 ms | 52.227, 70.626, 63.587, 48.797, 50.778 | 387,760,128 B |
| 512 | Tiled | 18.704 ms | 31.623, 39.987, 18.442, 15.622, 18.704 | 395,329,536 B |
| 1024 | Int64 | 254.000 ms | 210.858, 207.400, 255.557, 256.084, 254.000 | 381,845,504 B |
| 1024 | Tiled | 60.350 ms | 60.350, 56.938, 72.492, 79.553, 59.933 | 406,650,880 B |

All ten final paired comparisons improve: 1.65–3.45x at 512 positions, 3.22–4.24x at 1024.
The ratios of medians are 2.79x and 4.21x, respectively. Within-process variation still
exceeds a 3% noise floor: scalar CV spans 2.0–12.9% and tiled 4.1–41.9% at 512, versus
1.3–6.2% and 4.9–18.1% at 1024. The repeated paired gains justify this component path;
these numbers do not predict whole-model performance under the original full load.
RSS includes the Python process, libraries and allocator reuse, not just live tile scratch;
there is no measured process-memory reduction here.

Every synthetic result agrees across modes. SHA-256 is
`c9155026d1c174d1af3bc8b6a71a40d0c58a067ce50c1037d3b2ceca11ec48fd` at 512 positions and
`07ac357c38477c53e02b257591810f9ff6d5bf93e3a45d2beea00698cc83ef32` at 1024.
An earlier controlled cohort, before matching the scalar process's helper imports, also
improved all ten pairs; its records are retained separately and its RSS is not compared here.

## Reproduction

From `python/`, with the locked CPU development environment, run each mode in a fresh process:

```sh
PYTHONPATH=. uv run --no-sync python benchmarks/integer_attention.py scalar --tokens 512 --warmups 3 --iterations 10
PYTHONPATH=. uv run --no-sync python benchmarks/integer_attention.py tiled --tokens 512 --warmups 3 --iterations 10
PYTHONPATH=. uv run --no-sync python benchmarks/integer_attention.py scalar --tokens 1024 --warmups 3 --iterations 10
PYTHONPATH=. uv run --no-sync python benchmarks/integer_attention.py tiled --tokens 1024 --warmups 3 --iterations 10
```

The seed is 621; each workload has four query heads, two KV heads and head dimension 64.
Queries have the token-major transpose strides used by the reference; keys use its GQA
repeat and transpose. Each process uses two PyTorch CPU threads. The previous output is
released before each measured call. JSON reports raw samples, median, peak process RSS and
a digest of synthetic output only. Alternate mode order between successive pairs.
