# Integer reference matmul tiles

Issue [#615](https://github.com/AbdelStark/tessera/issues/615) accelerates logits-only CPU
`servable` inference while preserving RFC-0005's integer semantics (SC-7, INV-DM-6). The full
trace retains the int64 implementation as a differential oracle; `minimal` retains its wide
route. Wide values and missing/unsupported integer kernels fall back to the existing scalar
path. Allocator failures propagate instead of retrying with a full int64 weight allocation.

## Exactness and residency

For each signed INT8 weight, `hi = w >> 1`, `lo = w & 1`, and `w = 2*hi + lo`, including -128.
`hi` lies in [-64, 63], and `lo` in [0, 1]. A pair's absolute product sum is at most 16384;
even an unsigned-offset activation backend has a pair bound of `2*255*64 = 32640`, within
signed int16. An adversarial saturating-pair backend gives a wrong direct product on the
test's extremes, while the two-digit route matches the integer oracle.

Tiles have at most 128 rows, 1024 columns and 1024 inner entries. High/low dot products fit
INT32: absolute bounds `2^23` and `2^17`, respectively. Widen both partials to int64 before
doubling/adding. The full result and every partial sum have the public absolute bound
`128*128*K < 2^63`; operands outside INT8 fall back before conversion. No floating-point
conversion or approximation is introduced.

Only a bounded weight tile is copied from the original INT8 array, including read-only mapped
D artifacts. The necessary output remains int64; packing and accumulation scratch grow with
tile dimensions instead of the whole weight matrix. The existing embedding gather still has
an int64 conversion. No tensor/key cache survives a call; no new Python constant-time or
zeroization guarantee is claimed. Values reach neither errors nor logs (05 §3).

Tests cover signed extremes, zero axes and all three kinds of block/tail crossing, readonly
inputs, an odd result above `2^32`, int64 accumulation, wide-value refusal, kernel absence,
allocator errors, and tile-sized INT8 copies. A native-backend test is separate from always-run
contract/oracle cases; unsupported machines explicitly skip native acceleration evidence.
All 395 Python cases pass on this M1 Max with no skips, alongside strict lint and types.

An isolated no-op control passes. Nine planted faults fail assertions: wrong high digit;
missing low digit; INT32 total; truncation of wide activations; incomplete inner blocks;
oversized column blocks; zero output for an unsupported kernel; retry after allocator failure;
and copying the whole weight matrix. Block-end mutations run across the parametrized shapes,
including inputs larger than a tile, where the changed endpoint is actually reached.

## Synthetic measurement

Measured 2026-10-29 on Apple M1 Max/64 GiB, Python 3.13.18, PyTorch 2.14.1 and NumPy 2.5.4,
with two PyTorch CPU threads. Five alternating fresh processes per mode and shape; calibration
and fuzzing ran concurrently. The scalar baseline includes weight conversion and matmul;
it omits the scalar forward's separate magnitude scan. RSS includes Python and libraries.

| Rows x inner x columns | Mode | Median elapsed | Elapsed range | Median peak RSS |
|---|---|---|---|---|
| 512 x 2048 x 2048 | Scalar | 1.853 s | 1.736–2.446 s | 393,838,592 B |
| 512 x 2048 x 2048 | Tiled | 0.195 s | 0.156–0.461 s | 379,748,352 B |
| 16 x 2048 x 32768 | Scalar | 0.957 s | 0.926–1.299 s | 947,077,120 B |
| 16 x 2048 x 32768 | Tiled | 0.107 s | 0.104–0.154 s | 426,000,384 B |

The first workload has high operand reuse and targets integer compute; the second exposes
conversion residency. Median peak RSS falls 55.0% for the wider weight matrix. Every mode's
result matches its partner: SHA-256 `c8c80ce9b6cdc4a4b1b87c4e5044f5c9c9c0aa2e9e1e769923798a936bc9bfb0`
for the first shape, `6d6eae8084ea46786ef84e87e5457043501647013e3bb344cdd00e9e39de3f9e` for the second.
These are component measurements, not isolated hardware limits or proof throughput.

Reproduce each mode in a separate process from `python/` with locked CPU development dependencies:

```sh
PYTHONPATH=. uv run --no-sync python benchmarks/integer_matmul.py scalar
PYTHONPATH=. uv run --no-sync python benchmarks/integer_matmul.py tiled
PYTHONPATH=. uv run --no-sync python benchmarks/integer_matmul.py scalar --rows 16 --columns 32768
PYTHONPATH=. uv run --no-sync python benchmarks/integer_matmul.py tiled --rows 16 --columns 32768
```

## Actual 1B candidate and proof status

The [provisional exported 1B candidate](requested-silu.md#provisional-real-1b-diagnostic) was
read back from its real D artifact and run on six positions of the public diagnostic prompt.
Every logit matches the scalar fallback, with all 113 weight products accelerated and none
falling back. A paired run observed 17.933 s scalar and 4.831 s tiled, with other jobs active;
its 6,584,074,240 B peak RSS includes both routes. It is numerical differential evidence, not
a controlled end-to-end throughput or fidelity qualification.

The unchanged production build also completed actual `ni-100` weight commitment: 10 matrix
roots, 679.94 s and 7,465,140,224 B peak RSS. The following real-model proof attempt used a
private TokenFile containing a one-position prompt and its greedy continuation, with source
checkpoint/calibration provenance checked. It failed before producing a proof, after 1263.72 s
and 10,399,580,160 B peak RSS, with `ERR-INTERNAL`:

> a site's real layers read the same tensors over the same axes (RFC-0045 'One claim per input')

The failure is under investigation. The guard remains in place; these changes affect reference
inference only. There is no accepted real-model proof yet. Full held-out alpha selection,
fidelity gates, independent runtime soundness/verification and prover resource qualification
remain unfinished. Protocol/1 remains non-hiding.
