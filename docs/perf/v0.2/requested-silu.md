# Requested SiLU evaluation

Issue [#613](https://github.com/AbdelStark/tessera/issues/613) removes dense SiLU core generation
from the logits-only integer reference. RFC-0008 defines a lookup's value by its class and
`h = key >> r_class`; inference needs the values at the keys actually read, rather than every
possible key's core value and a table digest. Full table generation, its digest and the full
trace route remain available unchanged for valid public parameters.

## Bounds and equivalence

`silu_lookup` shares parameter validation with full table generation: class count, packed key
width, shifts, the `2^24` core limit, and fixed-point input/output exponent bounds. It validates
the signed key and class before evaluating values, even on an empty request. Explicitly reject
negative shifts, which the unsigned TCE field cannot encode. Both paths use the same exact
`silu_value` fixed-point formula, floor right shift on signed keys, half-up output rounding and
INT8 clamp. No floating point enters inference.

Per call, flatten shifted keys, compute distinct values and an inverse map, evaluate the formula
once per distinct `h`, and restore the input's shape. Arrays and sorting scratch grow with the
input window and requested distinct keys. There is no global cache indexed by activations.
This changes runtime and allocation strategy, not a table definition, proof format or verifier
predicate. Python reference code makes no new constant-time or zeroization guarantee; keys and
values stay out of errors and logs (05 §3).

Tests compare independent oracle lookup vectors, complete small signed domains with different
classes/shifts (including zero), non-contiguous and empty views, and shape/dtype preservation.
Parameter, key and class refusals occur before value generation with `ERR-INTERNAL`. A work
counter limits a request to five evaluations even though its possible core has `2^23` values,
and verifies that a later call computes its own values. The logits-only route is held to every
conformance model's oracle logits with dense generation forbidden. Existing full trace and
encoded-D comparisons continue to pass.

An isolated no-op control passes; five planted faults fail on assertions: round instead of
floor the shifted key, omit parameter validation, omit signed-key validation, restore dense
generation in the logits-only route, and recompute repeated keys. The Python suite passes
379 cases, with strict Ruff and pyright checks.

## Synthetic measurement

Measured 2026-10-29 on Apple M1 Max/64 GiB, Python 3.13.18, PyTorch 2.14.1, NumPy 2.5.4 and
Transformers 5.20.1. Five alternating fresh processes per mode, with other CPU work active.
The synthetic table has `b = 24`, one shift of 4, input exponent -3 and output exponent -2:
`2^20` possible `h` values. Six width-8192 rows request only 256 distinct `h` values.

| Mode | Median elapsed | Elapsed range | Median peak RSS |
|---|---|---|---|
| Dense core, digest and lookup | 6.826 s | 5.798–8.931 s | 385,433,600 B |
| Requested values | 0.00372 s | 0.00357–0.00541 s | 339,017,728 B |

RSS includes Python/libraries. Every run produces the identical result SHA-256
`9726f9cc4432f02f458fb22af3ed0ea534f57d6bc90880e75c5377c3c4977434`.
Reproduce in separate processes from `python/`, with locked CPU development dependencies:

```sh
PYTHONPATH=. uv run --no-sync python benchmarks/silu_requested.py dense
PYTHONPATH=. uv run --no-sync python benchmarks/silu_requested.py sparse
```

These measure a request needing a small fraction of its possible core. Sorting and formula
work grow with the request; this is not a universal throughput ratio or a proof benchmark.

## Provisional real 1B diagnostic

The local complete [NousResearch/Llama-3.2-1B checkpoint](https://huggingface.co/NousResearch/Llama-3.2-1B/tree/a9745ffc3556f145a830ac0c203509ba860582a3) at revision
`a9745ffc3556f145a830ac0c203509ba860582a3` has 1,235,814,400 physical parameters and tied output
weights. An external normalization copied its embedding bit for bit into an explicit LM head
and disabled tying; two BF16 probes had identical logits before/after. This is not native tied
checkpoint support. The normalized source was calibrated on the full pinned RFC-0005 C4 draw:
512 sequences of 2048 positions, with the prescribed 64 held-out sequences reserved.

At servable/affine alpha `1/2`, a provisional integer candidate contains 1,498,415,104 folded
weight entries and exports as a 1,499,047,757-byte D with real checkpoint/calibration provenance.
The current compiler accepts it with `ni-100`; debug and release produce identical claim-list
and execution-plan bytes. Release compilation took 379.37 s and peaked at 3,001,335,808 B RSS.
These are actual runs, not the cost report's extrapolated times.

An in-memory candidate diagnostic constructing the same weights and forwarding six positions
through all 16 layers completed in 872.21 s with dense SiLU and 27.51 s with requested values.
Peak RSS was 11,268,079,616 and 11,217,747,968 B, respectively, dominated by candidate
construction. Each includes transforms/scale selection; concurrent jobs differed. These single
runs establish that the route completes, not a controlled end-to-end speedup or numerical
comparison on that real prompt. Numerical equivalence is established by the oracle/domain
tests above. The diagnostic's placeholder in-memory provenance was never exported; the separate
D export has its actual provenance.

The full held-out alpha selection, fidelity qualification, proof generation and independent
verification of a real-model proof remain unfinished. The default compiler report has over
8.4 billion lookup-table rows: sparse reference evaluation does not shrink those proof tables,
the witness floor or prover stage memory. Protocol/1 remains non-hiding.
