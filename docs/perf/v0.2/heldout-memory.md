# Held-out scoring memory

RFC-0005's default held-out split has 64 sequences of 2048 positions. At vocabulary 128256,
retaining all reference logits in float64 alone requires 134,486,163,456 bytes (125.25 GiB),
before the concatenation copy and KL scratch. Issue [#611](https://github.com/AbdelStark/tessera/issues/611)
removes that allocation from `quantize` without changing scale policy or integer semantics.

## Lifetime and arithmetic

`iter_reference_logits` builds one BF16 model and yields CPU float64 logits by window. Scoring
closes it on success and failure, releasing the model and restoring deterministic-algorithm
and gradient modes. The reference's first window runs before reading transformed checkpoint
groups. Reference inference repeats for every candidate alpha, trading compute for capacity.

Each integer window runs through the existing int64 kernels and range checks, without
recording sites, residual boundaries, norm roots, or attention matrices for later layers.
`trace` retains its complete witness and remains the conformance/differential route. The
embedding's temporary int64 matrix is released immediately after the token gather.

Integer logits convert to float64 **before** multiplication by their real dyadic scale:
PyTorch's int64 tensor multiplied by a Python float otherwise loses odd integers above
float32's exact range. KL keeps its original float64 per-row computation. Only one scalar per
position survives each window; concatenate these scalars in position order before the final
mean. Averaging window means instead changes ragged-window weighting and floating-point
rounding. Alpha tie-breaking remains the existing smallest-alpha rule.

## Measurement

Measured on 2026-10-29 on an Apple M1 Max, 64 GiB RAM, using Python 3.13.18, PyTorch 2.14.1,
Transformers 5.20.1 and two PyTorch CPU threads. Five alternating fresh processes per mode;
peak RSS includes the interpreter and libraries. Other CPU work ran concurrently, so elapsed
times describe these runs and do not establish isolated throughput.

The KL workload has 32 synthetic windows, 128 positions and vocabulary 8192. The reference
workload repeats the two-layer integer conformance model to 64 layers, at 256 positions.
The latter is a synthetic memory workload, not a trained 64-layer model or proof.

| Mode | Median peak RSS | RSS range | Median elapsed |
|---|---|---|---|
| Eager KL logits and concatenations | 2,837,053,440 B | 2,796,699,648–2,860,285,952 B | 1.406 s |
| Streamed KL rows | 577,732,608 B | 488,079,360–625,672,192 B | 1.059 s |
| Integer reference with full trace | 676,724,736 B | 667,549,696–685,195,264 B | 3.957 s |
| Integer reference with logits only | 418,168,832 B | 408,354,816–438,386,688 B | 4.272 s |

Median peak RSS falls by 79.6% for KL and 38.2% for the integer reference. All ten KL runs
produce the identical float64 result `0x1.b78d8158a4c87p+2`; all ten reference runs give logits
SHA-256 `3b9d1d0cbf555c33683b012f22da332b9277bbf3c182df73521fcb8ffb828c69`.

Reproduce each mode in a separate process from `python/`, with locked CPU development dependencies:

```sh
PYTHONPATH=. uv run --no-sync python benchmarks/heldout_memory.py kl-eager
PYTHONPATH=. uv run --no-sync python benchmarks/heldout_memory.py kl-stream
PYTHONPATH=. uv run --no-sync python benchmarks/heldout_memory.py reference-trace
PYTHONPATH=. uv run --no-sync python benchmarks/heldout_memory.py reference-logits
```

## Equivalence and adversarial evidence

The existing independent integer-oracle cases exercise logits-only and full-trace paths.
`test_assemble.py` compares selected alpha and encoded D against eager BF16/trace scoring
across all four profile/scale combinations. Dedicated cases check ragged position weighting,
the original final reduction order, missing/extra source windows, iterator cleanup, window
and attention-matrix lifetimes, and odd large integer logits before scaling.

An isolated worktree's no-op control passes. Each of five planted faults fails on its own
test assertion: eagerly retain every BF16 window; restore full trace retention; average window
means; multiply integer logits before widening; and omit the extra-window check. The normal
Python suite passes all 357 cases, together with strict lint/types and repository policy gates.

The adversarial reading confirms that the recording flag controls retained outputs only:
all matmuls, norm/lookup checks, requantizations and residual results remain common to both
routes (SC-7). Source count/shape errors fail closed before choosing alpha. No verifier code,
proof encoding, transcript label, parameter set, masking rule or integer bound changes.
Existing Python secret-data classification remains: statistics, weights, tokens, activations
and KL rows are not added to errors or logs. These changes make no new zeroization guarantee.

## Remaining real-model work

Memory still grows with the INT8 model and BF16 reference; one window retains logits and KL
scratch proportional to positions times vocabulary. Large lookup cores and int64 matmul
costs remain. These synthetic measurements do not qualify full 1B/8B quantization, fidelity,
prover working sets, or a certifying-parameter proof checked by the independent verifier.
Protocol/1 remains non-hiding.
