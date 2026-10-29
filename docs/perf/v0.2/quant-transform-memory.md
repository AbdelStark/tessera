# Checkpoint transform memory

RFC-0005's folding, equalization and migration act within one transformer layer, or on the
final norm and LM head. Issue [#605](https://github.com/AbdelStark/tessera/issues/605) removes
whole-model float64 dictionaries from `quantize`: it caches one transformed group, while
retaining the INT8 output and small channel statistics. The embedding is a separate group.
The BF16 reference runs before these float64 transformations.

## Measurement

Measured on 2026-10-29 on an Apple M1 Max, 64 GiB RAM, with two PyTorch CPU threads,
Python 3.15.2, PyTorch 2.14.1 and Transformers 5.20.1. Five alternating fresh processes
per implementation read the same warm-cache synthetic safetensors checkpoint. Other CPU
work was running, so the times describe this run rather than isolated throughput.

The checkpoint has 50,348,544 parameters: 16 layers, width 512, FFN width 1536, eight
query heads, two KV heads and vocabulary 2048. Each run transforms at `alpha = 3/5` and
hashes every transformed weight and statistic. RSS includes the interpreter and libraries.

| Implementation | Median peak RSS | Range | Median elapsed | Range |
|---|---|---|---|---|
| Eager source and whole-model transforms | 1,334,165,504 B | 1,317,748,736–1,373,995,008 B | 0.371 s | 0.370–0.512 s |
| One transformed group cached | 631,635,968 B | 572,080,128–663,093,248 B | 0.540 s | 0.533–0.547 s |

Median peak RSS decreases by 52.7%; this small transform scan takes 45.6% longer because
group revisits reread and transform weights. It is a capacity improvement, not a throughput
claim. Every run gives SHA-256
`9886acf65d759ed15307b949a3417b467ed01802c8aedfcfae215ccbe1e407ed`.

Reproduce from `python/` with the locked CPU development environment:

```sh
PYTHONPATH=.:tests uv run --no-sync python benchmarks/checkpoint_transforms.py prepare /tmp/tessera-transform-bench
PYTHONPATH=.:tests uv run --no-sync python benchmarks/checkpoint_transforms.py eager /tmp/tessera-transform-bench
PYTHONPATH=.:tests uv run --no-sync python benchmarks/checkpoint_transforms.py stream /tmp/tessera-transform-bench
```

Repeat the last two commands in separate processes. The benchmark uses synthetic data only.
The memory result is for transforms, not full calibration, quantization or proving. In
particular, scale selection can retain a borrowed tensor and allocate matrix-sized rounding
scratch, the INT8 output grows with the model, and the BF16 reference is still resident.

## Equivalence and regression evidence

`python/tests/test_assemble.py` compares streamed transformations with the existing eager
float64 implementation at all four migration strengths, in reverse and mixed group order.
It compares complete encoded integer models, including provenance and selected alpha, across
`servable`/`minimal` and `pot`/`affine`. These comparisons use nonzero checkpoints and two
alpha candidates, rather than only zero tensors or a fixed choice.

Weak references check that both a transformed weight and transformed statistics have been
released before the next group's source is read. A separate test refuses checkpoint reads
before the BF16 reference. In an isolated worktree, the ten new cases pass unchanged and
catch four planted faults: retaining the previous group, restoring eager source loading,
returning unmigrated statistics, and returning a stale group's cache. Each mutant fails on
a test assertion, without an import or syntax failure.

## Real-model status

The complete public [NousResearch Llama-3.1-8B checkpoint](https://huggingface.co/NousResearch/Meta-Llama-3.1-8B)
at revision `1f47e50cdbe801ad8a5174156ec3a0655108fb9f` was downloaded as safetensors and
accepted by the importer: 291 tensors, 8,030,261,248 parameters. This checks actual shard
headers and file bounds, rather than sparse placeholders. No remote Python code or pickle
weights are loaded.

With Python 3.13.18 and the same locked PyTorch/Transformers versions, constructing the BF16
reference and running a two-token forward through all 32 layers took 24.80 s and peaked at
26,256,949,248 B RSS. The `[2, 128256]` logits were finite. This is reference inference only.
The full Python suite passed all 341 tests on Python 3.13, together with strict type and
format checks. A separate Python 3.15 full-suite attempt crashed in the third-party threaded
checkpoint loader used by `test_capture`; the supported-version run is the validation result.

This does not establish the full fidelity gates of [#93](https://github.com/AbdelStark/tessera/issues/93),
the full engine differential of [#312](https://github.com/AbdelStark/tessera/issues/312), or a
billion-parameter proof. This measurement preceded the
[regenerated weight commitment](../v0.3/weight-regeneration.md), which replaced encoded-weight
residency. The witness and large lookup working sets remain separate costs; the
[later real-model attempt](reference-int8-tiles.md#actual-1b-candidate-and-proof-status) records
the exported candidate, actual commitment and proof-planning failure. Protocol/1 remains non-hiding.
