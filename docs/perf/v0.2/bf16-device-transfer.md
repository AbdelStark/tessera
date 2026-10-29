# BF16 transfer before CPU conversion

Issue [#609](https://github.com/AbdelStark/tessera/issues/609) fixes two device transfers in
`tessera_quant.capture`: calibration observations and held-out reference logits.
[RFC-0005](../../rfcs/RFC-0005-integer-model.md#calibration) requires float64 statistics
reduced on CPU, so the reference's BF16 values must reach that conversion unchanged.

## Native reproduction

On an Apple M1 Max with Python 3.13.18, locked PyTorch 2.14.1 and Transformers 5.20.1:

```python
import torch

bits = torch.arange(65536, dtype=torch.int32).to(torch.int16)
values = bits.view(torch.bfloat16)
values = values[torch.isfinite(values)]
expected = values.to(torch.float64)
device_values = values.to("mps")
combined = device_values.to("cpu", torch.float64)
separate = device_values.to("cpu").to(torch.float64)
assert torch.equal(separate.view(torch.int64), expected.view(torch.int64))
```

All 65,280 finite BF16 patterns, including both signed zeros, survive the separate transfer
and CPU conversion bit for bit. The raw BF16 round trip is also exact. The combined
device/dtype operation fails this comparison and produces nonfinite values. Source embedding
finiteness, raw Metal embedding copies and gathers were checked independently; the combined
float64 copy is the failing step.

`python/tests/test_device_transfer.py` tests the actual observation and reference-logit
entry points against the CPU oracle, for both CPU and native Metal. Both Metal cases failed
on their assertions before the fix; both CPU cases passed. Metal cases skip when no device is
available, so Linux CI alone does not qualify this native path. No source activation arithmetic,
quantization scale policy or artifact encoding changes.

## Real-model probe

The complete public [Llama-3.2-1B snapshot](https://huggingface.co/NousResearch/Llama-3.2-1B/tree/a9745ffc3556f145a830ac0c203509ba860582a3)
was pinned to `a9745ffc3556f145a830ac0c203509ba860582a3` and its safetensors SHA-256 checked.
For the existing importer's untied layout, the shared embedding was explicitly copied into a
separate LM-head matrix and the configuration's tying flag changed. The original and copied
BF16 models gave identical logits on two probe sequences; this external normalization is not
native tied-checkpoint importer support or a fidelity result.

A full 2,048-token C4 window through all 16 layers failed calibration on Metal with
`ERR-QUANT-CALIB` before the transfer correction. The isolated correction produced all 194
finite statistics points in 8.51 seconds, including model loading. The CPU run's first window
took about 213 seconds after loading. These observations were made with concurrent fuzzing;
they are a practical path check, not a general speed ratio or a completed calibration result.

The full 512-sequence C4 calibration is separate and must be checked at actual completion.
Neither this transfer fix nor its probes establish a real-model integrity proof, fidelity
qualification, a Metal prover, or zero knowledge.
