# tessera-quant

The quantization front-end of Tessera. It turns a Llama-3.1 checkpoint into the integer model `D` that the prover proves ([RFC-0005](../docs/rfcs/RFC-0005-integer-model.md)). Its version is the prover release train's ([09 §1](../docs/spec/09-release-and-versioning.md#1-what-is-versioned)), and `cargo xtask lint` keeps it equal to the workspace `Cargo.toml`'s.

## Setup

[uv](https://docs.astral.sh/uv/) builds the environment from `uv.lock`. PyTorch comes from its own index through exactly one of two extras, which `uv.lock` pins to the same release:

```sh
uv sync --extra cpu      # CPU wheels: CI and the tiny conformance models
uv sync --extra cu130    # CUDA 13.0 wheels, Linux only: calibrating 8B and up
```

## Use

```python
from pathlib import Path

from tessera_quant import import_checkpoint

# A Hugging Face checkpoint directory: config.json and safetensors shards. Pickle files are refused.
model = import_checkpoint(Path("Llama-3.1-8B"))
q = model.tensor("model.layers.0.self_attn.q_proj.weight")  # BF16, read on demand
```

The front-end and the Rust tools exchange TCE artifacts only ([RFC-0003](../docs/rfcs/RFC-0003-canonical-encoding.md)). `tessera_quant.tce` writes them as secret files, streamed and renamed into place, and reads them mapped, digest first:

```python
from tessera_quant.integer_model import INTEGER_MODEL
from tessera_quant.tce import INTEGER_MODEL_LIMITS, read_artifact, write_artifact

d = read_artifact(Path("model.d"), INTEGER_MODEL, limits=INTEGER_MODEL_LIMITS)
d.weights[0].data  # INT8, a read-only view of the mapped file
write_artifact(Path("copy.d"), INTEGER_MODEL, d, limits=INTEGER_MODEL_LIMITS)  # mode 0600
```

`tessera_quant.tables` generates the lookup tables of `D` exactly as RFC-0008 defines them, with the digests the Rust generators compute:

```python
from fractions import Fraction

import numpy as np

from tessera_quant.tables import Exp, lookup_table

softmax = lookup_table(0, Exp(key_bits=22, sigma=16, scale=Fraction(1, 2896)))
softmax.lookup(np.array([0, 2896, 100000]))  # array([65536, 24109, 0]): round(2^16 · e^(-k/2896))
```

`run_reference` computes `D`'s logits on a sequence in int64, exactly as RFC-0005 defines them, and `tessera_quant.reference.trace` also returns every tensor `D` commits, for differential tests. Both refuse a model whose values could leave int64 rather than wrap:

```python
import numpy as np

from tessera_quant import run_reference

logits = run_reference(d, np.array([128000, 791, 6864], dtype=np.uint32))  # int64, [3, vocab]
```

## Checks

The `python` CI job runs these on Python 3.11, 3.12 and 3.13:

```sh
uv sync --locked --extra cpu
uv run --no-sync ruff check
uv run --no-sync ruff format --check
uv run --no-sync pyright      # strict, as pyproject.toml configures it
uv run --no-sync pytest
```
