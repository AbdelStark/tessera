# tessera-quant

The quantization front-end of Tessera. It turns a Llama-3.1 checkpoint into the integer model `D` that the prover proves ([RFC-0005](../docs/rfcs/RFC-0005-integer-model.md)). Its version is the prover release train's ([09 §1](../docs/spec/09-release-and-versioning.md#1-what-is-versioned)), and `cargo xtask lint` keeps it equal to the workspace `Cargo.toml`'s.

## Setup

[uv](https://docs.astral.sh/uv/) builds the environment from `uv.lock`. PyTorch comes from its own index through exactly one of two extras, which `uv.lock` pins to the same release:

```sh
uv sync --extra cpu      # CPU wheels: CI and the tiny conformance models
uv sync --extra cu130    # CUDA 13.0 wheels, Linux only: calibrating 8B and up
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
