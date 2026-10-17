# SPDX-License-Identifier: Apache-2.0 OR MIT
"""Tiny Llama-3.1 checkpoints for the tests: configurations, weights and writers."""

import json
import struct
from fractions import Fraction
from pathlib import Path
from typing import Any, cast

import safetensors.torch
import torch

from tessera_quant import (
    ArchSpec,
    RopeSpec,
    tensor_shapes,
)

Weights = dict[str, torch.Tensor]

TINY: dict[str, Any] = {
    "architectures": ["LlamaForCausalLM"],
    "model_type": "llama",
    "hidden_act": "silu",
    "hidden_size": 32,
    "intermediate_size": 64,
    "num_attention_heads": 4,
    "num_key_value_heads": 2,
    "num_hidden_layers": 2,
    "vocab_size": 64,
    "max_position_embeddings": 256,
    "rms_norm_eps": 1e-05,
    "rope_theta": 500000.0,
    "rope_scaling": {
        "factor": 8.0,
        "low_freq_factor": 1.0,
        "high_freq_factor": 4.0,
        "original_max_position_embeddings": 128,
        "rope_type": "llama3",
    },
    "attention_bias": False,
    "mlp_bias": False,
    "tie_word_embeddings": False,
    "torch_dtype": "bfloat16",
}

# Llama-3.1-8B's published config.json, the keys `D` reads.
LLAMA_3_1_8B: dict[str, Any] = TINY | {
    "hidden_size": 4096,
    "intermediate_size": 14336,
    "num_attention_heads": 32,
    "num_key_value_heads": 8,
    "num_hidden_layers": 32,
    "vocab_size": 128256,
    "max_position_embeddings": 131072,
    "rope_scaling": TINY["rope_scaling"] | {"original_max_position_embeddings": 8192},
}

TINY_ARCH = ArchSpec(
    n_layers=2,
    d_model=32,
    n_heads=4,
    n_kv_heads=2,
    d_head=8,
    d_ff=64,
    vocab=64,
    max_context=256,
    rope=RopeSpec(theta=500000, factor=8, low_freq_factor=1, high_freq_factor=4, original_max=128),
    rms_eps=Fraction(1, 100000),
)

LAYER0_Q = "model.layers.0.self_attn.q_proj.weight"


def tensors(arch: ArchSpec, seed: int = 7) -> Weights:
    generator = torch.Generator().manual_seed(seed)
    return {
        name: torch.randn(shape, generator=generator).to(torch.bfloat16)
        for name, shape in tensor_shapes(arch).items()
    }


def save(weights: Weights, path: Path) -> None:
    # safetensors' torch helpers are only partly typed.
    cast(Any, safetensors.torch).save_file(weights, str(path))


def write(directory: Path, config: dict[str, Any], weights: Weights, shards: int = 1) -> Path:
    directory.mkdir(parents=True, exist_ok=True)
    (directory / "config.json").write_text(json.dumps(config), encoding="utf-8")
    if shards == 1:
        save(weights, directory / "model.safetensors")
        return directory
    names = list(weights)
    weight_map: dict[str, str] = {}
    for shard in range(shards):
        file = f"model-{shard + 1:05}-of-{shards:05}.safetensors"
        part = names[shard::shards]
        save({name: weights[name] for name in part}, directory / file)
        weight_map |= dict.fromkeys(part, file)
    index = {"metadata": {}, "weight_map": weight_map}
    (directory / "model.safetensors.index.json").write_text(json.dumps(index), encoding="utf-8")
    return directory


def write_header_only(path: Path, shapes: dict[str, tuple[int, ...]], dtype: str = "BF16") -> None:
    """A safetensors file with the right header and a sparse data region of the right length."""
    header: dict[str, Any] = {}
    offset = 0
    for name, shape in shapes.items():
        size = 2 * int(torch.Size(shape).numel())
        header[name] = {
            "dtype": dtype,
            "shape": list(shape),
            "data_offsets": [offset, offset + size],
        }
        offset += size
    raw = json.dumps(header).encode()
    raw += b" " * (-len(raw) % 8)
    with path.open("wb") as file:
        file.write(struct.pack("<Q", len(raw)))
        file.write(raw)
        file.truncate(8 + len(raw) + offset)
