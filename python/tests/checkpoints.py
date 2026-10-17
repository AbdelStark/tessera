# SPDX-License-Identifier: Apache-2.0 OR MIT
"""Tiny Llama-3.1 checkpoints for the tests: configurations, weights and writers."""

import json
import struct
from fractions import Fraction
from pathlib import Path
from typing import Any, cast

import safetensors.torch
import tokenizers
import tokenizers.models
import tokenizers.pre_tokenizers
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


def llama_like(seed: int) -> Weights:
    """Random weights at Llama's magnitudes: unit embeddings, matrices near 0.02, gains near 1."""
    generator = torch.Generator().manual_seed(seed + 1)
    weights = tensors(TINY_ARCH, seed)
    for name, t in weights.items():
        if t.dim() == 1:
            gain = 1 + 0.1 * torch.randn(t.shape, generator=generator, dtype=torch.float64)
            weights[name] = gain.to(torch.bfloat16)
        elif name != "model.embed_tokens.weight":
            weights[name] = (t.to(torch.float64) * 0.02).to(torch.bfloat16)
    return weights


CHARACTERS = "abcdefghijklmnopqrstuvwxyz0123456789 .,"
"""What the character tokenizer knows: token `2 + i` is `CHARACTERS[i]`, 0 unknown, 1 `<s>`."""


def char_tokenizer(directory: Path) -> None:
    """Writes `tokenizer.json`: one token per character, as `tokenizers` saves it."""
    vocab = {"[UNK]": 0, "<s>": 1} | {c: 2 + i for i, c in enumerate(CHARACTERS)}
    tokenizer = tokenizers.Tokenizer(tokenizers.models.WordLevel(vocab, "[UNK]"))
    tokenizer.pre_tokenizer = tokenizers.pre_tokenizers.Split(
        tokenizers.Regex("."), behavior="isolated"
    )
    tokenizer.save(str(directory / "tokenizer.json"))


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
