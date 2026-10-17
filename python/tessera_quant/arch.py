# SPDX-License-Identifier: Apache-2.0 OR MIT
"""The architecture of `D` (03 §2): `ArchSpec`, read exactly from a Llama-3.1 configuration."""

import json
from collections.abc import Mapping
from dataclasses import dataclass
from decimal import Decimal
from enum import Enum
from fractions import Fraction
from typing import Any, cast

from tessera_quant.errors import ArchitectureError

ROPE_CONSTANT_BITS = 14
"""The bits of `D`'s fixed-point RoPE constants (03 §2, RFC-0005 'Integer semantics')."""


class ArchFamily(Enum):
    """The model family (03 §2)."""

    LlamaV3 = 0
    """Llama-3 and Llama-3.1, the only family of v1."""


@dataclass(frozen=True)
class RopeSpec:
    """Llama-3.1's RoPE (03 §2): the integer parameters of RFC-0008's `Rope` table.

    The base and the `llama3` frequency scaling, each an integer, as `D` requires.
    """

    theta: int
    """The base `θ`, below 2^32."""
    factor: int
    """The scaling `factor`, below 2^16."""
    low_freq_factor: int
    """`low_freq_factor`, below 2^16."""
    high_freq_factor: int
    """`high_freq_factor`, below 2^16."""
    original_max: int
    """`original_max_position_embeddings`, below 2^32."""
    bits: int = ROPE_CONSTANT_BITS
    """The precision of the fixed-point constants."""


@dataclass(frozen=True)
class ArchSpec:
    """The architecture parameters of a Llama-3.1 model (03 §2), each below 2^32.

    `rms_eps` is exact.
    """

    n_layers: int
    d_model: int
    n_heads: int
    n_kv_heads: int
    d_head: int
    d_ff: int
    vocab: int
    max_context: int
    rope: RopeSpec
    rms_eps: Fraction
    family: ArchFamily = ArchFamily.LlamaV3


def read_config(text: str) -> ArchSpec:
    """The `ArchSpec` of a Hugging Face `config.json` for a Llama-3.1 model.

    Decimal values are read as written, so `"rms_norm_eps": 1e-05` becomes exactly 1/100000.

    Raises:
        ArchitectureError: for a configuration of another family, or with a value `D` does not
            support.
    """
    try:
        config: Any = json.loads(text, parse_float=Decimal)
    except ValueError as error:
        raise ArchitectureError(f"config.json is not JSON: {error}") from None
    if not isinstance(config, dict):
        raise ArchitectureError("config.json is not an object")
    return _arch(_Config(cast(dict[str, Any], config)))


class _Config:
    """Typed reads of a configuration's keys, each failure an `ArchitectureError` naming the key."""

    def __init__(self, values: Mapping[str, Any]) -> None:
        self._values = values

    def value(self, key: str) -> Any:
        if key not in self._values:
            raise ArchitectureError(f"config.json has no {key}")
        return self._values[key]

    def integer(self, key: str, bits: int = 32) -> int:
        """A positive integer below `2^bits`, the width of its `ArchSpec` field."""
        value = self.value(key)
        if isinstance(value, bool) or not isinstance(value, int) or not 0 < value < 1 << bits:
            raise ArchitectureError(f"config.json's {key} is not a positive integer below 2^{bits}")
        return value

    def whole(self, key: str, bits: int) -> int:
        """An integer below `2^bits`, written with or without a fractional part: `500000.0`."""
        exact = self.exact(key)
        if exact.denominator != 1 or exact >= 1 << bits:
            raise ArchitectureError(f"config.json's {key} is not an integer below 2^{bits}")
        return exact.numerator

    def exact(self, key: str) -> Fraction:
        value = self.value(key)
        if isinstance(value, bool) or not isinstance(value, int | Decimal):
            raise ArchitectureError(f"config.json's {key} is not a number")
        exact = Fraction(value)
        if exact <= 0:
            raise ArchitectureError(f"config.json's {key} is not positive")
        return exact

    def expect(self, key: str, expected: object, *, default: object = None) -> None:
        value = self._values.get(key, default)
        if value != expected:
            raise ArchitectureError(f"config.json's {key} is {value!r}, where D needs {expected!r}")

    def section(self, key: str) -> "_Config":
        value = self.value(key)
        if not isinstance(value, dict):
            raise ArchitectureError(f"config.json's {key} is not an object")
        return _Config(cast(dict[str, Any], value))


def _arch(config: _Config) -> ArchSpec:
    config.expect("model_type", "llama")
    config.expect("hidden_act", "silu")
    config.expect("attention_bias", False, default=False)
    config.expect("mlp_bias", False, default=False)
    config.expect("tie_word_embeddings", False, default=False)
    config.expect("torch_dtype", "bfloat16")
    d_model = config.integer("hidden_size")
    n_heads = config.integer("num_attention_heads")
    n_kv_heads = config.integer("num_key_value_heads")
    if d_model % n_heads != 0 or n_heads % n_kv_heads != 0:
        raise ArchitectureError("the heads do not divide hidden_size, or the KV heads the heads")
    d_head = d_model // n_heads
    config.expect("head_dim", d_head, default=d_head)
    if d_head % 2 != 0:
        raise ArchitectureError("RoPE rotates pairs, and the head dimension is odd")
    scaling = config.section("rope_scaling")
    scaling.expect("rope_type", "llama3")
    rope = RopeSpec(
        theta=config.whole("rope_theta", 32),
        factor=scaling.whole("factor", 16),
        low_freq_factor=scaling.whole("low_freq_factor", 16),
        high_freq_factor=scaling.whole("high_freq_factor", 16),
        original_max=scaling.integer("original_max_position_embeddings"),
    )
    return ArchSpec(
        n_layers=config.integer("num_hidden_layers"),
        d_model=d_model,
        n_heads=n_heads,
        n_kv_heads=n_kv_heads,
        d_head=d_head,
        d_ff=config.integer("intermediate_size"),
        vocab=config.integer("vocab_size"),
        max_context=config.integer("max_position_embeddings"),
        rope=rope,
        rms_eps=config.exact("rms_norm_eps"),
    )


def tensor_shapes(arch: ArchSpec) -> dict[str, tuple[int, ...]]:
    """Every tensor of a Llama-3.1 checkpoint of `arch`, by its Hugging Face name, with its shape.

    These are exactly the tensors a checkpoint must hold, no more and no fewer.
    """
    d, q, kv = arch.d_model, arch.n_heads * arch.d_head, arch.n_kv_heads * arch.d_head
    shapes: dict[str, tuple[int, ...]] = {"model.embed_tokens.weight": (arch.vocab, d)}
    for layer in range(arch.n_layers):
        prefix = f"model.layers.{layer}"
        shapes |= {
            f"{prefix}.input_layernorm.weight": (d,),
            f"{prefix}.self_attn.q_proj.weight": (q, d),
            f"{prefix}.self_attn.k_proj.weight": (kv, d),
            f"{prefix}.self_attn.v_proj.weight": (kv, d),
            f"{prefix}.self_attn.o_proj.weight": (d, q),
            f"{prefix}.post_attention_layernorm.weight": (d,),
            f"{prefix}.mlp.gate_proj.weight": (arch.d_ff, d),
            f"{prefix}.mlp.up_proj.weight": (arch.d_ff, d),
            f"{prefix}.mlp.down_proj.weight": (d, arch.d_ff),
        }
    shapes |= {"model.norm.weight": (d,), "lm_head.weight": (arch.vocab, d)}
    return shapes
