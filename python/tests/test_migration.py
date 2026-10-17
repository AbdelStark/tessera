# SPDX-License-Identifier: Apache-2.0 OR MIT
"""Gain folding, QK equalization and SmoothQuant migration (RFC-0005 'Pipeline', steps 2 and 4).

Each transform keeps the model's logits, in float64 on 64 sequences, to well within the 1e-3 the
issue allows, and its statistics are the transformed model's own: an independent recorder, over
the same sequences, finds the maxima the transform predicts. Equalization balances the queries
and keys of each pair and group, and migration at `alpha = 1/2` balances each input channel with its
weight column. The transforms act layer by layer, leave their inputs untouched, and handle zero
channels. `select_alpha` and `mean_kl` follow RFC-0005 'Calibration'.
"""

import math
from collections.abc import Callable, Mapping
from fractions import Fraction
from typing import Any, cast

import pytest
import torch
from checkpoints import TINY, TINY_ARCH, tensors
from transformers import LlamaConfig, LlamaForCausalLM
from transformers.models.llama import modeling_llama

from tessera_quant import TesseraQuantError
from tessera_quant.migration import (
    ALPHAS,
    equalize,
    fold,
    mean_kl,
    migrate,
    select_alpha,
    transform,
)

ARCH = TINY_ARCH
GROUP = ARCH.n_heads // ARCH.n_kv_heads
POINTS = ("q", "k", "v", "attn_out", "up", "h")

# Hugging Face's RoPE, taken before any test wraps it: each recording wraps this one.
ROPE: Any = cast(Any, modeling_llama).apply_rotary_pos_emb


def float64_weights(seed: int = 7) -> dict[str, torch.Tensor]:
    return {name: t.to(torch.float64) for name, t in tensors(ARCH, seed).items()}


def model_of(state: Mapping[str, torch.Tensor]) -> LlamaForCausalLM:
    """Hugging Face's Llama in float64 with eager attention, holding `state`."""
    # transformers types `from_dict` loosely.
    config: Any = cast(Any, LlamaConfig).from_dict(TINY)
    config._attn_implementation = "eager"
    llama: Any = LlamaForCausalLM(config)
    llama = llama.to(torch.float64)
    llama.load_state_dict(dict(state), strict=True)
    return cast(LlamaForCausalLM, llama.eval())


def sequences(count: int = 64, length: int = 16, seed: int = 3) -> torch.Tensor:
    generator = torch.Generator().manual_seed(seed)
    return torch.randint(0, ARCH.vocab, (count, length), generator=generator)


def logits_of(state: Mapping[str, torch.Tensor], tokens: torch.Tensor) -> torch.Tensor:
    with torch.no_grad():
        return cast(Any, model_of(state))(input_ids=tokens).logits


def maxima_of(
    state: Mapping[str, torch.Tensor], tokens: torch.Tensor, monkeypatch: pytest.MonkeyPatch
) -> dict[str, torch.Tensor]:
    """Each transformed point's per-channel maxima in the model `state`, recorded in float64
    from the projections and RoPE's own output: independent of the capture's code."""
    found: dict[str, torch.Tensor] = {}

    def observe(name: str, values: torch.Tensor) -> None:
        maxima = values.abs().reshape(-1, values.shape[-1]).amax(dim=0)
        held = found.get(name)
        found[name] = maxima if held is None else torch.maximum(held, maxima)

    model: Any = model_of(state)
    layer = [0]

    def recording_rope(q: torch.Tensor, k: torch.Tensor, *args: Any, **kwargs: Any) -> Any:
        q_embed, k_embed = ROPE(q, k, *args, **kwargs)
        for name, values in (("q", q_embed), ("k", k_embed)):
            batch, heads, positions, d_head = values.shape
            rows = values.transpose(1, 2).reshape(batch * positions, heads * d_head)
            observe(f"layers.{layer[0]}.{name}", rows)
        return q_embed, k_embed

    monkeypatch.setattr(modeling_llama, "apply_rotary_pos_emb", recording_rope)
    handles: list[Any] = []
    for index, block in enumerate(model.model.layers):
        prefix = f"layers.{index}"

        def at(_module: Any, _args: Any, index: int = index) -> None:
            layer[0] = index

        def output(name: str) -> Callable[[Any, Any, torch.Tensor], None]:
            def hook(_module: Any, _args: Any, out: torch.Tensor) -> None:
                observe(name, out)

            return hook

        def input_of(name: str) -> Callable[[Any, tuple[torch.Tensor, ...]], None]:
            def hook(_module: Any, args: tuple[torch.Tensor, ...]) -> None:
                observe(name, args[0])

            return hook

        handles += [
            block.self_attn.register_forward_pre_hook(at),
            block.self_attn.v_proj.register_forward_hook(output(f"{prefix}.v")),
            block.self_attn.o_proj.register_forward_pre_hook(input_of(f"{prefix}.attn_out")),
            block.mlp.up_proj.register_forward_hook(output(f"{prefix}.up")),
            block.mlp.down_proj.register_forward_pre_hook(input_of(f"{prefix}.h")),
        ]
    try:
        with torch.no_grad():
            model(input_ids=tokens)
    finally:
        for handle in handles:
            handle.remove()
    return found


def relative(a: torch.Tensor, b: torch.Tensor) -> float:
    return float((a - b).abs().max() / b.abs().max())


def test_each_transform_keeps_the_logits(monkeypatch: pytest.MonkeyPatch) -> None:
    weights, tokens = float64_weights(), sequences()
    reference = logits_of(weights, tokens)
    maxima = maxima_of(weights, tokens, monkeypatch)
    folded = fold(ARCH, weights)
    assert relative(logits_of(folded, tokens), reference) < 1e-12
    equalized, equalized_maxima = equalize(ARCH, folded, maxima)
    assert relative(logits_of(equalized, tokens), reference) < 1e-12
    for alpha in ALPHAS:
        migrated, _ = migrate(ARCH, equalized, equalized_maxima, alpha)
        assert relative(logits_of(migrated, tokens), reference) < 1e-12, alpha
    # The transforms change the weights, so the comparison is not of a model with itself.
    q = "model.layers.0.self_attn.q_proj.weight"
    assert relative(folded[q], weights[q]) > 0.1
    assert relative(equalized[q], folded[q]) > 0.01


def test_the_predicted_statistics_are_the_transformed_models_own(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    weights, tokens = float64_weights(seed=13), sequences(seed=5)
    maxima = maxima_of(weights, tokens, monkeypatch)
    assert sorted(maxima) == sorted(
        f"layers.{layer}.{point}" for layer in range(ARCH.n_layers) for point in POINTS
    )
    for alpha in ALPHAS:
        transformed, predicted = transform(ARCH, weights, maxima, alpha)
        recorded = maxima_of(transformed, tokens, monkeypatch)
        for name in maxima:
            assert torch.allclose(predicted[name], recorded[name], rtol=1e-10, atol=0), name
        # Migration changed the values, the attention output, up and their product.
        for point in ("v", "attn_out", "up", "h"):
            name = f"layers.0.{point}"
            assert relative(predicted[name], maxima[name]) > 0.01, (alpha, name)


def test_equalization_balances_each_pair_of_each_group(monkeypatch: pytest.MonkeyPatch) -> None:
    weights, tokens = float64_weights(seed=21), sequences(seed=8)
    maxima = maxima_of(weights, tokens, monkeypatch)
    _, balanced = equalize(ARCH, fold(ARCH, weights), maxima)
    half = ARCH.d_head // 2
    for layer in range(ARCH.n_layers):
        q = balanced[f"layers.{layer}.q"].reshape(ARCH.n_kv_heads, GROUP, 2, half)
        k = balanced[f"layers.{layer}.k"].reshape(ARCH.n_kv_heads, 2, half)
        # Over each pair and each KV head's group of query heads: max|q| = max|k|.
        assert torch.allclose(q.amax(dim=(1, 2)), k.amax(dim=1), rtol=1e-12)


def test_migration_at_one_half_balances_inputs_with_weight_columns(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    weights, tokens = float64_weights(seed=34), sequences(seed=9)
    maxima = maxima_of(weights, tokens, monkeypatch)
    migrated, balanced = migrate(ARCH, weights, maxima, Fraction(1, 2))
    for layer in range(ARCH.n_layers):
        prefix = f"model.layers.{layer}"
        # The down projection: max|h_j| = max|W_down[:, j]| for every channel.
        down = migrated[f"{prefix}.mlp.down_proj.weight"].abs().amax(dim=0)
        assert torch.allclose(balanced[f"layers.{layer}.h"], down, rtol=1e-12)
        # The O projection, per value channel: the maxima over its group of query heads.
        o = migrated[f"{prefix}.self_attn.o_proj.weight"].abs().amax(dim=0)
        x = balanced[f"layers.{layer}.attn_out"]
        by_group = [t.reshape(ARCH.n_kv_heads, GROUP, ARCH.d_head).amax(dim=1) for t in (x, o)]
        assert torch.allclose(by_group[0], by_group[1], rtol=1e-12)


def test_layers_transform_alone_as_they_do_together(monkeypatch: pytest.MonkeyPatch) -> None:
    # A quantizer can stream a model a layer at a time, then the LM head with the final norm.
    weights, tokens = float64_weights(seed=55), sequences(count=8)
    maxima = maxima_of(weights, tokens, monkeypatch)
    alpha = Fraction(7, 10)
    whole, whole_maxima = transform(ARCH, weights, maxima, alpha)
    merged: dict[str, torch.Tensor] = {}
    merged_maxima: dict[str, torch.Tensor] = {}
    for layer in range(ARCH.n_layers):
        part = {n: t for n, t in weights.items() if n.startswith(f"model.layers.{layer}.")}
        layer_weights, layer_maxima = transform(ARCH, part, maxima, alpha)
        merged |= layer_weights
        merged_maxima |= {n: v for n, v in layer_maxima.items() if n.startswith(f"layers.{layer}.")}
    head = {
        n: weights[n] for n in ("model.embed_tokens.weight", "model.norm.weight", "lm_head.weight")
    }
    merged |= transform(ARCH, head, maxima, alpha)[0]
    assert sorted(merged) == sorted(whole)
    assert all(torch.equal(merged[n], whole[n]) for n in whole)
    assert sorted(merged_maxima) == sorted(whole_maxima)
    assert all(torch.equal(merged_maxima[n], whole_maxima[n]) for n in whole_maxima)


def test_the_inputs_are_left_untouched_and_zero_channels_scale_by_one() -> None:
    weights = float64_weights(seed=89)
    q_width, kv_width = ARCH.n_heads * ARCH.d_head, ARCH.n_kv_heads * ARCH.d_head
    half = ARCH.d_head // 2
    widths = {"q": q_width, "k": kv_width, "v": kv_width, "attn_out": q_width}
    generator = torch.Generator().manual_seed(4)
    maxima = {
        f"layers.{layer}.{point}": 0.5
        + torch.rand(widths.get(point, ARCH.d_ff), generator=generator, dtype=torch.float64)
        for layer in range(ARCH.n_layers)
        for point in POINTS
    }
    # Layer 0: the queries of pair 0 of KV head 0's whole group, the attention output of value
    # channel 1 over the group, and the product's channel 2 are zero; so is W_down's column 3.
    for head in range(GROUP):
        maxima["layers.0.q"][head * ARCH.d_head] = 0.0
        maxima["layers.0.q"][head * ARCH.d_head + half] = 0.0
        maxima["layers.0.attn_out"][head * ARCH.d_head + 1] = 0.0
    maxima["layers.0.h"][2] = 0.0
    weights["model.layers.0.mlp.down_proj.weight"][:, 3] = 0.0
    before = {n: t.clone() for n, t in weights.items()}
    before_maxima = {n: t.clone() for n, t in maxima.items()}
    transformed, predicted = transform(ARCH, weights, maxima, Fraction(1, 2))
    assert all(torch.equal(weights[n], before[n]) for n in weights)
    assert all(torch.equal(maxima[n], before_maxima[n]) for n in maxima)
    assert all(bool(torch.isfinite(t).all()) for t in [*transformed.values(), *predicted.values()])
    assert all(t.dtype == torch.float64 for t in transformed.values())
    # Each zero leaves its factor at 1: pair 0's keys, value channel 1, and up's rows 2 and 3 keep
    # their folded weights, while their neighbours scale.
    folded = fold(ARCH, weights)
    prefix = "model.layers.0"
    k, v, up = (
        f"{prefix}.self_attn.k_proj.weight",
        f"{prefix}.self_attn.v_proj.weight",
        f"{prefix}.mlp.up_proj.weight",
    )
    for name, kept, scaled in ((k, [0, half], 1), (v, [1], 0), (up, [2, 3], 0)):
        for row in kept:
            assert torch.equal(transformed[name][row], folded[name][row]), (name, row)
        assert not torch.equal(transformed[name][scaled], folded[name][scaled]), name


def test_migration_refuses_a_strength_outside_zero_to_one() -> None:
    weights = float64_weights()
    maxima = {f"layers.0.{p}": torch.ones(1) for p in POINTS}
    for alpha in (Fraction(-1, 10), Fraction(11, 10)):
        with pytest.raises(TesseraQuantError, match=r"alpha lies in \[0, 1\]"):
            migrate(ARCH, weights, maxima, alpha)


def test_alpha_is_the_least_divergence_and_the_smallest_of_ties() -> None:
    seen: list[Fraction] = []

    def divergence(values: dict[Fraction, float]) -> Callable[[Fraction], float]:
        def at(alpha: Fraction) -> float:
            seen.append(alpha)
            return values[alpha]

        return at

    assert select_alpha(divergence(dict(zip(ALPHAS, (0.3, 0.2, 0.25, 0.9), strict=True)))) == (
        Fraction(3, 5)
    )
    assert seen == list(ALPHAS)
    ties = dict(zip(ALPHAS, (0.4, 0.1, 0.1, 0.1), strict=True))
    assert select_alpha(divergence(ties)) == Fraction(3, 5)
    with pytest.raises(TesseraQuantError, match="not finite"):
        select_alpha(divergence(dict(zip(ALPHAS, (0.1, math.nan, 0.1, 0.1), strict=True))))
    with pytest.raises(TesseraQuantError, match="at least one candidate"):
        select_alpha(lambda _: 0.0, ())


def test_mean_kl_is_the_divergence_of_the_reference_from_the_logits() -> None:
    # p = (1/2, 1/2) and q = (3/4, 1/4): KL(p ‖ q) = ln(4/3) / 2.
    p = torch.tensor([[0.0, 0.0]], dtype=torch.float64)
    q = torch.tensor([[math.log(3.0), 0.0]], dtype=torch.float64)
    assert mean_kl(p, q) == pytest.approx(math.log(4 / 3) / 2, rel=1e-12)
    logits = torch.randn(3, 5, 7, generator=torch.Generator().manual_seed(1))
    assert mean_kl(logits, logits) == pytest.approx(0.0, abs=1e-15)
    assert mean_kl(logits, logits.flip(-1)) > 0
    with pytest.raises(TesseraQuantError, match="same shape"):
        mean_kl(logits, logits[:, :, :6])


def test_migration_moves_the_strength_alpha_of_each_channel_to_the_weights(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # s = x^alpha / w^(1-alpha) leaves x' = x^(1-alpha) w^(1-alpha) and w' = x^alpha w^alpha: at
    # alpha = 4/5 the input keeps a fifth of the product's logarithm, the weight four fifths.
    weights, tokens = float64_weights(seed=144), sequences(seed=12)
    maxima = maxima_of(weights, tokens, monkeypatch)
    alpha = Fraction(4, 5)
    migrated, moved = migrate(ARCH, weights, maxima, alpha)
    for layer in range(ARCH.n_layers):
        x = maxima[f"layers.{layer}.h"]
        w = weights[f"model.layers.{layer}.mlp.down_proj.weight"].abs().amax(dim=0)
        new_w = migrated[f"model.layers.{layer}.mlp.down_proj.weight"].abs().amax(dim=0)
        a = float(alpha)
        assert torch.allclose(moved[f"layers.{layer}.h"], (x * w).pow(1 - a), rtol=1e-12)
        assert torch.allclose(new_w, (x * w).pow(a), rtol=1e-12)
