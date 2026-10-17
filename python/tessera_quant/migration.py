# SPDX-License-Identifier: Apache-2.0 OR MIT
"""Gain folding, QK equalization and SmoothQuant migration (RFC-0005 'Pipeline', steps 2 and 4).

Each transform changes the weights without changing the model's outputs in real arithmetic, and
changes the statistics with them, so that the quantizer reads the activations the transformed
model computes. Each acts within one layer, or on the LM head, so a model can stream a layer at a
time:

- **Fold.** Each RMSNorm gain multiplies the columns of the weights that read its output: Q, K and
  V for the attention norm, gate and up for the MLP norm, the LM head for the final norm. The
  norms are left with a gain of 1, as `D`'s norm has none.
- **Equalize.** For each KV head and RoPE pair `(i, i + d_head/2)`, the rows of `W_q` of every
  query head of the group scale by `λ` and those of `W_k` by `1/λ`, with `λ = √(max|k| / max|q|)`
  over the pair and the group. The scores do not change: RoPE turns each pair by a rotation,
  which commutes with a common scale. The queries' and keys' maxima become equal.
- **Migrate.** SmoothQuant (Xiao et al. 2023) divides a linear layer's input channel `j` by
  `s_j = max|x_j|^alpha / max|w_j|^{1-alpha}`, `w_j` the weight's column `j`, and multiplies
  the column by `s_j`. The division needs a per-channel factor in the input's producer:
  - the O projection's input, the attention output, is a weighted sum of values, so the rows of
    `W_v` take it, one `s` per value channel: the maxima over its group of query heads;
  - the down projection's input, `SiLU(gate) · up`, scales with `up`, so the rows of `W_up` take
    it.

  The inputs of Q, K, V, gate, up and the LM head are norm outputs, and `D`'s norm applies no
  per-channel factor, so they are not migrated (#289).
- **alpha** is chosen from `{1/2, 3/5, 7/10, 4/5}` by the least mean KL divergence to the BF16
  reference on the held-out split (RFC-0005 'Calibration'). `select_alpha` takes that
  divergence as a function of `alpha`, since computing it needs the quantized model.

Everything runs in float64. Weights and statistics are secret-class (05 §3): nothing here logs or
reports a value.
"""

import math
from collections.abc import Callable, Mapping
from fractions import Fraction

import torch

from tessera_quant.arch import ArchSpec
from tessera_quant.errors import TesseraQuantError

ALPHAS: tuple[Fraction, ...] = (Fraction(1, 2), Fraction(3, 5), Fraction(7, 10), Fraction(4, 5))
"""The migration strengths RFC-0005 'Calibration' chooses from."""

Tensors = dict[str, torch.Tensor]


def fold(arch: ArchSpec, weights: Mapping[str, torch.Tensor]) -> Tensors:
    """`weights` with every RMSNorm gain folded into the columns of the weights reading its output.

    Each layer whose tensors `weights` holds is folded, and so is the LM head with the final norm
    when they are present; the norms' gains become 1.
    """
    folded = _float64(weights)
    for layer in _layers(arch, folded):
        prefix = f"model.layers.{layer}"
        for norm, consumers in (
            ("input_layernorm", ("self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj")),
            ("post_attention_layernorm", ("mlp.gate_proj", "mlp.up_proj")),
        ):
            gain = folded[f"{prefix}.{norm}.weight"]
            for consumer in consumers:
                name = f"{prefix}.{consumer}.weight"
                folded[name] = folded[name] * gain
            folded[f"{prefix}.{norm}.weight"] = torch.ones_like(gain)
    if "lm_head.weight" in folded:
        gain = folded["model.norm.weight"]
        folded["lm_head.weight"] = folded["lm_head.weight"] * gain
        folded["model.norm.weight"] = torch.ones_like(gain)
    return folded


def equalize(
    arch: ArchSpec, weights: Mapping[str, torch.Tensor], maxima: Mapping[str, torch.Tensor]
) -> tuple[Tensors, Tensors]:
    """`weights` and `maxima` after QK equalization of every layer `weights` holds.

    The queries' and keys' maxima are those after RoPE, channel `head · d_head + i`, as the
    capture records them.
    """
    weights, maxima = _float64(weights), _float64(maxima)
    for layer in _layers(arch, weights):
        q, k = f"layers.{layer}.q", f"layers.{layer}.k"
        q_max = _pairs(arch, maxima[q], arch.n_heads).amax(dim=1)
        k_max = _pairs(arch, maxima[k], arch.n_kv_heads)[:, 0]
        # [n_kv_heads, d_head / 2]: one λ per KV head and pair; 1 where either side is all zero.
        usable = (q_max > 0) & (k_max > 0)
        ratio = torch.where(usable, k_max / torch.where(usable, q_max, 1.0), 1.0)
        lam = ratio.sqrt()
        group = arch.n_heads // arch.n_kv_heads
        q_scale = _unpair(lam.repeat_interleave(group, dim=0))
        k_scale = _unpair(lam)
        prefix = f"model.layers.{layer}.self_attn"
        weights[f"{prefix}.q_proj.weight"] = weights[f"{prefix}.q_proj.weight"] * q_scale[:, None]
        weights[f"{prefix}.k_proj.weight"] = weights[f"{prefix}.k_proj.weight"] / k_scale[:, None]
        maxima[q] = maxima[q] * q_scale
        maxima[k] = maxima[k] / k_scale
    return weights, maxima


def migrate(
    arch: ArchSpec,
    weights: Mapping[str, torch.Tensor],
    maxima: Mapping[str, torch.Tensor],
    alpha: Fraction,
) -> tuple[Tensors, Tensors]:
    """`weights` and `maxima` after SmoothQuant migration of strength `alpha`.

    In every layer `weights` holds, the O projection's input migrates through `W_v`, and the down
    projection's through `W_up`.

    Raises:
        TesseraQuantError: for a strength outside `[0, 1]`.
    """
    if not 0 <= alpha <= 1:
        raise TesseraQuantError("the migration strength alpha lies in [0, 1]")
    weights, maxima = _float64(weights), _float64(maxima)
    group = arch.n_heads // arch.n_kv_heads
    for layer in _layers(arch, weights):
        prefix, point = f"model.layers.{layer}", f"layers.{layer}"
        # The attention output, channel `head · d_head + i`, per value channel `kv · d_head + i`.
        o_name, v_name = f"{prefix}.self_attn.o_proj.weight", f"{prefix}.self_attn.v_proj.weight"
        x = _groups(arch, maxima[f"{point}.attn_out"], group)
        w = _groups(arch, weights[o_name].abs().amax(dim=0), group)
        s = _smoothing(x, w, alpha)
        weights[v_name] = weights[v_name] / s[:, None]
        per_head = _per_head(arch, s, group)
        weights[o_name] = weights[o_name] * per_head
        maxima[f"{point}.v"] = maxima[f"{point}.v"] / s
        maxima[f"{point}.attn_out"] = maxima[f"{point}.attn_out"] / per_head
        # The down projection's input `SiLU(gate) · up`, through the rows of `W_up`.
        up_name, down_name = f"{prefix}.mlp.up_proj.weight", f"{prefix}.mlp.down_proj.weight"
        s = _smoothing(maxima[f"{point}.h"], weights[down_name].abs().amax(dim=0), alpha)
        weights[up_name] = weights[up_name] / s[:, None]
        weights[down_name] = weights[down_name] * s
        maxima[f"{point}.up"] = maxima[f"{point}.up"] / s
        maxima[f"{point}.h"] = maxima[f"{point}.h"] / s
    return weights, maxima


def transform(
    arch: ArchSpec,
    weights: Mapping[str, torch.Tensor],
    maxima: Mapping[str, torch.Tensor],
    alpha: Fraction,
) -> tuple[Tensors, Tensors]:
    """Folding, equalization and migration of strength `alpha`, in that order."""
    folded = fold(arch, weights)
    equalized, maxima = equalize(arch, folded, maxima)
    return migrate(arch, equalized, maxima, alpha)


def select_alpha(
    divergence: Callable[[Fraction], float], candidates: tuple[Fraction, ...] = ALPHAS
) -> Fraction:
    """The candidate of least divergence, the smallest of equal ones (RFC-0005 'Calibration').

    `divergence(alpha)` is the mean KL divergence to the BF16 reference, on the held-out split,
    of the model migrated with `alpha` and quantized.

    Raises:
        TesseraQuantError: when no candidate is given, or a divergence is not finite.
    """
    if not candidates:
        raise TesseraQuantError("alpha is chosen among at least one candidate")
    scored: list[tuple[float, Fraction]] = []
    for alpha in sorted(candidates):
        value = divergence(alpha)
        if not math.isfinite(value):
            raise TesseraQuantError(f"the divergence at alpha = {alpha} is not finite")
        scored.append((value, alpha))
    return min(scored)[1]


def mean_kl(reference: torch.Tensor, logits: torch.Tensor) -> float:
    """The mean, over every position, of `KL(softmax(reference) ‖ softmax(logits))`, in nats.

    Both are logits `[..., vocab]` of the same positions, the reference's first.
    """
    if reference.shape != logits.shape:
        raise TesseraQuantError("the divergence compares logits of the same shape")
    log_p = torch.log_softmax(reference.to(torch.float64), dim=-1)
    log_q = torch.log_softmax(logits.to(torch.float64), dim=-1)
    per_position = (log_p.exp() * (log_p - log_q)).sum(dim=-1)
    return float(per_position.mean())


def _float64(tensors: Mapping[str, torch.Tensor]) -> Tensors:
    return {name: tensor.to(torch.float64) for name, tensor in tensors.items()}


def _layers(arch: ArchSpec, weights: Mapping[str, torch.Tensor]) -> list[int]:
    """The layers whose Q projection `weights` holds."""
    return [
        layer
        for layer in range(arch.n_layers)
        if f"model.layers.{layer}.self_attn.q_proj.weight" in weights
    ]


def _pairs(arch: ArchSpec, values: torch.Tensor, heads: int) -> torch.Tensor:
    """`[heads · d_head]` as `[heads / group, group, d_head / 2]` maxima over each RoPE pair."""
    half = arch.d_head // 2
    by_head = values.reshape(heads, 2, half).amax(dim=1)
    group = heads // arch.n_kv_heads
    return by_head.reshape(arch.n_kv_heads, group, half)


def _unpair(per_pair: torch.Tensor) -> torch.Tensor:
    """`[heads, d_head / 2]` to `[heads · d_head]`, each pair's two channels equal."""
    return per_pair.repeat(1, 2).reshape(-1)


def _groups(arch: ArchSpec, values: torch.Tensor, group: int) -> torch.Tensor:
    """`[n_heads · d_head]` to `[n_kv_heads · d_head]`: the maximum over each KV head's group."""
    by_head = values.reshape(arch.n_kv_heads, group, arch.d_head)
    return by_head.amax(dim=1).reshape(-1)


def _per_head(arch: ArchSpec, per_value: torch.Tensor, group: int) -> torch.Tensor:
    """`[n_kv_heads · d_head]` to `[n_heads · d_head]`: each query head takes its KV head's."""
    by_kv = per_value.reshape(arch.n_kv_heads, 1, arch.d_head).expand(-1, group, -1)
    return by_kv.reshape(-1)


def _smoothing(x: torch.Tensor, w: torch.Tensor, alpha: Fraction) -> torch.Tensor:
    """`s = x^alpha / w^{1-alpha}` per channel, 1 where either maximum is zero."""
    usable = (x > 0) & (w > 0)
    a = float(alpha)
    s = x.clamp(min=1e-300).pow(a) / w.clamp(min=1e-300).pow(1 - a)
    return torch.where(usable, s, 1.0)
