# SPDX-License-Identifier: Apache-2.0 OR MIT
"""W8A8 quantization (RFC-0005 'Scale policy', 'Integer semantics').

A tiny Llama-3.1 goes through the pipeline: its BF16 statistics are captured, the migration
transforms it, and `choose` gives `D`'s weights and constants, in both profiles.

- **Weights.** No weight clips, each channel's scale is the least that holds it, a tensor has at
  most four exponents, RoPE pairs share theirs, and the embedding, the LM head and `minimal`'s up
  projection have one.
- **Shifts.** Every site's classes are the shifts RFC-0005's semantics give from the exponents,
  each in `[1, 63]`; the norms use RFC-0023's kappa, and the score scale is `2^(e_q + e_k)`
  times the multiple of `2^-40` nearest `1 / sqrt(d_head)`. SiLU's input exponent is the finest
  its table allows. The residual holds every token's embedding, seen in calibration or not.
- **Fidelity.** The model run in floats with every weight and activation rounded to its scale,
  as `D` holds them, gives logits close to the float model's.
- **affine** switches the named weights only, states each switched site and its shift in
  `affine_sites` (RFC-0025), keeps every multiplier within 16 bits where the largest rounds up,
  and refuses what cannot take it, a weight named in some layers and not others among them.
- **Every layer alike** (RFC-0038). A site whose channels share one shift in a layer, while
  another layer's need several, takes one class more there that no channel uses: above the
  shift, or below 63. `D` computes the logits it computes without that class, and a residual site
  of one shift in one layer and several in another takes two steps in both.
"""

import math
from collections.abc import Mapping
from dataclasses import replace
from fractions import Fraction
from pathlib import Path
from typing import Any, cast

import numpy as np
import pytest
import torch
from checkpoints import TINY, TINY_ARCH, tensors, write
from hypothesis import given
from hypothesis import strategies as st
from transformers import LlamaConfig, LlamaForCausalLM
from transformers.models.llama import modeling_llama

from tessera_quant import CalibrationError, TesseraQuantError, import_checkpoint, tensor_shapes
from tessera_quant.calibration import DatasetIdentity, draw
from tessera_quant.capture import capture, points
from tessera_quant.integer_model import (
    CalibrationIdentity,
    EmbedPoint,
    FinalNormPoint,
    IntegerModel,
    LayerPoint,
    Profile,
    Provenance,
    QuantPoint,
    Site,
    WeightType,
    norm_kappa,
)
from tessera_quant.migration import transform
from tessera_quant.reference import run_reference, trace
from tessera_quant.scales import Quantized, choose, exponent

ARCH = TINY_ARCH
SOURCE = DatasetIdentity(
    name="fixture", config="en", split="validation", revision="0" * 40, files=("fixture",)
)
ROPE: Any = cast(Any, modeling_llama).apply_rotary_pos_emb


def llama_like(seed: int) -> dict[str, torch.Tensor]:
    """Random weights at Llama's magnitudes: an embedding of unit scale, matrices near 0.02 and
    norm gains near 1. The residual stream is then coarser than the updates' accumulators, as in
    Llama, so that every residual site's shift is positive."""
    generator = torch.Generator().manual_seed(seed + 1)
    weights = tensors(ARCH, seed)
    for name, t in weights.items():
        if t.dim() == 1:
            gain = 1 + 0.1 * torch.randn(t.shape, generator=generator, dtype=torch.float64)
            weights[name] = gain.to(torch.bfloat16)
        elif name != "model.embed_tokens.weight":
            weights[name] = (t.to(torch.float64) * 0.02).to(torch.bfloat16)
    return weights


def migrated(
    tmp_path: Path, seed: int = 7
) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor]]:
    """A tiny checkpoint's weights and statistics, migrated at alpha = 1/2."""
    path = write(tmp_path / f"checkpoint-{seed}", TINY, llama_like(seed))
    model = import_checkpoint(path)
    documents = [f"document {i} " + "the quick brown fox jumps" * (1 + i % 3) for i in range(40)]
    calibration = draw(
        documents,
        lambda text: [2 + ord(c) % 60 for c in text],
        1,
        source=SOURCE,
        seed=11,
        count=6,
        length=16,
        held_out=0,
    )
    maxima = dict(capture(model, calibration).maxima)
    weights = {name: model.tensor(name).to(torch.float64) for name in model.names}
    return transform(ARCH, weights, maxima, Fraction(1, 2))


def act(quantized: Quantized) -> dict[QuantPoint, int]:
    return dict(quantized.quant.public.act_exponents)


def sets(quantized: Quantized) -> dict[QuantPoint, list[int]]:
    return dict(quantized.quant.public.shift_class_sets)


def affine_shifts(quantized: Quantized) -> dict[QuantPoint, int]:
    return dict(quantized.quant.public.affine_sites)


def weight(quantized: Quantized, ty: WeightType, layer: int = 0) -> torch.Tensor:
    found = next(w for w in quantized.weights if w.ty is ty and w.layer == layer)
    data: Any = found.data.astype("int64")
    return torch.tensor(data).reshape(found.rows, found.cols)


def row_exponents(quantized: Quantized, name: str) -> list[int]:
    return [round(math.log2(float(s))) for s in quantized.scales[name]]


@pytest.fixture(scope="module")
def pipeline(tmp_path_factory: pytest.TempPathFactory) -> dict[Profile, Any]:
    weights, maxima = migrated(tmp_path_factory.mktemp("quantize"))
    return {
        profile: (weights, maxima, choose(ARCH, profile, weights, maxima)) for profile in Profile
    }


@given(st.floats(min_value=1e-30, max_value=1e30, allow_nan=False), st.sampled_from([127, 1 << 18]))
def test_an_exponent_is_the_least_that_holds_its_maximum(maximum: float, limit: int) -> None:
    e = exponent(maximum, limit)
    assert Fraction(maximum) <= limit * Fraction(2) ** e
    assert Fraction(maximum) > limit * Fraction(2) ** (e - 1)


def test_an_exponent_needs_a_finite_positive_maximum() -> None:
    assert exponent(127.0, 127) == 0
    assert exponent(63.5, 127) == -1
    for bad in (0.0, -1.0, math.inf, math.nan):
        with pytest.raises(CalibrationError, match="finite, positive maximum"):
            exponent(bad, 127)


def test_no_weight_clips_and_each_scale_is_the_least_that_holds_its_row(
    pipeline: dict[Profile, Any],
) -> None:
    weights, _, quantized = pipeline[Profile.Servable]
    for w in quantized.weights:
        assert w.data.dtype.name == "int8"
        assert int(abs(w.data.astype("int64")).max()) <= 127
    # A row at the top of its tensor's window has its least scale: its largest entry rounds to at
    # least 64. Every row dequantizes to within half a step of the float weight.
    for name in ("model.layers.0.mlp.down_proj.weight", "model.layers.1.self_attn.v_proj.weight"):
        ty = WeightType.Down if "down" in name else WeightType.V
        layer = int(name.split(".")[2])
        q = weight(quantized, ty, layer).to(torch.float64)
        scale = quantized.scales[name][:, None]
        assert bool(((q * scale - weights[name]).abs() <= scale / 2).all())
        exponents = row_exponents(quantized, name)
        top = [k for k, e in enumerate(exponents) if e == max(exponents)]
        assert all(int(q[k].abs().max()) >= 64 for k in top)


def test_a_weight_takes_at_most_four_exponents_and_rope_pairs_share_theirs(
    pipeline: dict[Profile, Any],
) -> None:
    _, _, quantized = pipeline[Profile.Servable]
    for name, scale in quantized.scales.items():
        assert len({float(x) for x in scale}) <= 4, name
    half = ARCH.d_head // 2
    for name in (
        "model.layers.0.self_attn.q_proj.weight",
        "model.layers.1.self_attn.k_proj.weight",
    ):
        by_head = torch.tensor(row_exponents(quantized, name)).reshape(-1, 2, half)
        assert torch.equal(by_head[:, 0], by_head[:, 1]), name
    # A weight whose rows span 2^0 to 2^9 is raised into a window of four.
    weights, maxima, _ = pipeline[Profile.Servable]
    wide = dict(weights)
    gate = "model.layers.0.mlp.gate_proj.weight"
    rows = wide[gate].clone()
    for k in range(rows.shape[0]):
        rows[k] = rows[k] / rows[k].abs().max() * 2.0 ** (k % 10)
    wide[gate] = rows
    spread = choose(ARCH, Profile.Servable, wide, maxima)
    assert len({float(x) for x in spread.scales[gate]}) == 4
    assert len(sets(spread)[LayerPoint(0, Site.Silu)]) == 4


def test_the_embedding_lm_head_and_minimals_up_projection_have_one_exponent(
    pipeline: dict[Profile, Any],
) -> None:
    for profile, names in (
        (Profile.Servable, ("model.embed_tokens.weight", "lm_head.weight")),
        (
            Profile.Minimal,
            ("model.layers.0.mlp.up_proj.weight", "model.layers.1.mlp.up_proj.weight"),
        ),
    ):
        _, _, quantized = pipeline[profile]
        for name in names:
            assert len({float(x) for x in quantized.scales[name]}) == 1, name
        # No hidden classes for weights without a site of several.
        assert all(
            c.ty not in (WeightType.Embed, WeightType.LmHead) for c in quantized.quant.channels
        )
    _, _, minimal = pipeline[Profile.Minimal]
    assert all(c.ty is not WeightType.Up for c in minimal.quant.channels)
    # The logits stand for l * 2^(e_y + e_lm), e_lm the least exponent that holds the folded LM
    # head within 127 (RFC-0027).
    for profile in Profile:
        weights, _, quantized = pipeline[profile]
        e_lm = exponent(float(weights["lm_head.weight"].abs().max()), 127)
        e_y = act(quantized)[FinalNormPoint()]
        assert quantized.quant.public.logits_exponent == e_y + e_lm, profile
        assert float(quantized.scales["lm_head.weight"][0]) == 2.0**e_lm


def expected_sets(quantized: Quantized, profile: Profile) -> dict[QuantPoint, list[int]]:
    """Every site's classes from the exponents, by RFC-0005 'Integer semantics' written out."""
    a = act(quantized)
    public = quantized.quant.public
    kappa, e_res, sigma2, bits = (
        public.rsqrt_kappa,
        public.residual_exponent,
        2 * public.exp_sigma,
        ARCH.rope.bits,
    )
    out: dict[QuantPoint, list[int]] = {}

    def classes(offset: int, name: str) -> list[int]:
        return sorted({offset - e for e in row_exponents(quantized, name)})

    for layer in range(ARCH.n_layers):
        p = f"model.layers.{layer}"

        def at(site: Site, layer: int = layer) -> int:
            return a[LayerPoint(layer, site)]

        y, y2 = at(Site.AttnNorm), at(Site.MlpNorm)
        out[LayerPoint(layer, Site.AttnNorm)] = [kappa + y]
        out[LayerPoint(layer, Site.RopeQ)] = classes(
            at(Site.RopeQ) - y + bits, f"{p}.self_attn.q_proj.weight"
        )
        out[LayerPoint(layer, Site.RopeK)] = classes(
            at(Site.RopeK) - y + bits, f"{p}.self_attn.k_proj.weight"
        )
        out[LayerPoint(layer, Site.V)] = classes(at(Site.V) - y, f"{p}.self_attn.v_proj.weight")
        if profile is Profile.Servable:
            out[LayerPoint(layer, Site.AttnOut)] = [at(Site.AttnOut) - at(Site.V) + sigma2]
            o_in = at(Site.AttnOut)
        else:
            o_in = at(Site.V) - sigma2
        out[LayerPoint(layer, Site.AttnResidual)] = classes(
            e_res - o_in, f"{p}.self_attn.o_proj.weight"
        )
        out[LayerPoint(layer, Site.MlpNorm)] = [kappa + y2]
        in_exp = dict(quantized.quant.silu_in_exponents)[layer]
        out[LayerPoint(layer, Site.Silu)] = classes(in_exp - y2, f"{p}.mlp.gate_proj.weight")
        if profile is Profile.Servable:
            out[LayerPoint(layer, Site.Up)] = classes(at(Site.Up) - y2, f"{p}.mlp.up_proj.weight")
            out[LayerPoint(layer, Site.Product)] = [at(Site.Product) - at(Site.Silu) - at(Site.Up)]
            d_in = at(Site.Product)
        else:
            (e_up,) = set(row_exponents(quantized, f"{p}.mlp.up_proj.weight"))
            d_in = at(Site.Silu) + y2 + e_up
        out[LayerPoint(layer, Site.MlpResidual)] = classes(
            e_res - d_in, f"{p}.mlp.down_proj.weight"
        )
    out[FinalNormPoint()] = [kappa + a[FinalNormPoint()]]
    return out


def test_every_sites_classes_are_the_semantics_shifts(pipeline: dict[Profile, Any]) -> None:
    for profile in Profile:
        weights, maxima, quantized = pipeline[profile]
        public = quantized.quant.public
        assert sets(quantized) == expected_sets(quantized, profile)
        assert all(1 <= r <= 63 for shifts in sets(quantized).values() for r in shifts)
        assert [p for p, _ in public.shift_class_sets] == sorted(
            sets(quantized),
            key=lambda p: (
                (0, 0, 0)
                if isinstance(p, EmbedPoint)
                else (1, p.layer, p.site.value)
                if isinstance(p, LayerPoint)
                else (2, 0, 0)
            ),
        )
        # The residual keeps a bit of headroom over its stream and the embedding table, its scale
        # sets e, and kappa follows RFC-0023.
        residual = max(
            [
                float(maxima[n].max())
                for n in maxima
                if n.endswith(("attn_input", "mlp_input")) or n == "final.input"
            ]
            + [float(weights["model.embed_tokens.weight"].abs().max())]
        )
        assert public.residual_exponent == exponent(residual, 1 << 18)
        e = math.ceil(ARCH.d_model * ARCH.rms_eps * Fraction(4) ** -public.residual_exponent)
        assert public.rms_eps_int == e
        assert public.rsqrt_kappa == norm_kappa(ARCH.d_model, e)
        # The embedding enters the residual by a shift of at least 0.
        assert act(quantized)[EmbedPoint()] >= public.residual_exponent
        # Each hidden class indexes its site's shifts.
        for params in quantized.quant.channels:
            assert params.multiplier.size == 0
            assert int(params.class_.max()) < 4


def test_score_scales_and_silu_exponents_are_as_fine_as_allowed(
    pipeline: dict[Profile, Any],
) -> None:
    _, _, quantized = pipeline[Profile.Servable]
    # Each score scale is 2^e * n / 2^40, n the integer nearest 2^40 / sqrt(d_head), that is
    # |2n - 2^41 / sqrt(d)| < 1, at the tiny head size and at Llama's 128.
    arch, weights, maxima = synthetic(torch.full((4, 1024), 0.5, dtype=torch.float64))
    for model, d in (
        (quantized, ARCH.d_head),
        (choose(arch, Profile.Servable, weights, maxima), 128),
    ):
        exponents = act(model)
        for layer, scale in model.quant.public.score_scales:
            e = exponents[LayerPoint(layer, Site.RopeQ)] + exponents[LayerPoint(layer, Site.RopeK)]
            n = scale * Fraction(2) ** (40 - e)
            assert n.denominator == 1
            assert (2 * n.numerator - 1) ** 2 * d < 4**41 < (2 * n.numerator + 1) ** 2 * d
    a = act(quantized)
    b = (127 * 128 * ARCH.d_model).bit_length() + 1
    for layer, in_exp in quantized.quant.silu_in_exponents:
        y2 = a[LayerPoint(layer, Site.MlpNorm)]
        gate = set(row_exponents(quantized, f"model.layers.{layer}.mlp.gate_proj.weight"))

        def core(in_exp: int, y2: int = y2, gate: set[int] = gate) -> int:
            return sum(1 << max(b - (in_exp - y2 - e), 0) for e in gate)

        assert core(in_exp) <= 1 << 24
        assert min(in_exp - y2 - e for e in gate) >= 1
        # One finer would break a shift's floor of 1 or the core's 2^24 values.
        finer = in_exp - 1
        assert min(finer - y2 - e for e in gate) < 1 or core(finer) > 1 << 24


def fake_quantized_logits(
    quantized: Quantized,
    weights: Mapping[str, torch.Tensor],
    tokens: torch.Tensor,
    profile: Profile,
    monkeypatch: pytest.MonkeyPatch,
) -> torch.Tensor:
    """The migrated model in float64 with its weights dequantized and each INT8 activation
    rounded and saturated at its scale, as `D` holds it."""
    a = act(quantized)
    state = dict(weights)
    for name, scale in quantized.scales.items():
        q = weight_by_name(quantized, name).to(torch.float64)
        state[name] = (
            (q * scale[:, None]).T if name == "model.embed_tokens.weight" else q * scale[:, None]
        )
    config: Any = cast(Any, LlamaConfig).from_dict(TINY)
    config._attn_implementation = "eager"
    model: Any = cast(Any, LlamaForCausalLM(config)).to(torch.float64)
    model.load_state_dict(state, strict=True)
    model.eval()

    def rounded(x: torch.Tensor, e: int) -> torch.Tensor:
        return (x / 2.0**e).round().clamp(-128, 127) * 2.0**e

    layer = [0]

    def rope(q: torch.Tensor, k: torch.Tensor, *args: Any, **kwargs: Any) -> Any:
        q, k = ROPE(q, k, *args, **kwargs)
        point = LayerPoint(layer[0], Site.RopeQ), LayerPoint(layer[0], Site.RopeK)
        return rounded(q, a[point[0]]), rounded(k, a[point[1]])

    monkeypatch.setattr(modeling_llama, "apply_rotary_pos_emb", rope)
    handles: list[Any] = []

    def pre(e: int) -> Any:
        def hook(_module: Any, args: tuple[torch.Tensor, ...]) -> tuple[torch.Tensor]:
            return (rounded(args[0], e),)

        return hook

    def post(e: int) -> Any:
        def hook(_module: Any, _args: Any, out: torch.Tensor) -> torch.Tensor:
            return rounded(out, e)

        return hook

    for index, block in enumerate(model.model.layers):

        def at(_m: Any, _a: Any, index: int = index) -> None:
            layer[0] = index

        def point(site: Site, index: int = index) -> int:
            return a[LayerPoint(index, site)]

        handles.append(block.self_attn.register_forward_pre_hook(at))
        attention = (block.self_attn.q_proj, block.self_attn.k_proj, block.self_attn.v_proj)
        handles.extend(p.register_forward_pre_hook(pre(point(Site.AttnNorm))) for p in attention)
        handles.append(block.self_attn.v_proj.register_forward_hook(post(point(Site.V))))
        if profile is Profile.Servable:
            handles.append(
                block.self_attn.o_proj.register_forward_pre_hook(pre(point(Site.AttnOut)))
            )
            handles.append(block.mlp.up_proj.register_forward_hook(post(point(Site.Up))))
            handles.append(block.mlp.down_proj.register_forward_pre_hook(pre(point(Site.Product))))
        mlp = (block.mlp.gate_proj, block.mlp.up_proj)
        handles.extend(p.register_forward_pre_hook(pre(point(Site.MlpNorm))) for p in mlp)
        handles.append(block.mlp.act_fn.register_forward_hook(post(point(Site.Silu))))
    handles.append(model.lm_head.register_forward_pre_hook(pre(a[FinalNormPoint()])))
    try:
        with torch.no_grad():
            return model(input_ids=tokens).logits
    finally:
        for handle in handles:
            handle.remove()


def weight_by_name(quantized: Quantized, name: str) -> torch.Tensor:
    parts = name.split(".")
    if name == "model.embed_tokens.weight":
        return weight(quantized, WeightType.Embed)
    if name == "lm_head.weight":
        return weight(quantized, WeightType.LmHead)
    kinds = {
        "q_proj": WeightType.Q,
        "k_proj": WeightType.K,
        "v_proj": WeightType.V,
        "o_proj": WeightType.O,
        "gate_proj": WeightType.Gate,
        "up_proj": WeightType.Up,
        "down_proj": WeightType.Down,
    }
    return weight(quantized, kinds[parts[4]], int(parts[2]))


def test_the_quantized_model_keeps_the_float_models_predictions(
    pipeline: dict[Profile, Any], monkeypatch: pytest.MonkeyPatch
) -> None:
    tokens = torch.randint(2, ARCH.vocab, (32, 16), generator=torch.Generator().manual_seed(4))
    for profile in Profile:
        weights, _, quantized = pipeline[profile]
        config: Any = cast(Any, LlamaConfig).from_dict(TINY)
        config._attn_implementation = "eager"
        reference: Any = cast(Any, LlamaForCausalLM(config)).to(torch.float64)
        reference.load_state_dict(dict(weights), strict=True)
        with torch.no_grad():
            expected = reference.eval()(input_ids=tokens).logits
        found = fake_quantized_logits(quantized, weights, tokens, profile, monkeypatch)
        log_p = torch.log_softmax(expected, dim=-1)
        log_q = torch.log_softmax(found, dim=-1)
        kl = float((log_p.exp() * (log_p - log_q)).sum(dim=-1).mean())
        agree = float((expected.argmax(-1) == found.argmax(-1)).double().mean())
        # The tiny model's logits are nearly flat, so their error relative to the largest is the
        # sensitive measure: about 2% here, where a misplaced scale gives 50% or more.
        error = float((found - expected).abs().max() / expected.abs().max())
        assert error < 0.05, (profile, error)
        assert kl < 1e-4, (profile, kl)
        assert agree > 0.97, (profile, agree)


def test_affine_switches_the_named_weights_only(pipeline: dict[Profile, Any]) -> None:
    weights, maxima, _ = pipeline[Profile.Servable]
    downs = [f"model.layers.{layer}.mlp.down_proj.weight" for layer in range(ARCH.n_layers)]
    down = downs[0]
    quantized = choose(ARCH, Profile.Servable, weights, maxima, affine=downs)
    affine = [c for c in quantized.quant.channels if c.multiplier.size]
    assert [(c.ty, c.layer) for c in affine] == [(WeightType.Down, 0), (WeightType.Down, 1)]
    params = affine[0]
    assert params.multiplier.size == ARCH.d_model
    assert int(params.multiplier.min()) >= 1
    assert params.class_.size == 0
    # Each row's largest weight is exactly 127 at its scale max|w_c| / 127.
    q = weight(quantized, WeightType.Down, 0)
    assert bool((q.abs().amax(dim=1) == 127).all())
    # The sites and their one shift are public, in affine_sites rather than shift_class_sets.
    site = LayerPoint(0, Site.MlpResidual)
    assert list(affine_shifts(quantized)) == [site, LayerPoint(1, Site.MlpResidual)]
    assert site not in sets(quantized)
    r = affine_shifts(quantized)[site]
    assert 1 <= r <= 63
    # The site's factor M_c / 2^r is sigma_c / 2^offset, with offset = e_res - e_h.
    public = quantized.quant.public
    offset = public.residual_exponent - act(quantized)[LayerPoint(0, Site.Product)]
    sigmas = [float(x) for x in quantized.scales[down]]
    for m, sigma in zip(params.multiplier.tolist(), sigmas, strict=True):
        assert abs(int(m) / 2**r - sigma / 2.0**offset) <= 2.0 ** -(r + 1) * 1.0000001
    # Everything else is as without affine.
    plain = choose(ARCH, Profile.Servable, weights, maxima)
    for found, before in zip(quantized.weights, plain.weights, strict=True):
        if found.ty is not WeightType.Down:
            assert (found.data == before.data).all(), (found.ty, found.layer)


def test_affine_multipliers_fit_16_bits_and_ignore_zero_rows(
    pipeline: dict[Profile, Any],
) -> None:
    weights, maxima, plain = pipeline[Profile.Servable]
    down = "model.layers.0.mlp.down_proj.weight"
    offset = plain.quant.public.residual_exponent - act(plain)[LayerPoint(0, Site.Product)]
    # Row 0, now the largest, gets the factor sigma_0 / 2^offset = 65535.75 * 2^(j - offset). At
    # the shift offset - j its multiplier would round to 2^16, so the shift is one less, and the
    # multiplier 32768.
    rows = weights[down].clone()
    j = math.ceil(math.log2(float(rows.abs().max()) / (127 * 65535.75))) + 1
    rows[0] = rows[0] / rows[0].abs().max() * (127 * 65535.75 * 2.0**j)
    # Row 1 is zero, which holds at any scale and must move no other channel's multiplier.
    zero = rows.clone()
    zero[1] = 0
    found: dict[str, tuple[int, torch.Tensor]] = {}
    downs = [f"model.layers.{layer}.mlp.down_proj.weight" for layer in range(ARCH.n_layers)]
    for case, w in (("top", rows), ("zero", zero)):
        quantized = choose(ARCH, Profile.Servable, {**weights, down: w}, maxima, affine=downs)
        (params,) = [c for c in quantized.quant.channels if c.multiplier.size and c.layer == 0]
        r = affine_shifts(quantized)[LayerPoint(0, Site.MlpResidual)]
        found[case] = (r, torch.tensor(params.multiplier.astype("int64")))
    r, m = found["top"]
    assert r == offset - j - 1
    assert int(m[0]) == int(m.max()) == 32768
    r_zero, m_zero = found["zero"]
    assert r_zero == r
    assert int(m_zero[1]) == 32768
    assert torch.equal(m_zero[2:], m[2:])


def test_affine_refuses_what_cannot_take_it(pipeline: dict[Profile, Any]) -> None:
    weights, maxima, _ = pipeline[Profile.Servable]
    with pytest.raises(TesseraQuantError, match="affine applies to Q, K, V, O, up and down"):
        choose(
            ARCH, Profile.Servable, weights, maxima, affine=["model.layers.0.mlp.gate_proj.weight"]
        )
    with pytest.raises(TesseraQuantError, match="lacks"):
        choose(
            ARCH, Profile.Servable, weights, maxima, affine=["model.layers.9.mlp.up_proj.weight"]
        )
    # RFC-0038: a weight takes affine in every layer or in none.
    with pytest.raises(TesseraQuantError, match=r"mlp\.down_proj in layers \[1\] only"):
        choose(
            ARCH, Profile.Servable, weights, maxima, affine=["model.layers.1.mlp.down_proj.weight"]
        )

    def every(module: str) -> list[str]:
        return [f"model.layers.{layer}.{module}.weight" for layer in range(ARCH.n_layers)]

    weights_m, maxima_m, _ = pipeline[Profile.Minimal]
    with pytest.raises(TesseraQuantError, match="minimal's up projection"):
        choose(ARCH, Profile.Minimal, weights_m, maxima_m, affine=every("mlp.up_proj"))
    for name in ("self_attn.o_proj", "mlp.down_proj"):
        with pytest.raises(TesseraQuantError, match="minimal's O and down keys exceed 62 bits"):
            choose(ARCH, Profile.Minimal, weights_m, maxima_m, affine=every(name))
    # The sites minimal keeps after Q, K and V take it.
    names = [name for n in "qkv" for name in every(f"self_attn.{n}_proj")]
    quantized = choose(ARCH, Profile.Minimal, weights_m, maxima_m, affine=names)
    assert list(affine_shifts(quantized)) == [
        LayerPoint(layer, site)
        for layer in range(ARCH.n_layers)
        for site in (Site.RopeQ, Site.RopeK, Site.V)
    ]


def test_quantization_is_deterministic_and_refuses_missing_statistics(
    pipeline: dict[Profile, Any],
) -> None:
    weights, maxima, quantized = pipeline[Profile.Servable]
    again = choose(ARCH, Profile.Servable, weights, maxima)
    assert again.quant.public == quantized.quant.public
    assert all(
        (x.data == y.data).all() for x, y in zip(again.weights, quantized.weights, strict=True)
    )
    partial = {n: v for n, v in maxima.items() if n != "layers.1.silu"}
    with pytest.raises(CalibrationError, match=r"lack layers\.1\.silu"):
        choose(ARCH, Profile.Servable, weights, partial)


def test_each_channels_class_is_its_own_shift(pipeline: dict[Profile, Any]) -> None:
    # Every hidden class vector maps each output channel to the shift its exponent gives, and a
    # site's classes run ascending.
    for profile in Profile:
        _, _, quantized = pipeline[profile]
        expected = expected_sets(quantized, profile)
        names = {
            WeightType.Q: ("self_attn.q_proj", Site.RopeQ),
            WeightType.K: ("self_attn.k_proj", Site.RopeK),
            WeightType.V: ("self_attn.v_proj", Site.V),
            WeightType.O: ("self_attn.o_proj", Site.AttnResidual),
            WeightType.Gate: ("mlp.gate_proj", Site.Silu),
            WeightType.Up: ("mlp.up_proj", Site.Up),
            WeightType.Down: ("mlp.down_proj", Site.MlpResidual),
        }
        for shifts in sets(quantized).values():
            assert shifts == sorted(shifts)
        assert quantized.quant.channels
        for params in quantized.quant.channels:
            suffix, site = names[params.ty]
            point = LayerPoint(params.layer, site)
            shifts = sets(quantized)[point]
            exponents = row_exponents(quantized, f"model.layers.{params.layer}.{suffix}.weight")
            per_channel = [shifts[int(c)] for c in params.class_]
            # The shift falls by one per step of exponent: offset - e_c.
            offsets = {r + e for r, e in zip(per_channel, exponents, strict=True)}
            assert len(offsets) == 1, (profile, params.ty, params.layer)
            assert set(per_channel) == set(expected[point])


def counts(quantized: Quantized) -> dict[Site, list[int]]:
    """Each layer site's number of classes, layer by layer."""
    out: dict[Site, list[int]] = {}
    for point, shifts in quantized.quant.public.shift_class_sets:
        if isinstance(point, LayerPoint):
            out.setdefault(point.site, []).append(len(shifts))
    return out


def equalized(w: torch.Tensor) -> torch.Tensor:
    """`w` with every row's largest magnitude the tensor's: one exponent for all its rows."""
    return w / w.abs().amax(dim=1, keepdim=True) * w.abs().max()


def integer_model(profile: Profile, quantized: Quantized) -> IntegerModel:
    """`D` of `quantized`, under a provenance the reference does not read."""
    identity = CalibrationIdentity(
        "fixture", "en", "validation", "0" * 40, ["fixture"], 11, 6, 0, 16, bytes(32)
    )
    provenance = Provenance("0", [], identity, bytes(32), Fraction(1, 2), {})
    return IntegerModel(ARCH, profile, quantized.quant, quantized.weights, provenance)


def one_class(model: IntegerModel, point: LayerPoint, ty: WeightType) -> IntegerModel:
    """`model` with the site at `point` back to the one shift its channels take, and no class
    vector for the weight `ty` before it."""
    public = model.quant.public
    (params,) = (c for c in model.quant.channels if (c.ty, c.layer) == (ty, point.layer))
    shifts = dict(public.shift_class_sets)[point]
    used = sorted({shifts[int(k)] for k in params.class_})
    sets = [(p, used if p == point else s) for p, s in public.shift_class_sets]
    channels = [c for c in model.quant.channels if c is not params]
    quant = replace(model.quant, public=replace(public, shift_class_sets=sets), channels=channels)
    return replace(model, quant=quant)


@pytest.mark.parametrize(
    ("seed", "module", "point", "ty"),
    [
        # Seed 1 draws layer 1's K within one exponent, and layer 0's within two.
        (1, None, LayerPoint(1, Site.RopeK), WeightType.K),
        (7, "mlp.gate_proj", LayerPoint(1, Site.Silu), WeightType.Gate),
        (7, "self_attn.o_proj", LayerPoint(0, Site.AttnResidual), WeightType.O),
        # Scaled to the shift 63, which has no class above it.
        (7, "mlp.down_proj", LayerPoint(1, Site.MlpResidual), WeightType.Down),
    ],
    ids=["rope-k", "silu", "attn-residual", "steep-residual"],
)
def test_a_site_of_one_class_takes_another_where_a_layer_needs_several(
    tmp_path: Path, seed: int, module: str | None, point: LayerPoint, ty: WeightType
) -> None:
    site = point.site
    name = None if module is None else f"model.layers.{point.layer}.{module}.weight"
    weights, maxima = migrated(tmp_path, seed)
    steep = site is Site.MlpResidual
    sequence = np.array([1, 7, 9, 3, 12], dtype=np.uint32)
    for profile in Profile:
        w = dict(weights)
        if name is not None:
            # Every row at the tensor's largest magnitude: one exponent, so one shift at the site.
            w[name] = equalized(weights[name])
        if name is not None and steep:
            (r,) = expected_sets(choose(ARCH, profile, w, maxima), profile)[point]
            w[name] = w[name] * 2.0 ** (r - 63)
        quantized = choose(ARCH, profile, w, maxima)
        for each, n in counts(quantized).items():
            assert all(k == 1 for k in n) or all(k > 1 for k in n), (profile, each, n)
        (r,) = expected_sets(quantized, profile)[point]
        assert r == 63 if steep else r < 63
        shifts = sets(quantized)[point]
        assert shifts == ([62, 63] if steep else [r, r + 1]), (profile, shifts)
        # No channel takes the class added, so D computes what it computes without it.
        (params,) = (c for c in quantized.quant.channels if (c.ty, c.layer) == (ty, point.layer))
        assert {shifts[int(k)] for k in params.class_} == {r}
        d = integer_model(profile, quantized)
        alone = one_class(d, point, ty)
        assert (run_reference(d, sequence) == run_reference(alone, sequence)).all()
        if site in (Site.AttnResidual, Site.MlpResidual):
            # Two steps in every layer, as the compiler lowers the site.
            found = trace(d, sequence).updates
            layers = {p.layer for p in found if isinstance(p, LayerPoint) and p.site is site}
            assert layers == set(range(ARCH.n_layers)), profile


def test_the_embedding_is_no_finer_than_the_residual(pipeline: dict[Profile, Any]) -> None:
    # A residual far larger than the embedding: the embedding's exponent rises to the residual's,
    # so that its shift into the residual is 0 rather than negative.
    weights, maxima, _ = pipeline[Profile.Servable]
    large = {
        n: v * 2.0**20 if n.endswith(("attn_input", "mlp_input")) or n == "final.input" else v
        for n, v in maxima.items()
    }
    quantized = choose(ARCH, Profile.Servable, weights, large)
    public = quantized.quant.public
    embed = float(weights["model.embed_tokens.weight"].abs().max())
    assert exponent(embed, 127) < public.residual_exponent
    assert act(quantized)[EmbedPoint()] == public.residual_exponent
    assert {float(x) for x in quantized.scales["model.embed_tokens.weight"]} == {
        2.0**public.residual_exponent
    }


def test_the_residual_holds_every_tokens_embedding(pipeline: dict[Profile, Any]) -> None:
    # The calibration's tokens lie in [2, 61], so token 63's row enters the residual unseen. A
    # row 2^12 times the others' still fits INT20: the residual's scale covers the table, so the
    # shift s_e passes the compiler's check 127 * 2^s_e < 2^(bits - 1).
    weights, maxima, _ = pipeline[Profile.Servable]
    embed = weights["model.embed_tokens.weight"].clone()
    embed[ARCH.vocab - 1] *= 2.0**12
    quantized = choose(
        ARCH, Profile.Servable, {**weights, "model.embed_tokens.weight": embed}, maxima
    )
    public = quantized.quant.public
    shift = act(quantized)[EmbedPoint()] - public.residual_exponent
    assert shift >= 0
    assert 127 << shift < 1 << (public.residual_bits - 1)
    assert float(embed.abs().max()) <= 2.0 ** (public.residual_exponent + public.residual_bits - 2)


def synthetic(
    *gate_rows: torch.Tensor,
) -> tuple[Any, dict[str, torch.Tensor], dict[str, torch.Tensor]]:
    """A model of width 1024, a layer per gate given, whose gate key takes 25 bits: SiLU's core
    can bind."""
    arch = replace(
        ARCH,
        n_layers=len(gate_rows),
        d_model=1024,
        n_heads=8,
        n_kv_heads=8,
        d_head=128,
        d_ff=4,
        vocab=8,
    )
    generator = torch.Generator().manual_seed(3)
    shapes = tensor_shapes(arch)
    weights = {
        n: torch.randn(s, generator=generator, dtype=torch.float64) * (1.0 if len(s) == 1 else 0.02)
        for n, s in shapes.items()
    }
    weights["model.embed_tokens.weight"] *= 50
    for layer, rows in enumerate(gate_rows):
        weights[f"model.layers.{layer}.mlp.gate_proj.weight"] = rows
    # Activations near 1, and a residual near 2^10 as Llama's outliers make it.
    maxima = {
        name: torch.ones(width, dtype=torch.float64)
        * (2.0**10 if name.endswith(("attn_input", "mlp_input", "final.input")) else 1.0)
        for name, width in points(arch).items()
    }
    return arch, weights, maxima


def test_silu_takes_the_finest_input_exponent_its_core_allows() -> None:
    # Gate rows of one exponent: the floor r = 1 gives a core of exactly 2^24, which fits.
    rows = torch.full((4, 1024), 0.5, dtype=torch.float64)
    arch, weights, maxima = synthetic(rows)
    quantized = choose(arch, Profile.Servable, weights, maxima)
    ((_, in_exp),) = quantized.quant.silu_in_exponents
    assert sets(quantized)[LayerPoint(0, Site.Silu)] == [1]
    y2 = act(quantized)[LayerPoint(0, Site.MlpNorm)]
    (e,) = {
        round(math.log2(float(s))) for s in quantized.scales["model.layers.0.mlp.gate_proj.weight"]
    }
    assert in_exp == y2 + e + 1
    # Two exponents: the floor's shifts 1 and 2 give 2^24 + 2^23 values, so h coarsens by a bit.
    two = rows.clone()
    two[2:] *= 2
    arch, weights, maxima = synthetic(two)
    quantized = choose(arch, Profile.Servable, weights, maxima)
    assert sets(quantized)[LayerPoint(0, Site.Silu)] == [2, 3]
    # RFC-0038: beside a layer of two exponents, a layer of one takes a class more, whose 2^23
    # values pass the core too: that layer's h coarsens by a bit, to the shift 2 beside 3.
    arch, weights, maxima = synthetic(two, rows)
    quantized = choose(arch, Profile.Servable, weights, maxima)
    assert sets(quantized)[LayerPoint(1, Site.Silu)] == [2, 3]
    in_exp = dict(quantized.quant.silu_in_exponents)[1]
    assert in_exp == act(quantized)[LayerPoint(1, Site.MlpNorm)] + e + 2


def test_the_epsilon_term_rounds_up_only_what_is_not_whole(pipeline: dict[Profile, Any]) -> None:
    # epsilon = 2^-20 makes d * epsilon / sigma_res^2 a whole number: its ceiling is itself.
    weights, maxima, quantized = pipeline[Profile.Servable]
    e_res = quantized.quant.public.residual_exponent
    exact = replace(ARCH, rms_eps=Fraction(1, 1 << 20))
    term = choose(exact, Profile.Servable, weights, maxima).quant.public.rms_eps_int
    assert term == ARCH.d_model * Fraction(1, 1 << 20) * Fraction(4) ** -e_res
