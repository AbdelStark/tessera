# SPDX-License-Identifier: Apache-2.0 OR MIT
"""`D`'s scales and shifts for a migrated model (RFC-0005 'Scale policy', 'Integer semantics').

`choose` turns the float64 weights and statistics of `tessera_quant.migration.transform` into
`D`'s INT8 weights and its `QuantSpec`, for one migration strength. `tessera_quant.quantize`
chooses the strength and assembles `D`:

- **Weights**, per output channel: `sigma_c = 2^e_c`, the least power of two that keeps every
  weight of the channel within +-127, rounded to nearest even, so no weight clips. A tensor holds
  at most four consecutive exponents: channels below the window take its lowest. The two rows of
  each RoPE pair share an exponent, since the rotation mixes them. Three tensors take one
  exponent for all their rows, because nothing after them applies a per-channel factor: the
  embedding, whose rows enter the residual through one shift; the LM head, whose logits compare
  across the vocabulary; and, in `minimal`, the up projection, whose accumulator multiplies
  SiLU's output unrequantized.
- **Activations.** Each INT8 tensor's exponent is the least that keeps its calibration maximum
  within +-127. The residual's keeps within `2^(bits - 2)`, one bit of headroom, both the
  calibration's residual stream and the whole embedding table, since any token's row enters it,
  seen in calibration or not. The embedding's exponent is at least the residual's, so that its
  shift `s_e` into the residual is not negative, and the table's maximum bounds `s_e` by 12:
  `127 * 2^s_e` stays inside INT20.
- **Shifts.** Each site moves its accumulator's scale to its output's, `r_c = e_out - e_in -
  e_c`, with RFC-0005's factors: `2^kappa` at the norms, the RoPE constants' `2^bits`, the
  attention output's `2^(2 sigma)`. A site's classes are its distinct shifts, ascending, and
  each must lie in `[1, 63]`.
- **Constants.** kappa follows RFC-0023, RMSNorm's term is `ceil(d * eps / sigma_res^2)`, and each
  layer's score scale `2^(e_q + e_k) / sqrt(d_head)` is the rational `2^(e_q + e_k) * n / 2^40`,
  `n` the integer nearest `2^40 / sqrt(d_head)`. SiLU's input exponent is the finest whose table
  core, over the gate's classes, holds at most `2^24` values of `h`, for the key width the
  compiler gives the table (RFC-0006 'Llama lowering').
- **`affine`** (RFC-0005 'Scale policy', RFC-0025): a weight named in `affine` takes per-channel
  scales `max|w_c| / 127`, and its site a 16-bit multiplier per channel with one shift, the
  largest at which every multiplier still fits. The site goes in `affine_sites` with its shift,
  and the multipliers in the weight's `ChannelParams`, with no classes. Q, K, V, O, up and down
  can, except `minimal`'s O and down, whose keys would exceed 62 bits. `tessera_quant.quantize`
  with `affine` scales names every weight that can take them.
- **Every layer alike** (RFC-0038), so that the layers lower to one template: `affine` names a
  weight in every layer or in none, and a `pot` site whose channels share one shift `r` in a
  layer, while they take several in another, takes one class more there that no channel uses:
  `r + 1`, or 62 beside 63. At SiLU that class counts toward the table's core.

Weights and statistics are secret-class (05 §3): errors name tensors and sites, never a value.
"""

import math
from collections.abc import Collection, Mapping
from dataclasses import dataclass, field
from fractions import Fraction

import numpy as np
import torch

from tessera_quant.arch import ArchSpec
from tessera_quant.errors import CalibrationError, QuantizationInfeasibleError, TesseraQuantError
from tessera_quant.integer_model import (
    ChannelParams,
    EmbedPoint,
    FinalNormPoint,
    LayerPoint,
    Profile,
    PublicQuantSpec,
    QuantPoint,
    QuantSpec,
    Site,
    WeightTensor,
    WeightType,
    norm_kappa,
)

EXP_SIGMA = 16
"""sigma, the softmax exponentials' precision (RFC-0005 'Integer semantics')."""

RESIDUAL_BITS = 20
"""The residual stream's width: INT20."""

LOGITS_BITS = 28
"""The logits' width, as the claim list states it."""

WINDOW = 4
"""The most exponent classes a weight takes (RFC-0005 'Scale policy')."""

SILU_CORE = 1 << 24
"""The most values of `h` a `Silu` table's core holds (RFC-0008)."""

_SCORE_BITS = 40
"""The score scale's precision: `1 / sqrt(d_head)` rounds to a multiple of `2^-40`."""

_AFFINE_NAMES = frozenset({"q_proj", "k_proj", "v_proj", "o_proj", "up_proj", "down_proj"})
"""The Hugging Face names of the weights `affine` can take."""

_AFFINE = frozenset(
    {WeightType.Q, WeightType.K, WeightType.V, WeightType.O, WeightType.Up, WeightType.Down}
)

_CLASSED = (
    (Site.RopeQ, "self_attn.q_proj", True),
    (Site.RopeK, "self_attn.k_proj", True),
    (Site.V, "self_attn.v_proj", False),
    (Site.AttnResidual, "self_attn.o_proj", False),
    (Site.Silu, "mlp.gate_proj", False),
    (Site.Up, "mlp.up_proj", False),
    (Site.MlpResidual, "mlp.down_proj", False),
)
"""Each site whose classes follow a weight's row exponents: the weight's module, and whether the
rows of its RoPE pairs share an exponent."""


@dataclass(frozen=True, eq=False)
class Quantized:
    """`D`'s quantization constants and INT8 weights, ordered by `(ty, layer)`: secret-class."""

    quant: QuantSpec
    weights: list[WeightTensor]
    scales: dict[str, torch.Tensor] = field(repr=False)
    """Each weight's per-row scale, by its Hugging Face name: `2^e_c` in `pot` mode, so that
    `q * scale` is the dequantized weight. Not part of `D`, which needs only the shifts."""


def exponent(maximum: float, limit: int) -> int:
    """The least `e` with `maximum <= limit * 2^e`, exactly.

    Raises:
        CalibrationError: for a maximum that is not finite and positive (`ERR-QUANT-CALIB`).
    """
    if not (math.isfinite(maximum) and maximum > 0):
        raise CalibrationError("a scale needs a finite, positive maximum")
    value = Fraction(maximum) / limit
    _, e = math.frexp(float(value))
    while Fraction(2) ** (e - 1) >= value:
        e -= 1
    while Fraction(2) ** e < value:
        e += 1
    return e


def choose(
    arch: ArchSpec,
    profile: Profile,
    weights: Mapping[str, torch.Tensor],
    maxima: Mapping[str, torch.Tensor],
    *,
    affine: Collection[str] = (),
) -> Quantized:
    """`D`'s INT8 weights and constants for the migrated `weights` and their statistics `maxima`.

    `affine` names, by their Hugging Face names, the weights quantized in `affine` mode.

    Raises:
        CalibrationError: for missing, non-finite or degenerate statistics (`ERR-QUANT-CALIB`).
        QuantizationInfeasibleError: for derived shifts, multipliers, score rationals or SiLU
            scales outside the fixed bounds (also `ERR-QUANT-CALIB`).
        TesseraQuantError: for a weight named in `affine` that cannot take it.
    """
    return _Quantizer(arch, profile, weights, maxima, frozenset(affine)).run()


class _Quantizer:
    def __init__(
        self,
        arch: ArchSpec,
        profile: Profile,
        weights: Mapping[str, torch.Tensor],
        maxima: Mapping[str, torch.Tensor],
        affine: frozenset[str],
    ) -> None:
        self.arch = arch
        self.profile = profile
        self.weights = weights
        self.maxima = maxima
        self.affine = affine
        self.act: list[tuple[QuantPoint, int]] = []
        self.sets: list[tuple[QuantPoint, list[int]]] = []
        self.affine_sites: list[tuple[QuantPoint, int]] = []
        self.channels: list[ChannelParams] = []
        self.tensors: list[WeightTensor] = []
        self.scales: dict[str, torch.Tensor] = {}
        self.exponents: dict[str, list[int]] = {}
        self.several: frozenset[Site] = frozenset()

    def run(self) -> Quantized:
        arch = self.arch
        unknown = self.affine - set(self.weights)
        if unknown:
            raise TesseraQuantError(f"affine names tensors the model lacks: {sorted(unknown)}")
        for name in sorted(self.affine):
            if name.rsplit(".", 2)[-2] not in _AFFINE_NAMES:
                raise TesseraQuantError(f"{name}: affine applies to Q, K, V, O, up and down")
        layers = range(arch.n_layers)
        for _, module, _ in _CLASSED:
            named = [layer for layer in layers if _name(layer, module) in self.affine]
            if named and len(named) < arch.n_layers:
                raise TesseraQuantError(
                    f"affine names {module} in layers {named} only: every layer or none (RFC-0038)"
                )
        # A site takes several classes in every layer once one layer's channels need several.
        several: set[Site] = set()
        # Layer-major traversal keeps a streamed checkpoint's transformed group hot. Only
        # channel exponents survive this scan, never another copy of its float64 weights.
        for layer in layers:
            for site, module, pairs in _CLASSED:
                if (
                    site not in several
                    and (site is not Site.Up or self.profile is Profile.Servable)
                    and _name(layer, module) not in self.affine
                    and len(set(self._exponents(_name(layer, module), pairs))) > 1
                ):
                    several.add(site)
        self.several = frozenset(several)
        residual = max(
            self._maximum(name)
            for layer in range(arch.n_layers)
            for name in (f"layers.{layer}.attn_input", f"layers.{layer}.mlp_input")
        )
        embed = self._float("model.embed_tokens.weight")
        # Every token's row enters the residual, whether the calibration saw it or not.
        top = float(embed.abs().max())
        residual = max(residual, self._maximum("final.input"), top)
        self.e_res = exponent(residual, 1 << (RESIDUAL_BITS - 2))
        rms_eps = math.ceil(arch.d_model * arch.rms_eps * Fraction(4) ** -self.e_res)
        self.kappa = norm_kappa(arch.d_model, rms_eps)
        e_embed = max(exponent(top, 127), self.e_res)
        # The transposed embedding: its rows are the residual's channels.
        self._store(
            "model.embed_tokens.weight", WeightType.Embed, 0, embed.T, [e_embed] * arch.d_model
        )
        self.act.append((EmbedPoint(), e_embed))
        scales: list[tuple[int, Fraction]] = []
        silu: list[tuple[int, int]] = []
        for layer in range(arch.n_layers):
            e_q, e_k, in_exp = self._layer(layer)
            scales.append((layer, _score_scale(e_q + e_k, arch.d_head)))
            silu.append((layer, in_exp))
        e_final = self._act("final.norm")
        self.act.append((FinalNormPoint(), e_final))
        self.sets.append((FinalNormPoint(), [self._shift(self.kappa + e_final, "the final norm")]))
        lm_head = self._float("lm_head.weight")
        e_lm = _whole(lm_head)
        self._store("lm_head.weight", WeightType.LmHead, 0, lm_head, [e_lm] * arch.vocab)
        public = PublicQuantSpec(
            act_exponents=sorted(self.act, key=lambda item: _order(item[0])),
            residual_exponent=self.e_res,
            residual_bits=RESIDUAL_BITS,
            rsqrt_kappa=self.kappa,
            rms_eps_int=rms_eps,
            exp_sigma=EXP_SIGMA,
            score_scales=scales,
            shift_class_sets=sorted(self.sets, key=lambda item: _order(item[0])),
            affine_sites=sorted(self.affine_sites, key=lambda item: _order(item[0])),
            rope=arch.rope,
            logits_bits=LOGITS_BITS,
            logits_exponent=e_final + e_lm,
        )
        channels = sorted(self.channels, key=lambda c: (c.ty.value, c.layer))
        tensors = sorted(self.tensors, key=lambda w: (w.ty.value, w.layer))
        return Quantized(QuantSpec(public, silu, channels), tensors, self.scales)

    def _layer(self, layer: int) -> tuple[int, int, int]:
        """Quantizes layer `layer`, and returns its query and key exponents and SiLU's `in_exp`."""
        arch = self.arch
        prefix = f"model.layers.{layer}"

        def point(site: Site) -> LayerPoint:
            return LayerPoint(layer, site)

        servable = self.profile is Profile.Servable
        sigma2 = 2 * EXP_SIGMA

        # Attention.
        e_y = self._act(f"layers.{layer}.attn_norm")
        self.act.append((point(Site.AttnNorm), e_y))
        self.sets.append((point(Site.AttnNorm), [self._shift(self.kappa + e_y, "a norm")]))
        e_q, e_k, e_v = (self._act(f"layers.{layer}.{p}") for p in ("q", "k", "v"))
        for site, e in ((Site.RopeQ, e_q), (Site.RopeK, e_k), (Site.V, e_v)):
            self.act.append((point(site), e))
        bits = arch.rope.bits
        self._weighted(
            WeightType.Q,
            layer,
            f"{prefix}.self_attn.q_proj.weight",
            Site.RopeQ,
            e_q - e_y + bits,
            pairs=True,
        )
        self._weighted(
            WeightType.K,
            layer,
            f"{prefix}.self_attn.k_proj.weight",
            Site.RopeK,
            e_k - e_y + bits,
            pairs=True,
        )
        self._weighted(WeightType.V, layer, f"{prefix}.self_attn.v_proj.weight", Site.V, e_v - e_y)
        if servable:
            e_o = self._act(f"layers.{layer}.attn_out")
            self.act.append((point(Site.AttnOut), e_o))
            self.sets.append(
                (point(Site.AttnOut), [self._shift(e_o - e_v + sigma2, "the attention output")])
            )
            e_in = e_o
        else:
            e_in = e_v - sigma2
        self._weighted(
            WeightType.O,
            layer,
            f"{prefix}.self_attn.o_proj.weight",
            Site.AttnResidual,
            self.e_res - e_in,
        )

        # MLP.
        e_y2 = self._act(f"layers.{layer}.mlp_norm")
        self.act.append((point(Site.MlpNorm), e_y2))
        self.sets.append((point(Site.MlpNorm), [self._shift(self.kappa + e_y2, "a norm")]))
        e_a = self._act(f"layers.{layer}.silu")
        self.act.append((point(Site.Silu), e_a))
        in_exp = self._silu(layer, f"{prefix}.mlp.gate_proj.weight", e_y2, e_a)
        up = f"{prefix}.mlp.up_proj.weight"
        if servable:
            e_u = self._act(f"layers.{layer}.up")
            e_h = self._act(f"layers.{layer}.h")
            self.act += [(point(Site.Up), e_u), (point(Site.Product), e_h)]
            self._weighted(WeightType.Up, layer, up, Site.Up, e_u - e_y2)
            self.sets.append((point(Site.Product), [self._shift(e_h - e_a - e_u, "the product")]))
            e_in = e_h
        else:
            # `a * u` sums over the down projection's columns: one exponent for all of `W_up`.
            if up in self.affine:
                raise TesseraQuantError(f"{up}: minimal's up projection has no site to take affine")
            w_up = self._float(up)
            e_up = _whole(w_up)
            self._store(up, WeightType.Up, layer, w_up, [e_up] * arch.d_ff)
            e_in = e_a + e_y2 + e_up
        self._weighted(
            WeightType.Down,
            layer,
            f"{prefix}.mlp.down_proj.weight",
            Site.MlpResidual,
            self.e_res - e_in,
        )
        return e_q, e_k, in_exp

    def _weighted(
        self,
        ty: WeightType,
        layer: int,
        name: str,
        site: Site,
        offset: int,
        *,
        pairs: bool = False,
    ) -> None:
        """Quantizes the weight `name` and the site after it.

        The site's shift for a channel of exponent `e_c` is `offset - e_c`.
        """
        w = self._float(name)
        point = LayerPoint(layer, site)
        if name in self.affine:
            if ty not in _AFFINE:
                raise TesseraQuantError(f"{name}: affine applies to Q, K, V, O, up and down")
            if self.profile is Profile.Minimal and ty in (WeightType.O, WeightType.Down):
                # Their accumulators reach about 2^60 and 2^58, unrequantized (RFC-0025).
                raise TesseraQuantError(
                    f"{name}: minimal's O and down keys exceed 62 bits with a multiplier"
                )
            self._affine(name, ty, layer, w, point, offset, pairs=pairs)
            return
        exponents = self._exponents(name, pairs)
        shifts = [self._shift(offset - e, f"{name}'s site") for e in exponents]
        self._store(name, ty, layer, w, exponents)
        self._classes(ty, layer, point, shifts)

    def _affine(
        self,
        name: str,
        ty: WeightType,
        layer: int,
        w: torch.Tensor,
        point: QuantPoint,
        offset: int,
        *,
        pairs: bool,
    ) -> None:
        """`affine` mode: scales `max|w_c| / 127`, and multipliers `M_c` with one shift `r`.

        `M_c / 2^r` is the factor `sigma_c / 2^offset` that the site applies to the accumulator,
        as the `pot` shift `offset - e_c` applies `2^e_c / 2^offset`. A zero row holds at any
        scale: it takes the largest, which leaves `r` and the other multipliers as they are.
        """
        maxima = w.abs().amax(dim=1)
        if pairs:
            maxima = _pair_maxima(maxima, self.arch.d_head)
        largest = float(maxima.max())
        fallback = Fraction(largest) / 127 if largest > 0 else Fraction(1)
        scales = [Fraction(float(m)) / 127 if m > 0 else fallback for m in maxima]
        per_row = torch.tensor([float(s) for s in scales], dtype=torch.float64)
        q = torch.round(w / per_row[:, None])
        factors = [s / Fraction(2) ** offset for s in scales]
        top = max(factors)
        # The largest shift at which every multiplier fits 16 bits. Rounding is monotone, so the
        # largest factor's decides, and it can round up to 2^16 from below.
        r = 0
        while round(top * 2 ** (r + 1)) < 1 << 16:
            r += 1
        multipliers = [round(f * 2**r) for f in factors]
        if not 1 <= r <= 63 or min(multipliers) < 1:
            raise QuantizationInfeasibleError(
                "an affine site's multipliers fit 16 bits and one shift in [1, 63]"
            )
        self.tensors.append(_tensor(ty, layer, q))
        self.scales[name] = per_row
        self.affine_sites.append((point, r))
        self.channels.append(
            ChannelParams(
                ty,
                layer,
                np.zeros(0, dtype=np.uint8),
                np.asarray(multipliers, dtype=np.uint16),
            )
        )

    def _silu(self, layer: int, name: str, e_y2: int, e_a: int) -> int:
        """Quantizes the gate, and returns the `Silu` table's `in_exp`.

        `in_exp` is the finest for which the table's core, `2^(b - r_c)` values of `h` per class,
        holds at most `2^24`.
        """
        w = self._float(name)
        exponents = self._exponents(name, False)
        self._store(name, WeightType.Gate, layer, w, exponents)
        # The compiler's key width: the gate accumulator of INT8 activations lies within
        # +-127 * 128 * d_model (RFC-0006 'Llama lowering').
        b = (127 * 128 * self.arch.d_model).bit_length() + 1
        distinct = sorted(set(exponents))

        def classes(in_exp: int) -> list[int]:
            return self._padded(Site.Silu, sorted(in_exp - e_y2 - e for e in distinct))

        in_exp = max(e_y2 + e + 1 for e in distinct)
        while sum(1 << max(b - r, 0) for r in classes(in_exp)) > SILU_CORE:
            in_exp += 1
        shifts = [in_exp - e_y2 - e for e in exponents]
        if not (-64 <= in_exp <= 0 and 1 <= 63 - in_exp + e_a <= 126 and max(classes(in_exp)) < b):
            raise QuantizationInfeasibleError(
                f"layer {layer}: no Silu table takes the gate's scales"
            )
        self._classes(WeightType.Gate, layer, LayerPoint(layer, Site.Silu), shifts)
        return in_exp

    def _classes(self, ty: WeightType, layer: int, point: LayerPoint, shifts: list[int]) -> None:
        """Records the site's classes, its distinct shifts ascending, `_padded`.

        With several, each channel's class joins the hidden `ChannelParams`.
        """
        distinct = self._padded(point.site, sorted(set(shifts)))
        self.sets.append((point, distinct))
        if len(distinct) > 1:
            index = {r: k for k, r in enumerate(distinct)}
            classes = np.asarray([index[r] for r in shifts], dtype=np.uint8)
            self.channels.append(ChannelParams(ty, layer, classes, np.zeros(0, dtype=np.uint16)))

    def _padded(self, site: Site, distinct: list[int]) -> list[int]:
        """A site's classes: its distinct shifts, and one more if another layer's need several.

        No channel takes the extra class, so `D` computes what it computed without it (RFC-0038
        'What the quantizer can do').
        """
        if len(distinct) > 1 or site not in self.several:
            return distinct
        (r,) = distinct
        return [r, r + 1] if r < 63 else [r - 1, r]

    def _exponents(self, name: str, pairs: bool) -> list[int]:
        """The weight's row exponents, `_row_exponents`, computed once."""
        if name not in self.exponents:
            d_head = self.arch.d_head if pairs else None
            self.exponents[name] = _row_exponents(self._float(name), d_head)
        return self.exponents[name]

    def _store(
        self, name: str, ty: WeightType, layer: int, w: torch.Tensor, exponents: list[int]
    ) -> None:
        scale = torch.tensor([2.0**e for e in exponents], dtype=torch.float64)
        self.tensors.append(_tensor(ty, layer, torch.round(w / scale[:, None])))
        self.scales[name] = scale

    def _shift(self, r: int, what: str) -> int:
        if not 1 <= r <= 63:
            raise QuantizationInfeasibleError(f"{what}: a shift lies in [1, 63]")
        return r

    def _act(self, name: str) -> int:
        return exponent(self._maximum(name), 127)

    def _maximum(self, name: str) -> float:
        try:
            return float(self.maxima[name].max())
        except KeyError:
            raise CalibrationError(f"the statistics lack {name}") from None

    def _float(self, name: str) -> torch.Tensor:
        return self.weights[name].to(torch.float64)


def _row_exponents(w: torch.Tensor, d_head: int | None) -> list[int]:
    """Each row's least exponent keeping it within +-127.

    The two rows of each RoPE pair share theirs when `d_head` is given, and every exponent is
    raised into a window of `WINDOW` below the largest.
    """
    maxima = w.abs().amax(dim=1)
    if d_head is not None:
        maxima = _pair_maxima(maxima, d_head)
    exponents = [exponent(float(m), 127) if m > 0 else None for m in maxima]
    present = [e for e in exponents if e is not None]
    if not present:
        return [0] * len(exponents)
    low = max(present) - (WINDOW - 1)
    return [low if e is None else max(e, low) for e in exponents]


def _name(layer: int, module: str) -> str:
    """The Hugging Face name of a layer's weight in `module`."""
    return f"model.layers.{layer}.{module}.weight"


def _pair_maxima(maxima: torch.Tensor, d_head: int) -> torch.Tensor:
    """Each row's maximum raised to its RoPE pair's: rows `i` and `i + d_head/2` of each head."""
    half = d_head // 2
    by_head = maxima.reshape(-1, 2, half)
    return by_head.amax(dim=1, keepdim=True).expand(-1, 2, -1).reshape(-1)


def _whole(w: torch.Tensor) -> int:
    """One exponent for every row of `w`."""
    return exponent(float(w.abs().max()), 127)


def _tensor(ty: WeightType, layer: int, q: torch.Tensor) -> WeightTensor:
    if bool((q.abs() > 127).any()):
        raise TesseraQuantError(f"{ty.name} of layer {layer}: a weight clips")
    rows, cols = q.shape
    data = q.to(torch.int8).numpy().reshape(-1)
    return WeightTensor(ty, layer, int(rows), int(cols), data)


def _score_scale(e: int, d_head: int) -> Fraction:
    """`2^e / sqrt(d_head)` within `sqrt(d_head) * 2^-(m + 1)` relative, `m = _SCORE_BITS`.

    The rational is `2^e * n / 2^m`, `n` the integer nearest `2^m / sqrt(d_head)`.

    Raises:
        QuantizationInfeasibleError: when the rational leaves the claim list's 64 bits.
    """
    # round(2^m / sqrt(d)) = floor((isqrt(4^(m+1) // d) + 1) / 2), m = _SCORE_BITS.
    m = _SCORE_BITS
    numerator = (math.isqrt((1 << (2 * m + 2)) // d_head) + 1) // 2
    scale = Fraction(numerator, 1 << m) * Fraction(2) ** e
    if scale.numerator >= 1 << 63 or scale.denominator >= 1 << 64:
        raise QuantizationInfeasibleError("a score scale fits a rational of 64 bits")
    return scale


def _order(point: QuantPoint) -> tuple[int, int, int]:
    """03 §2's order: the embedding, the layers with their sites, the final norm."""
    if isinstance(point, EmbedPoint):
        return (0, 0, 0)
    if isinstance(point, LayerPoint):
        return (1, point.layer, point.site.value)
    return (2, 0, 0)
