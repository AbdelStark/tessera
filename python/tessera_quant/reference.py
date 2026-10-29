# SPDX-License-Identifier: Apache-2.0 OR MIT
"""The integer reference for `D`: its forward pass in PyTorch's int64 (RFC-0005).

This is an independent implementation of `D`'s 'Integer semantics' as RFC-0022 (the saturating
residual) and RFC-0023 (`kappa`) amend them. The CPU engine of `tessera-witness` is checked
against it bit for bit (SC-7, 07 §5), and the fidelity harness runs it.

- **Exact.** Every value is a true integer in int64. Before each product, its bound is computed
  from its operands' largest magnitudes in Python's integers, and a product that could reach
  `2^63` is refused, so nothing wraps silently. The norm's root, whose `d * 2^(2 kappa)` reaches
  `2^94`, is computed in Python's integers, row by row.
- **Tables.** `Exp`, `Silu` and RoPE's constants come from `tessera_quant.tables`, whose
  generators equal the Rust ones (RFC-0008), with the parameters the lowering gives them
  (RFC-0006 'Llama lowering'). The last few tables generated are kept for the next sequence.
  `rq` is its closed form.
- **One sequence** at positions 0 to `n - 1`, as one lowering covers.
- **Affine sites** rescale `x_c * M_c` with one shift, `M_c` the multiplier of the weight's output
  channel (RFC-0025).

Weights, tokens and every tensor computed are secret-class (05 §3). A `Trace` shows no values,
and errors name tensors and sites, never a value.
"""

import math
from dataclasses import dataclass, field
from fractions import Fraction
from functools import lru_cache

import numpy as np
import numpy.typing as npt
import torch

from tessera_quant.errors import TesseraQuantError, UsageError
from tessera_quant.integer_model import (
    EmbedPoint,
    FinalNormPoint,
    IntegerModel,
    LayerPoint,
    Profile,
    QuantPoint,
    Site,
    WeightType,
)
from tessera_quant.tables import Exp, LookupTable, Rope, RopeConstants, Silu, lookup_table

INT8 = (-128, 127)
"""The range of `D`'s INT8 tensors."""

_LIMIT = 1 << 63
"""int64's bound: every value and product stays below it in magnitude."""


@dataclass(frozen=True, eq=False)
class Trace:
    """Every tensor `D` commits on one sequence, for differential tests (SC-7): secret-class.

    The residual stream at each layer boundary, each site's output, each two-step residual
    update, each norm's root, each layer's attention maxima, exponentials and reciprocals, and the
    logits, all int64.
    """

    residuals: list[torch.Tensor] = field(repr=False)
    """`[n_layers + 1]` tensors of `[n, d_model]`: after the embedding, then after each layer."""
    sites: dict[QuantPoint, torch.Tensor] = field(repr=False)
    """Each rescale site's output, `[n, channels]`, by its point."""
    updates: dict[QuantPoint, torch.Tensor] = field(repr=False)
    """Each residual site of two steps, `[n, d_model]`: the update `u = rq(acc, r_c, -(2^b - 1),
    2^b - 1)` committed before the sum, `b` the residual's bits (RFC-0022). A site of one shift
    below 63 in every layer has none (RFC-0038)."""
    roots: dict[QuantPoint, torch.Tensor] = field(repr=False)
    """Each norm's root `rho`, `[n]`, by the point of the norm's output."""
    maxima: dict[int, torch.Tensor] = field(repr=False)
    """Each layer's score maxima `c`, `[n_heads, n]`: the max certificates' committed values."""
    exponentials: dict[int, torch.Tensor] = field(repr=False)
    """Each layer's attention exponentials `e`, `[n_heads, n, n]`, 0 at masked positions."""
    reciprocals: dict[int, torch.Tensor] = field(repr=False)
    """Each layer's attention reciprocals `R = floor(2^(2 sigma) / s)`, `[n_heads, n]`."""
    logits: torch.Tensor = field(repr=False)
    """`[n, vocab]`."""


def rq(x: torch.Tensor, r: torch.Tensor | int, lo: int, hi: int) -> torch.Tensor:
    """`rq(x, r, lo, hi) = clamp(floor((x + 2^(r-1)) / 2^r), lo, hi)` of RFC-0005, elementwise.

    `r`, from 1 to 63, is one shift or one per element of `x`'s last axis. Rounding is half up,
    for negative ties too. `x + 2^(r-1)` must stay below `2^63`, which every key of `D` does.
    """
    shift = torch.as_tensor(r, dtype=torch.int64)
    half = torch.bitwise_left_shift(torch.ones_like(shift), shift - 1)
    return torch.clamp(torch.bitwise_right_shift(x + half, shift), lo, hi)


def run_reference(model: IntegerModel, tokens: npt.NDArray[np.uint32]) -> npt.NDArray[np.int64]:
    """`D`'s logits, `[n, vocab]`, on the sequence `tokens` at positions 0 to `n - 1` (02 §4).

    Raises:
        TesseraQuantError: as `trace`.
    """
    return _Forward(model, tokens, record=False).run().logits.numpy()


def trace(model: IntegerModel, tokens: npt.NDArray[np.uint32]) -> Trace:
    """Every tensor `D` commits on the sequence `tokens` at positions 0 to `n - 1`.

    Deterministic: CPU int64 arithmetic only.

    Raises:
        UsageError: for a token outside the vocabulary, or a sequence empty or longer than the
            context.
        TesseraQuantError: for a product that could leave int64.
    """
    return _Forward(model, tokens).run()


class _Forward:
    """One forward pass of `D` over one sequence."""

    def __init__(
        self, model: IntegerModel, tokens: npt.NDArray[np.uint32], *, record: bool = True
    ) -> None:
        arch = model.arch
        public = model.quant.public
        n = len(tokens)
        if not 1 <= n <= arch.max_context:
            raise UsageError("a sequence has 1 to max_context tokens")
        if int(tokens.max()) >= arch.vocab:
            raise UsageError("every token lies in [0, vocab)")
        self.model = model
        self.record = record
        self.tokens = torch.as_tensor(tokens.astype(np.int64))
        self.n = n
        self.shifts = dict(public.shift_class_sets)
        self.affine = dict(public.affine_sites)
        self.exponents = dict(public.act_exponents)
        self.classes = {(c.ty, c.layer): c.class_ for c in model.quant.channels if len(c.class_)}
        self.multipliers = {
            (c.ty, c.layer): c.multiplier for c in model.quant.channels if len(c.multiplier)
        }
        self.weights = {(w.ty, w.layer): w for w in model.weights}

        def one_shift(point: LayerPoint) -> bool:
            shifts = [self.affine[point]] if point in self.affine else self.shifts.get(point, [])
            return len(shifts) == 1 and shifts[0] <= 62

        # RFC-0038: a residual site takes one route in every layer.
        self.two_steps = {
            site
            for site in (Site.AttnResidual, Site.MlpResidual)
            if not all(one_shift(LayerPoint(layer, site)) for layer in range(arch.n_layers))
        }
        rope = arch.rope
        constants = RopeConstants.of(
            Rope(
                rope.theta,
                rope.factor,
                rope.low_freq_factor,
                rope.high_freq_factor,
                rope.original_max,
                rope.bits,
                arch.max_context,
                arch.d_head,
            )
        )
        cos, sin = constants.pairs(np.arange(n, dtype=np.int64))
        self.cos, self.sin = torch.as_tensor(cos), torch.as_tensor(sin)
        self.half = 1 << (public.residual_bits - 1)
        self.sites: dict[QuantPoint, torch.Tensor] = {}
        self.updates: dict[QuantPoint, torch.Tensor] = {}
        self.roots: dict[QuantPoint, torch.Tensor] = {}
        self.maxima: dict[int, torch.Tensor] = {}
        self.exponentials: dict[int, torch.Tensor] = {}
        self.reciprocals: dict[int, torch.Tensor] = {}

    # -- arithmetic ---------------------------------------------------------------------------

    def weight(self, ty: WeightType, layer: int) -> torch.Tensor:
        """The weight `(ty, layer)` as `[rows, cols]` int64, converted when it is used."""
        w = self.weights[ty, layer]
        return torch.as_tensor(w.data.astype(np.int64)).reshape(w.rows, w.cols)

    def matmul(self, x: torch.Tensor, ty: WeightType, layer: int) -> torch.Tensor:
        """`x W^T` for the weight `(ty, layer)`: refused if a sum could reach `2^63`."""
        w = self.weight(ty, layer)
        _bounded(_largest(x) * _largest(w) * w.shape[1], f"{ty.name}'s product")
        return x @ w.T

    def per_channel(self, point: QuantPoint, ty: WeightType | None, layer: int) -> torch.Tensor:
        """Each output channel's shift at `point`: its class's, or the site's one shift."""
        shifts = torch.tensor(self.shifts[point], dtype=torch.int64)
        classes = self.classes.get((ty, layer)) if ty is not None else None
        if classes is None:
            if len(shifts) != 1:
                raise TesseraQuantError(f"{point}: a site of several classes has channel classes")
            return shifts
        return shifts[torch.as_tensor(classes.astype(np.int64))]

    def key(
        self, x: torch.Tensor, point: QuantPoint, ty: WeightType | None, layer: int
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """What the site at `point` rescales, with each channel's shift.

        At a `pot` site, `x` and its classes' shifts; at an affine site, `x_c * M_c` and the one
        shift (RFC-0025).
        """
        if point not in self.affine:
            return x, self.per_channel(point, ty, layer)
        if ty is None:
            raise TesseraQuantError(f"{point}: an affine site follows a weight")
        m = torch.as_tensor(self.multipliers[ty, layer].astype(np.int64))
        _bounded(_largest(x) * _largest(m), f"{point}'s affine key")
        return x * m, torch.tensor(self.affine[point], dtype=torch.int64)

    def requant(
        self,
        x: torch.Tensor,
        point: QuantPoint,
        ty: WeightType | None,
        layer: int,
        bounds: tuple[int, int] = INT8,
    ) -> torch.Tensor:
        """The site at `point`: `rq` of its key with each channel's shift, into `bounds`."""
        y = rq(*self.key(x, point, ty, layer), *bounds)
        if self.record:
            self.sites[point] = y
        return y

    def norm(self, res: torch.Tensor, point: QuantPoint) -> torch.Tensor:
        """RMSNorm: `rho = max{rho : rho^2 (s + e) <= d 2^(2 kappa)}`, `y = rq(res rho, r_n)`."""
        public = self.model.quant.public
        d, kappa, eps = self.model.arch.d_model, public.rsqrt_kappa, public.rms_eps_int
        if eps < 1:
            raise TesseraQuantError("RMSNorm's epsilon term is at least 1")
        _bounded(_largest(res) ** 2 * d, "a sum of squares")
        squares = [int(s) for s in (res * res).sum(dim=1).numpy()]
        top = d << (2 * kappa)
        roots = [math.isqrt(top // (s + eps)) for s in squares]
        rho = torch.tensor(roots, dtype=torch.int64)
        _bounded(_largest(res) * max(roots, default=0), f"{point}'s norm product")
        if self.record:
            self.roots[point] = rho
        return self.requant(res * rho[:, None], point, None, 0)

    def rope(self, x: torch.Tensor, heads: int) -> torch.Tensor:
        """Each head's pairs `(x_i, x_(i + d_head/2))` rotated: `(a C - b S, a S + b C)`."""
        dh = self.model.arch.d_head
        half = dh // 2
        _bounded(_largest(x) * 2 * max(_largest(self.cos), _largest(self.sin)), "RoPE's product")
        split = x.reshape(self.n, heads, dh)
        a, b = split[..., :half], split[..., half:]
        c, s = self.cos[:, None, :], self.sin[:, None, :]
        return torch.cat((a * c - b * s, a * s + b * c), dim=2).reshape(self.n, heads * dh)

    def attention(
        self, layer: int, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor
    ) -> torch.Tensor:
        """Causal attention, per query head: the scores and each row's max certificate.

        The exponentials come from the layer's `Exp` table; then `R = floor(2^(2 sigma) / s)` and
        `o = R * sum_i e_i v_i`.
        """
        arch = self.model.arch
        public = self.model.quant.public
        n, dh, heads = self.n, arch.d_head, arch.n_heads
        group = heads // arch.n_kv_heads
        sigma = public.exp_sigma
        scale = dict(public.score_scales)[layer]
        key_bits = (128 * 255 * dh).bit_length()
        table = _exp_table(key_bits, sigma, scale)
        qh = q.reshape(n, heads, dh).transpose(0, 1)
        kh = k.reshape(n, arch.n_kv_heads, dh).transpose(0, 1).repeat_interleave(group, dim=0)
        vh = v.reshape(n, arch.n_kv_heads, dh).transpose(0, 1).repeat_interleave(group, dim=0)
        _bounded(_largest(qh) * _largest(kh) * dh, "a score")
        scores = qh @ kh.transpose(1, 2)
        causal = torch.ones(n, n, dtype=torch.bool).tril()
        lowest = torch.full_like(scores, -_LIMIT)
        top = torch.where(causal, scores, lowest).amax(dim=2, keepdim=True)
        keys = torch.where(causal, top - scores, torch.zeros_like(scores))
        e = _lookup(table, keys) * causal
        total = e.sum(dim=2)
        if bool((total <= 0).any()):
            raise TesseraQuantError(f"layer {layer}: a row's maximum keeps its exponential")
        r = (1 << (2 * sigma)) // total
        _bounded(_largest(e) * _largest(vh) * n, "an attention sum")
        weighted = e @ vh
        _bounded(_largest(weighted) * _largest(r), "the attention output")
        if self.record:
            self.maxima[layer] = top.squeeze(2)
            self.exponentials[layer] = e
            self.reciprocals[layer] = r
        o = weighted * r[:, :, None]
        return o.transpose(0, 1).reshape(n, heads * dh)

    def silu(self, layer: int, g: torch.Tensor) -> torch.Tensor:
        """`a = Silu(class, g)`: the layer's `Silu` table at each channel's class."""
        arch = self.model.arch
        point = LayerPoint(layer, Site.Silu)
        shifts = self.shifts[point]
        b = (127 * 128 * arch.d_model).bit_length() + 1
        in_exp = dict(self.model.quant.silu_in_exponents)[layer]
        table = _silu_table(b, tuple(shifts), in_exp, self.exponents[point])
        classes = self.classes.get((WeightType.Gate, layer))
        per_channel = (
            np.zeros(arch.d_ff, dtype=np.int64) if classes is None else classes.astype(np.int64)
        )
        keys = g.numpy()
        a = np.empty_like(keys)
        for c in range(len(shifts)):
            channels = per_channel == c
            a[:, channels] = table.lookup(keys[:, channels], c)
        y = torch.as_tensor(a)
        if self.record:
            self.sites[point] = y
        return y

    def residual(
        self, res: torch.Tensor, acc: torch.Tensor, point: LayerPoint, ty: WeightType
    ) -> torch.Tensor:
        """`res <- clamp(res + floor((acc + 2^(r-1)) / 2^r), INT20)`, `r` the channel's shift.

        A site sums in two steps (RFC-0022) unless it has one shift of at most 62 in every layer
        (RFC-0038): its update clamped within `2^b - 1` first, which changes no sum. An affine
        site rescales `acc_c * M_c` (RFC-0025).
        """
        key, shift = self.key(acc, point, ty, point.layer)
        half = torch.bitwise_left_shift(torch.ones_like(shift), shift - 1)
        update = torch.bitwise_right_shift(key + half, shift)
        if self.record and point.site in self.two_steps:
            wide = 2 * self.half - 1
            self.updates[point] = torch.clamp(update, -wide, wide)
        out = torch.clamp(res + update, -self.half, self.half - 1)
        if self.record:
            self.sites[point] = out
        return out

    # -- the model ----------------------------------------------------------------------------

    def run(self) -> Trace:
        model, arch = self.model, self.model.arch
        servable = model.profile is Profile.Servable
        shift = self.exponents[EmbedPoint()] - model.quant.public.residual_exponent
        if not 0 <= shift <= 23:
            raise TesseraQuantError("the embedding's shift into the residual lies in [0, 23]")
        # x0 is column `token` of the transposed embedding: res = x0 * 2^s_e.
        embed = self.weight(WeightType.Embed, 0)
        res = embed[:, self.tokens].T << shift
        del embed
        residuals = [res] if self.record else []
        for layer in range(arch.n_layers):
            y = self.norm(res, LayerPoint(layer, Site.AttnNorm))
            q = self.rope(self.matmul(y, WeightType.Q, layer), arch.n_heads)
            q = self.requant(q, LayerPoint(layer, Site.RopeQ), WeightType.Q, layer)
            k = self.rope(self.matmul(y, WeightType.K, layer), arch.n_kv_heads)
            k = self.requant(k, LayerPoint(layer, Site.RopeK), WeightType.K, layer)
            v = self.matmul(y, WeightType.V, layer)
            v = self.requant(v, LayerPoint(layer, Site.V), WeightType.V, layer)
            o = self.attention(layer, q, k, v)
            if servable:
                o = self.requant(o, LayerPoint(layer, Site.AttnOut), None, layer)
            acc = self.matmul(o, WeightType.O, layer)
            res = self.residual(res, acc, LayerPoint(layer, Site.AttnResidual), WeightType.O)
            y2 = self.norm(res, LayerPoint(layer, Site.MlpNorm))
            a = self.silu(layer, self.matmul(y2, WeightType.Gate, layer))
            u = self.matmul(y2, WeightType.Up, layer)
            if servable:
                u = self.requant(u, LayerPoint(layer, Site.Up), WeightType.Up, layer)
            _bounded(_largest(a) * _largest(u), "the MLP's product")
            h = a * u
            if servable:
                h = self.requant(h, LayerPoint(layer, Site.Product), None, layer)
            acc = self.matmul(h, WeightType.Down, layer)
            res = self.residual(res, acc, LayerPoint(layer, Site.MlpResidual), WeightType.Down)
            if self.record:
                residuals.append(res)
        y = self.norm(res, FinalNormPoint())
        logits = self.matmul(y, WeightType.LmHead, 0)
        return Trace(
            residuals,
            self.sites,
            self.updates,
            self.roots,
            self.maxima,
            self.exponentials,
            self.reciprocals,
            logits,
        )


def _largest(x: torch.Tensor) -> int:
    """The largest magnitude in `x`, as a Python integer: 0 for an empty tensor."""
    return int(x.abs().max()) if x.numel() else 0


def _bounded(bound: int, what: str) -> None:
    """Refuses a computation whose values could reach `2^63`."""
    if bound >= _LIMIT:
        raise TesseraQuantError(f"{what} could leave int64")


# Layers of one definition share their table, and a model's tables serve every sequence: a few
# are kept, at most 2^24 values each.
@lru_cache(maxsize=4)
def _exp_table(key_bits: int, sigma: int, scale: Fraction) -> LookupTable:
    return lookup_table(0, Exp(key_bits, sigma, scale))


@lru_cache(maxsize=4)
def _silu_table(b: int, shifts: tuple[int, ...], in_exp: int, out_exp: int) -> LookupTable:
    return lookup_table(0, Silu(b, list(shifts), in_exp, out_exp))


def _lookup(table: LookupTable, keys: torch.Tensor) -> torch.Tensor:
    """`table`'s values at `keys`, through its NumPy generator."""
    return torch.as_tensor(table.lookup(keys.numpy()))
