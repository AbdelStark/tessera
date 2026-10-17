# SPDX-License-Identifier: Apache-2.0 OR MIT
"""The integer reference for `D` (RFC-0005 'Integer semantics') against independent values.

- **`rq`.** Ties round half up, negative ones too; values saturate; shifts of 1, 2, 7 and 40, and
  one shift per channel, agree with Python's integers.
- **The conformance models.** On every sequence of `spec/vectors/models/`, the reference gives
  the oracle's residual at every layer boundary and its logits, bit for bit. The oracle,
  `generate.py`, computes `D` in Python's integers and shares no code with this package.
- **The trace.** It holds every site's output, every norm's root and each layer's maxima,
  exponentials and reciprocals. Each row of exponentials holds its maximum's `2^sigma` and
  nothing past the diagonal. Two runs are identical, and the trace shows no value.
- **A zero row.** A token whose embedding is zero gives a residual row of zeros, whose root comes
  from the norm's `e` alone (RFC-0022 'Complete norms').
- **Refusals.** An affine site, a token outside the vocabulary, an empty sequence or one past
  the context, and a model whose sums of squares would leave int64.
"""

import json
import math
from dataclasses import replace
from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt
import pytest
import torch

from tessera_quant.errors import TesseraQuantError
from tessera_quant.integer_model import (
    INTEGER_MODEL,
    EmbedPoint,
    FinalNormPoint,
    IntegerModel,
    LayerPoint,
    Profile,
    Site,
    WeightType,
)
from tessera_quant.reference import rq, run_reference, trace
from tessera_quant.tce import INTEGER_MODEL_LIMITS, unseal

MODELS = Path(__file__).resolve().parents[2] / "spec" / "vectors" / "models"
INDEX: dict[str, Any] = json.loads((MODELS / "models.json").read_text(encoding="utf-8"))

SITES = {
    Profile.Servable: list(Site),
    Profile.Minimal: [
        Site.AttnNorm,
        Site.RopeQ,
        Site.RopeK,
        Site.V,
        Site.AttnResidual,
        Site.MlpNorm,
        Site.Silu,
        Site.MlpResidual,
    ],
}


def load(entry: dict[str, Any]) -> IntegerModel:
    data = (MODELS / entry["file"]).read_bytes()
    return unseal(data, INTEGER_MODEL, limits=INTEGER_MODEL_LIMITS)


def tokens(values: list[int]) -> npt.NDArray[np.uint32]:
    return np.array(values, dtype=np.uint32)


def listed(x: torch.Tensor) -> Any:
    """`x`'s values as nested lists of Python integers."""
    return x.numpy().tolist()


def exact_rq(x: int, r: int, lo: int, hi: int) -> int:
    """RFC-0005's `rq` in Python's integers, whose `>>` floors."""
    return min(max((x + (1 << (r - 1))) >> r, lo), hi)


def test_rq_rounds_half_up_and_saturates() -> None:
    # Ties: 2/4 rounds up to 1, -2/4 up to 0, 6/4 to 2 and -6/4 to -1.
    assert listed(rq(torch.tensor([2, -2, 6, -6]), 2, -128, 127)) == [1, 0, 2, -1]
    # Saturation at both ends.
    assert listed(rq(torch.tensor([1024, -1024, 510, -514]), 2, -128, 127)) == [
        127,
        -128,
        127,
        -128,
    ]
    for r in (1, 2, 7, 40):
        unit, half = 1 << r, 1 << (r - 1)
        values = [0, half - 1, half, half + 1, -half - 1, -half, -half + 1, 3 * half, -3 * half]
        values += [k * unit + half for k in (-129, -128, 127, 128)]
        values += [(1 << 61) - 1, -(1 << 61)]
        for lo, hi in ((-128, 127), (-(1 << 19), (1 << 19) - 1)):
            found = listed(rq(torch.tensor(values), r, lo, hi))
            assert found == [exact_rq(x, r, lo, hi) for x in values], (r, lo, hi)
    # One shift per channel, broadcast along the last axis.
    x = torch.tensor([[6, 6, 6], [-6, -6, -6]])
    shifts = torch.tensor([1, 2, 3])
    assert listed(rq(x, shifts, -128, 127)) == [
        [exact_rq(v, r, -128, 127) for v, r in zip(row, (1, 2, 3), strict=True)]
        for row in ((6, 6, 6), (-6, -6, -6))
    ]


@pytest.mark.parametrize("entry", INDEX["models"], ids=lambda entry: entry["name"])
def test_the_conformance_models_give_the_oracles_values(entry: dict[str, Any]) -> None:
    model = load(entry)
    for sequence in entry["sequences"]:
        found = trace(model, tokens(sequence["tokens"]))
        assert len(found.residuals) == model.arch.n_layers + 1
        for layer, (residual, expected) in enumerate(
            zip(found.residuals, sequence["residuals"], strict=True)
        ):
            assert listed(residual) == expected, (entry["name"], layer)
        assert listed(found.logits) == sequence["logits"], entry["name"]
        logits: Any = run_reference(model, tokens(sequence["tokens"])).tolist()
        assert logits == sequence["logits"]


@pytest.mark.parametrize("entry", INDEX["models"][:2], ids=lambda entry: entry["name"])
def test_the_trace_holds_every_committed_tensor_and_repeats(entry: dict[str, Any]) -> None:
    model = load(entry)
    arch, sigma = model.arch, model.quant.public.exp_sigma
    sequence = tokens(entry["sequences"][0]["tokens"])
    first, second = trace(model, sequence), trace(model, sequence)
    n = len(sequence)
    points = [
        LayerPoint(layer, site) for layer in range(arch.n_layers) for site in SITES[model.profile]
    ]
    assert set(first.sites) == {*points, FinalNormPoint()}
    norms = {Site.AttnNorm, Site.MlpNorm}
    assert set(first.roots) == {p for p in points if p.site in norms} | {FinalNormPoint()}
    layers = set(range(arch.n_layers))
    assert set(first.maxima) == set(first.exponentials) == set(first.reciprocals) == layers
    for e in first.exponentials.values():
        assert e.shape == (arch.n_heads, n, n)
        # The max certificate: each row's maximum has key 0, so e = Exp[0] = 2^sigma there, and
        # the causal mask leaves nothing past the diagonal.
        assert bool((e.amax(dim=2) == 1 << sigma).all())
        assert not bool(e.triu(diagonal=1).any())
    tensors = [
        *zip(first.residuals, second.residuals, strict=True),
        *((first.sites[p], second.sites[p]) for p in first.sites),
        *((first.roots[p], second.roots[p]) for p in first.roots),
        *((first.maxima[k], second.maxima[k]) for k in first.maxima),
        *((first.exponentials[k], second.exponentials[k]) for k in first.exponentials),
        *((first.reciprocals[k], second.reciprocals[k]) for k in first.reciprocals),
        (first.logits, second.logits),
    ]
    assert all(a.dtype == torch.int64 and torch.equal(a, b) for a, b in tensors)
    assert repr(first) == "Trace()"


def test_a_zero_row_takes_its_root_from_epsilon_alone() -> None:
    model = load(INDEX["models"][0])
    arch, public = model.arch, model.quant.public
    embed = next(w for w in model.weights if w.ty is WeightType.Embed)
    data = embed.data.copy().reshape(embed.rows, embed.cols)
    data[:, 7] = 0
    zero = replace(embed, data=data.reshape(-1))
    weights = [zero if w is embed else w for w in model.weights]
    found = trace(replace(model, weights=weights), tokens([7, 3]))
    root = math.isqrt((arch.d_model << (2 * public.rsqrt_kappa)) // public.rms_eps_int)
    assert found.residuals[0][0].abs().max() == 0
    assert int(found.roots[LayerPoint(0, Site.AttnNorm)][0]) == root
    assert found.sites[LayerPoint(0, Site.AttnNorm)][0].abs().max() == 0


def test_what_the_reference_cannot_compute_is_refused() -> None:
    entry = INDEX["models"][0]
    model = load(entry)
    sequence = tokens(entry["sequences"][0]["tokens"])
    public = model.quant.public
    point = LayerPoint(0, Site.V)
    affine = replace(
        public,
        shift_class_sets=[item for item in public.shift_class_sets if item[0] != point],
        affine_sites=[(point, 9)],
    )
    with pytest.raises(TesseraQuantError, match="affine site"):
        trace(replace(model, quant=replace(model.quant, public=affine)), sequence)
    with pytest.raises(TesseraQuantError, match=r"\[0, vocab\)"):
        trace(model, tokens([0, model.arch.vocab]))
    with pytest.raises(TesseraQuantError, match="1 to max_context"):
        trace(model, tokens([]))
    with pytest.raises(TesseraQuantError, match="1 to max_context"):
        trace(model, tokens([0] * (model.arch.max_context + 1)))
    # A 31-bit residual with the embedding shifted 23 bits: the largest embedding entry's square,
    # summed over 64 channels, passes 2^63.
    embed = next(w for w in model.weights if w.ty is WeightType.Embed)
    column = embed.data.astype(np.int64).reshape(embed.rows, embed.cols)
    widest = int(np.abs(column).max(axis=0).argmax())
    assert (int(np.abs(column[:, widest]).max()) << 23) ** 2 * model.arch.d_model >= 1 << 63
    exponent = dict(public.act_exponents)[EmbedPoint()]
    wide = replace(public, residual_bits=31, residual_exponent=exponent - 23)
    with pytest.raises(TesseraQuantError, match="could leave int64"):
        trace(replace(model, quant=replace(model.quant, public=wide)), tokens([widest]))
