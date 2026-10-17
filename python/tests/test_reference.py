# SPDX-License-Identifier: Apache-2.0 OR MIT
"""The integer reference for `D` (RFC-0005 'Integer semantics') against independent values.

- **`rq`.** Ties round half up, negative ones too; values saturate; shifts of 1, 2, 7 and 40, and
  one shift per channel, agree with Python's integers.
- **The conformance models (SC-7).** On every sequence of `spec/vectors/models/`, the trace holds
  every tensor of the oracle's `traces.json`, bit for bit: every tensor `D` commits, the
  residual stream before the first layer and the logits. Only the one-hot tokens, the circuit's
  encoding of the prompt, are not the reference's to compute. The oracle, `generate.py`,
  computes `D` in Python's integers and shares no code with this package; the CPU engine is
  held to the same traces (`crates/tessera-e2e/tests/differential.rs`).
- **Located.** A divergence is reported at its first tensor, in the trace's order, and its index:
  one seeded at a drawn index of each tensor in turn is found there, and a channel of `V` moved
  to another class shows up at layer 1's `V`, in that channel.
- **The trace.** Each row of exponentials holds its maximum's `2^sigma` and nothing past the
  diagonal. Two runs are identical, and the trace shows no value.
- **A zero row.** A token whose embedding is zero gives a residual row of zeros, whose root comes
  from the norm's `e` alone (RFC-0022 'Complete norms').
- **Refusals.** A token outside the vocabulary, an empty sequence or one past
  the context, and a model whose sums of squares would leave int64.
"""

import json
import math
from dataclasses import dataclass, replace
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
    QuantPoint,
    Site,
    WeightType,
)
from tessera_quant.reference import Trace, rq, run_reference, trace
from tessera_quant.tce import INTEGER_MODEL_LIMITS, unseal

MODELS = Path(__file__).resolve().parents[2] / "spec" / "vectors" / "models"
INDEX: dict[str, Any] = json.loads((MODELS / "models.json").read_text(encoding="utf-8"))
TRACES: dict[str, Any] = json.loads((MODELS / "traces.json").read_text(encoding="utf-8"))

NOT_COMPUTED = {"one-hot"}
"""The traced tensors the reference does not compute: the circuit's encoding of the prompt."""


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


def traces(entry: dict[str, Any]) -> list[list[dict[str, Any]]]:
    """The oracle's trace of each sequence of the model `entry`: its tensors, in order."""
    (model,) = (m for m in TRACES["models"] if m["name"] == entry["name"])
    return [sequence["tensors"] for sequence in model["sequences"]]


def label(point: QuantPoint) -> str:
    """A site's name in the traces: "layer 0 RopeQ", or "FinalNorm"."""
    if isinstance(point, LayerPoint):
        return f"layer {point.layer} {point.site.name}"
    assert isinstance(point, FinalNormPoint)
    return "FinalNorm"


def named(found: Trace) -> dict[str, torch.Tensor]:
    """Each tensor of `found` under its name in the traces.

    The residual stream after each layer is that layer's `MlpResidual` output.
    """
    for layer, residual in enumerate(found.residuals[1:]):
        assert residual is found.sites[LayerPoint(layer, Site.MlpResidual)]
    tensors = {"embedding": found.residuals[0], "logits": found.logits}
    tensors |= {label(p): t for p, t in found.sites.items()}
    tensors |= {f"{label(p)} update": t for p, t in found.updates.items()}
    tensors |= {f"{label(p)} root": t for p, t in found.roots.items()}
    for layer in found.maxima:
        tensors[f"layer {layer} maxima"] = found.maxima[layer]
        tensors[f"layer {layer} exponentials"] = found.exponentials[layer]
        tensors[f"layer {layer} reciprocals"] = found.reciprocals[layer]
    return tensors


@dataclass(frozen=True)
class Divergence:
    """The first place where the reference's values differ from the oracle's."""

    tensor: str
    index: tuple[int, ...]
    oracle: int
    reference: int


def first_divergence(
    traced: list[dict[str, Any]], found: dict[str, torch.Tensor]
) -> Divergence | None:
    """The first divergence of `found` from the trace, tensor by tensor in the trace's order.

    The reference must compute exactly the traced tensors it computes at all, each of its shape.
    """
    assert {t["name"] for t in traced} - NOT_COMPUTED == set(found)
    for t in traced:
        if t["name"] in NOT_COMPUTED:
            continue
        values = found[t["name"]]
        assert values.dtype == torch.int64
        assert list(values.shape) == t["shape"], t["name"]
        expected = np.array(t["values"], dtype=np.int64)
        flat: npt.NDArray[np.int64] = values.reshape(-1).numpy()
        differ = expected != flat
        if bool(differ.any()):
            # The first difference: argmax takes the first of equal maxima.
            i = int(differ.argmax())
            index = tuple(int(k) for k in np.unravel_index(i, t["shape"]))
            return Divergence(t["name"], index, int(expected[i]), int(flat[i]))
    return None


@pytest.mark.parametrize("entry", INDEX["models"], ids=lambda entry: entry["name"])
def test_the_reference_computes_every_committed_tensor_of_the_oracle(
    entry: dict[str, Any],
) -> None:
    model = load(entry)
    for sequence, traced in zip(entry["sequences"], traces(entry), strict=True):
        found = trace(model, tokens(sequence["tokens"]))
        divergence = first_divergence(traced, named(found))
        assert divergence is None, f"{entry['name']}: {divergence}"
        logits: Any = run_reference(model, tokens(sequence["tokens"])).tolist()
        assert logits == sequence["logits"]
        assert repr(found) == "Trace()"


def test_a_divergence_seeded_in_any_tensor_is_found_where_it_was_seeded() -> None:
    rng = np.random.Generator(np.random.PCG64(91))
    seeded = 0
    for entry in INDEX["models"]:
        sequence, traced = entry["sequences"][1], traces(entry)[1]
        found = named(trace(load(entry), tokens(sequence["tokens"])))
        for k, t in enumerate(traced):
            if t["name"] in NOT_COMPUTED:
                continue
            i = int(rng.integers(len(t["values"])))
            values = list(t["values"])
            values[i] += 1
            wrong = [*traced[:k], {**t, "values": values}, *traced[k + 1 :]]
            index = tuple(int(x) for x in np.unravel_index(i, t["shape"]))
            expected = Divergence(t["name"], index, values[i], t["values"][i])
            assert first_divergence(wrong, found) == expected, entry["name"]
            seeded += 1
    # Every traced tensor but the one-hot tokens.
    assert seeded == (41 - 1) + (35 - 1) + (39 - 1) + (73 - 1)


def test_a_channel_moved_to_another_class_diverges_at_its_site_and_channel() -> None:
    entry = INDEX["models"][0]
    model = load(entry)
    sequence, traced = entry["sequences"][1], traces(entry)[1]
    params = next(c for c in model.quant.channels if c.ty is WeightType.V and c.layer == 1)
    classes = int(params.class_.max()) + 1
    assert classes > 1
    channel = int(np.random.Generator(np.random.PCG64(92)).integers(len(params.class_)))
    moved = params.class_.copy()
    moved[channel] = (moved[channel] + 1) % classes
    channels = [replace(c, class_=moved) if c is params else c for c in model.quant.channels]
    other = replace(model, quant=replace(model.quant, channels=channels))
    divergence = first_divergence(traced, named(trace(other, tokens(sequence["tokens"]))))
    assert divergence is not None
    assert (divergence.tensor, divergence.index[1]) == ("layer 1 V", channel), divergence
    assert divergence.oracle != divergence.reference


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
    public = model.quant.public
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
