# SPDX-License-Identifier: Apache-2.0 OR MIT
"""Fidelity evaluation (RFC-0005 'Fidelity evaluation') against hand-computed values.

- **Token metrics.** On a vocabulary of three, with distributions written out, each position's
  negative log-likelihoods, KL divergence and agreement are the ones computed by hand.
- **Intervals.** The Wilson interval of 9 out of 10 is the textbook one, and its bounds stay in
  `[0, 1]`. A bootstrap over units of one ratio has no width. A bootstrap's percentiles bracket
  its estimate, and the same seed gives the same interval.
- **A tiny Llama.** The report records the seed, the data's identities and digests, the library
  versions and `D`'s provenance. Its rate is its agreeing tokens over its trials, two runs give
  the same report, and its shape is the committed example's, whose schema `cargo xtask lint`
  checks.
- **Hygiene.** An evaluation document that the calibration used, empty data, no continuation and
  a reference of another architecture are refused.
"""

import json
import math
import statistics
from dataclasses import replace
from pathlib import Path
from typing import cast

import numpy as np
import pytest
import torch
from checkpoints import TINY, TINY_ARCH, tensors, write

from tessera_quant import (
    CalibrationError,
    TesseraQuantError,
    calibrate,
    import_checkpoint,
    quantize,
)
from tessera_quant.calibration import DatasetIdentity, document_hash, draw
from tessera_quant.checkpoint import FloatModel
from tessera_quant.fidelity import (
    EvalSet,
    EvalSpec,
    bootstrap_ratio,
    evaluate_fidelity,
    token_totals,
    wilson,
)
from tessera_quant.integer_model import IntegerModel

EXAMPLE = Path(__file__).parents[2] / "docs" / "perf" / "examples" / "fidelity-tiny.json"
SOURCE = DatasetIdentity(
    name="fixture", config="en", split="validation", revision="0" * 40, files=("fixture",)
)


def encode(text: str) -> list[int]:
    return [2 + ord(c) % 60 for c in text]


def test_token_metrics_are_the_hand_computed_ones() -> None:
    # Position 0 predicts token 0: the reference gives it 1/2, D 1/4, and D prefers token 1.
    # Position 1 predicts token 2, which both give 3/5.
    p0, q0 = [0.5, 0.25, 0.25], [0.25, 0.625, 0.125]
    p1 = [0.2, 0.2, 0.6]
    reference = torch.log(torch.tensor([p0, p1, p1], dtype=torch.float64))
    model = torch.log(torch.tensor([q0, p1, p1], dtype=torch.float64))
    totals = token_totals([reference], [model * 1.0], [(5, 0, 2)], [0])
    ln2 = math.log(2)
    assert totals.reference_nll[0] == pytest.approx(ln2 + math.log(5 / 3))
    assert totals.model_nll[0] == pytest.approx(2 * ln2 + math.log(5 / 3))
    # KL(p0 ‖ q0), not KL(q0 ‖ p0), which differs; equal distributions add nothing.
    forward = sum(a * math.log(a / b) for a, b in zip(p0, q0, strict=True))
    backward = sum(b * math.log(b / a) for a, b in zip(p0, q0, strict=True))
    assert totals.kl[0] == pytest.approx(forward)
    assert abs(forward - backward) > 0.01
    assert (int(totals.agreeing[0]), int(totals.tokens[0])) == (1, 2)
    # From position 1 on, the first prediction is left out.
    late = token_totals([reference], [model], [(5, 0, 2)], [2])
    assert (int(late.agreeing[0]), int(late.tokens[0])) == (1, 1)
    assert late.kl[0] == pytest.approx(0.0, abs=1e-15)


def test_intervals_have_their_textbook_values() -> None:
    nine = wilson(9, 10)
    assert nine.estimate == 0.9
    assert (round(nine.low, 4), round(nine.high, 4)) == (0.5958, 0.9821)
    assert wilson(0, 10).low == 0.0
    assert wilson(10, 10).high == 1.0
    with pytest.raises(TesseraQuantError):
        wilson(1, 0)
    rng = np.random.Generator(np.random.PCG64(0))
    tokens = np.array([3.0, 5.0, 2.0])
    flat = bootstrap_ratio(0.5 * tokens, tokens, 100, rng)
    assert (flat.estimate, flat.low, flat.high) == (0.5, 0.5, 0.5)
    values = np.array([0.1, 0.9, 0.4, 0.7, 0.3])
    ones = np.ones(5)
    first = bootstrap_ratio(values, ones, 500, np.random.Generator(np.random.PCG64(7)))
    again = bootstrap_ratio(values, ones, 500, np.random.Generator(np.random.PCG64(7)))
    assert first == again
    assert first.estimate == pytest.approx(0.48)
    # The bounds are the 2.5% and 97.5% percentiles of the resampled means.
    picks = np.random.Generator(np.random.PCG64(7)).integers(0, 5, size=(500, 5))
    means = values[picks].mean(axis=1)
    # Python's own quantiles, interpolated as numpy's linear method: cut points at every 2.5%.
    cuts = statistics.quantiles([float(m) for m in means], n=40, method="inclusive")
    assert (first.low, first.high) == pytest.approx((cuts[0], cuts[-1]))
    assert first.low <= first.estimate <= first.high
    assert first.low > 0.1
    assert first.high < 0.9


@pytest.fixture(scope="module")
def evaluated(
    tmp_path_factory: pytest.TempPathFactory,
) -> tuple[FloatModel, IntegerModel, EvalSpec]:
    generator = torch.Generator().manual_seed(6)
    weights = tensors(TINY_ARCH, 5)
    for name, t in weights.items():
        if t.dim() == 1:
            gain = 1 + 0.1 * torch.randn(t.shape, generator=generator, dtype=torch.float64)
            weights[name] = gain.to(torch.bfloat16)
        elif name != "model.embed_tokens.weight":
            weights[name] = (t.to(torch.float64) * 0.02).to(torch.bfloat16)
    path = write(tmp_path_factory.mktemp("fidelity") / "checkpoint", TINY, weights)
    model = import_checkpoint(path)
    documents = [f"document {i} " + "the quick brown fox jumps" * (1 + i % 3) for i in range(40)]
    calibration = draw(documents, encode, 1, source=SOURCE, seed=11, count=6, length=16, held_out=3)
    d = quantize(model, calibrate(model, calibration))
    texts = [f"{i} lorem ipsum dolor sit amet, {i * 7} consectetur" for i in range(10)]
    sequences = tuple((1, *encode(t)[:24]) for t in texts)
    hashes = frozenset(document_hash(t) for t in texts)
    spec = EvalSpec(
        documents=EvalSet(SOURCE, sequences, hashes),
        prompts=EvalSet(SOURCE, tuple(s[:8] for s in sequences), hashes),
        continuation=6,
        calibration_hashes=calibration.document_hashes,
        resamples=300,
    )
    return model, d, spec


def shape(value: object) -> object:
    """The keys of `value`, all the way down: what two reports with other numbers share."""
    if isinstance(value, dict):
        return {k: shape(v) for k, v in cast(dict[str, object], value).items()}
    return type(value).__name__


def test_a_tiny_llamas_report_records_what_it_measured(
    evaluated: tuple[FloatModel, IntegerModel, EvalSpec],
) -> None:
    model, d, spec = evaluated
    report = evaluate_fidelity(d, model, spec, seed=3)
    found = report.as_json()
    assert found == evaluate_fidelity(d, model, spec, seed=3).as_json()
    assert found["seed"] == 3
    assert found["resamples"] == 300
    p = d.provenance
    assert found["model"]["statistics_digest"] == p.statistics.hex()
    assert found["model"]["alpha"] == f"{p.alpha.numerator}/{p.alpha.denominator}"
    perplexity, agreement = found["perplexity"], found["agreement"]
    assert perplexity["digest"] == spec.documents.digest.hex()
    assert perplexity["tokens"] == sum(len(s) - 1 for s in spec.documents.sequences)
    assert agreement["tokens"] == report.trials == 10 * 6
    assert agreement["top1"]["estimate"] == report.agreeing / report.trials
    for interval in (perplexity["ratio"], agreement["top1"], agreement["kl"]):
        assert interval["low"] <= interval["estimate"] <= interval["high"]
    ratio = perplexity["model"] / perplexity["reference"]
    assert perplexity["ratio"]["estimate"] == pytest.approx(ratio)
    # D follows its reference closely, as it must at its logits' real scale.
    assert 0.98 < ratio < 1.02
    assert agreement["top1"]["estimate"] >= 0.9
    assert agreement["kl"]["estimate"] < 1e-3
    assert set(found["environment"]["libraries"]) == {
        "blake3",
        "numpy",
        "safetensors",
        "torch",
        "transformers",
    }
    # The committed example has this report's shape; its numbers depend on the platform.
    example = json.loads(EXAMPLE.read_text(encoding="utf-8"))
    assert shape(example) == shape(found)


def test_evaluation_data_is_disjoint_from_the_calibration(
    evaluated: tuple[FloatModel, IntegerModel, EvalSpec],
) -> None:
    model, d, spec = evaluated
    shared = replace(
        spec, calibration_hashes=spec.calibration_hashes | spec.prompts.document_hashes
    )
    with pytest.raises(CalibrationError, match="share a document"):
        evaluate_fidelity(d, model, shared)
    empty = replace(spec, documents=replace(spec.documents, sequences=()))
    with pytest.raises(CalibrationError, match="empty"):
        evaluate_fidelity(d, model, empty)
    with pytest.raises(CalibrationError, match="at least one"):
        evaluate_fidelity(d, model, replace(spec, continuation=0))
    other = replace(d, arch=replace(d.arch, max_context=d.arch.max_context + 1))
    with pytest.raises(TesseraQuantError, match="another architecture"):
        evaluate_fidelity(other, model, spec)
