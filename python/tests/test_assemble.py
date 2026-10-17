# SPDX-License-Identifier: Apache-2.0 OR MIT
"""`D` from a checkpoint and its calibration (RFC-0005 'Pipeline', steps 4 to 6; 02 §4).

A tiny Llama-3.1 at Llama's magnitudes is calibrated, quantized and exported.

- **The strength.** Each candidate `alpha` is quantized alone. Its divergence from the BF16
  reference on the held-out split is computed here, `D`'s logits scaled by the LM head's
  exponent as RFC-0005's rule gives it, independently of `quantize`. `quantize` keeps the least
  divergent, the smallest of equal ones, and gives that candidate's `D` exactly.
- **Provenance.** Every field is what it names: each checkpoint file's BLAKE3, the calibration's
  identity, the statistics digest, the strength chosen and the library versions.
- **Determinism.** Two runs give the same bytes. The export is a secret file whose last 32 bytes
  are the digest returned, and reads back to the same `D`.
- **Exactness.** No float reaches `D`: every field is an integer, a rational, text or bytes.
- **Refusals.** An empty held-out split, `affine` scales until #301, and an unknown profile.
"""

import math
import stat
from dataclasses import fields, is_dataclass, replace
from enum import Enum
from fractions import Fraction
from pathlib import Path
from typing import Any, cast

import blake3
import numpy as np
import pytest
import torch
from checkpoints import TINY, TINY_ARCH, llama_like, write

from tessera_quant import (
    CalibrationError,
    CalibrationResult,
    TesseraQuantError,
    calibrate,
    export,
    import_checkpoint,
    quantize,
)
from tessera_quant.calibration import DatasetIdentity, draw
from tessera_quant.capture import reference_logits
from tessera_quant.checkpoint import FloatModel
from tessera_quant.integer_model import INTEGER_MODEL, FinalNormPoint, IntegerModel, Profile
from tessera_quant.migration import ALPHAS, fold, mean_kl
from tessera_quant.reference import trace
from tessera_quant.scales import exponent
from tessera_quant.tce import INTEGER_MODEL_LIMITS, encode, read_artifact

SOURCE = DatasetIdentity(
    name="fixture", config="en", split="validation", revision="0" * 40, files=("fixture",)
)
LIBRARIES = {
    "blake3",
    "datasets",
    "numpy",
    "python",
    "safetensors",
    "tokenizers",
    "torch",
    "transformers",
}


@pytest.fixture(scope="module")
def calibrated(tmp_path_factory: pytest.TempPathFactory) -> tuple[FloatModel, CalibrationResult]:
    path = write(tmp_path_factory.mktemp("assemble") / "checkpoint", TINY, llama_like(5))
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
        held_out=3,
    )
    return model, calibrate(model, calibration)


def divergence(model: FloatModel, calib: CalibrationResult, d: IntegerModel) -> float:
    """`D`'s mean KL divergence from the BF16 reference on the held-out split.

    `D`'s logits are `y * W_lm` at `2^(e_y + e_lm)`: the final norm's exponent, and the LM head's,
    the least `e` with its folded maximum within `127 * 2^e` (RFC-0005 'Scale policy'). The fold
    does not depend on the strength: no migration reaches the LM head.
    """
    held_out = calib.calibration.held_out
    reference = reference_logits(model, held_out, calib.calibration.seed)
    weights = {name: model.tensor(name).to(torch.float64) for name in model.names}
    head = fold(TINY_ARCH, weights)["lm_head.weight"]
    e_lm = exponent(float(head.abs().max()), 127)
    e_y = dict(d.quant.public.act_exponents)[FinalNormPoint()]
    found = torch.cat([trace(d, np.array(w.tokens, dtype=np.uint32)).logits for w in held_out])
    return mean_kl(reference, found.to(torch.float64) * 2.0 ** (e_y + e_lm))


def test_quantize_keeps_the_least_divergent_strength(
    calibrated: tuple[FloatModel, CalibrationResult],
) -> None:
    model, calib = calibrated
    alone = {alpha: quantize(model, calib, alphas=(alpha,)) for alpha in ALPHAS}
    scores = {alpha: divergence(model, calib, d) for alpha, d in alone.items()}
    assert all(math.isfinite(s) for s in scores.values())
    best = min(sorted(scores), key=lambda alpha: scores[alpha])
    chosen = quantize(model, calib)
    assert chosen.provenance.alpha == best
    assert encode(INTEGER_MODEL.codec, chosen) == encode(INTEGER_MODEL.codec, alone[best])
    # The strengths give different models, so the choice is not vacuous.
    assert len({encode(INTEGER_MODEL.codec, d) for d in alone.values()}) > 1


def test_the_provenance_names_what_d_comes_from(
    calibrated: tuple[FloatModel, CalibrationResult],
) -> None:
    model, calib = calibrated
    d = quantize(model, calib, alphas=(Fraction(3, 5),))
    p = d.provenance
    files = sorted(set(model.files.values()), key=lambda path: path.name)
    assert p.checkpoint == [(f.name, blake3.blake3(f.read_bytes()).digest()) for f in files]
    calibration = calib.calibration
    assert (p.calibration.dataset, p.calibration.config, p.calibration.split) == (
        "fixture",
        "en",
        "validation",
    )
    assert p.calibration.revision == "0" * 40
    assert p.calibration.files == ["fixture"]
    assert p.calibration.seed == 11
    assert (p.calibration.sequences, p.calibration.held_out) == (6, 3)
    assert p.calibration.tokens_per_sequence == 16
    assert p.calibration.selection == calibration.selection_digest
    assert p.statistics == calib.digest
    assert p.alpha == Fraction(3, 5)
    assert set(p.libraries) == LIBRARIES
    assert all(p.libraries.values())
    # The profile asked for is `D`'s.
    assert d.profile is Profile.Servable
    minimal = quantize(model, calib, profile="minimal", alphas=(Fraction(3, 5),))
    assert minimal.profile is Profile.Minimal


def floats(value: object) -> list[str]:
    """The paths of every float, or float array, inside `value`."""
    if isinstance(value, float):
        return ["."]
    if isinstance(value, np.ndarray):
        kind = cast(np.ndarray[Any, np.dtype[Any]], value).dtype
        return [] if np.issubdtype(kind, np.integer) else ["."]
    if isinstance(value, (bool, int, str, bytes, Fraction, Enum)):
        return []
    if isinstance(value, dict):
        entries = cast(dict[object, object], value).items()
        return [f"[{k}]{p}" for k, v in entries for p in floats(k) + floats(v)]
    if isinstance(value, (list, tuple)):
        items = cast(list[object], list(cast(Any, value)))
        return [f"[{i}]{p}" for i, v in enumerate(items) for p in floats(v)]
    if is_dataclass(value):
        return [f".{f.name}{p}" for f in fields(value) for p in floats(getattr(value, f.name))]
    return [f": {type(value).__name__}"]


def test_d_is_reproducible_exact_and_exported_as_a_secret(
    calibrated: tuple[FloatModel, CalibrationResult], tmp_path: Path
) -> None:
    model, calib = calibrated
    first, second = quantize(model, calib), quantize(model, calib)
    assert encode(INTEGER_MODEL.codec, first) == encode(INTEGER_MODEL.codec, second)
    assert floats(first) == []
    path = tmp_path / "model.d"
    digest = export(first, path)
    data = path.read_bytes()
    assert digest == data[-32:]
    assert stat.S_IMODE(path.stat().st_mode) == 0o600
    again = read_artifact(path, INTEGER_MODEL, limits=INTEGER_MODEL_LIMITS)
    assert encode(INTEGER_MODEL.codec, again) == encode(INTEGER_MODEL.codec, first)
    assert export(second, tmp_path / "again.d") == digest


def test_what_quantize_cannot_do_is_refused(
    calibrated: tuple[FloatModel, CalibrationResult],
) -> None:
    model, calib = calibrated
    with pytest.raises(TesseraQuantError, match="#301"):
        quantize(model, calib, scales="affine")
    with pytest.raises(TesseraQuantError, match="no profile"):
        quantize(model, calib, profile="tiny")  # pyright: ignore[reportArgumentType]
    empty = CalibrationResult(replace(calib.calibration, held_out=()), calib.statistics, calib.data)
    with pytest.raises(CalibrationError, match="held-out split"):
        quantize(model, empty)
