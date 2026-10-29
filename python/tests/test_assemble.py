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
- **Affine scales.** Every weight that can take them does, in both profiles: its site in
  `affine_sites`, its multipliers in place of classes. The reference runs the result.
- **Refusals.** An empty held-out split, and an unknown profile.
"""

import gc
import math
import stat
import weakref
from collections.abc import Generator, Mapping
from contextlib import closing
from dataclasses import fields, is_dataclass, replace
from enum import Enum
from fractions import Fraction
from pathlib import Path
from typing import Any, Literal, cast

import blake3
import numpy as np
import pytest
import torch
from checkpoints import TINY, TINY_ARCH, llama_like, write

from tessera_quant import (
    CalibrationError,
    CalibrationResult,
    TesseraQuantError,
    assemble,
    calibrate,
    export,
    import_checkpoint,
    quantize,
)
from tessera_quant._streaming import transformed_checkpoint
from tessera_quant.calibration import DatasetIdentity, draw
from tessera_quant.capture import reference_logits
from tessera_quant.checkpoint import FloatModel
from tessera_quant.integer_model import (
    INTEGER_MODEL,
    FinalNormPoint,
    IntegerModel,
    LayerPoint,
    Profile,
    Site,
)
from tessera_quant.migration import ALPHAS, fold, kl_rows, mean_kl, transform
from tessera_quant.reference import run_reference, trace
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


@pytest.mark.parametrize(
    ("profile", "scales"),
    [("servable", "pot"), ("servable", "affine"), ("minimal", "pot"), ("minimal", "affine")],
)
def test_quantize_keeps_the_least_divergent_strength(
    calibrated: tuple[FloatModel, CalibrationResult],
    profile: Literal["servable", "minimal"],
    scales: Literal["pot", "affine"],
) -> None:
    model, calib = calibrated
    alone = {
        alpha: quantize(model, calib, profile=profile, scales=scales, alphas=(alpha,))
        for alpha in ALPHAS
    }
    scores = {alpha: divergence(model, calib, d) for alpha, d in alone.items()}
    assert all(math.isfinite(s) for s in scores.values())
    best = min(sorted(scores), key=lambda alpha: scores[alpha])
    chosen = quantize(model, calib, profile=profile, scales=scales)
    assert chosen.provenance.alpha == best
    assert encode(INTEGER_MODEL.codec, chosen) == encode(INTEGER_MODEL.codec, alone[best])
    # The strengths give different models, so the choice is not vacuous.
    assert len({encode(INTEGER_MODEL.codec, d) for d in alone.values()}) > 1


def test_streamed_kl_preserves_ragged_position_weighting_and_final_reduction() -> None:
    generator = torch.Generator().manual_seed(611)
    references = [torch.randn(n, 65, generator=generator, dtype=torch.float64) for n in [1, 17, 33]]
    logits = [torch.randn(n, 65, generator=generator, dtype=torch.float64) for n in [1, 17, 33]]
    expected = mean_kl(torch.cat(references), torch.cat(logits))
    rows = [kl_rows(r, q) for r, q in zip(references, logits, strict=True)]
    assert float(torch.cat(rows).mean()) == expected
    assert float(torch.tensor([float(row.mean()) for row in rows]).mean()) != expected


def test_quantization_does_not_accumulate_reference_windows(
    calibrated: tuple[FloatModel, CalibrationResult], monkeypatch: pytest.MonkeyPatch
) -> None:
    model, calib = calibrated
    original = assemble.iter_reference_logits
    live: list[weakref.ReferenceType[torch.Tensor]] = []

    def references(*args: Any, **kwargs: Any) -> Generator[torch.Tensor, None, None]:
        with closing(original(*args, **kwargs)) as source:
            for row in source:
                live.append(weakref.ref(row))
                assert sum(ref() is not None for ref in live) <= 2, "held-out windows stayed live"
                yield row

    monkeypatch.setattr(assemble, "iter_reference_logits", references)
    quantize(model, calib, alphas=(ALPHAS[0],))
    assert len(live) == len(calib.calibration.held_out)
    assert not any(ref() is not None for ref in live)


@pytest.mark.parametrize("difference", [-1, 1])
def test_a_reference_with_missing_or_extra_windows_is_refused(
    calibrated: tuple[FloatModel, CalibrationResult],
    monkeypatch: pytest.MonkeyPatch,
    difference: int,
) -> None:
    model, calib = calibrated
    original = assemble.iter_reference_logits
    closed: list[bool] = []

    def references(*args: Any, **kwargs: Any) -> Generator[torch.Tensor, None, None]:
        with closing(original(*args, **kwargs)) as source:
            try:
                if difference < 0:
                    for _ in range(len(calib.calibration.held_out) - 1):
                        yield next(source)
                else:
                    yield from source
                    yield torch.zeros(1, model.arch.vocab, dtype=torch.float64)
            finally:
                closed.append(True)

    monkeypatch.setattr(assemble, "iter_reference_logits", references)
    with pytest.raises(TesseraQuantError, match="one row block per window") as refused:
        quantize(model, calib, alphas=(ALPHAS[0],))
    assert refused.value.code == "ERR-INTERNAL"
    assert closed == [True]


def test_ragged_windows_choose_alpha_by_positions_instead_of_window_means(
    calibrated: tuple[FloatModel, CalibrationResult], monkeypatch: pytest.MonkeyPatch
) -> None:
    model, calib = calibrated
    windows = tuple(
        replace(window, tokens=(window.tokens[0],) * n)
        for window, n in zip(calib.calibration.held_out, [1, 17, 33], strict=True)
    )
    calib = replace(calib, calibration=replace(calib.calibration, held_out=windows))
    candidate = -1
    position = 0

    def references(*args: Any, **kwargs: Any) -> Generator[torch.Tensor, None, None]:
        nonlocal candidate, position
        candidate += 1
        position = 0
        for window in windows:
            yield torch.zeros(len(window.tokens), model.arch.vocab, dtype=torch.float64)

    def found(d: IntegerModel, tokens: Any) -> Any:
        return np.zeros((len(tokens), d.arch.vocab), dtype=np.int64)

    def divergence(reference: torch.Tensor, logits: torch.Tensor) -> torch.Tensor:
        nonlocal position
        value = ([10.0, 0.0, 0.0], [0.0, 1.0, 1.0])[candidate][position]
        position += 1
        return torch.full((reference.shape[0],), value, dtype=torch.float64)

    monkeypatch.setattr(assemble, "iter_reference_logits", references)
    monkeypatch.setattr(assemble, "run_reference", found)
    monkeypatch.setattr(assemble, "kl_rows", divergence)
    chosen = quantize(model, calib, alphas=ALPHAS[:2])
    assert chosen.provenance.alpha == ALPHAS[0]


def test_integer_logits_are_widened_before_real_scale_multiplication(
    calibrated: tuple[FloatModel, CalibrationResult], monkeypatch: pytest.MonkeyPatch
) -> None:
    model, calib = calibrated
    d = quantize(model, calib, alphas=(ALPHAS[0],))
    expected = ((1 << 25) + 1) * 2.0**d.quant.public.logits_exponent
    original_reference = assemble.run_reference
    original_kl = assemble.kl_rows

    def found(*args: Any, **kwargs: Any) -> Any:
        values = original_reference(*args, **kwargs)
        # Odd integers above float32's exact range must survive the real-scale operation.
        return values * 0 + (1 << 25) + 1

    def checked(reference: torch.Tensor, logits: torch.Tensor) -> torch.Tensor:
        assert logits.dtype == torch.float64
        assert bool((logits == expected).all()), "integer logit bits were lost before scaling"
        return original_kl(reference, logits)

    monkeypatch.setattr(assemble, "run_reference", found)
    monkeypatch.setattr(assemble, "kl_rows", checked)
    quantize(model, calib, alphas=(ALPHAS[0],))


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
    with pytest.raises(TesseraQuantError, match="no profile"):
        quantize(model, calib, profile="tiny")  # pyright: ignore[reportArgumentType]
    empty = CalibrationResult(replace(calib.calibration, held_out=()), calib.statistics, calib.data)
    with pytest.raises(CalibrationError, match="held-out split"):
        quantize(model, empty)


def test_affine_scales_give_every_weight_that_can_take_them_a_multiplier(
    calibrated: tuple[FloatModel, CalibrationResult],
) -> None:
    model, calib = calibrated
    every = [Site.RopeQ, Site.RopeK, Site.V, Site.AttnResidual, Site.Up, Site.MlpResidual]
    # minimal's O and down keys would pass 62 bits, and its up projection has no site.
    cases: list[tuple[Literal["servable", "minimal"], list[Site]]] = [
        ("servable", every),
        ("minimal", every[:3]),
    ]
    for profile, sites in cases:
        d = quantize(model, calib, profile=profile, scales="affine", alphas=(Fraction(3, 5),))
        public = d.quant.public
        points = [p for p, _ in public.affine_sites]
        assert points == [LayerPoint(layer, s) for layer in range(2) for s in sites], profile
        assert not {p for p, _ in public.shift_class_sets} & set(points)
        with_multipliers = [c for c in d.quant.channels if len(c.multiplier)]
        assert len(with_multipliers) == len(points)
        assert all(len(c.class_) == 0 for c in with_multipliers)
        logits = run_reference(d, np.array([1, 7, 9], dtype=np.uint32))
        assert logits.shape == (3, TINY_ARCH.vocab)
        assert encode(INTEGER_MODEL.codec, d) == encode(
            INTEGER_MODEL.codec,
            quantize(model, calib, profile=profile, scales="affine", alphas=(Fraction(3, 5),)),
        )


@pytest.mark.parametrize("alpha", ALPHAS)
def test_lazy_transforms_equal_eager_even_when_accesses_cross_groups(
    calibrated: tuple[FloatModel, CalibrationResult], alpha: Fraction
) -> None:
    model, calib = calibrated
    source = {name: model.tensor(name).to(torch.float64) for name in model.names}
    expected, expected_maxima = transform(model.arch, source, calib.statistics.maxima, alpha)
    weights, maxima = transformed_checkpoint(model, calib.statistics.maxima, alpha)
    assert tuple(weights) == tuple(expected)
    assert len(weights) == len(expected)
    assert tuple(maxima) == tuple(expected_maxima)
    assert len(maxima) == len(expected_maxima)
    for name in reversed(expected):
        assert torch.equal(weights[name], expected[name]), name
    for name in reversed(expected_maxima):
        assert torch.equal(maxima[name], expected_maxima[name]), name
    # A weights access after statistics has changed the cache still reads the right group.
    for name in expected:
        assert torch.equal(weights[name], expected[name]), name
    for mapping in (weights, maxima):
        with pytest.raises(KeyError):
            _ = mapping["not-a-tensor"]


def test_borrowed_groups_are_released_before_the_next_group_is_read(
    calibrated: tuple[FloatModel, CalibrationResult], monkeypatch: pytest.MonkeyPatch
) -> None:
    model, calib = calibrated
    weights, maxima = transformed_checkpoint(model, calib.statistics.maxima, Fraction(3, 5))
    previous = [
        weakref.ref(weights["model.layers.0.self_attn.q_proj.weight"]),
        weakref.ref(maxima["layers.0.q"]),
    ]
    original = FloatModel.tensor

    def checked_read(self: FloatModel, name: str) -> torch.Tensor:
        gc.collect()
        assert all(value() is None for value in previous), "another group's tensors stayed live"
        return original(self, name)

    monkeypatch.setattr(FloatModel, "tensor", checked_read)
    _ = weights["model.layers.1.mlp.down_proj.weight"]


@pytest.mark.parametrize(
    ("profile", "scales"),
    [("servable", "pot"), ("servable", "affine"), ("minimal", "pot"), ("minimal", "affine")],
)
def test_streamed_d_is_byte_identical_to_eager_d(
    calibrated: tuple[FloatModel, CalibrationResult],
    monkeypatch: pytest.MonkeyPatch,
    profile: Literal["servable", "minimal"],
    scales: Literal["pot", "affine"],
) -> None:
    model, calib = calibrated
    streamed = quantize(model, calib, profile=profile, scales=scales, alphas=ALPHAS[:2])

    def eager(
        source: FloatModel, maxima: Mapping[str, torch.Tensor], alpha: Fraction
    ) -> tuple[Mapping[str, torch.Tensor], Mapping[str, torch.Tensor]]:
        weights = {name: source.tensor(name).to(torch.float64) for name in source.names}
        return transform(source.arch, weights, maxima, alpha)

    monkeypatch.setattr(assemble, "transformed_checkpoint", eager)
    expected = quantize(model, calib, profile=profile, scales=scales, alphas=ALPHAS[:2])
    assert encode(INTEGER_MODEL.codec, streamed, limits=INTEGER_MODEL_LIMITS) == encode(
        INTEGER_MODEL.codec, expected, limits=INTEGER_MODEL_LIMITS
    )


def test_bf16_reference_runs_before_any_float64_checkpoint_read(
    calibrated: tuple[FloatModel, CalibrationResult], monkeypatch: pytest.MonkeyPatch
) -> None:
    model, calib = calibrated
    original_read = FloatModel.tensor
    original_reference = assemble.iter_reference_logits
    reference_ran = False
    in_reference = False

    def reference(*args: object, **kwargs: object) -> Generator[torch.Tensor, None, None]:
        nonlocal reference_ran, in_reference
        in_reference = True
        with closing(
            original_reference(model, calib.calibration.held_out, calib.calibration.seed)
        ) as refs:
            for result in refs:
                in_reference = False
                reference_ran = True
                yield result
                in_reference = True
        in_reference = False

    def read(self: FloatModel, name: str) -> torch.Tensor:
        assert in_reference or reference_ran, "the full checkpoint was read before the reference"
        return original_read(self, name)

    monkeypatch.setattr(FloatModel, "tensor", read)
    monkeypatch.setattr(assemble, "iter_reference_logits", reference)
    quantize(model, calib, alphas=(Fraction(3, 5),))
