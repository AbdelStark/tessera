# SPDX-License-Identifier: Apache-2.0 OR MIT
"""The integer model `D` from a checkpoint and its calibration (RFC-0005 'Pipeline', 4 to 6).

`quantize` migrates the checkpoint's weights at each strength `alpha` RFC-0005 allows, and
chooses `D`'s scales for each (`tessera_quant.scales`). It keeps the strength whose `D` diverges
least from the BF16 reference on the held-out split: the mean KL divergence of their next-token
distributions, `D` run by the integer reference (`tessera_quant.reference`) and its logits taken
at their real scale, `2^logits_exponent` (RFC-0027). It then assembles `D` with its
`Provenance`, and `export` writes it as a secret artifact.

Statistics to artifact is float64 and integer arithmetic on the CPU, so it is byte-reproducible
(RFC-0005 'Determinism'). The BF16 reference's logits only rank the strengths, and run under
deterministic algorithms.
"""

import platform
from collections.abc import Callable, Mapping
from contextlib import closing
from fractions import Fraction
from importlib.metadata import version
from pathlib import Path
from typing import Literal

import blake3
import numpy as np
import torch

from tessera_quant._streaming import transformed_checkpoint
from tessera_quant.arch import ArchSpec
from tessera_quant.capture import CalibrationResult, iter_reference_logits, points
from tessera_quant.checkpoint import FloatModel
from tessera_quant.errors import (
    CalibrationError,
    QuantizationInfeasibleError,
    TesseraQuantError,
    UsageError,
)
from tessera_quant.integer_model import (
    INTEGER_MODEL,
    CalibrationIdentity,
    IntegerModel,
    Profile,
    Provenance,
)
from tessera_quant.migration import ALPHAS, kl_rows, select_alpha
from tessera_quant.reference import run_reference
from tessera_quant.scales import Quantized, choose
from tessera_quant.tce import INTEGER_MODEL_LIMITS, write_artifact

LIBRARIES = (
    "blake3",
    "datasets",
    "numpy",
    "python",
    "safetensors",
    "tokenizers",
    "torch",
    "transformers",
)
"""The libraries `Provenance` records, by name (RFC-0005 'Determinism')."""

_CHUNK = 1 << 24


def quantize(
    model: FloatModel,
    calib: CalibrationResult,
    *,
    profile: Literal["servable", "minimal"] = "servable",
    scales: Literal["pot", "affine"] = "pot",
    alphas: tuple[Fraction, ...] = ALPHAS,
    device: str = "cpu",
) -> IntegerModel:
    """The integer model `D` of `model`, calibrated by `calib` (02 §4).

    The migration strength is the admissible one of `alphas` whose `D` diverges least from the
    BF16 reference on the held-out split, the smallest of equal ones (RFC-0005 'Calibration').
    All candidates are preflighted before evaluation, retaining only their strengths. With `affine`
    scales, every weight that can take them does (RFC-0025): Q, K and V, and in `servable` O, up
    and down too.

    Raises:
        CalibrationError: for an empty held-out split, or as `tessera_quant.scales.choose`.
        UsageError: as `check_options`.
        TesseraQuantError: as `choose`.
    """
    the_profile = check_options(profile, scales)
    held_out = calib.calibration.held_out
    if not held_out:
        raise CalibrationError(
            "the migration strength is chosen on the held-out split: it is empty"
        )
    arch = model.arch
    _check_statistics(arch, calib.statistics.maxima)
    # The files are hashed once, before their tensors are read.
    checkpoint = checkpoint_digests(model)
    maxima = calib.statistics.maxima

    def assembled(q: Quantized, alpha: Fraction) -> IntegerModel:
        provenance = _provenance(checkpoint, calib, alpha)
        return IntegerModel(arch, the_profile, q.quant, q.weights, provenance)

    affine = affine_weights(arch, the_profile) if scales == "affine" else ()

    def quantized(alpha: Fraction) -> Quantized:
        migrated, migrated_maxima = transformed_checkpoint(model, maxima, alpha)
        return choose(arch, the_profile, migrated, migrated_maxima, affine=affine)

    def divergence(alpha: Fraction) -> float:
        rows: list[torch.Tensor] = []
        # RFC-0005: construction scratch is gone before the BF16 model/window is resident.
        # Complete preflight above still precedes every held-out forward (#625).
        d = assembled(quantized(alpha), alpha)
        with closing(
            iter_reference_logits(model, held_out, calib.calibration.seed, device)
        ) as refs:

            def next_reference() -> torch.Tensor:
                try:
                    return next(refs)
                except StopIteration:
                    raise TesseraQuantError(
                        "the reference yields one row block per window"
                    ) from None

            reference: torch.Tensor | None = None
            # Avoid chain/zip retaining the first window or a previously yielded pair.
            for window in held_out:
                if reference is None:
                    reference = next_reference()
                found = torch.as_tensor(
                    run_reference(d, np.array(window.tokens, dtype=np.uint32))
                ).to(torch.float64)
                rows.append(kl_rows(reference, found * 2.0**d.quant.public.logits_exponent))
                reference = None
                del found
            if next(refs, None) is not None:
                raise TesseraQuantError("the reference yields one row block per window")
        # Retain only one scalar per position, preserving the original final mean's order.
        return float(torch.cat(rows).mean())

    admissible = _admissible_alphas(quantized, alphas)
    alpha = select_alpha(divergence, admissible)
    return assembled(quantized(alpha), alpha)


def _check_statistics(arch: ArchSpec, maxima: Mapping[str, torch.Tensor]) -> None:
    """Reject malformed original statistics before any candidate can be omitted."""
    expected = points(arch)
    if set(maxima) != set(expected):
        raise CalibrationError("the statistics have exactly the quantization points")
    for name, channels in expected.items():
        values = maxima[name]
        if (
            values.shape != (channels,)
            or not bool(torch.isfinite(values).all())
            or not bool((values >= 0).all())
            or not bool((values > 0).any())
        ):
            raise CalibrationError(f"{name}: statistics are finite nonnegative channel maxima")


def _admissible_alphas(
    build: Callable[[Fraction], Quantized], candidates: tuple[Fraction, ...]
) -> tuple[Fraction, ...]:
    """Preflight only derived bounds, releasing each candidate and exception immediately."""
    if not candidates:
        raise TesseraQuantError("alpha is chosen among at least one candidate")
    admissible: list[Fraction] = []
    for alpha in sorted(candidates):
        try:
            # Do not bind the result or exception: both can own model-sized allocations.
            build(alpha)
        except QuantizationInfeasibleError:
            continue
        admissible.append(alpha)
    if not admissible:
        raise CalibrationError("no migration strength satisfies the scale/table bounds")
    return tuple(admissible)


def check_options(profile: str, scales: str) -> Profile:
    """The profile `quantize` builds for `profile`, once `profile` and `scales` are its options.

    Raises:
        UsageError: for scales other than `pot` and `affine`, or an unknown profile.
    """
    if scales not in ("pot", "affine"):
        raise UsageError(f"no scales {scales!r}: pot or affine")
    profiles = {"servable": Profile.Servable, "minimal": Profile.Minimal}
    if profile not in profiles:
        raise UsageError(f"no profile {profile!r}: servable or minimal")
    return profiles[profile]


def affine_weights(arch: ArchSpec, profile: Profile) -> tuple[str, ...]:
    """The Hugging Face names of the weights that can take `affine` scales (RFC-0025).

    Q, K and V in every layer, and in `servable` O, up and down too. `minimal`'s O and down keys
    would exceed 62 bits, and its up projection has no site.
    """
    attention = ("q_proj", "k_proj", "v_proj")
    names: list[str] = []
    for layer in range(arch.n_layers):
        names += [f"model.layers.{layer}.self_attn.{w}.weight" for w in attention]
        if profile is Profile.Servable:
            names.append(f"model.layers.{layer}.self_attn.o_proj.weight")
            names += [f"model.layers.{layer}.mlp.{w}.weight" for w in ("up_proj", "down_proj")]
    return tuple(names)


def checkpoint_digests(model: FloatModel) -> list[tuple[str, bytes]]:
    """Each file of `model`'s checkpoint and its BLAKE3, by name: what `Provenance` records."""
    paths = {path.name: path for path in model.files.values()}
    return [(name, file_digest(paths[name])) for name in sorted(paths)]


def export(model: IntegerModel, out: Path) -> bytes:
    """Writes `D` to `out` as an `IntegerModel` artifact, a secret file, and returns its digest.

    The digest is the envelope's: the file's last 32 bytes (RFC-0003 'Envelope').
    """
    return write_artifact(out, INTEGER_MODEL, model, limits=INTEGER_MODEL_LIMITS)


def file_digest(path: Path) -> bytes:
    """BLAKE3 of a file's bytes, read in chunks: a checkpoint file's digest in `Provenance`."""
    hasher = blake3.blake3()
    with path.open("rb") as file:
        while chunk := file.read(_CHUNK):
            hasher.update(chunk)
    return hasher.digest()


def libraries() -> dict[str, str]:
    """The version of each library of `LIBRARIES`, by name."""
    return {
        name: platform.python_version() if name == "python" else version(name) for name in LIBRARIES
    }


def _provenance(
    checkpoint: list[tuple[str, bytes]], calib: CalibrationResult, alpha: Fraction
) -> Provenance:
    calibration = calib.calibration
    source = calibration.source
    return Provenance(
        tessera_quant=version("tessera-quant"),
        checkpoint=checkpoint,
        calibration=CalibrationIdentity(
            dataset=source.name,
            config=source.config,
            split=source.split,
            revision=source.revision,
            files=list(source.files),
            seed=calibration.seed,
            sequences=len(calibration.calibration),
            held_out=len(calibration.held_out),
            tokens_per_sequence=calibration.length,
            selection=calibration.selection_digest,
        ),
        statistics=calib.digest,
        alpha=alpha,
        libraries=libraries(),
    )
