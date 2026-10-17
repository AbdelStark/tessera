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
from fractions import Fraction
from importlib.metadata import version
from pathlib import Path
from typing import Literal

import blake3
import numpy as np
import torch

from tessera_quant.capture import CalibrationResult, reference_logits
from tessera_quant.checkpoint import FloatModel
from tessera_quant.errors import CalibrationError, UsageError
from tessera_quant.integer_model import (
    INTEGER_MODEL,
    CalibrationIdentity,
    IntegerModel,
    Profile,
    Provenance,
)
from tessera_quant.migration import ALPHAS, mean_kl, select_alpha, transform
from tessera_quant.reference import trace
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

    The migration strength is the one of `alphas` whose `D` diverges least from the BF16 reference
    on the held-out split, the smallest of equal ones (RFC-0005 'Calibration').

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
    # The files are hashed once, before their tensors are read.
    checkpoint = checkpoint_digests(model)
    weights = {name: model.tensor(name).to(torch.float64) for name in model.names}
    maxima = calib.statistics.maxima
    reference = reference_logits(model, held_out, calib.calibration.seed, device)

    def assembled(q: Quantized, alpha: Fraction) -> IntegerModel:
        provenance = _provenance(checkpoint, calib, alpha)
        return IntegerModel(arch, the_profile, q.quant, q.weights, provenance)

    def quantized(alpha: Fraction) -> Quantized:
        migrated, migrated_maxima = transform(arch, weights, maxima, alpha)
        return choose(arch, the_profile, migrated, migrated_maxima)

    def divergence(alpha: Fraction) -> float:
        q = quantized(alpha)
        d = assembled(q, alpha)
        found = torch.cat(
            [trace(d, np.array(w.tokens, dtype=np.uint32)).logits for w in held_out]
        ).to(torch.float64)
        return mean_kl(reference, found * 2.0**d.quant.public.logits_exponent)

    alpha = select_alpha(divergence, alphas)
    return assembled(quantized(alpha), alpha)


def check_options(profile: str, scales: str) -> Profile:
    """The profile `quantize` builds for `profile`, once `profile` and `scales` are its options.

    Raises:
        UsageError: for `affine` scales, which the reference computes only with #301, or an
            unknown profile.
    """
    if scales != "pot":
        raise UsageError(
            "affine scales are chosen once the integer reference computes affine sites (#301)"
        )
    profiles = {"servable": Profile.Servable, "minimal": Profile.Minimal}
    if profile not in profiles:
        raise UsageError(f"no profile {profile!r}: servable or minimal")
    return profiles[profile]


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
