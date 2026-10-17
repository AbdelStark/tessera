# SPDX-License-Identifier: Apache-2.0 OR MIT
"""The errors of the quantization front-end (04 §4): each carries a stable code of 04 §3."""


class TesseraQuantError(Exception):
    """The base of every error the package raises.

    Its message holds public data only, such as names, shapes and configuration keys, never a
    weight or activation value (05 §3).
    """

    code: str = "ERR-INTERNAL"
    """The error's stable code, as 04 §3 spells it."""


class ArchitectureError(TesseraQuantError):
    """A checkpoint outside what `ArchSpec` supports, or not the expected one (`ERR-QUANT-ARCH`)."""

    code = "ERR-QUANT-ARCH"


class CalibrationError(TesseraQuantError):
    """Calibration that cannot be trusted (`ERR-QUANT-CALIB`).

    Too few usable documents, an overlap with evaluation data, a malformed dataset file, or
    non-finite or degenerate statistics.
    """

    code = "ERR-QUANT-CALIB"
