# SPDX-License-Identifier: Apache-2.0 OR MIT
"""The errors of the quantization front-end (04 §4): each carries a stable code of 04 §3."""


class TesseraQuantError(Exception):
    """The base of every error the package raises.

    Its message holds public data only, such as names, shapes and configuration keys, never a
    weight or activation value (05 §3).
    """

    code: str = "ERR-INTERNAL"
    """The error's stable code, as 04 §3 spells it."""


class UsageError(TesseraQuantError):
    """Invalid arguments or configuration (`ERR-USAGE`).

    An option the package does not offer yet, a malformed specification file, or input that a
    call's contract excludes, such as a token outside the vocabulary.
    """

    code = "ERR-USAGE"


class ArchitectureError(TesseraQuantError):
    """A checkpoint outside what `ArchSpec` supports, or not the expected one (`ERR-QUANT-ARCH`)."""

    code = "ERR-QUANT-ARCH"


class CalibrationError(TesseraQuantError):
    """Calibration that cannot be trusted (`ERR-QUANT-CALIB`).

    Too few usable documents, an overlap with evaluation data, a malformed dataset file, or
    non-finite or degenerate statistics.
    """

    code = "ERR-QUANT-CALIB"


class CodecError(TesseraQuantError):
    """Bytes outside the Tessera Canonical Encoding (RFC-0003), or a value too large to read back.

    `code` is one of the `ERR-CODEC-*` codes of 04 §3, and `offset` is the first byte the
    failing check reads, counted from the start of the input: the code and offset RFC-0003
    'Limits' and 'Envelope' fix, so that every reader reports the same. The message holds both
    and nothing read from the input, which may be secret-class data (05 §3).

    A writer raises it, with `ERR-CODEC-LIMIT`, for a value its readers would refuse under the
    limits it writes for, at the offset where they would.
    """

    offset: int
    """The offset of the first byte the failing check reads."""

    def __init__(self, code: str, offset: int) -> None:
        """An error with `code` at `offset`."""
        super().__init__(f"{code} at byte {offset}")
        self.code = code
        self.offset = offset


class EncodeError(TesseraQuantError):
    """A value its TCE type cannot hold (`ERR-INTERNAL`).

    An integer out of range, text outside printable ASCII, an array of another dtype: a bug in
    the code that built the value, which nothing converts silently. The message names the type,
    never the value (05 §3).
    """


class TableError(TesseraQuantError):
    """Table parameters RFC-0008 does not allow, or a key outside its table (`ERR-INTERNAL`).

    The compiler and the quantizer choose the parameters, and an honest witness's keys lie in their
    tables (INV-DM-2), so either is a bug. The message names the kind and the rule, never a key.
    """
