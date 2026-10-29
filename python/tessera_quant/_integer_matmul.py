# SPDX-License-Identifier: Apache-2.0 OR MIT
"""Bounded exact INT8 tiles for the CPU integer reference (RFC-0005, SC-7).

The full trace keeps int64 matmul as an equivalence oracle. These tiles use integer operations
only and retain no cache of secret operands; unsupported kernels leave the caller its oracle.
"""

from collections.abc import Callable
from typing import cast

import numpy as np
import numpy.typing as npt
import torch

_ROWS = 128
_COLUMNS = 1024
_INNER = 1024


def try_matmul(x: torch.Tensor, weights: npt.NDArray[np.int8]) -> torch.Tensor | None:
    """Try exact `x @ weights.T`, returning None for wide operands or unsupported kernels.

    Weight digits `hi = w >> 1` and `lo = w & 1` satisfy `w = 2*hi + lo`, including -128.
    Here `hi` lies in [-64, 63] and `lo` in [0, 1]. A signed INT8 pair contributes at most
    2*128*64 = 16384 in magnitude. Even kernels using unsigned-offset operands have a pair
    bound 2*255*64 = 32640, within signed int16. Each 1024-term partial is at most 2^23 (high)
    or 2^17 (low); widen to int64 before doubling/adding. The full sum's public bound is
    128*128*K < 2^63, including every partial sum in any order. No float conversion is used.

    Scratch is bounded by public tile sizes beside the necessary output and original operands.
    Copy only each INT8 weight tile, including when the source is a read-only mapped D.
    """
    if (
        x.device.type != "cpu"
        or x.dtype != torch.int64
        or x.ndim != 2
        or weights.ndim != 2
        or weights.dtype != np.int8
        or x.shape[1] != weights.shape[1]
    ):
        return None
    rows, inner = x.shape
    columns = weights.shape[0]
    if 128 * 128 * inner >= 1 << 63:
        return None
    if x.numel() and (int(x.min()) < -128 or int(x.max()) > 127):
        return None
    operator = getattr(torch, "_int_mm", None)
    if not callable(operator):
        return None
    kernel = cast(Callable[[torch.Tensor, torch.Tensor], torch.Tensor], operator)
    result = torch.empty((rows, columns), dtype=torch.int64)
    for row in range(0, rows, _ROWS):
        for column in range(0, columns, _COLUMNS):
            end_row, end_column = min(row + _ROWS, rows), min(column + _COLUMNS, columns)
            tile = torch.zeros((end_row - row, end_column - column), dtype=torch.int64)
            for start in range(0, inner, _INNER):
                end = min(start + _INNER, inner)
                activation = x[row:end_row, start:end].to(torch.int8).contiguous()
                # `tensor` copies this bounded tile without borrowing a read-only NumPy buffer.
                weight = torch.tensor(weights[column:end_column, start:end], dtype=torch.int8)
                high, low = weight >> 1, weight & 1
                try:
                    high_product = kernel(activation, high.T)
                    low_product = kernel(activation, low.T)
                except torch.OutOfMemoryError:
                    raise
                except RuntimeError as error:
                    # CPU allocators can report OOM as an ordinary RuntimeError. A fallback
                    # must not retry such a failure by allocating a whole int64 weight matrix.
                    message = str(error).lower()
                    if "alloc" in message or "out of memory" in message:
                        raise
                    return None
                tile.add_(high_product.to(torch.int64), alpha=2)
                tile.add_(low_product.to(torch.int64))
            result[row:end_row, column:end_column] = tile
    return result
