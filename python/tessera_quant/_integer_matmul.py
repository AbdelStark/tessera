# SPDX-License-Identifier: Apache-2.0 OR MIT
"""Bounded exact INT8 tiles for the CPU integer reference (RFC-0005, SC-7).

The full trace keeps int64 matmul as an equivalence oracle. These tiles use integer operations
only and retain no cache of secret operands; unsupported kernels leave the caller its oracle.
"""

from collections.abc import Callable
from itertools import zip_longest
from math import prod
from typing import cast

import numpy as np
import numpy.typing as npt
import torch

_ROWS = 128
_COLUMNS = 1024
_INNER = 1024
_Kernel = Callable[[torch.Tensor, torch.Tensor], torch.Tensor]
_Pack = Callable[[int, int, int, int], torch.Tensor]


def _narrow(x: torch.Tensor) -> bool:
    return not x.numel() or (int(x.min()) >= -128 and int(x.max()) <= 127)


def _kernel() -> _Kernel | None:
    operator = getattr(torch, "_int_mm", None)
    return cast(_Kernel, operator) if callable(operator) else None


def _batch_shape(left: torch.Size, right: torch.Size) -> tuple[int, ...] | None:
    """Broadcast only public leading dimensions, including zero axes, without meta tensors."""
    shape: list[int] = []
    for a, b in zip_longest(reversed(left), reversed(right), fillvalue=1):
        if a == 1:
            shape.append(b)
        elif b == 1 or a == b:
            shape.append(a)
        else:
            return None
    return tuple(reversed(shape))


def _multiply(x: torch.Tensor, result: torch.Tensor, pack: _Pack, kernel: _Kernel) -> bool:
    """Fill one output matrix using bounded operand copies; discard it on unsupported kernels."""
    rows, columns = result.shape
    inner = x.shape[1]
    for row in range(0, rows, _ROWS):
        for column in range(0, columns, _COLUMNS):
            end_row, end_column = min(row + _ROWS, rows), min(column + _COLUMNS, columns)
            tile = torch.zeros((end_row - row, end_column - column), dtype=torch.int64)
            for start in range(0, inner, _INNER):
                end = min(start + _INNER, inner)
                activation = x[row:end_row, start:end].to(torch.int8).contiguous()
                weight = pack(column, end_column, start, end)
                high, low = weight >> 1, weight & 1
                try:
                    high_product = kernel(activation, high.T)
                    low_product = kernel(activation, low.T)
                except torch.OutOfMemoryError:
                    raise
                except RuntimeError as error:
                    # Ordinary CPU allocator errors must not retry with larger operands.
                    message = str(error).lower()
                    if "alloc" in message or "out of memory" in message:
                        raise
                    return False
                tile.add_(high_product.to(torch.int64), alpha=2)
                tile.add_(low_product.to(torch.int64))
            result[row:end_row, column:end_column] = tile
    return True


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
    if not _narrow(x):
        return None
    kernel = _kernel()
    if kernel is None:
        return None
    result = torch.empty((rows, columns), dtype=torch.int64)

    def pack(column: int, end_column: int, start: int, end: int) -> torch.Tensor:
        # Copy without borrowing a read-only mapped D buffer.
        return torch.tensor(weights[column:end_column, start:end], dtype=torch.int8)

    return result if _multiply(x, result, pack, kernel) else None


def try_bmm(left: torch.Tensor, right: torch.Tensor) -> torch.Tensor | None:
    """Try exact CPU int64 `left @ right`, preserving batch broadcasts and operand strides.

    Both operands must lie in [-128,127] before any INT8 conversion. The two-digit proof and
    public accumulator bounds of `try_matmul` apply to each matrix. Expanding batch axes only
    creates views: index each logical batch without flattening/materializing its operands.
    Pack each RHS tile from its original tensor, never an entire converted head/batch. The
    necessary int64 output is written directly, with tile-bounded scratch and no activation
    cache. An unsupported kernel discards the entire result, including earlier batches.
    """
    if (
        left.device.type != "cpu"
        or right.device.type != "cpu"
        or left.dtype != torch.int64
        or right.dtype != torch.int64
        or left.ndim < 2
        or right.ndim < 2
        or left.shape[-1] != right.shape[-2]
    ):
        return None
    inner = left.shape[-1]
    if 128 * 128 * inner >= 1 << 63:
        return None
    batch = _batch_shape(left.shape[:-2], right.shape[:-2])
    if batch is None:
        return None
    if not _narrow(left) or not _narrow(right):
        return None
    kernel = _kernel()
    if kernel is None:
        return None
    left_view = left.expand(*batch, *left.shape[-2:])
    right_view = right.expand(*batch, *right.shape[-2:])
    result = torch.empty((*batch, left.shape[-2], right.shape[-1]), dtype=torch.int64)
    if not result.numel():
        return result
    if not inner:
        return result.zero_()
    # range is lazy; itertools.product would snapshot each potentially huge index pool.
    # Unflatten one logical index at a time, with metadata scratch bounded by the rank.
    for ordinal in range(prod(batch)):
        remaining = ordinal
        reverse_index: list[int] = []
        for size in reversed(batch):
            remaining, coordinate = divmod(remaining, size)
            reverse_index.append(coordinate)
        index = tuple(reversed(reverse_index))
        matrix = right_view[index]

        def pack(
            column: int, end_column: int, start: int, end: int, source: torch.Tensor = matrix
        ) -> torch.Tensor:
            return source[start:end, column:end_column].T.to(torch.int8).contiguous()

        if not _multiply(left_view[index], result[index], pack, kernel):
            return None
    return result
