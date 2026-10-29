# SPDX-License-Identifier: Apache-2.0 OR MIT
"""Exact attention products: binding, broadcasts, narrowing, allocation and failure controls."""

import math
from typing import Any

import numpy as np
import pytest
import torch

from tessera_quant import _integer_matmul as tiled


@pytest.mark.parametrize("route", ["tensor", "weights"])
@pytest.mark.parametrize(
    ("rows", "inner", "columns"), [(1, 7, 5), (3, 1, 5), (3, 7, 1), (3, 1025, 5)]
)
def test_singleton_tiles_have_valid_blas_leading_dimensions(
    monkeypatch: pytest.MonkeyPatch, route: str, rows: int, inner: int, columns: int
) -> None:
    calls = 0

    def blas(x: torch.Tensor, right: torch.Tensor) -> torch.Tensor:
        nonlocal calls
        calls += 1
        # Match the pinned oneDNN dispatcher: stride(1)==1 selects row-major 'N'.
        # A size-1 axis may be contiguous with a smaller/zero leading stride; oneDNN
        # rejects that descriptor and the upstream wrapper can leave output unwritten.
        for operand in (x, right):
            transposed = operand.stride(1) != 1
            leading = operand.stride(1) if transposed else operand.stride(0)
            required = operand.shape[0] if transposed else operand.shape[1]
            assert leading >= required
        return exact_kernel(x, right)

    monkeypatch.setattr(torch, "_int_mm", blas)
    left = torch.arange(rows * inner).reshape(rows, inner) % 256 - 128
    if rows == 1:
        left = left.as_strided(left.shape, (0, 1))
    weights = (np.arange(columns * inner, dtype=np.int64) % 256 - 128).astype(np.int8)
    weights = weights.reshape(columns, inner)
    right = torch.tensor(weights, dtype=torch.int64).T
    expected = left @ right
    if route == "weights":
        found = tiled.try_matmul(left, weights)
    else:
        found = tiled.try_bmm(left.unsqueeze(0), right.unsqueeze(0))
        if found is not None:
            found = found[0]
    assert found is not None
    assert calls > 0
    assert torch.equal(found, expected)


def exact_kernel(x: torch.Tensor, right: torch.Tensor) -> torch.Tensor:
    assert x.dtype == right.dtype == torch.int8
    assert x.shape[0] <= 128
    assert x.shape[1] <= 1024
    assert right.shape[1] <= 1024
    return (x.to(torch.int64) @ right.to(torch.int64)).to(torch.int32)


@pytest.mark.parametrize(
    ("left_shape", "right_shape"),
    [
        ((3, 7), (7, 5)),
        ((2, 3, 7), (2, 7, 5)),
        ((2, 1, 3, 7), (1, 4, 7, 5)),
        ((2, 3, 7), (7, 5)),
        ((3, 7), (2, 7, 5)),
        ((0, 3, 7), (1, 7, 5)),
        ((2, 0, 7), (2, 7, 5)),
        ((2, 3, 0), (2, 0, 5)),
        ((2, 3, 7), (2, 7, 0)),
        ((2, 129, 3), (2, 3, 1027)),
        ((2, 3, 2051), (2, 2051, 7)),
    ],
)
def test_shapes_broadcasts_and_tile_tails(
    monkeypatch: pytest.MonkeyPatch,
    left_shape: tuple[int, ...],
    right_shape: tuple[int, ...],
) -> None:
    monkeypatch.setattr(torch, "_int_mm", exact_kernel)
    left = (torch.arange(math.prod(left_shape)).reshape(left_shape) % 256 - 128).to(torch.int64)
    right = (torch.arange(math.prod(right_shape)).reshape(right_shape) * 31 % 256 - 128).to(
        torch.int64
    )
    expected = left @ right
    original_empty = torch.empty

    def poisoned(*args: Any, **kwargs: Any) -> torch.Tensor:
        result = original_empty(*args, **kwargs)
        if result.dtype == torch.int64:
            result.fill_(1 << 62)  # An unwritten tail must not pass by allocator reuse.
        return result

    monkeypatch.setattr(torch, "empty", poisoned)
    found = tiled.try_bmm(left, right)
    assert found is not None
    assert found.dtype == torch.int64
    assert found.shape == expected.shape
    assert torch.equal(found, expected)


def test_strided_gqa_binds_each_query_to_its_own_kv_head(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(torch, "_int_mm", exact_kernel)
    n, heads, kv, d = 7, 6, 2, 5
    q = (torch.arange(n * heads * d).reshape(n, heads, d) % 256 - 128).to(torch.int64)
    k = (torch.arange(n * kv * d).reshape(n, kv, d) * 73 % 256 - 128).to(torch.int64)
    left = q.transpose(0, 1)
    right = k.transpose(0, 1).repeat_interleave(heads // kv, dim=0).transpose(1, 2)
    assert not left.is_contiguous()
    assert not right.is_contiguous()
    found = tiled.try_bmm(left, right)
    assert found is not None
    # Literal integer oracle includes the GQA head map; no batched matmul or kernel under test.
    for head in range(heads):
        for row in range(n):
            for column in range(n):
                expected = sum(
                    int(q[row, head, j]) * int(k[column, head // 3, j]) for j in range(d)
                )
                assert int(found[head, row, column]) == expected


def test_broadcast_views_are_not_reshaped_into_a_materialized_batch(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    left = torch.arange(3 * 7).reshape(1, 3, 7) % 256 - 128
    right = torch.arange(2 * 4 * 7 * 5).reshape(2, 4, 7, 5) % 256 - 128
    expected = left @ right

    def forbidden(*args: Any, **kwargs: Any) -> Any:
        pytest.fail("broadcast operands must be indexed as views, without a batch reshape")

    monkeypatch.setattr(torch, "_int_mm", exact_kernel)
    monkeypatch.setattr(torch.Tensor, "reshape", forbidden)
    original_contiguous = torch.Tensor.contiguous

    def bounded_contiguous(self: torch.Tensor, *args: Any, **kwargs: Any) -> torch.Tensor:
        assert self.ndim == 2  # Only matrix tiles, never an expanded logical batch.
        return original_contiguous(self, *args, **kwargs)

    monkeypatch.setattr(torch.Tensor, "contiguous", bounded_contiguous)
    found = tiled.try_bmm(left, right)
    assert found is not None
    assert torch.equal(found, expected)


def test_empty_output_does_not_enumerate_a_huge_logical_batch(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    left = torch.empty((1_000_000_000, 0, 3), dtype=torch.int64)
    right = torch.ones((1, 3, 5), dtype=torch.int64)

    def forbidden(*args: Any, **kwargs: Any) -> Any:
        pytest.fail("an empty result must not materialize or traverse a batch index pool")

    monkeypatch.setattr(torch, "_int_mm", forbidden)
    # product() snapshots each input pool; intercept it so a regression cannot allocate a
    # billion Python integers before this test can reject the resource violation.
    monkeypatch.setattr(tiled, "product", forbidden, raising=False)
    original = torch.Tensor.__getitem__

    def no_empty_head(self: torch.Tensor, index: Any) -> torch.Tensor:
        assert self.numel() > 0  # No per-head indexing is needed for an empty output.
        return original(self, index)

    monkeypatch.setattr(torch.Tensor, "__getitem__", no_empty_head)
    found = tiled.try_bmm(left, right)
    assert found is not None
    assert found.shape == (1_000_000_000, 0, 5)
    assert found.numel() == 0


def test_rhs_and_activation_conversions_are_bounded_before_narrowing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    left = torch.full((2, 129, 1025), 127, dtype=torch.int64)
    right = torch.full((2, 1025, 1027), -128, dtype=torch.int64)
    original = torch.Tensor.to
    copies: list[tuple[int, ...]] = []

    def bounded(self: torch.Tensor, *args: Any, **kwargs: Any) -> torch.Tensor:
        if args and args[0] == torch.int8:
            assert self.ndim == 2
            assert self.shape[0] <= 1024
            assert self.shape[1] <= 1024
            copies.append(tuple(self.shape))
        return original(self, *args, **kwargs)

    monkeypatch.setattr(torch, "_int_mm", exact_kernel)
    monkeypatch.setattr(torch.Tensor, "to", bounded)
    found = tiled.try_bmm(left, right)
    assert found is not None
    assert bool((found == -128 * 127 * 1025).all())
    assert len(copies) == 32  # Two operands per tile, two batches and eight matrix tiles.


def test_signed_digits_are_required_even_for_a_tensor_rhs(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def saturating(x: torch.Tensor, right: torch.Tensor) -> torch.Tensor:
        products = (x.to(torch.int64) + 128)[:, :, None] * right.to(torch.int64)[None, :, :]
        if x.shape[1] % 2:
            products = torch.cat([products, torch.zeros_like(products[:, :1, :])], dim=1)
        paired = products[:, ::2, :] + products[:, 1::2, :]
        return (paired.clamp(-32768, 32767).sum(1) - 128 * right.to(torch.int64).sum(0)).to(
            torch.int32
        )

    left = torch.tensor([[[-128] * 33, [127] * 33]], dtype=torch.int64)
    right = torch.tensor([[-128, -127, 126, 127]] * 33, dtype=torch.int64).unsqueeze(0)
    expected = left @ right
    assert not torch.equal(saturating(left[0].to(torch.int8), right[0].to(torch.int8)), expected[0])
    monkeypatch.setattr(torch, "_int_mm", saturating)
    found = tiled.try_bmm(left, right)
    assert found is not None
    assert torch.equal(found, expected)


def test_odd_large_totals_do_not_wrap_an_int32_accumulator(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(torch, "_int_mm", exact_kernel)
    inner = 262145
    left = torch.full((2, 1, inner), -128, dtype=torch.int64)
    right = torch.full((2, inner, 1), -128, dtype=torch.int64)
    left[0, 0, 0] = right[0, 0, 0] = 127
    found = tiled.try_bmm(left, right)
    assert found is not None
    expected = 16384 * (inner - 1) + 127 * 127
    assert expected > 2**32
    assert expected % 2 == 1
    assert int(found[0, 0, 0]) == expected
    assert int(found[1, 0, 0]) == 16384 * inner


@pytest.mark.parametrize("side", [0, 1])
def test_wide_operands_are_refused_before_conversion(
    monkeypatch: pytest.MonkeyPatch, side: int
) -> None:
    def forbidden(*args: Any) -> Any:
        pytest.fail("wide operand reached an INT8 kernel")

    monkeypatch.setattr(torch, "_int_mm", forbidden)
    for value in [-129, 128, 65536, (1 << 25) + 1]:
        operands = [torch.ones((2, 3, 3), dtype=torch.int64) for _ in range(2)]
        operands[side][1, 0, 0] = value
        before = [x.clone() for x in operands]
        assert tiled.try_bmm(*operands) is None
        assert all(torch.equal(x, y) for x, y in zip(operands, before, strict=True))


def test_missing_and_partially_unsupported_backend_discard_all_partial_results(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    left = torch.ones((2, 1, 3), dtype=torch.int64)
    right = torch.full((2, 3, 2), 127, dtype=torch.int64)
    monkeypatch.setattr(torch, "_int_mm", None)
    assert tiled.try_bmm(left, right) is None
    calls = 0

    def partial(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        nonlocal calls
        calls += 1
        if calls == 3:
            raise NotImplementedError("second batch unsupported")
        return exact_kernel(x, y)

    monkeypatch.setattr(torch, "_int_mm", partial)
    assert tiled.try_bmm(left, right) is None
    assert calls == 3
    assert bool((left == 1).all())
    assert bool((right == 127).all())


@pytest.mark.parametrize(
    "error", [torch.OutOfMemoryError("synthetic OOM"), RuntimeError("CPU allocation failed")]
)
def test_allocator_failure_is_not_a_backend_fallback(
    monkeypatch: pytest.MonkeyPatch, error: RuntimeError
) -> None:
    def failed(*args: Any) -> Any:
        raise error

    monkeypatch.setattr(torch, "_int_mm", failed)
    with pytest.raises(type(error)):
        tiled.try_bmm(
            torch.ones((2, 3, 5), dtype=torch.int64), torch.ones((2, 5, 7), dtype=torch.int64)
        )


def test_invalid_metadata_and_unrepresentable_sum_leave_the_original_operation() -> None:
    assert tiled.try_bmm(torch.ones(3), torch.ones(3)) is None
    assert tiled.try_bmm(torch.ones(2, 3), torch.ones(3, 2)) is None  # float tensors
    assert (
        tiled.try_bmm(
            torch.ones((2, 3, 5), dtype=torch.int64), torch.ones((3, 5, 2), dtype=torch.int64)
        )
        is None
    )
    assert (
        tiled.try_bmm(torch.ones((2, 3), dtype=torch.int64), torch.ones((4, 2), dtype=torch.int64))
        is None
    )
    assert (
        tiled.try_bmm(
            torch.empty((0, 1 << 49), dtype=torch.int64),
            torch.empty((1 << 49, 0), dtype=torch.int64),
        )
        is None
    )
    assert (
        tiled.try_bmm(
            torch.empty((2, 3), dtype=torch.int64, device="meta"),
            torch.ones((3, 2), dtype=torch.int64),
        )
        is None
    )


def test_native_tensor_rhs_backend_has_the_exact_signed_and_strided_contract() -> None:
    left = torch.arange(3 * 2 * 8193).reshape(3, 2, 8193) % 256 - 128
    right = (torch.arange(3 * 5 * 8193).reshape(3, 5, 8193) % 256 - 128).transpose(1, 2)
    expected = left @ right
    found = tiled.try_bmm(left, right)
    if found is None:
        pytest.skip("this CPU does not provide the optional integer backend")
    assert torch.equal(found, expected)
