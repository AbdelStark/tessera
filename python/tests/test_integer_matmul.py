# SPDX-License-Identifier: Apache-2.0 OR MIT
"""Exact bounded integer tiles, with int64 and adversarial saturating-kernel oracles (SC-7)."""

from typing import Any, cast

import numpy as np
import numpy.typing as npt
import pytest
import torch

from tessera_quant import _integer_matmul as tiled


def exact_kernel(x: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    """The integer kernel contract, independent of availability of a machine's acceleration."""
    assert x.dtype == weight.dtype == torch.int8
    assert x.shape[0] <= 128
    assert x.shape[1] <= 1024
    assert weight.shape[1] <= 1024
    return (x.to(torch.int64) @ weight.to(torch.int64)).to(torch.int32)


@pytest.mark.parametrize(
    ("rows", "inner", "columns"),
    [(0, 3, 5), (2, 0, 3), (2, 3, 0), (1, 1, 1), (3, 7, 5), (129, 1025, 1027), (2, 2051, 17)],
)
def test_tiles_preserve_signed_products_shapes_and_every_tail(
    monkeypatch: pytest.MonkeyPatch, rows: int, inner: int, columns: int
) -> None:
    monkeypatch.setattr(torch, "_int_mm", exact_kernel)
    x = torch.arange(rows * inner, dtype=torch.int64).reshape(rows, inner) % 256 - 128
    weights = (np.arange(columns * inner, dtype=np.int64) % 256 - 128).astype(np.int8)
    weights = weights.reshape(columns, inner)
    weights.flags.writeable = False
    expected = x @ torch.tensor(weights, dtype=torch.int64).T
    found = tiled.try_matmul(x, weights)
    assert found is not None
    assert found.dtype == torch.int64
    assert torch.equal(found, expected)


def test_large_odd_results_are_accumulated_in_int64(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(torch, "_int_mm", exact_kernel)
    inner = 262145  # The final result exceeds 2^32, while each partial stays within 2^23.
    x = torch.full((1, inner), -128, dtype=torch.int64)
    weights = np.full((1, inner), -128, dtype=np.int8)
    x[0, 0], weights[0, 0] = 127, 127
    expected = 16384 * (inner - 1) + 127 * 127
    assert expected > 2**32
    assert expected % 2 == 1
    found = tiled.try_matmul(x, weights)
    assert found is not None
    assert found.shape == (1, 1)
    assert int(found[0, 0]) == expected


def test_weight_copies_are_bounded_int8_tiles(monkeypatch: pytest.MonkeyPatch) -> None:
    x = torch.full((129, 1025), 127, dtype=torch.int64)
    weights = np.full((1027, 1025), -128, dtype=np.int8)
    weights.flags.writeable = False
    original = torch.tensor
    copies: list[tuple[int, ...]] = []

    def bounded(data: Any, *args: Any, **kwargs: Any) -> torch.Tensor:
        assert isinstance(data, np.ndarray)
        array = cast(npt.NDArray[np.int8], data)
        assert array.dtype == np.int8
        assert array.shape[0] <= 1024
        assert array.shape[1] <= 1024
        assert kwargs["dtype"] == torch.int8
        copies.append(array.shape)
        return original(array, *args, **kwargs)

    monkeypatch.setattr(torch, "_int_mm", exact_kernel)
    monkeypatch.setattr(torch, "tensor", bounded)
    found = tiled.try_matmul(x, weights)
    assert found is not None
    assert len(copies) == 8  # Two row tiles x two column tiles x two inner tiles.
    assert bool((found == -128 * 127 * 1025).all())


def test_seven_bit_digits_survive_a_saturating_pair_backend(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def saturating(x: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
        # Model unsigned-offset x signed-byte products with saturating int16 pair sums.
        unsigned = x.to(torch.int64) + 128
        signed = weight.to(torch.int64)
        products = unsigned[:, :, None] * signed[None, :, :]
        if x.shape[1] % 2:
            products = torch.cat([products, torch.zeros_like(products[:, :1, :])], dim=1)
        pairs = products[:, ::2, :] + products[:, 1::2, :]
        return (pairs.clamp(-32768, 32767).sum(1) - 128 * signed.sum(0)).to(torch.int32)

    x = torch.tensor([[-128] * 33, [127] * 33], dtype=torch.int64)
    weights = np.array([[-128] * 33, [-127] * 33, [126] * 33, [127] * 33], dtype=np.int8)
    expected = x @ torch.tensor(weights, dtype=torch.int64).T
    assert not torch.equal(saturating(x.to(torch.int8), torch.tensor(weights).T), expected)
    monkeypatch.setattr(torch, "_int_mm", saturating)
    found = tiled.try_matmul(x, weights)
    assert found is not None
    assert torch.equal(found, expected)


def test_wide_activations_are_never_truncated(monkeypatch: pytest.MonkeyPatch) -> None:
    def forbidden(*args: Any) -> Any:
        pytest.fail("wide activation reached INT8 conversion/kernel")

    monkeypatch.setattr(torch, "_int_mm", forbidden)
    for value in [-129, 128, (1 << 25) + 1]:
        assert tiled.try_matmul(torch.tensor([[value]]), np.array([[127]], dtype=np.int8)) is None


@pytest.mark.parametrize("missing", [False, True])
def test_unsupported_integer_kernels_leave_the_exact_fallback(
    monkeypatch: pytest.MonkeyPatch, missing: bool
) -> None:
    def unsupported(*args: Any) -> Any:
        raise NotImplementedError("integer kernel unavailable")

    monkeypatch.setattr(torch, "_int_mm", None if missing else unsupported)
    found = tiled.try_matmul(torch.ones(3, 5, dtype=torch.int64), np.ones((7, 5), dtype=np.int8))
    assert found is None


@pytest.mark.parametrize(
    "error",
    [
        torch.OutOfMemoryError("synthetic memory failure"),
        RuntimeError("DefaultCPUAllocator: cannot allocate memory"),
    ],
)
def test_memory_failure_is_not_retried_with_wider_allocations(
    monkeypatch: pytest.MonkeyPatch, error: RuntimeError
) -> None:
    def no_memory(*args: Any) -> Any:
        raise error

    monkeypatch.setattr(torch, "_int_mm", no_memory)
    with pytest.raises(type(error)):
        tiled.try_matmul(torch.ones(3, 5, dtype=torch.int64), np.ones((7, 5), dtype=np.int8))


def test_native_integer_backend_at_extremes_and_an_odd_large_output() -> None:
    # This is separate from the always-run contract/oracle tests: unsupported machines use
    # the fallback, and must not be reported as having native acceleration evidence.
    x = torch.full((3, 8193), -128, dtype=torch.int64)
    weights = np.full((5, 8193), -128, dtype=np.int8)
    x[0, 0], weights[0, 0] = 127, 127
    expected = x @ torch.tensor(weights, dtype=torch.int64).T
    found = tiled.try_matmul(x, weights)
    if found is None:
        pytest.skip("this CPU does not provide the optional integer backend")
    assert int(expected[0, 0]) > 2**24
    assert int(expected[0, 0]) % 2 == 1
    assert torch.equal(found, expected)
