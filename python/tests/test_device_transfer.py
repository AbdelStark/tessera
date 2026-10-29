# SPDX-License-Identifier: Apache-2.0 OR MIT
"""BF16 device transfers preserve calibration values and reference logits (RFC-0005, #609)."""

from types import SimpleNamespace
from typing import Any, cast

import pytest
import torch

from tessera_quant import capture


def finite_patterns(device: str) -> tuple[torch.Tensor, torch.Tensor]:
    """Every finite BF16 bit pattern, including both signed zeros, with its CPU float64 oracle."""
    if device == "mps" and not torch.backends.mps.is_available():
        pytest.skip("native Metal BF16 transfer regression requires an MPS device")
    bits = torch.arange(65536, dtype=torch.int32).to(torch.int16)
    values = bits.view(torch.bfloat16)
    values = values[torch.isfinite(values)].reshape(255, 256)
    assert values.numel() == 65280
    return values.to(device), values.to(torch.float64)


@pytest.mark.parametrize("device", ["cpu", "mps"])
def test_observation_preserves_all_finite_bf16_channel_maxima(device: str) -> None:
    values, expected = finite_patterns(device)
    recorder = capture._Recorder()  # pyright: ignore[reportPrivateUsage]
    recorder.observe("synthetic", values)
    found = recorder.maxima["synthetic"]
    assert found.device.type == "cpu"
    assert found.dtype == torch.float64
    assert torch.equal(found, expected.abs().amax(dim=0))


@pytest.mark.parametrize("device", ["cpu", "mps"])
def test_reference_logits_preserve_every_finite_bf16_bit_pattern(device: str) -> None:
    values, expected = finite_patterns(device)

    class FixedLogits:
        # Fix the model output so this tests transport independently of GEMM/reduction rounding.
        def __call__(self, *, input_ids: torch.Tensor) -> SimpleNamespace:
            assert input_ids.shape == (1, 255)
            return SimpleNamespace(logits=values[None])

    (found,) = capture.logits_of(cast(Any, FixedLogits()), [[0] * 255], 0, device=device)
    assert found.device.type == "cpu"
    assert found.dtype == torch.float64
    # Compare representations too: ordinary equality hides a lost negative zero.
    assert torch.equal(found.view(torch.int64), expected.view(torch.int64))
