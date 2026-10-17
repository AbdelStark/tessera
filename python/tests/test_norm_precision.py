# SPDX-License-Identifier: Apache-2.0 OR MIT
"""The precision kappa of `D`'s norms (RFC-0023).

`norm_kappa` is the largest kappa, at most 31, whose root `floor(sqrt(d * 2^(2 kappa) / e))` has
at most 32 bits, the one lookup of RFC-0022's range check: checked against its defining
inequalities over random widths and epsilon terms, at its cap, and at Llama-3.1's shapes.
"""

from math import isqrt

import pytest
from hypothesis import given
from hypothesis import strategies as st

from tessera_quant.integer_model import norm_kappa


def root_bits(d_model: int, eps: int, kappa: int) -> int:
    """The bit length of the largest root of a row: RFC-0022's `w`."""
    return isqrt((d_model << (2 * kappa)) // eps).bit_length()


@given(st.integers(1, (1 << 32) - 1), st.integers(1, 1 << 40))
def test_kappa_is_the_largest_whose_root_fits_one_lookup(d_model: int, eps: int) -> None:
    kappa = norm_kappa(d_model, eps)
    # d < 2^32 always lets kappa = 16 fit.
    assert 16 <= kappa <= 31
    assert root_bits(d_model, eps, kappa) <= 32
    if kappa < 31:
        assert root_bits(d_model, eps, kappa + 1) > 32


def test_llama_shapes_take_thirty_and_small_ratios_the_cap() -> None:
    # e = ceil(d * 10^-5 * 2^14), a residual scale of 2^-7: 672 at 8B, 2685 at 405B.
    for d_model, eps in ((4096, 672), (16384, 2685)):
        assert norm_kappa(d_model, eps) == 30
        assert root_bits(d_model, eps, 30) == 32
    # d / e < 4: kappa = 31 keeps the root within 32 bits.
    assert norm_kappa(4096, 1025) == 31
    assert norm_kappa(4096, 1024) == 30
    # The widest residual with e = 1: d * 4^kappa < 2^64.
    assert norm_kappa((1 << 32) - 1, 1) == 16


def test_kappa_refuses_widths_and_epsilons_out_of_range() -> None:
    width, epsilon = r"width lies in \[1, 2\^32\)", "epsilon term is at least 1"
    for d_model, eps, message in ((0, 1, width), (1 << 32, 1, width), (16, 0, epsilon)):
        with pytest.raises(ValueError, match=message):
            norm_kappa(d_model, eps)
