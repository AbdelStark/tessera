# SPDX-License-Identifier: Apache-2.0 OR MIT
"""Exact 128-bit arithmetic on arrays of unsigned 64-bit integers, for the table generators.

NumPy has no 128-bit integer. A `Wide` value is a pair `(hi, lo)` of `uint64` arrays standing for
`hi·2^64 + lo` modulo `2^128`: a negative value is its two's complement, and its sign is `hi`'s
top bit. Every operation is exact. `divmod_by` estimates its quotient in floating point and
corrects it with exact remainders, so floating point decides nothing but where to start.
"""

from typing import TypeAlias

import numpy as np
import numpy.typing as npt

U64: TypeAlias = npt.NDArray[np.uint64]
Wide: TypeAlias = tuple[U64, U64]

_MASK32 = np.uint64(0xFFFF_FFFF)
_32 = np.uint64(32)
_63 = np.uint64(63)
_TWO64 = 2.0**64
# The largest float below 2^64 that converts to `uint64` exactly: estimates are cut to it.
_BELOW_TWO64 = 2.0**64 - 2.0**12


def u64(x: int) -> np.uint64:
    """A Python integer in `[0, 2^64)` as a NumPy scalar."""
    return np.uint64(x)


def wide(x: int, like: U64) -> Wide:
    """The integer `x` in `[0, 2^128)` broadcast to the shape of `like`."""
    return (np.full_like(like, x >> 64), np.full_like(like, x & (2**64 - 1)))


def mul(a: U64, b: U64 | np.uint64) -> Wide:
    """The full product `a·b` of two `uint64`s, by 32-bit halves: no partial product overflows."""
    a0, a1 = a & _MASK32, a >> _32
    b0, b1 = b & _MASK32, b >> _32
    low, cross_a, cross_b, high = a0 * b0, a0 * b1, a1 * b0, a1 * b1
    middle = (low >> _32) + (cross_a & _MASK32) + (cross_b & _MASK32)
    lo = (low & _MASK32) | (middle << _32)
    hi = high + (cross_a >> _32) + (cross_b >> _32) + (middle >> _32)
    return hi, lo


def add(a: Wide, b: Wide) -> Wide:
    """`a + b` modulo `2^128`."""
    lo = a[1] + b[1]
    return a[0] + b[0] + (lo < a[1]).astype(np.uint64), lo


def sub(a: Wide, b: Wide) -> Wide:
    """`a - b` modulo `2^128`."""
    return a[0] - b[0] - (a[1] < b[1]).astype(np.uint64), a[1] - b[1]


def negative(a: Wide) -> npt.NDArray[np.bool_]:
    """Whether `a`, read as a signed 128-bit integer, is below 0."""
    return (a[0] >> _63).astype(np.bool_)


def at_least(a: Wide, b: Wide) -> npt.NDArray[np.bool_]:
    """Whether `a ≥ b`, both read as unsigned 128-bit integers."""
    return (a[0] > b[0]) | ((a[0] == b[0]) & (a[1] >= b[1]))


def shift_right(a: Wide, s: int) -> Wide:
    """`a >> s` for an unsigned `a` and `0 ≤ s < 64`."""
    if s == 0:
        return a
    right, left = np.uint64(s), np.uint64(64 - s)
    return a[0] >> right, (a[1] >> right) | (a[0] << left)


def select(condition: npt.NDArray[np.bool_], a: U64, b: U64) -> U64:
    """`a` where `condition` holds and `b` elsewhere, by masks: no branch on the values."""
    mask = np.uint64(0) - condition.astype(np.uint64)
    return b ^ ((a ^ b) & mask)


def divmod_by(a: Wide, d: int) -> tuple[U64, U64]:
    """`(⌊a / d⌋, a mod d)` for `1 ≤ d < 2^64` and `0 ≤ a < d·2^64`, so the quotient fits 64 bits.

    The quotient is first estimated in floating point, within `2^13` of the truth. The exact
    remainder `a - q·d`, signed and below `2^77` in size, gives a second estimate of the
    correction, within one, and a last exact step either way settles it.
    """
    dv, df = u64(d), float(d)
    estimate = (a[0].astype(np.float64) * _TWO64 + a[1].astype(np.float64)) / df
    q = np.minimum(estimate, _BELOW_TWO64).astype(np.uint64)
    rem = sub(a, mul(q, dv))
    # The remainder's magnitude, converted, then signed: converting its two's complement words
    # would cancel, `-2^64 + (2^64 - 406)` rounding to 0.
    sign = negative(rem)
    zero = np.zeros_like(q)
    flipped = sub((zero, zero), rem)
    magnitude = (select(sign, flipped[0], rem[0]), select(sign, flipped[1], rem[1]))
    size = magnitude[0].astype(np.float64) * _TWO64 + magnitude[1].astype(np.float64)
    rem_float = size * (1.0 - 2.0 * sign.astype(np.float64))
    correction = np.floor(rem_float / df).astype(np.int64)
    q = (q.view(np.int64) + correction).view(np.uint64)
    step = mul(np.abs(correction).astype(np.uint64), dv)
    plus, minus = add(rem, step), sub(rem, step)
    down = correction < 0
    rem = (select(down, plus[0], minus[0]), select(down, plus[1], minus[1]))
    below = negative(rem)
    q = q - below.astype(np.uint64)
    rem = add(rem, (zero, below.astype(np.uint64) * dv))
    above = ~negative(rem) & ((rem[0] > 0) | (rem[1] >= dv))
    q = q + above.astype(np.uint64)
    rem = sub(rem, (zero, above.astype(np.uint64) * dv))
    return q, rem[1]
