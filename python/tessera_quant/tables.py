# SPDX-License-Identifier: Apache-2.0 OR MIT
"""The lookup tables of `D` and their digests (RFC-0008), computed in Python.

RFC-0008 'Generators and digests' defines every table by integers alone, and three
implementations compute them: `tessera-tables` for the prover, `tv-tables` for the verifier, and
this module, for the integer reference and the front-end. It generates the kinds `D` uses,
`Requant`, `Silu`, `Exp` and RoPE's constants, and each `TableSpec` with its digest, which must
equal the other implementations'.

- `Requant`'s values are `rq` of RFC-0005 'Integer semantics', a closed form: `requant` computes
  it for an array of keys, and the digest binds no value.
- `Exp`'s core is vectorized: its fixed-point steps run on `uint64` arrays with exact 128-bit
  intermediates (`tessera_quant._wide`).
- `Silu`'s core runs on Python's integers.
- RoPE's pairs are vectorized like `Exp`, computed on demand for any positions.

`spec/vectors/tables/`, written by an independent oracle, pins every specification, digest and
value this module computes. A Q64 number `v` stands for `v / 2^64`; every division is floor
division.
"""

from collections.abc import Iterator
from dataclasses import dataclass, field
from enum import Enum
from fractions import Fraction
from typing import TypeAlias

import blake3
import numpy as np
import numpy.typing as npt

from tessera_quant import _wide
from tessera_quant.errors import TableError
from tessera_quant.tce import (
    DIGEST,
    I8,
    I32,
    RATIONAL,
    U8,
    U16,
    U32,
    IntVec,
    Seq,
    Struct,
    Tagged,
    UnitEnum,
    Variant,
    Writer,
)

Ints: TypeAlias = npt.NDArray[np.int64]

LN2 = 0xB172_17F7_D1CF_79AB
"""`⌊ln 2 · 2^64⌋`, pinned in `spec/vectors/tables/constants.json`."""

TAU = 0x6_487E_D511_0B46_11A6
"""`⌊2π · 2^64⌋`, pinned in `spec/vectors/tables/constants.json`."""

HALF_PI = 0x1_921F_B544_42D1_8469
"""`⌊π/2 · 2^64⌋`, pinned in `spec/vectors/tables/constants.json`."""

MAX_CORE = 1 << 24
"""The longest dense core a table may have (RFC-0008 'Structured evaluation')."""

P = 2**64 - 2**32 + 1
"""The Goldilocks prime: table values are field elements in the centered representation."""

_ONE = 1 << 64
_ONE_62 = 1 << 62
_TERMS = 20
_ATANH_TERMS = 22
_TRIG_TERMS = 13
# Keys per vectorized step: a few megabytes of temporaries each.
_CHUNK = 1 << 18


class BitOp(Enum):
    """`Bitwise8`'s operation."""

    Xor = 0
    And = 1


@dataclass(frozen=True)
class Range:
    """`x ∈ [0, 2^bits)`, no value."""

    bits: int


@dataclass(frozen=True)
class Requant:
    """`rq(k, r_c, lo, hi)` for the class `c` and the key `k` of `b` bits (RFC-0005)."""

    b: int
    r_by_class: list[int]
    lo: int
    hi: int


@dataclass(frozen=True)
class Pow2:
    """`2^i` for `i < max_exp`."""

    max_exp: int


@dataclass(frozen=True)
class Bitwise8:
    """`a op b` for two bytes."""

    op: BitOp


@dataclass(frozen=True)
class AesSbox:
    """The AES S-box."""


@dataclass(frozen=True)
class Gf256Mul:
    """`c · x` in GF(2^8)."""

    c: int


@dataclass(frozen=True)
class Exp:
    """`round(2^sigma · e^{-k·scale})` for keys of `key_bits` bits."""

    key_bits: int
    sigma: int
    scale: Fraction


@dataclass(frozen=True)
class Exp2Frac:
    """`⌊2^{f/2^frac_bits} · 2^out_bits⌋`."""

    frac_bits: int
    out_bits: int


@dataclass(frozen=True)
class Silu:
    """`silu(h · 2^in_exp) / 2^out_exp`, rounded and clamped to INT8, for `h = k >> r_c`."""

    b: int
    r_by_class: list[int]
    in_exp: int
    out_exp: int


@dataclass(frozen=True)
class Rope:
    """Llama-3.1's RoPE constants: `(C, S)` per position and frequency."""

    theta: int
    factor: int
    low_freq_factor: int
    high_freq_factor: int
    original_max: int
    bits: int
    max_context: int
    d_head: int


TableKind: TypeAlias = (
    Range | Requant | Pow2 | Bitwise8 | AesSbox | Gf256Mul | Exp | Exp2Frac | Silu | Rope
)
"""A kind of table (RFC-0008 'Table specifications'), its TCE tag its declaration index."""

TABLE_KIND = Tagged[TableKind](
    [
        Variant("Range", Range, [("bits", U8)]),
        Variant("Requant", Requant, [("b", U8), ("r_by_class", Seq(U8)), ("lo", I32), ("hi", I32)]),
        Variant("Pow2", Pow2, [("max_exp", U8)]),
        Variant("Bitwise8", Bitwise8, [("op", UnitEnum(BitOp))]),
        Variant("AesSbox", AesSbox),
        Variant("Gf256Mul", Gf256Mul, [("c", U8)]),
        Variant("Exp", Exp, [("key_bits", U8), ("sigma", U8), ("scale", RATIONAL)]),
        Variant("Exp2Frac", Exp2Frac, [("frac_bits", U8), ("out_bits", U8)]),
        Variant(
            "Silu", Silu, [("b", U8), ("r_by_class", Seq(U8)), ("in_exp", I8), ("out_exp", I8)]
        ),
        Variant(
            "Rope",
            Rope,
            [
                ("theta", U32),
                ("factor", U16),
                ("low_freq_factor", U16),
                ("high_freq_factor", U16),
                ("original_max", U32),
                ("bits", U8),
                ("max_context", U32),
                ("d_head", U16),
            ],
        ),
    ]
)
"""`TableKind`'s TCE codec: tier A, so variants are appended, never reordered."""


@dataclass(frozen=True)
class TableSpec:
    """A table as a claim list names it: its index, kind, `⌈log2 T⌉` and digest (RFC-0008)."""

    id: int
    kind: TableKind
    log2_rows: int
    digest: bytes


TABLE_SPEC = Struct(
    TableSpec, [("id", U32), ("kind", TABLE_KIND), ("log2_rows", U8), ("digest", DIGEST)]
)
"""`TableSpec`'s TCE codec."""


def _refuse(kind: str, rule: str) -> TableError:
    return TableError(f"a {kind} table needs {rule}")


def _ceil_log2(n: int) -> int:
    return (n - 1).bit_length()


# ----- Fixed-point arithmetic on Python integers (RFC-0008 'Fixed-point generators') ----------


def exp_neg(x: int) -> tuple[int, int]:
    """`e^{-x}` for a Q64 `x ≥ 0`: `(y, n)` with `e^{-x/2^64} ≈ y · 2^{-64-n}` and `y ≤ 2^64`.

    With `n = ⌊x / LN2⌋` and `r = x - n·LN2`, Horner's rule over the Taylor series of `e^{-r}`:
    from `y = 2^64`, for `i` from 20 down to 1, `y ← 2^64 - ⌊r·y / (i·2^64)⌋`.
    """
    n, r = divmod(x, LN2)
    y = _ONE
    for i in range(_TERMS, 0, -1):
        y = _ONE - r * y // (i * _ONE)
    return y, n


def ln(n: int) -> int:
    """`ln n` in Q64 for an integer `n ≥ 1`, by `atanh` of `(u - 1)/(u + 1)`, `n = 2^m · u`."""
    m = n.bit_length() - 1
    top = 1 << m
    z = ((n - top) << 64) // (n + top)
    w = (z * z) >> 64
    a = _ONE // (2 * _ATANH_TERMS + 1)
    for k in range(_ATANH_TERMS - 1, -1, -1):
        a = _ONE // (2 * k + 1) + ((w * a) >> 64)
    return m * LN2 + ((z * a) >> 63)


def silu_value(h: int, in_exp: int, out_exp: int) -> int:
    """`Silu`'s value at `h`: `silu(h · 2^in_exp) / 2^out_exp`, rounded half up, clamped to INT8.

    `e^{-|x|}` in Q64 from `exp_neg`, the logistic `sigma(x)` in Q63 from it, then `h · sigma`
    shifted by `s = 63 - in_exp + out_exp` with rounding.
    """
    y, n = exp_neg(abs(h) << (64 + in_exp))
    e = y >> n if n < 128 else 0
    sigma = (1 << 127) // (_ONE + e) if h >= 0 else (e << 63) // (_ONE + e)
    s = 63 - in_exp + out_exp
    return max(-128, min(127, (h * sigma + (1 << (s - 1))) >> s))


def requant(keys: Ints, r: int, lo: int, hi: int) -> Ints:
    """`rq(k, r, lo, hi) = clamp(⌊(k + 2^{r-1}) / 2^r⌋, lo, hi)` (RFC-0005) for keys `|k| < 2^31`.

    The sum stays below `2^62 + 2^31`, and NumPy's shift of a signed integer is floor division.
    """
    shifted = (keys + np.int64(1 << (r - 1))) >> np.int64(r)
    return np.minimum(np.maximum(shifted, np.int64(lo)), np.int64(hi))


# ----- Exp, vectorized --------------------------------------------------------------------------


def _exp_argument(k: int, num: int, den: int) -> int | None:
    """`⌊k · num/den · 2^64⌋`, or `None` once its integer part reaches `2^20`."""
    q, m = divmod(k * num, den)
    if q >= 1 << 20:
        return None
    return (q << 64) + (m << 64) // den


def _zero_tail(rows: int, sigma: int, num: int, den: int) -> int:
    """The first key of `Exp`'s zero tail, or `rows` when it has none.

    The keys with an integer part of `k·scale` of `2^20` or more, or `sigma + 2` halvings or
    more, form a suffix, which a binary search finds.
    """

    def in_tail(k: int) -> bool:
        x = _exp_argument(k, num, den)
        return x is None or x // LN2 >= sigma + 2

    lo, hi = 0, rows
    while lo < hi:
        mid = (lo + hi) // 2
        if in_tail(mid):
            hi = mid
        else:
            lo = mid + 1
    return lo


def _exp_neg_wide(x: _wide.Wide) -> tuple[_wide.U64, _wide.U64]:
    """`exp_neg` on arrays, for `x < LN2 · 2^64`: `(d, n)` with `y = 2^64 - d`.

    `y` may be `2^64`, one past `uint64`, so the loop keeps its deficit `d < 2^64`. From
    `y = 2^64 - d`, `⌊r·y / 2^64⌋ = r - ⌈r·d / 2^64⌉`, and floor division by `i` after it is the
    floor of `r·y / (i·2^64)`: the step `y ← 2^64 - ⌊r·y / (i·2^64)⌋` is `d ← ⌊(r - ⌈r·d/2^64⌉)/i⌋`.
    """
    n, r = _wide.divmod_by(x, LN2)
    d = np.zeros_like(r)
    for i in range(_TERMS, 0, -1):
        hi, lo = _wide.mul(r, d)
        d = (r - hi - (lo != 0).astype(np.uint64)) // np.uint64(i)
    return d, n


def _exp_values(keys: _wide.U64, sigma: int, num: int, den: int) -> Ints:
    """`Exp`'s values at keys below its zero tail.

    `x = q·2^64 + ⌊m·2^64/den⌋` for `k·num = q·den + m`, then `(y·2^sigma + 2^{63+n}) >> (64 + n)`.
    """
    q, m = _wide.divmod_by(_wide.mul(keys, _wide.u64(num)), den)
    fraction, _ = _wide.divmod_by((m, np.zeros_like(m)), den)
    d, n = _exp_neg_wide((q, fraction))
    # With t = sigma - n ≥ 0 the value is (y·2^t + 2^63) >> 64, and y·2^t = 2^{64+t} - d·2^t.
    t = np.int64(sigma) - n.astype(np.int64)
    shift = np.minimum(np.maximum(t, np.int64(0)), np.int64(63)).astype(np.uint64)
    zero = np.zeros_like(d)
    scaled = (d >> (np.uint64(64) - shift), d << shift)
    top = (np.left_shift(np.ones_like(d), shift), zero)
    rounded = _wide.add(_wide.sub(top, scaled), (zero, np.full_like(d, 1 << 63)))
    value = rounded[0].astype(np.int64)
    # t = -1: (y + 2^64) >> 65, so 1 exactly when y = 2^64; t < -1 is the zero tail.
    last = ((t == -1) & (d == 0)).astype(np.int64)
    return last + (value - last) * (t >= 0).astype(np.int64)


# ----- Lookup tables ------------------------------------------------------------------------------


@dataclass(frozen=True, eq=False)
class LookupTable:
    """A lookup table of `D`: its specification, digest included, and the values the digest binds.

    `core` holds `Exp`'s values below its zero tail, or `Silu`'s values at every `h` of every
    class, class by class; it is empty for `Requant`, whose values are a closed form.
    """

    spec: TableSpec
    rows: int
    """`T`, the rows before padding."""
    core: Ints = field(repr=False)

    def lookup(self, keys: Ints, cls: int = 0) -> Ints:
        """The values that lookups of `keys` in class `cls` return.

        Raises:
            TableError: for a key or class outside the table, which an honest witness never has.
        """
        kind = self.spec.kind
        if isinstance(kind, Requant):
            return requant(_signed_keys(keys, kind.b, cls, kind.r_by_class), *_class(kind, cls))
        if isinstance(kind, Silu):
            _signed_keys(keys, kind.b, cls, kind.r_by_class)
            start = sum(1 << (kind.b - r) for r in kind.r_by_class[:cls])
            r = kind.r_by_class[cls]
            index = (keys >> np.int64(r)) + np.int64(start + (1 << (kind.b - 1 - r)))
            return self.core[index]
        if isinstance(kind, Exp):
            if cls != 0 or (keys.size > 0 and (keys.min() < 0 or keys.max() >= self.rows)):
                raise TableError("an Exp lookup needs class 0 and a key in [0, 2^key_bits)")
            # Past the core, the zero tail.
            inside = (keys < len(self.core)).astype(np.int64)
            return self.core[np.minimum(keys, np.int64(len(self.core) - 1))] * inside
        raise TableError(f"tessera_quant.tables generates no {type(kind).__name__} lookups")


def _signed_keys(keys: Ints, b: int, cls: int, r_by_class: list[int]) -> Ints:
    half = 1 << (b - 1)
    if not 0 <= cls < len(r_by_class) or (
        keys.size > 0 and (keys.min() < -half or keys.max() >= half)
    ):
        raise TableError("a lookup needs a class of the table and a key in [-2^{b-1}, 2^{b-1})")
    return keys


def _class(kind: Requant, cls: int) -> tuple[int, int, int]:
    return kind.r_by_class[cls], kind.lo, kind.hi


def lookup_table(table_id: int, kind: Requant | Silu | Exp) -> LookupTable:
    """The table of `kind` with index `table_id`, as RFC-0008 'Table specifications' defines it.

    Its parameters are checked, its core generated, and its `log2_rows` and digest computed.

    Raises:
        TableError: for parameters the kind does not allow.
    """
    if isinstance(kind, Requant):
        rows = _classes_rows("Requant", kind.b, kind.r_by_class)
        if any(not 1 <= r <= 63 for r in kind.r_by_class):
            raise _refuse("Requant", "1 ≤ r_c ≤ 63 for every class")
        if kind.lo > kind.hi:
            raise _refuse("Requant", "lo ≤ hi")
        return _finish(table_id, kind, rows, [], np.zeros(0, dtype=np.int64))
    if isinstance(kind, Silu):
        rows = _classes_rows("Silu", kind.b, kind.r_by_class)
        if any(r >= kind.b for r in kind.r_by_class):
            raise _refuse("Silu", "r_c < b for every class")
        if sum(1 << (kind.b - r) for r in kind.r_by_class) > MAX_CORE:
            raise _refuse("Silu", "at most 2^24 values of h over the classes")
        if not -64 <= kind.in_exp <= 0:
            raise _refuse("Silu", "-64 ≤ in_exp ≤ 0")
        if not 1 <= 63 - kind.in_exp + kind.out_exp <= 126:
            raise _refuse("Silu", "1 ≤ 63 - in_exp + out_exp ≤ 126")
        values = [
            silu_value(h, kind.in_exp, kind.out_exp)
            for r in kind.r_by_class
            for h in range(-(1 << (kind.b - 1 - r)), 1 << (kind.b - 1 - r))
        ]
        return _finish(table_id, kind, rows, [], np.array(values, dtype=np.int64))
    if not 1 <= kind.key_bits <= 32:
        raise _refuse("Exp", "1 ≤ key_bits ≤ 32")
    if kind.sigma > 32:
        raise _refuse("Exp", "sigma ≤ 32")
    if kind.scale <= 0:
        raise _refuse("Exp", "a positive scale")
    rows = 1 << kind.key_bits
    num, den = kind.scale.numerator, kind.scale.denominator
    tail = _zero_tail(rows, kind.sigma, num, den)
    if tail > MAX_CORE:
        raise _refuse("Exp", "a zero tail from key 2^24 or earlier")
    core = np.empty(tail, dtype=np.int64)
    for start in range(0, tail, _CHUNK):
        keys = np.arange(start, min(start + _CHUNK, tail), dtype=np.uint64)
        core[start : start + len(keys)] = _exp_values(keys, kind.sigma, num, den)
    return _finish(table_id, kind, rows, [tail], core)


def _classes_rows(kind: str, b: int, r_by_class: list[int]) -> int:
    classes = len(r_by_class)
    if not 1 <= classes <= 4:
        raise _refuse(kind, "one to four classes")
    if b < 1 or b + _ceil_log2(classes) > 32:
        raise _refuse(kind, "b ≥ 1 and b + ⌈log2 n⌉ ≤ 32 for n classes")
    return classes << b


def _finish(
    table_id: int, kind: Requant | Silu | Exp, rows: int, pieces: list[int], core: Ints
) -> LookupTable:
    log2_rows = _ceil_log2(rows)
    digest = _digest(table_id, kind, log2_rows, pieces, len(core), iter([core]))
    return LookupTable(TableSpec(table_id, kind, log2_rows, digest), rows, core)


def _field_elements(values: Ints) -> _wide.U64:
    """True integers as the canonical `Fp` of the centered representation: `v mod P`."""
    unsigned = values.astype(np.uint64)
    # For a negative v, as uint64 it is v + 2^64; v + P is that minus 2^32 - 1.
    return unsigned - (values < 0).astype(np.uint64) * np.uint64(2**32 - 1)


def _digest(
    table_id: int,
    kind: TableKind,
    log2_rows: int,
    pieces: list[int],
    core_len: int,
    core: Iterator[Ints],
) -> bytes:
    """`H("table", TCE(id, kind, log2_rows) ‖ TCE(pieces) ‖ TCE(core))` (RFC-0008, RFC-0004).

    The pieces are a `Vec<u64>`, and the core a `Vec<Fp>` streamed chunk by chunk.
    """
    hasher = blake3.blake3(derive_key_context="tessera/v1/table")
    out = Writer(hasher.update)
    U32.encode(table_id, out)
    TABLE_KIND.encode(kind, out)
    U8.encode(log2_rows, out)
    IntVec(np.uint64).encode(np.array(pieces, dtype=np.uint64), out)
    out.length(core_len)
    for chunk in core:
        out.put(_field_elements(chunk).data)
    return hasher.digest()


# ----- RoPE ---------------------------------------------------------------------------------------


@dataclass(frozen=True)
class RopeConstants:
    """RoPE's constants (RFC-0008 'Fixed-point generators').

    Its scaled frequencies `ω_i` in Q64, and its pairs `(C, S)`, which `pairs` computes on demand.
    """

    kind: Rope
    frequencies: list[int]
    """`ω_i`, at most `2^64`: the intermediate RFC-0008 pins."""

    @staticmethod
    def of(kind: Rope) -> "RopeConstants":
        """The constants of `kind`, its parameters checked: the frequencies computed once.

        Raises:
            TableError: for parameters RFC-0008 does not allow.
        """
        if kind.theta < 2:
            raise _refuse("Rope", "θ ≥ 2")
        if kind.factor < 1 or not 1 <= kind.low_freq_factor < kind.high_freq_factor:
            raise _refuse("Rope", "factor ≥ 1 and 1 ≤ low_freq_factor < high_freq_factor")
        if kind.original_max < 1:
            raise _refuse("Rope", "original_max ≥ 1")
        if not 1 <= kind.bits <= 30:
            raise _refuse("Rope", "1 ≤ bits ≤ 30")
        if kind.d_head < 2 or kind.d_head % 2 != 0:
            raise _refuse("Rope", "an even d_head of at least 2")
        if kind.max_context < 1 or kind.max_context * (kind.d_head // 2) > MAX_CORE:
            raise _refuse("Rope", "1 ≤ max_context and max_context · d_head / 2 ≤ 2^24")
        ln_theta = ln(kind.theta)
        return RopeConstants(kind, [_omega(kind, i, ln_theta) for i in range(kind.d_head // 2)])

    def pairs(self, positions: Ints) -> tuple[Ints, Ints]:
        """`(C, S)` at every position of `positions` and every frequency.

        Each has shape `(len(positions), d_head / 2)`: `cos_sin(p·ω_i)` rounded half up to `bits`.
        """
        if positions.size and (positions.min() < 0 or positions.max() >= self.kind.max_context):
            raise TableError("a RoPE position needs to be in [0, max_context)")
        p = positions.astype(np.uint64)
        shape = (len(positions), len(self.frequencies))
        cos, sin = np.empty(shape, dtype=np.int64), np.empty(shape, dtype=np.int64)
        for i, omega in enumerate(self.frequencies):
            cos[:, i], sin[:, i] = _cos_sin(_angle(p, omega))
        # (v·2^bits + 2^61) >> 62 = (v + 2^{61-bits}) >> (62 - bits), with no 128-bit product.
        shift, half = np.int64(62 - self.kind.bits), np.int64(1 << (61 - self.kind.bits))
        return (cos + half) >> shift, (sin + half) >> shift


def _omega(kind: Rope, i: int, ln_theta: int) -> int:
    """`ω_i = θ^{-2i/d_head}` in Q64, scaled in Llama-3's three bands by exact comparisons."""
    y, n = exp_neg(2 * i * ln_theta // kind.d_head)
    omega = y >> n if n < 128 else 0
    l_omega = kind.original_max * omega
    low, high, factor = kind.low_freq_factor, kind.high_freq_factor, kind.factor
    if TAU * high < l_omega:
        return omega
    if TAU * low > l_omega:
        return omega // factor
    smooth = ((l_omega - low * TAU) << 40) // ((high - low) * TAU)
    return omega // factor + ((omega * smooth) >> 40) * (factor - 1) // factor


def _angle(p: _wide.U64, omega: int) -> _wide.Wide:
    """`p·ω` for `ω ≤ 2^64`: `ω`'s bit 64, set only for `ω = 2^64`, adds `p` to the high word."""
    hi, lo = _wide.mul(p, _wide.u64(omega & (_ONE - 1)))
    return hi + p * np.uint64(omega >> 64), lo


def _cos_sin(angle: _wide.Wide) -> tuple[Ints, Ints]:
    """`cos_sin` of RFC-0008 on arrays: `(cos θ, sin θ)` in Q62 for Q64 angles below `2^100`.

    The angle is reduced modulo `TAU` with a quotient below `2^36`, estimated in floating point
    and corrected exactly, then to its quadrant: `TAU` is 2 above `4·HALF_PI`, so a fifth quadrant
    of two values exists, which `quadrant mod 4` folds onto the first. Horner's rule keeps every
    operand non-negative but the last `cos`, so `uint64` products and floor divisions suffice.
    """
    zero = np.zeros_like(angle[0])
    estimate = (angle[0].astype(np.float64) * 2.0**64 + angle[1].astype(np.float64)) / float(TAU)
    turns = estimate.astype(np.uint64)
    turn = _wide.sub(angle, _times(turns, TAU))
    below = _wide.negative(turn)
    turn = _wide.add(turn, _times(below.astype(np.uint64), TAU))
    above = _wide.at_least(turn, _wide.wide(TAU, zero))
    turn = _wide.sub(turn, _times(above.astype(np.uint64), TAU))
    quadrant = sum(
        (
            _wide.at_least(turn, _wide.wide(k * HALF_PI, zero)).astype(np.uint64)
            for k in range(1, 5)
        ),
        zero,
    )
    within = _wide.sub(turn, _times(quadrant, HALF_PI))
    r = _wide.shift_right(within, 2)[1]
    s = _wide.shift_right(_wide.mul(r, r), 62)[1]
    c = t = np.full_like(r, _ONE_62)
    cos = np.zeros_like(r, dtype=np.int64)
    for k in range(_TRIG_TERMS, 0, -1):
        product = _wide.shift_right(_wide.mul(s, c), 62)[1] // np.uint64((2 * k - 1) * (2 * k))
        cos = np.int64(_ONE_62) - product.astype(np.int64)
        # Non-negative before the last step: the terms after the first are below 1/4.
        c = cos.astype(np.uint64)
        t = np.uint64(_ONE_62) - (
            _wide.shift_right(_wide.mul(s, t), 62)[1] // np.uint64((2 * k) * (2 * k + 1))
        )
    sin = _wide.shift_right(_wide.mul(r, t), 62)[1].astype(np.int64)
    # Quadrants 0 to 3 give (c, s), (-s, c), (-c, -s), (s, -c): exactly one term survives.
    folded = quadrant % np.uint64(4)
    q0, q1, q2, q3 = ((folded == k).astype(np.int64) for k in range(4))
    return cos * (q0 - q2) + sin * (q3 - q1), sin * (q0 - q2) + cos * (q1 - q3)


def _times(n: _wide.U64, constant: int) -> _wide.Wide:
    """`n · constant` for a constant below `2^128` and a product below `2^128`."""
    hi, lo = _wide.mul(n, _wide.u64(constant & (_ONE - 1)))
    return hi + n * np.uint64(constant >> 64), lo


@dataclass(frozen=True)
class RopeTable:
    """RoPE's constants with their specification: the digest binds every pair, in row order."""

    spec: TableSpec
    constants: RopeConstants


def rope_table(table_id: int, kind: Rope) -> RopeTable:
    """The RoPE table of `kind` with index `table_id`, its digest over all its pairs streamed.

    Raises:
        TableError: for parameters RFC-0008 does not allow.
    """
    constants = RopeConstants.of(kind)
    pairs = kind.max_context * (kind.d_head // 2)
    log2_rows = _ceil_log2(pairs)
    step = max(1, _CHUNK // (kind.d_head // 2))

    def chunks() -> Iterator[Ints]:
        for start in range(0, kind.max_context, step):
            positions = np.arange(start, min(start + step, kind.max_context), dtype=np.int64)
            cos, sin = constants.pairs(positions)
            interleaved = np.empty((*cos.shape, 2), dtype=np.int64)
            interleaved[..., 0], interleaved[..., 1] = cos, sin
            yield interleaved.reshape(-1)

    digest = _digest(table_id, kind, log2_rows, [], 2 * pairs, chunks())
    return RopeTable(TableSpec(table_id, kind, log2_rows, digest), constants)
