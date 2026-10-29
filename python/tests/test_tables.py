# SPDX-License-Identifier: Apache-2.0 OR MIT
"""The table generators of `D` (RFC-0008), against the independent oracle's vectors.

Every `Requant`, `Silu`, `Exp` and RoPE vector regenerates to its specification's exact TCE bytes,
digest included, with its pieces, every listed value, every lookup at the edges of its key range
(a key outside is refused), and RoPE's frequencies and 4,008 pairs of Llama-3.1's 131,072-position
table. The exact 128-bit arithmetic under the vectorized generators equals Python's integers on
random and extreme operands, and the vectorized `exp_neg` and `cos_sin` equal the definitions run
on Python's integers, at the fifth quadrant that `TAU = 4·HALF_PI + 2` leaves too. A `2^24`-key
`Exp` core generates within the 30 s the issue allows.
"""

import time
from fractions import Fraction
from typing import Any

import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st
from tce_vectors import load

from tessera_quant import (
    TableError,
    _wide,  # pyright: ignore[reportPrivateUsage]
    tables,
)
from tessera_quant.tables import (
    HALF_PI,
    LN2,
    TABLE_SPEC,
    TAU,
    Exp,
    Requant,
    Rope,
    RopeConstants,
    Silu,
    lookup_table,
    rope_table,
    silu_lookup,
)
from tessera_quant.tce import encode

VECTORS = load("tables/tables.json")
KINDS = ("Requant", "Silu", "Exp", "Rope")
CASES = [c for c in VECTORS["cases"] if next(iter(c["kind"])) in KINDS]


def kind_of(entry: dict[str, Any]) -> Requant | Silu | Exp | Rope:
    ((name, fields),) = entry.items()
    if name == "Requant":
        return Requant(**fields)
    if name == "Silu":
        return Silu(**fields)
    if name == "Exp":
        num, den = fields["scale"]
        return Exp(fields["key_bits"], fields["sigma"], Fraction(num, den))
    return Rope(**fields)


def table_id(case: dict[str, Any]) -> int:
    return int.from_bytes(bytes.fromhex(case["spec"])[:4], "little")


def test_every_kind_of_d_has_vectors() -> None:
    assert sorted({next(iter(c["kind"])) for c in CASES}) == sorted(KINDS)
    assert len(CASES) == 21


@pytest.mark.parametrize("case", CASES, ids=lambda c: c["name"])
def test_each_vector_regenerates_to_its_bytes_values_and_lookups(case: dict[str, Any]) -> None:
    kind = kind_of(case["kind"])
    if isinstance(kind, Rope):
        rope = rope_table(table_id(case), kind)
        assert encode(TABLE_SPEC, rope.spec) == bytes.fromhex(case["spec"])
        assert rope.constants.frequencies == [int(f, 16) for f in case["frequencies"]]
        if "values" in case:
            cos, sin = rope.constants.pairs(np.arange(kind.max_context, dtype=np.int64))
            pairs = zip(cos.reshape(-1).tolist(), sin.reshape(-1).tolist(), strict=True)
            assert [[c, s] for c, s in pairs] == case["values"]
        return
    table = lookup_table(table_id(case), kind)
    assert encode(TABLE_SPEC, table.spec) == bytes.fromhex(case["spec"])
    assert table.spec.digest.hex() == case["digest"]
    assert (table.rows, table.spec.log2_rows) == (case["rows"], case["log2_rows"])
    assert ([len(table.core)] if isinstance(kind, Exp) else []) == case["pieces"]
    if "values" in case:
        assert values_of(table) == case["values"]
    for lookup in case["lookups"]:
        keys = np.array([lookup["key"]], dtype=np.int64)
        if lookup["value"] is None:
            with pytest.raises(TableError, match="lookup needs"):
                table.lookup(keys, lookup["class"])
        else:
            assert table.lookup(keys, lookup["class"]).tolist() == [lookup["value"]], lookup


def values_of(table: tables.LookupTable) -> list[int]:
    """Every row's value, row `j` being class `⌊j/2^b⌋` and key `(j mod 2^b) - 2^{b-1}`."""
    kind = table.spec.kind
    if isinstance(kind, Exp):
        return table.lookup(np.arange(table.rows, dtype=np.int64)).tolist()
    assert isinstance(kind, Requant | Silu)
    keys = np.arange(1 << kind.b, dtype=np.int64) - (1 << (kind.b - 1))
    return [v for c in range(len(kind.r_by_class)) for v in table.lookup(keys, c).tolist()]


def test_requested_silu_lookups_match_independent_vectors() -> None:
    for case in CASES:
        kind = kind_of(case["kind"])
        if not isinstance(kind, Silu):
            continue
        for lookup in case["lookups"]:
            keys = np.array([lookup["key"]], dtype=np.int64)
            if lookup["value"] is None:
                with pytest.raises(TableError, match="lookup needs"):
                    silu_lookup(kind, keys, lookup["class"])
            else:
                assert silu_lookup(kind, keys, lookup["class"]).tolist() == [lookup["value"]]


@pytest.mark.parametrize("in_exp", [-64, -3, 0])
@pytest.mark.parametrize("out_exp", [-2, -1])
def test_requested_silu_lookups_equal_complete_dense_domains(in_exp: int, out_exp: int) -> None:
    # All signed keys, three different classes including shift zero, and unaligned/empty views.
    kind = Silu(6, [0, 1, 5], in_exp, out_exp)
    dense = lookup_table(0, kind)
    full = np.arange(-32, 32, dtype=np.int64).reshape(4, 16)
    for cls in range(3):
        for keys in [full, full[:, ::3], full[:, :0], full.reshape(-1)[:13]]:
            found = silu_lookup(kind, keys, cls)
            assert found.dtype == np.int64
            assert found.shape == keys.shape
            assert found.tolist() == dense.lookup(keys, cls).tolist()


@pytest.mark.parametrize(
    "kind",
    [
        Silu(6, [], -2, -2),
        Silu(31, [1, 2, 3], -2, -2),
        Silu(6, [-1], -2, -2),
        Silu(6, [6], -2, -2),
        Silu(30, [5], -2, -2),
        Silu(6, [1], -65, -2),
        Silu(6, [1], 1, -2),
        Silu(6, [1], 0, -63),
        Silu(6, [1], -64, 0),
    ],
)
def test_sparse_and_dense_silu_refuse_parameters_before_generating_values(
    kind: Silu, monkeypatch: pytest.MonkeyPatch
) -> None:
    def forbidden(*args: Any) -> int:
        pytest.fail("an invalid SiLU generated a value")

    monkeypatch.setattr(tables, "silu_value", forbidden)
    with pytest.raises(TableError) as dense:
        lookup_table(0, kind)
    with pytest.raises(TableError) as sparse:
        silu_lookup(kind, np.zeros(0, dtype=np.int64))
    assert sparse.value.code == dense.value.code == "ERR-INTERNAL"
    assert str(sparse.value) == str(dense.value)


@pytest.mark.parametrize(("key", "cls"), [(-33, 0), (32, 0), (0, -1), (0, 3)])
def test_requested_silu_lookup_checks_signed_key_and_class_before_values(
    monkeypatch: pytest.MonkeyPatch, key: int, cls: int
) -> None:
    def forbidden(*args: Any) -> int:
        pytest.fail("an invalid SiLU lookup generated a value")

    monkeypatch.setattr(tables, "silu_value", forbidden)
    with pytest.raises(TableError, match="a lookup needs a class") as refused:
        silu_lookup(Silu(6, [0, 1, 5], -3, -2), np.array([key], dtype=np.int64), cls)
    assert refused.value.code == "ERR-INTERNAL"


def test_requested_silu_work_is_bounded_by_distinct_shifted_keys(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    kind = Silu(24, [1], -3, -2)  # A dense core would evaluate 2^23 values.
    keys = np.arange(970, dtype=np.int64).reshape(97, 10) % 10 - 4
    calls: list[int] = []
    original = tables.silu_value

    def counted(h: int, in_exp: int, out_exp: int) -> int:
        calls.append(h)
        assert len(calls) <= 5, "SiLU evaluated unrequested or repeated shifted keys"
        return original(h, in_exp, out_exp)

    monkeypatch.setattr(tables, "silu_value", counted)
    found = silu_lookup(kind, keys)
    assert calls == [-2, -1, 0, 1, 2]
    expected = [[original(int(k) // 2, -3, -2) for k in row] for row in keys]
    assert found.tolist() == expected
    calls.clear()
    assert silu_lookup(kind, keys).tolist() == expected
    assert calls == [-2, -1, 0, 1, 2], "a later call reused a cache indexed by secret keys"


def test_llama_3_1s_rope_constants_are_the_oracles_at_every_sampled_pair() -> None:
    entry = VECTORS["rope_llama31_131072"]
    kind = kind_of(entry["kind"])
    assert isinstance(kind, Rope)
    constants = RopeConstants.of(kind)
    samples = entry["samples"]
    assert len(samples) == 4008
    positions = np.array([s["position"] for s in samples], dtype=np.int64)
    cos, sin = constants.pairs(positions)
    found = [
        [int(cos[i, s["frequency"]]), int(sin[i, s["frequency"]])] for i, s in enumerate(samples)
    ]
    assert found == [s["constants"] for s in samples]


def test_parameters_outside_rfc_0008_are_refused() -> None:
    for kind, rule in [
        (Requant(4, [], 0, 1), "one to four classes"),
        (Requant(31, [1, 1, 1], 0, 1), "b ≥ 1 and b \\+ ⌈log2 n⌉ ≤ 32"),
        (Requant(4, [64], 0, 1), "1 ≤ r_c ≤ 63"),
        (Requant(4, [1], 2, 1), "lo ≤ hi"),
        (Silu(4, [4], 0, 0), "r_c < b"),
        (Silu(30, [5], 0, 0), "at most 2\\^24 values"),
        (Silu(8, [1], 1, 0), "-64 ≤ in_exp ≤ 0"),
        (Exp(33, 16, Fraction(1, 2)), "1 ≤ key_bits ≤ 32"),
        (Exp(8, 33, Fraction(1, 2)), "sigma ≤ 32"),
        (Exp(32, 16, Fraction(1, 2**32)), "a zero tail from key 2\\^24 or earlier"),
    ]:
        with pytest.raises(TableError, match=rule) as refused:
            lookup_table(0, kind)
        assert refused.value.code == "ERR-INTERNAL"
    with pytest.raises(TableError, match="an even d_head"):
        RopeConstants.of(Rope(10000, 1, 1, 4, 32, 14, 16, 7))


def test_a_two_to_the_24_key_exp_core_generates_within_30_seconds() -> None:
    kind = Exp(key_bits=24, sigma=16, scale=Fraction(1, 1 << 25))
    start = time.perf_counter()
    table = lookup_table(0, kind)
    elapsed = time.perf_counter() - start
    assert len(table.core) == 1 << 24, "no zero tail: every key is in the core"
    assert elapsed < 30, f"{elapsed:.1f} s"
    # The vectorized core is the definition, run on Python's integers, at keys across it.
    for k in [0, 1, 2**12 + 7, 2**23 - 1, 2**24 - 1]:
        assert table.core[k] == reference_exp(k, kind), k


def reference_exp(k: int, kind: Exp) -> int:
    """RFC-0008's `Exp` value of key `k`, on Python's integers."""
    q, m = divmod(k * kind.scale.numerator, kind.scale.denominator)
    if q >= 1 << 20:
        return 0
    y, n = tables.exp_neg((q << 64) + (m << 64) // kind.scale.denominator)
    if n >= kind.sigma + 2:
        return 0
    return ((y << kind.sigma) + (1 << (63 + n))) >> (64 + n)


U64 = st.integers(0, 2**64 - 1)


def array(values: list[int]) -> _wide.U64:
    return np.array(values, dtype=np.uint64)


def joined(w: _wide.Wide) -> list[int]:
    return [(int(h) << 64) | int(lo) for h, lo in zip(w[0].tolist(), w[1].tolist(), strict=True)]


EXTREMES = [0, 1, 2**32 - 1, 2**32, 2**63, 2**64 - 1]


@given(st.lists(st.tuples(U64, U64), min_size=1, max_size=64))
def test_wide_products_sums_and_differences_are_python_integers(
    pairs: list[tuple[int, int]],
) -> None:
    pairs += [(a, b) for a in EXTREMES for b in EXTREMES]
    a, b = array([x for x, _ in pairs]), array([y for _, y in pairs])
    assert joined(_wide.mul(a, b)) == [x * y for x, y in pairs]
    zero = np.zeros_like(a)
    assert joined(_wide.add((a, b), (b, a))) == [
        ((x << 64) + y + (y << 64) + x) % 2**128 for x, y in pairs
    ]
    assert joined(_wide.sub((zero, a), (zero, b))) == [(x - y) % 2**128 for x, y in pairs]


# Divisions whose floating-point estimates leave a last step up, then down, to the exact quotient.
LAST_STEPS = [
    (0x7840F89719E81694C09314BBBB1CA4EE, 0xFA529BA3FE3BFADB),
    (0x1ABBDDE5077AA4DD03B1BA681AE9EB, 0x460D692ED65412),
    (0x5369E23BC699D7121B7D65AA047123C7, 0xBB2D420F0F88080C),
    (0xFF34B43DA23B40E47DE29AA72760B63, 0x19F9919C895FD7B4),
]


def check_division(values: list[int], d: int) -> None:
    hi = array([v >> 64 for v in values])
    lo = array([v & (2**64 - 1) for v in values])
    q, r = _wide.divmod_by((hi, lo), d)
    assert list(zip(q.tolist(), r.tolist(), strict=True)) == [divmod(v, d) for v in values]


@given(
    st.integers(1, 2**64 - 1),
    st.lists(st.integers(0, 2**128 - 1), min_size=1, max_size=64),
    st.lists(st.integers(0, 2**64 - 1), min_size=1, max_size=16),
)
def test_wide_division_is_python_integer_division(
    d: int, values: list[int], quotients: list[int]
) -> None:
    # Below d·2^64, so the quotient fits 64 bits; exact multiples and their neighbours are where
    # the estimates fall on the wrong side of an integer.
    near = [q * d + s for q in quotients for s in (0, 1, d - 1)]
    edges = [0, d - 1, d, (d << 64) - 1]
    check_division([v % (d << 64) for v in values + near] + edges, d)


def test_wide_division_takes_its_last_steps_where_the_estimates_need_them() -> None:
    for value, d in LAST_STEPS:
        check_division([value], d)


@settings(max_examples=50)
@given(st.lists(st.integers(0, LN2 * 2**64 - 1), min_size=1, max_size=32))
def test_the_vectorized_exp_neg_is_its_definition(xs: list[int]) -> None:
    xs += [0, 1, LN2 - 1, LN2, 20 * LN2, LN2 * 2**64 - 1]
    d, n = tables._exp_neg_wide((array([x >> 64 for x in xs]), array([x % 2**64 for x in xs])))  # pyright: ignore[reportPrivateUsage]
    found = [(2**64 - int(di), int(ni)) for di, ni in zip(d.tolist(), n.tolist(), strict=True)]
    assert found == [tables.exp_neg(x) for x in xs]


def reference_cos_sin(angle: int) -> tuple[int, int]:
    """RFC-0008's `cos_sin` on Python's integers, written from its text."""
    turn = angle % TAU
    quadrant = turn // HALF_PI
    r = (turn - quadrant * HALF_PI) >> 2
    one = 1 << 62
    s = (r * r) >> 62
    c = t = one
    for k in range(13, 0, -1):
        c = one - (s * c) // ((2 * k - 1) * (2 * k) * one)
        t = one - (s * t) // ((2 * k) * (2 * k + 1) * one)
    sin = (r * t) >> 62
    return [(c, sin), (-sin, c), (-c, -sin), (sin, -c)][quadrant % 4]


@settings(max_examples=50)
@given(st.lists(st.integers(0, 2**96 - 1), min_size=1, max_size=32))
def test_the_vectorized_cos_sin_is_its_definition(angles: list[int]) -> None:
    # Each quadrant's edges, and the fifth quadrant's two angles, 4·HALF_PI and 4·HALF_PI + 1.
    edges = [k * HALF_PI + e for k in range(5) for e in (-1, 0, 1) if k * HALF_PI + e >= 0]
    angles += [*edges, TAU - 1, TAU, 7 * TAU + 4 * HALF_PI + 1, 2**96 - 1]
    hi = array([a >> 64 for a in angles])
    lo = array([a % 2**64 for a in angles])
    cos, sin = tables._cos_sin((hi, lo))  # pyright: ignore[reportPrivateUsage]
    found = list(zip(cos.tolist(), sin.tolist(), strict=True))
    assert found == [reference_cos_sin(a) for a in angles]
