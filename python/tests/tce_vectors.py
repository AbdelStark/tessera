# SPDX-License-Identifier: Apache-2.0 OR MIT
"""The conformance vectors' language (spec/vectors/codec/README.md), for the TCE tests.

`parse` turns a type name into the package's codecs, and `to_json` and `from_json` move values
between their Python form and the vectors' language-neutral JSON, by walking the codec. Both
follow the README: integers as decimal strings, a struct as an object keyed by field name, an
enum value as `{"Variant": fields}`, a map as its entries in canonical order.
"""

import json
import keyword
import re
from dataclasses import dataclass
from fractions import Fraction
from pathlib import Path
from typing import Any, cast

import numpy as np
import numpy.typing as npt

from tessera_quant.tce import (
    ASCII,
    BOOL,
    BYTES,
    DIGEST,
    FP,
    FP2,
    I8,
    I16,
    I32,
    I64,
    RATIONAL,
    U8,
    U16,
    U32,
    U64,
    Array,
    ByteArray,
    Codec,
    Int,
    IntVec,
    Limits,
    MapOf,
    Opt,
    Pair,
    Reader,
    Seq,
    Some,
    Struct,
    Tagged,
    UnitEnum,
    Variant,
    Writer,
)

VECTORS = Path(__file__).parents[2] / "spec" / "vectors"


def load(name: str) -> dict[str, Any]:
    """A vector file of `spec/vectors/`."""
    return cast(dict[str, Any], json.loads((VECTORS / name).read_text(encoding="utf-8")))


def limits(entry: dict[str, Any]) -> Limits:
    """A file's `limits` entry."""
    return Limits(
        max_total_bytes=int(entry["max_total_bytes"]),
        max_vec_len=int(entry["max_vec_len"]),
        max_depth=int(entry["max_depth"]),
    )


@dataclass(frozen=True, eq=False)
class Sample:
    """The `Sample` test schema."""

    flag: bool
    small: int
    list: npt.NDArray[np.uint16]
    maybe: Some[int] | None
    name: str
    digest: bytes


@dataclass(frozen=True)
class Empty:
    """`Shape::Empty`."""


@dataclass(frozen=True)
class Point:
    """`Shape::Point(u32)`: a tuple variant, its field held as `_0`."""

    _0: int


@dataclass(frozen=True)
class Box:
    """`Shape::Box { width, height }`."""

    width: int
    height: int


@dataclass(frozen=True)
class Nested:
    """`Shape::Nested(Vec<Shape>)`."""

    _0: list[Any]


class Lazy(Codec[Any]):
    """A codec named before it exists, for a recursive type: its size is declared."""

    def __init__(self, resolve: Any, min_size: int) -> None:
        self.resolve = resolve
        self.min_size = min_size

    def encode(self, value: Any, out: Writer) -> None:
        self.resolve().encode(value, out)

    def decode(self, reader: Reader) -> Any:
        return self.resolve().decode(reader)


SAMPLE = Struct(
    Sample,
    [
        ("flag", BOOL),
        ("small", U8),
        ("list", IntVec(np.uint16)),
        ("maybe", Opt(I32)),
        ("name", ASCII),
        ("digest", DIGEST),
    ],
)

SHAPE: Tagged[Any] = Tagged(
    [
        Variant("Empty", Empty),
        Variant("Point", Point, [("_0", U32)]),
        Variant("Box", Box, [("width", U16), ("height", U16)]),
        Variant("Nested", Nested, [("_0", Seq(Lazy(lambda: SHAPE, min_size=2)))]),
    ]
)

PRIMITIVES: dict[str, Codec[Any]] = {
    "u8": U8,
    "u16": U16,
    "u32": U32,
    "u64": U64,
    "i8": I8,
    "i16": I16,
    "i32": I32,
    "i64": I64,
    "bool": BOOL,
    "Fp": FP,
    "Fp2": FP2,
    "Ascii": ASCII,
    "Rational": RATIONAL,
    "Bytes": BYTES,
    "Digest": DIGEST,
    "Sample": SAMPLE,
    "Shape": SHAPE,
}

INTEGERS: dict[str, type[np.integer[Any]]] = {
    "u8": np.uint8,
    "u16": np.uint16,
    "u32": np.uint32,
    "u64": np.uint64,
    "i8": np.int8,
    "i16": np.int16,
    "i32": np.int32,
    "i64": np.int64,
}


def _split(arguments: str) -> list[str]:
    """Top-level comma-separated parts of a type's arguments."""
    parts: list[str] = []
    depth = 0
    start = 0
    for i, c in enumerate(arguments):
        if c in "<([":
            depth += 1
        elif c in ">)]":
            depth -= 1
        elif c == "," and depth == 0:
            parts.append(arguments[start:i].strip())
            start = i + 1
    parts.append(arguments[start:].strip())
    return parts


def parse(name: str) -> Codec[Any]:
    """The codec of a type name: `Vec<T>` of an integer type is an `IntVec`, as in schemas."""
    name = name.strip()
    if name in PRIMITIVES:
        return PRIMITIVES[name]
    if m := re.fullmatch(r"Option\^(\d+)<(.+)>", name):
        codec = parse(m.group(2))
        for _ in range(int(m.group(1))):
            codec = Opt(codec)
        return codec
    if m := re.fullmatch(r"\[u8; 1\]\^(\d+)", name):
        codec: Codec[Any] = U8
        for _ in range(int(m.group(1))):
            codec = Array[Any](codec, 1)
        return codec
    if m := re.fullmatch(r"pair\^(\d+)", name):
        codec = U8
        for _ in range(int(m.group(1))):
            codec = Pair[Any, int](codec, U8)
        return codec
    if m := re.fullmatch(r"Vec<(.+)>", name):
        inner = m.group(1).strip()
        return IntVec(INTEGERS[inner]) if inner in INTEGERS else Seq(parse(inner))
    if m := re.fullmatch(r"Option<(.+)>", name):
        return Opt(parse(m.group(1)))
    if m := re.fullmatch(r"Map<(.+)>", name):
        key, value = _split(m.group(1))
        return MapOf(parse(key), parse(value))
    if m := re.fullmatch(r"\((.+)\)", name):
        first, second = _split(m.group(1))
        return Pair(parse(first), parse(second))
    if m := re.fullmatch(r"\[(.+); (\d+)\]", name):
        element, n = m.group(1).strip(), int(m.group(2))
        return ByteArray(n) if element == "u8" else Array(parse(element), n)
    raise ValueError(f"unknown type {name!r}")


def _field_name(attribute: str) -> str:
    """The schema's name of a field: `class_` holds `class`."""
    stem = attribute.removesuffix("_")
    return stem if stem != attribute and keyword.iskeyword(stem) else attribute


def _is_tuple(fields: tuple[tuple[str, Codec[Any]], ...]) -> bool:
    return bool(fields) and all(re.fullmatch(r"_\d+", name) for name, _ in fields)


def to_json(codec: Codec[Any], value: Any) -> Any:
    """`value` in the vectors' JSON."""
    if isinstance(codec, Lazy):
        return to_json(codec.resolve(), value)
    if isinstance(codec, Int) or codec is FP:
        return str(value)
    if codec is BOOL or codec is ASCII:
        return value
    if codec is FP2:
        return [str(value[0]), str(value[1])]
    if codec is RATIONAL:
        return {"num": str(value.numerator), "den": str(value.denominator)}
    if codec is BYTES or isinstance(codec, ByteArray):
        return [str(b) for b in value]
    if isinstance(codec, IntVec):
        return [str(x) for x in cast(npt.NDArray[Any], value).tolist()]
    if isinstance(codec, Seq | Array):
        element = cast(Codec[Any], codec.element)  # pyright: ignore[reportUnknownMemberType]
        return [to_json(element, item) for item in value]
    if isinstance(codec, Opt):
        element = cast(Codec[Any], codec.element)  # pyright: ignore[reportUnknownMemberType]
        return None if value is None else {"some": to_json(element, value.value)}
    if isinstance(codec, Pair):
        pair = cast(Pair[Any, Any], codec)
        return [to_json(pair.first, value[0]), to_json(pair.second, value[1])]
    if isinstance(codec, MapOf):
        entries = cast(MapOf[Any, Any], codec)
        return [[to_json(entries.key, k), to_json(entries.value, v)] for k, v in value.items()]
    if isinstance(codec, Struct):
        return {_field_name(n): to_json(c, getattr(value, n)) for n, c in codec.fields}
    if isinstance(codec, UnitEnum):
        return {value.name: None}
    if isinstance(codec, Tagged):
        variant = next(v for v in codec.variants if type(value) is v.cls)
        fields = tuple(variant.fields)
        if not fields:
            return {variant.name: None}
        if _is_tuple(fields):
            return {variant.name: [to_json(c, getattr(value, n)) for n, c in fields]}
        return {variant.name: {_field_name(n): to_json(c, getattr(value, n)) for n, c in fields}}
    raise TypeError(f"no JSON form for {codec!r}")


def from_json(codec: Codec[Any], value: Any) -> Any:
    """The Python value of the vectors' JSON `value`."""
    if isinstance(codec, Lazy):
        return from_json(codec.resolve(), value)
    if isinstance(codec, Int) or codec is FP:
        return int(value)
    if codec is BOOL or codec is ASCII:
        return value
    if codec is FP2:
        return (int(value[0]), int(value[1]))
    if codec is RATIONAL:
        return Fraction(int(value["num"]), int(value["den"]))
    if codec is BYTES or isinstance(codec, ByteArray):
        return bytes(int(b) for b in value)
    if isinstance(codec, IntVec):
        return np.array([int(x) for x in value], dtype=cast(IntVec[Any], codec).dtype)
    if isinstance(codec, Seq | Array):
        element = cast(Codec[Any], codec.element)  # pyright: ignore[reportUnknownMemberType]
        return [from_json(element, item) for item in value]
    if isinstance(codec, Opt):
        element = cast(Codec[Any], codec.element)  # pyright: ignore[reportUnknownMemberType]
        return None if value is None else Some(from_json(element, value["some"]))
    if isinstance(codec, Pair):
        pair = cast(Pair[Any, Any], codec)
        return (from_json(pair.first, value[0]), from_json(pair.second, value[1]))
    if isinstance(codec, MapOf):
        entries = cast(MapOf[Any, Any], codec)
        return {from_json(entries.key, k): from_json(entries.value, v) for k, v in value}
    if isinstance(codec, Struct):
        return codec.make(**{n: from_json(c, value[_field_name(n)]) for n, c in codec.fields})
    if isinstance(codec, UnitEnum):
        (name,) = value
        return next(member for member in codec.members if member.name == name)
    if isinstance(codec, Tagged):
        ((name, fields_value),) = value.items()
        variant = next(v for v in codec.variants if v.name == name)
        fields = tuple(variant.fields)
        if _is_tuple(fields):
            values = zip(fields, fields_value, strict=True)
            return variant.cls(**{n: from_json(c, x) for (n, c), x in values})
        return variant.cls(**{n: from_json(c, fields_value[_field_name(n)]) for n, c in fields})
    raise TypeError(f"no Python form for {codec!r}")
