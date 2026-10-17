# SPDX-License-Identifier: Apache-2.0 OR MIT
"""The Tessera Canonical Encoding and its envelope (RFC-0003).

Python and the Rust tools exchange TCE artifacts only (DEP-4). A value has exactly one encoding
and a reader accepts nothing else: for every input it reports the code and offset that RFC-0003
'Limits' and 'Envelope' fix, as the prover's and the verifier's readers do, which the vectors of
`spec/vectors/codec/` check.

A `Codec` reads and writes one TCE type. The primitives are constants: `U8` to `I64`, `BOOL`,
`FP`, `FP2`, `ASCII`, `RATIONAL`, `BYTES` and `DIGEST`. Composites are built from them with
`Seq`, `Array`, `ByteArray`, `Opt`, `Pair`, `MapOf`, `Struct`, `UnitEnum` and `Tagged`, and
`IntVec` holds a sequence of integers in a NumPy array. A `Schema` names an artifact of 03 §5
and the codec of its current version; `TOKEN_FILE` and `LOGITS_FILE` are RFC-0003's auxiliary
schemas, and `tessera_quant.integer_model` defines `IntegerModel`.

`write_artifact` streams an envelope to a secret file. A first pass sizes the payload for the
header; the second writes it and hashes it on the way, and arrays go to the file from their own
memory, so a memory-mapped weight tensor of hundreds of gigabytes is never copied.
`read_artifact` maps the file, checks its digest over the mapping, then decodes it, and the
arrays it returns are read-only views of the mapping.
"""

import itertools
import math
import mmap
import operator
import os
import re
from abc import ABC, abstractmethod
from collections.abc import Callable, Generator, Sequence
from contextlib import contextmanager
from dataclasses import dataclass, field
from enum import Enum
from fractions import Fraction
from pathlib import Path
from typing import Any, BinaryIO, Generic, SupportsIndex, TypeAlias, TypeVar, cast

import blake3
import numpy as np
import numpy.typing as npt

from tessera_quant.errors import CodecError, EncodeError, TesseraQuantError

T = TypeVar("T")
A = TypeVar("A")
B = TypeVar("B")
K = TypeVar("K")
V = TypeVar("V")
E = TypeVar("E", bound=Enum)
S = TypeVar("S", bound=np.integer[Any])

Buffer: TypeAlias = bytes | bytearray | memoryview | mmap.mmap
"""Bytes a reader takes: in memory, or a mapped file."""

MAGIC = b"TSRA"
"""The first four bytes of every envelope."""

ENVELOPE_VERSION = 1
"""The envelope layout this module reads and writes."""

HEADER_LEN = 20
"""The header's length: the magic, four `u16` fields and the `u64` payload length."""

DIGEST_LEN = 32
"""The digest's length."""

DIGEST_CONTEXT = "tessera/v1/envelope"
"""The BLAKE3 key-derivation context of the digest, over every byte before it (RFC-0004)."""

P = 2**64 - 2**32 + 1
"""The Goldilocks prime (RFC-0002)."""

DEPTH_CEILING = 256
"""The deepest nesting any reader accepts, whatever its limits say (RFC-0003 'Limits')."""

_CODE_MAGIC = "ERR-CODEC-MAGIC"
_CODE_VERSION = "ERR-CODEC-VERSION"
_CODE_NONCANONICAL = "ERR-CODEC-NONCANONICAL"
_CODE_TRUNCATED = "ERR-CODEC-TRUNCATED"
_CODE_DIGEST = "ERR-CODEC-DIGEST"
_CODE_LIMIT = "ERR-CODEC-LIMIT"

# The unit in which a writer buffers small values, and in which it hashes and writes large ones.
_CHUNK = 1 << 16
_SLICE = 1 << 24


@dataclass(frozen=True)
class Limits:
    """The bounds a reader enforces before it allocates (RFC-0003 'Limits')."""

    max_total_bytes: int
    """The most bytes the whole input may hold."""
    max_vec_len: int
    """The longest length a single sequence or map may declare."""
    max_depth: int
    """The deepest nesting of composite values; above `DEPTH_CEILING` it counts as that."""


MESSAGE_LIMITS = Limits(max_total_bytes=8 << 30, max_vec_len=1 << 34, max_depth=32)
"""RFC-0003's defaults for messages: 8 GiB in all, 2^34 elements in a sequence, 32 levels."""

INTEGER_MODEL_LIMITS = Limits(max_total_bytes=1 << 40, max_vec_len=1 << 34, max_depth=32)
"""RFC-0003's defaults for `IntegerModel`: 1 TiB in all, the message defaults otherwise."""


class Reader:
    """A cursor over bytes being decoded, which enforces `Limits` (RFC-0003 'Limits').

    Offsets count from the start of the buffer, so a payload decoded in place inside its
    envelope reports offsets from the envelope's start. Every declared length is checked against
    the bytes left before anything is allocated (INV-TCE-3).
    """

    def __init__(
        self, data: memoryview, limits: Limits, start: int = 0, end: int | None = None
    ) -> None:
        """A reader of `data[start:end]`, the whole of `data` by default."""
        self._data = data
        self._position = start
        self._end = len(data) if end is None else end
        self._limits = limits
        self._ceiling = min(limits.max_depth, DEPTH_CEILING)
        self._depth = 0

    @property
    def position(self) -> int:
        """The offset of the next byte."""
        return self._position

    @property
    def remaining(self) -> int:
        """The bytes not yet read."""
        return self._end - self._position

    def enter(self) -> None:
        """Enters a composite value, `ERR-CODEC-LIMIT` at its first byte beyond `max_depth`."""
        if self._depth >= self._ceiling:
            raise CodecError(_CODE_LIMIT, self._position)
        self._depth += 1

    def leave(self) -> None:
        """Leaves the composite value entered last."""
        self._depth -= 1

    def require(self, n: int, at: int) -> None:
        """`ERR-CODEC-TRUNCATED` at `at`, the value's first byte, unless `n` bytes remain."""
        if n > self.remaining:
            raise CodecError(_CODE_TRUNCATED, at)

    def take(self, n: int) -> memoryview:
        """The `n` bytes of a value that starts here, without a copy."""
        start = self._position
        self.require(n, start)
        self._position = start + n
        return self._data[start : start + n]

    def since(self, start: int) -> bytes:
        """The bytes read since offset `start`."""
        return bytes(self._data[start : self._position])

    def integer(self, width: int, *, signed: bool = False) -> int:
        """A little-endian integer of `width` bytes, read whole before anything checks it."""
        return int.from_bytes(self.take(width), "little", signed=signed)

    def sequence_len(self, min_size: int) -> int:
        """A sequence's length prefix, checked in the order RFC-0003 'Limits' fixes.

        `ERR-CODEC-LIMIT` above `max_vec_len`, then `ERR-CODEC-TRUNCATED` when the elements, of
        `min_size` bytes each at least, cannot fit in the bytes left: both at the length's first
        byte, before any element is read.
        """
        at = self._position
        n = self.integer(8)
        if n > self._limits.max_vec_len:
            raise CodecError(_CODE_LIMIT, at)
        self.require(n * min_size, at)
        return n

    def finish(self) -> None:
        """`ERR-CODEC-NONCANONICAL` at the first byte left over, if any (INV-TCE-4)."""
        if self._position != self._end:
            raise CodecError(_CODE_NONCANONICAL, self._position)


def _noncanonical(at: int) -> CodecError:
    return CodecError(_CODE_NONCANONICAL, at)


class Writer:
    """Where an encoding goes, with the checks that keep it readable under `limits`.

    With limits, the first sequence longer than `max_vec_len` or value nested deeper than
    `max_depth` is kept in `refusal`, as the `ERR-CODEC-LIMIT` a reader would report at the same
    offset. A reader checks the input's size first, which is known only once everything is
    written, so the writer's owner checks the size, then raises `refusal`: it refuses exactly what
    a reader under the same limits would refuse, where it would.
    """

    def __init__(
        self, sink: Callable[[memoryview], object], limits: Limits | None = None, start: int = 0
    ) -> None:
        """A writer to `sink` whose first byte is at offset `start` of what readers will read."""
        self._sink = sink
        self._limits = limits
        self._ceiling = DEPTH_CEILING if limits is None else min(limits.max_depth, DEPTH_CEILING)
        self._position = start
        self._depth = 0
        self.refusal: CodecError | None = None
        """The first limit the value breaks, if any."""

    @property
    def position(self) -> int:
        """The offset of the next byte."""
        return self._position

    def _refuse(self) -> None:
        if self.refusal is None:
            self.refusal = CodecError(_CODE_LIMIT, self._position)

    def put(self, data: bytes | memoryview) -> None:
        """Appends `data`."""
        view = memoryview(data).cast("B")
        self._position += view.nbytes
        self._sink(view)

    def integer(self, value: int, width: int, *, signed: bool = False) -> None:
        """Appends `value` as a little-endian integer of `width` bytes."""
        self.put(value.to_bytes(width, "little", signed=signed))

    def enter(self) -> None:
        """Enters a composite value."""
        if self._limits is not None and self._depth >= self._ceiling:
            self._refuse()
        self._depth += 1

    def leave(self) -> None:
        """Leaves the composite value entered last."""
        self._depth -= 1

    def length(self, n: int) -> None:
        """Appends a sequence's length prefix."""
        if self._limits is not None and n > self._limits.max_vec_len:
            self._refuse()
        self.integer(n, 8)

    def check(self, total: int, at: int) -> None:
        """Raises what a reader would refuse, if anything.

        `ERR-CODEC-LIMIT` at `at` when the input's `total` bytes are too many, else the first
        limit the value breaks.
        """
        if self._limits is not None and total > self._limits.max_total_bytes:
            raise CodecError(_CODE_LIMIT, at)
        if self.refusal is not None:
            raise self.refusal


class Codec(ABC, Generic[T]):
    """A TCE type: how its values encode and decode."""

    min_size: int
    """The fewest bytes a value encodes to, which bounds a sequence's length (INV-TCE-3)."""

    @abstractmethod
    def encode(self, value: T, out: Writer) -> None:
        """Appends the encoding of `value`, or raises `EncodeError` if the type cannot hold it."""

    @abstractmethod
    def decode(self, reader: Reader) -> T:
        """Decodes one value, or raises `CodecError` with the code and offset RFC-0003 fixes."""


def _integer(value: object, name: str) -> int:
    """`value` as an `int`: a Python or NumPy integer, never a bool or a float."""
    if isinstance(value, bool | np.bool_):
        raise EncodeError(f"a {name} is an integer, not a bool")
    try:
        return operator.index(cast(SupportsIndex, value))
    except TypeError:
        raise EncodeError(f"a {name} is an integer") from None


class Int(Codec[int]):
    """A fixed-width integer: little-endian, two's complement when signed."""

    def __init__(self, name: str, width: int, *, signed: bool) -> None:
        """The integer type `name`, of `width` bytes."""
        self.name = name
        self.width = width
        self.signed = signed
        self.min_size = width
        bits = 8 * width
        self._low = -(1 << (bits - 1)) if signed else 0
        self._high = (1 << (bits - 1)) - 1 if signed else (1 << bits) - 1

    def encode(self, value: int, out: Writer) -> None:
        number = _integer(value, self.name)
        if not self._low <= number <= self._high:
            raise EncodeError(f"a {self.name} lies in [{self._low}, {self._high}]")
        out.integer(number, self.width, signed=self.signed)

    def decode(self, reader: Reader) -> int:
        return reader.integer(self.width, signed=self.signed)


U8 = Int("u8", 1, signed=False)
U16 = Int("u16", 2, signed=False)
U32 = Int("u32", 4, signed=False)
U64 = Int("u64", 8, signed=False)
I8 = Int("i8", 1, signed=True)
I16 = Int("i16", 2, signed=True)
I32 = Int("i32", 4, signed=True)
I64 = Int("i64", 8, signed=True)


class _Bool(Codec[bool]):
    """One byte, 0 or 1."""

    min_size = 1

    def encode(self, value: bool, out: Writer) -> None:
        if not isinstance(cast(object, value), bool):
            raise EncodeError("a bool is True or False")
        out.integer(int(value), 1)

    def decode(self, reader: Reader) -> bool:
        at = reader.position
        byte = reader.integer(1)
        if byte > 1:
            raise _noncanonical(at)
        return byte == 1


BOOL = _Bool()


class _Fp(Codec[int]):
    """A Goldilocks element: a `u64` below `P`."""

    min_size = 8

    def encode(self, value: int, out: Writer) -> None:
        number = _integer(value, "Fp")
        if not 0 <= number < P:
            raise EncodeError("an Fp lies in [0, P)")
        out.integer(number, 8)

    def decode(self, reader: Reader) -> int:
        at = reader.position
        number = reader.integer(8)
        if number >= P:
            raise _noncanonical(at)
        return number


FP = _Fp()


class _Fp2(Codec[tuple[int, int]]):
    """An element `c0 + c1·X` of `Fp[X]/(X^2 - 7)`: `c0`, then `c1`."""

    min_size = 16

    def encode(self, value: tuple[int, int], out: Writer) -> None:
        _tuple(value, "an Fp2")
        c0, c1 = value
        out.enter()
        FP.encode(c0, out)
        FP.encode(c1, out)
        out.leave()

    def decode(self, reader: Reader) -> tuple[int, int]:
        reader.enter()
        value = (FP.decode(reader), FP.decode(reader))
        reader.leave()
        return value


FP2 = _Fp2()

_NOT_PRINTABLE_BYTE = re.compile(rb"[^\x20-\x7e]")
_NOT_PRINTABLE_CHARACTER = re.compile(r"[^\x20-\x7e]")


class _Ascii(Codec[str]):
    """Printable ASCII text, every byte in `0x20..=0x7E`, encoded as `Vec<u8>`."""

    min_size = 8

    def encode(self, value: str, out: Writer) -> None:
        if not isinstance(cast(object, value), str):
            raise EncodeError("Ascii text is a str")
        bad = _NOT_PRINTABLE_CHARACTER.search(value)
        if bad is not None:
            raise EncodeError(f"Ascii text is printable ASCII, and character {bad.start()} is not")
        data = value.encode("ascii")
        out.enter()
        out.length(len(data))
        out.put(data)
        out.leave()

    def decode(self, reader: Reader) -> str:
        reader.enter()
        n = reader.sequence_len(1)
        start = reader.position
        data = bytes(reader.take(n))
        bad = _NOT_PRINTABLE_BYTE.search(data)
        if bad is not None:
            raise _noncanonical(start + bad.start())
        reader.leave()
        return data.decode("ascii")


ASCII = _Ascii()


class _Rational(Codec[Fraction]):
    """An exact fraction: an `i64` numerator, then a `u64` denominator, in lowest terms."""

    min_size = 16

    def encode(self, value: Fraction, out: Writer) -> None:
        if not isinstance(cast(object, value), Fraction):
            raise EncodeError("a Rational is a Fraction, never a float (INV-DM-6)")
        if not -(1 << 63) <= value.numerator < 1 << 63 or value.denominator >= 1 << 64:
            raise EncodeError("a Rational's numerator is an i64 and its denominator a u64")
        out.enter()
        out.integer(value.numerator, 8, signed=True)
        out.integer(value.denominator, 8)
        out.leave()

    def decode(self, reader: Reader) -> Fraction:
        reader.enter()
        at = reader.position
        numerator = reader.integer(8, signed=True)
        denominator = reader.integer(8)
        if denominator == 0 or math.gcd(numerator, denominator) != 1:
            raise _noncanonical(at)
        reader.leave()
        return Fraction(numerator, denominator)


RATIONAL = _Rational()


def _tuple(value: object, name: str) -> None:
    if not isinstance(value, tuple) or len(cast(tuple[object, ...], value)) != 2:
        raise EncodeError(f"{name} is a tuple of two")


def _list(value: object, name: str) -> None:
    if not isinstance(value, list):
        raise EncodeError(f"{name} is a list")


def _bytes(value: object, name: str) -> bytes:
    if not isinstance(value, bytes):
        raise EncodeError(f"{name} is bytes")
    return value


class _Bytes(Codec[bytes]):
    """`Vec<u8>`: raw bytes after a `u64` length."""

    min_size = 8

    def encode(self, value: bytes, out: Writer) -> None:
        data = _bytes(value, "Bytes")
        out.enter()
        out.length(len(data))
        out.put(data)
        out.leave()

    def decode(self, reader: Reader) -> bytes:
        reader.enter()
        data = bytes(reader.take(reader.sequence_len(1)))
        reader.leave()
        return data


BYTES = _Bytes()


class ByteArray(Codec[bytes]):
    """`[u8; N]`: `N` raw bytes, with no length."""

    def __init__(self, n: int) -> None:
        """The array of `n` bytes."""
        self.n = n
        self.min_size = n

    def encode(self, value: bytes, out: Writer) -> None:
        data = _bytes(value, f"a [u8; {self.n}]")
        if len(data) != self.n:
            raise EncodeError(f"a [u8; {self.n}] holds {self.n} bytes")
        out.enter()
        out.put(data)
        out.leave()

    def decode(self, reader: Reader) -> bytes:
        reader.enter()
        data = bytes(reader.take(self.n))
        reader.leave()
        return data


DIGEST = ByteArray(32)
"""A BLAKE3 digest, `[u8; 32]`."""


# NumPy's stubs type a buffer as `collections.abc.Buffer`, which Python 3.11's typeshed lacks.
_frombuffer = cast(Callable[..., npt.NDArray[Any]], np.frombuffer)  # pyright: ignore[reportUnknownMemberType]


class IntVec(Codec[npt.NDArray[S]]):
    """`Vec<T>` of an integer type, held in a one-dimensional NumPy array.

    Decoding returns a read-only view of the input, without a copy. Encoding writes the array's
    own memory, so the array must have the type's little-endian dtype already: nothing is cast.
    """

    def __init__(self, scalar: type[S]) -> None:
        """The sequence of `scalar`, a NumPy integer type."""
        self.dtype: np.dtype[S] = np.dtype(scalar).newbyteorder("<")
        self.name = f"{'i' if self.dtype.kind == 'i' else 'u'}{8 * self.dtype.itemsize}"
        self.min_size = 8

    def encode(self, value: npt.NDArray[S], out: Writer) -> None:
        array = cast(object, value)
        if not isinstance(array, np.ndarray) or value.dtype != self.dtype or value.ndim != 1:
            raise EncodeError(f"a Vec<{self.name}> is a one-dimensional array of {self.dtype}")
        data = np.ascontiguousarray(value)
        out.enter()
        out.length(data.shape[0])
        out.put(data.data)
        out.leave()

    def decode(self, reader: Reader) -> npt.NDArray[S]:
        reader.enter()
        n = reader.sequence_len(self.dtype.itemsize)
        view = reader.take(n * self.dtype.itemsize)
        array = cast(npt.NDArray[S], _frombuffer(view, dtype=self.dtype))
        array.flags.writeable = False
        reader.leave()
        return array


class Seq(Codec[list[T]]):
    """`Vec<T>`: a `u64` length `n`, then `n` encodings."""

    def __init__(self, element: Codec[T]) -> None:
        """The sequence of `element`, whose values encode to one byte at least."""
        if element.min_size < 1:
            raise ValueError("a sequence's elements encode to one byte at least (RFC-0003)")
        self.element = element
        self.min_size = 8

    def encode(self, value: list[T], out: Writer) -> None:
        _list(value, "a Vec")
        out.enter()
        out.length(len(value))
        for item in value:
            self.element.encode(item, out)
        out.leave()

    def decode(self, reader: Reader) -> list[T]:
        reader.enter()
        n = reader.sequence_len(self.element.min_size)
        items = [self.element.decode(reader) for _ in range(n)]
        reader.leave()
        return items


class Array(Codec[list[T]]):
    """`[T; N]`: `N` encodings, with no length."""

    def __init__(self, element: Codec[T], n: int) -> None:
        """The array of `n` values of `element`."""
        self.element = element
        self.n = n
        self.min_size = n * element.min_size

    def encode(self, value: list[T], out: Writer) -> None:
        _list(value, "an array")
        if len(value) != self.n:
            raise EncodeError(f"an array of {self.n} holds {self.n} values")
        out.enter()
        for item in value:
            self.element.encode(item, out)
        out.leave()

    def decode(self, reader: Reader) -> list[T]:
        reader.enter()
        reader.require(self.min_size, reader.position)
        items = [self.element.decode(reader) for _ in range(self.n)]
        reader.leave()
        return items


@dataclass(frozen=True)
class Some(Generic[T]):
    """A present `Option` value; an absent one is `None`, so nested options stay distinct."""

    value: T


class Opt(Codec[Some[T] | None]):
    """`Option<T>`: the enum `{ None, Some(T) }`, a `u16` tag 0 or 1, then the payload."""

    def __init__(self, element: Codec[T]) -> None:
        """The option of `element`."""
        self.element = element
        self.min_size = 2

    def encode(self, value: Some[T] | None, out: Writer) -> None:
        if value is not None and not isinstance(cast(object, value), Some):
            raise EncodeError("an Option is None or Some")
        out.enter()
        out.integer(0 if value is None else 1, 2)
        if value is not None:
            self.element.encode(value.value, out)
        out.leave()

    def decode(self, reader: Reader) -> Some[T] | None:
        reader.enter()
        at = reader.position
        tag = reader.integer(2)
        if tag > 1:
            raise _noncanonical(at)
        value = None if tag == 0 else Some(self.element.decode(reader))
        reader.leave()
        return value


class Pair(Codec[tuple[A, B]]):
    """`(A, B)`: both encodings, in order."""

    def __init__(self, first: Codec[A], second: Codec[B]) -> None:
        """The pair of `first` and `second`."""
        self.first = first
        self.second = second
        self.min_size = first.min_size + second.min_size

    def encode(self, value: tuple[A, B], out: Writer) -> None:
        _tuple(value, "a pair")
        first, second = value
        out.enter()
        self.first.encode(first, out)
        self.second.encode(second, out)
        out.leave()

    def decode(self, reader: Reader) -> tuple[A, B]:
        reader.enter()
        value = (self.first.decode(reader), self.second.decode(reader))
        reader.leave()
        return value


class MapOf(Codec[dict[K, V]]):
    """`Map<K, V>`: `Vec<(K, V)>` sorted strictly by the encoding of `K` (RFC-0003).

    The order is the keys' encodings', not their values': for little-endian integers it is not
    numeric order, and for text the length prefix sorts first. Decoding returns the entries in
    that order and needs keys Python can hash; encoding sorts them, so a dict in any order
    encodes canonically.
    """

    def __init__(self, key: Codec[K], value: Codec[V]) -> None:
        """The map from `key` to `value`."""
        if key.min_size + value.min_size < 1:
            raise ValueError("a map's entries encode to one byte at least (RFC-0003)")
        self.key = key
        self.value = value
        self.min_size = 8

    def encode(self, value: dict[K, V], out: Writer) -> None:
        if not isinstance(cast(object, value), dict):
            raise EncodeError("a Map is a dict")
        entries = sorted(
            ((encode(self.key, k), k, v) for k, v in value.items()), key=lambda entry: entry[0]
        )
        if any(a[0] == b[0] for a, b in itertools.pairwise(entries)):
            raise EncodeError("two keys of a map encode the same")
        out.enter()
        out.length(len(entries))
        for _, k, v in entries:
            out.enter()
            self.key.encode(k, out)
            self.value.encode(v, out)
            out.leave()
        out.leave()

    def decode(self, reader: Reader) -> dict[K, V]:
        reader.enter()
        n = reader.sequence_len(self.key.min_size + self.value.min_size)
        entries: dict[K, V] = {}
        previous: bytes | None = None
        for _ in range(n):
            # Each entry is one level, as the pair it is in `Vec<(K, V)>`.
            reader.enter()
            start = reader.position
            key = self.key.decode(reader)
            encoded = reader.since(start)
            # Strictly ascending: an equal key is a duplicate, a smaller one out of order.
            if previous is not None and encoded <= previous:
                raise _noncanonical(start)
            previous = encoded
            entries[key] = self.value.decode(reader)
            reader.leave()
        reader.leave()
        return entries


class Struct(Codec[T]):
    """A struct: its fields' encodings in the schema's order, with no tags.

    `fields` pairs each field's attribute with its codec, and `make` builds a value from them by
    keyword.
    """

    def __init__(self, make: Callable[..., T], fields: Sequence[tuple[str, Codec[Any]]]) -> None:
        """The struct `make` builds, of `fields` in order."""
        self.make = make
        self.fields = tuple(fields)
        self.min_size = sum(codec.min_size for _, codec in self.fields)

    def encode(self, value: T, out: Writer) -> None:
        if isinstance(self.make, type) and not isinstance(value, self.make):
            raise EncodeError(f"{self.make.__name__}: a value of another class")
        out.enter()
        for name, codec in self.fields:
            codec.encode(getattr(value, name), out)
        out.leave()

    def decode(self, reader: Reader) -> T:
        reader.enter()
        values = {name: codec.decode(reader) for name, codec in self.fields}
        reader.leave()
        return self.make(**values)


class UnitEnum(Codec[E]):
    """An enum whose variants have no fields: the variant's `u16` tag, its declaration index.

    The Python `Enum` declares its members in the schema's order, valued 0, 1 and so on.
    """

    def __init__(self, cls: type[E]) -> None:
        """The enum `cls`."""
        self.cls = cls
        self.members = list(cls)
        if [member.value for member in self.members] != list(range(len(self.members))):
            raise ValueError(f"{cls.__name__}'s values are not its declaration indices")
        self.min_size = 2

    def encode(self, value: E, out: Writer) -> None:
        if not isinstance(value, self.cls):
            raise EncodeError(f"a {self.cls.__name__} is one of its members")
        out.enter()
        out.integer(self.members.index(value), 2)
        out.leave()

    def decode(self, reader: Reader) -> E:
        reader.enter()
        at = reader.position
        tag = reader.integer(2)
        if tag >= len(self.members):
            raise _noncanonical(at)
        reader.leave()
        return self.members[tag]


@dataclass(frozen=True)
class Variant:
    """A variant of a `Tagged` enum: its name in the schema, its class, and its fields."""

    name: str
    cls: type[Any]
    fields: Sequence[tuple[str, Codec[Any]]] = ()


class Tagged(Codec[T]):
    """An enum: a `u16` tag, the variant's declaration index, then its fields, as a struct's.

    Each variant is a class of its own, and a value's class gives its tag.
    """

    def __init__(self, variants: Sequence[Variant]) -> None:
        """The enum of `variants`, in the schema's order."""
        self.variants = tuple(variants)
        self._tags = {variant.cls: tag for tag, variant in enumerate(self.variants)}
        if len(self._tags) != len(self.variants):
            raise ValueError("each variant has a class of its own")
        # The tag, then the smallest variant.
        sizes = (sum(codec.min_size for _, codec in v.fields) for v in self.variants)
        self.min_size = 2 + min(sizes, default=0)

    def encode(self, value: T, out: Writer) -> None:
        tag = self._tags.get(type(value))
        if tag is None:
            raise EncodeError("a value of none of the enum's variants")
        out.enter()
        out.integer(tag, 2)
        for name, codec in self.variants[tag].fields:
            codec.encode(getattr(value, name), out)
        out.leave()

    def decode(self, reader: Reader) -> T:
        reader.enter()
        at = reader.position
        tag = reader.integer(2)
        if tag >= len(self.variants):
            raise _noncanonical(at)
        variant = self.variants[tag]
        values = {name: codec.decode(reader) for name, codec in variant.fields}
        reader.leave()
        return cast(T, variant.cls(**values))


def encode(codec: Codec[T], value: T, *, limits: Limits | None = None) -> bytes:
    """The encoding of `value`, which a reader under `limits` would accept, if they are given."""
    out = bytearray()
    writer = Writer(out.extend, limits)
    codec.encode(value, writer)
    writer.check(len(out), 0)
    return bytes(out)


def decode(codec: Codec[T], data: Buffer, *, limits: Limits) -> T:
    """The value whose encoding is exactly `data`, read under `limits`."""
    view = memoryview(data).cast("B")
    if len(view) > limits.max_total_bytes:
        raise CodecError(_CODE_LIMIT, 0)
    reader = Reader(view, limits)
    value = codec.decode(reader)
    reader.finish()
    return value


@dataclass(frozen=True)
class Schema(Generic[T]):
    """An artifact schema of 03 §5: its id, its writers' version, and that version's codec."""

    name: str
    schema_id: int
    version: int
    codec: Codec[T]

    def reads(self, version: int) -> bool:
        """Whether its readers accept `version`: the current one or the one before, never 0."""
        return version != 0 and version in {self.version, self.version - 1}


@dataclass(frozen=True)
class Envelope:
    """An envelope whose header, length and digest have been checked."""

    data: memoryview = field(repr=False)
    """The whole envelope."""
    schema_id: int
    version: int
    """The schema version found: the reader's, or the one before."""
    payload_len: int

    def decode(self, codec: Codec[T], *, limits: Limits) -> T:
        """The payload decoded as `codec`, which it must fill (INV-TCE-4).

        Offsets count from the start of the envelope (RFC-0003 'Envelope', step 4).
        """
        reader = Reader(self.data, limits, HEADER_LEN, HEADER_LEN + self.payload_len)
        value = codec.decode(reader)
        reader.finish()
        return value


def _header(schema: Schema[Any], payload_len: int) -> bytes:
    return b"".join(
        (
            MAGIC,
            ENVELOPE_VERSION.to_bytes(2, "little"),
            schema.schema_id.to_bytes(2, "little"),
            schema.version.to_bytes(2, "little"),
            bytes(2),
            payload_len.to_bytes(8, "little"),
        )
    )


def _hasher() -> blake3.blake3:
    return blake3.blake3(derive_key_context=DIGEST_CONTEXT, max_threads=blake3.blake3.AUTO)


def _whole(payload_len: int) -> int:
    """The size of the envelope of a payload of `payload_len` bytes."""
    return HEADER_LEN + payload_len + DIGEST_LEN


def _field(view: memoryview, at: int, width: int) -> bytes:
    """The header field of `width` bytes at `at`, `ERR-CODEC-TRUNCATED` there if cut short."""
    if len(view) < at + width:
        raise CodecError(_CODE_TRUNCATED, at)
    return bytes(view[at : at + width])


def _number(view: memoryview, at: int, width: int) -> int:
    return int.from_bytes(_field(view, at, width), "little")


def open_envelope(data: Buffer, schema: Schema[Any], *, limits: Limits) -> Envelope:
    """Checks an envelope of `schema` in the order RFC-0003 'Envelope' fixes.

    1. The header, field by field as it is read: the magic, the envelope version, the schema id,
       a version the schema's readers accept, a zero `reserved`, and a payload length that keeps
       the envelope within `max_total_bytes`.
    2. The rest of the input: too short is `ERR-CODEC-TRUNCATED` at offset 12, the length that
       promised the bytes; too long is `ERR-CODEC-NONCANONICAL` past the digest.
    3. The digest, over everything before it.

    The payload is then ready to decode, as the version found.
    """
    view = memoryview(data).cast("B")
    if _field(view, 0, 4) != MAGIC:
        raise CodecError(_CODE_MAGIC, 0)
    if _number(view, 4, 2) != ENVELOPE_VERSION:
        raise CodecError(_CODE_VERSION, 4)
    if _number(view, 6, 2) != schema.schema_id:
        raise CodecError(_CODE_MAGIC, 6)
    version = _number(view, 8, 2)
    if not schema.reads(version):
        raise CodecError(_CODE_VERSION, 8)
    if _number(view, 10, 2) != 0:
        raise CodecError(_CODE_NONCANONICAL, 10)
    payload_len = _number(view, 12, 8)
    if _whole(payload_len) > limits.max_total_bytes:
        raise CodecError(_CODE_LIMIT, 12)
    expected = payload_len + DIGEST_LEN
    found = len(view) - HEADER_LEN
    if found < expected:
        raise CodecError(_CODE_TRUNCATED, 12)
    if found > expected:
        raise CodecError(_CODE_NONCANONICAL, HEADER_LEN + expected)
    digest_at = HEADER_LEN + payload_len
    hasher = _hasher()
    hasher.update(view[:digest_at])
    if hasher.digest() != bytes(view[digest_at:]):
        raise CodecError(_CODE_DIGEST, digest_at)
    return Envelope(view, schema.schema_id, version, payload_len)


def unseal(data: Buffer, schema: Schema[T], *, limits: Limits) -> T:
    """The value in an envelope of the current version of `schema`.

    The codec is one version's layout, so the version before, which `open_envelope` accepts, is
    `ERR-CODEC-VERSION` here.
    """
    envelope = open_envelope(data, schema, limits=limits)
    if envelope.version != schema.version:
        raise CodecError(_CODE_VERSION, 8)
    return envelope.decode(schema.codec, limits=limits)


def seal(schema: Schema[T], value: T, *, limits: Limits) -> bytes:
    """The envelope of `value` as the current version of `schema`, readable under `limits`."""
    out = bytearray(HEADER_LEN)
    writer = Writer(out.extend, limits, HEADER_LEN)
    schema.codec.encode(value, writer)
    payload_len = len(out) - HEADER_LEN
    writer.check(_whole(payload_len), 12)
    out[:HEADER_LEN] = _header(schema, payload_len)
    hasher = _hasher()
    hasher.update(out)
    out += hasher.digest()
    return bytes(out)


@contextmanager
def secret_file(path: Path) -> Generator[BinaryIO]:
    """A secret artifact being written (06 §7): mode 0600, renamed into place when complete.

    The file is never readable by others, even while it is written, and appears whole or not at
    all: if the block raises, the partial file is removed.
    """
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    descriptor = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    try:
        with os.fdopen(descriptor, "wb") as file:
            yield file
            file.flush()
            os.fsync(file.fileno())
        os.replace(temporary, path)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise
    directory = os.open(path.parent, os.O_RDONLY)
    try:
        os.fsync(directory)
    finally:
        os.close(directory)


class _Stream:
    """Writes to a file, hashing on the way.

    Small values go through a buffer; large ones go in slices, each hashed and written while it
    is in cache.
    """

    def __init__(self, file: BinaryIO) -> None:
        self._file = file
        self._hasher = _hasher()
        self._buffer = bytearray()

    def put(self, data: memoryview) -> None:
        if len(self._buffer) + data.nbytes <= _CHUNK:
            self._buffer += data
            return
        self._flush()
        if data.nbytes < _CHUNK:
            self._buffer += data
            return
        for start in range(0, data.nbytes, _SLICE):
            part = data[start : start + _SLICE]
            self._hasher.update(part)
            self._file.write(part)

    def _flush(self) -> None:
        if self._buffer:
            self._hasher.update(self._buffer)
            self._file.write(self._buffer)
            self._buffer.clear()

    def digest(self) -> bytes:
        self._flush()
        return self._hasher.digest()


def write_artifact(path: Path, schema: Schema[T], value: T, *, limits: Limits) -> bytes:
    """Writes the envelope of `value` to `path` as a secret artifact, and returns its digest.

    A first pass sizes the payload for the header; the second writes it and hashes it on the
    way, so memory stays at a few chunks whatever the payload's size. A value its readers would
    refuse under `limits` is refused before the file is created.
    """
    sizing = Writer(lambda _: None, limits, HEADER_LEN)
    schema.codec.encode(value, sizing)
    payload_len = sizing.position - HEADER_LEN
    sizing.check(_whole(payload_len), 12)
    with secret_file(path) as file:
        stream = _Stream(file)
        stream.put(memoryview(_header(schema, payload_len)))
        writer = Writer(stream.put, start=HEADER_LEN)
        schema.codec.encode(value, writer)
        if writer.position != HEADER_LEN + payload_len:
            raise TesseraQuantError("the value changed between the sizing and the writing pass")
        digest = stream.digest()
        file.write(digest)
    return digest


def read_artifact(path: Path, schema: Schema[T], *, limits: Limits) -> T:
    """Reads an artifact of `schema` from `path`: maps it, checks it, and decodes it.

    The digest is checked over the whole file before anything is decoded. Integer sequences are
    read-only views of the mapping, so the file must not change while they are in use. The
    digest detects corruption, not tampering: whoever can write the file can rewrite its digest.
    """
    with path.open("rb") as file:
        size = os.fstat(file.fileno()).st_size
        data: Buffer = b"" if size == 0 else mmap.mmap(file.fileno(), 0, access=mmap.ACCESS_READ)
    return unseal(data, schema, limits=limits)


@dataclass(frozen=True, eq=False)
class TokenFile:
    """Token sequences (RFC-0003 'Auxiliary schemas', schema `0x0601`): secret-class (05 §3)."""

    sequences: list[npt.NDArray[np.uint32]] = field(repr=False)


@dataclass(frozen=True, eq=False)
class LogitsFile:
    """The integer logits of `D` (RFC-0003 'Auxiliary schemas', schema `0x0602`): secret-class."""

    vocab: int
    positions: list[npt.NDArray[np.int64]] = field(repr=False)


TOKEN_FILE: Schema[TokenFile] = Schema(
    "TokenFile", 0x0601, 1, Struct(TokenFile, [("sequences", Seq(IntVec(np.uint32)))])
)
"""`TokenFile`, as its writers emit it."""

LOGITS_FILE: Schema[LogitsFile] = Schema(
    "LogitsFile",
    0x0602,
    1,
    Struct(LogitsFile, [("vocab", U32), ("positions", Seq(IntVec(np.int64)))]),
)
"""`LogitsFile`, as its writers emit it."""
