# SPDX-License-Identifier: Apache-2.0 OR MIT
"""Explicitly copy a tied BF16 checkpoint into an untied one before RFC-0005 import.

From python/: PYTHONPATH=. uv run python tools/materialize_tied_head.py prepare SOURCE OUT
Use `verify SOURCE PREPARED` to check an existing conversion, including all original bits.
This tool changes no accepted import rule, integer model, proof or fidelity qualification.
"""

import argparse
import json
import math
import os
import struct
from collections.abc import Sequence
from dataclasses import dataclass
from decimal import Decimal, DecimalException
from pathlib import Path
from typing import Any, BinaryIO, cast

from tessera_quant import ArchSpec, import_checkpoint, read_config, tensor_shapes

# This repository-only converter reuses the existing TB-5 parser, not a second weaker parser.
from tessera_quant.checkpoint import MAX_HEADER_BYTES, _placement, _read_header  # pyright: ignore[reportPrivateUsage]
from tessera_quant.cli import EXIT_CODES
from tessera_quant.errors import ArchitectureError, TesseraQuantError, UsageError

_CHUNK = 1024 * 1024
_CONFIG_LIMIT = 1024 * 1024
_EMBED = "model.embed_tokens.weight"
_HEAD = "lm_head.weight"
_AUXILIARY = (
    "tokenizer.json",
    "tokenizer_config.json",
    "special_tokens_map.json",
    "generation_config.json",
    "README.md",
    "LICENSE",
    "LICENSE.txt",
    "NOTICE",
    "USE_POLICY.md",
)
_Report = dict[str, str | int]


@dataclass(frozen=True)
class _Slice:
    path: Path
    start: int
    size: int


@dataclass(frozen=True)
class _Source:
    directory: Path
    arch: ArchSpec
    text: str
    config: dict[str, Any]
    tensors: dict[str, _Slice]


def _config_text(path: Path) -> str:
    with path.open("rb") as stream:
        raw = stream.read(_CONFIG_LIMIT + 1)
    if len(raw) > _CONFIG_LIMIT:
        raise ArchitectureError("configuration exceeds preparation limit")
    try:
        return raw.decode("utf-8")
    except UnicodeError:
        raise ArchitectureError("configuration is not UTF-8") from None


def _unique(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ArchitectureError("duplicate configuration member")
        result[key] = value
    return result


def _finite_constant(_: str) -> Any:
    raise ArchitectureError("non-finite configuration number")


def _config(text: str) -> dict[str, Any]:
    try:
        value: Any = json.loads(
            text, parse_float=Decimal, parse_constant=_finite_constant, object_pairs_hook=_unique
        )
    except (ValueError, RecursionError, DecimalException):
        raise ArchitectureError("configuration is not a JSON object") from None
    if not isinstance(value, dict):
        raise ArchitectureError("configuration is not a JSON object")
    return cast(dict[str, Any], value)


def _untied(text: str) -> str:
    fields = _config(text)
    if fields.get("tie_word_embeddings") is not True:
        raise ArchitectureError("preparation requires tie_word_embeddings=true")
    # Decode member boundaries, preserving every other byte and number lexeme. Regex or
    # float parsing could rewrite an escaped/nested key or a long exact decimal (INV-DM-6).
    decoder = json.JSONDecoder(parse_float=Decimal)
    position = text.index("{") + 1
    while True:
        position = _whitespace(text, position)
        key, position = decoder.raw_decode(text, position)
        position = _whitespace(text, position) + 1  # Validated member's colon.
        start = _whitespace(text, position)
        _, end = decoder.raw_decode(text, start)
        if key == "tie_word_embeddings":
            return text[:start] + "false" + text[end:]
        position = _whitespace(text, end) + 1  # Validated member's comma.


def _whitespace(text: str, position: int) -> int:
    while text[position] in " \t\r\n":
        position += 1
    return position


def _ranges(path: Path) -> dict[str, _Slice]:
    checked = _read_header(path)  # TB-5: validate the complete header before tensor data.
    with path.open("rb") as stream:
        prefix = stream.read(8)
        if len(prefix) != 8:
            raise ArchitectureError("checkpoint header changed during preparation")
        size = struct.unpack("<Q", prefix)[0]
        if size > MAX_HEADER_BYTES:
            raise ArchitectureError("checkpoint header changed during preparation")
        raw = stream.read(size)
    try:
        header: Any = json.loads(raw)
        result = {
            name: _Slice(path, 8 + size + int(header[name]["data_offsets"][0]), length)
            for name, (_, _, length) in checked.items()
        }
    except (ValueError, KeyError, TypeError, IndexError):
        raise ArchitectureError("checkpoint header changed during preparation") from None
    return result


def _source(directory: Path) -> _Source:
    directory = directory.resolve()
    text = _config_text(directory / "config.json")
    converted = _untied(text)
    arch = read_config(converted)
    expected = tensor_shapes(arch)
    del expected[_HEAD]
    placement = _placement(directory)
    if placement.keys() != expected.keys():
        raise ArchitectureError("source must hold every expected tensor except lm_head.weight")
    tensors: dict[str, _Slice] = {}
    for filename in sorted(set(placement.values())):
        path = directory / filename
        header = _read_header(path)
        names = {name for name, file in placement.items() if file == filename}
        if header.keys() != names:
            raise ArchitectureError("source shard does not match its index")
        for name, (dtype, shape, length) in header.items():
            if dtype != "BF16" or shape != expected[name] or length != 2 * math.prod(shape):
                raise ArchitectureError("source tensor has incompatible BF16 shape or length")
        tensors.update(_ranges(path))
    return _Source(directory, arch, converted, _config(converted), tensors)


def _snapshot(source: _Source) -> dict[Path, tuple[int, int, int, int, int]]:
    paths = {entry.path for entry in source.tensors.values()} | {source.directory / "config.json"}
    paths |= {
        source.directory / name
        for name in (*_AUXILIARY, "model.safetensors.index.json")
        if (source.directory / name).is_file()
    }
    result: dict[Path, tuple[int, int, int, int, int]] = {}
    for path in paths:
        s = path.stat()
        result[path] = (s.st_dev, s.st_ino, s.st_size, s.st_mtime_ns, s.st_ctime_ns)
    return result


def _copy_exact(reader: BinaryIO, writer: BinaryIO, size: int) -> None:
    remaining = size
    while remaining:
        block = reader.read(min(_CHUNK, remaining))
        if not block:
            raise ArchitectureError("checkpoint ended during preparation")
        if writer.write(block) != len(block):
            raise OSError("incomplete checkpoint write")
        remaining -= len(block)


def _private(path: Path) -> BinaryIO:
    return os.fdopen(os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600), "wb")


def _write(path: Path, data: bytes) -> None:
    with _private(path) as stream:
        if stream.write(data) != len(data):
            raise OSError("incomplete metadata write")


def _copy_file(source: Path, target: Path) -> None:
    with source.open("rb") as reader, _private(target) as writer:
        _copy_exact(reader, writer, source.stat().st_size)


def _same_config(left: Any, right: Any) -> bool:
    if type(left) is not type(right):
        return False
    if isinstance(left, dict):
        a, b = cast(dict[str, Any], left), cast(dict[str, Any], right)
        return a.keys() == b.keys() and all(_same_config(v, b[k]) for k, v in a.items())
    if isinstance(left, list):
        items, other = cast(list[Any], left), cast(list[Any], right)
        return len(items) == len(other) and all(
            _same_config(a, b) for a, b in zip(items, other, strict=True)
        )
    return bool(left == right)


def _equal(left: _Slice, right: _Slice) -> None:
    if left.size != right.size:
        raise ArchitectureError("prepared tensor length differs")
    with left.path.open("rb") as a, right.path.open("rb") as b:
        a.seek(left.start)
        b.seek(right.start)
        remaining = left.size
        while remaining:
            size = min(_CHUNK, remaining)
            first, second = a.read(size), b.read(size)
            if len(first) != size or first != second:
                raise ArchitectureError("prepared bytes differ from source")
            remaining -= size


def verify(source: Path, prepared: Path) -> _Report:
    """Check an explicit untied conversion against every original tensor's BF16 bits."""
    original = _source(source)
    frozen = _snapshot(original)
    if not _same_config(original.config, _config(_config_text(prepared / "config.json"))):
        raise ArchitectureError("prepared configuration differs from source")
    model = import_checkpoint(prepared, expected=original.arch)
    ranges: dict[str, _Slice] = {}
    for file in set(model.files.values()):
        ranges.update(_ranges(file))
    for name in model.names:
        _equal(original.tensors[_EMBED if name == _HEAD else name], ranges[name])
    for name in _AUXILIARY:
        a, b = original.directory / name, prepared / name
        if a.is_file() and (name == "tokenizer.json" or b.exists()):
            _equal(_Slice(a, 0, a.stat().st_size), _Slice(b, 0, b.stat().st_size))
    if frozen != _snapshot(original):
        raise ArchitectureError("source changed during preparation")
    return {
        "status": "verified",
        "source_tensors": len(original.tensors),
        "prepared_tensors": len(model.names),
        "logical_parameters": sum(math.prod(s) for s in tensor_shapes(model.arch).values()),
        "tensor_bytes": sum(entry.size for entry in ranges.values()),
    }


def materialize(source: Path, out: Path) -> _Report:
    """Stream a new private checkpoint, retaining non-importable partial evidence on failure.

    Validate every source header first. Stage files in an exclusively created directory;
    publish config.json last, only after byte equality and unchanged-source checks pass.
    Existing destinations and paths inside the source are refused without writing.
    """
    source, out = source.resolve(), out.absolute()
    resolved = out.resolve()
    if resolved == source or source in resolved.parents or out.exists() or out.is_symlink():
        raise UsageError("destination must be new and outside the source")
    original = _source(source)
    frozen = _snapshot(original)
    try:
        out.mkdir(mode=0o700)
    except FileExistsError:
        raise UsageError("destination already exists") from None
    _write(out / ".incomplete", b"Checkpoint preparation is incomplete.\n")
    stage = out / ".staging"
    stage.mkdir(mode=0o700)
    files = sorted({entry.path for entry in original.tensors.values()})
    count = len(files) + 1
    renamed: dict[Path, str] = {}
    for i, file in enumerate(files, 1):
        name = f"model-{i:05}-of-{count:05}.safetensors"
        _copy_file(file, stage / name)
        renamed[file] = name
    head_name = f"model-{count:05}-of-{count:05}.safetensors"
    embedding = original.tensors[_EMBED]
    header = {
        "__metadata__": {"format": "pt"},
        _HEAD: {
            "dtype": "BF16",
            "shape": list(tensor_shapes(original.arch)[_HEAD]),
            "data_offsets": [0, embedding.size],
        },
    }
    raw = json.dumps(header, separators=(",", ":")).encode("utf-8")
    raw += b" " * (-len(raw) % 8)
    with embedding.path.open("rb") as reader, _private(stage / head_name) as writer:
        writer.write(struct.pack("<Q", len(raw)))
        writer.write(raw)
        reader.seek(embedding.start)
        _copy_exact(reader, writer, embedding.size)
    placement = {name: renamed[entry.path] for name, entry in original.tensors.items()}
    placement[_HEAD] = head_name
    index = {
        "metadata": {"total_size": sum(t.size for t in original.tensors.values()) + embedding.size},
        "weight_map": placement,
    }
    _write(stage / "model.safetensors.index.json", (json.dumps(index, indent=2) + "\n").encode())
    _write(stage / "config.json", original.text.encode("utf-8"))
    for name in _AUXILIARY:
        if (source / name).is_file():
            _copy_file(source / name, stage / name)
    result = verify(source, stage)
    if frozen != _snapshot(original):
        raise ArchitectureError("source changed during preparation")
    _write(stage / "preparation.json", (json.dumps(result, indent=2) + "\n").encode())
    for path in sorted(stage.iterdir()):
        if path.name != "config.json":
            path.rename(out / path.name)
    (stage / "config.json").rename(out / "config.json")
    stage.rmdir()
    (out / ".incomplete").unlink()
    return result


def main(argv: Sequence[str] | None = None) -> int:
    """Run explicit preparation or verification; print public metadata and typed errors only."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("operation", choices=("prepare", "verify"))
    parser.add_argument("source", type=Path)
    parser.add_argument("prepared", type=Path)
    args = parser.parse_args(argv)
    try:
        function = materialize if args.operation == "prepare" else verify
        result = function(args.source, args.prepared)
    except TesseraQuantError as error:
        print(json.dumps({"error": {"code": error.code, "message": str(error)}}))
        return EXIT_CODES.get(error.code, 70)
    except OSError:
        print(json.dumps({"error": {"code": "ERR-RES-IO", "message": "checkpoint I/O failed"}}))
        return EXIT_CODES["ERR-RES-IO"]
    except MemoryError:
        print(json.dumps({"error": {"code": "ERR-RES-MEMORY", "message": "out of memory"}}))
        return EXIT_CODES["ERR-RES-MEMORY"]
    except Exception as error:  # OBS-1: unexpected failures reveal their type, never input values.
        print(json.dumps({"error": {"code": "ERR-INTERNAL", "message": type(error).__name__}}))
        return 70
    print(json.dumps(result))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
