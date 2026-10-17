# SPDX-License-Identifier: Apache-2.0 OR MIT
"""Importing a Llama-3.1 checkpoint (RFC-0005 'Pipeline', step 1).

A checkpoint is untrusted input (06 §3, TB-5). Only safetensors files are read, through headers this
module checks before any tensor is loaded, and nothing is ever unpickled.
"""

import json
import struct
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

import tokenizers
import torch
from safetensors import safe_open

from tessera_quant.arch import ArchSpec, read_config, tensor_shapes
from tessera_quant.errors import ArchitectureError

MAX_HEADER_BYTES = 100 * 2**20
"""The largest safetensors header read, as the safetensors library itself bounds it."""

INDEX = "model.safetensors.index.json"
SINGLE = "model.safetensors"
TOKENIZER = "tokenizer.json"
PICKLE_SUFFIXES = frozenset({".bin", ".pt", ".pth", ".pkl", ".pickle", ".ckpt"})


@dataclass(frozen=True)
class FloatModel:
    """A checked BF16 checkpoint: its architecture, and the file of each tensor, read on demand."""

    arch: ArchSpec
    files: Mapping[str, Path]
    """Each tensor's file, in the order of `tensor_shapes`."""

    @property
    def names(self) -> tuple[str, ...]:
        """Every tensor's name, in the order of `tensor_shapes`."""
        return tuple(self.files)

    def tensor(self, name: str) -> torch.Tensor:
        """The tensor `name`, in BF16, read from its file.

        Raises:
            KeyError: for a name the checkpoint does not hold.
        """
        # safetensors' stubs leave `safe_open` untyped.
        opened: Any = safe_open(str(self.files[name]), framework="pt")
        with opened as file:
            return cast(torch.Tensor, file.get_tensor(name))


def import_checkpoint(path: Path, *, expected: ArchSpec | None = None) -> FloatModel:
    """Checks the Llama-3.1 checkpoint in the directory `path` and returns it (02 §4).

    The directory holds `config.json` and either `model.safetensors` or
    `model.safetensors.index.json` with the shards it names. The configuration must give an
    `ArchSpec`, equal to `expected` when that is given. The files must hold exactly the tensors of
    that architecture, each in BF16 with its shape. Headers are checked whole before anything is
    loaded: every tensor's bytes inside the file, the tensors tiling the data, the header bounded.

    Raises:
        ArchitectureError: for anything else, including a directory with pickle files and no
            safetensors checkpoint, which is never opened.
    """
    directory = Path(path)
    if not directory.is_dir():
        raise ArchitectureError(f"{directory} is not a checkpoint directory")
    config = directory / "config.json"
    if not config.is_file():
        raise ArchitectureError(f"{directory} has no config.json")
    arch = read_config(config.read_text(encoding="utf-8"))
    if expected is not None and arch != expected:
        fields = ArchSpec.__dataclass_fields__
        differ = [f for f in fields if getattr(arch, f) != getattr(expected, f)]
        raise ArchitectureError(
            f"the checkpoint's architecture differs from the expected in {differ}"
        )
    shapes = tensor_shapes(arch)
    placement = _placement(directory)
    missing = sorted(shapes.keys() - placement.keys())
    extra = sorted(placement.keys() - shapes.keys())
    if missing or extra:
        raise ArchitectureError(f"tensors missing {missing[:5]}, unexpected {extra[:5]}")
    by_file: dict[str, set[str]] = {}
    for name, file in placement.items():
        by_file.setdefault(file, set()).add(name)
    for file, names in sorted(by_file.items()):
        header = _read_header(directory / file)
        if header.keys() != names:
            raise ArchitectureError(f"{file} does not hold exactly the tensors placed in it")
        for name in sorted(names):
            dtype, shape, size = header[name]
            if dtype != "BF16":
                raise ArchitectureError(f"{name} is {dtype}, not BF16")
            if shape != shapes[name]:
                raise ArchitectureError(f"{name} has shape {shape}, not {shapes[name]}")
            if size != 2 * _elements(shape):
                raise ArchitectureError(f"{name} does not hold its shape's bytes in {file}")
    return FloatModel(arch, {name: directory / placement[name] for name in shapes})


@dataclass(frozen=True)
class Tokenizer:
    """What drawing calibration and evaluation data needs of a checkpoint's tokenizer."""

    encode: Callable[[str], Sequence[int]]
    """A text's tokens, without special tokens."""
    bos: int
    """The beginning-of-text token."""


def read_tokenizer(path: Path, arch: ArchSpec) -> Tokenizer:
    """The tokenizer of the checkpoint in the directory `path`, whose architecture is `arch`.

    It is the directory's `tokenizer.json`, a JSON file that the `tokenizers` library parses
    without running code, with the beginning-of-text token `bos_token_id` of its `config.json`, as
    Llama-3.1 checkpoints ship them. Every token it can give lies in the vocabulary.

    Raises:
        ArchitectureError: for a missing or malformed tokenizer, a configuration without one
            integer `bos_token_id`, or a token outside the vocabulary.
    """
    directory = Path(path)
    file = directory / TOKENIZER
    if not file.is_file():
        raise ArchitectureError(f"{directory} has no {TOKENIZER}")
    try:
        tokenizer = tokenizers.Tokenizer.from_file(str(file))
    except Exception as error:  # the library raises its own exception types
        raise ArchitectureError(f"{TOKENIZER} is not a tokenizer: {error}") from None
    try:
        config: object = json.loads((directory / "config.json").read_text(encoding="utf-8"))
    except ValueError as error:
        raise ArchitectureError(f"config.json is not JSON: {error}") from None
    bos = cast(dict[str, object], config).get("bos_token_id") if isinstance(config, dict) else None
    if not isinstance(bos, int) or isinstance(bos, bool):
        raise ArchitectureError("config.json has no integer bos_token_id")
    if not 0 <= bos < arch.vocab or tokenizer.get_vocab_size(with_added_tokens=True) > arch.vocab:
        raise ArchitectureError(
            f"the tokenizer's tokens lie outside the vocabulary of {arch.vocab}"
        )

    def encode(text: str) -> list[int]:
        return tokenizer.encode(text, add_special_tokens=False).ids

    return Tokenizer(encode, bos)


def _placement(directory: Path) -> dict[str, str]:
    """Each tensor's file name, from the index or the single file."""
    index = directory / INDEX
    if index.is_file():
        try:
            document: Any = json.loads(index.read_text(encoding="utf-8"))
        except ValueError as error:
            raise ArchitectureError(f"{INDEX} is not JSON: {error}") from None
        fields = cast(dict[str, Any], document) if isinstance(document, dict) else {}
        weight_map: Any = fields.get("weight_map")
        if not isinstance(weight_map, dict):
            raise ArchitectureError(f"{INDEX} has no weight_map")
        placement: dict[str, str] = {}
        for name, file in cast(dict[Any, Any], weight_map).items():
            if not isinstance(name, str) or not isinstance(file, str) or not _plain(file):
                raise ArchitectureError(f"{INDEX} places a tensor in {file!r}, not a shard here")
            if not (directory / file).is_file():
                raise ArchitectureError(f"{INDEX} names {file}, which is missing")
            placement[name] = file
        return placement
    single = directory / SINGLE
    if single.is_file():
        return dict.fromkeys(_read_header(single), SINGLE)
    if any(p.suffix in PICKLE_SUFFIXES for p in directory.iterdir()):
        raise ArchitectureError(
            f"{directory} holds a pickle-format checkpoint and no safetensors one; pickle is "
            "never loaded (06 §3, TB-5): convert it to safetensors"
        )
    raise ArchitectureError(f"{directory} holds no safetensors checkpoint")


def _plain(file: str) -> bool:
    """A shard's name in its directory: no path, no leading dot, the safetensors suffix."""
    return (
        file.endswith(".safetensors")
        and "/" not in file
        and "\\" not in file
        and not file.startswith(".")
    )


def _read_header(path: Path) -> dict[str, tuple[str, tuple[int, ...], int]]:
    """Each tensor's dtype, shape and byte length, from a safetensors header checked whole.

    The header is at most `MAX_HEADER_BYTES`, and the tensors' byte ranges tile the data that
    follows it, from its first byte to the end of the file, in some order.
    """
    with path.open("rb") as file:
        prefix = file.read(8)
        if len(prefix) != 8:
            raise ArchitectureError(f"{path.name} is too short for a safetensors header")
        (size,) = struct.unpack("<Q", prefix)
        if size > MAX_HEADER_BYTES:
            raise ArchitectureError(f"{path.name}'s header is larger than {MAX_HEADER_BYTES} bytes")
        raw = file.read(size)
    if len(raw) != size:
        raise ArchitectureError(f"{path.name} ends inside its header")
    data = path.stat().st_size - 8 - size
    try:
        header: Any = json.loads(raw)
    except ValueError as error:
        raise ArchitectureError(f"{path.name}'s header is not JSON: {error}") from None
    if not isinstance(header, dict):
        raise ArchitectureError(f"{path.name}'s header is not an object")
    entries: dict[str, tuple[str, tuple[int, ...], int]] = {}
    ranges: list[tuple[int, int]] = []
    for name, entry in cast(dict[Any, Any], header).items():
        if name == "__metadata__":
            continue
        dtype, shape, begin, end = _entry(path.name, name, entry)
        entries[name] = (dtype, shape, end - begin)
        ranges.append((begin, end))
    position = 0
    for begin, end in sorted(ranges):
        if begin != position:
            raise ArchitectureError(f"{path.name}'s tensors overlap or leave a gap in the data")
        position = end
    if position != data:
        raise ArchitectureError(f"{path.name}'s tensors do not end where the file does")
    return entries


def _entry(file: str, name: object, entry: object) -> tuple[str, tuple[int, ...], int, int]:
    """One header entry: dtype, shape and byte range, each of the right type."""
    if not isinstance(name, str) or not isinstance(entry, dict):
        raise ArchitectureError(f"{file}'s header has a malformed entry")
    fields = cast(dict[str, Any], entry)
    dtype, shape, offsets = fields.get("dtype"), fields.get("shape"), fields.get("data_offsets")
    if not isinstance(dtype, str) or not _integers(shape) or not _integers(offsets):
        raise ArchitectureError(f"{file}'s entry for {name} is malformed")
    begin_end = cast(list[int], offsets)
    if len(begin_end) != 2 or not 0 <= begin_end[0] <= begin_end[1]:
        raise ArchitectureError(f"{file}'s entry for {name} has a malformed byte range")
    return dtype, tuple(cast(list[int], shape)), begin_end[0], begin_end[1]


def _integers(value: object) -> bool:
    """A list of non-negative integers, booleans excluded."""
    return isinstance(value, list) and all(
        isinstance(v, int) and not isinstance(v, bool) and v >= 0 for v in cast(list[Any], value)
    )


def _elements(shape: tuple[int, ...]) -> int:
    count = 1
    for dim in shape:
        count *= dim
    return count
