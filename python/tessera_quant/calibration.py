# SPDX-License-Identifier: Apache-2.0 OR MIT
"""The calibration set, drawn with a fixed seed from a pinned dataset (RFC-0005 'Calibration').

The draw is defined exactly, so that the same seed, documents and tokenizer give the same
sequences on any machine and with any library version:

- **Uniform integers** come from a SplitMix64 stream seeded with the seed. `uniform(n)` draws
  `x` and returns `x mod n` once `x < 2^64 - (2^64 mod n)`, drawing again otherwise, so every
  value is exactly equally likely.
- **Order.** The documents are visited in the order of a Fisher-Yates shuffle of their indices:
  for `i` from `N - 1` down to 1, swap positions `i` and `uniform(i + 1)`.
- **Sequences.** A visited document is tokenized without special tokens. One of at least `L - 1`
  tokens gives a sequence: the beginning-of-text token, then its `L - 1` tokens from position
  `uniform(t - L + 2)`, `t` its token count. Shorter documents are skipped.
- **Splits.** The first `count` sequences form the calibration set, and the next `held_out` the
  split on which the migration strength is chosen. No document gives two sequences.

The selection is identified by its digest. It covers every selected document's index, window
start, text and tokens, in draw order, so equal digests mean equal sequences. Each document
also has a content hash, by which calibration and evaluation data are checked disjoint (RFC-0005
'Fidelity evaluation', hygiene rules). Both digests are BLAKE3 in key-derivation mode, under
contexts of their own.
"""

import gzip
import json
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

import blake3

from tessera_quant.errors import CalibrationError

C4_DATASET = "allenai/c4"
"""The Hugging Face dataset of the default calibration set."""

C4_REVISION = "1588ec454efa1a09f29cd18ddd04fe05fc8653a2"
"""The pinned revision of `allenai/c4`: the commit of 2024-01-09."""

C4_VALIDATION_FILES = tuple(f"en/c4-validation.{i:05}-of-00008.json.gz" for i in range(8))
"""The files of the `en` validation split, in order."""

DEFAULT_COUNT = 512
"""Sequences in the default calibration set."""

DEFAULT_LENGTH = 2048
"""Tokens per sequence, the beginning-of-text token included."""

DEFAULT_HELD_OUT = 64
"""Sequences in the split held out for choosing the migration strength."""

_SELECTION_CONTEXT = "tessera-quant 2026-10-17 calibration selection v1"
_DOCUMENT_CONTEXT = "tessera-quant 2026-10-17 document v1"

_MASK = (1 << 64) - 1


class SplitMix64:
    """The SplitMix64 generator.

    A 64-bit state advances by the golden-ratio increment, and each output mixes it by two
    multiply-xorshift rounds, as Vigna's reference `splitmix64.c` does.
    """

    def __init__(self, seed: int) -> None:
        """A stream whose state starts at `seed`, an integer in `[0, 2^64)`."""
        if not 0 <= seed <= _MASK:
            raise ValueError(f"a seed is in [0, 2^64), not {seed}")
        self._state = seed

    def next_u64(self) -> int:
        """The next output."""
        self._state = (self._state + 0x9E3779B97F4A7C15) & _MASK
        z = self._state
        z = ((z ^ (z >> 30)) * 0xBF58476D1CE4E5B9) & _MASK
        z = ((z ^ (z >> 27)) * 0x94D049BB133111EB) & _MASK
        return z ^ (z >> 31)

    def uniform(self, n: int) -> int:
        """An integer uniform on `[0, n)`.

        It is `x mod n` for the first output `x` below the largest multiple of `n` that fits 64
        bits: rejecting the others makes every value exactly equally likely.
        """
        if not 1 <= n <= 1 << 64:
            raise ValueError(f"a range has 1 to 2^64 values, not {n}")
        limit = (1 << 64) - (1 << 64) % n
        while True:
            x = self.next_u64()
            if x < limit:
                return x % n


@dataclass(frozen=True)
class DatasetIdentity:
    """Where the documents come from: a dataset, its revision and the files read, in order."""

    name: str
    config: str
    split: str
    revision: str
    files: tuple[str, ...]


@dataclass(frozen=True)
class Window:
    """One sequence: the document it comes from, by index, and where its tokens start."""

    document: int
    start: int
    tokens: tuple[int, ...]
    """The beginning-of-text token, then the document's tokens from `start` on."""


@dataclass(frozen=True)
class CalibrationSet:
    """The sequences drawn for calibration and for the held-out split, and what identifies them."""

    source: DatasetIdentity
    seed: int
    length: int
    calibration: tuple[Window, ...]
    held_out: tuple[Window, ...]
    selection_digest: bytes
    """The digest of every selected document's index, window start, text and tokens, in draw
    order."""
    document_hashes: frozenset[bytes]
    """The content hash of every selected document, both splits."""


def document_hash(text: str) -> bytes:
    """A document's content hash: BLAKE3 of its UTF-8 bytes, under its own key-derivation context.

    Calibration and evaluation data are disjoint when no document of one has the hash of a
    document of the other.
    """
    return blake3.blake3(text.encode("utf-8"), derive_key_context=_DOCUMENT_CONTEXT).digest()


def draw(
    documents: Sequence[str],
    encode: Callable[[str], Sequence[int]],
    bos: int,
    *,
    source: DatasetIdentity,
    seed: int,
    count: int = DEFAULT_COUNT,
    length: int = DEFAULT_LENGTH,
    held_out: int = DEFAULT_HELD_OUT,
) -> CalibrationSet:
    """Draws the calibration set and the held-out split from `documents`.

    `count` calibration sequences and `held_out` more, of `length` tokens each, as the module
    documentation defines the draw. `encode` tokenizes a text without special tokens, and `bos`
    is the beginning-of-text token.

    Raises:
        CalibrationError: when the documents hold fewer than `count + held_out` that are long
            enough (`ERR-QUANT-CALIB`).
        ValueError: for a length below 2 or a negative count.
    """
    if length < 2:
        raise ValueError(f"a sequence holds at least 2 tokens, not {length}")
    if count < 0 or held_out < 0:
        raise ValueError("the split sizes are not negative")
    rng = SplitMix64(seed)
    order = list(range(len(documents)))
    for i in range(len(order) - 1, 0, -1):
        j = rng.uniform(i + 1)
        order[i], order[j] = order[j], order[i]
    wanted = count + held_out
    windows: list[Window] = []
    for index in order:
        if len(windows) == wanted:
            break
        tokens = encode(documents[index])
        span = len(tokens) - (length - 1)
        if span < 0:
            continue
        start = rng.uniform(span + 1)
        windows.append(Window(index, start, (bos, *tokens[start : start + length - 1])))
    if len(windows) < wanted:
        raise CalibrationError(
            f"{len(documents)} documents give {len(windows)} sequences of {length} tokens, "
            f"not the {wanted} drawn"
        )
    selection = blake3.blake3(derive_key_context=_SELECTION_CONTEXT)
    selection.update(len(windows).to_bytes(8, "little"))
    for window in windows:
        text = documents[window.document].encode("utf-8")
        selection.update(window.document.to_bytes(8, "little"))
        selection.update(window.start.to_bytes(8, "little"))
        selection.update(len(text).to_bytes(8, "little"))
        selection.update(text)
        for token in window.tokens:
            selection.update(token.to_bytes(4, "little"))
    return CalibrationSet(
        source=source,
        seed=seed,
        length=length,
        calibration=tuple(windows[:count]),
        held_out=tuple(windows[count:]),
        selection_digest=selection.digest(),
        document_hashes=frozenset(document_hash(documents[w.document]) for w in windows),
    )


def check_disjoint(calibration: CalibrationSet, evaluation: Iterable[str], name: str) -> None:
    """Checks that the evaluation set `name` shares no document with the calibration set.

    This is the hygiene rule of RFC-0005 'Fidelity evaluation': documents are compared by
    content hash.

    Raises:
        CalibrationError: naming the evaluation set and how many of its documents the
            calibration set shares (`ERR-QUANT-CALIB`).
    """
    shared = sum(1 for text in evaluation if document_hash(text) in calibration.document_hashes)
    if shared:
        raise CalibrationError(
            f"the evaluation set {name} shares {shared} documents with the calibration set"
        )


def read_json_lines(path: Path) -> list[str]:
    """The `text` field of every line of a gzip-compressed JSON-lines file, in order: a C4 file.

    Raises:
        CalibrationError: for a line that is not a JSON object with a string `text`.
    """
    texts: list[str] = []
    with gzip.open(path, "rt", encoding="utf-8") as lines:
        for number, line in enumerate(lines, 1):
            try:
                record: object = json.loads(line)
            except ValueError as error:
                raise CalibrationError(f"{path.name}:{number} is not JSON: {error}") from None
            text = cast(dict[str, object], record).get("text") if isinstance(record, dict) else None
            if not isinstance(text, str):
                raise CalibrationError(f"{path.name}:{number} has no string `text`")
            texts.append(text)
    return texts


def fetch_c4_validation(cache: Path | None = None) -> tuple[DatasetIdentity, list[str]]:
    """The documents of C4's `en` validation split at the pinned revision.

    The files come through the Hugging Face Hub, which checks each against the revision's
    recorded hash.

    Raises:
        CalibrationError: as `read_json_lines`.
    """
    import huggingface_hub

    # huggingface_hub types some parameters of `hf_hub_download` loosely.
    hub: Any = huggingface_hub
    documents: list[str] = []
    for file in C4_VALIDATION_FILES:
        local = cast(
            str,
            hub.hf_hub_download(
                C4_DATASET, file, repo_type="dataset", revision=C4_REVISION, cache_dir=cache
            ),
        )
        documents.extend(read_json_lines(Path(local)))
    identity = DatasetIdentity(
        name=C4_DATASET,
        config="en",
        split="validation",
        revision=C4_REVISION,
        files=C4_VALIDATION_FILES,
    )
    return identity, documents
