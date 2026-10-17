# SPDX-License-Identifier: Apache-2.0 OR MIT
"""The calibration set (RFC-0005 'Calibration'): a draw defined exactly, so that a seed and the
documents fix the sequences on any machine, an identity that changes with any of them, and the
hygiene check that keeps evaluation data out of calibration.
"""

import gzip
import json
from pathlib import Path

import pytest

from tessera_quant import CalibrationError
from tessera_quant.calibration import (
    DatasetIdentity,
    SplitMix64,
    check_disjoint,
    document_hash,
    draw,
    read_json_lines,
)

SOURCE = DatasetIdentity(
    name="fixture", config="en", split="validation", revision="0" * 40, files=("fixture",)
)
BOS = 1_000_000


def encode(text: str) -> list[int]:
    """One token per character: enough to check windows against their documents."""
    return [ord(c) for c in text]


def corpus(n: int = 60) -> list[str]:
    """Documents of 3 to 39 characters, distinct, some shorter than a sequence of 16."""
    return [f"doc {i:03} " + "abcdefghijklmnopqrstuvwxyz"[: (7 * i) % 30] for i in range(n)]


def test_splitmix64_is_the_published_generator() -> None:
    # The first output of SplitMix64 seeded with 0, as Vigna's reference splitmix64.c gives it.
    assert SplitMix64(0).next_u64() == 0xE220A8397B1DCDAF


class Scripted(SplitMix64):
    """A stream that returns given outputs: a draw that rejection meets only once in 2^64."""

    def __init__(self, outputs: list[int]) -> None:
        super().__init__(0)
        self._outputs = outputs

    def next_u64(self) -> int:
        return self._outputs.pop(0)


def test_uniform_rejects_the_values_that_would_bias_it() -> None:
    # 2^64 mod 3 = 1: the one value 2^64 - 1 would make 0 likelier than 1 and 2, so it is redrawn.
    assert Scripted([(1 << 64) - 1, 5]).uniform(3) == 2
    assert Scripted([(1 << 64) - 2]).uniform(3) == ((1 << 64) - 2) % 3
    # 2^64 mod 2^k = 0: nothing is rejected.
    assert Scripted([(1 << 64) - 1]).uniform(1 << 10) == 1023


def test_every_sequence_is_a_window_of_its_own_document() -> None:
    documents = corpus()
    chosen = draw(documents, encode, BOS, source=SOURCE, seed=7, count=8, length=16, held_out=3)
    assert len(chosen.calibration) == 8
    assert len(chosen.held_out) == 3
    windows = chosen.calibration + chosen.held_out
    assert len({w.document for w in windows}) == 11, "a document gives one sequence at most"
    for w in windows:
        tokens = encode(documents[w.document])
        assert len(w.tokens) == 16
        assert w.tokens[0] == BOS
        assert list(w.tokens[1:]) == tokens[w.start : w.start + 15]
    assert chosen.document_hashes == {document_hash(documents[w.document]) for w in windows}


def test_the_seed_and_the_documents_fix_the_selection() -> None:
    documents = corpus()

    def chosen(seed: int, docs: list[str] = documents) -> bytes:
        return draw(
            docs, encode, BOS, source=SOURCE, seed=seed, count=8, length=16, held_out=3
        ).selection_digest

    assert chosen(7) == chosen(7)
    assert chosen(7) != chosen(8)
    edited = documents.copy()
    first = draw(documents, encode, BOS, source=SOURCE, seed=7, count=8, length=16, held_out=3)
    edited[first.calibration[0].document] += "!"
    assert chosen(7, edited) != chosen(7), "an edited document changes the identity"
    # The identity's encoding is part of a model's provenance: pinned, so that it never moves.
    assert chosen(7).hex() == "deb12264c549c38737ca745ee0d658c068b87164d313b671889af6f0379685e8"


def test_too_few_long_documents_are_refused() -> None:
    with pytest.raises(CalibrationError) as refused:
        draw(corpus(), encode, BOS, source=SOURCE, seed=1, count=200, length=16, held_out=0)
    assert refused.value.code == "ERR-QUANT-CALIB"


def test_an_evaluation_set_that_shares_a_document_is_refused() -> None:
    documents = corpus()
    chosen = draw(documents, encode, BOS, source=SOURCE, seed=7, count=8, length=16, held_out=3)
    unused = [
        d
        for i, d in enumerate(documents)
        if i not in {w.document for w in chosen.calibration + chosen.held_out}
    ]
    check_disjoint(chosen, unused, "unused documents")
    shared = documents[chosen.held_out[0].document]
    with pytest.raises(CalibrationError) as refused:
        check_disjoint(chosen, [*unused[:5], shared], "overlap")
    assert refused.value.code == "ERR-QUANT-CALIB"
    assert "shares 1 documents" in str(refused.value)


def test_a_c4_file_reads_its_texts_in_order_and_refuses_a_malformed_line(tmp_path: Path) -> None:
    good = tmp_path / "good.json.gz"
    with gzip.open(good, "wt", encoding="utf-8") as out:
        for text in ["first", "second é", "third"]:
            out.write(json.dumps({"text": text, "url": "u", "timestamp": "t"}) + "\n")
    assert read_json_lines(good) == ["first", "second é", "third"]
    for line, reason in [("not json\n", "is not JSON"), ('{"text": 3}\n', "has no string `text`")]:
        bad = tmp_path / "bad.json.gz"
        with gzip.open(bad, "wt", encoding="utf-8") as out:
            out.write(json.dumps({"text": "fine"}) + "\n" + line)
        with pytest.raises(CalibrationError, match=f"bad.json.gz:2 {reason}"):
            read_json_lines(bad)
