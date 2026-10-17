# SPDX-License-Identifier: Apache-2.0 OR MIT
"""The tiny conformance models of `spec/vectors/models/` (07 §5), as tessera-quant reads them.

Each artifact decodes under the `IntegerModel` limits, with the digest the index states, into the
shapes the index states, and encodes back to its bytes. `test_reference.py` checks that the
integer reference gives each sequence's residuals and logits.
"""

import json
from pathlib import Path
from typing import Any

import pytest

from tessera_quant.integer_model import INTEGER_MODEL, Profile
from tessera_quant.tce import INTEGER_MODEL_LIMITS, seal, unseal

MODELS = Path(__file__).resolve().parents[2] / "spec" / "vectors" / "models"
INDEX: dict[str, Any] = json.loads((MODELS / "models.json").read_text(encoding="utf-8"))


def test_the_index_lists_every_model() -> None:
    assert (INDEX["schema"], INDEX["version"]) == ("0x0101", "1")
    names = [entry["name"] for entry in INDEX["models"]]
    assert names == ["tiny-2-servable", "tiny-2-minimal", "tiny-4-servable"]


@pytest.mark.parametrize("entry", INDEX["models"], ids=lambda entry: entry["name"])
def test_each_model_decodes_into_its_shapes_and_encodes_back(entry: dict[str, Any]) -> None:
    data = (MODELS / entry["file"]).read_bytes()
    assert len(data) == entry["bytes"]
    assert data[-32:].hex() == entry["digest"]
    model = unseal(data, INTEGER_MODEL, limits=INTEGER_MODEL_LIMITS)
    assert seal(INTEGER_MODEL, model, limits=INTEGER_MODEL_LIMITS) == data
    assert model.profile is Profile[entry["profile"]]
    for field, value in entry["shape"].items():
        assert getattr(model.arch, field) == value, field
    vocab, d = model.arch.vocab, model.arch.d_model
    for sequence in entry["sequences"]:
        n = len(sequence["tokens"])
        assert all(0 <= t < vocab for t in sequence["tokens"])
        assert len(sequence["residuals"]) == model.arch.n_layers + 1
        assert all([len(row) for row in cp] == [d] * n for cp in sequence["residuals"])
        assert [len(row) for row in sequence["logits"]] == [vocab] * n
