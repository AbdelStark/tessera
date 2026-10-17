# SPDX-License-Identifier: Apache-2.0 OR MIT
"""Calibration statistics (RFC-0005 'Calibration', 'Determinism').

The BF16 reference built from the checked files computes what Hugging Face's own loader gives,
bit for bit. The statistics are the activations' per-channel maxima, checked against an
independent pass. The same inputs give the same bytes. Non-finite or degenerate statistics are
refused, a reference whose RMSNorm departs from the capture's recomputation is caught, and the
statistics file is written as a secret artifact, beside a report that names them by digest only.
"""

import json
import stat
from pathlib import Path
from typing import Any, cast

import pytest
import torch
from checkpoints import TINY, TINY_ARCH, Weights, tensors, write
from transformers import LlamaForCausalLM
from transformers.models.llama import modeling_llama

from tessera_quant import CalibrationError, TesseraQuantError, import_checkpoint
from tessera_quant.calibration import CalibrationSet, DatasetIdentity, draw
from tessera_quant.capture import (
    REPORT_FILE,
    STATISTICS_FILE,
    calibrate,
    capture,
    points,
    reference_model,
    statistics_digest,
    write_secret,
)

EXAMPLE = Path(__file__).parents[2] / "docs" / "perf" / "examples" / "calibration-tiny.json"

SOURCE = DatasetIdentity(
    name="fixture", config="en", split="validation", revision="0" * 40, files=("fixture",)
)


def sequences(count: int = 3, length: int = 12) -> CalibrationSet:
    """A calibration set of token ids inside the tiny vocabulary of 64."""
    documents = [f"document {i} " + "the quick brown fox jumps" * (1 + i % 3) for i in range(20)]
    return draw(
        documents,
        lambda text: [2 + ord(c) % 60 for c in text],
        1,
        source=SOURCE,
        seed=11,
        count=count,
        length=length,
        held_out=0,
    )


def hugging_face(path: Path) -> LlamaForCausalLM:
    """Hugging Face's own loader, on the test's own files: the reference to compare with."""
    # transformers types `from_pretrained` loosely.
    loader: Any = LlamaForCausalLM
    model = cast(
        LlamaForCausalLM,
        loader.from_pretrained(
            path,
            dtype=torch.bfloat16,
            attn_implementation="eager",
            local_files_only=True,
            use_safetensors=True,
        ),
    )
    return model.eval()


def checkpoint(tmp_path: Path, edit: Any = None) -> Path:
    weights: Weights = tensors(TINY_ARCH)
    if edit is not None:
        edit(weights)
    return write(tmp_path / "checkpoint", TINY, weights)


def test_the_reference_computes_what_hugging_faces_loader_gives(tmp_path: Path) -> None:
    path = checkpoint(tmp_path)
    ours = reference_model(import_checkpoint(path))
    theirs = hugging_face(path)
    tokens = torch.tensor([sequences().calibration[0].tokens])
    with torch.no_grad():
        assert torch.equal(ours(input_ids=tokens).logits, theirs(input_ids=tokens).logits)


def channel_maxima(values: torch.Tensor) -> torch.Tensor:
    return values.to(torch.float64).abs().reshape(-1, values.shape[-1]).amax(dim=0)


def test_statistics_are_the_activations_per_channel_maxima(tmp_path: Path) -> None:
    path = checkpoint(tmp_path)
    calibration = sequences()
    statistics = capture(import_checkpoint(path), calibration)
    assert {name: v.shape[0] for name, v in statistics.maxima.items()} == points(TINY_ARCH)
    assert all(v.dtype == torch.float64 for v in statistics.maxima.values())

    # An independent pass: Hugging Face's hidden states, and layer 0's queries after RoPE.
    model = hugging_face(path)
    layer0: Any = cast(Any, model.model.layers)[0]
    inputs: dict[int, list[torch.Tensor]] = {}
    queries: list[torch.Tensor] = []
    # transformers types `apply_rotary_pos_emb` loosely.
    rope: Any = cast(Any, modeling_llama).apply_rotary_pos_emb
    with torch.no_grad():
        for window in calibration.calibration:
            tokens = torch.tensor([window.tokens])
            out: Any = model(input_ids=tokens, output_hidden_states=True)
            for layer in range(TINY_ARCH.n_layers):
                inputs.setdefault(layer, []).append(out.hidden_states[layer])
            x = layer0.input_layernorm(out.hidden_states[0])
            q = layer0.self_attn.q_proj(x).view(1, len(window.tokens), -1, TINY_ARCH.d_head)
            q = q.transpose(1, 2)
            positions = torch.arange(len(window.tokens)).unsqueeze(0)
            cos, sin = cast(Any, model.model.rotary_emb)(x, positions)
            rotated: Any = rope(q, q, cos, sin)[0]
            queries.append(rotated.transpose(1, 2).reshape(len(window.tokens), -1))
    for layer, found in inputs.items():
        expected = torch.stack([channel_maxima(h) for h in found]).amax(dim=0)
        assert torch.equal(statistics.maxima[f"layers.{layer}.attn_input"], expected), layer
    expected_q = torch.stack([channel_maxima(q) for q in queries]).amax(dim=0)
    assert torch.equal(statistics.maxima["layers.0.q"], expected_q)


def test_the_same_inputs_give_the_same_statistics_bytes(tmp_path: Path) -> None:
    model = import_checkpoint(checkpoint(tmp_path))
    calibration = sequences()
    first = capture(model, calibration).to_bytes(calibration)
    second = capture(model, calibration).to_bytes(calibration)
    assert first == second
    assert statistics_digest(first) == statistics_digest(second)
    other = sequences(count=4)
    assert capture(model, other).to_bytes(other) != first


def infinite_weight(weights: Weights) -> None:
    weights["model.layers.0.self_attn.q_proj.weight"][0, 0] = float("inf")


def silent_gate(weights: Weights) -> None:
    weights["model.layers.1.mlp.gate_proj.weight"].zero_()


@pytest.mark.parametrize(
    ("edit", "reason"),
    [
        (infinite_weight, "layers.0.q: a channel's maximum is not finite"),
        (silent_gate, "layers.1.gate: every channel is zero"),
    ],
)
def test_untrustworthy_statistics_are_refused(tmp_path: Path, edit: Any, reason: str) -> None:
    model = import_checkpoint(checkpoint(tmp_path, edit))
    with pytest.raises(CalibrationError, match=reason) as refused:
        capture(model, sequences())
    assert refused.value.code == "ERR-QUANT-CALIB"


def test_a_reference_norm_the_capture_does_not_recompute_is_caught(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    model = import_checkpoint(checkpoint(tmp_path))
    original = modeling_llama.LlamaRMSNorm.forward

    def shifted(self: Any, hidden_states: torch.Tensor) -> torch.Tensor:
        return original(self, hidden_states) * 2

    monkeypatch.setattr(modeling_llama.LlamaRMSNorm, "forward", shifted)
    with pytest.raises(TesseraQuantError, match="no longer computes"):
        capture(model, sequences())


def test_the_statistics_file_is_written_as_a_secret_artifact(tmp_path: Path) -> None:
    path = tmp_path / "statistics.safetensors"
    path.write_bytes(b"an earlier file")
    write_secret(path, b"the statistics")
    assert path.read_bytes() == b"the statistics"
    assert stat.S_IMODE(path.stat().st_mode) == 0o600
    assert sorted(p.name for p in tmp_path.iterdir()) == ["statistics.safetensors"]


def floats(value: object) -> list[float]:
    if isinstance(value, float):
        return [value]
    if isinstance(value, dict):
        return [f for item in cast(dict[str, object], value).values() for f in floats(item)]
    if isinstance(value, list):
        return [f for item in cast(list[object], value) for f in floats(item)]
    return []


def test_calibrate_keeps_the_statistics_secret_and_reports_them_by_digest(tmp_path: Path) -> None:
    model = import_checkpoint(checkpoint(tmp_path))
    out = tmp_path / "out"
    out.mkdir()
    report = calibrate(model, sequences(), out)
    data = (out / STATISTICS_FILE).read_bytes()
    assert stat.S_IMODE((out / STATISTICS_FILE).stat().st_mode) == 0o600
    assert report["statistics_digest"] == statistics_digest(data).hex()
    assert json.loads((out / REPORT_FILE).read_text(encoding="utf-8")) == report
    # Integers, strings and booleans only: no statistic, which would be a float, reaches it.
    assert floats(report) == []
    # The committed example, whose schema `cargo xtask lint` checks, is this report, but for the
    # statistics digest and the environment, which depend on hardware and libraries.
    example = json.loads(EXAMPLE.read_text(encoding="utf-8"))
    bound = {"statistics_digest", "environment"}
    assert {k: v for k, v in report.items() if k not in bound} == {
        k: v for k, v in example.items() if k not in bound
    }
