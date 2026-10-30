# SPDX-License-Identifier: Apache-2.0 OR MIT
"""Bit-preserving explicit preparation before RFC-0005's unchanged importer (TB-5)."""

import hashlib
import json
import os
import stat
import struct
from collections.abc import Callable
from pathlib import Path
from typing import Any, cast

import pytest
import torch
from checkpoints import TINY, TINY_ARCH, char_tokenizer, tensors, write
from transformers import LlamaForCausalLM

from tessera_quant import ArchitectureError, import_checkpoint
from tessera_quant.checkpoint import FloatModel
from tessera_quant.errors import UsageError
from tools import materialize_tied_head as preparation

EMBED = "model.embed_tokens.weight"
HEAD = "lm_head.weight"


def source_at(path: Path, shards: int = 1) -> Path:
    weights = tensors(TINY_ARCH)
    del weights[HEAD]
    weights[EMBED].view(torch.int16).flatten()[:4] = torch.tensor(
        [0, -32768, 32705, -1], dtype=torch.int16
    )  # Include signed zero and distinct NaN payloads: compare bits, never float equality.
    source = write(path, TINY | {"tie_word_embeddings": True}, weights, shards=shards)
    char_tokenizer(source)
    (source / "README.md").write_text("Fixture checkpoint.\n")
    return source


def digest_files(directory: Path) -> dict[str, str]:
    return {
        p.name: hashlib.sha256(p.read_bytes()).hexdigest()
        for p in directory.iterdir()
        if p.is_file()
    }


@pytest.mark.parametrize("shards", [1, 3])
def test_prepare_preserves_every_bit_config_and_source(tmp_path: Path, shards: int) -> None:
    source = source_at(tmp_path / "source", shards)
    raw = (
        (source / "config.json")
        .read_text()
        .replace('"rms_norm_eps": 1e-05', '"rms_norm_eps": 1.000000000000000001e-05')
    )
    (source / "config.json").write_text(raw)
    before = digest_files(source)
    out = tmp_path / "prepared"
    previous = os.umask(0)
    try:
        result = preparation.materialize(source, out)
    finally:
        os.umask(previous)
    assert result["status"] == "verified"
    assert (out / "config.json").read_text() == raw.replace(
        '"tie_word_embeddings": true', '"tie_word_embeddings": false'
    )
    assert digest_files(source) == before
    assert not (out / "model.safetensors").exists()  # HF must discover the complete shard index.
    assert not (out / ".incomplete").exists()
    assert not (out / ".staging").exists()
    assert stat.S_IMODE(out.stat().st_mode) == 0o700
    assert all(stat.S_IMODE(p.stat().st_mode) == 0o600 for p in out.iterdir())
    normalized = import_checkpoint(out)
    assert torch.equal(
        normalized.tensor(HEAD).view(torch.int16), normalized.tensor(EMBED).view(torch.int16)
    )
    assert preparation.verify(source, out)["status"] == "verified"
    with pytest.raises(ArchitectureError, match="tie_word_embeddings"):
        import_checkpoint(source)  # Accepted importer semantics are not relaxed.


@pytest.mark.parametrize(
    "change", ["head", "embedding", "other-weight", "config", "metadata", "tokenizer"]
)
def test_verify_rejects_each_changed_prepared_input(tmp_path: Path, change: str) -> None:
    source = source_at(tmp_path / "source")
    out = tmp_path / "prepared"
    preparation.materialize(source, out)
    if change == "config":
        config = out / "config.json"
        config.write_text(
            config.read_text().replace('"rms_norm_eps": 1e-05', '"rms_norm_eps": 2e-05')
        )
    elif change == "tokenizer":
        (out / "tokenizer.json").write_text("{}")
    elif change == "metadata":
        config = out / "config.json"
        config.write_text(config.read_text().rstrip()[:-1] + ', "operator_note": "changed"}')
    else:
        model = import_checkpoint(out)
        name = {"head": HEAD, "embedding": EMBED, "other-weight": "model.norm.weight"}[change]
        entry = preparation._ranges(model.files[name])[name]  # pyright: ignore[reportPrivateUsage]
        with entry.path.open("r+b") as stream:
            stream.seek(entry.start)
            original = stream.read(1)
            stream.seek(entry.start)
            stream.write(bytes([original[0] ^ 1]))
    with pytest.raises(ArchitectureError):
        preparation.verify(source, out)


@pytest.mark.parametrize("change", ["head-present", "missing", "extra", "dtype", "unsafe-index"])
def test_all_source_metadata_is_validated_before_output(tmp_path: Path, change: str) -> None:
    weights = tensors(TINY_ARCH)
    if change != "head-present":
        del weights[HEAD]
    if change == "missing":
        del weights["model.norm.weight"]
    elif change == "extra":
        weights["unexpected.weight"] = torch.zeros(1, dtype=torch.bfloat16)
    elif change == "dtype":
        weights["model.norm.weight"] = weights["model.norm.weight"].float()
    source = write(tmp_path / "source", TINY | {"tie_word_embeddings": True}, weights, shards=2)
    if change == "unsafe-index":
        index = source / "model.safetensors.index.json"
        external = tmp_path / "model-00001-of-00002.safetensors"
        external.write_bytes((source / external.name).read_bytes())
        index.write_text(index.read_text().replace("model-00001", "../model-00001"))
    out = tmp_path / "prepared"
    with pytest.raises(ArchitectureError):
        preparation.materialize(source, out)
    assert not out.exists()


@pytest.mark.parametrize("nested", [False, True])
def test_duplicate_config_members_are_refused(tmp_path: Path, nested: bool) -> None:
    source = source_at(tmp_path / "source")
    config = source / "config.json"
    raw = config.read_text()
    if nested:
        raw = raw.replace('"factor": 8.0', '"factor": 8.0, "factor": 8.0')
    else:
        raw = raw.replace(
            '"tie_word_embeddings": true',
            '"tie_word_embeddings": true, "tie_word_embeddings": false',
        )
    config.write_text(raw)
    with pytest.raises(ArchitectureError, match="duplicate"):
        preparation.materialize(source, tmp_path / "prepared")


@pytest.mark.parametrize("location", ["source", "inside", "existing"])
def test_destinations_are_exclusive_and_outside_source(tmp_path: Path, location: str) -> None:
    source = source_at(tmp_path / "source")
    out = {"source": source, "inside": source / "prepared", "existing": tmp_path / "existing"}[
        location
    ]
    if location == "existing":
        out.mkdir()
        (out / "keep").write_text("untouched")
    before = digest_files(source)
    with pytest.raises(UsageError):
        preparation.materialize(source, out)
    assert digest_files(source) == before
    if location == "existing":
        assert (out / "keep").read_text() == "untouched"


def test_failure_retains_private_partial_evidence_without_importable_output(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source = source_at(tmp_path / "source")
    out = tmp_path / "prepared"
    before = digest_files(source)

    def fail(*args: Any, **kwargs: Any) -> None:
        raise OSError("synthetic write failure")

    monkeypatch.setattr(preparation, "_copy_exact", fail)
    with pytest.raises(OSError, match="synthetic"):
        preparation.materialize(source, out)
    assert (out / ".incomplete").is_file()
    assert (out / ".staging").is_dir()
    assert not (out / "config.json").exists()
    assert digest_files(source) == before
    with pytest.raises(ArchitectureError):
        import_checkpoint(out)


def test_preparation_does_not_load_model_tensors(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source = source_at(tmp_path / "source", 3)

    def forbidden(*args: Any, **kwargs: Any) -> Any:
        pytest.fail("checkpoint conversion must stream bytes without loading a model tensor")

    monkeypatch.setattr(FloatModel, "tensor", forbidden)
    monkeypatch.setattr(torch, "load", forbidden)
    monkeypatch.setattr(preparation, "_CHUNK", 97)
    assert preparation.materialize(source, tmp_path / "prepared")["status"] == "verified"


def test_every_tensor_read_is_bounded_before_copy_or_comparison(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source = source_at(tmp_path / "source", 3)
    original = cast(Callable[..., Any], Path.open)
    reads: list[int] = []

    class Bounded:
        def __init__(self, stream: Any) -> None:
            self.stream = stream
            prefix = stream.read(8)
            self.header = struct.unpack("<Q", prefix)[0]
            stream.seek(0)

        def read(self, size: int = -1) -> bytes:
            assert size >= 0, "unbounded reads can materialize a whole checkpoint"
            if not (self.stream.tell() == 8 and size == self.header):
                assert size <= 97
                reads.append(size)
            return self.stream.read(size)

        def __enter__(self) -> "Bounded":
            return self

        def __exit__(self, *args: Any) -> None:
            self.stream.close()

        def seek(self, position: int) -> int:
            return self.stream.seek(position)

    def bounded(path: Path, *args: Any, **kwargs: Any) -> Any:
        stream: Any = original(path, *args, **kwargs)
        return Bounded(stream) if path.suffix == ".safetensors" and args == ("rb",) else stream

    monkeypatch.setattr(Path, "open", bounded)
    monkeypatch.setattr(preparation, "_CHUNK", 97)
    assert preparation.materialize(source, tmp_path / "prepared")["status"] == "verified"
    assert 97 in reads


def test_a_source_changed_during_copy_is_not_published(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source = source_at(tmp_path / "source", 3)
    original = preparation._copy_file  # pyright: ignore[reportPrivateUsage]
    changed = False

    def changing(path: Path, target: Path) -> None:
        nonlocal changed
        original(path, target)
        if not changed and path.suffix == ".safetensors":
            changed = True
            # Source content stays equal; an identity/mtime change must still abort the run.
            status = path.stat()
            path.touch()
            assert path.stat().st_mtime_ns != status.st_mtime_ns

    monkeypatch.setattr(preparation, "_copy_file", changing)
    out = tmp_path / "prepared"
    with pytest.raises(ArchitectureError, match="source changed"):
        preparation.materialize(source, out)
    assert not (out / "config.json").exists()
    assert (out / ".incomplete").exists()


def test_overlapping_tensor_ranges_and_pickle_only_sources_are_refused(tmp_path: Path) -> None:
    source = source_at(tmp_path / "source")
    path = source / "model.safetensors"
    with path.open("r+b") as stream:
        size = struct.unpack("<Q", stream.read(8))[0]
        header = json.loads(stream.read(size))
        name = next(n for n in header if n != "__metadata__")
        header[name]["data_offsets"][0] += 1
        encoded = json.dumps(header, separators=(",", ":")).encode()
        assert len(encoded) <= size
        stream.seek(8)
        stream.write(encoded + b" " * (size - len(encoded)))
    with pytest.raises(ArchitectureError):
        preparation.materialize(source, tmp_path / "bad-range")
    pickle = tmp_path / "pickle"
    pickle.mkdir()
    (pickle / "config.json").write_text(json.dumps(TINY | {"tie_word_embeddings": True}))
    (pickle / "pytorch_model.bin").write_bytes(b"never unpickle")
    with pytest.raises(ArchitectureError, match="pickle"):
        preparation.materialize(pickle, tmp_path / "pickle-out")


def test_escaped_and_nested_config_keys_do_not_change_the_wrong_boolean(tmp_path: Path) -> None:
    source = source_at(tmp_path / "source")
    raw = (source / "config.json").read_text()
    raw = raw.replace('"tie_word_embeddings": true', '"tie\\u005fword_embeddings": true')
    raw = raw[:-1] + ', "metadata": {"tie_word_embeddings": true}}'
    (source / "config.json").write_text(raw)
    out = tmp_path / "prepared"
    preparation.materialize(source, out)
    assert (out / "config.json").read_text() == raw.replace(
        '"tie\\u005fword_embeddings": true', '"tie\\u005fword_embeddings": false'
    )


def test_hugging_face_loads_every_prepared_shard_and_preserves_bf16_logits(tmp_path: Path) -> None:
    weights = tensors(TINY_ARCH)
    del weights[HEAD]
    source = write(tmp_path / "source", TINY | {"tie_word_embeddings": True}, weights)
    out = tmp_path / "prepared"
    preparation.materialize(source, out)
    models: list[LlamaForCausalLM] = []
    loader: Any = LlamaForCausalLM
    for path in (source, out):
        loaded, info = loader.from_pretrained(
            path,
            dtype=torch.bfloat16,
            attn_implementation="eager",
            local_files_only=True,
            use_safetensors=True,
            output_loading_info=True,
        )
        assert not any(info[key] for key in ("missing_keys", "unexpected_keys", "mismatched_keys"))
        models.append(cast(LlamaForCausalLM, loaded).eval())
    with torch.no_grad():
        tokens = torch.tensor([[1, 3, 5, 7, 9]])
        a, b = models[0](tokens).logits, models[1](tokens).logits
    assert torch.equal(a, b)


@pytest.mark.parametrize(
    "error", [RuntimeError("private-weight-canary"), OSError("private-weight-canary")]
)
def test_cli_errors_do_not_echo_private_exception_payloads(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    error: Exception,
) -> None:
    def fail(*args: Any) -> Any:
        raise error

    monkeypatch.setattr(preparation, "materialize", fail)
    assert preparation.main(["prepare", str(tmp_path / "source"), str(tmp_path / "out")]) != 0
    captured = capsys.readouterr()
    assert "private-weight-canary" not in captured.out + captured.err
    result = json.loads(captured.out)
    assert result["error"]["code"] == (
        "ERR-RES-IO" if isinstance(error, OSError) else "ERR-INTERNAL"
    )
