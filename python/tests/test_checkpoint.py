# SPDX-License-Identifier: Apache-2.0 OR MIT
"""Importing checkpoints (RFC-0005 'Pipeline', step 1; 06 §3, TB-5).

A tiny Llama-3.1 checkpoint imports with its exact architecture, single-file or sharded, and its
tensors read back bit for bit. A checkpoint with the published Llama-3.1-8B configuration and
shapes imports without its 16 GB of data being written: the files are sparse, and the import
reads headers only. Every departure from the architecture or the format raises
`ERR-QUANT-ARCH`, and a pickle file is never opened.
"""

import dataclasses
import json
import shutil
import struct
from collections.abc import Callable
from fractions import Fraction
from pathlib import Path
from typing import Any, cast

import pytest
import safetensors.torch
import torch

from tessera_quant import (
    ArchitectureError,
    ArchSpec,
    RopeSpec,
    TesseraQuantError,
    import_checkpoint,
    read_config,
    tensor_shapes,
)

Weights = dict[str, torch.Tensor]

TINY: dict[str, Any] = {
    "architectures": ["LlamaForCausalLM"],
    "model_type": "llama",
    "hidden_act": "silu",
    "hidden_size": 32,
    "intermediate_size": 64,
    "num_attention_heads": 4,
    "num_key_value_heads": 2,
    "num_hidden_layers": 2,
    "vocab_size": 64,
    "max_position_embeddings": 256,
    "rms_norm_eps": 1e-05,
    "rope_theta": 500000.0,
    "rope_scaling": {
        "factor": 8.0,
        "low_freq_factor": 1.0,
        "high_freq_factor": 4.0,
        "original_max_position_embeddings": 128,
        "rope_type": "llama3",
    },
    "attention_bias": False,
    "mlp_bias": False,
    "tie_word_embeddings": False,
    "torch_dtype": "bfloat16",
}

# Llama-3.1-8B's published config.json, the keys `D` reads.
LLAMA_3_1_8B: dict[str, Any] = TINY | {
    "hidden_size": 4096,
    "intermediate_size": 14336,
    "num_attention_heads": 32,
    "num_key_value_heads": 8,
    "num_hidden_layers": 32,
    "vocab_size": 128256,
    "max_position_embeddings": 131072,
    "rope_scaling": TINY["rope_scaling"] | {"original_max_position_embeddings": 8192},
}

TINY_ARCH = ArchSpec(
    n_layers=2,
    d_model=32,
    n_heads=4,
    n_kv_heads=2,
    d_head=8,
    d_ff=64,
    vocab=64,
    max_context=256,
    rope=RopeSpec(
        theta=Fraction(500000),
        factor=Fraction(8),
        low_freq_factor=Fraction(1),
        high_freq_factor=Fraction(4),
        original_max_position_embeddings=128,
    ),
    rms_eps=Fraction(1, 100000),
)

LAYER0_Q = "model.layers.0.self_attn.q_proj.weight"


def tensors(arch: ArchSpec, seed: int = 7) -> Weights:
    generator = torch.Generator().manual_seed(seed)
    return {
        name: torch.randn(shape, generator=generator).to(torch.bfloat16)
        for name, shape in tensor_shapes(arch).items()
    }


def save(weights: Weights, path: Path) -> None:
    # safetensors' torch helpers are only partly typed.
    cast(Any, safetensors.torch).save_file(weights, str(path))


def write(directory: Path, config: dict[str, Any], weights: Weights, shards: int = 1) -> Path:
    directory.mkdir(parents=True, exist_ok=True)
    (directory / "config.json").write_text(json.dumps(config), encoding="utf-8")
    if shards == 1:
        save(weights, directory / "model.safetensors")
        return directory
    names = list(weights)
    weight_map: dict[str, str] = {}
    for shard in range(shards):
        file = f"model-{shard + 1:05}-of-{shards:05}.safetensors"
        part = names[shard::shards]
        save({name: weights[name] for name in part}, directory / file)
        weight_map |= dict.fromkeys(part, file)
    index = {"metadata": {}, "weight_map": weight_map}
    (directory / "model.safetensors.index.json").write_text(json.dumps(index), encoding="utf-8")
    return directory


def write_header_only(path: Path, shapes: dict[str, tuple[int, ...]], dtype: str = "BF16") -> None:
    """A safetensors file with the right header and a sparse data region of the right length."""
    header: dict[str, Any] = {}
    offset = 0
    for name, shape in shapes.items():
        size = 2 * int(torch.Size(shape).numel())
        header[name] = {
            "dtype": dtype,
            "shape": list(shape),
            "data_offsets": [offset, offset + size],
        }
        offset += size
    raw = json.dumps(header).encode()
    raw += b" " * (-len(raw) % 8)
    with path.open("wb") as file:
        file.write(struct.pack("<Q", len(raw)))
        file.write(raw)
        file.truncate(8 + len(raw) + offset)


def test_a_tiny_checkpoint_imports_with_its_exact_architecture(tmp_path: Path) -> None:
    weights = tensors(TINY_ARCH)
    single = import_checkpoint(write(tmp_path / "single", TINY, weights))
    assert single.arch == TINY_ARCH
    assert single.arch.rms_eps == Fraction(1, 100000), "the decimal 1e-05 read as written"
    assert single.names == tuple(tensor_shapes(TINY_ARCH))
    sharded_dir = write(tmp_path / "sharded", TINY, weights, shards=3)
    sharded = import_checkpoint(sharded_dir, expected=TINY_ARCH)
    assert sharded.arch == TINY_ARCH
    assert sharded.names == single.names
    for model in (single, sharded):
        for name in ("model.embed_tokens.weight", "model.layers.1.mlp.down_proj.weight"):
            tensor = model.tensor(name)
            assert tensor.dtype == torch.bfloat16
            assert torch.equal(tensor, weights[name]), name


def test_a_checkpoint_of_the_published_8b_shapes_imports(tmp_path: Path) -> None:
    arch = read_config(json.dumps(LLAMA_3_1_8B))
    assert (arch.d_model, arch.n_layers, arch.d_head, arch.n_kv_heads) == (4096, 32, 128, 8)
    shapes = tensor_shapes(arch)
    assert len(shapes) == 291
    directory = tmp_path / "llama-3.1-8b"
    directory.mkdir()
    (directory / "config.json").write_text(json.dumps(LLAMA_3_1_8B), encoding="utf-8")
    names = list(shapes)
    weight_map: dict[str, str] = {}
    for shard in range(4):
        file = f"model-{shard + 1:05}-of-00004.safetensors"
        part = names[shard::4]
        write_header_only(directory / file, {name: shapes[name] for name in part})
        weight_map |= dict.fromkeys(part, file)
    index = {"metadata": {}, "weight_map": weight_map}
    (directory / "model.safetensors.index.json").write_text(json.dumps(index), encoding="utf-8")
    model = import_checkpoint(directory, expected=arch)
    assert len(model.names) == 291
    # The sparse data reads as zeros: the model reads tensors on demand, from the shard holding it.
    norm = model.tensor("model.norm.weight")
    assert norm.shape == (4096,)
    assert not norm.any()


def wrong_shape(w: Weights) -> None:
    w[LAYER0_Q] = torch.zeros(32, 31, dtype=torch.bfloat16)


def wrong_dtype(w: Weights) -> None:
    w[LAYER0_Q] = w[LAYER0_Q].float()


def missing(w: Weights) -> None:
    del w[LAYER0_Q]


def unexpected(w: Weights) -> None:
    w["model.layers.2.mlp.up_proj.weight"] = w[LAYER0_Q].clone()


@pytest.mark.parametrize(
    ("config", "change", "reason"),
    [
        (TINY, wrong_shape, "has shape"),
        (TINY, wrong_dtype, "not BF16"),
        (TINY, missing, "tensors missing"),
        (TINY, unexpected, "unexpected"),
        (TINY | {"model_type": "mistral"}, None, "model_type"),
        (
            TINY | {"rope_scaling": TINY["rope_scaling"] | {"rope_type": "default"}},
            None,
            "rope_type",
        ),
        (TINY | {"attention_bias": True}, None, "attention_bias"),
        (TINY | {"tie_word_embeddings": True}, None, "tie_word_embeddings"),
        (TINY | {"num_attention_heads": 5}, None, "do not divide"),
        (TINY | {"num_key_value_heads": 3}, None, "do not divide"),
        (TINY | {"hidden_size": 36, "head_dim": 9}, None, "odd"),
        (TINY | {"head_dim": 16}, None, "head_dim"),
        (TINY | {"num_hidden_layers": 2.0}, None, "num_hidden_layers"),
        ({k: v for k, v in TINY.items() if k != "vocab_size"}, None, "vocab_size"),
        (TINY | {"torch_dtype": "float32"}, None, "torch_dtype"),
    ],
)
def test_departures_from_the_architecture_are_refused_for_their_reason(
    tmp_path: Path,
    config: dict[str, Any],
    change: Callable[[Weights], None] | None,
    reason: str,
) -> None:
    weights = tensors(TINY_ARCH)
    if change is not None:
        change(weights)
    with pytest.raises(ArchitectureError, match=reason) as error:
        import_checkpoint(write(tmp_path, config, weights))
    assert error.value.code == "ERR-QUANT-ARCH"
    assert isinstance(error.value, TesseraQuantError)


def test_an_architecture_other_than_the_expected_is_refused(tmp_path: Path) -> None:
    directory = write(tmp_path, TINY, tensors(TINY_ARCH))
    other = dataclasses.replace(TINY_ARCH, max_context=512)
    with pytest.raises(ArchitectureError, match="max_context"):
        import_checkpoint(directory, expected=other)


def test_a_pickle_checkpoint_is_refused_unopened(tmp_path: Path) -> None:
    marker = tmp_path / "unpickled"
    # A protocol-0 pickle that would create `marker` if anything loaded it.
    payload = b"cbuiltins\nopen\n(S'" + str(marker).encode() + b"'\nS'w'\ntR."
    directory = tmp_path / "pickled"
    directory.mkdir()
    (directory / "config.json").write_text(json.dumps(TINY), encoding="utf-8")
    (directory / "pytorch_model.bin").write_bytes(payload)
    with pytest.raises(ArchitectureError, match="pickle"):
        import_checkpoint(directory)
    assert not marker.exists()


def single_file(directory: Path, body: Callable[[Path], object]) -> Path:
    """A checkpoint directory whose one safetensors file `body` writes."""
    directory.mkdir()
    (directory / "config.json").write_text(json.dumps(TINY), encoding="utf-8")
    body(directory / "model.safetensors")
    return directory


def truncated(path: Path) -> None:
    write_header_only(path, tensor_shapes(TINY_ARCH))
    with path.open("r+b") as file:
        file.truncate(path.stat().st_size - 1)


def overlapping(path: Path) -> None:
    write_header_only(path, tensor_shapes(TINY_ARCH))
    raw = path.read_bytes()
    (size,) = struct.unpack("<Q", raw[:8])
    header: dict[str, Any] = json.loads(raw[8 : 8 + size])
    first, second = list(header)[:2]
    header[second]["data_offsets"][0] = header[first]["data_offsets"][0]
    body = json.dumps(header).encode()
    body += b" " * (-len(body) % 8)
    path.write_bytes(struct.pack("<Q", len(body)) + body + raw[8 + size :])


def too_short(path: Path) -> None:
    path.write_bytes(b"\x01\x02")


def huge_header(path: Path) -> None:
    path.write_bytes(struct.pack("<Q", 2**40))


def cut_header(path: Path) -> None:
    path.write_bytes(struct.pack("<Q", 64))


def float16(path: Path) -> None:
    write_header_only(path, tensor_shapes(TINY_ARCH), dtype="F16")


def trailing(path: Path) -> None:
    write_header_only(path, tensor_shapes(TINY_ARCH))
    with path.open("ab") as file:
        file.write(b"\x00")


def long_tensor(path: Path) -> None:
    """The norm's byte range two bytes longer than its shape, every later range shifted: the ranges
    still tile the data, so only the length tells."""
    write_header_only(path, tensor_shapes(TINY_ARCH))
    raw = path.read_bytes()
    (size,) = struct.unpack("<Q", raw[:8])
    header: dict[str, Any] = json.loads(raw[8 : 8 + size])
    begin = header["model.norm.weight"]["data_offsets"][0]
    for entry in header.values():
        offsets = entry["data_offsets"]
        offsets[1] += 2 if offsets[0] >= begin else 0
        offsets[0] += 2 if offsets[0] > begin else 0
    body = json.dumps(header).encode()
    body += b" " * (-len(body) % 8)
    path.write_bytes(struct.pack("<Q", len(body)) + body + raw[8 + size :] + b"\x00\x00")


@pytest.mark.parametrize(
    ("case", "body", "match"),
    [
        ("too short", too_short, "too short"),
        ("a huge header", huge_header, "larger than"),
        ("cut in its header", cut_header, "ends inside"),
        ("truncated data", truncated, "do not end where the file does"),
        ("overlapping tensors", overlapping, "overlap or leave a gap"),
        ("half-precision tensors", float16, "not BF16"),
        ("trailing bytes", trailing, "do not end where the file does"),
        ("a byte range that is not its shape's", long_tensor, "its shape's bytes"),
    ],
)
def test_malformed_safetensors_files_are_refused(
    tmp_path: Path, case: str, body: Callable[[Path], object], match: str
) -> None:
    with pytest.raises(ArchitectureError, match=match):
        import_checkpoint(single_file(tmp_path / "checkpoint", body))
    assert case


def test_an_index_placing_tensors_elsewhere_is_refused(tmp_path: Path) -> None:
    weights = tensors(TINY_ARCH)

    traversal = write(tmp_path / "traversal", TINY, weights, shards=2)
    index_path = traversal / "model.safetensors.index.json"
    index: dict[str, Any] = json.loads(index_path.read_text(encoding="utf-8"))
    index["weight_map"][LAYER0_Q] = "../model-00001-of-00002.safetensors"
    index_path.write_text(json.dumps(index), encoding="utf-8")
    with pytest.raises(ArchitectureError, match="not a shard here"):
        import_checkpoint(traversal)

    misplaced = write(tmp_path / "misplaced", TINY, weights, shards=2)
    index_path = misplaced / "model.safetensors.index.json"
    index = json.loads(index_path.read_text(encoding="utf-8"))
    files = sorted(set(index["weight_map"].values()))
    name = next(n for n, f in index["weight_map"].items() if f == files[0])
    index["weight_map"][name] = files[1]
    index_path.write_text(json.dumps(index), encoding="utf-8")
    with pytest.raises(ArchitectureError, match="exactly the tensors placed in it"):
        import_checkpoint(misplaced)

    for elsewhere in ["sub/model-00001-of-00002.safetensors", "/model-00001-of-00002.safetensors"]:
        placed = write(tmp_path / f"placed-{len(elsewhere)}", TINY, weights, shards=2)
        (placed / "sub").mkdir()
        shutil.copy(placed / "model-00001-of-00002.safetensors", placed / "sub")
        index_path = placed / "model.safetensors.index.json"
        index = json.loads(index_path.read_text(encoding="utf-8"))
        index["weight_map"][LAYER0_Q] = elsewhere
        index_path.write_text(json.dumps(index), encoding="utf-8")
        with pytest.raises(ArchitectureError, match="not a shard here"):
            import_checkpoint(placed)

    # A shard holding a tensor the index places nowhere.
    stowaway = write(tmp_path / "stowaway", TINY, weights, shards=2)
    shard = stowaway / "model-00001-of-00002.safetensors"
    index = json.loads((stowaway / "model.safetensors.index.json").read_text(encoding="utf-8"))
    held = {n: weights[n] for n, f in index["weight_map"].items() if f == shard.name}
    save(held | {"extra": torch.zeros(2, dtype=torch.bfloat16)}, shard)
    with pytest.raises(ArchitectureError, match="exactly the tensors placed in it"):
        import_checkpoint(stowaway)

    with pytest.raises(ArchitectureError, match="not a checkpoint directory"):
        import_checkpoint(tmp_path / "absent")
