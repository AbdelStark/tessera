# SPDX-License-Identifier: Apache-2.0 OR MIT
"""Calibration statistics from a deterministic BF16 forward pass (RFC-0005 'Calibration').

The BF16 reference is Hugging Face's Llama implementation, built from the checkpoint's
configuration and loaded from the checked safetensors files of `import_checkpoint`: never through
`from_pretrained`, which would unpickle a `.bin` file it found (06 §3, TB-5). Each calibration
sequence runs alone, with deterministic algorithms, and every quantization point of `D` records
the absolute maximum of each channel, reduced in float64 on the CPU. A maximum is exact in any
order, so the statistics depend on the activations alone.

The points of layer `l`, named `layers.{l}.<point>`, follow RFC-0005 'Integer semantics':

| Point | Channels | What |
|---|---|---|
| `attn_input`, `mlp_input` | `d` | the residual stream entering each norm |
| `attn_norm`, `mlp_norm` | `d` | the norm's output before its gain, which folds into the weights |
| `q`, `k` | heads x `d_head` | the queries and keys after RoPE, channel `head * d_head + i` |
| `v` | KV heads x `d_head` | the values |
| `attn_out` | heads x `d_head` | the attention output, the O projection's input |
| `gate`, `up`, `silu`, `h` | `d_ff` | the gate and up projections, SiLU of the gate, the product |

and `final.input` and `final.norm` around the final norm. The statistics are secret-class: they
follow from the weights (05 §3). They are written as a secret artifact, mode 0600 by atomic
rename (06 §7), and a report names them by digest and shape only.
"""

import json
import os
import platform
from collections.abc import Generator, Mapping, Sequence
from contextlib import contextmanager
from dataclasses import dataclass, field
from importlib.metadata import version
from pathlib import Path
from typing import Any, cast

import blake3
import safetensors.torch
import torch
from transformers import AttentionInterface, LlamaConfig, LlamaForCausalLM, masking_utils
from transformers.models.llama.modeling_llama import LlamaRotaryEmbedding, eager_attention_forward

from tessera_quant.arch import ArchSpec
from tessera_quant.calibration import CalibrationSet, Window
from tessera_quant.checkpoint import FloatModel
from tessera_quant.errors import CalibrationError, TesseraQuantError
from tessera_quant.tce import secret_file

STATISTICS_SCHEMA = "tessera.calibration-statistics/1"
"""The schema the statistics file names in its metadata."""

REPORT_SCHEMA = "tessera.calibration/1"
"""The calibration report's schema (03 §5), `spec/schemas/calibration-v1.schema.json`."""

STATISTICS_FILE = "statistics.safetensors"
"""The statistics file `write_calibration` writes, a secret artifact."""

REPORT_FILE = "calibration.json"
"""The report `write_calibration` writes beside it."""

_STATISTICS_CONTEXT = "tessera-quant 2026-10-17 calibration statistics v1"
_ATTENTION = "tessera_capture"
_RECORDER = "_tessera_recorder"


def points(arch: ArchSpec) -> dict[str, int]:
    """Every quantization point of `arch` and its number of channels, in forward order."""
    heads, kv = arch.n_heads * arch.d_head, arch.n_kv_heads * arch.d_head
    per_layer = {
        "attn_input": arch.d_model,
        "attn_norm": arch.d_model,
        "q": heads,
        "k": kv,
        "v": kv,
        "attn_out": heads,
        "mlp_input": arch.d_model,
        "mlp_norm": arch.d_model,
        "gate": arch.d_ff,
        "up": arch.d_ff,
        "silu": arch.d_ff,
        "h": arch.d_ff,
    }
    found = {
        f"layers.{layer}.{name}": channels
        for layer in range(arch.n_layers)
        for name, channels in per_layer.items()
    }
    return found | {"final.input": arch.d_model, "final.norm": arch.d_model}


@dataclass(frozen=True)
class Statistics:
    """The per-channel absolute maxima at every quantization point, in float64."""

    maxima: Mapping[str, torch.Tensor]

    def to_bytes(self, calibration: CalibrationSet) -> bytes:
        """The statistics file: a safetensors file of the maxima, one float64 vector per point.

        Its metadata is one entry, `tessera`, of canonical JSON: the schema and the calibration
        set's selection digest, seed and sizes. safetensors orders several metadata entries by a
        hash map's whim, so one entry is what keeps equal statistics at equal bytes.
        """
        identity = {
            "calibration_selection": calibration.selection_digest.hex(),
            "schema": STATISTICS_SCHEMA,
            "seed": calibration.seed,
            "sequences": len(calibration.calibration),
            "tokens_per_sequence": calibration.length,
        }
        metadata = {"tessera": json.dumps(identity, sort_keys=True, separators=(",", ":"))}
        # safetensors' torch helpers are only partly typed.
        return cast(bytes, cast(Any, safetensors.torch).save(dict(self.maxima), metadata=metadata))


def statistics_digest(data: bytes) -> bytes:
    """The digest that names a statistics file in `Provenance`.

    BLAKE3 of the file's bytes, under its own key-derivation context.
    """
    return blake3.blake3(data, derive_key_context=_STATISTICS_CONTEXT).digest()


class _Recorder:
    """Running per-channel absolute maxima, in float64."""

    def __init__(self) -> None:
        self.maxima: dict[str, torch.Tensor] = {}

    def observe(self, name: str, values: torch.Tensor) -> None:
        channels = values.shape[-1]
        found = values.detach().to("cpu", torch.float64).abs().reshape(-1, channels).amax(dim=0)
        held = self.maxima.get(name)
        self.maxima[name] = found if held is None else torch.maximum(held, found)

    def observe_heads(self, name: str, values: torch.Tensor) -> None:
        # (batch, heads, positions, d_head) to one row per position, channel head * d_head + i.
        batch, heads, positions, d_head = values.shape
        rows = values.transpose(1, 2).reshape(batch * positions, heads * d_head)
        self.observe(name, rows)


def _capturing_attention(
    module: torch.nn.Module,
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    attention_mask: torch.Tensor | None,
    **kwargs: Any,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Eager attention, which first records the queries and keys after RoPE and the values."""
    recorder = cast(_Recorder | None, getattr(module, _RECORDER, None))
    if recorder is not None:
        layer = cast(int, cast(Any, module).layer_idx)
        recorder.observe_heads(f"layers.{layer}.q", query)
        recorder.observe_heads(f"layers.{layer}.k", key)
        recorder.observe_heads(f"layers.{layer}.v", value)
    return eager_attention_forward(module, query, key, value, attention_mask, **kwargs)


# The attention implementation, and the mask it takes: transformers builds no mask at all for an
# implementation its mask registry does not name, which would make the attention non-causal.
cast(Any, AttentionInterface).register(_ATTENTION, _capturing_attention)
# transformers types its mask functions loosely.
_masks: Any = masking_utils
_masks.AttentionMaskInterface.register(_ATTENTION, _masks.eager_mask)


def reference_model(model: FloatModel, device: str = "cpu") -> LlamaForCausalLM:
    """Hugging Face's Llama model in BF16, with the checkpoint's configuration and weights.

    The model is built on the meta device and filled from `model`'s safetensors files, read
    through `FloatModel`: no file is opened any other way. RoPE's frequencies are not weights,
    so they are computed again once the model has a device.
    """
    directories = {path.parent for path in model.files.values()}
    (directory,) = directories
    config_values = json.loads((directory / "config.json").read_text(encoding="utf-8"))
    # transformers types `from_dict` and `to` loosely.
    config = cast(LlamaConfig, cast(Any, LlamaConfig).from_dict(config_values))
    config._attn_implementation = _ATTENTION  # pyright: ignore[reportPrivateUsage]
    with torch.device("meta"):
        meta: Any = LlamaForCausalLM(config)
    llama = cast(LlamaForCausalLM, meta.to(dtype=torch.bfloat16).to_empty(device=device))
    state = {name: model.tensor(name) for name in model.names}
    llama.load_state_dict(state, strict=True)
    llama.model.rotary_emb = LlamaRotaryEmbedding(config).to(device)
    llama.eval()
    return llama


def reference_logits(
    model: FloatModel, windows: Sequence[Window], seed: int, device: str = "cpu"
) -> torch.Tensor:
    """The BF16 reference's logits on each window, run alone, in float64: `[positions, vocab]`.

    The windows' positions follow one another, window by window. They run under deterministic
    algorithms and `seed`, as the capture does.
    """
    llama = reference_model(model, device)
    rows: list[torch.Tensor] = []
    with _deterministic(seed), torch.no_grad():
        for window in windows:
            tokens = torch.tensor([window.tokens], dtype=torch.long, device=device)
            output: Any = llama(input_ids=tokens)
            rows.append(cast(torch.Tensor, output.logits)[0].to("cpu", torch.float64))
    return torch.cat(rows)


@contextmanager
def _deterministic(seed: int) -> Generator[None]:
    """Deterministic algorithms and seeded generators, as RFC-0005 'Determinism' requires."""
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    previous = torch.are_deterministic_algorithms_enabled()
    torch.use_deterministic_algorithms(True)
    cast(Any, torch).manual_seed(seed)
    try:
        yield
    finally:
        torch.use_deterministic_algorithms(previous)


def _normalized(module: torch.nn.Module, x: torch.Tensor) -> torch.Tensor:
    """An RMSNorm's output before its gain, computed as the reference computes it."""
    epsilon = cast(float, cast(Any, module).variance_epsilon)
    h = x.to(torch.float32)
    return (h * torch.rsqrt(h.pow(2).mean(-1, keepdim=True) + epsilon)).to(x.dtype)


def _norm_hook(recorder: _Recorder, input_name: str, norm_name: str) -> Any:
    def hook(module: torch.nn.Module, args: tuple[torch.Tensor, ...], output: torch.Tensor) -> None:
        x = args[0]
        normalized = _normalized(module, x)
        gain = cast(torch.Tensor, cast(Any, module).weight)
        recomputed = gain * normalized
        # NaN, which a non-finite weight spreads, is equal to NaN here: the finiteness check
        # reports it once the pass ends.
        same = (recomputed == output) | (recomputed.isnan() & output.isnan())
        if not bool(same.all()):
            raise TesseraQuantError(
                f"{norm_name}: the reference's RMSNorm no longer computes the gain times the "
                "normalized input that the capture records"
            )
        recorder.observe(input_name, x)
        recorder.observe(norm_name, normalized)

    return hook


def _input_hook(recorder: _Recorder, name: str) -> Any:
    def hook(_module: torch.nn.Module, args: tuple[torch.Tensor, ...]) -> None:
        recorder.observe(name, args[0])

    return hook


def _output_hook(recorder: _Recorder, name: str) -> Any:
    def hook(
        _module: torch.nn.Module, _args: tuple[torch.Tensor, ...], output: torch.Tensor
    ) -> None:
        recorder.observe(name, output)

    return hook


def capture(model: FloatModel, calibration: CalibrationSet, device: str = "cpu") -> Statistics:
    """The calibration statistics of `model` over the calibration set's sequences.

    Each sequence runs alone through the BF16 reference, under deterministic algorithms and the
    calibration set's seed, so the same checkpoint, sequences, hardware and library versions give
    the same statistics.

    Raises:
        CalibrationError: when a maximum is not finite, or every channel of a point is zero, which
            leaves its scale undefined (`ERR-QUANT-CALIB`).
        TesseraQuantError: when the reference's RMSNorm departs from what the capture recomputes.
    """
    recorder = _Recorder()
    llama = reference_model(model, device)
    handles: list[Any] = []
    for layer, block in enumerate(cast(Any, llama.model.layers)):
        prefix = f"layers.{layer}"
        setattr(block.self_attn, _RECORDER, recorder)
        handles += [
            block.input_layernorm.register_forward_hook(
                _norm_hook(recorder, f"{prefix}.attn_input", f"{prefix}.attn_norm")
            ),
            block.post_attention_layernorm.register_forward_hook(
                _norm_hook(recorder, f"{prefix}.mlp_input", f"{prefix}.mlp_norm")
            ),
            block.self_attn.o_proj.register_forward_pre_hook(
                _input_hook(recorder, f"{prefix}.attn_out")
            ),
            block.mlp.gate_proj.register_forward_hook(_output_hook(recorder, f"{prefix}.gate")),
            block.mlp.up_proj.register_forward_hook(_output_hook(recorder, f"{prefix}.up")),
            block.mlp.act_fn.register_forward_hook(_output_hook(recorder, f"{prefix}.silu")),
            block.mlp.down_proj.register_forward_pre_hook(_input_hook(recorder, f"{prefix}.h")),
        ]
    handles.append(
        cast(Any, llama.model.norm).register_forward_hook(
            _norm_hook(recorder, "final.input", "final.norm")
        )
    )
    try:
        with _deterministic(calibration.seed), torch.no_grad():
            for window in calibration.calibration:
                tokens = torch.tensor([window.tokens], dtype=torch.long, device=device)
                llama(input_ids=tokens)
    finally:
        for handle in handles:
            handle.remove()
    expected = points(model.arch)
    if sorted(recorder.maxima) != sorted(expected):
        raise TesseraQuantError("the capture did not record exactly the quantization points")
    maxima = {name: recorder.maxima[name] for name in expected}
    for name, values in maxima.items():
        if not bool(torch.isfinite(values).all()):
            raise CalibrationError(f"{name}: a channel's maximum is not finite")
        if not bool((values > 0).any()):
            raise CalibrationError(f"{name}: every channel is zero, so the scale is undefined")
    return Statistics(maxima)


def write_secret(path: Path, data: bytes) -> None:
    """Writes a secret artifact: mode 0600, through a temporary file renamed into place (06 §7).

    The file appears whole or not at all, and is never readable by others, even while written.
    """
    with secret_file(path) as file:
        file.write(data)


def calibration_report(
    model: FloatModel, calibration: CalibrationSet, statistics: bytes, device: str
) -> dict[str, Any]:
    """The `tessera.calibration/1` report of a calibration: public data only.

    It names the model's architecture, the data and its selection, the statistics by digest and
    shape, and the environment that reproduces them. No maximum appears: the statistics follow
    from the weights, so they stay in their secret file (05 §3).
    """
    arch = model.arch
    source = calibration.source
    libraries = ("torch", "transformers", "safetensors", "numpy", "blake3")
    return {
        "schema": REPORT_SCHEMA,
        "tessera_quant": version("tessera-quant"),
        "architecture": {
            "family": arch.family.name,
            "layers": arch.n_layers,
            "d_model": arch.d_model,
            "d_ff": arch.d_ff,
            "heads": arch.n_heads,
            "kv_heads": arch.n_kv_heads,
            "d_head": arch.d_head,
            "vocab": arch.vocab,
        },
        "dataset": {
            "name": source.name,
            "config": source.config,
            "split": source.split,
            "revision": source.revision,
            "files": list(source.files),
        },
        "seed": calibration.seed,
        "sequences": len(calibration.calibration),
        "held_out": len(calibration.held_out),
        "tokens_per_sequence": calibration.length,
        "selection_digest": calibration.selection_digest.hex(),
        "statistics_digest": statistics_digest(statistics).hex(),
        "points": [{"name": name, "channels": channels} for name, channels in points(arch).items()],
        "environment": {
            "device": device,
            "batch_size": 1,
            "deterministic_algorithms": True,
            "python": platform.python_version(),
            "libraries": {name: version(name) for name in libraries},
        },
    }


@dataclass(frozen=True, eq=False)
class CalibrationResult:
    """A calibration: its sequences and the statistics captured on them (02 §4): secret-class."""

    calibration: CalibrationSet
    statistics: Statistics = field(repr=False)
    data: bytes = field(repr=False)
    """The statistics file's bytes, which `digest` names."""

    @property
    def digest(self) -> bytes:
        """The statistics digest that `Provenance` records (RFC-0005 'Calibration')."""
        return statistics_digest(self.data)


def calibrate(
    model: FloatModel, calibration: CalibrationSet, device: str = "cpu"
) -> CalibrationResult:
    """Captures the statistics of `model` on the calibration set (02 §4, RFC-0005 'Pipeline', 3).

    Raises:
        CalibrationError: as `capture`.
        TesseraQuantError: as `capture`.
    """
    statistics = capture(model, calibration, device)
    return CalibrationResult(calibration, statistics, statistics.to_bytes(calibration))


def write_calibration(
    model: FloatModel, result: CalibrationResult, directory: Path, device: str = "cpu"
) -> dict[str, Any]:
    """Writes a calibration's statistics, with their report, into `directory`.

    The statistics go to `statistics.safetensors` as a secret artifact, and the report to
    `calibration.json`. Returns the report. `device` is the one the statistics were captured on,
    which the report records.
    """
    report = calibration_report(model, result.calibration, result.data, device)
    write_secret(directory / STATISTICS_FILE, result.data)
    text = json.dumps(report, indent=2, ensure_ascii=False) + "\n"
    (directory / REPORT_FILE).write_text(text, encoding="utf-8")
    return report
