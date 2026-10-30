# SPDX-License-Identifier: Apache-2.0 OR MIT
"""Controlled real-8B lifetime probe; stops before integer inference, no alpha selection."""

import argparse
import gc
import hashlib
import json
import os
import resource
import time
import weakref
from contextlib import closing
from fractions import Fraction
from pathlib import Path

import safetensors.torch
import torch
from tessera_quant import assemble, capture, import_checkpoint
from tessera_quant.calibration import (
    C4_DATASET,
    C4_REVISION,
    C4_VALIDATION_FILES,
    DatasetIdentity,
    draw,
    read_json_lines,
)
from tessera_quant.capture import CalibrationResult, Statistics, statistics_digest
from tessera_quant.checkpoint import FloatModel, read_tokenizer

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--label", choices=["baseline", "candidate"], required=True)
parser.add_argument("--output", type=Path, required=True)

for name in ("checkpoint", "calibration", "c4"):
    parser.add_argument("--" + name, type=Path, required=True)
parser.add_argument(
    "--expected-receipt",
    type=Path,
    default=Path(__file__).parent.parent / "qualification-8b/export-receipt.json",
)
args = parser.parse_args()
os.umask(0o077)
args.output.mkdir(mode=0o700, exist_ok=False)
start = time.monotonic()
models = []
maps = []
events = []
in_loading = False
preflight = False
reference_windows = 0
artifact = None


class Overlap(Exception):
    pass


class Boundary(Exception):
    pass


source = Path(assemble.__file__).parent
source_hash = {
    p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in source.glob("*.py")
}
script_sha = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()


def status(stage, **extra):
    value = {
        "pid": os.getpid(),
        "stage": stage,
        "seconds": time.monotonic() - start,
        "process_peak_rss_bytes": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        "mps_allocated_bytes": torch.mps.current_allocated_memory(),
        "mps_driver_bytes": torch.mps.driver_allocated_memory(),
        **extra,
    }
    events.append(value)
    temp = args.output / "status.tmp"
    temp.write_text(json.dumps(value, indent=2) + "\n")
    temp.replace(args.output / "status.json")
    print(json.dumps(value), flush=True)


try:
    torch.set_num_threads(2)
    torch.use_deterministic_algorithms(True)
    torch.mps.set_per_process_memory_fraction(0.5)
    status("checking-inputs")
    cal = args.calibration
    report = json.loads((cal / "calibration.json").read_text())
    data = (cal / "statistics.safetensors").read_bytes()
    assert statistics_digest(data).hex() == report["statistics_digest"]
    maxima = safetensors.torch.load(data)
    model_dir = args.checkpoint
    model = import_checkpoint(model_dir)
    expected = json.loads(args.expected_receipt.read_text())
    assert model.arch.n_layers == 32 and model.arch.d_model == 4096
    status("drawing-original-windows")
    documents = []
    for name in C4_VALIDATION_FILES:
        documents.extend(read_json_lines(args.c4 / name))
    tok = read_tokenizer(model_dir, model.arch)
    calib = draw(
        documents,
        tok.encode,
        tok.bos,
        source=DatasetIdentity(
            C4_DATASET, "en", "validation", C4_REVISION, C4_VALIDATION_FILES
        ),
        seed=0,
        count=512,
        length=2048,
        held_out=64,
    )
    assert calib.selection_digest.hex() == report["selection_digest"]
    del documents
    gc.collect()
    original_transform = assemble.transformed_checkpoint
    original_model = capture.reference_model
    original_read = FloatModel.tensor
    original_refs = assemble.iter_reference_logits
    original_preflight = assemble._admissible_alphas

    def transformed(*a, **kw):
        gc.collect()
        live = sum(r() is not None for r in models)
        status("constructing-candidate", preflight=preflight, bf16_models_live=live)
        if live:
            raise Overlap()
        result = original_transform(*a, **kw)
        maps.extend(weakref.ref(p) for p in result)
        return result

    def load_reference(*a, **kw):
        global in_loading
        gc.collect()
        assert not any(r() is not None for r in maps)
        status("loading-bf16", migration_maps_live=0)
        in_loading = True
        try:
            llama = original_model(*a, **kw)
        finally:
            in_loading = False
        models.append(weakref.ref(llama))
        status("bf16-loaded")
        return llama

    def read(self, name):
        if not in_loading and any(r() is not None for r in models):
            raise Overlap()
        return original_read(self, name)

    def references(*a, **kw):
        global reference_windows
        with closing(original_refs(*a, **kw)) as refs:
            for row in refs:
                assert tuple(row.shape) == (2048, 128256) and bool(row.isfinite().all())
                reference_windows += 1
                status("bf16-window-returned", reference_windows=reference_windows)
                yield row
                del row

    def preflighted(build, alphas):
        global preflight
        preflight = True
        try:
            return original_preflight(build, alphas)
        finally:
            preflight = False

    def stop(d, tokens):
        global artifact
        assert (
            args.label == "candidate" and reference_windows == 1 and len(tokens) == 2048
        )
        status("checking-fixed-D")
        digest = assemble.export(d, args.output / "candidate.d")
        assert digest.hex() == expected["artifact_envelope_digest"]
        with (args.output / "candidate.d").open("rb") as f:
            sha = hashlib.file_digest(f, "sha256").hexdigest()
        assert sha == expected["artifact_sha256"]
        artifact = {
            "bytes": (args.output / "candidate.d").stat().st_size,
            "envelope_digest": digest.hex(),
            "sha256": sha,
            "matches_prechange_fixed_D": True,
        }
        raise Boundary()

    assemble.transformed_checkpoint = transformed
    capture.reference_model = load_reference
    FloatModel.tensor = read
    assemble.iter_reference_logits = references
    assemble._admissible_alphas = preflighted
    assemble.run_reference = stop
    try:
        assemble.quantize(
            model,
            CalibrationResult(calib, Statistics(maxima), data),
            profile="servable",
            scales="affine",
            alphas=(Fraction(1, 2),),
            device="mps",
        )
    except Overlap:
        assert args.label == "baseline"
        outcome = "baseline-overlap-guard-fired"
    except Boundary:
        assert args.label == "candidate"
        outcome = "candidate-reached-integer-boundary"
    else:
        raise AssertionError("probe must stop before integer inference")
    gc.collect()
    assert not any(r() is not None for r in models + maps)
    assert source_hash == {
        p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in source.glob("*.py")
    }
    status("complete", outcome=outcome)
    receipt = {
        "scope": "controlled full-statistics 8B lifetime probe; no integer inference, full alpha selection, fidelity or proof",
        "label": args.label,
        "outcome": outcome,
        "reference_windows": reference_windows,
        "full_requested_preflight_strengths": ["1/2"],
        "full_prior_16_case_preflight": "PR634",
        "statistics_digest": report["statistics_digest"],
        "selection_digest": calib.selection_digest.hex(),
        "source_sha256": source_hash,
        "script_sha256": script_sha,
        "artifact": artifact,
        "events": events,
        "all_bf16_models_and_migration_maps_released": True,
        "memory_scope": "Darwin peak process RSS and event samples of per-process MPS allocated/driver bytes; not a whole-run device peak or before/after speedup",
        "mps_memory_fraction": 0.5,
        "seconds": time.monotonic() - start,
    }
    (args.output / "receipt.json").write_text(json.dumps(receipt, indent=2) + "\n")
# Every unexpected failure is fatal; redact messages that could carry secret values.
except BaseException as error:  # noqa: BLE001
    status("failed", error_type=type(error).__name__)
    raise SystemExit(1) from None
