# SPDX-License-Identifier: Apache-2.0 OR MIT
"""Full Servable/affine 8B alpha selection using the unchanged production scorer.

Uses the original 512+64 windows. This does not run the separate fidelity gates or prove D.
Run with PYTHONPATH pointing at a frozen tessera_quant package, not a changing checkout.
"""

import argparse
import gc
import hashlib
import importlib.util
import json
import math
import os
import resource
import shutil
import subprocess
import time
from fractions import Fraction
from pathlib import Path

import numpy as np
import safetensors.torch
import torch
from tessera_quant import assemble, import_checkpoint
from tessera_quant.calibration import (
    C4_DATASET,
    C4_REVISION,
    C4_VALIDATION_FILES,
    DatasetIdentity,
    draw,
    read_json_lines,
)
from tessera_quant.capture import CalibrationResult, Statistics, statistics_digest
from tessera_quant.checkpoint import read_tokenizer
from tessera_quant.errors import QuantizationInfeasibleError
from tessera_quant.integer_model import INTEGER_MODEL
from tessera_quant.migration import ALPHAS
from tessera_quant.tce import INTEGER_MODEL_LIMITS, read_artifact

parser = argparse.ArgumentParser(description=__doc__)
for name in ("checkpoint", "calibration", "c4", "output"):
    parser.add_argument("--" + name, type=Path, required=True)
parser.add_argument(
    "--helper",
    type=Path,
    default=Path(__file__).parent.parent / "qualification-8b/export_fixed.py",
)
args = parser.parse_args()
os.umask(0o077)
args.output.mkdir(mode=0o700, exist_ok=False)
started = time.monotonic()
alpha = None
windows = 0
scores = []
attempts = []
source = Path(assemble.__file__).parent
hashes = {
    p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in source.glob("*.py")
}
versions = assemble.libraries()
helper = args.output / "source-helper.py"
shutil.copyfile(args.helper, helper)
launch = {
    "source_git_head": subprocess.check_output(
        ["git", "rev-parse", "HEAD"], text=True
    ).strip(),
    "source_sha256": hashes,
    "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    "libraries": versions,
    "comparison_helper_sha256": hashlib.sha256(helper.read_bytes()).hexdigest(),
    "scope": "full admitted-strength Servable/affine selection; not both-profile/mode fidelity or proof",
}
(args.output / "launch-source.json").write_text(json.dumps(launch, indent=2) + "\n")


def status(stage, **extra):
    value = {
        "pid": os.getpid(),
        "stage": stage,
        "alpha": alpha,
        "integer_windows_returned": windows,
        "windows_per_alpha": 64,
        "tokens_per_window": 2048,
        "seconds": time.monotonic() - started,
        "process_peak_rss_bytes": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        "mps_allocated_bytes": torch.mps.current_allocated_memory(),
        "mps_driver_bytes": torch.mps.driver_allocated_memory(),
        **extra,
    }
    temp = args.output / "status.tmp"
    temp.write_text(json.dumps(value, indent=2) + "\n")
    temp.replace(args.output / "status.json")
    print(json.dumps(value), flush=True)


try:
    spec = importlib.util.spec_from_file_location("fixed_helper", helper)
    assert spec is not None and spec.loader is not None
    comparison = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(comparison)
    torch.set_num_threads(2)
    torch.use_deterministic_algorithms(True)
    torch.mps.set_per_process_memory_fraction(0.5)
    status("checking-inputs")
    report = json.loads((args.calibration / "calibration.json").read_text())
    data = (args.calibration / "statistics.safetensors").read_bytes()
    assert statistics_digest(data).hex() == report["statistics_digest"]
    assert (
        report["seed"],
        report["sequences"],
        report["held_out"],
        report["tokens_per_sequence"],
    ) == (0, 512, 64, 2048)
    maxima = safetensors.torch.load(data)
    model = import_checkpoint(args.checkpoint)
    assert (model.arch.n_layers, model.arch.d_model) == (32, 4096)
    status("drawing-original-windows")
    documents = []
    for name in C4_VALIDATION_FILES:
        documents.extend(read_json_lines(args.c4 / name))
    tokenizer = read_tokenizer(args.checkpoint, model.arch)
    calibration = draw(
        documents,
        tokenizer.encode,
        tokenizer.bos,
        source=DatasetIdentity(
            C4_DATASET, "en", "validation", C4_REVISION, C4_VALIDATION_FILES
        ),
        seed=0,
        count=512,
        length=2048,
        held_out=64,
    )
    assert calibration.selection_digest.hex() == report["selection_digest"]
    del documents
    gc.collect()
    original_preflight = assemble._admissible_alphas
    original_transform = assemble.transformed_checkpoint
    original_select = assemble.select_alpha
    original_integer = assemble.run_reference

    def preflight(build, candidates):
        def checked(a):
            global alpha
            alpha = str(a)
            status("preflight-candidate")
            try:
                found = build(a)
            except QuantizationInfeasibleError:
                attempts.append(
                    {"alpha": str(a), "state": "infeasible", "code": "ERR-QUANT-CALIB"}
                )
                status("candidate-refused")
                raise
            attempts.append({"alpha": str(a), "state": "admissible"})
            status("candidate-admitted")
            return found

        return original_preflight(checked, candidates)

    def transform(*a, **kw):
        # Operator resource policy: discard only inactive Metal cache pages between builds.
        # This changes no tensor, candidate, score, limit or rejection rule.
        gc.collect()
        assert torch.mps.current_allocated_memory() == 0
        torch.mps.empty_cache()
        status("candidate-construction")
        return original_transform(*a, **kw)

    def select(divergence, candidates):
        def measured(a):
            global alpha, windows
            alpha, windows = str(a), 0
            status("scoring-alpha")
            value = divergence(a)
            assert windows == 64 and math.isfinite(value)
            scores.append(
                {
                    "alpha": str(a),
                    "mean_per_position_kl_nats": value,
                    "windows": windows,
                }
            )
            (args.output / "score-progress.json").write_text(
                json.dumps(scores, indent=2) + "\n"
            )
            status("alpha-scored")
            return value

        selected = original_select(measured, candidates)
        global alpha
        alpha = str(selected)
        status("winner-reconstruction")
        return selected

    def integer(d, tokens):
        global windows
        assert len(tokens) == 2048
        status("integer-reference")
        found = original_integer(d, tokens)
        assert found.shape == (2048, 128256) and found.dtype == np.int64
        windows += 1
        status("integer-window-returned")
        return found

    assemble._admissible_alphas = preflight
    assemble.transformed_checkpoint = transform
    assemble.select_alpha = select
    assemble.run_reference = integer
    status("full-production-selection")
    d = assemble.quantize(
        model,
        CalibrationResult(calibration, Statistics(maxima), data),
        profile="servable",
        scales="affine",
        alphas=ALPHAS,
        device="mps",
    )
    assert len(attempts) == 4 and len(scores) == sum(
        a["state"] == "admissible" for a in attempts
    )
    best = min(
        scores, key=lambda s: (s["mean_per_position_kl_nats"], Fraction(s["alpha"]))
    )
    assert str(d.provenance.alpha) == best["alpha"]
    assert assemble.checkpoint_digests(model) == d.provenance.checkpoint
    assert (
        statistics_digest(
            (args.calibration / "statistics.safetensors").read_bytes()
        ).hex()
        == report["statistics_digest"]
    )
    status("exporting-selected-D")
    digest = assemble.export(d, args.output / "selected.d")
    assert (args.output / "selected.d").stat().st_mode & 0o777 == 0o600
    status("checking-full-readback")
    decoded = read_artifact(
        args.output / "selected.d", INTEGER_MODEL, limits=INTEGER_MODEL_LIMITS
    )
    assert comparison.same(d.arch, decoded.arch) and d.profile == decoded.profile
    assert comparison.same(d.quant, decoded.quant) and comparison.same(
        d.provenance, decoded.provenance
    )
    assert len(d.weights) == len(decoded.weights)
    for expected, found in zip(d.weights, decoded.weights, strict=True):
        assert (expected.ty, expected.layer, expected.rows, expected.cols) == (
            found.ty,
            found.layer,
            found.rows,
            found.cols,
        )
        assert expected.data.size == found.data.size == found.rows * found.cols
        for offset in range(0, found.data.size, 8 * 1024 * 1024):
            assert np.array_equal(
                expected.data[offset : offset + 8 * 1024 * 1024],
                found.data[offset : offset + 8 * 1024 * 1024],
            )
    assert hashes == {
        p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in source.glob("*.py")
    }
    assert versions == assemble.libraries()
    with (args.output / "selected.d").open("rb") as stream:
        artifact_sha = hashlib.file_digest(stream, "sha256").hexdigest()
    result = {
        "scope": launch["scope"],
        "launch": launch,
        "attempts": attempts,
        "scores": scores,
        "selected_alpha": str(d.provenance.alpha),
        "statistics_digest": report["statistics_digest"],
        "selection_digest": calibration.selection_digest.hex(),
        "artifact_bytes": (args.output / "selected.d").stat().st_size,
        "artifact_envelope_digest": digest.hex(),
        "artifact_sha256": artifact_sha,
        "complete_weight_and_metadata_readback": True,
        "fidelity_qualified": False,
        "proof_generated": False,
        "seconds": time.monotonic() - started,
    }
    (args.output / "selection-result.json").write_text(
        json.dumps(result, indent=2) + "\n"
    )
    status("complete")
# No failure may omit a scored candidate, reduce the windows, or expose secret values.
except BaseException as error:  # noqa: BLE001
    status("failed", error_type=type(error).__name__)
    raise SystemExit(1) from None
