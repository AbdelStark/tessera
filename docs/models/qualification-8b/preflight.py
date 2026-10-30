# SPDX-License-Identifier: Apache-2.0 OR MIT
"""Portable replay of the full existing-policy preflight measured for #633."""

import argparse
import gc
import hashlib
import json
import os
import resource
import subprocess
import time
from pathlib import Path

import safetensors.torch
import torch
from tessera_quant import assemble, import_checkpoint
from tessera_quant._streaming import transformed_checkpoint
from tessera_quant.capture import statistics_digest
from tessera_quant.errors import CalibrationError, QuantizationInfeasibleError
from tessera_quant.integer_model import Profile
from tessera_quant.migration import ALPHAS
from tessera_quant.scales import choose

os.umask(0o077)
parser = argparse.ArgumentParser(description=__doc__)
for name in ("checkpoint", "calibration", "output"):
    parser.add_argument("--" + name, type=Path, required=True)
args = parser.parse_args()
ROOT = args.output
ROOT.mkdir(mode=0o700, exist_ok=False)
CAL = args.calibration
START = time.monotonic()
completed = 0
results = []
SOURCE = Path(assemble.__file__).parent
hashes = {
    p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in SOURCE.glob("*.py")
}
launch = {
    "git_head": subprocess.check_output(
        ["git", "rev-parse", "HEAD"], text=True
    ).strip(),
    "source_sha256": hashes,
    "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    "scope": "actual 8B existing scale-policy preflight; no held-out scoring, selected alpha, fidelity or proof",
}
(ROOT / "launch-source.json").write_text(json.dumps(launch, indent=2) + "\n")


def status(stage, **extra):
    value = {
        "pid": os.getpid(),
        "stage": stage,
        "completed_candidates": completed,
        "requested_candidates": 16,
        "elapsed_seconds": time.monotonic() - START,
        "process_peak_rss_bytes": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        **extra,
    }
    temp = ROOT / "status.tmp"
    temp.write_text(json.dumps(value, indent=2) + "\n")
    temp.replace(ROOT / "status.json")
    print(json.dumps(value), flush=True)


try:
    torch.set_num_threads(2)
    torch.use_deterministic_algorithms(True)
    status("checking-inputs")
    report = json.loads((CAL / "calibration.json").read_text())
    verification = json.loads((CAL / "verification.json").read_text())
    data = (CAL / "statistics.safetensors").read_bytes()
    assert (
        statistics_digest(data).hex()
        == report["statistics_digest"]
        == verification["statistics_digest"]
    )
    assert (
        report["seed"],
        report["sequences"],
        report["held_out"],
        report["tokens_per_sequence"],
    ) == (0, 512, 64, 2048)
    maxima = safetensors.torch.load(data)
    model = import_checkpoint(args.checkpoint)
    assert (model.arch.n_layers, model.arch.d_model) == (32, 4096)
    assemble._check_statistics(model.arch, maxima)
    for profile in [Profile.Servable, Profile.Minimal]:
        for mode in ["pot", "affine"]:
            attempts = []
            affine = (
                assemble.affine_weights(model.arch, profile) if mode == "affine" else ()
            )

            def build(
                alpha, profile=profile, mode=mode, affine=affine, attempts=attempts
            ):
                global completed
                started = time.monotonic()
                status(
                    "preflighting", profile=profile.name, scales=mode, alpha=str(alpha)
                )
                weights, stats = transformed_checkpoint(model, maxima, alpha)
                try:
                    q = choose(model.arch, profile, weights, stats, affine=affine)
                except QuantizationInfeasibleError:
                    attempts.append(
                        {
                            "alpha": str(alpha),
                            "state": "infeasible",
                            "code": "ERR-QUANT-CALIB",
                            "error_type": "QuantizationInfeasibleError",
                            "seconds": time.monotonic() - started,
                        }
                    )
                    completed += 1
                    status(
                        "candidate-refused",
                        profile=profile.name,
                        scales=mode,
                        alpha=str(alpha),
                    )
                    raise
                entries = sum(w.data.size for w in q.weights)
                attempts.append(
                    {
                        "alpha": str(alpha),
                        "state": "admissible",
                        "weight_entries": entries,
                        "weight_tensors": len(q.weights),
                        "seconds": time.monotonic() - started,
                    }
                )
                completed += 1
                status(
                    "candidate-admitted",
                    profile=profile.name,
                    scales=mode,
                    alpha=str(alpha),
                )
                return q

            try:
                admissible = assemble._admissible_alphas(build, ALPHAS)
            except CalibrationError:
                if len(attempts) != 4 or any(
                    v["state"] != "infeasible" for v in attempts
                ):
                    raise
                admissible = ()
            assert len(attempts) == 4
            results.append(
                {
                    "profile": profile.name,
                    "scales": mode,
                    "admissible": list(map(str, admissible)),
                    "attempts": attempts,
                }
            )
            (ROOT / "preflight-results.json").write_text(
                json.dumps({"scope": launch["scope"], "results": results}, indent=2)
                + "\n"
            )
            gc.collect()
    assert completed == 16
    assert hashes == {
        p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in SOURCE.glob("*.py")
    }
    receipt = {
        "scope": launch["scope"],
        "model_source": "NousResearch/Meta-Llama-3.1-8B",
        "model_revision": "1f47e50cdbe801ad8a5174156ec3a0655108fb9f",
        "calibration_statistics_digest": report["statistics_digest"],
        "calibration_report_sha256": hashlib.sha256(
            (CAL / "calibration.json").read_bytes()
        ).hexdigest(),
        "source_sha256": hashes,
        "results": results,
        "seconds": time.monotonic() - START,
        "process_peak_rss_bytes": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
    }
    (ROOT / "preflight-results.json").write_text(json.dumps(receipt, indent=2) + "\n")
    status("complete")
# Fatal failures are redacted; never dump tensors through an exception message.
except BaseException as e:  # noqa: BLE001
    status("failed", error_type=type(e).__name__)
    raise SystemExit(1) from None
