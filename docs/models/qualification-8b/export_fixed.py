# SPDX-License-Identifier: Apache-2.0 OR MIT
"""Export and check the fixed diagnostic candidate for #633, not an alpha winner.

Run in the pinned tessera-quant environment; see ../llama31-8b-calibration.md.
Only public metadata is written to the receipt. Model and statistics stay private.
"""

import argparse
import dataclasses
import gc
import hashlib
import json
import os
import resource
import subprocess
import time
from fractions import Fraction
from pathlib import Path

import numpy as np
import safetensors.torch
import torch
from tessera_quant import assemble, import_checkpoint
from tessera_quant._streaming import transformed_checkpoint
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
from tessera_quant.integer_model import INTEGER_MODEL, IntegerModel, Profile
from tessera_quant.scales import choose
from tessera_quant.tce import INTEGER_MODEL_LIMITS, read_artifact


def require(condition, message):
    if not condition:
        raise ValueError(message)


def same(left, right):
    """Compare complete decoded metadata without rendering private channel arrays."""
    if type(left) is not type(right):
        return False
    if isinstance(left, np.ndarray):
        return left.dtype == right.dtype and np.array_equal(left, right)
    if dataclasses.is_dataclass(left):
        return all(
            same(getattr(left, f.name), getattr(right, f.name))
            for f in dataclasses.fields(left)
        )
    if isinstance(left, (list, tuple)):
        return len(left) == len(right) and all(
            same(a, b) for a, b in zip(left, right, strict=True)
        )
    if isinstance(left, dict):
        return left.keys() == right.keys() and all(
            same(left[k], right[k]) for k in left
        )
    return left == right


def sha256_file(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def run(args):
    os.umask(0o077)
    args.output.mkdir(mode=0o700, exist_ok=False)
    started = time.monotonic()
    source = Path(assemble.__file__).parent
    hashes = {p.name: sha256_file(p) for p in source.glob("*.py")}
    launch = {
        "git_head": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True
        ).strip(),
        "source_sha256": hashes,
        "script_sha256": sha256_file(Path(__file__)),
    }
    (args.output / "launch-source.json").write_text(json.dumps(launch, indent=2) + "\n")

    def status(stage, **extra):
        value = {
            "pid": os.getpid(),
            "stage": stage,
            "seconds": time.monotonic() - started,
            "process_peak_rss_bytes": resource.getrusage(
                resource.RUSAGE_SELF
            ).ru_maxrss,
            **extra,
        }
        temp = args.output / "status.tmp"
        temp.write_text(json.dumps(value, indent=2) + "\n")
        temp.replace(args.output / "status.json")
        print(json.dumps(value), flush=True)

    try:
        torch.set_num_threads(2)
        torch.use_deterministic_algorithms(True)
        status("checking-inputs")
        preflight = json.loads(args.preflight.read_text())
        require(preflight["source_sha256"] == hashes, "preflight source differs")
        results = preflight["results"]
        require(
            len(results) == 4 and sum(len(r["attempts"]) for r in results) == 16,
            "complete preflight required",
        )
        require(
            {(r["profile"], r["scales"]) for r in results}
            == {(p, m) for p in ("Servable", "Minimal") for m in ("pot", "affine")},
            "preflight profile/mode coverage differs",
        )
        for result in results:
            require(
                [a["alpha"] for a in result["attempts"]]
                == ["1/2", "3/5", "7/10", "4/5"],
                "preflight strength coverage differs",
            )
            require(
                all(
                    a["state"] in ("admissible", "infeasible")
                    for a in result["attempts"]
                )
                and result["admissible"]
                == [
                    a["alpha"] for a in result["attempts"] if a["state"] == "admissible"
                ],
                "preflight outcomes differ",
            )
        require(
            any(
                r["profile"] == "Servable"
                and r["scales"] == "affine"
                and "1/2" in r["admissible"]
                for r in results
            ),
            "fixed candidate not admitted",
        )
        report_path = args.calibration / "calibration.json"
        report_sha = sha256_file(report_path)
        require(
            report_sha == preflight["calibration_report_sha256"],
            "calibration report differs",
        )
        report = json.loads(report_path.read_text())
        data = (args.calibration / "statistics.safetensors").read_bytes()
        stats_digest = statistics_digest(data).hex()
        require(
            stats_digest
            == report["statistics_digest"]
            == preflight["calibration_statistics_digest"],
            "statistics differ",
        )
        maxima = safetensors.torch.load(data)
        model = import_checkpoint(args.checkpoint)
        require(
            (model.arch.n_layers, model.arch.d_model, model.arch.d_ff, model.arch.vocab)
            == (32, 4096, 14336, 128256),
            "unexpected checkpoint architecture",
        )
        assemble._check_statistics(model.arch, maxima)
        pins = Path(__file__).parent.parent / "checkpoints.sha256"
        expected_sha256 = {
            Path(name).name: digest
            for line in pins.read_text().splitlines()
            if line and not line.startswith("#")
            for digest, name in [line.split()]
            if name.startswith("llama-3.1-8b/")
        }
        paths = {path.name: path for path in model.files.values()}
        require(paths.keys() == expected_sha256.keys(), "checkpoint shard set differs")
        require(
            all(
                sha256_file(paths[name]) == digest
                for name, digest in expected_sha256.items()
            ),
            "checkpoint LFS checksum differs",
        )
        checkpoint = assemble.checkpoint_digests(model)
        status("drawing-original-calibration")
        documents = []
        for filename in C4_VALIDATION_FILES:
            documents.extend(read_json_lines(args.c4 / filename))
        tokenizer = read_tokenizer(args.checkpoint, model.arch)
        source_identity = DatasetIdentity(
            C4_DATASET, "en", "validation", C4_REVISION, C4_VALIDATION_FILES
        )
        calibration = draw(
            documents,
            tokenizer.encode,
            tokenizer.bos,
            source=source_identity,
            seed=0,
            count=512,
            length=2048,
            held_out=64,
        )
        require(
            calibration.selection_digest.hex() == report["selection_digest"],
            "original selection differs",
        )
        del documents
        gc.collect()
        alpha = Fraction(1, 2)
        provenance = assemble._provenance(
            checkpoint, CalibrationResult(calibration, Statistics(maxima), data), alpha
        )
        status("constructing-fixed-candidate")
        weights, stats = transformed_checkpoint(model, maxima, alpha)
        q = choose(
            model.arch,
            Profile.Servable,
            weights,
            stats,
            affine=assemble.affine_weights(model.arch, Profile.Servable),
        )
        candidate = IntegerModel(
            model.arch, Profile.Servable, q.quant, q.weights, provenance
        )
        del weights, stats, q
        gc.collect()
        artifact = args.output / "candidate-alpha-1of2.d"
        status("streaming-export")
        digest = assemble.export(candidate, artifact)
        require(artifact.stat().st_mode & 0o777 == 0o600, "artifact permissions differ")
        status("checking-full-readback")
        decoded = read_artifact(artifact, INTEGER_MODEL, limits=INTEGER_MODEL_LIMITS)
        require(
            same(candidate.arch, decoded.arch)
            and candidate.profile == decoded.profile
            and same(candidate.quant, decoded.quant)
            and same(provenance, decoded.provenance),
            "metadata readback differs",
        )
        require(len(candidate.weights) == len(decoded.weights), "tensor count differs")
        entries = 0
        for expected, found in zip(candidate.weights, decoded.weights, strict=True):
            require(
                (expected.ty, expected.layer, expected.rows, expected.cols)
                == (found.ty, found.layer, found.rows, found.cols),
                "tensor geometry differs",
            )
            require(
                found.data.dtype == np.int8 and not found.data.flags.writeable,
                "readback tensor representation differs",
            )
            require(
                expected.data.size == found.data.size == found.rows * found.cols,
                "tensor length differs",
            )
            for offset in range(0, found.data.size, 8 * 1024 * 1024):
                require(
                    np.array_equal(
                        expected.data[offset : offset + 8 * 1024 * 1024],
                        found.data[offset : offset + 8 * 1024 * 1024],
                    ),
                    "weight readback differs",
                )
            entries += found.data.size
        del decoded
        require(
            assemble.checkpoint_digests(model) == checkpoint,
            "checkpoint changed during run",
        )
        require(sha256_file(report_path) == report_sha, "report changed during run")
        require(
            statistics_digest(
                (args.calibration / "statistics.safetensors").read_bytes()
            ).hex()
            == stats_digest,
            "statistics changed during run",
        )
        require(
            hashes == {p.name: sha256_file(p) for p in source.glob("*.py")},
            "Python source changed during run",
        )
        require(
            sha256_file(Path(__file__)) == launch["script_sha256"],
            "export runner changed during run",
        )
        receipt = {
            "scope": "fixed Servable/affine alpha=1/2 diagnostic; not selected alpha, fidelity or proof",
            "source": launch,
            "checkpoint_blake3": [(name, digest.hex()) for name, digest in checkpoint],
            "checkpoint_sha256": expected_sha256,
            "calibration_report_sha256": report_sha,
            "statistics_digest": stats_digest,
            "selection_digest": calibration.selection_digest.hex(),
            "profile": "Servable",
            "scales": "affine",
            "alpha": "1/2",
            "artifact_bytes": artifact.stat().st_size,
            "artifact_sha256": sha256_file(artifact),
            "artifact_envelope_digest": digest.hex(),
            "weight_tensors": len(candidate.weights),
            "weight_entries": entries,
            "full_weight_and_metadata_readback": True,
            "seconds": time.monotonic() - started,
            "process_peak_rss_bytes": resource.getrusage(
                resource.RUSAGE_SELF
            ).ru_maxrss,
            "rss_units": "bytes on Darwin; KiB on Linux",
        }
        (args.output / "export-receipt.json").write_text(
            json.dumps(receipt, indent=2) + "\n"
        )
        status("complete")
    # Every failure is fatal; redact exception text that might contain secret values.
    except BaseException as error:  # noqa: BLE001
        status("failed", error_type=type(error).__name__)
        raise SystemExit(1) from None


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("checkpoint", "calibration", "c4", "preflight", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    run(parser.parse_args())
