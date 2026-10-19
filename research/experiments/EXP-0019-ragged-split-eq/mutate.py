#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0 OR MIT
"""Compile isolated source faults; retain exact rejection, exits and source hashes.

No mutation touches the measured source, production crates, or historical experiment.
"""
from pathlib import Path
import hashlib
import json
import os
import shutil
import subprocess
import tempfile


def main():
    root = Path(__file__).resolve().parent
    repo = root.parents[2]
    scratch = Path(tempfile.mkdtemp(prefix="tessera-exp19-mutations-"))
    (scratch / "src").mkdir()
    manifest = (root / "Cargo.toml").read_text().replace('"../../../', '"' + str(repo) + '/')
    (scratch / "Cargo.toml").write_text(manifest)
    shutil.copyfile(root / "Cargo.lock", scratch / "Cargo.lock")
    for src in (root / "src").glob("*.rs"):
        text = src.read_text().replace('"../../EXP-0009-analytic-padding-blocks/results/amendment-run.json"', '"' + str(repo / "research/experiments/EXP-0009-analytic-padding-blocks/results/amendment-run.json") + '"')
        (scratch / "src" / src.name).write_text(text)
    baseline = (scratch / "src/kernel.rs").read_text()
    measured_hashes = {str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted((root / "src").glob("*.rs"))}
    records = []
    env = dict(os.environ, CARGO_TARGET_DIR=str(scratch / "target"))
    for fault in ["None", "Tail", "Odd", "Order", "Packed", "Fallback"]:
        text = baseline.replace("const FAULT: Fault = Fault::None;", f"const FAULT: Fault = Fault::{fault};")
        assert text != baseline or fault == "None"
        (scratch / "src/kernel.rs").write_text(text)
        run = subprocess.run(["cargo", "run", "--release", "--offline", "--locked", "--", "small"], cwd=scratch, env=env, text=True, capture_output=True)
        (root / f"results/mutation-{fault}.txt").write_text(run.stdout + "\nSTDERR:\n" + run.stderr)
        record = {"fault": fault, "exit_code": run.returncode, "kernel_sha256": hashlib.sha256(text.encode()).hexdigest(), "stdout": run.stdout, "stderr": run.stderr}
        records.append(record)
        assert run.returncode == 0 if fault == "None" else run.returncode != 0, fault
        if fault != "None":
            assert "differs" in run.stderr, (fault, "must fail exact output check, not build/runtime plumbing")
        print(fault, "exit", run.returncode, flush=True)
    for name, expected in measured_hashes.items():
        assert hashlib.sha256((root / name).read_bytes()).hexdigest() == expected
    result = {"experiment": "EXP-0019", "scratch": str(scratch), "measured_source_unchanged": True, "measured_hashes": measured_hashes, "records": records}
    (root / "results/mutations.json").write_text(json.dumps(result, indent=2) + "\n")


if __name__ == "__main__":
    main()
