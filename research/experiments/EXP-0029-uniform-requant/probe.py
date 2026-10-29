#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0 OR MIT
"""Build the frozen scratch-only metadata overlay, retaining all production leaf checks."""

import hashlib
import io
import json
import os
import shutil
import subprocess
import tarfile
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent
REPO = ROOT.parents[2]
BASE = "a407755bc2c03736466b0c1b45dae38c2fb2bcb5"
AMENDMENT_COMMIT = "2ff6ab9"
amendment = "research/experiments/EXP-0029-uniform-requant/results/README.amendment-preregistered.txt"
assert (
    subprocess.check_output(
        ["git", "show", AMENDMENT_COMMIT + ":" + amendment], cwd=REPO
    )
    == (REPO / amendment).read_bytes()
)

scratch = Path(tempfile.mkdtemp(prefix="tessera-617-layout-overlay-"))
archive = subprocess.check_output(["git", "archive", BASE], cwd=REPO)
with tarfile.open(fileobj=io.BytesIO(archive)) as tf:
    tf.extractall(scratch, filter="data")
experiment = scratch / ROOT.relative_to(REPO)
shutil.copytree(
    ROOT,
    experiment,
    dirs_exist_ok=True,
    ignore=shutil.ignore_patterns("target", "__pycache__"),
)

compiler = scratch / "crates/tessera-compiler/src/select.rs"
original = compiler.read_text()
needle = "    reads: BTreeMap<NodeId, Vec<TableId>>,"
replacement = "    /// Scratch-only diagnostic access to the ordered lookup metadata.\n    pub reads: BTreeMap<NodeId, Vec<TableId>>,"
assert original.count(needle) == 1
compiler.write_text(original.replace(needle, replacement))

guards = (
    "crates/tessera-prover/src/leaf/model.rs",
    "crates/tessera-prover/src/leaf/layout.rs",
)
guard_hashes = {}
for path in guards:
    expected = subprocess.check_output(["git", "show", BASE + ":" + path], cwd=REPO)
    assert (scratch / path).read_bytes() == expected == (REPO / path).read_bytes()
    guard_hashes[path] = hashlib.sha256(expected).hexdigest()

env = dict(os.environ, CARGO_BUILD_JOBS="2", CARGO_TARGET_DIR=str(REPO / "target"))
receipt = {
    "scope": "E0 scratch public metadata only; no production change or proof",
    "base": BASE,
    "scratch": str(scratch),
    "leaf_source_sha256": guard_hashes,
    "runs": [],
}
for mode in ("tiny", "real"):
    command = [
        "cargo",
        "run",
        "--release",
        "--offline",
        "--locked",
        "--manifest-path",
        str(experiment / "Cargo.toml"),
        "--",
        mode,
    ]
    log = ROOT / "results" / f"layout-{mode}.log"
    with log.open("w") as output:
        completed = subprocess.run(
            command,
            cwd=scratch,
            env=env,
            stdout=output,
            stderr=subprocess.STDOUT,
            check=False,
        )
    receipt["runs"].append(
        {"mode": mode, "exit_code": completed.returncode, "command": command}
    )
    (ROOT / "results/layout-receipt.json").write_text(
        json.dumps(receipt, indent=2) + "\n"
    )
    assert completed.returncode == 0, f"{mode}: inspect {log}"
    print(log.read_text(), flush=True)
for path, digest in guard_hashes.items():
    assert hashlib.sha256((scratch / path).read_bytes()).hexdigest() == digest
assert compiler.read_text() == original.replace(
    needle, "    pub reads: BTreeMap<NodeId, Vec<TableId>>,"
)
receipt["leaf_sources_unchanged_after_execution"] = True
receipt["compiler_change"] = (
    "visibility/documentation of Selection.reads only; no algorithm or check changed"
)
(ROOT / "results/layout-receipt.json").write_text(json.dumps(receipt, indent=2) + "\n")
print("all frozen layout cases pass; original leaf source bytes unchanged", flush=True)
