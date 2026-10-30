#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0 OR MIT
"""Run only frozen public geometry in a private scratch visibility overlay."""

import hashlib
import io
import json
import os
import platform
import shutil
import subprocess
import sys
import tarfile
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent
REPO = ROOT.parents[2]
BASE = "51fbd9353a42259e0fdc07973aae6f3857a90f6a"
REGISTRATION = "4d25903"
FROZEN = ROOT.relative_to(REPO) / "results/README.preregistered.txt"
assert subprocess.check_output(["git", "show", f"{REGISTRATION}:{FROZEN}"], cwd=REPO) == (REPO / FROZEN).read_bytes()
assert hashlib.sha256((REPO / FROZEN).read_bytes()).hexdigest() == "24b68bbfd9bc3dc64dbc5b8ab0adb4f24ecc8de89f8240ddb0a7964fe34639f0"

if len(sys.argv) > 1:
    prior = json.loads((ROOT / "results/initial-build-run-receipt.json").read_text())
    scratch = Path(sys.argv[1])
    assert str(scratch) == prior["scratch"] and prior["base"] == BASE
    assert prior["exit_code"] != 0
else:
    scratch = Path(tempfile.mkdtemp(prefix="tessera-627-runtime-overlay-"))
    archive = subprocess.check_output(["git", "archive", BASE], cwd=REPO)
    with tarfile.open(fileobj=io.BytesIO(archive)) as tar:
        tar.extractall(scratch, filter="data")
experiment = scratch / ROOT.relative_to(REPO)
shutil.copytree(ROOT, experiment, dirs_exist_ok=True, ignore=shutil.ignore_patterns("target", "__pycache__"))

patches = {
    "crates/tessera-compiler/src/select.rs": [
        ("    reads: BTreeMap<NodeId, Vec<TableId>>,", "    /// Scratch-only public metadata injection.\n    pub reads: BTreeMap<NodeId, Vec<TableId>>,"),
    ],
    "crates/tessera-prover/src/cpu/mod.rs": [
        ("mod backend;", "/// Scratch-only access to the existing public-metadata weight charge.\npub mod backend;"),
    ],
    "crates/tessera-prover/src/cpu/backend.rs": [
        ("fn weight_memory(c: &ClaimList)", "/// Scratch-only public metadata charge; body unchanged.\npub fn weight_memory(c: &ClaimList)"),
    ],
    "crates/tessera-prover/src/native.rs": [
        ("    pub(crate) fn window_shape(", "    /// Scratch-only metadata visibility.\n    pub fn window_shape("),
    ],
    "crates/tessera-prover/src/stages/stream.rs": [
        ("    pub(crate) fn window_shapes(", "    /// Scratch-only metadata visibility.\n    pub fn window_shapes("),
    ],
    "crates/tessera-prover/src/cpu/plan.rs": [
        ("pub(crate) fn new(", "pub fn new("),
        ("    pub(crate) fn staged(", "    /// Scratch-only visibility; metadata algorithm unchanged.\n    pub fn staged("),
        ("pub(crate) fn for_budget(", "pub fn for_budget("),
        ("pub(crate) fn grouped(", "pub fn grouped("),
    ],
}
if len(sys.argv) > 1:
    # Validate every owned source before restoring an earlier failed setup; no partial cleanup.
    restores = {}
    for path in set(prior["patches"]) | set(patches):
        original = subprocess.check_output(["git", "show", f"{BASE}:{path}"], cwd=REPO).decode()
        assert original == (REPO / path).read_text()
        variants = {original}
        for recipe in [prior["patches"].get(path, []), patches.get(path, [])]:
            changed = original
            for before, after in recipe:
                assert changed.count(before) == 1
                changed = changed.replace(before, after)
            variants.add(changed)
        assert (scratch / path).read_text() in variants, path
        restores[path] = original
    for path, text in restores.items():
        (scratch / path).write_text(text)

# The normal module is exactly the already-studied helper bodies, with visibility only.
prior_adapter = (scratch / "research/experiments/EXP-0029-uniform-requant/layout_probe.rs").read_text()
helper_body = prior_adapter[prior_adapter.index("fn table("):prior_adapter.index("fn main()")]
helper_body = helper_body.replace("fn patch(", "pub(super) fn patch(")
module_text = (ROOT / "candidate.rs").read_text()
assert module_text[module_text.index("fn table("):].rstrip() == helper_body.rstrip()

originals = {}
prepared_patches = {}
for path, replacements in patches.items():
    original = (scratch / path).read_text()
    expected = subprocess.check_output(["git", "show", f"{BASE}:{path}"], cwd=REPO).decode()
    assert original == expected == (REPO / path).read_text()
    changed = original
    for before, after in replacements:
        assert changed.count(before) == 1, (path, before)
        changed = changed.replace(before, after)
    prepared_patches[path] = changed
    originals[path] = original
for path, changed in prepared_patches.items():
    (scratch / path).write_text(changed)

protected = [
    "crates/tessera-prover/src/leaf/model.rs", "crates/tessera-prover/src/leaf/layout.rs",
    "crates/tessera-compiler/src/advice.rs",
    "crates/tessera-sumcheck/src/prover/window/memory.rs",
]
hashes = {p: hashlib.sha256((scratch / p).read_bytes()).hexdigest() for p in protected}
for path in protected:
    assert (scratch / path).read_bytes() == subprocess.check_output(["git", "show", f"{BASE}:{path}"], cwd=REPO) == (REPO / path).read_bytes()
receipt = {"scope": "E0 public metadata; no production adoption, full G-4, witness or proof", "base": BASE, "registration": REGISTRATION, "scratch": str(scratch), "platform": platform.platform(), "python": platform.python_version(), "protected_sha256": hashes, "patches": patches, "adapter_sha256": {p:hashlib.sha256((ROOT/p).read_bytes()).hexdigest() for p in ["probe.rs","candidate.rs","Cargo.toml","Cargo.lock","run.py"]}, "modified_sha256": {p: hashlib.sha256((scratch / p).read_bytes()).hexdigest() for p in patches}}
(ROOT / "results/run-receipt.json").write_text(json.dumps(receipt, indent=2) + "\n")
env = dict(os.environ, CARGO_BUILD_JOBS="2", CARGO_TARGET_DIR=str(scratch / "target"))
command = ["cargo", "run", "--release", "--offline", "--locked", "--manifest-path", str(experiment / "Cargo.toml")]
print("Scratch", scratch, flush=True)
with (ROOT / "results/run.log").open("w") as log:
    completed = subprocess.run(command, cwd=scratch, env=env, stdout=log, stderr=subprocess.STDOUT, check=False)
receipt["command"] = command
receipt["exit_code"] = completed.returncode
receipt["rustc"] = subprocess.check_output(["rustc", "--version"], cwd=REPO, text=True).strip()
for path in protected:
    assert hashlib.sha256((scratch / path).read_bytes()).hexdigest() == hashes[path]
for path, replacements in patches.items():
    restored = (scratch / path).read_text()
    for before, after in reversed(replacements):
        assert restored.count(after) == 1
        restored = restored.replace(after, before)
    assert restored == originals[path] == (REPO / path).read_text()
receipt["all_algorithm_sources_preserved"] = True
(ROOT / "results/run-receipt.json").write_text(json.dumps(receipt, indent=2) + "\n")
assert completed.returncode == 0, "inspect results/run.log; do not change the frozen domain"
shutil.copyfile(experiment / "results/counts.json", ROOT / "results/counts.json")
print("Complete counts copied; source restoration checks passed.", flush=True)
