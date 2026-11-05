#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0 OR MIT
"""Prepare a visibility-only reference overlay and run EXP-0031 stages."""

import argparse
import datetime
import difflib
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import tarfile

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
BASE = "646f7addc7bc737f2fbb8970c865df8333877093"
REGISTRATION = "ede585df1fc290df27c32ba2a7b4a6672a69cc74"
FROZEN = "972efa1bdf13f49e271ab58f41edde3a6b54d17605d00e4b9d39bd8e02bea2c0"


def digest(data):
    return hashlib.sha256(data).hexdigest()


def verify_registration():
    relative = HERE.relative_to(REPO) / "results/README.preregistered.txt"
    committed = subprocess.check_output(
        ["git", "show", f"{REGISTRATION}:{relative}"], cwd=REPO
    )
    if digest(committed) != FROZEN or (HERE / "results/README.preregistered.txt").read_bytes() != committed:
        raise SystemExit("frozen registration differs from its pre-code commit")
    changed = subprocess.check_output(
        ["git", "diff", BASE, "--", "crates", "verifier"], cwd=REPO
    )
    if changed:
        raise SystemExit("production/verifier source differs from registered base")


def save(path, value):
    path.write_text(json.dumps(value, indent=2) + "\n")


def prepare(scratch):
    if scratch.exists():
        raise SystemExit("scratch already exists; inspect it rather than overwrite")
    scratch.mkdir(mode=0o700, parents=True)
    reference = scratch / "reference"
    reference.mkdir()
    process = subprocess.Popen(
        ["git", "archive", "--format=tar", BASE, "--", ".", ":(exclude)research"],
        cwd=REPO,
        stdout=subprocess.PIPE,
    )
    with tarfile.open(fileobj=process.stdout, mode="r|") as archive:
        archive.extractall(reference, filter="data")
    if process.wait() != 0:
        raise SystemExit("registered source archive failed")
    edits = {
        "crates/tessera-prover/src/leaf/mod.rs": [("mod grouped;", "pub mod grouped;")],
        "crates/tessera-prover/src/leaf/grouped.rs": [
            (
                "fn group_columns_from<",
                "/// Reference-only visibility for EXP-0031; algorithm unchanged.\npub fn group_columns_from<",
            ),
            ("fn forest_share(", "pub fn forest_share("),
        ],
    }
    receipt = {"base": BASE, "registration": REGISTRATION, "frozen_sha256": FROZEN, "edits": {}}
    for relative, replacements in edits.items():
        path = reference / relative
        before = path.read_text()
        after = before
        for old, new in replacements:
            if after.count(old) != 1:
                raise SystemExit("reference visibility anchor is not unique: " + relative)
            after = after.replace(old, new, 1)
        path.write_text(after)
        receipt["edits"][relative] = {
            "before_sha256": digest(before.encode()),
            "after_sha256": digest(after.encode()),
            "diff": "".join(difflib.unified_diff(before.splitlines(True), after.splitlines(True), fromfile=relative, tofile=relative)),
        }
    save(scratch / "reference-overlay.json", receipt)
    sync(scratch)


def sync(scratch):
    if not (scratch / "reference-overlay.json").is_file():
        raise SystemExit("reference overlay is missing")
    receipt = json.loads((scratch / "reference-overlay.json").read_text())
    for relative, row in receipt["edits"].items():
        if digest((scratch / "reference" / relative).read_bytes()) != row["after_sha256"]:
            raise SystemExit("reference overlay changed: " + relative)
    destination = scratch / "reference" / HERE.relative_to(REPO)
    destination.mkdir(parents=True, exist_ok=True)
    files = [HERE / "Cargo.toml", *sorted(HERE.glob("*.rs"))]
    if (HERE / "Cargo.lock").is_file():
        files.append(HERE / "Cargo.lock")
    hashes = {}
    for source in files:
        target = destination / source.name
        shutil.copyfile(source, target)
        hashes[source.name] = digest(source.read_bytes())
    save(scratch / "harness-source.json", hashes)
    return destination


def command(scratch, label, argv, cwd):
    environment = os.environ.copy()
    environment["CARGO_TARGET_DIR"] = str(scratch / "target")
    environment["CARGO_BUILD_JOBS"] = "2"
    attempt = datetime.datetime.now(datetime.timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    log = scratch / (label + "-" + attempt + ".log")
    with log.open("w") as output:
        result = subprocess.run(argv, cwd=cwd, env=environment, stdout=output, stderr=subprocess.STDOUT, umask=0o022)
    receipt = {
        "observed_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "command": argv,
        "exit_code": result.returncode,
        "log": str(log),
        "harness_sources": json.loads((scratch / "harness-source.json").read_text()),
        "runner_sha256": digest(Path(__file__).read_bytes()),
        "scope": "partial stage; no full experiment or production verdict",
    }
    save(scratch / (label + "-" + attempt + "-receipt.json"), receipt)
    save(scratch / (label + "-receipt.json"), receipt)
    if result.returncode:
        print("\n".join(log.read_text().splitlines()[-35:]))
        raise SystemExit(result.returncode)
    print(label + " completed; receipt: " + str(scratch / (label + "-receipt.json")))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("prepare", "build", "clippy", "source", "merge"))
    parser.add_argument("scratch", type=Path)
    args = parser.parse_args()
    verify_registration()
    scratch = args.scratch.resolve()
    if args.stage == "prepare":
        prepare(scratch)
    elif args.stage in ("build", "clippy"):
        destination = sync(scratch)
        argv = ["cargo", args.stage, "--manifest-path", str(destination / "Cargo.toml")]
        if (destination / "Cargo.lock").exists():
            argv.append("--locked")
        if args.stage == "clippy":
            argv += ["--", "-D", "warnings"]
        command(scratch, args.stage, argv, destination)
        if args.stage == "build":
            shutil.copyfile(destination / "Cargo.lock", HERE / "Cargo.lock")
    else:
        destination = scratch / "reference" / HERE.relative_to(REPO)
        previous = scratch / (args.stage + "-partial.json")
        if previous.is_file():
            stamp = datetime.datetime.now(datetime.timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
            shutil.copyfile(previous, scratch / (args.stage + "-partial-before-" + stamp + ".json"))
        command(scratch, args.stage, [str(scratch / "target/debug/bounded-grouped-source"), args.stage + "-only", str(previous)], destination)


if __name__ == "__main__":
    main()
