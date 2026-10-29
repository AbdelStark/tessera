# SPDX-License-Identifier: Apache-2.0 OR MIT
"""Prepare release metadata and compare independent builds (09 §9)."""

import argparse
import hashlib
import json
import re
import subprocess
import tomllib
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
TAG = re.compile(
    r"(verify-)?v(0|[1-9][0-9]*)\.(0|[1-9][0-9]*)\.(0|[1-9][0-9]*)(-rc(?:\.[1-9][0-9]*)?)?"
)
TARGETS = ("x86_64-unknown-linux-gnu", "aarch64-unknown-linux-gnu")


def metadata(root, tag, dry_run):
    """Reject unsupported tags and version mismatches before building."""
    match = TAG.fullmatch(tag)
    if match is None:
        raise ValueError("expected vX.Y.Z or verify-vX.Y.Z (optionally -rc or -rc.N)")
    verifier = bool(match[1])
    manifest = root / ("verifier/Cargo.toml" if verifier else "Cargo.toml")
    version = tomllib.loads(manifest.read_text())["workspace"]["package"]["version"]
    requested = ".".join(match.group(i) for i in (2, 3, 4)) + (match[5] or "")
    # The issue's prescribed rehearsal must work without changing a release version.
    if requested != version and not (dry_run and tag == "verify-v0.0.1-rc"):
        raise ValueError(
            f"tag version {requested} differs from manifest version {version}"
        )
    return {"tag": tag, "verifier": verifier, "version": version, "dry_run": dry_run}


def changelog_section(text, version, dry_run=True):
    """Prefer the released section; use Unreleased for a rehearsal."""
    headings = (
        (f"## [{version}]", "## [Unreleased]") if dry_run else (f"## [{version}]",)
    )
    for heading in headings:
        for match in re.finditer(r"^## .*", text, re.MULTILINE):
            if match[0] == heading or match[0].startswith(heading + " - "):
                tail = text[match.end() :]
                return re.split(r"^## ", tail, maxsplit=1, flags=re.MULTILINE)[
                    0
                ].strip()
    raise ValueError(
        "changelog lacks the required release section (Unreleased is allowed only for dry runs)"
    )


def prepare(root, tag, dry_run, out):
    info = metadata(root, tag, dry_run)
    commit = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=root, text=True
    ).strip()
    out.mkdir(parents=True, exist_ok=True)
    info["commit"] = commit
    (out / "release.json").write_text(json.dumps(info, indent=2) + "\n")
    changelog = root / ("verifier/CHANGELOG.md" if info["verifier"] else "CHANGELOG.md")
    notes = (
        f"# {tag}\n\nCommit: `{commit}`. Built package version: `{info['version']}`.\n\n"
        + ("Dry run; these artifacts are a rehearsal.\n\n" if dry_run else "")
        + "Draft only. Before publication, link the completed [release checklist]"
        "(https://github.com/AbdelStark/tessera/blob/"
        + commit
        + "/docs/release/checklist.md)"
        " and record the supported protocol versions (09 §3), required reviews and gate results.\n\n"
        + changelog_section(changelog.read_text(), info["version"], dry_run)
        + "\n"
    )
    (out / "notes.md").write_text(notes)
    return info


def compare(builds, out):
    """Require both builders' full binary bytes and provenance to agree per target."""
    records = []
    for target in TARGETS:
        left = builds / f"verifier-{target}-1"
        right = builds / f"verifier-{target}-2"
        name = f"tessera-verify-{target}"
        first = (left / name).read_bytes()
        if not first or first != (right / name).read_bytes():
            raise ValueError(f"independent binaries differ or are empty: {target}")
        provenance = (left / "build.json").read_bytes()
        if provenance != (right / "build.json").read_bytes():
            raise ValueError(f"independent provenance differs: {target}")
        info = json.loads(provenance)
        expected_commit = json.loads((out / "release.json").read_text())["commit"]
        if info["commit"] != expected_commit or info["target"] != target:
            raise ValueError(
                f"build provenance is for another commit or target: {target}"
            )
        digest = hashlib.sha256(first).hexdigest()
        records.append(f"{digest}  {name}\n")
        (out / name).write_bytes(first)
        (out / name).chmod(0o755)
        (out / f"{name}.build.json").write_bytes(provenance)
    (out / "SHA256SUMS").write_text("".join(records))
    with (out / "notes.md").open("a") as notes:
        notes.write("\n## SHA-256 (two independent builders agree)\n\n```text\n")
        notes.writelines(records)
        notes.write("```\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    prep = commands.add_parser("prepare")
    prep.add_argument("tag")
    prep.add_argument("--dry-run", action="store_true")
    prep.add_argument("--out", type=Path, required=True)
    check = commands.add_parser("compare")
    check.add_argument("builds", type=Path)
    check.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    try:
        if args.command == "prepare":
            print(json.dumps(prepare(ROOT, args.tag, args.dry_run, args.out)))
        else:
            compare(args.builds, args.out)
    except (ValueError, KeyError, OSError) as error:
        parser.exit(1, f"release: {error}\n")


if __name__ == "__main__":
    main()
