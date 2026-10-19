#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0 OR MIT
"""Check three arithmetic fault injections in an isolated temporary workspace.

Scratch files are retained for inspection. No production or experiment file is edited.
Run after timing, since compiling mutants would disturb wall-clock samples.
"""

from pathlib import Path
import json
import re
import subprocess
import tempfile


def main() -> None:
    here = Path(__file__).resolve().parent
    scratch = Path(tempfile.mkdtemp(prefix="tessera-exp12-mutants-"))
    (scratch / "src").mkdir()
    manifest = (here / "Cargo.toml").read_text()
    manifest = re.sub(
        r'path = "(\.\./\.\./\.\./[^"\n]+)"',
        lambda m: 'path = "' + str((here / m[1]).resolve()) + '"',
        manifest,
    )
    (scratch / "Cargo.toml").write_text(manifest)
    (scratch / "Cargo.lock").write_bytes((here / "Cargo.lock").read_bytes())
    original = (here / "src/main.rs").read_text()
    cases = [
        ("control", None, None),
        ("sign-correction", "- Fp::from_u64_reduced(((x >> 127) as u64) & (eps * eps))", "- Fp::ZERO"),
        ("integer-grid", "= 2 * y - x;", "= 3 * y - x;"),
        ("quadratic-collapse", "quadratic_weight * (c - b - b + a)", "quadratic_weight * (c - b + a)"),
    ]
    outcomes = []
    for name, before, after in cases:
        text = original
        if before is not None:
            assert original.count(before) == 1, (name, "mutation anchor changed")
            text = original.replace(before, after)
        (scratch / "src/main.rs").write_text(text)
        run = subprocess.run(
            ["cargo", "run", "--release", "--offline", "--locked", "--manifest-path", str(scratch / "Cargo.toml"), "--", "check"],
            text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, check=False,
        )
        (here / "results" / f"mutation-{name}.txt").write_text(run.stdout)
        if name == "control":
            assert run.returncode == 0, run.stdout[-3000:]
        else:
            assert run.returncode != 0 and "panicked at" in run.stdout, run.stdout[-3000:]
            assert "assertion" in run.stdout or "messages differ" in run.stdout, run.stdout[-3000:]
        outcomes.append({"name": name, "exit_code": run.returncode, "expected": "pass" if name == "control" else "assertion failure"})
    result = {"experiment": "EXP-0012", "control_passed": True, "mutants_caught": 3, "mutants_total": 3, "outcomes": outcomes, "scratch": str(scratch)}
    (here / "results/mutations.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
