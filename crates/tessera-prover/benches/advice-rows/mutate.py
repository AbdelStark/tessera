#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0 OR MIT
"""Plant row-streaming faults in a scratch repository; retain controls and every log."""

import argparse
import hashlib
import json
import os
import subprocess
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--target-dir", type=Path, required=True)
    args = parser.parse_args()
    root = args.repo.resolve()
    assert root != Path(__file__).resolve().parents[4], "use a scratch copy"
    out = args.out.resolve()
    out.mkdir(parents=True, exist_ok=False)
    env = dict(
        os.environ,
        CARGO_BUILD_JOBS="2",
        CARGO_TARGET_DIR=str(args.target_dir.resolve()),
    )
    paths = {
        "pcs": root / "crates/tessera-pcs/src/layout.rs",
        "prover": root / "crates/tessera-prover/src/advice.rs",
    }
    originals = {key: path.read_text() for key, path in paths.items()}
    commands = {
        "pcs": [
            "cargo",
            "test",
            "-p",
            "tessera-pcs",
            "--test",
            "source_rows",
            "--locked",
        ],
        "prover": [
            "cargo",
            "test",
            "-p",
            "tessera-prover",
            "--lib",
            "advice::tests",
            "--locked",
        ],
    }
    mutations = [
        (
            "dense-axis-order",
            "pcs",
            "for &axis in order.iter().rev()",
            "for &axis in order.iter()",
        ),
        (
            "score-query-shift",
            "pcs",
            "axes[1].shift = key_bits;",
            "axes[1].shift = key_bits + 1;",
        ),
        (
            "score-key-high-bits",
            "pcs",
            "axes[2].high_mask = (1u128 << (key_vars - key_bits)) - 1;",
            "axes[2].high_mask = 0;",
        ),
        (
            "source-rank",
            "pcs",
            "if shape.len() != rank",
            "if false && shape.len() != rank",
        ),
        ("natural-stride", "pcs", "a.stride = stride;", "a.stride = 1;"),
        (
            "metadata-reservation",
            "pcs",
            "axes.try_reserve_exact(rank)",
            "axes.try_reserve_exact(usize::MAX)",
        ),
        ("source-padding", "pcs", "coordinate < a.len", "coordinate <= a.len"),
        ("tensor-row-offset", "pcs", ".checked_sub(self.first_row)", ".checked_sub(0)"),
        ("row-end-bound", "pcs", "r < self.row_count()", "r <= self.row_count()"),
        (
            "source-axis-bound",
            "pcs",
            "if a.len as u128 > capacity",
            "if false && a.len as u128 > capacity",
        ),
        ("trailing-row-padding", "pcs", "self.vars.min(self.log2_k)", "self.log2_k"),
        (
            "row-zeroing",
            "prover",
            "fields.fill(Goldilocks::ZERO);",
            "// omitted row zeroing",
        ),
        (
            "row-order",
            "prover",
            "place.nonzero().iter().cloned().flatten()",
            "place.nonzero().iter().cloned().flatten().rev()",
        ),
        ("source-index", "prover", "values.get(index)", "values.get(index + 1)"),
        (
            "field-sign",
            "prover",
            "Goldilocks::from_i64(*values.get(index).ok_or(Error::Internal)?)",
            "Goldilocks::from_i64(values.get(index).ok_or(Error::Internal)?.wrapping_neg())",
        ),
        ("sink-error", "prover", "sink(row, &fields)?;", "let _ = sink(row, &fields);"),
        (
            "row-lifetime",
            "prover",
            "    Ok(())",
            "    core::mem::forget(fields);\n    Ok(())",
        ),
        (
            "field-allocation-per-row",
            "prover",
            "    let mut fields = zeros(width, Goldilocks::ZERO)?;\n    for row in place.nonzero().iter().cloned().flatten() {",
            "    for row in place.nonzero().iter().cloned().flatten() {\n        let mut fields = zeros(width, Goldilocks::ZERO)?;",
        ),
        (
            "whole-column-allocation",
            "prover",
            "zeros(width, Goldilocks::ZERO)?",
            "zeros(place.row_count() as usize * width, Goldilocks::ZERO)?",
        ),
    ]
    report = {
        "complete": False,
        "sources": {
            key: hashlib.sha256(value.encode()).hexdigest()
            for key, value in originals.items()
        },
        "commands": commands,
        "runs": [],
    }

    def run(name, key, mutant=False):
        log = out / (name + ".log")
        try:
            with log.open("w") as handle:
                result = subprocess.run(
                    commands[key],
                    check=False,
                    cwd=root,
                    env=env,
                    stdout=handle,
                    stderr=subprocess.STDOUT,
                    timeout=600,
                )
            body = log.read_text()
            killed = (
                result.returncode != 0
                and "test result: FAILED" in body
                and "error: test failed" in body
            )
            passed = result.returncode == 0 and "test result: ok" in body
            entry = {
                "name": name,
                "exit_code": result.returncode,
                "status": "killed"
                if mutant and killed
                else "passed"
                if not mutant and passed
                else "invalid-or-survived",
                "log_sha256": hashlib.sha256(log.read_bytes()).hexdigest(),
            }
        except subprocess.TimeoutExpired:
            entry = {"name": name, "status": "timeout-not-evidence"}
        report["runs"].append(entry)
        (out / "report.json").write_text(json.dumps(report, indent=2) + "\n")
        print(json.dumps(entry), flush=True)
        assert entry["status"] == ("killed" if mutant else "passed"), entry

    try:
        for key in paths:
            run("original-" + key, key)
        for name, key, before, after in mutations:
            full = originals[key]
            if key == "pcs":
                start = full.index("    pub fn row_indices(")
                end = full.index("    /// The row and column", start)
            else:
                start = full.index("fn column_rows(")
                end = full.index("\nfn table_rows(", start)
            body = full[start:end]
            assert body.count(before) == 1, (name, body.count(before))
            paths[key].write_text(
                full[:start] + body.replace(before, after, 1) + full[end:]
            )
            try:
                run(name, key, mutant=True)
            finally:
                paths[key].write_text(originals[key])
    finally:
        for key, path in paths.items():
            path.write_text(originals[key])
        for key in paths:
            run("restored-" + key, key)

    report["complete"] = True
    (out / "report.json").write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
