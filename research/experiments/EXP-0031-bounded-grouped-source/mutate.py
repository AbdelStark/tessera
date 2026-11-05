#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0 OR MIT
"""Run the registered compiled source/accounting/identity fault controls serially."""

import argparse
import datetime
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import re
import subprocess
import sys

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location("exp31runner", HERE / "run.py")
runner = importlib.util.module_from_spec(spec)
spec.loader.exec_module(runner)

# Each pattern must identify exactly one implementation site before running.
FAULTS = [
    ("swap-token-layer-bits", "adapter.rs", r"let token = h as u64 & \(\(1 << gt\) - 1\);\s*let layer = h as u64 >> gt;", "let token = h as u64 >> gt; let layer = h as u64 & ((1 << gt) - 1);", "source", ["canonical keys", "canonical forest share", "forest source indices", "source index transport"]),
    ("omit-request-slot-gap", "fixture.rs", r"if lookup.token == Some\(axis\) && !self.real_token\(at\)", "if false && lookup.token == Some(axis) && !self.real_token(at)", "source", ["independent canonical column values/padding"]),
    ("omit-row-zero-padding", "adapter.rs", r"v.resize\(n, G::from_u64_reduced\(3\)\);", "v.resize(n, G::from_u64_reduced(0));", "source", ["canonical values/padding", "independent canonical column values/padding"]),
    ("repeat-no-token-role", "fixture.rs", r"lookup\(vec!\[3\], None, 2, None, false\)", "lookup(vec![3], Some(0), 2, None, false)", "source", ["held roles, ordering or offsets", "independent canonical column values/padding"]),
    ("misplace-outer-site-identity", "adapter.rs", r"group: h as u64,\s*table: instance.table.0,", "group: if block.site != 1 { (h ^ (1 << gt)) as u64 } else { h as u64 }, table: instance.table.0,", "source", ["source index transport"]),
    ("wrong-forest-query-bit", "adapter.rs", r"token >> \(bit - forest.queries\)", "token >> (bit - forest.queries + 1)", "source", ["canonical forest share", "forest source indices"]),
    ("wrong-forest-layer-bit", "adapter.rs", r"layer >> \(bit - forest.layers\)", "layer >> (bit - forest.layers + 1)", "source", ["canonical forest share", "forest source indices"]),
    ("wrong-split-coordinate-order", "merge_probe.rs", r"original\s*\.coords\s*\.iter\(\)", "original.coords.iter().rev()", "merge", ["split coordinate order"]),
    ("omit-earlier-output-overlap", "memory.rs", r"let peak = add\(add\(add\(columns, forest\)\?, records\)\?, earlier\)\?;", "let peak = add(add(columns, forest)?, records)?;", "source", ["source peak within public bill", "public capacity/lifetime recount"]),
    ("omit-original-p2b-source", "merge_probe.rs", r"let callback_bill\s*=\s*growing::<G>\(full_max\)\.expect\(\"original P2b regeneration/copy\"\);", "let callback_bill = 0u64;", "merge", ["P2b engine and original callback source overlap", "original P2b source/copy charge"]),
    ("omit-nested-record-capacity", "memory.rs", r"growing::<IndexRecord>\(row_records\)\?,", "0,", "source", ["source peak within public bill", "public capacity/lifetime recount"]),
    ("wrap-size-addition", "memory.rs", r"a.checked_add\(b\).ok_or\(Refusal::Memory\)", "Ok(a.wrapping_add(b))", "source", ["assertion `left == right` failed"]),
    ("full-original-during-group", "merge_probe.rs", r"let mut values =\s*SecretBuf::try_with_capacity\(1 << shape.free_vars\(\)\)", "let _eager = self.values(i, shape); let mut values = SecretBuf::try_with_capacity(1 << shape.free_vars())", "merge", ["no full original source during group pass", "merge input and split scratch bill"]),
    ("underdeclare-message-dimensions", "merge_probe.rs", r"let driver = GroupedMessageMemory::new", "let message_vars = n_max; let driver = GroupedMessageMemory::new", "merge", ["message dimension upper bound"]),
]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("scratch", type=Path)
    args = parser.parse_args()
    runner.verify_registration()
    scratch = args.scratch.resolve()
    destination = runner.sync(scratch)
    stamp = datetime.datetime.now(datetime.timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    out = scratch / ("mutation-campaign-" + stamp)
    out.mkdir(mode=0o700)
    environment = os.environ.copy()
    environment["CARGO_TARGET_DIR"] = str(scratch / "target")
    environment["CARGO_BUILD_JOBS"] = "2"
    baseline = {p.name: p.read_text() for p in HERE.glob("*.rs")}
    hashes = {p: hashlib.sha256(text.encode()).hexdigest() for p, text in baseline.items()}
    for name in ["Cargo.toml","Cargo.lock","audit.py","run.py","mutate.py"]:
        hashes[name]=hashlib.sha256((HERE/name).read_bytes()).hexdigest()
    replacements = []
    for name, file, pattern, replacement, stage, expected in FAULTS:
        changed, count = re.subn(pattern, lambda _: replacement, baseline[file])
        if count != 1:
            raise SystemExit(f"mutation anchor not unique: {name} ({count})")
        replacements.append((name, file, changed, stage, expected))
    rows = []

    def status(stage, **extra):
        runner.save(scratch / "mutations-status.json", {"stage":stage,"pid":os.getpid(),"observed_utc":datetime.datetime.now(datetime.timezone.utc).isoformat(),"directory":str(out),"records":len(rows),**extra})
        runner.save(out / "results.json", rows)

    def execute(label, command):
        log = out / (label + ".log")
        with log.open("w") as handle:
            process = subprocess.run(command, cwd=destination, env=environment, stdout=handle, stderr=subprocess.STDOUT, umask=0o022)
        return process.returncode, log.read_text(), log

    def build(label):
        code, text, log = execute(label, ["cargo","build","--locked","--manifest-path",str(destination / "Cargo.toml")])
        if code:
            status("compile-failed-not-detection", label=label, log=str(log))
            raise SystemExit(text[-6000:])

    binary = str(scratch / "target/debug/bounded-grouped-source")

    def stage(label, kind):
        path = out / (label + ".json")
        code, text, log = execute(label, [binary,kind+"-only",str(path)])
        return code, text, log, path

    def audit(label, source, merge=None):
        command = [sys.executable,"-B","-I",str(HERE / "audit.py"),str(source),str(out/(label+".json"))]
        if merge is not None:
            command += ["--merge",str(merge)]
        return execute(label, command)

    def control(prefix):
        build(prefix+"-build")
        for kind in ["source","merge"]:
            code, text, _, path = stage(prefix+"-"+kind, kind)
            if code:
                status("control-failed", control=prefix)
                raise SystemExit(text[-6000:])
            if kind == "source":
                source = path
            else:
                merge = path
        code, text, _ = audit(prefix+"-audit",source,merge)
        if code:
            status("control-audit-failed", control=prefix)
            raise SystemExit(text[-6000:])
        rows.append({"kind":"control","name":prefix,"exit_code":0})
        return source

    status("running-controls", source_sha256=hashes)
    try:
        source_control = control("before")
        for name, file, changed, kind, expected in replacements:
            status("running-mutant", mutant=name)
            path = destination / file
            path.write_text(changed)
            try:
                build(name+"-build")
                code, text, log, result = stage(name,kind)
                detection = "compiled Rust assertion"
                if code == 0:
                    code, text, log = audit(name+"-audit", result if kind=="source" else source_control, result if kind=="merge" else None)
                    detection = "independent Python assertion"
                if code == 0 or not any(message in text for message in expected):
                    status("mutant-escaped-or-unexpected-failure",mutant=name,log=str(log))
                    raise SystemExit(f"{name}: not detected by its registered check\n{text[-5000:]}")
                if "AssertionError" not in text and "panicked at" not in text:
                    raise SystemExit(name+": non-assertion failure cannot count")
                rows.append({"kind":"mutant","name":name,"compiled":True,"detected":True,"detector":detection,"exit_code":code,"log":str(log)})
                print(name+": caught by "+detection,flush=True)
            finally:
                path.write_text(baseline[file])
        control("restored")
        for name, text in baseline.items():
            assert (destination/name).read_text()==text, "scratch candidate not restored"
            assert (HERE/name).read_text()==text, "canonical experiment source changed during campaign"
        runner.verify_registration()
        for name, sha in hashes.items():
            assert hashlib.sha256((HERE/name).read_bytes()).hexdigest()==sha, "experiment source changed during campaign"
        status("all-passed",mutants=len(FAULTS),registered_mutants=12,additional_review_controls=len(FAULTS)-12,controls=2,source_restored=True,source_sha256=hashes)
    except BaseException:
        for name, text in baseline.items():
            (destination/name).write_text(text)
        raise


if __name__ == "__main__":
    main()
