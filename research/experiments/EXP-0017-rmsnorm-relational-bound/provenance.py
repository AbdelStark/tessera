#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0 OR MIT
"""Capture exact EXP-0017 source, harness and saved-output provenance."""

from datetime import datetime, timezone
from hashlib import sha256
import json
from pathlib import Path
import platform
import subprocess

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
RESULTS = HERE / "results"
MAIN = "709c37c622caf0351df5ccb9acebcaa6cc09ab6f"
PRECOMMIT = "d4bca74b501d389c3b1a11f89484b1e00ebd8929"


def command(*args):
    return subprocess.check_output(args,cwd=ROOT,text=True).strip()


def digest(path):
    return sha256(path.read_bytes()).hexdigest()


frozen = RESULTS / "README.preregistered.txt"
relative = str(frozen.relative_to(ROOT))
committed = subprocess.check_output(["git","show",f"{PRECOMMIT}:{relative}"],cwd=ROOT)
assert committed == frozen.read_bytes()
paths = []
for folder in ("crates/tessera-compiler/src","crates/tessera-ir/src","crates/tessera-tables/src"):
    paths.extend(sorted((ROOT / folder).glob("*.rs")))
paths.extend(ROOT / p for p in (
    "crates/tessera-bench/tests/commitment_report.rs",
    "docs/perf/v0.2/commitment-report-405b-servable-c2-b32.json",
    "docs/rfcs/RFC-0005-integer-model.md","docs/rfcs/RFC-0006-compiler-ir.md",
    "docs/rfcs/RFC-0009-gadgets.md","docs/rfcs/RFC-0022-saturating-residual.md",
    "docs/rfcs/RFC-0023-norm-precision.md","docs/rfcs/RFC-0028-balanced-split.md",
    "docs/rfcs/RFC-0035-padded-rows.md","python/tessera_quant/integer_model.py",
    "research/tools/cost_model.py"))
source = {str(p.relative_to(ROOT)):digest(p) for p in paths}
critical = ["crates/tessera-compiler/src/"+p+".rs" for p in ("range","select","report","lower")]
critical += ["crates/tessera-bench/tests/commitment_report.rs","docs/rfcs/RFC-0023-norm-precision.md"]
unchanged = {}
for p in critical:
    historical = subprocess.check_output(["git","show",f"{MAIN}:{p}"],cwd=ROOT)
    unchanged[p] = sha256(historical).hexdigest() == source[p]
assert all(unchanged.values()), "relevant code changed; registration amendment required"
harness = {str(p.relative_to(HERE)):digest(p) for p in sorted(HERE.rglob("*"))
           if p.is_file() and "target" not in p.parts and "__pycache__" not in p.parts
           and (p.suffix in (".rs",".py") or p.name in ("Cargo.toml","Cargo.lock"))}
environment = dict(UTC=datetime.now(timezone.utc).isoformat(),CPU=command("sysctl","-n","machdep.cpu.brand_string"),
                   memory_bytes=command("sysctl","-n","hw.memsize"),OS=platform.platform(),
                   python=platform.python_version(),rustc=command("rustc","--version"),cargo=command("cargo","--version"),
                   executing_HEAD=command("git","rev-parse","HEAD"),normative_main=MAIN,
                   preregistration_commit=PRECOMMIT,preregistration_SHA256=digest(frozen),
                   critical_inputs_match_normative_main=unchanged,source_SHA256=source,harness_SHA256=harness,
                   HEAD_delta_note="Later main de5a70c/RFC0047 advice-order/layout validation does not alter these norm/range/selection inputs; no criterion amendment.",
                   evidence="E0; no timing or GPU run")
(RESULTS / "environment.json").write_text(json.dumps(environment,indent=2)+"\n")
saved = {str(p.relative_to(HERE)):digest(p) for p in sorted(RESULTS.iterdir())
         if p.is_file() and p.name != "SHA256SUMS.json"}
(RESULTS / "SHA256SUMS.json").write_text(json.dumps(saved,indent=2)+"\n")
print(json.dumps(dict(harness_files=len(harness),source_files=len(source),saved_files=len(saved),
                     all_critical_inputs_unchanged=True,preregistration_SHA256=digest(frozen)),indent=2))
