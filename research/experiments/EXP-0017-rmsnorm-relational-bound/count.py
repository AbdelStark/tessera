#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0 OR MIT
"""Whole-model table differences and conditional current-model E0 projections."""

from dataclasses import asdict, replace
import importlib.util
import json
from pathlib import Path
import sys

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]


def module(name, path):
    spec = importlib.util.spec_from_file_location(name,path)
    obj = importlib.util.module_from_spec(spec)
    sys.modules[name] = obj
    spec.loader.exec_module(obj)
    return obj


check = module("normcheck",HERE / "check.py")
model = module("costmodel",ROOT / "research/tools/cost_model.py")
compiler = json.loads((HERE / "results/compiler.json").read_text())
summaries = check.inspect_compiler(compiler)
cases = []
for case, summary in zip(compiler["cases"],summaries):
    before = {check.canonical(t["kind"]):t for t in case["baseline_tables"]}
    after = {check.canonical(t["kind"]):t for t in case["candidate_tables"]}
    deleted = [before[k] for k in sorted(before.keys()-after.keys())]
    added = [after[k] for k in sorted(after.keys()-before.keys())]
    assert sum(t["rows"] for t in added)-sum(t["rows"] for t in deleted) == summary["M_delta"]
    cases.append(dict(summary=summary,removed=deleted,added=added,
                      retained=[before[k] for k in sorted(before.keys()&after.keys())]))
current = next(c for c in cases if (c["summary"]["d"],c["summary"]["mode"]) == (16384,30))
projections = {}
for profile in ("corrected","counted"):
    baseline = model.PROFILES[profile]
    s = current["summary"]
    candidate = replace(baseline,commits_per_token=s["C"],lookups_per_token=s["K"],table_rows=s["M_after"])
    records = []
    for context in model.DEFAULT_CONTEXTS:
        b, c = model.mu(baseline,context),model.mu(candidate,context)
        records.append(dict(context=context,baseline=asdict(b),candidate=asdict(c),delta_percent=100*(c.total/b.total-1)))
    projections[profile] = dict(baseline_inputs=asdict(baseline),candidate_inputs=asdict(candidate),contexts=records)
result = dict(cases=cases,projections=projections,
              limitations=["E0 conditional current planning model; no GPU measurement",
                           "Remaining RFC0018 P4 and full P5/streaming implementation omitted or bounded as current model declares",
                           "RsqrtRow estimates unchanged; omitted row-gadget range-table rows unchanged",
                           "Logical table row reduction is not a measured memory/throughput result"])
(HERE / "results/count.json").write_text(json.dumps(result,indent=2)+"\n")
print(json.dumps(result,indent=2))
