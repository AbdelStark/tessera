#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0 OR MIT
"""Reproduce EXP-0013 verdicts and E0 scenarios from the saved measurements."""
import json
import re
import statistics
import subprocess
from pathlib import Path

HERE = Path(__file__).resolve().parent
RESULTS = HERE / "results"
ROOT = HERE.parents[2]


def main():
    run = json.loads((RESULTS / "run.json").read_text())
    count = json.loads((RESULTS / "count.json").read_text())
    pairs = {p["name"]: p for p in run["pairs"]}
    leaf = pairs["fraction leaf 2^21 point0: scale-first"]
    generic = pairs["fraction generic 2^21 point0: scale-first"]
    needed = [leaf, generic]
    clean = lambda ps: all(p["base_cv"] <= .03 and p["cand_cv"] <= .03 for p in ps)
    verdict26 = "inconclusive"
    if clean(needed):
        if leaf["sigma"] >= .15 and generic["sigma"] >= .08:
            verdict26 = "validated"
        elif leaf["sigma"] < .10:
            verdict26 = "refuted"
    forests = [p for p in run["pairs"] if p["name"].startswith("forest")]
    relation = [p for p in run["pairs"] if p["name"].endswith("relation-first")]
    assert len(forests) == 3 and len(relation) == 2
    verdict30 = "inconclusive"
    if clean(forests + relation):
        if all(p["sigma"] >= .05 for p in forests) and all(p["sigma"] >= .03 for p in relation):
            verdict30 = "validated"
        elif all(p["sigma"] < .02 for p in forests + relation):
            verdict30 = "refuted"
    raw, batch_cases = [], []
    for line in (RESULTS / "run.txt").read_text().splitlines():
        match = re.match(r"RAW batch_seconds=([0-9.e-]+) executions=(\d+)", line)
        if match:
            raw.append({"seconds": float(match[1]), "executions": int(match[2])})
        elif "ns/" in line and " min " in line:
            batch_cases.append({"name": line.split("  ")[0], "batches": raw})
            raw = []
    assert not raw and len(batch_cases) == len(run["cases"])
    for measured, reported in zip(batch_cases, run["cases"]):
        batches = measured["batches"]
        assert len(batches) == (26 if reported["repeated"] else 13)
        assert all(b["seconds"] >= .020 and b["executions"] > 0 for b in batches)
        samples = [b["seconds"] / b["executions"] for b in batches[-10:]]
        assert abs(statistics.median(samples) * 1e9 / reported["elements"] - reported["median_ns"]) < .001
    (RESULTS / "raw-batches.json").write_text(json.dumps(batch_cases, indent=2) + "\n")
    saving = lambda a: sum(
        1 - c["scale-first"]["int32_ops_per_unit_acc4_5_8"][a] / c["split"]["int32_ops_per_unit_acc4_5_8"][a]
        for c in count["fraction"].values()
    ) / 2
    forest = count["forest"]["2^8x2^14"]
    params = {
        **{f"lookup_count_acc{a}": 101 * (1 - .755 * saving(i)) for i, a in enumerate((4, 5, 8))},
        "lookup_cpu_ratio": 101 * (1 - .755 * (leaf["sigma"] + generic["sigma"]) / 2),
        "leaf_cpu_ratio": 41.8 * (1 - pairs["forest 2^8x2^14"]["sigma"]),
        "leaf_count_acc5": 41.8 * forest["deferred"]["int32_ops_per_unit_acc4_5_8"][1] / forest["split"]["int32_ops_per_unit_acc4_5_8"][1],
    }
    (RESULTS / "projection-parameters.json").write_text(json.dumps(params, indent=2) + "\n")
    scenarios = {
        "reference": {},
        "lookup_count": {"lookup_ps": params["lookup_count_acc5"]},
        "lookup_cpu_ratio": {"lookup_ps": params["lookup_cpu_ratio"]},
        "leaf_count": {"leaf_ps": params["leaf_count_acc5"]},
        "leaf_cpu_ratio": {"leaf_ps": params["leaf_cpu_ratio"]},
        "both_count": {"lookup_ps": params["lookup_count_acc5"], "leaf_ps": params["leaf_count_acc5"]},
        "both_cpu_ratio": {"lookup_ps": params["lookup_cpu_ratio"], "leaf_ps": params["leaf_cpu_ratio"]},
    }
    for name, overrides in scenarios.items():
        args = ["python3", "-I", str(ROOT / "research/tools/cost_model.py"), "--profile", "counted", "--set", "leaf_ps=41.8"]
        for k, v in overrides.items():
            args += ["--set", f"{k}={v}"]
        for extension, extra in (("txt", []), ("json", ["--json"])):
            result = subprocess.run(args + extra, check=True, capture_output=True, text=True)
            (RESULTS / f"cost_model_{name}.{extension}").write_text(result.stdout)
    summary = {"experiment": "EXP-0013", "verdict": verdict26, "IDEA-0026": verdict26, "IDEA-0030": verdict30, "all_retained_CV_at_most_3_percent": all(r["cv"] <= .03 for r in run["cases"]), "minimum_raw_batch_seconds": min(b["seconds"] for c in batch_cases for b in c["batches"]), "raw_batches": sum(len(c["batches"]) for c in batch_cases)}
    (RESULTS / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
