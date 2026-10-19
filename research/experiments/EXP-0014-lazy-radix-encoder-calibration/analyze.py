#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0 OR MIT
"""Audit raw measurements and apply the frozen EXP-0014 calibration criterion."""
import json
import math
import statistics
from pathlib import Path

HERE = Path(__file__).resolve().parent


def main():
    r = HERE / "results"
    run = json.loads((r / "run.json").read_text())
    asm = json.loads((r / "asm.json").read_text())
    count = json.loads((r / "count.json").read_text())
    assert len(run["cases"]) == 77
    rows = {}
    for case in run["cases"]:
        attempts = case["attempts"]
        assert len(attempts) in (1, 2)
        assert case["retained_attempt"] == len(attempts)-1
        if len(attempts) == 2:
            assert attempts[0]["cv"] > .03
        for attempt in attempts:
            batches = attempt["batches"]
            assert len(batches) == 13
            assert [b["warmup"] for b in batches] == [True]*3 + [False]*10
            assert all(b["seconds"] >= .020 and b["executions"] >= 1 for b in batches)
            samples = [b["seconds"]/b["executions"] for b in batches[3:]]
            assert samples == attempt["samples_seconds_per_execution"]
            assert math.isclose(statistics.median(samples), attempt["median_seconds"], rel_tol=1e-14)
            cv = statistics.stdev(samples)/statistics.mean(samples)
            assert math.isclose(cv, attempt["cv"], abs_tol=1e-14)
        a = attempts[-1]
        rows[case["name"]] = {"ns_per_element": a["median_seconds"]*1e9/case["elements"], "min_ns": a["min_seconds"]*1e9/case["elements"], "max_ns": a["max_seconds"]*1e9/case["elements"], "cv": a["cv"], "repeated": len(attempts)==2}
    pairs = {}
    necessary = []
    for prefix in ("encode 2^22", "ntt 2^22", "ntt 2^23"):
        base, cand = rows[prefix+" production"], rows[prefix+" lazy16"]
        pairs[prefix] = {"sigma": 1-cand["ns_per_element"]/base["ns_per_element"], "base": base, "candidate": cand}
        necessary += [base, cand]
    clean = all(row["cv"] <= .03 for row in necessary)
    branches = any(a["conditional_or_call_instructions"] for a in asm["anchors"].values())
    ratio = asm["ratio"]
    enc, n22, n23 = [pairs[p]["sigma"] for p in ("encode 2^22", "ntt 2^22", "ntt 2^23")]
    verdict = "inconclusive"
    if clean and not branches:
        if enc >= .25 and n22 >= .15 and n23 >= .15 and ratio <= .90:
            verdict = "validated"
        elif min(enc, n22, n23) < .15 or ratio > .90:
            verdict = "refuted"
    baseline_ntt = count["historical_radix2"]["ntt_ops"]
    floor_ntt = count["historical_radix16_floor"]["ntt_ops"]
    out = {"experiment": "EXP-0014", "verdict": verdict, "necessary_CVs_clean": clean,
           "anchor_instruction_ratio": ratio, "anchors_have_branch_or_call": branches,
           "judged_pairs": pairs, "all_rows": rows,
           "CPU_time_saving_divided_by_historical_arithmetic_gap": enc/(1-floor_ntt/baseline_ntt),
           "arithmetic_gap_fraction_E0": count["encoder"]["16"]["gap_fraction_versus_historical_floor"],
           "conditional_commit_ps_interval_E0": count["encoder"]["16"]["conditional_commit_ps_interval"],
           "scope": "CPU realization and conditional E0 calibration only; no GPU timing or mu claim"}
    (r / "summary.json").write_text(json.dumps(out, indent=2)+"\n")
    print(json.dumps({k:v for k,v in out.items() if k != "all_rows"}, indent=2))


if __name__ == "__main__":
    main()
