#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0 OR MIT
"""Independently recompute the frozen timing/noise rules from all retained raw blocks."""
from pathlib import Path
import json
import math
import statistics
import sys


def close(a, b):
    assert math.isfinite(a) and math.isfinite(b)
    assert math.isclose(a, b, rel_tol=2e-12, abs_tol=1e-14), (a, b)


def statistics_of(values, recorded):
    expected = {"median": statistics.median(values), "min": min(values), "max": max(values),
                "mean": statistics.mean(values), "cv": statistics.stdev(values)/statistics.mean(values)}
    for key, value in expected.items():
        close(value, recorded[key])
    assert len(values) == 10
    for a, b in zip(values, recorded["values"], strict=True):
        close(a, b)
    return expected


def audit(data):
    assert data["experiment"] == "EXP-0019"
    assert data["protocol"] == {"warmups": 3, "samples": 10, "min_sample_ms": 20,
                                "cv_limit": 0.03, "max_reruns": 1, "exclusive_cpu_authorized": True}
    u = next(c for c in data["cohorts"] if c["config"] == "U23")
    denominators = {"S1": (16777216,8390656), "S2": (11184812,8390656), "S3": (16777216,10485760), "S4": (22369632,10490880), "U23": (8388608,8388608), "U24": (16777216,16777216), "S1-1K": (1048576,524800), "S2-1K": (699052,524800)}
    assert [c["config"] for c in data["cohorts"]] == list(denominators)
    assert data["sham"]["config"] == "S2"
    summaries = []
    last_end = data["start_unix_seconds"]
    for c in [data["sham"], *data["cohorts"]]:
        sham = c["sham"]
        assert (c["padded_leaves"], c["real_leaves"]) == denominators[c["config"]]
        required = sham or c["judged"]
        if not required:
            assert len(c["attempts"]) == 1, "secondary cohorts have no registered rerun"
        assert c["selected_attempt"] == len(c["attempts"]) in (1, 2)
        outcomes = []
        for number, attempt in enumerate(c["attempts"], 1):
            assert attempt["attempt"] == number
            warm = attempt["fresh_warmups"]
            blocks = attempt["measured"]
            assert len(warm) == 3 and len(blocks) == 10
            for index, b in enumerate([*warm, *blocks]):
                assert b["index"] == index
                order = ("AD" if index%2 == 0 else "DA") if sham else (["ABU", "BAU", "UBA", "UAB", "AUB", "BUA"][index%6] if c["judged"] else ("AB" if index%2 == 0 else "BA"))
                assert b["order"] == order
                assert "".join(s["arm"] for s in b["samples"]) == order
                for s in b["samples"]:
                    assert s["elapsed_seconds"] >= 0.020 and s["repetitions"] >= 1
                    assert s["start_unix_seconds"] >= last_end, "overlapping or reordered timing samples"
                    assert s["end_unix_seconds"] >= s["start_unix_seconds"]
                    assert s["elapsed_seconds"] <= s["end_unix_seconds"] - s["start_unix_seconds"] + 2e-6
                    last_end = s["end_unix_seconds"]
                    close(s["elapsed_seconds"]/s["repetitions"], s["seconds_per_proof"])
            recorded = attempt["analysis"]
            values = {arm: [next(s["seconds_per_proof"] for s in b["samples"] if s["arm"] == arm) for b in blocks]
                      for arm in blocks[0]["order"]}
            noise = True
            aggregate = {}
            for arm, vals in values.items():
                s = statistics_of(vals, recorded["arms"][arm])
                noise &= s["cv"] <= .03
                samples = [s for b in blocks for s in b["samples"] if s["arm"] == arm]
                elapsed = sum(s["elapsed_seconds"] for s in samples)
                repetitions = sum(s["repetitions"] for s in samples)
                aggregate[arm] = elapsed/repetitions
                close(aggregate[arm], recorded["aggregate"][arm]["seconds_per_proof"])
                denominator = u if arm == "U" else c
                close(aggregate[arm]*1e9/denominator["real_leaves"], recorded["aggregate"][arm]["ns_per_real"])
                close(aggregate[arm]*1e9/denominator["padded_leaves"], recorded["aggregate"][arm]["ns_per_padded"])
            rvalues = [b["samples"][0]["seconds_per_proof"]/b["samples"][1]["seconds_per_proof"] for b in blocks] if sham else [a/b for a, b in zip(values["A"], values["B"], strict=True)]
            r = statistics_of(rvalues, recorded["R"])
            noise &= r["cv"] <= .03
            rag = aggregate["A"]/aggregate["D" if sham else "B"]
            close(rag, recorded["R_agg"])
            factor = c["padded_leaves"]/c["real_leaves"]
            close(factor, c["factor"])
            close(r["median"]/factor, recorded["R_over_f"])
            close(rag/factor, recorded["R_agg_over_f"])
            if sham:
                noise &= .97 <= r["median"] <= 1.03
            if c["judged"]:
                hvalues = [(b/c["real_leaves"])/(v/u["real_leaves"]) for b, v in zip(values["B"], values["U"], strict=True)]
                h = statistics_of(hvalues, recorded["H"])
                noise &= h["cv"] <= .03
                hag = (aggregate["B"]/c["real_leaves"])/(aggregate["U"]/u["real_leaves"])
                close(hag, recorded["H_agg"])
            assert bool(noise) == recorded["noise_pass"]
            outcomes.append(bool(noise))
        if required:
            assert not outcomes[0] if len(outcomes) == 2 else outcomes[0]
        else:
            assert len(outcomes) == 1  # A noisy secondary remains reportable, never deciding.
        summaries.append({"config": c["config"], "sham": sham, "selected_attempt": len(outcomes), "noise_pass": outcomes[-1], "analysis": c["attempts"][-1]["analysis"]})
    assert last_end <= data["end_unix_seconds"]
    required = [s for s in summaries if s["sham"] or next(c for c in data["cohorts"] if c["config"] == s["config"])["judged"]]
    noise = all(s["noise_pass"] for s in required)
    primary = [c for c in data["cohorts"] if c["judged"]]
    assert len(primary) == 4 and [c["config"] for c in primary] == ["S1", "S2", "S3", "S4"]
    selected = [(c, c["attempts"][-1]["analysis"]) for c in primary]
    validated = all(.9 <= s["R_over_f"] <= 1.1 and .9 <= s["R_agg_over_f"] <= 1.1 and s["H"]["median"] <= 1.05 and s["H_agg"] <= 1.05 for c, s in selected)
    refuted = any(s["R"]["median"] <= 1 + (c["factor"]-1)/2 for c, s in selected)
    verdict = "inconclusive" if not noise else "validated" if validated else "refuted" if refuted else "inconclusive"
    assert verdict == data["verdict"] and noise == data["noise_pass"]
    return {"experiment": "EXP-0019", "independent_raw_audit": "pass", "verdict": verdict, "noise_pass": noise,
            "all_samples_ge_20ms": True, "all_reruns_fresh_3_warmups": True, "cohorts": summaries}


def main():
    path = Path(sys.argv[1] if len(sys.argv)>1 else "results/run.json")
    result = audit(json.loads(path.read_text()))
    Path("results/audit.json").write_text(json.dumps(result, indent=2) + "\n")
    print(result["independent_raw_audit"], result["verdict"], "noise_pass", result["noise_pass"])


if __name__ == "__main__":
    main()
