#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0 OR MIT
"""Independent arithmetic audit of paired raw samples and frozen verdict rule."""
import json
import math
import statistics
from pathlib import Path

ROOT = Path(__file__).resolve().parent
ORDERS = [(0, 1, 2), (1, 2, 0), (2, 0, 1), (2, 1, 0), (1, 0, 2),
          (0, 2, 1), (0, 1, 2), (1, 2, 0), (2, 0, 1), (2, 1, 0)]


def stats(values):
    return {"median": statistics.median(values), "minimum": min(values),
            "maximum": max(values), "mean": statistics.mean(values),
            "sample_sd": statistics.stdev(values),
            "cv": statistics.stdev(values) / abs(statistics.mean(values))}


def main():
    raw = json.loads((ROOT / "results/run.json").read_text())
    journal = [json.loads(line) for line in (ROOT / "results/run.json.samples.ndjson").read_text().splitlines()]
    rows = []
    journal_count = 0
    assert all(a["end_unix_ns"] <= b["start_unix_ns"] for a, b in zip(journal, journal[1:]))
    for row in raw["rows"]:
        assert len(row["warmups"]) == 9
        assert [(w["arm"], w["warmup"]) for w in row["warmups"]] == [
            (arm, warmup) for arm in range(3) for warmup in range(3)]
        fixed_reps = [max(1, (60_000_000 + row["warmups"][3 * arm + 2]["elapsed_ns"] - 1)
                          // row["warmups"][3 * arm + 2]["elapsed_ns"]) for arm in range(3)]
        audited_attempts = []
        assert len(row["attempts"]) in (1, 2)
        for attempt in row["attempts"]:
            arms = [[], [], []]
            marginal = []
            ratio = []
            elapsed_floor = True
            for i, block in enumerate(attempt["blocks"]):
                assert block["block"] == i
                assert tuple(s["arm"] for s in block["samples_in_order"]) == ORDERS[i]
                times = [None, None, None]
                for sample in block["samples_in_order"]:
                    assert sample["reps"] == fixed_reps[sample["arm"]]
                    value = sample["elapsed_ns"] / 1e9 / sample["reps"]
                    assert math.isclose(value, sample["seconds_per_execution"], rel_tol=1e-13)
                    assert sample["end_unix_ns"] >= sample["start_unix_ns"]
                    elapsed_floor &= sample["elapsed_ns"] >= 20_000_000
                    times[sample["arm"]] = value
                    arms[sample["arm"]].append(value)
                    match = [j for j in journal if j["case"] == row["case"] and j["shape"] == row["shape"]
                             and j["attempt"] == attempt["attempt"] and j["block"] == i and j["arm"] == sample["arm"]]
                    assert len(match) == 1 and all(match[0][key] == sample[key] for key in
                        ("elapsed_ns", "reps", "start_unix_ns", "end_unix_ns"))
                    journal_count += 1
                m = times[2] - times[1]
                r = m / times[0]
                assert math.isclose(m, block["marginal"], rel_tol=1e-13)
                assert math.isclose(r, block["ratio"], rel_tol=1e-13)
                marginal.append(m)
                ratio.append(r)
            assert len(marginal) == 10
            arm_stats = [stats(arm) for arm in arms]
            marginal_stats = stats(marginal)
            ratio_stats = stats(ratio)
            noise_ok = elapsed_floor and marginal_stats["cv"] <= .03 and all(s["cv"] <= .03 for s in arm_stats)
            positive = all(m > 0 for m in marginal)
            verdict = "inconclusive"
            if noise_ok and positive:
                if max(ratio) < .90:
                    verdict = "validated"
                elif min(ratio) >= .90:
                    verdict = "refuted"
            assert verdict == attempt["verdict"] and noise_ok == attempt["noise_passed"]
            mean_f, mean_b = statistics.mean(arms[1]), statistics.mean(arms[2])
            covariance = sum((f - mean_f) * (b - mean_b) for f, b in zip(arms[1], arms[2])) / 9
            var_m = statistics.variance(marginal)
            identity = statistics.variance(arms[2]) + statistics.variance(arms[1]) - 2 * covariance
            assert math.isclose(var_m, identity, rel_tol=1e-10, abs_tol=1e-20)
            denom = statistics.stdev(arms[1]) * statistics.stdev(arms[2])
            audited_attempts.append({
                "attempt": attempt["attempt"], "arm_stats_seconds": arm_stats,
                "marginal_stats_seconds": marginal_stats, "ratio_stats": ratio_stats,
                "duration_floor_passed": elapsed_floor, "all_marginals_positive": positive,
                "noise_passed": noise_ok, "verdict": verdict,
                "paired_F_B_covariance_seconds_squared": covariance,
                "paired_F_B_correlation": covariance / denom if denom else None,
                "variance_identity_residual": var_m - identity,
                "subtraction_scale_amplification": (abs(mean_b) + abs(mean_f)) / abs(statistics.mean(marginal)),
                "arm_last3_over_first3_minus1": [statistics.mean(a[-3:]) / statistics.mean(a[:3]) - 1 for a in arms],
            })
        assert len(audited_attempts) == (1 if audited_attempts[0]["noise_passed"] else 2)
        rows.append({"case": row["case"], "shape": row["shape"], "judged": row["judged"],
                     "attempts": audited_attempts, "verdict": audited_attempts[-1]["verdict"],
                     "warmup_records_audited": 9, "warmups_per_arm_before_first_series": 3,
                     "fresh_warmups_before_registered_rerun": False, "fixed_reps": fixed_reps})
    assert journal_count == len(journal)
    verdicts = {}
    for case in ("production", "proxy"):
        judged = [row for row in rows if row["case"] == case and row["judged"]]
        assert len(judged) == 3
        words = [row["verdict"] for row in judged]
        verdicts[case] = words[0] if len(set(words)) == 1 else "inconclusive"
    result = {"evidence": "E1 CPU", "primary_verdict": verdicts["production"],
              "proxy_verdict": verdicts["proxy"], "empirical_band_is_not_confidence_interval": True,
              "raw_journal_samples_audited": journal_count, "rows": rows,
              "warmup_records_audited": 9 * len(raw["rows"]),
              "warmup_method": "Three per arm once per comparison; immediate full ten-block rerun without fresh warmups; same fixed repetitions and arm orders.",
              "gpu_evidence": False, "changes_model_leaf_unit_cost": False}
    (ROOT / "results/analysis.json").write_text(json.dumps(result, indent=2) + "\n")
    print(f"primary={verdicts['production']} proxy={verdicts['proxy']} audited_raw_samples={journal_count}")
    for row in rows:
        final = row["attempts"][-1]
        print(row["case"], row["shape"], "attempts", len(row["attempts"]), "ratio", final["ratio_stats"]["median"],
              "band", [final["ratio_stats"]["minimum"], final["ratio_stats"]["maximum"]],
              "arm CV", [s["cv"] for s in final["arm_stats_seconds"]],
              "M CV", final["marginal_stats_seconds"]["cv"], row["verdict"])


if __name__ == "__main__":
    main()
