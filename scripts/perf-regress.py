#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0 OR MIT
"""Compare a run's per-stage profile with a recorded one (#744).

    scripts/perf-regress.py RECORD.json RUN.json [--threshold 0.05] [--min-seconds 1]
    scripts/perf-regress.py --self-test

Both files are `tessera.run-manifest/1` (05 §6) from `tessera prove`. The profile is every
phase's, pass's and stage's time, each pass's device bytes and memory peak, the host memory
peak and the proof's parts (05 §4). Prints one line per entry and exits 1 when any grows by more
than the threshold. Times under --min-seconds in both runs are too short to compare and only
printed.
"""

import argparse
import json
import sys

# (metric, label): the profile's entries, keyed by the label's value.
TIMES = [("tessera_phase_seconds", "phase"), ("tessera_pass_seconds", "pass"),
         ("tessera_stage_seconds", "stage")]
SIZES = [("tessera_pass_device_bytes", "pass"), ("tessera_pass_device_memory_peak_bytes", "pass"),
         ("tessera_device_memory_peak_bytes", "device"), ("tessera_host_memory_peak_bytes", None),
         ("tessera_proof_bytes", "part")]


def profile(manifest: dict) -> dict[str, tuple[float, bool]]:
    """`{entry: (value, is_time)}`: a histogram's sum, or a counter's or gauge's value."""
    if manifest.get("schema") != "tessera.run-manifest/1":
        sys.exit("perf-regress: not a tessera.run-manifest/1")
    out = {}
    for sample in manifest.get("metrics", []):
        for metric, label in TIMES + SIZES:
            if sample["name"] != metric:
                continue
            key = metric if label is None else f"{metric}{{{sample['labels'].get(label)}}}"
            value = sample["sum"] if sample["type"] == "histogram" else sample["value"]
            out[key] = (float(value), (metric, label) in TIMES)
    return out


def compare(record: dict, run: dict, threshold: float, min_seconds: float) -> tuple[list, list]:
    """Report lines and regressed entries."""
    old, new = profile(record), profile(run)
    lines, regressed = [], []
    for key in sorted(old.keys() | new.keys()):
        (a, time_a), (b, time_b) = old.get(key, (0.0, False)), new.get(key, (0.0, False))
        time = time_a or time_b
        if key not in old or key not in new:
            lines.append(f"  {key}: {'new' if key not in old else 'gone'} ({a:g} -> {b:g})")
            continue
        change = (b - a) / a if a else (0.0 if b == 0 else float("inf"))
        short = time and max(a, b) < min_seconds
        flag = "" if short or change <= threshold else "  REGRESSION"
        lines.append(f"  {key}: {a:g} -> {b:g} ({change:+.1%}){' (short)' if short else ''}{flag}")
        if flag:
            regressed.append(key)
    return lines, regressed


def self_test() -> None:
    """Check the threshold, the short-time floor and that sizes have no floor."""
    def manifest(p2: float, p3: float, peak: float) -> dict:
        def h(name: str, labels: dict, total: float) -> dict:
            return {"name": name, "labels": labels, "type": "histogram", "sum": total}

        return {"schema": "tessera.run-manifest/1", "metrics": [
            h("tessera_pass_seconds", {"pass": "p2"}, p2),
            h("tessera_pass_seconds", {"pass": "p3"}, p3),
            {"name": "tessera_pass_device_memory_peak_bytes", "labels": {"pass": "p2"},
             "type": "gauge", "value": peak}]}
    base = manifest(100.0, 0.5, 1e9)
    assert compare(base, manifest(104.9, 0.9, 1e9), 0.05, 1.0)[1] == []
    assert compare(base, manifest(105.1, 0.5, 1e9), 0.05, 1.0)[1] == ["tessera_pass_seconds{p2}"]
    # A short time may double; memory has no floor.
    assert compare(base, manifest(90.0, 0.9, 1.1e9), 0.05, 1.0)[1] == [
        "tessera_pass_device_memory_peak_bytes{p2}"]
    print("perf-regress: self-test passed")


def main() -> None:
    """Compare two manifests, or run the self-test."""
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("record", nargs="?")
    p.add_argument("run", nargs="?")
    p.add_argument("--threshold", type=float, default=0.05)
    p.add_argument("--min-seconds", type=float, default=1.0)
    p.add_argument("--self-test", action="store_true")
    a = p.parse_args()
    if a.self_test:
        return self_test()
    if not (a.record and a.run):
        p.error("RECORD and RUN are required")
    with open(a.record, encoding="utf-8") as f, open(a.run, encoding="utf-8") as g:
        lines, regressed = compare(json.load(f), json.load(g), a.threshold, a.min_seconds)
    print("\n".join(lines))
    if regressed:
        print(f"perf-regress: {len(regressed)} regressed by more than {a.threshold:.0%}: "
              + ", ".join(regressed))
        sys.exit(1)
    print("perf-regress: no regression")


if __name__ == "__main__":
    main()
