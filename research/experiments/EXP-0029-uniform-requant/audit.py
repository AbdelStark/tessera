#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0 OR MIT
"""Recount frozen inputs and check targeted refusals separately from the run loop."""

import hashlib
import importlib.util
import json
import subprocess
import sys
from pathlib import Path

root = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location(
    "uniform_requant_experiment", root / "run.py"
)
experiment = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = experiment
spec.loader.exec_module(experiment)

source = json.loads((root / "real-1b-public-requants.json").read_text())
claim_hash = hashlib.sha256((root / "real-1b-claims.tcl").read_bytes()).hexdigest()
assert claim_hash == source["source_claim_sha256"]
relative = (
    "research/experiments/EXP-0029-uniform-requant/results/README.preregistered.txt"
)
committed = subprocess.check_output(["git", "show", "e56dda7:" + relative])
assert committed == (root / "results/README.preregistered.txt").read_bytes()
result = json.loads((root / "results/result.json").read_text())
totals = result["totals"]
# Distinct class shifts: 63,62,61,60 valid starting shifts, plus three wide-spread sets.
singletons = 62 * (63 + 62 + 61 + 60) * 5 * 2 + 62 * 3 * 5 * 2
assert singletons == 154380 == totals["singleton_profiles"]
assert totals["sites"] == singletons + 512 + 5 + 3
assert totals["admitted"] + totals["refused"] == totals["sites"]
assert totals["unchanged"] <= totals["admitted"]
assert len(source["rows"]) == 161
assert len([r for r in source["rows"] if r["position"] == 56]) == 16
receipts = json.loads((root / "results/mutations.json").read_text())
assert receipts[0]["exit_code"] == receipts[-1]["exit_code"] == 0
assert receipts[0]["source_sha256"] == receipts[-1]["source_sha256"]
assert (
    receipts[0]["source_sha256"]
    == hashlib.sha256((root / "run.py").read_bytes()).hexdigest()
)
assert len(receipts[1:-1]) == 10 and all(r["exit_code"] != 0 for r in receipts[1:-1])

Profile = experiment.Profile
impossible = [
    Profile(38, (1,), -1048575, 1048575, 32),
    Profile(38, (63,), -1048575, 1048575, 32),
]
assert experiment.promote(impossible) is None
for limit in (0, 33):
    assert experiment.promote([Profile(20, (8,), -128, 127, limit)]) is None
assert experiment.promote([Profile(20, (8,), -128, -1, 32)]) is None
assert experiment.promote([Profile(20, (8,), -(1 << 32), 127, 32)]) is None

# Count all 16 actual layers; every B in the registration must have a complete plan.
for site in result["required_real_sites"]:
    assert site["limit"] in (30, 31, 32)
    assert len(site["profiles"]) == len(site["plans"]) == 16
    assert all(plan["kind"] == "Wide" for plan in site["plans"])

print(
    json.dumps(
        {
            "scope": "E0 audit; no proof or global security claim",
            "frozen_plan_sha256": hashlib.sha256(committed).hexdigest(),
            "source_claim_sha256": claim_hash,
            "singleton_recount": singletons,
            "planned_sites_recount": singletons + 512 + 5 + 3,
            "mutation_sources_restored": True,
            "impossible_common_plan": "refused",
        },
        indent=2,
    )
)
