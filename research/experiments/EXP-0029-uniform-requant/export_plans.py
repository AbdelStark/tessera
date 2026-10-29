#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0 OR MIT
"""Export diagnostic Wide plans only; never a production claim list or proof."""

import importlib.util
import json
import sys
from pathlib import Path

root = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location(
    "uniform_requant_experiment", root / "run.py"
)
module = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = module
spec.loader.exec_module(module)
rows = json.loads((root / "real-1b-public-requants.json").read_text())["rows"]
lines = []
for name, limits in (("tiny", (32,)), ("real", (30, 31, 32))):
    for limit in limits:
        if name == "tiny":
            profiles = [
                module.Profile(39, (r,), -524288, 524287, limit) for r in (2, 12)
            ]
        else:
            profiles = [
                module.Profile(x["b"], tuple(x["shifts"]), x["lo"], x["hi"], limit)
                for x in rows
                if x["position"] == 56
            ]
        plans = module.promote(profiles)
        assert plans is not None
        for layer, (profile, plan) in enumerate(zip(profiles, plans)):
            assert module.valid(profile, plan) and plan.kind == "Wide"
            lines.append(f"{name}\t{limit}\t{layer}\t{plan.b}\t{plan.s}\t{plan.t}\n")
(root / "results/layout-plans.tsv").write_text("".join(lines))
print(f"{len(lines)} diagnostic Wide plans exported")
