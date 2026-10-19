#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0 OR MIT
"""Run the lemma self-check with planted faults in isolated scratch files."""

import json
from pathlib import Path
import subprocess
import sys
import tempfile

here = Path(__file__).resolve().parent
text = (here / "check.py").read_text()
scratch = Path(tempfile.mkdtemp(prefix="tessera-exp15-mutants-"))
mutations = [
    ("control", None, None),
    ("allow_zero_count", "range(psi, psi + 4)", "range(0, psi + 4)"),
    ("drop_output_range", "self.o_limbs.get((output + 32) % P, ())", "((0, 0, 0),)"),
    ("masked_lookup_at_zero_key", "else 31 for x, live", "else 0 for x, live"),
    ("widen_top_limb", "top_bits = bits - 2 * (count - 1)", "top_bits = bits - 2 * (count - 1) + 1"),
]
records = []
for name, before, after in mutations:
    if before is not None:
        assert text.count(before) == 1, name
    path = scratch / (name + ".py")
    path.write_text(text if before is None else text.replace(before, after))
    result = subprocess.run([sys.executable, "-I", str(path), "--self-test"],
                            text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    assert (result.returncode == 0) == (name == "control"), (name, result.stdout)
    records.append({"name": name, "exit": result.returncode, "output": result.stdout})
report = {"control_passes": True, "mutants_caught": 4, "records": records}
(here / "results/mutations.json").write_text(json.dumps(report, indent=2) + "\n")
print("Control passes; 4/4 planted faults caught.")
