#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0 OR MIT
"""Scratch saved-record mutations against the independent EXP-0017 checker."""

from copy import deepcopy
import importlib.util
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location("normcheck", HERE / "check.py")
check = importlib.util.module_from_spec(spec)
spec.loader.exec_module(check)
audit = json.loads((HERE / "results/audit.json").read_text())
compiler = json.loads((HERE / "results/compiler.json").read_text())


def change_extremum(a, c):
    a["summaries"][3]["ordinary"]["hi"] += 1


def change_width(a, c):
    a["summaries"][3]["ordinary"]["width"] -= 1


def change_multiplicity(a, c):
    c["cases"][3]["norms"][1]["candidate"]["commits_per_element"] = 1


def change_rows(a, c):
    c["cases"][3]["candidate_tables"][-1]["rows"] += 1


def change_verdict(a, c):
    a["verdict"] = "validated"


def change_lookup(a, c):
    a["requant_keys"] -= 1


records = []
for name, mutation in [("no-op", lambda a,c: None), ("extremum",change_extremum),
                       ("width",change_width),("split multiplicity",change_multiplicity),
                       ("unique table rows",change_rows),("verdict",change_verdict),
                       ("requant coverage",change_lookup)]:
    a,c = deepcopy(audit),deepcopy(compiler)
    mutation(a,c)
    try:
        check.verify(a,c)
    except AssertionError as exc:
        accepted = False
        error = str(exc)
    else:
        accepted = True
        error = None
    assert accepted == (name == "no-op"), name
    records.append(dict(name=name,accepted=accepted,expected_accept=name=="no-op",error=error))
result = dict(no_op_passed=True,mutations_caught=len(records)-1,mutations_total=len(records)-1,records=records,
              scope="saved-record integrity only; not protocol/compiler fault injection")
(HERE / "results/mutations.json").write_text(json.dumps(result,indent=2)+"\n")
print(json.dumps(result,indent=2))
