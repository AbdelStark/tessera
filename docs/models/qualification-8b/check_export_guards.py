# SPDX-License-Identifier: Apache-2.0 OR MIT
"""Check isolated #633 receipt faults and runtime mutants against a valid baseline.

Uses existing private calibration files, but never imports a checkpoint or writes D.
This is a qualification-script check, not a proof-verifier test.
"""

import argparse
import copy
import importlib.util
import json
import os
from pathlib import Path
from types import SimpleNamespace

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--calibration", type=Path, required=True)
parser.add_argument("--preflight", type=Path, required=True)
parser.add_argument("--output", type=Path, required=True)
args = parser.parse_args()
os.umask(0o077)
args.output.mkdir(mode=0o700, exist_ok=False)
runner = Path(__file__).with_name("export_fixed.py")
spec = importlib.util.spec_from_file_location("fixed_export", runner)
assert spec is not None and spec.loader is not None
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)
base = json.loads(args.preflight.read_text())
cases = []
x = copy.deepcopy(base)
x["source_sha256"] = {}
cases.append(("changed-source", "preflight source differs", x))
x = copy.deepcopy(base)
x["results"].pop()
cases.append(("incomplete", "complete preflight required", x))
x = copy.deepcopy(base)
x["results"][3] = copy.deepcopy(x["results"][0])
cases.append(("duplicate-profile-mode", "preflight profile/mode coverage differs", x))
x = copy.deepcopy(base)
x["results"][3]["attempts"][3]["alpha"] = "1/2"
cases.append(("duplicate-strength", "preflight strength coverage differs", x))
x = copy.deepcopy(base)
x["results"][3]["admissible"] = []
cases.append(("inconsistent-outcomes", "preflight outcomes differ", x))
x = copy.deepcopy(base)
x["calibration_report_sha256"] = "0" * 64
cases.append(("wrong-report-digest", "calibration report differs", x))
original = m.require


def no_model(*_args, **_kwargs):
    raise RuntimeError("test boundary: no model construction")


m.import_checkpoint = no_model


def attempt(name, receipt, skip=None):
    errors = []

    def checked(condition, message):
        if not condition:
            if message == skip:
                return
            errors.append(message)
        original(condition, message)

    m.require = checked
    path = args.output / (name + ".json")
    path.write_text(json.dumps(receipt))
    out = args.output / name
    try:
        m.run(
            SimpleNamespace(
                output=out,
                preflight=path,
                calibration=args.calibration,
                checkpoint=Path("unused"),
                c4=Path("unused"),
            )
        )
    except SystemExit as error:
        assert error.code == 1
    else:
        raise AssertionError("unexpected completion")
    status = json.loads((out / "status.json").read_text())
    assert not (out / "candidate-alpha-1of2.d").exists()
    return errors, status["error_type"]


# An invalid baseline would let an unrelated later refusal conceal a missing guard.
# Require exact failed rules, and see each assertion fail when its guard is disabled.
errors, error_type = attempt("positive-input-control", base)
assert errors == [] and error_type == "RuntimeError"
rows = []
for name, expected, receipt in cases:
    errors, error_type = attempt(name, receipt)
    assert errors == [expected] and error_type == "ValueError"
    errors, error_type = attempt(name + "-guard-disabled", receipt, skip=expected)
    assert errors != [expected], "mutation survived"
    rows.append(
        {
            "case": name,
            "exact_failed_rule": expected,
            "guard_disabled_is_detected": True,
        }
    )
result = {
    "scope": "fixed-export public qualification receipt guards; not proof verifier tests",
    "valid_input_reaches_model_boundary": True,
    "isolated_corruptions_rejected": 6,
    "runtime_guard_mutants_caught": 6,
    "no_model_constructed_or_artifact_written": True,
    "cases": rows,
    "runner_sha256": m.sha256_file(runner),
    "test_runner_sha256": m.sha256_file(Path(__file__)),
}
(args.output / "export-guard-checks.json").write_text(
    json.dumps(result, indent=2) + "\n"
)
print(json.dumps(result))
