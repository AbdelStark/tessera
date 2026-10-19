# SPDX-License-Identifier: Apache-2.0 OR MIT
"""CI-only constraint tree framing, independent Python oracle (RFC-0007)."""
import importlib.util
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location("reference_hash", HERE.parent / "hash/generate.py")
reference = importlib.util.module_from_spec(spec)
spec.loader.exec_module(reference)
CONTEXT = "tessera/ci/constraint-system-digest/1"


def frame(tree):
    records = []

    def visit(path, value):
        if value is None:
            tag, payload = "null", ""
        elif isinstance(value, bool):
            tag, payload = "bool", str(int(value))
        elif isinstance(value, int):
            assert -(2**63) <= value < 2**64
            tag, payload = "integer", str(value)
        elif isinstance(value, str):
            tag, payload = "string", value
        elif isinstance(value, list):
            tag, payload = "array", str(len(value))
        else:
            assert isinstance(value, dict)
            tag, payload = "object", str(len(value))
        records.append((path, tag, payload))
        if isinstance(value, list):
            for index, child in enumerate(value):
                visit(f"{path}/{index}", child)
        if isinstance(value, dict):
            for key in sorted(value):
                escaped = key.replace("~", "~0").replace("/", "~1")
                visit(f"{path}/{escaped}", value[key])

    visit("", tree)
    return b"".join(len(s.encode()).to_bytes(8, "little") + s.encode()
                    for row in records for s in row)


def main():
    trees = [None, False, True, 0, -2**63, 2**64-1, "", [], {},
             ["2", 2, None], {"a/b": {"~": True}, "é": "🦀"},
             {"coefficients": [str(-(2**127)), str(2**127-1)],
              "nodes": [{"inputs": [0, 1], "op": "Mul"}]}]
    cases = []
    for tree in trees:
        data = frame(tree)
        cases.append({"tree": tree, "frame": data.hex(),
                      "digest": reference.derive_key(CONTEXT, data).hex()})
    (HERE / "framing.json").write_text(json.dumps({"version": 1, "context": CONTEXT,
                                                  "cases": cases}, indent=2, ensure_ascii=False)+"\n")


if __name__ == "__main__":
    main()
