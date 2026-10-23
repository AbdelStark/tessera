# SPDX-License-Identifier: Apache-2.0 OR MIT
"""05 §2: `tessera-quant`'s JSON lines are `tessera.log-event/1` objects; readable lines are
unchanged; events take snake-case names and typed public fields only."""

import io
import json
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path
from typing import Any, cast

import pytest

from tessera_quant import cli, log

SCHEMA = Path(__file__).resolve().parents[2] / "spec" / "schemas" / "log-event-v1.schema.json"


def _errors(schema: Any, root: Any, value: Any, at: str = "") -> list[str]:
    """The JSON Schema subset of `spec/schemas/README.md`, enough for this schema."""
    if schema is True:
        return []
    out: list[str] = []
    if "$ref" in schema:
        out += _errors(root["$defs"][schema["$ref"].removeprefix("#/$defs/")], root, value, at)
    kinds = {
        "object": dict,
        "array": list,
        "string": str,
        "boolean": bool,
        "integer": int,
    }
    if "type" in schema:
        names = schema["type"] if isinstance(schema["type"], list) else [schema["type"]]
        ok = any(
            isinstance(value, kinds[n]) and not (n == "integer" and isinstance(value, bool))
            for n in names
        )
        if not ok:
            return [f"{at}: expected {names}"]
    if "enum" in schema and value not in schema["enum"]:
        out.append(f"{at}: not one of {schema['enum']}")
    if isinstance(value, str) and not (
        schema.get("minLength", 0) <= len(value) <= schema.get("maxLength", 1 << 30)
    ):
        out.append(f"{at}: length {len(value)}")
    if isinstance(value, list) and "items" in schema:
        for i, item in enumerate(cast(list[Any], value)):
            out += _errors(schema["items"], root, item, f"{at}/{i}")
    if isinstance(value, dict):
        members = cast(dict[str, Any], value)
        out += [f"{at}: missing {k}" for k in schema.get("required", []) if k not in members]
        for key, item in members.items():
            if key in schema.get("properties", {}):
                out += _errors(schema["properties"][key], root, item, f"{at}/{key}")
            elif "additionalProperties" in schema:
                extra = schema["additionalProperties"]
                if extra is False:
                    out.append(f"{at}: unexpected {key}")
                else:
                    out += _errors(extra, root, item, f"{at}/{key}")
    return out


def _run(arguments: list[str]) -> tuple[int, str]:
    err = io.StringIO()
    with redirect_stdout(io.StringIO()), redirect_stderr(err):
        status = cli.main(arguments)
    log.configure()
    return status, err.getvalue()


def test_json_lines_validate_against_the_schema() -> None:
    schema = json.loads(SCHEMA.read_text())
    status, stderr = _run(
        [
            "reference",
            "--model",
            "/nonexistent/model.d",
            "--tokens",
            "t.ttk",
            "--out",
            "o.tlg",
            "--log-format",
            "json",
        ]
    )
    assert status == 20
    lines = [json.loads(line) for line in stderr.splitlines()]
    assert lines, stderr
    for line in lines:
        assert _errors(schema, schema, line) == [], line
    failed = lines[-1]
    assert failed["event"] == "tessera.quant.command_failed"
    assert failed["level"] == "error"
    assert failed["target"] == "tessera_quant.cli"
    assert failed["fields"]["code"] == "ERR-RES-IO"
    assert failed["fields"]["message"].startswith("tessera-quant: ERR-RES-IO: ")
    # A usage error, before parsing finishes, is logged in the format asked for too.
    status, stderr = _run(["compile", "--log-format", "json"])
    assert status == 64
    line = json.loads(stderr.splitlines()[-1])
    assert _errors(schema, schema, line) == []
    assert line["fields"]["code"] == "ERR-USAGE"


def test_readable_lines_are_unchanged() -> None:
    status, stderr = _run(
        ["reference", "--model", "/nonexistent/model.d", "--tokens", "t.ttk", "--out", "o.tlg"]
    )
    assert status == 20
    assert stderr.startswith("tessera-quant: ERR-RES-IO: ")
    assert stderr.count("\n") == 1


def test_templates_and_fields() -> None:
    assert log.render("wrote {n} rows to {dir}", {"n": 3, "dir": "out"}) == "wrote 3 rows to out"
    assert log.render("{missing} { {ok}", {"ok": True}) == "{missing} { true"
    with pytest.raises(ValueError, match="snake case"):
        log.event("info", "NotSnake", "m")
    for bad in [1.5, b"bytes", [1], None]:
        with pytest.raises(TypeError, match="typed kind"):
            log.event("info", "progress", "m", value=bad)
