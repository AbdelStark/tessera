# SPDX-License-Identifier: Apache-2.0 OR MIT
"""Tessera adapter on CPU: the image rules (RFC-0014 B1 to B4), `info`'s build record (B5, B6), the
verdict map of `tessera-verify`'s exit codes (RFC-0005 §Verdict normalisation, INV-ADAPTER-9), the
recorded commitment reaching the verifier (INV-ADAPTER-5) and the forbidden-flag guard.

The upstream tools are stand-ins (`stub.py`): `statement_pin` prints a statement digest and the
weights root, and `tessera-verify` exits with the code the test chooses, as `tessera-verify`'s
exit codes at the pin define them (docs/spec/02-public-api.md §2.4: 0 accept, 10 reject,
64 usage). None of their output is captured from an upstream run."""

import importlib.machinery
import importlib.util
import json
import stat
from pathlib import Path
from types import ModuleType
from typing import Any

import pytest

from iproofs.provenance import dockerfile_problems
from iproofs.schema import SchemaRegistry

HERE = Path(__file__).resolve().parent
SYSTEM = HERE.parent
REGISTRY = SchemaRegistry.from_tree(SYSTEM.parents[1] / "schema")
COMMIT = "3c75bdf78e4cc5c9ed2140900ff109d5933e3006"
ROOT = "ab" * 32
STATEMENT = "cd" * 32

STUB = """#!/usr/bin/env python3
import json, os, sys
log = os.environ["TESSERA_STUB_LOG"]
with open(log, "a") as f:
    f.write(json.dumps(sys.argv) + "\\n")
if sys.argv[0].endswith("statement_pin"):
    print("{statement} {root}")
    sys.exit(0)
print('{{"verdict": "stub"}}')
sys.exit(int(os.environ["TESSERA_STUB_EXIT"]))
"""


@pytest.fixture
def adapter(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> ModuleType:
    io = tmp_path / "io"
    io.mkdir()
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    for name in ("statement_pin", "tessera-verify", "tessera"):
        tool = bin_dir / name
        tool.write_text(STUB.format(statement=STATEMENT, root=ROOT))
        tool.chmod(tool.stat().st_mode | stat.S_IXUSR)
    monkeypatch.setenv("IPROOFS_IO", str(io))
    monkeypatch.setenv("IPROOFS_INPUTS", str(HERE / "fixtures/inputs.json"))
    monkeypatch.setenv("TESSERA_REPO", str(tmp_path))
    monkeypatch.setenv("TESSERA_PIN", str(bin_dir / "statement_pin"))
    monkeypatch.setenv("TESSERA_VERIFY", str(bin_dir / "tessera-verify"))
    monkeypatch.setenv("TESSERA_BIN", str(bin_dir / "tessera"))
    monkeypatch.setenv("TESSERA_STUB_LOG", str(tmp_path / "calls.jsonl"))
    loader = importlib.machinery.SourceFileLoader("tessera_adapter", str(SYSTEM / "entrypoint"))
    spec = importlib.util.spec_from_loader("tessera_adapter", loader)
    assert spec is not None
    module = importlib.util.module_from_spec(spec)
    loader.exec_module(module)
    return module


def run(adapter: ModuleType, sub: str, **fields: Any) -> dict[str, Any]:
    io = adapter.host("/io")
    job = {"schema": "adapter-job/1", "contract": "adapter/1", "subcommand": sub, "run_id": "r",
           "workload": {"id": "l31-8b-c1024", "version": 1}, **fields}  # fmt: skip
    (io / "job.json").write_text(json.dumps(job))
    adapter.sys.argv = ["entrypoint", sub, "/io/job.json", "/io/out.json"]
    assert adapter.main() == 0
    out: dict[str, Any] = json.loads((io / "out.json").read_text())
    assert REGISTRY.errors("adapter-out/1", out) == []
    return out


def calls(tmp_path: Path) -> list[list[str]]:
    log = tmp_path / "calls.jsonl"
    return [json.loads(line) for line in log.read_text().splitlines()] if log.exists() else []


def test_dockerfile_rules() -> None:
    lock = (SYSTEM / "inputs.lock").read_text()
    assert dockerfile_problems((SYSTEM / "Dockerfile").read_text(), lock) == []


def test_upstream_pinned_and_checked() -> None:
    """INV-ADAPTER-2: the fetch stage adds the pin and checks it; B4: nothing edits the source."""
    dockerfile = (SYSTEM / "Dockerfile").read_text()
    assert f"tessera.git#{COMMIT}" in dockerfile
    assert f'test "$(git -C /opt/tessera rev-parse HEAD)" = {COMMIT}' in dockerfile
    build = dockerfile.split("\nFROM fetch", 1)[1]
    assert "patch" not in build
    assert "sed " not in build
    for card in ("system.yaml", "claim.yaml"):
        assert COMMIT in (SYSTEM / card).read_text()


def test_info_reports_the_build_record(
    adapter: ModuleType, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("IPROOFS_ADAPTER_COMMIT", "a" * 40)
    monkeypatch.setenv("IPROOFS_ADAPTER_DIRTY", "false")
    out = run(adapter, "info")
    inputs = json.loads((HERE / "fixtures/inputs.json").read_text())["inputs"]
    assert (out["adapter_commit"], out["adapter_dirty"], out["image_inputs"]) == (
        "a" * 40,
        False,
        inputs,
    )


def verify_job(tmp_path: Path, adapter: ModuleType) -> dict[str, Any]:
    io = adapter.host("/io")
    (io / "state").mkdir(exist_ok=True)
    for name in ("claims.tcl", "weights.twr"):
        (io / "state" / name).write_bytes(b"x")
    (io / "proofs/req-0").mkdir(parents=True)
    (io / "proofs/req-0/proof.tnp").write_bytes(b"x")
    return {"verifier_files": ["/io/state/claims.tcl", "/io/state/weights.twr"],
            "proof_dir": "/io/proofs/req-0", "request": [128000, 791, 4320],
            "claimed_output": [13, 578], "commitment_id": ROOT}  # fmt: skip


@pytest.mark.parametrize(("code", "verdict"), [(0, "accept"), (10, "reject"), (64, "invalid_call")])
def test_verdict_map(
    adapter: ModuleType, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, code: int, verdict: str
) -> None:
    monkeypatch.setenv("TESSERA_STUB_EXIT", str(code))
    out = run(adapter, "verify", **verify_job(tmp_path, adapter))
    assert (out["verdict"], out["raw_exit_code"]) == (verdict, code)


def test_unmapped_exit_stops_the_adapter(
    adapter: ModuleType, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Never guess a verdict: an exit code outside the table stops the adapter."""
    monkeypatch.setenv("TESSERA_STUB_EXIT", "3")
    job = verify_job(tmp_path, adapter)
    io = adapter.host("/io")
    (io / "job.json").write_text(json.dumps({"subcommand": "verify", **job}))
    adapter.sys.argv = ["entrypoint", "verify", "/io/job.json", "/io/out.json"]
    with pytest.raises(SystemExit, match="unmapped tessera-verify exit 3"):
        adapter.main()


def test_verify_uses_the_recorded_commitment_and_the_public_tokens(
    adapter: ModuleType, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """INV-ADAPTER-5: the weights root is the harness's commitment_id; --tokens pins the stream."""
    monkeypatch.setenv("TESSERA_STUB_EXIT", "0")
    run(adapter, "verify", **verify_job(tmp_path, adapter))
    pin, verify = calls(tmp_path)
    assert pin[-2:] == ["3", "2"]  # prompt and continuation lengths
    assert verify[verify.index("--weights-root") + 1] == ROOT
    assert verify[verify.index("--statement-digest") + 1] == STATEMENT
    assert "--tokens" in verify
    logged = adapter.host("/io/commands.jsonl").read_text().splitlines()
    assert len(logged) == 2
    assert all("env" not in json.loads(line) for line in logged)


def test_missing_proof_is_invalid_call(
    adapter: ModuleType, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("TESSERA_STUB_EXIT", "0")
    job = verify_job(tmp_path, adapter)
    (adapter.host("/io/proofs/req-0") / "proof.tnp").unlink()
    out = run(adapter, "verify", **job)
    assert out["verdict"] == "invalid_call"
    assert calls(tmp_path) == []


@pytest.mark.parametrize("flag", ["--test-rng", "--params=test-small"])
def test_forbidden_flag_refused_outside_a_control_job(adapter: ModuleType, flag: str) -> None:
    """RFC-0005 rule 4: a weakening flag reaches no upstream process outside a control job."""
    io = adapter.host("/io")
    job = {"subcommand": "prove", "control": False, "control_flags": [flag]}
    (io / "job.json").write_text(json.dumps(job))
    adapter.sys.argv = ["entrypoint", "prove", "/io/job.json", "/io/out.json"]
    with pytest.raises(SystemExit, match="outside a control job"):
        adapter.main()
    assert not (io / "commands.jsonl").exists()
