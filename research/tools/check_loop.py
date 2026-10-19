#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0 OR MIT
"""Exercise runner failure and overlap handling without calling an API."""

import os
from pathlib import Path
import shutil
import subprocess
import tempfile
import time

with tempfile.TemporaryDirectory(prefix="tessera-loop-check-") as scratch:
    root = Path(scratch)
    script = root / "research/tools/loop.sh"
    script.parent.mkdir(parents=True)
    shutil.copyfile(Path(__file__).with_name("loop.sh"), script)
    fake = root / "bin/claude"
    fake.parent.mkdir()
    fake.write_text('#!/bin/sh\necho stub-run\nif [ -n "${WAIT_FILE:-}" ]; then\n touch "$WAIT_FILE"\n while [ -f "$WAIT_FILE" ]; do sleep 0.05; done\nfi\nexit "${STUB_STATUS:-0}"\n')
    fake.chmod(0o755)
    for command in ("python3", "cargo"):
        gate = fake.with_name(command)
        gate.write_text('#!/bin/sh\necho gate-stub\nexit "${GATE_STATUS:-0}"\n')
        gate.chmod(0o755)
    env = {**os.environ, "PATH": f"{fake.parent}:{os.environ['PATH']}", "LOOP_LOG_DIR": str(root / "logs")}
    failed = subprocess.run(["sh", str(script)], env={**env, "STUB_STATUS": "17"}, capture_output=True, text=True)
    assert failed.returncode == 17, failed
    assert "done," not in failed.stdout and "stub-run" in failed.stdout
    assert not (root / ".autoresearch.lock").exists()
    bad_gate = subprocess.run(["sh", str(script)], env={**env, "GATE_STATUS": "23"}, capture_output=True, text=True)
    assert bad_gate.returncode == 23 and "done," not in bad_gate.stdout, bad_gate
    marker = root / "waiting"
    first = subprocess.Popen(["sh", str(script)], env={**env, "WAIT_FILE": str(marker)}, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    try:
        for _ in range(100):
            if marker.exists():
                break
            time.sleep(0.01)
        assert marker.exists(), "stub did not start"
        second = subprocess.run(["sh", str(script)], env={**env, "LOOP_LOG_DIR": str(root / "other-logs")}, capture_output=True)
        assert second.returncode == 75, second
    finally:
        marker.unlink(missing_ok=True)
        first.communicate(timeout=10)
    assert first.returncode == 0
    assert not (root / ".autoresearch.lock").exists()
print("ok: CLI/gate failures propagate, overlap rejected across log dirs, lock released; no API used")
