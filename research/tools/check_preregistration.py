#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0 OR MIT
"""Negative fixtures for the frozen-plan integrity check."""

import hashlib
import importlib.util
from pathlib import Path
import tempfile

spec = importlib.util.spec_from_file_location("check_kb", Path(__file__).with_name("check_kb.py"))
checker = importlib.util.module_from_spec(spec)
spec.loader.exec_module(checker)

with tempfile.TemporaryDirectory(prefix="tessera-prereg-check-") as scratch:
    results = Path(scratch)
    for frozen, marker in (("README.preregistered.txt", "preregistration.txt"),
                           ("README.amendment-preregistered.txt", "preregistration-amendment.txt"),
                           ("README.amendment-fraction.txt", "preregistration-amendment.txt")):
        snapshot, registration = results / frozen, results / marker
        snapshot.write_text("planned hypothesis and criterion\n")
        problems = []
        checker.check_frozen(results, problems)
        assert any("missing" in p for p in problems)
        registration.write_text(hashlib.sha256(snapshot.read_bytes()).hexdigest())
        problems = []
        checker.check_frozen(results, problems)
        assert not problems, problems
        snapshot.write_text("changed after observing results\n")
        problems = []
        checker.check_frozen(results, problems)
        assert any("digest mismatch" in p for p in problems)
        snapshot.unlink()
        registration.unlink()
print("ok: missing and altered preregistrations rejected; original and amendment formats pass")
