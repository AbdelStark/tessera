#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0 OR MIT
"""Consistency check of the research knowledge base (research/README.md rules).

`cargo xtask lint` checks Markdown links; this checks what links cannot: that every citation
key resolves to a bibliography entry, that every survey, idea and experiment on disk is in the
ledger and vice versa, that the ledger's counters are ahead of every identifier in use, that
status words come from the README's vocabulary and agree between a note and its ledger row, and
that every experiment report carries the pre-registration sections and names the idea it tests or the issue it
answers.

Usage:
    python3 -I research/tools/check_kb.py            # exit 1 on any failure
    python3 -I research/tools/check_kb.py --verbose  # also list unused bibliography keys
"""

from __future__ import annotations

import argparse
import hashlib
import re
import sys
from pathlib import Path

RESEARCH = Path(__file__).resolve().parent.parent

# A citation key: letters, digits and '+', ending in a two-digit year and an optional letter,
# in square brackets and not followed by '(' (which would make it a Markdown link).
CITATION = re.compile(r"\[([A-Za-z][A-Za-z0-9+]*?[0-9]{2}[a-z]?)\](?!\()")
SPECIAL_KEYS = {"E0-this", "Key2026"}  # a survey's own derivation; the README's placeholder
CODE_SPAN = re.compile(r"`[^`\n]*`")
FENCE = re.compile(r"```.*?```", re.S)
IDENTIFIER = re.compile(r"\b(SUR|IDEA|EXP)-(\d{4})\b")
BIB_ENTRY = re.compile(r"^- \[([^\]]+)\]")

IDEA_STATES = {"proposed", "triaged", "experimenting", "validated", "refuted", "inconclusive", "promoted", "shelved", "rejected"}
EXPERIMENT_STATES = {"planned", "running", "done: validated", "done: refuted", "done: inconclusive"}
EXPERIMENT_SECTIONS = ("## Hypothesis", "## Success criterion", "## Method", "## Results", "## Verdict")


def fail(problems: list[str], message: str) -> None:
    problems.append(message)


def status_token(line: str, states: set[str]) -> str | None:
    """The first vocabulary word inside backticks on a `| Status |` line."""
    for match in re.finditer(r"`([^`]+)`", line):
        token = match.group(1)
        for state in sorted(states, key=len, reverse=True):
            if token.startswith(state):
                return state
    return None


def check_frozen(results: Path, problems: list[str]) -> None:
    """Check frozen plans where present; older experiments retain their historical provenance."""
    for frozen, marker in (("README.preregistered.txt", "preregistration.txt"),
                           ("README.amendment-preregistered.txt", "preregistration-amendment.txt"),
                           ("README.amendment-fraction.txt", "preregistration-amendment.txt")):
        snapshot = results / frozen
        if not snapshot.exists():
            continue
        registration = results / marker
        if not registration.exists():
            fail(problems, f"{results.parent.name}: missing {marker} for frozen plan")
            continue
        digest = hashlib.sha256(snapshot.read_bytes()).hexdigest()
        if digest not in re.findall(r"\b[0-9a-f]{64}\b", registration.read_text()):
            fail(problems, f"{results.parent.name}: frozen pre-registration digest mismatch ({frozen})")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--verbose", action="store_true")
    args = ap.parse_args(argv)
    problems: list[str] = []

    md_files = sorted(p for p in RESEARCH.rglob("*.md") if "target" not in p.parts)
    bibliography = (RESEARCH / "bibliography.md").read_text(encoding="utf-8")
    defined = {m.group(1) for line in bibliography.splitlines() if (m := BIB_ENTRY.match(line))}

    # 1. Citations resolve.
    cited: dict[str, set[str]] = {}
    for path in md_files:
        if path.name == "bibliography.md":
            continue
        text = CODE_SPAN.sub("", FENCE.sub("", path.read_text(encoding="utf-8")))
        for key in CITATION.findall(text):
            cited.setdefault(key, set()).add(str(path.relative_to(RESEARCH)))
    for key, where in sorted(cited.items()):
        if key not in defined and key not in SPECIAL_KEYS:
            fail(problems, f"citation [{key}] has no bibliography entry (in {', '.join(sorted(where))})")
    if args.verbose:
        for key in sorted(defined - set(cited)):
            print(f"note: bibliography key [{key}] is never cited")

    # 2. Files on disk against the ledger, and counters.
    ledger = (RESEARCH / "LEDGER.md").read_text(encoding="utf-8")
    on_disk: dict[str, set[str]] = {"SUR": set(), "IDEA": set(), "EXP": set()}
    for path in (RESEARCH / "surveys").glob("SUR-*.md"):
        on_disk["SUR"].add(path.name[:8])
    for path in (RESEARCH / "ideas").glob("IDEA-*.md"):
        on_disk["IDEA"].add(path.name[:9])
    for path in (RESEARCH / "experiments").glob("EXP-*"):
        if path.is_dir():
            on_disk["EXP"].add(path.name[:8])
            if not (path / "README.md").exists():
                fail(problems, f"{path.name} has no README.md")

    def ledger_section(title: str) -> str:
        start = ledger.find(f"\n## {title}\n")
        if start < 0:
            fail(problems, f"ledger has no section '{title}'")
            return ""
        end = ledger.find("\n## ", start + 1)
        return ledger[start : end if end > 0 else len(ledger)]

    sections = {"SUR": ledger_section("Surveys"), "IDEA": ledger_section("Ideas"), "EXP": ledger_section("Experiments")}
    for kind, files in on_disk.items():
        rows = {f"{kind}-{num}" for k, num in IDENTIFIER.findall(sections[kind]) if k == kind}
        linked = {f"{kind}-{num}" for k, num in IDENTIFIER.findall("\n".join(line for line in sections[kind].splitlines() if line.startswith("| [")) ) if k == kind}
        for ident in sorted(files - linked):
            fail(problems, f"{ident} exists on disk but has no row in the ledger's table")
        for ident in sorted(linked - files):
            fail(problems, f"{ident} has a ledger row but no file on disk")
        del rows

    counters = dict(re.findall(r"^\| (Survey|Idea|Experiment) \| (\S+) \|$", ledger, flags=re.M))
    for label, kind in (("Survey", "SUR"), ("Idea", "IDEA"), ("Experiment", "EXP")):
        nxt = counters.get(label)
        if not nxt:
            fail(problems, f"ledger counters have no row for {label}")
            continue
        used = [int(i.split("-")[1]) for i in on_disk[kind]]
        if used and int(nxt.split("-")[1]) <= max(used):
            fail(problems, f"ledger counter {nxt} is not ahead of the highest identifier in use ({kind}-{max(used):04d})")

    # 3. Status words: vocabulary, and note against ledger row.
    def ledger_status(ident: str, section: str) -> str | None:
        for line in section.splitlines():
            if line.startswith(f"| [{ident}]"):
                cells = [c.strip() for c in line.strip().strip("|").split("|")]
                for cell in cells:
                    token = status_token(cell, IDEA_STATES | EXPERIMENT_STATES | {"current", "stale"})
                    if token and cell.startswith("`"):
                        return token
        return None

    for path in sorted((RESEARCH / "ideas").glob("IDEA-*.md")):
        text = path.read_text(encoding="utf-8")
        line = next((l for l in text.splitlines() if l.startswith("| Status |")), None)
        if line is None:
            fail(problems, f"{path.name}: no `| Status |` line")
            continue
        note_state = status_token(line, IDEA_STATES)
        if note_state is None:
            fail(problems, f"{path.name}: status line carries no vocabulary word: {line[:80]}")
            continue
        row_state = ledger_status(path.name[:9], sections["IDEA"])
        if row_state is not None and row_state != note_state:
            fail(problems, f"{path.name[:9]}: note says `{note_state}`, ledger row says `{row_state}`")

    for path in sorted((RESEARCH / "experiments").glob("EXP-*/README.md")):
        text = path.read_text(encoding="utf-8")
        check_frozen(path.parent / "results", problems)
        line = next((l for l in text.splitlines() if l.startswith("| Status |")), None)
        ident = path.parent.name[:8]
        if line is None:
            fail(problems, f"{ident}: no `| Status |` line")
        else:
            note_state = status_token(line, EXPERIMENT_STATES)
            if note_state is None:
                fail(problems, f"{ident}: status line carries no vocabulary word: {line[:80]}")
            else:
                row_state = ledger_status(ident, sections["EXP"])
                if row_state is not None and row_state != note_state:
                    fail(problems, f"{ident}: report says `{note_state}`, ledger row says `{row_state}`")
        for section in EXPERIMENT_SECTIONS:
            if section not in text:
                fail(problems, f"{ident}: report lacks the section '{section}'")
        if "| Pre-registered on |" not in text:
            fail(problems, f"{ident}: report has no pre-registration date")

    for path in sorted((RESEARCH / "surveys").glob("SUR-*.md")):
        text = path.read_text(encoding="utf-8")
        line = next((l for l in text.splitlines() if l.startswith("| Status |")), None)
        if line is None or status_token(line, {"current", "stale"}) is None:
            fail(problems, f"{path.name}: status line missing or outside the vocabulary")

    # 4. Every idea links its experiment and every experiment its idea.
    for path in sorted((RESEARCH / "experiments").glob("EXP-*/README.md")):
        text = path.read_text(encoding="utf-8")
        idea_line = next((l for l in text.splitlines() if l.startswith("| Idea |")), "")
        ideas = {f"IDEA-{num}" for k, num in IDENTIFIER.findall(idea_line) if k == "IDEA"}
        if not ideas and not re.search(r"issue #\d+", idea_line):
            # An experiment answers an idea note or, for a question raised upstream (a soundness or
            # zero-knowledge issue), the issue itself; either must be named on the line.
            fail(problems, f"{path.parent.name[:8]}: no idea or issue named on its `| Idea |` line")
        for ident in ideas:
            if ident not in on_disk["IDEA"]:
                fail(problems, f"{path.parent.name[:8]} names {ident}, which has no note")

    for problem in problems:
        print(f"FAIL  {problem}")
    print(f"{'ok' if not problems else 'FAIL'}: {len(md_files)} Markdown files, {len(defined)} bibliography entries, "
          f"{len(cited)} distinct citation keys, {len(on_disk['SUR'])} surveys, {len(on_disk['IDEA'])} ideas, {len(on_disk['EXP'])} experiments")
    return 1 if problems else 0


if __name__ == "__main__":
    sys.exit(main())
