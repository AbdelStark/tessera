#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0 OR MIT
"""Targeted fault-injection (mutation testing) runner for Tessera.

Discovers single-site mutations in a Rust source file (comparison flips,
boolean flips, off-by-one bounds, and deleted `absorb_` calls), runs the test
command for each mutant, restores the original file unconditionally, and exits
non-zero if any mutant survives.
"""

from __future__ import annotations

import argparse
import dataclasses
import pathlib
import re
import subprocess
import sys
from typing import Iterable


@dataclasses.dataclass(frozen=True)
class Mutant:
    line_no: int
    kind: str
    original_line: str
    mutated_line: str


PATTERNS: tuple[tuple[str, re.Pattern[str], str], ...] = (
    ("eq-to-ne", re.compile(r"(?<![=!<>])==(?!=)"), "!="),
    ("ne-to-eq", re.compile(r"!="), "=="),
    ("lt-to-le", re.compile(r"(?<![<>=!])<(?![<=])"), "<="),
    ("le-to-lt", re.compile(r"(?<![<>=!])<="), "<"),
    ("gt-to-ge", re.compile(r"(?<![<>=!-])>(?![>=])"), ">="),
    ("ge-to-gt", re.compile(r"(?<![<>=!-])>="), ">"),
    ("and-to-or", re.compile(r"&&"), "||"),
    ("or-to-and", re.compile(r"\|\|"), "&&"),
)


def is_test_or_comment_line(stripped: str) -> bool:
    return (
        not stripped
        or stripped.startswith("//")
        or stripped.startswith("/*")
        or stripped.startswith("*")
        or stripped.startswith("#[")
        or stripped.startswith("assert")
        or stripped.startswith("debug_assert")
    )


def discover_mutants(lines: list[str], line_range: tuple[int, int] | None) -> Iterable[Mutant]:
    in_test_mod = False
    for idx, line in enumerate(lines):
        line_no = idx + 1
        stripped = line.strip()
        if stripped.startswith("#[cfg(test)]"):
            in_test_mod = True
        if in_test_mod and line_range is None:
            continue
        if line_range is not None and not (line_range[0] <= line_no <= line_range[1]):
            continue
        if is_test_or_comment_line(stripped):
            continue

        # Strip trailing line comment before matching operators
        code_part, sep, comment_part = line.partition("//")

        # 1. Transcript absorb deletion mutant
        if ".absorb" in code_part and stripped.endswith(";"):
            indent = line[: len(line) - len(line.lstrip())]
            yield Mutant(
                line_no=line_no,
                kind="delete-absorb",
                original_line=line,
                mutated_line=f"{indent}// MUTANT: deleted absorb\n",
            )

        # 2. Relational / logical operator flips
        for kind, pattern, replacement in PATTERNS:
            for match in pattern.finditer(code_part):
                # Skip turbofish / generic brackets like Vec<T> or Result<T, E>
                if kind in ("lt-to-le", "gt-to-ge"):
                    before = code_part[: match.start()]
                    after = code_part[match.end() :]
                    if not (before.endswith(" ") and after.startswith(" ")):
                        continue
                mutated_code = (
                    code_part[: match.start()]
                    + replacement
                    + code_part[match.end() :]
                )
                mutated_line = mutated_code + (sep + comment_part if sep else "")
                yield Mutant(
                    line_no=line_no,
                    kind=kind,
                    original_line=line,
                    mutated_line=mutated_line,
                )


def parse_range(spec: str | None) -> tuple[int, int] | None:
    if not spec:
        return None
    start_s, _, end_s = spec.partition(":")
    start = int(start_s)
    end = int(end_s) if end_s else start
    return (start, end)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--file", required=True, type=pathlib.Path, help="Rust source file to mutate")
    parser.add_argument("--lines", default=None, help="Optional 1-indexed START:END line range")
    parser.add_argument("--cmd", required=True, help="Test command that must fail on each mutant")
    parser.add_argument("--max-mutants", type=int, default=50, help="Cap number of mutants to execute")
    parser.add_argument("--dry-run", action="store_true", help="List discovered mutants without running tests")
    args = parser.parse_args()

    target: pathlib.Path = args.file.resolve()
    original_text = target.read_text(encoding="utf-8")
    lines = original_text.splitlines(keepends=True)
    line_range = parse_range(args.lines)

    mutants = list(discover_mutants(lines, line_range))[: args.max_mutants]
    print(f"Discovered {len(mutants)} mutant(s) in {target}")

    if args.dry_run:
        for i, m in enumerate(mutants, 1):
            print(f"[{i}/{len(mutants)}] line {m.line_no} ({m.kind}):")
            print(f"  - {m.original_line.strip()}")
            print(f"  + {m.mutated_line.strip()}")
        return 0

    survived: list[Mutant] = []
    try:
        for i, m in enumerate(mutants, 1):
            mutated_lines = list(lines)
            mutated_lines[m.line_no - 1] = m.mutated_line
            target.write_text("".join(mutated_lines), encoding="utf-8")

            try:
                proc = subprocess.run(
                    args.cmd,
                    shell=True,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                    text=True,
                    timeout=30,
                )
                if proc.returncode == 0:
                    print(f"SURVIVED [{i}/{len(mutants)}] line {m.line_no} ({m.kind}): {m.original_line.strip()}")
                    survived.append(m)
                else:
                    print(f"KILLED   [{i}/{len(mutants)}] line {m.line_no} ({m.kind})")
            except subprocess.TimeoutExpired:
                print(f"KILLED   [{i}/{len(mutants)}] line {m.line_no} ({m.kind}) (timeout)")
    finally:
        target.write_text(original_text, encoding="utf-8")

    if survived:
        print(f"\nFAIL: {len(survived)}/{len(mutants)} mutant(s) survived.")
        return 1

    print(f"\nPASS: all {len(mutants)} mutant(s) killed.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
