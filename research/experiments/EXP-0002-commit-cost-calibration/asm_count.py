#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0 OR MIT
"""EXP-0002: the ARM64 instruction counts of the harness's two `#[inline(never)]` anchors.

Runs `cargo rustc --release --locked -- --emit asm` in this directory (one codegen unit, so one
assembly file), finds `mul_anchor` (one Goldilocks multiplication with `reduce128`) and
`ct_butterfly_anchor` (one Cooley–Tukey butterfly), and prints each body with its instruction
count: the CPU-side input of the int32 translation in count.py. Directives, labels and comments
are not counted; the final `ret` is.

    python3 -I asm_count.py
"""

from __future__ import annotations

import glob
import os
import re
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ANCHORS = ("mul_anchor", "ct_butterfly_anchor")


def main() -> int:
    subprocess.run(
        ["cargo", "rustc", "--release", "--locked", "--quiet", "--", "--emit", "asm"],
        cwd=HERE,
        check=True,
    )
    files = sorted(glob.glob(os.path.join(HERE, "target", "release", "deps", "calib-*.s")), key=os.path.getmtime)
    if not files:
        print("no assembly file under target/release/deps", file=sys.stderr)
        return 1
    path = files[-1]
    with open(path, encoding="utf-8") as f:
        lines = f.read().splitlines()
    print("assembly: %s" % os.path.relpath(path, HERE))
    for anchor in ANCHORS:
        label = re.compile(r"^_?_[RZ]N?\w*%s\w*:$" % re.escape(anchor))
        start = next((i for i, line in enumerate(lines) if label.match(line.strip()) or (anchor in line and line.rstrip().endswith(":") and not line.startswith("."))), None)
        if start is None:
            print("%s: not found" % anchor)
            continue
        body, count = [], 0
        for line in lines[start + 1 :]:
            text = line.split(";")[0].rstrip()
            if not text.strip():
                continue
            if text.strip().startswith(".") or text.rstrip().endswith(":"):
                if text.strip().startswith(".cfi_endproc"):
                    break
                continue
            count += 1
            body.append(text.strip())
            if text.strip() == "ret":
                break
        print()
        print("%s: %d instructions" % (anchor, count))
        for text in body:
            print("    " + text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
