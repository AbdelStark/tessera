#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0 OR MIT
"""Inspect equivalent, unrolled sixteen-point blocks; assembly is evidence, not a GPU cost."""
from pathlib import Path
import json
import re
import subprocess

HERE = Path(__file__).resolve().parent


def main():
    subprocess.run(["cargo", "rustc", "--release", "--locked", "--", "--emit", "asm"], cwd=HERE, check=True)
    path = max((HERE / "target/release/deps").glob("radixcal-*.s"), key=lambda p: p.stat().st_mtime)
    lines = path.read_text().splitlines()
    out = {"architecture": "aarch64", "block_points": 16, "radix2_equivalent_butterflies": 32, "anchors": {}}
    for name in ("radix2_anchor", "lazy_radix16_anchor"):
        matches = [i for i, s in enumerate(lines) if name in s and s.endswith(":") and not s.startswith(".")]
        assert len(matches) == 1, (name, matches)
        body = []
        for line in lines[matches[0]+1:]:
            text = line.split(";")[0].strip()
            if not text or text.startswith(".") or text.endswith(":"):
                continue
            if text == "ret":
                break
            body.append(text)
        branches = [s for s in body if re.match(r"(?:b\.|cbz|cbnz|tbz|tbnz|b\s|bl\s)", s)]
        stack = [s for s in body if re.search(r"\[sp[,\]]", s)]
        out["anchors"][name] = {"instructions": len(body), "per_butterfly_equivalent": len(body)/32, "conditional_or_call_instructions": branches, "stack_accesses": stack, "body": body}
        print(f"{name}: {len(body)} instructions, {len(stack)} stack accesses, {len(branches)} branch/call instructions")
    out["ratio"] = out["anchors"]["lazy_radix16_anchor"]["instructions"] / out["anchors"]["radix2_anchor"]["instructions"]
    print(f"static ratio A={out['ratio']:.6f}; no dynamic issue or GPU inference")
    (HERE / "results/asm.json").write_text(json.dumps(out, indent=2)+"\n")


if __name__ == "__main__":
    main()
