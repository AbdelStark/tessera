# SPDX-License-Identifier: Apache-2.0 OR MIT
"""RFC-0009 Lemma 7: Python integer oracle, no production code imports."""
from math import isqrt
from pathlib import Path
import importlib.util
import sys

sys.dont_write_bytecode = True


def main():
    lines = ["# d kappa epsilon u real root lower_product upper_product"]
    seed = 126
    for d, kappa, epsilon in [(1, 0, 1), (4096, 30, 672),
                              (16384, 30, 2685), (1, 31, 1),
                              (2**32 - 1, 16, 1)]:
        n = d * 2**(2*kappa)
        values = [epsilon, epsilon+1, 2**55-1]
        for _ in range(8):
            seed = (seed*6364136223846793005 + 1442695040888963407) % 2**64
            values.append(epsilon + seed % (2**55-epsilon))
        for u in values:
            r = isqrt(n//u)
            assert r*r*u <= n < (r+1)*(r+1)*u
            lines.append(f"{d} {kappa} {epsilon} {u} 1 {r} {r*r*u} {(r+1)*(r+1)*u}")
        lines.append(f"{d} {kappa} {epsilon} 0 0 0 0 0")
    Path(__file__).with_name("roots.txt").write_text("\n".join(lines)+"\n")
    # RFC-0050's independent schema predates the model and scalar integrations.
    path = Path(__file__).resolve().parent.parent / "advice/generate.py"
    spec = importlib.util.spec_from_file_location("advice_schema", path)
    advice = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(advice)
    schema = ["# width d kappa epsilon columns requests followed by ordered Range widths"]
    for width in range(1, 33):
        d, kappa = (4, 30) if width == 32 else (1, width-1)
        roles, requests = advice.root_schema(d, kappa, 1)
        values = [width, d, kappa, 1, len(roles)+1, len(requests), *requests]
        schema.append(" ".join(map(str, values)))
    Path(__file__).with_name("canonical.txt").write_text("\n".join(schema)+"\n")


if __name__ == "__main__":
    main()
