#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0 OR MIT
"""EXP-0036's fit: per N, the exponent a_N of f against p, with a binomial bootstrap interval."""

import math
import random
import sys

SEED = 0x0036
RESAMPLES = 2000


def exponent(points):
    """Minus the least-squares slope of ln f on ln p."""
    k = len(points)
    mx = sum(x for x, _ in points) / k
    my = sum(y for _, y in points) / k
    num = sum((x - mx) * (y - my) for x, y in points)
    return -num / sum((x - mx) ** 2 for x, _ in points)


def main(path):
    """Print each N's counts, a_N and its 95% bootstrap interval."""
    cells = {}
    with open(path) as f:
        for line in f:
            t = line.split()
            if len(t) == 6 and t[0] == "T" and t[1].isdigit():
                cells.setdefault(int(t[1]), []).append((int(t[3]), int(t[4]), int(t[5])))
    rng = random.Random(SEED)
    low = min(f for row in cells.values() for _, _, f in row)
    print(f"smallest cell count: {low}" + (" (below 20: inconclusive by the criterion)" if low < 20 else ""))
    for n in sorted(cells):
        row = cells[n]
        a = exponent([(math.log(p), math.log(f / t)) for p, t, f in row])
        boot = sorted(
            exponent([(math.log(p), math.log(max(1, rng.binomialvariate(t, f / t)) / t)) for p, t, f in row])
            for _ in range(RESAMPLES)
        )
        lo, hi = boot[int(0.025 * RESAMPLES)], boot[int(0.975 * RESAMPLES) - 1]
        counts = ", ".join(f"p={p}: {f}/{t}" for p, t, f in row)
        print(f"N = {n}: {counts}; a = {a:.3f}, 95% interval [{lo:.3f}, {hi:.3f}]")


if __name__ == "__main__":
    main(sys.argv[1])
