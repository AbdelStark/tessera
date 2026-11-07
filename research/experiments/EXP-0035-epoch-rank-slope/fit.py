#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0 OR MIT
"""EXP-0035's fit: the slope of ln f against ln N, with a parametric bootstrap interval."""

import math
import random
import sys

SEED = 0x0035
RESAMPLES = 2000


def slope(points):
    """Least-squares slope of y on x."""
    k = len(points)
    mx = sum(x for x, _ in points) / k
    my = sum(y for _, y in points) / k
    num = sum((x - mx) * (y - my) for x, y in points)
    return num / sum((x - mx) ** 2 for x, _ in points)


def main(path):
    """Print the counts, the slope and its 95% bootstrap interval."""
    cells = []
    with open(path) as f:
        for line in f:
            t = line.split()
            if len(t) == 6 and t[0] == "T" and t[1].isdigit():
                cells.append((int(t[1]), int(t[4]), int(t[5])))
    assert [n for n, _, _ in cells] == [8, 16, 32, 64], cells
    for n, trials, fails in cells:
        print(f"N = {n}: {fails} / {trials}")
    if min(f for _, _, f in cells) < 10:
        print("fewer than 10 failures at some N: inconclusive by the criterion")
    s = slope([(math.log(n), math.log(f / t)) for n, t, f in cells])
    rng = random.Random(SEED)
    boot = []
    for _ in range(RESAMPLES):
        pts = []
        for n, t, f in cells:
            # Binomial at the observed rate, as pre-registered; a zero draw is kept at 1 for the log.
            draw = max(1, rng.binomialvariate(t, f / t))
            pts.append((math.log(n), math.log(draw / t)))
        boot.append(slope(pts))
    boot.sort()
    lo, hi = boot[int(0.025 * RESAMPLES)], boot[int(0.975 * RESAMPLES) - 1]
    print(f"slope s = {s:.3f}, 95% bootstrap interval [{lo:.3f}, {hi:.3f}]")


if __name__ == "__main__":
    main(sys.argv[1])
