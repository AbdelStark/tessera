#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0 OR MIT
"""Reference endpoints of 99% Clopper-Pearson intervals, for the soundness-calibration harness
(docs/testing/soundness-calibration.md). This script is the independent oracle: it sums the
binomial tails term by term in mpmath at 50 digits, where the harness sums logarithms in
doubles, and bisects to 10^-40.

Usage: python3 oracle.py K,N [K,N ...]   (needs mpmath; prints one Rust tuple per pair)
"""

import sys

import mpmath as mp

mp.mp.dps = 50
TAIL = mp.mpf("0.005")


def cdf(k, n, p):
    """P(X <= k) for X ~ Binomial(n, p)."""
    if k >= n:
        return mp.mpf(1)
    q = 1 - p
    term = q**n
    total = term
    for i in range(k):
        term = term * (n - i) / (i + 1) * p / q
        total += term
    return total


def root(f):
    """The root in (0, 1) of a decreasing f."""
    low, high = mp.mpf(0), mp.mpf(1)
    for _ in range(140):
        mid = (low + high) / 2
        if f(mid) > 0:
            low = mid
        else:
            high = mid
    return (low + high) / 2


def interval(k, n):
    low = mp.mpf(0) if k == 0 else root(lambda p: cdf(k - 1, n, p) - (1 - TAIL))
    high = mp.mpf(1) if k == n else root(lambda p: cdf(k, n, p) - TAIL)
    return low, high


def literal(x):
    return "0.0" if x == 0 else "1.0" if x == 1 else repr(float(x))


for pair in sys.argv[1:]:
    k, n = (int(v) for v in pair.split(","))
    low, high = interval(k, n)
    print(f"({k}, {n}, {literal(low)}, {literal(high)}),")
