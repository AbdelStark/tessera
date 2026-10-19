#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0 OR MIT
"""Independent exact arithmetic for the 2026-10-19 RFC-0021 review.

This review artifact changes no historical EXP-0005 criterion or result. Radical
ceilings use integer/Fraction predicates; Decimal logarithms are display values.
The weight cases retain P=65536 from the registered masked planning profiles,
not protocol/1's operational P=0 (RFC-0049). No Tessera arithmetic is imported.
"""

from __future__ import annotations

import argparse
import json
from math import isqrt
from decimal import Decimal, localcontext
from fractions import Fraction
from pathlib import Path
from typing import Callable


def least_true(predicate: Callable[[int], bool], upper: int) -> int:
    """Return the least nonnegative satisfying integer in a monotone interval."""
    if upper < 0 or not predicate(upper):
        raise ValueError("upper endpoint does not satisfy the predicate")
    lower = 0
    while lower < upper:
        middle = (lower + upper) // 2
        if predicate(middle):
            upper = middle
        else:
            lower = middle + 1
    return lower


def ceil_fraction(value: Fraction) -> int:
    return -(-value.numerator // value.denominator)


def decimal(value: Fraction) -> Decimal:
    return Decimal(value.numerator) / Decimal(value.denominator)


def negative_log2(value: Decimal) -> Decimal:
    return -value.ln() / Decimal(2).ln()


def sqrt_interval(value: Fraction, precision: int = 160) -> tuple[Fraction, Fraction]:
    """Enclose a nonnegative square root with exact rational endpoints."""
    if value < 0 or precision < 0:
        raise ValueError("square root and precision must be nonnegative")
    scale = 1 << precision
    lower = isqrt(value.numerator * scale**2 // value.denominator)
    exact = lower**2 * value.denominator == value.numerator * scale**2
    return Fraction(lower, scale), Fraction(lower if exact else lower + 1, scale)


def query_union() -> dict:
    """Check query-only eleven-commitment examples, without choosing SP-1 policy."""
    advice_rho = Fraction(2**22 + 160 - 1, 2**23)
    weight_rho = Fraction(2**20 + 2**16 - 1, 2**22)
    advice_lo, advice_hi = (x + Fraction(3, 1000) for x in sqrt_interval(advice_rho))
    weight_lo, weight_hi = (x + Fraction(1, 200) for x in sqrt_interval(weight_rho))
    threshold = Fraction(1, 2**64)
    results = []
    with localcontext() as context:
        context.prec = 90
        advice_alpha = decimal(advice_rho).sqrt() + decimal(Fraction(3, 1000))
        weight_alpha = decimal(weight_rho).sqrt() + decimal(Fraction(1, 200))
        for advice_tau, weight_tau in ((130, 68), (138, 71), (138, 72)):
            lower = advice_lo**advice_tau + 10 * weight_lo**weight_tau
            upper = advice_hi**advice_tau + 10 * weight_hi**weight_tau
            status = "pass" if upper <= threshold else "fail" if lower > threshold else "unresolved"
            displayed = advice_alpha**advice_tau + 10 * weight_alpha**weight_tau
            results.append(
                {
                    "tau_advice": advice_tau,
                    "tau_weights": weight_tau,
                    "query_bits_decimal90": str(negative_log2(displayed)),
                    "exact_interval_status_vs_2pow_minus64": status,
                    "upper_ratio_to_2pow_minus64_decimal90": str(decimal(upper / threshold)),
                }
            )
    return {
        "scope": "query-only sum for one advice plus ten weights; masked planning P_W=65536, not protocol/1 calibration or a policy decision",
        "sum": "alpha_A^tau_A + 10*alpha_W^tau_W",
        "proof": "exact Fraction powers using 160-bit rational radical enclosures; Decimal90 bits for display",
        "alpha_advice_interval": [str(advice_lo), str(advice_hi)],
        "alpha_weight_interval": [str(weight_lo), str(weight_hi)],
        "cases": results,
    }


def calculate() -> dict:
    """Recompute the four review profiles with the actual Goldilocks square."""
    field_p = 2**64 - 2**32 + 1
    field_q = field_p**2
    cases = (
        ("live advice", 2**22, 160, 2**23, Fraction(3, 1000), 130),
        ("NI advice", 2**22, 200, 2**23, Fraction(3, 1000), 163),
        ("live weights", 2**20, 2**16, 2**22, Fraction(1, 200), 68),
        ("NI weights", 2**20, 2**16, 2**22, Fraction(1, 200), 85),
    )
    results = []
    with localcontext() as context:
        context.prec = 90
        for name, k, mask, n, eta, tau in cases:
            degree = k + mask - 1
            rho = Fraction(degree, n)
            agreement = least_true(
                lambda a: a >= eta * n and (Fraction(a) - eta * n) ** 2 >= degree * n,
                n,
            )
            multiplicity = max(3, least_true(lambda m: (2 * eta * m) ** 2 >= rho, n))
            shifted = Fraction(2 * multiplicity + 1, 2)
            mu = least_true(lambda value: value**2 * rho >= shifted**2, n) - 1
            height = ceil_fraction(shifted**2 / (3 * rho)) - 1
            theta = Fraction(n - degree, agreement - degree)
            exceptions = (
                (2 * mu - 1) * height
                + theta * (height + mu + 4 * degree * mu * height)
                + (n - degree - 1) * mu
            )
            list_bound = n * (agreement - degree) // (agreement**2 - n * degree)
            alpha = decimal(rho).sqrt() + decimal(eta)
            hypotheses = {
                "positive_degree": degree >= 1,
                "degree_below_n_minus_2": degree <= n - 2,
                "positive_eta": eta > 0,
                "agreement_below_one": alpha <= 1,
                "dimension_threshold": degree + 1 <= alpha * n,
                "pairwise_threshold": degree + 1 <= agreement,
                "pairwise_positive": n * degree < agreement**2,
                "separating_50_pair_union_lt_field": 50 * 49 // 2 < field_q,
            }
            results.append(
                {
                    "name": name,
                    "k": k,
                    "P": mask,
                    "n": n,
                    "D": degree,
                    "eta": str(eta),
                    "A": agreement,
                    "m": multiplicity,
                    "mu": mu,
                    "h": height,
                    "L": list_bound,
                    "E0": str(exceptions),
                    "threshold_margin": str(alpha * n - (degree + 1)),
                    "field_bits": str(negative_log2(decimal(exceptions / (field_q - 1)))),
                    "query_bits": str(tau * negative_log2(alpha)),
                    "claim_bits_T256_G16": str(
                        negative_log2(decimal(Fraction(16 * 256 * list_bound, field_q)))
                    ),
                    "claim_bits_terms_conservative": str(
                        negative_log2(decimal(Fraction(16 * 257 * list_bound, field_q)))
                    ),
                    "consistency_bits": str(tau * negative_log2(decimal(rho))),
                    "hypotheses": hypotheses,
                }
            )
    return {
        "field_p": field_p,
        "field_q": field_q,
        "oracle": "independent Fraction integer comparisons; Decimal precision90 for displayed logarithms",
        "cases": results,
    }


def self_test() -> None:
    """Check exact boundary ceilings and pinned independent-review values."""
    for radicand, expected in ((0, 0), (1, 1), (2, 2), (4, 2), (5, 3)):
        found = least_true(lambda x: x * x >= radicand, 10)
        if found != expected:
            raise ValueError(f"radical boundary: expected {expected}, got {found}")
    for fraction, expected in ((Fraction(3, 2), 2), (Fraction(2), 2), (Fraction(-3, 2), -1)):
        if ceil_fraction(fraction) != expected:
            raise ValueError("rational ceiling boundary mismatch")
    if sqrt_interval(Fraction(4)) != (Fraction(2), Fraction(2)):
        raise ValueError("perfect-square interval mismatch")
    lower, upper = sqrt_interval(Fraction(2))
    if not lower**2 < 2 < upper**2:
        raise ValueError("irrational interval does not enclose the radical")
    union_statuses = [case["exact_interval_status_vs_2pow_minus64"] for case in query_union()["cases"]]
    if union_statuses != ["fail", "fail", "pass"]:
        raise ValueError("query-only union boundary status mismatch")
    result = calculate()
    expected_thresholds = (5_956_920, 5_956_949, 2_182_666, 2_182_666)
    for case, expected in zip(result["cases"], expected_thresholds, strict=True):
        if case["A"] != expected or case["L"] != 49:
            raise ValueError(f"pinned threshold/list mismatch in {case['name']}")
        if not all(case["hypotheses"].values()):
            raise ValueError(f"theorem hypothesis failed in {case['name']}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--self-test", action="store_true")
    parser.add_argument("--check-summary", type=Path, help="compare every arithmetic field with review JSON")
    parser.add_argument("--query-union", action="store_true", help="print conditional query-only union examples")
    args = parser.parse_args()
    if args.self_test:
        self_test()
        print("self-test: exact boundaries and four pinned profiles pass")
    if args.check_summary:
        summary = json.loads(args.check_summary.read_text())
        if calculate() != summary["arithmetic"]:
            raise SystemExit("review arithmetic differs from summary")
        if "query_union" in summary and query_union() != summary["query_union"]:
            raise SystemExit("review query-only union differs from summary")
        print("summary: all exact arithmetic and displayed decimal fields match")
    if args.query_union:
        print(json.dumps(query_union(), indent=2))
    elif not args.self_test and not args.check_summary:
        print(json.dumps(calculate(), indent=2))


if __name__ == "__main__":
    main()
