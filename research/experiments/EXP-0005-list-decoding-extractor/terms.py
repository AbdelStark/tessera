#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0 OR MIT
"""EXP-0005: the extra error terms of the list-decoding extractor at Tessera's parameter sets.

For each set and slack `η`: the agreement threshold `A`, the machine-checked list bound
`L = ⌊n (A − D) / (A² − n D)⌋` ([ArkLib26] `closePolynomialSet_finite_and_ncard_le_pairwiseJohnson`),
the recipe's candidate degree `μ` (`johnsonMu`, the looser bound the first version of the report
used), `log2` of a historical independent-per-claim comparison (`G · L / q'`) and a conservative
powers-of-one-challenge envelope (`G · L · (s + 1) / q'`). Current RFC-0021 pins powers gamma^(t+1)
for s claims, degree at most s, so its exact claim-binding bound uses s rather than this legacy
s+1 envelope. Retain the envelope for historical-table reproduction; it is not the current exact
formula. Also tabulate `(ρ₋)^τ` (group consistency),
the legacy `1/q'` comparison (a zero first coefficient under the old all-random generator;
the current fixed-row generator has no such event) and `E₀/(q' − 1)` (the
mutual-correlated-agreement error, from EXP-0003's tool). `G` and `s` are bounds here (RFC-0011
'Sizes' gives `G + 1 ≤ 17`; `s = 2^8` is generous); the soundness calculators take both from the
claim list. Rows: the `η` of EXP-0003's best admissible `τ`, the `η = 0.002` the first Draft RFC-0021
selected, and the `η = 0.003` the revised Draft selects (second reading, item 5).
"""

import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "tools"))
import proximity  # noqa: E402

FIELD_BITS = 128
GROUPS_MAX = 16  # RFC-0011 'Sizes': G + 1 ≤ 17 combined rows
CLAIMS_PER_GROUP_MAX = 2**8  # generous; only the powers-of-one-challenge shape depends on it

CASES = (
    ("live-64 advice", 0.001, 129),
    ("live-64 advice", 0.002, 130),
    ("live-64 advice", 0.003, 130),
    ("live-64 weights", 0.005, 68),
    ("ni-100 advice", 0.002, 162),
    ("ni-100 advice", 0.003, 163),
    ("ni-100 weights", 0.005, 85),
)

print(
    f"{'set':<18} {'η':>6} {'τ':>4} {'A_thr':>9} {'L':>5} {'μ':>5} "
    f"{'log2 G·L/q′':>12} {'log2 G·L·(s+1)/q′':>18} {'log2 (ρ₋)^τ':>12} {'log2 1/q′':>10} {'log2 ε_MCA':>11}"
)
for name, eta, tau in CASES:
    code = proximity.PARAMETER_SETS[name]
    r = proximity.arklib_johnson(code, eta)
    independent = math.log2(GROUPS_MAX) + math.log2(r["list_bound"]) - FIELD_BITS
    powers = independent + math.log2(CLAIMS_PER_GROUP_MAX + 1)
    consistency = tau * math.log2(r["rho"])
    print(
        f"{name:<18} {eta:>6.3f} {tau:>4} {r['A']:>9} {r['list_bound']:>5} {r['mu']:>5} "
        f"{independent:>12.1f} {powers:>18.1f} {consistency:>12.1f} {-FIELD_BITS:>10} {-r['affine_bits']:>11.1f}"
    )
print()
print("every extra term is below 2^-100 when each log2 value in the four middle columns is below -100;")
print("ε_MCA earns the seed's grinding credit only because ξ is squeezed right after the seed (RFC-0021 order 3b);")
print("the claim-binding term is re-rollable in both modes at one or more hashes per attempt and is budgeted as λ = −log2(term)")
