#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0 OR MIT
"""EXP-0012 exact structural counts (E0), with explicit illustrative int32 costs.

No instruction-count or H100 timing claim: compiler register allocation, carry chains,
signed reduction and memory traffic require the E2 implementation. Shared unchanged
forest levels cancel; these counts compare the replaced leaf window only.
"""

from __future__ import annotations

import json
from pathlib import Path


def counts(n: int, v: int) -> dict[str, object]:
    pairs = 1 << n
    blocks = pairs >> v
    cells = 3**v
    low_bits = (n - v + 1) // 2
    high_groups = 1 << (n - v - low_bits)
    low_entries = 1 << low_bits
    cadence = 8 if v == 5 else 32
    reductions = 2 * cells * high_groups * ((low_entries + cadence - 1) // cadence)
    bound = cadence * (2**64 - 1) * (2**22 - 1) ** 2 * 9**v
    assert bound < 2**127
    extrapolations = 2 * blocks * sum(2 ** (v-k-1) * 3**k for k in range(v))
    product = blocks * cells
    # Direct once-binding: two half tables, one mul_by_base per original entry.
    bind_mul_by_base = 2 * pairs
    # Window message sums: 3 nodes for every Boolean suffix, after each collapse.
    message_fp2_mul = sum(3 * (1 << (v-k-1)) for k in range(v))
    # Quadratic collapse as written: r*(b-a) and quadratic_weight*second per output.
    collapse_fp2_mul = 2 * sum(3 ** (v-k-1) for k in range(v))
    baseline = {name: 0 for name in ("fp_mul", "fp_add", "fp_sub", "fp2_mul", "fp2_mul_by_base", "fp2_add", "fp2_sub")}
    for k in range(v):
        active_pairs = pairs >> (k + 1)
        rem = n - k - 1
        lo, hi = (rem + 1) // 2, rem // 2
        eq_entries = (1 << lo) + (1 << hi) - 2
        baseline["fp2_mul"] += eq_entries + 2 * (1 << hi)
        baseline["fp2_sub"] += eq_entries
        baseline["fp2_add"] += 2 * (1 << hi)
        rule = {"fp_mul": 2, "fp_add": 4, "fp_sub": 4, "fp2_mul_by_base": 4, "fp2_add": 2} if k == 0 else {"fp2_mul": 6, "fp2_add": 6, "fp2_sub": 4}
        for key, value in rule.items():
            baseline[key] += value * active_pairs
    baseline_equivalents = baseline["fp_mul"] + 3 * baseline["fp2_mul"] + 2 * baseline["fp2_mul_by_base"]
    result: dict[str, object] = {
        "pair_variables": n,
        "v": v,
        "pairs": pairs,
        "blocks": blocks,
        "grid_cells": cells,
        "cadence_max_blocks": cadence,
        "exact_signed_bound": bound,
        "integer_extrapolations": extrapolations,
        "signed_small_products": product,
        "signed_small_big_products": 2 * product,
        "signed_i128_accumulations": 2 * product,
        "signed_component_reductions": reductions,
        "high_group_fp2_multiplications": cells * high_groups,
        "once_bind_mul_by_base": bind_mul_by_base,
        "window_message_fp2_multiplications": message_fp2_mul,
        "quadratic_collapse_fp2_multiplications": collapse_fp2_mul,
        "quadratic_weight_fp2_multiplications": v,
        "quadratic_weight_mul_by_base": v,
        "ternary_grid_bytes": 16 * cells,
        "split_baseline_replaced_rounds_and_folds": baseline,
        "split_baseline_fp_mul_equivalents_per_pair": baseline_equivalents / pairs,
        "common_per_round_claim_recovery_excluded_both_arms": True,
    }
    # Explicit E0 assumptions: 2 int32 ops for the small product, 8 per 64x64->128
    # scaling, 8 per signed 128-bit add, 32 per signed reduction, 2 per extrapolation.
    # Includes the work omitted by the idea's 135-op headline: both components,
    # high-group scaling, once-binding and interpolation are separately visible.
    result["illustrative_integer_int32_ops_per_pair"] = (
        2 * extrapolations + 2 * product + 8 * 2 * product
        + 8 * 2 * product + 32 * reductions
    ) / pairs
    return result


def main() -> None:
    rows = [counts(n, v) for n in (19, 20, 21) for v in (1, 3, 4, 5)]
    assert counts(21, 4)["signed_small_products"] == 81 * (1 << 17)
    assert counts(21, 5)["cadence_max_blocks"] == 8
    print(json.dumps({
        "experiment": "EXP-0012", "evidence": "E0",
        "scope": "replaced leaf-level window; unchanged forest levels and tail cancel",
        "instruction_costs_are_assumptions": True,
        "excluded_from_integer_int32_column": [
            "once-binding Fp2-by-Fp operations", "high eq-group Fp2 scaling",
            "eq-table construction", "grid collapse and message sums", "allocations and copies",
        ],
        "rows": rows,
        "fraction_window_case": [
            {
                "v": v, "leaves_log": leaves_log,
                "grid_points_per_pair": (3 / 2)**v,
                "grid_input_passes": 1, "once_bind_input_passes": 1,
                "counterfactual_regenerated_input_passes": v,
                "counterfactual_pass_ratio_including_bind": v / 2,
                "grid_bytes_read": (1 << leaves_log) * 16,
                "once_bind_bytes_read": (1 << leaves_log) * 16,
                "actual_regeneration_measured": False,
            }
            for leaves_log in range(20, 25) for v in (2, 3, 4, 5)
        ],
    }, indent=2))


if __name__ == "__main__":
    main()
