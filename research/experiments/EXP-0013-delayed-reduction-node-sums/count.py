#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0 OR MIT
"""E0 exact counts; import the committed EXP-0008 rules instead of duplicating them.

An extension deferred product costs four wide products and four 192-bit adds.
An extension-by-base deferred product costs two of each. At block end each active
Acc192 costs three canonicalizations, one mul_small(EPSILON), one full multiply,
one add and one subtract; AccExt2 adds one mul_small(7) and one add.
The EPSILON mul_small uses the same 8/13 lane assumption as mul_small(7).
No E1 unit exists for a 192-bit add or wide product: do not invent one.
"""
from __future__ import annotations

import importlib.util
import json
import sys
from dataclasses import dataclass
from pathlib import Path

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location(
    "exp8_count", HERE.parent / "EXP-0008-split-eq-gruen-level-kernel/count.py"
)
assert spec and spec.loader
old = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = old
spec.loader.exec_module(old)
Ops = old.Ops


@dataclass
class Count:
    reduced: object
    wide_products: int = 0
    accumulates192: int = 0
    block_reductions192: int = 0
    ext_blocks: int = 0

    def lanes(self, accumulate: int, end: int = 0) -> float:
        # Each 64x64 product is 8 int32 operations (18 minus 10 for reduction).
        cost = old.int32(self.reduced.expand(), end)
        cost += self.wide_products * 8 + self.accumulates192 * accumulate
        add, sub, mul7 = [old.FIELD_OPS[k][end] for k in ("add", "sub", "mul7")]
        # Canonicalization charged conservatively as a base add's cost.
        reduce192 = 3 * add + mul7 + old.FIELD_OPS["mul"][end] + add + sub
        return cost + self.block_reductions192 * reduce192 + self.ext_blocks * (mul7 + add)

    def report(self, units: int) -> dict:
        expanded = self.reduced.expand()
        return {
            "full_multiplications_reduce128": expanded["mul"] / units,
            "reduced_mul_small": expanded["mul7"] / units,
            "wide_products": self.wide_products / units,
            "accumulates192": self.accumulates192 / units,
            "block_reductions192": self.block_reductions192 / units,
            "ext_blocks": self.ext_blocks / units,
            "int32_ops_per_unit_acc4_5_8": [self.lanes(a) / units for a in (4, 5, 8)],
            "known_reduced_operations_ns": self.reduced.ns() / units,
        }


def level(n: int, kind: str, arm: str) -> Count:
    c = Count(Ops())
    for k in range(n):
        pairs = 1 << (n - k - 1)
        rem = n - k - 1
        lo, hi = (rem + 1) // 2, rem // 2
        groups = 1 << hi
        shared = old.BASE["eq_entry"] * ((1 << lo) + groups - 2)
        shared += old.CAND["split_group"] * groups + old.CAND["split_round_overhead"]
        if kind == "product":
            first = k == 0
            rule = old.CAND["split_round1" if first else "split_round_ext"]
            fold = old.BASE["bind_base" if first else "bind_ext"] * 2
        else:
            first = False
            rule = old.CAND[f"fraction_{kind}_round"]
            fold = old.BASE["bind_ext"] * (2 if kind == "leaf" else 4)
        if arm != "split":
            if kind == "product" and first:
                rule += Ops(fp2_mulb=-2, fp2_add=-2)
                wide, reducers, blocks = 4, 4, 0
            else:
                deferred = 4 if arm == "scale-first" else 2
                removed_add = 4 if arm == "scale-first" else 2
                rule += Ops(fp2_mul=-deferred, fp2_add=-removed_add)
                wide, reducers, blocks = 4 * deferred, 6, 2
            if arm == "scale-first":
                # One extra per-round w*gamma table, not one per pair.
                shared += Ops(fp2_mul=1 << lo)
            c.wide_products += wide * pairs
            c.accumulates192 += wide * pairs
            c.block_reductions192 += reducers * groups
            c.ext_blocks += blocks * groups
        c.reduced += shared + (rule + fold) * pairs
    return c


def forest(trees: int, height: int, arm: str) -> Count:
    total = Count(Ops())
    for l in range(height):
        c = level(trees + l, "product", arm)
        total.reduced += c.reduced
        total.wide_products += c.wide_products
        total.accumulates192 += c.accumulates192
        total.block_reductions192 += c.block_reductions192
        total.ext_blocks += c.ext_blocks
    original = old.forest(trees, height, True)
    total.reduced += original["root"] + original["build"]
    return total


def main() -> None:
    out = {"evidence": "E0 primary analytic counts", "fraction": {}, "forest": {}}
    for kind in ("leaf", "generic"):
        out["fraction"][kind] = {
            arm: level(21, kind, arm).report(1 << 21)
            for arm in ("split", "scale-first", "relation-first")
        }
        # Assert the denominator exactly matches EXP-0008's existing fraction count.
        ours = level(21, kind, "split").reduced
        assert ours == old.fraction_levels(1 << 21)[kind + " candidate"]
    for trees, height in ((8, 14), (10, 12), (0, 22)):
        units = 1 << (trees + height)
        arms = {a: forest(trees, height, a) for a in ("split", "deferred")}
        assert arms["split"].reduced == old.forest(trees, height, True)["total"]
        record = {a: c.report(units) for a, c in arms.items()}
        fused, unfused = old.bytes_per_leaf(trees, height, True)
        for a, c in arms.items():
            record[a]["H100_ps_interval_acc4_lazy_acc8_written"] = [
                max(old.ps(c.lanes(4) / units, old.INT32_PEAK_OPS), old.ps(fused, old.HBM_PEAK_BYTES)),
                max(old.ps(c.lanes(8, 1) / units, old.INT32_PEAK_OPS * old.INT_UTILIZATION), old.ps(unfused, old.HBM_PEAK_BYTES * old.HBM_UTILIZATION)),
            ]
        out["forest"][f"2^{trees}x2^{height}"] = record
    print(json.dumps(out, indent=2))
    (HERE / "results/count.json").write_text(json.dumps(out, indent=2) + "\n")


if __name__ == "__main__":
    main()
