#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0 OR MIT
"""E0 exact arithmetic counts; GPU lane translations remain explicit assumptions.

The second lazy carry/borrow fixes each cost an assumed two extra int32 lanes.
Negative cheap roots need an additional lazy subtraction; exponents >=64 use
two widened shifts and a widened subtraction. These were absent in EXP-0002's
hypothetical floor and are charged here. Precomputed public powers are excluded
from kernels and listed separately; no H100 performance is measured.
"""
from collections import Counter
from pathlib import Path
import importlib.util
import json
import sys

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location("exp2_counts", HERE.parent / "EXP-0002-commit-cost-calibration/count.py")
old = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = old
spec.loader.exec_module(old)


def reverse(i, bits):
    return int(f"{i:0{bits}b}"[::-1], 2) if bits else 0


def transform(log, radix_bits, inverse=False):
    n = 1 << log
    c = Counter(full=0, add=0, sub=0, cheap=0, cheap_negative=0, cheap_wide_difference=0,
                logical_power_mults_if_not_precomputed=0, traversed_value_bytes=0)
    remaining = log
    groups = 1
    exponents = Counter()
    while remaining:
        bits = min(radix_bits, remaining)
        r = 1 << bits
        columns = 1 << (remaining - bits)
        c["full"] += (groups - 1) * columns * (r - 1)
        c["add"] += n * bits // 2
        c["sub"] += n * bits // 2
        c["logical_power_mults_if_not_precomputed"] += groups * max(0, r - 2)
        c["traversed_value_bytes"] += n * 16
        root_exp = {1: 0, 2: 48, 3: 120, 4: 156}[bits]
        for level in range(bits):
            half = r >> (level + 1)
            for g in range(1 << level):
                e = root_exp * reverse(g, bits - 1) % 192
                if inverse:
                    e = -e % 192
                if e:
                    exponents[e] += groups * columns * half
        groups *= r
        remaining -= bits
    c["cheap"] = sum(exponents.values())
    c["cheap_negative"] = sum(v for e, v in exponents.items() if e >= 96)
    c["cheap_wide_difference"] = sum(v for e, v in exponents.items() if e % 96 >= 64)
    return c, dict(exponents)


def encoder(log_k, bits):
    k = 1 << log_k
    inv, _ = transform(log_k, bits, True)
    forward, _ = transform(log_k + 1, bits, False)
    c = inv + forward
    c["full"] += k + 3 * 200  # spread; scaled mask, shifted mask, public 7^i update
    c["sub"] += 200
    c["canonicalize"] = 2 * k
    c["traversed_value_bytes"] += 16 * k + 24 * k + 16 * (2*k) + 32*200  # input copy, spread, final canon
    return c


def lanes(c, lazy, high=False):
    if not lazy:
        # Canonical controls use the same reduction as production; cheap constants differ.
        mul, add, sub, cheap = 26, 8, 5, 12
    else:
        # 18 is EXP-0002's optimistic schedule, 22 removes only final canonicalize from 26.
        mul, add, sub, cheap = (22 if high else 18), 6, 6, (12 if high else 8)
    return (c["full"]*mul + c["add"]*add + c["sub"]*sub + c["cheap"]*cheap
            + c["cheap_negative"]*sub + c["cheap_wide_difference"]*4
            + c["canonicalize"]*4)


def main():
    k = 1 << 22
    original = old.ntt_structure()
    radix2_inv, _ = transform(22, 1, True)
    radix2_forward, _ = transform(23, 1)
    assert radix2_inv["full"] + radix2_forward["full"] + k == round((original["intt_butterflies"] - original["intt_unit"] + original["ntt_butterflies"] - original["ntt_unit"] + original["spread_mults"])*k)
    assert radix2_inv["add"] + radix2_forward["add"] == round((original["intt_butterflies"] + original["ntt_butterflies"])*k)
    reference = old.ops_high(original, old.hash_structure())
    floor = old.ops_low(original, old.hash_structure())
    plan_mults = sum(x//2-1 + sum((x//r)*(r-1) for r in (4,8,16)) for x in (k,2*k)) + k-1 + 2*k-1
    out = {"plan_precomputation_full_multiplications_excluding_root_and_inverse_setup":plan_mults, "evidence": "E0 primary arithmetic counts with conditional hardware translations", "historical_radix2": reference, "historical_radix16_floor": floor, "encoder": {}}
    hash_lo, hash_hi = floor["hash_ops"], reference["hash_ops"]
    for bits in (2, 3, 4):
        c = encoder(22, bits)
        lo, hi = lanes(c, True)/k, lanes(c, True, True)/k
        out["encoder"][str(1 << bits)] = {
            "exact_counts_per_element": {key: value/k for key, value in c.items()},
            "lazy_ntt_int32_assumed_interval": [lo, hi],
            "canonical_ntt_int32_assumed": lanes(c, False)/k,
            "gap_fraction_versus_historical_floor": (reference["ntt_ops"]-lo)/(reference["ntt_ops"]-floor["ntt_ops"]),
            "conditional_commit_ps_interval": [max(old.ps(lo+hash_lo, old.INT32_PEAK_OPS), old.ps(48, old.HBM_PEAK_BYTES)), max(old.ps(hi+hash_hi, old.INT32_PEAK_OPS*.6), old.ps(c["traversed_value_bytes"]/k+16, old.HBM_PEAK_BYTES*.8))],
            "hash_int32_interval_unchanged": [hash_lo, hash_hi],
        }
    # Confirm the stale assumed root differs from production; all counts use exponent120.
    p = 2**64-2**32+1
    assert pow(7, (p-1)//8, p) == pow(2,120,p) == p - 2**24
    assert pow(7, (p-1)//16, p) == pow(2,156,p) == p - 2**60
    text = json.dumps(out, indent=2)
    (HERE / "results/count.json").write_text(text+"\n")
    print(text)


if __name__ == "__main__":
    main()
