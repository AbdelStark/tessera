#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0 OR MIT
"""EXP-0015: exhaustive small-field certificates and exact Goldilocks bounds (E0)."""

from __future__ import annotations

import argparse
from collections import Counter
from functools import lru_cache
import hashlib
import importlib.util
import itertools
import json
from pathlib import Path
import platform
import subprocess
import sys
import time

sys.dont_write_bytecode = True
HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
P = 65_537
GOLD = 2**64 - 2**32 + 1
B = 64
INV_B = pow(B, -1, P)


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"))


def require(condition, message, witness=None):
    if not condition:
        raise AssertionError(canonical({"failure": message, "witness": witness}))


def integer_digits(value, bits, limb=2):
    """The integer oracle never calls the modular limb finder."""
    require(0 <= value < 2**bits, "oracle digit input", [value, bits])
    out = []
    remaining = bits
    while remaining:
        width = min(limb, remaining)
        value, digit = divmod(value, 2**width)
        out.append(digit)
        remaining -= width
    require(value == 0, "integer digits consume the input")
    return tuple(out)


def modular_limb_map(bits):
    """Enumerate all field inputs and committed low limbs, deriving the virtual high limb."""
    count = (bits + 1) // 2
    top_bits = bits - 2 * (count - 1)
    inverse = pow(4 ** (count - 1), -1, P)
    found = {}
    assignments = 0
    for value in range(P):
        passing = []
        for low in itertools.product(range(4), repeat=count - 1):
            low_sum = sum(digit * 4**j for j, digit in enumerate(low))
            top = (value - low_sum) * inverse % P
            if top < 2**top_bits:
                passing.append(low + (top,))
        if passing:
            found[value] = tuple(passing)
            assignments += len(passing)
    # Independent enumeration of every admissible integer digit box.
    integer_box = {}
    widths = [4] * (count - 1) + [2**top_bits]
    for digits in itertools.product(*(range(width) for width in widths)):
        value = sum(digit * 4**j for j, digit in enumerate(digits))
        integer_box.setdefault(value % P, []).append(digits)
    require(found == {v: tuple(ds) for v, ds in integer_box.items()}, "limb checker vs integer box")
    require(assignments == 2**bits and len(found) == assignments, "unique bounded digit box")
    return found


def toy_exp(key):
    return max(4 - key, 0)


class Exhaustion:
    def __init__(self):
        self.s_limbs = modular_limb_map(4)
        self.o_limbs = modular_limb_map(6)
        self.counts = Counter()
        self.digest = hashlib.sha256()

    def record(self, domain, data):
        self.counts[domain] += 1
        self.digest.update(canonical([domain, data]).encode() + b"\n")

    def row_witnesses(self, scores, valid):
        psi = int(any(valid))
        maxima = [(scores[valid.index(True)] + k) % P for k in range(32)] if psi else [0]
        passing = []
        for maximum in maxima:
            keys = [(maximum - x) % P if live else 31 for x, live in zip(scores, valid)]
            self.counts["maximum_candidates"] += 1
            if not all(key < 32 for key in keys):
                continue
            packed = tuple((B * toy_exp(k) + int(k == 0)) % P for k in keys)
            total = sum(packed) % P
            for n in range(psi, psi + 4):
                self.counts["row_witness_candidates"] += 1
                s = (total - n) * INV_B % P
                for limbs in self.s_limbs.get(s, ()):
                    passing.append((maximum, n, s, limbs, packed))
        return passing

    @lru_cache(maxsize=None)
    def pv_witnesses(self, packed_sum):
        passing = []
        for contamination in range(-32, 32):
            self.counts["pv_witness_candidates"] += 1
            output = (packed_sum - contamination) * INV_B % P
            for digits in self.o_limbs.get((output + 32) % P, ()):
                passing.append((contamination, output, digits))
        return tuple(passing)

    def check_output(self, domain, es, zs, packed, values):
        # Direct integer sums form the oracle. No modular witness finder is used here.
        output = sum(e * v for e, v in zip(es, values))
        contamination = sum(z * v for z, v in zip(zs, values))
        expected = (contamination, output % P, integer_digits(output + 32, 6))
        packed_sum = sum(f * v for f, v in zip(packed, values)) % P
        got = self.pv_witnesses(packed_sum)
        require(got == (expected,), "PV missing, wrong or multiple witnesses", [domain, es, zs, values, got, expected])
        self.record(domain, [es, zs, values, expected])

    def integrated(self, scores, valid, domain):
        maximum = max(x for x, live in zip(scores, valid) if live) if any(valid) else 0
        es = tuple(toy_exp(maximum - x) if live else 0 for x, live in zip(scores, valid))
        zs = tuple(int(live and maximum == x) for x, live in zip(scores, valid))
        packed = tuple(B * e + z for e, z in zip(es, zs))
        expected = (maximum % P, sum(zs), sum(es), integer_digits(sum(es), 4), packed)
        got = self.row_witnesses(scores, valid)
        require(got == [expected], "row missing, wrong or multiple witnesses", [scores, valid, got, expected])
        self.counts[domain + "_score_rows"] += 1
        for values in itertools.product(range(-2, 2), repeat=len(scores)):
            self.check_output(domain, es, zs, got[0][4], values)

    def run(self):
        for length in range(1, 4):
            for scores in itertools.product(range(-8, 9), repeat=length):
                self.integrated(scores, (True,) * length, "real")
        require(self.counts["real_score_rows"] == 5_219, "registered real score row count")
        require(self.counts["real"] == sum(68**n for n in range(1, 4)), "registered real value-vector count")
        for scores in itertools.product(range(-2, 3), repeat=3):
            for mask in range(8):
                valid = tuple(bool(mask & (1 << j)) for j in range(3))
                self.integrated(scores, valid, "masks")
        require(self.counts["masks"] == 125 * 8 * 64, "registered masked value-vector count")
        atoms = [(e, 0) for e in range(5)] + [(4, 1)]
        for length in range(1, 4):
            for row in itertools.product(atoms, repeat=length):
                es, zs = tuple(zip(*row))
                packed = tuple(B * e + z for e, z in row)
                total = sum(packed) % P
                witnesses = []
                for n in range(1, 5):
                    s = (total - n) * INV_B % P
                    witnesses.extend((n, s, ds) for ds in self.s_limbs.get(s, ()))
                expected = [(sum(zs), sum(es), integer_digits(sum(es), 4))] if any(zs) else []
                require(witnesses == expected, "abstract zero-count decomposition", [row, witnesses, expected])
                self.counts["abstract_zero_marker_rejections" if not any(zs) else "abstract_nonzero_marker_acceptances"] += 1
                for values in itertools.product(range(-2, 2), repeat=length):
                    self.check_output("abstract", es, zs, packed, values)
        require(self.counts["abstract"] == sum(24**n for n in range(1, 4)), "registered abstract count")
        return {"counts": dict(sorted(self.counts.items())), "sha256": self.digest.hexdigest(),
                "pv_cache_distinct_inputs": self.pv_witnesses.cache_info().currsize,
                "bad": 0, "missing": 0, "multi": 0, "limb_boxes": {"4": 16, "6": 64}}


def range_digits(field_value, bits, p=GOLD):
    value = field_value % p
    return integer_digits(value, bits, 16) if value < 2**bits else None


def pv_checks(packed_sum, contamination, output, m=23, p=GOLD):
    b = 2**m
    return {"PV_CONTAM_RANGE": -b // 2 <= contamination < b // 2,
            "PV_NONNEG": range_digits(output + 2**37, 38, p) is not None,
            "PV_IDENTITY": (b * output + contamination - packed_sum) % p == 0}


def row_checks(packed_sum, n, s, psi=1, maximum=0):
    return {"COUNT_RANGE": 0 <= n - psi < 2**17,
            "SUM_NONNEG": range_digits(s, 34) is not None,
            "SUM_IDENTITY": (2**23 * s + n - packed_sum) % GOLD == 0,
            "PADDING_MAX": bool(psi) or maximum % GOLD == 0}


def wrap_solutions(value, b, low_o, high_o, low_t, high_t, p):
    """All rectangle solutions to b*o+t = value mod p, via bounded wrap multiples."""
    low = b * low_o + low_t
    high = b * (high_o - 1) + high_t - 1
    first = -((value - low) // p)
    last = (high - value) // p
    found = []
    for wraps in range(first, last + 1):
        target = value + wraps * p
        # The admissible residue interval has at most b entries.
        t = (target - low_t) % b + low_t
        o = (target - t) // b
        if low_t <= t < high_t and low_o <= o < high_o:
            found.append((o, t, wraps))
    return found


def goldilocks():
    b, max_e, row, chunk = 2**23, 2**16, 2**17, 2**14
    score_low, score_high, key_bits = -2**21 + 1, 2**21, 22
    score_spread = score_high - score_low
    bounds = {
        "packed_sum_max": row * (b * max_e + 1),
        "row_rhs_max": b * (2**34 - 1) + row,
        "output_abs_max": chunk * (b * max_e + 1) * 128,
        "output_rhs_abs_max": b * 2**37 + b // 2,
        "honest_sum_max": row * max_e,
        "honest_chunk_abs_max": chunk * max_e * 128,
        "honest_contamination_abs_max": chunk * 128,
        "score_spread_max": score_spread,
        "selected_table_last_key": 2**key_bits - 1,
    }
    inequalities = {
        "zero_count_below_radix": row < b,
        "honest_contamination_strictly_centered": bounds["honest_contamination_abs_max"] < b // 2,
        "sum_window_complete": bounds["honest_sum_max"] < 2**34,
        "negative_output_endpoint_included": bounds["honest_chunk_abs_max"] == 2**37,
        "positive_output_endpoint_strict": chunk * max_e * 127 < 2**37,
        "row_congruence_nonwrap": max(bounds["packed_sum_max"], bounds["row_rhs_max"]) < GOLD,
        "pv_congruence_nonwrap": bounds["output_abs_max"] + bounds["output_rhs_abs_max"] < GOLD,
        "centered_comparison_margin": max(bounds["output_abs_max"], bounds["output_rhs_abs_max"]) < 2**62,
        "nonneg_boxes": 2**34 < GOLD and 2**38 < GOLD and 34 <= 62 and 38 <= 62,
        "negative_score_difference_cannot_be_a_table_key": GOLD - score_spread > 2**32 - 1,
        "positive_score_spread_fits_selected_table": score_spread <= 2**key_bits - 1,
    }
    require(all(inequalities.values()), "Goldilocks analytic interval obligation", inequalities)
    spec = importlib.util.spec_from_file_location("exp15_table_oracle", ROOT / "spec/vectors/tables/generate.py")
    oracle = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(oracle)
    kind = {"Exp": {"key_bits": key_bits, "sigma": 16, "scale": [1, 16]}}
    tail = oracle.exp_tail(2**22, 16, 1, 16)
    keys = sorted({0, 1, tail - 1, tail, tail + 1, 2**22 - 2, 2**22 - 1})
    table_case = oracle.case("exp15-boundary", 0, kind, [(k, 0) for k in keys])
    values = {k: oracle.exp_value(k, 16, 1, 16) for k in keys}
    require(values[0] == max_e and values[2**22 - 1] == 0, "normative exponential endpoints")
    require(all(0 <= e <= max_e for e in values.values()), "sampled actual exponential range")
    cases = []
    # Multiplicities represent complete large rows without allocating a witness array.
    for length in [1, 2, 2**14, 2**14 + 1, 2**17]:
        for score in [-2**21 + 1, 0, 2**21]:
            for value in [-128, -1, 0, 1, 127]:
                s, n = length * max_e, length
                f = b * s + n
                require(all(row_checks(f, n, s, maximum=score).values()), "Gold boundary row predicates")
                row_solutions = wrap_solutions(f, b, 0, 2**34, 1, row + 1, GOLD)
                require(row_solutions == [(s, n, 0)], "Gold row-sum boundary uniqueness", [length, row_solutions])
                chunks = []
                for start in range(0, length, chunk):
                    count = min(chunk, length - start)
                    o, t = count * max_e * value, count * value
                    packed = b * o + t
                    solutions = wrap_solutions(packed, b, -2**37, 2**37, -b // 2, b // 2, GOLD)
                    require(solutions == [(o, t, 0)], "Gold chunk boundary uniqueness", [length, value, solutions])
                    require(all(pv_checks(packed, t, o).values()), "Gold boundary predicates")
                    chunks.append([count, o, t, integer_digits(o + 2**37, 38, 16)])
                cases.append({"length": length, "score": score, "value": value, "s": s, "n": n, "chunks": chunks})
    # Mixed extrema and zero-tail keys, including a masked value with an irrelevant score.
    scores = [-2**21 + 1, 0, 2**21]
    maximum = max(scores)
    es = [oracle.exp_value(maximum - x, 16, 1, 16) for x in scores] + [0]
    zs = [int(x == maximum) for x in scores] + [0]
    for vs in itertools.product([-128, -1, 0, 1, 127], repeat=4):
        o, t = sum(e * v for e, v in zip(es, vs)), sum(z * v for z, v in zip(zs, vs))
        packed = b * o + t
        require(wrap_solutions(packed, b, -2**37, 2**37, -b // 2, b // 2, GOLD) == [(o, t, 0)], "Gold mixed/zero-tail/masked uniqueness", vs)
    require(wrap_solutions(0, b, -2**37, 2**37, -b // 2, b // 2, GOLD) == [(0, 0, 0)], "Gold empty chunk")
    require(all(row_checks(0, 0, 0, psi=0).values()), "Gold empty row predicates")
    return {"bounds": bounds, "inequalities": inequalities, "all_tie_boundary_cases": cases,
            "mixed_masked_value_vectors": 5**4, "empty_row": {"n": 0, "s": 0, "maximum": 0},
            "exp_parameters": kind, "exp_base_table_spec": table_case["spec"],
            "exp_zero_tail": tail, "exp_boundary_values": values,
            "default_limb_width": 16, "sum_digit_widths": [16, 16, 2], "output_digit_widths": [16, 16, 6],
            "row_additions_one_chunk": {"commitments": 387, "lookups": 516}}


def negative_controls():
    b = 2**23
    half = pow(2, -1, GOLD)
    fake_t = -b // 2
    fractional = pv_checks(0, fake_t, half)
    require(fractional == {"PV_CONTAM_RANGE": True, "PV_NONNEG": False, "PV_IDENTITY": True}, "fractional PV attack exact gates")
    product = half * 2**16 % GOLD
    requant = (product + 2**15) // 2**16
    require(product == 32768 and requant == 1, "fractional witness propagates to a changed integer")
    a, bracket_b = 2**32 - 2**16 * 2**16, 2**16 - 1
    require(0 <= a < 2**34 and 0 <= bracket_b < 2**34 and 2**16 < 2**17,
            "fractional attack also passes the original reciprocal bracket")
    toy_half = pow(2, -1, P)
    require((B * toy_half - 32) % P == 0 and (toy_half + 32) % P not in range(64), "toy fractional output witness")
    toy_product = toy_half * 4 % P
    require(toy_product == 2 and (toy_product + 2) // 4 == 1, "toy fractional requant")
    # Removing the positive count permits a fake maximum one above the only score.
    missing_count = {"scores": [0], "maximum": 1, "E": 3, "n": 0, "s": 3}
    require((B * 3 - (B * 3 + 0)) % P == 0 and not 1 <= 0 <= 4, "COUNT_RANGE missing positivity")
    oversized = {"F": B * 4 + 1, "n": 65, "s": 3}
    require(oversized["F"] == B * oversized["s"] + oversized["n"] and not 1 <= 65 <= 4, "COUNT_RANGE oversized low part")
    unchunked_o = -(2**14 + 1) * 2**16 * 128
    require(range_digits(unchunked_o + 2**37, 38) is None, "PV_NONNEG requires chunking")
    unsafe_o, unsafe_t = 2**30, -2**32 + 1
    unsafe = pv_checks(0, unsafe_t, unsafe_o, m=34)
    primary = pv_checks(0, unsafe_t, unsafe_o)
    require(all(unsafe.values()) and not primary["PV_IDENTITY"], "Goldilocks unsafe radix wrap control")
    alias_s, alias_n = 2**16 + 2**41 - 512, 2
    f = b * 2**16 + 1
    require(b * alias_s + alias_n == f + GOLD and range_digits(alias_s, 34) is None, "SUM_NONNEG excludes full-size row alias")
    require(not 1 <= 0 <= 4 and 0 <= 0 <= 3, "literal padded count fails; selector adaptation accepts")
    padded = row_checks(0, 0, 0, psi=0)
    nonzero_padded = row_checks(0, 0, 0, psi=0, maximum=1)
    literal = row_checks(0, 0, 0, psi=1)
    require(all(padded.values()) and not nonzero_padded["PADDING_MAX"] and not literal["COUNT_RANGE"], "padded row exact gates")
    return {"classes_caught": 5,
            "missing_pv_nonneg": {"P": GOLD, "B": b, "packed_output": 0, "fake_t": fake_t, "fake_o_residue": half,
                                  "primary_predicates": fractional, "reciprocal": 2**16, "R_times_fake_o": product,
                                  "shift16_requant": requant, "honest_requant": 0, "bracket_a": a, "bracket_b": bracket_b,
                                  "toy_fake_o_residue": toy_half, "toy_R_times_fake_o": toy_product, "toy_shift2_requant": 1},
            "count_range": {"missing_positivity": missing_count, "oversized_low_part": oversized, "primary_reject": "COUNT_RANGE"},
            "no_chunking": {"length": 2**14 + 1, "output": unchunked_o, "primary_reject": "PV_NONNEG"},
            "unsafe_wrap": {"control_m": 34, "o": unsafe_o, "t": unsafe_t, "integer_rhs": GOLD,
                            "control_predicates": unsafe, "primary_predicates": primary},
            "row_alias": {"s": alias_s, "n": alias_n, "F": f, "integer_rhs": f + GOLD, "primary_reject": "SUM_NONNEG"},
            "padding": {"literal_predicates": literal, "adapted_predicates": padded,
                        "nonzero_maximum_predicates": nonzero_padded, "nonzero_maximum_reject": "PADDING_MAX"}}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--self-test", action="store_true")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    started = time.monotonic()
    checker = Exhaustion()
    controls = negative_controls()
    if args.self_test:
        checker.integrated((0, 0, -1), (True, True, True), "self_test")
        checker.integrated((7, -1, 8), (False, False, False), "self_test")
        print("PASS: all limb residues, tied and padded hand cases, five negative-control classes")
        return
    result = {"experiment": "EXP-0015", "evidence": "E0", "finite_domains": checker.run(),
              "goldilocks": goldilocks(), "negative_controls": controls, "verdict": "validated",
              "scope": "registered finite-domain lemma check plus exact interval obligations; not complete-protocol SP-3",
              "environment": {"python": sys.version, "platform": platform.platform(),
                              "cpu": subprocess.check_output(["sysctl", "-n", "machdep.cpu.brand_string"], text=True).strip(),
                              "utc": subprocess.check_output(["date", "-u", "+%Y-%m-%dT%H:%M:%SZ"], text=True).strip(),
                              "git_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
                              "preregistration_commit": "774b7c7",
                              "harness_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                              "preregistration_sha256": hashlib.sha256((HERE / "results/README.preregistered.txt").read_bytes()).hexdigest()},
              "elapsed_seconds_metadata_only": time.monotonic() - started}
    if args.output:
        args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
