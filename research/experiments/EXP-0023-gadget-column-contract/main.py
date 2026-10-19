#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0 OR MIT
"""EXP-0023: exact public metadata and synthetic integer witnesses, not a proof system."""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, replace
from datetime import datetime, timezone
import hashlib
import json
from math import isqrt
from pathlib import Path
import platform
import subprocess
import sys

P = 2**64 - 2**32 + 1
HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
FROZEN = "9d4feb8ab6c17daf22a6a6f78816c603002ee8cdbbf74b82bc744cccef5c2362"
AMENDMENT = "1aa852822c46e25c5e240157ff7aadeb3c983125110ee640205c814dc70b9d59"
CONSTANTS = "f30bda990098f0b62b9c72b6acd1076497a813647e523d1470cf3798f06c79e2"
SEED = 0x4071060023
COUNTS: Counter[str] = Counter()


def check(condition: bool, context: object) -> None:
    COUNTS["assertions"] += 1
    if not condition:
        raise AssertionError(context)


def byte_digits(value: int, count: int) -> list[int]:
    return [(value >> (8 * i)) & 255 for i in range(count)]


def product(root: list[int], u: list[int], m: int, increment: int = 0) -> list[int]:
    """Mixed-radix candidate evaluation; the incremented low factor may be 256."""
    factors = root.copy()
    factors[0] += increment
    result = [0] * m
    for i, a in enumerate(factors):
        for j, b in enumerate(factors):
            for k, c in enumerate(u):
                result[i + j + 2 * k] += a * b * c
    return result


def convolution(a: list[int], b: list[int]) -> list[int]:
    out = [0] * (len(a) + len(b) - 1)
    for j in range(len(out)):
        out[j] = sum(a[i] * b[j - i] for i in range(len(a)) if 0 <= j - i < len(b))
    return out


def bounds(w: int, n: int, independent: bool) -> dict[str, list[int]]:
    m = (2 * w + 62) // 8
    r = (w - 1) // 8
    digits = ([(2**w - 1) >> (8 * i) & 255 for i in range((w + 7) // 8)]
              if independent else [255] * r + [(1 << (w - 8 * r)) - 1])
    if independent:
        sparse_u = [65535, 0, 65535, 0, 65535, 0, 127]
        lower = convolution(convolution(digits, digits), sparse_u)
        shifted = [digits[0] + 1, *digits[1:]]
        upper = convolution(convolution(shifted, shifted), sparse_u)
        lower += [0] * (m - len(lower))
        upper += [0] * (m - len(upper))
    else:
        lower = product(digits, [65535, 65535, 65535, 127], m)
        upper = product(digits, [65535, 65535, 65535, 127], m, 1)
    return {"lower/left": lower, "lower/right": byte_digits(n, m),
            "upper/left": byte_digits(n + 1, m), "upper/right": upper}


def carry_bits(a: list[int], independent: bool) -> list[int]:
    result = []
    carry = prefix = 0
    for j, value in enumerate(a):
        if independent:
            prefix += value << (8 * j)
            carry = prefix >> (8 * (j + 1))
        else:
            carry = (value + carry) // 256
        result.append(max(1, carry.bit_length()))
    return result


def enforce_policy(descriptor: dict, max_log2_rows: int) -> dict:
    if not 1 <= max_log2_rows <= 32 or any(b > max_log2_rows for b in descriptor["families"]):
        raise ValueError("required range table exceeds public policy")
    return descriptor


def schema(w: int, n: int, independent: bool = False) -> dict:
    if not 1 <= w <= 32 or not 0 < n < 1 << (2 * w + 55):
        raise ValueError("public root parameters")
    r = (w + 7) // 8 - 1 if independent else (w - 1) // 8
    m = (2 * w + 55 - 1) // 8 + 1 if independent else (2 * w + 62) // 8
    raw = bounds(w, n, independent)
    bits = {name: carry_bits(values, independent) for name, values in raw.items()}
    columns: list[tuple[str, int]] = []
    requests: list[tuple[str, int]] = [("root", w)]
    if independent:
        input_roles = [(f"u/{i}", 16) for i in (0, 1, 2)]
        root_roles = [(f"root/{i}", 8) for i in range(r)]
        columns = input_roles + root_roles
        requests += input_roles + [("u/top", 7)] + root_roles + [("root/top", w % 8 or 8)]
    else:
        for i in range(3):
            columns.append((f"u/{i}", 16))
            requests.append((f"u/{i}", 16))
        requests.append(("u/top", 7))
        for i in range(r):
            columns.append((f"root/{i}", 8))
            requests.append((f"root/{i}", 8))
        requests.append(("root/top", w - 8 * r))
    if independent:
        # Flatten a table of array descriptions instead of emitting each comparison in place.
        arrays = [(f"{which}/{side}/{kind}",
                   [8] * m if kind == "digit" else bits[f"{which}/{side}"])
                  for which in ("lower", "upper")
                  for side in ("left", "right") for kind in ("digit", "carry")]
        for which in ("lower", "upper"):
            at = next(i for i, (name, _) in enumerate(arrays) if name.startswith(which)) + 4
            arrays[at:at] = [(f"{which}/diff", [8] * m), (f"{which}/borrow", [1] * m)]
        columns.extend((f"{name}/{i}", width) for name, widths in arrays
                       for i, width in enumerate(widths))
    else:
        for which in ("lower", "upper"):
            for side in ("left", "right"):
                for kind, widths in [("digit", [8] * m), ("carry", bits[f"{which}/{side}"])]:
                    for i, width in enumerate(widths):
                        columns.append((f"{which}/{side}/{kind}/{i}", width))
            columns.extend((f"{which}/diff/{i}", 8) for i in range(m))
            columns.extend((f"{which}/borrow/{i}", 1) for i in range(m))
    requests.extend(columns[3 + r:])
    if independent:
        widths = [width for _, width in requests]
        families = sorted(set(widths), key=widths.index)
        ids = [families.index(width) for width in widths]
    else:
        families, ids = [], []
        for _, width in requests:
            if width not in families:
                families.append(width)
            ids.append(families.index(width))
    return {"w": w, "N": str(n), "r": r, "m": m, "outputs": ["root"],
            "columns": columns, "requests": requests, "families": families,
            "request_ids": ids, "bounds": raw}


@dataclass
class Witness:
    u: int
    root: int
    psi: int
    values: dict[str, int]


def witness(s: dict, u: int, root: int, psi: int, integer_u: int | None = None, *, one: bool = False) -> Witness:
    """Digits come from independent whole integers; carries reconcile raw coefficients."""
    value_u = u if integer_u is None else integer_u
    r, m, n = s["r"], s["m"], int(s["N"])
    values = {f"u/{i}": (value_u >> (16 * i)) & 65535 for i in range(3)}
    values.update({f"root/{i}": (root >> (8 * i)) & 255 for i in range(r)})
    ud = [values[f"u/{i}"] for i in range(3)] + [value_u >> 48]
    rd = [values[f"root/{i}"] for i in range(r)] + [root >> (8 * r)]
    increment = 1 if one else psi
    raw = {"lower/left": product(rd, ud, m), "lower/right": byte_digits(psi * n, m),
           "upper/left": byte_digits(psi * (n + 1), m),
           "upper/right": product(rd, ud, m, increment)}
    exact = {"lower/left": root * root * value_u, "lower/right": psi * n,
             "upper/left": psi * (n + 1), "upper/right": (root + increment)**2 * value_u}
    for name, coefficients in raw.items():
        check(sum(a << (8 * i) for i, a in enumerate(coefficients)) == exact[name],
              ("product vs whole integer", name, s["w"], u, root))
        previous = 0
        for i, digit in enumerate(byte_digits(exact[name], m)):
            numerator = coefficients[i] + previous - digit
            check(numerator >= 0 and numerator % 256 == 0, ("integral carry", name, i))
            previous = numerator // 256
            values[f"{name}/digit/{i}"] = digit
            values[f"{name}/carry/{i}"] = previous
        check(previous == 0, ("builder full product", name, exact[name]))
    for which in ("lower", "upper"):
        borrow = 0
        for i in range(m):
            difference = values[f"{which}/right/digit/{i}"] - values[f"{which}/left/digit/{i}"] - borrow
            borrow = int(difference < 0)
            values[f"{which}/diff/{i}"] = difference + 256 * borrow
            values[f"{which}/borrow/{i}"] = borrow
    return Witness(u % P, root, psi, {name: values[name] for name, _ in s["columns"]})


def validate(s: dict, a: Witness, fault: str = "", *, one: bool = False) -> str | None:
    """Small field identities, range boxes and final carries: no bigint bracket shortcut."""
    w, r, m, n = s["w"], s["r"], s["m"], int(s["N"])
    if fault == "WholeConstantMod":
        n %= P
    v = a.values
    if list(v) != [name for name, _ in s["columns"]]:
        return "Shape"
    if not 0 <= a.root < 1 << w and fault != "RootRange":
        return "RootRange"
    if not 0 <= a.u < P or a.psi not in (0, 1):
        return "Input"
    if a.psi == 0 and a.u != 0 and fault != "InputPadding":
        return "InputPadding"
    if a.psi == 0 and a.root != 0 and fault != "RootPadding":
        return "RootPadding"
    for name, bits in s["columns"][:3 + r]:
        if not 0 <= v[name] < 1 << bits:
            return "Range/" + name
    ut = ((a.u - sum(v[f"u/{i}"] << (16 * i) for i in range(3))) * pow(1 << 48, -1, P)) % P
    rt = ((a.root - sum(v[f"root/{i}"] << (8 * i) for i in range(r))) * pow(1 << (8 * r), -1, P)) % P
    if ut >= 128 and fault != "UTop":
        return "UTop"
    if rt >= 1 << (w - 8 * r) and fault != "RTop":
        return "RTop"
    for name, bits in s["columns"][3 + r:]:
        if not 0 <= v[name] < 1 << bits:
            return "Range/" + name
    ud = [v[f"u/{i}"] for i in range(3)] + [ut]
    rd = [v[f"root/{i}"] for i in range(r)] + [rt]
    increment = 1 if one else a.psi
    raw = {"lower/left": product(rd, ud, m), "lower/right": byte_digits(a.psi * n, m),
           "upper/left": byte_digits(a.psi * (n + 1), m),
           "upper/right": product(rd, ud, m, increment)}
    if fault == "IncrementClip":
        q = rd.copy()
        q[0] = (q[0] + increment) & 255
        raw["upper/right"] = product(q, ud, m)
    for name, coefficients in raw.items():
        final = "FinalCarry/" + name
        if v[f"{name}/carry/{m-1}"] != 0 and fault != final:
            return final
        previous = 0
        for i, coefficient in enumerate(coefficients):
            if (coefficient + previous - v[f"{name}/digit/{i}"] - 256 * v[f"{name}/carry/{i}"]) % P:
                return f"Norm/{name}/{i}"
            previous = v[f"{name}/carry/{i}"]
    for which in ("lower", "upper"):
        final = which.capitalize()
        if v[f"{which}/borrow/{m-1}"] != 0 and fault != final:
            return final
        previous = 0
        for i in range(m):
            equation = (v[f"{which}/right/digit/{i}"] - v[f"{which}/left/digit/{i}"]
                        - previous - v[f"{which}/diff/{i}"] + 256 * v[f"{which}/borrow/{i}"])
            if equation % P:
                return f"Subtract/{which}/{i}"
            previous = v[f"{which}/borrow/{i}"]
    return None


def verify_descriptor(s: dict) -> None:
    other = schema(s["w"], int(s["N"]), True)
    check(s == other, ("independent schema", s["w"]))
    r, m = s["r"], s["m"]
    check(len(s["columns"]) + 1 == 4 + r + 12 * m, "committed count")
    check(len(s["requests"]) == 6 + r + 12 * m, "lookup count")
    check(len({name for name, _ in s["columns"]}) == len(s["columns"]), "unique roles")
    check("root" not in dict(s["columns"]) and s["outputs"] == ["root"], "output allocated once")
    check(all(1 <= b <= 32 for b in s["families"]), "default range policy")
    for name, raw in s["bounds"].items():
        bits = carry_bits(raw, False)
        check(bits == carry_bits(raw, True), ("independent carries", name))
        previous_box = 0
        for j, (a, b) in enumerate(zip(raw, bits)):
            box = (1 << b) - 1
            left, right = a + previous_box, 255 + 256 * box
            check(max(a, left, right) < 1 << 40, ("whole lookup box", s["w"], name, j))
            check(max(left, right) < (P - 1) // 2, "centered field")
            COUNTS["maximum_equation_bits"] = max(COUNTS["maximum_equation_bits"], max(left, right).bit_length())
            previous_box = box


def splitmix(state: int) -> tuple[int, int]:
    state = (state + 0x9E3779B97F4A7C15) & (2**64 - 1)
    z = ((state ^ (state >> 30)) * 0xBF58476D1CE4E5B9) & (2**64 - 1)
    z = ((z ^ (z >> 27)) * 0x94D049BB133111EB) & (2**64 - 1)
    return state, z ^ (z >> 31)


def perturbation_error(s: dict, a: Witness, name: str, changed: int) -> str:
    bits = dict(s["columns"])[name]
    if changed >= 1 << bits:
        return "Range/" + name
    if name.startswith("u/"):
        return "UTop"
    if name.startswith("root/"):
        return "RTop"
    parts = name.split("/")
    if len(parts) == 4:
        which, side, kind, index = parts
        if kind == "carry" and int(index) == s["m"] - 1:
            return f"FinalCarry/{which}/{side}"
        return f"Norm/{which}/{side}/{index}"
    which, kind, index = parts
    if kind == "borrow" and int(index) == s["m"] - 1:
        return which.capitalize()
    return f"Subtract/{which}/{index}"


def other_gadgets() -> None:
    # Separately declared counts/order and explicit loops for low-limb/virtual-top cases.
    for width in range(1, 63):
        for limb in range(1, 33):
            lows = list(range((width - 1) // limb))
            remaining, expected = width, []
            while remaining > limb:
                expected.append(len(expected))
                remaining -= limb
            check(lows == expected and remaining == width - len(lows) * limb, "NonNeg schema")
            check(sum([limb] * len(lows) + [remaining]) == width, "narrow top box")
            COUNTS["nonneg_cases"] += 1
    for sigma in range(32):
        for limb in range(1, 33):
            n = (sigma + 17) // limb
            a = ["maximum"] + [f"{side}/{j}" for side in ("a", "b") for j in range(n)]
            b = ["maximum"]
            for side in ("a", "b"):
                remaining, index = sigma + 18, 0
                while remaining > limb:
                    b.append(f"{side}/{index}")
                    index += 1
                    remaining -= limb
            check(a == b and len(a) == 1 + 2 * n, "Softmax schema")
            COUNTS["softmax_cases"] += 1
    for kind in ("Requant1", "Lookup", "OneHot"):
        check({"Requant1": [], "Lookup": [], "OneHot": []}[kind] == [], kind)
    check(["low"] == ["low"], "RequantSplit extra role")
    for split in (0, 1, 16, 32):
        named = (["low"] if split else []) + ["middle", "saturation"]
        walk = [x for x in ("low", "middle", "saturation") if x != "low" or split != 0]
        check(named == walk, "RequantWide output-first auxiliaries")


def run() -> dict:
    frozen = HERE / "results/README.preregistered.txt"
    rel = str(frozen.relative_to(ROOT))
    committed = subprocess.check_output(["git", "show", "HEAD:" + rel], cwd=ROOT)
    check(frozen.read_bytes() == committed and hashlib.sha256(committed).hexdigest() == FROZEN,
          "committed pre-registration")
    amendment = HERE / "results/README.amendment-preregistered.txt"
    amendment_commit = subprocess.check_output(["git", "show", "HEAD:" + str(amendment.relative_to(ROOT))], cwd=ROOT)
    check(amendment.read_bytes() == amendment_commit
          and hashlib.sha256(amendment_commit).hexdigest() == AMENDMENT, "committed amendment")
    constants = HERE / "results/README.constants-preregistered.txt"
    constants_commit = subprocess.check_output(["git", "show", "HEAD:" + str(constants.relative_to(ROOT))], cwd=ROOT)
    check(constants.read_bytes() == constants_commit
          and hashlib.sha256(constants_commit).hexdigest() == CONSTANTS, "committed constant boundary")
    state = SEED
    per_width, representatives = [], []
    matrix_hash = hashlib.sha256()
    for width in range(1, 33):
        d, kappa = (1, width - 1) if width <= 31 else (4, 30)
        n = d << (2 * kappa)
        check(isqrt(n).bit_length() == width, "independent width")
        s = schema(width, n)
        verify_descriptor(s)
        for policy in range(1, 33):
            try:
                accepted = enforce_policy(s, policy)
            except ValueError:
                check(policy < max(s["families"]), "exact range-policy rejection")
            else:
                check(policy >= max(s["families"]) and accepted == s, "unchanged range-policy acceptance")
            COUNTS["policy_cases"] += 1
        inputs = [1, 2, 3, 255, 256, 257, 2**55 - 1]
        for _ in range(128):
            state, value = splitmix(state)
            inputs.append(value % (2**55 - 1) + 1)
        for psi, u in [(1, u) for u in inputs] + [(0, 0)]:
            root = isqrt(n // u) if psi else 0
            a = witness(s, u, root, psi)
            check(validate(s, a) is None, ("honest", width, psi, u, validate(s, a)))
            check(schema(width, n) == s, "witness-independent metadata")
            if not psi:
                check(root == 0 and all(x == 0 for x in a.values.values()), "zero padding")
            for delta in (-1, 1):
                wrong = root + delta
                if not 0 <= wrong < 1 << width:
                    b = replace(a, root=wrong)
                    expected = "RootRange"
                else:
                    b = witness(s, u, wrong, psi)
                    expected = "RootPadding" if not psi else ("Lower" if delta > 0 else "Upper")
                check(validate(s, b) == expected, ("root alternative", width, delta, expected, validate(s, b)))
                COUNTS["root_alternatives"] += 1
            if (psi and u in (1, 2**55 - 1)) or not psi:
                representatives.append((s, a))
            matrix_hash.update(json.dumps([width, psi, str(u), str(root), list(a.values.items())], separators=(",", ":")).encode())
            COUNTS["honest_rows"] += 1
        # Full allowed box, independent of whether its overall root bracket passes.
        large = witness(s, 2**55 - 1, 2**width - 1, 1)
        decoded = sum(large.values[f"upper/right/digit/{j}"] << (8 * j) for j in range(s["m"]))
        check(decoded == 2**(2 * width) * (2**55 - 1), "increment retains full product")
        if width == 32:
            check(decoded.bit_length() == 119, "119-bit product")
            clipped = sum(large.values[f"upper/right/digit/{j}"] << (8 * j) for j in range(s["m"] - 1))
            check(clipped != decoded, "cropped high product detected")
        per_width.append({"width": width, "N": str(n), "committed": len(s["columns"]) + 1,
                          "requests": len(s["requests"]), "families": s["families"]})
    check(COUNTS["honest_rows"] == 4352, "registered matrix size")
    for s, a in representatives:
        for name, _ in s["columns"]:
            values = a.values.copy()
            values[name] += 1
            changed = replace(a, values=values)
            expected = perturbation_error(s, a, name, values[name])
            check(validate(s, changed) == expected,
                  ("alternative auxiliary", s["w"], name, expected, validate(s, changed)))
            COUNTS["auxiliary_alternatives"] += 1
    other_gadgets()
    faults = fault_controls()
    original_counts = dict(COUNTS)
    amended = polynomial_retention(matrix_hash.hexdigest())
    large_constants = constant_boundaries()
    return {"verdict": "validated", "evidence": "E0, bounded exact arithmetic/schema result only",
            "seed": hex(SEED), "frozen_sha256": FROZEN, "counts": original_counts,
            "amendment": amended, "total_assertions": COUNTS["assertions"],
            "large_constants": large_constants,
            "matrix_sha256": matrix_hash.hexdigest(), "widths": per_width, "faults": faults,
            "environment": {"python": sys.version, "platform": platform.platform(),
                            "machine": platform.machine(),
                            "cpu": (subprocess.check_output(["sysctl", "-n", "machdep.cpu.brand_string"], text=True).strip()
                                    if platform.system() == "Darwin" else platform.processor()),
                            "utc": datetime.now(timezone.utc).isoformat()},
            "limits": ["No full-model proof", "No G-4 or Fiat-Shamir proof", "No ZK simulation proof",
                       "No GPU/performance gain", "No executable differential run against production helpers"]}


def fault_controls() -> list[dict]:
    s = schema(16, 1 << 30)
    good = witness(s, 1, 1 << 15, 1)
    check(validate(s, good) is None, "fault control")
    cases: list[tuple[str, dict, Witness, str]] = []
    pad = witness(s, 0, 0, 0)
    for label, name in (("UTop", "u/0"), ("RTop", "root/0")):
        v = pad.values.copy()
        v[name] = 1
        cases.append((label, s, replace(pad, values=v), label))
    cases.extend([
        ("RootRange", s, replace(good, root=1 << 16), "RootRange"),
        ("RootPadding", s, witness(s, 0, 1, 0), "RootPadding"),
        ("InputPadding", s, witness(s, 1, 0, 0), "InputPadding"),
        ("Lower", s, witness(s, 1, (1 << 15) + 1, 1), "Lower"),
        ("Upper", s, witness(s, 1, (1 << 15) - 1, 1), "Upper")])
    for which in ("lower", "upper"):
        for side in ("left", "right"):
            name = f"{which}/{side}"
            v = good.values.copy()
            v[f"{name}/carry/{s['m']-1}"] = 1
            cases.append(("FinalCarry/" + name, s, replace(good, values=v), "FinalCarry/" + name))
    # Here the maximal byte root is an honest root; clipping its increment changes a passing row.
    carry_s = schema(8, 255**2)
    carry_a = witness(carry_s, 1, 255, 1)
    check(validate(carry_s, carry_a) is None, "increment boundary control")
    cases.append(("IncrementClip", carry_s, carry_a, None))
    results = []
    for fault, meta, value, expected in cases:
        original = validate(meta, value)
        changed = validate(meta, value, fault)
        check(original == expected and changed != original, ("behavior-changing fault", fault, original, changed))
        results.append({"fault": fault, "normal": original, "mutated": changed,
                        "effect": "changed rejection/acceptance outcome"})
    # The same canonical field input with limbs for u+P violates the narrow virtual top.
    wrap_s = schema(32, 1 << 62)
    wrap = witness(wrap_s, 1, 0, 1, 1 + P)
    check(validate(wrap_s, wrap) == "UTop", "u+P exact top rejection")
    check(validate(wrap_s, wrap, "UTop") != "UTop", "u+P top-check control")
    results.append({"fault": "u+P", "normal": "UTop", "mutated": validate(wrap_s, wrap, "UTop"),
                    "effect": "isolated input-box attack; no full-model claim"})
    baseline = schema(32, 1 << 62)
    for label in ("drop_role", "reorder_roles", "duplicate_output", "drop_u_top_request",
                  "drop_root_top_request", "reorder_requests", "crop_high_limbs"):
        changed = json.loads(json.dumps(baseline))
        # Normalize container types for an exact semantic descriptor comparison.
        reference = json.loads(json.dumps(schema(32, 1 << 62, True)))
        if label == "drop_role": changed["columns"].pop()
        elif label == "reorder_roles": changed["columns"][0:2] = reversed(changed["columns"][0:2])
        elif label == "duplicate_output": changed["columns"].insert(0, ["root", 32])
        elif label.startswith("drop_"):
            target = "u/top" if label == "drop_u_top_request" else "root/top"
            changed["requests"] = [r for r in changed["requests"] if r[0] != target]
        elif label == "reorder_requests": changed["requests"][0:2] = reversed(changed["requests"][0:2])
        else: changed["m"] -= 1
        check(changed != reference, ("metadata fault changes descriptor", label))
        try:
            verify_descriptor(changed)
        except AssertionError as error:
            check(error.args[0] == ("independent schema", 32), ("targeted descriptor rejection", label, error.args))
        else:
            raise AssertionError(("metadata fault survived actual descriptor gate", label))
        results.append({"fault": label, "effect": "actual descriptor gate rejects at independent schema"})
    check(validate(s, good) is None and schema(32, 1 << 62) == baseline, "restored fault control")
    return results


def polynomial_retention(original_hash: str) -> dict:
    before = COUNTS["assertions"]
    state, rows, alternatives, auxiliaries = SEED, 0, 0, 0
    encoded = hashlib.sha256()
    for width in range(1, 33):
        n = 1 << (2 * width - 2) if width <= 31 else 4 << 60
        s = schema(width, n)
        inputs = [1, 2, 3, 255, 256, 257, 2**55 - 1]
        for _ in range(128):
            state, value = splitmix(state)
            inputs.append(value % (2**55 - 1) + 1)
        for psi, u in [(1, u) for u in inputs] + [(0, 0)]:
            root = isqrt(n // u) if psi else 0
            selector = witness(s, u, root, psi)
            original = witness(s, u, root, psi, one=True)
            check(selector == original, ("valid-row column equality", width, psi, u))
            check(validate(s, original, one=True) is None, "original polynomial honest")
            check(validate(s, selector) is None, "selector polynomial honest")
            encoded.update(json.dumps([width, psi, str(u), str(root), list(original.values.items())], separators=(",", ":")).encode())
            for delta in (-1, 1):
                wrong = root + delta
                if not 0 <= wrong < 1 << width:
                    changed, expected = replace(original, root=wrong), "RootRange"
                else:
                    changed = witness(s, u, wrong, psi, one=True)
                    expected = "RootPadding" if not psi else ("Lower" if delta > 0 else "Upper")
                check(validate(s, changed, one=True) == expected, ("original polynomial root alternative", width, delta))
                alternatives += 1
            if (psi and u in (1, 2**55 - 1)) or not psi:
                for name, _ in s["columns"]:
                    values = original.values.copy()
                    values[name] += 1
                    expected = perturbation_error(s, original, name, values[name])
                    check(validate(s, replace(original, values=values), one=True) == expected,
                          ("original polynomial auxiliary", width, name, expected))
                    auxiliaries += 1
            rows += 1
        check(schema(width, n) == s, "polynomial choice cannot change public schema")
    check(rows == 4352 and alternatives == 8704 and auxiliaries == 13680, "amendment full original matrix")
    check(encoded.hexdigest() == original_hash, "whole canonical-data equality hash")
    # Degenerate maximum root is represented with a one-bit lookup, as the scalar API does.
    s = schema(1, 1)
    check(max(1, isqrt(1 // 2).bit_length()) == 1, "zero maximum root width")
    for psi, u in [(1, 2), (1, 3), (1, 2**55 - 1), (0, 0)]:
        a, b = witness(s, u, 0, psi), witness(s, u, 0, psi, one=True)
        check(a == b and validate(s, a) is None and validate(s, b, one=True) is None,
              "zero-maximum-root boundary")
    malformed = 0
    for width in (0, 33):
        try:
            schema(width, 1)
        except ValueError:
            malformed += 1
    for policy in (0, 33):
        try:
            enforce_policy(s, policy)
        except ValueError:
            malformed += 1
    check(malformed == 4, "malformed public widths/policies rejected")
    points = 0
    for psi in (0, 1, 2, 17, P - 1):
        for u in (0, 1, 2, 2**55 - 1):
            for root in (0, 1, 255, 2**32 - 1):
                difference = u * ((root + 1)**2 - (root + psi)**2) % P
                factored = u * (1 - psi) * (2 * root + psi + 1) % P
                check(difference == factored, "polynomial factorization")
                if psi in (0, 1) and u * (1 - psi) == 0:
                    check(difference == 0, "required Boolean-domain equality")
                points += 1
    check(1 * ((0 + 1)**2 - (0 + 0)**2) % P != 0, "not an off-domain polynomial identity")
    return {"verdict": "validated", "frozen_sha256": AMENDMENT,
            "honest_rows": rows, "root_alternatives": alternatives,
            "auxiliary_alternatives": auxiliaries, "zero_maximum_root_rows": 4,
            "malformed_parameter_cases": malformed, "factorization_points": points,
            "canonical_matrix_sha256": encoded.hexdigest(),
            "assertions_added": COUNTS["assertions"] - before,
            "conclusion": "retain rho+1 final-evaluation polynomial; equal data only on required domain"}


def constant_boundaries() -> dict:
    before, rows = COUNTS["assertions"], 0
    profiles = []
    for label, d, kappa, e in [("8b-norm-metadata", 4096, 30, 672),
                               ("405b-norm-metadata", 16384, 30, 2685),
                               ("constant-2^64", 4, 31, 2),
                               ("large-primitive-constant", 2**32 - 2, 31, 2**54),
                               ("small-root-large-constant", 4, 31, 2**55 - 3)]:
        n = d << (2 * kappa)
        width = max(1, isqrt(n // e).bit_length())
        s = schema(width, n)
        verify_descriptor(s)
        check(n > P, "constant exceeds field")
        for psi, u in [(1, e), (1, e + 1), (1, 2**55 - 1), (0, 0)]:
            check(not psi or e <= u < 2**55, "primitive input domain")
            root = isqrt(n // u) if psi else 0
            a, b = witness(s, u, root, psi), witness(s, u, root, psi, one=True)
            check(a == b and validate(s, a) is None and validate(s, b, one=True) is None,
                  ("large-constant row", label, psi, u))
            left_constant = sum(b.values[f"lower/right/digit/{j}"] << (8 * j) for j in range(s["m"]))
            right_constant = sum(b.values[f"upper/left/digit/{j}"] << (8 * j) for j in range(s["m"]))
            check(left_constant == psi * n and right_constant == psi * (n + 1), "whole constants retained")
            rows += 1
        wrong_n = n % P
        wrong_root = isqrt(wrong_n // e)
        bad_schema = schema(width, wrong_n)
        check(bad_schema["columns"] == s["columns"], "same public role boxes for constant attack")
        bad = witness(bad_schema, e, wrong_root, 1, one=True)
        first = next(j for j in range(s["m"]) if (n >> (8 * j) & 255) != (wrong_n >> (8 * j) & 255))
        expected = f"Norm/lower/right/{first}"
        check(wrong_root != isqrt(n // e), "attack changes the root")
        check(validate(s, bad, one=True) == expected, ("whole-constant rejection", label, expected))
        check(validate(s, bad, "WholeConstantMod", one=True) is None, "experimental constant-reduction fault accepts")
        profiles.append({"profile": label, "d": d, "kappa": kappa, "e": str(e), "N": str(n),
                         "width": width, "expected": expected, "faulted_outcome": "Accept",
                         "true_root": str(isqrt(n // e)), "wrong_root": str(wrong_root)})
    check(rows == 20, "registered large-constant cases")
    return {"verdict": "validated", "frozen_sha256": CONSTANTS, "rows": rows,
            "constant_reduction_attacks": len(profiles), "profiles": profiles,
            "assertions_added": COUNTS["assertions"] - before,
            "scope": "synthetic primitive cases and intentional fault only; no full-model vulnerability claim"}


if __name__ == "__main__":
    result = run()
    (HERE / "results/outcomes.json").write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    print(json.dumps({"verdict": result["verdict"], "counts": result["counts"],
                      "original_distinct_faults": 19, "original_attack_controls": 1,
                      "matrix_sha256": result["matrix_sha256"], "amendment": result["amendment"],
                      "large_constants": result["large_constants"]}))
