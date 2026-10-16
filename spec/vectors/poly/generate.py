#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0 OR MIT
"""Reference vectors for the multilinear extensions of RFC-0010 'Multilinear extensions':
evaluations of dense tables at a point, binding of the lowest variable, eq tables, the
less-than predicates and the causal selector, and round-polynomial interpolation.

This script is the independent oracle. Its arithmetic is Python's arbitrary-precision integers,
with Fp2 = Fp[X]/(X^2 - 7) multiplied by the schoolbook formula of RFC-0002, and every value is
computed from its definition: f(r) as the sum of f[x] eq(r, x) over the hypercube, eq tables
entry by entry, LT, LT2 and the causal selector by enumerating the integers or the pairs they
sum over, and p(r) by Horner's rule on the coefficients the round polynomial was drawn from. It
shares no code or algorithm with crates/tessera-poly, which checks its output.

Usage: python3 spec/vectors/poly/generate.py
(rewrites poly.json)

Inputs come from SplitMix64 with a fixed seed, so the file is reproducible byte for byte.
"""

import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
SEED = 0x7E55E7A_2026_0046  # fixed; changing it changes every vector
P = 2**64 - 2**32 + 1  # Goldilocks (RFC-0002 'Constants')
W = 7  # Fp2 = Fp[X]/(X^2 - W)

# Which sizes get which vectors; poly.json stays near 300 KB.
FP_TABLE_VARS = list(range(11)) + [12]  # Fp tables: 2^0 .. 2^10 and 2^12 entries
FP2_TABLE_VARS = list(range(10))  # Fp2 tables: 2^0 .. 2^9 entries
BOUND_MAX_VARS = 8  # the bound table is given for tables of at most 2^8 entries
EQ_TABLE_VARS = list(range(11))  # eq tables for points of 0 .. 10 coordinates
LT_ALL_VARS = list(range(5))  # LT(r, a) for every a in 0 ..= 2^n
LT_SAMPLED_VARS = [5, 8, 12, 16, 32, 63, 64, 65, 80]  # LT(r, a) at sampled a
LT_ENUMERATED_MAX = 16  # above this, LT comes from its dyadic decomposition
LT2_VARS = list(range(11))  # LT2 and the causal selector, by enumerating 4^n pairs
ROUND_DEGREES = list(range(1, 8))
ROUND_CASES = 4


def require(condition, message):
    """A check that holds under `python -O` too, unlike `assert`."""
    if not condition:
        raise SystemExit(f"generate.py: {message}")


def splitmix64(state):
    """Yields 64-bit outputs of SplitMix64 (Steele, Lea, Flood 2014) from `state`."""
    mask = (1 << 64) - 1
    while True:
        state = (state + 0x9E3779B97F4A7C15) & mask
        z = state
        z = ((z ^ (z >> 30)) * 0xBF58476D1CE4E5B9) & mask
        z = ((z ^ (z >> 27)) * 0x94D049BB133111EB) & mask
        yield z ^ (z >> 31)


# ----- Fp2 = Fp[X]/(X^2 - 7) on pairs (c0, c1); Fp embeds as (x, 0) ---------------------------

ZERO, ONE = (0, 0), (1, 0)


def add(a, b):
    return ((a[0] + b[0]) % P, (a[1] + b[1]) % P)


def sub(a, b):
    return ((a[0] - b[0]) % P, (a[1] - b[1]) % P)


def mul(a, b):
    """(a0 + a1 X)(b0 + b1 X) with X^2 = W, schoolbook."""
    return ((a[0] * b[0] + W * a[1] * b[1]) % P, (a[0] * b[1] + a[1] * b[0]) % P)


def lift(x):
    return (x % P, 0)


def total(values):
    out = ZERO
    for v in values:
        out = add(out, v)
    return out


class Inputs:
    """Uniform field elements from SplitMix64, by rejecting words at or above P."""

    def __init__(self, seed):
        self.words = splitmix64(seed)

    def fp(self):
        while True:
            x = next(self.words)
            if x < P:
                return x

    def fp2(self):
        return (self.fp(), self.fp())

    def point(self, n):
        return [self.fp2() for _ in range(n)]

    def below(self, bound):
        """A uniform integer in [0, bound), bound <= 2^64, by rejection."""
        while True:
            x = next(self.words)
            if x < bound * ((1 << 64) // bound):
                return x % bound


# ----- Definitions (RFC-0010 'Multilinear extensions') -----------------------------------------


def bit(x, i):
    return (x >> i) & 1


def eq_at(r, x):
    """eq(r, x) for an integer x read as the Boolean point (x_1, ..., x_n), x_1 its lowest bit:
    the product of r_i where x_i = 1 and 1 - r_i where x_i = 0."""
    out = ONE
    for i, ri in enumerate(r):
        out = mul(out, ri if bit(x, i) else sub(ONE, ri))
    return out


def eq_points(a, b):
    """eq(a, b) = prod_i (a_i b_i + (1 - a_i)(1 - b_i)) for two field points."""
    out = ONE
    for ai, bi in zip(a, b):
        out = mul(out, add(mul(ai, bi), mul(sub(ONE, ai), sub(ONE, bi))))
    return out


def mle(table, r):
    """f(r) = sum over the hypercube of f[x] eq(r, x); entries are Fp2 pairs."""
    require(len(table) == 1 << len(r), "a table of 2^n entries takes n coordinates")
    return total(mul(table[x], eq_at(r, x)) for x in range(len(table)))


def bind_lowest(table, r1):
    """f'[k] = f[2k] + r1 (f[2k+1] - f[2k]): the table of f(r1, x_2, ..., x_n)."""
    return [add(table[2 * k], mul(r1, sub(table[2 * k + 1], table[2 * k]))) for k in range(len(table) // 2)]


def lt_enumerated(r):
    """[LT(r, a) for a in 0 ..= 2^n], LT(r, a) = sum of eq(r, h) over the integers h < a."""
    out, running = [ZERO], ZERO
    for h in range(1 << len(r)):
        running = add(running, eq_at(r, h))
        out.append(running)
    return out


def lt_dyadic(r, a):
    """LT(r, a) from the dyadic blocks of [0, a): for each set bit j of a, the integers that
    agree with a above bit j and have bit j clear. The eq weights of a block's free low bits sum
    to 1, so the block weighs (1 - r_j) times the eq factors of a's bits above j. Coordinates at
    or beyond bit 64 see zero bits."""
    n = len(r)
    if a == 1 << n:
        return ONE
    out = ZERO
    for j in range(n):
        if not bit(a, j):
            continue
        block = sub(ONE, r[j])
        for i in range(j + 1, n):
            block = mul(block, r[i] if bit(a, i) else sub(ONE, r[i]))
        out = add(out, block)
    return out


def pair_sums(q, i):
    """By enumerating all 4^n pairs (q', i'): the sums of eq(q, q') eq(i, i') over i' < q', over
    q' < i' and over i' = q'."""
    n = len(q)
    eq_q = [eq_at(q, x) for x in range(1 << n)]
    eq_i = [eq_at(i, x) for x in range(1 << n)]
    # Unreduced integer sums of the schoolbook products, reduced once at the end.
    sums = {"lt_iq": [0, 0], "lt_qi": [0, 0], "eq": [0, 0]}
    for qp, (a0, a1) in enumerate(eq_q):
        for ip, (b0, b1) in enumerate(eq_i):
            key = "lt_iq" if ip < qp else "lt_qi" if qp < ip else "eq"
            sums[key][0] += a0 * b0 + W * a1 * b1
            sums[key][1] += a0 * b1 + a1 * b0
    return {key: (c0 % P, c1 % P) for key, (c0, c1) in sums.items()}


def horner(coefficients, x):
    out = ZERO
    for c in reversed(coefficients):
        out = add(mul(out, x), c)
    return out


# ----- Rendering ---------------------------------------------------------------------------------


def hex64(x):
    return f"0x{x:016x}"


def fp(x):
    return hex64(x)


def fp2(a):
    return [hex64(a[0]), hex64(a[1])]


def points(values):
    return [fp2(v) for v in values]


def section(cases):
    return "[\n" + ",\n".join(json.dumps(case, separators=(",", ":")) for case in cases) + "\n  ]"


# ----- Vectors -----------------------------------------------------------------------------------


def multilinear_cases(rng):
    cases = []
    for field, sizes in (("Fp", FP_TABLE_VARS), ("Fp2", FP2_TABLE_VARS)):
        for n in sizes:
            if field == "Fp":
                raw = [rng.fp() for _ in range(1 << n)]
                table, shown = [lift(x) for x in raw], [fp(x) for x in raw]
            else:
                table = [rng.fp2() for _ in range(1 << n)]
                shown = points(table)
            r = rng.point(n)
            value = mle(table, r)
            case = {"field": field, "n": n, "table": shown, "point": points(r), "value": fp2(value)}
            if 1 <= n <= BOUND_MAX_VARS:
                bound = bind_lowest(table, r[0])
                require(mle(bound, r[1:]) == value, "binding x_1 then evaluating is evaluating")
                case["bound"] = points(bound)
            cases.append(case)
    return cases


def eq_table_cases(rng):
    cases = []
    for n in EQ_TABLE_VARS:
        r = rng.point(n)
        table = [eq_at(r, x) for x in range(1 << n)]
        require(total(table) == ONE, "the eq weights of a point sum to 1")
        cases.append({"n": n, "point": points(r), "table": points(table)})
    return cases


def less_than_cases(rng):
    cases = []
    for n in LT_ALL_VARS + LT_SAMPLED_VARS:
        r = rng.point(n)
        cube = 1 << n
        if n in LT_ALL_VARS:
            bounds = list(range(cube + 1))
        else:
            top = min(cube, 1 << 64)  # a is a u64, so a <= 2^64 - 1 for n >= 64
            bounds = [0, 1, 2, top - 1, top // 2, top // 2 + 1]
            if cube <= (1 << 64) - 1:
                bounds.append(cube)
            bounds += [rng.below(min(cube, 1 << 64)) for _ in range(3)]
        if n <= LT_ENUMERATED_MAX:
            prefix = lt_enumerated(r)
            values = [prefix[a] for a in bounds]
            if n <= 12:
                for a in bounds:
                    require(lt_dyadic(r, a) == prefix[a], "the dyadic form of LT is the sum")
        else:
            values = [lt_dyadic(r, a) for a in bounds]
        require(all(a <= cube and a < 1 << 64 for a in bounds), "a <= 2^n and a fits 64 bits")
        if 0 in bounds:
            require(values[bounds.index(0)] == ZERO, "LT(r, 0) = 0")
        if cube in bounds:
            require(values[bounds.index(cube)] == ONE, "LT(r, 2^n) = 1")
        cases.append({"n": n, "point": points(r), "values": [[hex64(a), fp2(v)] for a, v in zip(bounds, values)]})
    return cases


def less_than_points_cases(rng):
    cases = []
    for n in LT2_VARS:
        q, i = rng.point(n), rng.point(n)
        sums = pair_sums(q, i)
        require(sums["eq"] == eq_points(q, i), "the diagonal sum is eq(q, i)")
        require(add(add(sums["lt_iq"], sums["lt_qi"]), sums["eq"]) == ONE, "every pair is <, > or =")
        causal = add(sums["lt_iq"], sums["eq"])
        cases.append({
            "n": n,
            "q": points(q),
            "i": points(i),
            "lt_iq": fp2(sums["lt_iq"]),
            "lt_qi": fp2(sums["lt_qi"]),
            "causal": fp2(causal),
            "eq": fp2(sums["eq"]),
        })
    return cases


def round_poly_cases(rng):
    cases = []
    for d in ROUND_DEGREES:
        for c in range(ROUND_CASES):
            coefficients = [rng.fp2() for _ in range(d + 1)]
            # The last case of each degree evaluates at a node, where interpolation must be exact
            # too: 0, 1 (the derived value) or d, by degree.
            r = lift((1, d, 0)[d % 3]) if c == ROUND_CASES - 1 else rng.fp2()
            sent = [horner(coefficients, lift(0))] + [horner(coefficients, lift(k)) for k in range(2, d + 1)]
            claim = add(horner(coefficients, ONE), horner(coefficients, ZERO))
            cases.append({
                "degree": d,
                "evaluations": points(sent),
                "claim": fp2(claim),
                "r": fp2(r),
                "value": fp2(horner(coefficients, r)),
            })
    return cases


def main():
    rng = Inputs(SEED)
    header = {
        "description": "RFC-0010 'Multilinear extensions' reference vectors, written by "
        "spec/vectors/poly/generate.py; see README.md for the fields of each section",
        "modulus": hex64(P),
        "non_residue": W,
        "seed": hex64(SEED),
    }
    sections = {
        "multilinear": multilinear_cases(rng),
        "eq_tables": eq_table_cases(rng),
        "less_than": less_than_cases(rng),
        "less_than_points": less_than_points_cases(rng),
        "round_polys": round_poly_cases(rng),
    }
    text = json.dumps(header, indent=2)[:-2]
    for name, cases in sections.items():
        text += f',\n  "{name}": ' + section(cases)
    text += "\n}\n"
    json.loads(text)  # the file must parse
    (HERE / "poly.json").write_text(text)


if __name__ == "__main__":
    main()
