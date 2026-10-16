#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0 OR MIT
"""Reference vectors for the opening checks of the advice commitment, RFC-0011 'Opening protocol':
honest openings of small committed matrices, which a verifier must accept, and openings changed in
exactly one place, which it must reject with the reason, group and column that RFC-0011 steps 6
and 8 give.

This script is the independent oracle: an honest prover of RFC-0011 commitments and openings,
written from RFC-0011 ('Canonical advice layout', 'Encoding', 'Column digests and root', 'Opening
protocol') and RFC-0004 ('Column digests', 'Merkle trees') alone, computing with Python's
arbitrary-precision integers. Its BLAKE3, H(domain, bytes), Merkle tree, canonical multiproof and
column chains are those of ../hash/generate.py, whose BLAKE3 is checked against the official test
vectors before anything is written; its roots of unity and masked encoding are those of
../encoding/generate.py; its Fp2 product, inverse and roots of unity are checked against the pinned
vectors of ../field/. It shares no code with crates/ or verifier/, which check its output. Where
RFC-0011 has the prover and the verifier use NTTs, it computes from definitions instead:

- A row's polynomial is interpolated on H by Lagrange's formula on coefficient lists, checked
  against the inverse DFT computed from its definition, and every codeword symbol is evaluated on
  its own by Horner's rule: no fast transform anywhere.
- A claim's value is the multilinear extension of the whole matrix, zero rows included, at the
  claim's point, checked against the per-row sum of RFC-0011.
- Every opening is checked again, from the fields written to the file, by a separate checker of
  RFC-0011 steps 6 and 8: it evaluates enc(u) by the barycentric formula on H, eq by its product
  formula and M by powers, recomputes each leaf from its opened column, and verifies the multiproof
  by rebuilding the root from the opened leaves.

Usage: python3 spec/vectors/pcs/generate.py
(rewrites opening.json)

Inputs come from SplitMix64 with the fixed seed below, so the file is reproducible byte for byte.
"""

import copy
import importlib.util
import json
import sys
from pathlib import Path

sys.dont_write_bytecode = True  # importing the oracles leaves nothing behind
HERE = Path(__file__).resolve().parent


def load(name, path):
    """Imports another oracle as a module; its main() does not run."""
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


hashing = load("hash_oracle", HERE.parent / "hash" / "generate.py")  # BLAKE3, H, Merkle trees, column chains
encoding = load("encoding_oracle", HERE.parent / "encoding" / "generate.py")  # roots of unity, masked encoding
P, H, require = hashing.P, hashing.H, hashing.require
require(encoding.P == P, "the oracles share the modulus")
SEED = 0x7E55E7A_0C05_0061  # fixed; changing it changes every vector
CHECK_SEED = 0x7E55E7A_C4EC_0061  # the self-checks' own stream: changing the checks changes no vector
W = 7  # Fp2 = Fp[X]/(X^2 - W) (RFC-0002)
SHIFT = 7  # the codeword lives on the coset 7.L (RFC-0011 'Encoding')
LOG2_INV_RATE = 1  # n = 2k


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


def inverse(a):
    """(a0 - a1 X) / (a0^2 - 7 a1^2), for a non-zero a."""
    norm = (a[0] * a[0] - W * a[1] * a[1]) % P
    require(norm != 0, "no inverse of 0")
    scale = pow(norm, P - 2, P)
    return (a[0] * scale % P, -a[1] * scale % P)


def check_field_vectors():
    """The arithmetic that the prover and the checker below share, against the pinned vectors of
    ../field/: every product and inverse of goldilocks_ext2.json, and every root of unity of
    constants.json as w_m = 7^((P - 1)/m). Returns the number of Fp2 cases checked."""
    constants = json.loads((HERE.parent / "field" / "constants.json").read_text())["goldilocks"]
    require(int(constants["modulus"], 16) == P and constants["extension_non_residue"] == W
            and constants["multiplicative_generator"] == SHIFT, "the field's constants")
    for log_m, text in enumerate(constants["root_of_unity_2_k"]):
        require(encoding.root(1 << log_m) == pow(7, (P - 1) >> log_m, P) == int(text, 16), f"w_(2^{log_m}) is pinned")
    ext2 = json.loads((HERE.parent / "field" / "goldilocks_ext2.json").read_text())
    require(int(ext2["modulus"], 16) == P and ext2["non_residue"] == W, "the extension's constants")
    for case in ext2["cases"]:
        a0, a1, b0, b1, c0, c1 = (int(x, 16) for x in case[:6])
        require(mul((a0, a1), (b0, b1)) == (c0, c1), "an Fp2 product of goldilocks_ext2.json")
        if case[6] is not None:
            require(inverse((a0, a1)) == (int(case[6], 16), int(case[7], 16)), "an Fp2 inverse of goldilocks_ext2.json")
        else:
            require((a0, a1) == ZERO, "only 0 has no inverse")
    return len(ext2["cases"])


# ----- Inputs ------------------------------------------------------------------------------------


class Inputs:
    """Uniform values from SplitMix64: field elements by rejecting words at or above P."""

    def __init__(self, seed):
        self.words = hashing.splitmix64(seed)

    def word(self):
        return next(self.words)

    def fp(self):
        while True:
            x = self.word()
            if x < P:
                return x

    def nonzero_fp(self):
        while True:
            x = self.fp()
            if x != 0:
                return x

    def fp2(self):
        return (self.fp(), self.fp())

    def digest(self):
        return b"".join(self.word().to_bytes(8, "little") for _ in range(4))

    def index(self, n):
        """Uniform in [0, n) for n a power of two: a word's low bits."""
        require(n & (n - 1) == 0, "a power-of-two range")
        return self.word() % n


# ----- Polynomials over Fp, as coefficient lists, constant first ----------------------------------

horner = encoding.horner


def times_linear(poly, a):
    """poly(X) (X - a)."""
    out = [0] * (len(poly) + 1)
    for i, c in enumerate(poly):
        out[i + 1] = (out[i + 1] + c) % P
        out[i] = (out[i] - a * c) % P
    return out


def divide_linear(poly, a):
    """poly(X) / (X - a) for a root a of poly, by synthetic division from the top coefficient."""
    quotient = [0] * (len(poly) - 1)
    carry = 0
    for i in range(len(poly) - 1, 0, -1):
        carry = (poly[i] + a * carry) % P
        quotient[i - 1] = carry
    require((poly[0] + a * carry) % P == 0, "the divisor's point is a root")
    return quotient


def interpolate(xs, ys):
    """The polynomial of degree below len(xs) through the points (xs[t], ys[t]), by Lagrange's
    formula: basis t is prod_{s != t} (X - x_s), the product over every point divided by X - x_t,
    scaled by y_t over its value at x_t."""
    require(len(set(xs)) == len(xs) == len(ys), "distinct interpolation points, one value each")
    product = [1]
    for x in xs:
        product = times_linear(product, x)
    out = [0] * len(xs)
    for xt, yt in zip(xs, ys):
        basis = divide_linear(product, xt)
        factor = yt * pow(horner(basis, xt), P - 2, P) % P
        for i, c in enumerate(basis):
            out[i] = (out[i] + factor * c) % P
    return out


def inverse_dft(values):
    """f_i = k^-1 sum_t values[t] w_k^(-it), from the definition of the inverse transform: the
    coefficients of the polynomial of degree below k that takes `values` on H."""
    k = len(values)
    w_inv = pow(encoding.root(k), P - 2, P)
    k_inv = pow(k, P - 2, P)
    return [k_inv * sum(v * pow(w_inv, i * t, P) for t, v in enumerate(values)) % P for i in range(k)]


# ----- The honest prover (RFC-0011) --------------------------------------------------------------


def encode_row(values, mask, k, n):
    """A row's polynomial f, interpolated on H, and its codeword: symbol j is
    g(x_j) = f(x_j) + (x_j^k - 1) m(x_j) at x_j = 7 w_n^j ('Encoding')."""
    h = encoding.points(k, 1)
    f = interpolate(h, values)
    require(f == inverse_dft(values), "Lagrange's formula and the inverse DFT give the same f")
    require(encoding.evaluate_at(f, h) == values, "f takes the row's values on H")
    xs = encoding.points(n, SHIFT)
    codeword = [(horner(f, x) + (pow(x, k, P) - 1) * horner(mask, x)) % P for x in xs]
    require(encoding.evaluate_at(encoding.masked(f, mask, k), xs) == codeword,
            "the codeword is g = f + (X^k - 1) m, built as one polynomial, on 7.L")
    return f, codeword


def commit(commitment_id, codewords, segments, row_block, n):
    """leaf_j = H("column-merge", d_j^(1) || ... || d_j^(S)), each d_j^(s) RFC-0004's chain of
    segment s's non-zero rows at column j, and the Merkle tree over the n leaves."""
    digests = [hashing.column_leaves(commitment_id, [codewords[r] for r in segment], n, row_block) for segment in segments]
    leaves = [H("column-merge", b"".join(d[j] for d in digests)) for j in range(n)]
    return leaves, hashing.merkle_levels(leaves)


def eq_table(point):
    """eq(point, x) for every x < 2^len(point), coordinate t setting bit t of x, built one
    coordinate at a time: the entries with bit t clear take 1 - r_t, those with it set take r_t."""
    table = [ONE]
    for r in point:
        table = [mul(e, sub(ONE, r)) for e in table] + [mul(e, r) for e in table]
    return table


def claim_value(values, k, row_vars, column_point, row_point):
    """A(column_point || row_point): the extension of the whole matrix, 2^row_vars rows of k values
    with the zero rows included, entry row * k + i holding position i of the row."""
    weights = eq_table(column_point + row_point)
    out = ZERO
    for row in range(1 << row_vars):
        for i in range(k):
            out = add(out, mul(lift(values[row][i] if row in values else 0), weights[row * k + i]))
    # RFC-0011's form: sum over the non-zero rows of eq(row_point, row) sum_i A_row[i] eq(column_point, i).
    by_rows, by_columns = eq_table(row_point), eq_table(column_point)
    per_row = total(mul(by_rows[row], total(mul(lift(a), by_columns[i]) for i, a in enumerate(values[row]))) for row in values)
    require(out == per_row, "the claim's value is the per-row sum of RFC-0011")
    return out


def combine(coefficients, rows, values, masks, k, pa):
    """u = sum_row c(row) A_row and M = sum_row c(row) m_row over the non-zero rows."""
    u = [total(mul(c, lift(values[r][i])) for c, r in zip(coefficients, rows)) for i in range(k)]
    mask = [total(mul(c, lift(masks[r][t])) for c, r in zip(coefficients, rows)) for t in range(pa)]
    return u, mask


def combined_at(u, mask, x, k):
    """enc(u)(x) + Z_H(x) M(x): enc(u) interpolated on H coordinate by coordinate, then Horner."""
    h = encoding.points(k, 1)
    z = (pow(x, k, P) - 1) % P
    return tuple(
        (horner(interpolate(h, [a[c] for a in u]), x) + z * horner([a[c] for a in mask], x)) % P for c in (0, 1)
    )


def build(rng, case):
    """An honest commitment and opening of a random matrix with `case`'s layout, claims and draws.
    Returns the opening as written to the file and the prover's state, for the changed openings."""
    log2_k, pa, row_block, row_vars = case["log2_k"], case["mask_coeffs"], case["row_block"], case["row_vars"]
    k, n = 1 << log2_k, 1 << (log2_k + LOG2_INV_RATE)
    segments = case["segments"]
    rows = [r for segment in segments for r in segment]
    require(all(a < b for a, b in zip(rows, rows[1:])), f"{case['name']}: rows increase within and across segments")
    require(rows and all(0 <= r < 1 << row_vars for r in rows), f"{case['name']}: rows in range")
    require(1 <= pa <= k and row_block >= 1, f"{case['name']}: 1 <= P_A <= k")
    commitment_id = rng.digest()
    values, masks, polys, codewords = {}, {}, {}, {}
    for r in rows:  # layout order: the row's values, then its mask coefficients m_0, m_1, ...
        values[r] = [rng.fp() for _ in range(k)]
        masks[r] = [rng.fp() for _ in range(pa)]
        polys[r], codewords[r] = encode_row(values[r], masks[r], k, n)
    leaves, levels = commit(commitment_id, codewords, segments, row_block, n)
    groups = []
    for patterns in case["groups"]:
        column_point = [rng.fp2() for _ in range(log2_k)]
        claims = []
        for pattern in patterns:  # one character per row coordinate, lowest first: r random, 0 or 1 Boolean
            require(len(pattern) == row_vars, f"{case['name']}: a row point has row_vars coordinates")
            row_point = [rng.fp2() if c == "r" else lift(int(c)) for c in pattern]
            gamma = rng.fp2()
            claims.append((row_point, gamma, claim_value(values, k, row_vars, column_point, row_point)))
        groups.append((column_point, claims))
    xi_point = [rng.fp2() for _ in range(row_vars)]
    draws = case["queries"]
    queries = list(draws) if isinstance(draws, list) else [rng.index(n) for _ in range(draws)]
    require(queries and all(0 <= q < n for q in queries), f"{case['name']}: draws in range")
    # Row coefficients: c_g(row) = sum_claims gamma eq(row_point, row), then xi_row = eq(xi*, row).
    coefficients = []
    for _, claims in groups:
        c = [ZERO] * (1 << row_vars)
        for row_point, gamma, _ in claims:
            c = [add(cv, mul(gamma, e)) for cv, e in zip(c, eq_table(row_point))]
        coefficients.append([c[r] for r in rows])
    xi_table = eq_table(xi_point)
    coefficients.append([xi_table[r] for r in rows])
    combined = [combine(cs, rows, values, masks, k, pa) for cs in coefficients]
    xs = encoding.points(n, SHIFT)
    for (column_point, claims), (u, _) in zip(groups, combined):
        require(total(mul(g, v) for _, g, v in claims) == total(mul(a, e) for a, e in zip(u, eq_table(column_point))),
                f"{case['name']}: the evaluation identity holds")
    for cs, (u, mask) in zip(coefficients, combined):
        for j, x in enumerate(xs):  # every column, not only the queried ones
            require(total(mul(c, lift(codewords[r][j])) for c, r in zip(cs, rows)) == combined_at(u, mask, x, k),
                    f"{case['name']}: the combined rows agree with the columns at column {j}")
    distinct = sorted(set(queries))
    opening = {
        "name": case["name"],
        "log2_k": log2_k,
        "log2_inv_rate": LOG2_INV_RATE,
        "mask_coeffs": pa,
        "row_block": row_block,
        "row_vars": row_vars,
        "commitment_id": commitment_id.hex(),
        "segments": segments,
        # Test data only: in a real commitment the rows and masks are secret (05-observability, section 3).
        "witness": {
            "rows": [[hex64(v) for v in values[r]] for r in rows],
            "masks": [[hex64(c) for c in masks[r]] for r in rows],
        },
        "root": levels[-1][0].hex(),
        "groups": [
            {
                "column_point": elements(column_point),
                "claims": [{"row_point": elements(rp), "gamma": fp2(g), "value": fp2(v)} for rp, g, v in claims],
            }
            for column_point, claims in groups
        ],
        "xi_point": elements(xi_point),
        "combined": [{"u": elements(u), "mask": elements(mask)} for u, mask in combined],
        "queries": queries,
        "columns": [[hex64(codewords[r][j]) for r in rows] for j in distinct],
        "siblings": [d.hex() for d in hashing.multiproof(levels, distinct)],
        "intermediate": {
            "row_coefficients": [elements(cs) for cs in coefficients],
            "points": [hex64(xs[j]) for j in distinct],
            "leaves": [leaves[j].hex() for j in distinct],
            "combined_at_queries": [elements([combined_at(u, mask, xs[j], k) for j in distinct]) for u, mask in combined],
        },
    }
    state = {
        "k": k, "n": n, "pa": pa, "row_block": row_block, "commitment_id": commitment_id, "segments": segments,
        "rows": rows, "values": values, "masks": masks, "polys": polys, "codewords": codewords,
        "coefficients": coefficients, "distinct": distinct, "queries": queries,
    }
    return opening, state


# ----- Self-checks of the code -------------------------------------------------------------------


def check_codewords(state, picks):
    """Every codeword lies in the code: interpolating all n symbols gives g = f + (X^k - 1) m, of
    degree below k + P_A, and interpolating any k + P_A of them reproduces the others (checked on
    the first, the last and two random sets of k + P_A positions). The mask vanishes on H: g takes
    the row's values there."""
    k, n, pa = state["k"], state["n"], state["pa"]
    xs, h = encoding.points(n, SHIFT), encoding.points(k, 1)
    require(all(pow(x, k, P) != 1 for x in xs), "7.L misses H")
    require(all(pow(x, k, P) == 1 for x in h), "Z_H vanishes on H")
    for r in state["rows"]:
        codeword = state["codewords"][r]
        g = interpolate(xs, codeword)
        require(g == encoding.masked(state["polys"][r], state["masks"][r], k), "the codeword interpolates to f + Z_H m")
        require(all(c == 0 for c in g[k + pa:]), "the codeword has degree below k + P_A")
        require(encoding.evaluate_at(g, h) == state["values"][r], "the mask vanishes on H")
        size = k + pa
        if size < n:
            subsets = [list(range(size)), list(range(n - size, n))]
            for _ in range(2):
                order = list(range(n))
                for i in range(n - 1, 0, -1):  # Fisher-Yates, from the self-checks' stream
                    j = picks.word() % (i + 1)
                    order[i], order[j] = order[j], order[i]
                subsets.append(sorted(order[:size]))
            for subset in subsets:
                poly = interpolate([xs[j] for j in subset], [codeword[j] for j in subset])
                require(all(horner(poly, xs[j]) == codeword[j] for j in range(n) if j not in subset),
                        "any k + P_A symbols determine the others")


def in_code(word, k, pa, n):
    """Whether a word of n symbols on 7.L is the evaluation of a polynomial of degree below k + P_A."""
    return all(c == 0 for c in interpolate(encoding.points(n, SHIFT), word)[k + pa:])


# ----- The checker: RFC-0011 steps 6 and 8, from an opening's fields alone ----------------------


def parse_fp(text):
    require(isinstance(text, str) and len(text) == 18 and text[:2] == "0x" and text == text.lower(), "an Fp is 0x and 16 hex digits")
    value = int(text[2:], 16)
    require(value < P, "an Fp is canonical")
    return value


def parse_fp2(pair):
    require(isinstance(pair, list) and len(pair) == 2, "an Fp2 is [c0, c1]")
    return (parse_fp(pair[0]), parse_fp(pair[1]))


def parse_digest(text):
    require(isinstance(text, str) and len(text) == 64 and text == text.lower(), "a digest is 64 lowercase hex digits")
    return bytes.fromhex(text)


def eq_bits(point, x):
    """eq(point, x) = prod_t (r_t x_t + (1 - r_t)(1 - x_t)), x_t bit t of x."""
    out = ONE
    for t, r in enumerate(point):
        bit = (x >> t) & 1
        out = mul(out, add(mul(r, lift(bit)), mul(sub(ONE, r), lift(1 - bit))))
    return out


def enc_at(u, x, k):
    """enc(u)(x) for x outside H, by the barycentric formula: the Lagrange basis of H at x is
    L_i(x) = (x^k - 1) w^i / (k (x - w^i)), w = 7^((P - 1)/k)."""
    w = pow(7, (P - 1) // k, P)
    z = (pow(x, k, P) - 1) % P
    require(z != 0, "x lies outside H")
    out = ZERO
    for i, ui in enumerate(u):
        wi = pow(w, i, P)
        out = add(out, mul(ui, lift(z * wi * pow(k * (x - wi), P - 2, P))))
    return out


def mask_at(mask, x):
    """M(x) = sum_i M_i x^i, each power on its own."""
    return total(mul(mi, lift(pow(x, i, P))) for i, mi in enumerate(mask))


def leaf_of(commitment_id, j, column, sizes, row_block):
    """leaf_j from an opened column: each segment's chain over its symbols, then the merge."""
    digests, start = b"", 0
    for m in sizes:
        chain = H("column-init", commitment_id + j.to_bytes(8, "little"))
        symbols = column[start:start + m]
        for step in range(-(-m // row_block)):  # ceil(m / B) steps, the last one partial
            block = symbols[step * row_block:(step + 1) * row_block]
            chain = H("column-step", chain + b"".join(s.to_bytes(8, "little") for s in block))
        digests += chain
        start += m
    require(start == len(column), "a column has one symbol per non-zero row")
    return H("column-merge", digests)


def verify_multiproof(root, n, opened, siblings):
    """Rebuilds the root from the opened (index, leaf) pairs, level by level, taking each sibling
    it cannot compute from the proof, in increasing index order within a level. False if the proof
    runs out, has digests left over, or reaches another root."""
    if not opened:
        return False
    level, taken, width = dict(opened), 0, n
    while width > 1:
        parents = {}
        for i in sorted(level):
            if i >> 1 in parents:
                continue  # the right child of a pair already joined
            if i ^ 1 in level:
                pair = (level[i], level[i ^ 1])
            elif taken == len(siblings):
                return False
            else:
                sibling, taken = siblings[taken], taken + 1
                pair = (level[i], sibling) if i % 2 == 0 else (sibling, level[i])
            parents[i >> 1] = H("merkle-node", pair[0] + pair[1])
        level, width = parents, width >> 1
    return taken == len(siblings) and level == {0: root}


def check(o):
    """The verdict of RFC-0011 steps 6 and 8 on opening `o`, checks in the vectors' order, the
    first failure deciding: (reason, group, column), with the step that decided it and, for an
    accepted opening, the values the checks computed, as `intermediate` writes them."""
    require(o["log2_inv_rate"] == LOG2_INV_RATE, "rate 1/2")
    log2_k, pa, row_block, row_vars = o["log2_k"], o["mask_coeffs"], o["row_block"], o["row_vars"]
    k, n = 1 << log2_k, 1 << (log2_k + o["log2_inv_rate"])
    require(1 <= pa <= k and row_block >= 1, "1 <= P_A <= k and B >= 1")
    segments = o["segments"]
    rows = [r for segment in segments for r in segment]
    require(all(a < b for a, b in zip(rows, rows[1:])) and all(0 <= r < 1 << row_vars for r in rows), "the layout")
    commitment_id, root = parse_digest(o["commitment_id"]), parse_digest(o["root"])
    groups = []
    for group in o["groups"]:
        column_point = [parse_fp2(c) for c in group["column_point"]]
        require(len(column_point) == log2_k and group["claims"], "a group's column point and claims")
        claims = [([parse_fp2(c) for c in claim["row_point"]], parse_fp2(claim["gamma"]), parse_fp2(claim["value"])) for claim in group["claims"]]
        require(all(len(rp) == row_vars for rp, _, _ in claims), "a row point has row_vars coordinates")
        groups.append((column_point, claims))
    xi_point = [parse_fp2(c) for c in o["xi_point"]]
    require(len(xi_point) == row_vars, "xi* has row_vars coordinates")
    queries = o["queries"]
    require(queries and all(isinstance(q, int) and 0 <= q < n for q in queries), "draws in range")

    # Step 6: the combined rows' shape, then each group's sum, in group order.
    combined = o["combined"]
    if len(combined) != len(groups) + 1 or any(len(c["u"]) != k or len(c["mask"]) != pa for c in combined):
        return ("Evaluation", None, None), "6, shape", None
    us = [[parse_fp2(v) for v in c["u"]] for c in combined]
    masks = [[parse_fp2(v) for v in c["mask"]] for c in combined]
    for g, (column_point, claims) in enumerate(groups):
        claimed = total(mul(gamma, value) for _, gamma, value in claims)
        if claimed != total(mul(us[g][i], eq_bits(column_point, i)) for i in range(k)):
            return ("Evaluation", g, None), "6, sum", None

    # Step 8.1: one column per distinct queried index, one symbol per non-zero row.
    smallest, distinct = min(queries), sorted(set(queries))
    if len(o["columns"]) != len(distinct) or any(len(c) != len(rows) for c in o["columns"]):
        return ("MerklePath", None, smallest), "8.1", None
    columns = [[parse_fp(s) for s in c] for c in o["columns"]]

    # Step 8.2: the leaves, from the columns, and the multiproof of the distinct indices.
    sizes = [len(segment) for segment in segments]
    leaves = [leaf_of(commitment_id, j, column, sizes, row_block) for j, column in zip(distinct, columns)]
    if not verify_multiproof(root, n, list(zip(distinct, leaves)), [parse_digest(d) for d in o["siblings"]]):
        return ("MerklePath", None, smallest), "8.2", None

    # Step 8.3: at each distinct j in increasing order, every group's combined row, then xi's.
    coefficients = [[total(mul(gamma, eq_bits(rp, r)) for rp, gamma, _ in claims) for r in rows] for _, claims in groups]
    coefficients.append([eq_bits(xi_point, r) for r in rows])
    w_n = pow(7, (P - 1) // n, P)
    points, at_queries = [], [[] for _ in combined]
    for j, column in zip(distinct, columns):
        x = SHIFT * pow(w_n, j, P) % P
        points.append(x)
        z = (pow(x, k, P) - 1) % P
        for c, (cs, u, mask) in enumerate(zip(coefficients, us, masks)):
            expected = add(enc_at(u, x, k), mul(lift(z), mask_at(mask, x)))
            if total(mul(cv, lift(s)) for cv, s in zip(cs, column)) != expected:
                return ("Proximity", None, j), "8.3", None
            at_queries[c].append(expected)
    computed = {
        "row_coefficients": [elements(cs) for cs in coefficients],
        "points": [hex64(x) for x in points],
        "leaves": [leaf.hex() for leaf in leaves],
        "combined_at_queries": [elements(values) for values in at_queries],
    }
    return ("Accept", None, None), "accept", computed


# ----- Reproduction: a second prover, from the witness and the public inputs alone ---------------
#
# It computes what an honest prover sends, by other algorithms than the prover above: codewords by
# radix-2 NTTs, the chains by a streaming committer that pushes rows in blocks of B, the root by
# recursion, the multiproof as the nodes whose subtree holds no opened leaf while their parent's
# does, eq by selecting r_t or 1 - r_t, and the claims' values by binding one variable at a time.

INPUTS = ("log2_k", "log2_inv_rate", "mask_coeffs", "row_block", "row_vars", "commitment_id", "segments", "witness",
          "xi_point", "queries")


def ntt(coefficients, w):
    """The values at w^0, ..., w^(n-1) of the polynomial with these n coefficients, n a power of two
    and w of order n: iterative Cooley-Tukey on the bit-reversed input, output in natural order."""
    a, n = list(coefficients), len(coefficients)
    j = 0
    for i in range(1, n):
        bit = n >> 1
        while j & bit:
            j ^= bit
            bit >>= 1
        j |= bit
        if i < j:
            a[i], a[j] = a[j], a[i]
    size = 2
    while size <= n:
        step, half = pow(w, n // size, P), size // 2
        for start in range(0, n, size):
            twiddle = 1
            for t in range(start, start + half):
                low, high = a[t], a[t + half] * twiddle % P
                a[t], a[t + half] = (low + high) % P, (low - high) % P
                twiddle = twiddle * step % P
        size <<= 1
    return a


def intt(values, w):
    """The coefficients of the polynomial of degree below n with these values at w^0, ..., w^(n-1)."""
    n_inv = pow(len(values), P - 2, P)
    return [v * n_inv % P for v in ntt(values, pow(w, P - 2, P))]


def coset_values(coefficients, n, w_n):
    """The polynomial's values at 7 w_n^j for j < n: coefficient i scaled by 7^i, then an NTT."""
    scaled, power = [0] * n, 1
    for i, c in enumerate(coefficients):
        scaled[i] = c * power % P
        power = power * SHIFT % P
    return ntt(scaled, w_n)


def with_mask(values, mask, k, n, w_k):
    """The coefficients of f + (X^k - 1) m, f taking `values` on H: m is added at X^k and taken
    away at X^0."""
    g = intt(values, w_k) + [0] * (n - k)
    for i, c in enumerate(mask):
        g[i] = (g[i] - c) % P
        g[k + i] = (g[k + i] + c) % P
    return g


def streamed_leaves(commitment_id, n, segments, row_block, codewords):
    """leaf_j for every column, as a streaming committer computes them: each segment's chains start
    from column-init and advance by one column-step per block of B rows, pushed in layout order,
    the last block possibly partial; the segment digests then merge."""
    digests = []
    for segment in segments:
        chains = [H("column-init", commitment_id + j.to_bytes(8, "little")) for j in range(n)]
        for start in range(0, len(segment), row_block):
            block = [codewords[r] for r in segment[start:start + row_block]]
            chains = [H("column-step", chains[j] + b"".join(cw[j].to_bytes(8, "little") for cw in block)) for j in range(n)]
        digests.append(chains)
    return [H("column-merge", b"".join(chains[j] for chains in digests)) for j in range(n)]


def subtree_root(leaves, low, high):
    """The root of the subtree over leaves low .. high - 1, by recursion."""
    if high - low == 1:
        return leaves[low]
    middle = (low + high) // 2
    return H("merkle-node", subtree_root(leaves, low, middle) + subtree_root(leaves, middle, high))


def multiproof_by_subtrees(leaves, opened):
    """The canonical multiproof as the nodes whose subtree holds no opened leaf while their parent's
    does, ordered by level from the leaves up, then by position."""
    found = []

    def visit(low, high, height):
        middle = (low + high) // 2
        for a, b in ((low, middle), (middle, high)):
            if any(a <= j < b for j in opened):
                if b - a > 1:
                    visit(a, b, height - 1)
            else:
                found.append((height - 1, a >> (height - 1), subtree_root(leaves, a, b)))

    if len(leaves) > 1:
        visit(0, len(leaves), len(leaves).bit_length() - 1)
    return [digest for _, _, digest in sorted(found)]


def eq_select(point, x):
    """eq(point, x) as the product of r_t where bit t of x is set and 1 - r_t where it is clear."""
    out = ONE
    for t, r in enumerate(point):
        out = mul(out, r if (x >> t) & 1 else sub(ONE, r))
    return out


def bound(table, point):
    """A table's multilinear extension at `point`, binding the lowest variable first:
    T'[x] = T[2x] + r (T[2x + 1] - T[2x])."""
    for r in point:
        table = [add(low, mul(r, sub(high, low))) for low, high in zip(table[0::2], table[1::2])]
    require(len(table) == 1, "a point binds every variable")
    return table[0]


def reproduce(inputs, groups):
    """What an honest prover sends for `inputs`, the witness and public fields of an opening, and the
    groups' points and batching challenges: the root, combined rows, columns and multiproof, the
    claims' values and the intermediate values, as the file writes them."""
    require(set(inputs) == set(INPUTS), "the inputs only")
    log2_k, pa, row_block, row_vars = inputs["log2_k"], inputs["mask_coeffs"], inputs["row_block"], inputs["row_vars"]
    k = 1 << log2_k
    n = k << inputs["log2_inv_rate"]
    w_n = pow(7, (P - 1) // n, P)
    w_k = pow(w_n, n // k, P)
    rows = [r for segment in inputs["segments"] for r in segment]
    witness = inputs["witness"]
    require(list(witness) == ["rows", "masks"] and len(witness["rows"]) == len(witness["masks"]) == len(rows),
            "one row of values and one mask per non-zero row")
    values = {r: [parse_fp(v) for v in vs] for r, vs in zip(rows, witness["rows"])}
    masks = {r: [parse_fp(c) for c in cs] for r, cs in zip(rows, witness["masks"])}
    require(all(len(values[r]) == k and len(masks[r]) == pa for r in rows), "k values and P_A mask coefficients per row")
    codewords = {r: coset_values(with_mask(values[r], masks[r], k, n, w_k), n, w_n) for r in rows}
    leaves = streamed_leaves(bytes.fromhex(inputs["commitment_id"]), n, inputs["segments"], row_block, codewords)
    distinct = sorted(set(inputs["queries"]))
    # The claims' values: the whole matrix, zero rows included, entry row * k + i.
    matrix = [lift(values[row][i]) if row in values else ZERO for row in range(1 << row_vars) for i in range(k)]
    claim_values = [[bound(matrix, column_point + row_point) for row_point, _ in claims] for column_point, claims in groups]
    # Combined rows, accumulated claim by claim: gamma times the rows weighted by eq(row_point, row).
    coefficients = []
    for _, claims in groups:
        cs = [ZERO] * len(rows)
        for row_point, gamma in claims:
            cs = [add(c, mul(gamma, eq_select(row_point, r))) for c, r in zip(cs, rows)]
        coefficients.append(cs)
    coefficients.append([eq_select([parse_fp2(c) for c in inputs["xi_point"]], r) for r in rows])
    combined, at_queries = [], []
    for cs in coefficients:
        u = [total(mul(c, lift(values[r][i])) for c, r in zip(cs, rows)) for i in range(k)]
        mask = [total(mul(c, lift(masks[r][t])) for c, r in zip(cs, rows)) for t in range(pa)]
        combined.append({"u": elements(u), "mask": elements(mask)})
        # enc(u) + Z_H M on 7.L, each coordinate of Fp2 on its own.
        on_coset = [coset_values(with_mask([a[c] for a in u], [a[c] for a in mask], k, n, w_k), n, w_n) for c in (0, 1)]
        at_queries.append(elements([(on_coset[0][j], on_coset[1][j]) for j in distinct]))
    points, x = [], SHIFT
    for j in range(n):
        if j in distinct:
            points.append(hex64(x))
        x = x * w_n % P
    return {
        "root": subtree_root(leaves, 0, n).hex(),
        "combined": combined,
        "columns": [[hex64(codewords[r][j]) for r in rows] for j in distinct],
        "siblings": [d.hex() for d in multiproof_by_subtrees(leaves, distinct)],
        "values": [[fp2(v) for v in vs] for vs in claim_values],
        "intermediate": {
            "row_coefficients": [elements(cs) for cs in coefficients],
            "points": points,
            "leaves": [leaves[j].hex() for j in distinct],
            "combined_at_queries": at_queries,
        },
    }


def check_reproduction(opening):
    """The opening's root, combined rows, columns, multiproof, claims' values and intermediate values
    are what the second prover computes from its witness and public inputs alone."""
    inputs = {key: opening[key] for key in INPUTS}
    groups = [
        ([parse_fp2(c) for c in group["column_point"]],
         [([parse_fp2(c) for c in claim["row_point"]], parse_fp2(claim["gamma"])) for claim in group["claims"]])
        for group in opening["groups"]
    ]
    got = reproduce(inputs, groups)
    for key in ("root", "combined", "columns", "siblings", "intermediate"):
        require(got[key] == opening[key], f"{opening['name']}: the witness reproduces {key}")
    require(got["values"] == [[claim["value"] for claim in group["claims"]] for group in opening["groups"]],
            f"{opening['name']}: the witness reproduces the claims' values")


# ----- Changed openings --------------------------------------------------------------------------


def differences(a, b, path=()):
    """The paths at which b differs from a. A list that lost or gained one element is reported
    once, as (..., "removed" or "inserted", the first position at which the two differ)."""
    if isinstance(a, dict) and isinstance(b, dict):
        require(list(a) == list(b), "the same fields in the same order")
        return [d for key in a for d in differences(a[key], b[key], path + (key,))]
    if isinstance(a, list) and isinstance(b, list):
        if len(a) == len(b):
            return [d for i, (x, y) in enumerate(zip(a, b)) for d in differences(x, y, path + (i,))]
        i = next((i for i, (x, y) in enumerate(zip(a, b)) if x != y), min(len(a), len(b)))
        if len(b) == len(a) - 1 and a[:i] + a[i + 1:] == b:
            return [path + ("removed", i)]
        if len(b) == len(a) + 1 and b[:i] + b[i + 1:] == a:
            return [path + ("inserted", i)]
        return [path + ("resized",)]
    return [] if a == b else [path]


def bump_fp(text):
    """An Fp plus 1."""
    return hex64((parse_fp(text) + 1) % P)


def bump(pair, c):
    """An Fp2 with 1 added to coordinate c."""
    out = list(pair)
    out[c] = bump_fp(out[c])
    return out


def flip(digest):
    """A digest with the lowest bit of its first byte flipped."""
    raw = bytearray(bytes.fromhex(digest))
    raw[0] ^= 1
    return raw.hex()


def without_intermediate(opening):
    """An honest opening as a rejection carries it: without its intermediate values and witness."""
    out = copy.deepcopy(opening)
    del out["intermediate"]
    del out["witness"]
    return out


def rejection(name, honest, changed, verdict, expected, step):
    """A changed opening with its expected verdict, after checking that it differs from `honest`
    at exactly the `expected` paths and that the checker stops at `step` with that verdict."""
    base = without_intermediate(honest)
    found = differences(base, changed)
    require(found == expected, f"{name}: changed at {found}, not at {expected}")
    got, at, _ = check(changed)
    require((got, at) == (verdict, step), f"{name}: the checker gives {got} at step {at}, not {verdict} at step {step}")
    reason, group, column = verdict
    return {"name": name, "opening": changed, "reason": reason, "group": group, "column": column}, step


def cheat(rng, name, honest, state, row, changed_columns):
    """A cheating prover: row `row`'s codeword changed at `changed_columns` (the set D) and committed,
    so the root, columns and multiproof are those of the changed codewords, while the claims and
    combined rows are the honest ones, from the row's message."""
    n, k, pa, rows, distinct = state["n"], state["k"], state["pa"], state["rows"], state["distinct"]
    hit = sorted(set(changed_columns) & set(distinct))
    require(hit and hit[0] > min(state["queries"]), f"{name}: the draws hit D, but not at their smallest index")
    codewords = {r: list(cw) for r, cw in state["codewords"].items()}
    for j in sorted(changed_columns):
        codewords[row][j] = (codewords[row][j] + rng.nonzero_fp()) % P
    require([j for j in range(n) if codewords[row][j] != state["codewords"][row][j]] == sorted(changed_columns),
            f"{name}: the codeword changes at exactly D")
    require(not in_code(codewords[row], k, pa, n), f"{name}: the changed codeword is not in the code")
    leaves, levels = commit(state["commitment_id"], codewords, state["segments"], state["row_block"], n)
    changed = without_intermediate(honest)
    changed["root"] = levels[-1][0].hex()
    changed["columns"] = [[hex64(codewords[r][j]) for r in rows] for j in distinct]
    changed["siblings"] = [d.hex() for d in hashing.multiproof(levels, distinct)]
    position = rows.index(row)
    expected = [("root",)] + [("columns", distinct.index(j), position) for j in hit]
    found = differences(without_intermediate(honest), changed)
    require([d for d in found if d[0] != "siblings"] == expected, f"{name}: changes the root and row {row}'s symbols in D only")
    return changed, expected + [d for d in found if d[0] == "siblings"]


def rejections_of(rng, built):
    """Every changed opening, each with its expected verdict and the step that decides it."""
    by_name = {opening["name"]: (opening, state) for opening, state in built}
    single, _ = by_name[NAME_SINGLE]
    main, state = by_name[NAME_MAIN]
    every, _ = by_name[NAME_EVERY]
    k, pa, rows, distinct = state["k"], state["pa"], state["rows"], state["distinct"]
    smallest = min(state["queries"])
    require(all(ZERO in cs for cs in state["coefficients"][:-1]), "each c_g of the main opening vanishes on some non-zero row")
    out = []

    def add_case(name, honest, edit, verdict, step):
        changed = without_intermediate(honest)
        expected = edit(changed)
        out.append(rejection(name, honest, changed, verdict, expected, step))

    # Evaluation, at the first group whose sums differ.
    def value_c0(o):
        o["groups"][0]["claims"][0]["value"] = bump(o["groups"][0]["claims"][0]["value"], 0)
        return [("groups", 0, "claims", 0, "value", 0)]

    add_case("a claim's value changed: 1 added to c0 of the only claim", single, value_c0, ("Evaluation", 0, None), "6, sum")

    # The next two move one side of a group's sum by exactly X, then by exactly 1, so that a
    # verifier comparing one coordinate of Fp2 accepts one of them.
    claim = main["groups"][1]["claims"][0]
    gamma, value = parse_fp2(claim["gamma"]), parse_fp2(claim["value"])
    moved = add(value, mul(inverse(gamma), (0, 1)))
    require(mul(gamma, sub(moved, value)) == (0, 1), "gamma v moves by X")
    require(moved[0] != value[0] and moved[1] != value[1], "the value changes in both coordinates")

    def value_by_x(o):
        o["groups"][1]["claims"][0]["value"] = fp2(moved)
        return [("groups", 1, "claims", 0, "value", 0), ("groups", 1, "claims", 0, "value", 1)]

    add_case("a claim's value changed by X/gamma: group 1's claimed sum moves by X, its c0 unchanged", main, value_by_x,
             ("Evaluation", 1, None), "6, sum")

    weight = eq_bits([parse_fp2(c) for c in main["groups"][2]["column_point"]], 2)
    entry = parse_fp2(main["combined"][2]["u"][2])
    shifted = add(entry, inverse(weight))
    require(mul(weight, sub(shifted, entry)) == ONE, "u_i eq(column point, i) moves by 1")
    require(shifted[0] != entry[0] and shifted[1] != entry[1], "the entry changes in both coordinates")

    def u_entry(o):
        o["combined"][2]["u"][2] = fp2(shifted)
        return [("combined", 2, "u", 2, 0), ("combined", 2, "u", 2, 1)]

    add_case("a group's u changed at entry 2 by 1/eq(column point, 2): group 2's opened sum moves by 1, its c1 unchanged",
             main, u_entry, ("Evaluation", 2, None), "6, sum")

    # Evaluation, the combined rows' shape.
    def no_xi(o):
        o["combined"].pop()
        return [("combined", "removed", 3)]

    add_case("the proximity row missing: G combined rows", main, no_xi, ("Evaluation", None, None), "6, shape")

    def u_short(o):
        o["combined"][0]["u"].pop()
        return [("combined", 0, "u", "removed", k - 1)]

    add_case("group 0's u of k - 1 values, its last entry dropped", main, u_short, ("Evaluation", None, None), "6, shape")

    def mask_long(o):
        o["combined"][1]["mask"].append(fp2(ZERO))
        return [("combined", 1, "mask", "inserted", pa)]

    add_case("group 1's mask of P_A + 1 coefficients, a zero coefficient appended, so M is the same polynomial", main,
             mask_long, ("Evaluation", None, None), "6, shape")

    def xi_u_long(o):
        o["combined"][3]["u"].append(fp2(ZERO))
        return [("combined", 3, "u", "inserted", k)]

    add_case("the proximity row's u of k + 1 values, a zero appended", main, xi_u_long, ("Evaluation", None, None), "6, shape")

    def xi_mask_short(o):
        o["combined"][3]["mask"].pop()
        return [("combined", 3, "mask", "removed", pa - 1)]

    add_case("the proximity row's mask of P_A - 1 coefficients, its last dropped", main, xi_mask_short,
             ("Evaluation", None, None), "6, shape")

    def extra_row(o):
        o["combined"].append(copy.deepcopy(o["combined"][3]))
        return [("combined", "inserted", 4)]

    add_case("an extra combined row: the proximity row sent twice, G + 2 rows", main, extra_row, ("Evaluation", None, None), "6, shape")

    # Proximity, at the smallest distinct query: a change that every column sees.
    def group_mask(o):
        o["combined"][1]["mask"][0] = bump(o["combined"][1]["mask"][0], 1)
        return [("combined", 1, "mask", 0, 1)]

    add_case("a group's mask changed: 1 added to c1 of coefficient 0 of group 1's mask", main, group_mask,
             ("Proximity", None, smallest), "8.3")

    def xi_u(o):
        o["combined"][3]["u"][3] = bump(o["combined"][3]["u"][3], 1)
        return [("combined", 3, "u", 3, 1)]

    add_case("the proximity row's u changed: 1 added to c1 of entry 3", main, xi_u, ("Proximity", None, smallest), "8.3")

    def xi_mask(o):
        o["combined"][3]["mask"][1] = bump(o["combined"][3]["mask"][1], 0)
        return [("combined", 3, "mask", 1, 0)]

    add_case("the proximity row's mask changed: 1 added to c0 of coefficient 1", main, xi_mask, ("Proximity", None, smallest), "8.3")

    # MerklePath, at the smallest queried index: steps 8.1 and 8.2 fail as a whole.
    merkle = ("MerklePath", None, smallest)
    largest = len(distinct) - 1

    def symbol(o):
        o["columns"][largest][-1] = bump_fp(o["columns"][largest][-1])
        return [("columns", largest, len(rows) - 1)]

    add_case(f"a column symbol changed: 1 added to row {rows[-1]}'s symbol in column {distinct[-1]}, the largest queried index",
             main, symbol, merkle, "8.2")

    def sibling(o):
        o["siblings"][1] = flip(o["siblings"][1])
        return [("siblings", 1)]

    add_case("a sibling changed: one bit of sibling 1 flipped", main, sibling, merkle, "8.2")

    def siblings_swapped(o):
        o["siblings"][0], o["siblings"][1] = o["siblings"][1], o["siblings"][0]
        return [("siblings", 0), ("siblings", 1)]

    add_case("the siblings out of canonical order: siblings 0 and 1 swapped", main, siblings_swapped, merkle, "8.2")

    # The multiproof's siblings per level, leaves first: the last one sits at the second level.
    known, per_level = distinct, []
    while len(per_level) < state["n"].bit_length() - 1:
        per_level.append(len([i for i in known if i ^ 1 not in known]))
        known = sorted({i >> 1 for i in known})
    require(sum(per_level) == len(main["siblings"]) and per_level[1] > 0 and not any(per_level[2:]),
            "the main multiproof's last sibling is at the second level")

    def sibling_missing(o):
        o["siblings"].pop()
        return [("siblings", "removed", len(o["siblings"]))]

    add_case("a sibling missing: the last one, at the second level, dropped", main, sibling_missing, merkle, "8.2")

    extra_digest = rng.digest().hex()

    def sibling_extra(o):
        o["siblings"].append(extra_digest)
        return [("siblings", "inserted", len(o["siblings"]) - 1)]

    add_case("an extra sibling: a random digest appended", main, sibling_extra, merkle, "8.2")

    def root(o):
        o["root"] = flip(o["root"])
        return [("root",)]

    add_case("the root changed: one bit flipped", main, root, merkle, "8.2")

    draw_order = list(dict.fromkeys(state["queries"]))
    require(draw_order != distinct, "the main opening's distinct draws are not increasing")

    def in_draw_order(o):
        o["columns"] = [o["columns"][distinct.index(j)] for j in draw_order]
        return [("columns", i, r) for i, j in enumerate(draw_order) if j != distinct[i] for r in range(len(rows))]

    add_case(f"the distinct columns in draw order ({', '.join(map(str, draw_order))}), not in increasing order",
             main, in_draw_order, merkle, "8.2")

    repeated = [j for j in distinct if state["queries"].count(j) > 1]
    require(len(repeated) == 1, "the main draw repeats one index")
    at = distinct.index(repeated[0])

    def twice(o):
        o["columns"].insert(at + 1, copy.deepcopy(o["columns"][at]))
        return [("columns", "inserted", at + 1)]

    add_case(f"the repeated column ({repeated[0]}) sent twice, next to itself", main, twice, merkle, "8.1")

    def per_draw(o):
        o["columns"] = [copy.deepcopy(o["columns"][distinct.index(j)]) for j in state["queries"]]
        return [("columns", "resized")]

    add_case(f"one column per draw, in draw order ({', '.join(map(str, state['queries']))}), the repeated one twice",
             main, per_draw, merkle, "8.1")

    def short(o):
        o["columns"][1].pop()
        return [("columns", 1, "removed", len(rows) - 1)]

    add_case(f"column {distinct[1]} one symbol short, its last dropped", main, short, merkle, "8.1")

    extra_symbol = hex64(rng.fp())

    def long(o):
        o["columns"][largest].append(extra_symbol)
        return [("columns", largest, "inserted", len(rows))]

    add_case(f"column {distinct[-1]} one symbol long, a random symbol appended", main, long, merkle, "8.1")

    def column_missing(o):
        o["columns"].pop()
        return [("columns", "removed", largest)]

    add_case(f"a column missing: the last one, column {distinct[-1]}", main, column_missing, merkle, "8.1")

    unqueried = next(j for j in range(state["n"]) if j not in distinct and j > distinct[0])
    slot = next(i for i, j in enumerate(distinct) if j > unqueried)

    def column_extra(o):
        o["columns"].insert(slot, [hex64(state["codewords"][r][unqueried]) for r in rows])
        return [("columns", "inserted", slot)]

    add_case(f"an extra column: the committed column {unqueried}, not queried, inserted in increasing order", main,
             column_extra, merkle, "8.1")

    def every_extra(o):
        o["siblings"].append(extra_digest)
        return [("siblings", "inserted", 0)]

    require(every["siblings"] == [], "an opening of every column has an empty multiproof")
    add_case("an extra sibling given to the empty multiproof of an opening of every column", every, every_extra,
             ("MerklePath", None, min(every["queries"])), "8.2")

    # Proximity, from a cheating prover whose committed row is not the codeword of its message.
    for row, changed_columns, why in CHEATS:
        cs = [state["coefficients"][g][rows.index(row)] for g in range(len(state["coefficients"]))]
        touched = [g for g, c in enumerate(cs[:-1]) if c != ZERO]
        require(cs[-1] != ZERO, "xi touches every row")
        require(bool(touched) == (why == "groups"), f"row {row}: the groups that touch it")
        name = (f"a cheating prover: row {row}'s codeword changed at columns {', '.join(map(str, changed_columns))} and committed, "
                + (f"c_g(row) non-zero for groups {', '.join(map(str, touched))} only" if touched else "no claim touches the row, only xi"))
        changed, expected = cheat(rng, name, main, state, row, changed_columns)
        hit = min(set(changed_columns) & set(distinct))
        out.append(rejection(name, main, changed, ("Proximity", None, hit), expected, "8.3"))
    return out


# ----- Rendering ---------------------------------------------------------------------------------


def hex64(x):
    return f"0x{x:016x}"


def fp2(a):
    return [hex64(a[0]), hex64(a[1])]


def elements(values):
    return [fp2(v) for v in values]


def compact(value):
    return json.dumps(value, separators=(",", ":"))


def render_opening(opening, pad):
    """One field per line; the groups, combined rows and columns one per line; the intermediate
    values one list per line; the witness's rows and masks one per line."""
    fields = []
    for key, value in opening.items():
        if key in ("groups", "combined", "columns") and value:
            items = ",\n".join(f"{pad}    {compact(item)}" for item in value)
            fields.append(f'{pad}  "{key}": [\n{items}\n{pad}  ]')
        elif key == "intermediate":
            items = ",\n".join(f'{pad}    "{name}": {compact(v)}' for name, v in value.items())
            fields.append(f'{pad}  "{key}": {{\n{items}\n{pad}  }}')
        elif key == "witness":
            parts = []
            for name, lists in value.items():
                items = ",\n".join(f"{pad}      {compact(item)}" for item in lists)
                parts.append(f'{pad}    "{name}": [\n{items}\n{pad}    ]' if lists else f'{pad}    "{name}": []')
            fields.append(f'{pad}  "{key}": {{\n' + ",\n".join(parts) + f"\n{pad}  }}")
        else:
            fields.append(f'{pad}  "{key}": {compact(value)}')
    return "{\n" + ",\n".join(fields) + f"\n{pad}}}"


def without_witness_lines(text):
    """The file with the lines of every opening's witness removed: from its opening line to its
    closing one, which ends with the comma that separates it from the next field."""
    out, inside = [], False
    for line in text.split("\n"):
        if line == '      "witness": {':
            inside = True
        elif inside:
            inside = line != "      },"
        else:
            out.append(line)
    return "\n".join(out)


def render_rejection(r, pad):
    return (
        "{\n"
        + f'{pad}  "name": {compact(r["name"])},\n'
        + f'{pad}  "opening": {render_opening(r["opening"], pad + "  ")},\n'
        + f'{pad}  "reason": {compact(r["reason"])},\n'
        + f'{pad}  "group": {compact(r["group"])},\n'
        + f'{pad}  "column": {compact(r["column"])}\n'
        + f"{pad}}}"
    )


def render_text(header, openings, rejections):
    text = json.dumps(header, indent=2)[:-2]
    text += ',\n  "openings": [\n' + ",\n".join("    " + render_opening(o, "    ") for o in openings) + "\n  ]"
    return text + ',\n  "rejections": [\n' + ",\n".join("    " + render_rejection(r, "    ") for r in rejections) + "\n  ]\n}\n"


def render(header, openings, rejections):
    text = render_text(header, openings, rejections)
    document = json.loads(text)
    require(document == {**header, "openings": openings, "rejections": rejections}, "the file parses back")
    # The witnesses were added to a file that had none: removing their lines gives that file byte
    # for byte, so every other value is unchanged.
    bare = [{key: value for key, value in o.items() if key != "witness"} for o in openings]
    require(all("witness" in o for o in openings) and not any("witness" in r["opening"] for r in rejections),
            "every honest opening carries its witness, and no rejection does")
    require(without_witness_lines(text) == render_text(header, bare, rejections), "removing the witnesses gives the file without them")
    return text, document


# ----- Vectors -----------------------------------------------------------------------------------

NAME_SINGLE = "one row (row_vars = 0) and one segment: one group of one claim, one draw, of column n - 1"
NAME_MAIN = ("two segments: zero rows between them, inside the second and after it; three groups of 2, 1 and 3 claims "
             "on whole tensors, so that each c_g vanishes on some rows and no claim touches row 6; four draws, "
             "6, 1, 6, 4: index 6 repeats and the distinct indices in draw order are not increasing")
NAME_EVERY = "every column queried: eight draws without a repeat, in no order, so the multiproof is empty"

# Each case: its layout, one row-point pattern per claim and group (a character per row coordinate,
# lowest first: r random, 0 or 1 Boolean, as the row-offset bits of a claimed tensor are), and its
# draws: a list, or a count of uniform draws.
CASES = [
    {"name": NAME_SINGLE, "log2_k": 2, "mask_coeffs": 1, "row_block": 1, "row_vars": 0,
     "segments": [[0]], "groups": [[""]], "queries": [7]},
    {"name": NAME_MAIN, "log2_k": 2, "mask_coeffs": 2, "row_block": 2, "row_vars": 3,
     "segments": [[0, 1, 2], [4, 6]], "groups": [["rr0", "rr0"], ["r01"], ["rr0", "r01", "rr0"]],
     "queries": [6, 1, 6, 4]},
    {"name": "three segments, the middle one empty, after a leading zero row; P_A = k; B = 32, one step per segment; "
             "Boolean row points that select single rows; two draws, 5 then 0",
     "log2_k": 2, "mask_coeffs": 4, "row_block": 32, "row_vars": 2,
     "segments": [[1], [], [2, 3]], "groups": [["rr"], ["rr", "10", "r1", "01"]], "queries": [5, 0]},
    {"name": "one segment of four rows, whole blocks of B = 2; two groups of 5 and 2 claims at random points; "
             "three draws, all of column n - 1",
     "log2_k": 3, "mask_coeffs": 1, "row_block": 2, "row_vars": 2,
     "segments": [[0, 1, 2, 3]], "groups": [["rr"] * 5, ["rr"] * 2], "queries": [15, 15, 15]},
    {"name": "three segments with zero rows inside, between and after them; B = 32; four groups of 1, 2, 3 and 5 "
             "claims; five uniform draws",
     "log2_k": 3, "mask_coeffs": 3, "row_block": 32, "row_vars": 4,
     "segments": [[0, 2], [5, 6, 7], [12]],
     "groups": [["rrrr"], ["rr00", "rr10"], ["0011", "rrr0", "r010"], ["rrrr"] * 5], "queries": 5},
    {"name": "two segments of one row each; P_A = k; B = 1; one group of 3 claims; six uniform draws",
     "log2_k": 3, "mask_coeffs": 8, "row_block": 1, "row_vars": 1,
     "segments": [[0], [1]], "groups": [["r", "r", "1"]], "queries": 6},
    {"name": "one segment with zero rows inside and after it; B = 32; three groups of 4, 1 and 2 claims; "
             "seven uniform draws",
     "log2_k": 4, "mask_coeffs": 1, "row_block": 32, "row_vars": 3,
     "segments": [[0, 1, 3, 4, 6]], "groups": [["rrr", "rr0", "r1r", "rrr"], ["rrr"], ["0r1", "rrr"]], "queries": 7},
    {"name": "two segments with a zero row between them; B = 1; two groups of 2 and 5 claims; eight uniform draws",
     "log2_k": 4, "mask_coeffs": 5, "row_block": 1, "row_vars": 2,
     "segments": [[0, 1], [3]], "groups": [["rr", "r0"], ["rr", "rr", "rr", "rr", "11"]], "queries": 8},
    {"name": "three segments, the first one empty; P_A = k; B = 2 with partial last blocks; four groups of 5, 1, 2 "
             "and 3 claims; six uniform draws",
     "log2_k": 4, "mask_coeffs": 16, "row_block": 2, "row_vars": 4,
     "segments": [[], [2, 3, 5], [8, 9, 10, 11, 13]],
     "groups": [["rrrr", "rrrr", "rrrr", "rr01", "rrr0"], ["rrr1"], ["0100", "rrrr"], ["r1rr", "rrrr", "1011"]],
     "queries": 6},
    {"name": NAME_EVERY, "log2_k": 2, "mask_coeffs": 3, "row_block": 1, "row_vars": 1,
     "segments": [[0, 1]], "groups": [["r"], ["0"]], "queries": [3, 7, 0, 5, 1, 6, 2, 4]},
]

# The cheating provers of the main opening: (row, D, which combined rows see the change).
CHEATS = [(4, [4, 6, 7], "groups"), (6, [2, 6], "xi")]


def main():
    checked = hashing.check_official_vectors()
    products = check_field_vectors()
    rng, picks = Inputs(SEED), Inputs(CHECK_SEED)
    built = [build(rng, case) for case in CASES]
    for _, state in built:
        check_codewords(state, picks)
    cases = rejections_of(rng, built)
    openings = [opening for opening, _ in built]
    rejections = [r for r, _ in cases]
    header = {
        "description": "RFC-0011 advice-opening reference vectors: honest openings, which a verifier accepts, and openings "
        "changed in one place, which it rejects with the stated reason; written by spec/vectors/pcs/generate.py; see README.md",
        "modulus": hex64(P),
        "non_residue": W,
        "coset_shift": SHIFT,
        "seed": hex64(SEED),
        "official_blake3_cases_checked": checked,
    }
    text, document = render(header, openings, rejections)
    # The checks again, on the file as written: every opening accepted with its intermediate values,
    # every rejection rejected at its stated check.
    for opening in document["openings"]:
        verdict, step, computed = check(opening)
        require(verdict == ("Accept", None, None), f"{opening['name']}: accepted, not {verdict} at step {step}")
        require(computed == opening["intermediate"], f"{opening['name']}: the intermediate values are the checks' own")
        check_reproduction(opening)
    for r, (_, step) in zip(document["rejections"], cases):
        require(check(r["opening"])[:2] == ((r["reason"], r["group"], r["column"]), step), f"{r['name']}: its verdict")
        require(r["opening"]["name"] in {o["name"] for o in openings}, f"{r['name']}: names the honest opening it changes")
    require(len(text.encode()) < 1_000_000, "the file stays under 1 MB")
    (HERE / "opening.json").write_text(text)
    print(f"wrote opening.json: {len(openings)} openings, {len(rejections)} rejections, {len(text)} bytes; "
          f"checked {checked} official BLAKE3 vectors and {products} Fp2 products")


if __name__ == "__main__":
    main()
