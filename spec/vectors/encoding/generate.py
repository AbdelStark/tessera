#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0 OR MIT
"""Reference vectors for the Reed-Solomon encoding of RFC-0011 'Encoding': the NTT of RFC-0002
'Constants' on a subgroup and on the coset 7.L, and rows of k values encoded with a vanishing
mask of P coefficients into codewords of n = 2k symbols.

This script is the independent oracle. Its arithmetic is Python's arbitrary-precision integers,
and every value is computed from its definition: the root w_m = 7^((P - 1)/m), each evaluation
by Horner's rule at its point, a row as the values of a random polynomial f of degree below k at
w_k^0, ..., w_k^(k-1), the masked polynomial g = f + (X^k - 1) m by polynomial arithmetic on
coefficient lists, and symbol j of the codeword as g(7 w_n^j). It uses no fast transform and
shares no code or algorithm with crates/tessera-pcs, which checks its output.

Usage: python3 spec/vectors/encoding/generate.py
(rewrites encoding.json)

Inputs come from SplitMix64 with a fixed seed, so the file is reproducible byte for byte.
"""

import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
SEED = 0x7E55E7A_2026_0047  # fixed; changing it changes every vector
P = 2**64 - 2**32 + 1  # Goldilocks (RFC-0002 'Constants')
G = 7  # generates Fp^*; also the codeword's coset shift (RFC-0011 'Encoding')
PRIMES_OF_P_MINUS_1 = [2, 3, 5, 17, 257, 65537]  # P - 1 = 2^32 * 3 * 5 * 17 * 257 * 65537

NTT_LOG_SIZES = list(range(1, 9))  # transforms of 2^1 .. 2^8 points
ROW_LOG_SIZES = list(range(7))  # rows of k = 2^0 .. 2^6 values


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


class Inputs:
    """Uniform field elements from SplitMix64, by rejecting words at or above P."""

    def __init__(self, seed):
        self.words = splitmix64(seed)

    def fp(self):
        while True:
            x = next(self.words)
            if x < P:
                return x

    def fps(self, n):
        return [self.fp() for _ in range(n)]


# ----- Definitions (RFC-0002 'Constants', RFC-0011 'Encoding') ---------------------------------


def root(m):
    """w_m = 7^((P - 1)/m), a primitive m-th root of unity for m a power of two up to 2^32."""
    require(m & (m - 1) == 0 and (P - 1) % m == 0, "m is a power of two dividing P - 1")
    w = pow(G, (P - 1) // m, P)
    require(pow(w, m, P) == 1, "w_m^m = 1")
    if m > 1:
        require(pow(w, m // 2, P) == P - 1, "w_m has order exactly m")
    return w


def horner(coefficients, x):
    """sum_i coefficients[i] x^i."""
    out = 0
    for c in reversed(coefficients):
        out = (out * x + c) % P
    return out


def points(n, shift):
    """shift w_n^0, ..., shift w_n^(n-1), each power computed on its own."""
    w = root(n)
    return [shift * pow(w, j, P) % P for j in range(n)]


def evaluate_at(coefficients, xs):
    return [horner(coefficients, x) for x in xs]


def masked(f, m, k):
    """g = f + (X^k - 1) m as a coefficient list of length 2k: X^k m shifts m up by k, and the
    -m term subtracts it at the bottom."""
    g = [0] * (2 * k)
    for i, c in enumerate(f):
        g[i] = (g[i] + c) % P
    for i, c in enumerate(m):
        g[i] = (g[i] - c) % P
        g[k + i] = (g[k + i] + c) % P
    return g


# ----- Rendering ---------------------------------------------------------------------------------


def hex64(x):
    return f"0x{x:016x}"


def elements(values):
    return [hex64(v) for v in values]


def section(cases):
    return "[\n" + ",\n".join(json.dumps(case, separators=(",", ":")) for case in cases) + "\n  ]"


# ----- Vectors -----------------------------------------------------------------------------------


def ntt_cases(rng):
    cases = []
    for log_n in NTT_LOG_SIZES:
        n = 1 << log_n
        coefficients = rng.fps(n)
        values = evaluate_at(coefficients, points(n, 1))
        coset = evaluate_at(coefficients, points(n, G))
        cases.append({
            "n": n,
            "coefficients": elements(coefficients),
            "values": elements(values),
            "coset_values": elements(coset),
        })
    return cases


def mask_counts(k):
    """P in {0, 1, k/2, k}, where meaningful: distinct and at most k."""
    return sorted({p for p in (0, 1, k // 2, k) if p <= k})


def encoding_cases(rng):
    cases = []
    for log_k in ROW_LOG_SIZES:
        k = 1 << log_k
        n = 2 * k
        for p in mask_counts(k):
            f = rng.fps(k)
            m = rng.fps(p)
            row = evaluate_at(f, points(k, 1))
            g = masked(f, m, k)
            xs = points(n, G)
            codeword = evaluate_at(g, xs)
            # The mask vanishes on H: g and f agree there.
            require(evaluate_at(g, points(k, 1)) == row, "g = f on H")
            # 7.L is disjoint from H: no codeword point is a k-th root of unity.
            require(all(pow(x, k, P) != 1 for x in xs), "7.L misses H")
            # g has degree below k + P, and m sits in its coefficients k .. k + P - 1.
            require(all(c == 0 for c in g[k + p:]), "deg g < k + P")
            require(g[k:k + p] == m, "the mask is g's top coefficients")
            cases.append({
                "k": k,
                "mask_coeffs": p,
                "f": elements(f),
                "row": elements(row),
                "mask": elements(m),
                "codeword": elements(codeword),
            })
    return cases


def main():
    for q in PRIMES_OF_P_MINUS_1:
        require((P - 1) % q == 0 and pow(G, (P - 1) // q, P) != 1, f"7 generates Fp^*: q = {q}")
    rng = Inputs(SEED)
    header = {
        "description": "RFC-0011 'Encoding' reference vectors, written by "
        "spec/vectors/encoding/generate.py; see README.md for the fields of each section",
        "modulus": hex64(P),
        "generator": G,
        "coset_shift": G,
        "seed": hex64(SEED),
    }
    sections = {
        "ntt": ntt_cases(rng),
        "encoding": encoding_cases(rng),
    }
    text = json.dumps(header, indent=2)[:-2]
    for name, cases in sections.items():
        text += f',\n  "{name}": ' + section(cases)
    text += "\n}\n"
    json.loads(text)  # the file must parse
    (HERE / "encoding.json").write_text(text)


if __name__ == "__main__":
    main()
