#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0 OR MIT
"""Reference vectors for RFC-0002 field arithmetic, computed with Python's arbitrary-precision
integers. This script is the independent oracle: it shares no code or algorithm with the Rust
implementations in crates/tessera-field and verifier/crates/tv-field, which both check its
output.

Usage: python3 spec/vectors/field/generate.py   (rewrites constants.json and goldilocks.json)

Inputs come from SplitMix64 with a fixed seed so that the files are reproducible and any
implementation can regenerate the same inputs.
"""

import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
SEED = 0x7E55E7A_2026_1015  # fixed; changing it changes every vector
CASES = 10_000


def splitmix64(state):
    """Yields 64-bit outputs of SplitMix64 (Steele, Lea, Flood 2014) from `state`."""
    mask = (1 << 64) - 1
    while True:
        state = (state + 0x9E3779B97F4A7C15) & mask
        z = state
        z = ((z ^ (z >> 30)) * 0xBF58476D1CE4E5B9) & mask
        z = ((z ^ (z >> 27)) * 0x94D049BB133111EB) & mask
        yield z ^ (z >> 31)


def prime_factors(n):
    factors, d = [], 2
    while d * d <= n:
        while n % d == 0:
            if d not in factors:
                factors.append(d)
            n //= d
        d += 1
    if n > 1:
        factors.append(n)
    return factors


def field_constants(p, generator, nonresidue):
    """Constants of RFC-0002 'Constants', each checked from its definition."""
    two_adicity = ((p - 1) & -(p - 1)).bit_length() - 1
    odd = (p - 1) >> two_adicity
    factors = prime_factors(p - 1)
    for q in factors:
        assert pow(generator, (p - 1) // q, p) != 1, f"{generator} is not a generator of F_{p}^*"
    assert pow(nonresidue, (p - 1) // 2, p) == p - 1, f"{nonresidue} is a square mod {p}"
    top = pow(generator, odd, p)  # primitive 2^two_adicity-th root of unity
    roots = [pow(top, 1 << (two_adicity - k), p) for k in range(two_adicity + 1)]
    for k, w in enumerate(roots):
        assert pow(w, 1 << k, p) == 1
        assert k == 0 or pow(w, 1 << (k - 1), p) == p - 1
    return {
        "modulus": p,
        "two_adicity": two_adicity,
        "p_minus_1_prime_factors": factors,
        "multiplicative_generator": generator,
        "extension_non_residue": nonresidue,
        "root_of_unity_2_k": roots,
    }


def hex64(x):
    return f"0x{x:016x}"


def main():
    p = 2**64 - 2**32 + 1
    constants = {
        "description": "RFC-0002 constants, derived and checked by spec/vectors/field/generate.py",
        "goldilocks": field_constants(p, 7, 7),
        "f65537": field_constants(2**16 + 1, 3, 3),
    }

    def render(c):
        out = dict(c)
        out["modulus"] = hex64(c["modulus"])
        out["root_of_unity_2_k"] = [hex64(w) for w in c["root_of_unity_2_k"]]
        return out

    for name in ("goldilocks", "f65537"):
        constants[name] = render(constants[name])
    (HERE / "constants.json").write_text(json.dumps(constants, indent=2) + "\n")

    # Operation vectors: edge pairs first, then CASES uniform pairs. Each case is
    # [a, b, a + b, a - b, a * b, a^(-1) or null for a = 0], all mod p.
    edge = [0, 1, p - 1, 2**32 - 1, 2**32, 2**63, p - 2**32]
    rng = splitmix64(SEED)
    pairs = [(a, b) for a in edge for b in edge]
    while len(pairs) < len(edge) ** 2 + CASES:
        a, b = next(rng), next(rng)
        if a < p and b < p:  # rejection keeps inputs uniform on [0, p)
            pairs.append((a, b))
    lines = []
    for a, b in pairs:
        inv = hex64(pow(a, p - 2, p)) if a else None
        case = [hex64(a), hex64(b), hex64((a + b) % p), hex64((a - b) % p), hex64(a * b % p), inv]
        lines.append(json.dumps(case))
    body = ",\n".join(lines)
    header = {
        "description": "Goldilocks reference vectors: [a, b, a+b, a-b, a*b, a^-1] mod p "
        "(a^-1 is null for a = 0); the first 49 cases are all pairs of edge values",
        "modulus": hex64(p),
        "seed": hex64(SEED),
        "edge_values": [hex64(x) for x in edge],
    }
    text = json.dumps(header, indent=2)[:-2] + ',\n  "cases": [\n' + body + "\n  ]\n}\n"
    json.loads(text)  # the file must parse
    (HERE / "goldilocks.json").write_text(text)


if __name__ == "__main__":
    main()
