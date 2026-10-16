#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0 OR MIT
"""Reference vectors for the arithmetic table kinds of RFC-0008 'Table specifications': each
table's TCE specification, row count, digest, rows and lookups at the edges of its key range.

This script is the independent oracle. It is written from RFC-0008 ('Table specifications',
'Generators and digests'), RFC-0003 (the encodings) and RFC-0005 ('Integer semantics', rq)
alone, computing with Python's integers. Its H(domain, bytes) is that of ../hash/generate.py,
whose BLAKE3 is checked against the official test vectors before anything is written. It shares
no code with crates/ or verifier/, which check its output, and where the Rust generator computes
it computes otherwise:

- rq by floor division, not by an arithmetic shift;
- products in GF(2^8) by carry-less multiplication and long division by the AES polynomial, not
  by shift-and-add;
- the S-box from inverses found by search and FIPS-197's bitwise affine formula, not from an
  exponentiation chain and rotations.

It checks those against FIPS-197 itself before writing anything: the products of §4.2 and
§4.2.1, the S-box entries of §5.1.1 and Figure 7, and the AES-128 encryptions of Appendices B and
C.1, run with this S-box and these products.

Usage: python3 spec/vectors/tables/generate.py
(rewrites tables.json)
"""

import importlib.util
import json
import sys
from pathlib import Path

sys.dont_write_bytecode = True  # importing the hash oracle leaves nothing behind

HERE = Path(__file__).resolve().parent
_spec = importlib.util.spec_from_file_location("hash_oracle", HERE.parent / "hash" / "generate.py")
hash_oracle = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(hash_oracle)

P = 2**64 - 2**32 + 1


def require(condition, message):
    if not condition:
        raise SystemExit(f"tables oracle: {message}")


# RFC-0003: fixed-width little-endian integers, two's complement for signed ones; a sequence is a
# u64 length then its elements; an enum is a u16 tag, its declaration index, then its fields.
def u8(x):
    require(0 <= x < 2**8, f"u8 out of range: {x}")
    return x.to_bytes(1, "little")


def u16(x):
    return x.to_bytes(2, "little")


def u32(x):
    return x.to_bytes(4, "little")


def u64(x):
    return x.to_bytes(8, "little")


def i32(x):
    require(-(2**31) <= x < 2**31, f"i32 out of range: {x}")
    return (x % 2**32).to_bytes(4, "little")


def fp(x):
    """A true integer as a canonical Fp in the centered representation (RFC-0002)."""
    require(-(P // 2) <= x <= P // 2, f"value outside the centered range: {x}")
    return (x % P).to_bytes(8, "little")


def sequence(encoded_elements):
    return u64(len(encoded_elements)) + b"".join(encoded_elements)


KIND_TAGS = {"Range": 0, "Requant": 1, "Pow2": 2, "Bitwise8": 3, "AesSbox": 4, "Gf256Mul": 5}
BITOP_TAGS = {"Xor": 0, "And": 1}


def encode_kind(kind):
    (name, params), = kind.items()
    out = u16(KIND_TAGS[name])
    if name == "Range":
        out += u8(params["bits"])
    elif name == "Requant":
        out += u8(params["b"]) + sequence([u8(r) for r in params["r_by_class"]])
        out += i32(params["lo"]) + i32(params["hi"])
    elif name == "Pow2":
        out += u8(params["max_exp"])
    elif name == "Bitwise8":
        out += u16(BITOP_TAGS[params["op"]])
    elif name == "Gf256Mul":
        out += u8(params["c"])
    return out


# GF(2^8) of FIPS-197 §4.2: bytes are polynomials over GF(2), reduced modulo x^8+x^4+x^3+x+1.
AES_POLY = 0x11B


def clmul(a, b):
    product = 0
    for i in range(8):
        if (b >> i) & 1:
            product ^= a << i
    return product


def reduce(p):
    for degree in range(14, 7, -1):
        if (p >> degree) & 1:
            p ^= AES_POLY << (degree - 8)
    return p


def gf_mul(a, b):
    return reduce(clmul(a, b))


INVERSE = {0: 0}
for _x in range(1, 256):
    (_found,) = [y for y in range(1, 256) if gf_mul(_x, y) == 1]
    INVERSE[_x] = _found


def sbox(x):
    """FIPS-197 §5.1.1: b'_i = b_i ⊕ b_(i+4) ⊕ b_(i+5) ⊕ b_(i+6) ⊕ b_(i+7) ⊕ c_i, c = 0x63."""
    b = INVERSE[x]
    out = 0
    for i in range(8):
        bit = (b >> i) ^ (b >> ((i + 4) % 8)) ^ (b >> ((i + 5) % 8))
        bit ^= (b >> ((i + 6) % 8)) ^ (b >> ((i + 7) % 8)) ^ (0x63 >> i)
        out |= (bit & 1) << i
    return out


def aes128_encrypt(plaintext, key):
    """FIPS-197 §5.1 and §5.2 for Nk = 4, Nr = 10, with the S-box and products above."""
    words = [list(key[4 * i : 4 * i + 4]) for i in range(4)]
    rcon = 1
    for i in range(4, 44):
        temp = list(words[i - 1])
        if i % 4 == 0:
            temp = [sbox(b) for b in temp[1:] + temp[:1]]
            temp[0] ^= rcon
            rcon = gf_mul(rcon, 2)
        words.append([a ^ b for a, b in zip(words[i - 4], temp)])
    state = [list(plaintext[4 * c : 4 * c + 4]) for c in range(4)]  # state[column][row]

    def add_round_key(rnd):
        for c in range(4):
            for r in range(4):
                state[c][r] ^= words[4 * rnd + c][r]

    add_round_key(0)
    for rnd in range(1, 11):
        for c in range(4):
            state[c] = [sbox(b) for b in state[c]]
        shifted = [[state[(c + r) % 4][r] for r in range(4)] for c in range(4)]
        state[:] = shifted
        if rnd != 10:
            for c in range(4):
                a = state[c]
                state[c] = [
                    gf_mul(a[0], 2) ^ gf_mul(a[1], 3) ^ a[2] ^ a[3],
                    a[0] ^ gf_mul(a[1], 2) ^ gf_mul(a[2], 3) ^ a[3],
                    a[0] ^ a[1] ^ gf_mul(a[2], 2) ^ gf_mul(a[3], 3),
                    gf_mul(a[0], 3) ^ a[1] ^ a[2] ^ gf_mul(a[3], 2),
                ]
        add_round_key(rnd)
    return bytes(b for column in state for b in column)


def check_products():
    """Every product against a second algorithm: logarithms to the base {03}, a generator of the
    multiplicative group, whose powers come from FIPS-197's xtime alone (§4.2.1)."""
    exp, x = [], 1
    for _ in range(255):
        exp.append(x)
        x = (((x << 1) ^ (0x1B if x & 0x80 else 0)) & 0xFF) ^ x  # x • {03} = xtime(x) ⊕ x
    require(sorted(exp) == list(range(1, 256)), "{03} generates every non-zero byte")
    log = {value: power for power, value in enumerate(exp)}
    for a in range(256):
        for b in range(256):
            expected = 0 if a == 0 or b == 0 else exp[(log[a] + log[b]) % 255]
            require(gf_mul(a, b) == expected, f"{a:#04x} • {b:#04x} by logarithms")


def check_fips197():
    check_products()
    # §4.2 and §4.2.1: {57} • {83} = {c1}; {57} times {02}, {04}, {08}, {10} and {13}.
    require(gf_mul(0x57, 0x83) == 0xC1, "FIPS-197 §4.2: {57} • {83} = {c1}")
    for factor, product in [(0x02, 0xAE), (0x04, 0x47), (0x08, 0x8E), (0x10, 0x07), (0x13, 0xFE)]:
        require(gf_mul(0x57, factor) == product, f"FIPS-197 §4.2.1: {{57}} • {factor:#04x}")
    # §5.1.1 and Figure 7: S({53}) = {ed}; Figure 7's first row and last row.
    require(sbox(0x53) == 0xED, "FIPS-197 §5.1.1: S({53}) = {ed}")
    first = bytes.fromhex("637c777bf26b6fc53001672bfed7ab76")
    last = bytes.fromhex("8ca1890dbfe6426841992d0fb054bb16")
    require(bytes(sbox(x) for x in range(16)) == first, "FIPS-197 Figure 7, row 0")
    require(bytes(sbox(x) for x in range(0xF0, 0x100)) == last, "FIPS-197 Figure 7, row f")
    require(sorted(sbox(x) for x in range(256)) == list(range(256)), "the S-box is a permutation")
    # Appendix B, and Appendix C.1 (AES-128).
    for plaintext, key, ciphertext in [
        ("3243f6a8885a308d313198a2e0370734", "2b7e151628aed2a6abf7158809cf4f3c",
         "3925841d02dc09fbdc118597196a0b32"),
        ("00112233445566778899aabbccddeeff", "000102030405060708090a0b0c0d0e0f",
         "69c4e0d86a7b0430d8cdb78070b4c55a"),
    ]:
        out = aes128_encrypt(bytes.fromhex(plaintext), bytes.fromhex(key))
        require(out.hex() == ciphertext, f"FIPS-197 AES-128 of {plaintext}: {out.hex()}")


def rq(k, r, lo, hi):
    """RFC-0005: clamp(floor((k + 2^(r−1)) / 2^r), lo, hi), r ≥ 1."""
    return min(max((k + 2 ** (r - 1)) // 2**r, lo), hi)


def table(kind):
    """T, the value of each row j < T, and whether the digest's core holds the values."""
    (name, p), = kind.items()
    if name == "Range":
        require(1 <= p["bits"] <= 32, "Range bits")
        return 2 ** p["bits"], lambda j: 0, False
    if name == "Requant":
        b, shifts, lo, hi = p["b"], p["r_by_class"], p["lo"], p["hi"]
        n = len(shifts)
        require(1 <= n <= 4 and b >= 1 and b + (n - 1).bit_length() <= 32, "Requant shape")
        require(all(1 <= r <= 63 for r in shifts) and lo <= hi, "Requant parameters")
        return n * 2**b, lambda j: rq(j % 2**b - 2 ** (b - 1), shifts[j // 2**b], lo, hi), False
    if name == "Pow2":
        require(1 <= p["max_exp"] <= 63, "Pow2 max_exp")
        return p["max_exp"], lambda j: 2**j, True
    if name == "Bitwise8":
        op = {"Xor": lambda a, b: a ^ b, "And": lambda a, b: a & b}[p["op"]]
        return 2**16, lambda j: op(j % 256, j // 256), True
    if name == "AesSbox":
        return 2**8, sbox, True
    if name == "Gf256Mul":
        return 2**8, lambda j: gf_mul(p["c"], j), True
    raise SystemExit(f"tables oracle: unknown kind {name}")


def pack(kind, key, cls):
    """The row a lookup of `key` in `cls` hits, or None (RFC-0008 'Table specifications')."""
    (name, p), = kind.items()
    rows, _, _ = table(kind)
    if name == "Requant":
        half = 2 ** (p["b"] - 1)
        if -half <= key < half and 0 <= cls < len(p["r_by_class"]):
            return key + half + 2 ** p["b"] * cls
        return None
    return key if 0 <= key < rows and cls == 0 else None


def case(name, table_id, kind, lookups, values=True):
    rows, value, enumerated = table(kind)
    log2_rows = (rows - 1).bit_length()
    definition = u32(table_id) + encode_kind(kind) + u8(log2_rows)
    core = sequence([fp(value(j)) for j in range(rows)]) if enumerated else u64(0)
    digest = hash_oracle.H("table", definition + u64(0) + core)
    out = {
        "name": name,
        "kind": kind,
        "spec": (definition + digest).hex(),
        "rows": rows,
        "log2_rows": log2_rows,
        "digest": digest.hex(),
    }
    if values:
        out["values"] = [value(j) for j in range(rows)]
    out["lookups"] = []
    for key, cls in lookups:
        row = pack(kind, key, cls)
        out["lookups"].append(
            {"key": key, "class": cls, "row": row, "value": None if row is None else value(row)}
        )
    return out


def edges(low, high, classes=1):
    """Keys at and just past each end of [low, high), in the first and last class and past them."""
    keys = [low, high - 1, low - 1, high, (low + high) // 2]
    return [(k, c) for c in sorted({0, classes - 1}) for k in keys] + [(low, classes), (-(2**62), 0)]


def main():
    hash_oracle.check_official_vectors()
    check_fips197()
    requant = lambda b, shifts, lo, hi: {"Requant": {"b": b, "r_by_class": shifts, "lo": lo, "hi": hi}}
    cases = [
        case("range-1", 0, {"Range": {"bits": 1}}, edges(0, 2)),
        case("range-8", 1, {"Range": {"bits": 8}}, edges(0, 256)),
        case("range-32", 2, {"Range": {"bits": 32}}, edges(0, 2**32), values=False),
        case("requant-b4-one-class", 3, requant(4, [1], -128, 127), edges(-8, 8)),
        case("requant-b5-three-classes-saturating", 4, requant(5, [1, 2, 3], -2, 3), edges(-16, 16, 3)),
        case("requant-b8-four-classes", 5, requant(8, [1, 2, 7, 8], -128, 127), edges(-128, 128, 4)),
        case("requant-b3-shift-63", 6, requant(3, [63], -1, 1), edges(-4, 4)),
        case("requant-lo-equals-hi", 7, requant(4, [2], 5, 5), edges(-8, 8)),
        case("requant-b31-two-classes", 8, requant(31, [1, 40], -(2**31), 2**31 - 1),
             edges(-(2**30), 2**30, 2), values=False),
        case("requant-b30-three-classes", 9, requant(30, [1, 2, 3], -(2**19), 2**19 - 1),
             edges(-(2**29), 2**29, 3), values=False),
        case("pow2-1", 10, {"Pow2": {"max_exp": 1}}, edges(0, 1)),
        case("pow2-5", 11, {"Pow2": {"max_exp": 5}}, edges(0, 5)),
        case("pow2-63", 12, {"Pow2": {"max_exp": 63}}, edges(0, 63)),
        case("bitwise8-xor", 13, {"Bitwise8": {"op": "Xor"}}, edges(0, 2**16) + [(0xCA53, 0)],
             values=False),
        case("bitwise8-and", 14, {"Bitwise8": {"op": "And"}}, edges(0, 2**16) + [(0xCA53, 0)],
             values=False),
        case("aes-sbox", 15, {"AesSbox": {}}, edges(0, 256) + [(0x53, 0)]),
        case("gf256-mul-0", 16, {"Gf256Mul": {"c": 0}}, edges(0, 256)),
        case("gf256-mul-2", 17, {"Gf256Mul": {"c": 2}}, edges(0, 256)),
        case("gf256-mul-3", 18, {"Gf256Mul": {"c": 3}}, edges(0, 256)),
        case("gf256-mul-57", 19, {"Gf256Mul": {"c": 0x57}}, edges(0, 256) + [(0x83, 0), (0x13, 0)]),
    ]
    document = {
        "description": "Arithmetic table kinds of RFC-0008 'Table specifications': each table's TCE "
        "specification with its digest, row count, log2_rows, the value of every row where "
        "'values' is present (the digest binds the value column of Pow2, Bitwise8, AesSbox and "
        "Gf256Mul), and lookups at the edges of the key range, with the row and value they hit "
        "or null for none. Written by generate.py.",
        "cases": cases,
    }
    text = json.dumps(document, indent=1) + "\n"
    (HERE / "tables.json").write_text(text, encoding="utf-8")


if __name__ == "__main__":
    main()
