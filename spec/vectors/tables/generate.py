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
from decimal import ROUND_FLOOR, Decimal, getcontext, localcontext
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


KIND_TAGS = {
    "Range": 0, "Requant": 1, "Pow2": 2, "Bitwise8": 3, "AesSbox": 4, "Gf256Mul": 5,
    "Exp": 6, "Exp2Frac": 7, "Silu": 8, "Rope": 9,
}
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
    elif name == "Exp":
        num, den = params["scale"]
        require(num > 0 and den > 0, "a positive scale")
        out += u8(params["key_bits"]) + u8(params["sigma"])
        out += (num % 2**64).to_bytes(8, "little") + u64(den)
    elif name == "Exp2Frac":
        out += u8(params["frac_bits"]) + u8(params["out_bits"])
    elif name == "Silu":
        out += u8(params["b"]) + sequence([u8(r) for r in params["r_by_class"]])
        out += (params["in_exp"] % 256).to_bytes(1, "little")
        out += (params["out_exp"] % 256).to_bytes(1, "little")
    elif name == "Rope":
        out += u32(params["theta"]) + u16(params["factor"]) + u16(params["low_freq_factor"])
        out += u16(params["high_freq_factor"]) + u32(params["original_max"]) + u8(params["bits"])
        out += u32(params["max_context"]) + u16(params["d_head"])
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


# RFC-0008 'Fixed-point generators'. A Q64 number v stands for v / 2^64.
ONE = 2**64
LN2 = 0xB17217F7D1CF79AB
TERMS = 20


TAU = 0x6487ED5110B4611A6
HALF_PI = 0x1921FB54442D18469


def decimal_pi():
    """π by Machin's formula, 16·atan(1/5) − 4·atan(1/239), at the context's precision."""
    def atan_inv(n):
        x = Decimal(1) / n
        term, total, k = x, x, 1
        while True:
            term *= -x * x
            k += 2
            if abs(term / k) < Decimal(10) ** -(getcontext().prec + 5):
                return total
            total += term / k
    return 16 * atan_inv(5) - 4 * atan_inv(239)


def check_constants():
    """LN2, TAU and HALF_PI are the floors of ln 2, 2π and π/2 times 2^64, computed here with 100
    significant digits (about 330 bits)."""
    with localcontext() as context:
        context.prec = 100
        ln2 = Decimal(2).ln()
        floor = lambda v: int((v * ONE).to_integral_value(rounding=ROUND_FLOOR))
        require(floor(ln2) == LN2, "LN2")
        pi = decimal_pi()
        require(str(pi).startswith("3.14159265358979323846264338327950288419716939937510"), "π")
        require(floor(2 * pi) == TAU and floor(pi / 2) == HALF_PI, "TAU and HALF_PI")
    constants = {
        "description": "The pinned constants of RFC-0008 'Fixed-point generators', each the floor "
        "of the constant times 2^64, computed with Python's decimal module at 100 significant "
        "digits by generate.py.",
        "LN2": hex(LN2),
        "TAU": hex(TAU),
        "HALF_PI": hex(HALF_PI),
    }
    (HERE / "constants.json").write_text(json.dumps(constants, indent=1) + "\n", encoding="utf-8")


def exp_neg(x):
    """(y, n) with e^(−x/2^64) ≈ y · 2^(−64−n): reduction by LN2, then Horner from the top term."""
    n, r = divmod(x, LN2)
    y = ONE
    for i in range(TERMS, 0, -1):
        y = ONE - (r * y) // (i * ONE)
    require(0 < y <= ONE and r * ONE < 2**128, "exp_neg stays in range")
    return y, n


def exp_pos(x):
    """e^(x/2^64) in Q62 for 0 ≤ x ≤ LN2, every step rounded down."""
    require(0 <= x <= LN2, "exp_pos's argument")
    y = 2**62
    for i in range(TERMS, 0, -1):
        y = 2**62 + (x * y) // (i * ONE)
    require(y <= 2**63, "exp_pos stays below 2")
    return y


def exp_value(k, sigma, num, den):
    """round(2^σ · e^(−k·num/den)), half up, in the fixed point of RFC-0008."""
    whole, rem = divmod(k * num, den)
    if whole >= 2**20:
        return 0
    x = whole * ONE + (rem * ONE) // den
    y, n = exp_neg(x)
    if n >= sigma + 2:
        return 0
    return (y * 2**sigma + 2 ** (63 + n)) >> (64 + n)


def exp_tail(rows, sigma, num, den):
    """The first key of Exp's zero tail, in closed form: x(k) = ⌊k·num·2^64/den⌋ reaches
    (σ+2)·LN2, or k·num/den reaches 2^20, from the least such k on."""
    by_halvings = -(-(sigma + 2) * LN2 * den // (num * ONE))
    by_size = -(-(2**20) * den // num)
    return min(rows, by_halvings, by_size)


def exp2frac_value(f, frac_bits, out_bits):
    """⌊2^out · 2^(f/2^frac_bits)⌋ from below: exp_pos of ⌊f·LN2 / 2^frac_bits⌋."""
    return exp_pos((f * LN2) >> frac_bits) >> (62 - out_bits)


def ln_int(n):
    """ln n in Q64 for an integer n ≥ 1: m·LN2 + 2·atanh(z), z = (u − 1)/(u + 1), n = 2^m·u."""
    m = n.bit_length() - 1
    top = 1 << m
    z = ((n - top) << 64) // (n + top)
    w = (z * z) >> 64
    a = ONE // 45
    for k in range(21, -1, -1):
        a = ONE // (2 * k + 1) + ((w * a) >> 64)
    return m * LN2 + ((z * a) >> 63)


def cos_sin(angle):
    """(cos, sin) in Q62 of a Q64 angle: reduced by TAU, then to its quadrant, then Horner."""
    turn = angle % TAU
    quadrant, rest = divmod(turn, HALF_PI)
    r = rest >> 2
    one = 1 << 62
    s = (r * r) >> 62
    c, t = one, one
    for k in range(13, 0, -1):
        c = one - (s * c) // ((2 * k - 1) * (2 * k) * one)
        t = one - (s * t) // ((2 * k) * (2 * k + 1) * one)
    sin = (r * t) >> 62
    return [(c, sin), (-sin, c), (-c, -sin), (sin, -c)][quadrant % 4]


def rope_omegas(p):
    """Each frequency θ^(−2i/d), scaled as Llama-3 scales it, in Q64."""
    ln_theta = ln_int(p["theta"])
    factor, low, high = p["factor"], p["low_freq_factor"], p["high_freq_factor"]
    omegas = []
    for i in range(p["d_head"] // 2):
        y, n = exp_neg(2 * i * ln_theta // p["d_head"])
        omega = y >> n if n < 128 else 0
        l_omega = p["original_max"] * omega
        if TAU * high < l_omega:
            omegas.append(omega)
        elif TAU * low > l_omega:
            omegas.append(omega // factor)
        else:
            smooth = ((l_omega - low * TAU) << 40) // ((high - low) * TAU)
            omegas.append(omega // factor + ((omega * smooth) >> 40) * (factor - 1) // factor)
    return omegas


def rope_pair(omega, position, bits):
    c, s = cos_sin(position * omega)
    round_half_up = lambda v: ((v << bits) + (1 << 61)) >> 62
    return round_half_up(c), round_half_up(s)


def rope_reference(p, position, i):
    """(2^bits·cos, 2^bits·sin) of HF's Llama-3 RoPE at 40 digits, the scaling in real numbers."""
    with localcontext() as context:
        context.prec = 40
        pi = decimal_pi()
        d, theta = p["d_head"], Decimal(p["theta"])
        omega = (-(Decimal(2 * i) / d) * theta.ln()).exp()
        wavelen = 2 * pi / omega
        L = Decimal(p["original_max"])
        low, high, factor = (Decimal(p[k]) for k in ("low_freq_factor", "high_freq_factor", "factor"))
        if wavelen < L / high:
            scaled = omega
        elif wavelen > L / low:
            scaled = omega / factor
        else:
            smooth = (L / wavelen - low) / (high - low)
            scaled = (1 - smooth) * omega / factor + smooth * omega
        angle = (position * scaled) % (2 * pi)
        # Taylor series of cos and sin at the reduced angle, to 45 digits.
        cos, sin, term, k = Decimal(1), Decimal(0), Decimal(1), 0
        sin_term = angle
        sin = sin_term
        while True:
            k += 2
            term *= -angle * angle / ((k - 1) * k)
            sin_term *= -angle * angle / (k * (k + 1))
            if abs(term) < Decimal(10) ** -45 and abs(sin_term) < Decimal(10) ** -45:
                break
            cos += term
            sin += sin_term
        scale = Decimal(2) ** p["bits"]
        return cos * scale, sin * scale


def check_rope(p, pairs):
    """Each listed pair within one unit of the 40-digit reference, both coordinates."""
    omegas = rope_omegas(p)
    for position, i in pairs:
        got = rope_pair(omegas[i], position, p["bits"])
        want = rope_reference(p, position, i)
        for g, w in zip(got, want):
            require(abs(Decimal(g) - w) <= 1, f"Rope({position}, {i}) = {got} against {want}")


def silu_value(h, in_exp, out_exp):
    """clamp(round(silu(h · 2^in_exp) / 2^out_exp), −128, 127): σ(x) in Q63 from e^(−|x|) in Q64."""
    y, n = exp_neg(abs(h) << (64 + in_exp))
    e = y >> n if n < 128 else 0
    sigma = (1 << 127) // (ONE + e) if h >= 0 else (e << 63) // (ONE + e)
    v = h * sigma
    require(abs(v) < 2**94, "Silu's product stays below 2^94")
    s = 63 - in_exp + out_exp
    return min(max((v + (1 << (s - 1))) >> s, -128), 127)


def silu_domain(b, r):
    """Every h = k >> r for k in [−2^(b−1), 2^(b−1)), r < b: from −2^(b−1−r) to 2^(b−1−r) − 1."""
    half = 2 ** (b - 1 - r)
    return range(-half, half)


def check_silu(b, shifts, in_exp, out_exp):
    """Every h of every class within one unit of silu(h · 2^in_exp) / 2^out_exp, clamped to INT8,
    computed with 60 digits."""
    with localcontext() as context:
        context.prec = 60
        scale_in, scale_out = Decimal(2) ** in_exp, Decimal(2) ** out_exp
        for r in shifts:
            for h in silu_domain(b, r):
                x = Decimal(h) * scale_in
                true = x / (1 + (-x).exp()) / scale_out
                true = min(max(true, Decimal(-128)), Decimal(127))
                got = silu_value(h, in_exp, out_exp)
                require(abs(Decimal(got) - true) <= 1, f"Silu(h={h}) = {got} against {true}")


def check_exp(sigma, num, den, keys):
    """Every key's value within one unit of 2^σ · e^(−k·scale) computed with 60 digits."""
    with localcontext() as context:
        context.prec = 60
        for k in keys:
            true = Decimal(2**sigma) * (Decimal(-k * num) / Decimal(den)).exp()
            got = exp_value(k, sigma, num, den)
            require(abs(Decimal(got) - true) <= 1, f"Exp at key {k}: {got} against {true}")


def check_exp2frac(frac_bits, out_bits):
    """Every value at or below ⌊2^out · 2^(f/2^frac_bits)⌋, and at most one unit below it."""
    with localcontext() as context:
        context.prec = 60
        ln2 = Decimal(2).ln()
        for f in range(2**frac_bits):
            true = Decimal(2**out_bits) * (Decimal(f) / Decimal(2**frac_bits) * ln2).exp()
            floor = int(true.to_integral_value(rounding=ROUND_FLOOR))
            got = exp2frac_value(f, frac_bits, out_bits)
            require(got <= floor, f"Exp2Frac({f}) = {got} above the true floor {floor}")
            require(floor - got <= 1, f"Exp2Frac({f}) = {got} more than one below {floor}")


def table(kind):
    """T, the value of each row j < T, and the digest's pieces and core length."""
    rows, value, pieces, core = _table(kind)
    return rows, value, pieces, core


def _table(kind):
    (name, p), = kind.items()
    if name == "Range":
        require(1 <= p["bits"] <= 32, "Range bits")
        return 2 ** p["bits"], lambda j: 0, [], 0
    if name == "Requant":
        b, shifts, lo, hi = p["b"], p["r_by_class"], p["lo"], p["hi"]
        n = len(shifts)
        require(1 <= n <= 4 and b >= 1 and b + (n - 1).bit_length() <= 32, "Requant shape")
        require(all(1 <= r <= 63 for r in shifts) and lo <= hi, "Requant parameters")
        return n * 2**b, lambda j: rq(j % 2**b - 2 ** (b - 1), shifts[j // 2**b], lo, hi), [], 0
    if name == "Pow2":
        require(1 <= p["max_exp"] <= 63, "Pow2 max_exp")
        return p["max_exp"], lambda j: 2**j, [], p["max_exp"]
    if name == "Bitwise8":
        op = {"Xor": lambda a, b: a ^ b, "And": lambda a, b: a & b}[p["op"]]
        return 2**16, lambda j: op(j % 256, j // 256), [], 2**16
    if name == "AesSbox":
        return 2**8, sbox, [], 2**8
    if name == "Gf256Mul":
        return 2**8, lambda j: gf_mul(p["c"], j), [], 2**8
    if name == "Exp":
        key_bits, sigma, (num, den) = p["key_bits"], p["sigma"], p["scale"]
        require(1 <= key_bits <= 32 and sigma <= 32 and num > 0, "Exp parameters")
        tail = exp_tail(2**key_bits, sigma, num, den)
        require(tail <= 2**24, "Exp's zero tail starts by key 2^24")
        return 2**key_bits, lambda k: exp_value(k, sigma, num, den), [tail], tail
    if name == "Silu":
        b, shifts, in_exp, out_exp = p["b"], p["r_by_class"], p["in_exp"], p["out_exp"]
        n = len(shifts)
        require(1 <= n <= 4 and b >= 1 and b + (n - 1).bit_length() <= 32, "Silu shape")
        require(all(r < b for r in shifts), "Silu shifts")
        require(sum(2 ** (b - r) for r in shifts) <= 2**24, "Silu core size")
        require(-64 <= in_exp <= 0 and 1 <= 63 - in_exp + out_exp <= 126, "Silu exponents")

        def silu_row(j):
            c, k = divmod(j, 2**b)
            return silu_value((k - 2 ** (b - 1)) >> shifts[c], in_exp, out_exp)

        core = sum(2 ** (b - r) for r in shifts)
        return n * 2**b, silu_row, [], core
    if name == "Rope":
        d, theta = p["d_head"], p["theta"]
        require(theta >= 2 and p["factor"] >= 1, "Rope θ and factor")
        require(1 <= p["low_freq_factor"] < p["high_freq_factor"], "Rope frequency factors")
        require(p["original_max"] >= 1 and 1 <= p["bits"] <= 30, "Rope context and bits")
        require(d >= 2 and d % 2 == 0, "Rope d_head")
        pairs = p["max_context"] * (d // 2)
        require(1 <= p["max_context"] and pairs <= 2**24, "Rope pair count")
        omegas = rope_omegas(p)

        def rope_row(j):
            position, i = divmod(j, d // 2)
            return rope_pair(omegas[i], position, p["bits"])

        return pairs, rope_row, [], pairs
    if name == "Exp2Frac":
        frac_bits, out_bits = p["frac_bits"], p["out_bits"]
        require(1 <= frac_bits <= 16 and 1 <= out_bits <= 56, "Exp2Frac parameters")
        return 2**frac_bits, lambda f: exp2frac_value(f, frac_bits, out_bits), [], 2**frac_bits
    raise SystemExit(f"tables oracle: unknown kind {name}")


def pack(kind, key, cls):
    """The row a lookup of `key` in `cls` hits, or None (RFC-0008 'Table specifications')."""
    (name, p), = kind.items()
    rows, _, _, _ = table(kind)
    if name in ("Requant", "Silu"):
        half = 2 ** (p["b"] - 1)
        if -half <= key < half and 0 <= cls < len(p["r_by_class"]):
            return key + half + 2 ** p["b"] * cls
        return None
    return key if 0 <= key < rows and cls == 0 else None


def case(name, table_id, kind, lookups, values=True):
    rows, value, pieces, core_len = table(kind)
    log2_rows = (rows - 1).bit_length()
    definition = u32(table_id) + encode_kind(kind) + u8(log2_rows)
    (name_of_kind, params), = kind.items()
    if name_of_kind == "Rope":
        core_values = [v for j in range(core_len) for v in value(j)]
    elif name_of_kind == "Silu":
        core_values = [
            silu_value(h, params["in_exp"], params["out_exp"])
            for r in params["r_by_class"]
            for h in silu_domain(params["b"], r)
        ]
        require(len(core_values) == core_len, "Silu core length")
    else:
        core_values = [value(j) for j in range(core_len)]
    core = sequence([fp(v) for v in core_values])
    digest = hash_oracle.H("table", definition + sequence([u64(x) for x in pieces]) + core)
    out = {
        "name": name,
        "kind": kind,
        "spec": (definition + digest).hex(),
        "rows": rows,
        "log2_rows": log2_rows,
        "pieces": pieces,
        "digest": digest.hex(),
    }
    if values:
        out["values"] = [list(value(j)) if name_of_kind == "Rope" else value(j) for j in range(rows)]
    out["lookups"] = []
    if name_of_kind == "Rope":
        return out
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
    check_constants()
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
    exp = lambda key_bits, sigma, num, den: {
        "Exp": {"key_bits": key_bits, "sigma": sigma, "scale": [num, den]}
    }
    # Test parameters, every key checked; then production ones (σ = 16, 2^22 keys, scales near
    # σ_q σ_k / √128 for σ_q σ_k = 2^-8 and 2^-10), every key of the core checked: past it the
    # true value is below 1/4 and the table holds 0.
    exp_cases = [
        ("exp-k6-s8-quarter", exp(6, 8, 1, 4), True),
        ("exp-k4-s16-no-tail", exp(4, 16, 1, 1000), True),
        ("exp-k8-s16-unit-scale", exp(8, 16, 1, 1), True),
        ("exp-k10-s0", exp(10, 0, 3, 7), True),
        ("exp-k22-s16-405b-wide", exp(22, 16, 1, 2896), False),
        ("exp-k22-s16-405b-narrow", exp(22, 16, 1, 11585), False),
    ]
    for offset, (name, kind, listed) in enumerate(exp_cases):
        params = kind["Exp"]
        rows, _, (tail,), _ = table(kind)
        keys = range(rows) if listed else range(tail + 1)
        check_exp(params["sigma"], *params["scale"], keys)
        if not listed:
            check_exp(params["sigma"], *params["scale"], range(tail, rows, rows // 4096))
        lookups = edges(0, rows) + [(max(tail - 1, 0), 0), (tail, 0)]
        cases.append(case(name, 20 + offset, kind, lookups, values=listed))
    exp2 = lambda frac_bits, out_bits: {"Exp2Frac": {"frac_bits": frac_bits, "out_bits": out_bits}}
    for offset, (name, frac_bits, out_bits, listed) in enumerate([
        ("exp2frac-f8-o16-rfc0017", 8, 16, True),
        ("exp2frac-f1-o1", 1, 1, True),
        ("exp2frac-f16-o56", 16, 56, False),
    ]):
        check_exp2frac(frac_bits, out_bits)
        kind = exp2(frac_bits, out_bits)
        cases.append(case(name, 26 + offset, kind, edges(0, 2**frac_bits), values=listed))
    silu = lambda b, shifts, in_exp, out_exp: {
        "Silu": {"b": b, "r_by_class": shifts, "in_exp": in_exp, "out_exp": out_exp}
    }
    # Test parameters with every row listed; then a 405B-like gate (keys below 2^29, four classes
    # of shifts 14 to 17, 122,880 values of h), every value checked.
    for offset, (name, kind, listed) in enumerate([
        ("silu-b6-two-classes", silu(6, [0, 2], -3, -4), True),
        ("silu-b8-saturating", silu(8, [1], 0, -2), True),
        ("silu-b4-four-classes-extreme-scales", silu(4, [0, 1, 2, 3], -64, -62), True),
        ("silu-b30-four-classes-405b", silu(30, [14, 15, 16, 17], -10, -4), False),
    ]):
        p = kind["Silu"]
        check_silu(p["b"], p["r_by_class"], p["in_exp"], p["out_exp"])
        classes = len(p["r_by_class"])
        lookups = edges(-(2 ** (p["b"] - 1)), 2 ** (p["b"] - 1), classes)
        cases.append(case(name, 29 + offset, kind, lookups, values=listed))
    rope = lambda theta, factor, low, high, original_max, bits, max_context, d_head: {
        "Rope": {
            "theta": theta, "factor": factor, "low_freq_factor": low, "high_freq_factor": high,
            "original_max": original_max, "bits": bits, "max_context": max_context,
            "d_head": d_head,
        }
    }
    # Test tables, every pair listed and checked, with frequencies in all three of Llama-3's
    # bands; then Llama-3.1's RoPE: an 8K-context table pinned by its digest, and sampled pairs
    # of the full 131,072-position table checked against the reference and listed for the Rust
    # test, which checks 10^6 more against an f64 reference.
    for offset, (name, kind, listed) in enumerate([
        ("rope-small-three-bands", rope(10000, 4, 1, 2, 32, 14, 16, 16), True),
        ("rope-small-no-scaling", rope(10000, 1, 1, 4, 64, 14, 12, 6), True),
        ("rope-llama31-8k", rope(500000, 8, 1, 4, 8192, 14, 8192, 128), False),
    ]):
        p = kind["Rope"]
        rows = p["max_context"] * (p["d_head"] // 2)
        if listed:
            check_rope(p, [divmod(j, p["d_head"] // 2) for j in range(rows)])
        entry = case(name, 33 + offset, kind, [], values=listed)
        # The scaled frequencies in Q64, pinned before any rounding to `bits`.
        entry["frequencies"] = [hex(w) for w in rope_omegas(p)]
        cases.append(entry)
    llama = rope(500000, 8, 1, 4, 8192, 14, 131072, 128)["Rope"]
    rng = hash_oracle.splitmix64(0x7E55_E7A0_0071_0003)
    sample = [(next(rng) % 131072, next(rng) % 64) for _ in range(4000)]
    sample += [(131071, i) for i in (0, 28, 29, 34, 35, 63)] + [(0, 0), (1, 1)]
    check_rope(llama, sample)
    omegas = rope_omegas(llama)
    rope_samples = [
        {"position": position, "frequency": i, "constants": list(rope_pair(omegas[i], position, 14))}
        for position, i in sample
    ]
    document = {
        "description": "Table kinds of RFC-0008 'Table specifications': each table's TCE "
        "specification with its digest, row count, log2_rows and the digest's pieces, the value "
        "of every row where 'values' is present (the digest binds the core: the value column of "
        "Pow2, Bitwise8, AesSbox, Gf256Mul and Exp2Frac, Exp's values below its zero tail, and "
        "Silu's values at every h of every class), "
        "and lookups at the edges of the key range, with the row and value they hit or null for "
        "none. Every Exp, Exp2Frac and Silu value is checked against a 60-digit decimal "
        "reference. "
        "Written by generate.py.",
        "cases": cases,
        "rope_llama31_131072": {"kind": rope(500000, 8, 1, 4, 8192, 14, 131072, 128),
                                "samples": rope_samples},
    }
    text = json.dumps(document, indent=1) + "\n"
    (HERE / "tables.json").write_text(text, encoding="utf-8")


if __name__ == "__main__":
    main()
