# SPDX-License-Identifier: Apache-2.0 OR MIT
"""Strict Ed25519 verification vectors: RFC 8032 under RFC-0060 'Signatures'.

An independent oracle in standard-library Python, written from two documents only:

- RFC 8032, Section 5.1: Ed25519's field, curve, point encoding (5.1.2), decoding (5.1.3), point
  addition and doubling (5.1.4), key generation (5.1.5) and signing (5.1.6). Section 7.1's TEST 1,
  TEST 2, TEST 3 and TEST SHA(abc) are checked before anything is written.
- docs/rfcs/RFC-0060-log-manifests.md, 'Signatures': the acceptance set of strict verification.
  A signature (R, S) on a message M under the key A verifies when S < L, R and A decode to points
  (RFC 8032 5.1.3, except that a y in [p, 2^255) is accepted and reduced modulo p), neither point
  is of small order, and R's 32 bytes equal the canonical encoding of [S]B - [k]A, with
  k = SHA-512(R || A || M) reduced modulo L before it multiplies A.

`strict_verify` returns the first rule that fails, in the order of CHECK_ORDER, or "accept". Every
case is built to reach one outcome, which `classify` predicts without the decoder (Euler's
criterion and the small-order y values), and an assertion checks that `strict_verify` agrees.
Every pseudo-random choice is SHA-512 of a fixed label and a counter, so the output is the same
on every run.

Run from the repository root: python3 -I -B spec/vectors/ed25519/generate.py
It writes spec/vectors/ed25519/strict.json.
"""

import hashlib
import json
from pathlib import Path

# RFC 8032 5.1, Table 1.
P = 2**255 - 19
L = 2**252 + 27742317777372353535851937790883648493
D = -121665 * pow(121666, P - 2, P) % P
SQRT_M1 = pow(2, (P - 1) // 4, P)
Y_MASK = 2**255 - 1

CHECK_ORDER = [
    "s-out-of-range",
    "r-not-a-point",
    "key-not-a-point",
    "r-small-order",
    "key-small-order",
    "equation",
]
OUTCOMES = ["accept"] + CHECK_ORDER

LABEL = b"tessera/spec/vectors/ed25519/strict/"


def sha512(data):
    return hashlib.sha512(data).digest()


def le(data):
    return int.from_bytes(data, "little")


def le32(n):
    return n.to_bytes(32, "little")


def prf(label, counter):
    """64 pseudo-random bytes: SHA-512 of the set's prefix, a label and a counter."""
    return sha512(LABEL + label.encode() + b"/" + str(counter).encode())


def stream(label, n):
    """n pseudo-random bytes from consecutive counters of one label."""
    out = b""
    i = 0
    while len(out) < n:
        out += prf(label, i)
        i += 1
    return out[:n]


def scalar(label, counter):
    """A pseudo-random scalar in [1, L)."""
    s = le(prf(label, counter)) % L
    assert s != 0
    return s


# ---------------------------------------------------------------------------------------------
# The field and the curve (RFC 8032 5.1.1 to 5.1.4).


def inv(x):
    assert x % P != 0
    return pow(x, P - 2, P)


def sqrt_ratio(u, v):
    """RFC 8032 5.1.3 steps 2 and 3: a root x of x^2 = u/v (mod p), or None if there is none."""
    x = u * pow(v, 3, P) * pow(u * pow(v, 7, P), (P - 5) // 8, P) % P
    vxx = v * x * x % P
    if vxx == u % P:
        return x
    if vxx == (-u) % P:
        return x * SQRT_M1 % P
    return None


# Points in extended homogeneous coordinates (X, Y, Z, T): x = X/Z, y = Y/Z, xy = T/Z.
IDENTITY = (0, 1, 1, 0)


def from_affine(x, y):
    return (x % P, y % P, 1, x * y % P)


def add(p1, p2):
    """RFC 8032 5.1.4's addition, complete on edwards25519."""
    x1, y1, z1, t1 = p1
    x2, y2, z2, t2 = p2
    a = (y1 - x1) * (y2 - x2) % P
    b = (y1 + x1) * (y2 + x2) % P
    c = t1 * 2 * D * t2 % P
    d = z1 * 2 * z2 % P
    e, f, g, h = b - a, d - c, d + c, b + a
    return (e * f % P, g * h % P, f * g % P, e * h % P)


def double(p1):
    """RFC 8032 5.1.4's doubling."""
    x1, y1, z1, _ = p1
    a = x1 * x1 % P
    b = y1 * y1 % P
    c = 2 * z1 * z1 % P
    h = a + b
    e = h - (x1 + y1) * (x1 + y1)
    g = a - b
    f = c + g
    return (e * f % P, g * h % P, f * g % P, e * h % P)


def neg(p1):
    x1, y1, z1, t1 = p1
    return ((-x1) % P, y1, z1, (-t1) % P)


def mul(k, p1):
    """[k]p1 for an integer k >= 0, by double and add from the most significant bit."""
    assert k >= 0
    q = IDENTITY
    for bit in bin(k)[2:]:
        q = double(q)
        if bit == "1":
            q = add(q, p1)
    return q


def affine(p1):
    zi = inv(p1[2])
    return (p1[0] * zi % P, p1[1] * zi % P)


def equal(p1, p2):
    return (p1[0] * p2[2] - p2[0] * p1[2]) % P == 0 and (p1[1] * p2[2] - p2[1] * p1[2]) % P == 0


def is_identity(p1):
    return equal(p1, IDENTITY)


def on_curve(p1):
    """-x^2 + y^2 = 1 + d x^2 y^2."""
    x, y = affine(p1)
    return (-x * x + y * y - 1 - D * x * x * y * y) % P == 0


def is_small_order(p1):
    """[8]p1 is the identity."""
    return is_identity(double(double(double(p1))))


def torsion_order(p1):
    """The order of a point of small order: 1, 2, 4 or 8."""
    q = p1
    for n in (1, 2, 4, 8):
        if is_identity(q):
            return n
        q = double(q)
    raise AssertionError("not of small order")


def order_text(p1):
    """The order of any point of edwards25519, whose group is cyclic of order 8L, as text."""
    small = torsion_order(mul(L, p1))
    if is_small_order(p1):
        return str(small)
    return "L" if small == 1 else f"{small}L"


def encode(p1):
    """RFC 8032 5.1.2: y in little endian, the low bit of x in bit 255. Always canonical."""
    x, y = affine(p1)
    return le32(y | (x & 1) << 255)


def decode(data, x_zero_sign=False):
    """RFC 8032 5.1.3 as RFC-0060 amends it, or None if the bytes are not a point.

    Step 1 differs: a y in [p, 2^255) is accepted and reduced modulo p, where RFC 8032 refuses it.
    Step 4 is RFC 8032's: x = 0 with x_0 = 1 is refused. `x_zero_sign=True` accepts it as x = 0,
    for the constructions only (never in `strict_verify`).
    """
    assert len(data) == 32
    n = le(data)
    sign = n >> 255
    y = n & Y_MASK
    if y >= P:
        y -= P
    u = (y * y - 1) % P
    v = (D * y * y + 1) % P
    x = sqrt_ratio(u, v)
    if x is None:
        return None
    if x == 0 and sign == 1 and not x_zero_sign:
        return None
    if (x & 1) != sign:
        x = (P - x) % P
    return from_affine(x, y)


B = decode(le32(4 * inv(5) % P))
assert B is not None and affine(B)[0] & 1 == 0
assert affine(B) == (  # RFC 8032 5.1, Table 1
    15112221349535400772501151409588531511454012693041857206046113283949847762202,
    46316835694926478169428394003475163141307993866256225615783033603165251855960,
)
assert on_curve(B) and is_identity(mul(L, B)) and not is_identity(B)


# ---------------------------------------------------------------------------------------------
# Strict verification (RFC-0060 'Signatures') and the cofactored equation.


def challenge(r_bytes, a_bytes, message):
    """k = SHA-512(R || A || M) modulo L, over the bytes exactly as given."""
    return le(sha512(r_bytes + a_bytes + message)) % L


def strict_verify(a_bytes, message, signature):
    """The first failing rule of CHECK_ORDER, or "accept"."""
    assert len(a_bytes) == 32 and len(signature) == 64
    r_bytes = signature[:32]
    s = le(signature[32:])
    if s >= L:
        return "s-out-of-range"
    r_point = decode(r_bytes)
    if r_point is None:
        return "r-not-a-point"
    a_point = decode(a_bytes)
    if a_point is None:
        return "key-not-a-point"
    if is_small_order(r_point):
        return "r-small-order"
    if is_small_order(a_point):
        return "key-small-order"
    k = challenge(r_bytes, a_bytes, message)
    if encode(add(mul(s, B), neg(mul(k, a_point)))) != r_bytes:
        return "equation"
    return "accept"


def cofactored(a_bytes, message, signature):
    """[8][S]B = [8]R + [8][k]A, with S < L and both points decoding as `decode` does."""
    r_bytes = signature[:32]
    s = le(signature[32:])
    r_point = decode(r_bytes)
    a_point = decode(a_bytes)
    if s >= L or r_point is None or a_point is None:
        return False
    k = challenge(r_bytes, a_bytes, message)
    return is_small_order(add(mul(s, B), neg(add(r_point, mul(k, a_point)))))


# ---------------------------------------------------------------------------------------------
# Keys and signatures (RFC 8032 5.1.5 and 5.1.6).


def expand(secret):
    """The pruned secret scalar a and the prefix, from SHA-512 of the 32-byte secret key."""
    assert len(secret) == 32
    h = bytearray(sha512(secret))
    h[0] &= 0b11111000
    h[31] &= 0b01111111
    h[31] |= 0b01000000
    return le(bytes(h[:32])), bytes(h[32:])


def public_key(secret):
    a, _ = expand(secret)
    return encode(mul(a, B))


def nonce(secret, message):
    _, prefix = expand(secret)
    return le(sha512(prefix + message)) % L


def sign(secret, message):
    a, _ = expand(secret)
    a_bytes = public_key(secret)
    r = nonce(secret, message)
    r_bytes = encode(mul(r, B))
    k = challenge(r_bytes, a_bytes, message)
    return r_bytes + le32((r + k * a) % L)


# RFC 8032 7.1: TEST 1, TEST 2, TEST 3 and TEST SHA(abc), as secret key, public key, message and
# signature.
RFC8032_TESTS = [
    (
        "9d61b19deffd5a60ba844af492ec2cc44449c5697b326919703bac031cae7f60",
        "d75a980182b10ab7d54bfed3c964073a0ee172f3daa62325af021a68f707511a",
        "",
        "e5564300c360ac729086e2cc806e828a84877f1eb8e5d974d873e06522490155"
        "5fb8821590a33bacc61e39701cf9b46bd25bf5f0595bbe24655141438e7a100b",
    ),
    (
        "4ccd089b28ff96da9db6c346ec114e0f5b8a319f35aba624da8cf6ed4fb8a6fb",
        "3d4017c3e843895a92b70aa74d1b7ebc9c982ccf2ec4968cc0cd55f12af4660c",
        "72",
        "92a009a9f0d4cab8720e820b5f642540a2b27b5416503f8fb3762223ebdb69da"
        "085ac1e43e15996e458f3613d0f11d8c387b2eaeb4302aeeb00d291612bb0c00",
    ),
    (
        "c5aa8df43f9f837bedb7442f31dcb7b166d38535076f094b85ce3a2e0b4458f7",
        "fc51cd8e6218a1a38da47ed00230f0580816ed13ba3303ac5deb911548908025",
        "af82",
        "6291d657deec24024827e69c3abe01a30ce548a284743a445e3680d7db5ac3ac"
        "18ff9b538d16f290ae67f760984dc6594a7c15e9716ed28dc027beceea1ec40a",
    ),
    (
        "833fe62409237b9d62ec77587520911e9a759cec1d19755b7da901b96dca3d42",
        "ec172b93ad5e563bf4932c70e1245034c35467ef2efd4d64ebf819683467e2bf",
        "ddaf35a193617abacc417349ae20413112e6fa4e89a97ea20a9eeee64b55d39a"
        "2192992a274fc1a836ba3c23a3feebbd454d4423643ce80e2a9ac94fa54ca49f",
        "dc2a4459e7369633a52b1bf277839a00201009a3efbf3ecb69bea2186c26b589"
        "09351fc9ac90b3ecfdfbc7c66431e0303dca179c138ac17ad9bef1177331a704",
    ),
]


def check_rfc8032():
    for secret, public, message, signature in RFC8032_TESTS:
        secret, public = bytes.fromhex(secret), bytes.fromhex(public)
        message, signature = bytes.fromhex(message), bytes.fromhex(signature)
        assert public_key(secret) == public, "RFC 8032 7.1 key generation"
        assert sign(secret, message) == signature, "RFC 8032 7.1 signing"
        assert strict_verify(public, message, signature) == "accept", "RFC 8032 7.1 verification"
        assert cofactored(public, message, signature)
        assert strict_verify(public, message + b"\x00", signature) == "equation"
    assert sha512(b"abc").hex() == RFC8032_TESTS[3][2], "TEST SHA(abc)'s message is SHA-512(abc)"


# ---------------------------------------------------------------------------------------------
# The torsion subgroup: eight points, cyclic, generated by any point of order 8.


def find_order_8():
    """[L]P for the first decodable y = 2, 3, ... whose [L]P has order 8."""
    y = 2
    while True:
        pt = decode(le32(y))
        if pt is not None:
            t = mul(L, pt)
            if torsion_order(t) == 8:
                return t
        y += 1


T8 = find_order_8()
TORSION = [mul(i, T8) for i in range(8)]  # TORSION[i] = [i]T8, of order 8 / gcd(i, 8)
assert all(on_curve(t) for t in TORSION)
assert len({encode(t) for t in TORSION}) == 8
assert all(equal(add(TORSION[i], TORSION[j]), TORSION[(i + j) % 8]) for i in range(8) for j in range(8))
assert [torsion_order(t) for t in TORSION] == [1, 8, 4, 8, 2, 8, 4, 8]
assert affine(TORSION[4]) == (0, P - 1), "the point of order 2 is (0, -1)"
assert {affine(TORSION[2]), affine(TORSION[6])} == {(SQRT_M1, 0), (P - SQRT_M1, 0)}
# y^2 of an order-8 point solves d y^4 + 2 y^2 - 1 = 0: doubling it gives y = 0, so x^2 = -y^2.
ORDER_8_Y = {affine(TORSION[i])[1] for i in (1, 3, 5, 7)}
assert len(ORDER_8_Y) == 2 and all((D * y**4 + 2 * y * y - 1) % P == 0 for y in ORDER_8_Y)
SMALL_ORDER_Y = {0, 1, P - 1} | ORDER_8_Y
T_OF_ORDER = {2: TORSION[4], 4: TORSION[2], 8: TORSION[1]}


def classify(data):
    """What `decode` should make of 32 bytes, predicted without it: "not-a-point", "small"
    (a point of small order) or "large", from Euler's criterion on (y^2 - 1)/(d y^2 + 1) and the
    small-order y values."""
    n = le(data)
    sign = n >> 255
    y = n & Y_MASK
    if y >= P:
        y -= P
    if y in (1, P - 1):  # x = 0
        return "not-a-point" if sign else "small"
    if y in SMALL_ORDER_Y:
        return "small"
    w = (y * y - 1) * inv(D * y * y + 1) % P
    return "large" if pow(w, (P - 1) // 2, P) == 1 else "not-a-point"


# ---------------------------------------------------------------------------------------------
# The cases.

CASES = []


def case(name, group, a_bytes, message, signature, intended, cofactored_intended, comment):
    """Records a case after checking that strict verification and the cofactored equation give
    what its construction intends."""
    assert intended in OUTCOMES
    got = strict_verify(a_bytes, message, signature)
    assert got == intended, f"{name}: strict_verify gives {got}, the construction {intended}"
    cof = cofactored(a_bytes, message, signature)
    assert cof == cofactored_intended, f"{name}: cofactored is {cof}"
    assert all(c["name"] != name for c in CASES), f"duplicate case name {name}"
    assert comment.endswith(".") and "\n" not in comment
    CASES.append(
        {
            "name": name,
            "group": group,
            "public_key": a_bytes.hex(),
            "message": message.hex(),
            "signature": signature.hex(),
            "expected": intended,
            "cofactored": cof,
            "comment": comment,
        }
    )


def r_outcome(data):
    return {"not-a-point": "r-not-a-point", "small": "r-small-order"}.get(classify(data))


def key_outcome(data):
    return {"not-a-point": "key-not-a-point", "small": "key-small-order"}.get(classify(data))


SEEDS = [prf("seed", i)[:32] for i in range(4)]
KEYS = [public_key(seed) for seed in SEEDS]
SCALARS = [expand(seed)[0] for seed in SEEDS]
HONEST_LENGTHS = [0, 1, 3, 32, 63, 64, 65, 111, 112, 127, 128, 129, 200, 255, 256, 1000]


def honest_message(n):
    return stream(f"message/{n}", n)


def honest():
    """16 signatures of RFC 8032 5.1.6 from four seeds; returns them by message length."""
    out = {}
    for i, n in enumerate(HONEST_LENGTHS):
        seed = i % 4
        message = honest_message(n)
        signature = sign(SEEDS[seed], message)
        case(
            f"honest-seed{seed}-len{n}",
            "honest",
            KEYS[seed],
            message,
            signature,
            "accept",
            True,
            f"An RFC 8032 signature by seed {seed} on a {n}-byte message.",
        )
        out[n] = (seed, message, signature)
    return out


def message_group(signed):
    for n, where in [(1, "first"), (111, "middle"), (1000, "last")]:
        seed, message, signature = signed[n]
        bit = {"first": 0, "middle": 4 * n, "last": 8 * n - 1}[where]
        flipped = bytearray(message)
        flipped[bit // 8] ^= 1 << (bit % 8)
        case(
            f"message-{where}-bit-flipped-len{n}",
            "message",
            KEYS[seed],
            bytes(flipped),
            signature,
            "equation",
            False,
            f"The honest {n}-byte signature with bit {bit} of the message flipped, the {where} bit.",
        )


def scalar_group(signed):
    seed, message, signature = signed[32]
    r_bytes, s = signature[:32], le(signature[32:])
    variants = [
        ("s-plus-l", s + L, "S + L, the malleated signature RFC-0060 refuses"),
        ("s-plus-2l", s + 2 * L, "S + 2L"),
    ]
    if s + 15 * L < 2**256:
        variants.append(("s-plus-15l", s + 15 * L, "S + 15L, below 2^256"))
    variants += [
        ("s-equals-l", L, "S = L"),
        ("s-all-ones", 2**256 - 1, "S = 2^256 - 1"),
        ("s-bit-255-set", s | 2**255, "the honest S with bit 255 also set"),
    ]
    for name, value, text in variants:
        assert L <= value < 2**256
        case(
            f"scalar-{name}",
            "scalar",
            KEYS[seed],
            message,
            r_bytes + le32(value),
            "s-out-of-range",
            False,
            f"The honest 32-byte signature with {text}, which is not below L.",
        )
    for name, value in [("s-l-minus-1", L - 1), ("s-zero", 0)]:
        case(
            f"scalar-{name}",
            "scalar",
            KEYS[seed],
            message,
            r_bytes + le32(value),
            "equation",
            False,
            f"The honest R with S = {'L - 1' if value else '0'}, in range but not the signer's.",
        )


def noncanonical_encodings():
    """The 38 encodings with y in [p, 2^255), both sign bits, as (name, bytes, reduced y, sign)."""
    out = []
    for offset in range(19):
        for sign_bit in (0, 1):
            data = le32((P + offset) | sign_bit << 255)
            out.append((f"y-p-plus-{offset}-sign{sign_bit}", data, offset, sign_bit))
    assert le(out[-1][1]) == 2**256 - 1
    return out


def nonsquare_encodings():
    """Three canonical encodings of a y with no x: pseudo-random y below p, sign bits 0, 1, 0."""
    out = []
    counter = 0
    while len(out) < 3:
        y = le(prf("non-square-y", counter)[:32]) % P
        counter += 1
        if classify(le32(y)) != "not-a-point":
            continue
        sign_bit = 1 if len(out) == 1 else 0
        data = le32(y | sign_bit << 255)
        assert decode(data) is None
        out.append((f"y-non-square-{len(out)}-sign{sign_bit}", data))
    return out


def describe(data):
    pt = decode(data)
    if pt is None:
        if decode(data, x_zero_sign=True) is not None:
            return "is x = 0 with sign bit 1, which RFC 8032 5.1.3 refuses"
        return "has no x on the curve"
    return f"decodes to a point of order {order_text(pt)}"


def r_encoding_group():
    seed = 2
    a = SCALARS[seed]
    message = b"tessera ed25519 strict vectors: non-canonical R"
    decoded = {}
    for name, data, offset, sign_bit in noncanonical_encodings():
        k = challenge(data, KEYS[seed], message)
        signature = data + le32(k * a % L)
        kind = classify(data)
        intended = r_outcome(data) or "equation"
        case(
            f"r-noncanonical-{name}",
            "r-encoding",
            KEYS[seed],
            message,
            signature,
            intended,
            kind == "small",
            f"R has y = p + {offset} and sign bit {sign_bit}, which reduces to y = {offset} and "
            f"{describe(data)}; S = k*a makes [S]B - [k]A the identity.",
        )
        decoded[(offset, sign_bit)] = describe(data)
    for name, data in nonsquare_encodings():
        k = challenge(data, KEYS[seed], message)
        case(
            f"r-{name}",
            "r-encoding",
            KEYS[seed],
            message,
            data + le32(k * a % L),
            "r-not-a-point",
            False,
            "R is a canonical y below p whose (y^2 - 1)/(d y^2 + 1) is not a square.",
        )
    found = 0
    counter = 0
    while found < 2:
        data = prf("random-r", counter)[:32]
        counter += 1
        if classify(data) != "not-a-point":
            continue
        k = challenge(data, KEYS[seed], message)
        case(
            f"r-random-bytes-{found}",
            "r-encoding",
            KEYS[seed],
            message,
            data + le32(k * a % L),
            "r-not-a-point",
            False,
            "R is 32 pseudo-random bytes that do not decode to a point.",
        )
        found += 1
    return decoded


def key_encoding_group():
    message = b"tessera ed25519 strict vectors: non-canonical key"
    r = scalar("key-encoding/r", 0)
    r_bytes = encode(mul(r, B))
    signature = r_bytes + le32(r)
    for name, data, offset, sign_bit in noncanonical_encodings():
        kind = classify(data)
        case(
            f"key-noncanonical-{name}",
            "key-encoding",
            data,
            message,
            signature,
            key_outcome(data) or "equation",
            kind == "small",
            f"A has y = p + {offset} and sign bit {sign_bit}, which reduces to y = {offset} and "
            f"{describe(data)}; R = [r]B and S = r.",
        )
    for name, data in nonsquare_encodings():
        case(
            f"key-{name}",
            "key-encoding",
            data,
            message,
            signature,
            "key-not-a-point",
            False,
            "A is a canonical y below p whose (y^2 - 1)/(d y^2 + 1) is not a square.",
        )


def small_order_encodings():
    """The 14 encodings of small-order points, as (name, bytes, the point x = 0 allowed)."""
    canonical = []
    low_y8 = min(ORDER_8_Y)
    for t in TORSION:
        order = torsion_order(t)
        x, y = affine(t)
        if order == 1:
            name = "identity"
        elif order == 2:
            name = "order2"
        elif order == 4:
            name = f"order4-sign{x & 1}"
        else:
            name = f"order8-y{'low' if y == low_y8 else 'high'}-sign{x & 1}"
        canonical.append((f"canonical-{name}", encode(t)))
    noncanonical = [
        ("noncanonical-y-p-sign0", le32(P)),
        ("noncanonical-y-p-sign1", le32(P | 2**255)),
        ("noncanonical-y-p-plus-1-sign0", le32(P + 1)),
        ("noncanonical-y-p-plus-1-sign1", le32((P + 1) | 2**255)),
        ("negative-zero-identity", le32(1 | 2**255)),
        ("negative-zero-order2", le32((P - 1) | 2**255)),
    ]
    out = []
    for name, data in canonical + noncanonical:
        pt = decode(data, x_zero_sign=True)
        assert pt is not None and is_small_order(pt)
        out.append((name, data, pt))
    assert len({data for _, data, _ in out}) == 14
    assert sum(classify(data) == "not-a-point" for _, data, _ in out) == 3
    return out


def small_order_r_group():
    seed = 3
    a = SCALARS[seed]
    message = b"tessera ed25519 strict vectors: small-order R"
    for name, data, pt in small_order_encodings():
        k = challenge(data, KEYS[seed], message)
        signature = data + le32(k * a % L)
        decodes = classify(data) != "not-a-point"
        holds = encode(add(mul(k * a % L, B), neg(mul(k, decode(KEYS[seed]))))) == data
        assert holds == (name == "canonical-identity")
        what = (
            f"a point of order {torsion_order(pt)}"
            if decodes
            else "x = 0 with sign bit 1, refused by RFC 8032 5.1.3"
        )
        tail = "; the cofactorless equation holds too" if holds else ""
        case(
            f"small-order-r-{name}",
            "small-order-r",
            KEYS[seed],
            message,
            signature,
            "r-small-order" if decodes else "r-not-a-point",
            decodes,
            f"R encodes {what}, with S = k*a under an honest key{tail}.",
        )


def small_order_key_group():
    message = b"tessera ed25519 strict vectors: small-order key"
    for index, (name, data, pt) in enumerate(small_order_encodings()):
        order = torsion_order(pt)
        for counter in range(1024):
            r = scalar(f"small-order-key/{index}", counter)
            r_bytes = encode(mul(r, B))
            k = challenge(r_bytes, data, message)
            if k % order == 0:
                break
        else:
            raise AssertionError(f"no r makes k a multiple of {order} for {name}")
        # [S]B - [k]A = [r]B - [k]T = R: the cofactorless equation holds on the point itself.
        assert encode(add(mul(r, B), neg(mul(k, pt)))) == r_bytes
        decodes = classify(data) != "not-a-point"
        what = f"a point of order {order}" if decodes else "x = 0 with sign bit 1"
        case(
            f"small-order-key-{name}",
            "small-order-key",
            data,
            message,
            r_bytes + le32(r),
            "key-small-order" if decodes else "key-not-a-point",
            decodes,
            f"A encodes {what}, and R = [r]B, S = r with k a multiple of {order}, so "
            f"[S]B - [k]A = R.",
        )


def grind(label, accept_if):
    """The first counter's r in [1, L), with R = [r]B, for which accept_if(r, R bytes) holds."""
    for counter in range(1024):
        r = scalar(label, counter)
        r_bytes = encode(mul(r, B))
        if accept_if(r, r_bytes):
            return r, r_bytes
    raise AssertionError(f"grinding {label} found nothing")


def mixed_order_key_group():
    seed = 0
    a = SCALARS[seed]
    message = b"tessera ed25519 strict vectors: mixed-order key"
    for order, t in sorted(T_OF_ORDER.items()):
        a_point = add(mul(a, B), t)
        a_bytes = encode(a_point)
        assert not is_small_order(a_point) and order_text(a_point) == f"{order}L"
        for divides in (True, False):

            def wanted(r, r_bytes, a_bytes=a_bytes, order=order, divides=divides):
                return (challenge(r_bytes, a_bytes, message) % order == 0) == divides

            r, r_bytes = grind(f"mixed-order-key/{order}/{divides}", wanted)
            k = challenge(r_bytes, a_bytes, message)
            signature = r_bytes + le32((r + k * a) % L)
            if divides:
                name, intended = "k-multiple", "accept"
                text = f"a multiple of {order}, so [k]T vanishes and the cofactorless equation holds"
            else:
                name, intended = "k-not-multiple", "equation"
                text = f"not a multiple of {order}, so [S]B - [k]A = R - [k]T"
            case(
                f"mixed-order-key-order{order}-{name}",
                "mixed-order-key",
                a_bytes,
                message,
                signature,
                intended,
                True,
                f"A = [a]B + T with T of order {order} and S = r + k*a, k {text}.",
            )


def mixed_order_r_group():
    message = b"tessera ed25519 strict vectors: mixed-order R"
    seed = 1
    a = SCALARS[seed]
    for order, t in sorted(T_OF_ORDER.items()):
        r = scalar(f"mixed-order-r/{order}", 0)
        r_point = add(mul(r, B), t)
        r_bytes = encode(r_point)
        assert order_text(r_point) == f"{order}L"
        k = challenge(r_bytes, KEYS[seed], message)
        case(
            f"mixed-order-r-order{order}-honest-key",
            "mixed-order-r",
            KEYS[seed],
            message,
            r_bytes + le32((r + k * a) % L),
            "equation",
            True,
            f"R = [r]B + T with T of order {order} under an honest key, S = r + k*a: "
            f"[S]B - [k]A = [r]B, not R.",
        )
    # A mixed-order key A = [a]B + T_A with T_A of order 8, and R = [r]B + T_R ground so that
    # T_R = -[k]T_A: then [S]B - [k]A = [r]B - [k]T_A = R, and strict verification accepts.
    seed = 2
    a = SCALARS[seed]
    t_a = T_OF_ORDER[8]
    a_bytes = encode(add(mul(a, B), t_a))
    for order in (2, 4, 8):
        found = None
        for counter in range(256):
            r = scalar(f"mixed-order-r-accept/{order}", counter)
            r_base = mul(r, B)
            for t_r in TORSION:
                if torsion_order(t_r) != order:
                    continue
                r_point = add(r_base, t_r)
                r_bytes = encode(r_point)
                k = challenge(r_bytes, a_bytes, message)
                if is_identity(add(mul(k, t_a), t_r)):
                    found = (r, r_bytes, k)
                    break
            if found:
                break
        assert found, f"no mixed-order R of torsion order {order} within the bounded search"
        r, r_bytes, k = found
        case(
            f"mixed-order-r-order{order}-mixed-key-accept",
            "mixed-order-r",
            a_bytes,
            message,
            r_bytes + le32((r + k * a) % L),
            "accept",
            True,
            f"R = [r]B + T_R with T_R of order {order} and A = [a]B + T_A with T_A of order 8, "
            f"ground so that T_R = -[k]T_A: the cofactorless equation holds.",
        )


def wrong_r_group(signed):
    seed, message, signature = signed[65]
    r = nonce(SEEDS[seed], message)
    assert encode(mul(r, B)) == signature[:32]
    other = encode(mul(r + 1, B))
    case(
        "wrong-r-next-point",
        "wrong-r",
        KEYS[seed],
        message,
        other + signature[32:],
        "equation",
        False,
        "The honest 65-byte signature with R replaced by the encoding of [r+1]B.",
    )
    flipped = bytearray(signature[:32])
    flipped[31] ^= 0x80
    assert classify(bytes(flipped)) == "large"
    case(
        "wrong-r-negated",
        "wrong-r",
        KEYS[seed],
        message,
        bytes(flipped) + signature[32:],
        "equation",
        False,
        "The honest 65-byte signature with R's sign bit flipped, the encoding of -[r]B.",
    )


def swap_group(signed):
    seed, message, signature = signed[128]
    other = (seed + 1) % 4
    case(
        f"swap-seed{seed}-signature-under-seed{other}-key",
        "swap",
        KEYS[other],
        message,
        signature,
        "equation",
        False,
        f"The honest 128-byte signature of seed {seed} verified under seed {other}'s key.",
    )


def field_carry_group():
    """R whose y sends d * y^2 through the rarest carry of the usual reduction of a product.

    Elements of four 64-bit limbs multiply into eight, T = hi * 2^256 + lo, which reduces through
    2^256 = 2p + 38: first W = lo + 38 * hi = r4 * 2^256 + r with r4 < 39, then r + 38 * r4, which
    carries out of 2^256 only when r >= 2^256 - 38 * r4, for fewer than one random product in
    2^247. Decoding R multiplies d by y^2 mod p. Here W is chosen just below a multiple of 2^256,
    hi is solved for so that d divides T = hi * (2^256 - 38) + W and hi is T's top half, and
    y^2 = T / d must be a square whose y decodes to a point of large order. A reduction that
    dropped this carry would compute d * y^2 - 38 instead; the draw is kept only if that value
    leaves (y^2 - 1) / (d * y^2 + 1) without a square root, so such a reduction refuses R as not a
    point where strict verification reaches the equation.
    """
    two_p = 2**256 - 38
    seed = 2
    a = SCALARS[seed]
    message = b"tessera ed25519 strict vectors: field carry"
    for counter in range(1024):
        draw = le(prf("field-carry", counter))
        r4 = 1 + draw % 5
        w = (r4 + 1) * 2**256 - 1 - (draw >> 8) % (38 * r4)
        # T = hi * 2^256 + lo = hi * two_p + W, with 0 <= W - 38 * hi = lo < 2^256.
        hi = -w * pow(two_p, -1, D) % D
        lowest = -(-(w - 2**256 + 1) // 38)
        hi += -(-(lowest - hi) // D) * D
        if hi > w // 38:
            continue
        t = hi * two_p + w
        yy = t // D
        if yy >= P:
            continue
        y = sqrt_ratio(yy, 1)
        if y is None or classify(le32(y)) != "large":
            continue
        dropped_v = (t - 38 + 1) % P
        if dropped_v != 0 and pow((yy - 1) * inv(dropped_v) % P, (P - 1) // 2, P) == P - 1:
            break
    else:
        raise AssertionError("no y within the bounded search")
    lo = t % 2**256
    assert D * (y * y % P) == t and t >> 256 == hi and lo + 38 * hi == w
    assert (w % 2**256) + 38 * (w >> 256) >= 2**256, "the second fold carries"
    data = le32(y)
    k = challenge(data, KEYS[seed], message)
    case(
        "field-carry-r-second-fold",
        "field-carry",
        KEYS[seed],
        message,
        data + le32(k * a % L),
        "equation",
        False,
        f"R's y makes d*y^2 = hi*2^256 + lo fold to lo + 38*hi = {w >> 256}*2^256 + r with "
        f"r + 38*{w >> 256} >= 2^256, so the fold of the fifth limb carries again (draw {counter}), "
        "and without that carry R would not decode; S = k*a makes [S]B - [k]A the identity.",
    )


def main():
    check_rfc8032()
    signed = honest()
    message_group(signed)
    scalar_group(signed)
    r_encoding_group()
    key_encoding_group()
    small_order_r_group()
    small_order_key_group()
    mixed_order_key_group()
    mixed_order_r_group()
    wrong_r_group(signed)
    swap_group(signed)
    field_carry_group()
    assert {c["expected"] for c in CASES} == set(OUTCOMES), "every outcome has a case"
    doc = {
        "version": 1,
        "description": (
            "Strict Ed25519 verification, RFC 8032 under RFC-0060 'Signatures': each case's "
            "expected outcome is the first rule of check_order it fails, or accept. cofactored "
            "says whether [8][S]B = [8]R + [8][k]A holds with S < L and both points decoding."
        ),
        "check_order": CHECK_ORDER,
        "cases": CASES,
    }
    out = Path(__file__).with_name("strict.json")
    out.write_text(json.dumps(doc, indent=1) + "\n")


if __name__ == "__main__":
    main()
