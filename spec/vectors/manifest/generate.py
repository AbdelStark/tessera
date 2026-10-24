# SPDX-License-Identifier: Apache-2.0 OR MIT
"""RFC-0015 'Log format' and 'Signatures' with RFC-0060: standard-library Python, no production
imports.

Ed25519 is RFC 8032's reference arithmetic (Section 5.1), checked against its test vectors 1 to 3
before anything is written; SHA-256 and SHA-512 are hashlib's; the envelope digest that is a
manifest's identity is the BLAKE3 of spec/vectors/hash/; the entries are encoded by
spec/vectors/anchor/'s oracle. The Merkle tree is RFC 9162 §2.1, written here.
"""
import hashlib
import importlib.util
import json
import sys
from pathlib import Path

sys.dont_write_bytecode = True
HERE = Path(__file__).resolve().parent


def load_oracle(name):
    spec = importlib.util.spec_from_file_location(name, HERE.parent / name / "generate.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


hash_oracle = load_oracle("hash")
anchor = load_oracle("anchor")

# ----- Ed25519, RFC 8032 Section 5.1 --------------------------------------------------------------

P = 2**255 - 19
L = 2**252 + 27742317777372353535851937790883648493
D = -121665 * pow(121666, P - 2, P) % P
SQRT_M1 = pow(2, (P - 1) // 4, P)


def point_add(a, b):
    x1, y1, z1, t1 = a
    x2, y2, z2, t2 = b
    A = (y1 - x1) * (y2 - x2) % P
    B = (y1 + x1) * (y2 + x2) % P
    C = 2 * t1 * t2 * D % P
    Dd = 2 * z1 * z2 % P
    E, F, G, H = B - A, Dd - C, Dd + C, B + A
    return (E * F % P, G * H % P, F * G % P, E * H % P)


def point_mul(s, point):
    q = (0, 1, 1, 0)
    while s > 0:
        if s & 1:
            q = point_add(q, point)
        point = point_add(point, point)
        s >>= 1
    return q


def recover_x(y, sign):
    if y >= P:
        return None
    x2 = (y * y - 1) * pow(D * y * y + 1, P - 2, P)
    if x2 == 0:
        return None if sign else 0
    x = pow(x2, (P + 3) // 8, P)
    if (x * x - x2) % P != 0:
        x = x * SQRT_M1 % P
    if (x * x - x2) % P != 0:
        return None
    if (x & 1) != sign:
        x = P - x
    return x


G_Y = 4 * pow(5, P - 2, P) % P
G_X = recover_x(G_Y, 0)
G = (G_X, G_Y, 1, G_X * G_Y % P)


def compress(point):
    x, y, z, _ = point
    zinv = pow(z, P - 2, P)
    x, y = x * zinv % P, y * zinv % P
    return int.to_bytes(y | ((x & 1) << 255), 32, "little")


def sha512_int(data):
    return int.from_bytes(hashlib.sha512(data).digest(), "little")


def secret_expand(seed):
    h = hashlib.sha512(seed).digest()
    a = int.from_bytes(h[:32], "little")
    a &= (1 << 254) - 8
    a |= 1 << 254
    return a, h[32:]


def public_key(seed):
    a, _ = secret_expand(seed)
    return compress(point_mul(a, G))


def sign(seed, message):
    a, prefix = secret_expand(seed)
    A = compress(point_mul(a, G))
    r = sha512_int(prefix + message) % L
    R = compress(point_mul(r, G))
    h = sha512_int(R + A + message) % L
    s = (r + h * a) % L
    return R + int.to_bytes(s, 32, "little")


def decompress(encoding):
    y = int.from_bytes(encoding, "little")
    sign, y = y >> 255, y & ((1 << 255) - 1)
    x = recover_x(y, sign)
    assert x is not None
    return (x, y, 1, x * y % P)


def point_equal(a, b):
    x1, y1, z1, _ = a
    x2, y2, z2, _ = b
    return (x1 * z2 - x2 * z1) % P == 0 and (y1 * z2 - y2 * z1) % P == 0


def lax_verifies(public, message, signature):
    """The cofactorless equation [S]B = R + [k]A alone, which strict verification adds rules to."""
    R, S = decompress(signature[:32]), int.from_bytes(signature[32:], "little")
    A = decompress(public)
    k = sha512_int(signature[:32] + public + message) % L
    return S < L and point_equal(point_mul(S, G), point_add(R, point_mul(k, A)))


IDENTITY = bytes([1]) + bytes(31)  # the neutral point, of order 1


def strictness(seed, message):
    """Two signatures the cofactorless equation accepts and strict verification refuses, each
    by one rule alone: R of small order under the honest key, and a small-order key."""
    a, _ = secret_expand(seed)
    A = public_key(seed)
    k = sha512_int(IDENTITY + A + message) % L
    r_rule = IDENTITY + int.to_bytes(k * a % L, 32, "little")
    key_rule = compress(G) + int.to_bytes(1, 32, "little")
    assert lax_verifies(A, message, r_rule)
    assert lax_verifies(IDENTITY, message, key_rule)
    return {"r_rule": r_rule.hex(), "key_rule_public_key": IDENTITY.hex(),
            "key_rule": key_rule.hex()}


def check_ed25519():
    """RFC 8032 Section 7.1, tests 1 to 3."""
    for seed, pub, msg, sig in [
        ("9d61b19deffd5a60ba844af492ec2cc44449c5697b326919703bac031cae7f60",
         "d75a980182b10ab7d54bfed3c964073a0ee172f3daa62325af021a68f707511a", "",
         "e5564300c360ac729086e2cc806e828a84877f1eb8e5d974d873e06522490155"
         "5fb8821590a33bacc61e39701cf9b46bd25bf5f0595bbe24655141438e7a100b"),
        ("4ccd089b28ff96da9db6c346ec114e0f5b8a319f35aba624da8cf6ed4fb8a6fb",
         "3d4017c3e843895a92b70aa74d1b7ebc9c982ccf2ec4968cc0cd55f12af4660c", "72",
         "92a009a9f0d4cab8720e820b5f642540a2b27b5416503f8fb3762223ebdb69da"
         "085ac1e43e15996e458f3613d0f11d8c387b2eaeb4302aeeb00d291612bb0c00"),
        ("c5aa8df43f9f837bedb7442f31dcb7b166d38535076f094b85ce3a2e0b4458f7",
         "fc51cd8e6218a1a38da47ed00230f0580816ed13ba3303ac5deb911548908025", "af82",
         "6291d657deec24024827e69c3abe01a30ce548a284743a445e3680d7db5ac3ac"
         "18ff9b538d16f290ae67f760984dc6594a7c15e9716ed28dc027beceea1ec40a"),
    ]:
        seed, msg = bytes.fromhex(seed), bytes.fromhex(msg)
        assert public_key(seed).hex() == pub
        assert sign(seed, msg).hex() == sig


# ----- RFC 9162 §2.1 -----------------------------------------------------------------------------

def leaf(entry_tce):
    return hashlib.sha256(b"\x00" + entry_tce).digest()


def node(left, right):
    return hashlib.sha256(b"\x01" + left + right).digest()


def mth(leaves):
    if not leaves:
        return hashlib.sha256(b"").digest()
    if len(leaves) == 1:
        return leaves[0]
    k = 1
    while k * 2 < len(leaves):
        k *= 2
    return node(mth(leaves[:k]), mth(leaves[k:]))


def path(m, leaves):
    if len(leaves) <= 1:
        return []
    k = 1
    while k * 2 < len(leaves):
        k *= 2
    if m < k:
        return path(m, leaves[:k]) + [mth(leaves[k:])]
    return path(m - k, leaves[k:]) + [mth(leaves[:k])]


# ----- Manifests ---------------------------------------------------------------------------------

def le(value, n):
    return value.to_bytes(n, "little")


def body_tce(key_id, log_id, start, end, first_seq, count, root, prev):
    return key_id + log_id + le(start, 8) + le(end, 8) + le(first_seq, 8) + le(count, 8) + root + prev


def envelope(payload):
    """RFC-0003's envelope of schema 0x0402, version 1, and its digest."""
    head = b"TSRA" + le(1, 2) + le(0x0402, 2) + le(1, 2) + le(0, 2) + le(len(payload), 8)
    digest = hash_oracle.H("envelope", head + payload)
    return head + payload + digest, digest


def entries_of(period, first_seq, start):
    """A period's entries, numbered from `first_seq` and timed inside the period that starts at
    `start`: TCE encodings from the anchor oracle. Each session opens before its frames."""
    sid = bytes([0x30 + period] * 16)
    out, seq = [], first_seq
    kc = bytes([1 + period] * 32)
    if period == 0:
        out.append(anchor.entry_open(seq, start + 1000, sid, kc))
        seq += 1
        # Request 7: three inbound frames, then three outbound; it ends in the next period.
        for i in range(6):
            digest = hashlib.sha256(bytes([i])).digest()
            direction, index = (0, i) if i < 3 else (1, i - 3)
            out.append(anchor.entry_frame(seq, start + 1001 + i, sid, 7, direction, index,
                                          22 + 4 * i, i, digest))
            seq += 1
    elif period == 1:
        # Period 0's session is still open: the recorder carries it in first (RFC-0062). Its
        # request 7, whose frames all lie in period 0, ends here (RFC-0063), and the session
        # closes in this period.
        carried = bytes([0x30] * 16)
        out.append(anchor.entry_carry(seq, start, carried, bytes([1] * 32)))
        seq += 1
        out.append(anchor.entry_request_end(seq, start + 10, carried, 7))
        seq += 1
        out.append(anchor.entry_open(seq, start + 4999, sid, kc))
        seq += 1
        for i in range(4):
            digest = hashlib.sha256(bytes([0x80 + i])).digest()
            out.append(anchor.entry_frame(seq, start + 5000 + i, sid, 9, 1, i, 26, i, digest))
            seq += 1
        out.append(anchor.entry_request_end(seq, start + 5050, sid, 9))
        seq += 1
        out.append(anchor.entry_close(seq, start + 5100, sid))
        seq += 1
        out.append(anchor.entry_close(seq, start + 5200, carried))
    return out


def main():
    check_ed25519()
    seed = bytes(range(0x40, 0x60))
    pub = public_key(seed)
    key_id = hashlib.sha256(pub).digest()
    log_id = bytes.fromhex("5ca1ab1e000000000000000000000001")
    manifests, prev, first_seq, start = [], bytes(32), 0, 1_000_000_000
    for period in range(3):  # the third period is empty
        entries = entries_of(period, first_seq, start)
        leaves = [leaf(e) for e in entries]
        root = mth(leaves)
        end = start + 60_000_000_000
        body = body_tce(key_id, log_id, start, end, first_seq, len(entries), root, prev)
        message = b"tessera-manifest/v1" + body
        signature = sign(seed, message)
        envelope_tce = le(0, 2) + le(32, 8) + pub + le(64, 8) + signature
        manifest = body + envelope_tce
        sealed, identity = envelope(manifest)
        manifests.append({
            "period_start_ns": start, "period_end_ns": end, "first_seq": first_seq,
            "entries": [e.hex() for e in entries], "leaves": [x.hex() for x in leaves],
            "entries_root": root.hex(), "prev_manifest": prev.hex(),
            "signed_message": message.hex(), "signature": signature.hex(),
            "manifest": manifest.hex(), "envelope": sealed.hex(), "identity": identity.hex(),
            "proofs": [[p.hex() for p in path(i, leaves)] for i in range(len(leaves))],
        })
        prev, first_seq, start = identity, first_seq + len(entries), end
    first_message = bytes.fromhex(manifests[0]["signed_message"])
    document = {
        "version": 1, "seed": seed.hex(), "public_key": pub.hex(), "recorder_key_id": key_id.hex(),
        "log_id": log_id.hex(), "manifests": manifests,
        "strictness": strictness(seed, first_message),
    }
    (HERE / "manifests.json").write_text(json.dumps(document, indent=1) + "\n")


if __name__ == "__main__":
    main()
