# SPDX-License-Identifier: Apache-2.0 OR MIT
"""RFC-0017 'Sampler PRF', as RFC-0061 amends it, and RFC-0009 Lemma 10: standard-library Python,
no production imports.

The stream class's hash is hashlib's SHA-256. AES-128 is written from FIPS-197 alone: the S-box is computed from inverses in GF(2^8) and the
affine map of §5.1.1, not copied from a table, and the cipher is checked against the standard's
own examples (Appendix B and C.1) before any vector is written. The inverse-CDF sample is found
by bisection over the prefix sums, where the reference sampler scans them.
"""
from pathlib import Path
import hashlib
import sys

sys.dont_write_bytecode = True
HERE = Path(__file__).resolve().parent
SEED = 0x7E55E2A0136


def splitmix64(state):
    """Yields 64-bit outputs of SplitMix64 (Steele, Lea, Flood 2014) from `state`."""
    mask = (1 << 64) - 1
    while True:
        state = (state + 0x9E3779B97F4A7C15) & mask
        z = state
        z = ((z ^ (z >> 30)) * 0xBF58476D1CE4E5B9) & mask
        z = ((z ^ (z >> 27)) * 0x94D049BB133111EB) & mask
        yield z ^ (z >> 31)


def gf_mul(a, b):
    """The product in GF(2^8) modulo x^8 + x^4 + x^3 + x + 1 (FIPS-197 §4.2)."""
    p = 0
    while b:
        if b & 1:
            p ^= a
        a = ((a << 1) ^ 0x1B) & 0xFF if a & 0x80 else a << 1
        b >>= 1
    return p


def sbox():
    """FIPS-197 §5.1.1: the multiplicative inverse (0 for 0), then the affine map with 0x63."""
    out = []
    for x in range(256):
        inv = next((y for y in range(1, 256) if gf_mul(x, y) == 1), 0)
        s = 0x63
        for i in range(8):
            bit = 0
            for j in (0, 4, 5, 6, 7):
                bit ^= (inv >> ((i + j) % 8)) & 1
            s ^= bit << i
        out.append(s)
    return out


SBOX = sbox()


def expand_key(key):
    """FIPS-197 §5.2 for Nk = 4: 44 words, as 11 round keys of 16 bytes."""
    words = [list(key[4 * i:4 * i + 4]) for i in range(4)]
    rcon = 1
    for i in range(4, 44):
        w = list(words[i - 1])
        if i % 4 == 0:
            w = [SBOX[b] for b in w[1:] + w[:1]]
            w[0] ^= rcon
            rcon = gf_mul(rcon, 2)
        words.append([a ^ b for a, b in zip(words[i - 4], w)])
    return [sum(words[4 * r:4 * r + 4], []) for r in range(11)]


def aes128(key, block):
    """FIPS-197 §5.1 Cipher(); the state is column-major, byte `r + 4c` at row r, column c."""
    keys = expand_key(key)
    s = [a ^ b for a, b in zip(block, keys[0])]
    for rnd in range(1, 11):
        s = [SBOX[b] for b in s]
        s = [s[(r + 4 * (c + r)) % 16] for c in range(4) for r in range(4)]
        if rnd < 10:
            mixed = []
            for c in range(4):
                col = s[4 * c:4 * c + 4]
                for r in range(4):
                    mixed.append(gf_mul(col[r], 2) ^ gf_mul(col[(r + 1) % 4], 3)
                                 ^ col[(r + 2) % 4] ^ col[(r + 3) % 4])
            s = mixed
        s = [a ^ b for a, b in zip(s, keys[rnd])]
    return bytes(s)


def check_cipher():
    """FIPS-197 Appendix B and Appendix C.1, and the S-box's two anchors in §5.1.1."""
    assert SBOX[0x00] == 0x63 and SBOX[0x53] == 0xED
    examples = [
        ("2b7e151628aed2a6abf7158809cf4f3c", "3243f6a8885a308d313198a2e0370734",
         "3925841d02dc09fbdc118597196a0b32"),
        ("000102030405060708090a0b0c0d0e0f", "00112233445566778899aabbccddeeff",
         "69c4e0d86a7b0430d8cdb78070b4c55a"),
    ]
    for key, plain, cipher in examples:
        assert aes128(bytes.fromhex(key), bytes.fromhex(plain)).hex() == cipher, key


def derive(base, nonce, bits, from_ns):
    """RFC-0061: a pin's PRF key, the first 16 bytes of
    SHA-256("tessera-sampler-derive/v1" ‖ K_p ‖ 0^23 ‖ nonce ‖ u8(b) ‖ u64_le(from_ns)); the label
    and the base key fill the first block alone."""
    key_block = b"tessera-sampler-derive/v1" + base
    message = key_block + bytes(64 - len(key_block)) + nonce + bytes([bits])
    message += from_ns.to_bytes(8, "little")
    return hashlib.sha256(message).digest()[:16]


# The prompt of the derivation rows' end-to-end columns: K_p, nonce, b, from_ns → K → class → u_0.
CHAIN_PROMPT = [1, 2, 3]


def derive_cases(rng):
    bases = [bytes(16), bytes(range(16)), bytes([0xFF] * 16)]
    nonces = [bytes(32), bytes(range(32)), bytes([0xFF] * 32)]
    cases = [(base, nonce, bits, 0) for base in bases for nonce in nonces for bits in (0, 8, 64)]
    # The start alone changes the key: its low byte, its high word and its largest value.
    cases += [(bases[1], nonces[1], 8, f) for f in (1, 2**32, 2**64 - 1)]
    for _ in range(16):
        base = b"".join(next(rng).to_bytes(8, "little") for _ in range(2))
        nonce = b"".join(next(rng).to_bytes(8, "little") for _ in range(4))
        cases.append((base, nonce, next(rng) % 65, next(rng)))
    return cases


def stream_class(key, bits, prompt):
    """RFC-0061: the top `bits` bits of u64_le(h[0..8]), 0 for bits = 0, with
    h = SHA-256("tessera-stream-class/v1" ‖ K ‖ 0^25 ‖ u64_le(n) ‖ u32_le(x_0) ‖ … ‖ u32_le(x_{n−1})):
    the label and the key fill the first block with zeros, then come the count and the tokens."""
    if bits == 0:
        return 0
    key_block = b"tessera-stream-class/v1" + key
    message = key_block + bytes(64 - len(key_block)) + len(prompt).to_bytes(8, "little")
    message += b"".join(x.to_bytes(4, "little") for x in prompt)
    head = int.from_bytes(hashlib.sha256(message).digest()[:8], "little")
    return head >> (64 - bits)


def draw(key, cls, t):
    """RFC-0061: u32_le(AES-128_K(u64_le(class) ‖ u32_le(t) ‖ 0^4)[0..4]), RFC-0017's block with
    the class in the slot of request_id."""
    cipher = aes128(key, cls.to_bytes(8, "little") + t.to_bytes(4, "little") + bytes(4))
    return cipher, int.from_bytes(cipher[:4], "little")


def sample(weights, u):
    """RFC-0009 Lemma 10: min{v : C(v)·2^32 > u·Z}, by bisection; None when Z = 0."""
    prefix, total = [], 0
    for w in weights:
        total += w
        prefix.append(total)
    if total == 0:
        return None
    lo, hi = 0, len(prefix) - 1
    while lo < hi:
        mid = (lo + hi) // 2
        if prefix[mid] * 2**32 > u * total:
            hi = mid
        else:
            lo = mid + 1
    # The bracket of RFC-0009 'InverseCdf': C_< · 2^32 ≤ u·Z < C_≤ · 2^32.
    below = prefix[lo - 1] if lo else 0
    assert below * 2**32 <= u * total < prefix[lo] * 2**32
    return lo


def prf_cases(rng):
    u32, vocab = 2**32 - 1, 2**18
    keys = [bytes(16), bytes(range(16)), bytes([0xFF] * 16),
            bytes.fromhex("2b7e151628aed2a6abf7158809cf4f3c")]
    cases = []
    # b = 0: one stream per key, at the edges of each byte of u32_le(t). These are the rows of
    # RFC-0017's block at request_id = 0.
    for key in keys:
        for t in (0, 1, 255, 256, 2**16, 2**24, u32):
            cases.append((key, 0, [], t))
    # Classes: the empty prompt, prompts that differ only in length ([0], [0, 0]) or in one
    # token ([1, 2, 3], [1, 2, 4]), and the largest v1 token, at b = 0, 1, 8, 63 and 64.
    prompts = [[], [0], [0, 0], [1, 2, 3], [1, 2, 4], [vocab - 1]]
    for key in keys:
        for bits in (0, 1, 8, 63, 64):
            for prompt in prompts:
                for t in (0, u32):
                    cases.append((key, bits, prompt, t))
    # SHA-256's padding after the key block: 8 + 4n bytes fill one more block up to n = 11 and
    # spill the length field at n = 12; two more up to n = 27, three at n = 28; then a long prompt.
    for n in (11, 12, 27, 28, 4096):
        cases.append((keys[3], 64, [next(rng) % vocab for _ in range(n)], 0))
    for _ in range(32):
        key = b"".join(next(rng).to_bytes(8, "little") for _ in range(2))
        bits = next(rng) % 65
        prompt = [next(rng) % vocab for _ in range(next(rng) % 40)]
        cases.append((key, bits, prompt, next(rng) & u32))
    # One request's stream at b = 8: positions 0 to 63 under one random key.
    key = b"".join(next(rng).to_bytes(8, "little") for _ in range(2))
    prompt = [next(rng) % vocab for _ in range(16)]
    cases += [(key, 8, prompt, t) for t in range(64)]
    return cases


def cdf_cases(rng):
    u32, u64 = 2**32 - 1, 2**64 - 1
    cases = []
    for weights in ([1], [1, 0, 3], [0, 5], [5, 0], [0, 0, 7, 0], [1, 1], [3, 3, 3],
                    [u64], [u64, u64], [1, u64], [u64, 1], [2**40] * 5):
        total = sum(weights)
        us = {0, 1, u32 - 1, u32}
        # Each prefix sum's boundary: the least u past it, and the u before it.
        prefix = 0
        for w in weights:
            prefix += w
            edge = -(-prefix * 2**32 // total)
            us.update(u for u in (edge - 1, edge) if 0 <= u <= u32)
        cases += [(weights, u) for u in sorted(us)]
    cases += [([0], 0), ([0, 0, 0], u32)]
    for _ in range(160):
        size = 1 + next(rng) % 300
        bits = 1 + next(rng) % 64
        zeros = next(rng) % 4
        weights = []
        for _ in range(size):
            w = next(rng) % 2**bits
            # A quarter, half or three quarters of the weights are zero, as in a peaked softmax.
            weights.append(0 if next(rng) % 4 < zeros else w)
        cases.append((weights, next(rng) & u32))
    return cases


def main():
    check_cipher()
    # One stream per file, so that a change to one file's cases leaves the other's rows alone.
    rng = splitmix64(SEED)
    lines = ["# key bits prompt t class ciphertext u (prompt: comma-separated tokens, - when empty)"]
    for key, bits, prompt, t in prf_cases(rng):
        cls = stream_class(key, bits, prompt)
        cipher, u = draw(key, cls, t)
        tokens = ",".join(str(x) for x in prompt) or "-"
        lines.append(f"{key.hex()} {bits} {tokens} {t} {cls} {cipher.hex()} {u}")
    (HERE / "prf.txt").write_text("\n".join(lines) + "\n")
    lines = ["# u token weights... (token none: every weight is zero)"]
    for weights, u in cdf_cases(splitmix64(SEED + 1)):
        token = sample(weights, u)
        words = [str(u), "none" if token is None else str(token)] + [str(w) for w in weights]
        lines.append(" ".join(words))
    (HERE / "inverse_cdf.txt").write_text("\n".join(lines) + "\n")
    lines = ["# base_key nonce bits from_ns key class u_0 (class and u_0 of the prompt 1,2,3 at t = 0)"]
    for base, nonce, bits, from_ns in derive_cases(splitmix64(SEED + 2)):
        key = derive(base, nonce, bits, from_ns)
        cls = stream_class(key, bits, CHAIN_PROMPT)
        _, u = draw(key, cls, 0)
        lines.append(f"{base.hex()} {nonce.hex()} {bits} {from_ns} {key.hex()} {cls} {u}")
    (HERE / "derive.txt").write_text("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
