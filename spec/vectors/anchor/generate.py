# SPDX-License-Identifier: Apache-2.0 OR MIT
"""RFC-0015 'Frames' with RFC-0059's header and counter: standard-library Python, no production
imports.

AES-128 is the FIPS-197 oracle of spec/vectors/scoring/, itself checked against FIPS-197's
examples; CTR mode is checked here against NIST SP 800-38A F.5.1; SHA-256 is hashlib's. A
session's 48 drawn bytes, the ones the reference cipher draws from the operating system at open,
are its identifier, then k_s and salt_k: here a fixed identifier and a SplitMix64 stream written
out little-endian. The cipher's tests build the session from them; the keys are only test
data.
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


scoring = load_oracle("scoring")
scoring.check_cipher()
aes128 = scoring.aes128


def splitmix_bytes(seed, n):
    """`n` bytes of SplitMix64 from `seed`, each output little-endian."""
    out, state, mask = b"", seed, (1 << 64) - 1
    while len(out) < n:
        state = (state + 0x9E3779B97F4A7C15) & mask
        z = state
        z = ((z ^ (z >> 30)) * 0xBF58476D1CE4E5B9) & mask
        z = ((z ^ (z >> 27)) * 0x94D049BB133111EB) & mask
        out += (z ^ (z >> 31)).to_bytes(8, "little")
    return out[:n]


def ctr(key, initial, data):
    """SP 800-38A CTR from `initial`, incrementing the block's last 32 bits big-endian."""
    out, counter = bytearray(), bytearray(initial)
    for at in range(0, len(data), 16):
        pad = aes128(key, bytes(counter))
        out += bytes(a ^ b for a, b in zip(data[at:at + 16], pad))
        low = (int.from_bytes(counter[12:], "big") + 1) % 2**32
        counter[12:] = low.to_bytes(4, "big")
    return bytes(out)


def check_ctr():
    """NIST SP 800-38A F.5.1, CTR-AES128.Encrypt."""
    key = bytes.fromhex("2b7e151628aed2a6abf7158809cf4f3c")
    initial = bytes.fromhex("f0f1f2f3f4f5f6f7f8f9fafbfcfdfeff")
    plain = bytes.fromhex(
        "6bc1bee22e409f96e93d7e117393172aae2d8a571e03ac9c9eb76fac45af8e51"
        "30c81c46a35ce411e5fbc1191a0a52eff69f2445df4f9b17ad2b417be66c3710")
    cipher = bytes.fromhex(
        "874d6191b620e3261bef6864990db6ce9806f66b7970fdff8617187bb9fffdff"
        "5ae4df3edbd5d35e5b4f09020db03eab1e031dda2fbe03d1792170a0f3009cee")
    assert ctr(key, initial, plain) == cipher


def le(value, n):
    return value.to_bytes(n, "little")


def plaintext(request_id, direction, frame_index, tokens):
    """TCE(TokenFrame): u64, u16 tag, u32, u64 count, u32 tokens (RFC-0059 'The plaintext')."""
    out = le(request_id, 8) + le(direction, 2) + le(frame_index, 4) + le(len(tokens), 8)
    return out + b"".join(le(t, 4) for t in tokens)


def header(session_id, request_id, direction, frame_index, length, frame_seq):
    """The 41-byte header (RFC-0059 'The header')."""
    return (b"TSFR" + session_id + le(request_id, 8) + bytes([direction]) + le(frame_index, 4)
            + le(length, 4) + le(frame_seq, 4))


def commitment(label, key, salt):
    return hashlib.sha256(label + key + salt).digest()


def entry_open(seq, t_ns, session_id, kc):
    return le(0, 2) + le(seq, 8) + le(t_ns, 8) + session_id + kc


def entry_frame(seq, t_ns, session_id, request_id, direction, frame_index, length, frame_seq,
                digest):
    """DigestLogEntry::Frame: tag 1, then the fields in the header's order (RFC-0059)."""
    return (le(1, 2) + le(seq, 8) + le(t_ns, 8) + session_id + le(request_id, 8)
            + le(direction, 2) + le(frame_index, 4) + le(length, 4) + le(frame_seq, 4) + digest)


def entry_close(seq, t_ns, session_id):
    return le(2, 2) + le(seq, 8) + le(t_ns, 8) + session_id


def entry_request_end(seq, t_ns, session_id, request_id):
    """RFC-0063's RequestEnd, tag 4: the request-end frame's fields."""
    return le(4, 2) + le(seq, 8) + le(t_ns, 8) + session_id + le(request_id, 8)


def request_end_frame(session_id, request_id):
    """RFC-0063's cleartext request-end frame: TRQE ‖ session_id ‖ u64_le(request_id)."""
    return b"TRQE" + session_id + le(request_id, 8)


def entry_carry(seq, t_ns, session_id, kc):
    """RFC-0062's SessionCarry, tag 3: the session's key commitment, carried into a period."""
    return le(3, 2) + le(seq, 8) + le(t_ns, 8) + session_id + kc


SESSIONS = [
    # (seed, session_id, first counter, frames as (request, direction, index, tokens))
    (1, bytes(range(16)), 0, [
        (7, 0, 0, [1, 2, 3]),
        (7, 1, 0, [42]),
        (7, 1, 1, []),
        (8, 0, 0, [0, 2**32 - 1, 128_255, 65_536]),
        (7, 1, 2, list(range(40))),
        (8, 1, 0, [9] * 3),
    ]),
    (2**64 - 1, bytes([0xFF] * 16), 0, [
        (0, 0, 0, [5]),
        (0, 1, 0, [6, 7]),
        (2**64 - 1, 0, 0, [2**31]),
        (2**64 - 1, 1, 0, list(range(1000, 1033))),
    ]),
    (0x7E55E2A0137, bytes.fromhex("0123456789abcdeffedcba9876543210"), 0, [
        (3, 1, 0, [11, 12, 13, 14, 15]),
    ]),
    # Counters past 2^16 and 2^24, in every byte of the block's counter word, and the last one.
    (5, bytes([0x42] * 16), 0x01020304, [
        (1, 0, 0, list(range(10))),
        (1, 1, 0, [77]),
        (1, 1, 1, [78, 79]),
    ]),
    (6, bytes([0x99] * 8 + [0x11] * 8), 2**32 - 1, [
        (4, 1, 0, [2**32 - 1] * 6),
    ]),
]


def main():
    check_ctr()
    sessions, seq, t_ns = [], 0, 1_000_000
    for seed, session_id, first, frames in SESSIONS:
        drawn = session_id + splitmix_bytes(seed, 32)
        key, salt_k = drawn[16:32], drawn[32:48]
        kc = commitment(b"tessera-anchor-key/v1", key, salt_k)
        record = {
            "drawn": drawn.hex(), "first_seq": first,
            "session_id": session_id.hex(),
            "key": key.hex(), "key_salt": salt_k.hex(), "key_commitment": kc.hex(),
            "session_open": (b"TSOP" + session_id + kc).hex(),
            "open_entry": {"seq": seq, "t_ns": t_ns,
                           "tce": entry_open(seq, t_ns, session_id, kc).hex()},
            "frames": [],
        }
        seq, t_ns = seq + 1, t_ns + 17
        for frame_seq, (request_id, direction, frame_index, tokens) in enumerate(frames, first):
            plain = plaintext(request_id, direction, frame_index, tokens)
            initial = session_id[:8] + frame_seq.to_bytes(4, "big") + bytes(4)
            cipher = ctr(key, initial, plain)
            head = header(session_id, request_id, direction, frame_index, len(cipher), frame_seq)
            digest = hashlib.sha256(head + cipher).digest()
            record["frames"].append({
                "request_id": request_id, "direction": direction, "frame_index": frame_index,
                "tokens": tokens, "frame_seq": frame_seq, "header": head.hex(),
                "plaintext": plain.hex(), "ciphertext": cipher.hex(), "sha256": digest.hex(),
                "entry": {"seq": seq, "t_ns": t_ns,
                          "tce": entry_frame(seq, t_ns, session_id, request_id, direction,
                                             frame_index, len(cipher), frame_seq, digest).hex()},
            })
            seq, t_ns = seq + 1, t_ns + 17
        # Each request's end, after the session's frames, in the order the requests first appear.
        record["request_ends"] = []
        for request_id in dict.fromkeys(f[0] for f in frames):
            record["request_ends"].append({
                "request_id": request_id,
                "frame": request_end_frame(session_id, request_id).hex(),
                "entry": {"seq": seq, "t_ns": t_ns,
                          "tce": entry_request_end(seq, t_ns, session_id, request_id).hex()},
            })
            seq, t_ns = seq + 1, t_ns + 17
        # The carry a recorder would write had the session stayed open into another period.
        record["carry_entry"] = {"seq": seq + 1000, "t_ns": t_ns + 1000,
                                 "tce": entry_carry(seq + 1000, t_ns + 1000, session_id,
                                                    kc).hex()}
        record["close_entry"] = {"seq": seq, "t_ns": t_ns,
                                 "tce": entry_close(seq, t_ns, session_id).hex()}
        seq, t_ns = seq + 1, t_ns + 17
        sessions.append(record)
    document = {"version": 1, "sessions": sessions}
    (HERE / "frames.json").write_text(json.dumps(document, indent=1) + "\n")


if __name__ == "__main__":
    main()
