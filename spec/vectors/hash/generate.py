#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0 OR MIT
"""Reference vectors for RFC-0004 hashing: the hash domains, Merkle trees with canonical
multiproofs, chained column digests, and RFC-0031's commitment_id of a commitment descriptor.

This script is the independent oracle. It carries its own BLAKE3, written from the BLAKE3
specification and checked against the official test vectors in ../blake3/test_vectors.json
before anything is written. It shares no code with the Rust implementations in
crates/tessera-hash and verifier/crates/tv-hash, which both check its output.

Usage: python3 spec/vectors/hash/generate.py
(rewrites hash.json)

Inputs come from SplitMix64 with a fixed seed, so any implementation can regenerate them.
"""

import hashlib
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
SEED = 0x7E55E7A_4A5E_0004  # fixed; changing it changes every vector
# ../blake3/test_vectors.json, as published at BLAKE3 tag 1.8.7 (see ../blake3/README.md).
OFFICIAL_SHA256 = "dcb91ea8accc77e6d6e632af7cdc1a99a9f3ae78cf648da595c7d064db32f624"
OFFICIAL_CASES = 35
P = 2**64 - 2**32 + 1

# ----- BLAKE3 (the BLAKE3 specification, section 2) -------------------------------------------

IV = (0x6A09E667, 0xBB67AE85, 0x3C6EF372, 0xA54FF53A, 0x510E527F, 0x9B05688C, 0x1F83D9AB, 0x5BE0CD19)
MSG_PERMUTATION = (2, 6, 3, 10, 7, 0, 4, 13, 1, 11, 12, 5, 9, 14, 15, 8)
CHUNK_START, CHUNK_END, PARENT, ROOT = 1, 2, 4, 8
KEYED_HASH, DERIVE_KEY_CONTEXT, DERIVE_KEY_MATERIAL = 16, 32, 64
BLOCK_LEN, CHUNK_LEN, MASK = 64, 1024, 0xFFFFFFFF


def rotr(x, n):
    return ((x >> n) | (x << (32 - n))) & MASK


def g(s, a, b, c, d, mx, my):
    s[a] = (s[a] + s[b] + mx) & MASK
    s[d] = rotr(s[d] ^ s[a], 16)
    s[c] = (s[c] + s[d]) & MASK
    s[b] = rotr(s[b] ^ s[c], 12)
    s[a] = (s[a] + s[b] + my) & MASK
    s[d] = rotr(s[d] ^ s[a], 8)
    s[c] = (s[c] + s[d]) & MASK
    s[b] = rotr(s[b] ^ s[c], 7)


def compress(cv, block, counter, block_len, flags):
    """The 16-word output of the compression function."""
    s = list(cv) + list(IV[:4]) + [counter & MASK, (counter >> 32) & MASK, block_len, flags]
    m = list(block)
    for r in range(7):
        g(s, 0, 4, 8, 12, m[0], m[1])
        g(s, 1, 5, 9, 13, m[2], m[3])
        g(s, 2, 6, 10, 14, m[4], m[5])
        g(s, 3, 7, 11, 15, m[6], m[7])
        g(s, 0, 5, 10, 15, m[8], m[9])
        g(s, 1, 6, 11, 12, m[10], m[11])
        g(s, 2, 7, 8, 13, m[12], m[13])
        g(s, 3, 4, 9, 14, m[14], m[15])
        if r < 6:
            m = [m[i] for i in MSG_PERMUTATION]
    for i in range(8):
        s[i] ^= s[i + 8]
        s[i + 8] ^= cv[i]
    return s


def words(data):
    """The little-endian 32-bit words of at most one block, zero-padded."""
    data = data.ljust(BLOCK_LEN, b"\0")
    return [int.from_bytes(data[i : i + 4], "little") for i in range(0, len(data), 4)]


class Output:
    """A compression waiting to be finished as a chaining value or as root output."""

    def __init__(self, cv, block, counter, block_len, flags):
        self.cv, self.block, self.counter = cv, block, counter
        self.block_len, self.flags = block_len, flags

    def chaining_value(self):
        return compress(self.cv, self.block, self.counter, self.block_len, self.flags)[:8]

    def root_bytes(self, length):
        out = b""
        counter = 0
        while len(out) < length:
            s = compress(self.cv, self.block, counter, self.block_len, self.flags | ROOT)
            out += b"".join(w.to_bytes(4, "little") for w in s)
            counter += 1
        return out[:length]


def chunk_output(key, chunk, chunk_counter, flags):
    """The output of one chunk of at most 1024 bytes."""
    blocks = [chunk[i : i + BLOCK_LEN] for i in range(0, len(chunk), BLOCK_LEN)] or [b""]
    cv = list(key)
    for n, block in enumerate(blocks[:-1]):
        start = CHUNK_START if n == 0 else 0
        cv = compress(cv, words(block), chunk_counter, BLOCK_LEN, flags | start)[:8]
    start = CHUNK_START if len(blocks) == 1 else 0
    last = blocks[-1]
    return Output(cv, words(last), chunk_counter, len(last), flags | start | CHUNK_END)


def blake3(data, key=IV, flags=0, length=32):
    """BLAKE3 of `data` in the mode set by `key` and `flags`, extended to `length` bytes."""
    chunks = [data[i : i + CHUNK_LEN] for i in range(0, len(data), CHUNK_LEN)] or [b""]
    stack = []
    for counter, chunk in enumerate(chunks[:-1]):
        cv = chunk_output(key, chunk, counter, flags).chaining_value()
        total = counter + 1
        while total & 1 == 0:  # merge each completed subtree
            cv = Output(key, stack.pop() + cv, 0, BLOCK_LEN, flags | PARENT).chaining_value()
            total >>= 1
        stack.append(cv)
    output = chunk_output(key, chunks[-1], len(chunks) - 1, flags)
    while stack:
        output = Output(key, stack.pop() + output.chaining_value(), 0, BLOCK_LEN, flags | PARENT)
    return output.root_bytes(length)


def key_words(key_bytes):
    return [int.from_bytes(key_bytes[i : i + 4], "little") for i in range(0, 32, 4)]


def derive_key(context, material, length=32):
    context_key = blake3(context.encode("ascii"), IV, DERIVE_KEY_CONTEXT)
    return blake3(material, key_words(context_key), DERIVE_KEY_MATERIAL, length)


def require(condition, message):
    """A check that holds under `python -O` too, unlike `assert`."""
    if not condition:
        raise SystemExit(f"generate.py: {message}")


def check_official_vectors():
    """Every mode of every official BLAKE3 test vector, at its full extended length, from the
    file whose digest is pinned above."""
    raw = (HERE.parent / "blake3" / "test_vectors.json").read_bytes()
    require(hashlib.sha256(raw).hexdigest() == OFFICIAL_SHA256, "test_vectors.json is not the published file")
    official = json.loads(raw)
    require(len(official["cases"]) == OFFICIAL_CASES, "test_vectors.json lost cases")
    key = key_words(official["key"].encode("ascii"))
    for case in official["cases"]:
        data = bytes(i % 251 for i in range(case["input_len"]))
        length = len(case["hash"]) // 2
        for mode, out in [
            ("hash", blake3(data, IV, 0, length)),
            ("keyed_hash", blake3(data, key, KEYED_HASH, length)),
            ("derive_key", derive_key(official["context_string"], data, length)),
        ]:
            require(out.hex() == case[mode], f"{mode} of input length {case['input_len']} differs")
    return len(official["cases"])


# ----- Tessera (RFC-0004) ---------------------------------------------------------------------

PREFIX = "tessera/v1/"
# RFC-0004 'Hash function and domains', in table order.
DOMAINS = [
    "envelope",
    "commitment-id",
    "column-init",
    "column-step",
    "merkle-node",
    "column-merge",
    "weights-root",
    "transcript-init",
    "transcript-absorb",
    "transcript-squeeze",
    "transcript-ratchet",
    "pow",
    "coin-commit",
    "coin-toss",
    "table",
    "audit-sample",
    "audit-select",
    "claim-list",
    "statement",
]
# RFC-0004 secret-class per-proof domains (RFC-0018, #567): the verifier never computes them,
# so they live under `prover_streams` rather than `domains`.
PROVER_DOMAINS = [
    "advice-masks",
    "prover-share",
    "ni-session",
]


def H(domain, data):
    require(domain in DOMAINS, f"unknown domain {domain}")
    return derive_key(PREFIX + domain, data)


def H_prover(domain, data, length=32):
    require(domain in PROVER_DOMAINS, f"unknown prover domain {domain}")
    return derive_key(PREFIX + domain, data, length)


def stream_words(domain, data, word_count):
    raw = H_prover(domain, data, 8 * word_count)
    return [int.from_bytes(raw[8 * i : 8 * i + 8], "little") for i in range(word_count)]


def splitmix64(state):
    """Yields 64-bit outputs of SplitMix64 (Steele, Lea, Flood 2014) from `state`."""
    mask = (1 << 64) - 1
    while True:
        state = (state + 0x9E3779B97F4A7C15) & mask
        z = state
        z = ((z ^ (z >> 30)) * 0xBF58476D1CE4E5B9) & mask
        z = ((z ^ (z >> 27)) * 0x94D049BB133111EB) & mask
        yield z ^ (z >> 31)


def digest_from(rng):
    """32 bytes: four outputs, little-endian."""
    return b"".join(next(rng).to_bytes(8, "little") for _ in range(4))


def canonical(rng):
    """A uniform value of [0, P), by rejection."""
    while True:
        x = next(rng)
        if x < P:
            return x


def merkle_levels(leaves):
    """Every level of the tree, leaves first: node = H("merkle-node", left ‖ right)."""
    levels = [leaves]
    while len(levels[-1]) > 1:
        below = levels[-1]
        levels.append([H("merkle-node", below[i] + below[i + 1]) for i in range(0, len(below), 2)])
    return levels


def multiproof(levels, indices):
    """The canonical multiproof: the siblings of the known nodes that are not known themselves,
    level by level from the leaves upward, in increasing index order within a level."""
    known = sorted(set(indices))
    siblings = []
    for level in levels[:-1]:
        known_set = set(known)
        siblings += [level[i ^ 1] for i in known if i ^ 1 not in known_set]
        known = sorted({i >> 1 for i in known})
    return siblings


def column_leaves(commitment_id, rows, columns, block):
    """leaf_j for every column of a matrix of canonical Fp values (RFC-0004 'Column digests')."""
    leaves = []
    for j in range(columns):
        d = H("column-init", commitment_id + j.to_bytes(8, "little"))
        for start in range(0, len(rows), block):
            symbols = b"".join(row[j].to_bytes(8, "little") for row in rows[start : start + block])
            d = H("column-step", d + symbols)
        leaves.append(d)
    return leaves


def hex64(x):
    return f"0x{x:016x}"


# ----- commitment_id (RFC-0031) ---------------------------------------------------------------

# RFC-0012 'Layout', in table order: the declaration order of `WeightMatrix`.
WEIGHT_MATRICES = ["Embed", "Q", "K", "V", "O", "Gate", "Up", "Down", "LmHead", "ChannelParams"]


def le(x, width):
    """A fixed-width little-endian unsigned integer (RFC-0003)."""
    require(0 <= x < 2 ** (8 * width), f"{x} fits {width} bytes")
    return x.to_bytes(width, "little")


def tce_descriptor(d):
    """TCE(CommitmentDescriptor) of RFC-0031: a u16 variant tag, then the variant's fields in
    declaration order, with `AdviceLayoutParams` and `WeightLayoutParams` as RFC-0026 lays them
    out and `WeightMatrix` as a u16 tag."""
    layout = d["layout"]
    if d["kind"] == "Advice":
        require(len(d["statement_digest"]) == 64, "a 32-byte statement digest")
        return (
            le(0, 2)
            + bytes.fromhex(d["statement_digest"])
            + le(d["unit"], 4)
            + le(layout["log2_k"], 1)
            + le(layout["log2_inv_rate"], 1)
            + le(layout["mask_coeffs"], 4)
            + le(layout["row_block"], 4)
            + le(layout["segments"], 4)
            + le(d["rows"], 8)
        )
    require(d["kind"] == "WeightMatrix", "a descriptor kind")
    return (
        le(1, 2)
        + le(layout["log2_k"], 1)
        + le(layout["log2_inv_rate"], 1)
        + le(layout["mask_coeffs"], 4)
        + le(layout["blinding_rows"], 4)
        + le(WEIGHT_MATRICES.index(d["matrix"]), 2)
        + le(d["log2_k"], 1)
        + le(d["row_block"], 4)
        + le(d["rows"], 8)
        + le(d["epoch"], 4)
    )


def main():
    checked = check_official_vectors()
    rng = splitmix64(SEED)

    lengths = [0, 1, 64, 1025]
    domains = [
        {
            "name": name,
            "context": PREFIX + name,
            "digests": [H(name, bytes(i % 251 for i in range(n))).hex() for n in lengths],
        }
        for name in DOMAINS
    ]

    queries = {
        1: [[0]],
        2: [[0], [1], [0, 1]],
        8: [[0], [3], [7], [0, 7], [2, 3], [1, 2, 5, 6], list(range(8))],
        16: [[5], [0, 15], [4, 5, 6, 7], [1, 8, 9, 14]],
    }
    trees = []
    for n in (1, 2, 8, 16, 1024):
        leaves = [digest_from(rng) for _ in range(n)]
        levels = merkle_levels(leaves)
        sets = queries.get(n)
        if sets is None:
            sets = [sorted({next(rng) % n for _ in range(size)}) for size in (1, 2, 5, 33, 200)]
        trees.append(
            {
                "leaves": [leaf.hex() for leaf in leaves],
                "root": levels[-1][0].hex(),
                "openings": [
                    {"indices": s, "siblings": [d.hex() for d in multiproof(levels, s)]} for s in sets
                ],
            }
        )

    matrices = []
    for m, columns, block in [(0, 3, 4), (1, 1, 1), (5, 3, 2), (8, 4, 4), (7, 2, 32), (33, 5, 32), (65, 2, 64)]:
        commitment_id = digest_from(rng)
        rows = [[canonical(rng) for _ in range(columns)] for _ in range(m)]
        matrices.append(
            {
                "commitment_id": commitment_id.hex(),
                "columns": columns,
                "block": block,
                "rows": [[hex64(x) for x in row] for row in rows],
                "leaves": [leaf.hex() for leaf in column_leaves(commitment_id, rows, columns, block)],
            }
        )

    # test-small/1's layouts (RFC-0014 'Parameter sets') and production-like ones.
    small_advice = {"log2_k": 10, "log2_inv_rate": 1, "mask_coeffs": 16, "row_block": 4, "segments": 1}
    large_advice = {"log2_k": 22, "log2_inv_rate": 1, "mask_coeffs": 200, "row_block": 32, "segments": 8}
    small_weights = {"log2_k": 8, "log2_inv_rate": 2, "mask_coeffs": 64, "blinding_rows": 32}
    large_weights = {"log2_k": 20, "log2_inv_rate": 2, "mask_coeffs": 65536, "blinding_rows": 4096}
    descriptors = [
        {"kind": "Advice", "statement_digest": digest_from(rng).hex(), "unit": 0,
         "layout": small_advice, "rows": 37},
        {"kind": "Advice", "statement_digest": digest_from(rng).hex(), "unit": 3,
         "layout": large_advice, "rows": 580_000},
        {"kind": "WeightMatrix", "layout": small_weights, "matrix": "Q", "log2_k": 8,
         "row_block": 4, "rows": 4 + 32, "epoch": 0},
        # `servable` with two classes at 405B: O and Down have a ChannelParams, 8 rows each of
        # 16 layers, beside the 4,096 blinding rows (RFC-0012 'Layout').
        {"kind": "WeightMatrix", "layout": large_weights, "matrix": "ChannelParams", "log2_k": 20,
         "row_block": 32, "rows": 2 * 8 + 4096, "epoch": 7},
    ]
    commitment_ids = []
    for d in descriptors:
        tce = tce_descriptor(d)
        commitment_ids.append(
            {"descriptor": d, "tce": tce.hex(), "commitment_id": H("commitment-id", tce).hex()}
        )

    prover_domains = [
        {
            "name": name,
            "context": PREFIX + name,
            "digests": [H_prover(name, bytes(i % 251 for i in range(n))).hex() for n in lengths],
        }
        for name in PROVER_DOMAINS
    ]
    proof_seed = digest_from(rng)
    mask_words = stream_words("advice-masks", proof_seed, 32)
    mask_values = [w for w in mask_words if w < P][:24]
    require(len(mask_values) == 24, "24 canonical mask values from 32 words")
    # Pinned 32-byte seed whose first block in `advice-masks` has word 6 >= P (0xffffffffbc84bb6d).
    reject_seed = (384450091).to_bytes(8, "little") + bytes(24)
    reject_words = stream_words("advice-masks", reject_seed, 16)
    require(reject_words[6] >= P, "pinned advice-masks seed must hit rejection on word 6")
    reject_values = [w for w in reject_words if w < P]
    require(len(reject_values) == 15, "exactly one word rejected in the first two blocks")
    prover_streams = {
        "domains": prover_domains,
        "proof_seed": proof_seed.hex(),
        "prover_share": H_prover("prover-share", proof_seed).hex(),
        "ni_session": H_prover("ni-session", proof_seed)[:16].hex(),
        "advice_masks": [hex64(v) for v in mask_values],
        "rejection": {
            "seed": reject_seed.hex(),
            "rejected_word_index": 6,
            "words": [hex64(w) for w in reject_words],
            "values": [hex64(v) for v in reject_values],
        },
    }

    vectors = {
        "description": (
            "RFC-0004 hashing reference vectors from an independent BLAKE3 (checked against "
            f"{checked} official vectors): H(domain, x) = BLAKE3.derive_key(context, x) for every "
            "domain on inputs of length 0, 1, 64 and 1025 filled with i mod 251; Merkle trees with "
            "canonical multiproofs; chained column digests with row block `block`; commitment_id "
            "from the TCE of a CommitmentDescriptor (RFC-0031); and secret-class per-proof stream "
            "domains (#567)"
        ),
        "seed": hex64(SEED),
        "input_lengths": lengths,
        "domains": domains,
        "merkle": trees,
        "columns": matrices,
        "commitment_ids": commitment_ids,
        "prover_streams": prover_streams,
    }
    (HERE / "hash.json").write_text(json.dumps(vectors, indent=1) + "\n")


if __name__ == "__main__":
    main()
