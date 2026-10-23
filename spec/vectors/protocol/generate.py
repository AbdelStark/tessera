#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0 OR MIT
"""Reference vectors for the protocol messages of RFC-0036, schema version 1 (tessera-protocol/1):
a small instance of every message, every list filled but the weight rows' masks, which
tessera-protocol/1 leaves empty, with its TCE encoding, its envelope and its value; the coin toss of
a live session; and inputs every reader must refuse, each with the code and the offset RFC-0003
gives it.

This script is the independent oracle. It is written from RFC-0036 ('Versions and envelopes',
'Seed messages', 'Round1', 'Round2Response', 'ColumnQueries, Round3Response and NiProof', 'Matrix
ids'), RFC-0037 'Messages and transcript' (TreeProof and StageProof), 03 §4 and RFC-0017 (the
statement and its predictor), RFC-0031 (the WeightMatrix names), RFC-0003 (the encoding, its
limits and its envelope) and RFC-0004 (the coin-toss domains) alone. Its BLAKE3 and
H(domain, bytes) are those of ../hash/generate.py, whose BLAKE3 is checked against the official
test vectors before anything is written. It shares no code with crates/, verifier/ or python/,
which check its output.

Every message is written out below as a schema, field by field, and encoded by it. A separate
decoder of the same schema then checks every output, applying RFC-0003's checks in the order
RFC-0003 'Limits' fixes and RFC-0036's rule on matrix ids:

- every message decodes back to its value, and every proper prefix of it is ERR-CODEC-TRUNCATED
  at the offset of the first check that needs a byte past the prefix, which the encoder records
  in decoding order and the decoder must find on its own;
- every refused input is built by changing a valid encoding at a place the encoder recorded, and
  the decoder must find, on its own, the code and the offset it was built to fail with;
- the depth cases' offsets come from the depth the encoder gave each value, and the decoder must
  stop at the same byte.

Usage: python3 spec/vectors/protocol/generate.py
(rewrites protocol.json)

Inputs come from SplitMix64 with a fixed seed, so any implementation can regenerate them.
"""

import importlib.util
import json
import math
import sys
from pathlib import Path

sys.dont_write_bytecode = True  # importing the hash oracle leaves nothing behind

HERE = Path(__file__).resolve().parent
_spec = importlib.util.spec_from_file_location("hash_oracle", HERE.parent / "hash" / "generate.py")
hash_oracle = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(hash_oracle)

H, P = hash_oracle.H, hash_oracle.P
SEED = 0x7E55E7A_9A07_0036  # fixed; changing it changes every vector

TRUNCATED = "ERR-CODEC-TRUNCATED"
NONCANONICAL = "ERR-CODEC-NONCANONICAL"
LIMIT = "ERR-CODEC-LIMIT"
MAGIC = "ERR-CODEC-MAGIC"
VERSION = "ERR-CODEC-VERSION"
DIGEST_MISMATCH = "ERR-CODEC-DIGEST"

# RFC-0003 'Limits': the defaults for verifier messages, under which every case is read unless it
# states its own.
LIMITS = {"max_total_bytes": 8 << 30, "max_vec_len": 1 << 34, "max_depth": 32}
DEPTH_CEILING = 256

# RFC-0031 'The descriptor': WeightMatrix in declaration order. A message's u8 matrix id is the
# declaration index (RFC-0036 'Matrix ids').
WEIGHT_MATRICES = ["Embed", "Q", "K", "V", "O", "Gate", "Up", "Down", "LmHead", "ChannelParams"]


def require(condition, message):
    if not condition:
        raise SystemExit(f"protocol oracle: {message}")


# ----- Schemas (RFC-0003 'Primitive types' and 'Composite types') ------------------------------


class U:
    """An unsigned little-endian integer of `width` bytes."""

    def __init__(self, width):
        self.width = width


class Fp:
    """A Goldilocks element: a u64 below P."""


class Fp2:
    """c0 then c1, each an Fp; a composite of one level."""


class Rational:
    """An i64 numerator then a u64 denominator, reduced, the denominator positive."""


class Array:
    """[u8; n]: n raw bytes; a composite of one level."""

    def __init__(self, n):
        self.n = n


class Matrix:
    """A u8 matrix id: the declaration index of a WeightMatrix (RFC-0036 'Matrix ids')."""


class Vec:
    """A u64 length, then the elements."""

    def __init__(self, item):
        self.item = item


class ByMatrix:
    """A Vec of structs that start with a Matrix id, the ids strictly increasing: weight_combined,
    and the queries' and the answer's entries, which follow its order (RFC-0036)."""

    def __init__(self, item):
        self.item = item


class Struct:
    """Fields in declaration order."""

    def __init__(self, name, fields):
        self.name, self.fields = name, fields


class Newtype:
    """A struct of one unnamed field, such as StageId(u16): a struct, so one level."""

    def __init__(self, name, inner):
        self.name, self.inner = name, inner


class Enum:
    """A u16 tag, the variant's declaration index, then the variant's fields."""

    def __init__(self, name, variants):
        self.name, self.variants = name, variants


U8, U16, U32, U64 = U(1), U(2), U(4), U(8)
FP, FP2, RATIONAL, MATRIX = Fp(), Fp2(), Rational(), Matrix()
DIGEST = Array(32)
SESSION_ID = Array(16)

# RFC-0036 'Seed messages'.
SEED_COMMIT = Struct("SeedCommit", [("session_id", SESSION_ID), ("commitment", DIGEST)])
PROVER_SEED_SHARE = Struct("ProverSeedShare", [("session_id", SESSION_ID), ("share", Array(32))])
SEED_REVEAL = Struct("SeedReveal", [("session_id", SESSION_ID), ("share", Array(32))])

# RFC-0036 'Round1', with 03 §4's statement and RFC-0017's predictor.
DECODING = Enum("Decoding", [("Greedy", []), ("Temperature", [("t", RATIONAL)])])
PREDICTOR = Enum(
    "PredictorSpec",
    [("Exact", [("decoding", DECODING)]), ("Mixture", [("decoding", DECODING), ("eps_log2", U8)])],
)
REQUEST_REF = Struct(
    "RequestRef",
    [("request_id", U64), ("session_id", SESSION_ID), ("prompt_len", U32), ("output_len", U32)],
)
BATCH = Struct("BatchDescriptor", [("requests", Vec(REQUEST_REF))])
STATEMENT = Struct(
    "PublicStatement",
    [
        ("claim_list_digest", DIGEST),
        ("weights_root", DIGEST),
        ("weights_epoch", U32),
        ("manifest_digest", DIGEST),
        ("batch", BATCH),
        ("predictor", PREDICTOR),
    ],
)
ADVICE = Struct("AdviceCommitment", [("root", DIGEST), ("layout_digest", DIGEST)])
# RFC-0054 'Messages': g_T and g_L, whose sum is at most 14; a larger sum is ERR-CODEC-LIMIT at the
# struct's first byte, once both fields are read.
MAX_LEAF_GROUP_BITS = 14
LEAF_GROUPS = Struct("LeafGroups", [("token_bits", U8), ("layer_bits", U8)])
ROUND1 = Struct(
    "Round1",
    [
        ("protocol_version", U16),
        ("session_id", SESSION_ID),
        ("statement", STATEMENT),
        ("advice", ADVICE),
        ("leaf_groups", LEAF_GROUPS),
    ],
)

# RFC-0036 'Round2Response', with RFC-0037's TreeProof and StageProof, and RFC-0054's groups and
# merge in TreeProof.
SUMCHECK = Struct("SumcheckProof", [("rounds", Vec(Vec(FP2))), ("final_evals", Vec(FP2))])
LOGUP = Struct("LogUpProof", [("roots", Vec(FP2)), ("values", Vec(FP2))])
GROUP = Struct(
    "GroupProof",
    [("roots", Vec(FP2)), ("levels", Vec(SUMCHECK)), ("values", Vec(FP2)), ("splits", Vec(FP2))],
)
TREES = Struct(
    "TreeProof",
    [
        ("levels", Vec(SUMCHECK)),
        ("groups", Vec(GROUP)),
        ("table_roots", Vec(FP2)),
        ("merge", SUMCHECK),
        ("splits", Vec(FP2)),
    ],
)
STAGE = Struct(
    "StageProof", [("stage", Newtype("StageId", U16)), ("sumcheck", SUMCHECK), ("splits", Vec(FP2))]
)
COMBINED_ROW = Struct("CombinedRow", [("u", Vec(FP2)), ("mask", Vec(FP2))])
WEIGHT_ROWS = Struct(
    "WeightCombinedRows",
    [("matrix", MATRIX), ("evaluations", Vec(COMBINED_ROW)), ("proximity", COMBINED_ROW)],
)
SCORING = Struct("ScoringClaims", [("requests", Vec(Newtype("FixedBits", U64)))])
ROUND2 = Struct(
    "Round2Response",
    [
        ("logup", LOGUP),
        ("trees", TREES),
        ("stages", Vec(STAGE)),
        ("advice_combined", Vec(COMBINED_ROW)),
        ("weight_combined", ByMatrix(WEIGHT_ROWS)),
        ("scoring", SCORING),
    ],
)

# RFC-0036 'ColumnQueries, Round3Response and NiProof'.
MATRIX_QUERIES = Struct("MatrixQueries", [("matrix", MATRIX), ("indices", Vec(U32))])
COLUMN_QUERIES = Struct("ColumnQueries", [("advice", Vec(U32)), ("weights", ByMatrix(MATRIX_QUERIES))])
MULTIPROOF = Struct("MultiProof", [("siblings", Vec(DIGEST))])
MATRIX_COLUMNS = Struct(
    "MatrixColumns", [("matrix", MATRIX), ("columns", Vec(Vec(FP))), ("proof", MULTIPROOF)]
)
ROUND3 = Struct(
    "Round3Response",
    [("advice_columns", Vec(Vec(FP))), ("advice_proof", MULTIPROOF), ("weight_columns", ByMatrix(MATRIX_COLUMNS))],
)
NI_PROOF = Struct(
    "NiProof",
    [("round1", ROUND1), ("seed_nonce", U64), ("round2", ROUND2), ("query_nonce", U64), ("round3", ROUND3)],
)

# 03 §5: every message's schema id; version 1 is tessera-protocol/1.
MESSAGES = {
    "SeedCommit": (SEED_COMMIT, 0x0501),
    "ProverSeedShare": (PROVER_SEED_SHARE, 0x0502),
    "SeedReveal": (SEED_REVEAL, 0x0503),
    "Round1": (ROUND1, 0x0510),
    "Round2Response": (ROUND2, 0x0511),
    "ColumnQueries": (COLUMN_QUERIES, 0x0512),
    "Round3Response": (ROUND3, 0x0513),
    "NiProof": (NI_PROOF, 0x0520),
}
SCHEMA_VERSION = 1


def min_size(t):
    """The fewest bytes a value of `t` encodes to, the factor of RFC-0003's length check."""
    if isinstance(t, U):
        return t.width
    if isinstance(t, (Fp, Rational)):
        return 8 if isinstance(t, Fp) else 16
    if isinstance(t, Fp2):
        return 16
    if isinstance(t, Array):
        return t.n
    if isinstance(t, Matrix):
        return 1
    if isinstance(t, (Vec, ByMatrix)):
        return 8
    if isinstance(t, Struct):
        return sum(min_size(ft) for _, ft in t.fields)
    if isinstance(t, Newtype):
        return min_size(t.inner)
    if isinstance(t, Enum):
        return 2 + min(sum(min_size(ft) for _, ft in fields) for _, fields in t.variants)
    raise SystemExit(f"protocol oracle: no size for {t}")


def composite(t):
    """Whether a value of `t` takes a depth level while it is decoded (RFC-0003 'Limits')."""
    return not isinstance(t, (U, Fp, Matrix))


# ----- The encoder ----------------------------------------------------------------------------


class Encoder:
    """Encodes values by their schema, recording where each value starts (`at`, by path), the
    depth level of every composite value, by path (`level`) and in the order its encoding starts
    (`levels`), and the checks a decoder makes against the bytes left, in its order (`checks`):
    for each, the offset it reports and the bytes it needs the input to hold, RFC-0003 'Limits'.
    A fixed-width value is read whole; a sequence's length is read, then its elements' smallest
    size is checked against the bytes left; an array is checked whole."""

    def __init__(self):
        self.out = bytearray()
        self.at = {}
        self.level = {}
        self.levels = []
        self.checks = []

    def need(self, width):
        """A check at the current offset that needs `width` more bytes."""
        self.checks.append((len(self.out), len(self.out) + width))

    def put(self, data):
        self.out += data

    def value(self, t, v, path=(), depth=0):
        self.at[path] = len(self.out)
        level = depth + 1 if composite(t) else depth
        if composite(t):
            self.level[path] = level
            self.levels.append((len(self.out), level))
        if isinstance(t, U):
            require(isinstance(v, int) and 0 <= v < 2 ** (8 * t.width), f"{path}: u{8 * t.width} {v}")
            self.need(t.width)
            self.put(v.to_bytes(t.width, "little"))
        elif isinstance(t, Fp):
            require(isinstance(v, int) and 0 <= v < P, f"{path}: Fp {v}")
            self.need(8)
            self.put(v.to_bytes(8, "little"))
        elif isinstance(t, Fp2):
            self.value(FP, v[0], path + ("c0",), level)
            self.value(FP, v[1], path + ("c1",), level)
        elif isinstance(t, Rational):
            num, den = v
            require(den > 0 and math.gcd(abs(num), den) == 1, f"{path}: rational {num}/{den}")
            require(-(2**63) <= num < 2**63 and den < 2**64, f"{path}: rational range")
            self.need(8)
            self.put((num % 2**64).to_bytes(8, "little"))
            self.need(8)
            self.put(den.to_bytes(8, "little"))
        elif isinstance(t, Array):
            require(isinstance(v, bytes) and len(v) == t.n, f"{path}: [u8; {t.n}]")
            self.need(t.n)
            self.put(v)
        elif isinstance(t, Matrix):
            require(0 <= v < len(WEIGHT_MATRICES), f"{path}: matrix id {v}")
            self.need(1)
            self.put(v.to_bytes(1, "little"))
        elif isinstance(t, (Vec, ByMatrix)):
            if isinstance(t, ByMatrix):
                ids = [entry["matrix"] for entry in v]
                require(all(a < b for a, b in zip(ids, ids[1:])), f"{path}: matrix ids {ids}")
            self.need(8)
            self.need(8 + len(v) * min_size(t.item))
            self.put(len(v).to_bytes(8, "little"))
            for i, item in enumerate(v):
                self.value(t.item, item, path + (i,), level)
        elif isinstance(t, Struct):
            require(list(v) == [f for f, _ in t.fields], f"{path}: the fields of {t.name}")
            if t is LEAF_GROUPS:
                require(v["token_bits"] + v["layer_bits"] <= MAX_LEAF_GROUP_BITS, f"{path}: {v}")
            for field, ft in t.fields:
                self.value(ft, v[field], path + (field,), level)
        elif isinstance(t, Newtype):
            self.value(t.inner, v, path + ("0",), level)
        elif isinstance(t, Enum):
            name, fields = v
            names = [n for n, _ in t.variants]
            require(name in names, f"{path}: variant {name} of {t.name}")
            tag = names.index(name)
            self.need(2)
            self.put(tag.to_bytes(2, "little"))
            variant = t.variants[tag][1]
            require((fields is None) == (not variant), f"{path}: the fields of {name}")
            for field, ft in variant:
                self.value(ft, fields[field], path + (field,), level)
        else:
            raise SystemExit(f"protocol oracle: no encoding for {t}")


def encoded(t, v):
    encoder = Encoder()
    encoder.value(t, v)
    return encoder


# ----- The decoder: RFC-0003 'Limits', checked as the bytes are read ---------------------------


class Refused(Exception):
    def __init__(self, code, offset):
        super().__init__(f"{code} at {offset}")
        self.code, self.offset = code, offset


class Decoder:
    def __init__(self, data, limits):
        if len(data) > limits["max_total_bytes"]:
            raise Refused(LIMIT, 0)
        self.data, self.pos, self.depth = bytes(data), 0, 0
        self.max_vec_len = limits["max_vec_len"]
        self.max_depth = min(limits["max_depth"], DEPTH_CEILING)

    def take(self, n):
        """n bytes, or ERR-CODEC-TRUNCATED at the first of them."""
        if n > len(self.data) - self.pos:
            raise Refused(TRUNCATED, self.pos)
        out = self.data[self.pos : self.pos + n]
        self.pos += n
        return out

    def uint(self, width):
        return int.from_bytes(self.take(width), "little")

    def enter(self):
        """A composite value's level, taken before any of its bytes is read."""
        if self.depth >= self.max_depth:
            raise Refused(LIMIT, self.pos)
        self.depth += 1

    def leave(self):
        self.depth -= 1

    def read(self, t, previous=None):
        if isinstance(t, U):
            return self.uint(t.width)
        if isinstance(t, Fp):
            at = self.pos
            x = self.uint(8)
            if x >= P:
                raise Refused(NONCANONICAL, at)
            return x
        if isinstance(t, Matrix):
            at = self.pos
            m = self.uint(1)
            if m >= len(WEIGHT_MATRICES) or (previous is not None and m <= previous):
                raise Refused(NONCANONICAL, at)
            return m
        start = self.pos
        self.enter()
        if isinstance(t, Fp2):
            value = (self.read(FP), self.read(FP))
        elif isinstance(t, Rational):
            at = self.pos
            num = self.uint(8)
            num -= (num >> 63) << 64
            den = self.uint(8)
            if den == 0 or math.gcd(abs(num), den) != 1:
                raise Refused(NONCANONICAL, at)
            value = (num, den)
        elif isinstance(t, Array):
            value = self.take(t.n)
        elif isinstance(t, (Vec, ByMatrix)):
            at = self.pos
            n = self.uint(8)
            if n > self.max_vec_len:
                raise Refused(LIMIT, at)
            if n * min_size(t.item) > len(self.data) - self.pos:
                raise Refused(TRUNCATED, at)
            value, last = [], None
            for _ in range(n):
                item = self.read(t.item, last if isinstance(t, ByMatrix) else None)
                if isinstance(t, ByMatrix):
                    last = item["matrix"]
                value.append(item)
        elif isinstance(t, Struct):
            value = {}
            for field, ft in t.fields:
                value[field] = self.read(ft, previous if isinstance(ft, Matrix) else None)
            if t is LEAF_GROUPS and value["token_bits"] + value["layer_bits"] > MAX_LEAF_GROUP_BITS:
                raise Refused(LIMIT, start)
        elif isinstance(t, Newtype):
            value = self.read(t.inner)
        elif isinstance(t, Enum):
            at = self.pos
            tag = self.uint(2)
            if tag >= len(t.variants):
                raise Refused(NONCANONICAL, at)
            name, variant = t.variants[tag]
            fields = {field: self.read(ft) for field, ft in variant}
            value = (name, fields if variant else None)
        else:
            raise SystemExit(f"protocol oracle: no decoding for {t}")
        self.leave()
        return value


def decode(t, data, limits):
    decoder = Decoder(data, limits)
    value = decoder.read(t)
    if decoder.pos != len(decoder.data):
        raise Refused(NONCANONICAL, decoder.pos)
    return value


def outcome(t, data, limits):
    try:
        return ("value", decode(t, data, limits))
    except Refused as refused:
        return (refused.code, refused.offset)


# ----- The envelope (RFC-0003 'Envelope') ------------------------------------------------------

HEADER, DIGEST_LEN = 20, 32


def seal(schema_id, payload, version=SCHEMA_VERSION):
    header = b"TSRA" + (1).to_bytes(2, "little") + schema_id.to_bytes(2, "little")
    header += version.to_bytes(2, "little") + (0).to_bytes(2, "little") + len(payload).to_bytes(8, "little")
    body = header + payload
    return body + H("envelope", body)


def open_envelope(data, schema_id, t, limits):
    """The payload's value, checked in RFC-0003's order. A reader of version 1 accepts version 1
    only: no writer emits version 0."""

    def field(at, n):
        if len(data) < at + n:
            raise Refused(TRUNCATED, at)
        return int.from_bytes(data[at : at + n], "little")

    if len(data) < 4:
        raise Refused(TRUNCATED, 0)
    if data[:4] != b"TSRA":
        raise Refused(MAGIC, 0)
    if field(4, 2) != 1:
        raise Refused(VERSION, 4)
    if field(6, 2) != schema_id:
        raise Refused(MAGIC, 6)
    if field(8, 2) != SCHEMA_VERSION:
        raise Refused(VERSION, 8)
    if field(10, 2) != 0:
        raise Refused(NONCANONICAL, 10)
    n = field(12, 8)
    if n + HEADER + DIGEST_LEN > limits["max_total_bytes"]:
        raise Refused(LIMIT, 12)
    rest = len(data) - HEADER
    if rest < n + DIGEST_LEN:
        raise Refused(TRUNCATED, 12)
    if rest > n + DIGEST_LEN:
        raise Refused(NONCANONICAL, HEADER + n + DIGEST_LEN)
    if H("envelope", data[: HEADER + n]) != data[HEADER + n :]:
        raise Refused(DIGEST_MISMATCH, HEADER + n)
    try:
        return decode(t, data[HEADER : HEADER + n], limits)
    except Refused as refused:
        raise Refused(refused.code, refused.offset + HEADER) from None


# ----- Values as JSON (README.md 'Values') ------------------------------------------------------


def to_json(t, v):
    if isinstance(t, (U, Fp, Matrix)):
        return str(v)
    if isinstance(t, Fp2):
        return [str(v[0]), str(v[1])]
    if isinstance(t, Rational):
        return {"num": str(v[0]), "den": str(v[1])}
    if isinstance(t, Array):
        return v.hex()
    if isinstance(t, (Vec, ByMatrix)):
        return [to_json(t.item, x) for x in v]
    if isinstance(t, Struct):
        return {field: to_json(ft, v[field]) for field, ft in t.fields}
    if isinstance(t, Newtype):
        return to_json(t.inner, v)
    if isinstance(t, Enum):
        name, fields = v
        variant = dict(t.variants)[name]
        return {name: None if not variant else {f: to_json(ft, fields[f]) for f, ft in variant}}
    raise SystemExit(f"protocol oracle: no JSON for {t}")


# ----- The messages ----------------------------------------------------------------------------


class Inputs:
    def __init__(self, seed):
        self.rng = hash_oracle.splitmix64(seed)

    def digest(self):
        return hash_oracle.digest_from(self.rng)

    def session(self):
        return hash_oracle.digest_from(self.rng)[:16]

    def fp(self):
        return hash_oracle.canonical(self.rng)

    def fp2(self):
        return (self.fp(), self.fp())

    def fp2s(self, n):
        return [self.fp2() for _ in range(n)]

    def fps(self, n):
        return [self.fp() for _ in range(n)]


def messages(rng):
    """One small instance of every message, every list filled but the weight rows' masks, and the
    coin-toss inputs."""
    s_v, s_p, session = rng.digest(), rng.digest(), rng.session()
    seed_commit = {"session_id": session, "commitment": H("coin-commit", s_v)}
    prover_share = {"session_id": session, "share": s_p}
    seed_reveal = {"session_id": session, "share": s_v}

    def statement(requests, epoch, predictor):
        return {
            "claim_list_digest": rng.digest(),
            "weights_root": rng.digest(),
            "weights_epoch": epoch,
            "manifest_digest": rng.digest(),
            "batch": {"requests": requests},
            "predictor": predictor,
        }

    exact = statement(
        [
            {"request_id": 1, "session_id": rng.session(), "prompt_len": 17, "output_len": 5},
            {"request_id": 2**64 - 1, "session_id": rng.session(), "prompt_len": 2**32 - 1, "output_len": 0},
        ],
        7,
        ("Exact", {"decoding": ("Greedy", None)}),
    )
    # The claim list's protocol version: 0 until tessera-protocol/1 is assigned (09 §3).
    round1_exact = {
        "protocol_version": 0,
        "session_id": session,
        "statement": exact,
        "advice": {"root": rng.digest(), "layout_digest": rng.digest()},
        "leaf_groups": {"token_bits": 0, "layer_bits": 0},
    }
    mixture = statement(
        [
            {"request_id": 40, "session_id": rng.session(), "prompt_len": 3, "output_len": 9},
            {"request_id": 41, "session_id": rng.session(), "prompt_len": 1024, "output_len": 1},
            {"request_id": 0, "session_id": rng.session(), "prompt_len": 0, "output_len": 2**32 - 1},
        ],
        2**32 - 1,
        ("Mixture", {"decoding": ("Temperature", {"t": (7, 10)}), "eps_log2": 6}),
    )
    round1_mixture = {
        "protocol_version": 0,
        "session_id": rng.session(),
        "statement": mixture,
        "advice": {"root": rng.digest(), "layout_digest": rng.digest()},
        # RFC-0054's bound exactly: 2^6 token subsets by 2^8 layer subsets.
        "leaf_groups": {"token_bits": 6, "layer_bits": 8},
    }

    def weight_rows(matrix, k, m=1):
        return {
            "matrix": matrix,
            "evaluations": [{"u": rng.fp2s(k), "mask": []} for _ in range(m)],
            "proximity": {"u": rng.fp2s(k), "mask": []},
        }

    roots = rng.fp2s(8)
    roots[0] = (0, P - 1)  # both ends of the field
    round2 = {
        "logup": {"roots": roots, "values": rng.fp2s(1)},
        "trees": {
            "levels": [
                {"rounds": [], "final_evals": rng.fp2s(4)},
                {"rounds": [rng.fp2s(3)], "final_evals": rng.fp2s(4)},
                {"rounds": [rng.fp2s(3), rng.fp2s(3)], "final_evals": rng.fp2s(6)},
            ],
            # Two groups, the second holding no tree of any instance (RFC-0054 'Groups').
            "groups": [
                {
                    "roots": rng.fp2s(4),
                    "levels": [{"rounds": [], "final_evals": rng.fp2s(2)}, {"rounds": [rng.fp2s(3)], "final_evals": rng.fp2s(2)}],
                    "values": rng.fp2s(1),
                    "splits": rng.fp2s(2),
                },
                {"roots": [], "levels": [], "values": [], "splits": []},
            ],
            "table_roots": rng.fp2s(2),
            "merge": {"rounds": [rng.fp2s(2), rng.fp2s(2)], "final_evals": rng.fp2s(3)},
            "splits": rng.fp2s(2),
        },
        "stages": [
            {
                "stage": 1,
                "sumcheck": {"rounds": [rng.fp2s(2), rng.fp2s(2), rng.fp2s(2)], "final_evals": rng.fp2s(3)},
                "splits": rng.fp2s(2),
            },
            {"stage": 2, "sumcheck": {"rounds": [rng.fp2s(3)], "final_evals": rng.fp2s(2)}, "splits": rng.fp2s(1)},
        ],
        "advice_combined": [{"u": rng.fp2s(4), "mask": rng.fp2s(2)} for _ in range(3)],
        # tessera-protocol/1's weight encoding has no mask (RFC-0012 'Migration / Rollout').
        "weight_combined": [weight_rows(1, 2), weight_rows(8, 2), weight_rows(9, 2, 2)],
        "scoring": {"requests": [0, 1234]},
    }
    # The indices of a proof with n_A = 16 and n_T = 8, repeats included.
    queries = {
        "advice": [5, 0, 13, 5, 9],
        "weights": [
            {"matrix": 1, "indices": [3, 3, 0]},
            {"matrix": 8, "indices": [7, 1, 6]},
            {"matrix": 9, "indices": [0, 0, 0]},
        ],
    }
    queries_wide = {"advice": [2**32 - 1, 0, 2**31], "weights": []}
    advice_columns = [rng.fps(3) for _ in range(4)]  # distinct 0, 5, 9, 13
    advice_columns[0][0], advice_columns[3][2] = P - 1, 0
    round3 = {
        "advice_columns": advice_columns,
        "advice_proof": {"siblings": [rng.digest() for _ in range(3)]},
        "weight_columns": [
            {"matrix": 1, "columns": [rng.fps(2) for _ in range(2)], "proof": {"siblings": [rng.digest() for _ in range(2)]}},
            {"matrix": 8, "columns": [rng.fps(2) for _ in range(3)], "proof": {"siblings": [rng.digest()]}},
            {"matrix": 9, "columns": [rng.fps(2)], "proof": {"siblings": [rng.digest() for _ in range(3)]}},
        ],
    }
    ni_proof = {
        "round1": round1_exact,
        "seed_nonce": 0x0123456789ABCDEF,
        "round2": round2,
        "query_nonce": 2**64 - 1,
        "round3": round3,
    }
    cases = [
        ("the verifier's commitment to its share", "SeedCommit", seed_commit),
        ("the prover's share", "ProverSeedShare", prover_share),
        ("the verifier's share, which opens the commitment", "SeedReveal", seed_reveal),
        ("an exact, greedy statement of two requests", "Round1", round1_exact),
        ("a mixture predictor at temperature 7/10 over three requests", "Round1", round1_mixture),
        ("three tree levels, two stages, three advice rows and three opened matrices", "Round2Response", round2),
        ("queries for the advice and three matrices, with repeats", "ColumnQueries", queries),
        ("queries at the ends of u32, no matrix opened", "ColumnQueries", queries_wide),
        ("the answers to the first queries", "Round3Response", round3),
        ("a non-interactive proof of the same rounds", "NiProof", ni_proof),
    ]
    return cases, (s_v, s_p)


def coin_tosses(rng, shares):
    """`c_V = H("coin-commit", s_V)` and `seed = H("coin-toss", s_V ‖ s_P)` (RFC-0036 'Seed
    messages', RFC-0004)."""
    pairs = [shares, (bytes(32), b"\xff" * 32), (rng.digest(), rng.digest())]
    return [
        {
            "verifier_share": s_v.hex(),
            "prover_share": s_p.hex(),
            "commitment": H("coin-commit", s_v).hex(),
            "seed": H("coin-toss", s_v + s_p).hex(),
        }
        for s_v, s_p in pairs
    ]


# ----- Refused inputs ----------------------------------------------------------------------------


def changed(data, at, replacement):
    return data[:at] + replacement + data[at + len(replacement) :]


def le(x, width):
    return x.to_bytes(width, "little")


def refusals(cases):
    """Inputs built to fail one check each, with the code and offset they were built for, which the
    decoder must find too."""
    first = {}
    for _, ty, value in cases:
        first.setdefault(ty, value)
    built = {ty: encoded(MESSAGES[ty][0], value) for ty, value in first.items()}
    mixture = encoded(ROUND1, cases[4][2])
    refused, accepted = [], []

    def refuse(ty, data, code, offset, why, limits=None):
        found = outcome(MESSAGES[ty][0], data, limits or LIMITS)
        require(found == (code, offset), f"{why}: the decoder gives {found}, the case was built for {code} at {offset}")
        case = {"type": ty, "hex": data.hex(), "code": code, "offset": offset, "why": why}
        if limits:
            case["limits"] = limits
        refused.append(case)

    def accept(ty, data, why, limits):
        found = outcome(MESSAGES[ty][0], data, limits)
        require(found[0] == "value", f"{why}: refused with {found}")
        accepted.append({"type": ty, "hex": data.hex(), "limits": limits, "why": why})

    for ty, enc in built.items():
        data = bytes(enc.out)
        refuse(ty, data + b"\x00", NONCANONICAL, len(data), "a byte after the message (INV-TCE-4)")

    # Truncation inside a fixed-size value: an array is checked whole, an integer as it is read.
    commit = bytes(built["SeedCommit"].out)
    refuse("SeedCommit", commit[:36], TRUNCATED, 16, "cut inside the commitment, an array checked whole")
    ni = built["NiProof"]
    at = ni.at[("seed_nonce",)]
    refuse("NiProof", bytes(ni.out[: at + 3]), TRUNCATED, at, "cut inside seed_nonce")

    # Enum tags (RFC-0003 'Composite types').
    r1 = built["Round1"]
    data = bytes(r1.out)
    at = r1.at[("statement", "predictor")]
    refuse("Round1", changed(data, at, le(2, 2)), NONCANONICAL, at, "PredictorSpec tag 2: it has two variants")
    at = r1.at[("statement", "predictor", "decoding")]
    refuse("Round1", changed(data, at, le(2, 2)), NONCANONICAL, at, "Decoding tag 2: it has two variants")
    at = ni.at[("round1", "statement", "predictor")]
    refuse("NiProof", changed(bytes(ni.out), at, le(0xFFFF, 2)), NONCANONICAL, at, "PredictorSpec tag 65535 inside the proof")

    # RFC-0054's grouping: at most 14 group bits, checked once both fields are read.
    at = r1.at[("leaf_groups",)]
    refuse("Round1", changed(data, at, le(7, 1) + le(8, 1)), LIMIT, at, "leaf_groups 7 + 8 = 15 bits, one past RFC-0054's 14")
    refuse("Round1", changed(data, at, le(255, 1) + le(255, 1)), LIMIT, at, "leaf_groups 255 + 255: the sum, not each field, is bounded")
    # Sums that wrap a u8 to 0 and to 1: the bound is on the integer sum.
    refuse("Round1", changed(data, at, le(128, 1) + le(128, 1)), LIMIT, at, "leaf_groups 128 + 128, 0 modulo 256")
    refuse("Round1", changed(data, at, le(250, 1) + le(7, 1)), LIMIT, at, "leaf_groups 250 + 7, 1 modulo 256")
    refuse("Round1", data[: at + 1], TRUNCATED, at + 1, "cut inside leaf_groups: truncation before the bound")
    at = ni.at[("round1", "leaf_groups")]
    refuse("NiProof", changed(bytes(ni.out), at, le(15, 1)), LIMIT, at, "leaf_groups 15 + 0 inside the proof")

    # Rationals: reduced, the denominator positive.
    data = bytes(mixture.out)
    at = mixture.at[("statement", "predictor", "decoding", "t")]
    refuse("Round1", changed(data, at, le(14, 8) + le(20, 8)), NONCANONICAL, at, "the temperature 14/20, not reduced")
    refuse("Round1", changed(data, at, le(1, 8) + le(0, 8)), NONCANONICAL, at, "the temperature 1/0")

    # Lengths: max_vec_len first, then the bytes left, both before any element is read.
    data = bytes(r1.out)
    at = r1.at[("statement", "batch", "requests")]
    refuse("Round1", changed(data, at, le((1 << 34) + 1, 8)), LIMIT, at, "2^34 + 1 requests, past max_vec_len")
    refuse("Round1", changed(data, at, le(1 << 33, 8)), TRUNCATED, at, "2^33 requests of at least 32 bytes in what is left")
    r2 = built["Round2Response"]
    at = r2.at[("stages",)]
    refuse("Round2Response", changed(bytes(r2.out), at, le(2**64 - 1, 8)), LIMIT, at, "2^64 - 1 stages")
    r3 = built["Round3Response"]
    refuse("Round3Response", changed(bytes(r3.out), 0, le(1 << 63, 8)), LIMIT, 0, "2^63 advice columns")
    at = r3.at[("advice_proof", "siblings")]
    refuse("Round3Response", changed(bytes(r3.out), at, le(1 << 20, 8)), TRUNCATED, at, "2^20 siblings of 32 bytes in what is left")
    queries = built["ColumnQueries"]
    refuse("ColumnQueries", changed(bytes(queries.out), 0, le((1 << 34) + 1, 8)), LIMIT, 0, "2^34 + 1 advice indices")

    # Field elements below P (RFC-0002).
    data = bytes(r2.out)
    for path, word, why in [
        (("logup", "roots", 0, "c0"), P, "a root's c0 equal to P"),
        (("logup", "roots", 1, "c1"), 2**64 - 1, "a root's c1 equal to 2^64 - 1"),
        (("trees", "levels", 1, "rounds", 0, 2, "c0"), P + 1, "a tree round's value at P + 1"),
        (("trees", "groups", 0, "roots", 3, "c1"), P, "a group root's c1 at P"),
        (("trees", "groups", 0, "levels", 1, "rounds", 0, 0, "c0"), P, "a group level's round value at P"),
        (("trees", "table_roots", 1, "c0"), 2**64 - 1, "a table root's c0 at 2^64 - 1"),
        (("trees", "merge", "final_evals", 2, "c1"), P, "a merge evaluation at P"),
        (("stages", 0, "sumcheck", "final_evals", 0, "c1"), P, "a stage's final evaluation at P"),
        (("advice_combined", 2, "mask", 1, "c0"), P, "a combined mask coefficient at P"),
    ]:
        at = r2.at[path]
        refuse("Round2Response", changed(data, at, le(word, 8)), NONCANONICAL, at, why)
    at = r3.at[("advice_columns", 1, 0)]
    refuse("Round3Response", changed(bytes(r3.out), at, le(P, 8)), NONCANONICAL, at, "an advice symbol equal to P")

    # Matrix ids: a WeightMatrix index, strictly increasing (RFC-0036 'Matrix ids').
    for ty, enc, field in [
        ("Round2Response", r2, "weight_combined"),
        ("ColumnQueries", queries, "weights"),
        ("Round3Response", r3, "weight_columns"),
    ]:
        data = bytes(enc.out)
        for entry, matrix, why in [
            (1, 1, "the second entry repeats the first's matrix"),
            (1, 0, "the second entry's matrix is below the first's"),
            (2, 1, "the third entry's matrix is below the second's"),
            (0, 10, "matrix id 10 names no matrix"),
            (2, 255, "matrix id 255 names no matrix"),
        ]:
            at = enc.at[(field, entry)]
            require(enc.at[(field, entry, "matrix")] == at, "an entry starts with its matrix id")
            refuse(ty, changed(data, at, le(matrix, 1)), NONCANONICAL, at, f"{field}: {why}")
    at = ni.at[("round2", "weight_combined", 2)]
    refuse("NiProof", changed(bytes(ni.out), at, le(8, 1)), NONCANONICAL, at, "weight_combined: a repeated matrix inside the proof")

    # Depth (RFC-0003 'Limits'): one level short of each message's deepest value, and exactly at it.
    for ty, enc in built.items():
        data = bytes(enc.out)
        deepest = max(level for _, level in enc.levels)
        at = next(offset for offset, level in enc.levels if level == deepest)
        short = dict(LIMITS, max_depth=deepest - 1)
        refuse(ty, data, LIMIT, at, f"max_depth {deepest - 1}, one level short of its deepest value", short)
        accept(ty, data, f"max_depth {deepest}, its deepest value's level", dict(LIMITS, max_depth=deepest))

    # A struct of one unnamed field takes a level of its own (RFC-0003), but no input observes
    # StageId's or FixedBits' since RFC-0054: TreeProof's merge, always present, puts two
    # sequences at their level before them, so a reader that skipped their level would fail
    # where a correct one fails on every input. A Round2Response whose other lists are empty
    # still isolates the branches below.
    def bare(stages, scoring):
        return {
            "logup": {"roots": [], "values": []},
            "trees": {
                "levels": [],
                "groups": [],
                "table_roots": [],
                "merge": {"rounds": [], "final_evals": []},
                "splits": [],
            },
            "stages": stages,
            "advice_combined": [],
            "weight_combined": [],
            "scoring": {"requests": scoring},
        }

    # Isolate branches whose missing struct level would be hidden by the full packet's
    # deeper siblings. The encoder locates their first deepest value independently.
    combined_only = bare([], [])
    combined_only["advice_combined"] = [{"u": [(0, 1)], "mask": []}]
    logup_only = bare([], [])
    logup_only["logup"]["roots"] = [(0, 1)]
    for value, path, why in [
        (combined_only, ("advice_combined", 0, "u", 0), "CombinedRow's struct depth"),
        (logup_only, ("logup", "roots", 0), "LogUpProof's struct depth"),
    ]:
        enc = encoded(ROUND2, value)
        level, at = enc.level[path], enc.at[path]
        require(next(o for o, l in enc.levels if l >= level) == at, f"{why}: the first at its level")
        refuse("Round2Response", bytes(enc.out), LIMIT, at, f"{why}, one level short",
               dict(LIMITS, max_depth=level - 1))
        accept("Round2Response", bytes(enc.out), f"{why}, exactly at its deepest level",
               dict(LIMITS, max_depth=level))

    # The whole input against max_total_bytes, and a length exactly at max_vec_len.
    commit = bytes(built["SeedCommit"].out)
    refuse("SeedCommit", commit, LIMIT, 0, "48 bytes against max_total_bytes 47", dict(LIMITS, max_total_bytes=47))
    accept("SeedCommit", commit, "48 bytes at max_total_bytes 48", dict(LIMITS, max_total_bytes=48))
    data = bytes(queries.out)
    refuse("ColumnQueries", data, LIMIT, 0, "5 advice indices against max_vec_len 4", dict(LIMITS, max_vec_len=4))
    accept("ColumnQueries", data, "5 advice indices at max_vec_len 5", dict(LIMITS, max_vec_len=5))
    return refused, accepted


def envelope_refusals(cases):
    """Envelopes a reader of the schema must refuse, each for one fault (RFC-0003 'Envelope')."""
    first = {}
    for _, ty, value in cases:
        first.setdefault(ty, value)
    refused = []

    def refuse(schema, data, code, offset, why):
        t, schema_id = MESSAGES[schema]
        try:
            open_envelope(data, schema_id, t, LIMITS)
            found = "accepted"
        except Refused as r:
            found = (r.code, r.offset)
        require(found == (code, offset), f"{why}: the reader gives {found}, built for {code} at {offset}")
        refused.append({"schema": schema, "hex": data.hex(), "code": code, "offset": offset, "why": why})

    round1 = bytes(encoded(ROUND1, first["Round1"]).out)
    refuse("Round1", seal(0x0510, round1, version=2), VERSION, 8, "schema version 2, tessera-protocol/2: the protocols never mix")
    refuse("Round1", seal(0x0510, round1, version=0), VERSION, 8, "schema version 0, which no writer emits")
    reveal = bytes(encoded(SEED_REVEAL, first["SeedReveal"]).out)
    refuse("SeedCommit", seal(0x0503, reveal), MAGIC, 6, "a SeedReveal where a SeedCommit is expected")
    queries = encoded(COLUMN_QUERIES, first["ColumnQueries"])
    at = queries.at[("weights", 1)]
    unsorted = changed(bytes(queries.out), at, le(1, 1))
    refuse("ColumnQueries", seal(0x0512, unsorted), NONCANONICAL, HEADER + at, "a repeated matrix in the payload, offset from the envelope's start")
    round2 = bytes(encoded(ROUND2, first["Round2Response"]).out)
    sealed = seal(0x0511, round2)
    flipped = changed(sealed, HEADER + 5, bytes([sealed[HEADER + 5] ^ 1]))
    refuse("Round2Response", flipped, DIGEST_MISMATCH, HEADER + len(round2), "a payload byte changed under the digest")
    refuse("SeedCommit", seal(0x0501, bytes(encoded(SEED_COMMIT, first["SeedCommit"]).out))[:-1], TRUNCATED, 12, "one digest byte short")
    return refused


# ----- Output --------------------------------------------------------------------------------


def main():
    checked = hash_oracle.check_official_vectors()
    rng = Inputs(SEED)
    cases, shares = messages(rng)
    rendered = []
    for name, ty, value in cases:
        t, schema_id = MESSAGES[ty]
        enc = encoded(t, value)
        payload = bytes(enc.out)
        require(decode(t, payload, LIMITS) == value, f"{name}: the decoder does not give the value back")
        truncated = []
        for cut in range(len(payload)):
            found = outcome(t, payload[:cut], LIMITS)
            built = next(at for at, need in enc.checks if need > cut)
            require(found == (TRUNCATED, built), f"{name}: the prefix of {cut} bytes gives {found}, built for {built}")
            if not truncated or truncated[-1][1] != built:
                truncated.append([cut, built])
        envelope = seal(schema_id, payload)
        require(open_envelope(envelope, schema_id, t, LIMITS) == value, f"{name}: the envelope")
        case = {
            "name": name,
            "type": ty,
            "value": to_json(t, value),
            "payload": payload.hex(),
            "truncated": truncated,
            "envelope": envelope.hex(),
        }
        if ty == "Round1":
            case["statement_digest"] = H("statement", bytes(encoded(STATEMENT, value["statement"]).out)).hex()
        rendered.append(case)
    refused, accepted = refusals(cases)
    document = {
        "description": (
            "Protocol message vectors (RFC-0036, schema version 1, tessera-protocol/1), written by "
            "generate.py, an oracle independent of the Rust crates and the Python package. Each "
            "`messages` entry decodes from `payload` as `type` under `limits` to `value` (README.md "
            "'Values'), encodes back to `payload`, and every proper prefix of `payload` fails with "
            "ERR-CODEC-TRUNCATED at the offset `truncated` gives: runs [length, offset], each "
            "covering the prefixes from its length to the next run's; `envelope` is its RFC-0003 "
            "envelope at schema version 1, which a "
            "reader of the schema opens to `value`. A Round1's `statement_digest` is "
            "H(\"statement\", TCE(statement)). Each `coin_toss` entry gives "
            "commitment = H(\"coin-commit\", s_V) and seed = H(\"coin-toss\", s_V || s_P). Decoding a "
            "`refused` entry as `type`, under its own `limits` if it has them, fails with `code` at "
            "`offset`, the first byte the failing check reads (RFC-0003 'Limits'); an `accepted` "
            "entry sits exactly at its limits and decodes. Opening an `envelopes` entry as a "
            "message of `schema` fails with `code` at `offset`, counted from the envelope's start."
        ),
        "official_blake3_cases_checked": checked,
        "limits": LIMITS,
        "schemas": {ty: {"id": schema_id, "version": SCHEMA_VERSION} for ty, (_, schema_id) in MESSAGES.items()},
        "matrices": WEIGHT_MATRICES,
        "messages": rendered,
        "coin_toss": coin_tosses(rng, shares),
        "refused": refused,
        "accepted": accepted,
        "envelopes": envelope_refusals(cases),
    }
    (HERE / "protocol.json").write_text(json.dumps(document, indent=1) + "\n")
    print(
        f"wrote protocol.json: {len(rendered)} messages, {len(refused)} refused inputs, "
        f"{len(accepted)} at their limits, {len(document['envelopes'])} refused envelopes"
    )


if __name__ == "__main__":
    main()
