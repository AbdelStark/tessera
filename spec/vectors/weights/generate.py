#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0 OR MIT
"""Reference vectors for the weight commitment of RFC-0040 at test-small/1: for each conformance
model of ../models/ and for one synthetic model, each of the ten matrices' message width k_T,
committed rows, commitment_id and root, then R_W and the WeightCommitmentPublic artifact (.twr);
and rows encoded at rate 1/4, the code every weight matrix uses.

This script is the independent oracle. It is written from RFC-0040 ('The matrices', 'Encoding,
leaves and roots', 'WeightCommitmentPublic'), RFC-0012 'Layout', RFC-0031 'The descriptor',
RFC-0004 ('Column digests', 'Merkle trees'), RFC-0011 'Encoding', RFC-0003 and 03 §2 alone,
computing with Python's integers. It shares no code with crates/, verifier/ or python/, which
check its output. It takes from the other oracles:

- BLAKE3, H(domain, bytes), the Merkle levels and TCE(CommitmentDescriptor) from
  ../hash/generate.py, whose BLAKE3 is checked against the official test vectors first;
- the roots of unity, Horner's rule and the masked polynomial from ../encoding/generate.py;
- the radix-2 NTT from ../pcs/generate.py;
- the IntegerModel types and encoder from ../models/generate.py.

It decodes each model with its own TCE reader, checks that the models oracle's encoder writes the
artifact back byte for byte, and that its digest is the one ../models/models.json lists. Two parts
run faster than their definitions, and each is checked against them before anything is written:

- Codewords. A row's polynomial comes from an inverse NTT, and its codeword from NTTs of size k
  on the four cosets c_s H that make up 7.L. Before each matrix is hashed, its first and last
  committed rows are checked against the definition: the polynomial takes the row's values on
  H, and its codeword symbols are that polynomial at 7 w_n^j by Horner's rule, every one for the
  first row of each size.
- Hashing. Every column of a matrix advances its chain at the same time, so the column chains
  and the Merkle levels run as one BLAKE3 compression over many columns at once: each 32-bit word
  of the state holds one lane per column, 64 bits apart, in one Python integer. Every input here
  is a single block (column-init, column-step with B = 4, merkle-node), so one compression is one
  hash. The lanes are checked against the hash oracle's H on random inputs first, and for every
  matrix one column's chain and, for every model, one matrix's Merkle levels are recomputed with
  H itself.

Usage: python3 spec/vectors/weights/generate.py
(rewrites weights.json and the *.twr artifacts)
"""

import importlib.util
import json
import struct
import sys
from pathlib import Path

sys.dont_write_bytecode = True  # importing the other oracles leaves nothing behind

HERE = Path(__file__).resolve().parent


def _oracle(name):
    spec = importlib.util.spec_from_file_location(f"{name}_oracle", HERE.parent / name / "generate.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


hashing = _oracle("hash")  # BLAKE3, H, Merkle levels, TCE(CommitmentDescriptor)
encoding = _oracle("encoding")  # roots of unity, Horner, the masked polynomial
pcs = _oracle("pcs")  # the radix-2 NTT
models = _oracle("models")  # the IntegerModel types and encoder

P, H = hashing.P, hashing.H


def require(condition, message):
    """A check that holds under `python -O` too, unlike `assert`."""
    if not condition:
        raise SystemExit(f"weights oracle: {message}")


require(encoding.P == P and pcs.P == P, "the oracles share the modulus")
require(encoding.G == 7 and pcs.SHIFT == 7, "the codeword's coset is 7.L (RFC-0011 'Encoding')")

SEED = 0x7E55E7A_0040_0102  # fixed; changing it changes the encoding cases and the synthetic model

# test-small/1 (RFC-0014 'Parameter sets', RFC-0026 'Schema'): the claim list's WeightLayoutParams,
# the row block B and the first epoch (RFC-0012 'Layout').
PARAMS = "test-small/1"
LAYOUT = {"log2_k": 8, "log2_inv_rate": 2, "mask_coeffs": 64, "blinding_rows": 32}
ROW_BLOCK = 4
EPOCH = 0

# 03 §5: WeightCommitmentPublic.
SCHEMA_ID, SCHEMA_VERSION = 0x0301, 1

# 03 §2's weight types, in declaration order, and RFC-0031's WeightMatrix: the same nine, then
# ChannelParams.
WEIGHT_TYPES = ["Embed", "Q", "K", "V", "O", "Gate", "Up", "Down", "LmHead"]
MATRICES = WEIGHT_TYPES + ["ChannelParams"]
require(MATRICES == hashing.WEIGHT_MATRICES, "the hash oracle's WeightMatrix order")
require(WEIGHT_TYPES == models.WEIGHT_TYPES, "the models oracle's WeightType order")
PER_LAYER = WEIGHT_TYPES[1:8]
CHANNEL_TYPES = 16  # the ChannelParams tensor's types, 9 padded to 16 (RFC-0012 'Layout')

# The conformance models (../models/README.md), in their index's order.
MODEL_NAMES = ["tiny-2-servable", "tiny-2-minimal", "tiny-2-affine", "tiny-4-servable"]


# ----- RFC-0003: reading IntegerModel ------------------------------------------------------------
#
# The models oracle's types describe the schema: (primitive,), ("vec", T), ("tuple", T1, T2),
# ("map", K, V), ("struct", [(name, T), ...]) or ("enum", [(variant, T or None), ...]). A decoded
# value takes the form that oracle's encoder takes.

WIDTHS = {"u8": 1, "u16": 2, "u32": 4, "u64": 8, "i8": 1, "i16": 2, "i32": 4, "i64": 8}


class Located(list):
    """A decoded vector of integers, with the offset in the artifact of its first element."""

    offset = None


class Input:
    """The bytes of one artifact and a position in them."""

    def __init__(self, data, pos):
        self.data, self.pos = data, pos

    def take(self, n):
        require(self.pos + n <= len(self.data), f"truncated at {self.pos}")
        out = self.data[self.pos:self.pos + n]
        self.pos += n
        return out

    def integer(self, kind):
        return int.from_bytes(self.take(WIDTHS[kind]), "little", signed=kind[0] == "i")


def decode(ty, src):
    kind = ty[0]
    if kind in WIDTHS:
        return src.integer(kind)
    if kind == "digest":
        return list(src.take(32))
    if kind == "ascii":
        data = src.take(src.integer("u64"))
        require(all(0x20 <= b <= 0x7E for b in data), "printable ASCII")
        return data.decode("ascii")
    if kind == "rational":
        num, den = src.integer("i64"), src.integer("u64")
        return {"num": num, "den": den}
    if kind == "vec":
        n = src.integer("u64")
        element = ty[1]
        if element[0] in WIDTHS:
            out = Located()
            out.offset = src.pos
            width = WIDTHS[element[0]]
            data = src.take(n * width)
            out.extend(int.from_bytes(data[i:i + width], "little", signed=element[0][0] == "i")
                       for i in range(0, len(data), width))
            return out
        return [decode(element, src) for _ in range(n)]
    if kind == "tuple":
        return [decode(t, src) for t in ty[1:]]
    if kind == "map":
        return [[decode(ty[1], src), decode(ty[2], src)] for _ in range(src.integer("u64"))]
    if kind == "struct":
        return {name: decode(t, src) for name, t in ty[1]}
    if kind == "enum":
        tag = src.integer("u16")
        require(tag < len(ty[1]), f"enum tag {tag}")
        variant, fields = ty[1][tag]
        return {variant: None if fields is None else decode(fields, src)}
    raise SystemExit(f"weights oracle: unknown type {ty!r}")


def read_model(name, index):
    """The model in ../models/<name>.d: its envelope's header and size checked (RFC-0003
    'Envelope'), its digest the one ../models/models.json lists, its payload decoded, and the
    models oracle's encoder writing it back. (The models oracle computed that digest; recomputing
    it here would hash every model again in Python.)"""
    data = (HERE.parent / "models" / f"{name}.d").read_bytes()
    require(data[:4] == b"TSRA", f"{name}: magic")
    header = [int.from_bytes(data[4 + 2 * i:6 + 2 * i], "little") for i in range(4)]
    require(header == [1, models.SCHEMA_ID, models.SCHEMA_VERSION, 0], f"{name}: header")
    payload_len = int.from_bytes(data[12:20], "little")
    require(len(data) == 20 + payload_len + 32, f"{name}: length")
    listed = [m for m in index["models"] if m["name"] == name]
    require(len(listed) == 1 and listed[0]["file"] == f"{name}.d", f"{name}: listed once in models.json")
    require(listed[0]["digest"] == data[-32:].hex() and listed[0]["bytes"] == len(data), f"{name}: as models.json lists it")
    src = Input(data, 20)
    value = decode(models.INTEGER_MODEL, src)
    require(src.pos == 20 + payload_len, f"{name}: the value fills the payload")
    require(models.encode(models.INTEGER_MODEL, value) == data[20:-32], f"{name}: decodes and encodes back")
    return value, data


# ----- RFC-0040 'The matrices' -------------------------------------------------------------------


def pad(n):
    """The next power of two of n >= 1."""
    require(n >= 1, "a dimension is at least 1")
    return 1 << (n - 1).bit_length()


def log2(x):
    require(x >= 1 and x & (x - 1) == 0, f"{x} is a power of two")
    return x.bit_length() - 1


def shapes_of(arch):
    """(out, in, layers) of each weight type (03 §2): rows are output channels, Embed is
    d_model x vocab, transposed, and Embed and LmHead have one layer."""
    d, ff, vocab, n = arch["d_model"], arch["d_ff"], arch["vocab"], arch["n_layers"]
    q, kv = arch["n_heads"] * arch["d_head"], arch["n_kv_heads"] * arch["d_head"]
    return {"Embed": (d, vocab, 1), "Q": (q, d, n), "K": (kv, d, n), "V": (kv, d, n), "O": (d, q, n),
            "Gate": (ff, d, n), "Up": (ff, d, n), "Down": (d, ff, n), "LmHead": (vocab, d, 1)}


def weight_matrix(out, inputs, layers, k_w, tensors):
    """The matrix of one weight type: k_T = max(in_pad, min(k_W, out_pad in_pad)),
    a_T = log2(k_T / in_pad), h_T = log2 out_pad - a_T; W_l[o][i] at row l 2^h + o >> a and
    column (o mod 2^a) in_pad + i; a row is committed unless l >= layers or (o >> a) 2^a >= out.
    `tensors[l]` holds W_l row-major. Returns the layout and the committed rows' values."""
    out_pad, in_pad = pad(out), pad(inputs)
    k = max(in_pad, min(k_w, out_pad * in_pad))
    a = log2(k // in_pad)
    h = log2(out_pad) - a
    require(h >= 0, "a row holds at most every output")
    committed, rows = [], []
    for r in range(pad(layers) << h):
        layer, o_hi = r >> h, r & ((1 << h) - 1)
        if layer >= layers or o_hi << a >= out:
            continue  # structurally zero: a padded layer, or only padded outputs
        row = [0] * k
        data = tensors[layer]
        for o_lo in range(1 << a):
            o = (o_hi << a) + o_lo
            if o < out:
                for i in range(inputs):
                    row[o_lo * in_pad + i] = data[o * inputs + i] % P
        committed.append(r)
        rows.append(row)
    layout = {"out": out, "in": inputs, "layers": layers, "log2_k": log2(k), "a": a, "h": h,
              "row_vars": log2(pad(layers) << h)}
    return layout, committed, rows


def channel_matrix(channels, widths, layers, k_w):
    """The ChannelParams matrix: the tensor [16, L_pad, C] row-major, entry f = (t L_pad + l) C + c
    at row f // k and column f mod k, k = min(k_W, 16 L_pad C). C is the largest padded output
    width among the weights that have a ChannelParams, 1 when none has. A position is live when
    (t, l) has a ChannelParams and c is below its output width, and a row is committed when it
    holds a live position. `channels` maps (t, l) to its values, `widths` t to its output width."""
    l_pad = pad(layers)
    c_width = max((pad(widths[t]) for t, _ in channels), default=1)
    total = CHANNEL_TYPES * l_pad * c_width
    k = min(k_w, total)
    live = {}
    for (t, layer), values in sorted(channels.items()):
        require(layer == 0 if t in (0, 8) else layer < layers, "a ChannelParams' layer: 0 for Embed and LmHead")
        require(len(values) == widths[t], "one value per output channel")
        for c, v in enumerate(values):
            live[(t * l_pad + layer) * c_width + c] = v
    committed = sorted({f // k for f in live})
    rows = [[live.get(r * k + j, 0) % P for j in range(k)] for r in committed]
    layout = {"channel_width": c_width, "log2_k": log2(k), "row_vars": log2(total // k)}
    return layout, committed, rows


# ----- RFC-0011 'Encoding' at rate 1/4 -----------------------------------------------------------


POINTS = {}
CHECKED = set()


def points(n, shift):
    """shift w_n^j for j < n, each power on its own (the encoding oracle's points), kept."""
    if (n, shift) not in POINTS:
        POINTS[n, shift] = encoding.points(n, shift)
    return POINTS[n, shift]


def check_codeword(row, f, codeword, k, n):
    """The definition, by Horner's rule: f takes the row's values on H, and symbol j is f(7 w_n^j).
    Every position for the first row of each size, eight of each for the others."""
    full = (k, n) not in CHECKED
    CHECKED.add((k, n))
    on_h = range(k) if full else sorted({0, k - 1, *range(0, k, max(1, k // 6))})
    on_coset = range(n) if full else sorted({0, n - 1, *range(0, n, max(1, n // 6))})
    require(all(encoding.horner(f, points(k, 1)[i]) == row[i] for i in on_h), "f takes the row's values on H")
    require(all(encoding.horner(f, points(n, 7)[j]) == codeword[j] for j in on_coset), "symbol j is f(7 w_n^j)")


TWISTS = {}


def twists(k, n):
    """For s < n/k, the powers c_s^i, i < k, of c_s = 7 w_n^s."""
    if (k, n) not in TWISTS:
        w_n, table = encoding.root(n), []
        for s in range(n // k):
            c, power, powers = 7 * pow(w_n, s, P) % P, 1, []
            for _ in range(k):
                powers.append(power)
                power = power * c % P
            table.append(powers)
        TWISTS[k, n] = table
    return TWISTS[k, n]


def codewords(rows, k, n):
    """Each row's codeword: the polynomial f of degree below k taking the row's values on H, at
    7 w_n^j for j < n. No mask: tessera-protocol/1 masks no weight row (RFC-0040).

    f comes from an inverse NTT. With j = (n/k) q + s, the point 7 w_n^j is c_s w_k^q, since
    w_n^(n/k) = w_k: the codeword interleaves f's values on the n/k cosets c_s H, each the NTT of
    size k of the coefficients f_i c_s^i. The first and last rows are checked against the
    definition."""
    w_k, ratio = encoding.root(k), n // k
    table = twists(k, n)
    out = []
    for index, row in enumerate(rows):
        f = pcs.intt(row, w_k)
        cosets = [pcs.ntt([a * b % P for a, b in zip(f, powers)], w_k) for powers in table]
        out.append([cosets[j % ratio][j // ratio] for j in range(n)])
        if index in (0, len(rows) - 1):
            check_codeword(row, f, out[-1], k, n)
    return out


# ----- BLAKE3 over many inputs at once -----------------------------------------------------------
#
# One BLAKE3 compression (the BLAKE3 specification, section 2.2) over many single-block inputs in
# derive_key mode: a 32-bit word of the state is a Python integer holding one lane per input,
# lane l in bits 64 l to 64 l + 31. A sum of two or three words stays below 2^34, so masking each
# lane back to 32 bits after every addition keeps the lanes apart, and a rotation shifts within
# the lane's 64 bits before the mask.


class Lanes:
    def __init__(self, count):
        self.count = count
        self.ones = int.from_bytes((b"\x01" + b"\x00" * 7) * count, "little")
        self.mask = 0xFFFFFFFF * self.ones

    def pack(self, values):
        """The lanes `values`, each below 2^32."""
        return int.from_bytes(struct.pack(f"<{self.count}Q", *values), "little")

    def unpack(self, word):
        return struct.unpack(f"<{self.count}Q", word.to_bytes(8 * self.count, "little"))

    def derive_key(self, key, message, block_len):
        """BLAKE3 derive_key of single-block inputs: `key` is the context key's 8 words, the same
        for every lane; `message` the 16 packed words of the zero-padded block; every lane's input
        is `block_len` bytes. Returns the 8 packed words of the 32-byte outputs."""
        require(1 <= block_len <= 64 and len(message) == 16, "one block")
        mask, ones = self.mask, self.ones
        flags = hashing.CHUNK_START | hashing.CHUNK_END | hashing.ROOT | hashing.DERIVE_KEY_MATERIAL
        s = [w * ones for w in key] + [w * ones for w in hashing.IV[:4]] + [0, 0, block_len * ones, flags * ones]
        m = list(message)

        def g(a, b, c, d, mx, my):
            s[a] = (s[a] + s[b] + mx) & mask
            x = s[d] ^ s[a]
            s[d] = ((x >> 16) | (x << 16)) & mask
            s[c] = (s[c] + s[d]) & mask
            x = s[b] ^ s[c]
            s[b] = ((x >> 12) | (x << 20)) & mask
            s[a] = (s[a] + s[b] + my) & mask
            x = s[d] ^ s[a]
            s[d] = ((x >> 8) | (x << 24)) & mask
            s[c] = (s[c] + s[d]) & mask
            x = s[b] ^ s[c]
            s[b] = ((x >> 7) | (x << 25)) & mask

        for r in range(7):
            g(0, 4, 8, 12, m[0], m[1])
            g(1, 5, 9, 13, m[2], m[3])
            g(2, 6, 10, 14, m[4], m[5])
            g(3, 7, 11, 15, m[6], m[7])
            g(0, 5, 10, 15, m[8], m[9])
            g(1, 6, 11, 12, m[10], m[11])
            g(2, 7, 8, 13, m[12], m[13])
            g(3, 4, 9, 14, m[14], m[15])
            if r < 6:
                m = [m[i] for i in hashing.MSG_PERMUTATION]
        return [s[i] ^ s[i + 8] for i in range(8)]

    def digests(self, words):
        """The 32-byte digest of every lane, from 8 packed words."""
        columns = [self.unpack(w) for w in words]
        return [struct.pack("<8I", *(column[lane] for column in columns)) for lane in range(self.count)]


def context_key(domain):
    """The 8 words of BLAKE3's context key for tessera/v1/<domain>."""
    require(domain in hashing.DOMAINS, f"a domain: {domain}")
    key = hashing.blake3((hashing.PREFIX + domain).encode("ascii"), hashing.IV, hashing.DERIVE_KEY_CONTEXT)
    return hashing.key_words(key)


KEYS = {d: context_key(d) for d in ("column-init", "column-step", "merkle-node")}


def zero_words(count):
    return [0] * count


def column_leaves(commitment_id, codewords_, n):
    """leaf_j = d_j[ceil(m / B)] for every column j < n (RFC-0004 'Column digests'): one chain over
    the committed rows, no segments and no column-merge (RFC-0040 'Encoding, leaves and roots')."""
    lanes = Lanes(n)
    id_words = struct.unpack("<8I", commitment_id)
    # d_j[0] = H("column-init", commitment_id ‖ u64(j)): 40 bytes.
    message = [w * lanes.ones for w in id_words] + [lanes.pack(range(n)), 0] + zero_words(6)
    chain = lanes.derive_key(KEYS["column-init"], message, 40)
    for start in range(0, len(codewords_), ROW_BLOCK):
        block = codewords_[start:start + ROW_BLOCK]
        symbols = []
        for codeword in block:
            symbols.append(lanes.pack([x & 0xFFFFFFFF for x in codeword]))
            symbols.append(lanes.pack([x >> 32 for x in codeword]))
        message = chain + symbols + zero_words(8 - len(symbols))
        chain = lanes.derive_key(KEYS["column-step"], message, 32 + 8 * len(block))
    return lanes.digests(chain)


def merkle_root(leaves):
    """The root of the binary tree over the leaves (RFC-0004 'Merkle trees'), a level at a time."""
    level = leaves
    while len(level) > 1:
        lanes = Lanes(len(level) // 2)
        left = [struct.unpack("<8I", level[2 * p]) for p in range(lanes.count)]
        right = [struct.unpack("<8I", level[2 * p + 1]) for p in range(lanes.count)]
        message = [lanes.pack([w[i] for w in left]) for i in range(8)]
        message += [lanes.pack([w[i] for w in right]) for i in range(8)]
        level = lanes.digests(lanes.derive_key(KEYS["merkle-node"], message, 64))
    return level[0]


def check_lanes(rng):
    """The lanes give the hash oracle's H on random inputs of every length used here."""
    for domain, length in (("column-init", 40), ("column-step", 64), ("column-step", 48), ("merkle-node", 64)):
        count = 37
        inputs = [bytes(next(rng) & 0xFF for _ in range(length)) for _ in range(count)]
        lanes = Lanes(count)
        padded = [x + b"\x00" * (64 - length) for x in inputs]
        message = [lanes.pack([int.from_bytes(x[4 * i:4 * i + 4], "little") for x in padded]) for i in range(16)]
        got = lanes.digests(lanes.derive_key(KEYS[domain], message, length))
        require(got == [H(domain, x) for x in inputs], f"the lanes hash {domain} as H does")


def check_chain(commitment_id, codewords_, leaves, j):
    """Column j's chain, recomputed with the hash oracle's H, gives leaf_j."""
    d = H("column-init", commitment_id + j.to_bytes(8, "little"))
    for start in range(0, len(codewords_), ROW_BLOCK):
        symbols = b"".join(cw[j].to_bytes(8, "little") for cw in codewords_[start:start + ROW_BLOCK])
        d = H("column-step", d + symbols)
    require(d == leaves[j], f"column {j}'s chain")


# ----- RFC-0031, RFC-0040: commitment_id, roots, R_W, the artifact --------------------------------


def tce_layout(layout):
    return (layout["log2_k"].to_bytes(1, "little") + layout["log2_inv_rate"].to_bytes(1, "little")
            + layout["mask_coeffs"].to_bytes(4, "little") + layout["blinding_rows"].to_bytes(4, "little"))


def commit(name, matrix, layout, committed, rows):
    """The matrix's commitment_id, codewords, leaves and root."""
    k = 1 << layout["log2_k"]
    n = k << LAYOUT["log2_inv_rate"]
    descriptor = {"kind": "WeightMatrix", "layout": LAYOUT, "matrix": matrix, "log2_k": layout["log2_k"],
                  "row_block": ROW_BLOCK, "rows": len(committed), "epoch": EPOCH}
    tce = hashing.tce_descriptor(descriptor)
    commitment_id = H("commitment-id", tce)
    words = codewords(rows, k, n)
    leaves = column_leaves(commitment_id, words, n)
    check_chain(commitment_id, words, leaves, (MATRICES.index(matrix) * 97) % n)
    root = merkle_root(leaves)
    return {"descriptor": tce.hex(), "commitment_id": commitment_id.hex(), "root": root.hex()}, leaves


def public_artifact(roots):
    """TCE(WeightCommitmentPublic): the layout, the ten roots in WeightMatrix order, the epoch; and
    its envelope (RFC-0003 'Envelope', schema 0x0301 version 1)."""
    require(len(roots) == 10, "ten roots")
    payload = tce_layout(LAYOUT) + b"".join(roots) + EPOCH.to_bytes(4, "little")
    header = (b"TSRA" + (1).to_bytes(2, "little") + SCHEMA_ID.to_bytes(2, "little")
              + SCHEMA_VERSION.to_bytes(2, "little") + (0).to_bytes(2, "little") + len(payload).to_bytes(8, "little"))
    return payload, header + payload + H("envelope", header + payload)


def commitment(name, arch, tensors, channels):
    """Every matrix of one model, then R_W and the .twr artifact. `tensors` maps (type, layer) to
    W row-major, `channels` (type index, layer) to its class indices or multipliers."""
    shapes = shapes_of(arch)
    k_w = 1 << LAYOUT["log2_k"]
    matrices, roots, check_tree = [], [], None
    for ty in WEIGHT_TYPES:
        out, inputs, layers = shapes[ty]
        per_layer = [tensors[ty, layer] for layer in range(layers)]
        for data in per_layer:
            require(len(data) == out * inputs, f"{name}: {ty} holds out x in entries")
        layout, committed, rows = weight_matrix(out, inputs, layers, k_w, per_layer)
        result, leaves = commit(name, ty, layout, committed, rows)
        matrices.append({"matrix": ty, **layout, "rows": len(committed), "committed_rows": committed, **result})
        roots.append(bytes.fromhex(result["root"]))
        if check_tree is None or len(leaves) < len(check_tree):
            check_tree = leaves
    widths = {WEIGHT_TYPES.index(ty): shapes[ty][0] for ty in WEIGHT_TYPES}
    layout, committed, rows = channel_matrix(channels, widths, arch["n_layers"], k_w)
    result, leaves = commit(name, "ChannelParams", layout, committed, rows)
    matrices.append({"matrix": "ChannelParams", **layout, "rows": len(committed), "committed_rows": committed,
                     **result})
    roots.append(bytes.fromhex(result["root"]))
    # The tree, once per model, from the hash oracle's Merkle levels.
    require(merkle_root(check_tree) == hashing.merkle_levels(check_tree)[-1][0], f"{name}: a Merkle root")
    payload, artifact = public_artifact(roots)
    return matrices, payload, artifact


def conformance(name, index):
    """One conformance model's commitment, with where each weight and channel parameter lies in
    its artifact."""
    value, data = read_model(name, index)
    arch = value["arch"]
    tensors, located = {}, []
    for w in value["weights"]:
        ty = next(iter(w["ty"]))
        require((ty, w["layer"]) not in tensors, f"{name}: one tensor per (type, layer)")
        out, inputs, _ = shapes_of(arch)[ty]
        require((w["rows"], w["cols"]) == (out, inputs), f"{name}: {ty}'s shape follows the architecture")
        tensors[ty, w["layer"]] = w["data"]
        located.append({"matrix": ty, "layer": w["layer"], "rows": w["rows"], "cols": w["cols"],
                        "offset": w["data"].offset})
    channels, located_channels = {}, []
    for ch in value["quant"]["channels"]:
        ty = next(iter(ch["ty"]))
        require((len(ch["class"]) == 0) != (len(ch["multiplier"]) == 0), f"{name}: classes or multipliers")
        kind = "class" if ch["class"] else "multiplier"
        values = ch[kind]
        channels[WEIGHT_TYPES.index(ty), ch["layer"]] = values
        located_channels.append({"matrix": ty, "layer": ch["layer"], "kind": kind, "count": len(values),
                                 "offset": values.offset})
    matrices, payload, artifact = commitment(name, arch, tensors, channels)
    (HERE / f"{name}.twr").write_bytes(artifact)
    return {
        "name": name,
        "model": f"../models/{name}.d",
        "model_digest": data[-32:].hex(),
        "arch": {k: arch[k] for k in ("n_layers", "d_model", "n_heads", "n_kv_heads", "d_head", "d_ff", "vocab")},
        "tensors": located,
        "channels": located_channels,
        "matrices": matrices,
        "public": payload.hex(),
        "weights_root": H("weights-root", payload).hex(),
        "file": f"{name}.twr",
        "bytes": len(artifact),
        "digest": artifact[-32:].hex(),
    }


# A model with no ChannelParams, so C = 1 and the ChannelParams matrix commits no row; and an
# embedding wider than k_W, non-power-of-two widths and layers, and partly padded rows.
SYNTHETIC_ARCH = {"n_layers": 3, "d_model": 12, "n_heads": 2, "n_kv_heads": 1, "d_head": 6, "d_ff": 20,
                  "vocab": 300}


def synthetic(rng):
    """The synthetic model's weights, uniform in [-127, 127], and its commitment."""
    shapes = shapes_of(SYNTHETIC_ARCH)
    tensors, listed = {}, []
    for ty in WEIGHT_TYPES:
        out, inputs, layers = shapes[ty]
        for layer in range(layers):
            data = [next(rng) % 255 - 127 for _ in range(out * inputs)]
            if not listed:
                data[0], data[-1] = -127, 127  # the ends of INT8's centered range
            tensors[ty, layer] = data
            listed.append({"matrix": ty, "layer": layer, "rows": out, "cols": inputs, "data": data})
    matrices, payload, artifact = commitment("synthetic", SYNTHETIC_ARCH, tensors, {})
    (HERE / "synthetic.twr").write_bytes(artifact)
    return {
        "name": "synthetic",
        "arch": SYNTHETIC_ARCH,
        "tensors": listed,
        "channels": [],
        "matrices": matrices,
        "public": payload.hex(),
        "weights_root": H("weights-root", payload).hex(),
        "file": "synthetic.twr",
        "bytes": len(artifact),
        "digest": artifact[-32:].hex(),
    }


# ----- Rows encoded at rate 1/4, from the definition ---------------------------------------------


def encoding_cases(rng):
    """Rows of k values encoded into n = 4k symbols: g = f + (X^k - 1) m on 7.L, every symbol by
    Horner's rule. P = 0 is a weight row of tessera-protocol/1; the masked cases pin where a mask
    goes at this rate."""
    fp = encoding.Inputs(next(rng))
    cases = []
    shapes = [(1 << e, 0) for e in range(7)] + [(4, 1), (4, 2), (4, 4), (16, 1), (16, 8), (16, 16)]
    for k, p in shapes:
        n = 4 * k
        f, m = fp.fps(k), fp.fps(p)
        g = encoding.masked(f, m, k)
        require(all(c == 0 for c in g[k + p:]), "deg g < k + P")
        row = encoding.evaluate_at(f, encoding.points(k, 1))
        require(encoding.evaluate_at(g, encoding.points(k, 1)) == row, "the mask vanishes on H")
        cases.append({"k": k, "mask_coeffs": p, "row": [hex64(x) for x in row], "mask": [hex64(x) for x in m],
                      "codeword": [hex64(x) for x in encoding.evaluate_at(g, encoding.points(n, 7))]})
    return cases


# ----- Output ------------------------------------------------------------------------------------


def hex64(x):
    return f"0x{x:016x}"


def compact(value, indent=0):
    """JSON with every innermost list of numbers or strings on one line."""
    pad_ = " " * indent
    if isinstance(value, list) and all(isinstance(x, (int, str)) for x in value):
        return json.dumps(value, separators=(",", ":"))
    if isinstance(value, list):
        inner = ",\n".join(pad_ + " " + compact(x, indent + 1) for x in value)
        return "[\n" + inner + "\n" + pad_ + "]"
    if isinstance(value, dict):
        inner = ",\n".join(f"{pad_} {json.dumps(k)}: {compact(v, indent + 1)}" for k, v in value.items())
        return "{\n" + inner + "\n" + pad_ + "}"
    return json.dumps(value)


def main():
    checked = hashing.check_official_vectors()
    rng = hashing.splitmix64(SEED)
    check_lanes(rng)
    cases = encoding_cases(rng)
    index = json.loads((HERE.parent / "models" / "models.json").read_text(encoding="utf-8"))
    require([m["name"] for m in index["models"]] == MODEL_NAMES, "every conformance model, in order")
    entries = [conformance(name, index) for name in MODEL_NAMES]
    entries.append(synthetic(rng))
    document = {
        "description": "The weight commitment of RFC-0040 at test-small/1: for each model, each matrix's layout, "
        "committed rows, commitment_id and root, then R_W and its WeightCommitmentPublic artifact; and rows encoded at "
        "rate 1/4. Written by generate.py; see README.md.",
        "params": PARAMS,
        "layout": LAYOUT,
        "row_block": ROW_BLOCK,
        "epoch": EPOCH,
        "schema": f"0x{SCHEMA_ID:04x}",
        "version": str(SCHEMA_VERSION),
        "seed": hex64(SEED),
        "official_blake3_cases_checked": checked,
        "encoding": cases,
        "models": entries,
    }
    text = compact(document) + "\n"
    json.loads(text)  # the file must parse
    (HERE / "weights.json").write_text(text, encoding="utf-8")
    print(f"wrote weights.json: {len(entries)} models, {len(cases)} encodings, {len(text)} bytes")


if __name__ == "__main__":
    main()
