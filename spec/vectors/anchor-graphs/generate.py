#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0 OR MIT
"""RFC-0066's SHA-256 circuit, node by node, and its public inputs from bytes.

This script is the independent oracle. It was written from RFC-0066 'The SHA-256 circuit',
RFC-0072, RFC-0006 'Operation set', RFC-0044's notation and FIPS 180-4 alone, by an author who
read neither crates/ nor verifier/crates/. It writes:

- sha256-graphs.json: the circuit's graph for five parameter sets, in the lowering vectors'
  format (spec/vectors/lowering/README.md), with the tensors RFC-0072 makes public and the
  per-block counts of its sizes table;
- sha256-inputs.json: the circuit's public inputs (the message's bits with its committed bytes
  zeroed, the place mask, the initial value, the SHR masks, the digest's bits) and its committed
  bytes, for FIPS 180-4's examples, NIST CAVP's messages, and the anchor's key commitments and
  frames of spec/vectors/anchor/frames.json.

Before writing, it checks: its own SHA-256, constants derived from their definitions, against
hashlib, CAVP and FIPS 180-4's examples; that the public bits and the placed committed bits
make up the padded message; that each graph keeps RFC-0006's shape rules; that an interpreter
of the graph's operations, its Boolean functions as logic, gives FIPS 180-4's chaining words
on every input set, on the file's graph of its parameters or on one built for it, with every
BitDecompose input inside its width and every AssertEqual equal; and that the counts of public tensors, selections, BitFn and Mul
nodes and zero-checks are RFC-0072's sizes table.

Usage: python3 -I spec/vectors/anchor-graphs/generate.py
"""

import hashlib
import itertools
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
VECTORS = HERE.parent
M32 = (1 << 32) - 1


def require(condition, message):
    """A check that holds under `python -O` too, unlike `assert`."""
    if not condition:
        raise SystemExit(f"anchor-graphs/generate.py: {message}")


# ----- FIPS 180-4 -------------------------------------------------------------------------------


def primes(count):
    out, n = [], 2
    while len(out) < count:
        if all(n % p for p in out if p * p <= n):
            out.append(n)
        n += 1
    return out


def icbrt(n):
    """floor(n^(1/3)) by Newton's method from above."""
    x = 1 << ((n.bit_length() + 2) // 3)
    while True:
        y = (2 * x + n // (x * x)) // 3
        if y >= x:
            break
        x = y
    while x**3 > n:
        x -= 1
    while (x + 1) ** 3 <= n:
        x += 1
    return x


def isqrt(n):
    x = 1 << ((n.bit_length() + 1) // 2)
    while True:
        y = (x + n // x) // 2
        if y >= x:
            break
        x = y
    while x * x > n:
        x -= 1
    return x


# Section 4.2.2: the first 32 bits of the fractional parts of the cube roots of the first 64
# primes; section 5.3.3: of the square roots of the first 8.
K = [icbrt(p << 96) & M32 for p in primes(64)]
H0 = [isqrt(p << 64) & M32 for p in primes(8)]
require(K[0] == 0x428A2F98 and K[63] == 0xC67178F2, "the round constants' ends")
require(H0[0] == 0x6A09E667 and H0[7] == 0x5BE0CD19, "the initial value's ends")


def rotr(x, r):
    return ((x >> r) | (x << (32 - r))) & M32


def big_sigma0(x):
    return rotr(x, 2) ^ rotr(x, 13) ^ rotr(x, 22)


def big_sigma1(x):
    return rotr(x, 6) ^ rotr(x, 11) ^ rotr(x, 25)


def small_sigma0(x):
    return rotr(x, 7) ^ rotr(x, 18) ^ (x >> 3)


def small_sigma1(x):
    return rotr(x, 17) ^ rotr(x, 19) ^ (x >> 10)


def pad(message):
    """Section 5.1.1: the message, a 1 bit, zeros, and its length in bits as 64 bits."""
    zeros = (55 - len(message)) % 64
    return message + b"\x80" + bytes(zeros) + (8 * len(message)).to_bytes(8, "big")


def chaining(message):
    """Section 6.2: the hash value H^(i) after every block of the padded message."""
    data, state, out = pad(message), list(H0), []
    for at in range(0, len(data), 64):
        w = [int.from_bytes(data[at + 4 * t:at + 4 * t + 4], "big") for t in range(16)]
        for t in range(16, 64):
            w.append((small_sigma1(w[t - 2]) + w[t - 7] + small_sigma0(w[t - 15]) + w[t - 16]) & M32)
        a, b, c, d, e, f, g, h = state
        for t in range(64):
            t1 = (h + big_sigma1(e) + ((e & f) ^ (~e & g)) + K[t] + w[t]) & M32
            t2 = (big_sigma0(a) + ((a & b) ^ (a & c) ^ (b & c))) & M32
            h, g, f, e, d, c, b, a = g, f, e, (d + t1) & M32, c, b, a, (t1 + t2) & M32
        state = [(x + y) & M32 for x, y in zip(state, [a, b, c, d, e, f, g, h], strict=True)]
        out.append(state)
    return out


def sha256(message):
    digest = b"".join(w.to_bytes(4, "big") for w in chaining(message)[-1])
    require(digest == hashlib.sha256(message).digest(), "the oracle's SHA-256 is hashlib's")
    return digest


def cavp(name):
    """(message, digest) of each case of a CAVP response file."""
    cases, length, msg = [], None, None
    for line in (VECTORS / "sha256" / name).read_text().splitlines():
        key, _, value = (part.strip() for part in line.partition("="))
        if key == "Len":
            length = int(value)
        elif key == "Msg":
            msg = bytes.fromhex(value)[:length // 8]
        elif key == "MD":
            cases.append((msg, bytes.fromhex(value)))
    return cases


def check_sha256():
    """FIPS 180-4's examples (its Appendix B, as NIST's example document states them) and every
    CAVP message the vectors use."""
    examples = [
        (b"abc", "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad"),
        (b"abcdbcdecdefdefgefghfghighijhijkijkljklmklmnlmnomnopnopq",
         "248d6a61d20638b8e5c026930c3e6039a33ce45964ff2167f6ecedd419db06c1"),
    ]
    for message, digest in examples:
        require(sha256(message).hex() == digest, f"FIPS 180-4's example {message[:8]!r}")
    for name in ("SHA256ShortMsg.rsp", "SHA256LongMsg.rsp"):
        cases = cavp(name)
        require(len(cases) == (65 if "Short" in name else 64), f"{name}'s case count")
        for message, digest in cases[:70 if "Short" in name else 3]:
            require(sha256(message) == digest, f"{name}, a message of {len(message)} bytes")
    return examples


# ----- The graph ------------------------------------------------------------------------------

GADGETS = {"OneHot", "Requant", "TableMap", "SoftmaxRow", "RsqrtRow", "Bitwise", "BitDecompose",
           "Argmax", "InverseCdf"}  # gadget-backed (RFC-0072 'What is public')
SELECTIONS = {"GatherPublic", "Slice"}


def size(shape):
    out = 1
    for n in shape:
        out *= n
    return out


class Graph:
    """A graph under construction: ids dense in creation order (RFC-0044 'Ids'), each node
    checked against RFC-0006 'Operation set' as it is created."""

    def __init__(self):
        self.shapes, self.kinds = [], []
        self.inputs, self.nodes, self.constants = [], [], []
        self.block = None
        self.node_blocks = []

    def tensor(self, shape, kind):
        require(len(shape) >= 1 and all(n >= 1 for n in shape), "shapes have axes of length >= 1")
        self.shapes.append(list(shape))
        self.kinds.append(kind)
        return len(self.shapes) - 1

    def input(self, role, shape, kind, ty, lo, hi):
        t = self.tensor(shape, kind)
        self.inputs.append([t, role, list(shape), kind, ty, lo, hi])
        return t

    def constant(self, values):
        """A constant of its own, declared just before its node (RFC-0066 'The graph')."""
        self.constants.append(list(values))
        return len(self.constants) - 1

    def node(self, op, params, inputs, outputs, at):
        for t in inputs:
            require(0 <= t < len(self.shapes), "inputs exist before their readers")
        outs = [self.tensor(shape, kind) for shape, kind in outputs]
        self.nodes.append([len(self.nodes), op, params, list(inputs),
                           [[t, self.shapes[t], self.kinds[t]] for t in outs], at])
        self.node_blocks.append(self.block)
        return outs

    def shape(self, t):
        return self.shapes[t]

    # The operations, with RFC-0006's shape rules.

    def gather(self, x, index, at):
        xs = self.shape(x)
        require(len(index) >= 1 and all(0 <= i < xs[0] for i in index), "each index a row of x's outermost axis")
        c = self.constant(index)
        return self.node("GatherPublic", {"index": c}, [x], [([len(index)] + xs[1:], "v")], at)[0]

    def slice(self, x, axis, start, length, at):
        xs = self.shape(x)
        require(length >= 1 and start + length <= xs[axis], "start + len within the axis, len >= 1")
        ys = list(xs)
        ys[axis] = length
        return self.node("Slice", {"axis": axis, "start": start, "len": length}, [x], [(ys, "v")], at)[0]

    def reshape(self, x, shape, at):
        require(size(shape) == size(self.shape(x)), "a reshape keeps the element count")
        return self.node("Reshape", {}, [x], [(shape, "v")], at)[0]

    def transpose(self, x, perm, at):
        xs = self.shape(x)
        require(sorted(perm) == list(range(len(xs))), "perm is a permutation of x's axes")
        return self.node("Transpose", {"perm": perm}, [x], [([xs[p] for p in perm], "v")], at)[0]

    def add(self, a, b, at):
        require(self.shape(a) == self.shape(b), "Add's shapes are equal")
        return self.node("Add", {}, [a, b], [(self.shape(a), "v")], at)[0]

    def mul_trailing(self, a, b, at):
        sa, sb = self.shape(a), self.shape(b)
        require(sa[:len(sb)] == sb, "a Trailing operand's shape is a prefix of a's")
        return self.node("Mul", {"broadcast": "Trailing"}, [a, b], [(sa, "v")], at)[0]

    def const_mul(self, x, values, at):
        require(len(values) == 1 or len(values) == self.shape(x)[-1], "one value or one per channel")
        c = self.constant(values)
        return self.node("ConstMul", {"c": c}, [x], [(self.shape(x), "v")], at)[0]

    def const_add(self, x, values, at):
        require(len(values) == 1 or len(values) == self.shape(x)[-1], "one value or one per channel")
        c = self.constant(values)
        return self.node("ConstAdd", {"c": c}, [x], [(self.shape(x), "v")], at)[0]

    def bit_decompose(self, x, low, high, at):
        require(1 <= low and low + high <= 62, "1 <= low, low + high <= 62")
        xs = self.shape(x)
        outputs = [(xs + [low], "A")] + ([(xs + [high], "A")] if high else [])
        return self.node("BitDecompose", {"low": low, "high": high}, [x], outputs, at)

    def bitfn(self, f, inputs, at):
        require(len(inputs) == (2 if f in ("Xor", "And") else 3), f"{f}'s arity")
        require(all(self.shape(t) == self.shape(inputs[0]) for t in inputs), "BitFn's shapes are equal")
        return self.node("BitFn", {"f": f}, inputs, [(self.shape(inputs[0]), "v")], at)[0]

    def assert_equal(self, a, b, at):
        require(self.shape(a) == self.shape(b), "AssertEqual's shapes are equal")
        self.node("AssertEqual", {}, [a, b], [], at)


def rotation(r):
    """ROTR^r as a gather: row i taken from row (i + r) mod 32 (RFC-0066 'Layout')."""
    return [(i + r) % 32 for i in range(32)]


def assembly(n_blocks, offsets):
    """The message's gather (RFC-0066 'The message'): committed byte k's bit t, row 8k + t of
    the decomposed bytes, goes to its word position, and row 0 to every other row. Bit i of word
    j is bit i mod 8 of byte 4j + 3 - floor(i/8), so byte o's bit t is row
    32 (o // 4) + 8 (3 - o % 4) + t."""
    idx = [0] * (512 * n_blocks)
    for k, o in enumerate(offsets):
        for t in range(8):
            idx[32 * (o // 4) + 8 * (3 - o % 4) + t] = 8 * k + t
    return idx


SMALL_SIGMA = {0: ([7, 18], 3), 1: ([17, 19], 10)}  # FIPS 180-4 (4.6), (4.7)
BIG_SIGMA = {0: [2, 13, 22], 1: [6, 11, 25]}  # (4.4), (4.5)


def sha256_graph(n_blocks, m, offsets):
    """RFC-0066 'The graph', for M messages of N blocks with K = len(offsets) committed bytes."""
    g = Graph()
    k = len(offsets)
    message = g.input("message", [512 * n_blocks, m], "P", "u1", 0, 1)
    start = g.input("start", [256, m], "P", "u1", 0, 1)
    masks = g.input("masks", [2, 32], "P", "u1", 0, 1)
    if k:
        committed = g.input("committed", [k, m], "A", "u8", 0, 255)
        place = g.input("place", [512 * n_blocks], "P", "u1", 0, 1)

    # 1. Each SHR mask: a gather of its row of `masks`, SHR^3 row 0 and SHR^10 row 1, then a
    #    Reshape to [32].
    mask = {}
    for row, r in ((0, 3), (1, 10)):
        t = g.gather(masks, [row], f"SHR^{r} mask: row {row} of masks")
        mask[r] = g.reshape(t, [32], f"SHR^{r} mask")

    # 2. The message's bits.
    if k:
        b = g.bit_decompose(committed, 8, 0, "message: the committed bytes' bits")[0]
        t = g.transpose(b, [0, 2, 1], "message: bits to [K, 8, M]")
        t = g.reshape(t, [8 * k, m], "message: bits to [8K, M]")
        t = g.gather(t, assembly(n_blocks, offsets), "message: the committed bits at their rows")
        t = g.mul_trailing(t, place, "message: times place")
        bits = g.bitfn("Xor", [message, t], "message: Xor of the public and committed bits")
    else:
        bits = message

    # 3. The eight initial words.
    state = [g.slice(start, 0, 32 * j, 32, f"initial word {'abcdefgh'[j]}") for j in range(8)]

    def function(x, rots, shr, at):
        """A function: its rotations' gathers, each SHR a gather and a Mul, in its definition's
        order, then its Xor3."""
        parts = [g.gather(x, rotation(r), f"{at}: ROTR^{r}") for r in rots]
        if shr is not None:
            t = g.gather(x, rotation(shr), f"{at}: SHR^{shr} gather")
            parts.append(g.mul_trailing(t, mask[shr], f"{at}: SHR^{shr} mask"))
        return g.bitfn("Xor3", parts, f"{at}: Xor3")

    def word_sum(terms, constant, carry, at):
        """A sum: its rows' Adds left to right, the value's five levels, its ConstAdd when the
        constant is not 0, the BitDecompose, then the Transpose of its result bits."""
        rows = terms[0]
        for term in terms[1:]:
            rows = g.add(rows, term, f"{at}: rows")
        value = rows
        for level in range(5):
            n = g.shape(value)[0]
            even = g.gather(value, list(range(0, n, 2)), f"{at}: level {level} even rows")
            odd = g.gather(value, list(range(1, n, 2)), f"{at}: level {level} odd rows")
            scaled = g.const_mul(odd, [2 ** (2**level)], f"{at}: level {level} 2^{2**level} odd")
            value = g.add(even, scaled, f"{at}: level {level} sum")
        value = g.reshape(value, [m], f"{at}: value")
        if constant:
            value = g.const_add(value, [constant], f"{at}: constant")
        out = g.bit_decompose(value, 32, carry, f"{at}: bits and carry")[0]
        return g.transpose(out, [1, 0], f"{at}: word")

    chain = []
    for blk in range(n_blocks):
        g.block = blk
        w = [g.slice(bits, 0, 512 * blk + 32 * t, 32, f"block {blk} W{t}") for t in range(16)]
        for t in range(16, 64):
            s1 = function(w[t - 2], *SMALL_SIGMA[1], f"block {blk} W{t} sigma1")
            s0 = function(w[t - 15], *SMALL_SIGMA[0], f"block {blk} W{t} sigma0")
            w.append(word_sum([s1, w[t - 7], s0, w[t - 16]], 0, 2, f"block {blk} W{t}"))
        a, b, c, d, e, f, h7, h = state
        for t in range(64):
            at = f"block {blk} round {t}"
            s1 = function(e, BIG_SIGMA[1], None, f"{at} Sigma1")
            ch = g.bitfn("Ch", [e, f, h7], f"{at} Ch")
            s0 = function(a, BIG_SIGMA[0], None, f"{at} Sigma0")
            maj = g.bitfn("Maj", [a, b, c], f"{at} Maj")
            shared = g.add(h, s1, f"{at}: shared rows")
            shared = g.add(shared, ch, f"{at}: shared rows")
            shared = g.add(shared, w[t], f"{at}: shared rows")
            new_e = word_sum([shared, d], K[t], 3, f"{at} e")
            new_a = word_sum([shared, s0, maj], K[t], 3, f"{at} a")
            h, h7, f, e, d, c, b, a = h7, f, e, new_e, c, b, a, new_a
        state = [word_sum([state[j], v], 0, 1, f"block {blk} chaining {'abcdefgh'[j]}")
                 for j, v in enumerate([a, b, c, d, e, f, h7, h])]
        chain.append(state)
    g.block = None

    # 5. The digest, declared after the circuit's nodes, then per word its Slice and the
    #    AssertEqual of the computed word and the digest's rows.
    digest = g.input("digest", [256, m], "P", "u1", 0, 1)
    for j in range(8):
        rows = g.slice(digest, 0, 32 * j, 32, f"digest word {j}")
        g.assert_equal(state[j], rows, f"digest word {j}: binding")
    return g, chain


# ----- What RFC-0072 counts ----------------------------------------------------------------


def public_tensors(g):
    """RFC-0072 'What is public': a public graph input; the output of a node that is not
    gadget-backed whose inputs are all public, unless declared committed."""
    public = {t for t, _, _, kind, *_ in g.inputs if kind == "P"}
    for _, op, _, ins, outs, _ in g.nodes:
        if op not in GADGETS and all(t in public for t in ins):
            public.update(t for t, _, kind in outs if kind == "v")
    return public


def padded_positions(shape):
    return any(n & (n - 1) for n in shape)


def moved_input(g, t, producers):
    """The committed graph input t is, directly or moved by Reshape, Slice, Transpose,
    GatherPublic or Expand (RFC-0066 'A committed input's padding'), or None."""
    committed = {i[0] for i in g.inputs if i[3] == "A"}
    while t not in committed:
        node = producers.get(t)
        if node is None or node[1] not in ("Reshape", "Slice", "Transpose", "GatherPublic", "Expand"):
            return None
        t = node[3][0]
    return t


def counts(g):
    """Per block (None for the graph's own nodes) and in all: RFC-0072's sizes-table columns
    that the graph decides, the zero-checks in RFC-0072's table order."""
    public = public_tensors(g)
    producers = {out[0]: node for node in g.nodes for out in node[4]}
    by_block = {}
    for node, blk in zip(g.nodes, g.node_blocks, strict=True):
        _, op, params, ins, outs, _ = node
        row = by_block.setdefault(blk, {"zero_checks": 0, "input_padding": 0, "booleanity": 0,
                                        "bit_padding": 0, "recomposition": 0, "assert_equal": 0,
                                        "selections": 0, "bitfn": 0, "mul": 0, "nodes": 0})
        row["nodes"] += 1
        hidden = all(t in public for t, _, _ in outs) and outs
        if op in SELECTIONS and not hidden:
            row["selections"] += 1
        elif op == "BitFn" and not hidden:
            row["bitfn"] += 1
        elif op == "Mul" and not hidden:
            row["mul"] += 1
        elif op == "BitDecompose":
            u = moved_input(g, ins[0], producers)
            if u is not None and padded_positions(g.shape(u)):
                row["input_padding"] += 1
            row["booleanity"] += len(outs)
            row["bit_padding"] += sum(1 for n in (params["low"], params["high"]) if n and n & (n - 1))
            row["recomposition"] += 1
        elif op == "AssertEqual":
            row["assert_equal"] += 1
    for row in by_block.values():
        row["zero_checks"] = sum(row[key] for key in ("input_padding", "booleanity", "bit_padding",
                                                      "recomposition", "assert_equal"))
    out = {"blocks": [by_block[b] for b in sorted(k for k in by_block if k is not None)],
           "graph": by_block.get(None)}
    out["total"] = {key: sum(row[key] for row in by_block.values()) for key in out["graph"]}
    return out, public


# RFC-0072 'The circuits' sizes': (zero-checks, selections, BitFn, Mul).
LATER_BLOCK_PUBLIC = (680, 2441, 335, 79)
LATER_BLOCK_COMMITTED = (680, 2528, 352, 96)
FIRST_BLOCK_PUBLIC = (680, 2415, 331, 79)
KEY_COMMITMENT = (690, 2523, 349, 97)


def figures(row):
    return (row["zero_checks"], row["selections"], row["bitfn"], row["mul"])


# ----- The interpreter -------------------------------------------------------------------------


def evaluate(g, values):
    """Runs the graph on true integers, its tensors flat and row-major at their real shapes.
    BitFn is logic, not its polynomial; a BitDecompose input outside its width, a BitFn input
    that is not a bit, or an AssertEqual of unequal tensors stops the script."""
    vals = dict(values)
    for nid, op, params, ins, outs, at in g.nodes:
        xs = [vals[t] for t in ins]
        shapes = [g.shape(t) for t in ins]
        if op == "GatherPublic":
            row = size(shapes[0][1:])
            y = [v for i in g.constants[params["index"]] for v in xs[0][i * row:(i + 1) * row]]
        elif op == "Slice":
            axis, start, length = params["axis"], params["start"], params["len"]
            outer, n, inner = size(shapes[0][:axis]), shapes[0][axis], size(shapes[0][axis + 1:])
            y = [xs[0][(o * n + start + j) * inner + i]
                 for o in range(outer) for j in range(length) for i in range(inner)]
        elif op == "Reshape":
            y = xs[0]
        elif op == "Transpose":
            perm, shape = params["perm"], shapes[0]
            strides = [size(shape[a + 1:]) for a in range(len(shape))]
            y = [xs[0][sum(i * strides[perm[a]] for a, i in enumerate(out))]
                 for out in itertools.product(*[range(shape[p]) for p in perm])]
        elif op == "Add":
            y = [a + b for a, b in zip(xs[0], xs[1], strict=True)]
        elif op == "Mul":
            inner = size(shapes[0][len(shapes[1]):])
            y = [a * xs[1][i // inner] for i, a in enumerate(xs[0])]
        elif op in ("ConstMul", "ConstAdd"):
            c, last = g.constants[params["c"]], shapes[0][-1]
            y = [(v * c[i % last if len(c) > 1 else 0]) if op == "ConstMul" else
                 (v + c[i % last if len(c) > 1 else 0]) for i, v in enumerate(xs[0])]
        elif op == "BitDecompose":
            low, high = params["low"], params["high"]
            require(all(0 <= v < 1 << (low + high) for v in xs[0]), f"node {nid} ({at}): input within its width")
            vals[outs[0][0]] = [(v >> i) & 1 for v in xs[0] for i in range(low)]
            if high:
                vals[outs[1][0]] = [(v >> (low + j)) & 1 for v in xs[0] for j in range(high)]
            continue
        elif op == "BitFn":
            require(all(v in (0, 1) for x in xs for v in x), f"node {nid} ({at}): BitFn reads bits")
            logic = {"Xor": lambda a, b: a ^ b, "Xor3": lambda a, b, c: a ^ b ^ c,
                     "And": lambda a, b: a & b, "Ch": lambda e, f, g_: (e & f) ^ ((1 - e) & g_),
                     "Maj": lambda a, b, c: (a & b) ^ (a & c) ^ (b & c)}[params["f"]]
            y = [logic(*v) for v in zip(*xs, strict=True)]
        elif op == "AssertEqual":
            require(xs[0] == xs[1], f"node {nid} ({at}): the computed word is the digest's")
            continue
        else:
            raise SystemExit(f"anchor-graphs/generate.py: no rule for {op}")
        require(len(y) == size(outs[0][1]), f"node {nid} ({at}): its output's size")
        vals[outs[0][0]] = y
    return vals


def word_of(bits_flat, m, column):
    """The integer of a [32, M] word tensor's column, row i of weight 2^i."""
    return sum(bits_flat[i * m + column] << i for i in range(32))


# ----- Public inputs from bytes (V4) -----------------------------------------------------------


def bits_string(rows):
    return "".join(str(v) for row in rows for v in row)


def word_bits(words, m_words):
    """[32 * len, M]: row 32 j + i is bit i of word j, one column per message."""
    return [[(words[col][j] >> i) & 1 for col in range(m_words)] for j in range(len(words[0])) for i in range(32)]


def public_inputs(messages, offsets):
    """RFC-0066 'The message' and 'The output': the circuit's inputs for M messages of one
    block count whose bytes at `offsets` are committed."""
    padded = [pad(msg) for msg in messages]
    n_blocks = len(padded[0]) // 64
    m = len(messages)
    require(all(len(p) == 64 * n_blocks for p in padded), "the messages have one block count")
    require(all(o < len(msg) for o in offsets for msg in messages), "committed bytes are message bytes")
    require(offsets == sorted(set(offsets)), "committed offsets in order, each once")
    committed = set(offsets)

    def byte_of(row):
        j, i = divmod(row, 32)
        return 4 * j + 3 - i // 8, i % 8

    message = []
    for row in range(512 * n_blocks):
        byte, t = byte_of(row)
        message.append([0 if byte in committed else (p[byte] >> t) & 1 for p in padded])
    place = [1 if byte_of(row)[0] in committed else 0 for row in range(512 * n_blocks)]
    digests = [sha256(msg) for msg in messages]
    out = {
        "message": message,
        "start": word_bits([H0] * m, m),
        "masks": [[1 if i < 32 - r else 0 for i in range(32)] for r in (3, 10)],
        "committed": [[p[o] for p in padded] for o in offsets],
        "place": place,
        "digest": word_bits([[int.from_bytes(d[4 * j:4 * j + 4], "big") for j in range(8)] for d in digests], m),
    }
    # The public bits and the placed committed bits make up the padded message (their supports
    # are disjoint, so Xor is the sum).
    gathered = assembly(n_blocks, offsets)
    for row in range(512 * n_blocks):
        byte, t = byte_of(row)
        for col, p in enumerate(padded):
            committed_bit = (out["committed"][gathered[row] // 8][col] >> (gathered[row] % 8)) & 1 if offsets else 0
            require(out["message"][row][col] ^ (committed_bit * place[row]) == (p[byte] >> t) & 1,
                    "the message's bits are its public and placed committed bits")
    return n_blocks, digests, out


def graph_inputs(g, inputs):
    """The interpreter's input values, by tensor id, flat."""
    flat = {"message": [v for row in inputs["message"] for v in row],
            "start": [v for row in inputs["start"] for v in row],
            "masks": [v for row in inputs["masks"] for v in row],
            "committed": [v for row in inputs["committed"] for v in row],
            "place": inputs["place"],
            "digest": [v for row in inputs["digest"] for v in row]}
    return {t: flat[role] for t, role, *_ in g.inputs}


# ----- Cases ---------------------------------------------------------------------------------

GRAPHS = [
    # (name, N, M, committed offsets)
    ("key commitment", 1, 1, list(range(21, 53))),
    ("three key commitments", 1, 3, list(range(21, 53))),
    ("one public block", 1, 1, []),
    ("three public messages", 1, 3, []),
    ("two blocks, committed bytes crossing into the second", 2, 2, list(range(41, 67))),
]


def input_cases(examples):
    """(name, source, messages, committed offsets, digests in hex): every input set of
    sha256-inputs.json."""
    frames = json.loads((VECTORS / "anchor" / "frames.json").read_text())["sessions"]
    short, long = cavp("SHA256ShortMsg.rsp"), cavp("SHA256LongMsg.rsp")
    cases = []
    for message, digest in examples:
        cases.append((f"FIPS 180-4 example, {len(message)} bytes", "FIPS 180-4", [message], [], [digest]))
    for message, digest in short:
        cases.append((f"CAVP short message, {len(message)} bytes", "SHA256ShortMsg.rsp", [message], [], [digest.hex()]))
    for message, digest in long[:2]:
        cases.append((f"CAVP long message, {len(message)} bytes", "SHA256LongMsg.rsp", [message], [], [digest.hex()]))
    three = [short[0], short[3], short[55]]
    require([len(msg) for msg, _ in three] == [0, 3, 55], "three one-block CAVP messages")
    cases.append(("three CAVP short messages of 0, 3 and 55 bytes", "SHA256ShortMsg.rsp",
                  [msg for msg, _ in three], [], [d.hex() for _, d in three]))
    whole = short[64]
    require(len(whole[0]) == 64, "CAVP's 64-byte message")
    cases.append(("CAVP's 64-byte message with its first word and last three bytes committed",
                  "SHA256ShortMsg.rsp", [whole[0]], [0, 1, 2, 3, 61, 62, 63], [whole[1].hex()]))
    label = b"tessera-anchor-key/v1"
    for i, session in enumerate(frames):
        secret = bytes.fromhex(session["key"]) + bytes.fromhex(session["key_salt"])
        cases.append((f"key commitment of frames.json session {i}", "anchor/frames.json",
                      [label + secret], list(range(21, 53)), [session["key_commitment"]]))

    def frame(s, f):
        record = frames[s]["frames"][f]
        return bytes.fromhex(record["header"]) + bytes.fromhex(record["ciphertext"]), record["sha256"]

    secrets = [label + bytes.fromhex(s["key"]) + bytes.fromhex(s["key_salt"]) for s in frames[:3]]
    cases.append(("key commitments of frames.json sessions 0, 1 and 2", "anchor/frames.json",
                  secrets, list(range(21, 53)), [s["key_commitment"] for s in frames[:3]]))
    one_token = [(0, 1), (1, 0), (3, 1)]
    for pair in ((one_token[0], one_token[1]), (one_token[1], one_token[2])):
        msgs = [frame(*sf) for sf in pair]
        require(all(len(msg) == 67 for msg, _ in msgs), "frames of one token are 67 bytes")
        cases.append((f"frames of one token, sessions {pair[0][0]} and {pair[1][0]}", "anchor/frames.json",
                      [msg for msg, _ in msgs], list(range(41, 67)), [d for _, d in msgs]))
    for s, f, n_tokens in ((0, 2, 0), (0, 4, 40)):
        msg, digest = frame(s, f)
        require(len(msg) == 63 + 4 * n_tokens, "a frame hashes 63 + 4n bytes")
        cases.append((f"a frame of {n_tokens} tokens, session {s}", "anchor/frames.json",
                      [msg], list(range(41, len(msg))), [digest]))
    return cases


def render_rows(rows):
    return "[\n" + ",\n".join("    " + json.dumps(row, separators=(",", ":")) for row in rows) + "\n   ]"


def main():
    examples = check_sha256()
    graphs = []
    for name, n_blocks, m, offsets in GRAPHS:
        g, chain = sha256_graph(n_blocks, m, offsets)
        tally, public = counts(g)
        graphs.append((name, n_blocks, m, offsets, g, chain, tally, public))
    # RFC-0072's sizes table, from the graphs' public tensors.
    by_name = {name: (g, chain, tally) for name, _, _, _, g, chain, tally, _ in graphs}
    require(figures(by_name["key commitment"][2]["total"]) == KEY_COMMITMENT, "the key commitment's sizes")
    # At M = 3 the committed bytes [32, 3] have padded positions, so their decomposition adds
    # the input's padding check (RFC-0072 'The circuits' sizes', 'Per graph').
    three = by_name["three key commitments"][2]["total"]
    require(figures(three) == (KEY_COMMITMENT[0] + 1,) + KEY_COMMITMENT[1:] and three["input_padding"] == 1,
            "three key commitments: the key commitment's sizes and one input padding check")
    for name in ("one public block", "three public messages"):
        tally = by_name[name][2]
        require(figures(tally["blocks"][0]) == FIRST_BLOCK_PUBLIC, f"{name}: a first block's public figures")
        require(tally["total"]["zero_checks"] == 680 + 8, f"{name}: 680 zero-checks and 8 bindings")
    crossing = by_name["two blocks, committed bytes crossing into the second"][2]
    require(figures(crossing["blocks"][1]) == LATER_BLOCK_COMMITTED, "a later block of committed bytes")
    require(all(row["nodes"] == 6032 for _, _, _, _, _, _, t, _ in graphs for row in t["blocks"]),
            "6,032 nodes per block (RFC-0066 'Drawbacks')")
    # A later block of public bytes, from a graph the file does not hold.
    require(figures(counts(sha256_graph(2, 1, [])[0])[0]["blocks"][1]) == LATER_BLOCK_PUBLIC,
            "a later block of public bytes")

    inputs = []
    runs = 0
    for name, source, messages, offsets, expected in input_cases(examples):
        n_blocks, digests, values = public_inputs(messages, offsets)
        require([d.hex() for d in digests] == expected, f"{name}: the digests its source gives")
        m = len(messages)
        graph = next((gname for gname, gn, gm, goff, *_ in graphs if (gn, gm, goff) == (n_blocks, m, offsets)), None)
        # The interpreter runs every input set: on its graph in the file, or on a graph of its
        # parameters that the file does not hold.
        g, chain = by_name[graph][:2] if graph is not None else sha256_graph(n_blocks, m, offsets)
        vals = evaluate(g, graph_inputs(g, values))
        for col, msg in enumerate(messages):
            expected_chain = chaining(msg)
            for blk, words in enumerate(chain):
                require([word_of(vals[t], m, col) for t in words] == expected_chain[blk],
                        f"{name}: the graph's chaining words after block {blk}")
        runs += graph is not None
        inputs.append({
            "name": name, "source": source, "graph": graph,
            "N": n_blocks, "M": m, "K": len(offsets), "committed_offsets": offsets,
            "messages": [msg.hex() for msg in messages],
            "digests": [d.hex() for d in digests],
            "chaining": [[[f"{w:08x}" for w in state] for state in chaining(msg)] for msg in messages],
            "message": bits_string(values["message"]),
            "place": "".join(map(str, values["place"])) if offsets else None,
            "committed": values["committed"] if offsets else None,
            "start": bits_string(values["start"]),
            "masks": bits_string(values["masks"]),
            "digest": bits_string(values["digest"]),
        })
    require(runs >= len(GRAPHS), "the interpreter runs every graph")
    require(all(any(case["graph"] == name for case in inputs) for name, *_ in GRAPHS), "every graph has inputs")

    text = "{\n"
    text += ' "description": ' + json.dumps(
        "RFC-0066's SHA-256 circuit, node by node, for five parameter sets, written by "
        "spec/vectors/anchor-graphs/generate.py from RFC-0066 and RFC-0072 alone; "
        "README.md gives the format and the readings the oracle chose.") + ",\n"
    text += ' "round_constants": ' + json.dumps([f"{k:08x}" for k in K]) + ",\n"
    text += ' "initial_value": ' + json.dumps([f"{h:08x}" for h in H0]) + ",\n"
    text += ' "cases": [\n'
    blocks = []
    for name, n_blocks, m, offsets, g, chain, tally, public in graphs:
        body = "  {\n"
        body += f'   "name": {json.dumps(name)},\n'
        body += f'   "N": {n_blocks},\n   "M": {m},\n   "K": {len(offsets)},\n'
        body += f'   "committed_offsets": {json.dumps(offsets)},\n'
        body += f'   "tensors": {len(g.shapes)},\n'
        body += '   "inputs": ' + render_rows(g.inputs) + ",\n"
        body += '   "nodes": ' + render_rows(g.nodes) + ",\n"
        body += '   "constants": ' + render_rows(g.constants) + ",\n"
        body += f'   "chaining": {json.dumps(chain)},\n'
        body += f'   "public": {json.dumps(sorted(public), separators=(",", ":"))},\n'
        body += f'   "counts": {json.dumps(tally)}\n'
        body += "  }"
        blocks.append(body)
    text += ",\n".join(blocks) + "\n ]\n}\n"
    json.loads(text)
    (HERE / "sha256-graphs.json").write_text(text)

    header = {
        "description": "The SHA-256 circuit's inputs from bytes (RFC-0066 'The message', 'The output'), "
        "written by spec/vectors/anchor-graphs/generate.py; README.md gives the fields.",
    }
    text = json.dumps(header, indent=1)[:-2] + ',\n "cases": [\n'
    text += ",\n".join("  " + json.dumps(case, separators=(",", ":")) for case in inputs) + "\n ]\n}\n"
    json.loads(text)
    (HERE / "sha256-inputs.json").write_text(text)


if __name__ == "__main__":
    main()
