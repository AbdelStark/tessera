#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0 OR MIT
"""RFC-0012/0040/0049 weight-opening oracle, written before the independent verifier.

Only public RFC formulas and the existing independent hash/PCS oracles are used. Sparse
structural indices define evaluation coefficients; dense committed positions define proximity.
Polynomials are interpolated by Lagrange and evaluated by Horner, independently of Rust NTTs.
"""
import importlib.util
import math
import json
import struct
import sys
from pathlib import Path

sys.dont_write_bytecode = True
HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location("pcs_oracle", HERE.parent / "pcs" / "generate.py")
p = importlib.util.module_from_spec(spec)
spec.loader.exec_module(p)
P = p.P
NAMES = ["Embed", "Q", "K", "V", "O", "Gate", "Up", "Down", "LmHead", "ChannelParams"]
SEED = 0x7E55E7A_0049_0109


def pad(x):
    return 1 << (x - 1).bit_length()


def shapes(a):
    d, ff, v, n = a["d_model"], a["d_ff"], a["vocab"], a["n_layers"]
    q, kv = a["n_heads"] * a["d_head"], a["n_kv_heads"] * a["d_head"]
    return [(d, v, 1), (q, d, n), (kv, d, n), (kv, d, n), (d, q, n),
            (ff, d, n), (ff, d, n), (d, ff, n), (v, d, 1)]


def matrix(a, ty, kw, channels):
    """Enumerate small fixtures literally; Rust must derive a compact row map instead."""
    if ty < 9:
        out, inputs, layers = shapes(a)[ty]
        op, ip, lp = pad(out), pad(inputs), pad(layers)
        k = max(ip, min(kw, op * ip))
        per = k // ip
        stride = op // per
        capacity = lp * stride
        values = [[0] * k for _ in range(capacity)]
        live = []
        for layer in range(layers):
            for o in range(out):
                row = layer * stride + o // per
                if row not in live:
                    live.append(row)
                for i in range(inputs):
                    values[row][(o % per) * ip + i] = ((ty + 1) * 11 + layer * 7 + o * 5 + i * 3) % 31 - 15
        return k, capacity, live, [[x % P for x in values[r]] for r in live]
    lp = pad(a["n_layers"])
    c = max((pad(shapes(a)[t][0]) for t, _, _ in channels), default=1)
    total = 16 * lp * c
    k = min(kw, total)
    values = [[0] * k for _ in range(total // k)]
    live = set()
    for ty, layer, kind in channels:
        for ch in range(shapes(a)[ty][0]):
            f = (ty * lp + layer) * c + ch
            row, column = divmod(f, k)
            live.add(row)
            values[row][column] = ch % 2 if kind == "class" else ch + 1
    live = sorted(live)
    return k, total // k, live, [values[r] for r in live]


def fhex(x):
    return f"0x{x:016x}"


def pair(x):
    return [fhex(v) for v in x]


def point(rng, n):
    return [rng.fp2() for _ in range(n)]


def dot(values, pt):
    return p.total(p.mul(v, p.eq_bits(pt, i)) for i, v in enumerate(values))


def case(name, a, ty, log2_kw, channels, rng):
    kw = 1 << log2_kw
    k, capacity, live, rows = matrix(a, ty, kw, channels)
    layout = dict(log2_k=log2_kw, log2_inv_rate=2, mask_coeffs=64, blinding_rows=32)
    epoch, block = 7, 2
    descriptor = (struct.pack("<HBBIIHBIQI", 1, log2_kw, 2, 64, 32, ty,
                              k.bit_length() - 1, block, len(live), epoch))
    cid = p.H("commitment-id", descriptor)
    n = 4 * k
    words = [p.encode_row(row, [], k, n)[1] for row in rows]
    leaves = p.hashing.column_leaves(cid, words, n, block)
    levels = p.hashing.merkle_levels(leaves)
    rp = point(rng, capacity.bit_length() - 1)
    cp = point(rng, k.bit_length() - 1)
    xi = point(rng, (len(live) - 1).bit_length() if live else 0)
    gamma = (2, 3)
    claimed = p.total(p.mul(p.eq_bits(rp, r), dot([p.lift(x) for x in row], cp))
                      for r, row in zip(live, rows))
    if live:
        coefficients = [[p.mul(gamma, p.eq_bits(rp, r)) for r in live],
                        [p.eq_bits(xi, j) for j in range(len(live))]]
        combined = [[p.total(p.mul(c, p.lift(row[i])) for c, row in zip(cs, rows)) for i in range(k)]
                    for cs in coefficients]
        p.require(dot(combined[0], cp) == p.mul(gamma, claimed), "evaluation binds the public claim")
        queries = [n - 1, 0, n - 1, n // 2, 1]
        distinct = sorted(set(queries))
        columns = [[row[j] for row in words] for j in distinct]
        proof = p.hashing.multiproof(levels, distinct)
        at_queries = []
        for u, cs in zip(combined, coefficients):
            # Both extension coordinates use the independent Lagrange/Horner encoder.
            lanes = [p.encode_row([v[t] for v in u], [], k, n)[1] for t in (0, 1)]
            encoded = list(zip(*lanes))
            for j in range(n):
                expected = p.total(p.mul(c, p.lift(w[j])) for c, w in zip(cs, words))
                p.require(encoded[j] == expected, "encoding and row combination commute")
            at_queries.append([pair(encoded[j]) for j in distinct])
    else:
        p.require(claimed == p.ZERO, "an empty matrix's polynomial is zero")
        coefficients, combined, queries, columns, proof, at_queries = [], [], [], [], [], []
    short_columns = None
    if live:
        # Authenticate one symbol too few per column under the expected descriptor. Root
        # binding does not make a maliciously built commitment's column length canonical.
        short_words = words[:-1]
        short_levels = p.hashing.merkle_levels(p.hashing.column_leaves(cid, short_words, n, block))
        short_combined = [[p.total(p.mul(c, p.lift(row[i])) for c, row in zip(cs, rows[:-1])) for i in range(k)]
                          for cs in coefficients]
        for u, cs in zip(short_combined, coefficients):
            lanes = [p.encode_row([v[t] for v in u], [], k, n)[1] for t in (0, 1)]
            for j, encoded in enumerate(zip(*lanes)):
                p.require(encoded == p.total(p.mul(c, p.lift(word[j])) for c, word in zip(cs, short_words)),
                          "short-column codewords are algebraically consistent")
        short_columns = dict(root=short_levels[-1][0].hex(),
                             columns=[[fhex(word[j]) for word in short_words] for j in distinct],
                             siblings=[x.hex() for x in p.hashing.multiproof(short_levels, distinct)],
                             combined=[[pair(x) for x in u] for u in short_combined],
                             value=pair(p.mul(dot(short_combined[0], cp), p.inverse(gamma))))
    omitted_columns = None
    if live:
        # A valid authentication proof for a proper subset must not discharge all queries.
        omitted_columns = dict(columns=[[fhex(x) for x in col] for col in columns[:-1]],
                               siblings=[x.hex() for x in p.hashing.multiproof(levels, distinct[:-1])])
    ordered_faults = []
    if live and k >= 4:
        # Each row fails at a different authenticated query. Lagrange-domain polynomial
        # products make every other queried symbol agree, without using either Rust checker.
        h = p.encoding.points(k, 1)
        xs = p.encoding.points(n, p.SHIFT)
        for failed in [(3, 1), (1, 3)]:
            wrong = []
            for row, fail in zip(combined, failed):
                zeros = [xs[j] for index, j in enumerate(distinct) if index != fail]
                delta = lambda x: p.lift(math.prod(x - z for z in zeros) % P)
                changed = [p.add(v, delta(x)) for v, x in zip(row, h)]
                lanes = [p.encode_row([v[t] for v in changed], [], k, n)[1] for t in (0, 1)]
                encoded = list(zip(*lanes))
                old_lanes = [p.encode_row([v[t] for v in row], [], k, n)[1] for t in (0, 1)]
                old_encoded = list(zip(*old_lanes))
                p.require([index for index, j in enumerate(distinct) if encoded[j] != old_encoded[j]] == [fail],
                          "exactly the chosen query fails for this row")
                wrong.append(changed)
            value = p.mul(dot(wrong[0], cp), p.inverse(gamma))
            ordered_faults.append(dict(combined=[[pair(x) for x in row] for row in wrong],
                                       value=pair(value), first_failing_column=distinct[min(failed)]))
    return dict(name=name, short_columns=short_columns, omitted_columns=omitted_columns, ordered_faults=ordered_faults, arch=a, channels=[dict(matrix=t, layer=l, kind=z) for t, l, z in channels],
                matrix=ty, rows=[[fhex(x) for x in row] for row in rows], layout=layout, row_block=block, epoch=epoch, log2_k=k.bit_length() - 1,
                row_vars=capacity.bit_length() - 1, committed_rows=live, descriptor=descriptor.hex(),
                commitment_id=cid.hex(), root=levels[-1][0].hex(), row_point=[pair(x) for x in rp],
                column_point=[pair(x) for x in cp], xi_point=[pair(x) for x in xi],
                gamma=pair(gamma), value=pair(claimed), combined=[[pair(x) for x in u] for u in combined],
                queries=queries, columns=[[fhex(x) for x in col] for col in columns],
                siblings=[x.hex() for x in proof], coefficients=[[pair(x) for x in cs] for cs in coefficients],
                combined_at_queries=at_queries)


def main():
    p.check_field_vectors()
    rng = p.Inputs(SEED)
    a = dict(n_layers=3, d_model=5, n_heads=1, n_kv_heads=1, d_head=5, d_ff=9, vocab=7)
    scalar = dict(n_layers=1, d_model=1, n_heads=1, n_kv_heads=1, d_head=1, d_ff=1, vocab=1)
    cases = [case("wide-embedding", a, 0, 2, [], rng),
             case("gapped-gate", a, 5, 2, [], rng),
             case("wide-down", a, 7, 2, [], rng),
             case("partial-head", a, 8, 3, [], rng),
             case("channel-gaps", a, 9, 2, [(1, 0, "class"), (1, 2, "class"), (7, 1, "multiplier")], rng),
             case("scalar", scalar, 1, 0, [], rng),
             case("empty-channel", a, 9, 2, [], rng)]
    output = dict(description="Independent RFC-0012/0040/0049 protocol1 weight openings; no masks or blinding rows.",
                  seed=hex(SEED), cases=cases)
    (HERE / "openings.json").write_text(json.dumps(output, indent=2) + "\n")
    print(f"weights-opening: {len(cases)} independent cases")


if __name__ == "__main__":
    main()
