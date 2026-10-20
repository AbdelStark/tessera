#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0 OR MIT
"""Public model advice oracle, written before either model-layout implementation.

RFC-0011/0034/0035/0041/0046/0047/0050 define the geometry and gadget roles.
Input graphs are independent Python lowering fixtures, never Rust output or model
weights. This oracle expands every tile and allocates every value individually;
an implementation may instead count repeated tiles without expanding them.
"""
import importlib.util
import json
import math
import sys
from pathlib import Path

sys.dont_write_bytecode = True
HERE = Path(__file__).resolve().parent
P = 2**64 - 2**32 + 1
MAX_ROWS = 2**32


def require(ok, message):
    if not ok:
        raise ValueError(message)


def load_oracle(name):
    spec = importlib.util.spec_from_file_location(name, HERE.parent / name / "generate.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


ranges = load_oracle("ranges")


def log2_up(n):
    require(n > 0, "positive domain")
    return (n - 1).bit_length()


def signed_bits(bounds):
    require(bounds is not None, "bounded gadget input")
    lo, hi = bounds
    return max(1, (~lo).bit_length() + 1 if lo < 0 else 1,
               hi.bit_length() + 1 if hi >= 0 else 1)


def range_kind(bits):
    return {"Range": {"bits": bits}}


def requant_kind(bits, shifts, lo, hi):
    require(bits >= 1 and all(r >= 1 for r in shifts), "requant table domain")
    return {"Requant": {"b": bits, "r_by_class": shifts, "lo": lo, "hi": hi}}


def convolution(a, b):
    out = [0] * (len(a) + len(b) - 1)
    for i, x in enumerate(a):
        for j, y in enumerate(b):
            out[i + j] += x * y
    return out


def root_schema(dim, kappa, epsilon):
    """Whole integer constants; weighted-prefix carry bounds, not a field product."""
    require(dim >= 1 and 0 <= kappa <= 31 and 1 <= epsilon < 2**55, "root parameters")
    constant = dim << (2 * kappa)
    width = max(1, math.isqrt(constant // epsilon).bit_length())
    require(width <= 32, "root width")
    r = (width - 1) // 8
    m = (2 * width + 62) // 8
    roles = [["root.input", j] for j in range(3)] + [["root.low", j] for j in range(r)]
    requests = [width, 16, 16, 16, 7] + [8] * r + [width - 8 * r]
    root = [255] * r + [(1 << (width - 8 * r)) - 1]
    inputs = [65535, 0, 65535, 0, 65535, 0, 127]
    products = []
    for increment in (0, 1):
        factors = root.copy()
        factors[0] += increment
        raw = convolution(convolution(factors, factors), inputs)
        require(len(raw) <= m, "full product fits")
        products.append(raw + [0] * (m - len(raw)))
    for comparison, sides in (
        ("lower", (products[0], [(constant >> (8*j)) & 255 for j in range(m)])),
        ("upper", ([((constant+1) >> (8*j)) & 255 for j in range(m)], products[1])),
    ):
        require(constant + 1 < 256**m, "whole constant fits")
        carry_widths = []
        for raw in sides:
            prefix, previous_box, widths = 0, 0, []
            for j, coefficient in enumerate(raw):
                prefix += coefficient * 256**j
                carry = prefix // 256**(j+1)
                bits = max(1, carry.bit_length())
                box = (1 << bits) - 1
                require(max(coefficient + previous_box, 255 + 256*box) < 2**40,
                        "entire root normalization lookup box")
                previous_box = box
                widths.append(bits)
            require(carry == 0, "complete product has zero final carry")
            carry_widths.append(widths)
        for name, widths in (
            ("left.digit", [8]*m), ("left.carry", carry_widths[0]),
            ("right.digit", [8]*m), ("right.carry", carry_widths[1]),
            ("difference", [8]*m), ("borrow", [1]*m),
        ):
            roles.extend([[f"root.{comparison}.{name}", j] for j in range(m)])
            requests.extend(widths)
    require(len(roles) + 1 == 4 + r + 12*m, "root column count")
    require(len(requests) == 6 + r + 12*m, "root request count")
    return roles, requests


def catalog(case, maximum=32, limb=16):
    """RFC-0006 selection, then every request of the accepted RFC-0050 schema."""
    intervals = ranges.analyze(case)["ranges"]
    tables = json.loads(json.dumps(case["tables"]))
    choices = []

    def intern(kind):
        if "Range" in kind:
            rows = 1 << kind["Range"]["bits"]
        else:
            q = kind["Requant"]
            rows = len(q["r_by_class"]) << q["b"]
        bits = log2_up(rows)
        require(1 <= bits <= maximum, "Gadget: table policy")
        return find_table(kind, bits)

    def append_table(kind, bits):
        tid = len(tables)
        tables.append([tid, kind, bits])
        return tid

    def find_table(kind, bits):
        for tid, old, _ in tables:
            if old == kind:
                return tid
        return append_table(kind, bits)

    for nid, op, param, inputs, outputs, _ in case["nodes"]:
        roles, requests, pattern = [], [], [op]
        if op == "Requant":
            b = signed_bits(intervals[inputs[0]])
            shifts = case["shift_classes"][param["shift"]]
            classes = log2_up(len(shifts))
            lo, hi = param["lo"], param["hi"]
            if b + classes <= maximum:
                pattern = ["Requant1"]
                requests = [requant_kind(b, shifts, lo, hi)]
            else:
                low = b + classes - maximum
                high = min(min(shifts)-1, maximum)
                if b <= 62 and len(shifts)*2**b < P and low <= high:
                    s = min(high, max(low, (b+classes+1)//2))
                    require(1 <= s < b, "Gadget: positive high part")
                    pattern, roles = ["RequantSplit", s], [["requant.low", 0]]
                    requests = [range_kind(s), requant_kind(b-s, [r-s for r in shifts], lo, hi)]
                    require((1 << (b-1)) % (1 << s) == 0, "split padded witness is zero")
                else:
                    s = min(min(shifts)-1, maximum)
                    t = max(shifts)-s + log2_up(max(abs(lo), abs(hi), 1))
                    k = b-s-t
                    require(b <= 62 and k >= 1 and t+2+classes <= maximum
                            and k+1 <= maximum, "Gadget: wide preconditions")
                    pattern = ["RequantWide", s, t]
                    roles = ([["requant.low", 0]] if s else [])
                    roles += [["requant.middle", 0], ["requant.saturation", 0]]
                    requests = ([range_kind(s)] if s else []) + [range_kind(t), range_kind(k),
                        requant_kind(k+1, [1], -2, 1),
                        requant_kind(t+2, [r-s for r in shifts], lo, hi)]
                    require((1 << (b-1)) % (1 << (s+t)) == 0, "wide padded witness is zero")
        elif op == "RsqrtRow":
            roles, widths = root_schema(param["dim"], param["kappa"], param["eps_int"])
            requests = [range_kind(w) for w in widths]
        elif op == "SoftmaxRow":
            require(1 <= limb <= maximum, "Gadget: limb policy")
            sigma = param["sigma"]
            roles = [["softmax.maximum", 0]]
            requests = [param["exp_table"], range_kind(sigma+1)]
            low_count = (sigma+17)//limb
            for slack in ("a", "b"):
                roles += [[f"softmax.{slack}", j] for j in range(low_count)]
                requests += [range_kind(limb)]*low_count + [range_kind(sigma+18-low_count*limb)]
        elif op == "TableMap":
            pattern, requests = ["Lookup"], [param["table"]]
        elif op != "OneHot":
            continue
        ids = [r if isinstance(r, int) else intern(r) for r in requests]
        require(all(tables[t][2] <= maximum for t in ids), "Gadget: base table policy")
        choices.append({"node": nid, "pattern": pattern, "roles": roles, "tables": ids})
    return tables, choices


def geometry(shape, log_k):
    require(shape and all(0 < x <= 2**32 for x in shape), "shape")
    widths = [log2_up(x) for x in shape]
    order = sorted(range(len(shape)), key=lambda i: shape[i] == 1 << widths[i])
    starts = [sum(widths[i+1:]) for i in range(len(shape))]
    permutation = [bit for axis in reversed(order)
                   for bit in range(starts[axis], starts[axis]+widths[axis])]
    size = 1 << sum(widths)
    rows = max(1, size >> log_k)
    zero = rows
    if order and shape[order[0]] != 1 << widths[order[0]]:
        used = shape[order[0]] * (size >> widths[order[0]])
        zero = (used + (1 << log_k)-1) >> log_k
    return permutation, rows, zero


def model_layout(case, log_k, query_block, segments, maximum=32, limb=16):
    require(query_block > 0 and query_block & (query_block-1) == 0, "Layout")
    require(segments > 0, "Layout")
    tables, choices = catalog(case, maximum, limb)
    by_node = {c["node"]: c for c in choices}
    producers = {o[0]: node[0] for node in case["nodes"] for o in node[4]}
    boundaries = [producers[t] for t in case["checkpoints"]]
    layers = len(boundaries)-1
    cuts = [j*(layers//segments)+min(j, layers % segments) for j in range(segments+1)]
    rope = case["tables"][0][1]["Rope"]
    soft = next(node for node in case["nodes"] if node[1] == "SoftmaxRow")
    heads = soft[4][0][1][0]
    kv_rows = next(i[2][0] for i in case["inputs"] if i[1] == "weight K 0")
    kv_heads = kv_rows // rope["d_head"]
    group = heads // kv_heads
    n, N = case["n"], 1 << log2_up(case["n"])
    Q = min(query_block, N)
    require(group*Q <= 1 << log_k, "Layout: row too short for a key")
    w = min(log2_up(N), log_k-log2_up(group)-log2_up(Q))
    tiles, zeros, cursor = [], [], 0

    def zero(start, end):
        if start < end:
            zeros.append([start, end])

    def tile(key, segment, values):
        nonlocal cursor
        placed = []
        for source, shape, special in values:
            permutation, rows, zero_at = geometry(shape, log_k)
            if special == "score":
                ni, nq, nh = log2_up(N), log2_up(Q), log2_up(group)
                permutation = list(range(w)) + list(range(ni, ni+nq+nh)) + list(range(w, ni))
                zero_at = 0 if key[3]*Q >= n else min(rows, (min(n, (key[3]+1)*Q)+(1 << w)-1) >> w)
            elif special == "query" and key[3]*Q >= n:
                zero_at = 0
            base = ((cursor+rows-1)//rows)*rows
            zero(cursor, base)
            zero(base+zero_at, base+rows)
            placed.append([source, shape, permutation, base, rows, zero_at])
            cursor = base+rows
        tiles.append({"key": key, "segment": segment,
                      "rows": [placed[0][3], cursor], "values": placed})

    tile(["tokens"], 0, [(["tensor", case["inputs"][0][0]], [n], None)])
    for nid, op, param, inputs, outputs, _ in case["nodes"]:
        layer = next((i for i in range(layers) if boundaries[i] < nid <= boundaries[i+1]), None)
        segment = (0 if nid <= boundaries[0] else segments-1) if layer is None else next(
            j for j in range(segments) if cuts[j] <= layer < cuts[j+1])
        values = [(["tensor", tid], shape, None) for tid, shape, kind in outputs if kind != "v"]
        choice = by_node.get(nid)
        if choice:
            aux_shape = outputs[1][1] if op == "SoftmaxRow" else outputs[0][1]
            values += [(["aux", nid, *role], aux_shape, None) for role in choice["roles"]]
        if not values:
            continue
        if op == "SoftmaxRow":
            for kv in range(kv_heads):
                for qb in range(N//Q):
                    pieces = []
                    for i, (source, _, _) in enumerate(values):
                        pieces.append((source, [group, Q, N] if i == 0 else [group, Q],
                                       "score" if i == 0 else "query"))
                    tile(["attention", nid, kv, qb], segment, pieces)
        else:
            tile(["node", nid], segment, values)
    used_tables = {t for choice in choices for t in choice["tables"]}
    for tid, kind, log_rows in tables:
        if tid not in used_tables:
            continue
        if "Requant" in kind:
            q = kind["Requant"]
            length = len(q["r_by_class"]) << q["b"]
        elif "Silu" in kind:
            q = kind["Silu"]
            length = len(q["r_by_class"]) << q["b"]
        else:
            length = 1 << log_rows
        tile(["table", tid], segments-1, [(["multiplicity", tid], [length], None)])
    require(cursor <= MAX_ROWS, "Layout: maximum rows")
    merged = []
    for start, end in sorted(zeros):
        if merged and start <= merged[-1][1]:
            merged[-1][1] = max(merged[-1][1], end)
        else:
            merged.append([start, end])
    segment_edges = [0] + [next((t["rows"][0] for t in tiles if t["segment"] >= j), cursor)
                           for j in range(1, segments)] + [cursor]
    return {"model": case["model"], "file": case["file"], "variant": case.get("variant"),
            "n": n, "log2_k": log_k, "query_block": query_block, "segments": segments,
            "tables": tables, "choices": choices, "tiles": tiles, "rows": cursor,
            "nu": log_k+log2_up(cursor), "zero_rows": merged,
            "segment_rows": [[a, b] for a, b in zip(segment_edges, segment_edges[1:])]}


def self_check():
    for width in range(1, 33):
        roles, requests = root_schema(1 if width < 32 else 4, min(width-1, 30), 1)
        require(len(roles) == len({tuple(r) for r in roles}), "unique root roles")
        require(max(requests) <= 32, "root policy")
    require(len(root_schema(4, 30, 1)[0])+1 == 187, "width32 columns")
    require(root_schema(1, 0, 2)[1][0] == 1, "zero-root minimum width")
    order, rows, zero_at = geometry([16384, 53248], 22)
    require(rows == 256 and zero_at == 208 and order[:14] == list(range(16, 30)), "405B MLP")
    _, rows, _ = geometry([2**21, 2**21], 10)
    require(rows == MAX_ROWS and rows+1 > MAX_ROWS, "exact row limit")
    # Check the permutation against explicit Cartesian indices, not its construction.
    from itertools import product
    for shape in ([3, 4], [4, 3], [3, 5], [2, 3, 4], [1, 7, 2]):
        permutation, _, _ = geometry(shape, 2)
        padded = [1 << log2_up(x) for x in shape]
        axes = [i for i, (x, p) in enumerate(zip(shape, padded)) if x != p]
        axes += [i for i, (x, p) in enumerate(zip(shape, padded)) if x == p]
        for indices in product(*(range(p) for p in padded)):
            natural = 0
            for x, p in zip(indices, padded):
                natural = natural*p+x
            reference = 0
            for axis in axes:
                reference = reference*padded[axis]+indices[axis]
            actual = sum(((natural >> old) & 1) << new for new, old in enumerate(permutation))
            require(actual == reference, "axis permutation preserves index identity")


def render(document):
    compact = lambda value: json.dumps(value, separators=(",", ":"))
    cases = []
    for case in document["cases"]:
        fields = []
        for key, value in case.items():
            if key == "tiles":
                tiles = []
                for tile in value:
                    header = ",".join(compact(k)+":"+compact(v) for k, v in tile.items() if k != "values")
                    entries = ",\n".join("      "+compact(v) for v in tile["values"])
                    tiles.append("    {"+header+',"values":[\n'+entries+"\n    ]}")
                fields.append('   "tiles":[\n'+",\n".join(tiles)+"\n   ]")
            elif key in ("tables", "choices"):
                fields.append("   "+compact(key)+":[\n"+",\n".join("    "+compact(v) for v in value)+"\n   ]")
            else:
                fields.append("   "+compact(key)+":"+compact(value))
        cases.append("  {\n"+",\n".join(fields)+"\n  }")
    fields = []
    for key, value in document.items():
        if key == "cases":
            continue
        body = ("[\n"+",\n".join("  "+compact(v) for v in value)+"\n ]"
                if key == "roots" else compact(value))
        fields.append(" "+compact(key)+":"+body)
    text = "{\n"+",\n".join(fields+[' "cases":[\n'+",\n".join(cases)+"\n ]"])+"\n}\n"
    require(json.loads(text) == document, "rendered vector round trip")
    return text


def main():
    self_check()
    graphs = json.loads((HERE.parent / "lowering/lowering.json").read_text())["cases"]
    cases = [model_layout(c, k, q, s) for c in graphs for k, q, s in ((10, 32, 3), (22, 512, 8))]
    cases += [model_layout(c, 10, 1, 1) for c in graphs if c["n"] == 5]
    require(len(cases) == 54, "all independent lowering cases")
    roots = []
    for width in range(1, 33):
        roles, requests = root_schema(1 if width < 32 else 4, min(width-1, 30), 1)
        roots.append({"width": width, "roles": roles, "requests": requests})
    document = {"version": 1, "description": "Public advice layout and complete gadget columns; RFC-0011/0034/0035/0041/0046/0047/0050, independent expanded Python oracle.", "roots": roots, "cases": cases}
    (HERE / "layout.json").write_text(render(document))
    print(f"advice oracle: {len(cases)} cases; {sum(len(t['values']) for c in cases for t in c['tiles'])} values")


if __name__ == "__main__":
    main()
