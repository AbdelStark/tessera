#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0 OR MIT
"""RFC-0011/0031/0037 canonical grouping oracle, before the model opening checker.

Only independent Python oracles are imported. Full conformance layouts get exact
descriptor/count vectors. Complete opening vectors use a legal public micro-model
variant; synthetic rows are values of c(row)+d(row)*X on H, not inference witnesses.
Natural-coordinate enumeration and weighted mapped images independently agree.
"""
import copy
import hashlib
import importlib.util
import itertools
import json
import sys
from pathlib import Path

sys.dont_write_bytecode = True
HERE = Path(__file__).resolve().parent


def load(name):
    spec = importlib.util.spec_from_file_location(name, HERE.parent / name / "generate.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


pcs, advice = load("pcs"), load("advice")
lowering = advice.load_oracle("lowering")
P, ZERO, ONE = pcs.P, pcs.ZERO, pcs.ONE
add, mul, lift, total, require = pcs.add, pcs.mul, pcs.lift, pcs.total, pcs.require
SEED = 0x7E55E7A_0116_0011
LOG_K, MASKS = 6, 4


def descriptor(statement, unit, log_k, masks, block, segments, rows):
    return (bytes(2) + statement + unit.to_bytes(4, "little") + bytes([log_k, 1])
            + b"".join(x.to_bytes(4, "little") for x in (masks, block, segments))
            + rows.to_bytes(8, "little"))


def native_layouts(rng):
    """Complement public zero intervals; never enumerate millions of rows."""
    cases = json.loads((HERE.parent / "advice/layout.json").read_text())["cases"]
    output = []
    for index, case in enumerate(cases):
        counts = []
        for first, end in case["segment_rows"]:
            zero = sum(max(0, min(b, end)-max(a, first)) for a, b in case["zero_rows"])
            counts.append(end-first-zero)
        rows = sum(counts)
        require(rows == sum(v[5] for t in case["tiles"] for v in t["values"]), "independent row counts")
        statement = rng.digest()
        encoded = descriptor(statement, index, case["log2_k"], 3, 5, case["segments"], rows)
        output.append({"case": index, "statement_digest": statement.hex(), "unit": index,
                       "row_block": 5, "mask_coeffs": 3, "rows": rows,
                       "segment_counts": counts, "descriptor_tce": encoded.hex(),
                       "commitment_id": pcs.H("commitment-id", encoded).hex()})
    return output


def micro_public():
    """Public-only edits, identically made by each Rust consumer, with no private weights."""
    models = lowering.models
    spec = models.MODELS[0]
    model, sequences = models.build(spec)
    run = models.Run(model, spec["plan"])
    run.forward(sequences)
    original = models.integer_model(spec, model, run.c)
    public = {"arch": copy.deepcopy(original["arch"]), "profile": original["profile"],
              "quant": {"public": copy.deepcopy(original["quant"]["public"]),
                        "silu_in_exponents": original["quant"]["silu_in_exponents"]}}
    public["arch"].update(d_model=4, n_heads=2, n_kv_heads=1, d_head=2,
                          d_ff=6, vocab=8, max_context=8)
    quant = public["quant"]["public"]
    embed = next(e for p, e in quant["act_exponents"] if "Embed" in p)
    quant.update(residual_bits=8, residual_exponent=embed, rsqrt_kappa=0, exp_sigma=3)
    for _, shifts in quant["shift_class_sets"]:
        shifts[:] = [12]
    quant["score_scales"] = [[i, {"num": 1, "den": 1}] for i in range(public["arch"]["n_layers"])]
    return public


def bits(shape):
    return sum((n-1).bit_length() for n in shape)


def source_key(source):
    return tuple(source)


def c(row):
    return (17*row+5) % P


def d(row):
    return (row*row+3) % P


def mask(row):
    return [((row+1)*(i+1)+7) % P for i in range(MASKS)]


def row_value(row, column, k):
    return (c(row)+d(row)*pow(pcs.encoding.root(k), column, P)) % P


def symbol(row, x, k):
    return (c(row)+d(row)*x+(pow(x, k, P)-1)*pcs.horner(mask(row), x)) % P


def placed_point(local, location, nu, log_k):
    _, _, permutation, base, _, _ = location
    point = [local[i] for i in permutation]
    for bit in range(len(permutation), nu):
        point.append(lift((base >> (bit-log_k)) & 1) if bit >= log_k else ZERO)
    return point


def image_list(locations, point, group, N, Q, nu, log_k):
    """Fix high head/query bits, then rename the remaining local variables (RFC-0034/47)."""
    first_tile, first, ordinal = locations[0]
    if first_tile["key"][0] != "attention":
        require(len(locations) == 1, "ordinary source has one location")
        return [(ONE, placed_point(point, first, nu, log_k))]
    ni, nq, ng = (x.bit_length()-1 for x in (N, Q, group))
    key_count = ni if ordinal == 0 else 0
    local = point[:key_count] + point[key_count:key_count+nq] + point[key_count+ni:key_count+ni+ng]
    out = []
    for tile, location, _ in locations:
        _, _, kv, qb = tile["key"]
        high_query = point[key_count+nq:key_count+ni]
        high_head = point[key_count+ni+ng:]
        weight = mul(pcs.eq_bits(high_query, qb), pcs.eq_bits(high_head, kv))
        out.append((weight, placed_point(local, location, nu, log_k)))
    return out


def natural_value(locations, shape, point, group, N, Q, k):
    """Enumerate Boolean natural coordinates, independently of the image construction."""
    if locations[0][0]["key"][0] == "table" and bits(shape) >= k.bit_length()-1:
        # A one-dimensional table needs no permutation. Factor its local column
        # sum directly, without expanding every cell of a large multiplicity table.
        location = locations[0][1]
        width = k.bit_length()-1
        low = pcs.eq_table(point[:width])
        phi = total(mul(weight, lift(pow(pcs.encoding.root(k), j, P))) for j, weight in enumerate(low))
        high = pcs.eq_table(point[width:])
        return total(mul(high[r], add(lift(c(location[3]+r)), mul(lift(d(location[3]+r)), phi)))
                     for r in range(location[5]))
    padded = [1 << (axis-1).bit_length() for axis in shape]
    result = ZERO
    for coordinates in itertools.product(*(range(axis) for axis in padded)):
        natural_index = 0
        for index, axis in zip(coordinates, padded):
            natural_index = natural_index*axis+index
        tile, location, ordinal = locations[0]
        if tile["key"][0] == "attention":
            head, query = coordinates[:2]
            kv, qb = head//group, query//Q
            found = next((v for t, v, _ in locations if t["key"][2:] == [kv, qb]), None)
            if found is None:
                continue
            location = found
            local_index = (head % group)*Q+(query % Q)
            if ordinal == 0:
                local_index = local_index*N+coordinates[2]
        else:
            local_index = natural_index
        permutation, base, zero_at = location[2], location[3], location[5]
        physical = sum(((local_index >> old) & 1) << new for new, old in enumerate(permutation))
        row = base+physical//k
        if row-base >= zero_at:
            continue
        result = add(result, mul(pcs.eq_bits(point, natural_index), lift(row_value(row, physical % k, k))))
    return result


def coefficients_digest(values):
    return hashlib.sha256(b"".join(x.to_bytes(8, "little") for pair in values for x in pair)).hexdigest()


def opening_case(rng, unit, n, segments, block, group_count, claim_count):
    public = micro_public()
    graph, _ = lowering.case_of("opening micro", "tiny-2-servable.d", lowering.Public(public), n, None)
    layout = advice.model_layout(graph, LOG_K, 2, segments, 17, 8)
    k, columns = 1 << LOG_K, 2 << LOG_K
    row_vars = layout["nu"]-LOG_K
    rows_by_segment = [[] for _ in range(segments)]
    sources = {}
    for tile in layout["tiles"]:
        for ordinal, location in enumerate(tile["values"]):
            rows_by_segment[tile["segment"]].extend(range(location[3], location[3]+location[5]))
            sources.setdefault(source_key(location[0]), []).append((tile, location, ordinal))
    rows = [r for part in rows_by_segment for r in part]
    require(len(rows) < 1 << 15, "complete micro-model commitment stays small")
    statement = rng.digest()
    encoded = descriptor(statement, unit, LOG_K, MASKS, block, segments, len(rows))
    identifier = pcs.H("commitment-id", encoded)
    heads, group = public["arch"]["n_heads"], 2
    N, Q = 1 << (n-1).bit_length(), min(2, 1 << (n-1).bit_length())
    clusters = {}
    for source, locations in sources.items():
        tile, location, ordinal = locations[0]
        shape = location[1]
        category = source[0]
        if tile["key"][0] == "attention":
            category = "score" if ordinal == 0 else "query"
            shape = [heads, n, n] if ordinal == 0 else [heads, n]
        clusters.setdefault((category, tuple(shape)), []).append(source)
    categories = ["tensor", "score", "query", "multiplicity", "aux"]
    templates = []
    for g in range(group_count):
        category = categories[g % len(categories)]
        candidates = [(shape, members) for (kind, shape), members in clusters.items() if kind == category]
        require(candidates, "every kind has an opening obligation")
        shape, members = max(candidates, key=lambda pair: (len(pair[1]) >= 2, len(pair[0]), bits(pair[0])))
        templates.append((shape, members[:2], [rng.fp2() for _ in range(bits(shape))]))
    gamma = rng.fp2()
    require(gamma != ZERO, "nonzero main batching challenge; zero is checked separately")
    claims, groups = [], []
    natural_cache, image_cache, tables = {}, {}, {}
    # Round-robin claims force stable first-appearance grouping and per-group powers.
    for index in range(claim_count):
        shape, members, point = templates[index % group_count]
        pieces, images, value = [], [], ZERO
        for source in members:
            coefficient = rng.fp2()
            mapped = image_list(sources[source], point, group, N, Q, layout["nu"], LOG_K)
            natural_key = (source, tuple(point))
            if natural_key not in natural_cache:
                natural_cache[natural_key] = natural_value(sources[source], shape, point, group, N, Q, k)
            piece_value = natural_cache[natural_key]
            reconstructed = ZERO
            for weight, full in mapped:
                column, row_point = full[:LOG_K], full[LOG_K:]
                # Full tensor-product table, restricted to the public retained rows.
                key = tuple(full)
                if key not in image_cache:
                    table = pcs.eq_table(row_point)
                    tables[tuple(row_point)] = [table[r] for r in rows]
                    phi = total(mul(lift(pow(pcs.encoding.root(k), j, P)), e) for j, e in enumerate(pcs.eq_table(column)))
                    image_cache[key] = total(mul(table[r], add(lift(c(r)), mul(lift(d(r)), phi))) for r in rows)
                image_value = image_cache[key]
                reconstructed = add(reconstructed, mul(weight, image_value))
                images.append((mul(coefficient, weight), full, image_value))
            require(piece_value == reconstructed, "natural enumeration equals weighted image sum")
            pieces.append({"source": list(source), "coefficient": pcs.fp2(coefficient)})
            value = add(value, mul(coefficient, piece_value))
        column_point = images[0][1][:LOG_K]
        require(all(full[:LOG_K] == column_point for _, full, _ in images), "one claim has one column point")
        found = next((g for g in groups if g["point"] == column_point), None)
        if found is None:
            found = {"point": column_point, "claims": [], "images": [], "expected": ZERO, "power": ONE}
            groups.append(found)
        found["power"] = mul(found["power"], gamma)
        power = found["power"]
        found["expected"] = add(found["expected"], mul(power, value))
        found["claims"].append(index)
        found["images"].extend((mul(power, weight), full[LOG_K:], v) for weight, full, v in images)
        claims.append({"point": pcs.elements(point), "pieces": pieces, "value": pcs.fp2(value)})
    require(len(groups) == group_count, "all requested groups remain distinct")
    xi = [rng.fp2() for _ in range(row_vars)]
    coefficient_rows = []
    for g in groups:
        table = [ZERO] * len(rows)
        for weight, point, _ in g["images"]:
            table = [add(a, mul(weight, b)) for a, b in zip(table, tables[tuple(point)])]
        coefficient_rows.append(table)
    xi_table = pcs.eq_table(xi)
    coefficient_rows.append([xi_table[r] for r in rows])
    combined = []
    for weights in coefficient_rows:
        constant = total(mul(weight, lift(c(row))) for weight, row in zip(weights, rows))
        linear = total(mul(weight, lift(d(row))) for weight, row in zip(weights, rows))
        u = [add(constant, mul(linear, lift(pow(pcs.encoding.root(k), j, P)))) for j in range(k)]
        masks = [total(mul(weight, lift(mask(row)[i])) for weight, row in zip(weights, rows)) for i in range(MASKS)]
        combined.append({"u": pcs.elements(u), "mask": pcs.elements(masks)})
    for g, row in zip(groups, combined):
        u = [pcs.parse_fp2(x) for x in row["u"]]
        require(total(mul(a, b) for a, b in zip(u, pcs.eq_table(g["point"]))) == g["expected"], "group evaluation")
    # Check the closed row recipe against the independent Lagrange encoder.
    for row in rows[:2] + rows[-1:]:
        _, word = pcs.encode_row([row_value(row, j, k) for j in range(k)], mask(row), k, columns)
        require(word == [symbol(row, x, k) for x in pcs.encoding.points(columns, 7)], "masked row recipe")
    leaves = []
    xs = pcs.encoding.points(columns, 7)
    for column, x in enumerate(xs):
        parts = []
        for segment in rows_by_segment:
            chain = pcs.H("column-init", identifier+column.to_bytes(8, "little"))
            for first in range(0, len(segment), block):
                payload = b"".join(symbol(row, x, k).to_bytes(8, "little") for row in segment[first:first+block])
                chain = pcs.H("column-step", chain+payload)
            parts.append(chain)
        leaves.append(pcs.H("column-merge", b"".join(parts)))
    tree = pcs.hashing.merkle_levels(leaves)
    queries = [columns-1, 1, columns//2, 1]
    distinct = sorted(set(queries))
    return {"name": f"micro n={n} S={segments} B={block} G={group_count}",
            "n": n, "log2_k": LOG_K, "log2_inv_rate": 1,
            "segments": rows_by_segment, "row_vars": row_vars,
            "row_block": block, "mask_coeffs": MASKS, "statement_digest": statement.hex(),
            "unit": unit, "descriptor_tce": encoded.hex(), "commitment_id": identifier.hex(),
            "allocated_rows": layout["rows"], "root": tree[-1][0].hex(),
            "gamma": pcs.fp2(gamma), "claims": claims,
            "groups": [{"column_point": pcs.elements(g["point"]), "claim_indices": g["claims"],
                        "expected": pcs.fp2(g["expected"]),
                        "claims": [{"gamma": pcs.fp2(w), "row_point": pcs.elements(p), "value": pcs.fp2(v)}
                                   for w, p, v in g["images"]]} for g in groups],
            "coefficient_sha256": [coefficients_digest(values) for values in coefficient_rows],
            "xi_point": pcs.elements(xi), "combined": combined, "queries": queries,
            "columns": [[pcs.hex64(symbol(row, xs[j], k)) for row in rows] for j in distinct],
            "siblings": [v.hex() for v in pcs.hashing.multiproof(tree, distinct)]}


def main():
    rng = pcs.Inputs(SEED)
    native = native_layouts(rng)
    cases = []
    for unit, shape in enumerate([(1, 1, 32, 1, 3), (3, 2, 3, 16, 40), (5, 8, 5, 5, 13)]):
        cases.append(opening_case(rng, unit, *shape))
        print(cases[-1]["name"], sum(map(len, cases[-1]["segments"])), "rows", flush=True)
    document = {"version": 1, "seed": pcs.hex64(SEED), "layouts": native, "openings": cases}
    (HERE / "opening.json").write_text(json.dumps(document, separators=(",", ":"))+"\n")
    print("canonical advice opening oracle:", len(native), "native layouts;", len(cases), "complete openings", flush=True)


if __name__ == "__main__":
    main()
