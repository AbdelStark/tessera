# SPDX-License-Identifier: Apache-2.0 OR MIT
"""Model-derived requantization bindings, independently evaluated on Boolean rows.

Lookup keys come from whole-integer digits, not the verifier's affine forms.
Only the existing Python public-model, lowering and table oracles are imported.
"""
import copy
import importlib.util
from pathlib import Path


def cases(mle):
    path = Path(__file__).resolve().parent.parent / "advice-opening/generate.py"
    spec = importlib.util.spec_from_file_location("opening_oracle", path)
    opening = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(opening)
    base = opening.micro_public()
    base["arch"].update(d_model=2, n_heads=1, n_kv_heads=1, d_ff=6)
    variants = [("one", 32, [12]), ("split", 16, [12]),
                ("wide-zero-low", 24, [1]), ("wide-low", 24, [4]),
                ("four-classes", 24, [4, 5, 6, 7])]
    selected = set()
    out = []
    for name, maximum, shifts in variants:
        public = copy.deepcopy(base)
        if shifts != [12]:
            for point, values in public["quant"]["public"]["shift_class_sets"]:
                if "Layer" in point and next(iter(point["Layer"]["site"])) in \
                        opening.lowering.WEIGHT_BEFORE:
                    values[:] = shifts
        graph, _ = opening.lowering.case_of(name, "tiny-2-servable.d",
                                           opening.lowering.Public(public), 3, None)
        tables, choices = opening.advice.catalog(graph, maximum, 8)
        ranges = opening.advice.ranges.analyze(graph)["ranges"]
        sites = []
        for gadget, choice in enumerate(choices):
            pattern = choice["pattern"]
            if pattern[0] not in ("Requant1", "RequantSplit", "RequantWide", "Lookup"):
                continue
            selected.add(tuple(pattern))
            nid, op, param, inputs, outputs, _ = graph["nodes"][choice["node"]]
            final_kind = tables[choice["tables"][-1]][1]
            table_param = next(iter(final_kind.values()))
            classes = len(table_param["r_by_class"])
            b = table_param["b"] if op == "TableMap" else opening.advice.signed_bits(ranges[inputs[0]])
            output, shape, _ = outputs[0]
            channel_vars = (shape[-1]-1).bit_length()
            variables = sum((dim-1).bit_length() for dim in shape)
            point = [[11+3*i, 7+5*i] for i in range(variables)]
            sources = [["tensor", inputs[0]], ["tensor", output]]
            if classes > 1:
                sources.append(["tensor", inputs[1]])
            sources += [["auxiliary", nid, role] for role, _ in choice["roles"]]
            columns = [[] for _ in sources]
            keys, values = [[] for _ in choice["tables"]], [[] for _ in choice["tables"]]
            low, high = ranges[inputs[0]]
            for row in range(1 << variables):
                time, channel = divmod(row, 1 << channel_vars)
                valid = time < 3 and channel < shape[-1]
                c = channel % classes if channel < shape[-1] else 0
                probes = [low, high, 0, max(low, -1), min(high, 1), (low+high)//2,
                          max(low, -17), min(high, 255)]
                z = probes[(row+nid) % len(probes)] if valid else 0
                zoff = z + (1 << (b-1))
                witness, lookup_keys = [], []
                if pattern[0] in ("Requant1", "Lookup"):
                    lookup_keys = [c*(1 << b) + zoff]
                elif pattern[0] == "RequantSplit":
                    s = pattern[1]
                    top, lo = divmod(zoff, 1 << s)
                    witness = [lo]
                    lookup_keys = [lo, c*(1 << (b-s)) + top]
                else:
                    s, t = pattern[1:]
                    k = b-s-t
                    top, lo = divmod(zoff, 1 << s)
                    top, mid = divmod(top, 1 << t)
                    g = min(1, max(-2, top-(1 << (k-1))))
                    witness = ([lo] if s else []) + [mid, g]
                    lookup_keys = ([lo] if s else []) + [mid, top, 2*top,
                        c*(1 << (t+2)) + (1 << (t+1)) + mid+(1 << t)*g]
                lookup_values = []
                for tid, key in zip(choice["tables"], lookup_keys):
                    count, value, _, _ = opening.lowering.tables.table(tables[tid][1])
                    assert 0 <= key < count
                    lookup_values.append(value(key))
                y = lookup_values[-1]
                if op == "Requant":
                    actual_shifts = graph["shift_classes"][param["shift"]]
                    assert y == opening.lowering.tables.rq(z, actual_shifts[c], param["lo"], param["hi"])
                for column, value in zip(columns, [z, y]+([c] if classes > 1 else [])+witness):
                    column.append(value)
                for column, value in zip(keys, lookup_keys):
                    column.append(value)
                for column, value in zip(values, lookup_values):
                    column.append(value)
            evaluations = [mle(column[:1 << channel_vars], point[:channel_vars])
                           if classes > 1 and i == 2 else mle(column, point)
                           for i, column in enumerate(columns)]
            sites.append({"gadget": gadget, "node": nid, "pattern": pattern,
                          "tables": choice["tables"], "shape": shape, "sources": sources,
                          "point": point, "columns": columns, "evaluations": evaluations,
                          "keys": keys, "values": values,
                          "key_evals": [mle(v, point) for v in keys],
                          "value_evals": [mle(v, point) for v in values]})
        used = {tid for site in sites for tid in site["tables"]}
        out.append({"name": name, "maximum": maximum, "shifts": shifts,
                    "n": 3, "tables": [table for table in tables if table[0] in used],
                    "sites": sites})
    assert {"Requant1", "RequantSplit", "RequantWide", "Lookup"} <= {p[0] for p in selected}
    assert any(p[0] == "RequantWide" and p[1] == 0 for p in selected)
    return {"schema_version": 1, "description": "Independent public micro-model bindings and integer-row requantization lookups", "cases": out}
