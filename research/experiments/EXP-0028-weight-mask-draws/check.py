#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0 OR MIT
"""Independent saved-record reader for EXP-0028; imports no experiment code."""

import gzip
import hashlib
import itertools
import json
import math
import sys
from fractions import Fraction
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
TYPES = ("Embed", "Q", "K", "V", "O", "Gate", "Up", "Down", "LmHead", "ChannelParams")


def pad(n):
    x = 1
    while x < n:
        x *= 2
    return x


def rows_for(record, matrix):
    a, kw = record["arch"], record["kw"]
    widths = [
        (a["d_model"], a["vocab"]),
        (a["n_heads"] * a["d_head"], a["d_model"]),
        (a["n_kv_heads"] * a["d_head"], a["d_model"]),
        (a["n_kv_heads"] * a["d_head"], a["d_model"]),
        (a["d_model"], a["n_heads"] * a["d_head"]),
        (a["d_ff"], a["d_model"]),
        (a["d_ff"], a["d_model"]),
        (a["d_model"], a["d_ff"]),
        (a["vocab"], a["d_model"]),
    ]
    if matrix == "ChannelParams":
        ch = record["channels"]
        c = max([pad(widths[TYPES.index(x["matrix"])][0]) for x in ch] + [1])
        lp = pad(a["n_layers"])
        k = min(kw, 16 * lp * c)
        indices = set()
        for x in ch:
            t = TYPES.index(x["matrix"])
            first = (t * lp + x["layer"]) * c
            last = first + widths[t][0]
            # An interval hits each row between its first and last live position.
            indices.update(range(first // k, (last - 1) // k + 1))
        return k, 16 * lp * c // k, sorted(indices)
    t = TYPES.index(matrix)
    out, inp = widths[t]
    k = max(pad(inp), min(kw, pad(out) * pad(inp)))
    layer_count = 1 if t in (0, 8) else a["n_layers"]
    low_outputs = k // pad(inp)
    stride = pad(out) // low_outputs
    indices = [
        l * stride + h
        for l in range(layer_count)
        for h in range((out - 1) // low_outputs + 1)
    ]
    return k, pad(layer_count) * stride, indices


def expected(case, support, passes, epochs):
    masks = values = 0
    statuses = []
    for m in case["matrices"]:
        d, k, e = m["data_rows"], m["width"], m["claims"]
        bw, p = case["bw"], case["pw"]
        b = bw
        if m["cp"]:
            if e == 0:
                d, b = 0, 0
            else:
                # Repeated extra pairs for every proof that the base budget can hold.
                proofs = len(range(0, max(bw - 4, 0) - 5, 6))
                b += (e - 1) * 2 * proofs
        r = e * 2 + 4 if e else 0
        if m["cp"] and e == 0:
            status = "empty"
        elif p == 0 or case["kw"] + p >= 4 * case["kw"] or (d and k < case["kw"]):
            status = "invalid-masked-width"
        elif r > max(b - 4, 0):
            status = "unfit-reservation"
        else:
            status = "fit"
        statuses.append(status)
        for _ in range(epochs):
            for phase in range(passes):
                all_values = phase == 1 or support == "FULL"
                all_masks = phase == 1 or support != "SELECTED"
                masks += p * (d + (b if all_masks else r))
                values += k * (b if all_values else r)
    return {
        "mask_bytes": masks * 8,
        "value_bytes": values * 8,
        "accepted_bytes": (masks + values) * 8,
    }, statuses


def main(path):
    saved = Path(path).read_bytes()
    doc = json.loads(gzip.decompress(saved) if saved[:2] == b"\x1f\x8b" else saved)
    assert (
        doc["schema_version"] == 1
        and doc["experiment"] == "EXP-0028"
        and doc["verdict"] == "validated"
    )
    frozen = Path(__file__).with_name("results") / "README.preregistered.txt"
    assert doc["registration_sha256"] == hashlib.sha256(frozen.read_bytes()).hexdigest()
    for p, digest in doc["input_sha256"].items():
        assert hashlib.sha256((ROOT / p).read_bytes()).hexdigest() == digest
    for name, digest in doc.get("code_sha256", {}).items():
        assert (
            hashlib.sha256(Path(__file__).with_name(name).read_bytes()).hexdigest()
            == digest
        )
    checks = 0
    geometry = {g["name"]: g for g in doc["geometry"]}
    assert len(geometry) == 14
    fixture_names = set()
    for source in doc["input_sha256"]:
        fixture = json.loads((ROOT / source).read_text())
        for item in fixture.get("models", fixture.get("cases", [])):
            name = item["name"]
            fixture_names.add(name)
            g = geometry[name]
            assert g["source"] == source and g["arch"] == item["arch"]
            clean_channels = [
                {
                    "matrix": TYPES[c["matrix"]]
                    if isinstance(c["matrix"], int)
                    else c["matrix"],
                    "layer": c["layer"],
                }
                for c in item["channels"]
            ]
            assert g["channels"] == clean_channels
            assert g["kw"] == 2 ** fixture.get("layout", item.get("layout"))["log2_k"]
            recorded = item.get("matrices", [item])
            for m in recorded:
                name_t = (
                    TYPES[m["matrix"]] if isinstance(m["matrix"], int) else m["matrix"]
                )
                saved = next(x for x in g["matrices"] if x["matrix"] == name_t)
                digest = hashlib.sha256(
                    b"".join(r.to_bytes(8, "little") for r in m["committed_rows"])
                ).hexdigest()
                assert (
                    saved["width"],
                    saved["extent"],
                    saved["data_rows"],
                    saved["rows_sha256"],
                ) == (
                    2 ** m["log2_k"],
                    2 ** m["row_vars"],
                    len(m["committed_rows"]),
                    digest,
                )
                checks += 1
    assert set(geometry) == fixture_names | {"405B-tight"}
    assert geometry["405B-tight"]["arch"] == {
        "n_layers": 126,
        "d_model": 16384,
        "n_heads": 128,
        "n_kv_heads": 8,
        "d_head": 128,
        "d_ff": 53248,
        "vocab": 128256,
    }
    assert geometry["405B-tight"]["channels"] == [
        {"matrix": t, "layer": l} for t in TYPES[1:8] for l in range(126)
    ]
    for g in geometry.values():
        assert len(g["matrices"]) == 10
        for m in g["matrices"]:
            k, extent, indices = rows_for(g, m["matrix"])
            digest = hashlib.sha256(
                b"".join(i.to_bytes(8, "little") for i in indices)
            ).hexdigest()
            assert (m["width"], m["extent"], m["data_rows"], m["rows_sha256"]) == (
                k,
                extent,
                len(indices),
                digest,
            )
            checks += 1
    cases = {c["name"]: c for c in doc["cases"]}
    assert len(cases) == len(doc["cases"]) == 3671
    grid = list(
        itertools.product(
            [1, 3, 4, 9],
            [1, 2, 5],
            [1, 3],
            [2, 8, 16],
            [0, 1, 4],
            [0, 4, 8, 32],
            [0, 1, 3, 7],
        )
    )
    # Independent random-state replay; no import of the generating script.
    state = 0x5680028
    draws = []
    for _ in range(200 * 7):
        state = (state + 11400714819323198485) % (2**64)
        x = state
        x = ((x ^ (x >> 30)) * 13787848793156543929) % (2**64)
        x = ((x ^ (x >> 27)) * 10723151780598845931) % (2**64)
        draws.append(x ^ (x >> 31))
    for i in range(200):
        x = draws[i * 7 : i * 7 + 7]
        grid.append(
            (
                x[0] % 33 + 1,
                x[1] % 65 + 1,
                x[2] % 7 + 1,
                2 ** (x[3] % 9),
                [0, 1, 4, 16][x[4] % 4],
                [0, 4, 8, 32][x[5] % 4],
                [0, 1, 3, 7][x[6] % 4],
            )
        )
    for i, (out, inp, layers, kw, pw, bw, e) in enumerate(grid):
        case = cases[f"synthetic-{i}"]
        assert case["geometry"] == {"out": out, "inputs": inp, "layers": layers}
        assert (case["kw"], case["pw"], case["bw"]) == (kw, pw, bw)
        k = max(pad(inp), min(kw, pad(out) * pad(inp)))
        row_set = {
            (l * pad(out) + o) * pad(inp) // k
            for l in range(layers)
            for o in range(out)
        }
        assert case["matrices"] == [
            {
                "data_rows": len(row_set) if e else 0,
                "width": k,
                "claims": e,
                "cp": e != 1,
            }
        ]
        checks += 1
    for name, g in geometry.items():
        case = cases[name]
        assert (case["kw"], case["pw"], case["bw"]) == (
            g["kw"],
            65536 if name == "405B-tight" else 64,
            4096 if name == "405B-tight" else 32,
        )
        assert len(case["matrices"]) == 10
        for m, geom in zip(case["matrices"], g["matrices"]):
            assert (m["data_rows"], m["width"]) == (geom["data_rows"], geom["width"])
            cp = geom["matrix"] == "ChannelParams"
            assert m["cp"] == cp and m["claims"] == (
                7 if cp and m["data_rows"] else 0 if cp else 1
            )
        checks += 1
    assert (
        cases["405B-historical"]["matrices"][:-1]
        == cases["405B-tight"]["matrices"][:-1]
    )
    assert cases["405B-historical"]["matrices"][-1]["data_rows"] == 882
    assert cases["405B-historical"]["matrices"][-1]["claims"] == 7
    assert cases["405B-historical"]["matrices"][-1]["cp"] is True
    assert cases["405B-historical"]["matrices"][-1]["width"] == 2**20
    assert (
        cases["405B-historical"]["kw"],
        cases["405B-historical"]["pw"],
        cases["405B-historical"]["bw"],
    ) == (2**20, 2**16, 4096)
    assert doc["judgments"] == sum(
        len(c["outputs"]) * len(c["matrices"]) for c in cases.values()
    )
    for case in cases.values():
        seen = set()
        for out in case["outputs"]:
            key = out["support"], out["passes"], out["epochs"]
            assert key not in seen
            seen.add(key)
            counter, statuses = expected(case, *key)
            assert all(out[k] == v for k, v in counter.items())
            assert out["store_read_bytes"] == counter["accepted_bytes"]
            assert out["capacity"] == statuses
            checks += 1
        for key in itertools.product(
            ["FULL", "SELECTED", "MASK-ALL-SELECTED-VALUES"], [1, 2], [1, 2]
        ):
            _, statuses = expected(case, *key)
            should_exist = key[0] == "FULL" or all(
                s in ("fit", "empty") for s in statuses
            )
            assert (key in seen) == should_exist
    for key, fault in doc["faults"].items():
        assert fault["control"] != fault["faulty"]
        if key != "10":
            c = cases[fault["case"]]
            assert fault["matrix"] in c["matrices"]
            isolated = dict(c, matrices=[fault["matrix"]])
            control, _ = expected(
                isolated, fault["support"], fault["passes"], fault["epochs"]
            )
            assert control == fault["control"]
        checks += 1
    assert set(doc["faults"]) == {str(x) for x in range(1, 11)}
    assert doc["faults"]["10"] == {
        "case": "8-bit-field-251",
        "control": "256/251",
        "faulty": "1",
        "accepted_candidates": 251,
        "all_candidates": 256,
    }
    assert doc["goldilocks_expansion"] == [2**64, 2**64 - 2**32 + 1]
    assert doc["rates"][:4] == [
        {"input": s, "result": "invalid-rate"} for s in ["0", "-1", "nan", "inf"]
    ]
    for scenario in doc["scenarios"]:
        c = cases[scenario["case"]]
        count, _ = expected(c, scenario["support"], 1, 1)
        value = Fraction(count["accepted_bytes"], scenario["hypothetical_gbps"] * 10**9)
        assert scenario["exact_gpu_s"] == [value.numerator, value.denominator]
        assert math.isfinite(scenario["gpu_s"]) and scenario["gpu_s"] == float(value)
        checks += 1
    print(
        json.dumps(
            {
                "independent_checks": checks,
                "cases": len(cases),
                "geometry_cases": len(geometry),
                "faults": 10,
                "result": "pass",
                "evidence": "E0",
            }
        )
    )


if __name__ == "__main__":
    main(sys.argv[1])
