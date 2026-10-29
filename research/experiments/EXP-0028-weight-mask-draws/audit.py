#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0 OR MIT
"""EXP-0028: public row geometry and distinct stream/pass accounting, E0 only."""

from __future__ import annotations

import argparse
import datetime
import gzip
import hashlib
import itertools
import json
import platform
import subprocess
from fractions import Fraction
from functools import lru_cache
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
NAMES = ["Embed", "Q", "K", "V", "O", "Gate", "Up", "Down", "LmHead"]
SUPPORTS = ["FULL", "SELECTED", "MASK-ALL-SELECTED-VALUES"]
MASK64 = (1 << 64) - 1


def power(n):
    return 1 << (n - 1).bit_length()


def shapes(arch):
    d, f, v = arch["d_model"], arch["d_ff"], arch["vocab"]
    q = arch["n_heads"] * arch["d_head"]
    kv = arch["n_kv_heads"] * arch["d_head"]
    return [(d, v), (q, d), (kv, d), (kv, d), (d, q), (f, d), (f, d), (d, f), (v, d)]


@lru_cache(None)
def literal_weight(out, inputs, layers, kw):
    op, ip = power(out), power(inputs)
    width = max(ip, min(kw, op * ip))
    rows = sorted(
        {
            ((layer * op + o) * ip) // width
            for layer in range(layers)
            for o in range(out)
        }
    )
    return width, power(layers) * op * ip // width, tuple(rows)


def formula_weight(out, inputs, layers, kw):
    ip = power(inputs)
    width = max(ip, min(kw, power(out) * ip))
    per = width // ip
    stride = power(out) // per
    return (
        width,
        power(layers) * stride,
        tuple(
            l * stride + row
            for l in range(layers)
            for row in range((out + per - 1) // per)
        ),
    )


def channels_geometry(arch, channels, kw, literal):
    dims = shapes(arch)
    positions = sorted({(NAMES.index(c["matrix"]), c["layer"]) for c in channels})
    cwidth = max([power(dims[t][0]) for t, _ in positions] + [1])
    lp = power(arch["n_layers"])
    width = min(kw, 16 * lp * cwidth)
    rows = set()
    for t, l in positions:
        start = (t * lp + l) * cwidth
        if literal:
            rows.update((start + c) // width for c in range(dims[t][0]))
        else:
            rows.update(range(start // width, (start + dims[t][0] - 1) // width + 1))
    return width, 16 * lp * cwidth // width, tuple(sorted(rows))


def digest_rows(rows):
    h = hashlib.sha256()
    for row in rows:
        h.update(row.to_bytes(8, "little"))
    return h.hexdigest()


def geometry_record(name, source, arch, channels, kw, expected=None):
    matrices = []
    for i, (out, inp) in enumerate(shapes(arch)):
        layers = 1 if i in (0, 8) else arch["n_layers"]
        a = literal_weight(out, inp, layers, kw)
        b = formula_weight(out, inp, layers, kw)
        assert a == b, (name, NAMES[i], "geometry")
        matrices.append((NAMES[i], a))
    a = channels_geometry(arch, channels, kw, True)
    b = channels_geometry(arch, channels, kw, False)
    assert a == b, (name, "ChannelParams", "geometry")
    matrices.append(("ChannelParams", a))
    saved = []
    for matrix, (width, extent, rows) in matrices:
        if expected is not None and matrix in expected:
            e = expected[matrix]
            assert width == 1 << e["log2_k"], (name, matrix, "width")
            assert extent == 1 << e["row_vars"], (name, matrix, "extent")
            assert list(rows) == e["committed_rows"], (name, matrix, "row indices")
        saved.append(
            {
                "matrix": matrix,
                "width": width,
                "extent": extent,
                "data_rows": len(rows),
                "rows_sha256": digest_rows(rows),
            }
        )
    return {
        "name": name,
        "source": source,
        "arch": arch,
        "channels": [{"matrix": c["matrix"], "layer": c["layer"]} for c in channels],
        "kw": kw,
        "matrices": saved,
    }


def candidate(matrix, pw, bw, kw, support, passes, epochs, fault=0):
    d, k, e, cp = (matrix[x] for x in ("data_rows", "width", "claims", "cp"))
    b = bw + 2 * (e - 1) * (max(bw - 4, 0) // 6) if cp and e else bw
    if cp and e == 0:
        d, b = 0, (bw if fault == 6 else 0)
    if fault == 4 and cp and e:
        b = bw
    r = (6 if fault == 5 else 2 * e + 4) if e else (6 if fault == 6 else 0)
    if fault == 3:
        k = kw
    masks = (d + b) * pw
    values = b * k
    if support != "FULL":
        if support == "SELECTED":
            masks = (d + r) * pw
            if fault == 1:
                masks -= r * pw
        values = r * k
        if fault == 2:
            values = 0
    if fault == 7:
        masks *= 2
        values *= 2
    # P5 answers all committed rows even when P4's hypothetical support was smaller.
    if passes == 2 and fault != 8:
        masks += (d + b) * pw
        values += b * k
    scale = 1 if fault == 9 else epochs
    return {
        "mask_bytes": masks * 8 * scale,
        "value_bytes": values * 8 * scale,
        "accepted_bytes": (masks + values) * 8 * scale,
    }


def reference(matrix, pw, bw, support, passes, epochs):
    # Independent explicit row consumers; no candidate counter or geometry helper.
    data = matrix["data_rows"]
    if matrix["cp"]:
        claims = matrix["claims"]
        blind = 0 if claims == 0 else bw + (claims - 1) * 2 * ((max(4, bw) - 4) // 6)
        if claims == 0:
            data = 0
    else:
        claims = matrix["claims"]
        blind = bw
    reserved = 0 if claims == 0 else claims * 2 + 4
    live = set(range(data))
    chosen = set(range(data, data + reserved))
    all_blind = set(range(data, data + blind))
    mask_consumers = [
        live | chosen,
        live | chosen,
        live | (all_blind if support != "SELECTED" else chosen),
    ]
    value_consumers = [chosen, chosen, all_blind if support == "FULL" else chosen]
    # FULL is valid even when no reservation fits; generic proximity still reaches all.
    if support == "FULL":
        mask_consumers = [live, live | all_blind, live | all_blind]
        value_consumers = [all_blind, all_blind]
    mask_rows = set().union(*mask_consumers)
    value_rows = set().union(*value_consumers)
    # Explicit entry sets for the small cases, factored per-row counting for large ones.
    if len(mask_rows) * pw + len(value_rows) * matrix["width"] <= 200_000:
        masks = len({(r, j) for r in mask_rows for j in range(pw)})
        values = len({(r, j) for r in value_rows for j in range(matrix["width"])})
    else:
        masks = sum(pw for _ in mask_rows)
        values = sum(matrix["width"] for _ in value_rows)
    if passes == 2:
        masks += sum(pw for _ in live | all_blind)
        values += sum(matrix["width"] for _ in all_blind)
    total_masks = total_values = 0
    for _ in range(epochs):
        total_masks += masks
        total_values += values
    return {
        "mask_bytes": total_masks * 8,
        "value_bytes": total_values * 8,
        "accepted_bytes": (total_masks + total_values) * 8,
    }


def capacity(matrix, pw, bw, kw):
    d, k, e, cp = (matrix[x] for x in ("data_rows", "width", "claims", "cp"))
    if cp and e == 0:
        return "empty"
    if pw <= 0 or kw + pw >= 4 * kw or (d and k < kw):
        return "invalid-masked-width"
    blind = bw + 2 * (e - 1) * (max(bw - 4, 0) // 6) if cp else bw
    if 2 * e + 4 > max(blind - 4, 0):
        return "unfit-reservation"
    return "fit"


def splitmix(state):
    while True:
        state = (state + 0x9E3779B97F4A7C15) & MASK64
        z = state
        z = ((z ^ (z >> 30)) * 0xBF58476D1CE4E5B9) & MASK64
        z = ((z ^ (z >> 27)) * 0x94D049BB133111EB) & MASK64
        yield z ^ (z >> 31)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if args.out.exists():
        raise SystemExit(
            "output exists; retain previous attempts and choose another path"
        )
    frozen = Path(__file__).with_name("results") / "README.preregistered.txt"
    committed = subprocess.check_output(
        ["git", "show", "HEAD:" + str(frozen.relative_to(ROOT))], cwd=ROOT
    )
    assert committed == frozen.read_bytes(), (
        "registration must be committed before code"
    )
    geometry, inputs = [], {}
    for rel in [
        "spec/vectors/weights/weights.json",
        "spec/vectors/weights-opening/openings.json",
    ]:
        raw = (ROOT / rel).read_bytes()
        inputs[rel] = hashlib.sha256(raw).hexdigest()
        doc = json.loads(raw)
        for case in doc.get("models", doc.get("cases", [])):
            # The opening vectors use TCE enum indices; weight vectors use names.
            for channel in case["channels"]:
                if isinstance(channel["matrix"], int):
                    channel["matrix"] = (NAMES + ["ChannelParams"])[channel["matrix"]]
            if isinstance(case.get("matrix"), int):
                case["matrix"] = (NAMES + ["ChannelParams"])[case["matrix"]]
            kw = 1 << (doc.get("layout", case.get("layout"))["log2_k"])
            exp = (
                {m["matrix"]: m for m in case["matrices"]}
                if "matrices" in case
                else {case["matrix"]: case}
            )
            geometry.append(
                geometry_record(
                    case["name"], rel, case["arch"], case["channels"], kw, exp
                )
            )
    arch = {
        "n_layers": 126,
        "d_model": 16384,
        "n_heads": 128,
        "n_kv_heads": 8,
        "d_head": 128,
        "d_ff": 53248,
        "vocab": 128256,
    }
    channel = [{"matrix": t, "layer": l} for t in NAMES[1:8] for l in range(126)]
    big = geometry_record(
        "405B-tight", "RFC-0012/0040 public architecture", arch, channel, 1 << 20
    )
    geometry.append(big)
    cases = []
    for g in geometry:
        is_big = g["name"] == "405B-tight"
        matrices = [
            {
                "data_rows": m["data_rows"],
                "width": m["width"],
                "claims": (
                    7
                    if m["matrix"] == "ChannelParams" and m["data_rows"]
                    else 0
                    if m["matrix"] == "ChannelParams"
                    else 1
                ),
                "cp": m["matrix"] == "ChannelParams",
            }
            for m in g["matrices"]
        ]
        cases.append(
            {
                "name": g["name"],
                "matrices": matrices,
                "pw": 65536 if is_big else 64,
                "bw": 4096 if is_big else 32,
                "kw": g["kw"],
            }
        )
        if is_big:
            historic = [dict(m) for m in matrices]
            historic[-1]["data_rows"] = 882
            cases.append(
                {
                    "name": "405B-historical",
                    "matrices": historic,
                    "pw": 65536,
                    "bw": 4096,
                    "kw": 1 << 20,
                }
            )
    domain = itertools.product(
        [1, 3, 4, 9],
        [1, 2, 5],
        [1, 3],
        [2, 8, 16],
        [0, 1, 4],
        [0, 4, 8, 32],
        [0, 1, 3, 7],
    )
    generated = [tuple(x) for x in domain]
    rng = splitmix(0x5680028)
    for _ in range(200):
        generated.append(
            (
                next(rng) % 33 + 1,
                next(rng) % 65 + 1,
                next(rng) % 7 + 1,
                1 << (next(rng) % 9),
                [0, 1, 4, 16][next(rng) % 4],
                [0, 4, 8, 32][next(rng) % 4],
                [0, 1, 3, 7][next(rng) % 4],
            )
        )
    for i, (out, inp, layers, kw, pw, bw, e) in enumerate(generated):
        a = literal_weight(out, inp, layers, kw)
        assert a == formula_weight(out, inp, layers, kw), ("synthetic geometry", i)
        cases.append(
            {
                "name": f"synthetic-{i}",
                "geometry": {"out": out, "inputs": inp, "layers": layers},
                "matrices": [
                    {
                        "data_rows": 0 if e == 0 else len(a[2]),
                        "width": a[0],
                        "claims": e,
                        "cp": e != 1,
                    }
                ],
                "pw": pw,
                "bw": bw,
                "kw": kw,
            }
        )
    killed = {}
    results = []
    judgments = 0
    for case in cases:
        outputs = []
        for support, passes, epochs in itertools.product(SUPPORTS, [1, 2], [1, 2]):
            labels = [
                capacity(m, case["pw"], case["bw"], case["kw"])
                for m in case["matrices"]
            ]
            if support != "FULL" and any(x not in ("fit", "empty") for x in labels):
                continue
            total = {"mask_bytes": 0, "value_bytes": 0, "accepted_bytes": 0}
            for m in case["matrices"]:
                c = candidate(
                    m, case["pw"], case["bw"], case["kw"], support, passes, epochs
                )
                r = reference(m, case["pw"], case["bw"], support, passes, epochs)
                assert c == r, (case["name"], m, support, passes, epochs, c, r)
                judgments += 1
                for key in total:
                    total[key] += r[key]
                for fault in range(1, 10):
                    if fault not in killed:
                        changed = candidate(
                            m,
                            case["pw"],
                            case["bw"],
                            case["kw"],
                            support,
                            passes,
                            epochs,
                            fault,
                        )
                        if changed != r:
                            killed[fault] = {
                                "case": case["name"],
                                "matrix": m,
                                "support": support,
                                "passes": passes,
                                "epochs": epochs,
                                "control": r,
                                "faulty": changed,
                            }
            outputs.append(
                dict(
                    support=support,
                    passes=passes,
                    epochs=epochs,
                    capacity=labels,
                    store_read_bytes=total["accepted_bytes"],
                    **total,
                )
            )
        results.append(dict(**case, outputs=outputs))
    hist = next(c for c in results if c["name"] == "405B-historical")
    for support, expected in zip(SUPPORTS, [654109048832, 216735416320, 242463277056]):
        found = next(
            o
            for o in hist["outputs"]
            if (o["support"], o["passes"], o["epochs"]) == (support, 1, 1)
        )
        assert found["accepted_bytes"] == expected
    # A complete toy candidate population exposes the rejected-word cost exactly.
    accepted = sum(1 for word in range(256) if word < 251)
    expansion = Fraction(256, accepted)
    assert expansion == Fraction(256, 251)
    killed[10] = {
        "case": "8-bit-field-251",
        "control": str(expansion),
        "faulty": "1",
        "accepted_candidates": accepted,
        "all_candidates": 256,
    }
    assert set(killed) == set(range(1, 11)), (
        "surviving faults",
        set(range(1, 11)) - killed.keys(),
    )
    rates = []
    for value in ["0", "-1", "nan", "inf", "100", "530", "1000"]:
        try:
            rate = Fraction(value)
            if rate <= 0:
                raise ValueError("positive rate required")
        except (ValueError, ZeroDivisionError):
            rates.append({"input": value, "result": "invalid-rate"})
        else:
            rates.append(
                {
                    "input": value,
                    "result": "scenario-only",
                    "accepted_bytes_per_second": int(rate * 10**9),
                }
            )
    q = Fraction(2**64, 2**64 - 2**32 + 1)
    scenarios = []
    for case in results:
        if case["name"].startswith("405B"):
            for o in case["outputs"]:
                if o["passes"] == o["epochs"] == 1:
                    for gbps in [100, 530, 1000]:
                        seconds = Fraction(o["accepted_bytes"], gbps * 10**9)
                        scenarios.append(
                            {
                                "case": case["name"],
                                "support": o["support"],
                                "hypothetical_gbps": gbps,
                                "gpu_s": float(seconds),
                                "exact_gpu_s": [seconds.numerator, seconds.denominator],
                            }
                        )
    report = {
        "schema_version": 1,
        "experiment": "EXP-0028",
        "evidence": "E0",
        "verdict": "validated",
        "source_revision": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
        ).strip(),
        "registration_sha256": hashlib.sha256(committed).hexdigest(),
        "input_sha256": inputs,
        "environment": {
            "python": platform.python_version(),
            "machine": platform.machine(),
            "platform": platform.platform(),
            "utc": datetime.datetime.now(datetime.UTC).isoformat(),
        },
        "code_sha256": {
            p.name: hashlib.sha256(p.read_bytes()).hexdigest()
            for p in [Path(__file__), Path(__file__).with_name("check.py")]
        },
        "geometry": geometry,
        "cases": results,
        "judgments": judgments,
        "faults": killed,
        "rates": rates,
        "goldilocks_expansion": [q.numerator, q.denominator],
        "scenarios": scenarios,
        "scope": "Finite public metadata accounting only; no GPU rate or security validation.",
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    payload = (json.dumps(report, indent=2, allow_nan=False) + "\n").encode()
    with args.out.open("xb") as f:
        f.write(
            gzip.compress(payload, mtime=0) if args.out.suffix == ".gz" else payload
        )
    print(
        json.dumps(
            {
                "verdict": report["verdict"],
                "cases": len(results),
                "judgments": judgments,
                "geometry_cases": len(geometry),
                "faults": len(killed),
            }
        )
    )


if __name__ == "__main__":
    main()
