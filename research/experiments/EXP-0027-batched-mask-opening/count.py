#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0 OR MIT
"""EXP-0027 (d): the 405B count of mask claims, opening groups, blinding rows and proof bytes.

It reads the compiler's schedule counts (`schedule/`, run first) and the protocol calculator's
`SoftmaxRow` padded-layer reads (`crates/tessera-protocol/src/soundness/llama.rs`), and counts,
by the rules the pre-registration states:

- the masked sumchecks of `tessera-protocol/2` and their rounds;
- the tree-level mask claims (RFC-0032), their column sub-points, and the groups of B1 and B1';
- the mask block `M`'s layout, the batching cube's variables `m`, and `C`'s support;
- the proof bytes, blinding rows and committed elements before and after (live-64, ni-100);
- the soundness terms the batching adds.

    python3 -I count.py <schedule-405b.json> <count-405b.json>

Every number is E0: a count from the compiler's report and the RFCs' rules.
"""

from __future__ import annotations

import json
import math
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
COMMITTED = ROOT / "docs/perf/v0.2/commitment-report-405b-servable-c2-b32.json"
CALCULATOR = ROOT / "crates/tessera-protocol/src/soundness/llama.rs"

TOKENS = 1 << 14
CONTEXT = 1 << 14
LOG2_K = 22
K = 1 << LOG2_K
FP2_BYTES = 16
FP_BYTES = 8
GOLDILOCKS = 2**64 - 2**32 + 1
Q = GOLDILOCKS**2
PARAMS = {"live-64": {"tau": 155, "mask": 160}, "ni-100": {"tau": 193, "mask": 200}}
GROWTH_BITS = {"fixed": 0, "tokens": 14, "keys": 14, "tokens_and_context": 28}
LAYER_SUMCHECK = (22, 4)  # RFC-0068 at 405B: n = 22 rounds of degree 4.
IDENTITY = (2, 4)  # RFC-0032: two rounds of degree at most 4.
STAGE_DEGREE = 7  # RFC-0013 R4: a masking polynomial's g_j up to degree 7.
TREE_DEGREE, TREE_RAISED = 3, 7  # RFC-0010 tree levels; RFC-0032 the raised round.
SLOTS_PER_ROUND = 8


def ceil_log2(n: int) -> int:
    return 0 if n <= 1 else (n - 1).bit_length()


def calculator_tables() -> tuple[list[tuple[int, int, int]], int]:
    """The calculator's (lookups per token, rows, softmax reads per padded row) and Exp's N."""
    text = CALCULATOR.read_text()
    block = text[text.index("const TABLES"):text.index("];", text.index("const TABLES"))]
    rows = [tuple(int(x.replace("_", "")) for x in m.groups()) for m in re.finditer(r"\(([\d_]+), ([\d_]+), ([\d_]+)\)", block)]
    exp = re.search(r"const EXP_LOOKUPS: u64 = 1 << (\d+);", text)
    return rows, 1 << int(exp.group(1))


def main(argv: list[str]) -> int:
    sched_path, out_path = Path(argv[1]), Path(argv[2])
    sched = json.loads(sched_path.read_text())
    committed = json.loads(COMMITTED.read_text())
    calc, exp_lookups = calculator_tables()
    checks = {}

    # The compiler's counts against the committed report (docs/perf/v0.2).
    checks["tables_match_committed"] = sched["tables"] == committed["tables"]
    checks["stages_match_committed"] = sched["stages"] == committed["lowering"]["stages"]
    strip = lambda st: [{"stage": s["stage"], "tables": s["tables"], "residents": s["residents"]} for s in st]  # noqa: E731
    checks["stage_tables_match_committed"] = strip(sched["stage_tables"]) == strip(committed["stage_tables"])
    checks["calculator_matches_compiler"] = [(t["lookups_per_token"], t["rows"]) for t in sched["tables"]] == [(a, b) for a, b, _ in calc]

    n_layers, _d, n_heads = sched["shape"][0], sched["shape"][1], sched["shape"][2]
    layers_pad = 1 << ceil_log2(n_layers)
    heads_pad = 1 << ceil_log2(n_heads)

    # LogUp instances: one per table with lookups (RFC-0008 'Batching').
    instances = []
    for t, (_, _, softmax) in zip(sched["tables"], calc):
        if t["lookups_per_token"] == 0 and t["score_lookups"] == 0:
            continue
        if t["kind"] == "Exp":
            n = exp_lookups
        else:
            n = t["lookups_per_token"] * TOKENS + softmax * (layers_pad - n_layers) * heads_pad * TOKENS
        instances.append({"id": t["id"], "kind": t["kind"], "lookups": n, "rows": t["rows"]})

    softmax_s = ceil_log2(n_layers) + ceil_log2(n_heads) + ceil_log2(TOKENS)
    softmax_h = ceil_log2(CONTEXT)

    def tree_count(extra_height: int) -> dict:
        forests = []
        for inst in instances:
            forests.append({"kind": "fraction", "side": "lookup", "id": inst["id"], "s": 1, "h": max(2, ceil_log2(inst["lookups"]) + extra_height)})
            forests.append({"kind": "fraction", "side": "table", "id": inst["id"], "s": 1, "h": max(2, ceil_log2(inst["rows"]))})
        forests.append({"kind": "product", "side": "softmax", "id": None, "s": softmax_s, "h": softmax_h})
        height = max(f["h"] for f in forests)
        levels = []
        claims = []
        for ell in range(height):
            present = []
            for f in forests:
                t = ell - (height - f["h"])
                if 0 <= t <= f["h"] - 1:
                    n_i = f["s"] + t
                    present.append(n_i)
                    if f["kind"] == "fraction":
                        claims += [{"level": ell, "n": n_i, "c": min(n_i, LOG2_K)}] * 2
                    elif t >= 1:
                        claims.append({"level": ell, "n": n_i, "c": min(n_i, LOG2_K)})
            distinct = sorted({c["n"] for c in claims if c["level"] == ell})
            levels.append({"level": ell, "rounds": max(present), "instances": len(present), "claim_points": len(distinct)})
        return {"forests": forests, "height": height, "levels": levels, "claims": claims}

    def full_count(extra_height: int) -> dict:
        tree = tree_count(extra_height)
        sumchecks = []
        for st in sched["stage_tables"]:
            n = max(LOG2_K, max(t["log2_entries"] + GROWTH_BITS[t["growth"]] for t in st["tables"]))
            sumchecks.append({"name": f"stage {st['stage']}", "rounds": n, "degrees": [STAGE_DEGREE] * n})
        sumchecks.append({"name": "layer sumcheck (RFC-0068)", "rounds": LAYER_SUMCHECK[0], "degrees": [LAYER_SUMCHECK[1]] * LAYER_SUMCHECK[0]})
        sumchecks.append({"name": "LogUp identity (RFC-0032)", "rounds": IDENTITY[0], "degrees": [IDENTITY[1]] * IDENTITY[0]})
        for lv in tree["levels"]:
            n = lv["rounds"]
            sumchecks.append({"name": f"tree level {lv['level']}", "rounds": n, "degrees": [TREE_DEGREE] * (n - 1) + [TREE_RAISED]})
        n_sumchecks = len(sumchecks)
        rounds_total = sum(s["rounds"] for s in sumchecks)
        coefficients = sum(1 + sum(d + 1 for d in s["degrees"]) for s in sumchecks)
        slots = sum(1 + SLOTS_PER_ROUND * s["rounds"] for s in sumchecks)
        tree_claims = len(tree["claims"])
        tree_entries = sum(1 << c["c"] for c in tree["claims"])
        claims = n_sumchecks + tree_claims
        b1 = claims
        b1p = n_sumchecks + sum(lv["claim_points"] for lv in tree["levels"])
        coeff_block = 1 << ceil_log2(slots)
        rows_unpacked = tree_claims + math.ceil(coeff_block / K)
        rows_packed = math.ceil((tree_entries + coeff_block) / K)
        support = coefficients + tree_entries

        def bytes_for(groups: int, par: dict) -> int:
            return groups * (K * FP2_BYTES + par["mask"] * FP2_BYTES + FP2_BYTES + 2 * par["tau"] * FP_BYTES)

        layouts = {}
        for name, rows in (("unpacked", rows_unpacked), ("packed", rows_packed)):
            m = LOG2_K + ceil_log2(rows)
            per_param = {}
            for pname, par in PARAMS.items():
                before_b1 = bytes_for(b1, par)
                before_b1p = bytes_for(b1p, par)
                batch_messages = (2 * m + 2) * FP2_BYTES
                s_symbols = 2 * rows * par["tau"] * FP_BYTES
                after = bytes_for(1, par) + batch_messages + s_symbols
                per_param[pname] = {
                    "bytes_before_B1": before_b1,
                    "bytes_before_B1p": before_b1p,
                    "bytes_after": after,
                    "after_parts": {"one_group": bytes_for(1, par), "batching_messages": batch_messages, "S_opened_symbols": s_symbols},
                    "saving_vs_B1p": 1 - after / before_b1p,
                    "saving_vs_B1": 1 - after / before_b1,
                    "criterion_after_le_10pct_B1p": after <= 0.1 * before_b1p,
                }
            committed_s = 2 * rows * K
            layouts[name] = {
                "fp2_rows_M": rows,
                "fp_rows_S": 2 * rows,
                "m": m,
                "batching_rounds": m,
                "S_committed_elements": committed_s,
                "S_share_of_advice_rows": 2 * rows / sched["advice"]["rows"],
                "soundness_added_log2": math.log2((claims - 1 + 2 * m + 1) / Q),
                "soundness_added_terms": {"alpha": claims - 1, "rounds": 2 * m, "rho": 1},
                "verifier_c_at_r_terms": coefficients + tree_claims * m,
                "prover_batching_fp2_mults_bound": 8 * support,
                "params": per_param,
            }
        return {
            "height": tree["height"],
            "instances": len(instances),
            "fraction_forests": sum(f["kind"] == "fraction" for f in tree["forests"]),
            "softmax_forest": {"s": softmax_s, "h": softmax_h},
            "masked_sumchecks": n_sumchecks,
            "masked_sumcheck_list": [{"name": s["name"], "rounds": s["rounds"]} for s in sumchecks],
            "rounds_total": rounds_total,
            "masking_coefficients_fp2": coefficients,
            "coefficient_slots_fp2": slots,
            "tree_level_claims": tree_claims,
            "tree_level_claims_full_width": sum(c["c"] == LOG2_K for c in tree["claims"]),
            "tree_level_entries_fp2": tree_entries,
            "mask_claims": claims,
            "groups_B1": b1,
            "groups_B1p": b1p,
            "blinding_rows_before_B1": 2 * b1,
            "blinding_rows_before_B1p": 2 * b1p,
            "blinding_rows_after": 2,
            "C_support_fp2": support,
            "levels": tree["levels"],
            "layouts": layouts,
        }

    result = {
        "checks": checks,
        "inputs": {
            "schedule": sched["source"],
            "advice_rows": sched["advice"],
            "instances": instances,
            "tokens": TOKENS,
            "context": CONTEXT,
            "log2_k": LOG2_K,
        },
        "count": full_count(0),
        "sensitivity_lookup_heights_plus_1": full_count(1),
    }
    out_path.write_text(json.dumps(result, indent=1) + "\n")

    c = result["count"]
    print("checks:", checks)
    print(f"instances {c['instances']}, fraction forests {c['fraction_forests']}, softmax forest s={softmax_s} h={softmax_h}, tallest {c['height']}")
    print(f"masked sumchecks {c['masked_sumchecks']}, rounds {c['rounds_total']}, coefficients {c['masking_coefficients_fp2']} Fp2, slots {c['coefficient_slots_fp2']}")
    print(f"tree-level mask claims {c['tree_level_claims']} (full width {c['tree_level_claims_full_width']}), entries {c['tree_level_entries_fp2']} Fp2")
    print(f"mask claims {c['mask_claims']}; groups B1 {c['groups_B1']}, B1' {c['groups_B1p']}; C support {c['C_support_fp2']} Fp2")
    for name, lay in c["layouts"].items():
        lp = lay["params"]["live-64"]
        print(f"[{name}] M rows {lay['fp2_rows_M']} Fp2, m {lay['m']}, S {lay['fp_rows_S']} Fp rows ({100 * lay['S_share_of_advice_rows']:.3f}% of advice rows), "
              f"live-64 bytes before B1 {lp['bytes_before_B1'] / 2**30:.2f} GiB, B1' {lp['bytes_before_B1p'] / 2**30:.2f} GiB, after {lp['bytes_after'] / 2**20:.2f} MiB, "
              f"saving vs B1' {100 * lp['saving_vs_B1p']:.2f}%, soundness +2^{lay['soundness_added_log2']:.1f}")
    s = result["sensitivity_lookup_heights_plus_1"]
    print(f"sensitivity (+1 lookup heights): claims {s['mask_claims']}, B1 {s['groups_B1']}, B1' {s['groups_B1p']}, tallest {s['height']}")
    return 0 if all(checks.values()) else 1


if __name__ == "__main__":
    sys.exit(main(sys.argv))
