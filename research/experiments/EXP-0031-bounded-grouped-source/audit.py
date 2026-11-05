#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0 OR MIT
"""Independent finite integer/index recount; does not import the Rust candidate.

Requires blake3==1.0.11 for the ordered synthetic-column digest. Allocation
observations are measured producer inputs, not independently measured by Python.
"""

import argparse
import hashlib
import json
from pathlib import Path
import blake3

SEED = 0x5EED0031
MASK = (1 << 64) - 1
LENGTHS = [[1], [3], [5], [2, 1], [3, 2], [5, 1, 3]]
LAYERS = [2, 3, 5, 2, 3, 5]
PAIRS = [(0, 1), (1, 0), (1, 1), (2, 1)]


def width(n):
    return (n - 1).bit_length()


def padded(n):
    return 1 << width(n)


def mix(x):
    x = (x + 0x9E3779B97F4A7C15) & MASK
    x = ((x ^ (x >> 30)) * 0xBF58476D1CE4E5B9) & MASK
    x = ((x ^ (x >> 27)) * 0x94D049BB133111EB) & MASK
    return x ^ (x >> 31)


def spec(site, lookup, tokens):
    if site == 0:
        return [5], None, 1, 5, None
    if site == 2:
        return [tokens, 2], 0, 2, 6, None
    return [
        ([tokens, 3], 0, 1, 0, (1, [1])),
        ([2, tokens, 3], 1, 2, 3, (4, [0, 2])),
        ([3], None, 1, 2, None),
    ][lookup]


def natural_coordinates(domain, index):
    result = [0] * len(domain)
    for axis in reversed(range(len(domain))):
        w = width(domain[axis])
        result[axis] = index % (1 << w)
        index //= 1 << w
    assert index == 0, "full index outside padded domain"
    return result


def datum(case, site, layer, term, axes, domain, token_axis, coordinates):
    index = 0
    real = True
    for axis in axes:
        index = index * padded(domain[axis]) + coordinates[axis]
        real &= coordinates[axis] < domain[axis]
        if axis == token_axis:
            r, position = divmod(coordinates[axis], case["slot"])
            real &= r < len(case["lengths"]) and position < case["lengths"][r]
    if not real:
        return 0
    return mix(SEED ^ (case["fixture"] << 48) ^ (site << 40) ^ (layer << 32) ^ (term << 24) ^ index) & 3


def permutation(height, position):
    if height <= 1:
        return position
    upper = f"{position >> 1:0{height - 1}b}"[::-1]
    return (int(upper, 2) << 1) + position % 2


def grow(n, size):
    if not n:
        return 0
    minimum = 8 if size == 1 else 4 if size <= 1024 else 1
    return 4 * max(n, minimum) * size


def audit(data):
    assert data["stage"] == "source-prototype-only"
    assert data["registered_criteria_complete"] is False
    assert data["production_admission"] is False and data["real_model_acceptance"] is False
    expected_cases = set()
    expected_refusals = set()
    for fixture, lengths in enumerate(LENGTHS):
        slot = padded(max(lengths))
        tokens = lengths[0] if len(lengths) == 1 else len(lengths) * slot
        for height in [1, 2]:
            for gt, gl in PAIRS:
                key = (fixture, height, gt, gl)
                (expected_cases if gt <= width(tokens) and gl <= width(LAYERS[fixture]) else expected_refusals).add(key)
    actual_cases = {(c["fixture"], c["forest"]["height"], c["gt"], c["gl"]) for c in data["cases"]}
    assert actual_cases == expected_cases and len(actual_cases) == len(data["cases"])
    assert {(c["fixture"], c["height"], c["gt"], c["gl"]) for c in data["refusals"]} == expected_refusals
    groups = rows = reductions = 0
    for case in data["cases"]:
        assert case["lengths"] == LENGTHS[case["fixture"]]
        assert case["layers"] == LAYERS[case["fixture"]]
        assert case["slot"] == padded(max(case["lengths"]))
        tokens = case["lengths"][0] if len(case["lengths"]) == 1 else len(case["lengths"]) * case["slot"]
        assert case["tokens"] == tokens
        gt, gl = case["gt"], case["gl"]
        assert len(case["groups"]) == 1 << (gt + gl)
        forest = case["forest"]
        assert forest["queries"] == forest["height"]
        assert forest["layers"] == forest["queries"] + width(tokens)
        assert forest["vars"] == forest["layers"] + width(case["layers"]) + 1
        full_payload = (8 + case["layers"] * (12*padded(tokens)+16) + 2*padded(tokens))*8 + (1 << forest["vars"])*8
        assert full_payload <= case["baseline_source_retained"] <= case["baseline_source_construction_peak"], "baseline constructor/retained observation"
        earlier = 0
        for h, group in enumerate(case["groups"]):
            groups += 1
            assert group["h"] == h
            subset, layer_subset = h % (1 << gt), h // (1 << gt)
            tables = {}
            for site in range(3):
                for layer in range(case["layers"] if site == 1 else 1):
                    if (layer % (1 << gl) if site == 1 else 0) != layer_subset:
                        continue
                    for lookup in range(3 if site == 1 else 1):
                        domain, token_axis, table, _, _ = spec(site, lookup, tokens)
                        if token_axis is None and subset:
                            continue
                        full_vars = sum(map(width, domain))
                        variables = full_vars - (gt if token_axis is not None else 0)
                        tables.setdefault(table, []).append({"site":site,"layer":layer,"lookup":lookup,"vars":variables})
            expected_instances = []
            for table, blocks in sorted(tables.items()):
                blocks.sort(key=lambda b: (-b["vars"], b["site"], b["layer"], b["lookup"]))
                offset = 0
                for block in blocks:
                    block["offset"] = offset
                    offset += 1 << block["vars"]
                expected_instances.append({"table":table,"height":width(offset),"entries":offset,"blocks":blocks})
            assert group["instances"] == expected_instances, "held roles, ordering or offsets"
            expected_records = []
            digest = blake3.blake3()
            column_bytes = grow(len(tables), 56)
            payload = 0
            row_count = 0
            for instance in expected_instances:
                table = instance["table"]
                count = 1 << instance["height"]
                keys = [0] * count
                values = [3] * count if table == 2 else None
                column_bytes += grow(count, 8) * (2 if table == 2 else 1)
                payload += count * 8 * (2 if table == 2 else 1)
                for block in instance["blocks"]:
                    domain, token_axis, _, term, class_term = spec(block["site"], block["lookup"], tokens)
                    # Independent oracle: enumerate full coordinates then select the subset.
                    candidates = []
                    for original in range(1 << sum(map(width, domain))):
                        coordinates = natural_coordinates(domain, original)
                        if token_axis is None or coordinates[token_axis] % (1 << gt) == subset:
                            candidates.append((original, coordinates))
                    assert len(candidates) == 1 << block["vars"]
                    for position, (original, coordinates) in enumerate(candidates):
                        target = permutation(instance["height"], block["offset"] + position)
                        key = datum(case, block["site"], block["layer"], term, range(len(domain)), domain, token_axis, coordinates)
                        if class_term:
                            ct, axes = class_term
                            key += 4 * datum(case, block["site"], block["layer"], ct, axes, domain, token_axis, coordinates)
                        keys[target] = key
                        if values is not None:
                            values[target] = 2 * key + 3
                        expected_records.append({"group":h,"table":table,"site":block["site"],"layer":block["layer"],"lookup":block["lookup"],"position":position,"original":original,"leaf":target})
                        row_count += 1
                digest.update(table.to_bytes(4, "little"))
                digest.update(count.to_bytes(8, "little"))
                digest.update(bytes([values is not None]))
                for value in keys + (values or []):
                    digest.update(value.to_bytes(8, "little"))
            assert group["indices"] == expected_records, "source index transport"
            assert group["column_digest"] == digest.hexdigest(), "independent canonical column values/padding"
            forest_records = [{"group":h,"position":j,"original":x} for j, x in enumerate(x for x in range(1 << forest["vars"]) if (x >> forest["queries"]) % (1 << gt) == subset and (x >> forest["layers"]) % (1 << gl) == layer_subset)]
            assert group["forest_indices"] == forest_records, "forest source indices"
            n_forest = len(forest_records)
            records = grow(row_count, 64) + grow(n_forest, 24)
            expected_bill = {"columns":column_bytes,"forest":grow(n_forest,8),"records":records,"earlier":earlier,"peak":column_bytes+grow(n_forest,8)+records+earlier,"source_payload":payload+n_forest*8,"row_records":row_count,"forest_records":n_forest}
            assert group["bill"] == expected_bill, "public capacity/lifetime recount"
            assert group["baseline_source_payload"] == full_payload, "complete resident source payload"
            assert group["observed_peak"] <= expected_bill["peak"], "measured allocation exceeds bill"
            earlier += records
            rows += row_count + n_forest
        reductions += max(g["bill"]["source_payload"] for g in case["groups"]) < full_payload
    assert reductions > 0 and data["source_payload_reduction_cases"] == reductions, "strict maximum source payload reduction"
    return {"source_cases":len(actual_cases),"groups":groups,"index_records":rows,"negative_shapes":len(expected_refusals),"source_payload_reduction_cases":reductions,"scope":"independent integer/index/value-digest recount; measured allocator inputs and pinned Rust ABI sizes are producer observations; full experiment unjudged"}


def audit_merge(data, source):
    assert data["stage"] == "merge-prototype-only"
    assert data["registered_criteria_complete"] is False
    assert data["production_admission"] is False and data["real_model_acceptance"] is False
    source_cases = {(c["fixture"], c["forest"]["height"], c["gt"], c["gl"]): c for c in source["cases"]}
    assert len(data["cases"]) == len(source_cases)
    cuts_checked = 0
    for case in data["cases"]:
        key = (case["fixture"], case["height"], case["gt"], case["gl"])
        current = source_cases.pop(key)
        gt, gl = case["gt"], case["gl"]
        token_vars = width(current["tokens"])
        shapes = [{"vars":3,"group_vars":[]}]
        identities = [(0,0,5)]
        for layer in range(current["layers"]):
            shapes.extend([
                {"vars":token_vars+2,"group_vars":list(range(2,2+gt))},
                {"vars":2,"group_vars":[]},
                {"vars":token_vars+3,"group_vars":list(range(2,2+gt))},
                {"vars":3,"group_vars":[]},
                {"vars":2,"group_vars":[]},
            ])
            identities.extend((1,layer,t) for t in [0,1,3,4,2])
        shapes.append({"vars":token_vars+1,"group_vars":list(range(1,1+gt))})
        identities.append((2,0,6))
        forest = current["forest"]
        shapes.append({"vars":forest["vars"],"group_vars":list(range(forest["queries"],forest["queries"]+gt))+list(range(forest["layers"],forest["layers"]+gl))})
        assert case["shapes"] == shapes, "original polynomial identity/variable order"
        expected_claims = []
        for h in range(1 << (gt + gl)):
            token_subset, layer_subset = h % (1 << gt), h // (1 << gt)
            expected = []
            for i, (site, layer, term) in enumerate(identities):
                if (layer % (1 << gl) if site == 1 else 0) != layer_subset:
                    continue
                if (site == 0 or term == 2) and token_subset:
                    continue
                expected.append(i)
            expected.append(len(shapes)-1)
            expected_claims.append(expected)
        assert case["claims"] == expected_claims, "merge held-claim order"
        assert case["full_source_calls_during_group"] == 0, "original sources cannot materialize during group pass"
        maximum = max(p["vars"] for p in shapes)
        message_vars=max(maximum,4,max(instance["height"] for group in current["groups"] for instance in group["instances"]))
        assert case["message_vars"]==message_vars,"message dimension upper bound"
        full_max = 1 << maximum
        callback = grow(full_max,8)
        # Production component totals below are pinned producer inputs; this audit
        # independently recounts their public composition, not their Rust internals.
        group_max = max(1 << (p["vars"]-len(p["group_vars"])) for p in shapes)
        input_bill = case["group_layout_bill"] + case["driver_bill"] + grow(group_max,8) + grow(group_max,16) + grow(2*maximum,16)
        assert case["input_bill"] == input_bill, "merge input/callback composition"
        assert 0 <= case["input_retained"] <= case["input_observed_peak"] <= input_bill
        largest = maximum - max(len(p["group_vars"]) for p in shapes)
        assert [c["cut"] for c in case["cuts"]] == list(range(largest+1)), "every legal cut"
        block = max(grow(len(group),40) + sum(grow(1 << (shapes[i]["vars"]-len(shapes[i]["group_vars"])),8) for i in group) for group in expected_claims)
        transcripts = set()
        for cut in case["cuts"]:
            cuts_checked += 1
            assert cut["callback_bill"] == callback, "original P2b source/copy charge"
            assert cut["block_bill"] == block, "group callback block overlap"
            assert cut["construction_bill"] == cut["engine_construction"] + block
            assert cut["proving_bill"] == cut["engine_proving"] + callback
            assert cut["construction_observed"] <= cut["construction_bill"]
            assert 0 <= cut["kept_observed"] <= cut["kept_bill"]
            assert cut["proving_observed"] <= cut["proving_bill"]
            assert cut["regeneration_calls"] == [1]*len(shapes), "exact original regeneration count"
            transcripts.add(cut["transcript_digest"])
        assert len(transcripts) == 1, "merge transcript invariant across cuts"
    assert not source_cases, "no omitted source case at merge"
    return {"merge_cases":len(data["cases"]),"legal_cuts":cuts_checked,"scope":"independent public source/polynomial/claim/cut/callback composition; production MergeMemory and layout component totals and allocator observations are pinned producer inputs"}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("out", type=Path)
    parser.add_argument("--merge", type=Path)
    args = parser.parse_args()
    raw = args.source.read_bytes()
    result = audit(json.loads(raw))
    if args.merge:
        merge_raw = args.merge.read_bytes()
        result["merge"] = audit_merge(json.loads(merge_raw), json.loads(raw))
        result["merge_source_sha256"] = hashlib.sha256(merge_raw).hexdigest()
    result["source_sha256"] = hashlib.sha256(raw).hexdigest()
    args.out.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result))


if __name__ == "__main__":
    main()
