#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0 OR MIT
"""Independent integer loop/storage counts for EXP-0019, not clocks or DRAM traffic.

A tuple (s,h,lengths) specifies padded tree bits, height, public real-prefix lengths.
All sums use arbitrary precision. Counts model exactly the registered loops, not an
alternative implementation; source hashes accompany the measured run separately.
"""
from collections import Counter
import json
from pathlib import Path


def grouped(t, heads):
    out = []
    for h in range(t.bit_length() - 1, 0, -1):
        rows = [1, 2] if h == 1 else list(range((1 << (h - 1)) + 1, (1 << h) + 1))
        lengths = rows * heads
        out.append(((len(lengths) - 1).bit_length(), h, lengths))
    return out


def shapes():
    return {
        "S1": [(12, 12, list(range(1, 4097)))],
        "S2": grouped(4096, 1),
        "S3": [(10, 14, [1 << 14] * 640)],
        "S4": grouped(2048, 5),
        "U23": [(9, 14, [1 << 14] * 512)],
        "U24": [(10, 14, [1 << 14] * 1024)],
        "S1-1K": [(10, 10, list(range(1, 1025)))],
        "S2-1K": grouped(1024, 1),
    }


def count(forests):
    total = Counter()
    height = max(h for _, h, _ in forests)
    per_level = []
    total["original_real_leaves"] = sum(sum(lengths) for _, _, lengths in forests)
    total["original_padded_leaves"] = sum(1 << (s + h) for s, h, _ in forests)
    for level in range(height):
        row = Counter()
        active = [(s, h, lengths, level - (height - h) + 1)
                  for s, h, lengths in forests if height - h <= level]
        maximum = max(s + t - 1 for s, _, _, t in active)
        for s, h, lengths, t in active:
            m = t - 1
            n = s + m
            e = [(r + (1 << (h-t)) - 1) // (1 << (h-t)) for r in lengths]
            pairs = [(r + 1 >> 1, r >> 1) for r in e]
            row["candidate_stored_base_nodes"] += sum(e)
            row["production_stored_base_nodes"] += 1 << (s + t)
            row["candidate_leaf_copy_elements"] += sum(lengths) if t == h else 0
            row["production_leaf_copy_elements"] += (1 << (s+h)) if t == h else 0
            row["candidate_root_owned_alloc_bytes"] += 2 * len(lengths) * 8 if m == 0 else 0
            first_pairs = pairs if m else [(len(lengths), len(lengths))]
            row["candidate_bound_buffer_alloc_bytes"] += (sum((a+1)//2+(b+1)//2 for a,b in first_pairs)*16 if n else 0) + (2*len(lengths)*16 if m else 0)
            row["production_bound_buffer_alloc_bytes"] += (2*(1 << (n-1))*16 if n else 0)
            row["candidate_metadata_groups_created"] += len(lengths) if m else 1
            row["candidate_cached_tree_weight_multiplies"] += len(lengths) * s if m else 0
            row["production_build_base_multiplies"] += 0 if t == h else 1 << (s+t)
            row["candidate_build_base_multiplies"] += 0 if t == h else sum(e)
            row["front_padding_rounds"] += maximum - n
            row["public_inverse_attempts"] += n
            for k in range(n):
                rest = n-k-1
                eq = (1 << ((rest+1)//2)) + (1 << (rest//2))
                row["candidate_eq_table_elements_built"] += eq
                row["production_eq_table_elements_built"] += eq
                row["production_sum_input_reads"] += 4*(1 << rest)
                row["production_bind_input_reads"] += 4*(1 << rest)
                row["production_visited_pairs"] += 1 << rest
                row["production_bound_writes"] += 2 * (1 << rest)
                if k == m:
                    pairs = [(len(lengths), len(lengths))]
                unique = Counter(pairs)
                local_bits = m-k-1 if k < m else n-k-1
                for (el, ol), repeat in unique.items():
                    full = min(el//2, ol//2)
                    touched = max((el+1)//2, (ol+1)//2)
                    row["candidate_sum_input_reads"] += repeat*(el+ol)
                    row["candidate_bind_input_reads"] += repeat*(el+ol)
                    row["candidate_full_real_pairs"] += repeat * full
                    row["candidate_mixed_pairs"] += repeat * (touched-full)
                    row["candidate_visited_pairs"] += repeat * touched
                    row["candidate_bound_writes"] += repeat * ((el+1)//2 + (ol+1)//2)
                    row["candidate_boundary_folds"] += repeat * ((el%2)+(ol%2))
                    if touched < 1 << local_bits:
                        row["candidate_constant_run_contributions"] += 1
                        row["candidate_constant_pairs_not_visited"] += repeat * ((1 << local_bits)-touched)
                if k < m and len(lengths) < 1 << s:
                    row["candidate_constant_padded_tree_contributions"] += 1
                    row["candidate_constant_pairs_not_visited"] += ((1 << s)-len(lengths)) * (1 << local_bits)
                pairs = [((el+1)//2, (ol+1)//2) for el, ol in pairs]
        row["candidate_base_storage_bytes"] = row["candidate_stored_base_nodes"] * 8
        row["production_base_storage_bytes"] = row["production_stored_base_nodes"] * 8
        row["candidate_node_buffer_alloc_bytes"] = row["candidate_base_storage_bytes"] + row["candidate_root_owned_alloc_bytes"] + row["candidate_bound_buffer_alloc_bytes"]
        row["production_node_buffer_alloc_bytes"] = row["production_base_storage_bytes"] + row["production_bound_buffer_alloc_bytes"]
        row["candidate_bound_written_bytes"] = row["candidate_bound_writes"] * 16
        row["production_bound_written_bytes"] = row["production_bound_writes"] * 16
        # First bind allocates capacity for the writes; later rounds overwrite it in place.
        total.update(row)
        per_level.append({"level": level, **dict(row)})
    assert total["candidate_visited_pairs"] + total["candidate_constant_pairs_not_visited"] == total["production_visited_pairs"]
    assert total["candidate_eq_table_elements_built"] == total["production_eq_table_elements_built"]
    return {"total": dict(total), "per_level": per_level}


def main():
    assert count([(0, 1, [0])])["total"].get("candidate_visited_pairs", 0) == 0
    assert count([(0, 4, [16])])["total"]["candidate_visited_pairs"] == count([(0, 4, [16])])["total"]["production_visited_pairs"]
    result = {"evidence": "E0 exact integer reconstruction; writes are logical, not DRAM or allocator bytes",
              "unpriced": ["comparison/sort instructions", "public metadata allocator overhead", "tail bit recursion branches", "hash cost", "all allocator metadata/round-output allocations (node-buffer allocations above are an exact subset)"],
              "fallback": "inverse attempts counted; normal/scripted dynamic fallback counts recorded by diagnostics, never inferred from clock",
              "cases": {name: count(fs) for name, fs in shapes().items()}}
    Path("results/count.json").write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    for name, case in result["cases"].items():
        print(name, json.dumps(case["total"], sort_keys=True))


if __name__ == "__main__":
    main()
