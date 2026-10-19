#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0 OR MIT
"""Independently reconstruct EXP21 row domains and scalar judgment counts."""
from collections import Counter
from itertools import product
from pathlib import Path
import hashlib
import json


def main():
    results = Path(__file__).resolve().parent / "results"
    raw = results / "attempts/20261019T105827.081336Z-45083/summary.json"
    run = json.loads(raw.read_text())
    logical, physical, by_size = Counter(), Counter(), Counter()
    seen = set()
    for size in (1, 2, 4):
        windows = sorted(set(range(1, size + 1)) | {size + 1, 1024})
        for n, window, query, scores, layers, heads in product(
            range(size + 1), windows, range(size),
            product((-1, 0, 1), repeat=size), (1, 3), (1, 3)
        ):
            padded_layers = 1 << (layers - 1).bit_length()
            padded_heads = 1 << (heads - 1).bit_length()
            for layer, head in product(range(padded_layers), range(padded_heads)):
                real = layer < layers and head < heads and query < n
                selected = tuple(real and i <= query and query - i < window for i in range(size))
                maximum = max((x for x, keep in zip(scores, selected) if keep), default=0)
                candidates = 7 + (maximum == -1)
                counts = {
                    "row/contexts": 1,
                    "row/e_probes": 5 * size * candidates,
                    "row/c_probes": candidates,
                    "row/R_probes": 8,
                    "row/nonbase_R_probes": 2,
                    "judgments/pointwise": candidates * (5 * size + 1) + size + 17,
                }
                logical.update(counts)
                key = (scores, selected, real)
                if key not in seen:
                    seen.add(key)
                    physical.update(counts)
                    by_size[size] += 1
    logical["judgments/pointwise"] += 1
    physical["judgments/pointwise"] += 1
    for name, value in logical.items():
        assert run["counts"][name] == value, (name, value, run["counts"][name])
    for name in ("e_probes", "c_probes", "R_probes", "nonbase_R_probes"):
        assert run["counts"]["physical_row/" + name] == physical["row/" + name]
    assert len(seen) == 933 and physical["judgments/pointwise"] == 156570
    assert run["counts"]["cached_logical_judgments"] == (
        logical["judgments/pointwise"] - physical["judgments/pointwise"]
    )
    output = {
        "kind": "independent parent combinatorial reconstruction; no candidate module imported",
        "source": "public frozen domains and explicit scalar judgment inventory",
        "raw_summary_sha256": hashlib.sha256(raw.read_bytes()).hexdigest(),
        "logical_counts": dict(logical), "physical_counts": dict(physical),
        "unique_inputs_by_N": dict(by_size),
        "row_core_judgment_formula": "C*(5*N+1)+N+17; final context-count judgment +1",
        "all_match": True,
    }
    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()
