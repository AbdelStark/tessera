#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0 OR MIT
"""E0 exact eq-table loop counts, not complete kernel counts or time prices."""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent
rows = []
for s, h in [(8, 14), (10, 12), (0, 22), (6, 14), (7, 14)]:
    lengths = [1 << (s + level) for level in range(h)]
    # eq_table grows 1 -> 2^n; each old entry requires one multiplication
    # and one subtraction. Each bound public table emits 2^n-1 folded words.
    eq_work = sum(length - 1 for length in lengths)
    assert eq_work == (1 << (s + h)) - (1 << s) - h
    def split_build(n):
        return sum((1 << ((rest + 1) // 2)) + (1 << (rest // 2)) - 2
                   for rest in range(n))
    # level::State::add_round builds these separately for every active instance.
    # The forest's root-claim pass also builds one split table at its s-variable root.
    root_build = (1 << ((s + 1) // 2)) + (1 << (s // 2)) - 2
    production_s = root_build + sum(split_build(s + level) for level in range(h))
    production_f = sum(split_build(level) for level in range(s + h))
    rows.append({
        "shape": f"s{s}h{h}", "judged": s + h == 22,
        "production_split_eq_build_multiplications_S": production_s,
        "production_split_eq_build_multiplications_F": production_f,
        "production_split_eq_build_multiplications_B": production_s + production_f,
        "production_split_eq_build_multiplications_saved_vs_separate": 0,
        "level_table_lengths": lengths,
        "eq_build_multiplications_P": eq_work,
        "eq_build_multiplications_R": eq_work,
        "eq_build_multiplications_T": eq_work,
        "eq_build_multiplications_saved_vs_separate": eq_work,
        "eq_build_subtractions_saved_vs_separate": eq_work,
        "public_eq_fold_output_words_P": eq_work,
        "public_eq_fold_output_words_R": eq_work,
        "public_eq_fold_output_words_T": 2 * eq_work,
        "public_eq_fold_output_words_saved_vs_separate": 0,
        "derived_product_multiplications_per_input_type":
            (1 << (s + h)) - (1 << (s + 1)),
    })
(ROOT / "results/count.json").write_text(json.dumps({
    "evidence": "E0", "scope": "table build/fold loop counts; production split-eq and separate literal generic proxy",
    "source": "prover::State::new owns bound buffers; State::bind binds each read table in each state",
    "exclusions": ["kernel evaluation-node arithmetic", "allocations", "cache effects", "production split-eq pricing"],
    "rows": rows,
}, indent=2) + "\n")
print("E0: production builds additive; proxy saves one eq build and zero folding outputs; no GPU pricing")
