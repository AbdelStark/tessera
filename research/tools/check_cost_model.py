#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0 OR MIT
"""Differential check against tessera-bench, including P4/P5, tables and memory boundaries."""

import importlib.util
import math
from pathlib import Path
import subprocess
import sys
from dataclasses import replace

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location("cost_model", HERE / "cost_model.py")
model = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = model
spec.loader.exec_module(model)


def main():
    manifest = HERE / "model-oracle/Cargo.toml"
    subprocess.run(["cargo", "build", "--quiet", "--locked", "--manifest-path", str(manifest)], check=True)
    binary = HERE / "model-oracle/target/debug/research-model-oracle"
    fields = ("batch_tokens", "commit_ps", "lookup_ps", "leaf_ps", "glue", "weight_pass_gpu_s",
              "native_rate", "stream_pass_ps_per_score", "column_answer_ps", "row_combination_ps",
              "commits_per_token", "lookups_per_token", "scores_per_token_per_ctx", "native_ops_lin",
              "native_ops_att_per_ctx", "regen_full_equivalents", "table_rows", "score_block", "free_bytes",
              "bytes_per_leaf")
    terms = ("linear", "attention", "weights", "regen", "streaming", "column_answers",
             "row_combination", "tables", "total", "streamed_passes")
    checked = 0
    for base in model.PROFILES.values():
        for p in (base, replace(base, row_combination_ps=17, column_answer_ps=23, table_rows=2**32,
                                batch_tokens=4096, score_block=512, stream_pass_ps_per_score=7),
                  replace(base, free_bytes=base.scores_per_token_per_ctx * 1024 * base.batch_tokens * 32),
                  replace(base, free_bytes=base.scores_per_token_per_ctx * 1024 * base.batch_tokens * 32 - 1)):
            for context in (0, 16, 512, 1024, 16384, 131072):
                values = [str(int(p.batch_tokens)) if f == "batch_tokens" else str(getattr(p, f)) for f in fields]
                raw = subprocess.check_output([str(binary), str(context), *values], text=True)
                expected = list(map(float, raw.split()))
                got = model.mu(p, context)
                for term, want in zip(terms, expected, strict=True):
                    assert math.isclose(getattr(got, term), want, rel_tol=2e-13, abs_tol=1e-13), (term, got, want)
                checked += 1
    for change in ({"lookup_ps": -1}, {"commit_ps": math.nan}, {"native_rate": 0},
                   {"batch_tokens": 0}, {"free_bytes": 0}):
        try:
            model.mu(replace(model.Params(), **change), 1024)
        except ValueError:
            pass
        else:
            raise AssertionError(f"accepted invalid model input: {change}")
    print(f"ok: {checked} scenarios match normative tessera-bench; invalid inputs rejected")


if __name__ == "__main__":
    main()
