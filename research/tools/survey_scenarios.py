#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0 OR MIT
"""Current E0 scenarios requested by SUR-0005/0008; no measured GPU costs."""

from dataclasses import asdict, replace
import importlib.util
import json
import math
from pathlib import Path
import sys

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location("cost_model", HERE / "cost_model.py")
model = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = model
spec.loader.exec_module(model)


def main():
    base = model.PROFILES["counted"]
    scenarios = []
    for cost in (150, 240, 360):
        for couple_p5 in (False, True):
            p = replace(base, commit_ps=cost,
                        column_answer_ps=cost if couple_p5 else base.column_answer_ps)
            scenarios.append({"commit_ps": cost, "P5_full_reencode_coupled": couple_p5,
                              "parameters": asdict(p),
                              "rows": [asdict(row) for row in model.table(p)]})
    # RFC-0013 Cost gives an approximate mask count for one 16K-token proof.
    # This is an additive subset, not a full ZK accounting or a global factor.
    mask_elements = 16_000_000_000
    p = replace(base, commits_per_token=base.commits_per_token + mask_elements/base.batch_tokens)
    original, masked = model.mu(base, 16384), model.mu(p, 16384)
    delta = masked.total - original.total
    seconds = mask_elements * (base.glue*base.commit_ps + base.column_answer_ps) * 1e-12
    assert math.isclose(delta, seconds/(base.batch_tokens*original.native_s_per_token),
                        rel_tol=1e-12), "mask increment must include commitment and P5, without lookups"
    for i in range(0, len(scenarios), 2):
        fixed, coupled = scenarios[i:i+2]
        for a, b in zip(fixed["rows"], coupled["rows"], strict=True):
            expected = a["column_answers"] * (coupled["commit_ps"]/base.column_answer_ps - 1)
            assert math.isclose(b["total"]-a["total"], expected, rel_tol=1e-12)
    table_sharing = []
    for units in (1, 2, 4, 8):
        repeated = model.mu(replace(base, table_rows=base.table_rows*units), 16384)
        credit = repeated.total-original.total
        assert math.isclose(credit, (units-1)*original.tables, rel_tol=1e-12, abs_tol=1e-13)
        table_sharing.append({"units": units, "M_per_unit": base.table_rows,
                              "per_unit_tables_partial_mu": repeated.total,
                              "shared_table_partial_mu": original.total,
                              "table_only_credit_mu": credit,
                              "table_only_credit_fraction": credit/repeated.total})
    print(json.dumps({"evidence": "E0 conditional planning scenarios",
                      "normative_context": "main709c37c; compiler report from c2c8a69",
                      "baseline_parameters": asdict(base), "commitment_scenarios": scenarios,
                      "table_sharing_diagnostic_16K": table_sharing,
                      "zk_mask_subset_16K": {"estimated_base_field_elements": mask_elements,
                          "count_is_estimate_not_compiler_output": True,
                          "commitment_plus_P5_seconds": seconds, "delta_mu": delta,
                          "fraction_of_partial_baseline": delta/original.total,
                          "extra_sumcheck_work_and_weight_budget_not_priced": True},
                      "limitations": ["No H100 measurement or total-time bound",
                          "P4, remaining streaming and unallocated range tables remain unpriced",
                          "ZK subset is separate from 1.15 sumcheck glue; no universal 1.001 factor",
                          "Table-sharing diagnostic holds all other terms fixed, not a feasible unit plan or net gain",
                          "Mask count depends on stage schedule and must be replaced by compiler output"]}, indent=2))


if __name__ == "__main__":
    main()
