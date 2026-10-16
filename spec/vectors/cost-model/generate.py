#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0 OR MIT
"""Reference data for the executable cost model (RFC-0020 'Executable cost model', 08 §1): the
source's projections with the lines they come from, the inputs they rest on with their provenance,
and the model of 08 §1 evaluated on those inputs.

The source is `proofs-of-inference-sota-path-forward.md` at the repository root. Its §5 states the
cost model, the unit costs and the counts, and charts the projected proof multiplier (lines
95-134); its §8 tabulates proof time, sampling rate and detection probability (lines 238-249). The
figures below are copied from those lines. `tessera-bench`'s tests check the copy against the
document, so a typo here or an edit there fails `cargo test`.

This script is the independent oracle: it shares no code with crates/ and evaluates 08 §1 term by
term in Python's floating point,

    linear    = (1 + glue)·(C·c_c + K·c_ℓ)            / native
    attention = (1 + glue)·A·S·(c_c + c_ℓ + c_π)       / native
    weights   = (W/T)                                 / native,   native = (F_lin + F_att·S)/R,

with neither regeneration nor streaming, which the source's model does not have (RISK-9,
RISK-10). Contexts are binary: 1K is 1,024 tokens.

Provenance of the inputs:
- stated: c_c, c_ℓ, c_π, the glue, R, W and T (line 101); 137,216 committed elements and lookups
  per layer per token (line 97); 128 heads, 126 layers and 512 operations per score (line 99);
- derived from stated numbers: C = K = 137,216 × 126, A = 128 × 126 / 2 = 8,064, F_att = 512·A;
- reconstructed: F_lin, which the source writes as 8.1×10^11 (line 97), two significant digits.
  Two operations per parameter of Llama-3.1-405B, 2 × 405,853,388,800, is the reading that
  reproduces the chart: the attention column alone, which does not depend on C, pins F_lin within
  0.1% of it (checked below), while 8.1×10^11 misses that column by 0.11.

Known deviation: the chart's linear column corresponds to about 1.76×10^7 committed elements per
token (least squares, checked below), 1.8% above the stated 137,216 × 126 = 1.729×10^7. The model
keeps the stated count, and its linear term stays within 0.036 of the chart, inside the golden
test's ±0.05.

§8 follows from its own μ column, rounded to one decimal: the sampling rate for an amortized
overhead o at the prefill-to-serving ratio η is s = o/(μ·η) (RFC-0016 'Calculators'), the daily
detection of v violating requests is 1 − (1 − s)^v, and the proof time of a request of S tokens on
G GPUs is μ·S·(F_lin + F_att·S)/(G·R). From the table's μ these reproduce every printed figure;
from the model's μ, 16K detection would be 83.41% against the printed 83.3%, so the golden test
checks the calculators from the table's μ and the model against the chart.

Usage: python3 spec/vectors/cost-model/generate.py
(rewrites cost-model.json)
"""

import json
from decimal import ROUND_HALF_UP, Decimal
from pathlib import Path

HERE = Path(__file__).resolve().parent
SOURCE = "proofs-of-inference-sota-path-forward.md"

LAYERS, HEADS = 126, 128
PER_LAYER = 137_216
PARAMETERS = 405_853_388_800
OPS_PER_SCORE = 512
PEAK_INT8_OPS = 1979e12

INPUTS = {
    # name: (value, provenance, line, the line's own words, how the value follows from them)
    "commit_ps": (30.0, "stated", 101, "30 ps per committed 64-bit element", None),
    "lookup_ps": (40.0, "stated", 101, "40 ps per LogUp-GKR lookup", None),
    "product_leaf_ps": (12.0, "stated", 101, "12 ps per product-tree leaf", None),
    "glue_fraction": (0.15, "stated", 101, "+15% for sumcheck glue", None),
    "native_ops_per_s": (
        0.6 * PEAK_INT8_OPS,
        "stated",
        101,
        "native INT8 at 60% of 1,979 TOPS",
        "per GPU",
    ),
    "weight_pass_gpu_s": (
        0.24,
        "stated",
        101,
        "one 0.24 GPU-s weight pass per 16K-token proof batch",
        None,
    ),
    "batch_tokens": (16384.0, "stated", 101, "16K-token proof batch", "16K = 16,384"),
    "commits_per_token": (
        float(PER_LAYER * LAYERS),
        "derived",
        97,
        "137,216 per layer per token",
        "times the 126 layers of line 99",
    ),
    "lookups_per_token": (
        float(PER_LAYER * LAYERS),
        "derived",
        97,
        "one committed element and one lookup each",
        "137,216 per layer per token times 126 layers, as commits_per_token",
    ),
    "scores_per_token_per_ctx": (
        float(HEADS * LAYERS // 2),
        "derived",
        99,
        "128 heads × S/2 × 126 layers ≈ 8,064·S per token",
        None,
    ),
    "native_ops_att_per_ctx": (
        float(OPS_PER_SCORE * HEADS * LAYERS // 2),
        "derived",
        99,
        "Native work per score is 512 tensor operations",
        "512 times scores_per_token_per_ctx",
    ),
    "native_ops_lin": (
        float(2 * PARAMETERS),
        "reconstructed",
        97,
        "against 8.1×10^11 native INT8 operations",
        "two operations per parameter of Llama-3.1-405B, 2 × 405,853,388,800: the reading "
        "that reproduces the chart",
    ),
}

CONTEXTS = [1024 << k for k in range(8)]
# Lines 113-115, in the caption's order (line 118): total, attention, linear layers.
CHART_TOTAL = [3.21, 4.32, 6.51, 10.76, 18.75, 33.04, 56.26, 88.73]
CHART_ATTENTION = [1.13, 2.25, 4.46, 8.75, 16.82, 31.24, 54.68, 87.48]
CHART_LINEAR = [2.06, 2.05, 2.03, 1.99, 1.91, 1.78, 1.56, 1.24]
# Lines 127-134: the chart-data table's weight pass and its total to one decimal.
TABLE_WEIGHT_PASS = [0.021, 0.021, 0.021, 0.021, 0.020, 0.018, 0.016, 0.013]
TABLE_TOTAL = [3.2, 4.3, 6.5, 10.8, 18.8, 33.0, 56.3, 88.7]

# Lines 244-247 and the assumptions of lines 242 and 249.
GPUS, ETA, TARGET_OVERHEAD, VIOLATIONS = 8, 0.3, 0.1, 100
SECTION_8 = [
    # context, μ, proof time (s), its half unit (s), sampling (%), detection (%), detection approximate
    (1024, 3.2, 0.3, 0.05, 10.4, 100.0, True),
    (4096, 6.5, 2.3, 0.05, 5.1, 99.5, False),
    (16384, 18.8, 28.5, 0.05, 1.8, 83.3, False),
    (131072, 88.7, 28 * 60.0, 30.0, 0.38, 31.4, False),
]
CHART_TOLERANCE = 0.05
WEIGHT_PASS_TOLERANCE = 0.0005
PERCENTAGE_POINTS = 0.1


def value(name):
    return INPUTS[name][0]


def native(context):
    return (value("native_ops_lin") + value("native_ops_att_per_ctx") * context) / value(
        "native_ops_per_s"
    )


def model(context):
    """The terms of 08 §1, without regeneration or streaming, in units of the native time."""
    ps = 1e-12
    glue = 1.0 + value("glue_fraction")
    linear = glue * (
        value("commits_per_token") * value("commit_ps")
        + value("lookups_per_token") * value("lookup_ps")
    ) * ps
    per_score = value("commit_ps") + value("lookup_ps") + value("product_leaf_ps")
    attention = glue * value("scores_per_token_per_ctx") * context * per_score * ps
    weights = value("weight_pass_gpu_s") / value("batch_tokens")
    n = native(context)
    return {
        "context": context,
        "linear": linear / n,
        "attention": attention / n,
        "weights": weights / n,
        "total": (linear + attention + weights) / n,
    }


def calculators(context, mu):
    sampling = TARGET_OVERHEAD / (mu * ETA)
    return {
        "context": context,
        "mu": mu,
        "proof_time_s": mu * context * native(context) / GPUS,
        "sampling_rate": sampling,
        "detection": 1.0 - (1.0 - sampling) ** VIOLATIONS,
    }


def one_decimal(x):
    """`x` rounded to one decimal, halves up, as the source rounds 18.75 to 18.8."""
    return float(Decimal(repr(x)).quantize(Decimal("0.1"), rounding=ROUND_HALF_UP))


def check(rows, calcs):
    """The claims of the docstring, checked before anything is written."""
    for k, row in enumerate(rows):
        assert abs(row["linear"] - CHART_LINEAR[k]) <= 0.036, (k, "linear")
        assert abs(row["attention"] - CHART_ATTENTION[k]) <= CHART_TOLERANCE, (k, "attention")
        assert abs(row["total"] - CHART_TOTAL[k]) <= CHART_TOLERANCE, (k, "total")
        assert abs(row["weights"] - TABLE_WEIGHT_PASS[k]) <= WEIGHT_PASS_TOLERANCE, (k, "weights")
        assert one_decimal(CHART_TOTAL[k]) == TABLE_TOTAL[k], (k, "the table rounds the chart")
    for (context, mu, time, half, sampling, detection, _), calc in zip(SECTION_8, calcs):
        assert TABLE_TOTAL[CONTEXTS.index(context)] == mu, "§8's μ is the chart's, one decimal"
        assert abs(calc["proof_time_s"] - time) <= half, (context, "proof time")
        assert abs(100 * calc["sampling_rate"] - sampling) <= PERCENTAGE_POINTS, (context, "sampling")
        assert abs(100 * calc["detection"] - detection) <= PERCENTAGE_POINTS, (context, "detection")
    # The attention column pins F_lin: att(S) = b·S/(F + G·S) with b known, so two contexts give
    # F + G·S at each, and the 64K and 128K points give F.
    per_score = value("commit_ps") + value("lookup_ps") + value("product_leaf_ps")
    b = (1 + value("glue_fraction")) * value("scores_per_token_per_ctx") * per_score * 1e-12
    b *= value("native_ops_per_s")
    d64, d128 = (b * CONTEXTS[k] / CHART_ATTENTION[k] for k in (6, 7))
    g = (d128 - d64) / (CONTEXTS[7] - CONTEXTS[6])
    f_lin = d128 - g * CONTEXTS[7]
    assert abs(f_lin / value("native_ops_lin") - 1) < 1e-3, f_lin
    # The rounded 8.1×10^11 misses the attention column by more than the tolerance.
    rounded = [
        b * c / (8.1e11 + value("native_ops_att_per_ctx") * c) for c in CONTEXTS
    ]
    assert max(abs(r - a) for r, a in zip(rounded, CHART_ATTENTION)) > 0.1
    # Least-squares count per token behind the linear column, at the model's F_lin.
    unit = (1 + value("glue_fraction")) * (value("commit_ps") + value("lookup_ps")) * 1e-12
    k = [unit / native(c) for c in CONTEXTS]
    fitted = sum(x * y for x, y in zip(k, CHART_LINEAR)) / sum(x * x for x in k)
    assert abs(fitted / 1.76e7 - 1) < 1e-3, fitted
    return fitted


def main():
    rows = [model(c) for c in CONTEXTS]
    calcs = [calculators(c, mu) for (c, mu, *_rest) in SECTION_8]
    fitted = check(rows, calcs)
    document = {
        "description": (
            "Executable cost model reference (RFC-0020 'Executable cost model', 08 §1): the "
            "source's projections, the inputs they rest on, and the model evaluated on them; "
            "written by spec/vectors/cost-model/generate.py; see README.md"
        ),
        "source": SOURCE,
        "inputs": {
            name: {"value": v, "provenance": p, "line": line, "text": text}
            | ({"note": note} if note else {})
            for name, (v, p, line, text, note) in INPUTS.items()
        },
        "chart": {
            "lines": [111, 134],
            "contexts": CONTEXTS,
            "total": CHART_TOTAL,
            "attention": CHART_ATTENTION,
            "linear": CHART_LINEAR,
            "weight_pass": TABLE_WEIGHT_PASS,
            "total_one_decimal": TABLE_TOTAL,
            "fitted_commits_per_token": round(fitted, -3),
        },
        "section_8": {
            "lines": [242, 249],
            "gpus": GPUS,
            "eta": ETA,
            "target_overhead": TARGET_OVERHEAD,
            "violations": VIOLATIONS,
            "rows": [
                {
                    "context": c,
                    "mu": mu,
                    "proof_time_s": t,
                    "proof_time_half_unit_s": h,
                    "sampling_percent": s,
                    "detection_percent": d,
                    "detection_approximate": a,
                }
                for (c, mu, t, h, s, d, a) in SECTION_8
            ],
        },
        "tolerances": {
            "chart": CHART_TOLERANCE,
            "weight_pass": WEIGHT_PASS_TOLERANCE,
            "percentage_points": PERCENTAGE_POINTS,
        },
        "model": rows,
        "calculators": calcs,
    }
    (HERE / "cost-model.json").write_text(json.dumps(document, indent=2, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    main()
