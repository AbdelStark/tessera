#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0 OR MIT
"""Research cost model for Tessera's proof multiplier `μ` (08 §1, RFC-0020).

A planning tool for the autoresearch loop (`research/README.md`): it scores an idea by the
change it makes to `μ`, before anyone writes a kernel. It is not the normative model, which is
`tessera bench model` once RFC-0020 lands; when the two disagree, that one wins.

The formula is the source's (source §5) plus the corpus's corrections:

    μ(S) = [1.15 (C_lin c_c + K_lin c_ℓ) + 1.15 A S (c_c + c_ℓ + c_π) + W/T] / native(S)
           + n_regen + streaming(S)

    native(S) = (F_lin + F_att S) / R          (ideal teacher-forced INT8 forward pass, per token)
    streaming(S) = c(S) · p_s · A · S / native(S),  c(S) = max(0, ⌈log2(N_level · bytes / free)⌉)

Usage:
    python3 research/tools/cost_model.py                       # source and corrected baselines
    python3 research/tools/cost_model.py --check               # golden check against source §5 and §8
    python3 research/tools/cost_model.py --sensitivity         # share of μ per lever, gain per 2× lever
    python3 research/tools/cost_model.py --set lookup_ps=20 --set leaf_ps=6   # a scenario vs corrected
    python3 research/tools/cost_model.py --profile source --set commits_per_token=1e7
    python3 research/tools/cost_model.py --profile counted --sensitivity   # the counted unit costs (EXP-0002, SUR-0002, EXP-0006)
    python3 research/tools/cost_model.py --json

Every number this prints is a projection (08 §1). Floating point is acceptable here: the model
is a planning tool, not part of the protocol (RFC-0020).
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from dataclasses import asdict, dataclass, fields, replace
from pathlib import Path

LAYERS_405B = 126
H100_DENSE_INT8_TOPS = 1979e12
SOURCE_COMMITS_PER_LAYER_TOKEN = 137_216
# RISK-2's upper bound: the `minimal` profile under RFC-0006's gadget-selection rule (EXP-0011 reproduces it).
RISK2_COMMITS_PER_LAYER_TOKEN = 220_160
# The default `servable` profile with every split (five split rows in four groups: the two norm keys, the RoPE
# q and k keys and the 40-bit `o8` attention-output rescale), derived from RFC-0005's range cells by RFC-0006's
# rule at 405B shapes: EXP-0011 (E0, validated, 2026-10-17). The `corrected` and `counted` profiles use it;
# the compiler's commitment report replaces it (RFC-0020).
SERVABLE_COMMITS_PER_LAYER_TOKEN = 326_656
# The stated count, 137,216 per layer per token times 126 layers (source §5 and §7), as the pinned vector of
# spec/vectors/cost-model/cost-model.json records it (`inputs.commits_per_token`, provenance `derived`).
SOURCE_COMMITS_PER_TOKEN = float(SOURCE_COMMITS_PER_LAYER_TOKEN * LAYERS_405B)
# The least-squares fit of the chart's linear column (vector `chart.fitted_commits_per_token`), 1.8% above the
# stated count; reference only, the model uses the stated count and stays within the ±0.05 tolerance.
FITTED_COMMITS_PER_TOKEN = 17_599_000.0
# The pinned vector (RFC-0020 'Executable cost model'); absent on branches older than PR #240.
VECTOR_PATH = Path(__file__).resolve().parents[2] / "spec" / "vectors" / "cost-model" / "cost-model.json"
DEFAULT_CONTEXTS = (1024, 4096, 16384, 131072)
SOURCE_CONTEXTS = (1024, 2048, 4096, 8192, 16384, 32768, 65536, 131072)


@dataclass(frozen=True)
class Params:
    """One scenario. Units: picoseconds, operations, bytes, seconds."""

    # Unit costs (08 §2), per element.
    commit_ps: float = 30.0
    lookup_ps: float = 40.0
    leaf_ps: float = 12.0
    glue: float = 1.15
    # Counts per token (RFC-0006 commitment report): the stated 137,216 per layer per token times 126
    # layers, as the pinned vector records; the chart's linear column fits 1.760e7 (1.8% more), which the
    # LM-head rescales and per-row scalars could account for, and the stated count reproduces the chart
    # within ±0.05 anyway. The compiler's commitment report replaces it (RFC-0020).
    commits_per_token: float = SOURCE_COMMITS_PER_TOKEN
    lookups_per_token: float = SOURCE_COMMITS_PER_TOKEN
    scores_per_token_per_ctx: float = 8064.0  # A = heads × layers / 2
    # Native work (source §5): 2 ops per parameter (405.85e9) per token; 512 ops per score.
    native_ops_lin: float = 2 * 405.85e9
    native_ops_att_per_ctx: float = 512 * 8064
    native_rate: float = 0.60 * H100_DENSE_INT8_TOPS
    # Weight pass: 0.24 GPU-s per 16K-token batch (source §5).
    weight_pass_gpu_s: float = 0.24
    batch_tokens: float = 16384.0
    # RISK-9: witness regenerations in full-forward equivalents.
    regen_full_equivalents: float = 0.0
    # RISK-10: streamed leaf levels (RFC-0018 'Pass structure').
    stream_pass_ps_per_score: float = 0.0
    free_bytes: float = 0.2e12
    bytes_per_leaf: float = 32.0
    stream_tokens: float = 16384.0  # tokens in the proof unit whose leaves are counted


PROFILES = {
    # The source's model, which the RFC-0020 golden test reproduces.
    "source": Params(),
    # The corpus's corrections: the `servable` count of EXP-0011 (326,656 per layer per token, in place of RISK-2's
    # 220,160 until 2026-10-17), RISK-9 four regenerations, RISK-10 streaming.
    "corrected": Params(
        commits_per_token=float(SERVABLE_COMMITS_PER_LAYER_TOKEN * LAYERS_405B),
        lookups_per_token=float(SERVABLE_COMMITS_PER_LAYER_TOKEN * LAYERS_405B),
        regen_full_equivalents=4.0,
        stream_pass_ps_per_score=7.0,
    ),
    # The corrected profile with the unit costs at the centre of the operation-count intervals the
    # research loop derived (E0 counts fitted to E1 on an Apple M4 Max; the H100 int32 peak and the
    # 60% utilization of [VYTA25] are the translation): c_c 94 ps, the high count at the pipeline's
    # peak, inside EXP-0002's [58, 157]; c_ℓ 101 ps, the textbook fraction tree at peak, inside
    # SUR-0002's 76–169; c_π 73 ps, the centre of EXP-0006's [45, 117]. Planning values until the
    # RFC-0020 v0.1 gate measures them (E2); the `corrected` profile stays the RFC-0020 reference.
    "counted": Params(
        commits_per_token=float(SERVABLE_COMMITS_PER_LAYER_TOKEN * LAYERS_405B),
        lookups_per_token=float(SERVABLE_COMMITS_PER_LAYER_TOKEN * LAYERS_405B),
        commit_ps=94.0,
        lookup_ps=101.0,
        leaf_ps=73.0,
        regen_full_equivalents=4.0,
        stream_pass_ps_per_score=7.0,
    ),
}

# Source §5 chart-data table, the fallback when the pinned vector is absent: (context, linear, attention,
# weight pass, total to one decimal). The vector carries the two-decimal totals of the chart itself.
SOURCE_TABLE = {
    1024: (2.06, 1.13, 0.021, 3.2),
    2048: (2.05, 2.25, 0.021, 4.3),
    4096: (2.03, 4.46, 0.021, 6.5),
    8192: (1.99, 8.75, 0.021, 10.8),
    16384: (1.91, 16.82, 0.020, 18.8),
    32768: (1.78, 31.24, 0.018, 33.0),
    65536: (1.56, 54.68, 0.016, 56.3),
    131072: (1.24, 87.48, 0.013, 88.7),
}


@dataclass(frozen=True)
class Breakdown:
    """`μ` and its terms at one context length (RFC-0020 `MuBreakdown`)."""

    context: int
    linear: float
    attention: float
    weights: float
    regen: float
    streaming: float
    total: float
    streamed_passes: int
    native_s_per_token: float

    def row(self) -> str:
        return (
            f"{self.context // 1024:>4}K {self.linear:>8.2f} {self.attention:>10.2f} "
            f"{self.weights:>8.3f} {self.regen:>6.1f} {self.streaming:>10.2f} {self.total:>8.1f}"
        )


def streamed_passes(p: Params, context: int) -> int:
    """`c(S)`: streamed rounds until the folded leaf level fits in memory (RFC-0018)."""
    if p.stream_pass_ps_per_score <= 0:
        return 0
    leaves = p.scores_per_token_per_ctx * context * p.stream_tokens
    ratio = leaves * p.bytes_per_leaf / p.free_bytes
    return max(0, math.ceil(math.log2(ratio))) if ratio > 1 else 0


def mu(p: Params, context: int) -> Breakdown:
    """The proof multiplier at `context`, per 08 §1."""
    native = (p.native_ops_lin + p.native_ops_att_per_ctx * context) / p.native_rate
    ps = 1e-12
    linear = p.glue * (p.commits_per_token * p.commit_ps + p.lookups_per_token * p.lookup_ps) * ps
    scores = p.scores_per_token_per_ctx * context
    attention = p.glue * scores * (p.commit_ps + p.lookup_ps + p.leaf_ps) * ps
    weights = p.weight_pass_gpu_s / p.batch_tokens
    passes = streamed_passes(p, context)
    streaming = passes * p.stream_pass_ps_per_score * ps * scores
    terms = (linear / native, attention / native, weights / native, p.regen_full_equivalents, streaming / native)
    return Breakdown(context, *terms, sum(terms), passes, native)


def table(p: Params, contexts=DEFAULT_CONTEXTS) -> list[Breakdown]:
    return [mu(p, c) for c in contexts]


HEADER = "   S   linear  attention  weights  regen  streaming    total"


def print_table(title: str, rows: list[Breakdown], baseline: list[Breakdown] | None = None) -> None:
    print(title)
    print(HEADER)
    for i, r in enumerate(rows):
        line = r.row()
        if baseline is not None:
            b = baseline[i].total
            line += f"   Δ {100 * (r.total - b) / b:+6.1f}%"
        print(line)
    print()


def golden_rows(vector_path: Path | None = None) -> tuple[dict[int, tuple[float, float, float, float]], float, str]:
    """The reference rows, their tolerance on the total, and where they came from.

    The pinned vector (spec/vectors/cost-model/cost-model.json, PR #240; another file with `--vector`) when it
    exists: the chart's two-decimal series, so the total is held to the same ±0.05 as the terms. Otherwise
    the embedded chart-data table, whose totals are printed to one decimal and are held to ±0.1.
    """
    path = vector_path or VECTOR_PATH
    if path.exists():
        with path.open(encoding="utf-8") as f:
            vector = json.load(f)
        chart = vector["chart"]
        rows = {
            int(c): (lin, att, w, total)
            for c, lin, att, w, total in zip(chart["contexts"], chart["linear"], chart["attention"], chart["weight_pass"], chart["total"])
        }
        return rows, float(vector["tolerances"]["chart"]), f"pinned vector {path}"
    return SOURCE_TABLE, 0.1, "embedded chart-data table (the pinned vector is absent on this branch)"


def check(vector_path: Path | None = None) -> bool:
    """Golden check (RFC-0020): the source profile reproduces the source's chart within ±0.05 per term."""
    ok = True
    p = PROFILES["source"]
    rows, total_tolerance, origin = golden_rows(vector_path)
    print(f"golden check against {origin} (±0.05 per term, ±{total_tolerance:g} on the total)")
    for context, (lin, att, w, total) in rows.items():
        r = mu(p, context)
        worst = max(abs(r.linear - lin), abs(r.attention - att), abs(r.weights - w))
        passed = worst <= 0.05 and abs(r.total - total) <= total_tolerance
        status = "ok  " if passed else "FAIL"
        ok &= passed
        print(
            f"{status} {context // 1024:>4}K model {r.linear:5.2f} {r.attention:6.2f} {r.weights:6.3f} "
            f"{r.total:6.2f}   reference {lin:5.2f} {att:6.2f} {w:6.3f} {total:6.2f}   worst dev {worst:.3f}"
        )
    return ok


LEVERS = {
    # name: (field, how a 2× improvement changes it)
    "commit_ps (c_c)": ("commit_ps", 0.5),
    "lookup_ps (c_ℓ)": ("lookup_ps", 0.5),
    "leaf_ps (c_π)": ("leaf_ps", 0.5),
    "commits_per_token (C_lin)": ("commits_per_token", 0.5),
    "lookups_per_token (K_lin)": ("lookups_per_token", 0.5),
    "scores per token (A)": ("scores_per_token_per_ctx", 0.5),
    "glue factor → 1.0": ("glue", None),
    "regen → 0": ("regen_full_equivalents", None),
    "streaming → 0": ("stream_pass_ps_per_score", None),
}


def sensitivity(p: Params, contexts=DEFAULT_CONTEXTS) -> None:
    base = table(p, contexts)
    print("share of μ by unit cost (percent of total)")
    print("   S   c_c    c_ℓ    c_π   weights  regen  stream")
    for r in base:
        s = p.scores_per_token_per_ctx * r.context
        n = r.native_s_per_token
        ps = 1e-12
        cc = p.glue * (p.commits_per_token + s) * p.commit_ps * ps / n
        cl = p.glue * (p.lookups_per_token + s) * p.lookup_ps * ps / n
        cp = p.glue * s * p.leaf_ps * ps / n
        pct = lambda x: 100 * x / r.total  # noqa: E731
        print(
            f"{r.context // 1024:>4}K {pct(cc):5.1f}% {pct(cl):5.1f}% {pct(cp):5.1f}%  "
            f"{pct(r.weights):5.1f}%  {pct(r.regen):5.1f}% {pct(r.streaming):5.1f}%"
        )
    print()
    print("gain in μ from one lever (2× better, or removed where marked), percent of total")
    print(f"{'lever':<28}" + "".join(f"{c // 1024:>7}K" for c in contexts))
    for name, (field, factor) in LEVERS.items():
        value = getattr(p, field)
        new = value * factor if factor is not None else (1.0 if field == "glue" else 0.0)
        rows = table(replace(p, **{field: new}), contexts)
        deltas = "".join(f"{100 * (r.total - b.total) / b.total:+7.1f}%" for r, b in zip(rows, base))
        print(f"{name:<28}{deltas}")
    print()


def parse_set(values: list[str], p: Params) -> Params:
    names = {f.name for f in fields(Params)}
    for item in values:
        if "=" not in item:
            sys.exit(f"--set expects name=value, got {item!r}")
        name, raw = item.split("=", 1)
        if name not in names:
            sys.exit(f"unknown parameter {name!r}; known: {', '.join(sorted(names))}")
        p = replace(p, **{name: float(raw)})
    return p


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--profile", choices=sorted(PROFILES), default="corrected", help="baseline profile")
    ap.add_argument("--set", action="append", default=[], metavar="NAME=VALUE", help="override a parameter")
    ap.add_argument("--contexts", default=",".join(str(c) for c in DEFAULT_CONTEXTS), help="comma-separated tokens")
    ap.add_argument("--check", action="store_true", help="golden check against the pinned vector, or the embedded table")
    ap.add_argument("--vector", type=Path, default=None, help="another cost-model vector file for --check")
    ap.add_argument("--sensitivity", action="store_true", help="shares and 2× gains per lever")
    ap.add_argument("--json", action="store_true", help="machine-readable output")
    args = ap.parse_args(argv)

    contexts = tuple(int(c) for c in args.contexts.split(","))
    if args.check:
        return 0 if check(args.vector) else 1

    baseline = PROFILES[args.profile]
    scenario = parse_set(args.set, baseline)
    if args.json:
        out = {
            "profile": args.profile,
            "params": asdict(scenario),
            "baseline": [asdict(r) for r in table(baseline, contexts)],
            "scenario": [asdict(r) for r in table(scenario, contexts)] if args.set else None,
        }
        print(json.dumps(out, indent=2))
        return 0

    if args.set:
        print_table(f"baseline: profile {args.profile!r}", table(baseline, contexts))
        print_table("scenario: " + ", ".join(args.set), table(scenario, contexts), table(baseline, contexts))
    else:
        for name in ("source", "corrected") if args.profile == "corrected" else (args.profile,):
            print_table(f"profile {name!r}", table(PROFILES[name], contexts))
    if args.sensitivity:
        sensitivity(scenario, contexts)
    return 0


if __name__ == "__main__":
    sys.exit(main())
