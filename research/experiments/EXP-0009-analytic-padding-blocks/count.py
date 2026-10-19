#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0 OR MIT
"""EXP-0009: the power-of-two padding factors of Tessera's tree instances (E0, exact integers).

Three paddings make a tree instance larger than its real leaf count (IDEA-0019): a fraction
tree's lookups are padded to a power of two by repeats of table row 0 (RFC-0008 'Fraction
tree'); a forest's trees are padded to a power of two with trees whose roots are public
(RFC-0010 'Tree levels'); and a `SoftmaxRow` row of `q + 1` causal keys is padded to its
forest's height with the public constant 1 (RFC-0009 'SoftmaxRow', issue #247). This script
prints `padded / real` for every instance of the two target models at 1K, 4K and 16K tokens,
and the exact leaf counts of the harness's configurations.

Inputs: the eight rescaled tensors per layer per token of RISK-2 (docs/spec/00-overview.md §9:
16,384 + 16,384 + 1,024 + 1,024 + 16,384 + 16,384 + 53,248 + 16,384 at 405B, read as two norms,
two residual rescales, RoPE-Q, RoPE-K, V and SiLU), the layer and head counts of Llama 3.1
(405B: 126 layers, 128 heads, hidden 16,384, 8 KV heads of 128, intermediate 53,248; 70B: 80
layers, 64 heads, hidden 8,192, 8 KV heads of 128, intermediate 28,672), one `Exp` lookup and
one product-tree leaf per causal score (RFC-0009 'SoftmaxRow': one tree of `q + 1` leaves per
row), and the forest rule of RFC-0010 'Tree levels' (trees of one height stacked into one
forest, the count padded to a power of two). Every figure is E0 (own count, primary).

Two assumptions about which lookups share a table definition (one instance per rescaled tensor;
or norms, residual rescales and RoPE Q+K each sharing one definition) and two about how
`SoftmaxRow` rows are grouped (all rows of a model at one height `log2 T`; rows grouped by
height `⌈log2(q + 1)⌉`, with row 0 at height 1), since RFC-0009 and RFC-0010 fix neither.

    python3 -I count.py
"""

from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction
from collections import Counter
import sys


@dataclass(frozen=True)
class Model:
    name: str
    layers: int
    heads: int
    hidden: int
    kv: int
    intermediate: int


MODELS = [
    Model("405B", layers=126, heads=128, hidden=16_384, kv=1_024, intermediate=53_248),
    Model("70B", layers=80, heads=64, hidden=8_192, kv=1_024, intermediate=28_672),
]
TOKENS = [1 << 10, 1 << 12, 1 << 14]


def next_pow2(n: int) -> int:
    return 1 if n <= 1 else 1 << (n - 1).bit_length()


def factor(real: int) -> Fraction:
    return Fraction(next_pow2(real), real)


def lookup_instances(m: Model, shared: bool) -> dict[str, int]:
    """Per-layer-per-token lookup counts of the rescaled tensors (RISK-2), per instance."""
    if not shared:
        return {
            "norm 1": m.hidden,
            "norm 2": m.hidden,
            "residual rescale 1": m.hidden,
            "residual rescale 2": m.hidden,
            "RoPE-Q": m.hidden,
            "RoPE-K": m.kv,
            "V": m.kv,
            "SiLU": m.intermediate,
        }
    return {
        "norms (shared)": 2 * m.hidden,
        "residual rescales (shared)": 2 * m.hidden,
        "RoPE Q+K (shared)": m.hidden + m.kv,
        "V": m.kv,
        "SiLU": m.intermediate,
    }


def causal_scores(t: int) -> int:
    return t * (t + 1) // 2


def softmax_one_height(m: Model, t: int) -> tuple[int, int]:
    """All rows at one height log2 T: trees = layers·heads·T padded to a power of two, T leaves each."""
    trees = m.layers * m.heads * t
    return next_pow2(trees) * t, m.layers * m.heads * causal_scores(t)


def rows_at_height(h: int) -> int:
    return 2 if h == 1 else 1 << (h - 1)


def softmax_grouped(m: Model, t: int, pairs: int | None = None) -> tuple[int, int]:
    """Rows grouped by height h = 1..log2 T; each height's layers·heads·rows_h trees padded to a
    power of two with 2^h leaves each."""
    pairs = m.layers * m.heads if pairs is None else pairs
    padded = 0
    n = t.bit_length() - 1
    for h in range(1, n + 1):
        trees = pairs * rows_at_height(h)
        padded += next_pow2(trees) * (1 << h)
    return padded, pairs * causal_scores(t)


def harness_configs() -> list[tuple[str, int, int]]:
    """The exact leaf counts of the harness's configurations (README 'Success criterion')."""
    out = []
    t = 4096
    out.append(("S1 one height T=4096", next_pow2(t) * t, causal_scores(t)))
    p, r = softmax_grouped(Model("-", 1, 1, 0, 0, 0), t)
    out.append(("S2 grouped T=4096", p, r))
    out.append(("S3 70B forest 640 of 1024 x 2^14", 1024 << 14, 640 << 14))
    p, r = softmax_grouped(Model("-", 1, 1, 0, 0, 0), 2048, pairs=5)
    out.append(("S4 70B grouped T=2048 x5", p, r))
    out.append(("U23 2^9 x 2^14", 1 << 23, 1 << 23))
    out.append(("U24 2^10 x 2^14", 1 << 24, 1 << 24))
    t = 1024
    out.append(("S1-1K", next_pow2(t) * t, causal_scores(t)))
    p, r = softmax_grouped(Model("-", 1, 1, 0, 0, 0), t)
    out.append(("S2-1K", p, r))
    return out


def main() -> int:
    if "--amendment" in sys.argv:
        amendment_counts()
        return 0
    print("EXP-0009 counts: power-of-two padding factors, padded / real (E0, exact)")
    for m in MODELS:
        print()
        print("%s: %d layers (%s x 2^%d), %d heads, layers x heads = %d (%s x 2^%d)" % (
            m.name, m.layers, Fraction(m.layers, next_pow2(m.layers)), next_pow2(m.layers).bit_length() - 1,
            m.heads, m.layers * m.heads, Fraction(m.layers * m.heads, next_pow2(m.layers * m.heads)),
            next_pow2(m.layers * m.heads).bit_length() - 1))
        for shared in (False, True):
            print("  lookup instances, %s definitions (leaves = count x layers x T):" % ("shared" if shared else "per-tensor"))
            print("    %-28s %10s | %-24s | %-24s | %-24s" % ("instance", "per layer", "T = 1K", "T = 4K", "T = 16K"))
            for name, per_layer in lookup_instances(m, shared).items():
                cells = []
                for t in TOKENS:
                    real = per_layer * m.layers * t
                    f = factor(real)
                    cells.append("%14d  %.4f" % (real, float(f)))
                print("    %-28s %10d | %s" % (name, per_layer, " | ".join(cells)))
            total_real = {t: sum(c * m.layers * t for c in lookup_instances(m, shared).values()) for t in TOKENS}
            total_padded = {t: sum(next_pow2(c * m.layers * t) for c in lookup_instances(m, shared).values()) for t in TOKENS}
            print("    %-28s %10s | %s" % ("all lookups, weighted", "", " | ".join(
                "%14d  %.4f" % (total_real[t], total_padded[t] / total_real[t]) for t in TOKENS)))
        print("  Exp instance (one lookup per causal score, layers x heads x T(T+1)/2):")
        for t in TOKENS:
            real = m.layers * m.heads * causal_scores(t)
            print("    T = %5d: real %16d  padded %16d  factor %.4f" % (t, real, next_pow2(real), float(factor(real))))
        print("  SoftmaxRow forest (one product-tree leaf per causal score):")
        for t in TOKENS:
            p1, r1 = softmax_one_height(m, t)
            p2, r2 = softmax_grouped(m, t)
            assert r1 == r2
            print("    T = %5d: real %16d | one height: padded %16d  factor %.4f | grouped by height: padded %16d  factor %.4f" % (
                t, r1, p1, p1 / r1, p2, p2 / r2))
    print()
    print("Model-independent row padding (layers x heads = 1): Sum_q 2^ceil(log2(q+1)) / (T(T+1)/2), row 0 at height 1:")
    for t in TOKENS:
        p, r = softmax_grouped(Model("-", 1, 1, 0, 0, 0), t)
        p1, _ = softmax_one_height(Model("-", 1, 1, 0, 0, 0), t)
        print("    T = %5d: grouped %.5f (%d / %d), one height %.5f (%d / %d)" % (t, p / r, p, r, p1 / r, p1, r))
    print()
    print("Harness configurations (README 'Success criterion'):")
    print("    %-36s %12s %12s %10s %12s" % ("configuration", "padded", "real", "f", "1+(f-1)/2"))
    for name, padded, real in harness_configs():
        f = Fraction(padded, real)
        print("    %-36s %12d %12d %10.5f %12.4f" % (name, padded, real, float(f), 1 + (float(f) - 1) / 2))
    return 0


def amendment_counts() -> None:
    """Exact public-shape work counts; no throughput assumption or conversion to time."""
    def grouped(t: int, pairs: int = 1) -> list[tuple[int, list[int]]]:
        return [
            (h, list(range(1 if h == 1 else (1 << (h - 1)) + 1, (1 << h) + 1)) * pairs)
            for h in range(1, t.bit_length())
        ]

    configs = {
        "S1": [(12, list(range(1, 4097)))],
        "S2": grouped(4096),
        "S3": [(14, [1 << 14] * 640)],
        "S4": grouped(2048, 5),
        "U23": [(14, [1 << 14] * 512)],
        "U24": [(14, [1 << 14] * 1024)],
        "S1-1K": [(10, list(range(1, 1025)))],
        "S2-1K": grouped(1024),
    }
    print("EXP-0009 amendment E0 counts: exact tree-position-round work, synthetic shapes")
    print("Full pair arithmetic is unchanged; tail multiplications include N node scales and 3 coefficient products.")
    print("Four-lane folds include masked spare lanes; this count does not assert machine SIMD or speed.")
    for name, forests in configs.items():
        tallest = max(h for h, _ in forests)
        counts: Counter[str] = Counter()
        for h, leaves in forests:
            s = (next_pow2(len(leaves))).bit_length() - 1
            for t in range(2, h + 1):
                m = t - 1
                level = tallest - h + t - 1
                active_vars = [
                    level - (tallest - other_h) + (next_pow2(len(other_leaves))).bit_length() - 1
                    for other_h, other_leaves in forests if tallest - other_h <= level
                ]
                front_padded = m + s < max(active_vars)
                real_nodes = [(r + (1 << (h - t)) - 1) >> (h - t) for r in leaves]
                prefixes = [(r + 1 >> 1, r >> 1, r >> 1) for r in real_nodes]
                for j in range(m):
                    runs = Counter(prefixes)
                    counts["tree_position_rounds"] += len(leaves)
                    counts["equal_length_runs"] += len(runs)
                    nodes = 4 if j == 0 and front_padded else 3
                    pairs = 1 << (m - j - 1)
                    old_tail_thresholds = set()
                    for (e, o, q), multiplicity in runs.items():
                        full = min(e // 2, o // 2, q // 2)
                        padded = max((e + 1) // 2, (o + 1) // 2)
                        counts["full_pair_node_products"] += full * nodes * multiplicity
                        counts["boundary_pair_node_products"] += (padded - full) * nodes * multiplicity
                        if padded < pairs:
                            old_tail_thresholds.add(padded)
                            counts["tail_evaluations_amended"] += 1
                            counts["tail_recursion_fp2_products_amended"] += m - j - 1 + padded.bit_count()
                            counts["tail_scale_additions_amended"] += multiplicity
                            counts["tail_terms_original"] += multiplicity
                            counts["tail_terms_amended"] += 1
                            counts["tail_fp2_products_original"] += (3 + nodes) * multiplicity
                            counts["tail_fp2_products_amended"] += 3 + nodes
                        for length in (e, o, q):
                            counts["full_prefix_folds"] += length // 2 * multiplicity
                            if length % 2:
                                counts["odd_tail_folds"] += multiplicity
                                counts["odd_tail_four_lane_batches"] += (multiplicity + 3) // 4
                                counts["odd_tail_spare_lanes"] += 4 * ((multiplicity + 3) // 4) - multiplicity
                    counts["tail_evaluations_original"] += len(old_tail_thresholds)
                    counts["tail_recursion_fp2_products_original"] += sum(
                        m - j - 1 + threshold.bit_count() for threshold in old_tail_thresholds
                    )
                    prefixes = [tuple((length + 1) // 2 for length in row) for row in prefixes]
        assert counts["tail_terms_amended"] <= counts["tail_terms_original"]
        assert counts["odd_tail_folds"] + counts["odd_tail_spare_lanes"] == 4 * counts["odd_tail_four_lane_batches"]
        print(name, dict(sorted(counts.items())))


if __name__ == "__main__":
    raise SystemExit(main())
