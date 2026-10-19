#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0 OR MIT
"""EXP-0008: operation counts per leaf of the product-tree leaf stage, baseline against the
split-eq, Gruen-corrected candidate (E0), and the fraction proxy's two levels.

The baseline rules are EXP-0006's (crates/tessera-sumcheck: forest.rs, prover.rs, kernel.rs;
crates/tessera-poly: eq.rs, multilinear.rs), copied here so that the two counts share one
accounting. The candidate's rules read the harness (src/main.rs: `product_base_sums`,
`product_ext_sums`, `prove_split_batch`, `fraction_leaf_sums`, `fraction_generic_sums`).

Every rule is one line naming the code it translates; every figure is E0 (own count, primary).
The unit costs used for the time prediction are EXP-0006's E1 measurements on one Apple M4 Max
performance core (results/run.txt of that experiment, 2026-10-16), so the prediction is E0 over
E1 units. The peak rates are secondary (vendor documentation) and the utilizations EXP-0002's.

    python3 -I count.py
"""

from __future__ import annotations

import math
from dataclasses import dataclass, fields

# RFC-0020's c_π row shape: product-tree GKR batched over rows of 2^14 leaves; 2^8 such rows.
TREES_LOG = 8
HEIGHT = 14
N = 1 << (TREES_LOG + HEIGHT)

# H100 SXM5: 132 SMs × 64 int32 lanes × 1.98 GHz; HBM3 3.35 TB/s (as EXP-0006).
INT32_PEAK_OPS = 132 * 64 * 1.98e9
HBM_PEAK_BYTES = 3.35e12
INT_UTILIZATION = 0.60  # [VYTA25] via EXP-0002 and EXP-0006
HBM_UTILIZATION = 0.80

# EXP-0006's measured unit costs, ns per operation, in cache (E1, one M4 Max performance core).
UNITS_NS = {
    "fp_mul": 0.598,
    "fp_add": 0.342,
    "fp_sub": 0.500,
    "fp2_mul": 3.347,
    "fp2_mulb": 0.878,
    "fp2_add": 0.442,
    "fp2_sub": 0.267,
}


@dataclass(frozen=True)
class Ops:
    """Operation counts in the harness's units: Goldilocks multiplication, addition and
    subtraction; Fp2 multiplication (Karatsuba as ext2.rs implements it), Fp2-by-Fp
    multiplication (`mul_by_base`), Fp2 addition and subtraction."""

    fp_mul: float = 0.0
    fp_add: float = 0.0
    fp_sub: float = 0.0
    fp2_mul: float = 0.0
    fp2_mulb: float = 0.0
    fp2_add: float = 0.0
    fp2_sub: float = 0.0

    def __add__(self, other: "Ops") -> "Ops":
        return Ops(**{f.name: getattr(self, f.name) + getattr(other, f.name) for f in fields(Ops)})

    def __mul__(self, k: float) -> "Ops":
        return Ops(**{f.name: getattr(self, f.name) * k for f in fields(Ops)})

    __rmul__ = __mul__

    def expand(self) -> dict[str, float]:
        """Goldilocks operations: a Karatsuba Fp2 product is 3 multiplications, one
        `mul_small(7)`, 3 additions, 2 subtractions; `mul_base` is 2 multiplications; an Fp2
        addition or subtraction is 2 of its kind (EXP-0006)."""
        return {
            "mul": self.fp_mul + 3 * self.fp2_mul + 2 * self.fp2_mulb,
            "mul7": self.fp2_mul,
            "add": self.fp_add + 3 * self.fp2_mul + 2 * self.fp2_add,
            "sub": self.fp_sub + 2 * self.fp2_mul + 2 * self.fp2_sub,
        }

    def mul_equivalents(self) -> float:
        """IDEA-0008's unit: Goldilocks multiplications, an Fp2 multiplication counting 3 and a
        `mul_by_base` 2."""
        return self.fp_mul + 3 * self.fp2_mul + 2 * self.fp2_mulb

    def ns(self) -> float:
        """Predicted nanoseconds from EXP-0006's unit costs."""
        return sum(getattr(self, k) * v for k, v in UNITS_NS.items())


# --- baseline rules (EXP-0006 count.py, three nodes per round message) ---------------------------
BASE = {
    # kernel.rs `product_sums`, mixed path (base = [even, odd] over Fp, ext = [eq] over Fp2).
    "round1_mixed": Ops(fp_mul=3, fp_sub=2, fp_add=4, fp2_sub=1, fp2_add=5, fp2_mulb=3),
    # kernel.rs `zip3` over Fp2.
    "zip3_ext": Ops(fp2_sub=3, fp2_add=9, fp2_mul=6),
    # kernel.rs `zip2` over Fp2.
    "zip2_ext": Ops(fp2_sub=2, fp2_add=7, fp2_mul=3),
    # kernel.rs `bind_base`: per pair `r.mul_by_base(pair[1] − pair[0]).add_base(pair[0])`.
    "bind_base": Ops(fp_sub=1, fp_add=1, fp2_mulb=1),
    # kernel.rs `bind_ext` / `bind_in_place`: per pair `pair[0] + r·(pair[1] − pair[0])`.
    "bind_ext": Ops(fp2_sub=1, fp2_mul=1, fp2_add=1),
    # eq.rs `eq_table`: per entry one multiplication and one subtraction.
    "eq_entry": Ops(fp2_mul=1, fp2_sub=1),
    # forest.rs `level_above`: one Fp multiplication per entry of the level produced.
    "level_entry": Ops(fp_mul=1),
}

# --- candidate rules (src/main.rs), two nodes per round ---------------------------------------------
CAND = {
    # `product_base_sums`, per pair: lines `l[1] − l[0]`, `r[1] − r[0]` (2 Fp sub), `l[1] + dl`,
    # `r[1] + dr` (2 Fp add); products `l[0]·r[0]`, `(l[1] + dl)·(r[1] + dr)` (2 Fp mul);
    # scaling `w.mul_by_base(p0)`, `w.mul_by_base(p2)` (2 mul_by_base); accumulate (2 Fp2 add).
    "split_round1": Ops(fp_sub=2, fp_add=2, fp_mul=2, fp2_mulb=2, fp2_add=2),
    # `product_ext_sums`, per pair: lines (2 Fp2 sub, 2 Fp2 add), products (2 Fp2 mul), scaling
    # by `eq_lo` (2 Fp2 mul), accumulate (2 Fp2 add).
    "split_round_ext": Ops(fp2_sub=2, fp2_add=4, fp2_mul=4),
    # per x_hi group (2^lo pairs): `sums[z] += h · acc[z]` at two nodes.
    "split_group": Ops(fp2_mul=2, fp2_add=2),
    # per own round per instance: ℓ(0), ℓ(1) (2 Fp2 mul), the recovery (1 Fp2 mul, 1 inverse),
    # the three message products and three weightings (6 Fp2 mul), the claim update (about 4
    # Fp2 mul and one mul_by_base): about 13 Fp2 mul plus one Fp2 inversion, counted as 100 Fp
    # multiplications (a Fermat exponentiation in Fp plus the norm).
    "split_round_overhead": Ops(fp2_mul=13, fp2_mulb=1, fp_mul=100),
    # `fraction_leaf_sums`, per pair: lines (2 Fp2 sub, 2 Fp2 add); relation
    # `e·(o + γ) + γ²·o` at two nodes (2 × (2 Fp2 mul, 2 Fp2 add)); scaling (2 Fp2 mul);
    # accumulate (2 Fp2 add).
    "fraction_leaf_round": Ops(fp2_sub=2, fp2_add=8, fp2_mul=6),
    # `fraction_generic_sums`, per pair: lines for four tables at node 2 (4 Fp2 sub, 4 Fp2 add);
    # relation `dd·(ne + γ²·de) + γ·(no·de)` at two nodes (2 × (4 Fp2 mul, 2 Fp2 add));
    # scaling (2 Fp2 mul); accumulate (2 Fp2 add).
    "fraction_generic_round": Ops(fp2_sub=4, fp2_add=10, fp2_mul=10),
}

# int32 translation per Goldilocks operation, (low, high), as EXP-0002 and EXP-0006.
FIELD_OPS = {
    "mul": (18, 26),
    "add": (4, 8),
    "sub": (4, 5),
    "mul7": (8, 13),
}


def level_sizes(trees_log: int, height: int) -> list[int]:
    return [1 << (trees_log + level) for level in range(height)]


def base_level(m: int) -> Ops:
    """One level's sumcheck on tables of `m` entries (eq, even, odd), as EXP-0006 counts it."""
    half = m // 2
    rest = half - 1
    return (
        BASE["round1_mixed"] * half
        + BASE["bind_base"] * (2 * half)
        + BASE["bind_ext"] * half
        + BASE["zip3_ext"] * rest
        + BASE["bind_ext"] * (3 * rest)
    )


def cand_level(m: int) -> Ops:
    """One level's sumcheck on the candidate: `n = log2 m` own rounds; round `k` has `m / 2^(k+1)`
    pairs over `m' = n − k − 1` remaining variables, split into `lo = ⌈m'/2⌉` and `hi = ⌊m'/2⌋`;
    `eq_lo` and `eq_hi` rebuilt each round (`2^lo + 2^hi − 2` entries); `2^hi` groups; two folds of
    the value tables after every round but the last one needed (the engine folds after every
    round, the harness too: `bind_lowest` after round 0, `bind_lowest_in_place` after)."""
    n = m.bit_length() - 1
    total = Ops()
    for k in range(n):
        pairs = m >> (k + 1)
        rem = n - k - 1
        lo, hi = (rem + 1) // 2, rem // 2
        total = total + BASE["eq_entry"] * ((1 << lo) + (1 << hi) - 2)
        total = total + CAND["split_group"] * (1 << hi)
        total = total + CAND["split_round_overhead"]
        if k == 0:
            total = total + CAND["split_round1"] * pairs + BASE["bind_base"] * (2 * pairs)
        else:
            total = total + CAND["split_round_ext"] * pairs + BASE["bind_ext"] * (2 * pairs)
    return total


def forest(trees_log: int, height: int, candidate: bool) -> dict[str, Ops]:
    n = 1 << (trees_log + height)
    sizes = level_sizes(trees_log, height)
    sumchecks = Ops()
    eq = Ops()
    for m in sizes:
        if candidate:
            sumchecks = sumchecks + cand_level(m)
        else:
            sumchecks = sumchecks + base_level(m)
            eq = eq + BASE["eq_entry"] * (m - 1)
    build = BASE["level_entry"] * (n - (1 << (trees_log + 1)))
    # The candidate's root claim: 2^s products and a 2^s-entry eq table, once per forest.
    root = Ops(fp_mul=1 << trees_log, fp2_mulb=1 << trees_log, fp2_add=1 << trees_log) + BASE["eq_entry"] * ((1 << trees_log) - 1) if candidate else Ops()
    return {"sumchecks": sumchecks, "eq": eq, "build": build, "root": root, "total": sumchecks + eq + build + root}


def components(trees_log: int, height: int) -> tuple[dict[str, Ops], dict[str, Ops]]:
    """Per leaf of the whole tree, by component, baseline and candidate."""
    n = 1 << (trees_log + height)
    sizes = level_sizes(trees_log, height)
    half = sum(m // 2 for m in sizes)
    rest = sum(m // 2 - 1 for m in sizes)
    base = {
        "eq tables": BASE["eq_entry"] * sum(m - 1 for m in sizes),
        "round 1": BASE["round1_mixed"] * half,
        "fold after round 1": BASE["bind_base"] * (2 * half) + BASE["bind_ext"] * half,
        "rounds >= 2": BASE["zip3_ext"] * rest,
        "folds after rounds >= 2": BASE["bind_ext"] * (3 * rest),
        "level build": BASE["level_entry"] * (n - (1 << (trees_log + 1))),
    }
    eq = Ops()
    groups = Ops()
    overhead = Ops()
    r1 = Ops()
    f1 = Ops()
    rx = Ops()
    fx = Ops()
    for m in sizes:
        nvars = m.bit_length() - 1
        for k in range(nvars):
            pairs = m >> (k + 1)
            rem = nvars - k - 1
            lo, hi = (rem + 1) // 2, rem // 2
            eq = eq + BASE["eq_entry"] * ((1 << lo) + (1 << hi) - 2)
            groups = groups + CAND["split_group"] * (1 << hi)
            overhead = overhead + CAND["split_round_overhead"]
            if k == 0:
                r1 = r1 + CAND["split_round1"] * pairs
                f1 = f1 + BASE["bind_base"] * (2 * pairs)
            else:
                rx = rx + CAND["split_round_ext"] * pairs
                fx = fx + BASE["bind_ext"] * (2 * pairs)
    cand = {
        "eq tables (sqrt-size, rebuilt per round)": eq,
        "round 1 (two nodes, Fp products)": r1,
        "fold after round 1 (2 bind_base)": f1,
        "rounds >= 2 (two nodes, Fp2)": rx,
        "folds after rounds >= 2 (2 bind_in_place)": fx,
        "per-group eq_hi scaling": groups,
        "per-round recovery and claim update": overhead,
        "root claim": forest(trees_log, height, True)["root"],
        "level build": BASE["level_entry"] * (n - (1 << (trees_log + 1))),
    }
    scale = 1 / n
    return ({k: v * scale for k, v in base.items()}, {k: v * scale for k, v in cand.items()})


def fraction_levels(m: int) -> dict[str, Ops]:
    """The fraction proxy's two levels on tables of `m` entries over Fp2, baseline (EXP-0006's
    count: three instances, each binding its own eq) and candidate (one relation, two folds on the
    leaf level and four on the generic, sqrt-size eq tables)."""
    pairs = m - 1
    base_leaf = (BASE["zip3_ext"] + BASE["zip2_ext"] * 2) * pairs + BASE["bind_ext"] * (7 * pairs)
    base_generic = BASE["zip3_ext"] * (3 * pairs) + BASE["bind_ext"] * (9 * pairs)
    n = m.bit_length() - 1
    cand_leaf = Ops()
    cand_generic = Ops()
    for k in range(n):
        p = m >> (k + 1)
        rem = n - k - 1
        lo, hi = (rem + 1) // 2, rem // 2
        shared = BASE["eq_entry"] * ((1 << lo) + (1 << hi) - 2) + CAND["split_group"] * (1 << hi) + CAND["split_round_overhead"]
        cand_leaf = cand_leaf + shared + CAND["fraction_leaf_round"] * p + BASE["bind_ext"] * (2 * p)
        cand_generic = cand_generic + shared + CAND["fraction_generic_round"] * p + BASE["bind_ext"] * (4 * p)
    return {
        "leaf baseline": base_leaf,
        "leaf candidate": cand_leaf,
        "generic baseline": base_generic,
        "generic candidate": cand_generic,
    }


def int32(expanded: dict[str, float], end: int) -> float:
    return sum(expanded[op] * FIELD_OPS[op][end] for op in ("mul", "mul7", "add", "sub"))


def bytes_per_leaf(trees_log: int, height: int, candidate: bool) -> tuple[float, float]:
    """HBM bytes per leaf as EXP-0006 models them. Fused low end (levels built and read once, the
    rounds on chip): the same 40 B for both arms. Unfused high end, as the CPU code is written: the
    baseline per level of M entries writes and reads the eq table over Fp2 (32M), reads the even
    and odd halves (16M), writes the first fold of three tables (24M), reads and writes the later
    rounds (48M + 24M): 144M; the candidate has no eq table and two tables instead of three:
    reads 16M, writes the first fold 16M, later rounds read 32M and write 16M: 80M."""
    n = 1 << (trees_log + height)
    sizes = level_sizes(trees_log, height)
    fused = 8 + 8 + 8 + sum(2 * m * 8 for m in sizes) / n
    per_level = 80 if candidate else 144
    unfused = 24 + sum(per_level * m for m in sizes) / n
    return fused, unfused


def ps(ops: float, rate: float) -> float:
    return ops / rate * 1e12


def show(label: str, ops: Ops, width: int = 46) -> None:
    print(
        "  %-*s Fp mul %6.3f add %6.3f sub %6.3f | Fp2 mul %6.3f mul_by_base %6.3f add %6.3f sub %6.3f | %5.2f eq | %6.2f ns"
        % (width, label, ops.fp_mul, ops.fp_add, ops.fp_sub, ops.fp2_mul, ops.fp2_mulb, ops.fp2_add, ops.fp2_sub, ops.mul_equivalents(), ops.ns())
    )


def interval(total: Ops, trees_log: int, height: int, candidate: bool) -> tuple[float, float, float, float]:
    ex = total.expand()
    low_ops, high_ops = int32(ex, 0), int32(ex, 1)
    fused, unfused = bytes_per_leaf(trees_log, height, candidate)
    lo = max(ps(low_ops, INT32_PEAK_OPS), ps(fused, HBM_PEAK_BYTES))
    hi = max(ps(high_ops, INT32_PEAK_OPS * INT_UTILIZATION), ps(unfused, HBM_PEAK_BYTES * HBM_UTILIZATION))
    return low_ops, high_ops, lo, hi


def main() -> int:
    print("EXP-0008 counts: product-tree leaf stage, 2^%d trees of 2^%d leaves (N = 2^%d); baseline (EXP-0006's rules) against the split-eq Gruen candidate (src/main.rs)" % (TREES_LOG, HEIGHT, TREES_LOG + HEIGHT))
    print("Columns: operations per leaf of the whole tree; 'eq' = Fp-multiplication equivalents (Fp2 mul 3, mul_by_base 2); 'ns' = predicted from EXP-0006's E1 unit costs (%s)" % ", ".join("%s %.3f" % kv for kv in UNITS_NS.items()))
    print()
    base, cand = components(TREES_LOG, HEIGHT)
    print("Baseline, by component:")
    for label, ops in base.items():
        show(label, ops)
    base_total = forest(TREES_LOG, HEIGHT, False)["total"] * (1 / N)
    show("baseline total per leaf", base_total)
    print()
    print("Candidate, by component:")
    for label, ops in cand.items():
        show(label, ops)
    cand_total = forest(TREES_LOG, HEIGHT, True)["total"] * (1 / N)
    show("candidate total per leaf", cand_total)
    print()
    print("Fp-multiplication equivalents per leaf: baseline %.2f, candidate %.2f (%+.1f%%); IDEA-0021's Mechanism table predicted 25.5 -> about 15.5"
          % (base_total.mul_equivalents(), cand_total.mul_equivalents(), 100 * (cand_total.mul_equivalents() / base_total.mul_equivalents() - 1)))
    bx, cx = base_total.expand(), cand_total.expand()
    print("Goldilocks operations per leaf: baseline %.2f mul + %.2f mul7 + %.2f add + %.2f sub; candidate %.2f mul + %.2f mul7 + %.2f add + %.2f sub"
          % (bx["mul"], bx["mul7"], bx["add"], bx["sub"], cx["mul"], cx["mul7"], cx["add"], cx["sub"]))
    print("Predicted time per leaf from EXP-0006's unit costs (E0 over E1 units): baseline %.2f ns (EXP-0006 measured 31.70 on this shape), candidate %.2f ns; predicted ratio %.3f, predicted speed-up sigma %.1f%%"
          % (base_total.ns(), cand_total.ns(), base_total.ns() / cand_total.ns(), 100 * (1 - cand_total.ns() / base_total.ns())))
    print()
    for label, total, candidate in (("baseline", base_total, False), ("candidate", cand_total, True)):
        low_ops, high_ops, lo, hi = interval(total, TREES_LOG, HEIGHT, candidate)
        fused, unfused = bytes_per_leaf(TREES_LOG, HEIGHT, candidate)
        central = round(math.sqrt(lo * hi))
        print("H100 projection, %s: int32 ops per leaf %.0f (lazy) / %.0f (as written); HBM bytes %.0f (fused) / %.0f (unfused); compute %.1f ps at peak / %.1f at %.0f%%; memory %.1f / %.1f ps at %.0f%%; interval I = [%.1f, %.1f] ps, central %d, rho_pi = [%.2f, %.2f]"
              % (label, low_ops, high_ops, fused, unfused, ps(low_ops, INT32_PEAK_OPS), ps(high_ops, INT32_PEAK_OPS * INT_UTILIZATION), 100 * INT_UTILIZATION,
                 ps(fused, HBM_PEAK_BYTES * HBM_UTILIZATION), ps(unfused, HBM_PEAK_BYTES * HBM_UTILIZATION), 100 * HBM_UTILIZATION, lo, hi, central, lo / 12, hi / 12))
    _, _, blo, bhi = interval(base_total, TREES_LOG, HEIGHT, False)
    _, _, clo, chi = interval(cand_total, TREES_LOG, HEIGHT, True)
    print("Count ratio of the intervals: lo %.3f, hi %.3f; the counted central 73 ps scaled by the Fp-equivalent ratio: %.1f ps, by the int32 (lazy) ratio: %.1f ps"
          % (clo / blo, chi / bhi, 73 * cand_total.mul_equivalents() / base_total.mul_equivalents(), 73 * clo / blo))
    print()
    m = N // 2
    fr = fraction_levels(m)
    print("Fraction proxy (IDEA-0018 case), per entry of 2^21-entry Fp2 tables:")
    for label, ops in fr.items():
        show(label, ops * (1 / m))
    for level in ("leaf", "generic"):
        b, c = fr[level + " baseline"] * (1 / m), fr[level + " candidate"] * (1 / m)
        print("  %s level: Fp2-multiplication equivalents per pair-round (IDEA-0018's unit) baseline %.2f, candidate %.2f (%+.1f%%); predicted ns per entry %.2f -> %.2f (ratio %.3f, sigma %.1f%%)"
              % (level, b.mul_equivalents() / 3 * 2, c.mul_equivalents() / 3 * 2, 100 * (c.mul_equivalents() / b.mul_equivalents() - 1), b.ns(), c.ns(), b.ns() / c.ns(), 100 * (1 - c.ns() / b.ns())))
    print("  (IDEA-0018's Mechanism table: generic 27 -> 14, leaf 19 -> 8 Fp2 multiplications per pair-round, counted per pair; the figures above are per entry over all rounds, so they are those per-pair figures halved up to the folds' geometric tail)")
    print()
    print("Not counted: the gamma weights of the baseline (3 Fp2 multiplications per instance per round), the transcript's hashes per round, the de-interleaving copy of the leaves, the scratch rows (in L1), allocation and page zeroing.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
