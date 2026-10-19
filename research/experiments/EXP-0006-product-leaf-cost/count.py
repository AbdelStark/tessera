#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0 OR MIT
"""EXP-0006: operation and byte counts per leaf of the product-tree leaf stage (E0).

The leaf stage as specified (RFC-0010 'Tree levels') and as implemented (crates/tessera-sumcheck:
forest.rs `prove_forests`, prover.rs `prove_batch`, kernel.rs; crates/tessera-poly: eq.rs,
multilinear.rs): a forest of 2^s trees of height h, N = 2^(s+h) leaves over Fp; one degree-3
sumcheck per level ℓ = 0..h−1 over tables of M_ℓ = 2^(s+ℓ) entries (eq over Fp2, the even and
odd halves of the level below over Fp); the eq table built per level by doubling; the first
round in the prime subfield; every table folded after every round (`bind_base` on the first
binding of an Fp table, in-place Fp2 folds after). Σ_ℓ M_ℓ = N − 2^s, so the whole tree costs
about twice its leaf level, and every per-leaf figure below is per leaf of the whole tree.

Every rule is one line naming the code it translates; every figure is E0 (own count, primary).
The peak rates are secondary (vendor documentation) and the utilizations are EXP-0002's.

    python3 -I count.py
"""

from __future__ import annotations

from dataclasses import dataclass, fields

# RFC-0020's c_π row shape: product-tree GKR batched over rows of 2^14 leaves. The harness
# measures 2^8 such rows, 2^22 leaves, the size of RFC-0010's largest test case and the bench.
TREES_LOG = 8
HEIGHT = 14
N = 1 << (TREES_LOG + HEIGHT)

# H100 SXM5: 132 SMs × 64 int32 lanes × 1.98 GHz (CUDA C++ Programming Guide, compute capability
# 9.0; NVIDIA H100 datasheet for the SM count and boost clock). HBM3 3.35 TB/s.
INT32_PEAK_OPS = 132 * 64 * 1.98e9
HBM_PEAK_BYTES = 3.35e12
INT_UTILIZATION = 0.60  # [VYTA25] via SUR-0002 and EXP-0002: field-multiplication kernels
HBM_UTILIZATION = 0.80


@dataclass(frozen=True)
class Ops:
    """Operation counts in the units the harness measures: Goldilocks multiplication, addition
    and subtraction; Fp2 multiplication (Karatsuba as ext2.rs implements it), Fp2-by-Fp
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
        """Goldilocks operations: ext2.rs `Mul` (Karatsuba) is v0 = a0·b0, v1 = a1·b1,
        cross = (a0 + a1)·(b0 + b1), c0 = v0 + mul_small(7)(v1), c1 = cross − v0 − v1: 3
        multiplications, one `mul_small(7)`, 3 additions, 2 subtractions; `mul_base` is 2
        multiplications; an Fp2 addition or subtraction is 2 of its kind."""
        return {
            "mul": self.fp_mul + 3 * self.fp2_mul + 2 * self.fp2_mulb,
            "mul7": self.fp2_mul,
            "add": self.fp_add + 3 * self.fp2_mul + 2 * self.fp2_add,
            "sub": self.fp_sub + 2 * self.fp2_mul + 2 * self.fp2_sub,
        }

    def mul_equivalents(self, w_is_mul: bool = False) -> float:
        """IDEA-0008's unit: Goldilocks multiplications, an Fp2 multiplication counting 3 (4 when
        the multiplication by W = 7 is charged as a full one) and a `mul_by_base` 2."""
        per_fp2 = 4.0 if w_is_mul else 3.0
        return self.fp_mul + per_fp2 * self.fp2_mul + 2 * self.fp2_mulb


# --- per-pair rules of the prover's loops, three nodes per round message (p(0), p(2), p(3)) ------
# kernel.rs `line` / `each_node` with N = 3, SKIP_ONE: Δ = b − a (1 sub), b + Δ (1 add), one more
# addition for the third node (the last `value += delta` is dead): 1 sub, 2 adds per table.
RULES = {
    # kernel.rs `product_sums`, mixed path (base = [even, odd] over Fp, ext = [eq] over Fp2):
    # `first_factor(even)` 1 sub + 2 adds; `next_factor(odd)` 1 sub + 2 adds + 3 multiplications
    # (`row[m] *= value`); `first_factor(eq)` 1 Fp2 sub + 2 Fp2 adds; then per node
    # `sums[m] += ext_row[m].mul_by_base(base_row[m])`: 3 mul_by_base + 3 Fp2 adds.
    "round1_mixed": Ops(fp_mul=3, fp_sub=2, fp_add=4, fp2_sub=1, fp2_add=5, fp2_mulb=3),
    # kernel.rs `zip3` over Fp2 (homogeneous_sums, N = 3 ≤ ZIP_EXT_NODES): three lines (3 subs,
    # 6 adds), `*p *= v` for two factors at three nodes (6 multiplications), `*sum += p` (3 adds).
    "zip3_ext": Ops(fp2_sub=3, fp2_add=9, fp2_mul=6),
    # kernel.rs `zip2` over Fp2: two lines (2 subs, 4 adds), 3 multiplications, 3 sum adds.
    "zip2_ext": Ops(fp2_sub=2, fp2_add=7, fp2_mul=3),
    # kernel.rs `zip3` over Fp (the first own round of three Fp tables, summed in Fp).
    "zip3_base": Ops(fp_sub=3, fp_add=9, fp_mul=6),
    # kernel.rs `bind_base`: per pair `r.mul_by_base(pair[1] − pair[0]).add_base(pair[0])`.
    "bind_base": Ops(fp_sub=1, fp_add=1, fp2_mulb=1),
    # kernel.rs `bind_ext` / `bind_in_place` and multilinear.rs `fold_in_place`: per pair
    # `pair[0] + r·(pair[1] − pair[0])` over Fp2.
    "bind_ext": Ops(fp2_sub=1, fp2_mul=1, fp2_add=1),
    # eq.rs `eq_table`: per entry of the finished table one `*one = *zero * r` and one
    # `*zero -= *one` (2^n − 1 of each for 2^n entries).
    "eq_entry": Ops(fp2_mul=1, fp2_sub=1),
    # forest.rs `level_above`: one Fp multiplication per entry of the level produced.
    "level_entry": Ops(fp_mul=1),
}

# int32 translation per Goldilocks operation, (low, high), from EXP-0002's count.py, which reads
# them line by line from crates/tessera-field/src/goldilocks.rs: a 64-bit add or subtract is two
# 32-bit instructions with a carry, a 64-bit select two, a wide 32×32→64 product one. Low is lazy
# reduction (no canonicalize), high is the code as written.
FIELD_OPS = {
    "mul": (18, 26, "4 wide products + 5–6 carried adds (9–10); reduce128: 9–16 (canonicalize skipped lazily)"),
    "add": (4, 8, "overflowing_add (2), EPSILON on carry (2), canonicalize (4; skipped lazily)"),
    "sub": (4, 5, "overflowing_sub (2), EPSILON on borrow (2), one more if the mask is materialized"),
    "mul7": (8, 13, "mul_small(7) (goldilocks.rs 132–140): x·7 as 96 bits, 2 wide products + 2 carried adds (4), hi·EPSILON (1), lo + t (2), carry fix (1); canonicalize (4) and a materialized mask (1) at the high end"),
}


def level_instance(m: int, nodes4: bool = False) -> Ops:
    """One level's sumcheck on tables of `m` entries: eq (Fp2, public), even and odd (Fp).

    prover.rs: round 0 is the instance's first own round (no padding for a single forest), read
    unbound through `product_sums`' mixed path over m/2 pairs; `bind` then calls `bind_base` on
    the two Fp tables and `bind_ext` on eq (m/2 pairs each); rounds j ≥ 1 run `zip3` over three
    Fp2 tables of m/2^j entries (m/2^(j+1) pairs; m/2 − 1 pairs in all) and `bind_in_place` on
    each (the same pair counts). With `nodes4`, the four-node count IDEA-0008 used (the padded
    instance's first own round, `node_sums`): one more multiplication per factor and one more
    sum per pair."""
    half = m // 2
    rest = half - 1
    round1 = RULES["round1_mixed"]
    zip3 = RULES["zip3_ext"]
    if nodes4:
        round1 = round1 + Ops(fp_mul=1, fp2_mulb=1, fp2_add=1)
        zip3 = zip3 + Ops(fp2_mul=2, fp2_add=1)
    return (
        round1 * half
        + RULES["bind_base"] * (2 * half)
        + RULES["bind_ext"] * half
        + zip3 * rest
        + RULES["bind_ext"] * (3 * rest)
    )


def forest_levels(trees_log: int, height: int) -> list[int]:
    """The table sizes M_ℓ = 2^(s+ℓ) of the levels ℓ = 0..h−1 of one forest."""
    return [1 << (trees_log + level) for level in range(height)]


def forest(trees_log: int, height: int, nodes4: bool = False) -> dict[str, Ops]:
    """The whole tree, by component."""
    n = 1 << (trees_log + height)
    sizes = forest_levels(trees_log, height)
    sumchecks = Ops()
    eq = Ops()
    for m in sizes:
        sumchecks = sumchecks + level_instance(m, nodes4)
        eq = eq + RULES["eq_entry"] * (m - 1)
    # forest.rs `build_levels`: levels 1..h−1 are built, 2^(s+t) entries each; the leaves are
    # de-interleaved without arithmetic and the roots are never built: N − 2^(s+1) products.
    build = RULES["level_entry"] * (n - (1 << (trees_log + 1)))
    return {"sumchecks": sumchecks, "eq": eq, "build": build, "total": sumchecks + eq + build}


def components_per_leaf(trees_log: int, height: int) -> dict[str, Ops]:
    """IDEA-0008's breakdown of the leaf-stage cost, per leaf of the whole tree."""
    n = 1 << (trees_log + height)
    sizes = forest_levels(trees_log, height)
    half = sum(m // 2 for m in sizes)
    rest = sum(m // 2 - 1 for m in sizes)
    return {
        "eq build": RULES["eq_entry"] * sum(m - 1 for m in sizes) * (1 / n),
        "round 1 (eq × even × odd, Fp first round)": RULES["round1_mixed"] * half * (1 / n),
        "fold after round 1 (2 bind_base + bind_ext)": (RULES["bind_base"] * (2 * half) + RULES["bind_ext"] * half) * (1 / n),
        "rounds ≥ 2 (zip3 over Fp2)": RULES["zip3_ext"] * rest * (1 / n),
        "folds after rounds ≥ 2 (3 bind_in_place)": RULES["bind_ext"] * (3 * rest) * (1 / n),
        "level build": RULES["level_entry"] * (n - (1 << (trees_log + 1))) * (1 / n),
    }


def ext3(m: int) -> Ops:
    """`prove_batch` on three Fp2 tables of `m` entries, degree 3: every round `zip3` over Fp2
    (m − 1 pairs in all) and a fold of each table (bind_ext, then bind_in_place)."""
    return RULES["zip3_ext"] * (m - 1) + RULES["bind_ext"] * (3 * (m - 1))


def base3(m: int) -> Ops:
    """`prove_batch` on three Fp tables of `m` entries, degree 3: the first round `zip3` in Fp
    and three `bind_base`, then as `ext3` on m/2 entries."""
    half = m // 2
    return RULES["zip3_base"] * half + RULES["bind_base"] * (3 * half) + RULES["zip3_ext"] * (half - 1) + RULES["bind_ext"] * (3 * (half - 1))


def bind_pass(m: int) -> Ops:
    """multilinear.rs `bind_lowest`: one `bind_base` pass over m/2 pairs."""
    return RULES["bind_base"] * (m // 2)


def bind_every_round(m: int) -> Ops:
    """`bind_lowest`, then `bind_lowest_in_place` down to one entry: m/2 − 1 more pairs."""
    return bind_pass(m) + RULES["bind_ext"] * (m // 2 - 1)


def fraction_leaf_level(m: int) -> Ops:
    """The fraction-tree proxy's leaf level (numerators 1): a degree-3 batch of [eq, d_e, d_o],
    [eq, d_e], [eq, d_o] on Fp2 tables of `m` entries. Each instance binds its own tables, the
    public eq included (prover.rs `State::bind` binds every table of the instance): 3 + 2 + 2 = 7
    table folds per pair."""
    pairs = m - 1
    return (RULES["zip3_ext"] + RULES["zip2_ext"] * 2) * pairs + RULES["bind_ext"] * (7 * pairs)


def fraction_generic_level(m: int) -> Ops:
    """A generic fraction-tree level: [eq, n_e, d_o], [eq, n_o, d_e], [eq, d_e, d_o] on Fp2
    tables of `m` entries; 9 table folds per pair."""
    pairs = m - 1
    return RULES["zip3_ext"] * (3 * pairs) + RULES["bind_ext"] * (9 * pairs)


def fraction_build(n: int) -> Ops:
    """The fraction tree's values, built by the harness in tessera-field arithmetic: leaves
    d = α − key − β·value (one mul_by_base, two Fp2 subtractions), the first layer with
    numerators 1 (n = d_e + d_o, d = d_e·d_o per pair), the layers above
    (n_L·d_R + n_R·d_L, d_L·d_R) per node (3 multiplications, 1 addition)."""
    leaves = Ops(fp2_mulb=1, fp2_sub=2) * n
    first = Ops(fp2_add=1, fp2_mul=1) * (n // 2)
    upper = Ops(fp2_mul=3, fp2_add=1) * (n // 2 - 1)
    return leaves + first + upper


def int32(expanded: dict[str, float], end: int) -> float:
    return sum(expanded[op] * FIELD_OPS[op][end] for op in ("mul", "mul7", "add", "sub"))


def bytes_per_leaf(trees_log: int, height: int) -> tuple[float, float]:
    """HBM bytes per leaf. Fused low end: the leaves are read once to build the levels (8 B),
    the built levels written and read back once (8 + 8 B), and each level's values read once by
    its sumcheck (2·M_ℓ entries of 8 B per level: 16 B per leaf), the eq table and the rounds on
    chip. Unfused high end, as the CPU code is written: per level of M entries the eq table
    written and read over Fp2 (16M + 16M), the even and odd halves read (16M), the first fold
    written (3 tables of M/2 Fp2: 24M), the later rounds read (48M) and folds written (24M):
    144M, summed over levels, plus the 24 B of the build."""
    n = 1 << (trees_log + height)
    sizes = forest_levels(trees_log, height)
    fused = 8 + 8 + 8 + sum(2 * m * 8 for m in sizes) / n
    unfused = 24 + sum(144 * m for m in sizes) / n
    return fused, unfused


def ps(ops: float, rate: float) -> float:
    return ops / rate * 1e12


def show_ops(label: str, ops: Ops, width: int = 46) -> None:
    print(
        "  %-*s Fp mul %6.3f  add %6.3f  sub %6.3f | Fp2 mul %6.3f  mul_by_base %6.3f  add %6.3f  sub %6.3f"
        % (width, label, ops.fp_mul, ops.fp_add, ops.fp_sub, ops.fp2_mul, ops.fp2_mulb, ops.fp2_add, ops.fp2_sub)
    )


def main() -> int:
    print("EXP-0006 counts: product-tree leaf stage, 2^%d trees of 2^%d leaves (N = 2^%d)" % (TREES_LOG, HEIGHT, TREES_LOG + HEIGHT))
    print()
    print("Per-pair rules (kernel.rs, eq.rs, forest.rs), three nodes per round message:")
    for name, ops in RULES.items():
        show_ops(name, ops, 14)
    print()

    print("Per leaf of the whole tree, by component (IDEA-0008's breakdown):")
    comps = components_per_leaf(TREES_LOG, HEIGHT)
    for label, ops in comps.items():
        show_ops(label, ops)
    total = forest(TREES_LOG, HEIGHT)["total"] * (1 / N)
    show_ops("total per leaf", total)
    ex = total.expand()
    print()
    print("Expanded to Goldilocks operations per leaf: %.3f multiplications, %.3f mul_small(7), %.3f additions, %.3f subtractions (%.3f additions or subtractions)"
          % (ex["mul"], ex["mul7"], ex["add"], ex["sub"], ex["add"] + ex["sub"]))
    print("Fp-multiplication equivalents per leaf (IDEA-0008's unit): %.2f with W cheap, %.2f with W as a multiplication; IDEA-0008 counted 29 (Karatsuba) to 46 at four nodes per round"
          % (total.mul_equivalents(), total.mul_equivalents(w_is_mul=True)))
    four = forest(TREES_LOG, HEIGHT, nodes4=True)["total"] * (1 / N)
    print("The same count at four nodes per round (node_sums, as a padded instance's first own round computes): %.2f Fp-multiplication equivalents, %.3f additions or subtractions"
          % (four.mul_equivalents(), four.expand()["add"] + four.expand()["sub"]))
    print()

    print("int32 translation of goldilocks.rs (low, high):")
    for name, (lo, hi, why) in FIELD_OPS.items():
        print("  %-5s %3d  %3d   %s" % (name, lo, hi, why))
    low_ops, high_ops = int32(ex, 0), int32(ex, 1)
    fused, unfused = bytes_per_leaf(TREES_LOG, HEIGHT)
    print()
    print("%-26s %12s %12s" % ("per leaf", "low (lazy)", "high (as written)"))
    print("%-26s %12.1f %12.1f" % ("int32 ops", low_ops, high_ops))
    print("%-26s %12.1f %12.1f" % ("HBM bytes (fused/unfused)", fused, unfused))
    mul_only_low = total.mul_equivalents() * FIELD_OPS["mul"][0]
    print("%-26s %12.1f %12s" % ("int32 ops, mul only", mul_only_low, "–"))
    print()
    peak_lo, peak_hi = ps(low_ops, INT32_PEAK_OPS), ps(high_ops, INT32_PEAK_OPS)
    util_lo, util_hi = ps(low_ops, INT32_PEAK_OPS * INT_UTILIZATION), ps(high_ops, INT32_PEAK_OPS * INT_UTILIZATION)
    mem_peak_lo, mem_peak_hi = ps(fused, HBM_PEAK_BYTES), ps(unfused, HBM_PEAK_BYTES)
    mem_util_lo, mem_util_hi = ps(fused, HBM_PEAK_BYTES * HBM_UTILIZATION), ps(unfused, HBM_PEAK_BYTES * HBM_UTILIZATION)
    print("H100 projection, ps per leaf (int32 peak %.3g op/s, HBM %.3g B/s):" % (INT32_PEAK_OPS, HBM_PEAK_BYTES))
    print("  compute at peak:          %6.1f  (low)   %6.1f  (high)" % (peak_lo, peak_hi))
    print("  compute at %.0f%%:           %6.1f  (low)   %6.1f  (high)" % (100 * INT_UTILIZATION, util_lo, util_hi))
    print("  memory at peak:           %6.1f  (fused) %6.1f  (unfused)" % (mem_peak_lo, mem_peak_hi))
    print("  memory at %.0f%%:            %6.1f  (fused) %6.1f  (unfused)" % (100 * HBM_UTILIZATION, mem_util_lo, mem_util_hi))
    print("  multiplications only, 18 int32 each, at peak: %.1f ps (IDEA-0008's reading, for comparison with SUR-0002's c_ℓ count)" % ps(mul_only_low, INT32_PEAK_OPS))
    lo = max(peak_lo, mem_peak_lo)
    hi = max(util_hi, mem_util_hi)
    central = round((lo * hi) ** 0.5)
    print()
    print("interval I = [lo, hi] = [%.1f, %.1f] ps; rho_pi = [%.2f, %.2f] against 12 ps; central (geometric mean) %d ps" % (lo, hi, lo / 12, hi / 12, central))
    bands = []
    if lo <= 24:
        bands.append("proceed (rho <= 2)")
    if lo <= 48 and hi > 24:
        bands.append("re-plan (2 < rho <= 4)")
    if hi > 48:
        bands.append("stop and redesign (rho > 4)")
    print("RFC-0020 bands the interval touches: " + "; ".join(bands))
    print("08 §2 trigger c_pi > 24 ps: %s at lo, %s at hi" % ("fires" if lo > 24 else "silent", "fires" if hi > 24 else "silent"))
    print("SC-2 (cost_model.py, corrected profile): mu(16K) <= 45 needs c_pi <= 60 ps -> %s at lo, %s at hi; mu(1K) <= 12 needs c_pi <= 238 ps -> %s at lo, %s at hi"
          % ("holds" if lo <= 60 else "fails", "holds" if hi <= 60 else "fails", "holds" if lo <= 238 else "fails", "holds" if hi <= 238 else "fails"))
    print()

    m = N // 2
    print("Counts the Rust fit uses (the harness computes the same rules from the sizes):")
    show_ops("level instance, per entry (M = 2^%d)" % (TREES_LOG + HEIGHT - 1), level_instance(m) * (1 / m))
    show_ops("three Fp2 tables, per entry", ext3(m) * (1 / m))
    show_ops("three Fp tables, per entry", base3(m) * (1 / m))
    show_ops("bind_lowest pass, per entry (2^%d)" % (TREES_LOG + HEIGHT), bind_pass(N) * (1 / N))
    show_ops("binding every round, per entry", bind_every_round(N) * (1 / N))
    show_ops("eq tables of every level, per leaf", forest(TREES_LOG, HEIGHT)["eq"] * (1 / N))
    show_ops("level build, per leaf", forest(TREES_LOG, HEIGHT)["build"] * (1 / N))
    show_ops("fraction proxy: build, per lookup", fraction_build(N) * (1 / N))
    show_ops("fraction proxy: leaf level, per entry", fraction_leaf_level(m) * (1 / m))
    show_ops("fraction proxy: generic level, per entry", fraction_generic_level(m) * (1 / m))
    frac = (fraction_build(N) + fraction_leaf_level(m) + fraction_generic_level(m)) * (1 / N)
    show_ops("fraction proxy: tree, per lookup", frac)
    print()
    print("c_pi / c_l by multiplication count: product tree %.2f against the proxy's %.2f Fp-multiplication equivalents per lookup -> %.2f; against SUR-0002's 56–74 -> %.2f–%.2f (IDEA-0008 expected 0.4–0.8)"
          % (total.mul_equivalents(), frac.mul_equivalents(), total.mul_equivalents() / frac.mul_equivalents(), total.mul_equivalents() / 74, total.mul_equivalents() / 56))
    print()
    print("Not counted: the gamma weights (3 Fp2 multiplications per instance per round), the transcript's hashes per round, the de-interleaving copy of the leaves, the scratch rows (in L1), allocation.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
