#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0 OR MIT
"""EXP-0011: commitments and lookups per layer per token of RFC-0005's two profiles at 405B shapes.

One tensor table (RFC-0005 'Integer semantics': every rescaled or lookup tensor of a dense Llama
block, with the profiles that contain it and its stated range cell) and one rule (RFC-0006 'Gadget
selection' with `B_max = 32`) give the baseline arms (the source's count with every rescale a single
lookup, and `minimal` under the rule) and the candidate arm (`servable` under the rule) in the same
code. Every key width is also derived from public data alone by exact interval arithmetic (weights in
[−127, 127], activations over their type range, the public matmul bound `127 · fan_in · max|x|`, the
14-bit RoPE constants, the `SoftmaxRow` fact `R_q · Σ e ≤ 2^{2σ}`), so that the stated cells are
checked rather than copied. Seeded random draws at toy shapes and adversarial inputs at the 405B shapes
check that the bounds are not exceeded and are attained. Finally the cost model (research/tools) is
evaluated at the `servable` count.

Usage:
    python3 -I research/experiments/EXP-0011-servable-count/count.py          # the report's tables
    python3 -I research/experiments/EXP-0011-servable-count/count.py --json   # machine-readable

Exit status 1 when a pre-registered check fails (README 'Success criterion').
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from dataclasses import dataclass, field, replace
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "tools"))
import cost_model  # noqa: E402

# Llama 3.1 405B shapes (00 §9 RISK-2, SUR-0009): d, d_ff, 128 query heads of 128, 8 KV heads, 126 layers.
D = 16_384
D_FF = 53_248
N_Q_HEADS = 128
N_KV_HEADS = 8
D_HEAD = 128
LAYERS = 126
D_KV = N_KV_HEADS * D_HEAD  # 1,024
assert N_Q_HEADS * D_HEAD == D

# Integer model constants (RFC-0005 'Integer semantics', RFC-0008 `Rope` table).
INT8 = (-128, 127)
INT20 = (-(1 << 19), (1 << 19) - 1)
W_MAX = 127  # weights are quantized into [−127, 127]
KAPPA = 16  # RMSNorm `κ`
SIGMA = 16  # softmax `σ`: `T_exp[0] = 2^σ`, `R_q = ⌊2^{2σ} / s_q⌋`
ROPE_BITS = 14  # `(C, S)` 14-bit constants
MAX_CONTEXT = 1 << 17  # positions in [0, max_context), 131,072

# RFC-0006 'Gadget selection'.
B_MAX = 32
B_CEIL = 2 * B_MAX - 2  # 62: above it `ERR-COMPILE-RANGE`

# Known values the harness must reproduce (criterion (a)) and the idea note's hand count (criterion (b), (d), (e)).
KNOWN_SOURCE = cost_model.SOURCE_COMMITS_PER_LAYER_TOKEN  # 137,216 (source §5, 00 §9 RISK-2)
KNOWN_RISK2 = cost_model.RISK2_COMMITS_PER_LAYER_TOKEN  # 220,160 (00 §9 RISK-2)
STATED = {
    "servable_elements": 260_096,
    "servable_commitments": 326_656,
    "servable_split_added": 66_560,
    "after_idea_0011": 293_888,
    "after_rope_route": 276_480,
    "after_fused_rescale": 223_232,
    "mu": {"corrected": {1024: 10.4, 16384: 36.6}, "counted": {1024: 21.6, 16384: 82.7}},
    "mu_tolerance": 0.05,  # RFC-0020 per-term tolerance
}
ROUNDED_PER_TOKEN = 4.116e7  # the idea note's `--set` value

SEED = 0x5EED0033
TOY_SHAPES = (4, 8, 16)
TOY_DRAWS = 2_000

Interval = tuple[int, int]


# --- exact interval arithmetic -------------------------------------------------------------------


def imul(a: Interval, b: Interval) -> Interval:
    products = (a[0] * b[0], a[0] * b[1], a[1] * b[0], a[1] * b[1])
    return (min(products), max(products))


def iadd(a: Interval, b: Interval) -> Interval:
    return (a[0] + b[0], a[1] + b[1])


def isub(a: Interval, b: Interval) -> Interval:
    return (a[0] - b[1], a[1] - b[0])


def matmul_weight(x: Interval, fan_in: int) -> Interval:
    """RFC-0006 'Range analysis': `MatMulWeight` uses only the public bound `127 · fan_in · max|x|`."""
    m = W_MAX * fan_in * max(abs(x[0]), abs(x[1]))
    return (-m, m)


def signed_width(iv: Interval) -> int:
    """The smallest `b` with `[lo, hi] ⊆ [−2^{b−1}, 2^{b−1} − 1]` (RFC-0006: "signed range width")."""
    b = 1
    while not (-(1 << (b - 1)) <= iv[0] and iv[1] <= (1 << (b - 1)) - 1):
        b += 1
    return b


def gadget_class(b: int) -> str:
    if b <= B_MAX:
        return "single"
    if b <= B_CEIL:
        return "split"
    return "error"


def split_point(b: int) -> int | None:
    return b - B_MAX if gadget_class(b) == "split" else None


def cost(b: int) -> int:
    """Commitments (= lookups) per element under the rule; `None` class raises, as the compiler would."""
    c = gadget_class(b)
    if c == "error":
        raise ValueError(f"width {b} exceeds 2·B_max − 2 = {B_CEIL}: ERR-COMPILE-RANGE")
    return 1 if c == "single" else 2


# --- the derived intervals -----------------------------------------------------------------------

ACC_D = matmul_weight(INT8, D)  # Q, K, V, gate, up, servable O: INT8 operands, fan-in d
ACC_DFF_INT8 = matmul_weight(INT8, D_FF)  # servable down projection: INT8 operands, fan-in d_ff
ROPE_C_TIGHT = (-(1 << (ROPE_BITS - 1)), 1 << (ROPE_BITS - 1))  # signed 14-bit: |C|, |S| ≤ 2^13
ROPE_C_LOOSE = (-((1 << ROPE_BITS) - 1), (1 << ROPE_BITS) - 1)  # the looser reading: |C|, |S| < 2^14
ROPE_KEY_TIGHT = isub(imul(ACC_D, ROPE_C_TIGHT), imul(ACC_D, ROPE_C_TIGHT))  # a·C − b·S
ROPE_KEY_LOOSE = isub(imul(ACC_D, ROPE_C_LOOSE), imul(ACC_D, ROPE_C_LOOSE))
# Softmax: `e_qi = T_exp[c_q − x_qi] ∈ [0, 2^σ]` with `T_exp[0] = 2^σ` at the row maximum, so `s_q ≥ 2^σ`
# and `R_q = ⌊2^{2σ}/s_q⌋ ≤ 2^σ`; the fact `R_q · s_q ≤ 2^{2σ}` bounds `o = R_q Σ e v` by `2^{2σ} · max|v|`.
E_MAX = 1 << SIGMA
S_Q = (E_MAX, MAX_CONTEXT * E_MAX)  # at least the maximal element, at most every position maximal
R_Q_TIGHT = (1, E_MAX)  # using `s_q ≥ T_exp[0]`
R_Q_NAIVE = (1, 1 << (2 * SIGMA))  # using only `s_q ≥ 1`
O_FACT = imul((1, 1 << (2 * SIGMA)), INT8)  # `R_q · s_q ∈ [1, 2^{2σ}]` times `v ∈ [−128, 127]`
O_PRODUCT_TIGHT = imul(R_Q_TIGHT, imul(S_Q, INT8))  # the product bound the fact replaces, `R_q ≤ 2^σ`
O_PRODUCT_NAIVE = imul(R_Q_NAIVE, imul(S_Q, INT8))  # the same with `R_q ≤ 2^{2σ}`
ACC_O_MINIMAL = matmul_weight(O_FACT, D)  # `minimal`: `W_O o`
H_KEY = imul(INT8, INT8)  # `a · u8`
AU_MINIMAL = imul(INT8, ACC_D)  # `minimal`: `a · u` with `u` the unrescaled up accumulator
ACC_DOWN_MINIMAL = matmul_weight(AU_MINIMAL, D_FF)
NORM_FACT_KEY = (-(1 << KAPPA) * math.isqrt(D), (1 << KAPPA) * math.isqrt(D))  # IDEA-0011: `|res·ρ| ≤ 2^κ √d`
assert math.isqrt(D) ** 2 == D

# The norm key's class depends on the `QuantSpec` constant `e`, the exact integer form of `d·ε/σ_res²` rounded up
# (RFC-0005 'Integer semantics'), with Llama's `ε = 10^-5` and `σ_res = 2^{e_res}` chosen with one bit of headroom
# over the calibration maximum of the residual stream ('Scale policy'). `e ≤ 2^20` means `σ_res ≥ 2^-11.3`, a
# residual stream whose INT20 range spans at least ±206 real units; every smaller stream is far below anything
# reported for Llama (SUR-0009 Question 1: residual outliers in the thousands). The stated `< 2^37` cell sits at
# `e ∈ [1,024, 4,096)`. The first run of this script encoded the margin as `2^22`; the exact flip is 4,192,257
# (2^21.999), so that constant was wrong by 0.05% and is replaced by a derived interval (README 'Analysis').
RMS_NORM_EPS = 1e-5
E_ADMISSIBLE_MAX = 1 << 20


def sigma_res_for(e: int) -> float:
    """The residual scale `σ_res` at which `⌈d·ε/σ_res²⌉ = e`, as a power of two exponent (log2)."""
    return math.log2(math.sqrt(D * RMS_NORM_EPS / e))


def rho_max(e: int) -> int:
    """`ρ = max{ρ : ρ² (s + e) ≤ d · 2^{2κ}}` at the worst case `s = 0`, for the `QuantSpec` constant `e ≥ 1`."""
    return math.isqrt(D * (1 << (2 * KAPPA)) // e)


def norm_key(e: int) -> Interval:
    return imul(INT20, (0, rho_max(e)))


def norm_flip_e() -> int:
    """The smallest `e` at which the norm key's class under the rule stops being a split."""
    lo, hi = 1, 1 << 60
    assert gadget_class(signed_width(norm_key(lo))) == "split"
    assert gadget_class(signed_width(norm_key(hi))) == "single"
    while hi - lo > 1:
        mid = (lo + hi) // 2
        if gadget_class(signed_width(norm_key(mid))) == "split":
            lo = mid
        else:
            hi = mid
    return hi


# --- the tensor table ----------------------------------------------------------------------------


@dataclass(frozen=True)
class Tensor:
    name: str
    elements: int
    profiles: frozenset[str]  # subset of {"minimal", "servable"}
    stated_lt: int  # RFC-0005's cell `< 2^k`: this `k`
    derived: Interval  # the exact interval from public data
    kind: str = "requant"  # or "tablemap"
    note: str = ""
    alt_derived: dict[str, Interval] = field(default_factory=dict)  # other readings, for the report

    def stated_widths(self) -> tuple[int, int]:
        return (self.stated_lt, self.stated_lt + 1)

    def derived_width(self) -> int:
        return signed_width(self.derived)


BOTH = frozenset({"minimal", "servable"})
MIN = frozenset({"minimal"})
SRV = frozenset({"servable"})

NORM_E_STATED = 1 << 11  # the `e` at which the public-data derivation reproduces the stated `< 2^37` cell (reported)


def tensor_table() -> list[Tensor]:
    return [
        Tensor("attn norm y = rq(res·ρ)", D, BOTH, 37, norm_key(NORM_E_STATED), note="key depends on QuantSpec `e`; class reported over all admissible `e`",
               alt_derived={"e=1": norm_key(1)}),
        Tensor("RoPE q = rq(a·C − b·S)", D, BOTH, 44, ROPE_KEY_TIGHT, note="|C|,|S| ≤ 2^13 (signed 14-bit)", alt_derived={"|C|,|S| < 2^14": ROPE_KEY_LOOSE}),
        Tensor("RoPE k = rq(a·C − b·S)", D_KV, BOTH, 44, ROPE_KEY_TIGHT, note="as q", alt_derived={"|C|,|S| < 2^14": ROPE_KEY_LOOSE}),
        Tensor("v = rq(acc)", D_KV, BOTH, 29, ACC_D),
        Tensor("o8 = rq(o, r_o, −128, 127)", D, SRV, 39, O_FACT, note="`SoftmaxRow` fact `R_q s_q ≤ 2^{2σ}`, `|v| ≤ 128`",
               alt_derived={"product bound, R_q ≤ 2^σ": O_PRODUCT_TIGHT, "product bound, R_q ≤ 2^{2σ}": O_PRODUCT_NAIVE}),
        Tensor("residual rq(acc_O) [servable: acc = W_O o8]", D, SRV, 29, ACC_D),
        Tensor("residual rq(acc_O) [minimal: acc = W_O o]", D, MIN, 60, ACC_O_MINIMAL),
        Tensor("MLP norm y = rq(res·ρ)", D, BOTH, 37, norm_key(NORM_E_STATED), note="as attn norm", alt_derived={"e=1": norm_key(1)}),
        Tensor("SiLU a = T_silu[g]", D_FF, BOTH, 29, ACC_D, kind="tablemap"),
        Tensor("u8 = rq(u)", D_FF, SRV, 29, ACC_D),
        Tensor("h = rq(a·u8)", D_FF, SRV, 15, H_KEY),
        Tensor("residual rq(acc_down) [servable: acc = W_down h]", D, SRV, 30, ACC_DFF_INT8),
        Tensor("residual rq(acc_down) [minimal: acc = W_down (a·u)]", D, MIN, 58, ACC_DOWN_MINIMAL),
    ]


# --- the arms ------------------------------------------------------------------------------------


@dataclass(frozen=True)
class Arm:
    name: str
    elements: int
    commitments: int
    lookups: int
    splits: tuple[tuple[str, int], ...]  # (tensor, added commitments)
    rows: tuple[tuple[str, int, int, str, int], ...]  # (tensor, elements, width, class, commitments)


def count(name: str, profile: str, tensors: list[Tensor], width_of, force_single: bool = False) -> Arm:
    rows, splits = [], []
    elements = commitments = 0
    for t in tensors:
        if profile not in t.profiles:
            continue
        b = width_of(t)
        per = 1 if force_single else cost(b)
        c = "single (forced)" if force_single else gadget_class(b)
        rows.append((t.name, t.elements, b, c, per * t.elements))
        elements += t.elements
        commitments += per * t.elements
        if per == 2:
            splits.append((t.name, t.elements))
    return Arm(name, elements, commitments, commitments, tuple(splits), tuple(rows))


def stated_width(t: Tensor) -> int:
    """The idea note's reading: `< 2^k` is `k` bits of magnitude, `b = k + 1` (so `o8` is 40 and the norm 38)."""
    return t.stated_lt + 1


def derived_width(t: Tensor) -> int:
    return t.derived_width()


def fixed_points(tensors: list[Tensor]) -> dict[str, Arm]:
    def after_0011(ts: list[Tensor]) -> list[Tensor]:
        return [replace(t, derived=NORM_FACT_KEY, stated_lt=signed_width(NORM_FACT_KEY) - 1) if "norm" in t.name else t for t in ts]

    def after_rope(ts: list[Tensor]) -> list[Tensor]:
        # A route that brings `q`, `k` to one lookup (SUR-0009 Question 1): the key is modelled as `≤ B_max` bits.
        return [replace(t, derived=(-(1 << (B_MAX - 1)), (1 << (B_MAX - 1)) - 1), stated_lt=B_MAX - 1) if t.name.startswith("RoPE") else t for t in ts]

    def after_fused(ts: list[Tensor]) -> list[Tensor]:
        # `h = rq(a · u)` directly: the `u8` rescale disappears; `h`'s key `a · u` is `|a| · |u| < 2^36`, a split
        # unless a fact or placement bounds it, so the fixed point of the idea note (223,232) assumes `h` stays single.
        return [t for t in ts if not t.name.startswith("u8")]

    t1 = after_0011(tensors)
    t2 = after_rope(t1)
    t3 = after_fused(t2)
    return {
        "servable after IDEA-0011": count("servable after IDEA-0011", "servable", t1, derived_width),
        "servable after IDEA-0011 + RoPE route": count("servable after IDEA-0011 + RoPE route", "servable", t2, derived_width),
        "servable after IDEA-0011 + RoPE route + fused rescale": count("servable after all three", "servable", t3, derived_width),
        "minimal after IDEA-0011": count("minimal after IDEA-0011", "minimal", t1, derived_width),
        "minimal after IDEA-0011 + RoPE route": count("minimal after IDEA-0011 + RoPE route", "minimal", t2, derived_width),
    }


# --- attained extremes and seeded draws ----------------------------------------------------------


class SplitMix64:
    MASK = (1 << 64) - 1

    def __init__(self, seed: int) -> None:
        self.state = seed & self.MASK

    def next_u64(self) -> int:
        self.state = (self.state + 0x9E3779B97F4A7C15) & self.MASK
        z = self.state
        z = ((z ^ (z >> 30)) * 0xBF58476D1CE4E5B9) & self.MASK
        z = ((z ^ (z >> 27)) * 0x94D049BB133111EB) & self.MASK
        return z ^ (z >> 31)

    def below(self, n: int) -> int:
        return self.next_u64() % n

    def in_range(self, lo: int, hi: int) -> int:
        return lo + self.below(hi - lo + 1)


def rq(x: int, r: int, lo: int, hi: int) -> int:
    """RFC-0005: `clamp(floor((x + 2^{r−1}) / 2^r), lo, hi)`, `r ≥ 1`."""
    return min(max((x + (1 << (r - 1))) >> r, lo), hi)


def t_exp(k: int, theta: float) -> int:
    """`T_exp[k] = round(2^σ · exp(−k · θ))`, `θ = σ_q σ_k / √d_head` (a toy value here; only `T_exp[0] = 2^σ` matters)."""
    return round((1 << SIGMA) * math.exp(-k * theta))


def softmax_row_o(scores: list[int], values: list[int], theta: float) -> tuple[int, int, int]:
    c = max(scores)
    es = [t_exp(c - x, theta) for x in scores]
    s_q = sum(es)
    r_q = (1 << (2 * SIGMA)) // s_q
    o = r_q * sum(e * v for e, v in zip(es, values))
    return o, r_q, s_q


def extremes_and_draws() -> dict:
    rng = SplitMix64(SEED)
    out: dict = {"seed": hex(SEED), "toy_shapes": TOY_SHAPES, "draws_per_shape": TOY_DRAWS, "violations": 0, "attained": {}}
    # Matmul bound at toy shapes: random weights and activations never exceed it; the adversarial input attains it.
    for d in TOY_SHAPES:
        bound = matmul_weight(INT8, d)
        for _ in range(TOY_DRAWS):
            acc = sum(rng.in_range(-W_MAX, W_MAX) * rng.in_range(*INT8) for _ in range(d))
            if not bound[0] <= acc <= bound[1]:
                out["violations"] += 1
        adv = sum(W_MAX * INT8[0] for _ in range(d))
        out["attained"][f"matmul d={d}"] = adv == bound[0]
    # RoPE key at toy shapes with the tight constant reading.
    for d in TOY_SHAPES:
        acc = matmul_weight(INT8, d)
        key = isub(imul(acc, ROPE_C_TIGHT), imul(acc, ROPE_C_TIGHT))
        for _ in range(TOY_DRAWS):
            a, b = rng.in_range(*acc), rng.in_range(*acc)
            cc, ss = rng.in_range(*ROPE_C_TIGHT), rng.in_range(*ROPE_C_TIGHT)
            if not key[0] <= a * cc - b * ss <= key[1]:
                out["violations"] += 1
        out["attained"][f"rope d={d}"] = acc[1] * ROPE_C_TIGHT[1] - acc[0] * ROPE_C_TIGHT[1] == key[1]
    # Softmax row: random rows never exceed the fact or the `o` bound; the one-position row attains `o = −2^39`.
    theta = 1 / 64
    for n in (1, 2, 4, 8, 16):
        for _ in range(TOY_DRAWS // 4):
            scores = [rng.in_range(-(1 << 21) + 1, (1 << 21) - 1) for _ in range(n)]
            values = [rng.in_range(*INT8) for _ in range(n)]
            o, r_q, s_q = softmax_row_o(scores, values, theta)
            if r_q * s_q > (1 << (2 * SIGMA)) or not O_FACT[0] <= o <= O_FACT[1] or r_q > E_MAX or s_q < E_MAX:
                out["violations"] += 1
    o, r_q, s_q = softmax_row_o([0], [-128], theta)
    out["attained"]["o = −2^39 (one position, v = −128)"] = o == -(1 << 39) and r_q * s_q == 1 << (2 * SIGMA)
    out["o_extreme"] = {"o": o, "R_q": r_q, "s_q": s_q, "R_q*s_q": r_q * s_q}
    # The `h` key: `a · u8` reaches 128 · 128 = 2^14 exactly, so its signed width is 16 under the strict convention.
    out["attained"]["h key = 2^14"] = H_KEY[1] == 1 << 14
    # `rq` sanity: round half up and saturation at both ends.
    out["rq_checks"] = all([rq(0, 1, -128, 127) == 0, rq(1, 1, -128, 127) == 1, rq(-1, 1, -128, 127) == 0, rq(1 << 40, 8, -128, 127) == 127, rq(-(1 << 40), 8, -128, 127) == -128])
    return out


# --- the cost model ------------------------------------------------------------------------------


def run_cost_model(per_token: float) -> dict:
    results = {}
    for name in ("corrected", "counted"):
        base = cost_model.PROFILES[name]
        scen = replace(base, commits_per_token=per_token, lookups_per_token=per_token)
        rows = {}
        for ctx in cost_model.DEFAULT_CONTEXTS:
            b, s = cost_model.mu(base, ctx), cost_model.mu(scen, ctx)
            rows[ctx] = {"baseline": round(b.total, 3), "scenario": round(s.total, 3), "linear": round(s.linear, 3), "delta_pct": round(100 * (s.total - b.total) / b.total, 1)}
        results[name] = rows
    return results


# --- main ----------------------------------------------------------------------------------------


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--json", action="store_true")
    args = ap.parse_args(argv)

    tensors = tensor_table()
    arms = {
        "source (minimal list, every rescale one lookup)": count("source", "minimal", tensors, stated_width, force_single=True),
        "minimal (rule, derived widths)": count("minimal", "minimal", tensors, derived_width),
        "minimal (rule, stated cells)": count("minimal", "minimal", tensors, stated_width),
        "servable (rule, derived widths)": count("servable", "servable", tensors, derived_width),
        "servable (rule, stated cells)": count("servable", "servable", tensors, stated_width),
    }
    fps = fixed_points(tensors)
    servable = arms["servable (rule, derived widths)"]
    minimal = arms["minimal (rule, derived widths)"]
    source = arms["source (minimal list, every rescale one lookup)"]
    per_token_exact = servable.commitments * LAYERS

    # (c) derivation against stated cells, under both readings of `< 2^k`.
    derivation = []
    class_agreement = True
    for t in tensors:
        b = t.derived_width()
        classes = {gadget_class(b), gadget_class(t.stated_lt), gadget_class(t.stated_lt + 1)}
        agree = len(classes) == 1
        class_agreement &= agree
        derivation.append({
            "tensor": t.name, "elements": t.elements, "profiles": sorted(t.profiles), "stated": f"< 2^{t.stated_lt}",
            "derived_interval": [t.derived[0], t.derived[1]], "derived_b": b, "derived_class": gadget_class(b),
            "split_point": split_point(b), "stated_b_readings": [t.stated_lt, t.stated_lt + 1], "classes_agree": agree,
            "alt": {k: {"interval": [v[0], v[1]], "b": signed_width(v), "class": gadget_class(signed_width(v))} for k, v in t.alt_derived.items()},
            "note": t.note,
        })
    flip = norm_flip_e()
    norm_scan = {str(e): {"rho_max": rho_max(e), "b": signed_width(norm_key(e)), "class": gadget_class(signed_width(norm_key(e))),
                          "log2_sigma_res": round(sigma_res_for(e), 2), "residual_span_real": round((1 << 19) * 2 ** sigma_res_for(e), 1)}
                 for e in (1, 1 << 10, 1 << 11, 1 << 12, 1 << 16, E_ADMISSIBLE_MAX, flip - 1, flip, 1 << 22)}
    o8 = next(d for d in derivation if d["tensor"].startswith("o8"))

    extremes = extremes_and_draws()
    cm_exact = run_cost_model(float(per_token_exact))
    cm_rounded = run_cost_model(ROUNDED_PER_TOKEN)
    mu_ok = True
    mu_dev = {}
    for name, cells in STATED["mu"].items():
        for ctx, stated in cells.items():
            dev_exact = cm_exact[name][ctx]["scenario"] - stated
            dev_rounded = cm_rounded[name][ctx]["scenario"] - stated
            mu_dev[f"{name} {ctx // 1024}K"] = {"stated": stated, "exact": cm_exact[name][ctx]["scenario"], "rounded": cm_rounded[name][ctx]["scenario"],
                                                "dev_exact": round(dev_exact, 3), "dev_rounded": round(dev_rounded, 3)}
            mu_ok &= abs(dev_exact) <= STATED["mu_tolerance"] and round(cm_rounded[name][ctx]["scenario"], 1) == round(cm_exact[name][ctx]["scenario"], 1)

    checks = {
        "(a) source reproduces 137,216": source.commitments == KNOWN_SOURCE,
        "(a) minimal under the rule reproduces 220,160": minimal.commitments == KNOWN_RISK2 and arms["minimal (rule, stated cells)"].commitments == KNOWN_RISK2,
        "(b) servable elements 260,096": servable.elements == STATED["servable_elements"],
        "(b) servable commitments 326,656 = lookups": servable.commitments == STATED["servable_commitments"] == servable.lookups
        and arms["servable (rule, stated cells)"].commitments == STATED["servable_commitments"],
        "(b) four split groups adding 66,560": len(servable.splits) == 5 and sum(n for _, n in servable.splits) == STATED["servable_split_added"]
        and {n.split(" ")[0] for n, _ in servable.splits} == {"attn", "MLP", "RoPE", "o8"},
        "(c) every class agrees under both readings": class_agreement,
        "(c) o8 derives to b = 40 with an attained extreme": o8["derived_b"] == 40 and extremes["attained"]["o = −2^39 (one position, v = −128)"],
        "(c) norm key is a split for every admissible e (e ≤ 2^20; flip point reported)": flip > E_ADMISSIBLE_MAX
        and all(gadget_class(signed_width(norm_key(e))) == "split" for e in (1, E_ADMISSIBLE_MAX, flip - 1)),
        "(c) no random draw exceeded a bound": extremes["violations"] == 0 and all(extremes["attained"].values()) and extremes["rq_checks"],
        "(d) fixed points 293,888 / 276,480 / 223,232": (
            fps["servable after IDEA-0011"].commitments == STATED["after_idea_0011"]
            and fps["servable after IDEA-0011 + RoPE route"].commitments == STATED["after_rope_route"]
            and fps["servable after IDEA-0011 + RoPE route + fused rescale"].commitments == STATED["after_fused_rescale"]
        ),
        "(e) μ cells within ±0.05 and equal at the rounded count": mu_ok,
        "(e) --check passes on the source profile": None,  # filled below (prints)
    }

    if args.json:
        import io
        import contextlib

        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            checks["(e) --check passes on the source profile"] = cost_model.check()
        out = {
            "shapes": {"d": D, "d_ff": D_FF, "q_heads": N_Q_HEADS, "kv_heads": N_KV_HEADS, "d_head": D_HEAD, "layers": LAYERS},
            "rule": {"B_max": B_MAX, "ceiling": B_CEIL},
            "derivation": derivation,
            "norm_key_scan_over_e": norm_scan,
            "norm_flip_e": flip,
            "norm_e_admissible_max": E_ADMISSIBLE_MAX,
            "arms": {k: {"elements": a.elements, "commitments": a.commitments, "lookups": a.lookups, "splits": a.splits, "rows": a.rows} for k, a in arms.items()},
            "fixed_points": {k: {"elements": a.elements, "commitments": a.commitments, "splits": a.splits} for k, a in fps.items()},
            "per_token": {"servable_exact": per_token_exact, "rounded_used_by_idea": ROUNDED_PER_TOKEN, "minimal": minimal.commitments * LAYERS, "source": source.commitments * LAYERS},
            "ratios": {"servable/source": round(servable.commitments / source.commitments, 4), "servable/minimal": round(servable.commitments / minimal.commitments, 4)},
            "extremes": extremes,
            "cost_model_exact": cm_exact,
            "cost_model_rounded": cm_rounded,
            "mu_deviation": mu_dev,
            "check_output": buf.getvalue(),
            "checks": checks,
            "all_pass": all(checks.values()),
        }
        print(json.dumps(out, indent=2, default=list))
        return 0 if out["all_pass"] else 1

    print("EXP-0011: commitments and lookups per layer per token (E0, exact integers)")
    print(f"shapes: d={D:,} d_ff={D_FF:,} q_heads={N_Q_HEADS} kv_heads={N_KV_HEADS} d_head={D_HEAD} layers={LAYERS}; rule: B_max={B_MAX}, ceiling {B_CEIL}")
    print()
    print("Derivation of every key width from public data (RFC-0006 range analysis), against RFC-0005's stated cell")
    print(f"{'tensor':<52} {'elements':>8} {'stated':>8} {'derived b':>9} {'class':>7} {'s':>3}  agree  interval")
    for d in derivation:
        s = "" if d["split_point"] is None else str(d["split_point"])
        print(f"{d['tensor']:<52} {d['elements']:>8,} {d['stated']:>8} {d['derived_b']:>9} {d['derived_class']:>7} {s:>3}  {str(d['classes_agree']):<5}  [{d['derived_interval'][0]:,}, {d['derived_interval'][1]:,}]")
        for k, v in d["alt"].items():
            print(f"{'':<52} {'':>8} {'':>8} {v['b']:>9} {v['class']:>7}      reading: {k}  [{v['interval'][0]:,}, {v['interval'][1]:,}]")
    print()
    print(f"Norm key `res · ρ` as a function of the QuantSpec constant e (ρ_max = ⌊√(d·2^(2κ)/e)⌋, |res| ≤ 2^19); class flips to single at e = {flip:,} = 2^{math.log2(flip):.3f}; admissible e ≤ {E_ADMISSIBLE_MAX:,} = 2^20")
    for e, v in norm_scan.items():
        print(f"  e = {int(e):>10,}: ρ_max = {v['rho_max']:>10,}  b = {v['b']:>2}  {v['class']:<6}  σ_res = 2^{v['log2_sigma_res']:>6}  residual span ±{v['residual_span_real']:,} real units")
    print()
    for title, arm in list(arms.items()) + [(k, v) for k, v in fps.items()]:
        print(f"{title}: elements {arm.elements:,}, commitments {arm.commitments:,}, lookups {arm.lookups:,}, per token ×{LAYERS}: {arm.commitments * LAYERS:,}")
        for name, n, b, c, com in arm.rows:
            print(f"    {name:<52} {n:>8,}  b={b:>2}  {c:<16} {com:>8,}")
        if arm.splits:
            print(f"    split groups: {', '.join(f'{n} (+{k:,})' for n, k in arm.splits)}; added {sum(k for _, k in arm.splits):,}")
        print()
    print(f"ratios: servable/source {servable.commitments / source.commitments:.3f}×, servable/minimal {servable.commitments / minimal.commitments:.3f}×")
    print()
    print(f"Attained extremes and seeded draws (SplitMix64 seed {extremes['seed']}, toy shapes {TOY_SHAPES}, {TOY_DRAWS} draws per shape): violations {extremes['violations']}")
    for k, v in extremes["attained"].items():
        print(f"  attained: {k}: {v}")
    print(f"  o extreme: {extremes['o_extreme']}")
    print()
    for label, cm in (("exact", cm_exact), ("rounded 4.116e7", cm_rounded)):
        per_token = per_token_exact if label == "exact" else ROUNDED_PER_TOKEN
        print(f"Cost model at commits_per_token = lookups_per_token = {per_token:,.0f} ({label})")
        for name, rows in cm.items():
            print(f"  profile {name!r}: " + "; ".join(f"{ctx // 1024}K {r['baseline']:.1f} → {r['scenario']:.1f} ({r['delta_pct']:+.1f}%, linear {r['linear']:.2f})" for ctx, r in rows.items()))
    print()
    print("μ against the pre-registered cells (tolerance ±0.05):")
    for k, v in mu_dev.items():
        print(f"  {k:<14} stated {v['stated']:>5}  exact {v['exact']:>7.3f} (dev {v['dev_exact']:+.3f})  rounded {v['rounded']:>7.3f} (dev {v['dev_rounded']:+.3f})")
    print()
    checks["(e) --check passes on the source profile"] = cost_model.check()
    print()
    print("Pre-registered checks:")
    for k, v in checks.items():
        print(f"  {'ok  ' if v else 'FAIL'} {k}")
    ok = all(checks.values())
    print(f"\n{'all checks pass' if ok else 'A CHECK FAILED'}")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
