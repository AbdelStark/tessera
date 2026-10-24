#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0 OR MIT
"""Reference reports of the soundness calculator (RFC-0014 'Soundness calculator'): every row of
the soundness table with its bits, a FixedBits lower bound, its target and whether it passes.

The reports cover RFC-0014's three parameter sets at the 405B 16K-token, 16K-context statement;
statements and parameters made to miss exactly one target; the edges of the exact comparisons;
and random statements under random parameter sets. Each report is a `tessera.soundness/1` object
(spec/schemas/soundness-v1.schema.json), the one `tessera params check --json` prints.

This script is the independent oracle. It is written from RFC-0014 ('Soundness budget',
'Grinding', 'Parameter sets', 'Soundness calculator'), RFC-0049 'The calculator' (a weight
matrix's count in tessera-protocol/1), RFC-0026 'Schema' (the layouts), RFC-0040 'The matrices'
(the 405B weight matrices), RFC-0030 'Cost' (the Exp instance), RFC-0054 'Soundness' (the leaf
stage in token groups), RFC-0010 'Tree levels' (a level's rounds) and 06 §4 (SP-1) alone. It shares
no code with crates/, verifier/ or python/, which check its output. It states the 405B lookups of
docs/perf/v0.2/commitment-report-405b-servable-c2-b32.json, the compiler's report, which RFC-0014
'Soundness calculator' names as their source, and the lowering's 53,489 split values, which
crates/tessera-bench/tests/commitment_report.rs counts.

Every bit count is computed twice, and every decision once more:
- by LOG2 of RFC-0014 'Soundness calculator', with Python's integers: the FixedBits both
  calculators must reproduce bit for bit;
- as the true value λ = g + max(0, −log2 ε) with 60-digit decimals, against which the script
  checks that every FixedBits r satisfies r ≤ 256·λ < r + 1 + 2^-20, and every per-query bit count
  r' satisfies r' ≤ 256·b(ρ') < r' + 1 + 2^-37;
- every pass or fail with exact integers, c·2^T ≤ q and A^τ ≥ 2^T·B^τ, and again against the true
  value whenever it lies further than 10^-30 from the target.

Usage: python3 spec/vectors/soundness/generate.py
(rewrites soundness.json)

Random inputs come from SplitMix64 with the seed below, so any implementation can regenerate them.
"""

import json
from decimal import ROUND_FLOOR, Decimal, getcontext
from pathlib import Path

HERE = Path(__file__).resolve().parent
SEED = 0x50_0D_B1_75_01_03_14_00  # fixed; changing it changes every random case
SCHEMA = "tessera.soundness/1"

# RFC-0002: Goldilocks. The challenge field is Fp2, of q = p^2 elements (RFC-0014).
P = 2**64 - 2**32 + 1
Q = P * P

# RFC-0014 'Soundness calculator': LOG2's working precision and output fractions.
W = 62
FIELD_FRACTION = 8
QUERY_FRACTION = 48
UNIT = 256  # FixedBits: units of 2^-8 bits (RFC-0036)
MAX_QUERIES = 2**16
MAX_CODEWORD = 2**32
U128 = 2**128 - 1

getcontext().prec = 60
LN2 = Decimal(2).ln()
MATRICES = ["Embed", "Q", "K", "V", "O", "Gate", "Up", "Down", "LmHead", "ChannelParams"]


class SplitMix64:
    """Steele, Lea and Flood (2014)."""

    def __init__(self, seed):
        self.state = seed

    def next(self):
        self.state = (self.state + 0x9E3779B97F4A7C15) % 2**64
        z = self.state
        z = ((z ^ (z >> 30)) * 0xBF58476D1CE4E5B9) % 2**64
        z = ((z ^ (z >> 27)) * 0x94D049BB133111EB) % 2**64
        return z ^ (z >> 31)

    def below(self, n):
        """Uniform in [0, n), by rejection: the top bits of as many words as n needs."""
        if n <= 1:
            return 0
        bits = (n - 1).bit_length()
        words = -(-bits // 64)
        while True:
            x = 0
            for _ in range(words):
                x = (x << 64) | self.next()
            x >>= 64 * words - bits
            if x < n:
                return x

    def bits(self, most):
        """A value of a random bit length up to `most`, so that every scale is reached."""
        width = self.below(most + 1)
        return self.below(2**width) if width else 0


def require(condition, what):
    if not condition:
        raise SystemExit(f"soundness oracle: {what}")


# ---------------------------------------------------------------------------------------------
# RFC-0014 'Parameter sets' and RFC-0026 'Schema'.

def param_set(name, mode, release, advice, weights, queries, pow_bits, targets):
    return {
        "id": name,
        "mode": mode,
        "release": release,
        "advice": dict(zip(["log2_k", "log2_inv_rate", "mask_coeffs", "row_block", "segments"],
                           advice)),
        "weights": dict(zip(["log2_k", "log2_inv_rate", "mask_coeffs", "blinding_rows"], weights)),
        "advice_queries": queries[0],
        "weight_queries": queries[1],
        "seed_pow": pow_bits[0],
        "query_pow": pow_bits[1],
        "targets": None if targets is None else {"bits": targets[0], "one_shot": targets[1]},
    }


# SP-1: 100 bits for every row a prover can retry; 64 bits for live-64's one-shot rows, the
# LogUp identity (a coin-tossed seed) and the column queries (the verifier's draw). In
# non-interactive mode every row is retried, at the proofs of work's cost.
LIVE_64 = param_set("live-64/1", "live", True, (22, 1, 160, 32, 8), (20, 2, 2**16, 4096),
                    (155, 97), (0, 0), (100, 64))
NI_100 = param_set("ni-100/1", "non-interactive", True, (22, 1, 200, 32, 8), (20, 2, 2**16, 4096),
                   (193, 122), (20, 20), (100, 100))
TEST_SMALL = param_set("test-small/1", "either", False, (10, 1, 16, 4, 1), (8, 2, 64, 32),
                       (8, 8), (0, 0), None)
PARAM_SETS = [LIVE_64, NI_100, TEST_SMALL]


# ---------------------------------------------------------------------------------------------
# The 405B 16K-token, 16K-context statement (RFC-0014 'Soundness calculator', its table of
# sources).

TOKENS = 16_384
LAYERS, D_MODEL, HEADS, KV_HEADS, D_HEAD, D_FF, VOCAB = 126, 16_384, 128, 8, 128, 53_248, 128_256
SPLIT_VALUES = 53_489  # the lowering's ungrouped split values (RFC-0045), as RFC-0014 states them
MAX_GROUP_BITS = 14  # RFC-0054 'Messages': g_T + g_L ≤ 14
LEVEL_DEGREE = 3  # RFC-0010 'Tree levels': eq(r, x) times two evaluations
MERGE_DEGREE = 2  # RFC-0054 'The merge'
NO_GROUPS = {"count": 1, "level_rounds": [], "merge_rounds": 0, "merge_claims": 0}


def pow2_at_least(x):
    return 1 << (x - 1).bit_length()


def log2_exact(x):
    require(x > 0 and x & (x - 1) == 0, f"{x} is a power of two")
    return x.bit_length() - 1


def weight_matrices_405b(log2_kw):
    """RFC-0040 'The matrices' for Llama-3.1-405B: each matrix's message width k_T, its committed
    data rows m_T and its variables ν_T = log2 L_pad + log2 out_pad + log2 in_pad."""
    kw = 1 << log2_kw
    l_pad = pow2_at_least(LAYERS)
    matrices = []
    shapes = [
        ("Embed", D_MODEL, VOCAB, 1),  # d_model × vocab, transposed
        ("Q", HEADS * D_HEAD, D_MODEL, LAYERS),
        ("K", KV_HEADS * D_HEAD, D_MODEL, LAYERS),
        ("V", KV_HEADS * D_HEAD, D_MODEL, LAYERS),
        ("O", D_MODEL, HEADS * D_HEAD, LAYERS),
        ("Gate", D_FF, D_MODEL, LAYERS),
        ("Up", D_FF, D_MODEL, LAYERS),
        ("Down", D_MODEL, D_FF, LAYERS),
        ("LmHead", VOCAB, D_MODEL, 1),
    ]
    for name, out, inp, layers in shapes:
        out_pad, in_pad = pow2_at_least(out), pow2_at_least(inp)
        width = max(in_pad, min(kw, out_pad * in_pad))
        a = log2_exact(width // in_pad)
        # A row of o_hi = r holds outputs r·2^a .. r·2^a + 2^a − 1: committed when r·2^a < out.
        rows = layers * -(-out // 2**a)
        layer_bits = log2_exact(l_pad) if layers > 1 else 0
        variables = layer_bits + log2_exact(out_pad) + log2_exact(in_pad)
        matrices.append({"matrix": name, "width": width, "rows": rows, "variables": variables})
    # ChannelParams: the tensor [16, L_pad, C], C = 2^16 at 405B (RFC-0012 'Layout'), at most 882
    # committed rows (RFC-0012 'Column answers').
    c = 2**16
    width = min(kw, 16 * l_pad * c)
    variables = 4 + log2_exact(l_pad) + log2_exact(c)
    matrices.append({"matrix": "ChannelParams", "width": width, "rows": 882,
                     "variables": variables})
    return matrices


# The tables of the compiler's 405B report, docs/perf/v0.2/commitment-report-405b-servable-c2-b32.json
# (servable, two shift classes, B_max = 32), including RFC-0050, in id order: (kind, lookups per token, rows). The
# oracle runs on spec/vectors/ alone, so it states them; crates/tessera-protocol/tests/soundness.rs
# checks the calculator's default against the report itself.
REPORT_TABLES = [
    ('Rope', 0, 8_388_608),
    ('Exp', 0, 4_194_304),
    ('Silu', 6_709_248, 1_073_741_824),
    ('Range', 253, 4_294_967_296),
    ('Range', 65_271, 65_536),
    ('Range', 759, 128),
    ('Range', 23_782, 256),
    ('Range', 506, 16_777_216),
    ('Range', 1_012, 33_554_432),
    ('Range', 4_130_286, 67_108_864),
    ('Range', 2_530, 134_217_728),
    ('Range', 16_634, 131_072),
    ('Range', 506, 32_768),
    ('Range', 15_686, 2),
    ('Range', 2_209_792, 2_097_152),
    ('Requant', 16_384, 1_048_576),
    ('Requant', 2_193_408, 16_777_216),
    ('Requant', 6_838_272, 1_073_741_824),
    ('Range', 32_256, 4),
    ('Range', 2_064_384, 1_048_576),
    ('Requant', 2_064_384, 1_048_576),
    ('Requant', 2_064_384, 1_073_741_824),
    ('Requant', 4_128_768, 8_388_608),
    ('Requant', 4_128_768, 67_108_864),
    ('Requant', 6_709_248, 65_536),
    ('Requant', 2_064_384, 4_294_967_296),
]


def lookups_405b():
    """Each LogUp instance of the compiler's 405B report at 16K tokens: N = lookups per token ×
    16,384, T = its rows. The Exp instance looks up every position of the padded score tensor,
    2^7 layers × 2^7 heads × 2^14 queries × 2^14 keys = 2^42 (RFC-0030 'Cost')."""
    instances = []
    for kind, per_token, rows in REPORT_TABLES:
        if kind == "Exp":
            require(per_token == 0, "Exp is looked up by scores only")
            lookups = pow2_at_least(LAYERS) * pow2_at_least(HEADS) * TOKENS * TOKENS
        else:
            lookups = per_token * TOKENS
            # Shared Range families also serve roots/requants. Add only the
            # Softmax part's two padded layers, without padding those other reads.
            per_row = {2**17: 1, 2**16: 4, 2**2: 2}.get(rows, 0) if kind == "Range" else 0
            require(per_token >= LAYERS * HEADS * per_row, "Softmax share of a Range family")
            lookups += (pow2_at_least(LAYERS) - LAYERS) * HEADS * TOKENS * per_row
        instances.append({"lookups": lookups, "rows": rows})
    return instances


def statement_405b():
    lookups = lookups_405b()
    forests = [log2_exact(pow2_at_least(LAYERS) * HEADS * TOKENS)]  # SoftmaxRow: s = 28
    tallest = max(max(i["lookups"], i["rows"]) for i in lookups)
    levels = (tallest - 1).bit_length()  # the tallest tree's height: 42
    # RFC-0054 'Soundness': the most groups the verifier accepts, 2^{min(14, t + n_ℓ)}.
    t, n_layers = ceil_log2(TOKENS), ceil_log2(LAYERS)
    groups = 2**min(MAX_GROUP_BITS, t + n_layers)
    # RFC-0010 'Tree levels': level ℓ of the tallest tree, the Exp lookup tree, has ℓ rounds. The
    # SoftmaxRow forest's 2^s trees, of height t, join at level 42 − t and have as many: their
    # level ℓ binds ℓ − (42 − t) tree variables and s stacking variables, s = 42 − t.
    row_height = t
    require(forests == [levels - row_height], "the forest joins the tallest tree's levels")
    level_rounds = sum(range(levels))  # 861
    # The merged polynomials: one per split value and one per forest (RFC-0054 'The merge'); the
    # most variables, the forest leaf table's s + t, above any split tensor's.
    merge_rounds = max(s + row_height for s in forests)
    return {
        "protocol": 1,
        "rounds": [{"count": 10_000, "degree": 7}],
        "stages": [64, 64],
        # Each level batches every instance's two trees and every forest at most.
        "tree_levels": [2 * len(lookups) + len(forests)] * levels,
        "zero_checks": [42] * 8,
        "forests": forests,
        "lookups": lookups,
        "claim_groups": [64] * 16,
        # RFC-0011/0047/0050, servable c2/Q512: both independent layouts
        # derive 2,328,576 allocated rows, including gaps. The calculator
        # input is the padded row-index domain, hence 2^22 rows (theta=22).
        "advice_rows": pow2_at_least(2_328_576),
        "matrices": weight_matrices_405b(20),
        "groups": {
            "count": groups,
            "level_rounds": [{"count": level_rounds, "degree": LEVEL_DEGREE}],
            "merge_rounds": merge_rounds,
            "merge_claims": SPLIT_VALUES + len(forests),
        },
    }


# ---------------------------------------------------------------------------------------------
# RFC-0014 'Soundness calculator': LOG2, the rows, the targets, the decisions.

def log2_lower(a, b, f):
    """LOG2(a, b, F): an integer L ≤ 2^F·log2(a/b), for integers a ≥ b ≥ 1, every rounding down."""
    require(a >= b >= 1, "LOG2 takes a ≥ b ≥ 1")
    e = a.bit_length() - b.bit_length()
    if b << e > a:
        e -= 1
    y = (a << W) // (b << e)
    require(1 << W <= y < 1 << (W + 1), "LOG2 normalizes into [2^W, 2^(W+1))")
    result = e
    for _ in range(f):
        y = (y * y) >> W
        result *= 2
        if y >= 1 << (W + 1):
            y >>= 1
            result += 1
    return result


def log2_rounded_up(a, b, f, step):
    """LOG2 with one kind of step rounded up instead of down: `normalize`, `square` or `halve`.
    The oracle never reports it; it picks the inputs at which rounding down is what keeps LOG2
    below the bound."""
    e = a.bit_length() - b.bit_length()
    if b << e > a:
        e -= 1
    y = -(-(a << W) // (b << e)) if step == "normalize" else (a << W) // (b << e)
    result = e
    for _ in range(f):
        y = -(-(y * y) >> W) if step == "square" else (y * y) >> W
        result *= 2
        if y >= 1 << (W + 1):
            y = (y + 1) >> 1 if step == "halve" else y >> 1
            result += 1
    return result


def boundary(b, f, k):
    """The smallest a with a^(2^f) ≥ 2^k·b^(2^f): log2(a/b) reaches k/2^f at a, and a − 1 lies just
    below it, by about 2^-62 when b ≈ 2^62."""
    goal = (b ** (2**f)) << k
    lo, hi = b, b << (k // 2**f + 2)
    while lo < hi:
        mid = (lo + hi) // 2
        if mid ** (2**f) >= goal:
            hi = mid
        else:
            lo = mid + 1
    return lo


def ceil_log2(x):
    require(x >= 1, "a row count is at least 1")
    return (x - 1).bit_length()


def field_bits(c, g):
    """The FixedBits of ε = c/q under g bits of grinding: None for no term (c = 0)."""
    if c == 0:
        return None
    if c >= Q:
        return UNIT * g
    return log2_lower(Q, c, FIELD_FRACTION) + UNIT * g


def field_pass(c, g, t):
    """Exact: g + max(0, log2(q/c)) ≥ t ⟺ t ≤ g or c·2^(t−g) ≤ q."""
    if c == 0 or t <= g:
        return True
    return c * 2**(t - g) <= Q


def query_terms(n, k, mask):
    """(A, B) of ε = (B/A)^τ: A = 2n, B = n + k + P."""
    return 2 * n, n + k + mask


def query_bits(n, k, mask, tau, g):
    """(per-query FixedBits, the row's FixedBits) of ((n + k + P)/(2n))^τ under g bits."""
    a, b = query_terms(n, k, mask)
    if b >= a:
        return 0, UNIT * g
    lb = log2_lower(a, b, QUERY_FRACTION)
    shift = QUERY_FRACTION - FIELD_FRACTION
    return lb >> shift, ((tau * lb) >> shift) + UNIT * g


def query_pass(n, k, mask, tau, g, t):
    """Exact: g + max(0, τ·log2(A/B)) ≥ t ⟺ t ≤ g or A^τ ≥ 2^(t−g)·B^τ."""
    if t <= g:
        return True
    a, b = query_terms(n, k, mask)
    return a**tau >= 2**(t - g) * b**tau


def true_field(c, g):
    if c >= Q:
        return Decimal(g)
    return Decimal(g) + (Decimal(Q) / Decimal(c)).ln() / LN2


def true_per_query(n, k, mask):
    a, b = query_terms(n, k, mask)
    if b >= a:
        return Decimal(0)
    return (Decimal(a) / Decimal(b)).ln() / LN2


def check_bound(r, true, slack, what):
    """r ≤ 256·true < r + 1 + slack, with the true value at 60 digits."""
    scaled = true * UNIT
    require(Decimal(r) <= scaled, f"{what}: {r} is above 256·λ = {scaled}")
    require(scaled < Decimal(r) + 1 + Decimal(slack), f"{what}: {r} is too far below {scaled}")
    nearest = scaled.to_integral_value()
    require(abs(scaled - nearest) > Decimal(10)**-30 or scaled == nearest,
            f"{what}: 256·λ lies within 10^-30 of an integer, past what 60 digits decide")


def e30(true):
    """⌊256·λ·10^30⌋, a decimal string: the true value a test checks FixedBits against."""
    scaled = true * UNIT * Decimal(10)**30
    return str(int(scaled.to_integral_value(rounding=ROUND_FLOOR)))


def check_decision(passed, true, t, what):
    """The exact decision agrees with the true value, wherever the decimals decide it."""
    if abs(true - t) > Decimal(10)**-30:
        require(passed == (true >= t), f"{what}: the exact decision disagrees with λ = {true}")


def validate(params, statement):
    adv, wts = params["advice"], params["weights"]
    require(statement["protocol"] in (1, 2), "protocol 1 or 2")
    require(2**(adv["log2_k"] + adv["log2_inv_rate"]) <= MAX_CODEWORD, "advice codeword ≤ 2^32")
    for tau in (params["advice_queries"], params["weight_queries"]):
        require(1 <= tau <= MAX_QUERIES, "τ in 1..=2^16")
    require(statement["advice_rows"] >= 1, "advice rows ≥ 1")
    ids = [MATRICES.index(m["matrix"]) for m in statement["matrices"]]
    require(ids == sorted(set(ids)), "matrices strictly increasing")
    for m in statement["matrices"]:
        require(m["width"] >= 1 and m["width"] * 2**wts["log2_inv_rate"] <= MAX_CODEWORD,
                "matrix codeword in 1..=2^32")
        # tessera-protocol/1 bounds a matrix with no committed row: it has no proximity rows.
        if statement["protocol"] == 2:
            require(m["rows"] + wts["blinding_rows"] >= 1, "a matrix holds a row")
    groups = statement["groups"]
    g = groups["count"]
    require(g >= 1 and g & (g - 1) == 0 and g <= 2**MAX_GROUP_BITS, "G a power of two in 1..=2^14")
    require(g == 1 or statement["protocol"] == 1, "tessera-protocol/2 accepts no grouping")
    if g == 1:
        require(not groups["level_rounds"] and groups["merge_rounds"] == 0
                and groups["merge_claims"] == 0, "no grouping has no grouped sizes")


def report(params, statement, name, why):
    """The `tessera.soundness/1` report of `params` at `statement`."""
    validate(params, statement)
    adv, wts = params["advice"], params["weights"]
    targets = params["targets"]
    v2 = statement["protocol"] == 2
    rows = []
    truth = []  # per row: ⌊256·λ·10^30⌋, and for the queries ⌊256·b(ρ')·10^30⌋

    def target(covered):
        """The rows a proof of work covers take the set's one-shot target: the LogUp identity
        and the queries, which a live session draws once."""
        if targets is None:
            return None
        return targets["one_shot"] if covered else targets["bits"]

    def field_row(row, numerator, g, covered, extra=None):
        c = min(numerator, U128)  # numerators saturate at 2^128 − 1
        t = target(covered)
        bits = field_bits(c, g)
        entry = {"row": row}
        entry.update(extra or {})
        entry["numerator"] = str(c)
        entry.update(finish(bits, g, covered, t, None if t is None else field_pass(c, g, t)))
        if bits is not None:
            true = true_field(c, g)
            check_bound(bits, true, 2**-20, f"{name}: {row}")
            if t is not None:
                check_decision(entry["pass"], true, t, f"{name}: {row}")
            truth.append({"bits": e30(true)})
        else:
            truth.append({"bits": None})
        rows.append(entry)

    def finish(bits, g, covered, t, passed):
        return {
            "bits": bits,
            "grinding": g,
            "covered_by": covered,
            "target": t,
            "margin": None if bits is None or t is None else bits - UNIT * t,
            "pass": None if t is None else passed,
        }

    # 1. Sumcheck and tree rounds: the largest error of one challenge the stages and tree levels
    # squeeze (rounds, batching, zero-check points, lines, and the masks of tessera-protocol/2).
    rounds = [r["degree"] for r in statement["rounds"] if r["count"] > 0]
    challenges = rounds + statement["stages"] + statement["tree_levels"] + statement["zero_checks"]
    if statement["tree_levels"]:
        challenges.append(1)
    if v2 and (statement["stages"] or statement["tree_levels"]):
        challenges.append(1)
    # RFC-0054 'Soundness': the groups' and the table batch's levels are levels of the same
    # degrees; the merge has degree 2 and batches with error (P + G)/q.
    groups = statement["groups"]
    grouped = groups["count"] > 1
    if grouped:
        challenges += [r["degree"] for r in groups["level_rounds"] if r["count"] > 0]
        if groups["merge_rounds"] > 0:
            challenges.append(MERGE_DEGREE)
        challenges.append(groups["merge_claims"] + groups["count"])
    field_row("rounds", max(challenges, default=0), 0, None)

    # 2. All rounds, union bound: RFC-0010's (n·d + m) per stage, (n·d + m + 1) per tree level.
    union = (sum(r["count"] * r["degree"] for r in statement["rounds"])
             + sum(statement["stages"])
             + sum(m + 1 for m in statement["tree_levels"])
             + sum(statement["zero_checks"])
             + (len(statement["stages"]) + len(statement["tree_levels"]) if v2 else 0))
    if grouped:
        # G more copies of the leaf stage's levels, each group's and the table batch's being no
        # more than the stage's, and the merge: 2·n_max + P + G.
        levels = (sum(r["count"] * r["degree"] for r in groups["level_rounds"])
                  + sum(m + 1 for m in statement["tree_levels"]))
        union += (groups["count"] * levels + MERGE_DEGREE * groups["merge_rounds"]
                  + groups["merge_claims"] + groups["count"])
    field_row("union", union, 0, None)

    # 3. Product-forest root claims: s/q for a forest of 2^s trees.
    field_row("forests", max(statement["forests"], default=0), 0, None)

    # 4. The LogUp identity: (N + T)/q per instance, 2(N + T)/q in tessera-protocol/2.
    worst = max((i["lookups"] + i["rows"] for i in statement["lookups"]), default=0)
    field_row("logup", (2 if v2 else 1) * worst, params["seed_pow"], "seed-pow")

    # 5. Opening evaluation checks: T_g/q per claim group, 1/q per weight matrix.
    openings = statement["claim_groups"] + ([1] if statement["matrices"] else [])
    field_row("openings", max(openings, default=0), 0, None)

    # 6, 7. Proximity, per commitment: the queries, then the field term ϑ·n/q.
    commitments = [("advice", 2**adv["log2_k"], adv["log2_inv_rate"], adv["mask_coeffs"],
                    params["advice_queries"], ceil_log2(statement["advice_rows"]))]
    for m in statement["matrices"]:
        # RFC-0049 'The calculator': tessera-protocol/1 commits no blinding row and no mask for
        # the weights, so a matrix counts ϑ_T = ⌈log2 m_T⌉ at the unmasked rate, and one with no
        # committed row has no queries row and no field term. tessera-protocol/2 counts B_W and
        # P_W.
        count = m["rows"] + (wts["blinding_rows"] if v2 else 0)
        if count == 0:
            continue
        commitments.append((m["matrix"], m["width"], wts["log2_inv_rate"],
                            wts["mask_coeffs"] if v2 else 0, params["weight_queries"],
                            ceil_log2(count)))
    for commitment, k, inv_rate, mask, tau, theta in commitments:
        n = k << inv_rate
        g = params["query_pow"]
        t = target("query-pow")
        per_query, bits = query_bits(n, k, mask, tau, g)
        entry = {"row": "queries", "commitment": commitment, "codeword": n, "message": k,
                 "mask": mask, "queries": tau, "per_query_bits": per_query}
        passed = None if t is None else query_pass(n, k, mask, tau, g, t)
        entry.update(finish(bits, g, "query-pow", t, passed))
        b = true_per_query(n, k, mask)
        check_bound(per_query, b, 2**-37, f"{name}: per-query bits of {commitment}")
        true = Decimal(g) + tau * b
        check_bound(bits, true, 2**-20, f"{name}: queries of {commitment}")
        if t is not None:
            check_decision(passed, true, t, f"{name}: queries of {commitment}")
        truth.append({"bits": e30(true), "per_query": e30(b)})
        rows.append(entry)
        field_row("field", theta * n, 0, None, {"commitment": commitment, "theta": theta})

    # 8. Weight linking: ν/q for the largest matrix.
    field_row("linking", max((m["variables"] for m in statement["matrices"]), default=0), 0, None)

    # 9. Hash collisions: BLAKE3's 256-bit output, 128 bits.
    t = target(None)
    hash_row = {"row": "hash"}
    hash_row.update(finish(128 * UNIT, 0, None, t, None if t is None else 128 >= t))
    rows.append(hash_row)
    truth.append({"bits": e30(Decimal(128))})

    # The minimum: the least margin over its target, or the fewest bits without targets; the
    # first such row.
    def key(row):
        return row["margin"] if targets is not None else row["bits"]

    ranked = [(key(row), i) for i, row in enumerate(rows) if row["bits"] is not None]
    minimum = min(ranked)[1]
    meets = None if targets is None else all(row["pass"] for row in rows)
    return {
        "name": name,
        "why": why,
        "true": truth,
        "report": {
            "schema": SCHEMA,
            "params": params,
            "statement": statement,
            "rows": rows,
            "minimum": minimum,
            "meets_targets": meets,
        },
    }


# ---------------------------------------------------------------------------------------------
# The cases.

def with_changes(base, **changes):
    statement = json.loads(json.dumps(base))
    statement.update(changes)
    return statement


def matrix_changed(base, name, **changes):
    statement = json.loads(json.dumps(base))
    for m in statement["matrices"]:
        if m["matrix"] == name:
            m.update(changes)
    return statement


def params_changed(base, suffix, **changes):
    params = json.loads(json.dumps(base))
    params.update(changes)
    params["id"] = base["id"] + suffix
    return params


def adversarial(base):
    """One statement or parameter per row that misses exactly that row's target, at live-64 and
    ni-100. The rounds row cannot fail alone: the union bound counts every term it counts."""
    cases = []
    for params in (LIVE_64, NI_100):
        tau_a = params["advice_queries"]
        cases += [
            ("rounds", params, with_changes(base, rounds=base["rounds"] + [{"count": 1, "degree": 2**29}]),
             "a round of degree 2^29: 99 − ε bits, which the union bound counts too"),
            ("union", params, with_changes(base, rounds=base["rounds"] + [{"count": 2**28, "degree": 1}]),
             "2^28 more rounds of degree 1: the union bound falls below 100 bits"),
            ("forests", params, with_changes(base, forests=base["forests"] + [2**29]),
             "a forest of 2^(2^29) trees: s/q of 99 − ε bits"),
            ("openings", params, with_changes(base, claim_groups=base["claim_groups"] + [2**29]),
             "a claim group of 2^29 claims"),
            ("queries/advice", params_changed(params, " (τ_A − 1)", advice_queries=tau_a - 1), base,
             "one advice query fewer than the set's τ_A"),
            ("queries/Down", params,
             with_changes(matrix_changed(base, "Down", width=2**19), protocol=2,
                          groups=NO_GROUPS),
             "Down at half the message width under tessera-protocol/2, whose P_W mask "
             "coefficients count: ρ' = 0.281, 97 or 122 queries fall short"),
            ("field/advice", params, with_changes(base, advice_rows=2**33),
             "a 33-bit advice row index: ϑ·n/q = 33·2^23/q"),
            ("field/Down", params, matrix_changed(base, "Down", width=2**24),
             "Down at 2^24 columns: 17·2^26/q, ϑ = ⌈log2 m_T⌉ in tessera-protocol/1, while its "
             "queries still pass"),
            ("linking", params, matrix_changed(base, "Down", variables=2**29),
             "a matrix of 2^29 variables"),
        ]
        if params is NI_100:
            cases.append(("logup", params, with_changes(base, lookups=base["lookups"] + [
                {"lookups": 2**50 - 2**22, "rows": 2**22}]),
                "an instance of N + T = 2^50 (RFC-0014 'Testing Strategy'): 78 + 20 bits"))
        else:
            cases.append(("logup", params, with_changes(base, lookups=base["lookups"] + [
                {"lookups": 2**63, "rows": 2**63}]),
                "an instance of N + T = 2^64: 64 − ε one-shot bits"))
    return cases


def edges(base):
    """The exact comparisons at their boundaries, where rounded bits would decide wrongly."""
    below = Q // 2**100  # 2^28 − 1: c·2^100 ≤ q holds, and fails for c + 1
    empty = {"protocol": 1, "rounds": [], "stages": [], "tree_levels": [], "zero_checks": [],
             "forests": [], "lookups": [], "claim_groups": [], "advice_rows": 1, "matrices": [],
             "groups": NO_GROUPS}
    cases = [
        ("edge/numerator-pass", NI_100, with_changes(base, claim_groups=[below]),
         "T_g = ⌊q/2^100⌋: 100 + 4.7·10^-9 bits, which passes"),
        ("edge/numerator-fail", NI_100, with_changes(base, claim_groups=[below + 1]),
         "T_g = 2^28: 100 − 6.7·10^-10 bits, since q < 2^128"),
        ("edge/logup-2^50-live", LIVE_64, with_changes(base, lookups=base["lookups"] + [
            {"lookups": 2**50 - 2**22, "rows": 2**22}]),
         "N + T = 2^50 at live-64: 78 − ε one-shot bits pass its 64-bit target"),
        ("edge/protocol-2", NI_100, with_changes(base, protocol=2, groups=NO_GROUPS,
                                                 rounds=base["rounds"] + [{"count": 2, "degree": 4}],
                                                 stages=base["stages"] + [len(base["lookups"])]),
         "tessera-protocol/2: 2(N + T)/q, a mask per batched sumcheck, the identity sumcheck"),
        ("edge/empty", NI_100, {"protocol": 1, "rounds": [], "stages": [], "tree_levels": [],
                                "zero_checks": [], "forests": [], "lookups": [], "claim_groups": [],
                                "advice_rows": 1, "matrices": [], "groups": NO_GROUPS},
         "nothing but the advice and the hash: every other row has no term, and ϑ = 0"),
        ("edge/rate-one", params_changed(LIVE_64, " (rate 1)", weights={**LIVE_64["weights"], "log2_inv_rate": 0}),
         base, "weights at rate 1: ρ' ≥ 1, so their queries give no bit"),
        ("edge/line-only", NI_100, with_changes(empty, tree_levels=[0]),
         "a tree level that batches nothing: its line challenge alone, 1/q"),
        ("edge/mask-only", NI_100, with_changes(empty, protocol=2, stages=[0]),
         "a stage that batches nothing in tessera-protocol/2: its mask's ρ alone, 1/q"),
        ("edge/zero-count", NI_100, with_changes(empty, rounds=[{"count": 0, "degree": 2**30},
                                                              {"count": 1, "degree": 3}]),
         "no round of degree 2^30 counts: the rounds row is 3/q"),
        ("edge/hash-128", params_changed(NI_100, " (128-bit targets)",
                                         targets={"bits": 128, "one_shot": 128}), base,
         "targets of 128 bits: only the hash reaches one"),
        ("edge/q-below-2^128", LIVE_64, with_changes(base, lookups=[
            {"lookups": 2**63, "rows": 2**63 - 1}]),
         "N + T = 2^64 − 1: 64 − 6.7·10^-10 one-shot bits, which fail 64 since q < 2^128 − 1"),
        ("edge/empty-channel-params", LIVE_64, matrix_changed(base, "ChannelParams", rows=0),
         "no weight has channel parameters: in tessera-protocol/1 ChannelParams commits no row, "
         "so it has no queries row and no field term, and is not refused (RFC-0049)"),
        ("edge/groups-none", NI_100, with_changes(base, groups=NO_GROUPS),
         "the 405B statement were no grouping accepted: the ungrouped union row and rounds row "
         "(RFC-0054 'Soundness')"),
        ("edge/groups-union-miss", NI_100,
         with_changes(base, groups={**base["groups"], "level_rounds": [{"count": 6_000, "degree": 3}]}),
         "6,000 level rounds per group at G = 2^14: the union row misses 100 bits at the largest "
         "grouping, while the statement without groups passes (RFC-0054 'Testing Strategy')"),
        ("edge/groups-merge-only", NI_100, with_changes(empty, groups={
            "count": 2, "level_rounds": [], "merge_rounds": 1, "merge_claims": 1}),
         "two groups, no level, a one-round merge of one claim: the rounds row is P + G = 3, the "
         "union 2·1 + 1 + 2 = 5"),
    ]
    return cases


def random_params(rng, i):
    log2_k = rng.below(25)
    inv_rate = rng.below(min(5, 33 - log2_k))
    w_log2_k = rng.below(23)
    w_inv_rate = rng.below(5)
    targets = None if rng.below(5) == 0 else (60 + rng.below(70), 40 + rng.below(70))
    return param_set(f"random-{i}", ["live", "non-interactive", "either"][rng.below(3)],
                     rng.below(2) == 1,
                     (log2_k, inv_rate, rng.bits(18), 1 + rng.below(64), 1 + rng.below(8)),
                     (w_log2_k, w_inv_rate, rng.bits(18), rng.bits(13)),
                     (1 + rng.below(300), 1 + rng.below(300)), (rng.below(33), rng.below(33)),
                     targets)


def random_statement(rng, params):
    w_inv = params["weights"]["log2_inv_rate"]
    blinding = params["weights"]["blinding_rows"]
    ids = sorted(set(rng.below(10) for _ in range(rng.below(5))))
    matrices = []
    for j in ids:
        # Widths are powers of two at every set of RFC-0014; random widths test the general rule.
        if rng.below(2):
            width = 2**rng.below(33 - w_inv)
        else:
            width = max(1, rng.bits(32 - w_inv))
        rows = rng.bits(63)
        if rows + blinding == 0:
            rows = 1
        matrices.append({"matrix": MATRICES[j], "width": width, "rows": rows,
                         "variables": rng.bits(32)})
    statement = {
        "protocol": 1 + rng.below(2),
        "rounds": [{"count": rng.bits(40), "degree": rng.bits(12)} for _ in range(rng.below(4))],
        "stages": [rng.bits(20) for _ in range(rng.below(4))],
        "tree_levels": [rng.bits(16) for _ in range(rng.below(6))],
        "zero_checks": [rng.bits(7) for _ in range(rng.below(4))],
        "forests": [rng.bits(8) for _ in range(rng.below(3))],
        "lookups": [{"lookups": rng.bits(64), "rows": rng.bits(64)} for _ in range(rng.below(4))],
        "claim_groups": [rng.bits(32) for _ in range(rng.below(4))],
        "advice_rows": max(1, rng.bits(64)),
        "matrices": matrices,
        "groups": random_groups(rng),
    }
    if statement["protocol"] == 2:  # RFC-0054 'Messages': no grouping in tessera-protocol/2
        statement["groups"] = dict(NO_GROUPS)
    return statement


def random_groups(rng):
    """RFC-0054's grouped sizes, in tessera-protocol/1 only: G = 2^0..2^14."""
    count = 2**rng.below(MAX_GROUP_BITS + 1)
    if count == 1:
        return dict(NO_GROUPS)
    return {
        "count": count,
        "level_rounds": [{"count": rng.bits(20), "degree": rng.bits(4)} for _ in range(rng.below(3))],
        "merge_rounds": rng.bits(7),
        "merge_claims": rng.bits(24),
    }


def log2_cases(rng):
    """LOG2 at its edges and at random inputs: both calculators must give `result`."""
    cases = [(1, 1, 8), (2, 1, 8), (3, 2, 8), (Q, 1, 8), (Q, Q - 1, 8), (Q, 7, 8), (Q, 2**28, 8),
             (Q, 2**28 - 1, 8), (2**33, 2**32 + 1, 48), (2**24, 2**23 + 2**22 + 160, 48),
             (2**22, 2**21 + 2**20 + 2**16, 48), (2**64, 3, 0), (U128, 1, 8), (U128, U128, 8)]
    for _ in range(24):
        a = max(1, rng.bits(128))
        b = 1 + rng.below(a)
        cases.append((a, b, rng.below(56)))
    # q against 2^128 − 1: c = 2^64 − 1 gives 64 − 6.7·10^-10 bits under q, and 64 + 2^-64 under
    # 2^128 − 1.
    cases += [(Q, 2**64 - 1, 8), (Q, 2**64 - 2**33 + 3, 8), (Q, 2**64 - 2**33 + 2, 8)]
    # Inputs just below and at the boundary k/2^f, b near 2^62: for each step, up to four pairs at
    # which rounding that step up instead of down changes the result.
    wanted = {"normalize": 4, "square": 4, "halve": 4}
    tries = 0
    while any(wanted.values()):
        tries += 1
        require(tries < 20_000, "boundary cases for every step")
        f = 1 + rng.below(8)
        b = 2**61 + rng.below(2**62)
        k = 1 + rng.below(2 ** (f + 1) - 1)
        a = boundary(b, f, k)
        require(log2_lower(a - 1, b, f) <= k - 1, "LOG2 stays below the boundary")
        hit = [step for step, left in wanted.items() if left and any(
            log2_rounded_up(x, b, f, step) != log2_lower(x, b, f) for x in (a - 1, a))]
        if hit:
            wanted[hit[0]] -= 1
            cases += [(a - 1, b, f), (a, b, f)]
    return [{"a": str(a), "b": str(b), "fraction": f, "result": str(log2_lower(a, b, f))}
            for a, b, f in cases]


def compact(value):
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"))


def main():
    base = statement_405b()
    cases = []
    for params in PARAM_SETS:
        cases.append(report(params, base, f"405b/{params['id']}",
                            "the 405B 16K-token, 16K-context statement"))
    for params in PARAM_SETS:
        cases.append(report(params, with_changes(base, protocol=2, groups=NO_GROUPS),
                            f"405b-protocol-2/{params['id']}",
                            "the 405B statement under tessera-protocol/2: the weights count "
                            "their P_W mask coefficients and B_W blinding rows"))
    for row, params, statement, why in adversarial(base):
        cases.append(report(params, statement, f"adversarial/{params['id']}/{row}", why))
    for name, params, statement, why in edges(base):
        cases.append(report(params, statement, name, why))
    rng = SplitMix64(SEED)
    for i in range(48):
        params = random_params(rng, i) if rng.below(4) else PARAM_SETS[rng.below(3)]
        statement = random_statement(rng, params)
        cases.append(report(params, statement, f"random/{i}", "random sizes"))

    # What the reports must show, checked before anything is written.
    by_name = {c["name"]: c["report"] for c in cases}
    for params in (LIVE_64, NI_100):
        for prefix in ("405b", "405b-protocol-2"):
            require(by_name[f"{prefix}/{params['id']}"]["meets_targets"],
                    f"{params['id']} meets its targets at {prefix}")
    require(by_name["405b/test-small/1"]["meets_targets"] is None, "test-small is not held to targets")
    # RFC-0049: in tessera-protocol/1 every weight matrix counts its committed rows at the
    # unmasked rate; in tessera-protocol/2 its blinding rows and mask coefficients.
    for prefix, v2 in (("405b", False), ("405b-protocol-2", True)):
        for params in PARAM_SETS:
            r = by_name[f"{prefix}/{params['id']}"]
            wts = params["weights"]
            for m in base["matrices"]:
                rows = [row for row in r["rows"] if row.get("commitment") == m["matrix"]]
                require([row["row"] for row in rows] == ["queries", "field"], f"{prefix}: rows")
                require(rows[0]["mask"] == (wts["mask_coeffs"] if v2 else 0), f"{prefix}: mask")
                count = m["rows"] + (wts["blinding_rows"] if v2 else 0)
                require(rows[1]["theta"] == ceil_log2(count), f"{prefix}: ϑ of {m['matrix']}")
    empty_rows = by_name["edge/empty-channel-params"]["rows"]
    require(all(row.get("commitment") != "ChannelParams" for row in empty_rows),
            "an empty ChannelParams has no proximity rows in tessera-protocol/1")
    require(len(empty_rows) == len(by_name["405b/live-64/1"]["rows"]) - 2, "two rows fewer")
    require(by_name["edge/empty-channel-params"]["meets_targets"], "and meets live-64's targets")
    for name, r in by_name.items():
        if not name.startswith("adversarial/"):
            continue
        failed = [(row["row"], row.get("commitment")) for row in r["rows"] if row["pass"] is False]
        aimed = name.split("/", 3)[-1]
        kind, _, commitment = aimed.partition("/")
        expected = [(kind, commitment or None)]
        if kind == "rounds":
            expected.append(("union", None))
        require(sorted(failed, key=str) == sorted(expected, key=str), f"{name}: fails {failed}")
        # The row that fails is the minimum. The rounds row's union bound counts its terms and
        # more, so it has at most the rounds row's bits: it is the minimum when it has fewer, and
        # the rounds row, the first of the two, when they tie.
        least = r["rows"][r["minimum"]]
        want = (kind, commitment or None)
        if kind == "rounds":
            bits = {row["row"]: row["bits"] for row in r["rows"] if row["row"] in ("rounds", "union")}
            require(bits["union"] <= bits["rounds"], f"{name}: the union counts the rounds")
            if bits["union"] < bits["rounds"]:
                want = ("union", None)
        require((least["row"], least.get("commitment")) == want, f"{name}: the minimum")
    for name, verdict in [("edge/numerator-pass", True), ("edge/numerator-fail", False)]:
        openings = [row for row in by_name[name]["rows"] if row["row"] == "openings"][0]
        require(openings["pass"] is verdict, f"{name}: the decision")
    require(by_name["edge/logup-2^50-live"]["meets_targets"], "2^50 passes live-64's one-shot target")
    for name, numerator in [("edge/line-only", "1"), ("edge/mask-only", "1"), ("edge/zero-count", "3")]:
        rounds = by_name[name]["rows"][0]
        require(rounds["row"] == "rounds" and rounds["numerator"] == numerator, f"{name}: rounds")
    # RFC-0054 'Soundness': the 405B statement's rounds row is the merge's batching, P + G; the
    # union counts G more copies of the leaf stage's levels and the merge; without groups both
    # are the ungrouped statement's.
    groups = base["groups"]
    for params in PARAM_SETS:
        rows = by_name[f"405b/{params['id']}"]["rows"]
        require(rows[0]["numerator"] == str(groups["merge_claims"] + groups["count"]),
                "the 405B rounds row is the merge's batching")
    plain = {row["row"]: row for row in by_name["edge/groups-none"]["rows"][:2]}
    rows = {row["row"]: row for row in by_name["405b/ni-100/1"]["rows"][:2]}
    levels = (sum(r["count"] * r["degree"] for r in groups["level_rounds"])
              + sum(m + 1 for m in base["tree_levels"]))
    require(int(rows["union"]["numerator"]) == int(plain["union"]["numerator"])
            + groups["count"] * levels + 2 * groups["merge_rounds"] + groups["merge_claims"]
            + groups["count"], "the grouped union adds G copies and the merge")
    require(plain["rounds"]["numerator"] == str(max(base["stages"])), "ungrouped, a stage's m")
    miss = by_name["edge/groups-union-miss"]
    require([row["row"] for row in miss["rows"] if row["pass"] is False] == ["union"],
            "a grouped union bound alone misses its target")
    require(by_name["edge/groups-none"]["meets_targets"], "and without groups it passes")
    merge_only = by_name["edge/groups-merge-only"]["rows"]
    require([merge_only[0]["numerator"], merge_only[1]["numerator"]] == ["3", "5"],
            "a merge alone: P + G and 2·n_max + P + G")
    hash_128 = by_name["edge/hash-128"]["rows"]
    require([row["row"] for row in hash_128 if row["pass"]] == ["hash"], "only the hash reaches 128")
    logup = [row for row in by_name["edge/q-below-2^128"]["rows"] if row["row"] == "logup"][0]
    require(logup["bits"] == 64 * UNIT - 1 and logup["pass"] is False, "q < 2^128 − 1 decides")

    header = {
        "description": "Soundness reports (RFC-0014 'Soundness calculator'), written by "
        "spec/vectors/soundness/generate.py; see README.md",
        "field": {"p": str(P), "q": str(Q)},
        "log2": {"precision": W, "field_fraction": FIELD_FRACTION, "query_fraction": QUERY_FRACTION},
        "seed": f"0x{SEED:016x}",
    }
    text = json.dumps(header, indent=2, ensure_ascii=False)[:-2]
    text += ',\n  "log2_cases": [\n' + ",\n".join(compact(c) for c in log2_cases(rng)) + "\n  ]"
    text += ',\n  "parameter_sets": [\n' + ",\n".join(compact(p) for p in PARAM_SETS) + "\n  ]"
    text += ',\n  "cases": [\n' + ",\n".join(compact(c) for c in cases) + "\n  ]\n}\n"
    json.loads(text)  # the file must parse
    (HERE / "soundness.json").write_text(text)


if __name__ == "__main__":
    main()
