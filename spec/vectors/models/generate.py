#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0 OR MIT
"""Tiny conformance models of D (07 §5): Llama-3.1-architecture models with random INT8
weights, their IntegerModel artifacts, token sequences, and what D computes on them, the residual
stream at every layer boundary and the logits.

This script is the independent oracle. It is written from RFC-0005 ('Integer semantics',
'Profiles'), as RFC-0022 (the saturating residual) and RFC-0023 (the norm's κ) amend it, from
03 §2 (the model types) and RFC-0003 (the encoding) alone, computing with Python's integers. It
takes RFC-0008's tables, rq and the Exp, Silu and RoPE generators, from ../tables/generate.py,
and BLAKE3 from ../hash/generate.py. It shares no code with crates/, verifier/ or python/, which
check its output: the Rust engine and the PyTorch reference must reproduce every value here.

The models are built so that their values are worth checking:

- each site's shift is the least that keeps its outputs on the sequences here inside the output's
  range, so the INT8 tensors use their range rather than a few of its values;
- channels take one to four shift classes, the RoPE pairs sharing theirs, as RFC-0005's scale
  policy makes them;
- a few sites are pushed past their range on purpose, so that the vectors exercise every clamp:
  an INT8 requantization, the Silu table's, the residual's INT20 (RFC-0022), and an attention
  layer whose scale sends exponentials to 0.

Before writing anything it checks that its encoder writes the two models of ../model/ byte for
byte, and afterwards that every clamp it was asked for happened.

Usage: python3 spec/vectors/models/generate.py
(rewrites models.json and the *.d artifacts)
"""

import importlib.util
import json
import math
import sys
from fractions import Fraction
from pathlib import Path

sys.dont_write_bytecode = True  # importing the other oracles leaves nothing behind

HERE = Path(__file__).resolve().parent


def _oracle(name):
    spec = importlib.util.spec_from_file_location(f"{name}_oracle", HERE.parent / name / "generate.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


hash_oracle = _oracle("hash")
tables = _oracle("tables")


def require(condition, message):
    """A check that holds under `python -O` too, unlike `assert`."""
    if not condition:
        raise SystemExit(f"models oracle: {message}")


# ----- RFC-0003: the encoding --------------------------------------------------------------------
#
# A type is a tuple: (primitive,), ("vec", T), ("tuple", T1, T2), ("map", K, V),
# ("struct", [(name, T), ...]) or ("enum", [(variant, T or None), ...]). A value follows the codec
# vectors' conventions (../codec/README.md): an integer as an int or a decimal string, a struct as
# a dict by field name, an enum as {variant: fields}, a pair as a list, a map as a list of pairs,
# a digest as a list of 32 bytes.

WIDTHS = {"u8": 1, "u16": 2, "u32": 4, "u64": 8, "i8": 1, "i16": 2, "i32": 4, "i64": 8}


def integer(value):
    if isinstance(value, str):
        return int(value, 10)
    require(isinstance(value, int) and not isinstance(value, bool), f"not an integer: {value!r}")
    return value


def fixed(kind, value):
    width = WIDTHS[kind]
    x = integer(value)
    if kind[0] == "u":
        require(0 <= x < 1 << (8 * width), f"{kind} out of range: {x}")
        return x.to_bytes(width, "little")
    require(-(1 << (8 * width - 1)) <= x < 1 << (8 * width - 1), f"{kind} out of range: {x}")
    return x.to_bytes(width, "little", signed=True)


def length(n):
    return n.to_bytes(8, "little")


def encode(ty, value):
    kind = ty[0]
    if kind in WIDTHS:
        return fixed(kind, value)
    if kind == "digest":
        data = bytes(integer(b) for b in value)
        require(len(data) == 32, "a digest has 32 bytes")
        return data
    if kind == "ascii":
        data = value.encode("ascii")
        require(all(0x20 <= b <= 0x7E for b in data), f"not printable ASCII: {value!r}")
        return length(len(data)) + data
    if kind == "rational":
        num, den = integer(value["num"]), integer(value["den"])
        require(den > 0 and math.gcd(num, den) == 1, f"not a reduced rational: {num}/{den}")
        return fixed("i64", num) + fixed("u64", den)
    if kind == "vec":
        if ty[1] == ("i8",):  # weights: millions of entries
            data = [integer(x) for x in value]
            require(all(-128 <= x < 128 for x in data), "i8 out of range")
            return length(len(data)) + bytes(x & 0xFF for x in data)
        return length(len(value)) + b"".join(encode(ty[1], x) for x in value)
    if kind == "tuple":
        require(len(value) == len(ty) - 1, f"a {len(ty) - 1}-tuple")
        return b"".join(encode(t, x) for t, x in zip(ty[1:], value))
    if kind == "map":
        entries = [(encode(ty[1], k), encode(ty[2], v)) for k, v in value]
        keys = [k for k, _ in entries]
        require(all(a < b for a, b in zip(keys, keys[1:])), "a map's keys strictly ascend")
        return length(len(entries)) + b"".join(k + v for k, v in entries)
    if kind == "struct":
        names = [name for name, _ in ty[1]]
        require(sorted(value) == sorted(names), f"fields {sorted(value)} against {names}")
        return b"".join(encode(t, value[name]) for name, t in ty[1])
    if kind == "enum":
        require(len(value) == 1, "an enum value names one variant")
        ((variant, fields),) = value.items()
        names = [name for name, _ in ty[1]]
        require(variant in names, f"unknown variant {variant}")
        tag = names.index(variant)
        field_type = ty[1][tag][1]
        if field_type is None:
            require(fields is None, f"{variant} has no fields")
            return fixed("u16", tag)
        return fixed("u16", tag) + encode(field_type, fields)
    raise SystemExit(f"models oracle: unknown type {ty!r}")


U8, U16, U32, U64, I8, I64 = ("u8",), ("u16",), ("u32",), ("u64",), ("i8",), ("i64",)
ASCII, DIGEST, RATIONAL = ("ascii",), ("digest",), ("rational",)


def vec(t):
    return ("vec", t)


def pair(a, b):
    return ("tuple", a, b)


def struct(*fields):
    return ("struct", list(fields))


def enum(*variants):
    return ("enum", [(v, None) if isinstance(v, str) else v for v in variants])


# 03 §2, in declaration order.
SITES = ["AttnNorm", "RopeQ", "RopeK", "V", "AttnOut", "AttnResidual", "MlpNorm", "Silu", "Up",
         "Product", "MlpResidual"]
WEIGHT_TYPES = ["Embed", "Q", "K", "V", "O", "Gate", "Up", "Down", "LmHead"]
ROPE_SPEC = struct(("theta", U32), ("factor", U16), ("low_freq_factor", U16),
                   ("high_freq_factor", U16), ("original_max", U32), ("bits", U8))
ARCH_SPEC = struct(("family", enum("LlamaV3")), ("n_layers", U32), ("d_model", U32),
                   ("n_heads", U32), ("n_kv_heads", U32), ("d_head", U32), ("d_ff", U32),
                   ("vocab", U32), ("max_context", U32), ("rope", ROPE_SPEC), ("rms_eps", RATIONAL))
PROFILE = enum("Servable", "Minimal")
QUANT_POINT = enum("Embed", ("Layer", struct(("layer", U32), ("site", enum(*SITES)))), "FinalNorm")
WEIGHT_TYPE = enum(*WEIGHT_TYPES)
PUBLIC_QUANT_SPEC = struct(
    ("act_exponents", vec(pair(QUANT_POINT, I8))), ("residual_exponent", I8),
    ("residual_bits", U8), ("rsqrt_kappa", U8), ("rms_eps_int", U64), ("exp_sigma", U8),
    ("score_scales", vec(pair(U32, RATIONAL))), ("shift_class_sets", vec(pair(QUANT_POINT, vec(I8)))),
    ("rope", ROPE_SPEC), ("logits_bits", U8))
CHANNEL_PARAMS = struct(("ty", WEIGHT_TYPE), ("layer", U32), ("class", vec(U8)), ("multiplier", vec(U16)))
QUANT_SPEC = struct(("public", PUBLIC_QUANT_SPEC), ("silu_in_exponents", vec(pair(U32, I8))),
                    ("channels", vec(CHANNEL_PARAMS)))
WEIGHT_TENSOR = struct(("ty", WEIGHT_TYPE), ("layer", U32), ("rows", U32), ("cols", U32), ("data", vec(I8)))
CALIBRATION_IDENTITY = struct(
    ("dataset", ASCII), ("config", ASCII), ("split", ASCII), ("revision", ASCII), ("files", vec(ASCII)),
    ("seed", U64), ("sequences", U32), ("held_out", U32), ("tokens_per_sequence", U32), ("selection", DIGEST))
PROVENANCE = struct(("tessera_quant", ASCII), ("checkpoint", vec(pair(ASCII, DIGEST))),
                    ("calibration", CALIBRATION_IDENTITY), ("statistics", DIGEST), ("alpha", RATIONAL),
                    ("libraries", ("map", ASCII, ASCII)))
INTEGER_MODEL = struct(("arch", ARCH_SPEC), ("profile", PROFILE), ("quant", QUANT_SPEC),
                       ("weights", vec(WEIGHT_TENSOR)), ("provenance", PROVENANCE))

SCHEMA_ID, SCHEMA_VERSION = 0x0101, 1  # 03 §5: IntegerModel


def seal(payload):
    """RFC-0003 'Envelope': the header, the payload, and BLAKE3 derive_key over both."""
    header = b"TSRA" + fixed("u16", 1) + fixed("u16", SCHEMA_ID) + fixed("u16", SCHEMA_VERSION)
    header += fixed("u16", 0) + fixed("u64", len(payload))
    signed = header + payload
    return signed + hash_oracle.H("envelope", signed)


def check_encoder():
    """The two models of ../model/ encode to their bytes: every tag, order and width is right."""
    document = json.loads((HERE.parent / "model" / "integer-model.json").read_text(encoding="utf-8"))
    number = lambda x: int(x, 0) if isinstance(x, str) else x
    require(number(document["schema"]) == SCHEMA_ID, "IntegerModel's schema id")
    require(number(document["version"]) == SCHEMA_VERSION, "IntegerModel's version")
    for case in document["cases"]:
        written = seal(encode(INTEGER_MODEL, case["value"]))
        require(written == bytes.fromhex(case["hex"]), f"the encoder writes {case['name']} otherwise")
    require(len(document["cases"]) == 2, "both pinned models")


# ----- RFC-0005: D's integer semantics -----------------------------------------------------------

INT8 = (-128, 127)


def rq(x, r, lo, hi):
    return tables.rq(x, r, lo, hi)


def rounded(x, r):
    """⌊(x + 2^(r−1)) / 2^r⌋: rq before its clamp."""
    return (x + (1 << (r - 1))) >> r


def matvec(weight, x):
    """W x for a weight of `rows` rows of `cols` entries, row-major."""
    rows, cols, data = weight["rows"], weight["cols"], weight["data"]
    require(len(x) == cols, "a matmul's input has the weight's columns")
    return [sum(map(int.__mul__, data[i * cols:(i + 1) * cols], x)) for i in range(rows)]


KINDS = ("requant_clamped", "silu_clamped", "residual_saturated", "exp_zero")


class Events:
    """How often each clamp was reached, and where, so that the vectors are known to exercise it:
    a requantization that clamped, a Silu value at the table's clamp, a residual sum saturated at
    its width (RFC-0022), an attention weight whose exponential is 0."""

    def __init__(self):
        self.at = {}

    def add(self, kind, key, count):
        if count:
            self.at[kind, key] = self.at.get((kind, key), 0) + count

    def total(self, kind):
        return sum(n for (k, _), n in self.at.items() if k == kind)

    def as_dict(self):
        return {kind: self.total(kind) for kind in KINDS}


class Run:
    """D's forward pass over several sequences at once (RFC-0005 'Integer semantics').

    With `constants` None, the pass chooses each site's constants as it reaches the site, from the
    values every sequence gives there, and records them; otherwise it applies the recorded ones.
    The values of one sequence depend on that sequence alone either way.
    """

    def __init__(self, model, plan, constants=None):
        self.m = model
        self.plan = plan
        self.choosing = constants is None
        self.c = {} if constants is None else constants
        self.events = Events()

    # -- sites ----------------------------------------------------------------------------------

    def requant(self, key, values, classes, push=0, bounds=INT8):
        """y_c = rq(x_c, r_{class c}, lo, hi): the least base shift keeping every y inside
        [lo, hi], `push` bits less when the plan wants this site to clamp."""
        n = max(classes) + 1
        lo, hi = bounds
        if self.choosing:
            base = 1
            while any(not lo <= rounded(x, base + k) <= hi for row in values for x, k in zip(row, classes)):
                base += 1
            base -= push
            require(base >= 1 and base + n - 1 <= 63, f"{key}: a shift lies in [1, 63]")
            self.c[key] = [base + k for k in range(n)]
        shifts = self.c[key]
        out = []
        for row in values:
            y = [rounded(x, shifts[k]) for x, k in zip(row, classes)]
            self.events.add("requant_clamped", key, sum(1 for v in y if v < lo or v > hi))
            out.append([min(max(v, lo), hi) for v in y])
        return out

    def residual(self, key, res, acc, classes, push=0):
        """res ← clamp(res + ⌊(acc + 2^(r−1)) / 2^r⌋, INT20), r the channel's class shift
        (RFC-0022). The base is the least keeping updates within 2^13, or 2^(13 + push)."""
        n = max(classes) + 1
        if self.choosing:
            target = 1 << (13 + push)
            base = 1
            while any(abs(rounded(x, base + k)) > target for row in acc for x, k in zip(row, classes)):
                base += 1
            require(base + n - 1 <= 63, f"{key}: a shift lies in [1, 63]")
            self.c[key] = [base + k for k in range(n)]
        shifts = self.c[key]
        half = 1 << (self.m["residual_bits"] - 1)
        out = []
        for row, update in zip(res, acc):
            s = [r + rounded(x, shifts[k]) for r, x, k in zip(row, update, classes)]
            self.events.add("residual_saturated", key, sum(1 for v in s if v < -half or v > half - 1))
            out.append([min(max(v, -half), half - 1) for v in s])
        return out

    def norm(self, key, res):
        """ρ = max{ρ : ρ²(s + e) ≤ d·2^(2κ)}, y_i = rq(res_i·ρ, r_n, −128, 127)."""
        d, kappa, eps = self.m["d_model"], self.m["kappa"], self.m["rms_eps_int"]
        products = []
        for row in res:
            s = sum(x * x for x in row)
            rho = math.isqrt(((d << (2 * kappa)) // (s + eps)))
            require(rho * rho * (s + eps) <= d << (2 * kappa) < (rho + 1) ** 2 * (s + eps), "ρ's bracket")
            products.append([x * rho for x in row])
        return self.requant(key, products, [0] * d)

    def rope(self, values, n_heads, positions):
        """Each head's pairs (x_i, x_(i + d_head/2)): (a, b) ↦ (a·C − b·S, a·S + b·C)."""
        dh = self.m["d_head"]
        half = dh // 2
        out = []
        for row, p in zip(values, positions):
            row = list(row)
            for h in range(n_heads):
                for i in range(half):
                    c, s = self.m["rope_table"](p, i)
                    a, b = row[h * dh + i], row[h * dh + i + half]
                    row[h * dh + i], row[h * dh + i + half] = a * c - b * s, a * s + b * c
            out.append(row)
        return out

    def attention(self, layer, q, k, v, lengths):
        """Causal attention of each sequence: the scores, their max certificate's exponentials, the
        reciprocal R = ⌊2^(2σ)/s⌋ and o = R·Σ e·v, per query head (grouped-query)."""
        m = self.m
        dh, nh, group = m["d_head"], m["n_heads"], m["n_heads"] // m["n_kv_heads"]
        sigma = m["exp_sigma"]
        # Every score's spread below its row's max, per sequence.
        spreads = []
        start = 0
        for n in lengths:
            seq = []
            for h in range(nh):
                kv = h // group
                for t in range(n):
                    qt = q[start + t][h * dh:(h + 1) * dh]
                    scores = [sum(map(int.__mul__, qt, k[start + i][kv * dh:(kv + 1) * dh]))
                              for i in range(t + 1)]
                    top = max(scores)
                    seq.append((h, t, [top - x for x in scores]))
            spreads.append(seq)
            start += n
        key = ("score_scale", layer)
        if self.choosing:
            widest = max(max(row) for seq in spreads for _, _, row in seq)
            reach = self.plan["exp_reach"].get(layer, 8)
            self.c[key] = Fraction(reach, max(widest, 1))
        scale = self.c[key]
        out = []
        start = 0
        for n, seq in zip(lengths, spreads):
            rows = [[0] * (nh * dh) for _ in range(n)]
            for h, t, row in seq:
                kv = h // group
                e = [tables.exp_value(x, sigma, scale.numerator, scale.denominator) for x in row]
                self.events.add("exp_zero", ("Layer", layer, "Exp"), e.count(0))
                total = sum(e)
                require(total > 0, "the row's maximum keeps its exponential")
                r = (1 << (2 * sigma)) // total
                for j in range(dh):
                    rows[t][h * dh + j] = r * sum(w * v[start + i][kv * dh + j] for i, w in enumerate(e))
            out.extend(rows)
            start += n
        return out

    def silu(self, key, g, classes, in_exp, out_exp, reach):
        """a = Silu(class, g): h = g >> r_c, clamp(round(silu(h·2^in_exp)/2^out_exp), −128, 127).
        The base is the least keeping |h| within 2^reach."""
        n = max(classes) + 1
        b = (127 * 128 * self.m["d_model"]).bit_length() + 1
        if self.choosing:
            base = 1
            while any(abs(x >> (base + k)) > 1 << reach for row in g for x, k in zip(row, classes)):
                base += 1
            shifts = [base + k for k in range(n)]
            require(all(r < b for r in shifts), f"{key}: r_c < b")
            require(sum(1 << (b - r) for r in shifts) <= 1 << 24, f"{key}: the table's core")
            self.c[key] = shifts
        shifts = self.c[key]
        out = []
        for row in g:
            a = [tables.silu_value(x >> shifts[k], in_exp, out_exp) for x, k in zip(row, classes)]
            self.events.add("silu_clamped", key, sum(1 for x in a if x in INT8))
            out.append(a)
        return out

    # -- the model ------------------------------------------------------------------------------

    def forward(self, sequences):
        m, w = self.m, self.m["weights"]
        servable = m["profile"] == "Servable"
        lengths = [len(s) for s in sequences]
        tokens = [t for s in sequences for t in s]
        positions = [p for s in sequences for p in range(len(s))]
        d = m["d_model"]
        s_e = m["embed_exponent"] - m["residual_exponent"]
        embed = w["Embed", 0]
        # x0 is column `token` of the transposed embedding: res = x0 · 2^s_e.
        res = [[embed["data"][c * m["vocab"] + t] << s_e for c in range(d)] for t in tokens]
        checkpoints = [res]
        classes = m["classes"]
        for layer in range(m["n_layers"]):
            def site(name, layer=layer):
                return ("Layer", layer, name)

            push = self.plan["push"].get(layer, {})
            y = self.norm(site("AttnNorm"), res)
            q = self.rope([matvec(w["Q", layer], x) for x in y], m["n_heads"], positions)
            q = self.requant(site("RopeQ"), q, classes["Q", layer], push=push.get("RopeQ", 0))
            k = self.rope([matvec(w["K", layer], x) for x in y], m["n_kv_heads"], positions)
            k = self.requant(site("RopeK"), k, classes["K", layer], push=push.get("RopeK", 0))
            v = self.requant(site("V"), [matvec(w["V", layer], x) for x in y], classes["V", layer],
                             push=push.get("V", 0))
            o = self.attention(layer, q, k, v, lengths)
            if servable:
                o = self.requant(site("AttnOut"), o, [0] * d, push=push.get("AttnOut", 0))
            acc = [matvec(w["O", layer], x) for x in o]
            res = self.residual(site("AttnResidual"), res, acc, classes["O", layer],
                                push=push.get("AttnResidual", 0))
            y2 = self.norm(site("MlpNorm"), res)
            g = [matvec(w["Gate", layer], x) for x in y2]
            in_exp = m["silu_in_exp"]
            a = self.silu(site("Silu"), g, classes["Gate", layer], in_exp, m["silu_out_exp"],
                          self.plan["silu_reach"].get(layer, 10))
            u = [matvec(w["Up", layer], x) for x in y2]
            if servable:
                u8 = self.requant(site("Up"), u, classes["Up", layer], push=push.get("Up", 0))
                h = self.requant(site("Product"), [[x * z for x, z in zip(ar, ur)] for ar, ur in zip(a, u8)],
                                 [0] * m["d_ff"], push=push.get("Product", 0))
            else:
                h = [[x * z for x, z in zip(ar, ur)] for ar, ur in zip(a, u)]
            acc = [matvec(w["Down", layer], x) for x in h]
            res = self.residual(site("MlpResidual"), res, acc, classes["Down", layer],
                                push=push.get("MlpResidual", 0))
            checkpoints.append(res)
        y = self.norm(("FinalNorm",), res)
        logits = [matvec(w["LmHead", 0], x) for x in y]
        # Back to one list per sequence.
        def split(rows):
            parts, start = [], 0
            for n in lengths:
                parts.append(rows[start:start + n])
                start += n
            return parts

        return [split(c) for c in checkpoints], split(logits)


# ----- The models --------------------------------------------------------------------------------

LLAMA_ROPE = {"theta": 500000, "factor": 8, "low_freq_factor": 1, "high_freq_factor": 4,
              "original_max": 8192, "bits": 14}
RMS_EPS = Fraction(1, 100000)
RESIDUAL_EXPONENT = -10  # σ_res = 2^−10
EMBED_EXPONENT = -3      # so s_e = 7: 127 · 2^7 < 2^19
INT8_EXPONENT = -3       # the other INT8 points' exponents; D's arithmetic reads only Silu's
SILU_IN_EXP = -7         # h · 2^−7: |h| ≤ 2^10 spans silu over [−8, 8]

# name, seed, profile, shape, the classes of each weight's channels, the sequences' lengths, and
# what to push past its range.
MODELS = [
    {
        "name": "tiny-2-servable", "seed": 0x7E55_E7A0_0083_0002, "profile": "Servable",
        "shape": {"n_layers": 2, "d_model": 64, "n_heads": 4, "n_kv_heads": 2, "d_head": 16,
                  "d_ff": 192, "vocab": 512, "max_context": 64},
        "classes": {"Q": 3, "K": 2, "V": 4, "O": 4, "Gate": 2, "Up": 3, "Down": 4},
        "lengths": [6, 9],
        "plan": {"push": {1: {"RopeQ": 1, "Product": 1}}, "silu_reach": {1: 12}, "exp_reach": {}},
    },
    {
        "name": "tiny-2-minimal", "seed": 0x7E55_E7A0_0083_0003, "profile": "Minimal",
        "shape": {"n_layers": 2, "d_model": 64, "n_heads": 4, "n_kv_heads": 2, "d_head": 16,
                  "d_ff": 192, "vocab": 512, "max_context": 64},
        "classes": {"Q": 2, "K": 3, "V": 1, "O": 3, "Gate": 4, "Up": 1, "Down": 4},
        "lengths": [7, 4],
        "plan": {"push": {1: {"MlpResidual": 7}}, "silu_reach": {}, "exp_reach": {0: 24}},
    },
    {
        "name": "tiny-4-servable", "seed": 0x7E55_E7A0_0083_0004, "profile": "Servable",
        "shape": {"n_layers": 4, "d_model": 256, "n_heads": 8, "n_kv_heads": 2, "d_head": 32,
                  "d_ff": 384, "vocab": 512, "max_context": 64},
        "classes": {"Q": 4, "K": 4, "V": 2, "O": 1, "Gate": 3, "Up": 4, "Down": 2},
        "lengths": [5, 8],
        "residual_bits": 16,
        "plan": {"push": {2: {"V": 1}, 3: {"AttnResidual": 3}}, "silu_reach": {2: 12},
                 "exp_reach": {1: 24}},
    },
]


class Rng:
    """SplitMix64, integer-only, so that every platform draws the same weights."""

    def __init__(self, seed):
        self.source = hash_oracle.splitmix64(seed)

    def next(self):
        return next(self.source)

    def below(self, n):
        """Uniform in [0, n), by rejection."""
        limit = (1 << 64) - (1 << 64) % n
        while True:
            x = self.next()
            if x < limit:
                return x % n

    def weights(self, count):
        """Bell-shaped INT8 entries: the sum of four bytes folded into [−31, 31] each."""
        out = []
        while len(out) < count:
            x = self.next()
            for half in (x & 0xFFFFFFFF, x >> 32):
                out.append(sum(((half >> (8 * i)) & 0xFF) % 63 - 31 for i in range(4)))
        return out[:count]


def weight_shapes(shape):
    d, ff, vocab = shape["d_model"], shape["d_ff"], shape["vocab"]
    q, kv = shape["n_heads"] * shape["d_head"], shape["n_kv_heads"] * shape["d_head"]
    per_layer = {"Q": (q, d), "K": (kv, d), "V": (kv, d), "O": (d, q), "Gate": (ff, d), "Up": (ff, d),
                 "Down": (d, ff)}
    out = [("Embed", 0, d, vocab)]
    for ty in WEIGHT_TYPES[1:8]:
        for layer in range(shape["n_layers"]):
            out.append((ty, layer, *per_layer[ty]))
    out.append(("LmHead", 0, vocab, d))
    return out


def channel_classes(rng, rows, n, d_head=None):
    """Each output channel's class: every class used, the rest drawn; the two rows of a RoPE pair
    share theirs."""
    if d_head is None:
        classes = [c % n if c < n else rng.below(n) for c in range(rows)]
        return classes
    half = d_head // 2
    classes = [0] * rows
    pair = 0
    for head in range(rows // d_head):
        for i in range(half):
            k = pair % n if pair < n else rng.below(n)
            classes[head * d_head + i] = classes[head * d_head + i + half] = k
            pair += 1
    return classes


def kappa_for(d, eps):
    """RFC-0023: the largest κ, at most 31, with ⌊d·2^(2κ)/e⌋ < 2^64."""
    return next(k for k in range(31, -1, -1) if (d << (2 * k)) // eps < 1 << 64)


def build(spec):
    rng = Rng(spec["seed"])
    shape = spec["shape"]
    d, vocab = shape["d_model"], shape["vocab"]
    weights = {}
    for ty, layer, rows, cols in weight_shapes(shape):
        weights[ty, layer] = {"ty": ty, "layer": layer, "rows": rows, "cols": cols,
                              "data": rng.weights(rows * cols)}
    classes = {}
    for ty, n in spec["classes"].items():
        for layer in range(shape["n_layers"]):
            rows = weights[ty, layer]["rows"]
            rope = shape["d_head"] if ty in ("Q", "K") else None
            classes[ty, layer] = channel_classes(rng, rows, n, rope)
    if spec["profile"] == "Minimal":
        # No site follows W_Up in minimal: its channels have no class.
        for layer in range(shape["n_layers"]):
            classes["Up", layer] = [0] * shape["d_ff"]
    sequences = []
    for n in spec["lengths"]:
        sequences.append([rng.below(vocab) for _ in range(n)])
    # The ends of the vocabulary appear.
    sequences[0][0], sequences[-1][-1] = 0, vocab - 1
    eps = -(-d * RMS_EPS.numerator * 4 ** (-RESIDUAL_EXPONENT) // RMS_EPS.denominator)
    rope = dict(LLAMA_ROPE, d_head=shape["d_head"], max_context=shape["max_context"])
    omegas = tables.rope_omegas(rope)
    cache = {}

    def rope_table(position, i):
        if (position, i) not in cache:
            cache[position, i] = tables.rope_pair(omegas[i], position, rope["bits"])
        return cache[position, i]

    model = dict(shape, profile=spec["profile"], weights=weights, classes=classes,
                 residual_bits=spec.get("residual_bits", 20), residual_exponent=RESIDUAL_EXPONENT,
                 embed_exponent=EMBED_EXPONENT,
                 rms_eps_int=eps, kappa=kappa_for(d, eps), exp_sigma=16, silu_in_exp=SILU_IN_EXP,
                 silu_out_exp=INT8_EXPONENT, rope_table=rope_table)
    return model, sequences


def sites_of(profile):
    if profile == "Servable":
        return SITES
    return [s for s in SITES if s not in ("AttnOut", "Up", "Product")]


def point(key):
    if key[0] == "FinalNorm":
        return {"FinalNorm": None}
    return {"Layer": {"layer": key[1], "site": {key[2]: None}}}


def integer_model(spec, model, constants):
    """D as 03 §2 lays it out: the public constants, the hidden channel classes, the weights."""
    n_layers, profile = model["n_layers"], model["profile"]
    after = {"RopeQ": "Q", "RopeK": "K", "V": "V", "AttnResidual": "O", "Silu": "Gate", "Up": "Up",
             "MlpResidual": "Down"}
    act = [[{"Embed": None}, model["embed_exponent"]]]
    sets = []
    channels = []
    for layer in range(n_layers):
        for site in sites_of(profile):
            key = ("Layer", layer, site)
            if site not in ("AttnResidual", "MlpResidual"):
                act.append([point(key), INT8_EXPONENT])
            shifts = constants[key]
            sets.append([point(key), shifts])
            if len(shifts) > 1:
                ty = after[site]
                channels.append({"ty": {ty: None}, "layer": layer, "class": model["classes"][ty, layer],
                                 "multiplier": []})
    act.append([{"FinalNorm": None}, INT8_EXPONENT])
    sets.append([{"FinalNorm": None}, constants[("FinalNorm",)]])
    channels.sort(key=lambda c: (WEIGHT_TYPES.index(next(iter(c["ty"]))), c["layer"]))
    rope = {k: LLAMA_ROPE[k] for k in ("theta", "factor", "low_freq_factor", "high_freq_factor",
                                       "original_max", "bits")}
    scales = []
    for layer in range(n_layers):
        scale = constants[("score_scale", layer)]
        scales.append([layer, {"num": scale.numerator, "den": scale.denominator}])
    zero = [0] * 32
    return {
        "arch": {"family": {"LlamaV3": None}, **{k: model[k] for k in (
            "n_layers", "d_model", "n_heads", "n_kv_heads", "d_head", "d_ff", "vocab", "max_context")},
                 "rope": rope, "rms_eps": {"num": RMS_EPS.numerator, "den": RMS_EPS.denominator}},
        "profile": {profile: None},
        "quant": {
            "public": {
                "act_exponents": act, "residual_exponent": model["residual_exponent"],
                "residual_bits": model["residual_bits"], "rsqrt_kappa": model["kappa"],
                "rms_eps_int": model["rms_eps_int"], "exp_sigma": model["exp_sigma"],
                "score_scales": scales, "shift_class_sets": sets, "rope": rope, "logits_bits": 28,
            },
            "silu_in_exponents": [[layer, model["silu_in_exp"]] for layer in range(n_layers)],
            "channels": channels,
        },
        "weights": [{"ty": {ty: None}, "layer": layer, "rows": w["rows"], "cols": w["cols"],
                     "data": w["data"]}
                    for (ty, layer), w in sorted(model["weights"].items(),
                                                 key=lambda item: (WEIGHT_TYPES.index(item[0][0]), item[0][1]))],
        "provenance": {
            "tessera_quant": "spec/vectors/models/generate.py", "checkpoint": [],
            "calibration": {"dataset": "none", "config": "none", "split": "none", "revision": "0" * 40,
                            "files": [], "seed": spec["seed"], "sequences": len(spec["lengths"]),
                            "held_out": 0, "tokens_per_sequence": max(spec["lengths"]), "selection": zero},
            "statistics": zero, "alpha": {"num": 1, "den": 1}, "libraries": [],
        },
    }


# ----- Output ------------------------------------------------------------------------------------

def compact(value, indent=0):
    """JSON with every innermost list of integers on one line."""
    pad = " " * indent
    if isinstance(value, list) and all(isinstance(x, int) for x in value):
        return json.dumps(value, separators=(",", ":"))
    if isinstance(value, list):
        inner = ",\n".join(pad + " " + compact(x, indent + 1) for x in value)
        return "[\n" + inner + "\n" + pad + "]"
    if isinstance(value, dict):
        inner = ",\n".join(f"{pad} {json.dumps(k)}: {compact(v, indent + 1)}" for k, v in value.items())
        return "{\n" + inner + "\n" + pad + "}"
    return json.dumps(value)


def planned(spec, events):
    """Each site the plan pushes past its range clamps on these sequences."""
    plan = spec["plan"]
    for layer, sites in plan["push"].items():
        for site in sites:
            kind = "residual_saturated" if site.endswith("Residual") else "requant_clamped"
            require(events.at.get((kind, ("Layer", layer, site)), 0) > 0,
                    f"{spec['name']}: layer {layer}'s {site} does not clamp")
    for layer in plan["silu_reach"]:
        require(events.at.get(("silu_clamped", ("Layer", layer, "Silu")), 0) > 0,
                f"{spec['name']}: layer {layer}'s Silu does not clamp")
    for layer in plan["exp_reach"]:
        require(events.at.get(("exp_zero", ("Layer", layer, "Exp")), 0) > 0,
                f"{spec['name']}: layer {layer}'s exponentials do not reach 0")


def main():
    check_encoder()
    entries = []
    total = Events()
    for spec in MODELS:
        model, sequences = build(spec)
        chooser = Run(model, spec["plan"])
        checkpoints, logits = chooser.forward(sequences)
        # The constants chosen, applied afresh, give the same values: D is a function of them.
        replay = Run(model, spec["plan"], chooser.c)
        require(replay.forward(sequences) == (checkpoints, logits), f"{spec['name']} replays")
        events = replay.events
        planned(spec, events)
        for (kind, key), n in events.at.items():
            total.add(kind, key, n)
        artifact = seal(encode(INTEGER_MODEL, integer_model(spec, model, chooser.c)))
        file = f"{spec['name']}.d"
        (HERE / file).write_bytes(artifact)
        entries.append({
            "name": spec["name"],
            "file": file,
            "bytes": len(artifact),
            "digest": artifact[-32:].hex(),
            "profile": spec["profile"],
            "shape": spec["shape"],
            "events": events.as_dict(),
            "sequences": [
                {"tokens": tokens,
                 "residuals": [checkpoint[s] for checkpoint in checkpoints],
                 "logits": logits[s]}
                for s, tokens in enumerate(sequences)
            ],
        })
    # Every clamp D has is reached somewhere.
    for kind in KINDS:
        require(total.total(kind) > 0, f"no model reaches {kind}")
    document = {
        "description": "Tiny conformance models of D (07 §5): each artifact, an IntegerModel sealed in "
        "its envelope, with its digest, and what D computes on its token sequences: the residual "
        "stream before the first layer and after each, and the logits. 'events' counts the clamps "
        "the sequences reach. Written by generate.py; see README.md.",
        "schema": "0x0101",
        "version": "1",
        "models": entries,
    }
    (HERE / "models.json").write_text(compact(document) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
