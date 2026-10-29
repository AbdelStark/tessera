#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0 OR MIT
"""Reference vectors for RFC-0071 ('Batched Request Slots'):
- batch slot geometry and RFC-0044 Llama lowering shapes/positions for R = 1, 2 and 3 requests of
  different lengths (under both tessera-protocol/1 and tessera-protocol/2 slot widths);
- the slot row selector psi_tilde, the batch causal selector and the SoftmaxRow product-forest root
  claim at random Fp2 points and edge Boolean vertices;
- the exact-greedy runtime boundary's active windows, key columns, native schedule, per-request key
  time coefficients (RFC-0053 + RFC-0071), and per-request scoring window sums U_hat_r (RFC-0017 +
  RFC-0071).

This script is an independent oracle written from RFC-0071, RFC-0044, RFC-0010, RFC-0030, RFC-0035,
RFC-0053 and RFC-0017 alone. Multilinear values are computed directly from their hypercube sums
sum_x f(x) * eq(r, x) in Python arbitrary-precision integers over Fp2 = Fp[X]/(X^2 - 7), and
cross-checked against the O(R log S) closed forms before writing. It shares no code with crates/,
verifier/ or python/. Both workspaces check slots.json.

Usage: python3 spec/vectors/request-slots/generate.py
(rewrites slots.json)
"""

import importlib.util
import json
import sys
from pathlib import Path

sys.dont_write_bytecode = True

HERE = Path(__file__).resolve().parent
SEED = 0x7E55E7A_2026_0071
P = 2**64 - 2**32 + 1  # Goldilocks (RFC-0002)
W = 7  # Fp2 = Fp[X]/(X^2 - 7)


def _oracle(name):
    spec = importlib.util.spec_from_file_location(
        f"{name}_oracle", HERE.parent / name / "generate.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


lowering_oracle = _oracle("lowering")
models_oracle = lowering_oracle.models
tables_oracle = lowering_oracle.tables


def require(condition, message):
    if not condition:
        raise SystemExit(f"request-slots oracle: {message}")


def splitmix64(state):
    mask = (1 << 64) - 1
    while True:
        state = (state + 0x9E3779B97F4A7C15) & mask
        z = state
        z = ((z ^ (z >> 30)) * 0xBF58476D1CE4E5B9) & mask
        z = ((z ^ (z >> 27)) * 0x94D049BB133111EB) & mask
        yield z ^ (z >> 31)


class Rng:
    def __init__(self, seed):
        self._words = splitmix64(seed)

    def fp(self):
        while True:
            w = next(self._words)
            if w < P:
                return w

    def fp2(self):
        return [self.fp(), self.fp()]

    def point(self, k):
        return [self.fp2() for _ in range(k)]

    def below(self, bound):
        require(bound >= 1, f"below({bound})")
        limit = (1 << 64) - ((1 << 64) % bound)
        while True:
            w = next(self._words)
            if w < limit:
                return w % bound


# ----- Fp2 = Fp[X]/(X^2 - 7) -------------------------------------------------------------------

ZERO = [0, 0]
ONE = [1, 0]


def add(a, b):
    return [(a[0] + b[0]) % P, (a[1] + b[1]) % P]


def sub(a, b):
    return [(a[0] - b[0]) % P, (a[1] - b[1]) % P]


def mul(a, b):
    return [(a[0] * b[0] + W * a[1] * b[1]) % P, (a[0] * b[1] + a[1] * b[0]) % P]


def eq_vertex(point, index):
    acc = ONE
    for i, r_i in enumerate(point):
        bit = (index >> i) & 1
        term = r_i if bit else sub(ONE, r_i)
        acc = mul(acc, term)
    return acc


def mle_bool(indicator, point):
    acc = ZERO
    for x in range(1 << len(point)):
        if indicator(x):
            acc = add(acc, eq_vertex(point, x))
    return acc


def ceil_log2(n):
    require(n >= 1, f"ceil_log2({n})")
    return (n - 1).bit_length()


# ----- Closed-form O(R log S) cross-check ------------------------------------------------------

def lt_closed(point, bound):
    k = len(point)
    if bound <= 0:
        return ZERO
    if bound >= (1 << k):
        return ONE
    eq_pref = ONE
    acc = ZERO
    for i in range(k - 1, -1, -1):
        r_i = point[i]
        if (bound >> i) & 1:
            acc = add(acc, mul(eq_pref, sub(ONE, r_i)))
            eq_pref = mul(eq_pref, r_i)
        else:
            eq_pref = mul(eq_pref, sub(ONE, r_i))
    return acc


def causal_bound_closed(r_key, r_pos, bound):
    """Closed-form O(b) sum_{0 <= j <= p < bound} eq(r_key, j) * eq(r_pos, p)."""
    b = len(r_pos)
    require(len(r_key) == b, "causal_bound_closed axis mismatch")
    if bound <= 0:
        return ZERO
    if bound >= (1 << b):
        eq_lt, lt_lt, eq_eq, lt_eq = ONE, ZERO, ZERO, ZERO
    else:
        eq_lt, lt_lt, eq_eq, lt_eq = ZERO, ZERO, ONE, ZERO
    for i in range(b - 1, -1, -1):
        kc = r_key[i]
        qp = r_pos[i]
        not_kc = sub(ONE, kc)
        not_qp = sub(ONE, qp)
        w00 = mul(not_kc, not_qp)
        w01 = mul(not_kc, qp)
        w11 = mul(kc, qp)
        n_i = (bound >> i) & 1 if bound < (1 << b) else 0
        next_lt_lt = add(lt_lt, mul(eq_lt, w01))
        next_eq_lt = mul(eq_lt, add(w00, w11))
        if n_i == 0:
            next_lt_eq = mul(lt_eq, not_qp)
            next_eq_eq = mul(eq_eq, w00)
        else:
            next_lt_lt = add(next_lt_lt, mul(lt_eq, not_qp))
            next_lt_eq = add(mul(lt_eq, qp), mul(eq_eq, w01))
            next_eq_lt = add(next_eq_lt, mul(eq_eq, w00))
            next_eq_eq = mul(eq_eq, w11)
        eq_lt, lt_lt, eq_eq, lt_eq = next_eq_lt, next_lt_lt, next_eq_eq, next_lt_eq
    return add(eq_lt, lt_lt)


# ----- Slot geometry and batch lowering (RFC-0071 'The slots', 'Lowering') ---------------------

def slot_geometry(lengths, protocol):
    require(len(lengths) >= 1 and all(n >= 1 for n in lengths), f"invalid lengths {lengths}")
    require(protocol in (1, 2), f"invalid protocol {protocol}")
    r_count = len(lengths)
    n_max = max(lengths)
    raw_b = ceil_log2(n_max)
    b = raw_b if protocol == 1 else max(raw_b, 2)
    s = 1 << b
    if r_count == 1:
        n_tok = lengths[0]
        tok_vars = b
        key_vars = b
        padded = 1 << tok_vars
        real_rows = [tau < lengths[0] for tau in range(n_tok)]
        positions = list(range(lengths[0]))
    else:
        n_tok = r_count * s
        req_vars = ceil_log2(r_count)
        tok_vars = b + req_vars
        key_vars = b
        padded = 1 << tok_vars
        real_rows = [(tau % s) < lengths[tau // s] for tau in range(n_tok)]
        positions = [
            (tau % s) if (tau % s) < lengths[tau // s] else 0 for tau in range(n_tok)
        ]
    return {
        "r": r_count,
        "n_max": n_max,
        "slot_bits": b,
        "slot_len": s,
        "token_len": n_tok,
        "token_vars": tok_vars,
        "key_vars": key_vars,
        "padded_tokens": padded,
        "real_rows": real_rows,
        "positions": positions,
        "positions_bounds": [0, n_max - 1],
    }


def lower_batch_shapes(public, lengths, protocol):
    """RFC-0044 + RFC-0071 'Lowering': batch-lowered graph and its key tensor shapes."""
    geom = slot_geometry(lengths, protocol)
    a, q = public.arch, public.quant
    require(geom["n_max"] <= a["max_context"], "n_max <= max_context")
    n_tok = geom["token_len"]
    n_key = lengths[0] if geom["r"] == 1 else geom["n_max"]

    if geom["r"] == 1 and protocol == 1:
        g_single, _ = lowering_oracle.lower(public, lengths[0])
    else:
        g_single = None

    d, h, kv, dh, ff, vocab = (
        a[k] for k in ("d_model", "n_heads", "n_kv_heads", "d_head", "d_ff", "vocab")
    )
    g_ratio = h // kv
    b_res = q["residual_bits"]
    sigma = q["exp_sigma"]
    servable = public.profile == "Servable"
    g = lowering_oracle.Graph()

    rope = a["rope"]
    g.table({
        "Rope": {
            "theta": rope["theta"],
            "factor": rope["factor"],
            "low_freq_factor": rope["low_freq_factor"],
            "high_freq_factor": rope["high_freq_factor"],
            "original_max": rope["original_max"],
            "bits": rope["bits"],
            "max_context": a["max_context"],
            "d_head": dh,
        }
    })
    g.at = "inputs"
    tokens = g.input("tokens", [n_tok], "A", 0, vocab - 1)
    positions = g.input("positions", [n_tok], "P", 0, geom["n_max"] - 1)

    def weight_role(kind, ty, layer):
        return f"{kind} {ty} {layer}"

    def matmul(ty, layer, x, rows, cols):
        w = g.input(weight_role("weight", ty, layer), [rows, cols], "W", -127, 127, ty="i8")
        g.weights.append([ty, layer, w])
        return g.node("MatMulWeight", {"weight": [ty, layer]}, [x, w], ([n_tok, rows], "v"))

    def key(x, p, c):
        mode, value = public.site(p)
        if mode == "affine":
            m = g.input(
                weight_role("multipliers", lowering_oracle.WEIGHT_BEFORE[p[2]], p[1]),
                [c], "W", 0, 2**16 - 1,
            )
            z = g.node("Mul", {"broadcast": "Leading"}, [x, m], ([n_tok, c], "v"))
            return z, [value], None
        if len(value) == 1:
            return x, value, None
        kappa = g.input(
            weight_role("classes", lowering_oracle.WEIGHT_BEFORE[p[2]], p[1]),
            [c], "W", 0, len(value) - 1,
        )
        return x, value, kappa

    def requant(z, shifts, kappa, c, lo, hi):
        s = g.shift_class_set(shifts)
        inputs = [z] if kappa is None else [z, kappa]
        return g.node("Requant", {"shift": s, "lo": lo, "hi": hi}, inputs, ([n_tok, c], "A"))

    def rescale(x, p, c):
        g.at = f"{lowering_oracle.label(p)}: rescale"
        z, shifts, kappa = key(x, p, c)
        return requant(z, shifts, kappa, c, -128, 127)

    def norm(res, p):
        g.at = f"{lowering_oracle.label(p)}: norm"
        u = g.node("SumSquares", {"axis": 1}, [res], ([n_tok], "v"))
        rho = g.node(
            "RsqrtRow",
            {"kappa": q["rsqrt_kappa"], "eps_int": q["rms_eps_int"], "dim": d},
            [u], ([n_tok], "A"),
        )
        y = g.node("Mul", {"broadcast": "Trailing"}, [res, rho], ([n_tok, d], "v"))
        return rescale(y, p, d)

    def rope_(x, k):
        y = g.node("Reshape", {}, [x], ([n_tok, k, dh], "v"))
        y = g.node("Rotate", {"rope": 0}, [y, positions], ([n_tok, k, dh], "v"))
        return g.node("Reshape", {}, [y], ([n_tok, k * dh], "v"))

    def heads(x, k):
        y = g.node("Reshape", {}, [x], ([n_tok, k, dh], "v"))
        y = g.node("Transpose", {"perm": [1, 0, 2]}, [y], ([k, n_tok, dh], "v"))
        if k == h:
            return y
        y = g.node("Expand", {"axis": 1, "len": g_ratio}, [y], ([k, g_ratio, n_tok, dh], "v"))
        return g.node("Reshape", {}, [y], ([h, n_tok, dh], "v"))

    lo_res, hi_res = -(2 ** (b_res - 1)), 2 ** (b_res - 1) - 1

    def residual(acc, res, p):
        g.at = f"{lowering_oracle.label(p)}: residual"
        z, shifts, kappa = key(acc, p, d)
        if public.fused(p[2]):
            (r,) = shifts
            c = g.constant([2**r])
            t = g.node("ConstMul", {"c": c}, [res], ([n_tok, d], "v"))
            t = g.node("Add", {}, [z, t], ([n_tok, d], "v"))
            s = g.shift_class_set([r])
            return g.node("Requant", {"shift": s, "lo": lo_res, "hi": hi_res}, [t], ([n_tok, d], "A"))
        u = requant(z, shifts, kappa, d, 1 - 2**b_res, 2**b_res - 1)
        t = g.node("Add", {}, [res, u], ([n_tok, d], "v"))
        c = g.constant([2])
        t = g.node("ConstMul", {"c": c}, [t], ([n_tok, d], "v"))
        s = g.shift_class_set([1])
        return g.node("Requant", {"shift": s, "lo": lo_res, "hi": hi_res}, [t], ([n_tok, d], "A"))

    def exp_table(layer):
        num, den = public.scales[layer]
        for key_bits in range(lowering_oracle.bitlen(128 * 255 * dh), 33):
            kind = {"Exp": {"key_bits": key_bits, "sigma": sigma, "scale": [num, den]}}
            if g.has_table(kind):
                return g.table(kind)
            if tables_oracle.exp_value(2**key_bits - 1, sigma, num, den) == 0:
                return g.table(kind)
        raise SystemExit("no Exp table")

    def silu_table(layer):
        mode, shifts = public.site(("Layer", layer, "Silu"))
        require(mode == "pot", "Silu pot")
        return g.table({
            "Silu": {
                "b": lowering_oracle.bitlen(127 * 128 * d) + 1,
                "r_by_class": shifts,
                "in_exp": public.silu_in[layer],
                "out_exp": public.act["Layer", layer, "Silu"],
            }
        })

    g.at = "prologue"
    one_hot = g.node("OneHot", {"vocab": vocab}, [tokens], ([n_tok, vocab], "A"))
    x = matmul("Embed", 0, one_hot, d, vocab)
    s_e = public.act[("Embed",)] - q["residual_exponent"]
    c = g.constant([2**s_e])
    res = g.node("ConstMul", {"c": c}, [x], ([n_tok, d], "A"))

    layer0_shapes = None
    for layer in range(a["n_layers"]):
        def p(site, layer=layer):
            return ("Layer", layer, site)

        y = norm(res, p("AttnNorm"))
        g.at = f"layer {layer}: projections"
        q_ = matmul("Q", layer, y, h * dh, d)
        k_ = matmul("K", layer, y, kv * dh, d)
        v_ = matmul("V", layer, y, kv * dh, d)
        g.at = f"layer {layer}: rope"
        q_ = rope_(q_, h)
        k_ = rope_(k_, kv)
        q_ = rescale(q_, p("RopeQ"), h * dh)
        k_ = rescale(k_, p("RopeK"), kv * dh)
        v_ = rescale(v_, p("V"), kv * dh)
        g.at = f"layer {layer}: attention"
        qh, kh, vh = heads(q_, h), heads(k_, kv), heads(v_, kv)
        scores = g.node("MatMulAct", {"transpose_rhs": True}, [qh, kh], ([h, n_tok, n_key], "v"))
        t = exp_table(layer)
        e, recip = g.node(
            "SoftmaxRow", {"exp_table": t, "sigma": sigma}, [scores],
            ([h, n_tok, n_key], lowering_oracle.lookup(t)), ([h, n_tok], "A"),
        )
        pv = g.node("MatMulAct", {"transpose_rhs": False}, [e, vh], ([h, n_tok, dh], "v"))
        if layer == 0:
            layer0_shapes = {
                "q": list(g.tensors[qh][0]),
                "k": list(g.tensors[kh][0]),
                "v": list(g.tensors[vh][0]),
                "scores": list(g.tensors[scores][0]),
                "exp": list(g.tensors[e][0]),
                "reciprocal": list(g.tensors[recip][0]),
                "pv": list(g.tensors[pv][0]),
            }
        o = g.node("Mul", {"broadcast": "Trailing"}, [pv, recip], ([h, n_tok, dh], "v"))
        o = g.node("Transpose", {"perm": [1, 0, 2]}, [o], ([n_tok, h, dh], "v"))
        o = g.node("Reshape", {}, [o], ([n_tok, h * dh], "v"))
        if servable:
            o = rescale(o, p("AttnOut"), h * dh)
        g.at = f"layer {layer}: O"
        acc = matmul("O", layer, o, d, h * dh)
        res = residual(acc, res, p("AttnResidual"))

        y = norm(res, p("MlpNorm"))
        g.at = f"layer {layer}: gate and up"
        gate = matmul("Gate", layer, y, ff, d)
        up = matmul("Up", layer, y, ff, d)
        g.at = f"layer {layer}: silu"
        t_s = silu_table(layer)
        inputs = [gate]
        silu_shifts = public.shifts(p("Silu"))
        if len(silu_shifts) > 1:
            inputs.append(
                g.input(weight_role("classes", "Gate", layer), [ff], "W", 0, len(silu_shifts) - 1)
            )
        act = g.node(
            "TableMap", {"table": t_s}, inputs,
            ([n_tok, ff], lowering_oracle.lookup(t_s)),
        )
        if servable:
            u = rescale(up, p("Up"), ff)
            g.at = f"layer {layer}: product"
            prod = g.node("Mul", {"broadcast": "Trailing"}, [act, u], ([n_tok, ff], "v"))
            prod = rescale(prod, p("Product"), ff)
        else:
            g.at = f"layer {layer}: product"
            prod = g.node("Mul", {"broadcast": "Trailing"}, [act, up], ([n_tok, ff], "v"))
        g.at = f"layer {layer}: down"
        acc = matmul("Down", layer, prod, d, ff)
        res = residual(acc, res, p("MlpResidual"))

    y = norm(res, ("FinalNorm",))
    g.at = "epilogue: logits"
    logits = matmul("LmHead", 0, y, vocab, d)

    if g_single is not None:
        require(g.inputs == g_single.inputs, "R = 1 inputs match RFC-0044")
        require(g.nodes == g_single.nodes, "R = 1 nodes match RFC-0044")
        require(g.constants == g_single.constants, "R = 1 constants match RFC-0044")
        require(g.shift_classes == g_single.shift_classes, "R = 1 shift_classes match RFC-0044")
        require(g.weights == g_single.weights, "R = 1 weights match RFC-0044")
        require(g.tables == g_single.tables, "R = 1 tables match RFC-0044")

    return {
        "slot_bits": geom["slot_bits"],
        "slot_len": geom["slot_len"],
        "token_len": geom["token_len"],
        "padded_tokens": geom["padded_tokens"],
        "real_rows": geom["real_rows"],
        "positions": geom["positions"],
        "positions_bounds": geom["positions_bounds"],
        "shapes": {
            "tokens": list(g.tensors[tokens][0]),
            "positions": list(g.tensors[positions][0]),
            "one_hot": list(g.tensors[one_hot][0]),
            "embed": list(g.tensors[res][0]),
            **layer0_shapes,
            "logits": list(g.tensors[logits][0]),
        },
        "node_count": len(g.nodes),
        "tensor_count": len(g.tensors),
        "table_count": len(g.tables),
    }


# ----- Predicates & root claim (RFC-0071 'The selectors and the causal prefix') ----------------

def predicate_case(rng, name, lengths, protocol, heads, boolean_probe=None):
    geom = slot_geometry(lengths, protocol)
    b = geom["slot_bits"]
    s = geom["slot_len"]
    r_count = geom["r"]
    tok_vars = geom["token_vars"]
    key_vars = geom["key_vars"]
    head_vars = ceil_log2(heads)

    if boolean_probe is None:
        r_token = rng.point(tok_vars)
        r_key = rng.point(key_vars)
        r_head = rng.point(head_vars)
    else:
        tau, key_idx, h_idx = boolean_probe
        r_token = [ONE if ((tau >> i) & 1) else ZERO for i in range(tok_vars)]
        r_key = [ONE if ((key_idx >> i) & 1) else ZERO for i in range(key_vars)]
        r_head = [ONE if ((h_idx >> i) & 1) else ZERO for i in range(head_vars)]

    def is_real_tau(tau):
        if r_count == 1:
            return tau < lengths[0]
        beta = tau >> b
        p = tau & (s - 1)
        return beta < r_count and p < lengths[beta]

    # Direct hypercube evaluation (ground truth).
    psi_tok = mle_bool(is_real_tau, r_token)
    head_lt = mle_bool(lambda h_i: h_i < heads, r_head)
    psi_head_tok = mle_bool(
        lambda x: is_real_tau(x >> head_vars) and ((x & ((1 << head_vars) - 1)) < heads),
        r_head + r_token,
    )
    require(psi_head_tok == mul(head_lt, psi_tok), f"{name}: psi(h, tau) factorization")

    def is_causal(pair_idx):
        j = pair_idx & ((1 << key_vars) - 1)
        tau = pair_idx >> key_vars
        if r_count == 1:
            return tau < lengths[0] and j <= tau
        beta = tau >> b
        p = tau & (s - 1)
        return beta < r_count and p < lengths[beta] and j <= p

    causal = mle_bool(is_causal, r_key + r_token)
    softmax_root = sub(ONE, psi_head_tok)

    # Cross-check O(R log S) closed forms against direct hypercube evaluation.
    if r_count == 1:
        psi_closed = lt_closed(r_token, lengths[0])
        causal_closed = causal_bound_closed(r_key, r_token, lengths[0])
    else:
        r_p = r_token[:b]
        r_beta = r_token[b:]
        psi_closed = ZERO
        causal_closed = ZERO
        for req_idx, n_r in enumerate(lengths):
            eq_b = eq_vertex(r_beta, req_idx)
            psi_closed = add(psi_closed, mul(eq_b, lt_closed(r_p, n_r)))
            causal_closed = add(causal_closed, mul(eq_b, causal_bound_closed(r_key, r_p, n_r)))
    require(psi_closed == psi_tok, f"{name}: closed-form psi mismatch")
    require(causal_closed == causal, f"{name}: closed-form causal mismatch")

    return {
        "name": name,
        "protocol": protocol,
        "lengths": lengths,
        "heads": heads,
        "slot_bits": b,
        "slot_len": s,
        "r_token": r_token,
        "r_key": r_key,
        "r_head": r_head,
        "slot_selector": psi_tok,
        "head_row_selector": psi_head_tok,
        "causal_selector": causal,
        "head_causal_selector": mul(head_lt, causal),
        "softmax_root": softmax_root,
    }


# ----- Greedy windows, runtime boundary & U_hat_r (RFC-0053, RFC-0017, RFC-0071) ---------------

def greedy_case(rng, name, requests, vocab, protocol=1):
    require(len(requests) >= 1, "non-empty requests")
    lengths = [pr + out for pr, out in requests]
    require(all(pr >= 1 and out >= 0 for pr, out in requests), "prompt >= 1, output >= 0")
    geom = slot_geometry(lengths, protocol)
    b = geom["slot_bits"]
    s = geom["slot_len"]
    r_count = geom["r"]
    tok_vars = geom["token_vars"]
    padded = geom["padded_tokens"]
    vocab_vars = ceil_log2(vocab)

    r_time = rng.point(tok_vars)

    active_positions = []
    key_columns = []
    for req_idx, (prompt_len, output_len) in enumerate(requests):
        base = 0 if r_count == 1 else req_idx * s
        window = list(range(base + prompt_len - 1, base + prompt_len - 1 + output_len))
        active_positions.extend(window)
        if output_len > 0:
            row_vars = ceil_log2(output_len)
            p_row = rng.point(row_vars)
            # Direct definition: sum_{u < output_len} eq(p_row, u) * eq(r_time, base + prompt_len - 1 + u)
            coeff = ZERO
            win_mle = ZERO
            for u in range(output_len):
                eq_t = eq_vertex(r_time, base + prompt_len - 1 + u)
                win_mle = add(win_mle, eq_t)
                coeff = add(coeff, mul(eq_vertex(p_row, u), eq_t))
            key_columns.append({
                "request_index": req_idx,
                "prompt_len": prompt_len,
                "output_len": output_len,
                "entries": 1 << (row_vars + vocab_vars),
                "p_row": p_row,
                "key_time_coeff": coeff,
                "window_mle": win_mle,
            })

    active_set = set(active_positions)
    require(len(active_set) == len(active_positions), f"{name}: disjoint request windows")
    selected_active_mle = mle_bool(lambda t: t in active_set, r_time)

    has_maximum = len(key_columns) > 0
    if has_maximum:
        native_kinds = (
            ["tokens"]
            + [f"greedy_key:{col['request_index']}" for col in key_columns]
            + ["greedy_selected"]
        )
    else:
        native_kinds = ["tokens", "observe"]

    # Synthetic row bounds u_t (in 1/256-bit FixedBits units) across all 0..padded_tokens,
    # including nonzero decoys on prompt-only and padded positions so window slicing is tested.
    row_bounds = [rng.below(2048) + 1 for _ in range(padded)]
    window_u_hat = []
    for req_idx, (prompt_len, output_len) in enumerate(requests):
        base = 0 if r_count == 1 else req_idx * s
        window_u_hat.append(
            sum(row_bounds[base + prompt_len - 1 + u] for u in range(output_len))
        )

    return {
        "name": name,
        "protocol": protocol,
        "vocab": vocab,
        "requests": [
            {"prompt_len": pr, "output_len": out} for pr, out in requests
        ],
        "slot_bits": b,
        "slot_len": s,
        "token_len": geom["token_len"],
        "padded_tokens": padded,
        "has_maximum": has_maximum,
        "active_positions": active_positions,
        "native_kinds": native_kinds,
        "r_time": r_time,
        "selected_active_mle": selected_active_mle,
        "key_columns": key_columns,
        "greedy_u_hat": [0 for _ in requests],
        "row_bounds": row_bounds,
        "window_u_hat": window_u_hat,
    }


def main():
    rng = Rng(SEED)

    publics = {}
    for spec in models_oracle.MODELS:
        if spec["name"] in ("tiny-2-minimal", "tiny-4-servable"):
            model, sequences = models_oracle.build(spec)
            chooser = models_oracle.Run(model, spec["plan"])
            chooser.forward(sequences)
            d = models_oracle.integer_model(spec, model, chooser.c)
            publics[spec["name"]] = lowering_oracle.Public({
                "arch": d["arch"],
                "profile": d["profile"],
                "quant": {
                    "public": d["quant"]["public"],
                    "silu_in_exponents": d["quant"]["silu_in_exponents"],
                },
            })

    lowering_cases = []
    for model_name in ("tiny-2-minimal", "tiny-4-servable"):
        pub = publics[model_name]
        for case_name, protocol, lengths in [
            (f"{model_name}-r1-n5", 1, [5]),
            (f"{model_name}-r2-5-2", 1, [5, 2]),
            (f"{model_name}-r2-1-4", 1, [1, 4]),
            (f"{model_name}-r3-5-1-7", 1, [5, 1, 7]),
            (f"{model_name}-r2-p1-small", 1, [1, 2]),
            (f"{model_name}-r2-p2-small", 2, [1, 2]),
            (f"{model_name}-r3-p2-small", 2, [2, 1, 3]),
        ]:
            res = lower_batch_shapes(pub, lengths, protocol)
            lowering_cases.append({
                "name": case_name,
                "model": f"{model_name}.d",
                "protocol": protocol,
                "lengths": lengths,
                **res,
            })

    predicate_cases = [
        predicate_case(rng, "r1-n5-random", [5], 1, 4),
        predicate_case(rng, "r2-5-2-random-a", [5, 2], 1, 4),
        predicate_case(rng, "r2-5-2-random-b", [5, 2], 1, 3),
        predicate_case(rng, "r2-3-6-random", [3, 6], 1, 1),
        predicate_case(rng, "r3-5-1-7-random-a", [5, 1, 7], 1, 4),
        predicate_case(rng, "r3-5-1-7-random-b", [5, 1, 7], 1, 3),
        predicate_case(rng, "r3-2-4-3-random", [2, 4, 3], 1, 2),
        predicate_case(rng, "r2-small-p1", [1, 2], 1, 2),
        predicate_case(rng, "r2-small-p2", [1, 2], 2, 2),
        # Edge Boolean vertices: padded inter-slot row, cross-slot key index, real boundary, past-R slot.
        predicate_case(rng, "r2-5-2-bool-real-slot0", [5, 2], 1, 4, boolean_probe=(4, 4, 3)),
        predicate_case(rng, "r2-5-2-bool-padded-slot0", [5, 2], 1, 4, boolean_probe=(5, 0, 0)),
        predicate_case(rng, "r2-5-2-bool-real-slot1", [5, 2], 1, 4, boolean_probe=(9, 1, 2)),
        predicate_case(rng, "r2-5-2-bool-future-key-slot1", [5, 2], 1, 4, boolean_probe=(8, 1, 0)),
        predicate_case(rng, "r2-5-2-bool-padded-slot1", [5, 2], 1, 4, boolean_probe=(10, 0, 0)),
        predicate_case(rng, "r3-5-1-7-bool-past-r-slot3", [5, 1, 7], 1, 4, boolean_probe=(24, 0, 0)),
    ]

    greedy_cases = [
        greedy_case(rng, "r1-2p-3o", [(2, 3)], 11),
        greedy_case(rng, "r1-4p-0o", [(4, 0)], 11),
        greedy_case(rng, "r2-diff-both-outputs", [(2, 3), (1, 1)], 11),
        greedy_case(rng, "r2-first-zero-output", [(3, 0), (2, 4)], 16),
        greedy_case(rng, "r2-second-zero-output", [(1, 4), (3, 0)], 16),
        greedy_case(rng, "r2-both-zero-outputs", [(3, 0), (5, 0)], 11),
        greedy_case(rng, "r3-diff-all-outputs", [(2, 3), (4, 1), (1, 6)], 32),
        greedy_case(rng, "r3-mixed-zero-middle", [(2, 3), (1, 0), (3, 4)], 32),
        greedy_case(rng, "r3-all-zero-outputs", [(2, 0), (1, 0), (5, 0)], 32),
    ]

    document = {
        "description": (
            "Conformance vectors for RFC-0071 batched request slots: slot geometry, RFC-0044 "
            "batch-lowered shapes and public positions, multilinear slot/causal selectors and "
            "SoftmaxRow forest root claims over Fp2, and exact-greedy runtime windows, key "
            "coefficients and U_hat_r window sums for batches of 1, 2 and 3 requests."
        ),
        "lowering": lowering_cases,
        "predicates": predicate_cases,
        "greedy": greedy_cases,
    }
    text = json.dumps(document, indent=2) + "\n"
    (HERE / "slots.json").write_text(text, encoding="utf-8")


if __name__ == "__main__":
    main()
