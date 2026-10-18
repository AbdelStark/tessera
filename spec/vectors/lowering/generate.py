#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0 OR MIT
"""The Llama lowering of RFC-0044, node by node: for each conformance model of ../models/, and for
variants of two of them that take the forms the models miss, at the lengths n = 1, 4, 5 and
max_context, every graph input, node, constant, shift-class set, weight and table of the lowered
graph, each by its id.

This script is the independent oracle of RFC-0044 'Testing Strategy'. It is written from RFC-0044
alone, with the definitions it cites: RFC-0005 (the weight before each site), RFC-0006 (the
operations and their shapes), RFC-0008 (the table kinds), RFC-0022 and RFC-0038 (the residual's
routes, one template for every layer), RFC-0025 (affine sites), RFC-0030 (the width of an Exp
table), RFC-0037 (the head layouts and the committed first checkpoint), RFC-0042 (the SiLU input
exponents) and 03 §2-3 (the model and circuit types). It reads each model's public part only,
the architecture, the profile, PublicQuantSpec and the SiLU input exponents (RFC-0006 'Llama
lowering'), and never its weights or channel parameters.

It takes that public part from ../models/generate.py, which rebuilds each model, and RFC-0008's Exp
values and row counts from ../tables/generate.py. It shares no code with crates/, verifier/ or
python/. tessera-compiler's lowering must reproduce every id here
(crates/tessera-compiler/tests/lowering_vectors.rs), and the verifier's will (#107).

Before writing, it checks its own graphs against the rules of RFC-0006 'Operation set' (topological
order, one producer per tensor, each operation's shapes) and RFC-0037 'Layers and kinds' (every
layer the same template), which RFC-0044 must meet; and that the JSON it writes reads back as the
graphs it built.

Usage: python3 spec/vectors/lowering/generate.py
(rewrites lowering.json)
"""

import importlib.util
import json
import sys
from pathlib import Path

sys.dont_write_bytecode = True  # importing the other oracles leaves nothing behind

HERE = Path(__file__).resolve().parent


def _oracle(name):
    spec = importlib.util.spec_from_file_location(f"{name}_oracle", HERE.parent / name / "generate.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


models = _oracle("models")
tables = models.tables  # the tables oracle, as the models oracle loaded it


def require(condition, message):
    """A check that holds under `python -O` too, unlike `assert`."""
    if not condition:
        raise SystemExit(f"lowering oracle: {message}")


def bitlen(x):
    """RFC-0044 'The tables': the number of bits of x, so that bitlen(1) = 1."""
    require(x >= 1, f"bitlen of {x}")
    return x.bit_length()


# ----- The public part (03 §2, RFC-0006 'Llama lowering') ----------------------------------------

SITES = ["AttnNorm", "RopeQ", "RopeK", "V", "AttnOut", "AttnResidual", "MlpNorm", "Silu", "Up",
         "Product", "MlpResidual"]

# RFC-0044 'Notation': "the weight before p" is RFC-0005's.
WEIGHT_BEFORE = {"RopeQ": "Q", "RopeK": "K", "V": "V", "AttnResidual": "O", "Silu": "Gate",
                 "Up": "Up", "MlpResidual": "Down"}

# RFC-0025 'Public data': the sites an affine rescale may take.
AFFINE_SITES = ("RopeQ", "RopeK", "V", "AttnResidual", "Up", "MlpResidual")


def point_key(point):
    """A QuantPoint as a key: ("Layer", layer, site), ("Embed",) or ("FinalNorm",)."""
    require(isinstance(point, dict) and len(point) == 1, f"a QuantPoint: {point!r}")
    ((variant, fields),) = point.items()
    if variant == "Layer":
        require(isinstance(fields["site"], dict) and len(fields["site"]) == 1, "a Site")
        ((site, _),) = fields["site"].items()
        require(site in SITES, f"unknown site {site}")
        return ("Layer", fields["layer"], site)
    require(variant in ("Embed", "FinalNorm") and fields is None, f"unknown QuantPoint {variant}")
    return (variant,)


def label(key):
    return "FinalNorm" if key == ("FinalNorm",) else f"layer {key[1]} {key[2]}"


class Public:
    """The part of D the lowering reads (RFC-0006 'Llama lowering', RFC-0042 'The verifier's
    lowering'): the architecture, the profile, PublicQuantSpec and silu_in_exponents."""

    def __init__(self, model):
        self.arch = model["arch"]
        ((self.profile, _),) = model["profile"].items()
        require(self.profile in ("Servable", "Minimal"), f"unknown profile {self.profile}")
        quant = model["quant"]["public"]
        self.quant = quant
        self.act = {}
        for point, exponent in quant["act_exponents"]:
            key = point_key(point)
            require(key not in self.act, f"{key} has two exponents")
            self.act[key] = exponent
        self.rescale = {}  # point -> ("pot", shifts) or ("affine", r)
        for point, shifts in quant["shift_class_sets"]:
            key = point_key(point)
            require(key not in self.rescale, f"{key} is listed twice")
            self.rescale[key] = ("pot", list(shifts))
        for point, r in quant["affine_sites"]:
            key = point_key(point)
            require(key not in self.rescale, f"{key} is listed twice")
            self.rescale[key] = ("affine", r)
        self.scales = {}
        for layer, scale in quant["score_scales"]:
            require(layer not in self.scales, f"layer {layer} has two score scales")
            self.scales[layer] = (scale["num"], scale["den"])
        self.silu_in = {}
        for layer, exponent in model["quant"]["silu_in_exponents"]:
            require(layer not in self.silu_in, f"layer {layer} has two SiLU input exponents")
            self.silu_in[layer] = exponent
        self.validate()

    def sites(self):
        """The rescale sites of the profile (RFC-0005 'Profiles'): minimal has no AttnOut, Up or
        Product."""
        if self.profile == "Servable":
            return SITES
        return [s for s in SITES if s not in ("AttnOut", "Up", "Product")]

    def validate(self):
        """What the lowering's structure rests on (RFC-0006 'Validation', RFC-0025 'Validation',
        RFC-0038 'The rule'): a model breaking any of it is not one RFC-0044 lowers."""
        a = self.arch
        layers = range(a["n_layers"])
        require(a["d_model"] == a["n_heads"] * a["d_head"], "d_model = n_heads · d_head")
        require(a["n_heads"] % a["n_kv_heads"] == 0, "n_kv_heads divides n_heads")
        require(a["d_head"] % 2 == 0, "an even d_head")
        points = {("FinalNorm",)} | {("Layer", l, s) for l in layers for s in self.sites()}
        require(set(self.rescale) == points, "every rescale site of the profile listed once")
        require(set(self.scales) == set(layers), "a score scale per layer")
        require(set(self.silu_in) == set(layers), "a SiLU input exponent per layer")
        require(("Embed",) in self.act, "the embedding's exponent")
        for key, (mode, value) in self.rescale.items():
            shifts = [value] if mode == "affine" else value
            require(1 <= len(shifts) <= 4 and all(1 <= r <= 63 for r in shifts),
                    f"{label(key)}: one to four shifts in [1, 63]")
            site = key[2] if key[0] == "Layer" else "FinalNorm"
            require(mode == "pot" or site in AFFINE_SITES, f"{label(key)} cannot be affine")
            require(len(shifts) == 1 or site in WEIGHT_BEFORE,
                    f"{label(key)} follows no weight and has several classes")
        for site in self.sites():
            modes = {self.rescale["Layer", l, site][0] for l in layers}
            require(len(modes) == 1, f"RFC-0038: {site} is affine in some layers only")
            if modes == {"pot"}:
                several = {len(self.rescale["Layer", l, site][1]) > 1 for l in layers}
                require(len(several) == 1, f"RFC-0038: {site} has one class in some layers only")
        num_den = [self.scales[l] for l in layers]
        require(all(num > 0 and den > 0 for num, den in num_den), "positive score scales")

    def site(self, key):
        return self.rescale[key]

    def shifts(self, key):
        """The site's shifts: its classes', or an affine site's one."""
        mode, value = self.rescale[key]
        return [value] if mode == "affine" else value

    def fused(self, site):
        """RFC-0044 'residual' (RFC-0038 'The residual's route'): the fused key when in every
        layer the site has one shift r ≤ 62, of one class or affine; two steps otherwise."""
        return all(len(self.shifts(("Layer", l, site))) == 1 and self.shifts(("Layer", l, site))[0] <= 62
                   for l in range(self.arch["n_layers"]))


# ----- The graph (RFC-0044 'Ids') -----------------------------------------------------------------

def unsigned(lo, hi):
    """RFC-0044 'Graph inputs': the narrowest unsigned type holding [lo, hi], of at least one bit."""
    require(0 <= lo <= hi, f"an unsigned range: [{lo}, {hi}]")
    return f"u{max(hi.bit_length(), 1)}"


class Graph:
    """A lowered graph under construction. Tensors, nodes, constants, shift-class sets, weights
    and tables are each a dense list in creation order (RFC-0044 'Ids'): a graph input is created
    when RFC-0044 names it, a node's outputs with the node, in output order, and every mention of
    a constant or a shift-class set creates a new entry."""

    def __init__(self):
        self.tensors = []  # by tensor id: [shape, kind]
        self.producer = []  # by tensor id: the node that outputs it, or None for an input
        self.inputs = []
        self.nodes = []
        self.constants = []
        self.shift_classes = []
        self.weights = []
        self.tables = []
        self.at = ""

    def _tensor(self, shape, kind, producer):
        self.tensors.append([list(shape), kind])
        self.producer.append(producer)
        return len(self.tensors) - 1

    def input(self, role, shape, kind, lo, hi, ty=None):
        """A graph input. Its type is the narrowest unsigned one holding its values, of at least
        one bit, except a weight's, which is i8."""
        ty = unsigned(lo, hi) if ty is None else ty
        tensor = self._tensor(shape, kind, None)
        self.inputs.append([tensor, role, list(shape), kind, ty, lo, hi])
        return tensor

    def node(self, op, params, inputs, *outputs):
        """A node reading `inputs`; each output a (shape, kind), created with it in order.
        Returns the output's id, or the ids when there are several."""
        node = len(self.nodes)
        ids = [self._tensor(shape, kind, node) for shape, kind in outputs]
        self.nodes.append([node, op, params, list(inputs),
                           [[t, list(shape), kind] for t, (shape, kind) in zip(ids, outputs)], self.at])
        return ids[0] if len(ids) == 1 else ids

    def constant(self, values):
        self.constants.append(list(values))
        return len(self.constants) - 1

    def shift_class_set(self, shifts):
        self.shift_classes.append(list(shifts))
        return len(self.shift_classes) - 1

    def has_table(self, kind):
        return kind in self.tables

    def table(self, kind):
        """An earlier table of the same definition, or the next id."""
        if kind in self.tables:
            return self.tables.index(kind)
        self.tables.append(kind)
        return len(self.tables) - 1


def lookup(t):
    """RFC-0044 'Notation': L(t), LookupOutput { table: t }."""
    return f"L({t})"


# ----- RFC-0044 'Proposed Design' ------------------------------------------------------------------

def lower(public, n):
    """The graph RFC-0044 lowers from `public` for a sequence of length n, with its layers' node
    ranges: layer l is the nodes after c_l's producer, up to and including c_(l+1)'s
    (RFC-0037 'Layers and kinds')."""
    a, q = public.arch, public.quant
    n_layers = a["n_layers"]
    require(1 <= n <= a["max_context"], f"n = {n} within max_context")
    # 'Notation'.
    d, h, kv, dh, ff, V = (a[k] for k in ("d_model", "n_heads", "n_kv_heads", "d_head", "d_ff", "vocab"))
    g_ = h // kv
    b = q["residual_bits"]
    sigma = q["exp_sigma"]
    servable = public.profile == "Servable"
    g = Graph()

    # 'Ids': table 0 is RoPE's, with the architecture's RopeSpec, max_context and d_head.
    rope = a["rope"]
    g.table({"Rope": {"theta": rope["theta"], "factor": rope["factor"],
                      "low_freq_factor": rope["low_freq_factor"],
                      "high_freq_factor": rope["high_freq_factor"],
                      "original_max": rope["original_max"], "bits": rope["bits"],
                      "max_context": a["max_context"], "d_head": dh}})

    # 'Graph inputs'.
    g.at = "inputs"
    tokens = g.input("tokens", [n], "A", 0, V - 1)
    positions = g.input("positions", [n], "P", 0, n - 1)

    def weight_role(kind, ty, layer):
        return f"{kind} {ty} {layer}"

    # 'Procedures'.
    def matmul(ty, layer, x, rows, cols):
        w = g.input(weight_role("weight", ty, layer), [rows, cols], "W", -127, 127, ty="i8")
        g.weights.append([ty, layer, w])
        return g.node("MatMulWeight", {"weight": [ty, layer]}, [x, w], ([n, rows], "v"))

    def key(x, p, c):
        """What the site p rescales of x [n, c], with its shifts and its class vector."""
        mode, value = public.site(p)
        if mode == "affine":
            m = g.input(weight_role("multipliers", WEIGHT_BEFORE[p[2]], p[1]), [c], "W", 0, 2**16 - 1)
            z = g.node("Mul", {"broadcast": "Leading"}, [x, m], ([n, c], "v"))
            return z, [value], None
        if len(value) == 1:
            return x, value, None
        kappa = g.input(weight_role("classes", WEIGHT_BEFORE[p[2]], p[1]), [c], "W", 0, len(value) - 1)
        return x, value, kappa

    def requant(z, shifts, kappa, c, lo, hi):
        s = g.shift_class_set(shifts)
        inputs = [z] if kappa is None else [z, kappa]
        return g.node("Requant", {"shift": s, "lo": lo, "hi": hi}, inputs, ([n, c], "A"))

    def rescale(x, p, c):
        g.at = f"{label(p)}: rescale"
        z, shifts, kappa = key(x, p, c)
        return requant(z, shifts, kappa, c, -128, 127)

    def norm(res, p):
        g.at = f"{label(p)}: norm"
        u = g.node("SumSquares", {"axis": 1}, [res], ([n], "v"))
        rho = g.node("RsqrtRow", {"kappa": q["rsqrt_kappa"], "eps_int": q["rms_eps_int"], "dim": d},
                     [u], ([n], "A"))
        y = g.node("Mul", {"broadcast": "Trailing"}, [res, rho], ([n, d], "v"))
        return rescale(y, p, d)

    def rope_(x, k):
        y = g.node("Reshape", {}, [x], ([n, k, dh], "v"))
        y = g.node("Rotate", {"rope": 0}, [y, positions], ([n, k, dh], "v"))
        return g.node("Reshape", {}, [y], ([n, k * dh], "v"))

    def heads(x, k):
        y = g.node("Reshape", {}, [x], ([n, k, dh], "v"))
        y = g.node("Transpose", {"perm": [1, 0, 2]}, [y], ([k, n, dh], "v"))
        if k == h:
            return y
        y = g.node("Expand", {"axis": 1, "len": g_}, [y], ([k, g_, n, dh], "v"))
        return g.node("Reshape", {}, [y], ([h, n, dh], "v"))

    lo_res, hi_res = -(2 ** (b - 1)), 2 ** (b - 1) - 1

    def residual(acc, res, p):
        g.at = f"{label(p)}: residual"
        z, shifts, kappa = key(acc, p, d)
        if public.fused(p[2]):
            (r,) = shifts
            c = g.constant([2**r])
            t = g.node("ConstMul", {"c": c}, [res], ([n, d], "v"))
            t = g.node("Add", {}, [z, t], ([n, d], "v"))
            s = g.shift_class_set([r])
            return g.node("Requant", {"shift": s, "lo": lo_res, "hi": hi_res}, [t], ([n, d], "A"))
        u = requant(z, shifts, kappa, d, 1 - 2**b, 2**b - 1)
        t = g.node("Add", {}, [res, u], ([n, d], "v"))
        c = g.constant([2])
        t = g.node("ConstMul", {"c": c}, [t], ([n, d], "v"))
        s = g.shift_class_set([1])
        return g.node("Requant", {"shift": s, "lo": lo_res, "hi": hi_res}, [t], ([n, d], "A"))

    # 'The tables'.
    def exp_table(layer):
        num, den = public.scales[layer]
        for key_bits in range(bitlen(128 * 255 * dh), 33):
            kind = {"Exp": {"key_bits": key_bits, "sigma": sigma, "scale": [num, den]}}
            if g.has_table(kind):
                return g.table(kind)
            if tables.exp_value(2**key_bits - 1, sigma, num, den) == 0:
                return g.table(kind)
        raise SystemExit(f"lowering oracle: layer {layer}'s Exp table has no zero up to 32 bits")

    def silu_table(layer):
        mode, shifts = public.site(("Layer", layer, "Silu"))
        require(mode == "pot", "Silu is a pot site")
        return g.table({"Silu": {"b": bitlen(127 * 128 * d) + 1, "r_by_class": shifts,
                                 "in_exp": public.silu_in[layer],
                                 "out_exp": public.act["Layer", layer, "Silu"]}})

    # 'The prologue'.
    g.at = "prologue"
    x = g.node("OneHot", {"vocab": V}, [tokens], ([n, V], "A"))
    x = matmul("Embed", 0, x, d, V)
    s_e = public.act[("Embed",)] - q["residual_exponent"]
    require(s_e >= 0, "the embedding's shift s_e ≥ 0")
    c = g.constant([2**s_e])
    res = g.node("ConstMul", {"c": c}, [x], ([n, d], "A"))
    boundaries = [len(g.nodes) - 1]  # c_0's producer

    # 'A layer'.
    for layer in range(n_layers):
        def p(site, layer=layer):
            return ("Layer", layer, site)

        # 1.
        y = norm(res, p("AttnNorm"))
        # 2.
        g.at = f"layer {layer}: projections"
        q_ = matmul("Q", layer, y, h * dh, d)
        k_ = matmul("K", layer, y, kv * dh, d)
        v_ = matmul("V", layer, y, kv * dh, d)
        # 3.
        g.at = f"layer {layer}: rope"
        q_ = rope_(q_, h)
        k_ = rope_(k_, kv)
        # 4.
        q_ = rescale(q_, p("RopeQ"), h * dh)
        k_ = rescale(k_, p("RopeK"), kv * dh)
        v_ = rescale(v_, p("V"), kv * dh)
        # 5.
        g.at = f"layer {layer}: attention"
        qh, kh, vh = heads(q_, h), heads(k_, kv), heads(v_, kv)
        scores = g.node("MatMulAct", {"transpose_rhs": True}, [qh, kh], ([h, n, n], "v"))
        t = exp_table(layer)
        e, recip = g.node("SoftmaxRow", {"exp_table": t, "sigma": sigma}, [scores],
                          ([h, n, n], lookup(t)), ([h, n], "A"))
        o = g.node("MatMulAct", {"transpose_rhs": False}, [e, vh], ([h, n, dh], "v"))
        o = g.node("Mul", {"broadcast": "Trailing"}, [o, recip], ([h, n, dh], "v"))
        o = g.node("Transpose", {"perm": [1, 0, 2]}, [o], ([n, h, dh], "v"))
        o = g.node("Reshape", {}, [o], ([n, h * dh], "v"))
        # 6.
        if servable:
            o = rescale(o, p("AttnOut"), h * dh)
        # 7.
        g.at = f"layer {layer}: O"
        acc = matmul("O", layer, o, d, h * dh)
        res = residual(acc, res, p("AttnResidual"))
        # 8.
        y = norm(res, p("MlpNorm"))
        # 9.
        g.at = f"layer {layer}: gate and up"
        gate = matmul("Gate", layer, y, ff, d)
        up = matmul("Up", layer, y, ff, d)
        # 10.
        g.at = f"layer {layer}: silu"
        t = silu_table(layer)
        inputs = [gate]
        silu_shifts = public.shifts(p("Silu"))
        if len(silu_shifts) > 1:
            inputs.append(g.input(weight_role("classes", "Gate", layer), [ff], "W", 0, len(silu_shifts) - 1))
        act = g.node("TableMap", {"table": t}, inputs, ([n, ff], lookup(t)))
        # 11.
        if servable:
            u = rescale(up, p("Up"), ff)
            g.at = f"layer {layer}: product"
            prod = g.node("Mul", {"broadcast": "Trailing"}, [act, u], ([n, ff], "v"))
            prod = rescale(prod, p("Product"), ff)
        else:
            g.at = f"layer {layer}: product"
            prod = g.node("Mul", {"broadcast": "Trailing"}, [act, up], ([n, ff], "v"))
        # 12.
        g.at = f"layer {layer}: down"
        acc = matmul("Down", layer, prod, d, ff)
        res = residual(acc, res, p("MlpResidual"))
        boundaries.append(len(g.nodes) - 1)  # c_(l+1)'s producer

    # 'The epilogue'.
    y = norm(res, ("FinalNorm",))
    g.at = "epilogue: logits"
    logits = matmul("LmHead", 0, y, V, d)
    require(g.tensors[logits] == [[n, V], "v"], "the logits are [n, V], virtual")
    return g, boundaries


# ----- Self-checks: RFC-0006 'Operation set' and RFC-0037 'Layers and kinds' -------------------------

def numel(shape):
    out = 1
    for x in shape:
        out *= x
    return out


def check_graph(public, g, boundaries):
    """The graph keeps the rules RFC-0044's output must keep: inputs before their readers, one
    producer per tensor, each operation's shapes (RFC-0006 'Operation set'), each kind
    (RFC-0044 'Notation') and every layer one template (RFC-0037 'Layers and kinds')."""
    shape = lambda t: g.tensors[t][0]
    kind = lambda t: g.tensors[t][1]
    require(len(g.producer) == len(g.tensors), "a producer slot per tensor")
    for t, role, sh, k, ty, lo, hi in g.inputs:
        require(g.producer[t] is None and g.tensors[t] == [sh, k], f"input {role} is a graph input")
        require(ty == "i8" if role.startswith("weight ") else ty == unsigned(lo, hi), f"input {role}'s type")
    for node, op, params, inputs, outputs, at in g.nodes:
        where = f"node {node} ({op}, {at})"
        require(all(0 <= t < len(g.tensors) for t in inputs), f"{where}: its inputs exist")
        require(all(g.producer[t] is None or g.producer[t] < node for t in inputs),
                f"{where}: reads earlier tensors only (topological order)")
        require(all(g.producer[t] == node for t, _, _ in outputs), f"{where}: produces its outputs")
        require(all(len(s) >= 1 and all(x >= 1 for x in s) for _, s, _ in outputs), f"{where}: shapes")
        ins = [shape(t) for t in inputs]
        outs = [s for _, s, _ in outputs]
        if op == "MatMulWeight":
            x, w = ins
            require(kind(inputs[1]) == "W" and len(w) == 2 and x[-1] == w[1] and outs == [x[:-1] + [w[0]]],
                    f"{where}: W [out, in], x [..., in], y [..., out]")
        elif op == "Add":
            require(len(ins) == 2 and ins[0] == ins[1] == outs[0], f"{where}: all shapes equal")
        elif op == "ConstMul":
            c = g.constants[params["c"]]
            require(len(ins) == 1 and outs == ins and len(c) in (1, ins[0][-1]), f"{where}: y = x")
        elif op == "Rotate":
            x, pos = ins
            require(params["rope"] == 0 and kind(inputs[1]) == "P", f"{where}: positions are Public")
            require(pos == [x[0]] and x[-1] % 2 == 0 and outs == [x], f"{where}: x [n, ..., 2m], pos [n]")
        elif op == "SumSquares":
            axis = params["axis"]
            require(outs == [ins[0][:axis] + ins[0][axis + 1:] or [1]], f"{where}: y is x without the axis")
        elif op == "Reshape":
            require(len(ins) == 1 and numel(ins[0]) == numel(outs[0]), f"{where}: equal element counts")
        elif op == "Transpose":
            perm = params["perm"]
            require(sorted(perm) == list(range(len(ins[0]))) and outs == [[ins[0][i] for i in perm]],
                    f"{where}: y's axis i is x's axis perm[i]")
        elif op == "Expand":
            axis, n_ = params["axis"], params["len"]
            require(axis <= len(ins[0]) and n_ >= 1 and outs == [ins[0][:axis] + [n_] + ins[0][axis:]],
                    f"{where}: an axis of len inserted before axis")
        elif op == "MatMulAct":
            x, y = ins
            out = outs[0]
            if params["transpose_rhs"]:
                ok = x[:-2] == y[:-2] and x[-1] == y[-1] and out == x[:-1] + [y[-2]]
            else:
                ok = x[:-2] == y[:-2] and x[-1] == y[-2] and out == x[:-1] + [y[-1]]
            require(ok, f"{where}: a [..., m, k] by b [..., k, n] (or [..., n, k]) gives [..., m, n]")
        elif op == "Mul":
            x, y = ins
            if params["broadcast"] == "Trailing":
                ok = y == x[:len(y)]
            else:
                require(params["broadcast"] == "Leading", f"{where}: a broadcast")
                ok = y == x[-1:]
            require(ok and outs == [x], f"{where}: the {params['broadcast']} broadcast, y = a")
        elif op == "OneHot":
            require(outs == [ins[0] + [params["vocab"]]] and kind(outputs[0][0]) == "A", f"{where}: h is t with vocab")
        elif op == "Requant":
            classes = g.shift_classes[params["shift"]]
            require(len(ins) == (2 if len(classes) > 1 else 1) and outs == [ins[0]], f"{where}: y = x")
            if len(classes) > 1:
                require(ins[1] == [ins[0][-1]] and kind(inputs[1]) == "W", f"{where}: c [x's last axis]")
            require(params["lo"] <= params["hi"], f"{where}: lo ≤ hi")
        elif op == "TableMap":
            (table,) = g.tables[params["table"]]
            require(table == "Silu", f"{where}: S3, a Silu table")
            classes = g.tables[params["table"]]["Silu"]["r_by_class"]
            require(len(ins) == (2 if len(classes) > 1 else 1) and outs == [ins[0]], f"{where}: y = x")
            if len(classes) > 1:
                require(ins[1] == [ins[0][-1]] and kind(inputs[1]) == "W", f"{where}: c [x's last axis]")
            require(outputs[0][2] == lookup(params["table"]), f"{where}: y is LookupOutput {{ table }}")
        elif op == "SoftmaxRow":
            (s,) = ins
            require(len(s) >= 2 and s[-1] == s[-2] and outs == [s, s[:-1]], f"{where}: s [..., n, n], e = s, R")
            require("Exp" in g.tables[params["exp_table"]], f"{where}: an Exp table")
            require(outputs[0][2] == lookup(params["exp_table"]) and outputs[1][2] == "A", f"{where}: kinds")
        elif op == "RsqrtRow":
            require(outs == ins and outputs[0][2] == "A" and params["dim"] >= 1 and params["kappa"] <= 31,
                    f"{where}: ρ = u, advice")
        else:
            raise SystemExit(f"lowering oracle: {where}: RFC-0044 names no {op}")
    # Every weight once, each read by the MatMulWeight its WeightRef names.
    readers = {}
    for node, op, params, inputs, outputs, at in g.nodes:
        if op == "MatMulWeight":
            readers[inputs[1]] = tuple(params["weight"])
    require(len(g.weights) == len({(ty, l) for ty, l, _ in g.weights}), "every weight once")
    require(all(readers.get(t) == (ty, l) for ty, l, t in g.weights), "each weight read by its matmul")
    require(len(g.weights) == 2 + 7 * public.arch["n_layers"], "Embed, LmHead and seven per layer")
    # RFC-0037 'Layers and kinds': every layer the same nodes, in the same order, with equal
    # operations up to the layer's own weights, constants, shift classes and tables, and equal
    # shapes; each input the same position's output in its own layer, the layer's checkpoint, or
    # a graph input.
    roles = {t: role for t, role, *_ in g.inputs}

    def template(layer):
        first, last = boundaries[layer] + 1, boundaries[layer + 1]
        checkpoint = g.nodes[boundaries[layer]][4][0][0]
        position = {}
        rows = []
        for node, op, params, inputs, outputs, at in g.nodes[first:last + 1]:
            for t, _, _ in outputs:
                position[t] = node - first
            refs = []
            for t in inputs:
                if t in position:
                    refs.append(("node", position[t]))
                elif t == checkpoint:
                    refs.append(("checkpoint",))
                else:
                    require(g.producer[t] is None, f"node {node} reads a tensor of another layer")
                    role = roles[t]
                    require(role == "positions" or role.endswith(f" {layer}"),
                            f"node {node} reads {role}, not its layer's own input")
                    refs.append(("input", role.rsplit(" ", 1)[0], g.tensors[t][0], g.tensors[t][1]))
            kinds = [k if not k.startswith("L(") else "L" for _, _, k in outputs]
            # Equal up to the layer's own weights, constants, shift classes and tables.
            same = {k: v for k, v in params.items() if k not in ("c", "shift", "table", "exp_table")}
            if "weight" in same:
                same["weight"] = same["weight"][0]
            rows.append((op, same, refs, [s for _, s, _ in outputs], kinds))
        return rows
    first = template(0)
    for layer in range(1, public.arch["n_layers"]):
        require(template(layer) == first, f"RFC-0037: layer {layer} is layer 0's template")


# ----- Output ---------------------------------------------------------------------------------------

LENGTHS = (1, 4, 5)  # and each model's max_context


def table_entry(t, kind):
    """A table by id: its kind and its log2_rows, ⌈log2 T⌉ for the kind's T rows (RFC-0008
    'Table specifications'). The tables oracle counts the rows, and refuses parameters outside
    RFC-0008's, an Exp zero tail past key 2^24 among them."""
    rows, _, _, _ = tables.table(kind)
    log2_rows = (rows - 1).bit_length()
    require(log2_rows <= 32, f"table {t}: at most 2^32 rows")
    return [t, kind, log2_rows]


def forms(public, g):
    """The forms of RFC-0044's procedures this graph takes ('Testing Strategy', 'Every form')."""
    layers = range(public.arch["n_layers"])
    found = {public.profile.lower()}
    for key, (mode, value) in public.rescale.items():
        if key[0] != "Layer" or key[2] not in WEIGHT_BEFORE:
            continue
        if mode == "affine":
            found.add(f"affine {key[2]}")
        else:
            found.add("pot of one class" if len(value) == 1 else "pot of several classes")
        if key[2] == "Silu":
            found.add("Silu of one class" if len(value) == 1 else "Silu of several classes")
    for site in ("AttnResidual", "MlpResidual"):
        modes = {public.site(("Layer", l, site))[0] for l in layers}
        one = all(len(public.shifts(("Layer", l, site))) == 1 for l in layers)
        at = "an affine site" if modes == {"affine"} else (
            "a pot site of one class" if one else "a pot site of several classes")
        if public.fused(site):
            found |= {"fused key", f"fused key at {at}"}
        else:
            found |= {"two steps", f"two steps at {at}"}
            if any(public.shifts(("Layer", l, site)) == [r] for l in layers for r in range(1, 63)):
                found.add("two steps in a layer of one shift at most 62")
    expanded = any(op == "Expand" for _, op, _, _, _, _ in g.nodes)
    found.add("heads expanded" if expanded else "heads not expanded")
    exp_ids = [p["exp_table"] for _, op, p, _, _, _ in g.nodes if op == "SoftmaxRow"]
    silu_ids = [p["table"] for _, op, p, _, _, _ in g.nodes if op == "TableMap"]
    if len(set(exp_ids)) < len(exp_ids):
        found.add("shared Exp table")
    if len(set(silu_ids)) < len(silu_ids):
        found.add("shared Silu table")
    w = bitlen(128 * 255 * public.arch["d_head"])
    for kind in g.tables:
        if "Exp" in kind:
            e = kind["Exp"]
            rows, (num, den) = 2 ** e["key_bits"], e["scale"]
            if e["key_bits"] > w:
                found.add("Exp wider than the spread")
            if tables.exp_tail(rows, e["sigma"], num, den) == rows:
                found.add("Exp zero at its last key before its zero tail")
    return found


# Every form of RFC-0044's procedures ('Testing Strategy', 'Every form').
FORMS = {
    "servable", "minimal", "pot of one class", "pot of several classes",
    "affine RopeQ", "affine RopeK", "affine V", "affine AttnResidual", "affine Up", "affine MlpResidual",
    "Silu of one class", "Silu of several classes",
    "fused key", "fused key at an affine site", "fused key at a pot site of one class",
    "two steps", "two steps at an affine site", "two steps at a pot site of one class",
    "two steps at a pot site of several classes", "two steps in a layer of one shift at most 62",
    "heads expanded", "heads not expanded", "shared Exp table", "shared Silu table",
    "Exp wider than the spread", "Exp zero at its last key before its zero tail",
}

# RFC-0044 'Testing Strategy': the forms the conformance models cover.
COVERED = {"servable", "minimal", "pot of one class", "pot of several classes", "affine RopeQ",
           "affine RopeK", "affine AttnResidual", "affine Up", "fused key", "two steps"}


def case_of(name, file, public, n, variant):
    g, boundaries = lower(public, n)
    check_graph(public, g, boundaries)
    routes = {site: "fused key" if public.fused(site) else "two steps"
              for site in ("AttnResidual", "MlpResidual")}
    found = forms(public, g)
    case = {"model": name, "file": file}
    if variant is not None:
        case["variant"] = variant
    case.update({
        "profile": public.profile,
        "n": n,
        "routes": routes,
        "forms": sorted(found),
        "tensors": len(g.tensors),
        "checkpoints": [g.nodes[i][4][0][0] for i in boundaries],
        "inputs": g.inputs,
        "nodes": g.nodes,
        "constants": g.constants,
        "shift_classes": g.shift_classes,
        "weights": g.weights,
        "tables": [table_entry(t, kind) for t, kind in enumerate(g.tables)],
    })
    return case, found


# ----- Variants: the forms the conformance models miss ------------------------------------------------
#
# RFC-0044 'Testing Strategy': "The oracle's own test adds any form they miss." A variant is a
# conformance model with its public part edited, every layer still rescaled alike (RFC-0038):
#
# - `n_kv_heads`: the architecture's;
# - `rescale`: [layer, site, "pot", shifts] or [layer, site, "affine", r], the site's entry in
#   shift_class_sets or affine_sites at that layer, replacing the one it had;
# - `score_scales`: [layer, num, den], the layer's score scale.
#
# A variant is a public part and nothing more: the lowering reads nothing else.

def variants(publics):
    """The variants, with each edit's values, from the conformance models' public parts."""
    servable = publics["tiny-2-servable"]["quant"]["public"]
    scale0 = next(s for l, s in servable["score_scales"] if l == 0)
    return [
        ("tiny-2-servable", {
            # Affine V and MlpResidual, the latter at shift 63, so two steps at an affine site;
            # AttnResidual of one class whose layer 0 shift ≤ 62 still takes two steps, as layer
            # 1's 63 needs (RFC-0038 'The residual's route'); Silu of one class, read without a
            # class vector; and layer 1 with layer 0's score scale, which shares its Exp table.
            "rescale": [[0, "V", "affine", 20], [1, "V", "affine", 20],
                        [0, "AttnResidual", "pot", [5]], [1, "AttnResidual", "pot", [63]],
                        [0, "Silu", "pot", [6]], [1, "Silu", "pot", [6]],
                        [0, "MlpResidual", "affine", 63], [1, "MlpResidual", "affine", 63]],
            "score_scales": [[1, scale0["num"], scale0["den"]]],
        }, {"affine V", "affine MlpResidual", "Silu of one class", "two steps at an affine site",
            "two steps at a pot site of one class", "two steps in a layer of one shift at most 62",
            "shared Exp table"}),
        ("tiny-2-minimal", {
            # As many KV heads as query heads, so no Expand; RopeK of one class, so that K needs
            # no classes at its new width; layer 0's scale needs a 20-bit Exp table, since its
            # 19-bit table's last key is 11; layer 1's 19-bit table maps its last key to 0 by
            # rounding before its zero tail starts (RFC-0030: the check reads the value).
            "n_kv_heads": 4,
            "rescale": [[0, "RopeK", "pot", [22]], [1, "RopeK", "pot", [22]]],
            "score_scales": [[0, 1, 60000], [1, 1, 43330]],
        }, {"heads not expanded", "Exp wider than the spread",
            "Exp zero at its last key before its zero tail"}),
    ]


def point_value(layer, site):
    return {"Layer": {"layer": layer, "site": {site: None}}}


def point_order(entry):
    """QuantPoint's order, as declared (03 §2): Embed, the layers' sites, FinalNorm."""
    key = point_key(entry[0])
    if key[0] == "Embed":
        return (0,)
    if key[0] == "Layer":
        return (1, key[1], SITES.index(key[2]))
    return (2,)


def edit(public_part, variant):
    """The public part with a variant's edits."""
    out = json.loads(json.dumps(public_part))
    if "n_kv_heads" in variant:
        out["arch"]["n_kv_heads"] = variant["n_kv_heads"]
    quant = out["quant"]["public"]
    for layer, site, mode, value in variant.get("rescale", []):
        point = point_value(layer, site)
        quant["shift_class_sets"] = [e for e in quant["shift_class_sets"] if e[0] != point]
        quant["affine_sites"] = [e for e in quant["affine_sites"] if e[0] != point]
        require(mode in ("pot", "affine"), f"a rescale is pot or affine: {mode}")
        (quant["shift_class_sets"] if mode == "pot" else quant["affine_sites"]).append([point, value])
    quant["shift_class_sets"].sort(key=point_order)
    quant["affine_sites"].sort(key=point_order)
    for layer, num, den in variant.get("score_scales", []):
        require(any(l == layer for l, _ in quant["score_scales"]), f"layer {layer} has a score scale")
        quant["score_scales"] = [[l, {"num": num, "den": den} if l == layer else s]
                                 for l, s in quant["score_scales"]]
    return out


RECORDS = ("inputs", "nodes", "weights", "tables")


def dumps(value):
    return json.dumps(value, separators=(", ", ": "))


def render(document):
    """The document with one record per line: an input, a node, a weight or a table."""
    cases = []
    for case in document["cases"]:
        fields = []
        for key, value in case.items():
            if key in RECORDS:
                body = ",\n".join("    " + dumps(record) for record in value)
                fields.append(f"   {dumps(key)}: [\n{body}\n   ]")
            else:
                fields.append(f"   {dumps(key)}: {dumps(value)}")
        cases.append("  {\n" + ",\n".join(fields) + "\n  }")
    head = [f" {dumps(key)}: {dumps(value)}" for key, value in document.items() if key != "cases"]
    return "{\n" + ",\n".join(head + [' "cases": [\n' + ",\n".join(cases) + "\n ]"]) + "\n}\n"


def main():
    cases = []
    publics = {}
    covered = set()
    for spec in models.MODELS:
        model, sequences = models.build(spec)
        chooser = models.Run(model, spec["plan"])
        chooser.forward(sequences)
        d = models.integer_model(spec, model, chooser.c)
        # The lowering reads the public part only: drop the rest before it can be read.
        publics[spec["name"]] = {"arch": d["arch"], "profile": d["profile"],
                                 "quant": {"public": d["quant"]["public"],
                                           "silu_in_exponents": d["quant"]["silu_in_exponents"]}}
        public = Public(publics[spec["name"]])
        for n in LENGTHS + (public.arch["max_context"],):
            case, found = case_of(spec["name"], f"{spec['name']}.d", public, n, None)
            cases.append(case)
            covered |= found
    missing = COVERED - covered
    require(not missing, f"the conformance models miss {sorted(missing)}, which RFC-0044 says they cover")
    seen = set(covered)
    for name, variant, meant in variants(publics):
        public = Public(edit(publics[name], variant))
        for n in LENGTHS + (public.arch["max_context"],):
            case, found = case_of(f"{name} variant", f"{name}.d", public, n, variant)
            require(meant <= found, f"the variant of {name} misses {sorted(meant - found)}")
            cases.append(case)
            seen |= found
    require(seen == FORMS, f"forms missed: {sorted(FORMS - seen)}; unnamed: {sorted(seen - FORMS)}")
    document = {
        "description": "The Llama lowering of RFC-0044 for each conformance model of "
        "spec/vectors/models/, and for variants of two of them that take the forms the models "
        "miss, at each length n: every graph input, node, constant, shift-class set, weight and "
        "table, each by its id in creation order. Written by generate.py from RFC-0044 alone; "
        "README.md gives the format.",
        "cases": cases,
    }
    text = render(document)
    require(json.loads(text) == json.loads(json.dumps(document)), "the file reads back as the graphs")
    (HERE / "lowering.json").write_text(text, encoding="utf-8")


if __name__ == "__main__":
    main()
