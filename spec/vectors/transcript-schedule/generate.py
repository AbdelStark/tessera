#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0 OR MIT
"""Reference vectors for the transcript schedule of tessera-protocol/1 (RFC-0014 'Transcript
schedule' and 'Grinding'): for each case, the full ordered list of the labels one proof unit
absorbs and squeezes under, and, for a fixed test seed, every challenge, seed, column index and
proof-of-work nonce drawn at each label.

This script is the independent oracle. It expands the labels from RFC-0014's table as RFC-0010
('Tree levels', 'Batching within a stage', 'Sumcheck messages'), RFC-0035, RFC-0036 ('What the
transcript absorbs'), RFC-0037 ('Messages and transcript'), RFC-0045 ('The split') and RFC-0049
('The schedule') amend it, and runs the transcript of ../transcript/generate.py, whose BLAKE3 and
H(domain, bytes) are those of ../hash/generate.py, checked against the official test vectors
before anything is written. It derives the conformance models' layer point and weight matrices
from RFC-0040 with ../weights/generate.py's reader and layouts. It shares no code with crates/
or verifier/, which check its output.

Every conformance-model dimension is derived independently from the public lowering and
canonical gadget catalog below, including all lookup families, Softmax forests, OneHot and
gadget relations. No sizes are copied from Rust output. See README.md.

Usage: python3 spec/vectors/transcript-schedule/generate.py
(rewrites schedule.json)

Inputs come from SplitMix64 with a fixed seed, so any implementation can regenerate them.
"""

import copy
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


transcript = _oracle("transcript")  # RFC-0004's transcript and TCE(SessionContext)
hashing = transcript.oracle  # BLAKE3 and H, from ../hash/generate.py
weights = _oracle("weights")  # RFC-0040's matrices and the models' reader
lowering = _oracle("lowering")  # RFC-0044's public graph
advice = _oracle("advice")  # RFC-0050's complete allocation

H, P, require = hashing.H, hashing.P, hashing.require
u64, Transcript = transcript.u64, transcript.Transcript
SEED = 0x7E55E7A_0125_0014  # fixed; changing it changes every vector

# RFC-0014 'Parameter sets': the modes, the advice's log2 k and inverse rate, τ_A, τ_W, g_seed and
# g_query. Every weight matrix is encoded at rate 1/4 (RFC-0012 'Layout').
PARAMETER_SETS = {
    "live-64/1": {"modes": ["Live"], "log2_k": 22, "log2_inv_rate": 1, "tau_a": 155, "tau_w": 97,
                  "g_seed": 0, "g_query": 0},
    "ni-100/1": {"modes": ["NonInteractive"], "log2_k": 22, "log2_inv_rate": 1, "tau_a": 193, "tau_w": 122,
                 "g_seed": 20, "g_query": 20},
    "test-small/1": {"modes": ["Live", "NonInteractive"], "log2_k": 10, "log2_inv_rate": 1, "tau_a": 8,
                     "tau_w": 8, "g_seed": 0, "g_query": 0},
}
WEIGHT_LOG2_INV_RATE = 2

# RFC-0031's WeightMatrix, whose names a label's {T} spells (RFC-0036 'Matrix ids').
MATRICES = weights.MATRICES
require(MATRICES == hashing.WEIGHT_MATRICES, "the hash oracle's WeightMatrix order")


# ----- The schedule (RFC-0014 'Transcript schedule') -----------------------------------------------


def expand(sizes, params, mode):
    """The steps of one proof unit of `sizes`, in `mode`, at `params`: each a label and an
    operation, "absorb", "digest" (the non-interactive seed), "fp2" (`count` challenges in Fp2,
    one after the other: a challenge or a proximity point), "indices" (`count` of `log_n` bits)
    or "grind" (a proof of work of `bits`, whose nonce is absorbed)."""
    steps = []

    def absorb(label):
        steps.append({"label": label, "op": "absorb"})

    def fp2(label, count=1):
        steps.append({"label": label, "op": "fp2", "count": count})

    # Row 2: TCE(Round1).
    absorb("round1")
    # Row 3: the coin-tossed seed, or the seed's proof of work and the squeezed seed (RFC-0036).
    if mode == "Live":
        absorb("seed")
    else:
        steps.append({"label": "seed-pow", "op": "grind", "phase": "seed", "bits": params["g_seed"]})
        steps.append({"label": "seed", "op": "digest"})
    # RFC-0049: right after the seed, each matrix with committed rows, in RFC-0040's order, draws
    # its proximity point, one coordinate per bit of its committed rows' positions.
    for point in sizes["proximity"]:
        fp2(f"pcs/weights/{point['matrix']}/xi", point["vars"])
    # Row 4: α and β, then the fraction trees' roots, absorbed even when empty.
    fp2("logup/alpha")
    fp2("logup/beta")
    absorb("logup/roots")
    # Row 5 (RFC-0010 'Tree levels'): a tree of height h joins at level H - h; a tree of height 0
    # joins none.
    trees = sizes["trees"]
    tallest = max((t["height"] for t in trees), default=0)
    for level in range(tallest):
        joining = [t for t in trees if tallest - t["height"] == level]
        active = [t for t in trees if tallest - t["height"] <= level]
        # The forests of more than one tree that join here: the root point, s_max coordinates.
        for c in range(max((t.get("trees_log", 0) for t in joining), default=0)):
            fp2(f"tree/{level}/root/{c}")
        fp2(f"tree/{level}/gamma")
        if any(t["kind"] == "fraction" for t in active):
            fp2(f"tree/{level}/lambda")
        # A tree at its own level t is an instance on s + t variables; the level's rounds are its
        # largest instance's.
        rounds = max((t.get("trees_log", 0) + level - (tallest - t["height"]) for t in active), default=0)
        for j in range(rounds):
            absorb(f"tree/{level}/round/{j}")
            fp2(f"tree/{level}/r/{j}")
        absorb(f"tree/{level}/evals")
        fp2(f"tree/{level}/line")
    # The lookup trees' value evaluations, even when empty, then the leaf stage's split values,
    # stage 0's (RFC-0045 'The split').
    absorb("logup/values")
    absorb("stage/0/split")
    # Row 6 (RFC-0037 'Messages and transcript'): stage 1 draws the layer point, then any stage
    # its zero-check point, then γ; the rounds, the evaluations, the splits even when empty.
    for k, stage in enumerate(sizes["stages"], start=1):
        if k == 1:
            for c in range(sizes["layer_vars"]):
                fp2(f"stage/1/layer/{c}")
        for c in range(stage["zero_vars"]):
            fp2(f"stage/{k}/zero/{c}")
        fp2(f"stage/{k}/gamma")
        for j in range(stage["rounds"]):
            absorb(f"stage/{k}/round/{j}")
            fp2(f"stage/{k}/r/{j}")
        absorb(f"stage/{k}/evals")
        absorb(f"stage/{k}/split")
    # Row 7: tessera-protocol/1 has no blinding claim, so the opening's challenges are squeezed
    # directly; the weights' points are RFC-0049's, in row 3.
    fp2("pcs/advice/gamma")
    fp2("pcs/advice/xi", sizes["advice_vars"])
    for codeword in sizes["opened"]:
        fp2(f"pcs/weights/{codeword['matrix']}/gamma")
    # Row 8, then row 9.
    absorb("pcs/advice/combined")
    for codeword in sizes["opened"]:
        absorb(f"pcs/weights/{codeword['matrix']}/combined")
    absorb("scoring")
    # Row 10: the verifier's queries, or the queries' proof of work and the squeezed indices.
    if mode == "Live":
        absorb("queries")
    else:
        steps.append({"label": "query-pow", "op": "grind", "phase": "query", "bits": params["g_query"]})
        steps.append({"label": "pcs/advice/queries", "op": "indices",
                      "log_n": params["log2_k"] + params["log2_inv_rate"], "count": params["tau_a"]})
        for codeword in sizes["opened"]:
            steps.append({"label": f"pcs/weights/{codeword['matrix']}/queries", "op": "indices",
                          "log_n": log2(codeword["codeword"]), "count": params["tau_w"]})
    labels = [s["label"] for s in steps]
    require(len(set(labels)) == len(labels), "INV-FS-4: no label twice")
    return steps


def log2(x):
    require(x >= 1 and x & (x - 1) == 0, f"{x} is a power of two")
    return x.bit_length() - 1


# ----- Running a schedule ------------------------------------------------------------------------


def vec_fp2(rng):
    """A stand-in message: TCE(Vec<Fp2>) of one element."""
    return u64(1) + u64(hashing.canonical(rng)) + u64(hashing.canonical(rng))


def column_queries(rng, sizes, params):
    """A live verifier's queries, TCE(ColumnQueries) (RFC-0036): τ_A advice indices below n_A, then
    τ_W below each opened matrix's n_T, with the matrix's id, every index a u32."""
    n_a = 1 << (params["log2_k"] + params["log2_inv_rate"])

    def indices(count, n):
        return u64(count) + b"".join((next(rng) % n).to_bytes(4, "little") for _ in range(count))

    out = indices(params["tau_a"], n_a) + u64(len(sizes["opened"]))
    for codeword in sizes["opened"]:
        out += bytes([MATRICES.index(codeword["matrix"])]) + indices(params["tau_w"], codeword["codeword"])
    return out


def run(steps, context, sizes, params, rng):
    """Plays `steps` on the transcript of `context`, absorbing stand-in messages, until a step whose
    size is unknown. Returns the steps with what each absorbed or drew, the number played, and the
    case's coin toss in a live session."""
    t = Transcript(transcript.session_context(context))
    played, coin = [], None
    for step in steps:
        record = dict(step)
        op, label = step["op"], step["label"]
        if op == "fp2" and step["count"] is None:
            break
        if op == "absorb":
            if label == "seed":
                share_v, share_p = hashing.digest_from(rng), hashing.digest_from(rng)
                coin = {"verifier_share": share_v.hex(), "prover_share": share_p.hex()}
                tce = H("coin-toss", share_v + share_p)
            elif label == "queries":
                tce = column_queries(rng, sizes, params)
            else:
                tce = vec_fp2(rng)
            t.absorb(label, tce)
            record["tce"] = tce.hex()
        elif op == "digest":
            record["out"] = t.squeeze(label).hex()
        elif op == "fp2":
            values = []
            for _ in range(step["count"]):
                value, _ = t.challenge_fp2(label)
                values.append([str(c) for c in value])
            record["values"] = values
        elif op == "indices":
            record["values"] = [str(i) for i in t.challenge_indices(label, step["log_n"], step["count"])]
        elif op == "grind":
            record["nonce"] = str(t.grind(label, step["bits"]))
        else:
            raise SystemExit(f"generate.py: unknown op {op}")
        played.append(record)
    rest = [dict(s) for s in steps[len(played):]]
    return played + rest, len(played), t, coin


def case(name, sizes, params_id, mode, rng, pow_bits=None, provenance=None):
    params = dict(PARAMETER_SETS[params_id])
    require(mode in params["modes"], f"{params_id} allows {mode}")
    if pow_bits is not None:
        params["g_seed"], params["g_query"] = pow_bits
    context = {
        "protocol": "tessera-protocol/1",
        "mode": mode,
        "params": params_id,
        "statement_digest": hashing.digest_from(rng).hex(),
        "session_id": hashing.digest_from(rng)[:16].hex(),
    }
    steps = expand(sizes, params, mode)
    steps, replayed, t, coin = run(steps, context, sizes, params, rng)
    out = {
        "name": name,
        "mode": mode,
        "params": params_id,
        "seed_pow": params["g_seed"],
        "query_pow": params["g_query"],
        "sizes": sizes,
    }
    if provenance:
        out["provenance"] = provenance
    out.update({
        "context": context,
        "context_tce": transcript.session_context(context).hex(),
    })
    if coin:
        out["coin_toss"] = coin
    out.update({
        "replayed": replayed,
        "state": t.state.hex(),
        "counter": t.counter,
        "steps": steps,
    })
    return out


# ----- Synthetic sizes ---------------------------------------------------------------------------


def fraction(height):
    return {"kind": "fraction", "height": height}


def forest(trees_log, height):
    require(height >= 1, "a product forest's trees have a level")
    return {"kind": "forest", "trees_log": trees_log, "height": height}


def stage(rounds, zero_vars):
    require(zero_vars <= rounds, "a zero-check is one of the stage's instances")
    return {"rounds": rounds, "zero_vars": zero_vars}


# Every rule of the schedule: forests that join at levels 0 and 1, a single tree, a fraction tree
# that joins at the last level and one that joins none; a stage of no round and a zero-check
# point outside stage 1; a matrix with committed rows the proof does not open, and a point of no
# coordinate.
RICH = {
    "trees": [forest(1, 3), fraction(1), fraction(0), forest(2, 2), forest(0, 3)],
    "layer_vars": 2,
    "stages": [stage(3, 2), stage(0, 0), stage(2, 1)],
    "advice_vars": 3,
    "proximity": [{"matrix": "Q", "vars": 2}, {"matrix": "Down", "vars": 0}, {"matrix": "ChannelParams", "vars": 1}],
    "opened": [{"matrix": "Q", "codeword": 1 << 10}, {"matrix": "ChannelParams", "codeword": 1 << 9}],
}

BARE = {"trees": [], "layer_vars": 0, "stages": [], "advice_vars": 0, "proximity": [], "opened": []}

# Only fraction trees, of mixed heights: a level with λ and many rounds; and every tree of height 0.
FRACTIONS = {
    "trees": [fraction(4), fraction(2), fraction(0), fraction(4), fraction(1)],
    "layer_vars": 1,
    "stages": [stage(4, 4)],
    "advice_vars": 2,
    "proximity": [{"matrix": "Embed", "vars": 1}, {"matrix": "LmHead", "vars": 2}],
    "opened": [{"matrix": "Embed", "codeword": 1 << 8}, {"matrix": "LmHead", "codeword": 1 << 11}],
}

FLAT = {**BARE, "trees": [fraction(0), fraction(0)]}

# Codewords of live-64/1's weights: k_T = 2^20 at rate 1/4.
LIVE_64 = {**RICH, "opened": [{"matrix": "Q", "codeword": 1 << 22}, {"matrix": "ChannelParams", "codeword": 1 << 22}]}


# ----- The conformance models (07 §5) -------------------------------------------------------------

def model_geometry(case):
    """Walk the public expression graph, rather than a Rust-produced schedule.

    A nonlinear definition costs one stage; linear substitutions cost none. Gadget
    zero-checks run in stage 1. Root comparisons read u, hence its SumSquares
    dependency in stage 2; the top-input lookup also reads it directly in stage 1.
    Softmax's masked key/leaf relation precedes the score's MatMulAct (RFC-0030).
    """
    nodes = {node[0]: node for node in case["nodes"]}
    tensors = {item[0]: (item[2], item[3]) for item in case["inputs"]}
    producers = {}
    for node in nodes.values():
        for tid, shape, kind in node[4]:
            tensors[tid] = (shape, kind)
            producers[tid] = node
    bounds = [producers[t][0] for t in case["checkpoints"]]
    layer_bits = (len(bounds) - 2).bit_length()

    def stacked(node):
        return layer_bits if bounds[0] < node[0] <= bounds[-1] else 0

    def variables(shape):
        return sum((d - 1).bit_length() for d in shape)

    def domain(tensor):
        return variables(tensors[tensor][0])

    stages, seen = {}, set()

    def record(index, count, zero=False):
        entry = stages.setdefault(index, [0, 0])
        entry[0] = max(entry[0], count)
        if zero:
            entry[1] = max(entry[1], count)

    def follow(tensor, index):
        if (tensor, index) in seen or tensors[tensor][1] != "v":
            return
        seen.add((tensor, index))
        node = producers[tensor]
        _, op, params, inputs, _, _ = node
        if op == "MatMulWeight":
            own = (tensors[inputs[0]][0][-1] - 1).bit_length()
        elif op == "MatMulAct":
            shape = tensors[inputs[0]][0]
            own = variables(shape[:-2]) + (shape[-1] - 1).bit_length()
        elif op in ("SumSquares", "Rotate"):
            own = domain(inputs[0])
        elif op == "Mul" or (op == "ConstMul" and len(case["constants"][params["c"]]) != 1):
            own = domain(tensor)
        else:
            require(op in ("Add", "Sub", "Reshape", "Transpose", "Expand", "Sum", "ConstMul"),
                    "a supported linear substitution")
            for operand in inputs:
                follow(operand, index)
            return
        record(index, own + stacked(node))
        for operand in inputs:
            follow(operand, index + 1)

    gadgets = {"Requant", "TableMap", "Bitwise", "RsqrtRow", "SoftmaxRow", "OneHot"}
    for node in nodes.values():
        _, op, _, inputs, outputs, _ = node
        if op in gadgets:
            for tensor in inputs:
                follow(tensor, 2 if op == "SoftmaxRow" else 1)
            if op == "OneHot":
                record(1, domain(outputs[0][0]) + stacked(node), True)
            if op == "RsqrtRow":
                record(1, domain(outputs[0][0]) + stacked(node), True)
                # RFC-0050's simplified u binding retains residual feature/time/layer zeros.
                residual = producers[inputs[0]][3][0]
                record(1, domain(residual) + stacked(node), True)
                follow(inputs[0], 2)
            if op == "SoftmaxRow":
                record(1, domain(inputs[0]) + stacked(node))
                record(1, domain(outputs[1][0]) + stacked(node), True)
        elif op not in ("MatMulWeight", "MatMulAct"):
            for tensor, _, kind in outputs:
                if kind == "A":
                    record(1, domain(tensor) + stacked(node), True)
                    for operand in inputs:
                        follow(operand, 2)
    follow(case["nodes"][-1][4][0][0], 1)

    # RFC-0045: real layers' per-axis-padded blocks, then the combined column's tail.
    tables, choices = advice.catalog(case)
    counts = {}
    for choice in choices:
        node = nodes[choice["node"]]
        for table in choice["tables"]:
            tensor = (node[4][1][0] if node[1] == "SoftmaxRow"
                      and table != node[2]["exp_table"] else node[3][0])
            counts[table] = counts.get(table, 0) + (1 << domain(tensor))
    trees = [tree for tid, count in sorted(counts.items())
             for tree in (fraction((count - 1).bit_length()), fraction(tables[tid][2]))]
    # One forest per template Softmax site, including padded layer/head/query axes.
    for node in nodes.values():
        if node[1] == "SoftmaxRow" and bounds[0] < node[0] <= bounds[1]:
            heads, queries, keys = tensors[node[3][0]][0]
            if keys > 1:  # One key has no product-tree reduction level (RFC-0010).
                trees.append(forest(layer_bits + variables([heads, queries]), (keys - 1).bit_length()))
    return {"trees": trees, "stages": [stage(*stages[k]) for k in range(1, max(stages) + 1)],
            "lookup_entries": [[tid, count] for tid, count in sorted(counts.items())]}



def matrices(arch, channels):
    """Each matrix with committed rows at test-small/1, in RFC-0040's order: its committed rows m_T
    and message width k_T (RFC-0040 'The matrices'), with ../weights/generate.py's layout rules."""
    k_w = 1 << weights.LAYOUT["log2_k"]
    out = []
    for ty, (rows_out, inputs, layers) in weights.shapes_of(arch).items():
        out_pad, in_pad = weights.pad(rows_out), weights.pad(inputs)
        k = max(in_pad, min(k_w, out_pad * in_pad))
        a = weights.log2(k // in_pad)
        # A row (l, o_hi) is committed when l < layers and it holds an output below `out`.
        committed = layers * -(-rows_out // (1 << a))
        out.append((ty, committed, k))
    widths = {weights.WEIGHT_TYPES.index(ty): shape[0] for ty, shape in weights.shapes_of(arch).items()}
    layout, committed, _ = weights.channel_matrix(channels, widths, arch["n_layers"], k_w)
    out.append(("ChannelParams", len(committed), 1 << layout["log2_k"]))
    return [(ty, m, k) for ty, m, k in out if m > 0]


def conformance_sizes(name, index, n=None, query_block=512, padded=False):
    """A model's sizes and lookup counts, all from public data; λ = ⌈log2 L⌉; the proximity point of each
    matrix with committed rows, ⌈log2 m_T⌉ coordinates (RFC-0049); the opened matrices, every one
    with committed rows, each read by a claim of the schedule: the nine weight types by their
    MatMulWeight instances, ChannelParams by the sites that read its classes or multipliers."""
    value, _ = weights.read_model(name, index)
    if padded:
        value = copy.deepcopy(value)
        value["arch"].update(n_layers=3)
        q = value["quant"]["public"]
        for field in ("act_exponents", "shift_class_sets", "affine_sites"):
            for point, data in list(q[field]):
                if point.get("Layer", {}).get("layer") == 0:
                    point = copy.deepcopy(point)
                    point["Layer"]["layer"] = 2
                    q[field].append([point, data])
        q["score_scales"].append([2, q["score_scales"][0][1]])
        value["quant"]["silu_in_exponents"].append([2, value["quant"]["silu_in_exponents"][0][1]])
    arch = value["arch"]
    public = {"arch": arch, "profile": value["profile"],
              "quant": {"public": value["quant"]["public"],
                        "silu_in_exponents": value["quant"]["silu_in_exponents"]}}
    entry = next(m for m in index["models"] if m["name"] == name)
    n = len(entry["sequences"][0]["tokens"]) if n is None else n
    graph, _ = lowering.case_of(name, entry["file"], lowering.Public(public), n, None)
    # Public class/multiplier inputs determine rows; their values do not affect geometry.
    channels = {}
    for _, role, shape, *_ in graph["inputs"]:
        if role.startswith(("classes ", "multipliers ")):
            _, ty, layer = role.split()
            channels[weights.WEIGHT_TYPES.index(ty), int(layer)] = [0] * shape[0]
    committed = matrices(arch, channels)
    layout = advice.model_layout(graph, 10, query_block, 1)  # test-small/1, one segment
    geometry = model_geometry(graph)
    return {
        "trees": geometry["trees"],
        "layer_vars": (arch["n_layers"] - 1).bit_length(),
        "stages": geometry["stages"],
        "advice_vars": layout["nu"] - 10,
        "proximity": [{"matrix": ty, "vars": (m - 1).bit_length()} for ty, m, _ in committed],
        "opened": [{"matrix": ty, "codeword": k << WEIGHT_LOG2_INV_RATE} for ty, _, k in committed],
    }, geometry["lookup_entries"]


PROVENANCE = {
    "trees": "derived here: all RFC-0050 lookup requests, each real layer's per-axis-padded domain, "
    "one lookup/table tree pair per ascending TableId (RFC-0045), then Softmax forests stacked over "
    "padded layers/heads/queries (RFC-0010/0030)",
    "layer_vars": "derived here: ceil(log2 n_layers) (RFC-0037 'Layers and kinds')",
    "stages": "derived here by walking the public graph: nonlinear relations cost one stage; linear "
    "maps cost none; include OneHot/root/Softmax obligations and their dependencies (RFC-0009/0010/0030/0035/0037)",
    "advice_vars": "derived here through the independent public lowering and advice oracles: nu - log2 k, "
    "at test-small/1, stated Q (512 for native cases), one segment (RFC-0011/0047/0050)",
    "proximity": "derived here: each matrix with committed rows at test-small/1 (RFC-0040), ceil(log2 m_T) "
    "coordinates (RFC-0049)",
    "opened": "derived here: every matrix with committed rows, n_T = 4 k_T (RFC-0040, RFC-0012)",
}


# ----- Output ------------------------------------------------------------------------------------


def flat(value):
    """A scalar, or a list of scalars or of lists of scalars."""
    if isinstance(value, list):
        return all(not isinstance(x, (dict, list)) or (isinstance(x, list) and flat(x)) for x in value)
    return not isinstance(value, dict)


def compact(value, indent=0):
    """JSON with each step, each tree, stage and matrix of the sizes, and every list of scalars on
    one line."""
    pad = " " * indent
    if (isinstance(value, dict) and all(flat(v) for v in value.values())) or flat(value):
        return json.dumps(value, ensure_ascii=False)
    if isinstance(value, list):
        inner = ",\n".join(pad + " " + compact(x, indent + 1) for x in value)
        return "[\n" + inner + "\n" + pad + "]"
    if isinstance(value, dict):
        inner = ",\n".join(f"{pad} {json.dumps(k)}: {compact(v, indent + 1)}" for k, v in value.items())
        return "{\n" + inner + "\n" + pad + "}"
    return json.dumps(value)


def main():
    checked = hashing.check_official_vectors()
    rng = hashing.splitmix64(SEED)
    cases = [
        case("every rule, live", RICH, "test-small/1", "Live", rng),
        case("every rule, non-interactive", RICH, "test-small/1", "NonInteractive", rng),
        case("every rule, non-interactive, proofs of work of 8 and 10 bits", RICH, "test-small/1",
             "NonInteractive", rng, pow_bits=(8, 10)),
        case("no tree, no stage, no matrix, live", BARE, "test-small/1", "Live", rng),
        case("no tree, no stage, no matrix, non-interactive", BARE, "test-small/1", "NonInteractive", rng),
        case("fraction trees of mixed heights, non-interactive, proofs of work of 6 and 9 bits", FRACTIONS,
             "test-small/1", "NonInteractive", rng, pow_bits=(6, 9)),
        case("fraction trees of height 0, live", FLAT, "test-small/1", "Live", rng),
        case("every rule at live-64/1", LIVE_64, "live-64/1", "Live", rng),
        case("every rule at ni-100/1, proofs of work of 8 and 7 bits", LIVE_64, "ni-100/1", "NonInteractive", rng,
             pow_bits=(8, 7)),
    ]
    index = json.loads((HERE.parent / "models" / "models.json").read_text(encoding="utf-8"))
    require([m["name"] for m in index["models"]] == weights.MODEL_NAMES, "every conformance model, in order")
    models = []
    for name in weights.MODEL_NAMES:
        sizes, entries = conformance_sizes(name, index)
        for mode in ("Live", "NonInteractive"):
            model = case(f"{name}, {mode}", sizes, "test-small/1", mode, rng, provenance=PROVENANCE)
            model["lookup_entries"] = entries
            models.append(model)
    boundaries = []
    configs = [(name, 1, 512, False) for name in weights.MODEL_NAMES]
    configs += [("tiny-2-servable", n, q, False) for n, q in ((3, 1), (17, 4), (64, 512))]
    configs += [("tiny-2-servable", 3, 4, True)]
    for name, n, q, padded in configs:
        sizes, entries = conformance_sizes(name, index, n, q, padded)
        for mode in ("Live", "NonInteractive"):
            title = f"{name}, n={n}, Q={q}, padded={padded}, {mode}"
            item = case(title, sizes, "test-small/1", mode, rng, provenance=PROVENANCE)
            item.update(model=name, sequence_length=n, query_block=q, padded=padded, lookup_entries=entries)
            boundaries.append(item)
    document = {
        "description": "Transcript schedules of tessera-protocol/1 (RFC-0014 'Transcript schedule' and "
        "'Grinding', as RFC-0049 places pcs/weights/{T}/xi), written by generate.py, an oracle independent of "
        "the Rust crates; see README.md.",
        "seed": f"0x{SEED:016x}",
        "official_blake3_cases_checked": checked,
        "parameter_sets": PARAMETER_SETS,
        "cases": cases,
        "models": models,
        "boundaries": boundaries,
    }
    text = compact(document) + "\n"
    require(json.loads(text) == document, "the file parses back to the document")
    (HERE / "schedule.json").write_text(text, encoding="utf-8")
    steps = sum(len(c["steps"]) for c in cases + models + boundaries)
    print(f"wrote schedule.json: {len(cases)} cases, {len(models)} model cases, {len(boundaries)} boundary cases, {steps} steps, {len(text)} bytes")


if __name__ == "__main__":
    main()
