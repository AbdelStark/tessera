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

The conformance models' leaf trees and reduction stages are the prover's derivations, stated
below (DERIVED), and their advice proximity point has no derivation yet (#106): those cases run
the transcript up to `pcs/advice/xi` and list the labels after it. See README.md.

Usage: python3 spec/vectors/transcript-schedule/generate.py
(rewrites schedule.json)

Inputs come from SplitMix64 with a fixed seed, so any implementation can regenerate them.
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


transcript = _oracle("transcript")  # RFC-0004's transcript and TCE(SessionContext)
hashing = transcript.oracle  # BLAKE3 and H, from ../hash/generate.py
weights = _oracle("weights")  # RFC-0040's matrices and the models' reader

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

# What no oracle derives yet, from the prover's derivations at each model's first sequence of
# ../models/models.json: each LogUp instance's lookup tree and table tree heights (h_a, h_t), in
# ascending TableId, from tessera-prover's leaf layout of the requantization sites (RFC-0045, #367),
# the only gadgets laid out so far; and each reduction stage's rounds and zero-check coordinates
# from tessera-compiler's stage schedule (RFC-0037, #113). crates/tessera-prover/tests/
# transcript_sizes.rs derives them again and compares.
DERIVED = {
    "tiny-2-servable": {
        "instances": [(9, 23), (9, 22), (11, 19), (9, 19), (9, 18), (9, 23), (10, 20), (10, 20), (10, 23),
                      (11, 23), (11, 25), (11, 25), (12, 23), (11, 16), (9, 25), (9, 19), (11, 16), (9, 25)],
        "stages": [(12, 9), (9, 0)],
    },
    "tiny-2-minimal": {
        "instances": [(9, 23), (9, 22), (11, 19), (9, 18), (9, 19), (9, 21), (10, 28), (9, 27), (11, 23),
                      (11, 25), (11, 25), (9, 12), (10, 24), (9, 7), (9, 8), (10, 28), (9, 18), (9, 27),
                      (9, 5), (9, 14), (9, 15)],
        "stages": [(10, 9), (12, 0), (7, 0)],
    },
    "tiny-2-affine": {
        "instances": [(9, 23), (9, 22), (11, 26), (11, 26), (8, 23), (10, 17), (10, 20), (10, 20), (9, 21),
                      (11, 25), (11, 25), (12, 19), (11, 18), (12, 16), (9, 25), (10, 23), (8, 23), (9, 22),
                      (11, 18), (9, 25)],
        "stages": [(12, 9), (10, 0), (7, 0)],
    },
    "tiny-4-servable": {
        "instances": [(15, 23), (11, 22), (15, 20), (13, 20), (11, 24), (13, 20), (12, 24), (14, 23), (14, 25),
                      (14, 16), (12, 25), (13, 19), (12, 25), (11, 20), (9, 24), (11, 24), (12, 25), (12, 16),
                      (11, 24)],
        "stages": [(14, 11), (10, 0)],
    },
}


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


def conformance_sizes(name, index):
    """A model's sizes: its derived trees and stages; λ = ⌈log2 L⌉; the proximity point of each
    matrix with committed rows, ⌈log2 m_T⌉ coordinates (RFC-0049); the opened matrices, every one
    with committed rows, each read by a claim of the schedule: the nine weight types by their
    MatMulWeight instances, ChannelParams by the sites that read its classes or multipliers."""
    value, _ = weights.read_model(name, index)
    arch = value["arch"]
    channels = {}
    for ch in value["quant"]["channels"]:
        ty = next(iter(ch["ty"]))
        channels[weights.WEIGHT_TYPES.index(ty), ch["layer"]] = ch["class"] or ch["multiplier"]
    derived = DERIVED[name]
    trees = [tree for h_a, h_t in derived["instances"] for tree in (fraction(h_a), fraction(h_t))]
    committed = matrices(arch, channels)
    return {
        "trees": trees,
        "layer_vars": (arch["n_layers"] - 1).bit_length(),
        "stages": [stage(r, z) for r, z in derived["stages"]],
        "advice_vars": None,
        "proximity": [{"matrix": ty, "vars": (m - 1).bit_length()} for ty, m, _ in committed],
        "opened": [{"matrix": ty, "codeword": k << WEIGHT_LOG2_INV_RATE} for ty, _, k in committed],
    }


PROVENANCE = {
    "trees": "derived by the prover: tessera-prover's leaf layout of the requantization sites (RFC-0045, #367), "
    "a lookup tree and a table tree per instance in ascending TableId; the other gadgets' instances and the "
    "product forests join with complete gadget/leaf integration (#126, #130)",
    "layer_vars": "derived here: ceil(log2 n_layers) (RFC-0037 'Layers and kinds')",
    "stages": "derived by the prover: tessera-compiler's stage schedule (RFC-0037, #113); the gadgets' zero-checks "
    "(RFC-0035, OneHot) join with their gadgets",
    "advice_vars": "not derived: the canonical advice layout (#106) fixes the advice row index's bits; the transcript "
    "runs up to pcs/advice/xi",
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
        sizes = conformance_sizes(name, index)
        for mode in ("Live", "NonInteractive"):
            models.append(case(f"{name}, {mode}", sizes, "test-small/1", mode, rng, provenance=PROVENANCE))
    document = {
        "description": "Transcript schedules of tessera-protocol/1 (RFC-0014 'Transcript schedule' and "
        "'Grinding', as RFC-0049 places pcs/weights/{T}/xi), written by generate.py, an oracle independent of "
        "the Rust crates; see README.md.",
        "seed": f"0x{SEED:016x}",
        "official_blake3_cases_checked": checked,
        "parameter_sets": PARAMETER_SETS,
        "cases": cases,
        "models": models,
    }
    text = compact(document) + "\n"
    require(json.loads(text) == document, "the file parses back to the document")
    (HERE / "schedule.json").write_text(text, encoding="utf-8")
    steps = sum(len(c["steps"]) for c in cases + models)
    print(f"wrote schedule.json: {len(cases)} cases, {len(models)} model cases, {steps} steps, {len(text)} bytes")


if __name__ == "__main__":
    main()
