#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0 OR MIT
"""Canary model and tokens for the OBS-3 redaction scan, from the independent RFC-0005 oracle.

The model carries runs of eight INT8 weights whose bytes, read little-endian, form a distinctive
64-bit value, and the token file a prompt of distinctive token ids, followed by the model's own
greedy continuation. No Rust or tessera_quant code runs here; all values are synthetic test data.
"""
import argparse
import copy
import hashlib
import importlib.util
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
module = importlib.util.spec_from_file_location("model_oracle", HERE.parent / "models" / "generate.py")
oracle = importlib.util.module_from_spec(module)
module.loader.exec_module(oracle)

LAYERS = 2
VOCAB = 4096
PROMPT_LEN = 3
OUTPUT_LEN = 3
MISMATCH = 0xA3C

# (tensor, layer, offset into its row-major data, eight INT8 weights). Every byte is nonzero and
# the last one negative, so each run's 64-bit value has 19 or 20 decimal digits and no short form.
# The LM head's run is the rows of tokens 0xa3c to 0xa3f, one per quadrant and longer than the
# drawn rows, so that the greedy continuation picks distinctive ids.
RUNS = [
    ("Embed", 0, 1024, [90, -61, 110, -105, 47, -88, 101, -77]),
    ("Gate", 1, 2, [-99, 83, -52, 117, -41, 66, -114, 29]),
    ("LmHead", 0, 2 * 0xA3C, [117, 109, 103, -98, -111, 95, -106, -121]),
]


def value(run):
    """The run's bytes as one little-endian 64-bit value."""
    return int.from_bytes(bytes(w & 0xFF for w in run), "little")


def distinctive(token):
    """A token id whose decimal and hex forms are unlikely to occur in a log by chance: four
    decimal digits that do not read as a year, and a hex form mixing digits and letters, so that
    neither a count, a timestamp nor an English word spells it."""
    h = format(token, "x")
    return (1000 <= token < VOCAB and not 1900 <= token < 2200
            and any(c.isdigit() for c in h) and any(c.isalpha() for c in h))


def token_file(tokens):
    """A one-sequence TokenFile (RFC-0003), as the exact-cli fixture writes it."""
    payload = oracle.length(1) + oracle.length(len(tokens)) + b"".join(oracle.fixed("u32", x) for x in tokens)
    signed = b"TSRA" + oracle.fixed("u16", 1) + oracle.fixed("u16", 0x0601)
    signed += oracle.fixed("u16", 1) + oracle.fixed("u16", 0) + oracle.length(len(payload)) + payload
    return signed + oracle.hash_oracle.H("envelope", signed)


def calibrated(spec):
    model, calibration = oracle.build(spec)
    for ty, layer, offset, run in RUNS:
        data = model["weights"][ty, layer]["data"]
        oracle.require(offset + len(run) <= len(data), f"canary run fits {ty} {layer}")
        data[offset:offset + len(run)] = run
    model["residual_exponent"] = model["embed_exponent"]
    # The synthetic RMS epsilon and reciprocal precision of the exact-cli fixture: they keep the
    # RFC-0050 carry lookups within the 18-bit row policy the test compiles with.
    model["rms_eps_int"] = model["d_model"] * 4 ** (-model["residual_exponent"])
    model["kappa"] = 7
    model["exp_sigma"] = 3
    chooser = oracle.Run(model, spec["plan"])
    chooser.forward(calibration)
    constants = copy.deepcopy(chooser.c)
    # RFC-0038: one table/shift set and score scale per site across all layers.
    for site in oracle.sites_of("Servable"):
        common = [max(constants[("Layer", layer, site)][0] for layer in range(LAYERS))]
        for layer in range(LAYERS):
            constants[("Layer", layer, site)] = common
    scale = min(constants[("score_scale", layer)] for layer in range(LAYERS))
    for layer in range(LAYERS):
        constants[("score_scale", layer)] = scale
    return model, constants


def continuation(model, spec, constants, prompt):
    tokens = list(prompt) + [0] * OUTPUT_LEN
    for t in range(PROMPT_LEN, len(tokens)):
        _, logits = oracle.Run(model, spec["plan"], constants).forward([tokens])
        row = logits[0][t - 1]
        tokens[t] = row.index(max(row))
    return tokens


def artifacts():
    spec = {
        "name": "canary-reference", "seed": 0xCA4A_E7A0_0164_0001,
        "profile": "Servable", "residual_bits": 8,
        "shape": {"n_layers": LAYERS, "d_model": 2, "n_heads": 1, "n_kv_heads": 1,
                  "d_head": 2, "d_ff": 6, "vocab": VOCAB, "max_context": 8},
        "classes": {key: 1 for key in ["Q", "K", "V", "O", "Gate", "Up", "Down"]},
        "lengths": [3, 6],
        "plan": {"push": {}, "silu_reach": {}, "exp_reach": {}, "update_clamp": {}},
    }
    model, constants = calibrated(spec)
    # The first prompt of distinctive ids whose greedy continuation is distinctive too.
    rng = oracle.Rng(spec["seed"] ^ 0x70C0)
    for _ in range(1000):
        prompt = [rng.below(VOCAB) for _ in range(PROMPT_LEN)]
        if not all(distinctive(t) for t in prompt) or len(set(prompt)) < PROMPT_LEN:
            continue
        tokens = continuation(model, spec, constants, prompt)
        if all(distinctive(t) for t in tokens[PROMPT_LEN:]):
            break
    else:
        oracle.require(False, "a distinctive prompt and continuation")
    _, logits = oracle.Run(model, spec["plan"], constants).forward([tokens])
    for t in range(PROMPT_LEN, len(tokens)):
        row = logits[0][t - 1]
        oracle.require(tokens[t] == row.index(max(row)), "greedy continuation")
    integer = oracle.integer_model(spec, model, constants)
    integer["arch"]["rms_eps"] = {"num": 1, "den": 1}
    integer["provenance"]["tessera_quant"] = "spec/vectors/canary/generate.py"
    model_bytes = oracle.seal(oracle.encode(oracle.INTEGER_MODEL, integer))
    for _, _, _, run in RUNS:
        oracle.require(bytes(w & 0xFF for w in run) in model_bytes, "canary run in the model bytes")
    token_bytes = token_file(tokens)
    # The last output token replaced by another distinctive id that the model does not choose.
    altered = tokens[:-1] + [MISMATCH]
    oracle.require(distinctive(MISMATCH) and MISMATCH not in tokens, "a distinctive altered token")
    mismatch_bytes = token_file(altered)
    canaries = {
        "description": "OBS-3 canaries of a synthetic public model: 64-bit weight runs and token ids, from the independent Python oracle.",
        "model": "canary-reference.d", "tokens": "recorded.ttk", "layers": LAYERS, "vocab": VOCAB,
        "prompt_len": PROMPT_LEN, "output_len": OUTPUT_LEN,
        "weight_runs": [{"tensor": ty, "layer": layer, "offset": offset, "weights": run,
                         "value": format(value(run), "016x")} for ty, layer, offset, run in RUNS],
        "token_ids": tokens,
        "mismatch": {"tokens": "mismatch.ttk", "token_ids": altered},
        "model_sha256": hashlib.sha256(model_bytes).hexdigest(),
        "tokens_sha256": hashlib.sha256(token_bytes).hexdigest(),
    }
    return {"canary-reference.d": model_bytes, "recorded.ttk": token_bytes, "mismatch.ttk": mismatch_bytes,
            "canaries.json": (json.dumps(canaries, indent=2) + "\n").encode()}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    for name, data in artifacts().items():
        path = HERE / name
        if args.check:
            oracle.require(path.exists() and path.read_bytes() == data, f"regenerate canary/{name}")
        else:
            path.write_bytes(data)


if __name__ == "__main__":
    main()
