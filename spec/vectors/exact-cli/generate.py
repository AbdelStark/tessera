#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0 OR MIT
"""Small three-layer CLI fixture, from the independent RFC-0005 integer oracle.

No Rust or tessera_quant code runs here. All weights/tokens are synthetic public
test data. The existing model and TokenFile schemas are unchanged.
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


def artifacts():
    spec = {
        "name": "tiny-3-reference", "seed": 0x7E55_E7A0_0133_0003,
        "profile": "Servable", "residual_bits": 8,
        "shape": {"n_layers": 3, "d_model": 2, "n_heads": 1, "n_kv_heads": 1,
                  "d_head": 2, "d_ff": 6, "vocab": 8, "max_context": 8},
        "classes": {key: 1 for key in ["Q", "K", "V", "O", "Gate", "Up", "Down"]},
        "lengths": [3, 5],
        "plan": {"push": {}, "silu_reach": {}, "exp_reach": {}, "update_clamp": {}},
    }
    model, calibration = oracle.build(spec)
    model["residual_exponent"] = model["embed_exponent"]
    # A synthetic RMS epsilon of 1 keeps rho's public range small while retaining
    # nonzero normalized values; all arithmetic still follows RFC-0005 exactly.
    eps = model["d_model"] * 4 ** (-model["residual_exponent"])
    model["rms_eps_int"] = eps
    # rho < 32 keeps the complete RFC-0050 carry lookups within 18 bits.
    model["kappa"] = 7
    model["exp_sigma"] = 3
    chooser = oracle.Run(model, spec["plan"])
    chooser.forward(calibration)
    constants = copy.deepcopy(chooser.c)
    # RFC-0038: one table/shift set and score scale per site across all layers.
    for site in oracle.sites_of("Servable"):
        shifts = [constants[("Layer", layer, site)] for layer in range(3)]
        common = [max(values[0] for values in shifts)]
        for layer in range(3):
            constants[("Layer", layer, site)] = common
    scale = min(constants[("score_scale", layer)] for layer in range(3))
    for layer in range(3):
        constants[("score_scale", layer)] = scale

    tokens = [1, 2, 0, 0, 0]
    prompt = 2
    for t in range(prompt, len(tokens)):
        _, logits = oracle.Run(model, spec["plan"], constants).forward([tokens])
        row = logits[0][t - 1]
        tokens[t] = row.index(max(row))
    _, logits = oracle.Run(model, spec["plan"], constants).forward([tokens])
    oracle.require(any(v != 0 for row in logits[0] for v in row), "CLI fixture has nonzero logits")
    for t in range(prompt, len(tokens)):
        oracle.require(tokens[t] == logits[0][t - 1].index(max(logits[0][t - 1])), "greedy continuation")
    integer = oracle.integer_model(spec, model, constants)
    integer["arch"]["rms_eps"] = {"num": 1, "den": 1}
    integer["provenance"]["tessera_quant"] = "spec/vectors/exact-cli/generate.py"
    model_bytes = oracle.seal(oracle.encode(oracle.INTEGER_MODEL, integer))
    payload = oracle.length(1) + oracle.length(len(tokens)) + b"".join(oracle.fixed("u32", x) for x in tokens)
    signed = b"TSRA" + oracle.fixed("u16", 1) + oracle.fixed("u16", 0x0601)
    signed += oracle.fixed("u16", 1) + oracle.fixed("u16", 0) + oracle.length(len(payload)) + payload
    token_bytes = signed + oracle.hash_oracle.H("envelope", signed)
    expected = {
        "description": "Synthetic public three-layer exact-greedy CLI fixture; nonzero integer logits from the independent Python oracle.",
        "model": "tiny-3-reference.d", "tokens": "recorded.ttk", "layers": 3,
        "prompt_len": prompt, "output_len": len(tokens) - prompt,
        "request_id": 23, "request_session": "11" * 16, "manifest_digest": "13" * 32,
        "token_ids": tokens, "logits": logits[0],
        "model_sha256": hashlib.sha256(model_bytes).hexdigest(),
        "tokens_sha256": hashlib.sha256(token_bytes).hexdigest(),
    }
    return {"tiny-3-reference.d": model_bytes, "recorded.ttk": token_bytes,
            "expected.json": (json.dumps(expected, indent=2) + "\n").encode()}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    for name, data in artifacts().items():
        path = HERE / name
        if args.check:
            oracle.require(path.exists() and path.read_bytes() == data, f"regenerate exact-cli/{name}")
        else:
            path.write_bytes(data)


if __name__ == "__main__":
    main()
