# SPDX-License-Identifier: Apache-2.0 OR MIT
"""Measure synthetic held-out scoring and integer trace RSS in separate processes.

From python/: PYTHONPATH=. uv run python benchmarks/heldout_memory.py MODE
Modes kl-eager/kl-stream compare equal KL reductions; reference-trace/reference-logits compare
equal integer logits on a repeated conformance model. Neither workload is a real model proof.
"""

import argparse
import hashlib
import json
import resource
import sys
import time
from dataclasses import replace
from pathlib import Path

import numpy as np
import torch

from tessera_quant.integer_model import INTEGER_MODEL, IntegerModel, LayerPoint, WeightType
from tessera_quant.migration import kl_rows, mean_kl
from tessera_quant.reference import run_reference, trace
from tessera_quant.tce import INTEGER_MODEL_LIMITS, unseal


def _inputs(index: int) -> tuple[torch.Tensor, torch.Tensor]:
    positions = torch.arange(128 * 8192).reshape(128, 8192)
    return (
        ((positions + index) % 257 - 128).to(torch.float64) / 16,
        ((positions * 3 + index) % 251 - 125).to(torch.float64) / 16,
    )


def _repeated_model() -> IntegerModel:
    path = Path(__file__).resolve().parents[2] / "spec/vectors/models/tiny-2-servable.d"
    base = unseal(path.read_bytes(), INTEGER_MODEL, limits=INTEGER_MODEL_LIMITS)
    layers = 64
    public = base.quant.public

    def points(entries):
        shared = [(p, v) for p, v in entries if not isinstance(p, LayerPoint)]
        return shared + [
            (replace(p, layer=layer), v)
            for layer in range(layers)
            for p, v in entries
            if isinstance(p, LayerPoint) and p.layer == layer % base.arch.n_layers
        ]

    public = replace(
        public,
        act_exponents=points(public.act_exponents),
        shift_class_sets=points(public.shift_class_sets),
        affine_sites=points(public.affine_sites),
        score_scales=[
            (layer, dict(public.score_scales)[layer % base.arch.n_layers])
            for layer in range(layers)
        ],
    )
    shared_types = {WeightType.Embed, WeightType.LmHead}
    channels = [c for c in base.quant.channels if c.ty in shared_types] + [
        replace(c, layer=layer)
        for layer in range(layers)
        for c in base.quant.channels
        if c.ty not in shared_types and c.layer == layer % base.arch.n_layers
    ]
    weights = [w for w in base.weights if w.ty in shared_types] + [
        replace(w, layer=layer)
        for layer in range(layers)
        for w in base.weights
        if w.ty not in shared_types and w.layer == layer % base.arch.n_layers
    ]
    quant = replace(
        base.quant,
        public=public,
        channels=channels,
        silu_in_exponents=[
            (layer, dict(base.quant.silu_in_exponents)[layer % base.arch.n_layers])
            for layer in range(layers)
        ],
    )
    return replace(
        base,
        arch=replace(base.arch, n_layers=layers, max_context=256),
        quant=quant,
        weights=weights,
    )


def main() -> None:
    """Measure one mode's elapsed time, process peak RSS and exact result fingerprint."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "mode", choices=("kl-eager", "kl-stream", "reference-trace", "reference-logits")
    )
    mode = parser.parse_args().mode
    torch.set_num_threads(2)
    start = time.perf_counter()
    if mode == "kl-eager":
        pairs = [_inputs(i) for i in range(32)]
        result = mean_kl(torch.cat([p[0] for p in pairs]), torch.cat([p[1] for p in pairs])).hex()
    elif mode == "kl-stream":
        rows = []
        for i in range(32):
            reference, logits = _inputs(i)
            rows.append(kl_rows(reference, logits))
            del reference, logits
        result = float(torch.cat(rows).mean()).hex()
    else:
        model = _repeated_model()
        tokens = np.arange(256, dtype=np.uint32) % model.arch.vocab
        if mode == "reference-trace":
            recorded = trace(model, tokens)
            values = recorded.logits.numpy()
        else:
            values = run_reference(model, tokens)
        result = hashlib.sha256(values.tobytes()).hexdigest()
    rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    print(
        json.dumps(
            {
                "mode": mode,
                "seconds": time.perf_counter() - start,
                "peak_rss_bytes": rss if sys.platform == "darwin" else rss * 1024,
                "result": result,
            }
        )
    )


if __name__ == "__main__":
    main()
