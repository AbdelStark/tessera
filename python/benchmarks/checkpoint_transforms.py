# SPDX-License-Identifier: Apache-2.0 OR MIT
"""Measure RFC-0005 transform RSS on a synthetic 50M checkpoint (dev dependencies required).

From python/: PYTHONPATH=.:tests uv run python benchmarks/checkpoint_transforms.py prepare DIR
Then run eager and stream in separate processes against DIR. Only synthetic data belongs here.
"""

import argparse
import hashlib
import json
import math
import resource
import sys
import time
from dataclasses import replace
from fractions import Fraction
from pathlib import Path

import torch
from checkpoints import TINY, TINY_ARCH, tensors, write

from tessera_quant import import_checkpoint, tensor_shapes
from tessera_quant._streaming import transformed_checkpoint
from tessera_quant.capture import points
from tessera_quant.migration import transform


def main() -> None:
    """Prepare the fixed checkpoint, or hash one eager/streamed transformation and measure it."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("prepare", "eager", "stream"))
    parser.add_argument("checkpoint", type=Path)
    args = parser.parse_args()
    torch.set_num_threads(2)
    if args.mode == "prepare":
        arch = replace(
            TINY_ARCH,
            n_layers=16,
            d_model=512,
            d_ff=1536,
            n_heads=8,
            n_kv_heads=2,
            d_head=64,
            vocab=2048,
        )
        config = TINY | {
            "num_hidden_layers": 16,
            "hidden_size": 512,
            "intermediate_size": 1536,
            "num_attention_heads": 8,
            "num_key_value_heads": 2,
            "vocab_size": 2048,
        }
        write(args.checkpoint, config, tensors(arch), shards=4)
        print(json.dumps({"parameters": sum(math.prod(s) for s in tensor_shapes(arch).values())}))
        return
    model = import_checkpoint(args.checkpoint)
    maxima = {
        name: torch.linspace(0.01, 2, channels, dtype=torch.float64)
        for name, channels in points(model.arch).items()
    }
    start = time.perf_counter()
    if args.mode == "eager":
        source = {name: model.tensor(name).to(torch.float64) for name in model.names}
        weights, stats = transform(model.arch, source, maxima, Fraction(3, 5))
    else:
        weights, stats = transformed_checkpoint(model, maxima, Fraction(3, 5))
    digest = hashlib.sha256()
    for name in weights:
        digest.update(weights[name].numpy().tobytes())
    for name in stats:
        digest.update(stats[name].numpy().tobytes())
    rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    print(
        json.dumps(
            {
                "mode": args.mode,
                "seconds": time.perf_counter() - start,
                "peak_rss_bytes": rss if sys.platform == "darwin" else rss * 1024,
                "sha256": digest.hexdigest(),
            }
        )
    )


if __name__ == "__main__":
    main()
