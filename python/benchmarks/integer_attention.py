# SPDX-License-Identifier: Apache-2.0 OR MIT
"""Compare int64 attention products with bounded exact tiles on identical head tensors.

From python/: PYTHONPATH=. uv run python benchmarks/integer_attention.py scalar|tiled
This measures the score product, not a complete model, prover or GPU backend.
"""

import argparse
import hashlib
import json
import resource
import statistics
import sys
import time

import torch

from tessera_quant._integer_matmul import try_bmm


def main() -> None:
    """Measure one fresh-process mode on strided queries and GQA-repeated keys."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("scalar", "tiled"))
    parser.add_argument("--tokens", type=int, default=512)
    parser.add_argument("--heads", type=int, default=4)
    parser.add_argument("--kv-heads", type=int, default=2)
    parser.add_argument("--dimension", type=int, default=64)
    parser.add_argument("--warmups", type=int, default=0)
    parser.add_argument("--iterations", type=int, default=1)
    args = parser.parse_args()
    if args.heads % args.kv_heads or args.iterations < 1 or args.warmups < 0:
        raise ValueError("whole GQA groups and positive measurement count")
    torch.set_num_threads(2)
    generator = torch.Generator().manual_seed(621)
    q = torch.randint(
        -128, 128, (args.tokens, args.heads, args.dimension), generator=generator, dtype=torch.int64
    ).transpose(0, 1)
    k = torch.randint(
        -128,
        128,
        (args.tokens, args.kv_heads, args.dimension),
        generator=generator,
        dtype=torch.int64,
    ).transpose(0, 1)
    right = k.repeat_interleave(args.heads // args.kv_heads, dim=0).transpose(1, 2)
    samples = []
    found = None
    for iteration in range(args.warmups + args.iterations):
        found = None  # Release the previous output before timing/allocating the next one.
        start = time.perf_counter()
        if args.mode == "scalar":
            found = q @ right
        else:
            found = try_bmm(q, right)
            if found is None:
                raise RuntimeError("the optional integer backend is unavailable")
        if iteration >= args.warmups:
            samples.append(time.perf_counter() - start)
    if found is None:
        raise RuntimeError("the measurement produced no output")
    rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    print(
        json.dumps(
            {
                "mode": args.mode,
                "tokens": args.tokens,
                "heads": args.heads,
                "kv_heads": args.kv_heads,
                "dimension": args.dimension,
                "seconds": samples,
                "median_seconds": statistics.median(samples),
                "peak_rss_bytes": rss if sys.platform == "darwin" else rss * 1024,
                "sha256": hashlib.sha256(found.numpy().tobytes()).hexdigest(),
                "scope": "synthetic CPU score product; not whole-model or GPU qualification",
            }
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
