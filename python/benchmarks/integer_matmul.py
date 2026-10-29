# SPDX-License-Identifier: Apache-2.0 OR MIT
"""Measure scalar conversion/matmul and bounded INT8 tiles on identical synthetic inputs.

From python/: PYTHONPATH=. uv run python benchmarks/integer_matmul.py scalar|tiled
Use --rows 16 --columns 32768 to measure a wider weight matrix's conversion residency.
"""

import argparse
import hashlib
import json
import resource
import sys
import time

import numpy as np
import torch

from tessera_quant._integer_matmul import try_matmul


def main() -> None:
    """Measure one fresh-process mode, including its required weight conversion and output."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("scalar", "tiled"))
    parser.add_argument("--rows", type=int, default=512)
    parser.add_argument("--inner", type=int, default=2048)
    parser.add_argument("--columns", type=int, default=2048)
    args = parser.parse_args()
    torch.set_num_threads(2)
    generator = torch.Generator().manual_seed(615)
    x = torch.randint(-128, 128, (args.rows, args.inner), generator=generator, dtype=torch.int64)
    weights = np.random.default_rng(615).integers(
        -128, 128, (args.columns, args.inner), dtype=np.int8
    )
    weights.flags.writeable = False
    start = time.perf_counter()
    if args.mode == "scalar":
        converted = torch.tensor(weights, dtype=torch.int64)
        found = x @ converted.T
    else:
        found = try_matmul(x, weights)
        if found is None:
            raise RuntimeError("the optional integer backend is unavailable")
    elapsed = time.perf_counter() - start
    rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    print(
        json.dumps(
            {
                "mode": args.mode,
                "rows": args.rows,
                "inner": args.inner,
                "columns": args.columns,
                "seconds": elapsed,
                "peak_rss_bytes": rss if sys.platform == "darwin" else rss * 1024,
                "sha256": hashlib.sha256(found.numpy().tobytes()).hexdigest(),
            }
        )
    )


if __name__ == "__main__":
    main()
