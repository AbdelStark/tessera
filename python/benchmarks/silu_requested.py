# SPDX-License-Identifier: Apache-2.0 OR MIT
"""Measure exact dense/sparse SiLU evaluation on synthetic keys in separate processes.

From python/: PYTHONPATH=. uv run python benchmarks/silu_requested.py dense|sparse
The table has 2^20 possible h values; six width-8192 rows request only 256 distinct h values.
"""

import argparse
import hashlib
import json
import resource
import sys
import time

import numpy as np

from tessera_quant.tables import Silu, lookup_table, silu_lookup


def main() -> None:
    """Evaluate identical synthetic keys, reporting elapsed time, peak RSS and result hash."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("dense", "sparse"))
    mode = parser.parse_args().mode
    keys = (np.arange(6 * 8192, dtype=np.int64) % 4096 - 2048).reshape(6, 8192)
    kind = Silu(24, [4], -3, -2)
    start = time.perf_counter()
    if mode == "dense":
        table = lookup_table(0, kind)
        found = table.lookup(keys)
    else:
        found = silu_lookup(kind, keys)
    elapsed = time.perf_counter() - start
    rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    print(
        json.dumps(
            {
                "mode": mode,
                "seconds": elapsed,
                "peak_rss_bytes": rss if sys.platform == "darwin" else rss * 1024,
                "sha256": hashlib.sha256(found.tobytes()).hexdigest(),
            }
        )
    )


if __name__ == "__main__":
    main()
