# SPDX-License-Identifier: Apache-2.0 OR MIT
"""EXP-0032's fit and chunk planner, as pre-registered in README.md (IDEA-0043).

Reads the binary's JSON rows, fits rows(n) = a*n + b*n^2 by least squares, and plans
binary-aligned chunks of a long request under the 4 GiB gate of 08 §5.
"""

import json
import sys

GATE = 4 * 2**30  # bytes, 08 §5
COLUMN_BYTES_PER_ROW = 1_240  # RFC-0057's opened columns per non-zero row (E0)
NON_COLUMN = 2.02e9  # largest non-column part of RFC-0057's three rows (E0)
LINK = 0.07e9  # one combined row of 2^22 Fp2 plus slack (IDEA-0043, E0)
PER_EARLIER = 1e6  # tau_A short columns of one earlier context commitment (E0)
MAX_CHUNK = 65_536
MIN_CHUNK = 1_024


def fit(points):
    """Least-squares a, b for rows = a*n + b*n^2 through the origin."""
    s2 = sum(n**2 for n, _ in points)
    s3 = sum(n**3 for n, _ in points)
    s4 = sum(n**4 for n, _ in points)
    r1 = sum(n * r for n, r in points)
    r2 = sum(n * n * r for n, r in points)
    det = s2 * s4 - s3 * s3
    return (r1 * s4 - r2 * s3) / det, (s2 * r2 - s3 * r1) / det


def main(path):
    """Print the fit, its residuals and the chunk plan for the rows in `path`."""
    with open(path) as f:
        data = json.load(f)
    points = [(d["n"], d["rows"]) for d in data]
    measured = dict(points)
    a, b = fit(points)
    print(f"fit: a = {a:.4f} rows/token, b = {b:.6e} rows/token^2")
    worst = 0.0
    for n, r in points:
        rel = (a * n + b * n * n) / r - 1
        worst = max(worst, abs(rel))
        print(f"  n = {n:>6}: rows {r:>10,}  fit {a * n + b * n * n:>12,.0f}  residual {rel:+.2%}")
    print(f"largest relative residual: {worst:.2%}")

    def rows(n):
        """Measured rows at a measured size, the fit elsewhere."""
        return measured[n] if n in measured else a * n + b * n * n

    def size(c, p, k):
        """Proof bytes of a chunk of c tokens at offset p after k earlier chunks."""
        added = rows(p + c) - (rows(p) if p else 0)
        return NON_COLUMN + LINK + k * PER_EARLIER + COLUMN_BYTES_PER_ROW * added

    for s in (16_384, 32_768, 65_536, 131_072):
        p, k, out, stuck = 0, 0, [], None
        while p < s:
            c = MAX_CHUNK
            while c >= MIN_CHUNK and (p % c or p + c > s or size(c, p, k) > GATE):
                c //= 2
            if c < MIN_CHUNK:
                stuck = p
                break
            out.append((p, c, size(c, p, k)))
            p, k = p + c, k + 1
        if stuck is not None:
            print(f"S = {s}: no aligned chunk of >= {MIN_CHUNK} tokens fits at offset {stuck}")
            continue
        sizes = ", ".join(str(c) for _, c, _ in out)
        print(
            f"S = {s}: {len(out)} proofs, total {sum(x[2] for x in out) / 1e9:.1f} GB, "
            f"largest {max(x[2] for x in out) / 2**30:.3f} GiB; chunks {sizes}"
        )


if __name__ == "__main__":
    main(sys.argv[1])
