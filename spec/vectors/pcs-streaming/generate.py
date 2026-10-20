#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0 OR MIT
"""Independent RFC-0011 streaming commitment vectors, before the Rust committer.

Enumerate retained tensor coordinates to locate committed rows (RFC-0047), align each
tensor's padded domain, and assign layers to segments (RFC-0046). Interpolate
each row and evaluate its masked polynomial with the independent PCS oracle;
hash full columns in blocks, then merge every segment and build a dense tree.
No Rust code or output is used. All witness values are public synthetic data.
"""

import importlib.util
import itertools
import json
import sys
from pathlib import Path

sys.dont_write_bytecode = True
HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location("pcs_oracle", HERE.parent / "pcs/generate.py")
pcs = importlib.util.module_from_spec(spec)
spec.loader.exec_module(pcs)
require, H, hex64 = pcs.require, pcs.H, pcs.hex64
SEED = 0x7E55E7A_0115_0011


def geometry(tiles, layers, segments, positions, log_k):
    """Enumerate retained slots: RFC-0047 commits inner-axis length padding."""
    k, cursor = 1 << log_k, 0
    rows = [[] for _ in range(segments)]
    cuts = [j * (layers // segments) + min(j, layers % segments) for j in range(segments + 1)]
    past_layer = False
    for tile in tiles:
        layer = tile["layer"]
        if layer is None:
            segment = segments - 1 if past_layer else 0
        else:
            past_layer = True
            segment = next(s for s in range(segments) if cuts[s] <= layer < cuts[s + 1])
        for value in tile["values"]:
            if "dense" in value:
                shape = value["dense"]
                bits = [(axis - 1).bit_length() for axis in shape]
                order = sorted(range(len(shape)), key=lambda i: shape[i] == 1 << bits[i])
                outer = next((i for i in order if shape[i] != 1 << bits[i]), None)
                # Only the outermost length padding is skipped (RFC-0047 Padding).
                retained = [1 << bits[i] if shape[i] != 1 << bits[i] and i != outer else shape[i]
                            for i in range(len(shape))]
                cells = []
                for coordinates in itertools.product(*(range(axis) for axis in retained)):
                    index = 0
                    for axis in order:
                        index = (index << bits[axis]) + coordinates[axis]
                    cells.append(index)
                variables = sum(bits)
            else:
                # These cases have g=Q'=1: (i_hi,h,q,i_lo) is simply the key index.
                block = value["score"]
                variables = (positions - 1).bit_length()
                cells = list(range(block + 1)) if block < positions else []
            span = max(1, (1 << variables) // k)
            base = (cursor + span - 1) // span * span
            rows[segment].extend(sorted({base + cell // k for cell in cells}))
            cursor = base + span
    flattened = [row for segment in rows for row in segment]
    require(all(a < b for a, b in zip(flattened, flattened[1:])), "strict layout order")
    return cursor, rows


def tiles_for(kind, layers):
    dense = lambda *axes: {"dense": list(axes)}
    score = lambda block: {"score": block}
    if kind == "one":
        return [{"layer": 0, "values": [dense(1)]}]
    if kind == "empty":
        return [{"layer": layer, "values": [score(3)]} for layer in range(layers)]
    tiles = [{"layer": None, "values": [dense(1), dense(3, 16)]}]
    for layer in range(layers):
        values = [dense(3, 2), dense(5, 8)] if kind == "mixed" else [dense(3, 5)]
        tiles.append({"layer": layer, "values": values + [score(0), score(3)]})
    tiles.append({"layer": None, "values": [dense(1)]})
    return tiles


def case(rng, unit, kind, segments, block):
    layers = 1 if kind == "one" else (8 if kind == "sparse" else 3)
    positions = 1 if kind == "one" else (4 if kind == "sparse" else 3)
    log_k = 0 if kind in ("one", "sparse") else 3
    k, n = 1 << log_k, 2 << log_k
    masks_per_row = 1 if k == 1 else 3
    tiles = tiles_for(kind, layers)
    allocated, segments_rows = geometry(tiles, layers, segments, positions, log_k)
    indices = [row for segment in segments_rows for row in segment]
    statement = rng.digest()
    # RFC-0031 Advice descriptor: enum tag, statement, unit, layout fields, row count.
    descriptor = (bytes(2) + statement + unit.to_bytes(4, "little") + bytes([log_k, 1])
                  + b"".join(x.to_bytes(4, "little") for x in (masks_per_row, block, segments))
                  + len(indices).to_bytes(8, "little"))
    commitment_id = H("commitment-id", descriptor)
    values, masks, codewords = {}, {}, {}
    for row in indices:
        values[row] = [rng.fp() for _ in range(k)]
        masks[row] = [rng.fp() for _ in range(masks_per_row)]
        _, codewords[row] = pcs.encode_row(values[row], masks[row], k, n)
    leaves, tree = pcs.commit(commitment_id, codewords, segments_rows, block, n)
    digests = [pcs.hashing.column_leaves(commitment_id, [codewords[r] for r in rows], n, block)
               for rows in segments_rows]
    return {
        "name": f"{kind}, S={segments}, B={block}",
        "layout": {"log2_k": log_k, "segments": segments, "layers": layers,
                   "group": 1, "query_block": 1, "max_context": 4},
        "positions": positions, "tiles": tiles,
        "mask_coeffs": masks_per_row, "row_block": block,
        "statement_digest": statement.hex(), "unit": unit,
        "descriptor_tce": descriptor.hex(), "commitment_id": commitment_id.hex(),
        "allocated_rows": allocated, "row_vars": (allocated - 1).bit_length(),
        "segments": segments_rows,
        "rows": [[hex64(x) for x in values[r]] for r in indices],
        "masks": [[hex64(x) for x in masks[r]] for r in indices],
        "codewords": [[hex64(x) for x in codewords[r]] for r in indices],
        "segment_digests": [[d.hex() for d in part] for part in digests],
        "leaves": [d.hex() for d in leaves], "root": tree[-1][0].hex(),
    }


def main():
    official = pcs.hashing.check_official_vectors()
    rng = pcs.Inputs(SEED)
    configs = [("mixed", s, b) for s in (1, 2, 8) for b in (1, 3, 32)]
    configs += [(kind, s, 32) for kind in ("empty", "one") for s in (1, 2, 8)]
    configs.append(("sparse", 8, 5))
    cases = [case(rng, unit, *config) for unit, config in enumerate(configs)]
    require(any(not c["rows"] for c in cases), "empty commitment")
    require(any(any(not s for s in c["segments"]) for c in cases), "empty segment")
    require(any(any(len(s) % c["row_block"] for s in c["segments"]) for c in cases), "partial block")
    doc = {"description": "RFC-0011 synthetic streaming commitments; no soundness claim for test parameters",
           "seed": hex64(SEED), "official_blake3_cases_checked": official, "cases": cases}
    text = json.dumps(doc, indent=1) + "\n"
    require(json.loads(text) == doc, "JSON round trip")
    (HERE / "streaming.json").write_text(text, encoding="utf-8")
    print(f"wrote streaming.json: {len(cases)} cases, {sum(len(c['rows']) for c in cases)} rows")


if __name__ == "__main__":
    main()
