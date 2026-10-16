#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0 OR MIT
"""Seed corpora for the fuzz targets, from the conformance vectors of spec/vectors/.

Usage: python3 seeds.py OUT_DIR

Writes OUT_DIR/<target> for each target below, one file per seed, named by its SHA-1 as
libFuzzer names corpus files:

- decode: every input of codec/golden.json and codec/noncanonical.json behind every type selector
  and each of the target's four limits, so that each input meets every decoder.
- envelope: every codec/envelope.json input under each of the 32 reader configurations, with the
  payload types the envelope vectors use; and every golden and non-canonical input as the payload
  of a probe envelope, which the target reseals (configuration bit 4).
- multiproof: every opening of hash/hash.json, at most its first 64 indices, in the target's tree
  of the same size; alone, and followed by each change the target's script knows.
- inclusion: every tree size and leaf index of the golden proofs in rfc6962/inclusion/ that the
  target's log holds; alone, and followed by each change.

The selectors are the indices of `TYPES` in src/lib.rs, read from the source so that the seeds
follow any change to the list. The proof targets' input formats are those of `multiproof_case`
and `inclusion_case` in src/proofs.rs.
"""

import hashlib
import json
import pathlib
import re
import sys

HERE = pathlib.Path(__file__).resolve().parent
VECTORS = HERE.parent.parent / "spec" / "vectors" / "codec"
HASH_VECTORS = HERE.parent.parent / "spec" / "vectors" / "hash" / "hash.json"
INCLUSION_VECTORS = HERE.parent.parent / "spec" / "vectors" / "rfc6962" / "inclusion"

# The decode target's limits, selected by its second byte.
DECODE_LIMITS = 4
# Envelope configuration bit 4: the target makes the declared length and the digest match.
RESEAL = 0b1_0000
# Payload types of envelope.json, and the type the target uses for the deeper Option cases.
ENVELOPE_PAYLOADS = ("Bytes", "Vec<u64>", "Option^8<u8>")
# The proof targets' scripts (src/proofs.rs): the changes each knows, and the most indices a
# multiproof case opens and entries an inclusion case's log holds.
MULTIPROOF_CHANGES = 16
INCLUSION_CHANGES = 12
MAX_OPENED = 64
MAX_ENTRIES = 260


def type_names():
    source = (HERE / "src" / "lib.rs").read_text()
    body = re.search(r"pub const TYPES: \[&str; \d+\] = \[(.*?)\];", source, re.S)
    if body is None:
        sys.exit("seeds.py: TYPES not found in src/lib.rs")
    return re.findall(r'"([^"]*)"', body.group(1))


def probe_envelope(payload):
    """A probe-schema envelope (0xFFF0, version 3) around `payload`, with a zero digest."""
    header = (
        b"TSRA"
        + (1).to_bytes(2, "little")
        + (0xFFF0).to_bytes(2, "little")
        + (3).to_bytes(2, "little")
        + bytes(2)
        + len(payload).to_bytes(8, "little")
    )
    return header + payload + bytes(32)


def u16(value):
    return value.to_bytes(2, "little")


def multiproof_seeds():
    """A tree size (log2 n), a count and the indices of each opening, then each change."""
    vectors = json.loads(HASH_VECTORS.read_text())
    for tree in vectors["merkle"]:
        log_n = len(tree["leaves"]).bit_length() - 1
        for opening in tree["openings"]:
            indices = opening["indices"][:MAX_OPENED]
            base = bytes([log_n, len(indices)]) + b"".join(u16(i) for i in indices)
            yield base
            for change in range(MULTIPROOF_CHANGES):
                yield base + bytes([change, 1, 0, 3])


def inclusion_seeds():
    """A log of `size` entries (written size - 1) and a leaf index, then each change."""
    shapes = set()
    for path in sorted(INCLUSION_VECTORS.rglob("*.json")):
        case = json.loads(path.read_text())
        if case["leafIdx"] < case["treeSize"] <= MAX_ENTRIES:
            shapes.add((case["treeSize"], case["leafIdx"]))
    for size, index in sorted(shapes):
        base = u16(size - 1) + u16(index)
        yield base
        for change in range(INCLUSION_CHANGES):
            yield base + bytes([change, 1, 0, 3])


def main():
    if len(sys.argv) != 2:
        sys.exit(__doc__)
    out = pathlib.Path(sys.argv[1])
    names = type_names()

    def write(target, data):
        directory = out / target
        directory.mkdir(parents=True, exist_ok=True)
        (directory / hashlib.sha1(data).hexdigest()).write_bytes(data)

    inputs = []
    for file, keys in (("golden.json", ("cases",)), ("noncanonical.json", ("cases", "accepted"))):
        document = json.loads((VECTORS / file).read_text())
        inputs += [bytes.fromhex(case["hex"]) for key in keys for case in document[key]]
    envelopes = json.loads((VECTORS / "envelope.json").read_text())["cases"]

    for data in inputs:
        for selector in range(len(names)):
            for limits in range(DECODE_LIMITS):
                write("decode", bytes([selector, limits]) + data)
            write("envelope", bytes([RESEAL, selector]) + probe_envelope(data))
    payloads = [names.index(name) for name in ENVELOPE_PAYLOADS]
    for case in envelopes:
        data = bytes.fromhex(case["hex"])
        for config in range(32):
            for selector in payloads:
                write("envelope", bytes([config, selector]) + data)

    for seed in multiproof_seeds():
        write("multiproof", seed)
    for seed in inclusion_seeds():
        write("inclusion", seed)

    for target in ("decode", "envelope", "multiproof", "inclusion"):
        print(f"{target}: {sum(1 for _ in (out / target).iterdir())} seeds")


if __name__ == "__main__":
    main()
