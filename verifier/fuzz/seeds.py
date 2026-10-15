#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0 OR MIT
"""Seed corpora for the fuzz targets, from the conformance vectors of spec/vectors/codec/.

Usage: python3 seeds.py OUT_DIR

Writes OUT_DIR/decode and OUT_DIR/envelope, one file per seed, named by its SHA-1 as libFuzzer
names corpus files:

- decode: every input of golden.json and noncanonical.json behind every type selector, so that
  each input meets every decoder.
- envelope: every envelope.json input under each of the 32 reader configurations, with the payload
  types the envelope vectors use; and every golden and non-canonical input as the payload of a
  probe envelope, which the target reseals (configuration bit 4).

The selectors are the indices of `TYPES` in src/lib.rs, read from the source so that the seeds
follow any change to the list.
"""

import hashlib
import json
import pathlib
import re
import sys

HERE = pathlib.Path(__file__).resolve().parent
VECTORS = HERE.parent.parent / "spec" / "vectors" / "codec"

# Configuration bit 4: the target makes the declared length and the digest match.
RESEAL = 0b1_0000
# Payload types of envelope.json, and the type the target uses for the deeper Option cases.
ENVELOPE_PAYLOADS = ("Bytes", "Vec<u64>", "Option^8<u8>")


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
            write("decode", bytes([selector]) + data)
            write("envelope", bytes([RESEAL, selector]) + probe_envelope(data))
    payloads = [names.index(name) for name in ENVELOPE_PAYLOADS]
    for case in envelopes:
        data = bytes.fromhex(case["hex"])
        for config in range(32):
            for selector in payloads:
                write("envelope", bytes([config, selector]) + data)

    for target in ("decode", "envelope"):
        print(f"{target}: {sum(1 for _ in (out / target).iterdir())} seeds")


if __name__ == "__main__":
    main()
