#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0 OR MIT
"""Seed corpora for the fuzz targets, from the conformance vectors of spec/vectors/.

Usage: python3 seeds.py OUT_DIR

Writes OUT_DIR/<target> for each target below, one file per seed, named by its SHA-1 as
libFuzzer names corpus files:

- decode: every input of codec/golden.json and codec/noncanonical.json behind every type selector
  and each of the target's four limits, so that each input meets every decoder; and the payload of
  every weight commitment artifact below as a WeightCommitmentPublic, under each limit.
- envelope: every codec/envelope.json input under each of the 64 reader configurations, with the
  payload types the envelope vectors use; every golden and non-canonical input as the payload of a
  probe envelope, which the target reseals (configuration bit 5); and every weight commitment
  artifact (schema 0x0301: the .twr files of weights/ and the 0x0301 cases of
  codec/weight-commitment.json) under each configuration, as a WeightCommitmentPublic.
- multiproof: every opening of hash/hash.json, at most its first 64 indices, in the target's tree
  of the same size; alone, and followed by each change the target's script knows.
- inclusion: every tree size and leaf index of the golden proofs in rfc6962/inclusion/ that the
  target's log holds; alone, and followed by each change.
- sumcheck: the shape of every stage and tree batch of sumcheck/sumcheck.json that the target's
  limits admit, with a seed for its tables or leaves; alone, and followed by each change.

The selectors are the indices of `TYPES` in src/lib.rs, read from the source so that the seeds
follow any change to the list. The proof targets' input formats are those of `multiproof_case`
and `inclusion_case` in src/proofs.rs, and of `case` in src/sumcheck.rs.
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
SUMCHECK_VECTORS = HERE.parent.parent / "spec" / "vectors" / "sumcheck" / "sumcheck.json"
WEIGHT_VECTORS = HERE.parent.parent / "spec" / "vectors" / "weights"

# The decode target's limits, selected by its second byte.
DECODE_LIMITS = 4
# The envelope target's configurations: bits 0-2 the reader, 3-4 the limits, and bit 5, with
# which the target makes the declared length and the digest match.
ENVELOPE_CONFIGURATIONS = 64
RESEAL = 0b10_0000
# The weight commitment's public artifact: its schema, and its envelope's header and digest.
WEIGHTS_PUBLIC = 0x0301
HEADER_LEN = 20
DIGEST_LEN = 32
# Payload types of envelope.json, and the type the target uses for the deeper Option cases.
ENVELOPE_PAYLOADS = ("Bytes", "Vec<u64>", "Option^8<u8>")
# The proof targets' scripts (src/proofs.rs): the changes each knows, and the most indices a
# multiproof case opens and entries an inclusion case's log holds.
MULTIPROOF_CHANGES = 16
INCLUSION_CHANGES = 12
MAX_OPENED = 64
MAX_ENTRIES = 260
# The sumcheck target (src/sumcheck.rs): its changes, and the largest batch it builds.
SUMCHECK_CHANGES = 16
SUMCHECK_MAX_INSTANCES = 4
SUMCHECK_MAX_VARS = 6
SUMCHECK_MAX_DEGREE = 7
SUMCHECK_MAX_HEIGHT = 5


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


def sumcheck_seeds():
    """Each stage and tree batch of the sumcheck vectors that fits the target, as its shape and a
    seed (the stage's index or the batch's, as 8 bytes), alone and followed by each change: a
    stage's degree, instances (variables, tables) and index; a batch's trees (height, the zero
    leaf's index or 255). A tree case's change starts with the level it touches."""
    vectors = json.loads(SUMCHECK_VECTORS.read_text())
    tail = bytes([1, 0, 2, 0, 3, 0]) + (5).to_bytes(16, "little")
    stages = [stage for session in vectors["sessions"] for stage in session["stages"]]
    for number, stage in enumerate(stages):
        shapes = [(i["vars"], i["factors"]) for i in stage["instances"]]
        degree = stage["degree"]
        if len(shapes) > SUMCHECK_MAX_INSTANCES or degree > SUMCHECK_MAX_DEGREE:
            continue
        if any(vars > SUMCHECK_MAX_VARS for vars, _ in shapes):
            continue
        index = int(stage["prefix"].split("/")[1])
        base = bytes([0, degree - 1, len(shapes) - 1])
        base += b"".join(bytes([vars, tables - 1]) for vars, tables in shapes)
        base += bytes([index % 256]) + number.to_bytes(8, "little")
        yield base
        for change in range(SUMCHECK_CHANGES):
            yield base + bytes([change]) + tail
    for number, batch in enumerate(vectors["tree_batches"]):
        trees = batch["trees"]
        heights = [tree["height"] for tree in trees]
        if len(trees) > SUMCHECK_MAX_INSTANCES or max(heights) > SUMCHECK_MAX_HEIGHT:
            continue
        zeros = []
        for tree in trees:
            zero = [i for i, leaf in enumerate(tree["leaves"]) if int(leaf, 16) == 0]
            zeros.append(zero[0] if zero else 255)
        base = bytes([1, len(trees) - 1])
        base += b"".join(bytes([h - 1, z]) for h, z in zip(heights, zeros))
        base += number.to_bytes(8, "little")
        yield base
        for level in range(max(heights)):
            for change in range(SUMCHECK_CHANGES):
                yield base + bytes([level, change]) + tail


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
        for config in range(ENVELOPE_CONFIGURATIONS):
            for selector in payloads:
                write("envelope", bytes([config, selector]) + data)
    public = names.index("WeightCommitmentPublic")
    artifacts = [path.read_bytes() for path in sorted(WEIGHT_VECTORS.glob("*.twr"))]
    commitments = json.loads((VECTORS / "weight-commitment.json").read_text())["cases"]
    artifacts += [
        bytes.fromhex(case["hex"]) for case in commitments if case["schema"] == WEIGHTS_PUBLIC
    ]
    if not artifacts:
        sys.exit("seeds.py: no weight commitment artifact found")
    for data in artifacts:
        for config in range(ENVELOPE_CONFIGURATIONS):
            write("envelope", bytes([config, public]) + data)
        for limits in range(DECODE_LIMITS):
            write("decode", bytes([public, limits]) + data[HEADER_LEN:-DIGEST_LEN])

    for seed in multiproof_seeds():
        write("multiproof", seed)
    for seed in inclusion_seeds():
        write("inclusion", seed)
    for seed in sumcheck_seeds():
        write("sumcheck", seed)

    for target in ("decode", "envelope", "multiproof", "inclusion", "sumcheck"):
        print(f"{target}: {sum(1 for _ in (out / target).iterdir())} seeds")


if __name__ == "__main__":
    main()
