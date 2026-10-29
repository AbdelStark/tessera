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
- protocol: every protocol/1 packet, malformed payload and boundary-valid payload under each
  limit, plus all valid and malformed envelopes.
- multiproof: every opening of hash/hash.json, at most its first 64 indices, in the target's tree
  of the same size; alone, and followed by each change the target's script knows.
- inclusion: every tree size and leaf index of the golden proofs in rfc6962/inclusion/ that the
  target's log holds; alone, and followed by each change.
- sumcheck: the shape of every stage and tree batch of sumcheck/sumcheck.json that the target's
  limits admit, with a seed for its tables or leaves; alone, and followed by each change;
  GDG-prefixed gadget cases cover each family, model variant, affine change and nonlinear final.
- audit: every sample of audit/audit.json, as a payload (selector 0) and as an envelope
  (selector 2), its accepted and refused payloads, an empty ledger, payload (1) and envelope
  (3), a ledger payload deriving each sample, and a pending toss over each sample's manifest and
  policy, whose share opens its commitment, payload (4) and envelope (5), each under both of the
  target's limits (src/audit.rs). The toss's commitment and envelope digest are H of
  hash/generate.py, the hash oracle the audit vectors use.

The selectors are the indices of `TYPES` in src/lib.rs, read from the source so that the seeds
follow any change to the list. The proof targets' input formats are those of `multiproof_case`
and `inclusion_case` in src/proofs.rs, and of `case` in src/sumcheck.rs.
"""

import hashlib
import importlib.util
import json
import pathlib
import re
import sys

sys.dont_write_bytecode = True  # importing the hash oracle leaves nothing behind

HERE = pathlib.Path(__file__).resolve().parent
VECTORS = HERE.parent.parent / "spec" / "vectors" / "codec"
HASH_VECTORS = HERE.parent.parent / "spec" / "vectors" / "hash" / "hash.json"
INCLUSION_VECTORS = HERE.parent.parent / "spec" / "vectors" / "rfc6962" / "inclusion"
SUMCHECK_VECTORS = HERE.parent.parent / "spec" / "vectors" / "sumcheck" / "sumcheck.json"
AUDIT_VECTORS = HERE.parent.parent / "spec" / "vectors" / "audit" / "audit.json"
WEIGHT_VECTORS = HERE.parent.parent / "spec" / "vectors" / "weights"
HASH_ORACLE = HERE.parent.parent / "spec" / "vectors" / "hash" / "generate.py"

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
SUMCHECK_MAX_DEGREE = 10
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


def hash_oracle():
    """spec/vectors/hash/generate.py, whose BLAKE3 is checked against the official vectors."""
    spec = importlib.util.spec_from_file_location("hash_oracle", HASH_ORACLE)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


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
    # RFC-0013: protocol/2 reduction stages of every newly admitted degree. Header bit 1
    # selects the protocol without changing the legacy protocol/1 seeds (header 0).
    for degree in range(8, SUMCHECK_MAX_DEGREE + 1):
        base = bytes([2, degree - 1, 2, 3, degree - 1, 1, degree - 1, 0, degree - 1, 0])
        base += degree.to_bytes(8, "little")
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



def gadget_seeds():
    """GDG scripts of src/gadgets.rs: bounded metadata, fields, then shape controls."""
    def cells(n,start=0):
        return b"".join((0xffffffffffffffff-i).to_bytes(8,"little")+(7+13*i).to_bytes(8,"little") for i in range(start,start+n))
    def tagged(data):
        return b"GDG"+data
    for family in range(9):
        yield tagged(bytes([family]))
    for bits in range(1,63):
        for limb in (1,8,16,32):
            n=(bits+limb-1)//limb
            yield tagged(bytes([0,bits-1,limb-1,0])+cells(n))
    for x,y in ((0,0),(65535,1),(1,65535),(65535,65535)):
        for mode in range(4):
            yield tagged(bytes([1])+x.to_bytes(2,"little")*2+y.to_bytes(2,"little")*2+bytes([mode]))
    for width in range(1,33):
        n=(width+7)//8
        for j in sorted({0,(2*width+62)//8-1,2*(n-1)+6}):
            for increment in (0,1):
                for shape in range(4):
                    yield tagged(bytes([2,width-1,j,increment])+cells(n+4)+bytes([shape]))
    for family,file,sites_key in ((3,"requant.json","sites"),(4,"roots.json","roots"),(5,"softmax.json",None)):
        cases=json.loads((HERE.parent.parent/"spec/vectors/gadget-finals"/file).read_text())["cases"]
        for ci,case in enumerate(cases):
            sites=case[sites_key] if sites_key else [case]
            for si in sorted({0,len(sites)-1}):
                header=bytes([family])+ci.to_bytes(2,"little")+(si.to_bytes(2,"little") if sites_key else b"")
                for mode in range(8):
                    yield tagged(header+bytes([mode])+bytes(16))
                base=header+bytes(17)
                if family==4:
                    root=sites[si];n=(root["width"]+7)//8;m=(2*root["width"]+62)//8
                    for coefficient in (0,m-1,m,2*m-1):
                        yield tagged(base+bytes([0])+coefficient.to_bytes(2,"little")+cells(2+n+4)+bytes(2))
                    for kind in (1,2):
                        variables=2+((case["dim"]-1).bit_length() if kind==2 else 0)
                        yield tagged(base+bytes([kind])+cells(2*variables+1)+bytes(2))
                if family==5:
                    t=(case["n"]-1).bit_length();h=(case["heads"]-1).bit_length();full=2*t+h;row=t+h
                    for kind in range(4):
                        count=1 if kind==3 else 2
                        data=base+bytes([kind])+cells(full+2+(full if kind<2 else row)+count)+bytes(2)
                        if kind==3:
                            for padding in (0,1):
                                yield tagged(data+cells(row)+bytes([padding]))
                        else:
                            yield tagged(data)
    for relation in range(4):
        yield tagged(bytes([6])+cells(3)+bytes([relation])+cells(11)+bytes(2))
    for vocab,rows in ((1,1),(3,3),(8,4)):
        v=(vocab-1).bit_length();r=(rows-1).bit_length();n=v+r
        for kind,count in ((0,3),(1,2),(2,1)):
            yield tagged(bytes([7,vocab-1,rows-1])+cells(n+r+3)+bytes([kind])+cells(n+count,17)+bytes(2))
    for role in range(4):
        yield tagged(bytes([8,role])+cells((1 if role==2 else 4)+1+2+1)+bytes(2))
    greedy = json.loads((HERE.parent.parent / "spec/vectors/greedy-boundary/greedy-boundary.json").read_text())["cases"]
    for case in range(len(greedy)):
        for kind in range(3):
            for shape in range(16):
                yield tagged(bytes([9, case, kind, 0, shape]))
            for flags in range(1, 8):
                yield tagged(bytes([9, case, kind, flags, 0]) + cells(32))


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

    protocols = json.loads((HERE.parent.parent / "spec/vectors/protocol/protocol.json").read_text())
    packet_source = (HERE / "src/protocol.rs").read_text()
    packet_body = re.search(r"pub const TYPES: \[&str; \d+\] = \[(.*?)\];", packet_source, re.S)
    if packet_body is None:
        sys.exit("seeds.py: protocol TYPES not found")
    packets = re.findall(r'"([^"]*)"', packet_body.group(1))
    for key in ("messages", "refused", "accepted"):
        for case in protocols[key]:
            selector = packets.index(case["type"])
            payload = bytes.fromhex(case["payload" if key == "messages" else "hex"])
            for config in range(4):
                write("protocol", bytes([selector, config]) + payload)
            if key == "messages":
                for config in range(4, 8):
                    write("protocol", bytes([selector, config]) + bytes.fromhex(case["envelope"]))
    for case in protocols["envelopes"]:
        kind = case["schema"]
        for config in range(4, 8):
            write("protocol", bytes([packets.index(kind), config]) + bytes.fromhex(case["hex"]))

    for seed in multiproof_seeds():
        write("multiproof", seed)
    for seed in inclusion_seeds():
        write("inclusion", seed)
    for seed in sumcheck_seeds():
        write("sumcheck", seed)
    for seed in gadget_seeds():
        write("sumcheck", seed)

    audit = json.loads(AUDIT_VECTORS.read_text())
    samples = [(0, bytes.fromhex(case["payload"])) for key in ("samples", "refused", "accepted")
               for case in audit[key]]
    samples += [(2, bytes.fromhex(case["envelope"])) for case in audit["samples"]]
    # The empty ledger: five empty sequences (RFC-0067), and its envelope (schema 0x0701,
    # version 1) with a zero digest, which the target refuses at the digest. Then a ledger that
    # derived each sample: three empty sequences, the sample's manifest and identity, and no
    # record.
    empty = bytes(40)
    for case in audit["samples"]:
        identity = bytes.fromhex(case["envelope"])[-DIGEST_LEN:]
        manifest = bytes.fromhex(case["manifest_digest"])
        samples.append((1, bytes(24) + (1).to_bytes(8, "little") + manifest + identity + bytes(8)))
    header = b"TSRA" + u16(1) + u16(0x0701) + u16(1) + u16(0) + len(empty).to_bytes(8, "little")
    samples += [(1, empty), (3, header + empty + bytes(DIGEST_LEN))]
    # A pending toss over each sample's manifest and policy (RFC-0067 'The pending toss'): the
    # payload's bytes before its coin toss and `selected`, then H("coin-commit", share) and a
    # share; and its envelope, schema 0x0702 version 1.
    H = hash_oracle().H
    for case in audit["samples"]:
        payload = bytes.fromhex(case["payload"])
        policy_end = len(payload) - 96 - 8 - 24 * len(case["selected"])
        share = bytes([len(payload) % 256]) * 32
        toss = payload[:policy_end] + H("coin-commit", share) + share
        header = b"TSRA" + u16(1) + u16(0x0702) + u16(1) + u16(0) + len(toss).to_bytes(8, "little")
        samples += [(4, toss), (5, header + toss + H("envelope", header + toss))]
    for selector, data in samples:
        for limits in range(2):
            write("audit", bytes([selector, limits]) + data)

    # Complete claim validation: raw payload/envelope, both protocol selections,
    # and exact semantic refusal controls. Only public fixture bytes are used.
    claims = json.loads((HERE.parent.parent / "spec/vectors/claims/claim-lists.json").read_text())
    for case in claims["cases"]:
        envelope = bytes.fromhex(case["hex"])
        payload = envelope[20:-32]
        write("claims", bytes([0]) + payload)
        write("claims", bytes([1]) + envelope)
        write("claims", bytes([8]) + (2).to_bytes(2, "little") + payload[2:])
    for mode in range(2, 8):
        write("claims", bytes([mode]))

    # Two complete NI contexts, each with immutable original caller pins. Tamper
    # edits stay bound to the tamper base rather than the other recorded proof.
    recorded = HERE.parent / "crates/tv-core/tests/fixtures/driver"
    tamper = HERE.parent.parent / "spec/vectors/tamper"
    for context, path in [(0, recorded / "ni.tnp"), (64, tamper / "ni.tnp")]:
        proof = path.read_bytes()
        write("verify", bytes([context | 0x30]) + proof)
        write("verify", bytes([context | 1]) + proof[20:-32])
        for mode in range(2, 9):
            write("verify", bytes([context | mode]))
    corpus = json.loads((tamper / "protocol-1.json").read_text())
    for index, case in enumerate(corpus["mutations"]):
        edits = case.get("edits", [])
        if edits and all(e["target"] == "ni.tnp" for e in edits):
            write("verify", bytes([0x49]) + index.to_bytes(2, "little"))

    for target in ("decode", "envelope", "protocol", "multiproof", "inclusion", "sumcheck", "audit", "claims", "verify"):
        print(f"{target}: {sum(1 for _ in (out / target).iterdir())} seeds")


if __name__ == "__main__":
    main()
