#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0 OR MIT
"""The SC-4 tamper corpus (07 §4): mutations of a valid tiny proof, each with its verdict.

This script is the independent oracle. It parses the committed base messages with RFC-0003's
encoding and the protocol/1 message layouts of RFC-0014, chooses every mutation, and predicts
every verdict from RFC-0019's check order and the rejection rules of RFC-0008, RFC-0010,
RFC-0011, RFC-0033, RFC-0037, RFC-0043 and RFC-0045. It never runs Rust and reads no verdict
from a verifier. The base messages are synthetic producer output of the tessera-e2e `tamper-base`
example: inputs here, not evidence. Its envelope hash and transcript are those of
../transcript/generate.py.

Usage: python3 spec/vectors/tamper/generate.py [--check]
(rewrites protocol-1.json; --check compares instead)
"""

import argparse
import hashlib
import importlib.util
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
_spec = importlib.util.spec_from_file_location(
    "transcript_oracle", HERE.parent / "transcript" / "generate.py"
)
transcript = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(transcript)
H, P, require = transcript.H, transcript.P, transcript.require

# Schema ids (03 §5) of the base messages, by file.
BASES = {
    "ni.tnp": 0x0520,
    "live-round1.tce": 0x0510,
    "live-round2.tce": 0x0511,
    "live-round3.tce": 0x0513,
}
FP2, DIGEST = 16, 32


# ----- TCE reading (RFC-0003), with the byte range of every value -----------------------------


class Node:
    """A decoded value and the payload bytes `[lo, hi)` that encode it."""

    def __init__(self, value, lo, hi, field=False):
        self.value, self.lo, self.hi, self.field = value, lo, hi, field

    def __getitem__(self, key):
        return self.value[key]

    def __len__(self):
        return len(self.value)


class Reader:
    def __init__(self, data):
        self.data, self.at = data, 0

    def raw(self, n):
        start = self.at
        require(n <= len(self.data) - start, "truncated base")
        self.at += n
        return Node(self.data[start : self.at], start, self.at)

    def uint(self, n):
        node = self.raw(n)
        node.value = int.from_bytes(node.value, "little")
        return node

    def fp(self):
        node = self.uint(8)
        require(node.value < P, "non-canonical base field element")
        node.field = True
        return node

    def fp2(self):
        start = self.at
        return Node((self.fp().value, self.fp().value), start, self.at)

    def seq(self, read, min_size):
        start = self.at
        n = self.uint(8).value
        require(n * min_size <= len(self.data) - self.at, "sequence longer than its bytes")
        return Node([read() for _ in range(n)], start, self.at)

    def obj(self, fields):
        start = self.at
        return Node({name: read() for name, read in fields}, start, self.at)

    # protocol/1 messages, field by field in declaration order.

    def request(self):
        return self.obj(
            [
                ("id", lambda: self.uint(8)),
                ("session", lambda: self.raw(16)),
                ("prompt", lambda: self.uint(4)),
                ("outputs", lambda: self.uint(4)),
            ]
        )

    def predictor(self):
        start = self.at
        tag, decoding = self.uint(2), self.uint(2)
        require(tag.value == decoding.value == 0, "the base is Exact { Greedy }")
        return Node({"tag": tag, "decoding": decoding}, start, self.at)

    def round1(self):
        return self.obj(
            [
                ("protocol", lambda: self.uint(2)),
                ("session", lambda: self.raw(16)),
                ("claim_digest", lambda: self.raw(DIGEST)),
                ("weights_root", lambda: self.raw(DIGEST)),
                ("epoch", lambda: self.uint(4)),
                ("manifest", lambda: self.raw(DIGEST)),
                ("requests", lambda: self.seq(self.request, 32)),
                ("predictor", self.predictor),
                ("advice_root", lambda: self.raw(DIGEST)),
                ("layout_digest", lambda: self.raw(DIGEST)),
                ("token_bits", lambda: self.uint(1)),
                ("layer_bits", lambda: self.uint(1)),
            ]
        )

    def sumcheck(self):
        return self.obj(
            [
                ("rounds", lambda: self.seq(lambda: self.seq(self.fp2, FP2), 8)),
                ("evals", lambda: self.seq(self.fp2, FP2)),
            ]
        )

    def row(self):
        return self.obj(
            [("u", lambda: self.seq(self.fp2, FP2)), ("mask", lambda: self.seq(self.fp2, FP2))]
        )

    def weight_rows(self):
        return self.obj(
            [
                ("matrix", lambda: self.uint(1)),
                ("evals", lambda: self.seq(self.row, 16)),
                ("proximity", self.row),
            ]
        )

    def stage(self):
        return self.obj(
            [
                ("id", lambda: self.uint(2)),
                ("sumcheck", self.sumcheck),
                ("splits", lambda: self.seq(self.fp2, FP2)),
            ]
        )

    def group(self):
        return self.obj(
            [
                ("roots", lambda: self.seq(self.fp2, FP2)),
                ("levels", lambda: self.seq(self.sumcheck, 16)),
                ("values", lambda: self.seq(self.fp2, FP2)),
                ("splits", lambda: self.seq(self.fp2, FP2)),
            ]
        )

    def round2(self):
        # TreeProof's groups, table roots and merge sit between its levels and its splits
        # (RFC-0054 'Messages').
        return self.obj(
            [
                ("roots", lambda: self.seq(self.fp2, FP2)),
                ("values", lambda: self.seq(self.fp2, FP2)),
                ("levels", lambda: self.seq(self.sumcheck, 16)),
                ("groups", lambda: self.seq(self.group, 32)),
                ("table_roots", lambda: self.seq(self.fp2, FP2)),
                ("merge", self.sumcheck),
                ("splits", lambda: self.seq(self.fp2, FP2)),
                ("stages", lambda: self.seq(self.stage, 26)),
                ("advice", lambda: self.seq(self.row, 16)),
                ("weights", lambda: self.seq(self.weight_rows, 33)),
                ("scoring", lambda: self.seq(lambda: self.uint(8), 8)),
            ]
        )

    def columns(self):
        return self.seq(lambda: self.seq(self.fp, 8), 8)

    def siblings(self):
        return self.seq(lambda: self.raw(DIGEST), DIGEST)

    def round3(self):
        return self.obj(
            [
                ("columns", self.columns),
                ("siblings", self.siblings),
                (
                    "weights",
                    lambda: self.seq(
                        lambda: self.obj(
                            [
                                ("matrix", lambda: self.uint(1)),
                                ("columns", self.columns),
                                ("siblings", self.siblings),
                            ]
                        ),
                        17,
                    ),
                ),
            ]
        )

    def ni(self):
        return self.obj(
            [
                ("round1", self.round1),
                ("seed_nonce", lambda: self.uint(8)),
                ("round2", self.round2),
                ("query_nonce", lambda: self.uint(8)),
                ("round3", self.round3),
            ]
        )


def open_envelope(name):
    """RFC-0003 'Envelope': header, payload, digest; returns the payload and its parse."""
    data = (HERE / name).read_bytes()
    header = b"TSRA" + (1).to_bytes(2, "little") + BASES[name].to_bytes(2, "little")
    require(data[:12] == header + (1).to_bytes(2, "little") + bytes(2), f"{name} header")
    size = int.from_bytes(data[12:20], "little")
    require(len(data) == size + 52, f"{name} length")
    require(H("envelope", data[:-32]) == data[-32:], f"{name} digest")
    payload = data[20:-32]
    reader = Reader(payload)
    parse = {
        "ni.tnp": Reader.ni,
        "live-round1.tce": Reader.round1,
        "live-round2.tce": Reader.round2,
        "live-round3.tce": Reader.round3,
    }[name]
    tree = parse(reader)
    require(reader.at == len(payload), f"{name} has trailing bytes")
    return data, tree


# ----- Mutations -------------------------------------------------------------------------------


def changed(node):
    """The node's value, minimally and canonically changed: +1 on an integer, +1 modulo P on an
    Fp or an Fp2's c0, the low bit of a byte string's first byte."""
    if isinstance(node.value, bytes):
        return bytes([node.value[0] ^ 1]) + node.value[1:]
    if isinstance(node.value, tuple):
        c0, c1 = node.value
        return ((c0 + 1) % P).to_bytes(8, "little") + c1.to_bytes(8, "little")
    if node.field:
        return ((node.value + 1) % P).to_bytes(8, "little")
    return (node.value + 1).to_bytes(node.hi - node.lo, "little")


def u64(x):
    return x.to_bytes(8, "little")


def reject(code, **fields):
    """RFC-0043's verdict object: the reason's code and its payload fields."""
    return {"status": "reject", "reason": {"code": code, **fields}}


class Corpus:
    def __init__(self, files):
        self.files, self.cases = files, []

    def add(self, case_id, step, family, expected, edits, reseal=True):
        """`edits` are (target, node, after), (target, lo, hi, after) or a `shift`. With
        `reseal`, offsets count from the payload's first byte and each consumer recomputes
        `payload_len` and the digest; without, they count from the envelope's first byte and
        nothing is recomputed."""
        out = []
        for edit in edits:
            if edit[0] == "shift":
                _, target, vec, delta = edit
                require(reseal and len(vec) > 0, "a shift is a payload edit of a non-empty Vec")
                out.append(
                    {
                        "target": target,
                        "offset": vec.lo + 8,
                        "shift_fp2": {"count": len(vec), "delta": list(delta)},
                    }
                )
                continue
            if len(edit) == 3:
                target, node, after = edit
                lo, hi = node.lo, node.hi
            else:
                target, lo, hi, after = edit
            data = self.files[target]
            region = data[20:-32] if reseal else data
            before = region[lo:hi]
            require(before != after, f"{case_id}: an edit that changes nothing")
            out.append({"target": target, "offset": lo, "before": before.hex(), "after": after.hex()})
        out.sort(key=lambda e: (e["target"], -e["offset"]))
        for a, b in zip(out, out[1:]):
            if a["target"] == b["target"]:
                width = 16 * b["shift_fp2"]["count"] if "shift_fp2" in b else len(b["before"]) // 2
                require(b["offset"] + width <= a["offset"], f"{case_id} overlaps")
        self.cases.append(
            {
                "id": case_id,
                "step": step,
                "family": family,
                "reseal": reseal,
                "edits": out,
                "expected": expected,
            }
        )


def shift(target, vec, delta=(1, 0)):
    """Every Fp2 of `vec` plus `delta`, coordinate-wise modulo P: the consumer applies it."""
    return ("shift", target, vec, delta)


def round_length(target, proof, rounds):
    """The last round message one value short: (count edit, removal of its last value)."""
    last = rounds[-1]
    count = Node(len(last), last.lo, last.lo + 8)
    tail = last[len(last) - 1]
    return [(target, count, u64(len(last) - 1)), (target, tail, b"")]


def round2_cases(corpus, target, prefix, r2, tables):
    """Steps 7 to 9 of RFC-0019, which read only Round2. They come before any query, so a
    mutation here is judged identically in non-interactive and recorded live sessions."""
    roots = r2["roots"]
    require(len(roots) == 4 * len(tables), "four LogUp roots per instance (RFC-0008)")
    # 7.1: per instance (n_a, d_a, n_t, d_t), in ascending TableId (RFC-0045 'The check'). A
    # zero root (n, d) = (0, 0) keeps n_a·d_t + n_t·d_a = 0, so only d ≠ 0 rejects it.
    for i, table in enumerate(tables):
        for part, name in [(0, "lookup-root"), (2, "table-root")]:
            n, d = roots[4 * i + part], roots[4 * i + part + 1]
            require(d.value != (0, 0), "an honest denominator is nonzero")
            corpus.add(
                f"{prefix}-logup-{i}-{name}-zero",
                "7.1",
                "logup-root",
                reject("REJ-LOOKUP", table=table),
                [(target, n.lo, d.hi, bytes(2 * FP2))],
            )
        corpus.add(
            f"{prefix}-logup-{i}-identity",
            "7.1",
            "logup-root",
            reject("REJ-LOOKUP", table=table),
            [(target, roots[4 * i], changed(roots[4 * i]))],
        )
    # 7.2: RFC-0010 'Sumcheck messages' derives p_j(1) from the claim, so a round value that
    # changes is caught by the level's final check, round n_max; a short round message by its
    # own round; a round after round n_max by round n_max (RFC-0033).
    for level, proof in enumerate(r2["levels"].value):
        rounds, evals = proof["rounds"], proof["evals"]
        n = len(rounds)
        if n:
            corpus.add(
                f"{prefix}-tree-{level}-coefficient",
                "7.2",
                "tree-coefficient",
                reject("REJ-TREE", level=level, round=n),
                [(target, rounds[0][0], changed(rounds[0][0]))],
            )
            corpus.add(
                f"{prefix}-tree-{level}-round-length",
                "7.2",
                "tree-round-length",
                reject("REJ-TREE", level=level, round=n - 1),
                round_length(target, proof, rounds),
            )
        else:
            extra = u64(1) + u64(3) + bytes(3 * FP2)
            corpus.add(
                f"{prefix}-tree-{level}-extra-round",
                "7.2",
                "tree-round-length",
                reject("REJ-TREE", level=level, round=0),
                [(target, rounds, extra)],
            )
        # RFC-0010 order N0, N1, D0, D1: D0's partners N1 + λ·D1 are nonzero for honest trees.
        corpus.add(
            f"{prefix}-tree-{level}-evaluation",
            "7.2",
            "tree-evaluation",
            reject("REJ-TREE", level=level, round=n),
            [(target, evals[2], changed(evals[2]))],
        )
    # Stage 0's split check (RFC-0045 'The check'): the lookup values enter the key and value
    # claims, and the split values must give them; a mismatch is Sumcheck { 0, 0 }.
    for i, node in enumerate(r2["values"].value):
        corpus.add(
            f"{prefix}-logup-value-{i}",
            "7.3",
            "leaf-split",
            reject("REJ-SUMCHECK", stage=0, round=0),
            [(target, node, changed(node))],
        )
    splits = r2["splits"]
    for name, node in [("first", splits[0]), ("last", splits[len(splits) - 1])]:
        corpus.add(
            f"{prefix}-leaf-split-{name}",
            "7.3",
            "leaf-split",
            reject("REJ-SUMCHECK", stage=0, round=0),
            [(target, node, changed(node))],
        )
    # 8: the reduction stages, by the same message rules, numbered as their labels.
    for proof in r2["stages"].value:
        stage = proof["id"].value
        rounds, evals = proof["sumcheck"]["rounds"], proof["sumcheck"]["evals"]
        n = len(rounds)
        require(n > 0, f"stage {stage} has rounds")
        require(len(proof["splits"]) == 0, "the base's reduction stages split no claim")
        corpus.add(
            f"{prefix}-stage-{stage}-coefficient",
            "8",
            "stage-coefficient",
            reject("REJ-SUMCHECK", stage=stage, round=n),
            [(target, rounds[0][0], changed(rounds[0][0]))],
        )
        corpus.add(
            f"{prefix}-stage-{stage}-round-length",
            "8",
            "stage-round-length",
            reject("REJ-SUMCHECK", stage=stage, round=n - 1),
            round_length(target, proof["sumcheck"], rounds),
        )
        corpus.add(
            f"{prefix}-stage-{stage}-evaluation",
            "8",
            "stage-evaluation",
            reject("REJ-SUMCHECK", stage=stage, round=n),
            [(target, evals[0], changed(evals[0]))],
        )
    # 9: Σ γ·v = ⟨u'_g, eq(column sub-point)⟩ for every group, in order, advice first (RFC-0011
    # step 6). A sub-point can hold fixed bits of a tensor's offset, where eq gives some entries
    # weight 0, so one changed entry could reach step 11 instead. Shifting every entry by δ moves
    # the inner product by δ·Σ_j eq(x, j) = δ at any point x. The last advice row is proximity.
    advice = r2["advice"].value
    for group, row in enumerate(advice[:-1]):
        corpus.add(
            f"{prefix}-advice-group-{group}-combined",
            "9",
            "combined-row",
            reject("REJ-PCS-EVAL", commitment="Advice"),
            [shift(target, row["u"])],
        )
    for packet in r2["weights"].value:
        matrix = packet["matrix"].value
        for group, row in enumerate(packet["evals"].value):
            corpus.add(
                f"{prefix}-weight-{matrix}-group-{group}-combined",
                "9",
                "combined-row",
                reject("REJ-PCS-EVAL", commitment="Weight"),
                [shift(target, row["u"])],
            )


def statement_cases(corpus, target, prefix, r1):
    """Step 4: the proof's statement against the caller's, field by field in RFC-0019's order;
    then the advice commitment's layout digest (RFC-0031)."""
    fields = [
        ("claim_digest", "ClaimListDigest"),
        ("weights_root", "WeightsRoot"),
        ("epoch", "WeightsEpoch"),
        ("manifest", "ManifestDigest"),
        ("layout_digest", "LayoutDigest"),
    ]
    for key, field in fields:
        corpus.add(
            f"{prefix}-statement-{key}",
            "4",
            "statement",
            reject("REJ-STATEMENT", field=field),
            [(target, r1[key], changed(r1[key]))],
        )
    for key in ["id", "session", "prompt", "outputs"]:
        node = r1["requests"][0][key]
        corpus.add(
            f"{prefix}-request-{key}",
            "4",
            "statement",
            reject("REJ-STATEMENT", field="Batch"),
            [(target, node, changed(node))],
        )
    # RFC-0017: Mixture { decoding: Greedy, eps_log2: 1 }, a canonical other predictor.
    mixture = (1).to_bytes(2, "little") + (0).to_bytes(2, "little") + bytes([1])
    corpus.add(
        f"{prefix}-predictor",
        "4",
        "statement",
        reject("REJ-STATEMENT", field="Predictor"),
        [(target, r1["predictor"], mixture)],
    )


def render():
    base = json.loads((HERE / "base.json").read_text())
    files, trees = {}, {}
    for name in BASES:
        files[name], trees[name] = open_envelope(name)
    tables = base["lookup_tables"]
    require(tables == sorted(set(tables)), "instances in ascending TableId (RFC-0045)")
    corpus = Corpus(files)
    ni, live1 = trees["ni.tnp"], trees["live-round1.tce"]
    live2, live3 = trees["live-round2.tce"], trees["live-round3.tce"]
    require(ni["seed_nonce"].value == ni["query_nonce"].value == 0, "test-small grinds no bits")

    # Step 1: the prover's NiProof fails to decode (RFC-0043 'Malformed', RFC-0003 'Errors').
    ni_kind = BASES["ni.tnp"]
    root = ni["round2"]["roots"][0]
    corpus.add(
        "ni-decode-noncanonical-field",
        "1",
        "decode",
        reject("REJ-MALFORMED", artifact=ni_kind, error="ERR-CODEC-NONCANONICAL"),
        [("ni.tnp", root.lo, root.lo + 8, u64(P))],
    )
    end = ni.hi
    corpus.add(
        "ni-decode-trailing-byte",
        "1",
        "decode",
        reject("REJ-MALFORMED", artifact=ni_kind, error="ERR-CODEC-NONCANONICAL"),
        [("ni.tnp", end, end, b"\x00")],
    )
    corpus.add(
        "ni-decode-truncated",
        "1",
        "decode",
        reject("REJ-MALFORMED", artifact=ni_kind, error="ERR-CODEC-TRUNCATED"),
        [("ni.tnp", end - 1, end, b"")],
    )
    size = len(files["ni.tnp"])
    digest = files["ni.tnp"][size - 32 : size - 31]
    corpus.add(
        "ni-envelope-digest",
        "1",
        "decode",
        reject("REJ-MALFORMED", artifact=ni_kind, error="ERR-CODEC-DIGEST"),
        [("ni.tnp", size - 32, size - 31, bytes([digest[0] ^ 1]))],
        reseal=False,
    )

    # Step 1 too: RFC-0054's grouping, which this verifier checks no stage of yet ('Migration /
    # Rollout'), so any nonzero grouping is beyond its limits, as more than 14 group bits is
    # beyond the codec's. An ungrouped stage has no groups, no table roots and an empty merge,
    # lengths the grouping fixes ('Messages'), so a nonempty one is malformed.
    for prefix, r1, target, kind in [
        ("ni", ni["round1"], "ni.tnp", ni_kind),
        ("live", live1, "live-round1.tce", BASES["live-round1.tce"]),
    ]:
        require(r1["token_bits"].value == r1["layer_bits"].value == 0, "the base is ungrouped")
        for name, token, layer in [("token", 1, 0), ("layer", 0, 1), ("past-bound", 8, 7)]:
            corpus.add(
                f"{prefix}-leaf-groups-{name}",
                "1",
                "leaf-groups",
                reject("REJ-MALFORMED", artifact=kind, error="ERR-CODEC-LIMIT"),
                [(target, r1["token_bits"].lo, r1["layer_bits"].hi, bytes([token, layer]))],
            )
    for prefix, r2, target, kind in [
        ("ni", ni["round2"], "ni.tnp", ni_kind),
        ("live", live2, "live-round2.tce", BASES["live-round2.tce"]),
    ]:
        groups, table_roots, merge = r2["groups"], r2["table_roots"], r2["merge"]
        require(len(groups) == len(table_roots) == 0, "the base is ungrouped")
        require(len(merge["rounds"]) == len(merge["evals"]) == 0, "the base merges nothing")
        for name, vec, after in [
            # One group of four empty lists.
            ("group", groups, u64(1) + bytes(32)),
            ("table-root", table_roots, u64(1) + bytes(16)),
            ("merge-round", merge["rounds"], u64(1) + u64(0)),
            ("merge-eval", merge["evals"], u64(1) + bytes(16)),
        ]:
            corpus.add(
                f"{prefix}-ungrouped-{name}",
                "1",
                "leaf-groups",
                reject("REJ-MALFORMED", artifact=kind, error="ERR-CODEC-NONCANONICAL"),
                [(target, vec, after)],
            )

    # Step 4 in both sessions; the manifest digest is protocol/1's anchor-side input.
    statement_cases(corpus, "ni.tnp", "ni", ni["round1"])
    statement_cases(corpus, "live-round1.tce", "live", live1)

    # Step 6: the seed nonce is absorbed (INV-FS-1). Test-small grinds zero bits, so nonce 1
    # passes the predicate and changes every later challenge. Level 0 holds for any γ and λ on
    # honest roots, since g = n + λ·d at no variable; level 1 then meets a new line point and
    # fails its final check. Real 20-bit predicates: the `pow` boundaries below.
    corpus.add(
        "ni-seed-nonce-bound",
        "6",
        "fiat-shamir",
        reject("REJ-TREE", level=1, round=1),
        [("ni.tnp", ni["seed_nonce"], u64(1))],
    )

    # Steps 7 to 9 in both sessions.
    round2_cases(corpus, "ni.tnp", "ni", ni["round2"], tables)
    round2_cases(corpus, "live-round2.tce", "live", live2, tables)

    # Steps 11 and 12, recorded live only: the queries are fixed, so a Round2 change after the
    # last combined row leaves them and every column check unchanged (in NI it would move them).
    # Each queried set holds index 0, and both column checks report the smallest index that
    # fails (RFC-0011 'The column check'). A row's entries are its evaluations on H, its
    # codeword evaluations on the coset 7·L, disjoint from H (RFC-0011 'Encoding'). Changing
    # entry 0 adds δ·L_0, whose numerator X^k − 1 = ±7^k − 1 is nonzero on 7·L: every symbol of
    # the codeword moves, the first queried column among them.
    t2, t3 = "live-round2.tce", "live-round3.tce"
    openings = [("advice", live3, "Advice")]
    openings += [(f"weight-{p['matrix'].value}", p, "Weight") for p in live3["weights"].value]
    for name, packet, commitment in openings:
        for field, node in [("sibling", packet["siblings"][0]), ("symbol", packet["columns"][0][0])]:
            corpus.add(
                f"live-{name}-column-{field}",
                "11",
                "merkle-path",
                reject("REJ-PCS-MERKLE", commitment=commitment, column=0),
                [(t3, node, changed(node))],
            )
    proximity = [("advice", live2["advice"][len(live2["advice"]) - 1], "Advice")]
    proximity += [(f"weight-{p['matrix'].value}", p["proximity"], "Weight") for p in live2["weights"].value]
    for name, row, commitment in proximity:
        node = row["u"][0]
        corpus.add(
            f"live-{name}-proximity-row",
            "11",
            "proximity",
            reject("REJ-PCS-PROXIMITY", commitment=commitment, column=0),
            [(t2, node, changed(node))],
        )
    request = base["request"]
    uhat = live2["scoring"][0]
    require(uhat.value == 0, "an exact proof claims Û = 0")
    corpus.add(
        "live-exact-nonzero-uhat",
        "12",
        "scoring",
        reject("REJ-UHAT", request_id=request["request_id"], uhat_fixed=1, bound_fixed=0),
        [(t2, uhat, u64(1))],
    )

    # The first failure decides (RFC-0019 'Check order'): two mutations of adjacent checks.
    d_a0 = live2["roots"][1]
    levels = live2["levels"].value
    stages = {p["id"].value: p["sumcheck"] for p in live2["stages"].value}
    adv_row = live2["advice"][0]["u"]
    w_row = live2["weights"][0]["evals"][0]["u"]
    adv_prox = live2["advice"][len(live2["advice"]) - 1]["u"][0]
    split = live2["splits"][0]
    last = len(levels) - 1
    groups_lo, groups_hi = live1["token_bits"].lo, live1["layer_bits"].hi
    pairs = [
        ("leaf-groups-before-statement",
         reject("REJ-MALFORMED", artifact=BASES["live-round1.tce"], error="ERR-CODEC-LIMIT"),
         [("live-round1.tce", groups_lo, groups_hi, bytes([1, 0])),
          ("live-round1.tce", live1["manifest"], changed(live1["manifest"]))]),
        ("statement-before-logup", reject("REJ-STATEMENT", field="ManifestDigest"),
         [("live-round1.tce", live1["manifest"], changed(live1["manifest"])), (t2, d_a0, changed(d_a0))]),
        ("logup-before-tree", reject("REJ-LOOKUP", table=tables[0]),
         [(t2, d_a0, changed(d_a0)), (t2, levels[1]["evals"][2], changed(levels[1]["evals"][2]))]),
        ("tree-from-the-root", reject("REJ-TREE", level=1, round=1),
         [(t2, levels[1]["evals"][2], changed(levels[1]["evals"][2])),
          (t2, levels[2]["evals"][2], changed(levels[2]["evals"][2]))]),
        ("tree-before-split", reject("REJ-TREE", level=last, round=len(levels[last]["rounds"])),
         [(t2, levels[last]["evals"][2], changed(levels[last]["evals"][2])), (t2, split, changed(split))]),
        ("split-before-stages", reject("REJ-SUMCHECK", stage=0, round=0),
         [(t2, split, changed(split)), (t2, stages[1]["evals"][0], changed(stages[1]["evals"][0]))]),
        ("stages-in-order", reject("REJ-SUMCHECK", stage=1, round=len(stages[1]["rounds"])),
         [(t2, stages[1]["evals"][0], changed(stages[1]["evals"][0])),
          (t2, stages[2]["evals"][0], changed(stages[2]["evals"][0]))]),
        ("stages-before-openings", reject("REJ-SUMCHECK", stage=2, round=len(stages[2]["rounds"])),
         [(t2, stages[2]["evals"][0], changed(stages[2]["evals"][0])), shift(t2, adv_row)]),
        ("advice-before-weight-groups", reject("REJ-PCS-EVAL", commitment="Advice"),
         [shift(t2, adv_row), shift(t2, w_row)]),
        ("evaluations-before-columns", reject("REJ-PCS-EVAL", commitment="Weight"),
         [shift(t2, w_row), (t3, live3["siblings"][0], changed(live3["siblings"][0]))]),
        ("merkle-before-proximity", reject("REJ-PCS-MERKLE", commitment="Advice", column=0),
         [(t3, live3["columns"][0][0], changed(live3["columns"][0][0])), (t2, adv_prox, changed(adv_prox))]),
        ("proximity-before-scoring", reject("REJ-PCS-PROXIMITY", commitment="Advice", column=0),
         [(t2, adv_prox, changed(adv_prox)), (t2, uhat, u64(1))]),
    ]
    for name, expected, edits in pairs:
        corpus.add(f"live-order-{name}", "order", "check-order", expected, edits)
    # The grouping is judged once the proof has decoded, and before its ungrouped fields.
    ni_groups = (ni["round1"]["token_bits"].lo, ni["round1"]["layer_bits"].hi, bytes([1, 0]))
    corpus.add(
        "ni-order-decode-before-leaf-groups",
        "order",
        "check-order",
        reject("REJ-MALFORMED", artifact=ni_kind, error="ERR-CODEC-NONCANONICAL"),
        [("ni.tnp", *ni_groups), ("ni.tnp", ni.hi, ni.hi, b"\x00")],
    )
    corpus.add(
        "ni-order-leaf-groups-before-ungrouped",
        "order",
        "check-order",
        reject("REJ-MALFORMED", artifact=ni_kind, error="ERR-CODEC-LIMIT"),
        [("ni.tnp", *ni_groups), ("ni.tnp", ni["round2"]["groups"], u64(1) + bytes(32))],
    )
    ni_first, ni_manifest = ni["round2"]["roots"][1], ni["round1"]["manifest"]
    corpus.add(
        "ni-order-statement-before-logup",
        "order",
        "check-order",
        reject("REJ-STATEMENT", field="ClaimListDigest"),
        [
            ("ni.tnp", ni["round1"]["claim_digest"], changed(ni["round1"]["claim_digest"])),
            ("ni.tnp", ni_manifest, changed(ni_manifest)),
            ("ni.tnp", ni_first, changed(ni_first)),
        ],
    )

    # The real 20-bit predicates of ni-100 (RFC-0014 'Grinding'), on a fixed context: a valid
    # nonce and the next one, which fails, for both proof-of-work phases.
    pow_data = base["pow"]
    require(pow_data["bits"] == 20 and pow_data["params"] == "ni-100/1", "ni-100 grinds 20 bits")
    context = transcript.session_context(
        {
            "protocol": "tessera-protocol/1",
            "mode": "NonInteractive",
            "params": pow_data["params"],
            "statement_digest": pow_data["statement_digest"],
            "session_id": pow_data["session_id"],
        }
    )
    for case in pow_data["cases"]:
        label = {"seed": "seed-pow", "queries": "query-pow"}[case["phase"]]
        require(transcript.Transcript(context).check_grind(label, 20, case["valid"]), "valid nonce")
        require(not transcript.Transcript(context).check_grind(label, 20, case["invalid"]), "bad")

    ids = [c["id"] for c in corpus.cases]
    require(len(set(ids)) == len(ids), "case ids are unique")
    families = {}
    for case in corpus.cases:
        families[case["family"]] = families.get(case["family"], 0) + 1
    return {
        "schema": "tessera.tamper-corpus/1",
        "protocol": "tessera-protocol/1",
        "base": base,
        "base_sha256": {name: hashlib.sha256(files[name]).hexdigest() for name in sorted(files)},
        "coverage": {
            "lookup_instances": len(tables),
            "tree_levels": len(live2["levels"]),
            "stages": [p["id"].value for p in live2["stages"].value],
            "advice_groups": len(live2["advice"]) - 1,
            "weight_matrices": [p["matrix"].value for p in live2["weights"].value],
            "families": dict(sorted(families.items())),
            "cases": len(corpus.cases),
        },
        "mutations": corpus.cases,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    data = (json.dumps(render(), indent=1) + "\n").encode()
    target = HERE / "protocol-1.json"
    if args.check:
        require(target.read_bytes() == data, "stale: run python3 spec/vectors/tamper/generate.py")
    else:
        target.write_bytes(data)


if __name__ == "__main__":
    main()
