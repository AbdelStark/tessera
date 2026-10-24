#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0 OR MIT
"""Reference vectors for audit sampling, RFC-0016 with RFC-0063: a period's candidates and their
lengths from two periods' logs, the rate thresholds of a policy file, the coin toss over a
manifest, the sampling seed, each request's selection value and decision, the AuditSample
artifact (schema 0x0403) with its envelope and the inputs a reader must refuse, and the batches
first-fit decreasing packs.

This script is the independent oracle. It is written from RFC-0016 ('Policy', 'Sample
derivation', 'Batching'), RFC-0063 'Which requests a period draws from', RFC-0059 (a frame's
length is 22 + 4n for n tokens), RFC-0004 (the coin-commit, audit-sample and audit-select
domains), RFC-0003 (the encoding, its limits and its envelope) and 03 §5 (the artifact catalog)
alone. Its BLAKE3 and H(domain, bytes) are those of ../hash/generate.py, whose BLAKE3 is checked
against the official test vectors before anything is written. It shares no code with crates/,
verifier/ or python/, which check its output.

Usage: python3 spec/vectors/audit/generate.py
(rewrites audit.json)

Inputs come from SplitMix64 with a fixed seed, so any implementation can regenerate them.
"""

import importlib.util
import json
import sys
from fractions import Fraction
from pathlib import Path

sys.dont_write_bytecode = True  # importing the hash oracle leaves nothing behind

HERE = Path(__file__).resolve().parent
_spec = importlib.util.spec_from_file_location("hash_oracle", HERE.parent / "hash" / "generate.py")
hash_oracle = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(hash_oracle)

H = hash_oracle.H
SEED = 0xA0D1_7516_0063_0403  # fixed; changing it changes every vector

AUDIT_SAMPLE_SCHEMA = 0x0403
SCHEMA_VERSION = 1
U64 = (1 << 64) - 1
U32 = (1 << 32) - 1

TRUNCATED = "ERR-CODEC-TRUNCATED"
NONCANONICAL = "ERR-CODEC-NONCANONICAL"
LIMIT = "ERR-CODEC-LIMIT"


def require(condition, message):
    if not condition:
        raise SystemExit(f"audit oracle: {message}")


def le(x, width):
    require(0 <= x < 1 << (8 * width), f"{x} does not fit {width} bytes")
    return x.to_bytes(width, "little")


class Inputs:
    def __init__(self, seed):
        self.rng = hash_oracle.splitmix64(seed)

    def u64(self):
        return next(self.rng)

    def below(self, n):
        """A uniform integer of [0, n), by rejection."""
        require(n > 0, "an empty range")
        limit = (1 << 64) - ((1 << 64) % n)
        while True:
            x = next(self.rng)
            if x < limit:
                return x % n

    def bytes(self, n):
        out = b""
        while len(out) < n:
            out += next(self.rng).to_bytes(8, "little")
        return out[:n]


# ----- Rates (RFC-0016 'Policy') ----------------------------------------------------------------
# A policy file gives each rate as a decimal: 0, 1, or 0 or 1 followed by a point and one to 19
# digits, at most 1. Its threshold over 2^64 is floor(rate * 2^64), capped at 2^64 - 1.

def threshold(rate):
    whole, _, frac = rate.partition(".")
    require(whole in ("0", "1"), f"rate {rate}")
    require(("." not in rate) or (1 <= len(frac) <= 19 and frac.isdigit() and frac.isascii()), f"rate {rate}")
    value = Fraction(int(whole)) + (Fraction(int(frac), 10 ** len(frac)) if frac else 0)
    require(value <= 1, f"rate {rate} above 1")
    return min((value * (1 << 64)).numerator // (value * (1 << 64)).denominator, U64)


RATES = [
    "0", "1", "1.0", "1.0000000000000000000", "0.5", "0.1", "0.004", "0.25",
    "0.0000000000000000001", "0.0000000000000000542", "0.9999999999999999999", "0.3333333333333333333",
]
REFUSED_RATES = ["", ".5", "0.", "2", "1.1", "1.0000000000000000001", "0.12345678901234567890", "-0.1", "0,5", "0.1e1", " 0.1", "01", "0.1 "]


def is_rate(text):
    """The grammar above, read on its own."""
    if text in ("0", "1"):
        return True
    if len(text) < 3 or text[0] not in "01" or text[1] != ".":
        return False
    frac = text[2:]
    if not (1 <= len(frac) <= 19) or not all("0" <= c <= "9" for c in frac):
        return False
    return text[0] == "0" or set(frac) == {"0"}


# ----- The encoding (RFC-0003, RFC-0016's structs field by field) -------------------------------

def tce_key(key):
    session, request = key
    return session + le(request, 8)


def tce_policy(policy):
    out = le(policy["period_ns"], 8) + le(len(policy["buckets"]), 8)
    for max_tokens, thr in policy["buckets"]:
        out += le(max_tokens, 4) + le(thr, 8)
    out += le(policy["min_batch_tokens"], 4) + le(policy["max_batch_tokens"], 4)
    out += le(policy["proof_deadline_ns"], 8)
    params = policy["params"].encode("ascii")
    return out + le(len(params), 8) + params


def tce_sample(sample):
    toss = sample["coin_toss"]
    out = sample["manifest_digest"] + tce_policy(sample["policy"])
    out += toss["verifier_commitment"] + toss["prover_share"] + toss["verifier_share"]
    out += le(len(sample["selected"]), 8)
    for key in sample["selected"]:
        out += tce_key(key)
    return out


def seal(schema_id, payload, version=SCHEMA_VERSION):
    """RFC-0003 'Envelope': magic, envelope version 1, schema, version, flags 0, length, payload,
    then H("envelope", everything before)."""
    header = b"TSRA" + le(1, 2) + le(schema_id, 2) + le(version, 2) + le(0, 2) + le(len(payload), 8)
    body = header + payload
    return body + H("envelope", body)


# ----- A separate reader of AuditSample, RFC-0003's order with RFC-0016's rules ----------------
# It records the offset of every check, so the refusals below are found, not assumed.

LIMITS = {"max_total_bytes": 8 << 30, "max_vec_len": 1 << 34, "max_depth": 32}


class Refused(Exception):
    def __init__(self, code, at):
        super().__init__(f"{code} at {at}")
        self.code, self.at = code, at


class Reader:
    def __init__(self, data):
        self.data, self.pos = data, 0

    def take(self, n):
        if len(self.data) - self.pos < n:
            raise Refused(TRUNCATED, self.pos)
        out = self.data[self.pos:self.pos + n]
        self.pos += n
        return out

    def uint(self, width):
        return int.from_bytes(self.take(width), "little")

    def length(self, min_size):
        at = self.pos
        n = self.uint(8)
        if n > LIMITS["max_vec_len"]:
            raise Refused(LIMIT, at)
        if n * min_size > len(self.data) - self.pos:
            raise Refused(TRUNCATED, at)
        return n


def read_policy(r):
    period_ns = r.uint(8)
    n = r.length(12)
    buckets = []
    for _ in range(n):
        at = r.pos
        max_tokens = r.uint(4)
        if buckets and max_tokens <= buckets[-1][0]:
            raise Refused(NONCANONICAL, at)
        buckets.append((max_tokens, r.uint(8)))
    min_batch = r.uint(4)
    at = r.pos
    max_batch = r.uint(4)
    if max_batch == 0 or 2 * min_batch > max_batch:
        raise Refused(LIMIT, at)
    deadline = r.uint(8)
    m = r.length(1)
    at = r.pos
    params = r.take(m)
    for i, b in enumerate(params):
        if not 0x20 <= b <= 0x7E:
            raise Refused(NONCANONICAL, at + i)
    return {
        "period_ns": period_ns, "buckets": buckets, "min_batch_tokens": min_batch,
        "max_batch_tokens": max_batch, "proof_deadline_ns": deadline, "params": params.decode("ascii"),
    }


def read_sample(data):
    r = Reader(data)
    manifest = r.take(32)
    policy = read_policy(r)
    toss = {"verifier_commitment": r.take(32), "prover_share": r.take(32), "verifier_share": r.take(32)}
    n = r.length(24)
    selected = []
    for _ in range(n):
        at = r.pos
        key = (r.take(16), r.uint(8))
        if selected and key <= selected[-1]:
            raise Refused(NONCANONICAL, at)
        selected.append(key)
    if r.pos != len(data):
        raise Refused(NONCANONICAL, r.pos)
    return {"manifest_digest": manifest, "policy": policy, "coin_toss": toss, "selected": selected}


def outcome(data):
    try:
        return read_sample(data)
    except Refused as e:
        return (e.code, e.at)


# ----- The candidates (RFC-0063 'Which requests a period draws from') ---------------------------
# An entry is a dict with its kind: open, carry, close (a session), frame (with request,
# direction, index and length) or end (with request).

def frame_tokens(length):
    """RFC-0059: the ciphertext of n tokens is 22 + 4n bytes. Any other length has no count."""
    if length < 22 or (length - 22) % 4:
        return None
    return (length - 22) // 4


def candidates(previous, current):
    """Every request with an end in the period, and every one with a frame in the previous period
    and no end in either, with its length over all its frames in both: the sum of their token
    counts, or none (2^64 - 1, longer than every bucket) when a frame has no count."""
    def ends(log):
        return {(e["session"], e["request"]) for e in log if e["kind"] == "end"}

    def framed(log):
        return {(e["session"], e["request"]) for e in log if e["kind"] == "frame"}

    prev = previous or []
    ended = ends(current)
    unterminated = framed(prev) - ends(prev) - ended
    out = []
    for key in sorted(ended | unterminated):
        total = 0
        for e in prev + current:
            if e["kind"] == "frame" and (e["session"], e["request"]) == key:
                n = frame_tokens(e["length"])
                total = U64 if (n is None or total == U64) else min(total + n, U64)
        out.append({"key": key, "tokens": total, "ended": key in ended})
    return out


def random_log(rng, sessions, requests):
    log = []
    for _ in range(rng.below(12)):
        kind = ["frame", "frame", "frame", "end", "end", "open", "carry", "close"][rng.below(8)]
        session = sessions[rng.below(len(sessions))]
        e = {"kind": kind, "session": session}
        if kind in ("frame", "end"):
            e["request"] = requests[rng.below(len(requests))]
        if kind == "frame":
            e["direction"] = ["In", "Out"][rng.below(2)]
            e["index"] = rng.below(3)
            # Mostly well-formed lengths, sometimes one no frame has.
            e["length"] = 22 + 4 * rng.below(40) if rng.below(8) else [0, 21, 23, 25][rng.below(4)]
        log.append(e)
    return log


# ----- Selection (RFC-0016 'Sample derivation') --------------------------------------------------

def sample_seed(toss, manifest):
    return H("audit-sample", toss["verifier_share"] + toss["prover_share"] + manifest)


def selection_value(seed, key):
    session, request = key
    return int.from_bytes(H("audit-select", seed + session + le(request, 8))[:8], "little")


def bucket_threshold(policy, tokens):
    """The threshold of the first bucket whose maximum is at least `tokens`; None past them all."""
    for max_tokens, thr in policy["buckets"]:
        if tokens <= max_tokens:
            return thr
    return None


def selected(policy, seed, key, tokens):
    thr = bucket_threshold(policy, tokens)
    return True if thr is None else selection_value(seed, key) < thr


# ----- Batching (RFC-0016 'Batching') -------------------------------------------------------------

def batches(policy, requests):
    """First-fit decreasing within each group, groups in increasing order. A request goes to the
    first batch, in the order batches were opened, that it fits under the maximum; a request
    longer than the maximum is a batch on its own, which takes no other."""
    out = []
    for group in sorted({g for _, g, _ in requests}):
        items = sorted(((t, k) for k, g, t in requests if g == group), key=lambda x: (-x[0], x[1]))
        opened = []  # [total, keys, alone]
        for tokens, key in items:
            if tokens > policy["max_batch_tokens"]:
                opened.append([tokens, [key], True])
                continue
            for b in opened:
                if not b[2] and b[0] + tokens <= policy["max_batch_tokens"]:
                    b[0] += tokens
                    b[1].append(key)
                    break
            else:
                opened.append([tokens, [key], False])
        for total, keys, _ in opened:
            out.append({"group": group, "requests": sorted(keys), "tokens": total})
    return out


def check_batches(policy, requests, packed):
    """The bounds RFC-0016 promises, read from the batches alone."""
    seen = [k for b in packed for k in b["requests"]]
    require(sorted(seen) == sorted(k for k, _, _ in requests), "every request in exactly one batch")
    tokens = {k: t for k, _, t in requests}
    for group in {b["group"] for b in packed}:
        small = 0
        for b in packed:
            if b["group"] != group:
                continue
            require(b["tokens"] == sum(tokens[k] for k in b["requests"]), "a batch's total")
            require(b["tokens"] <= policy["max_batch_tokens"] or len(b["requests"]) == 1, "over the maximum")
            small += b["tokens"] < policy["min_batch_tokens"]
        require(small <= 1, "two batches of one group under the minimum")


# ----- The vectors ----------------------------------------------------------------------------

def key_json(key):
    return {"session_id": key[0].hex(), "request_id": key[1]}


def policy_json(policy):
    return {
        "period_ns": policy["period_ns"],
        "buckets": [{"max_tokens": m, "threshold": t} for m, t in policy["buckets"]],
        "min_batch_tokens": policy["min_batch_tokens"],
        "max_batch_tokens": policy["max_batch_tokens"],
        "proof_deadline_ns": policy["proof_deadline_ns"],
        "params": policy["params"],
    }


def toss_json(toss):
    return {k: v.hex() for k, v in toss.items()}


def entry_json(e):
    out = dict(e)
    out["session"] = e["session"].hex()
    return out


def toss_from(rng):
    s_v, s_p = rng.bytes(32), rng.bytes(32)
    return {"verifier_commitment": H("coin-commit", s_v), "prover_share": s_p, "verifier_share": s_v}


POLICY = {
    "period_ns": 86_400 * 10**9,
    "buckets": [(4096, threshold("0.5")), (32768, threshold("0.25")), (131072, threshold("0.1"))],
    "min_batch_tokens": 16384,
    "max_batch_tokens": 262144,
    "proof_deadline_ns": 7 * 86_400 * 10**9,
    "params": "live-64/1",
}


def main():
    hash_oracle.check_official_vectors()
    rng = Inputs(SEED)

    # Rates.
    for r in RATES:
        require(is_rate(r), f"the grammar refuses {r!r}")
    for r in REFUSED_RATES:
        require(not is_rate(r), f"the grammar accepts {r!r}")
    rates = [{"rate": r, "threshold": threshold(r)} for r in RATES]
    require(threshold("1") == U64 and threshold("0") == 0 and threshold("0.5") == 1 << 63, "rate edges")

    # Candidates over random two-period logs, and every frame of the previous period's requests
    # either in a candidate of this period or in an ended candidate of the previous one.
    sessions = [rng.bytes(16) for _ in range(3)]
    requests = [0, 1, 7, U64]
    candidate_cases = []

    def frame(session, request, length, direction="In", index=0):
        return {"kind": "frame", "session": session, "request": request, "direction": direction,
                "index": index, "length": length}

    def end(session, request):
        return {"kind": "end", "session": session, "request": request}

    s0 = sessions[0]
    fixed = [
        # A frame with no token count makes the length 2^64 - 1, wherever it lies: before a
        # well-formed frame, after one, or in the previous period.
        (None, [frame(s0, 1, 21), frame(s0, 1, 26, "Out"), end(s0, 1)]),
        (None, [frame(s0, 1, 26), frame(s0, 1, 21, "Out"), end(s0, 1)]),
        ([frame(s0, 2, 23)], [frame(s0, 2, 26, "Out"), end(s0, 2)]),
        # Well-formed frames in both periods, no end: an unterminated candidate of 1 + 2 tokens.
        ([frame(s0, 3, 26)], [frame(s0, 3, 30, "Out")]),
        # An end before its request's frames, which still makes it a candidate.
        ([], [end(s0, 4), frame(s0, 4, 26)]),
    ]
    for previous, current in fixed:
        found = candidates(previous, current)
        candidate_cases.append({
            "previous": None if previous is None else [entry_json(e) for e in previous],
            "current": [entry_json(e) for e in current],
            "candidates": [{"key": key_json(c["key"]), "tokens": c["tokens"], "ended": c["ended"]} for c in found],
        })
    require([c["tokens"] for c in candidates(*fixed[0])] == [U64], "a malformed frame first")
    require([c["tokens"] for c in candidates(*fixed[1])] == [U64], "a malformed frame last")
    require([c["tokens"] for c in candidates(*fixed[2])] == [U64], "a malformed frame before the boundary")
    require([(c["tokens"], c["ended"]) for c in candidates(*fixed[3])] == [(3, False)], "unterminated")
    for _ in range(40):
        previous = random_log(rng, sessions, requests) if rng.below(4) else None
        current = random_log(rng, sessions, requests)
        found = candidates(previous, current)
        keys = {c["key"] for c in found}
        for e in previous or []:
            if e["kind"] != "frame":
                continue
            k = (e["session"], e["request"])
            ended_before = any(x["kind"] == "end" and (x["session"], x["request"]) == k for x in previous)
            require(k in keys or ended_before, "a frame of the previous period escapes")
        for e in current:
            if e["kind"] == "end":
                require((e["session"], e["request"]) in keys, "an end makes no candidate")
        candidate_cases.append({
            "previous": None if previous is None else [entry_json(e) for e in previous],
            "current": [entry_json(e) for e in current],
            "candidates": [{"key": key_json(c["key"]), "tokens": c["tokens"], "ended": c["ended"]} for c in found],
        })

    # Seeds and selection values.
    seeds = []
    for _ in range(4):
        toss, manifest = toss_from(rng), rng.bytes(32)
        seed = sample_seed(toss, manifest)
        keys = [(rng.bytes(16), rng.u64()) for _ in range(6)] + [(b"\0" * 16, 0), (b"\xff" * 16, U64)]
        seeds.append({
            "coin_toss": toss_json(toss),
            "manifest_digest": manifest.hex(),
            "sample_seed": seed.hex(),
            "values": [{"key": key_json(k), "value": selection_value(seed, k)} for k in keys],
        })

    # Samples: random candidates, the policy's buckets and one past them all, derived and encoded.
    samples = []
    for name, n in [("empty", 0), ("one", 1), ("small", 9), ("larger", 60)]:
        toss, manifest = toss_from(rng), rng.bytes(32)
        seed = sample_seed(toss, manifest)
        cands = []
        for _ in range(n):
            tokens = [rng.below(4097), 4096, 4097, rng.below(131073), 131072, 131073, U64][rng.below(7)]
            cands.append(((rng.bytes(16), rng.below(1 << 20)), tokens))
        chosen = sorted({k for k, t in cands if selected(POLICY, seed, k, t)})
        for k, t in cands:
            if t > 131072:
                require(k in chosen, "a candidate past every bucket is not selected")
        sample = {"manifest_digest": manifest, "policy": POLICY, "coin_toss": toss, "selected": chosen}
        payload = tce_sample(sample)
        require(read_sample(payload) == sample, f"{name}: the reader does not give the sample back")
        truncated = sorted({outcome(payload[:cut])[1] for cut in range(len(payload))})
        for cut in range(len(payload)):
            require(outcome(payload[:cut])[0] == TRUNCATED, f"{name}: a prefix of {cut} bytes")
        samples.append({
            "name": name,
            "candidates": [{"key": key_json(k), "tokens": t} for k, t in cands],
            "coin_toss": toss_json(toss),
            "manifest_digest": manifest.hex(),
            "policy": policy_json(POLICY),
            "sample_seed": seed.hex(),
            "selected": [key_json(k) for k in chosen],
            "payload": payload.hex(),
            "truncated_at": truncated,
            "envelope": seal(AUDIT_SAMPLE_SCHEMA, payload).hex(),
        })

    # Refusals, each built on the larger sample's encoding and found by the reader on its own.
    base = bytes.fromhex(samples[-1]["payload"])
    sample = read_sample(base)
    require(len(sample["selected"]) >= 3, "the larger sample selects three requests")
    policy_at = 32
    buckets_at = policy_at + 8 + 8
    after_buckets = buckets_at + 12 * len(POLICY["buckets"])
    selected_at = after_buckets + 4 + 4 + 8 + 8 + len(POLICY["params"]) + 96 + 8

    def with_policy(**change):
        p = dict(POLICY, **change)
        return tce_sample(dict(sample, policy=p))

    refusals = [
        ("selected-repeated", base[:selected_at + 24] + base[selected_at:selected_at + 24] + base[selected_at + 48:],
         (NONCANONICAL, selected_at + 24)),
        ("selected-decreasing",
         base[:selected_at] + base[selected_at + 24:selected_at + 48] + base[selected_at:selected_at + 24] + base[selected_at + 48:],
         (NONCANONICAL, selected_at + 24)),
        ("buckets-repeated", with_policy(buckets=[(4096, 1), (4096, 2)]), (NONCANONICAL, buckets_at + 12)),
        ("buckets-decreasing", with_policy(buckets=[(4096, 1), (4095, 2)]), (NONCANONICAL, buckets_at + 12)),
        ("batch-max-zero", with_policy(min_batch_tokens=0, max_batch_tokens=0), (LIMIT, after_buckets + 4)),
        ("batch-min-over-half", with_policy(min_batch_tokens=16385, max_batch_tokens=32769), (LIMIT, after_buckets + 4)),
        ("params-not-ascii", with_policy(params="live-64/\x7f"), (NONCANONICAL, after_buckets + 4 + 4 + 8 + 8 + 8)),
        ("trailing", base + b"\0", (NONCANONICAL, len(base))),
    ]
    accepted = [
        ("batch-min-half", with_policy(min_batch_tokens=16384, max_batch_tokens=32768)),
        ("batch-min-zero", with_policy(min_batch_tokens=0, max_batch_tokens=1)),
        ("buckets-none", with_policy(buckets=[])),
    ]
    refused_json = []
    for name, data, expected in refusals:
        require(outcome(data) == expected, f"{name}: the reader finds {outcome(data)}, built for {expected}")
        refused_json.append({"name": name, "payload": data.hex(), "code": expected[0], "at": expected[1]})
    accepted_json = []
    for name, data in accepted:
        require(not isinstance(outcome(data), tuple), f"{name}: refused")
        accepted_json.append({"name": name, "payload": data.hex()})

    # Batches: a set at the edges, then random ones.
    batch_cases = []
    k = lambda i: (b"\0" * 16, i)
    edges = [(k(1), 0, 10), (k(2), 0, 4), (k(3), 0, 0), (k(4), 0, 0), (k(5), 1, U64), (k(6), 1, 3), (k(7), 1, 10)]
    policy = dict(POLICY, min_batch_tokens=1, max_batch_tokens=10)
    packed = batches(policy, edges)
    check_batches(policy, edges, packed)
    require([b["tokens"] for b in packed] == [10, 4, U64, 10, 3], "the edge batches")
    batch_cases.append({
        "name": "edges",
        "min_batch_tokens": 1,
        "max_batch_tokens": 10,
        "requests": [{"key": key_json(kk), "group": g, "tokens": t} for kk, g, t in edges],
        "batches": [{"group": b["group"], "requests": [key_json(kk) for kk in b["requests"]], "tokens": b["tokens"]} for b in packed],
    })
    for name, lo, hi, n, groups in [
        ("one-group", 16384, 262144, 30, 1), ("groups", 16384, 65536, 40, 3),
        ("tight", 50, 100, 25, 2), ("oversized", 1000, 2000, 12, 1),
    ]:
        policy = dict(POLICY, min_batch_tokens=lo, max_batch_tokens=hi)
        reqs = []
        keys = set()
        while len(reqs) < n:
            key = (rng.bytes(16), rng.below(64))
            if key in keys:
                continue
            keys.add(key)
            tokens = rng.below(hi // 2 + 1) if rng.below(6) else hi + rng.below(hi)
            if rng.below(5) == 0 and reqs:
                tokens = reqs[rng.below(len(reqs))][2]  # ties
            reqs.append((key, rng.below(groups), tokens))
        packed = batches(policy, reqs)
        check_batches(policy, reqs, packed)
        batch_cases.append({
            "name": name,
            "min_batch_tokens": lo,
            "max_batch_tokens": hi,
            "requests": [{"key": key_json(k), "group": g, "tokens": t} for k, g, t in reqs],
            "batches": [{"group": b["group"], "requests": [key_json(k) for k in b["requests"]], "tokens": b["tokens"]} for b in packed],
        })

    document = {
        "description": (
            "Audit sampling vectors (RFC-0016 with RFC-0063), written by generate.py, an oracle "
            "independent of the Rust crates and the Python package. See README.md for each section."
        ),
        "rates": rates,
        "refused_rates": REFUSED_RATES,
        "candidates": candidate_cases,
        "seeds": seeds,
        "samples": samples,
        "refused": refused_json,
        "accepted": accepted_json,
        "batches": batch_cases,
    }
    (HERE / "audit.json").write_text(json.dumps(document, indent=1) + "\n")


if __name__ == "__main__":
    main()
