#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0 OR MIT
"""Reference vectors for the RFC-0004 Fiat-Shamir transcript: scripted sessions of absorbs,
squeezes, challenges and proofs of work, with the state after every step.

This script is the independent oracle. Its BLAKE3 and its `H(domain, bytes)` are those of
../hash/generate.py, written from the BLAKE3 specification and checked against the official test
vectors before anything is written. Everything else is written here from RFC-0004 'Transcript'
and RFC-0003's encoding table. It shares no code with crates/tessera-hash, which checks its
output, nor with the verifier's own transcript (verifier/crates/tv-hash, #52), which is to.

Usage: python3 spec/vectors/transcript/generate.py
(rewrites transcript.json)

Inputs come from SplitMix64 with a fixed seed, so any implementation can regenerate them.
"""

import importlib.util
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
_spec = importlib.util.spec_from_file_location("hash_oracle", HERE.parent / "hash" / "generate.py")
oracle = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(oracle)

H, P, require = oracle.H, oracle.P, oracle.require
SEED = 0x7E55E7A_7A45_0045  # fixed; changing it changes every vector
MODES = {"Live": 0, "NonInteractive": 1}  # declaration order of `Mode` (RFC-0004)


# ----- TCE (RFC-0003) --------------------------------------------------------------------------


def u16(x):
    return x.to_bytes(2, "little")


def u64(x):
    return x.to_bytes(8, "little")


def ascii_(text):
    data = text.encode("ascii")
    require(all(0x20 <= b <= 0x7E for b in data), f"{text!r} is not printable ASCII")
    return u64(len(data)) + data


def session_context(context):
    """TCE(SessionContext): protocol, mode, params, statement digest, session id."""
    digest = bytes.fromhex(context["statement_digest"])
    session = bytes.fromhex(context["session_id"])
    require(len(digest) == 32 and len(session) == 16, "digest and session id sizes")
    return (
        ascii_(context["protocol"])
        + u16(MODES[context["mode"]])
        + ascii_(context["params"])
        + digest
        + session
    )


# ----- The transcript (RFC-0004 'Transcript') ----------------------------------------------------


def words(out):
    return [int.from_bytes(out[8 * i : 8 * i + 8], "little") for i in range(4)]


def leading_zero_bits(digest):
    """Zero bits before the first one, most significant bit of byte 0 first."""
    return 256 - int.from_bytes(digest, "big").bit_length()


class Transcript:
    def __init__(self, context_tce):
        self.state = H("transcript-init", context_tce)
        self.counter = 0

    def absorb(self, label, tce):
        self.state = H("transcript-absorb", self.state + ascii_(label) + tce)

    def squeeze(self, label):
        out = H("transcript-squeeze", self.state + ascii_(label) + u64(self.counter))
        self.counter += 1
        self.state = H("transcript-ratchet", self.state + out)
        return out

    def challenge_fp(self, label):
        """The challenge, and how many words at or above P it passed over."""
        rejected = 0
        while True:
            for word in words(self.squeeze(label)):
                if word < P:
                    return word, rejected
                rejected += 1

    def challenge_fp2(self, label):
        """The challenge, and how many squeezes it rejected."""
        rejected = 0
        while True:
            c0, c1, _, _ = words(self.squeeze(label))
            if c0 < P and c1 < P:
                return [c0, c1], rejected
            rejected += 1

    def challenge_indices(self, label, log_n, count):
        require(0 <= log_n <= 64, "log_n")
        mask = (1 << log_n) - 1
        indices = []
        while len(indices) < count:
            for word in words(self.squeeze(label))[: count - len(indices)]:
                indices.append(word & mask)
        return indices

    def pow_meets(self, label, bits, nonce):
        digest = H("pow", self.state + ascii_(label) + u64(nonce))
        return leading_zero_bits(digest) >= bits

    def grind(self, label, bits):
        nonce = 0
        while not self.pow_meets(label, bits, nonce):
            nonce += 1
        self.absorb(label, u64(nonce))
        return nonce

    def check_grind(self, label, bits, nonce):
        valid = self.pow_meets(label, bits, nonce)
        if valid:
            self.absorb(label, u64(nonce))
        return valid


# ----- Sessions ---------------------------------------------------------------------------------


def fp2_tce(rng):
    return u64(oracle.canonical(rng)) + u64(oracle.canonical(rng))


def nth_nonce(transcript, label, bits, valid, n):
    """The `n`-th nonce (from 0) that meets `bits`, or that does not when `valid` is false."""
    nonce, seen = 0, 0
    while True:
        if transcript.pow_meets(label, bits, nonce) == valid:
            if seen == n:
                return nonce
            seen += 1
        nonce += 1


def run(name, context, script):
    """Plays `script` on a fresh transcript and records each step with the state after it."""
    context_tce = session_context(context)
    t = Transcript(context_tce)
    case = {
        "name": name,
        "context": context,
        "context_tce": context_tce.hex(),
        "state": t.state.hex(),
        "steps": [],
    }
    for step in script:
        op, label = step["op"], step["label"]
        record = dict(step)
        if op == "absorb":
            t.absorb(label, bytes.fromhex(step["tce"]))
        elif op == "squeeze":
            record["out"] = t.squeeze(label).hex()
        elif op in ("challenge_fp", "challenge_fp2"):
            value, rejected = getattr(t, op)(label)
            record["value"] = str(value) if op == "challenge_fp" else [str(c) for c in value]
            require(rejected == step.get("rejected", 0), f"{name}: {op} rejected {rejected}")
            record["rejected"] = rejected
        elif op == "challenge_indices":
            record["value"] = [str(i) for i in t.challenge_indices(label, step["log_n"], step["count"])]
        elif op == "grind":
            record["nonce"] = str(t.grind(label, step["bits"]))
        elif op == "check_grind":
            # The nonce is chosen against the current state: the first invalid one, or the second
            # valid one, so that a non-minimal nonce is shown accepted.
            nonce = nth_nonce(t, label, step["bits"], step["valid"], 0 if not step["valid"] else 1)
            record["nonce"] = str(nonce)
            require(t.check_grind(label, step["bits"], nonce) == step["valid"], "check_grind")
        else:
            raise SystemExit(f"generate.py: unknown op {op}")
        record["state"] = t.state.hex()
        record["counter"] = t.counter
        case["steps"].append(record)
    return case


def main():
    checked = oracle.check_official_vectors()
    rng = oracle.splitmix64(SEED)

    def context(mode, params):
        return {
            "protocol": "tessera-protocol/1",
            "mode": mode,
            "params": params,
            "statement_digest": oracle.digest_from(rng).hex(),
            "session_id": oracle.digest_from(rng)[:16].hex(),
        }

    round1 = bytes(next(rng) & 0xFF for _ in range(100))
    non_interactive = run(
        "a non-interactive session",
        context("NonInteractive", "ni-100/1"),
        [
            {"op": "absorb", "label": "round1", "tce": (u64(len(round1)) + round1).hex()},
            {"op": "grind", "label": "seed-pow", "bits": 12},
            {"op": "squeeze", "label": "seed"},
            {"op": "challenge_fp2", "label": "logup/alpha"},
            {"op": "challenge_fp2", "label": "logup/beta"},
            {"op": "absorb", "label": "tree/0/round/0", "tce": (u64(3) + fp2_tce(rng) + fp2_tce(rng) + fp2_tce(rng)).hex()},
            {"op": "challenge_fp2", "label": "tree/0/r/0"},
            {"op": "absorb", "label": "stage/0/zk-sum", "tce": fp2_tce(rng).hex()},
            {"op": "challenge_fp", "label": "stage/0/zk-rho"},
            {"op": "challenge_fp", "label": "stage/0/gamma"},
            {"op": "grind", "label": "query-pow", "bits": 8},
            {"op": "challenge_indices", "label": "pcs/advice/queries", "log_n": 22, "count": 9},
        ],
    )
    # A message of 2,408 bytes, longer than a BLAKE3 chunk: Vec<Fp> of 300 elements.
    combined = u64(300) + b"".join(u64(oracle.canonical(rng)) for _ in range(300))
    live = run(
        "a live session",
        context("Live", "live-64/1"),
        [
            {"op": "absorb", "label": "round1", "tce": (u64(0)).hex()},
            {"op": "absorb", "label": "seed", "tce": oracle.digest_from(rng).hex()},
            {"op": "challenge_fp2", "label": "logup/alpha"},
            {"op": "challenge_indices", "label": "pcs/weights/0/queries", "log_n": 0, "count": 3},
            {"op": "challenge_indices", "label": "pcs/weights/1/queries", "log_n": 64, "count": 5},
            {"op": "challenge_indices", "label": "pcs/weights/2/queries", "log_n": 20, "count": 0},
            {"op": "squeeze", "label": ""},
            {"op": "challenge_fp", "label": "tree/3/r/12"},
            {"op": "absorb", "label": "pcs/advice/combined", "tce": combined.hex()},
            {"op": "challenge_fp", "label": "pcs/advice/gamma"},
        ],
    )
    proof_of_work = run(
        "proofs of work",
        context("NonInteractive", "test-small/1"),
        [
            {"op": "grind", "label": "seed-pow", "bits": 0},
            {"op": "check_grind", "label": "query-pow", "bits": 8, "valid": False},
            {"op": "check_grind", "label": "query-pow", "bits": 8, "valid": True},
            {"op": "grind", "label": "seed-pow", "bits": 10},
            {"op": "squeeze", "label": "seed"},
        ],
    )

    # Inputs found by search that reach the rejection branches, which BLAKE3 reaches with probability
    # about 2^-32 per word: after these absorbs, the first squeeze's word 0, or its word 1, is at or
    # above P. The checks above fail if they ever stop doing so.
    rejecting = {
        "protocol": "tessera-protocol/1",
        "mode": "NonInteractive",
        "params": "test-small/1",
        "statement_digest": "5a" * 32,
        "session_id": "07" * 16,
    }
    word_0 = u64(1440573241).hex()
    word_1 = u64(457009896).hex()
    rejections = [
        run(
            "challenge_fp passes over a word at or above P",
            rejecting,
            [
                {"op": "absorb", "label": "x", "tce": word_0},
                {"op": "challenge_fp", "label": "c", "rejected": 1},
            ],
        ),
        run(
            "challenge_fp2 squeezes again when c0 is at or above P",
            rejecting,
            [
                {"op": "absorb", "label": "x", "tce": word_0},
                {"op": "challenge_fp2", "label": "c", "rejected": 1},
            ],
        ),
        run(
            "challenge_fp2 squeezes again when c1 is at or above P",
            rejecting,
            [
                {"op": "absorb", "label": "x", "tce": word_1},
                {"op": "challenge_fp2", "label": "c", "rejected": 1},
            ],
        ),
    ]

    document = {
        "description": (
            "Fiat-Shamir transcript vectors (RFC-0004 'Transcript'), written by generate.py, an "
            "oracle independent of the Rust crates. Each case builds a SessionContext from "
            "`context`, whose TCE encoding is `context_tce`, and starts from `state`. Each step "
            "then runs on the same transcript and leaves `state` and `counter` (squeezes so far). "
            "`absorb` absorbs the TCE bytes `tce` under `label`. `squeeze` returns `out`. "
            "`challenge_fp`, `challenge_fp2` and `challenge_indices` return `value` (decimal "
            "strings); `rejected` counts the words (`Fp`) or squeezes (`Fp2`) rejected for being "
            "at or above P, which the last three cases reach. `grind` finds `nonce` for `bits`. `check_grind` checks `nonce` against "
            "`bits`, giving `valid`: a valid nonce is absorbed, an invalid one leaves the state "
            "unchanged, and the valid nonce shown is not the smallest."
        ),
        "official_blake3_cases_checked": checked,
        "cases": [non_interactive, live, proof_of_work, *rejections],
    }
    text = json.dumps(document, indent=2) + "\n"
    (HERE / "transcript.json").write_text(text)
    print(f"wrote transcript.json: {len(document['cases'])} cases")


if __name__ == "__main__":
    main()
