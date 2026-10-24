#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0 OR MIT
"""The measurements behind RFC-0061 'Motivation': what a sampling stream shared by many requests
lets an observer learn, and what per-prompt streams let a serving stack choose.

Three measurements, each from fixed seeds so that it prints the RFC's figures:

  shared          b = 0: every request under one key draws the same u_t. Each observed
                  (prompt, output) pair confines u_0 to one slice of the distribution; the slices
                  intersect, and the next prompt's first token is predicted.
  classes         b = 8: responses spread over 2^8 keyed classes the observer never sees. Two
                  responses of one class have slices that intersect at every position, so the
                  observer clusters them, learns each class's stream, and predicts a new
                  response's next token from the tokens seen so far.
  tokenizations   a tokenizer's ambiguity: the token sequences that spell the last words of one
                  text, and how many values of u_0's top byte they reach when each prompt draws
                  a stream of its own. It reads a SentencePiece-style tokenizer.json given on the
                  command line. RFC-0061's figure used a 32,000-entry Llama-style BPE
                  tokenizer.json with byte fallback, SHA-256
                  bcd04f0eadf90287bd26e1a183ac487d8a141b09b06aecb7725bbdd343640f2e.

The distributions are synthetic, softmaxes of Gaussian logits, and the observer holds them
exactly: an upper bound on what an observer with a proxy of D learns. `shared` and `classes` need
NumPy. Usage: python3 -I research/tools/sampling_streams.py {shared|classes|tokenizations PATH}
"""

import hashlib
import json
import sys

if sys.argv[1:2] != ["tokenizations"]:
    import numpy as np


def shared():
    """b = 0, over 4,096 tokens: u_0 learned from the intersection of observed slices."""
    vocab = 4096
    rng = np.random.default_rng(0x497)

    def distribution():
        scale = rng.uniform(1.0, 6.0)
        logits = rng.normal(0.0, scale, vocab)
        p = np.exp(logits - logits.max())
        return p / p.sum()

    def trial(n_obs, n_test, u):
        lo, hi = 0.0, 1.0
        for _ in range(n_obs):
            cdf = np.cumsum(distribution())
            y = int(np.searchsorted(cdf, u, side="right"))  # min{v : C(v) > u}
            below = cdf[y - 1] if y else 0.0
            lo, hi = max(lo, below), min(hi, cdf[y])
        hits = 0
        for _ in range(n_test):
            cdf = np.cumsum(distribution())
            guess = int(np.searchsorted(cdf, (lo + hi) / 2, side="right"))
            hits += guess == int(np.searchsorted(cdf, u, side="right"))
        return hi - lo, hits / n_test

    entropies = []
    for _ in range(200):
        p = distribution()
        nz = p[p > 0]
        entropies.append(float(-(nz * np.log2(nz)).sum()))
    print(f"mean entropy {np.mean(entropies):.2f} bits")
    for n_obs in (0, 1, 10, 100, 1000):
        widths, accs = [], []
        for _ in range(20):
            w, a = trial(n_obs, 200, rng.uniform())
            widths.append(w)
            accs.append(a)
        print(f"{n_obs:5d} observed responses: u_0 known to {-np.log2(np.mean(widths)):5.1f} bits, "
              f"next prompt's first token predicted {np.mean(accs):6.1%}")


def classes():
    """b = 8, over 1,024 tokens and 32 positions: classes clustered, then streams learned."""
    vocab, positions, bits = 1024, 32, 8
    count = 1 << bits
    rng = np.random.default_rng(0x4973)
    streams = rng.uniform(size=(count, positions))

    def cdfs(n):
        scale = rng.uniform(1.0, 6.0, size=(n, positions, 1))
        logits = rng.normal(0.0, 1.0, size=(n, positions, vocab)) * scale
        p = np.exp(logits - logits.max(axis=2, keepdims=True))
        p /= p.sum(axis=2, keepdims=True)
        return np.cumsum(p, axis=2), p

    def respond(n):
        cls = rng.integers(0, count, size=n)
        c, p = cdfs(n)
        u = streams[cls]
        y = np.minimum((c <= u[..., None]).sum(axis=2), vocab - 1)
        hi = np.take_along_axis(c, y[..., None], axis=2)[..., 0]
        below = np.take_along_axis(c, np.maximum(y - 1, 0)[..., None], axis=2)[..., 0]
        lo = np.where(y > 0, below, 0.0)
        return cls, lo, hi, c, y, p

    def cluster(lo, hi):
        LO, HI, size = np.zeros((0, positions)), np.zeros((0, positions)), []
        for i in range(lo.shape[0]):
            ok = (np.maximum(LO, lo[i]) < np.minimum(HI, hi[i])).all(axis=1)
            idx = np.flatnonzero(ok)
            if len(idx) == 0:
                LO, HI = np.vstack([LO, lo[i]]), np.vstack([HI, hi[i]])
                size.append(1)
            else:
                k = idx[np.argmax(np.array(size)[idx])]
                LO[k], HI[k] = np.maximum(LO[k], lo[i]), np.minimum(HI[k], hi[i])
                size[k] += 1
        return LO, HI, np.array(size, dtype=float)

    def predict(LO, HI, size, lo, hi, c, y, k):
        hits = 0
        for i in range(lo.shape[0]):
            if k:
                ok = (np.maximum(LO[:, :k], lo[i, :k]) < np.minimum(HI[:, :k], hi[i, :k])).all(axis=1)
            else:
                ok = np.ones(len(size), dtype=bool)
            if not ok.any():
                continue
            mid = (LO[ok, k] + HI[ok, k]) / 2
            guesses = (c[i, k][None, :] <= mid[:, None]).sum(axis=1)
            votes = np.bincount(np.minimum(guesses, vocab - 1), weights=size[ok], minlength=vocab)
            hits += int(np.argmax(votes)) == y[i, k]
        return hits / lo.shape[0]

    _, _, _, _, _, p = respond(200)
    entropy = -(p * np.log2(np.where(p > 0, p, 1))).sum(axis=2)
    print(f"mean entropy {entropy.mean():.2f} bits per position; "
          f"best guess under fresh randomness {p.max(axis=2).mean():.1%}")
    observed = 0
    lo_all, hi_all = np.zeros((0, positions)), np.zeros((0, positions))
    for target in (256, 1024, 4096, 16384):
        remaining = target - observed
        while remaining:
            n = min(remaining, 256)
            _, lo, hi, _, _, _ = respond(n)
            lo_all, hi_all = np.vstack([lo_all, lo]), np.vstack([hi_all, hi])
            remaining -= n
        observed = target
        LO, HI, size = cluster(lo_all, hi_all)
        _, tlo, thi, tc, ty, _ = respond(400)
        accs = {k: predict(LO, HI, size, tlo, thi, tc, ty, k) for k in (0, 1, 2, 4, 8)}
        print(f"{target:6d} observed responses, {len(size):5d} clusters: next token predicted "
              + ", ".join(f"after {k} seen {a:5.1%}" for k, a in accs.items()))


# The tokenizer.json behind RFC-0061's figure: 32,000 entries, Llama-style BPE with byte fallback.
TOKENIZER_SHA256 = "bcd04f0eadf90287bd26e1a183ac487d8a141b09b06aecb7725bbdd343640f2e"


def tokenizations(path):
    """The token sequences that spell the last words of one text, and the top bytes of u_0 they
    reach when each prompt draws its own stream: a keyed hash of the prompt, under two fixed
    keys."""
    data = open(path, "rb").read()
    digest = hashlib.sha256(data).hexdigest()
    if digest != TOKENIZER_SHA256:
        print(f"note: this tokenizer.json is {digest}, not the one RFC-0061 measured; "
              "its counts will differ")
    vocab = {k: v for k, v in json.loads(data)["model"]["vocab"].items()
             if not (k.startswith("<") and k.endswith(">"))}
    longest = max(len(k) for k in vocab)

    def segmentations(text):
        table = [[] for _ in range(len(text) + 1)]
        table[0] = [[]]
        for i in range(1, len(text) + 1):
            for j in range(max(0, i - longest), i):
                if table[j] and text[j:i] in vocab:
                    table[i].extend(s + [vocab[text[j:i]]] for s in table[j])
        return table[-1]

    words = "Summarize the following email in one sentence.".split(" ")
    head = segmentations("▁" + "▁".join(words[:-2]))[0]
    print(f"vocabulary of {len(vocab)} pieces")
    for run, key in enumerate((bytes(16), bytes(range(16)))):
        for tail in (1, 2):
            variants = segmentations("▁" + "▁".join(words[-tail:]))
            reached = set()
            for v in variants:
                prompt = [1] + head + v
                message = key + b"".join(x.to_bytes(4, "little") for x in prompt)
                reached.add(hashlib.sha256(message).digest()[3])
            print(f"run {run}: last {tail} word(s), {len(variants)} tokenizations: "
                  f"{len(reached)}/256 top bytes of u_0 with a stream per prompt, 1/256 at b = 0")


if __name__ == "__main__":
    command = sys.argv[1:2]
    if command == ["shared"]:
        shared()
    elif command == ["classes"]:
        classes()
    elif command == ["tokenizations"] and len(sys.argv) == 3:
        tokenizations(sys.argv[2])
    else:
        sys.exit(__doc__)
