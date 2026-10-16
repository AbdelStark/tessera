#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0 OR MIT
"""Reference vectors for the sumchecks of RFC-0010: honest proofs of single and batched
sumchecks of products of multilinear tables, front-padded, and of batches of product trees proven
level by level from their roots to their leaves.

This script is the independent oracle: an honest prover written from RFC-0010 ('Multilinear
extensions', 'Tree levels', 'Batching within a stage', 'Sumcheck messages') and RFC-0014
'Transcript schedule' alone, computing with Python's arbitrary-precision integers. Its transcript
is the oracle of ../transcript/generate.py, whose BLAKE3 is checked against the official test
vectors before anything is written. It shares no code with crates/ or verifier/, which check its
output, and its algorithms differ from theirs:

- A smaller instance is padded literally: its tables are extended to 2^n_max entries, constant in
  the lowest n_max - n_i variables, and one plain sumcheck runs over the padded tables.
- Every claim, final evaluation and tree claim it writes is computed from its definition, a sum
  over the hypercube weighted by eq, and checked against the value the sumcheck reaches.
- Each round polynomial is evaluated at its challenge by Lagrange's formula, and checked against
  the sum of the tables bound to that challenge.

Usage: python3 spec/vectors/sumcheck/generate.py
(rewrites sumcheck.json)

Inputs come from SplitMix64 with the fixed seed below, so the file is reproducible byte for byte.
"""

import importlib.util
import json
import sys
from pathlib import Path

sys.dont_write_bytecode = True  # importing the oracles leaves nothing behind
HERE = Path(__file__).resolve().parent
_spec = importlib.util.spec_from_file_location("transcript_oracle", HERE.parent / "transcript" / "generate.py")
transcript = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(transcript)

hashing = transcript.oracle  # ../hash/generate.py: BLAKE3, H and the official-vector check
P, require = hashing.P, hashing.require
SEED = 0x7E55E7A_5C0C_0059  # fixed; changing it changes every vector
W = 7  # Fp2 = Fp[X]/(X^2 - W)

SINGLE_FACTORS = [1, 2, 3, 4]  # products of d tables, the sumcheck's degree d
SINGLE_VARS = list(range(13))  # over 0 to 12 variables


# ----- Fp2 = Fp[X]/(X^2 - 7) on pairs (c0, c1); Fp embeds as (x, 0) ---------------------------

ZERO, ONE = (0, 0), (1, 0)


def add(a, b):
    return ((a[0] + b[0]) % P, (a[1] + b[1]) % P)


def sub(a, b):
    return ((a[0] - b[0]) % P, (a[1] - b[1]) % P)


def mul(a, b):
    """(a0 + a1 X)(b0 + b1 X) with X^2 = W, schoolbook."""
    return ((a[0] * b[0] + W * a[1] * b[1]) % P, (a[0] * b[1] + a[1] * b[0]) % P)


def lift(x):
    return (x % P, 0)


def total(values):
    out = ZERO
    for v in values:
        out = add(out, v)
    return out


def product(values):
    out = ONE
    for v in values:
        out = mul(out, v)
    return out


def power(a, e):
    out = ONE
    for _ in range(e):
        out = mul(out, a)
    return out


def inverse_fp(x):
    require(x % P != 0, "no inverse of 0")
    return pow(x, P - 2, P)


# ----- Inputs ------------------------------------------------------------------------------------


class Inputs:
    """Uniform field elements from SplitMix64, by rejecting words at or above P."""

    def __init__(self, seed):
        self.words = hashing.splitmix64(seed)

    def word(self):
        return next(self.words)

    def fp(self):
        while True:
            x = self.word()
            if x < P:
                return x

    def fp2(self):
        return (self.fp(), self.fp())

    def bytes(self, n):
        return b"".join(self.word().to_bytes(8, "little") for _ in range(n // 8))

    def context(self):
        return {
            "protocol": "tessera-protocol/1",
            "mode": "NonInteractive",
            "params": "test-small/1",
            "statement_digest": self.bytes(32).hex(),
            "session_id": self.bytes(16).hex(),
        }


# ----- Definitions (RFC-0010 'Multilinear extensions') -----------------------------------------


def eq_at(r, x):
    """eq(r, x) for an integer x read as the Boolean point (x_1, ..., x_n), x_1 its lowest bit:
    the product of r_i where x_i = 1 and 1 - r_i where x_i = 0."""
    out = ONE
    for i, ri in enumerate(r):
        out = mul(out, ri if (x >> i) & 1 else sub(ONE, ri))
    return out


def eq_points(a, b):
    """eq(a, b) = prod_i (a_i b_i + (1 - a_i)(1 - b_i)) for two field points."""
    require(len(a) == len(b), "eq of points of different lengths")
    return product(add(mul(ai, bi), mul(sub(ONE, ai), sub(ONE, bi))) for ai, bi in zip(a, b))


def mle(table, r):
    """f(r) = sum over the hypercube of f[x] eq(r, x)."""
    require(len(table) == 1 << len(r), "a table of 2^n entries takes n coordinates")
    return total(mul(table[x], eq_at(r, x)) for x in range(len(table)))


def lagrange(values, r):
    """p(r) for the polynomial through (k, values[k]), k = 0..d, by Lagrange's formula."""
    out = ZERO
    for i, value in enumerate(values):
        numerator, denominator = ONE, 1
        for m in range(len(values)):
            if m != i:
                numerator = mul(numerator, sub(r, lift(m)))
                denominator = denominator * (i - m) % P
        out = add(out, mul(value, mul(numerator, lift(inverse_fp(denominator)))))
    return out


# ----- TCE and the transcript (RFC-0003, RFC-0004, RFC-0014) -------------------------------------


def tce(values):
    """TCE(Vec<Fp2>): the u64 length, then c0 and c1 of each element, 8 bytes little-endian."""
    out = len(values).to_bytes(8, "little")
    for c0, c1 in values:
        out += c0.to_bytes(8, "little") + c1.to_bytes(8, "little")
    return out


def challenge(t, label):
    value, _ = t.challenge_fp2(label)
    return (value[0], value[1])


# ----- The prover --------------------------------------------------------------------------------


def batched_sum(weights, tables):
    """sum_i gamma^i sum_x prod_k T_ik[x] over tables of one size."""
    out = ZERO
    for weight, factors in zip(weights, tables):
        size = len(factors[0])
        out = add(out, mul(weight, total(product(t[x] for t in factors) for x in range(size))))
    return out


def sumcheck(t, prefix, degree, instances):
    """The honest prover of one batched sumcheck, from 'Batching within a stage' and 'Sumcheck
    messages'. Each instance is (tables, vars): its relation is the product of its tables, each
    of 2^vars entries, and its claim the sum of that product over its hypercube. Returns gamma,
    the round messages, the challenges, the claims claim_1 .. claim_{n_max + 1}, and each
    instance's tables bound to all n_max challenges."""
    n_max = max(n for _, n in instances)
    for tables, n in instances:
        require(all(len(table) == 1 << n for table in tables), "tables of 2^n entries")
        require(1 <= len(tables) <= degree, "a relation of degree at most the stage's")
    gamma = challenge(t, f"{prefix}/gamma")
    weights = [power(gamma, i) for i in range(len(instances))]
    # Front-padding, literally: T'[idx] = T[idx >> (n_max - n_i)], constant in the low variables.
    padded = [[[table[idx >> (n_max - n)] for idx in range(1 << n_max)] for table in tables] for tables, n in instances]
    claim = batched_sum(weights, padded)
    own = [total(product(table[x] for table in tables) for x in range(1 << n)) for tables, n in instances]
    require(claim == total(mul(w, mul(lift(1 << (n_max - n)), c)) for w, c, (_, n) in zip(weights, own, instances)),
            "the padded batch sums to sum_i gamma^i 2^(n_max - n_i) c_i")
    claims, rounds, challenges = [claim], [], []
    for j in range(n_max):
        half = 1 << (n_max - j - 1)

        def at(x_j):
            """p_j(x_j): the batched sum with x_j fixed and the later variables summed."""
            out = ZERO
            for weight, factors in zip(weights, padded):
                terms = (product(add(f[2 * x], mul(x_j, sub(f[2 * x + 1], f[2 * x]))) for f in factors) for x in range(half))
                out = add(out, mul(weight, total(terms)))
            return out

        values = [at(lift(k)) for k in range(degree + 1)]
        require(add(values[0], values[1]) == claims[-1], f"{prefix} round {j}: p(0) + p(1) = claim")
        message = [values[0]] + values[2:]
        t.absorb(f"{prefix}/round/{j}", tce(message))
        r = challenge(t, f"{prefix}/r/{j}")
        padded = [[[add(f[2 * x], mul(r, sub(f[2 * x + 1], f[2 * x]))) for x in range(half)] for f in factors] for factors in padded]
        claim = batched_sum(weights, padded)
        require(lagrange(values, r) == claim, f"{prefix} round {j}: p(r) is the bound tables' sum")
        rounds.append(message)
        challenges.append(r)
        claims.append(claim)
    bound = [[f[0] for f in factors] for factors in padded]
    return gamma, rounds, challenges, claims, bound


def suffix(challenges, n):
    """The last n challenges: an instance's own point (front-padding)."""
    return challenges[len(challenges) - n:]


def prove_stage(t, rng, prefix, degree, shapes):
    """A stage of product instances, each `(vars, factors)` with random Fp2 tables."""
    instances = [([[rng.fp2() for _ in range(1 << n)] for _ in range(k)], n) for n, k in shapes]
    gamma, rounds, challenges, claims, bound = sumcheck(t, prefix, degree, instances)
    evals, points, values, records = [], [], [], []
    for (tables, n), bound_i in zip(instances, bound):
        point = suffix(challenges, n)
        own = [mle(table, point) for table in tables]
        require(own == bound_i, f"{prefix}: each table's MLE at its point is the bound table")
        evals += own
        points.append(point)
        values.append(product(own))
        records.append({"vars": n, "factors": len(tables), "claim": fp2(total(product(tb[x] for tb in tables) for x in range(1 << n)))})
    t.absorb(f"{prefix}/evals", tce(evals))
    weights = [power(gamma, i) for i in range(len(instances))]
    require(total(mul(w, v) for w, v in zip(weights, values)) == claims[-1], f"{prefix}: the final check holds")
    return {
        "prefix": prefix,
        "degree": degree,
        "instances": records,
        "gamma": fp2(gamma),
        "rounds": [elements(m) for m in rounds],
        "challenges": elements(challenges),
        "claims": elements(claims),
        "final_evals": elements(evals),
        "points": [elements(p) for p in points],
        "values": elements(values),
        "state": t.state.hex(),
        "counter": t.counter,
    }


# ----- Product trees ('Tree levels') -------------------------------------------------------------


def tree_levels(leaves):
    """Every level of a product tree, the root first: V_l[x] = V_{l+1}[2x] V_{l+1}[2x + 1]."""
    levels = [leaves]
    while len(levels[0]) > 1:
        below = levels[0]
        levels.insert(0, [mul(below[2 * x], below[2 * x + 1]) for x in range(len(below) // 2)])
    return levels


def prove_trees(t, heights_and_leaves):
    """A batch of product trees, level by level from the roots (RFC-0010 'Tree levels'). The
    instances present at a level are the trees whose root level has been reached, in the order
    of the batch's list: a tree of height h joins at level H - h with its root claim."""
    trees = [tree_levels(leaves) for leaves in heights_and_leaves]
    heights = [len(levels) - 1 for levels in trees]
    tallest = max(heights)
    current = {}  # tree -> (own level, point, value)
    records = []
    for level in range(tallest):
        for tree, h in enumerate(heights):
            if tallest - h == level:
                current[tree] = (0, [], trees[tree][0][0])
        present = [tree for tree in range(len(trees)) if tree in current]
        prefix = f"tree/{level}"
        instances, entering = [], []
        for tree in present:
            own, point, value = current[tree]
            require(own == level - (tallest - heights[tree]), "a tree has l - (H - h) variables at level l")
            require(len(point) == own and mle(trees[tree][own], point) == value, f"{prefix}: each claim is its level's MLE at its point")
            below = trees[tree][own + 1]
            eq_table = [eq_at(point, x) for x in range(1 << own)]
            left = [below[2 * x] for x in range(1 << own)]
            right = [below[2 * x + 1] for x in range(1 << own)]
            require(total(mul(eq_table[x], mul(left[x], right[x])) for x in range(1 << own)) == value,
                    f"{prefix}: the level relation sums to the claim")
            instances.append(([eq_table, left, right], own))
            entering.append({"tree": tree, "point": elements(point), "value": fp2(value)})
        gamma, rounds, challenges, claims, bound = sumcheck(t, prefix, 3, instances)
        evals, nexts = [], []
        weights = [power(gamma, i) for i in range(len(present))]
        terms = []
        for tree, (tables, own), bound_i in zip(present, instances, bound):
            point = suffix(challenges, own)
            _, entry_point, _ = current[tree]
            u, w = mle(tables[1], point), mle(tables[2], point)
            require([eq_points(entry_point, point), u, w] == bound_i, f"{prefix}: eq, u and w are the bound tables")
            evals += [u, w]
            terms.append(mul(eq_points(entry_point, point), mul(u, w)))
        t.absorb(f"{prefix}/evals", tce(evals))
        require(total(mul(wt, term) for wt, term in zip(weights, terms)) == claims[-1], f"{prefix}: the final check holds")
        rho = challenge(t, f"{prefix}/line")
        for i, tree in enumerate(present):
            own = current[tree][0]
            u, w = evals[2 * i], evals[2 * i + 1]
            point = [rho] + suffix(challenges, own)
            value = add(u, mul(rho, sub(w, u)))
            require(mle(trees[tree][own + 1], point) == value, f"{prefix}: the next claim is the next level's MLE at rho || r'")
            current[tree] = (own + 1, point, value)
            nexts.append({"tree": tree, "point": elements(point), "value": fp2(value)})
        records.append({
            "prefix": prefix,
            "instances": entering,
            "gamma": fp2(gamma),
            "rounds": [elements(m) for m in rounds],
            "challenges": elements(challenges),
            "claims": elements(claims),
            "final_evals": elements(evals),
            "rho": fp2(rho),
            "next": nexts,
            "state": t.state.hex(),
            "counter": t.counter,
        })
    for tree, (own, point, value) in current.items():
        require(own == heights[tree] and mle(trees[tree][-1], point) == value, "every leaf claim is the leaves' MLE at its point")
    described = [{"height": h, "root": fp(levels[0][0]), "leaves": [fp(v) for v in levels[-1]]} for h, levels in zip(heights, trees)]
    return described, records


# ----- Rendering ---------------------------------------------------------------------------------


def hex64(x):
    return f"0x{x:016x}"


def fp(a):
    """An Fp element, as Fp2 (a, 0)."""
    require(a[1] == 0, "a base-field value")
    return hex64(a[0])


def fp2(a):
    return [hex64(a[0]), hex64(a[1])]


def elements(values):
    return [fp2(v) for v in values]


def compact(value):
    return json.dumps(value, separators=(",", ":"))


def render(header, sessions, batches):
    text = json.dumps(header, indent=2)[:-2] + ',\n  "sessions": [\n'
    text += ",\n".join(
        "    {" + f'"name":{compact(s["name"])},"context":{compact(s["context"])},"stages":[\n'
        + ",\n".join("      " + compact(stage) for stage in s["stages"]) + "\n    ]}"
        for s in sessions
    )
    text += '\n  ],\n  "tree_batches": [\n'
    text += ",\n".join(
        "    {" + f'"name":{compact(b["name"])},"context":{compact(b["context"])},"trees":[\n'
        + ",\n".join("      " + compact(tree) for tree in b["trees"]) + '\n    ],"levels":[\n'
        + ",\n".join("      " + compact(level) for level in b["levels"]) + "\n    ]}"
        for b in batches
    )
    text += "\n  ]\n}\n"
    require(json.loads(text) == {**header, "sessions": sessions, "tree_batches": batches}, "the file parses back")
    return text


# ----- Vectors -----------------------------------------------------------------------------------


def session(rng, name, stages):
    context = rng.context()
    t = transcript.Transcript(transcript.session_context(context))
    return {"name": name, "context": context, "stages": [prove_stage(t, rng, prefix, d, shapes) for prefix, d, shapes in stages]}


def tree_batch(rng, name, shapes):
    """`shapes` lists each tree's height, and the index of a leaf set to zero, or None."""
    context = rng.context()
    t = transcript.Transcript(transcript.session_context(context))
    leaves = []
    for height, zero_at in shapes:
        values = [lift(rng.fp()) for _ in range(1 << height)]
        if zero_at is not None:
            values[zero_at] = ZERO
        leaves.append(values)
    trees, levels = prove_trees(t, leaves)
    for (height, zero_at), tree in zip(shapes, trees):
        require((tree["root"] == hex64(0)) == (zero_at is not None), "only a tree with a zero leaf has root 0")
    return {"name": name, "context": context, "trees": trees, "levels": levels}


def main():
    checked = hashing.check_official_vectors()
    rng = Inputs(SEED)
    sessions = [
        session(rng, f"single instances, products of {d} table{'s' * (d > 1)} on 0 to 12 variables", [(f"stage/{n}", d, [(n, d)]) for n in SINGLE_VARS])
        for d in SINGLE_FACTORS
    ]
    sessions += [
        session(rng, "a batch of 3, 7 and 12 variables", [("stage/0", 3, [(3, 3), (7, 2), (12, 3)])]),
        session(rng, "a batch of 12, 3, 7 and 0 variables, products of 2, 4, 1 and 3 tables", [("stage/5", 4, [(12, 2), (3, 4), (7, 1), (0, 3)])]),
        session(rng, "a batch of equal sizes, without padding", [("stage/1", 2, [(5, 2), (5, 2), (5, 1)])]),
        session(rng, "batches of n_max = 0, then of a single instance, on one transcript", [
            ("stage/0", 3, [(0, 1), (0, 2), (0, 3)]),
            ("stage/1", 2, [(6, 2)]),
            ("stage/2", 1, [(0, 1)]),
        ]),
        session(rng, "degrees 5 to 7: products of 5, 6 and 7 tables, then a batch of degree 7", [
            ("stage/0", 5, [(3, 5)]),
            ("stage/1", 6, [(3, 6)]),
            ("stage/2", 7, [(3, 7)]),
            ("stage/3", 7, [(4, 7), (2, 3), (0, 1)]),
        ]),
    ]
    batches = [tree_batch(rng, f"a single product tree of height {h}", [(h, None)]) for h in range(1, 7)]
    batches += [
        tree_batch(rng, "trees of heights 3, 6, 1, 4, 6 and 2, the one of height 4 with a zero leaf", [(3, None), (6, None), (1, None), (4, 9), (6, None), (2, None)]),
        tree_batch(rng, "a tree of height 3 with a zero leaf, alone", [(3, 2)]),
        tree_batch(rng, "three trees of height 4", [(4, None), (4, None), (4, None)]),
        tree_batch(rng, "trees of heights 2 and 5, the shorter first, with a zero leaf", [(2, 0), (5, None)]),
    ]
    header = {
        "description": "RFC-0010 sumcheck reference vectors: honest proofs of batched sumchecks "
        "and of product-tree levels, written by spec/vectors/sumcheck/generate.py; see README.md",
        "modulus": hex64(P),
        "non_residue": W,
        "seed": hex64(SEED),
        "official_blake3_cases_checked": checked,
    }
    text = render(header, sessions, batches)
    (HERE / "sumcheck.json").write_text(text)
    stages = sum(len(s["stages"]) for s in sessions)
    levels = sum(len(b["levels"]) for b in batches)
    print(f"wrote sumcheck.json: {len(sessions)} sessions of {stages} stages, {len(batches)} tree batches of {levels} levels, {len(text)} bytes")


if __name__ == "__main__":
    main()
