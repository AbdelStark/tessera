#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0 OR MIT
"""Reference vectors for the sumchecks of RFC-0010: honest proofs of single and batched
sumchecks of products of multilinear tables, front-padded, of batches of product trees proven
level by level from their roots to their leaves, of batches of product forests, trees stacked
under a random root point, and of LogUp-GKR leaf stages, the fraction trees of RFC-0008 beside
product forests, honest and cheating.

This script is the independent oracle: an honest prover written from RFC-0010 ('Multilinear
extensions', 'Tree levels', 'Batching within a stage', 'Sumcheck messages'), RFC-0008 'Fraction
tree' and RFC-0014 'Transcript schedule' alone, computing with Python's arbitrary-precision
integers. Its transcript is the oracle of ../transcript/generate.py, whose BLAKE3 is checked
against the official test vectors before anything is written. It shares no code with crates/ or
verifier/, which check its output, and its algorithms differ from theirs:

- A smaller instance is padded literally: its tables are extended to 2^n_max entries, constant in
  the lowest n_max - n_i variables, and one plain sumcheck runs over the padded tables.
- Every claim, final evaluation and tree claim it writes is computed from its definition, a sum
  over the hypercube weighted by eq, and checked against the value the sumcheck reaches.
- Each round polynomial is evaluated at its challenge by Lagrange's formula, and checked against
  the sum of the tables bound to that challenge.
- A forest's levels are its trees' levels concatenated, and a forest's root claim is computed
  twice: as sum_y v_y eq(r, y), from the product formula of eq and the trees' products, and as
  the extension of the stacked root level.
- A LogUp batch is proven by the prover here and read back from the written file by a separate
  verifier, which rederives every challenge and claim from the messages, evaluates the tables by
  folding them variable by variable, and must find exactly the batch's expected verdict.

Usage: python3 spec/vectors/sumcheck/generate.py
(rewrites sumcheck.json)

Inputs come from SplitMix64 with the fixed seed below, so the file is reproducible byte for byte.
"""

import hashlib
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
TABLES_MAX_VARS = 5  # a stage whose instances have at most this many variables records its tables
# SHA-256 of the file as it was before logup_batches: sessions, tree_batches and forest_batches
# are pinned by it, and a deliberate change to them updates it.
BEFORE_LOGUP_SHA256 = "2e049d2450e985d1673d7008a02a0417aba798bf79c30ed121254187f206403c"


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
    """A stage of product instances, each `(vars, factors)` with random Fp2 tables. A stage whose
    instances all have at most TABLES_MAX_VARS variables also records each instance's tables, so
    that a prover can be checked against the whole proof; the tables come from the same draws."""
    instances = [([[rng.fp2() for _ in range(1 << n)] for _ in range(k)], n) for n, k in shapes]
    with_tables = max(n for n, _ in shapes) <= TABLES_MAX_VARS
    gamma, rounds, challenges, claims, bound = sumcheck(t, prefix, degree, instances)
    evals, points, values, records = [], [], [], []
    for (tables, n), bound_i in zip(instances, bound):
        point = suffix(challenges, n)
        own = [mle(table, point) for table in tables]
        require(own == bound_i, f"{prefix}: each table's MLE at its point is the bound table")
        evals += own
        points.append(point)
        values.append(product(own))
        record = {"vars": n, "factors": len(tables), "claim": fp2(total(product(tb[x] for tb in tables) for x in range(1 << n)))}
        if with_tables:
            record["tables"] = [elements(table) for table in tables]
        records.append(record)
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


# ----- Product forests ('Tree levels': 'Layout', 'Joining the batch', 'Root claim') --------------


def forest_levels(trees):
    """The levels of a forest of 2^s trees of one height, the root level first: level l is the
    table of 2^(s + l) entries whose entry y 2^l + x is entry x of tree y's level l, that is the
    trees' level-l tables concatenated in the order of y."""
    per_tree = [tree_levels(leaves) for leaves in trees]
    require(len(trees) & (len(trees) - 1) == 0, "a forest has 2^s trees")
    require(len({len(levels) for levels in per_tree}) == 1, "the trees of a forest have one height")
    height = len(per_tree[0]) - 1
    stacked = [[v for levels in per_tree for v in levels[level]] for level in range(height + 1)]
    for level in range(height):
        above, below = stacked[level], stacked[level + 1]
        require(len(above) == len(trees) << level, "level l of a forest has 2^(s + l) entries")
        require(above == [mul(below[2 * k], below[2 * k + 1]) for k in range(len(above))],
                "the stacked levels pair their entries as one tree's do, the pair bit lowest")
    return stacked


def boolean_point(y, n):
    """The integer y as the Boolean point (y_1, ..., y_n), y_1 its lowest bit."""
    return [lift((y >> i) & 1) for i in range(n)]


def root_claim(roots, point):
    """sum_y v_y eq(r, y) over the forest's public root values, eq from its product formula."""
    require(len(roots) == 1 << len(point), "a root point has s coordinates for 2^s trees")
    return total(mul(v, eq_points(point, boolean_point(y, len(point)))) for y, v in enumerate(roots))


def prove_forests(t, forests):
    """A batch of product forests, level by level from the roots (RFC-0010 'Tree levels'). Each
    forest is `(s, levels, roots)`: its stacked levels, the root level first, and the public values
    of its root level, one per tree. A forest of height h joins at level H - h. At a level where
    forests of more than one tree join, the challenges tree/{l}/root/{c}, c = 0 .. s_max - 1, are
    squeezed before gamma, and each joining forest's root point is the last s of them, in the
    order squeezed; a forest of one tree takes none and enters with its root value. The instances
    present at a level are the forests that have joined, in the order of the batch's list."""
    heights = [len(levels) - 1 for _, levels, _ in forests]
    tallest = max(heights)
    current = {}  # forest -> (own level, point, value)
    records = []
    for level in range(tallest):
        prefix = f"tree/{level}"
        joining = [f for f, h in enumerate(heights) if tallest - h == level]
        s_max = max((forests[f][0] for f in joining), default=0)
        root_challenges = [challenge(t, f"{prefix}/root/{c}") for c in range(s_max)]
        for f in joining:
            s, levels, roots = forests[f]
            require(roots == levels[0], "the public root values are the root level")
            point = root_challenges[s_max - s:]
            require(len(point) == s, "a forest's root point has s coordinates")
            value = root_claim(roots, point)
            require(value == mle(levels[0], point), f"{prefix}: the root claim is the root level's MLE at the root point")
            current[f] = (0, point, value)
        present = [f for f in range(len(forests)) if f in current]
        instances, entering = [], []
        for f in present:
            own, point, value = current[f]
            s, levels, _ = forests[f]
            n = s + own
            require(own == level - (tallest - heights[f]) and len(point) == n, "a forest has s + l - (H - h) variables at level l")
            require(mle(levels[own], point) == value, f"{prefix}: each claim is its forest level's MLE at its point")
            below = levels[own + 1]
            eq_table = [eq_at(point, x) for x in range(1 << n)]
            left = [below[2 * x] for x in range(1 << n)]
            right = [below[2 * x + 1] for x in range(1 << n)]
            require(total(mul(eq_table[x], mul(left[x], right[x])) for x in range(1 << n)) == value,
                    f"{prefix}: the level relation sums to the claim")
            instances.append(([eq_table, left, right], n))
            entering.append({"forest": f, "point": elements(point), "value": fp2(value)})
        gamma, rounds, challenges, claims, bound = sumcheck(t, prefix, 3, instances)
        evals, nexts, terms = [], [], []
        weights = [power(gamma, i) for i in range(len(present))]
        for f, (tables, n), bound_i in zip(present, instances, bound):
            point = suffix(challenges, n)
            _, entry_point, _ = current[f]
            u, w = mle(tables[1], point), mle(tables[2], point)
            require([eq_points(entry_point, point), u, w] == bound_i, f"{prefix}: eq, u and w are the bound tables")
            evals += [u, w]
            terms.append(mul(eq_points(entry_point, point), mul(u, w)))
        t.absorb(f"{prefix}/evals", tce(evals))
        require(total(mul(wt, term) for wt, term in zip(weights, terms)) == claims[-1], f"{prefix}: the final check holds")
        rho = challenge(t, f"{prefix}/line")
        for i, f in enumerate(present):
            own = current[f][0]
            n = forests[f][0] + own
            u, w = evals[2 * i], evals[2 * i + 1]
            point = [rho] + suffix(challenges, n)
            value = add(u, mul(rho, sub(w, u)))
            require(mle(forests[f][1][own + 1], point) == value, f"{prefix}: the next claim is the next level's MLE at rho || r'")
            current[f] = (own + 1, point, value)
            nexts.append({"forest": f, "point": elements(point), "value": fp2(value)})
        records.append({
            "prefix": prefix,
            "root_challenges": elements(root_challenges),
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
    for f, (own, point, value) in current.items():
        s, levels, _ = forests[f]
        require(own == heights[f] and len(point) == s + own and mle(levels[-1], point) == value,
                "every leaf claim is the stacked leaves' MLE at its point")
    return records


# ----- LogUp-GKR fraction trees (RFC-0008 'Fraction tree'; RFC-0010 'Tree levels') ---------------


def neg(a):
    return ((-a[0]) % P, (-a[1]) % P)


def inverse(a):
    """1/a in Fp2: (c0 - c1 X) / (c0^2 - 7 c1^2). The norm c0^2 - 7 c1^2 of a non-zero element is
    not 0, because 7 is not a square in Fp."""
    norm = (a[0] * a[0] - W * a[1] * a[1]) % P
    require(norm != 0, "no inverse of 0")
    k = inverse_fp(norm)
    return (a[0] * k % P, -a[1] * k % P)


def height_of(n):
    """h = ceil(log2 n) for n >= 1: the height of a tree whose n leaves are padded to 2^h."""
    require(n >= 1, "a tree has a leaf")
    return (n - 1).bit_length()


def compress(beta, row):
    """A tuple (key, value) of Fp as one Fp2 element, key + beta value ('Relation')."""
    return add(lift(row[0]), mul(beta, lift(row[1])))


def combine(left, right):
    """A fraction tree's node from its children (n, d), the left one the even entry:
    (n_L d_R + n_R d_L, d_L d_R)."""
    return (add(mul(left[0], right[1]), mul(right[0], left[1])), mul(left[1], right[1]))


def fraction_levels(leaves):
    """Every level of a fraction tree, the root first, each a list of (numerator, denominator):
    node x of level l combines entries 2x and 2x + 1 of level l + 1."""
    require(len(leaves) & (len(leaves) - 1) == 0, "a fraction tree has 2^h leaves")
    levels = [leaves]
    while len(levels[0]) > 1:
        below = levels[0]
        levels.insert(0, [combine(below[2 * x], below[2 * x + 1]) for x in range(len(below) // 2)])
    return levels


def fraction_sum(pairs):
    """sum_i n_i / d_i."""
    return total(mul(n, inverse(d)) for n, d in pairs)


def check_fraction_tree(levels):
    """A second reading of a fraction tree, from its leaves: no leaf denominator is 0, and every
    node x of level l, over the leaves x 2^(h - l) to (x + 1) 2^(h - l) - 1, has their product as
    its denominator and their fractions' sum as its fraction n/d. At the root, n/d is the sum of
    all the leaves' fractions, sum_i n_i / d_i."""
    leaves, h = levels[-1], len(levels) - 1
    require(all(d != ZERO for _, d in leaves), "no leaf denominator is 0")
    fractions = [mul(n, inverse(d)) for n, d in leaves]
    for level, nodes in enumerate(levels):
        span = 1 << (h - level)
        require(len(nodes) == 1 << level, "level l of a tree has 2^l nodes")
        for x, (n, d) in enumerate(nodes):
            require(d == product(leaf[1] for leaf in leaves[x * span:(x + 1) * span]), "a node's denominator is its leaves' product")
            require(mul(n, inverse(d)) == total(fractions[x * span:(x + 1) * span]), "a node's fraction is its leaves' sum")
    require(mul(levels[0][0][0], inverse(levels[0][0][1])) == fraction_sum(leaves), "the root's n/d is sum_i n_i / d_i")


def level_sumcheck(t, prefix, instances, weights):
    """The honest prover of one tree level's batched sumcheck of degree 3 ('Tree levels' step 2,
    'Sumcheck messages'), once gamma and lambda are drawn. Each instance is `(terms, n)`: its
    relation is a sum of terms, each a coefficient times the product of three tables of 2^n
    entries, and `weights` holds gamma^i. The tables are padded literally, as in `sumcheck`.
    Returns the round messages, the challenges, the claims, each the bound tables' sum, and each
    term's tables bound to every challenge."""
    n_max = max(n for _, n in instances)
    padded = [[(c, [[table[idx >> (n_max - n)] for idx in range(1 << n_max)] for table in tables]) for c, tables in terms]
              for terms, n in instances]

    def batched(instances):
        out = ZERO
        for weight, terms in zip(weights, instances):
            for c, tables in terms:
                out = add(out, mul(mul(weight, c), total(product(f[x] for f in tables) for x in range(len(tables[0])))))
        return out

    claims, rounds, challenges = [batched(padded)], [], []
    for j in range(n_max):
        half = 1 << (n_max - j - 1)

        def at(x_j):
            """p_j(x_j): the batched sum with x_j fixed and the later variables summed."""
            out = ZERO
            for weight, terms in zip(weights, padded):
                for c, tables in terms:
                    s = total(product(add(f[2 * x], mul(x_j, sub(f[2 * x + 1], f[2 * x]))) for f in tables) for x in range(half))
                    out = add(out, mul(mul(weight, c), s))
            return out

        values = [at(lift(k)) for k in range(4)]
        require(add(values[0], values[1]) == claims[-1], f"{prefix} round {j}: p(0) + p(1) = claim")
        message = [values[0]] + values[2:]
        t.absorb(f"{prefix}/round/{j}", tce(message))
        r = challenge(t, f"{prefix}/r/{j}")
        padded = [[(c, [[add(f[2 * x], mul(r, sub(f[2 * x + 1], f[2 * x]))) for x in range(half)] for f in tables]) for c, tables in terms]
                  for terms in padded]
        claim = batched(padded)
        require(lagrange(values, r) == claim, f"{prefix} round {j}: p(r) is the bound tables' sum")
        rounds.append(message)
        challenges.append(r)
        claims.append(claim)
    bound = [[[f[0] for f in tables] for _, tables in terms] for terms in padded]
    return rounds, challenges, claims, bound


def numerators(nodes):
    return [nd[0] for nd in nodes]


def denominators(nodes):
    return [nd[1] for nd in nodes]


def prove_leaf_stage(t, schedule):
    """The tree levels of one leaf stage, after logup/roots (RFC-0010 'Tree levels'). `schedule`
    lists the trees in schedule order, each a dict with `kind` ("lookup", "table" or "forest") and
    `levels`, the prover's own tree or stacked forest, the root level first. A forest also has `s`
    and its public `roots`; a fraction tree has `s` = 0 and `sent`, the root (n, d) the roots
    message gave it, which is what the verifier holds on its level 0. A tree of height h joins at
    level H - h, and the instances present at a level are the trees that have joined, in schedule
    order. Where a fraction tree is present, lambda follows gamma and a fraction tree enters with
    the claim p + lambda q. Returns the level records, whose claims are those the verifier derives
    from the messages, and every tree's claims after its last level."""
    heights = [len(tree["levels"]) - 1 for tree in schedule]
    tallest = max(heights)
    current = {}  # tree -> (own level, point, claim): a forest's value, a fraction tree's (p, q)
    records = []
    for level in range(tallest):
        prefix = f"tree/{level}"
        joining = [k for k, h in enumerate(heights) if tallest - h == level]
        s_max = max((schedule[k]["s"] for k in joining), default=0)
        root_challenges = [challenge(t, f"{prefix}/root/{c}") for c in range(s_max)]
        for k in joining:
            tree = schedule[k]
            if tree["kind"] == "forest":
                point = root_challenges[s_max - tree["s"]:]
                value = root_claim(tree["roots"], point)
                require(tree["roots"] == tree["levels"][0] and value == mle(tree["levels"][0], point),
                        f"{prefix}: the root claim is the root level's MLE at the root point")
                current[k] = (0, point, value)
            else:
                current[k] = (0, [], tree["sent"])
        present = [k for k in range(len(schedule)) if k in current]
        gamma = challenge(t, f"{prefix}/gamma")
        lam = challenge(t, f"{prefix}/lambda") if any(schedule[k]["kind"] != "forest" for k in present) else None
        weights = [power(gamma, i) for i in range(len(present))]
        instances, entering, held, owned = [], [], [], []
        for k in present:
            tree = schedule[k]
            own, point, claim = current[k]
            n = len(point)
            require(own == level - (tallest - heights[k]) and n == tree["s"] + own, "a tree has s + l - (H - h) variables at level l")
            above, below = tree["levels"][own], tree["levels"][own + 1]
            eq_table = [eq_at(point, x) for x in range(1 << n)]
            if tree["kind"] == "forest":
                truth = mle(above, point)
                left = [below[2 * x] for x in range(1 << n)]
                right = [below[2 * x + 1] for x in range(1 << n)]
                require(total(mul(eq_table[x], mul(left[x], right[x])) for x in range(1 << n)) == truth,
                        f"{prefix}: the level relation sums to the claim")
                instances.append(([(ONE, [eq_table, left, right])], n))
                entering.append({"tree": k, "point": elements(point), "value": fp2(claim)})
                held.append(claim)
                owned.append(truth)
            else:
                truth = (mle(numerators(above), point), mle(denominators(above), point))
                n0, n1 = [below[2 * x][0] for x in range(1 << n)], [below[2 * x + 1][0] for x in range(1 << n)]
                d0, d1 = [below[2 * x][1] for x in range(1 << n)], [below[2 * x + 1][1] for x in range(1 << n)]
                require(total(mul(eq_table[x], add(mul(n0[x], d1[x]), mul(n1[x], d0[x]))) for x in range(1 << n)) == truth[0]
                        and total(mul(eq_table[x], mul(d0[x], d1[x])) for x in range(1 << n)) == truth[1],
                        f"{prefix}: the level relation sums to the claims")
                instances.append(([(ONE, [eq_table, n0, d1]), (ONE, [eq_table, n1, d0]), (lam, [eq_table, d0, d1])], n))
                entering.append({"tree": k, "point": elements(point), "numerator": fp2(claim[0]), "denominator": fp2(claim[1])})
                held.append(add(claim[0], mul(lam, claim[1])))
                owned.append(add(truth[0], mul(lam, truth[1])))
        n_max = max(n for _, n in instances)
        scales = [lift(1 << (n_max - n)) for _, n in instances]
        rounds, challenges, own_claims, bound = level_sumcheck(t, prefix, instances, weights)
        require(own_claims[0] == total(mul(w, mul(s, c)) for w, s, c in zip(weights, scales, owned)),
                f"{prefix}: the padded batch sums to sum_i gamma^i 2^(n_max - n_i) c_i")
        # The claims as the verifier derives them from the claims it holds and the messages:
        # p_j(1) = claim_j - p_j(0), and claim_{j+1} = p_j(r_j) ('Sumcheck messages').
        claims = [total(mul(w, mul(s, c)) for w, s, c in zip(weights, scales, held))]
        for message, r in zip(rounds, challenges):
            claims.append(lagrange([message[0], sub(claims[-1], message[0])] + message[1:], r))
        honest = held == owned
        require(not honest or claims == own_claims, f"{prefix}: an honest level's claims are the bound tables' sums")
        evals, terms = [], []
        for k, (parts, n), bound_i in zip(present, instances, bound):
            point = suffix(challenges, n)
            e = eq_points(current[k][1], point)
            if schedule[k]["kind"] == "forest":
                _, left, right = parts[0][1]
                u, w = mle(left, point), mle(right, point)
                require(bound_i == [[e, u, w]], f"{prefix}: eq, u and w are the bound tables")
                evals += [u, w]
                terms.append(mul(e, mul(u, w)))
            else:
                (_, n0, d1), (_, n1, d0) = parts[0][1], parts[1][1]
                N0, N1, D0, D1 = mle(n0, point), mle(n1, point), mle(d0, point), mle(d1, point)
                require(bound_i == [[e, N0, D1], [e, N1, D0], [e, D0, D1]], f"{prefix}: eq, N_0, N_1, D_0 and D_1 are the bound tables")
                evals += [N0, N1, D0, D1]
                terms.append(mul(e, add(add(mul(N0, D1), mul(N1, D0)), mul(lam, mul(D0, D1)))))
        t.absorb(f"{prefix}/evals", tce(evals))
        final = total(mul(w, term) for w, term in zip(weights, terms))
        require(final == own_claims[-1], f"{prefix}: the prover's final check holds")
        require(not honest or final == claims[-1], f"{prefix}: an honest level passes the final check")
        rho = challenge(t, f"{prefix}/line")
        nexts, cursor = [], 0
        for k, (_, n) in zip(present, instances):
            tree, own = schedule[k], current[k][0]
            point = [rho] + suffix(challenges, n)
            below = tree["levels"][own + 1]
            if tree["kind"] == "forest":
                u, w = evals[cursor:cursor + 2]
                cursor += 2
                value = add(u, mul(rho, sub(w, u)))
                require(mle(below, point) == value, f"{prefix}: the next claim is the next level's MLE at rho || r'")
                current[k] = (own + 1, point, value)
                nexts.append({"tree": k, "point": elements(point), "value": fp2(value)})
            else:
                N0, N1, D0, D1 = evals[cursor:cursor + 4]
                cursor += 4
                claim = (add(N0, mul(rho, sub(N1, N0))), add(D0, mul(rho, sub(D1, D0))))
                require(claim == (mle(numerators(below), point), mle(denominators(below), point)),
                        f"{prefix}: the next claims are the next level's MLEs at rho || r'")
                current[k] = (own + 1, point, claim)
                nexts.append({"tree": k, "point": elements(point), "numerator": fp2(claim[0]), "denominator": fp2(claim[1])})
        records.append({
            "prefix": prefix,
            "root_challenges": elements(root_challenges),
            "instances": entering,
            "gamma": fp2(gamma),
            "lambda": None if lam is None else fp2(lam),
            "rounds": [elements(m) for m in rounds],
            "challenges": elements(challenges),
            "claims": elements(claims),
            "final_evals": elements(evals),
            "rho": fp2(rho),
            "next": nexts,
            "state": t.state.hex(),
            "counter": t.counter,
        })
    for k, tree in enumerate(schedule):
        if heights[k] == 0:
            # A tree of height 0 joins at level H: its root, its one leaf, is its leaf claim.
            require(tree["kind"] != "forest", "a forest has a level")
            current[k] = (0, [], tree["sent"])
        own, point, claim = current[k]
        leaves = tree["levels"][-1]
        require(own == heights[k] and len(point) == tree["s"] + own, "every tree reaches its leaves")
        if tree["kind"] == "forest":
            require(mle(leaves, point) == claim, "every leaf claim is the stacked leaves' MLE at its point")
        else:
            require(claim == (mle(numerators(leaves), point), mle(denominators(leaves), point)),
                    "every leaf claim is the leaves' numerators' and denominators' MLE at its point")
    return records, current


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


def render_batches(key, members, batches):
    """`"key": [...]`, one batch per line group: its name and context, then each of its `members`
    and each of its levels on a line of its own."""
    return f'"{key}": [\n' + ",\n".join(
        "    {" + f'"name":{compact(b["name"])},"context":{compact(b["context"])},"{members}":[\n'
        + ",\n".join("      " + compact(member) for member in b[members]) + '\n    ],"levels":[\n'
        + ",\n".join("      " + compact(level) for level in b["levels"]) + "\n    ]}"
        for b in batches
    ) + "\n  ]"


LOGUP_KEYS = ["name", "context", "alpha", "beta", "instances", "forests", "trees", "roots", "levels", "values", "leaf_checks",
              "state", "counter", "expected"]
LOGUP_LISTS = {"instances", "forests", "levels", "leaf_checks"}  # one member per line


def render_logup(batches):
    """`"logup_batches": [...]`, one batch per line group: its fields in order, and each of its
    instances, forests, levels and leaf checks on a line of its own."""
    def field(key, value):
        if key in LOGUP_LISTS and value:
            return f'"{key}":[\n' + ",\n".join("      " + compact(member) for member in value) + "\n    ]"
        return f'"{key}":{compact(value)}'

    for b in batches:
        require(list(b) == LOGUP_KEYS, "a LogUp batch's fields, in order")
    return '"logup_batches": [\n' + ",\n".join("    {" + ",".join(field(key, b[key]) for key in LOGUP_KEYS) + "}" for b in batches) + "\n  ]"


def render(header, sessions, batches, forest_batches, logup_batches):
    text = json.dumps(header, indent=2)[:-2] + ',\n  "sessions": [\n'
    text += ",\n".join(
        "    {" + f'"name":{compact(s["name"])},"context":{compact(s["context"])},"stages":[\n'
        + ",\n".join("      " + compact(stage) for stage in s["stages"]) + "\n    ]}"
        for s in sessions
    )
    text += "\n  ],\n  " + render_batches("tree_batches", "trees", batches)
    # The forests are one key appended to the file as it was before them: removing the key, with
    # its comma, gives that file byte for byte.
    without_forests = text + "\n}\n"
    text += ",\n  " + render_batches("forest_batches", "forests", forest_batches)
    # The LogUp batches are one more key appended the same way, to the file whose hash is pinned.
    without_logup = text + "\n}\n"
    text += ",\n  " + render_logup(logup_batches) + "\n}\n"
    require(json.loads(without_forests) == {**header, "sessions": sessions, "tree_batches": batches}, "the file parses back")
    require(json.loads(without_logup) == {**header, "sessions": sessions, "tree_batches": batches, "forest_batches": forest_batches},
            "the file parses back")
    require(hashlib.sha256(without_logup.encode()).hexdigest() == BEFORE_LOGUP_SHA256,
            "sessions, tree_batches and forest_batches are byte for byte the file before logup_batches")
    document = json.loads(text)
    require(document == {**header, "sessions": sessions, "tree_batches": batches, "forest_batches": forest_batches,
                         "logup_batches": logup_batches}, "the file parses back")
    require(list(document) == [*header, "sessions", "tree_batches", "forest_batches", "logup_batches"], "the LogUp batches come last")
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


def draw_forest(rng, s, h, zeros):
    """2^s trees of height h with random leaves, where `zeros` places zero leaves. "rows" gives
    every tree a zero leaf at a random position, as SoftmaxRow's rows have their maximum's key;
    "first" gives every tree a zero first leaf; a list of `(tree, leaf)` places those zeros only.
    Returns the forest as `(s, levels, roots)` and its record."""
    trees = [[lift(rng.fp()) for _ in range(1 << h)] for _ in range(1 << s)]
    if zeros == "rows":
        places = [(y, rng.word() % (1 << h)) for y in range(1 << s)]
    elif zeros == "first":
        places = [(y, 0) for y in range(1 << s)]
    else:
        places = zeros
    for y, x in places:
        trees[y][x] = ZERO
    # The public root values: each tree's product, computed directly from its leaves.
    roots = [product(leaves) for leaves in trees]
    zero_trees = {y for y, _ in places}
    for y, root in enumerate(roots):
        require((root == ZERO) == (y in zero_trees), "only the trees with a zero leaf have root 0")
    levels = forest_levels(trees)
    return (s, levels, roots), {"rows_log": s, "height": h, "roots": [fp(v) for v in roots], "leaves": [fp(v) for v in levels[-1]]}


def forest_batch(rng, name, shapes):
    """`shapes` lists each forest as `(s, h, zeros)`, drawn by `draw_forest`."""
    context = rng.context()
    t = transcript.Transcript(transcript.session_context(context))
    drawn = [draw_forest(rng, s, h, zeros) for s, h, zeros in shapes]
    return {"name": name, "context": context, "forests": [d for _, d in drawn], "levels": prove_forests(t, [f for f, _ in drawn])}


def trees_as_forests(batch):
    """A tree batch proven again as forests of one tree each, on a fresh transcript of its context:
    a single tree is a forest with s = 0, which takes no root challenge and enters with its root."""
    t = transcript.Transcript(transcript.session_context(batch["context"]))
    forests = []
    for tree in batch["trees"]:
        levels = tree_levels([lift(int(leaf, 16)) for leaf in tree["leaves"]])
        forests.append((0, levels, [levels[0][0]]))
    records = prove_forests(t, forests)
    for record in records:
        require(record.pop("root_challenges") == [], "a single tree takes no root challenge")
        for claim in record["instances"] + record["next"]:
            claim["tree"] = claim.pop("forest")
    return records


# ----- LogUp batches: one leaf stage each, honest or cheating -------------------------------------

WITNESS_CHEATS = {"outside", "wrong value", "multiplicity", "false roots", "numerator", "table", "phantom row",
                  "phantom row, zero padding"}  # each breaks the identity
MESSAGE_CHEATS = {"zero root", "false roots", "scaled roots"}  # a roots message other than the trees' roots
HARMLESS = {"padding multiplicity"}  # a witness other than the honest prover's that the verifier accepts
PADDING_COUNTS = {"phantom row", "phantom row, zero padding", "padding multiplicity"}  # non-zero padding multiplicities


def draw_instance(rng, rows, lookups, has_values):
    """A LogUp instance as its prover holds it: a public table of `rows` rows (key, value) of Fp and
    `lookups` lookups, each of a row drawn at random, with the multiplicities they give. A table
    with values has distinct random keys and random values; a table without, Range-like, has the
    keys 0 to T - 1 and every value 0. m_j counts the lookups of row j, row 0's also the lookup
    tree's 2^h_a - N padding leaves, and the 2^h_t - T padding rows hold 0. Another prover
    changes `lookups` (a row index, or the tuple of a lookup that is no row), `tuples`,
    `numerators`, `table` (the table its trees use), `padding` (the tuple its table tree pads
    with) or `multiplicities`; `rows` stays public."""
    if has_values:
        table, keys = [], set()
        while len(table) < rows:
            key = rng.fp()
            if key not in keys:
                keys.add(key)
                table.append((key, rng.fp()))
    else:
        table = [(j, 0) for j in range(rows)]
    indices = [rng.word() % rows for _ in range(lookups)]
    m = [0] * (1 << height_of(rows))
    for j in indices:
        m[j] += 1
    m[0] += (1 << height_of(lookups)) - lookups
    return {"rows": table, "has_values": has_values, "lookups": indices, "tuples": [table[j] for j in indices],
            "numerators": [ONE] * lookups, "table": list(table), "padding": "row 0", "multiplicities": m}


def padding_row(beta, table, padding):
    """The tuple a table's padding rows hold, compressed: a copy of row 0 (RFC-0008 'Fraction
    tree'), or 0 for a prover that pads with key and value 0, as the text did before #62 fixed it."""
    return ZERO if padding == "zero" else compress(beta, table[0])


def logup_trees(alpha, beta, inst):
    """The prover's two trees of an instance (RFC-0008 'Fraction tree'). The lookup tree's leaf
    i < N is (1, alpha - a_i), a_i = key + beta value of the lookup's tuple, and its 2^h_a - N
    padding leaves repeat row 0, (1, alpha - t_0). The table tree's leaf j < T is
    (-m_j, alpha - t_j), and its 2^h_t - T padding leaves repeat row 0, (-m_j, alpha - t_0),
    which an honest prover gives the multiplicity 0."""
    table, tuples, m = inst["table"], inst["tuples"], inst["multiplicities"]
    h_a, h_t = height_of(len(tuples)), height_of(len(table))
    require(len(m) == 1 << h_t, "2^h_t multiplicities")
    lookup = [(n, sub(alpha, compress(beta, tup))) for n, tup in zip(inst["numerators"], tuples)]
    lookup += [(ONE, sub(alpha, compress(beta, table[0])))] * ((1 << h_a) - len(tuples))
    rows = [(neg(lift(mj)), sub(alpha, compress(beta, row))) for mj, row in zip(m, table)]
    rows += [(neg(lift(mj)), sub(alpha, padding_row(beta, table, inst["padding"]))) for mj in m[len(table):]]
    return fraction_levels(lookup), fraction_levels(rows)


def lookup_side(inst):
    """The lookup side's key and value polynomials, of 2^h_a entries: each lookup's key and value,
    then row 0's in the padding slots."""
    tuples = inst["tuples"]
    slots = tuples + [inst["table"][0]] * ((1 << height_of(len(tuples))) - len(tuples))
    return [lift(key) for key, _ in slots], [lift(value) for _, value in slots]


def table_polynomial(beta, table, padding="row 0"):
    """The table polynomial t = key_t + beta value_t, of 2^h_t entries, its padding rows copies of
    row 0."""
    return [compress(beta, row) for row in table] + [padding_row(beta, table, padding)] * ((1 << height_of(len(table))) - len(table))


def identity(alpha, beta, inst):
    """'Relation' against the public table: sum_{i<N} 1/(alpha - a_i) - sum_{j<T} m_j/(alpha - t_j),
    from the lookups' tuples and the multiplicities. A padding row is a copy of row 0, so row 0's
    multiplicity takes the padding rows' multiplicities, and gives back the lookup tree's padding
    leaves, which repeat row 0. It is 0 exactly when the identity holds."""
    rows, tuples, m = inst["rows"], inst["tuples"], inst["multiplicities"]
    counts = m[:len(rows)]
    counts[0] += sum(m[len(rows):]) - ((1 << height_of(len(tuples))) - len(tuples))
    lookups = total(inverse(sub(alpha, compress(beta, tup))) for tup in tuples)
    return sub(lookups, total(mul(lift(c), inverse(sub(alpha, compress(beta, row)))) for c, row in zip(counts, rows)))


def outside_tuple(rng, inst):
    """A tuple that is no row of the table: the key T of a Range-like table, or a key that no row
    has, with a random value."""
    if not inst["has_values"]:
        return (len(inst["rows"]), 0)
    keys = {key for key, _ in inst["rows"]}
    while True:
        key = rng.fp()
        if key not in keys:
            return (key, rng.fp())


def cheat_witness(rng, inst, cheat):
    """The witness a cheating prover commits instead of the honest one; its trees stay honest
    trees of that witness."""
    kind = cheat["kind"]
    if kind in ("outside", "numerator"):
        # A lookup that is no row, which no multiplicity counts; "numerator" also gives its leaf
        # the numerator 0, so that the trees' roots sum to zero.
        i = cheat["lookup"]
        inst["multiplicities"][inst["lookups"][i]] -= 1
        inst["lookups"][i] = inst["tuples"][i] = outside_tuple(rng, inst)
        if kind == "numerator":
            inst["numerators"][i] = ZERO
    elif kind == "wrong value":
        # A lookup of its row's key with another value, still counted in that row.
        i = cheat["lookup"]
        require(inst["has_values"], "a wrong value needs a table with values")
        key, value = inst["rows"][inst["lookups"][i]]
        inst["lookups"][i] = inst["tuples"][i] = (key, (value + 1) % P)
    elif kind in ("multiplicity", "false roots"):
        inst["multiplicities"][cheat["row"]] += 1
    elif kind == "table":
        # The trees use a table whose row j, one the lookups use, has another value; the lookups of
        # row j take that tuple, which is no row of the public table.
        j = next(j for j in inst["lookups"] if j != 0)
        require(inst["has_values"], "a changed value needs a table with values")
        key, value = inst["rows"][j]
        inst["table"][j] = (key, (value + 1) % P)
        for i, lookup in enumerate(inst["lookups"]):
            if lookup == j:
                inst["lookups"][i] = inst["tuples"][i] = inst["table"][j]
    elif kind in ("phantom row", "phantom row, zero padding"):
        # The lookup (0, 0), which is no row, with the multiplicity 1 on the last padding row.
        # Under row 0's padding that multiplicity counts a lookup of row 0 and cannot balance it;
        # under the zero padding the text had before #62, the leaf (-1, alpha) balanced it.
        rows, i, last = inst["rows"], cheat["lookup"], len(inst["multiplicities"]) - 1
        require(len(rows) & (len(rows) - 1) and (0, 0) not in rows, "a table with padding rows and without the row (0, 0)")
        inst["multiplicities"][inst["lookups"][i]] -= 1
        inst["lookups"][i] = inst["tuples"][i] = (0, 0)
        inst["multiplicities"][last] += 1
        if kind == "phantom row, zero padding":
            inst["padding"] = "zero"
    elif kind == "padding multiplicity":
        # Not a cheat: the lookup tree's padding leaves, lookups of row 0, counted on the last
        # padding row, a copy of row 0, instead of on row 0.
        m, padding = inst["multiplicities"], (1 << height_of(len(inst["tuples"]))) - len(inst["tuples"])
        require(len(inst["rows"]) < len(m) and padding > 0, "padding on both sides")
        m[0] -= padding
        m[-1] += padding
    else:
        require(False, f"unknown cheat {kind}")


def cheat_roots(rng, roots, cheat):
    """The roots message of a cheating prover, from its trees' roots, each instance's
    ((n_a, d_a), (n_t, d_t))."""
    sent = list(roots)
    k, kind = cheat["instance"], cheat["kind"]
    lookup, table = sent[k]
    if kind == "zero root":
        # (0, 0) passes n_a d_t + n_t d_a = 0: only the denominator check rejects it.
        if cheat["side"] == "lookup":
            lookup = (ZERO, ZERO)
        else:
            table = (ZERO, ZERO)
    elif kind == "false roots":
        # The n_a that makes n_a d_t + n_t d_a = 0, hiding the multiplicity off by one.
        lookup = (neg(mul(mul(table[0], lookup[1]), inverse(table[1]))), lookup[1])
    elif kind == "scaled roots":
        # (c n_a, c d_a) is the root's fraction, so the roots sum to zero, but it is not the root.
        c = rng.fp2()
        require(c not in (ZERO, ONE), "a scale other than 0 and 1")
        lookup = (mul(c, lookup[0]), mul(c, lookup[1]))
    else:
        require(False, f"unknown cheat {kind}")
    sent[k] = (lookup, table)
    return sent


def describe_instance(inst):
    return {
        "rows": [[hex64(key), hex64(value)] for key, value in inst["rows"]],
        "has_values": inst["has_values"],
        "lookups": [j if isinstance(j, int) else [hex64(j[0]), hex64(j[1])] for j in inst["lookups"]],
        "multiplicities": [hex64(m % P) for m in inst["multiplicities"]],
        "h_a": height_of(len(inst["lookups"])),
        "h_t": height_of(len(inst["rows"])),
    }


def logup_batch(rng, name, shapes, forest_shapes, order, cheat=None):
    """One leaf stage on one transcript: the LogUp instances `shapes`, each (T, N, has_values), and
    the forests `forest_shapes`, as in `forest_batch`, proven in the schedule `order` of
    (kind, index) pairs, which is the vectors' choice. `cheat` is what another prover does instead
    of the honest one, a cheat or the harmless padding multiplicity, or None. Returns the batch and
    the checks a verifier must find failing, in check order."""
    context = rng.context()
    t = transcript.Transcript(transcript.session_context(context))
    instances = [draw_instance(rng, rows, lookups, has_values) for rows, lookups, has_values in shapes]
    drawn = [draw_forest(rng, s, h, zeros) for s, h, zeros in forest_shapes]
    kind = cheat["kind"] if cheat else None
    if kind in WITNESS_CHEATS or kind in HARMLESS:
        cheat_witness(rng, instances[cheat["instance"]], cheat)
    for k, inst in enumerate(instances):
        counted = kind in PADDING_COUNTS and cheat["instance"] == k
        require(counted == any(inst["multiplicities"][len(inst["rows"]):]), f"{name}: 0 in the padding rows unless a prover counts there")
    # RFC-0014 rows 4 and 5: alpha and beta, the roots, the levels, then the values.
    alpha, beta = challenge(t, "logup/alpha"), challenge(t, "logup/beta")
    trees = [logup_trees(alpha, beta, inst) for inst in instances]
    for k, (inst, (lookup, table)) in enumerate(zip(instances, trees)):
        check_fraction_tree(lookup)
        check_fraction_tree(table)
        breaks = kind in WITNESS_CHEATS and cheat["instance"] == k
        require((identity(alpha, beta, inst) == ZERO) != breaks, f"{name}: the identity holds exactly for an honest witness")
    roots = [(lookup[0][0], table[0][0]) for lookup, table in trees]
    sent = cheat_roots(rng, roots, cheat) if kind in MESSAGE_CHEATS else roots
    for k, ((n_a, d_a), (n_t, d_t)) in enumerate(sent):
        # Which of the root checks each instance fails: the identity-breaking witnesses fail the
        # sum alone, a root (0, 0) the denominator alone, and every other instance passes both.
        zero, sums = ZERO in (d_a, d_t), add(mul(n_a, d_t), mul(n_t, d_a)) == ZERO
        at_fault = cheat is not None and cheat["instance"] == k
        if at_fault and kind in ("outside", "wrong value", "multiplicity", "phantom row"):
            require(not zero and not sums, f"{name}: honest trees of a bad witness have non-zero roots that do not sum to zero")
        elif at_fault and kind == "zero root":
            require(zero and sums, f"{name}: a root (0, 0) passes n_a d_t + n_t d_a = 0 and fails only its denominator")
        else:
            require(not zero and sums, f"{name}: the roots of instance {k} pass")
    message = [x for lookup, table in sent for x in lookup + table]
    t.absorb("logup/roots", tce(message))
    require(sorted(order) == sorted([(side, k) for k in range(len(instances)) for side in ("lookup", "table")]
                                    + [("forest", f) for f in range(len(drawn))]), f"{name}: the schedule lists every tree once")
    schedule = []
    for side, index in order:
        if side == "forest":
            s, levels, public = drawn[index][0]
            schedule.append({"kind": side, "s": s, "levels": levels, "roots": public})
        else:
            which = 0 if side == "lookup" else 1
            schedule.append({"kind": side, "s": 0, "levels": trees[index][which], "sent": sent[index][which]})
    levels, final = prove_leaf_stage(t, schedule)
    place = {tree: k for k, tree in enumerate(order)}
    values, checks = [], []
    for k, inst in enumerate(instances):
        _, rho_a, (p_a, q_a) = final[place[("lookup", k)]]
        _, rho_t, (p_t, q_t) = final[place[("table", k)]]
        keys, vals = lookup_side(inst)
        value = mle(vals, rho_a)
        if inst["has_values"]:
            values.append(value)
        else:
            require(value == ZERO, "a table without values has the value 0")
        key = sub(sub(alpha, q_a), mul(beta, value))
        padding = (1 << len(rho_a)) - len(inst["tuples"])
        table_eval = mle(table_polynomial(beta, inst["rows"]), rho_t)
        require(p_a == mle(inst["numerators"] + [ONE] * padding, rho_a), "the numerator claim is the lookup leaves' numerators at rho_a")
        require(key == mle(keys, rho_a), "the key claim is the lookup side's key polynomial at rho_a")
        require(neg(p_t) == mle([lift(m) for m in inst["multiplicities"]], rho_t), "the multiplicity claim is the multiplicity polynomial at rho_t")
        require(q_t == sub(alpha, mle(table_polynomial(beta, inst["table"], inst["padding"]), rho_t)),
                "the denominator claim is alpha - t(rho_t) of the trees' table and padding")
        require((q_t == sub(alpha, table_eval)) == (inst["table"] == inst["rows"] and inst["padding"] == "row 0"),
                "the table evaluation holds exactly for the public table, padded with row 0")
        checks.append({
            "lookup": {"point": elements(rho_a), "numerator": fp2(p_a), "denominator": fp2(q_a),
                       "value": fp2(value) if inst["has_values"] else None, "key": fp2(key)},
            "table": {"point": elements(rho_t), "numerator": fp2(p_t), "multiplicity": fp2(neg(p_t)), "denominator": fp2(q_t),
                      "table_eval": fp2(table_eval)},
        })
    t.absorb("logup/values", tce(values))
    # The trees whose root in the message is not their root, and the level where each joins.
    tallest = max(len(tree["levels"]) - 1 for tree in schedule)
    falsified = [place[(side, k)] for k in range(len(instances)) for which, side in enumerate(("lookup", "table"))
                 if sent[k][which] != roots[k][which]]
    require(len(falsified) == (1 if kind in MESSAGE_CHEATS else 0), f"{name}: a message cheat falsifies one root, and no other batch any")
    joins = {tree: tallest - (len(schedule[tree]["levels"]) - 1) for tree in falsified}
    require(all(level < tallest for level in joins.values()), f"{name}: a falsified root is on a tree with a level")
    # The checks a verifier finds failing, in check order; the first is the verdict.
    if cheat is None or kind in HARMLESS:
        failing = []
    elif kind in ("outside", "wrong value", "multiplicity", "phantom row"):
        failing = [{"reason": "Lookup", "check": "roots", "instance": cheat["instance"]}]
    elif kind == "zero root":
        # The level where the tree joins rejects its (0, 0) claims too, but the roots come first.
        failing = [{"reason": "Lookup", "check": "roots", "instance": cheat["instance"]},
                   {"reason": "Sumcheck", "check": f"level {joins[falsified[0]]}"}]
    elif kind in ("false roots", "scaled roots"):
        failing = [{"reason": "Sumcheck", "check": f"level {joins[falsified[0]]}"}]
    elif kind == "numerator":
        failing = [{"reason": "Lookup", "check": "leaf numerator", "instance": cheat["instance"]}]
    else:
        require(kind in ("table", "phantom row, zero padding"), f"{name}: a cheat the leaf checks catch")
        failing = [{"reason": "Lookup", "check": "table evaluation", "instance": cheat["instance"]}]
    expected = dict(failing[0]) if failing else "accept"
    if failing and expected["reason"] == "Sumcheck":
        # The level's instance at fault: the falsified tree's place among the level's instances,
        # which the schedule makes also its LogUp instance and its index in `trees`.
        tree = falsified[0]
        at = [entry["tree"] for entry in levels[joins[tree]]["instances"]].index(tree)
        require(at == tree == order[tree][1] == cheat["instance"], f"{name}: the falsified tree's place is the same under every reading")
        expected["instance"] = at
    batch = {
        "name": name,
        "context": context,
        "alpha": fp2(alpha),
        "beta": fp2(beta),
        "instances": [describe_instance(inst) for inst in instances],
        "forests": [d for _, d in drawn],
        "trees": [{"kind": side, "index": index} for side, index in order],
        "roots": elements(message),
        "levels": levels,
        "values": elements(values),
        "leaf_checks": checks,
        "state": t.state.hex(),
        "counter": t.counter,
        "expected": expected,
    }
    return batch, failing


# ----- A verifier's reading of the LogUp batches ---------------------------------------------------


def parse_fp(text):
    require(isinstance(text, str) and len(text) == 18 and text[:2] == "0x", "an Fp is 0x and 16 hex digits")
    value = int(text[2:], 16)
    require(value < P, "an Fp is canonical")
    return value


def parse_fp2(pair):
    require(isinstance(pair, list) and len(pair) == 2, "an Fp2 is [c0, c1]")
    return (parse_fp(pair[0]), parse_fp(pair[1]))


def fold(table, point):
    """f~(point) by binding x_1 first, as a sumcheck's rounds do: each pair (f[2k], f[2k + 1])
    becomes f[2k] + r (f[2k + 1] - f[2k])."""
    require(len(table) == 1 << len(point), "a table of 2^n entries takes n coordinates")
    for r in point:
        table = [add(table[2 * k], mul(r, sub(table[2 * k + 1], table[2 * k]))) for k in range(len(table) // 2)]
    return table[0]


def check_logup_batch(batch):
    """A verifier's reading of one LogUp batch from the written file (RFC-0019 'Check order' step
    7), apart from the prover above. From the batch's public data (the tables and sizes, the
    forests' public roots, the schedule) and its messages (roots, rounds, evaluations, values)
    alone, it rederives every challenge, claim and point, requires each to be the one recorded,
    and returns every check that fails, in check order: each instance's roots, then each level,
    then each instance's leaf checks, the numerator before the table evaluation. A forest's root
    claim and a table's extension are computed by folding, not by the sum over eq the prover
    uses."""
    t = transcript.Transcript(transcript.session_context(batch["context"]))
    alpha, beta = challenge(t, "logup/alpha"), challenge(t, "logup/beta")
    require([fp2(alpha), fp2(beta)] == [batch["alpha"], batch["beta"]], "alpha and beta are the first two squeezes")
    instances, forests, schedule = batch["instances"], batch["forests"], batch["trees"]
    roots = [parse_fp2(x) for x in batch["roots"]]
    require(len(roots) == 4 * len(instances), "the roots message holds n_a, d_a, n_t, d_t per instance")
    t.absorb("logup/roots", tce(roots))
    failures = []
    for k in range(len(instances)):
        n_a, d_a, n_t, d_t = roots[4 * k:4 * k + 4]
        if d_a == ZERO or d_t == ZERO or add(mul(n_a, d_t), mul(n_t, d_a)) != ZERO:
            failures.append({"reason": "Lookup", "check": "roots", "instance": k})
    heights, roots_of = [], {}
    for k, tree in enumerate(schedule):
        if tree["kind"] == "forest":
            heights.append(forests[tree["index"]]["height"])
            continue
        inst = instances[tree["index"]]
        require(inst["h_a"] == height_of(len(inst["lookups"])) and inst["h_t"] == height_of(len(inst["rows"]))
                and len(inst["multiplicities"]) == 1 << inst["h_t"], "an instance's sizes")
        heights.append(inst["h_a"] if tree["kind"] == "lookup" else inst["h_t"])
        offset = 4 * tree["index"] + (0 if tree["kind"] == "lookup" else 2)
        roots_of[k] = (roots[offset], roots[offset + 1])
    tallest = max(heights)
    require(len(batch["levels"]) == tallest, "levels 0 to H - 1")
    held = {}  # tree -> (point, claim): a forest's value, a fraction tree's (p, q)
    for level, record in enumerate(batch["levels"]):
        prefix = f"tree/{level}"
        joining = [k for k, h in enumerate(heights) if tallest - h == level]
        rows_logs = {k: forests[schedule[k]["index"]]["rows_log"] for k in joining if schedule[k]["kind"] == "forest"}
        s_max = max(rows_logs.values(), default=0)
        root_challenges = [challenge(t, f"{prefix}/root/{c}") for c in range(s_max)]
        for k in joining:
            if k in rows_logs:
                point = root_challenges[s_max - rows_logs[k]:]
                held[k] = (point, fold([lift(parse_fp(v)) for v in forests[schedule[k]["index"]]["roots"]], point))
            else:
                held[k] = ([], roots_of[k])
        present = [k for k in range(len(schedule)) if k in held]
        forest = [schedule[k]["kind"] == "forest" for k in present]
        gamma = challenge(t, f"{prefix}/gamma")
        lam = None if all(forest) else challenge(t, f"{prefix}/lambda")
        n_max = max(len(held[k][0]) for k in present)
        claim, weight = ZERO, ONE
        for k, f in zip(present, forest):
            point, c = held[k]
            claim = add(claim, mul(weight, mul(lift(1 << (n_max - len(point))), c if f else add(c[0], mul(lam, c[1])))))
            weight = mul(weight, gamma)
        rounds = [[parse_fp2(x) for x in m] for m in record["rounds"]]
        require(len(rounds) == n_max and all(len(m) == 3 for m in rounds), f"{prefix}: n_max rounds of 3 values")
        claims, challenges = [claim], []
        for j, (p0, p2, p3) in enumerate(rounds):
            t.absorb(f"{prefix}/round/{j}", tce([p0, p2, p3]))
            r = challenge(t, f"{prefix}/r/{j}")
            claim = lagrange([p0, sub(claim, p0), p2, p3], r)
            claims.append(claim)
            challenges.append(r)
        evals = [parse_fp2(x) for x in record["final_evals"]]
        require(len(evals) == sum(2 if f else 4 for f in forest), f"{prefix}: u, w per forest, N_0, N_1, D_0, D_1 per fraction tree")
        t.absorb(f"{prefix}/evals", tce(evals))
        rho = challenge(t, f"{prefix}/line")
        combination, weight, cursor, entries, nexts = ZERO, ONE, 0, [], []
        for k, f in zip(present, forest):
            point, c = held[k]
            r_prime = challenges[len(challenges) - len(point):]
            if f:
                u, w = evals[cursor:cursor + 2]
                g = mul(u, w)
                entries.append({"tree": k, "point": elements(point), "value": fp2(c)})
                held[k] = ([rho] + r_prime, add(u, mul(rho, sub(w, u))))
                nexts.append({"tree": k, "point": elements(held[k][0]), "value": fp2(held[k][1])})
            else:
                N0, N1, D0, D1 = evals[cursor:cursor + 4]
                g = add(add(mul(N0, D1), mul(N1, D0)), mul(lam, mul(D0, D1)))
                entries.append({"tree": k, "point": elements(point), "numerator": fp2(c[0]), "denominator": fp2(c[1])})
                held[k] = ([rho] + r_prime, (add(N0, mul(rho, sub(N1, N0))), add(D0, mul(rho, sub(D1, D0)))))
                nexts.append({"tree": k, "point": elements(held[k][0]), "numerator": fp2(held[k][1][0]), "denominator": fp2(held[k][1][1])})
            cursor += 2 if f else 4
            combination = add(combination, mul(weight, mul(eq_points(point, r_prime), g)))
            weight = mul(weight, gamma)
        if combination != claim:
            failures.append({"reason": "Sumcheck", "check": f"level {level}"})
        derived = {"prefix": prefix, "root_challenges": elements(root_challenges), "instances": entries, "gamma": fp2(gamma),
                   "lambda": None if lam is None else fp2(lam), "rounds": record["rounds"], "challenges": elements(challenges),
                   "claims": elements(claims), "final_evals": record["final_evals"], "rho": fp2(rho), "next": nexts,
                   "state": t.state.hex(), "counter": t.counter}
        require(derived == record, f"{prefix}: every value of the level is the one the verifier derives")
    for k, h in enumerate(heights):
        if h == 0:
            # A tree of height 0 never joins a level: its root is its leaf claim, on no variable.
            require(k in roots_of, "a tree of height 0 is a fraction tree")
            held[k] = ([], roots_of[k])
    with_values = [k for k, inst in enumerate(instances) if inst["has_values"]]
    values = [parse_fp2(x) for x in batch["values"]]
    require(len(values) == len(with_values), "one value per instance with values")
    t.absorb("logup/values", tce(values))
    checks = []
    for k, inst in enumerate(instances):
        rho_a, (p_a, q_a) = held[schedule.index({"kind": "lookup", "index": k})]
        rho_t, (p_t, q_t) = held[schedule.index({"kind": "table", "index": k})]
        value = values[with_values.index(k)] if inst["has_values"] else ZERO
        rows = [(parse_fp(key), parse_fp(v)) for key, v in inst["rows"]]
        require(inst["has_values"] or not any(v for _, v in rows), "a table without values has every value 0")
        # The table polynomial, its padding rows copies of row 0, folded at rho_t.
        table = [add(lift(key), mul(beta, lift(v))) for key, v in rows]
        table_eval = fold(table + [table[0]] * ((1 << inst["h_t"]) - len(rows)), rho_t)
        if p_a != ONE:
            failures.append({"reason": "Lookup", "check": "leaf numerator", "instance": k})
        if q_t != sub(alpha, table_eval):
            failures.append({"reason": "Lookup", "check": "table evaluation", "instance": k})
        checks.append({
            "lookup": {"point": elements(rho_a), "numerator": fp2(p_a), "denominator": fp2(q_a),
                       "value": fp2(value) if inst["has_values"] else None, "key": fp2(sub(sub(alpha, q_a), mul(beta, value)))},
            "table": {"point": elements(rho_t), "numerator": fp2(p_t), "multiplicity": fp2(neg(p_t)), "denominator": fp2(q_t),
                      "table_eval": fp2(table_eval)},
        })
    require(checks == batch["leaf_checks"], "every leaf check is the one the verifier derives")
    require([t.state.hex(), t.counter] == [batch["state"], batch["counter"]], "the transcript after logup/values")
    return failures


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
    for batch in batches:
        require(trees_as_forests(batch) == batch["levels"], f"{batch['name']}: the trees, proven as forests of one tree, give the same levels")
    forest_batches = [
        forest_batch(rng, "a forest of 2 trees of height 1, alone", [(1, 1, [])]),
        forest_batch(rng, "a SoftmaxRow forest of 8 trees of height 3, every tree with a zero leaf, alone", [(3, 3, "rows")]),
        forest_batch(rng, "forests of 4 and 2 trees and a single tree, of height 4; a forest of 2 trees of height 2, one with a "
                     "zero leaf; two single trees of height 1", [(2, 4, []), (1, 4, []), (0, 4, []), (1, 2, [(0, 1)]), (0, 1, []), (0, 1, [])]),
        forest_batch(rng, "a single tree of height 3; a SoftmaxRow forest of 16 trees whose first leaves are zero and a forest "
                     "of 4 trees, of height 2", [(0, 3, []), (4, 2, "first"), (2, 2, [])]),
    ]
    # LogUp batches, drawn after the forests. The schedule of each is the vectors' choice.
    lookup, table, forest = "lookup", "table", "forest"

    def paired(count, forests=0):
        """Each instance's lookup tree, then its table tree, in instance order, then the forests."""
        return [(side, k) for k in range(count) for side in (lookup, table)] + [(forest, f) for f in range(forests)]

    logup = [
        logup_batch(rng, "one instance alone: N = 6 and T = 7, both trees of height 3", [(7, 6, True)], [], paired(1)),
        logup_batch(rng, "one instance alone with N = 1 and T = 1: both trees of height 0, so no level, and the values follow the roots",
                    [(1, 1, True)], [], paired(1)),
        logup_batch(rng, "three instances whose trees have heights 2 and 1, 1 and 2, 0 and 2, joining at levels 0 and 1, 1 and 0, "
                    "never and 0; the second table without values", [(2, 3, True), (4, 2, False), (3, 1, True)], [], paired(3)),
        logup_batch(rng, "four instances padded on both sides, N of 3, 4, 5 and 2 and T of 5, 4, 3 and 1; the lookup trees first, "
                    "then the table trees", [(5, 3, True), (4, 4, True), (3, 5, False), (1, 2, True)], [],
                    [(lookup, 0), (lookup, 1), (lookup, 2), (lookup, 3), (table, 0), (table, 1), (table, 2), (table, 3)]),
        logup_batch(rng, "three instances beside three forests, the schedule interleaving them: level 0 has a forest of 2 trees of "
                    "height 4 alone and no lambda; fraction trees join at levels 1, 2 and 3; at level 2 a SoftmaxRow forest of 4 "
                    "trees joins under two root challenges while fraction trees are present; the third instance has N = T = 1",
                    [(4, 5, True), (6, 2, False), (1, 1, True)], [(1, 4, []), (2, 2, "rows"), (0, 3, [])],
                    [(table, 1), (forest, 2), (lookup, 0), (forest, 0), (table, 0), (lookup, 2), (forest, 1), (lookup, 1), (table, 2)]),
        logup_batch(rng, "a lookup tree of height 10 and a table tree of height 5: N = 513 and T = 17, padded on both sides, "
                    "the table tree first", [(17, 513, True)], [], [(table, 0), (lookup, 0)]),
        logup_batch(rng, "two instances with N = 1 and T = 1 beside a forest of 2 trees of height 2: the forest is alone at every "
                    "level, so no lambda is squeezed", [(1, 1, True), (1, 1, False)], [(1, 2, [])],
                    [(forest, 0), (lookup, 0), (table, 0), (lookup, 1), (table, 1)]),
        logup_batch(rng, "not a cheat: the lookup tree's padding, lookups of row 0, counted on the table's padding row, a copy of "
                    "row 0, instead of on row 0; the batch verifies", [(3, 3, True)], [], paired(1),
                    {"kind": "padding multiplicity", "instance": 0}),
        logup_batch(rng, "one instance whose table has no values: logup/values is absorbed empty, after the last level",
                    [(3, 2, False)], [], paired(1)),
        logup_batch(rng, "no LogUp instance, a forest of 2 trees of height 1 alone: logup/roots and logup/values are absorbed empty",
                    [], [(1, 1, [])], [(forest, 0)]),
        logup_batch(rng, "cheating: a lookup outside a table without values, key T, with honest trees", [(1, 1, True), (2, 2, False)],
                    [], paired(2), {"kind": "outside", "instance": 1, "lookup": 1}),
        logup_batch(rng, "cheating: a lookup with a wrong value, counted in its key's row, with honest trees", [(2, 2, True)], [],
                    paired(1), {"kind": "wrong value", "instance": 0, "lookup": 1}),
        logup_batch(rng, "cheating: a multiplicity off by one, with honest trees, beside a forest", [(2, 2, True)], [(1, 1, [])],
                    paired(1, 1), {"kind": "multiplicity", "instance": 0, "row": 1}),
        logup_batch(rng, "cheating: the roots message gives a lookup tree the root (0, 0), which passes n_a d_t + n_t d_a = 0",
                    [(1, 1, True), (2, 2, True)], [], paired(2), {"kind": "zero root", "instance": 1, "side": lookup}),
        logup_batch(rng, "cheating: the roots message gives a table tree the root (0, 0), which passes n_a d_t + n_t d_a = 0",
                    [(2, 2, True)], [], paired(1), {"kind": "zero root", "instance": 0, "side": table}),
        logup_batch(rng, "cheating: false roots made to sum to zero, hiding a multiplicity off by one, with honest trees: level 0 fails",
                    [(2, 2, True)], [], paired(1), {"kind": "false roots", "instance": 0, "row": 1}),
        logup_batch(rng, "cheating: a lookup tree's root scaled by a constant, the same fraction, with honest trees; the tree joins at "
                    "level 1, which fails", [(2, 2, True), (2, 3, True)], [], paired(2), {"kind": "scaled roots", "instance": 0}),
        logup_batch(rng, "cheating: a lookup outside the table whose leaf numerator the prover set to 0, so that the roots sum to zero "
                    "and the levels pass", [(1, 1, True), (2, 2, False)], [(1, 1, [])], paired(2, 1),
                    {"kind": "numerator", "instance": 1, "lookup": 1}),
        logup_batch(rng, "cheating: a prover whose trees use a table with one row's value changed, the row its lookups use, with "
                    "honest trees over it", [(1, 1, False), (3, 4, True)], [], paired(2), {"kind": "table", "instance": 1}),
        logup_batch(rng, "cheating: the phantom row, a lookup of (0, 0), which is no row, and the multiplicity 1 on the padding row; "
                    "honest trees, whose padding leaf (-1, alpha - t_0) counts a lookup of row 0, so the roots fail",
                    [(3, 2, True)], [], paired(1), {"kind": "phantom row", "instance": 0, "lookup": 1}),
        logup_batch(rng, "cheating: the phantom row under the zero padding of the text before #62, the padding leaf (-1, alpha): the "
                    "roots, the levels and the numerator pass, and the table evaluation, padded with row 0, fails",
                    [(3, 2, True)], [], paired(1), {"kind": "phantom row, zero padding", "instance": 0, "lookup": 1}),
    ]
    logup_batches = [batch for batch, _ in logup]
    header = {
        "description": "RFC-0010 sumcheck reference vectors: honest proofs of batched sumchecks "
        "and of product-tree levels, written by spec/vectors/sumcheck/generate.py; see README.md",
        "modulus": hex64(P),
        "non_residue": W,
        "seed": hex64(SEED),
        "official_blake3_cases_checked": checked,
    }
    text = render(header, sessions, batches, forest_batches, logup_batches)
    # A verifier reads every LogUp batch back from the file and must stop where the batch says.
    for batch, (_, failing) in zip(json.loads(text)["logup_batches"], logup):
        failures = check_logup_batch(batch)
        require(failures == failing, f"{batch['name']}: the verifier finds exactly the failing checks {failing}, not {failures}")
        expected = batch["expected"]
        require(expected == "accept" if not failures else all(expected[key] == failures[0][key] for key in failures[0]),
                f"{batch['name']}: the expected verdict is the first failing check")
    (HERE / "sumcheck.json").write_text(text)
    stages = sum(len(s["stages"]) for s in sessions)
    levels = sum(len(b["levels"]) for b in batches)
    forest_levels_count = sum(len(b["levels"]) for b in forest_batches)
    logup_levels = sum(len(b["levels"]) for b in logup_batches)
    rejected = sum(b["expected"] != "accept" for b in logup_batches)
    print(f"wrote sumcheck.json: {len(sessions)} sessions of {stages} stages, {len(batches)} tree batches of {levels} levels, "
          f"{len(forest_batches)} forest batches of {forest_levels_count} levels, {len(logup_batches)} LogUp batches "
          f"({len(logup_batches) - rejected} accepted, {rejected} rejected) of {logup_levels} levels, {len(text)} bytes")


if __name__ == "__main__":
    main()
