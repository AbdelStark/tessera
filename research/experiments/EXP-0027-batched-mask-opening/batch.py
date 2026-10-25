#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0 OR MIT
"""EXP-0027: the masked batching sumcheck of IDEA-0040 (research/, a dependency of nothing).

Three parts, each exact or a count, run with `python3 -I`:

- `a`: the rank of the functionals of a pad `S` that the batching sumcheck reveals, against their
  count, over F65537 and Goldilocks, for protocol-like and adversarial `C`;
- `b`: an exact distinguisher over F65537 on a toy protocol (masked stages, tree-level-like mask
  rows, the batching), with the candidate pad and the negative controls;
- `c`: a Monte Carlo count of the cheating proofs the batching accepts over F65537, against the
  analytic bound, with order controls.

    python3 -I batch.py all <results-dir> [--workers N]
    python3 -I batch.py a|b|c <results-dir> [--workers N]

The method and the success criterion are the pre-registered ones in README.md.
"""

from __future__ import annotations

import argparse
import datetime
import json
import multiprocessing
import platform
import sys
import time
from pathlib import Path

Q65537 = 65537
GOLDILOCKS = 2**64 - 2**32 + 1
SEED_A = 0x45800270001
SEED_B = 0x45800270002
SEED_C = 0x45800270003
GOLDEN = 0x9E3779B97F4A7C15
MASK64 = (1 << 64) - 1


class SplitMix64:
    """SplitMix64, with uniform integers by rejection sampling."""

    def __init__(self, seed: int) -> None:
        self.state = seed & MASK64

    def next(self) -> int:
        self.state = (self.state + GOLDEN) & MASK64
        z = self.state
        z = ((z ^ (z >> 30)) * 0xBF58476D1CE4E5B9) & MASK64
        z = ((z ^ (z >> 27)) * 0x94D049BB133111EB) & MASK64
        return z ^ (z >> 31)

    def below(self, n: int) -> int:
        limit = (1 << 64) - ((1 << 64) % n)
        while True:
            x = self.next()
            if x < limit:
                return x % n

    def nonzero(self, p: int) -> int:
        return 1 + self.below(p - 1)

    def bits(self, k: int) -> int:
        return self.next() & ((1 << k) - 1)


def worker_seed(seed: int, w: int) -> int:
    """Worker w's seed: the (w + 1)-th output of SplitMix64 at `seed` (the 2026-10-25 amendment).

    The pre-registered `seed + w·GOLDEN` gave the stream of `seed` shifted by `w` draws."""
    master = SplitMix64(seed)
    out = 0
    for _ in range(w + 1):
        out = master.next()
    return out


# ---------------------------------------------------------------------------------------------
# Multilinear helpers. Position x of a cube has bit i as variable i + 1, the lowest first.


def eq_table(r: list[int], p: int) -> list[int]:
    t = [1]
    for ri in r:
        o = (1 - ri) % p
        t = [v * o % p for v in t] + [v * ri % p for v in t]
    return t


def mle(f: list[int], r: list[int], p: int) -> int:
    t = [x % p for x in f]
    for ri in r:
        t = [(t[2 * i] + ri * (t[2 * i + 1] - t[2 * i])) % p for i in range(len(t) // 2)]
    return t[0]


def inv(a: int, p: int) -> int:
    a %= p
    if a == 0:
        raise ZeroDivisionError("inverse of zero")
    return pow(a, p - 2, p)


def rank(vectors: list[list[int]], cols: list[int], p: int) -> int:
    """The rank over F_p of the vectors restricted to the columns `cols`."""
    pivots: list[tuple[int, list[int]]] = []
    for vec in vectors:
        row = [vec[c] % p for c in cols]
        for col, prow in pivots:
            f = row[col]
            if f:
                row = [(a - f * b) % p for a, b in zip(row, prow)]
        lead = next((i for i, a in enumerate(row) if a), None)
        if lead is not None:
            s = inv(row[lead], p)
            pivots.append((lead, [a * s % p for a in row]))
    return len(pivots)


# ---------------------------------------------------------------------------------------------
# The batching sumcheck over two tables, the lowest variable first.


def batch_rounds(ct: list, zt: list, r: list[int], p: int, combo) -> tuple[list, list]:
    """Round coefficients (A_j, B_j, Q_j) of Σ_x C̃(x)·z̃(x) and the final folded values.

    `ct` holds field elements; `zt` field elements or affine forms, combined by `combo`."""
    ct = list(ct)
    zt = list(zt)
    rounds = []
    for rj in r:
        half = len(ct) // 2
        a_terms, b_terms, q_terms = [], [], []
        for y in range(half):
            c0, c1 = ct[2 * y], ct[2 * y + 1]
            z0, z1 = zt[2 * y], zt[2 * y + 1]
            dc = (c1 - c0) % p
            if c0:
                a_terms.append((c0, z0))
                b_terms.append((c0, z1))
            b_terms.append(((dc - c0) % p, z0))
            if dc:
                q_terms.append((dc, z1))
                q_terms.append((-dc % p, z0))
        rounds.append((combo(a_terms), combo(b_terms), combo(q_terms)))
        o = (1 - rj) % p
        ct = [(ct[2 * y] * o + ct[2 * y + 1] * rj) % p for y in range(half)]
        zt = [combo([(o, zt[2 * y]), (rj, zt[2 * y + 1])]) for y in range(half)]
    return rounds, [ct[0], zt[0]]


def int_combo(p: int):
    def combo(pairs) -> int:
        return sum(s * x for s, x in pairs) % p

    return combo


def batch_rounds_int(ct: list[int], zt: list[int], r: list[int], p: int) -> tuple[list, list]:
    """`batch_rounds` on field elements only, for the soundness loop."""
    rounds = []
    for rj in r:
        c0s, c1s = ct[0::2], ct[1::2]
        z0s, z1s = zt[0::2], zt[1::2]
        a = b = q = 0
        for c0, c1, z0, z1 in zip(c0s, c1s, z0s, z1s):
            dc = c1 - c0
            a += c0 * z0
            b += c0 * z1 + (dc - c0) * z0
            q += dc * (z1 - z0)
        rounds.append((a % p, b % p, q % p))
        o = 1 - rj
        ct = [(c0 * o + c1 * rj) % p for c0, c1 in zip(c0s, c1s)]
        zt = [(z0 * o + z1 * rj) % p for z0, z1 in zip(z0s, z1s)]
    return rounds, [ct[0], zt[0]]


def functionals(c: list[int], r: list[int], p: int) -> dict[str, list[list[int]]]:
    """The vectors w with ⟨w, S⟩ equal to each revealed coefficient on z = S."""
    m = len(r)
    n = 1 << m
    out = {"gamma": [list(c)], "A": [], "B": [], "Q": [], "final": [eq_table(r, p)]}
    cf = list(c)
    for j in range(m):
        eqlow = eq_table(r[:j], p)
        low = (1 << j) - 1
        wa, wb, wq = [0] * n, [0] * n, [0] * n
        for x in range(n):
            e = eqlow[x & low]
            hx = x >> j
            b = hx & 1
            y = hx >> 1
            c0, c1 = cf[2 * y], cf[2 * y + 1]
            dc = (c1 - c0) % p
            if b:
                wq[x] = e * dc % p
                wb[x] = e * c0 % p
            else:
                wq[x] = -e * dc % p
                wb[x] = e * (dc - c0) % p
                wa[x] = e * c0 % p
        out["A"].append(wa)
        out["B"].append(wb)
        out["Q"].append(wq)
        o = (1 - r[j]) % p
        cf = [(cf[2 * y] * o + cf[2 * y + 1] * r[j]) % p for y in range(len(cf) // 2)]
    return out


# ---------------------------------------------------------------------------------------------
# Layouts shared by (a), (b) and (c).

SLOTS_PER_ROUND = 8


def poly_block(n: int) -> int:
    size = SLOTS_PER_ROUND * (n + 1)
    return 1 << (size - 1).bit_length()


def place(blocks: list[tuple]) -> list[tuple]:
    """Blocks (kind, size, payload) in decreasing size, each aligned at its size."""
    ordered = sorted(blocks, key=lambda b: -b[1])
    out, offset = [], 0
    for kind, size, payload in ordered:
        out.append((kind, size, payload, offset))
        offset += size
    return out


def layout_total(blocks: list[tuple]) -> int:
    return sum(b[1] for b in blocks)


# ---------------------------------------------------------------------------------------------
# (a) The rank experiment.

PROTO_SPECS = {
    6: ([2], [1, 2, 3, 4]),
    8: ([2, 3, 4], [1, 2, 3, 4, 5, 6]),
    10: ([2, 3, 4, 5, 6, 7], [1, 2, 3, 4, 5, 6, 7, 8, 7, 6]),
}


def protocol_like(m: int, polys: list[int], rows: list[int], rng: SplitMix64, p: int):
    """C and T_layout for masking polynomials (d_j = 3, d_n = 7) and eq rows."""
    blocks = [("poly", poly_block(n), n) for n in polys] + [("row", 1 << c, c) for c in rows]
    placed = place(blocks)
    assert layout_total(placed) <= 1 << m
    alpha = rng.nonzero(p)
    c = [0] * (1 << m)
    layout = []
    weight = 1
    for kind, _size, payload, off in placed:
        if kind == "poly":
            n = payload
            pt = [rng.below(p) for _ in range(n)]
            c[off] = weight
            layout.append(off)
            for j in range(1, n + 1):
                d = 7 if j == n else 3
                for e in range(d + 1):
                    pos = off + SLOTS_PER_ROUND * j + e
                    c[pos] = weight * pow(pt[j - 1], e, p) % p
                    layout.append(pos)
        else:
            width = payload
            pt = [rng.below(p) for _ in range(width)]
            t = eq_table(pt, p)
            for x in range(1 << width):
                c[off + x] = weight * t[x] % p
                layout.append(off + x)
        weight = weight * alpha % p
    return c, sorted(layout)


def cycling(m: int, values: list[int], size_of) -> list[int]:
    chosen, total, i = [], 0, 0
    while True:
        v = values[i % len(values)]
        s = size_of(v)
        if total + s > 1 << m:
            break
        chosen.append(v)
        total += s
        i += 1
    return chosen


def family_c(name: str, arg, m: int, rng: SplitMix64, p: int):
    n = 1 << m
    full = list(range(n))
    if name == "PROTO":
        polys, rows = PROTO_SPECS[m]
        return protocol_like(m, polys, rows, rng, p)
    if name == "POLY":
        return protocol_like(m, cycling(m, [2, 3, 4, 5, 6, 7], poly_block), [], rng, p)
    if name == "ROWS":
        return protocol_like(m, [], cycling(m, list(range(1, m - 1)), lambda c: 1 << c), rng, p)
    if name == "DENSE":
        return [rng.nonzero(p) for _ in range(n)], full
    if name == "SPARSE":
        c = [0] * n
        positions = set()
        while len(positions) < arg:
            positions.add(rng.below(n))
        for x in positions:
            c[x] = rng.nonzero(p)
        return c, full
    if name == "SUBCUBE":
        variables = []
        while len(variables) < arg:
            v = rng.below(m)
            if v not in variables:
                variables.append(v)
        bits = {v: rng.bits(1) for v in variables}
        c = [rng.nonzero(p) if all((x >> v) & 1 == b for v, b in bits.items()) else 0 for x in range(n)]
        return c, full
    if name == "SINGLE":
        c = [0] * n
        c[rng.below(n)] = rng.nonzero(p)
        return c, full
    if name == "EQ1":
        return eq_table([rng.below(p) for _ in range(m)], p), full
    if name == "BOOLPT":
        pt = [rng.below(p) for _ in range(m)]
        chosen = []
        while len(chosen) < arg:
            v = rng.below(m)
            if v not in chosen:
                chosen.append(v)
        for v in chosen:
            pt[v] = rng.bits(1)
        return eq_table(pt, p), full
    raise ValueError(name)


def families(m: int) -> list[tuple[str, object]]:
    out = [("PROTO", None), ("POLY", None), ("ROWS", None), ("DENSE", None)]
    out += [("SPARSE", s) for s in (2, 3, 5, 2 * m)]
    out += [("SUBCUBE", j) for j in (1, 2, m // 2)]
    out += [("SINGLE", None), ("EQ1", None), ("BOOLPT", 1), ("BOOLPT", 2)]
    return out


def spread_support(m: int) -> list[int]:
    c = m - 4
    return [x for x in range(1 << m) if bin(x >> c).count("1") <= 2]


def rank_cell(task):
    """One (field, m, family) cell of part (a)."""
    p, m, name, arg, draws, seed = task
    rng = SplitMix64(seed)
    n = 1 << m
    spread = spread_support(m)
    records = []
    combo = int_combo(p)
    for _ in range(draws):
        c, layout = family_c(name, arg, m, rng, p)
        r = [rng.below(p) for _ in range(m)]
        w = functionals(c, r, p)
        # Correctness: the executable prover on z = S against ⟨w, S⟩.
        s = [rng.below(p) for _ in range(n)]
        rounds, (cfin, sfin) = batch_rounds(c, s, r, p, combo)
        dot = lambda u: sum(a * b for a, b in zip(u, s)) % p  # noqa: E731
        ok = dot(w["gamma"][0]) == sum(a * b for a, b in zip(c, s)) % p
        ok &= dot(w["final"][0]) == sfin
        ok &= cfin == mle(c, r, p)
        for j in range(m):
            ok &= dot(w["A"][j]) == rounds[j][0]
            ok &= dot(w["B"][j]) == rounds[j][1]
            ok &= dot(w["Q"][j]) == rounds[j][2]
        listed = w["gamma"] + [v for j in range(m) for v in (w["B"][j], w["Q"][j])] + w["final"]
        supp = [x for x in range(n) if c[x]]
        lay = sorted(set(layout))
        rk = lambda cols: rank(listed, cols, p)  # noqa: E731
        r_full = rk(list(range(n)))
        r_l1 = rk(supp)
        r_l2 = rk(lay)
        r_l3 = rk(spread)
        r_l1u = rk(sorted(set(supp) | set(lay)))
        r_l3u = rk(sorted(set(spread) | set(lay)))
        records.append({
            "ok": ok,
            "full": r_full,
            "L1": r_l1,
            "L2": r_l2,
            "L3": r_l3,
            "leak_L1": r_l1u - r_l1,
            "leak_L2": r_l2 - r_l2,
            "leak_L3": r_l3u - r_l3,
            "supp": len(supp),
            "layout": len(lay),
            "c_at_r": int(cfin != 0),
        })
    return {"p": p, "m": m, "family": name, "arg": arg, "draws": draws, "records": records}


def summarize_rank(cell) -> dict:
    recs = cell["records"]
    m = cell["m"]
    count = 2 * m + 1

    def stats(key):
        vals = [r[key] for r in recs]
        return {"min": min(vals), "max": max(vals), "mode": max(set(vals), key=vals.count),
                "mode_share": vals.count(max(set(vals), key=vals.count)) / len(vals)}

    return {
        "field": "F65537" if cell["p"] == Q65537 else "Goldilocks",
        "m": m,
        "family": cell["family"] + ("" if cell["arg"] is None else f"({cell['arg']})"),
        "draws": cell["draws"],
        "count_listed": 2 * m + 2,
        "count_independent": count,
        "correct": all(r["ok"] for r in recs),
        "full": stats("full"),
        "L1": stats("L1"),
        "L2": stats("L2"),
        "L3": stats("L3"),
        "leak_L1": stats("leak_L1"),
        "leak_L2_max": max(r["leak_L2"] for r in recs),
        "leak_L3": stats("leak_L3"),
        "full_eq_count": sum(r["full"] == count for r in recs) / len(recs),
        "L1_eq_count": sum(r["L1"] == count for r in recs) / len(recs),
        "L3_eq_count": sum(r["L3"] == count for r in recs) / len(recs),
        "supp": stats("supp"),
        "layout": stats("layout"),
    }


def predictions(rows: list[dict]) -> list[dict]:
    """P-a1 to P-a5 of the pre-registration."""
    out = []
    for row in rows:
        fam, m, field = row["family"], row["m"], row["field"]
        need = 0.98 if field == "F65537" else 1.0
        cell = row["_cell"]
        recs = cell["records"]
        count = 2 * m + 1
        if fam == "DENSE":
            ok = sum(r["full"] == count and r["L1"] == count for r in recs) / len(recs)
            out.append({"id": "P-a1", "field": field, "m": m, "family": fam, "share": ok, "need": need, "holds": ok >= need})
        elif fam == "SINGLE":
            ok = sum(r["full"] == m + 1 and r["L1"] == 1 for r in recs) / len(recs)
            out.append({"id": "P-a2", "field": field, "m": m, "family": fam, "share": ok, "need": 1.0, "holds": ok >= 1.0})
        elif fam.startswith("SUBCUBE"):
            j = cell["arg"]
            ok = sum(r["full"] == count - j and r["L1"] == count - 2 * j for r in recs) / len(recs)
            out.append({"id": "P-a3", "field": field, "m": m, "family": fam, "share": ok, "need": need, "holds": ok >= need})
        elif fam in ("PROTO", "POLY", "ROWS"):
            ok = sum(r["L1"] == r["L2"] and r["leak_L1"] == 0 for r in recs) / len(recs)
            out.append({"id": "P-a4", "field": field, "m": m, "family": fam, "share": ok, "need": need, "holds": ok >= need})
        elif fam == "EQ1":
            ok = sum(r["full"] == m + 1 for r in recs) / len(recs)
            out.append({"id": "P-a5", "field": field, "m": m, "family": fam, "share": ok, "need": need, "holds": ok >= need})
    return out


def part_a(workers: int) -> dict:
    tasks = []
    index = 0
    for p in (Q65537, GOLDILOCKS):
        for m in (6, 8, 10):
            draws = 200 if m < 10 else 50
            for name, arg in families(m):
                tasks.append((p, m, name, arg, draws, worker_seed(SEED_A, index)))
                index += 1
    if workers > 1:
        with multiprocessing.Pool(workers) as pool:
            cells = pool.map(rank_cell, tasks, chunksize=1)
    else:
        cells = [rank_cell(t) for t in tasks]
    rows = []
    for cell in cells:
        row = summarize_rank(cell)
        row["_cell"] = cell
        rows.append(row)
    preds = predictions(rows)
    for row in rows:
        del row["_cell"]
    verdict = {
        "correct_all": all(r["correct"] for r in rows),
        "leak_L2_all_zero": all(r["leak_L2_max"] == 0 for r in rows),
        "draws": sum(r["draws"] for r in rows),
    }
    return {"cells": rows, "predictions": preds, "criteria": verdict}


# ---------------------------------------------------------------------------------------------
# (b) The toy protocol, over F65537, as affine forms in the masking randomness.

P_B = Q65537


class Form:
    """k + Σ v[i]·s_i over F65537: a value as an affine function of the masking randomness."""

    __slots__ = ("k", "v")

    def __init__(self, k: int = 0, v: dict | None = None) -> None:
        self.k = k
        self.v = v if v is not None else {}


def form_combo(pairs) -> int | Form:
    p = P_B
    k = 0
    v = None
    for s, x in pairs:
        s %= p
        if not s:
            continue
        if isinstance(x, Form):
            k += s * x.k
            if v is None:
                v = {}
            for var, c in x.v.items():
                v[var] = (v.get(var, 0) + s * c) % p
        else:
            k += s * x
    k %= p
    if v is None:
        return k
    v = {a: c for a, c in v.items() if c}
    return Form(k, v) if v else k


class FormSource:
    """Masking randomness as fresh variables."""

    def __init__(self, start: int = 0) -> None:
        self.count = start

    def rand(self) -> Form:
        f = Form(0, {self.count: 1})
        self.count += 1
        return f

    def fork(self) -> "FormSource":
        return FormSource(self.count)


class NumSource:
    """Masking randomness as uniform field elements, recorded in order."""

    def __init__(self, rng: SplitMix64, start: list | None = None) -> None:
        self.rng = rng
        self.drawn = list(start or [])

    def rand(self) -> int:
        x = self.rng.below(P_B)
        self.drawn.append(x)
        return x

    def fork(self) -> "NumSource":
        return NumSource(self.rng, self.drawn)


def vandermonde_inverse(d: int) -> list[list[int]]:
    """Coefficients from the evaluations at 0..d."""
    p = P_B
    size = d + 1
    mat = [[pow(x, e, p) for e in range(size)] + [int(i == x) for i in range(size)] for x in range(size)]
    for col in range(size):
        piv = next(r for r in range(col, size) if mat[r][col])
        mat[col], mat[piv] = mat[piv], mat[col]
        s = inv(mat[col][col], p)
        mat[col] = [a * s % p for a in mat[col]]
        for r in range(size):
            if r != col and mat[r][col]:
                f = mat[r][col]
                mat[r] = [(a - f * b) % p for a, b in zip(mat[r], mat[col])]
    return [row[size:] for row in mat]


VINV = {d: vandermonde_inverse(d) for d in (1, 2, 3, 4, 7)}


def stage_degrees(n: int) -> list[int]:
    return [2] * (n - 1) + [4]


class Public:
    """One draw's public data and challenges for (b), shared by both witnesses and every arm."""

    def __init__(self, config: dict, rng: SplitMix64) -> None:
        p = P_B
        self.config = config
        self.stages = []
        for n in config["stages"]:
            table = [rng.below(p) for _ in range(1 << n)]
            r = [rng.below(p) for _ in range(n)]
            self.stages.append({"n": n, "P": table, "rho": rng.nonzero(p), "r": r})
        self.rows = []
        for width, claimed in config["rows"]:
            q = [rng.below(p) for _ in range(max(width - 1, 1))]
            pt = [rng.below(p) for _ in range(width)]
            extra = [rng.below(p) for _ in range(width)]
            self.rows.append({"c": width, "claimed": claimed, "q": q, "p": pt, "p2": extra})
        self.alpha = rng.nonzero(p)
        self.rho = rng.nonzero(p)
        self.blocks = place(
            [("poly", poly_block(s["n"]), i) for i, s in enumerate(self.stages)]
            + [("row", 1 << rw["c"], i) for i, rw in enumerate(self.rows)]
        )
        if config.get("explicit"):
            self.blocks = config["explicit"](self)
        self.m = max(2, (layout_total(self.blocks) - 1).bit_length())
        self.r_batch = [rng.below(p) for _ in range(self.m)]


def explicit_unclaimed(pub: Public) -> list[tuple]:
    """UNCLAIMED: claimed rows at [0, 8), unclaimed rows at [8, 16)."""
    claimed = [i for i, rw in enumerate(pub.rows) if rw["claimed"]]
    unclaimed = [i for i, rw in enumerate(pub.rows) if not rw["claimed"]]
    out, off = [], 0
    for i in claimed + unclaimed:
        size = 1 << pub.rows[i]["c"]
        out.append(("row", size, i, off))
        off += size
    return out


def witness(pub: Public, rng: SplitMix64, base: dict | None = None) -> dict:
    """A witness, or a second one with the same public sums as `base`."""
    p = P_B
    stages = []
    for k, st in enumerate(pub.stages):
        size = 1 << st["n"]
        if base is None:
            stages.append([rng.below(p) for _ in range(size)])
        else:
            table = st["P"]
            delta = [rng.below(p) for _ in range(size)]
            i0 = next(i for i in range(size) if table[i])
            rest = sum(table[i] * delta[i] for i in range(size) if i != i0) % p
            delta[i0] = -rest * inv(table[i0], p) % p
            stages.append([(a + d) % p for a, d in zip(base["stages"][k], delta)])
    rows = [[rng.below(p) for _ in range(1 << rw["c"])] for rw in pub.rows]
    return {"stages": stages, "rows": rows}


def z_factor(pt: list[int]) -> int:
    p = P_B
    out = 1
    for x in pt:
        out = out * x % p * ((1 - x) % p) % p
    return out


def run_old(pub: Public, wit: dict, src) -> tuple[dict, dict, list]:
    """The view before the batching, M's values by position, and the claims (vector, value)."""
    p = P_B
    combo = form_combo
    view: dict[str, object] = {}
    mvals: dict[int, object] = {}
    claims = []
    offsets = {(kind, payload): off for kind, _s, payload, off in pub.blocks}
    for k, st in enumerate(pub.stages):
        n, table, rho, r = st["n"], st["P"], st["rho"], st["r"]
        a_tab = wit["stages"][k]
        degs = stage_degrees(n)
        a0 = src.rand()
        g = [[src.rand() for _ in range(d + 1)] for d in degs]
        r_a = [src.rand() for _ in range(1 << n)]
        off = offsets[("poly", k)]
        mvals[off] = a0
        for j in range(n):
            for e in range(degs[j] + 1):
                mvals[off + SLOTS_PER_ROUND * (j + 1) + e] = g[j][e]
        h = sum(x * y for x, y in zip(table, a_tab)) % p
        view[f"stage/{k}/H"] = h
        terms = [(1 << n, a0)]
        for j in range(n):
            terms.append((1 << n, g[j][0]))
            for e in range(1, degs[j] + 1):
                terms.append((1 << (n - 1), g[j][e]))
        gamma = combo(terms)
        view[f"stage/{k}/zk-sum"] = gamma

        def g_at(pt):
            pairs = [(1, a0)]
            for j in range(n):
                for e in range(degs[j] + 1):
                    pairs.append((pow(pt[j], e, p), g[j][e]))
            return combo(pairs)

        def f_at(pt):
            zf = z_factor(pt[1:])
            ra = combo(list(zip(eq_table(pt, p), r_a)))
            a_prime = combo([(1, mle(a_tab, pt, p)), (zf, ra)])
            return combo([(mle(table, pt, p), a_prime), (rho, g_at(pt))])

        for j in range(n):
            d = degs[j]
            evals = []
            for xv in range(d + 1):
                acc = []
                for tail in range(1 << (n - j - 1)):
                    pt = r[:j] + [xv] + [(tail >> i) & 1 for i in range(n - j - 1)]
                    acc.append((1, f_at(pt)))
                evals.append(combo(acc))
            vinv = VINV[d]
            for e in range(d + 1):
                view[f"stage/{k}/round/{j}/c{e}"] = combo(list(zip(vinv[e], evals)))
        zf = z_factor(r[1:])
        ra = combo(list(zip(eq_table(r, p), r_a)))
        view[f"stage/{k}/final"] = combo([(1, mle(a_tab, r, p)), (zf, ra)])
        vec = {off: 1}
        for j in range(n):
            for e in range(degs[j] + 1):
                vec[off + SLOTS_PER_ROUND * (j + 1) + e] = pow(r[j], e, p)
        claims.append((vec, g_at(r)))
    for t, rw in enumerate(pub.rows):
        width = rw["c"]
        vtab = wit["rows"][t]
        rmask = [src.rand() for _ in range(1 << width)]
        off = offsets[("row", t)]
        for x in range(1 << width):
            mvals[off + x] = rmask[x]
        if width == 1:
            q = rw["q"][0]
            pt = [q]
            view[f"row/{t}/y"] = combo([(mle(vtab, pt, p), 1), (q * (1 - q) % p, combo(list(zip(eq_table(pt, p), rmask))))])
        else:
            zq = z_factor(rw["q"])
            for b in (0, 1):
                pt = [b] + rw["q"]
                masked = combo(list(zip(eq_table(pt, p), rmask)))
                view[f"row/{t}/y{b}"] = combo([(mle(vtab, pt, p), 1), (zq, masked)])
        if rw["claimed"]:
            et = eq_table(rw["p"], p)
            value = combo(list(zip(et, rmask)))
            view[f"row/{t}/u"] = value
            claims.append(({off + x: et[x] for x in range(1 << width)}, value))
        else:
            # Its other opening, outside C.
            view[f"row/{t}/u2"] = combo(list(zip(eq_table(rw["p2"], p), rmask)))
    return view, mvals, claims


def batch_c(pub: Public, claims: list) -> list[int]:
    p = P_B
    c = [0] * (1 << pub.m)
    weight = 1
    for vec, _value in claims:
        for pos, coef in vec.items():
            c[pos] = (c[pos] + weight * coef) % p
        weight = weight * pub.alpha % p
    return c


def run_batch(pub: Public, arm: str, mvals: dict, claims: list, src) -> dict:
    """The batching view of `arm`."""
    p = P_B
    combo = form_combo
    size = 1 << pub.m
    c = batch_c(pub, claims)
    mvec = [mvals.get(x, 0) for x in range(size)]
    view: dict[str, object] = {}
    if arm == "NOS":
        zvec = mvec
    else:
        if arm in ("CAND", "SPLIT"):
            support = sorted(mvals)
        elif arm == "L1":
            support = [x for x in range(size) if c[x]]
        else:
            raise ValueError(arm)
        svec = [0] * size
        for x in support:
            svec[x] = src.rand()
        view["batch/zk-sum"] = combo([(c[x], svec[x]) for x in range(size)])
        zvec = [combo([(1, mvec[x]), (pub.rho, svec[x])]) for x in range(size)]
    rounds, (cfin, zfin) = batch_rounds(c, zvec, pub.r_batch, p, combo)
    for j, (a, b, q) in enumerate(rounds):
        view[f"batch/round/{j}/A"] = a
        view[f"batch/round/{j}/B"] = b
        view[f"batch/round/{j}/Q"] = q
    if arm == "SPLIT":
        et = eq_table(pub.r_batch, p)
        view["batch/M"] = combo(list(zip(et, mvec)))
        view["batch/S"] = combo(list(zip(et, svec)))
    else:
        view["batch/y"] = zfin
    return view


def verify_numeric(pub: Public, view: dict, arm: str, mvals: dict, svals: list | None, c_pub: list | None = None) -> bool:
    """The verifier's checks on a numeric view; the opening of y is ideal."""
    p = P_B
    ok = True
    claims_values = []
    for k, st in enumerate(pub.stages):
        n, rho, r = st["n"], st["rho"], st["r"]
        degs = stage_degrees(n)
        claim = (view[f"stage/{k}/H"] + rho * view[f"stage/{k}/zk-sum"]) % p
        for j in range(n):
            co = [view[f"stage/{k}/round/{j}/c{e}"] for e in range(degs[j] + 1)]
            ok &= (2 * co[0] + sum(co[1:])) % p == claim
            claim = sum(cf * pow(r[j], e, p) for e, cf in enumerate(co)) % p
        g_r = (claim - mle(st["P"], r, p) * view[f"stage/{k}/final"]) * inv(rho, p) % p
        claims_values.append(g_r)
    for t, rw in enumerate(pub.rows):
        if rw["claimed"]:
            claims_values.append(view[f"row/{t}/u"])
    if arm == "OLD":
        return ok
    v = 0
    weight = 1
    for val in claims_values:
        v = (v + weight * val) % p
        weight = weight * pub.alpha % p
    claim = v if arm == "NOS" else (v + pub.rho * view["batch/zk-sum"]) % p
    for j in range(pub.m):
        a, b, q = (view[f"batch/round/{j}/{x}"] for x in "ABQ")
        ok &= (2 * a + b + q) % p == claim
        rj = pub.r_batch[j]
        claim = (a + b * rj + q * rj * rj) % p
    # The verifier's own C̃(r*), from the claims' public functionals.
    if c_pub is None:
        _, _, claims = run_old(pub, {"stages": [[0] * (1 << s["n"]) for s in pub.stages], "rows": [[0] * (1 << r["c"]) for r in pub.rows]}, FormSource())
        c_pub = batch_c(pub, claims)
    c = c_pub
    c_r = mle(c, pub.r_batch, p)
    size = 1 << pub.m
    mvec = [mvals.get(x, 0) for x in range(size)]
    true_m = mle(mvec, pub.r_batch, p)
    true_s = mle(svals, pub.r_batch, p) if svals is not None else 0
    true_y = (true_m + (0 if arm == "NOS" else pub.rho * true_s)) % p
    if arm == "SPLIT":
        y = (view["batch/M"] + pub.rho * view["batch/S"]) % p
        ok &= view["batch/M"] == true_m and view["batch/S"] == true_s
    else:
        y = view["batch/y"]
    ok &= claim == c_r * y % p
    ok &= y == true_y
    return ok


CONFIGS = {
    "REAL": {"stages": [2, 3, 4], "rows": [(1, True), (2, True), (2, True), (3, True), (4, True), (5, True)]},
    "TIGHT": {"stages": [], "rows": [(1, True)] * 6},
    "UNCLAIMED": {"stages": [], "rows": [(1, True)] * 4 + [(1, False)] * 4, "explicit": explicit_unclaimed},
}
ARMS = {
    "REAL": ["OLD", "CAND", "L1", "NOS", "SPLIT"],
    "TIGHT": ["OLD", "CAND", "L1", "NOS", "SPLIT"],
    "UNCLAIMED": ["OLD", "CAND", "L1", "NOS"],
}


def matrix(views: list[dict], labels: list[str], nvars: int):
    rows, consts = [], []
    for lab in labels:
        row = []
        ks = []
        for view in views:
            x = view[lab]
            if isinstance(x, Form):
                vec = [0] * nvars
                for var, cf in x.v.items():
                    vec[var] = cf
                row.append(vec)
                ks.append(x.k)
            else:
                row.append([0] * nvars)
                ks.append(x % P_B)
        rows.append(row)
        consts.append(ks)
    return rows, consts


def distinguish(a_rows: list[list[int]], d: list[int]) -> tuple[bool, dict | None, int]:
    """Is d outside the column span of A? Then return λ with λ^T A = 0 and λ^T d ≠ 0."""
    p = P_B
    nrows = len(a_rows)
    pivots = []  # (col, row vector, d value, λ)
    for i in range(nrows):
        row = list(a_rows[i])
        dv = d[i]
        lam = {i: 1}
        for col, prow, pd, plam in pivots:
            f = row[col]
            if f:
                row = [(x - f * y) % p for x, y in zip(row, prow)]
                dv = (dv - f * pd) % p
                for key, val in plam.items():
                    lam[key] = (lam.get(key, 0) - f * val) % p
        lead = next((j for j, x in enumerate(row) if x), None)
        if lead is None:
            if dv:
                return True, {key: val for key, val in lam.items() if val}, len(pivots)
            continue
        s = inv(row[lead], p)
        pivots.append((lead, [x * s % p for x in row], dv * s % p, {key: val * s % p for key, val in lam.items()}))
    return False, None, len(pivots)


def toy_draw(task):
    """One (configuration, draw) of part (b): every arm's exact test."""
    config_name, draw, seed = task
    config = CONFIGS[config_name]
    rng = SplitMix64(seed)
    pub = Public(config, rng)
    w1 = witness(pub, rng)
    w2 = witness(pub, rng, base=w1)
    results = {}
    lambdas = {}
    for arm in ARMS[config_name]:
        views = []
        nvars = 0
        for wit in (w1, w2):
            src = FormSource()
            old, mvals, claims = run_old(pub, wit, src)
            view = dict(old)
            if arm != "OLD":
                view.update(run_batch(pub, arm, mvals, claims, src.fork()))
            views.append(view)
            nvars = max(nvars, src.count + (1 << pub.m) + 1)
        labels = sorted(views[0])
        assert labels == sorted(views[1])
        rows, consts = matrix(views, labels, nvars)
        same_a = all(r[0] == r[1] for r in rows)
        a_rows = [r[0] for r in rows]
        dvec = [(k[0] - k[1]) % P_B for k in consts]
        dist, lam, rk = distinguish(a_rows, dvec)
        results[arm] = {"distinguishable": dist, "same_A": same_a, "rank": rk, "view": len(labels)}
        if lam is not None:
            lambdas[arm] = {labels[i]: val for i, val in lam.items()}
    out = {"config": config_name, "draw": draw, "m": pub.m, "arms": results}
    if draw == 0:
        out["lambdas"] = lambdas
        out["seed"] = seed
    return out


def empirical(config_name: str, seed: int, lambdas: dict, samples: int) -> dict:
    """Numeric views at draw 0, the λ-test's advantage, and the affine model's check."""
    config = CONFIGS[config_name]
    rng = SplitMix64(seed)
    pub = Public(config, rng)
    w1 = witness(pub, rng)
    w2 = witness(pub, rng, base=w1)
    out = {}
    stat_rng = SplitMix64(seed ^ 0x5A5A5A5A)
    for arm in ARMS[config_name]:
        lam = lambdas.get(arm) or lambdas.get("NOS")
        source = "own" if arm in lambdas else ("NOS" if "NOS" in lambdas else "random")
        models = []
        for wit in (w1, w2):
            src = FormSource()
            old, mvals, claims = run_old(pub, wit, src)
            view = dict(old)
            if arm != "OLD":
                view.update(run_batch(pub, arm, mvals, claims, src.fork()))
            models.append(view)
        _, _, pub_claims = run_old(pub, w1, FormSource())
        c_pub = batch_c(pub, pub_claims)
        if lam is None:
            lam = {lab: stat_rng.below(P_B) for lab in models[0]}
        lam = {k: v for k, v in lam.items() if k in models[0]}
        target = sum(v * (models[0][k].k if isinstance(models[0][k], Form) else models[0][k]) for k, v in lam.items()) % P_B
        hits = [0, 0]
        accepted = 0
        model_ok = True
        num_rng = SplitMix64(worker_seed(seed ^ 0x77777777, ARMS[config_name].index(arm) + 1))
        for wi, wit in enumerate((w1, w2)):
            for sample in range(samples):
                src = NumSource(num_rng)
                old, mvals, claims = run_old(pub, wit, src)
                view = dict(old)
                svals = None
                if arm != "OLD":
                    bsrc = src.fork()
                    view.update(run_batch(pub, arm, mvals, claims, bsrc))
                    if arm != "NOS":
                        size = 1 << pub.m
                        c = batch_c(pub, claims)
                        support = sorted(mvals) if arm in ("CAND", "SPLIT") else [x for x in range(size) if c[x]]
                        svals = [0] * size
                        for idx, x in enumerate(support):
                            svals[x] = bsrc.drawn[len(src.drawn) + idx]
                    drawn = bsrc.drawn
                else:
                    drawn = src.drawn
                accepted += verify_numeric(pub, view, arm, mvals, svals, c_pub)
                if sample < 10:
                    model = models[wi]
                    for lab, val in view.items():
                        mv = model[lab]
                        pred = (mv.k + sum(cf * drawn[var] for var, cf in mv.v.items())) % P_B if isinstance(mv, Form) else mv % P_B
                        model_ok &= pred == val % P_B
                stat = sum(v * view[k] for k, v in lam.items()) % P_B
                hits[wi] += stat == target
        out[arm] = {
            "lambda": source,
            "hit_w1": hits[0] / samples,
            "hit_w2": hits[1] / samples,
            "advantage": (hits[0] - hits[1]) / samples,
            "accepted": accepted / (2 * samples),
            "model_matches": model_ok,
        }
    return out


def part_b(workers: int) -> dict:
    draws = 100
    tasks = []
    for ci, name in enumerate(CONFIGS):
        for draw in range(draws):
            tasks.append((name, draw, worker_seed(SEED_B, ci * 1000 + draw)))
    if workers > 1:
        with multiprocessing.Pool(workers) as pool:
            results = pool.map(toy_draw, tasks, chunksize=1)
    else:
        results = [toy_draw(t) for t in tasks]
    summary = {}
    emp = {}
    for name in CONFIGS:
        rs = [r for r in results if r["config"] == name]
        summary[name] = {
            "m": rs[0]["m"],
            "draws": len(rs),
            "arms": {
                arm: {
                    "distinguishable_share": sum(r["arms"][arm]["distinguishable"] for r in rs) / len(rs),
                    "same_A": all(r["arms"][arm]["same_A"] for r in rs),
                    "view_entries": rs[0]["arms"][arm]["view"],
                    "rank_mode": max(set(x := [r["arms"][arm]["rank"] for r in rs]), key=x.count),
                }
                for arm in ARMS[name]
            },
        }
        first = next(r for r in rs if r["draw"] == 0)
        emp[name] = empirical(name, first["seed"], first["lambdas"], 1000)
    crit = {}
    for name in CONFIGS:
        arms = summary[name]["arms"]
        crit[name] = {"OLD_never": arms["OLD"]["distinguishable_share"] == 0.0, "CAND_never": arms["CAND"]["distinguishable_share"] == 0.0}
    crit["power"] = {
        "TIGHT/NOS": summary["TIGHT"]["arms"]["NOS"]["distinguishable_share"],
        "TIGHT/SPLIT": summary["TIGHT"]["arms"]["SPLIT"]["distinguishable_share"],
        "UNCLAIMED/L1": summary["UNCLAIMED"]["arms"]["L1"]["distinguishable_share"],
        "UNCLAIMED/NOS": summary["UNCLAIMED"]["arms"]["NOS"]["distinguishable_share"],
    }
    return {"summary": summary, "empirical": emp, "criteria": crit}


# ---------------------------------------------------------------------------------------------
# (c) Soundness over F65537.

P_C = Q65537


def soundness_setup(name: str, rng: SplitMix64):
    """The claims' functionals and a committed M and S for layout `name`."""
    p = P_C
    pub = Public(CONFIGS[name], rng)
    src = NumSource(rng)
    w = witness(pub, rng)
    _, mvals, claims = run_old(pub, w, src)
    size = 1 << pub.m
    mvec = [mvals.get(x, 0) for x in range(size)]
    svec = [0] * size
    for x in sorted(mvals):
        svec[x] = rng.below(p)
    vecs = []
    for vec, _ in claims:
        full = [0] * size
        for pos, cf in vec.items():
            full[pos] = cf
        vecs.append(full)
    vals = [sum(a * b for a, b in zip(v, mvec)) % p for v in vecs]
    sig = [sum(a * b for a, b in zip(v, svec)) % p for v in vecs]
    return pub.m, vecs, mvec, svec, vals, sig


def poly_from_roots(roots: list[int], kappa: int, p: int) -> list[int]:
    coeffs = [kappa % p]
    for a in roots:
        nxt = [0] * (len(coeffs) + 1)
        for i, c in enumerate(coeffs):
            nxt[i] = (nxt[i] - a * c) % p
            nxt[i + 1] = (nxt[i + 1] + c) % p
        coeffs = nxt
    return coeffs


def soundness_chunk(task):
    name, mode, trials, seed = task
    p = P_C
    rng = SplitMix64(seed)
    m, vecs, mvec, svec, vals, sig = soundness_setup(name, SplitMix64(SEED_C))
    n_claims = len(vecs)
    size = 1 << m
    # Each position belongs to at most one claim (the layout's blocks are disjoint).
    owner = []
    for x in range(size):
        own = [(i, vecs[i][x]) for i in range(n_claims) if vecs[i][x]]
        assert len(own) <= 1
        owner.append(own[0] if own else (0, 0))
    counts = {"accepted": 0, "alpha": 0, "rho": 0, "round": [0] * m, "trials": trials}
    for _ in range(trials):
        roots = []
        while len(roots) < n_claims - 1:
            a = rng.below(p)
            if a not in roots:
                roots.append(a)
        errs = poly_from_roots(roots, rng.nonzero(p), p) if mode in ("cheat", "nc1") else [0] * n_claims
        alpha = rng.nonzero(p)
        if mode == "nc2":
            # alpha known before the claims: false claims whose combination is exact.
            errs = [0] * n_claims
            errs[1] = 1
            errs[0] = -alpha % p
        powers = [pow(alpha, i, p) for i in range(n_claims)]
        e_alpha = sum(e * w for e, w in zip(errs, powers)) % p
        v_true = sum(v * w for v, w in zip(vals, powers)) % p
        v_claim = (v_true + e_alpha) % p
        c = [powers[i] * cf % p for i, cf in owner]
        cs = sum(s * w for s, w in zip(sig, powers)) % p
        rho0 = rng.nonzero(p)
        rho = rng.below(p)
        if mode == "cheat":
            gamma_s = (cs - e_alpha * inv(rho0, p)) % p
        elif mode == "nc1":
            gamma_s = (cs - e_alpha * inv(rho, p)) % p if rho else cs
        else:
            gamma_s = cs
        claim = (v_claim + rho * gamma_s) % p
        z = [(a + rho * b) % p for a, b in zip(mvec, svec)]
        r = [rng.below(p) for _ in range(m)]
        rounds, (cfin, zfin) = batch_rounds_int(c, z, r, p)
        ok = True
        event = None
        if e_alpha == 0 and any(errs):
            event = "alpha"
        true_claim = (v_true + rho * cs) % p
        if event is None and claim == true_claim and any(errs):
            event = "rho"
        for j in range(m):
            a, b, q = rounds[j]
            honest = (2 * a + b + q) % p
            if honest != claim:
                ra, rb = rng.below(p), rng.below(p)
                while ra == rb or (ra * rb + (1 - ra) * (1 - rb)) % p == 0:
                    ra, rb = rng.below(p), rng.below(p)
                delta = (claim - honest) * inv(ra * rb + (1 - ra) * (1 - rb), p) % p
                a = (a + delta * ra * rb) % p
                b = (b - delta * (ra + rb)) % p
                q = (q + delta) % p
                cheated = True
            else:
                cheated = False
            ok &= (2 * a + b + q) % p == claim
            rj = r[j]
            claim = (a + b * rj + q * rj * rj) % p
            if cheated and claim == (rounds[j][0] + rounds[j][1] * rj + rounds[j][2] * rj * rj) % p and event is None:
                event = ("round", j)
        c_r = mle(c, r, p)
        ok &= claim == c_r * zfin % p
        ok &= zfin == mle(z, r, p)
        if ok:
            counts["accepted"] += 1
            if event == "alpha":
                counts["alpha"] += 1
            elif event == "rho":
                counts["rho"] += 1
            elif isinstance(event, tuple):
                counts["round"][event[1]] += 1
    return {"name": name, "mode": mode, "n_claims": n_claims, "m": m, **counts}


def merge(chunks: list[dict]) -> dict:
    out = dict(chunks[0])
    for key in ("accepted", "alpha", "rho", "trials"):
        out[key] = sum(ch[key] for ch in chunks)
    out["round"] = [sum(ch["round"][j] for ch in chunks) for j in range(len(chunks[0]["round"]))]
    return out


def part_c(workers: int) -> dict:
    q = P_C
    plan = [("REAL", "cheat", 10**6), ("TIGHT", "cheat", 2 * 10**6), ("REAL", "honest", 10**4), ("REAL", "nc1", 10**4), ("REAL", "nc2", 10**4)]
    tasks = []
    index = 0
    for name, mode, trials in plan:
        parts = max(1, workers) if trials >= 10**5 else 1
        per = trials // parts
        for w in range(parts):
            n = per if w < parts - 1 else trials - per * (parts - 1)
            tasks.append((name, mode, n, worker_seed(SEED_C, index + 1)))
            index += 1
    if workers > 1:
        with multiprocessing.Pool(workers) as pool:
            chunks = pool.map(soundness_chunk, tasks, chunksize=1)
    else:
        chunks = [soundness_chunk(t) for t in tasks]
    out = {}
    for name, mode, _trials in plan:
        res = merge([ch for ch in chunks if ch["name"] == name and ch["mode"] == mode])
        key = f"{name}/{mode}"
        if mode == "cheat":
            n, m, t = res["n_claims"], res["m"], res["trials"]
            pred = 1 - (1 - (n - 1) / q) * (1 - 1 / q) * (1 - 2 / q) ** m
            bound = (n + 2 * m + 1) / q
            k = res["accepted"]
            sigma = (pred * t * (1 - pred)) ** 0.5
            res.update({
                "predicted_rate": pred,
                "predicted_count": pred * t,
                "band": 4 * sigma,
                "bound_rate": bound,
                "bound_count_4sigma": bound * t + 4 * (bound * t) ** 0.5,
                "in_band": abs(k - pred * t) <= 4 * sigma,
                "under_bound": k <= bound * t + 4 * (bound * t) ** 0.5,
                "predicted_alpha": (n - 1) / q * t,
                "predicted_rho": (1 - (n - 1) / q) / q * t,
                "predicted_round_each": 2 / q * t,
            })
        else:
            res["rate"] = res["accepted"] / res["trials"]
        out[key] = res
    return out


# ---------------------------------------------------------------------------------------------


def environment() -> dict:
    return {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "machine": platform.machine(),
        "processor": platform.processor(),
        "utc": datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
    }


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("part", choices=["a", "b", "c", "all"])
    ap.add_argument("results", type=Path)
    ap.add_argument("--workers", type=int, default=1)
    args = ap.parse_args(argv)
    args.results.mkdir(parents=True, exist_ok=True)
    parts = ["a", "b", "c"] if args.part == "all" else [args.part]
    runners = {"a": part_a, "b": part_b, "c": part_c}
    for part in parts:
        start = time.time()
        env = environment()
        result = runners[part](args.workers)
        env["finished_utc"] = datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
        env["seconds"] = round(time.time() - start, 1)
        env["workers"] = args.workers
        path = args.results / f"run-{part}.json"
        path.write_text(json.dumps({"environment": env, "result": result}, indent=1) + "\n")
        print(f"part {part}: {env['seconds']} s -> {path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
