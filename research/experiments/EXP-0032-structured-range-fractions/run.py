#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0 OR MIT
"""EXP-0032: exact structured bottom levels for Range table fraction trees (E0).

Reference: dense leaves, levels by RFC-0008's rule, dense level sumcheck (RFC-0010 step 2,
pair bit lowest, lowest variable bound first). Candidate: dense levels t < c, structured
levels t >= c (IDEA-0043). Every value is exact over Goldilocks Fp2 = Fp[X]/(X^2 - 7).
"""

import argparse
import gzip
import json
import re
import sys
import time
from pathlib import Path

P = 2**64 - 2**32 + 1
W = 7
ROOT = Path(__file__).resolve().parents[3]
COUNT = {"full": 0, "scalar": 0}


# ---------- Fp2, every multiplication counted ----------
def add(x, y):
    return ((x[0] + y[0]) % P, (x[1] + y[1]) % P)


def sub(x, y):
    return ((x[0] - y[0]) % P, (x[1] - y[1]) % P)


def mul(x, y):
    COUNT["full"] += 1
    a0, a1 = x
    b0, b1 = y
    return ((a0 * b0 + W * a1 * b1) % P, (a0 * b1 + a1 * b0) % P)


def smul(s, x):  # base-field scalar times Fp2
    COUNT["scalar"] += 1
    return (s * x[0] % P, s * x[1] % P)


def small(k, x):  # small public integer times x by additions only (k in 0..3)
    out = ZERO
    for _ in range(k):
        out = add(out, x)
    return out


def inv(x):
    a0, a1 = x
    n = (a0 * a0 - W * a1 * a1) % P
    if n == 0:
        raise ZeroDivisionError("Fp2 inverse of zero")
    ni = pow(n, P - 2, P)
    return (a0 * ni % P, (-a1) * ni % P)


ZERO, ONE = (0, 0), (1, 0)


def fe(n):
    return (n % P, 0)


class SplitMix64:
    def __init__(self, seed):
        self.s = seed

    def next(self):
        self.s = (self.s + 0x9E3779B97F4A7C15) % 2**64
        z = self.s
        z = ((z ^ (z >> 30)) * 0xBF58476D1CE4E5B9) % 2**64
        z = ((z ^ (z >> 27)) * 0x94D049BB133111EB) % 2**64
        return z ^ (z >> 31)

    def below(self, n):
        return self.next() % n

    def fp2(self):
        return (self.below(P), self.below(P))


def check_field_vectors():
    d = json.loads((ROOT / "spec/vectors/field/goldilocks_ext2.json").read_text())
    assert int(d["modulus"], 16) == P and d["non_residue"] == W
    for case in d["cases"]:
        a0, a1, b0, b1, m0, m1, i0, i1, _norm = case
        a, b = (int(a0, 16), int(a1, 16)), (int(b0, 16), int(b1, 16))
        assert mul(a, b) == (int(m0, 16), int(m1, 16)), case
        if i0 is not None:
            assert inv(a) == (int(i0, 16), int(i1, 16)), case
    return len(d["cases"])


def eq1(r, bit):  # eq(r, bit) for a public or secret bit, arithmetic select
    return add(smul(bit, r), smul(1 - bit, sub(ONE, r)))


# ---------- tree (RFC-0008) ----------
def table_leaves(h, alpha, keys):
    counts = [0] * (1 << h)
    for a in keys:
        counts[a] += 1
    return [fe(-m) for m in counts], [sub(alpha, fe(j)) for j in range(1 << h)]


def level_above(n, d):
    up_n, up_d = [], []
    for x in range(len(n) // 2):
        up_n.append(add(mul(n[2 * x], d[2 * x + 1]), mul(n[2 * x + 1], d[2 * x])))
        up_d.append(mul(d[2 * x], d[2 * x + 1]))
    return up_n, up_d


def all_levels(n, d):
    levels = [(n, d)]
    while len(levels[-1][0]) > 1:
        levels.append(level_above(*levels[-1]))
    return levels[::-1]  # levels[t] has 2^t entries


# ---------- reference dense level sumcheck ----------
def eq_table(r):
    table = [ONE]
    for ri in r:  # variable i lands on index bit i: the lowest variable is bit 0
        table = [mul(e, sub(ONE, ri)) for e in table] + [mul(e, ri) for e in table]
    return table


def dense_level(N, D, r, lam, chal):
    """Level t: claims on 2^t entries; N, D the 2^{t+1} entries of level t + 1."""
    t = len(r)
    tables = [eq_table(r)] + [[T[2 * y + b] for y in range(1 << t)] for T in (N, D) for b in (0, 1)]
    msgs = []
    for j in range(t):
        s = []
        for X in range(4):
            acc = ZERO
            for u in range(len(tables[0]) // 2):
                e, n0, n1, d0, d1 = (add(T[2 * u], small(X, sub(T[2 * u + 1], T[2 * u]))) for T in tables)
                g = add(add(mul(n0, d1), mul(n1, d0)), mul(lam, mul(d0, d1)))
                acc = add(acc, mul(e, g))
            s.append(acc)
        msgs.append(s)
        rho = chal[j]
        tables = [[add(T[2 * u], mul(rho, sub(T[2 * u + 1], T[2 * u]))) for u in range(len(T) // 2)] for T in tables]
    return msgs, tuple(T[0] for T in tables[1:])


def dense_level_cost(t):
    """Closed form of dense_level's counted multiplications (full, scalar)."""
    full = sum(2 * (1 << i) for i in range(t))  # eq_table
    for j in range(t):
        pairs = 1 << (t - j - 1)
        full += 4 * pairs * 5 + 5 * pairs  # 4 X values x 5 mults; fold 5 tables
    scalar = 0
    return full, scalar


# ---------- candidate: structured bottom level ----------
def poly_linear_product(consts, gamma):
    """prod_s (consts[s] + gamma*y): coefficients in y. gamma is a public integer."""
    c = [ONE]
    for cs in consts:
        nxt = [ZERO] * (len(c) + 1)
        for k, ck in enumerate(c):
            nxt[k] = add(nxt[k], mul(cs, ck))
            nxt[k + 1] = add(nxt[k + 1], smul(gamma, ck))
        c = nxt
    return c


def shift_one(c):  # f(y) -> f(y + 1), classical division-free Taylor shift: additions only
    a = list(c)
    n = len(a) - 1
    for i in range(n):
        for j in range(n - 1, i - 1, -1):
            a[j] = add(a[j], a[j + 1])
    return a


def scale_pow2(c):  # f(y) -> f(2y)
    return [smul(pow(2, k, P), ck) for k, ck in enumerate(c)]


def even_odd(c):  # (f(2u), f(2u + 1))
    return scale_pow2(c), scale_pow2(shift_one(c))


def pmul(a, b):
    out = [ZERO] * (len(a) + len(b) - 1)
    for i, x in enumerate(a):
        for k, y in enumerate(b):
            out[i + k] = add(out[i + k], mul(x, y))
    return out


BINOM = {}


def binom_row(k):
    if k not in BINOM:
        row = [1]
        for i in range(k):
            row = [1] + [(row[a - 1] + row[a]) % P for a in range(1, len(row))] + [1]
        BINOM[k] = row
    return BINOM[k]


def suffix_moments(rs, K):
    """For every j: M_j[k] = sum_{u over the bits rs[j:]} eq(rs[j:], u) * u^k, k <= K."""
    out = [None] * (len(rs) + 1)
    S = [ONE] + [ZERO] * K
    out[len(rs)] = S
    for j in range(len(rs) - 1, -1, -1):
        r = rs[j]
        T = [smul(pow(2, a, P), S[a]) for a in range(K + 1)]  # (2u)^a
        nxt = []
        for k in range(K + 1):
            row = binom_row(k)
            acc = ZERO
            for a in range(k + 1):
                acc = add(acc, smul(row[a], T[a]))  # sum_a C(k,a) (2u)^a, the x = 1 branch
            nxt.append(add(mul(sub(ONE, r), T[k]), mul(r, acc)))
        S = nxt
        out[j] = S
    return out


def horner_base(c, x):  # evaluate at a base-field integer
    acc = ZERO
    for ck in reversed(c):
        acc = add(smul(x, acc), ck)
    return acc


def structured_level(h, t, alpha, keys, r, lam, chal, fault=None):
    B = 1 << (h - t - 1)
    off = (lambda b: 1 - b) if fault == "F1" else (lambda b: b)
    lo = 1 if fault == "F4" else 0
    H = []
    for b in (0, 1):
        consts = [sub(alpha, fe(B * off(b) + s)) for s in range(lo, lo + B)]
        H.append(poly_linear_product(consts, -2 * B % P))
    src = sorted(set(keys)) if fault == "F2" else keys
    entries = []  # [pair bit, node position, weight]; one per lookup slot
    for a in src:
        e = a >> (h - t - 1)
        base = e * B
        if fault == "F6":
            prod = ONE
            for s in range(B):
                prod = mul(prod, sub(alpha, fe(base + s)))
            w = mul(prod, inv(sub(alpha, fe(a))))
        else:
            w = ONE
            for s in range(lo, lo + B):
                d = sub(alpha, fe(base + s))
                hit = int(base + s == a)  # data, not control flow: arithmetic select
                w = mul(w, add(d, smul(hit, sub(ONE, d))))
        entries.append([e & 1, e >> 1, sub(ZERO, w)])
    moments = suffix_moments(r, 2 * B) if t else [[ONE]]
    msgs = []
    prefix = ONE
    for j in range(t):
        rest = len(r) - j - 1
        # per-entry eq over the remaining suffix bits
        for en in entries:
            pos = en[1] >> 1
            eqs = ONE
            for i in range(rest):
                eqs = mul(eqs, eq1(r[j + 1 + i], (pos >> i) & 1))
            en.append(eqs)
        EO = [even_odd(Hb) for Hb in H]
        M = moments[j + 1]
        s = []
        for X in range(4):
            HX = [[add(E[k], small(X, sub(O[k], E[k]))) for k in range(len(E))] for E, O in EO]
            prod = pmul(HX[0], HX[1])
            dd = ZERO
            for k, ck in enumerate(prod):
                dd = add(dd, mul(ck, M[k]))
            s.append(dd)
        cross = [ZERO] * 4
        for en in entries:
            b, u, w, eqs = en
            pos, odd = u >> 1, u & 1
            vals = []
            for E, O in EO:
                ev, ov = horner_base(E, pos), horner_base(O, pos)
                vals.append((ev, ov))
            weq = mul(w, eqs)
            for X in range(4):
                fx = small(X, ONE) if odd else sub(ONE, small(X, ONE))
                hv = [add(ev, small(X, sub(ov, ev))) for ev, ov in vals]
                other = add(smul(1 - b, hv[1]), smul(b, hv[0]))  # N_b pairs with D_{1-b}
                if fault == "F3" and b == 1:
                    continue
                cross[X] = add(cross[X], mul(mul(weq, other), fx))
            en.pop()
        msgs.append([mul(mul(prefix, eq1_small(r[j], X)), add(cross[X], mul(lam, s[X]))) for X in range(4)])
        rho = chal[j]
        prefix = mul(prefix, add(mul(sub(ONE, r[j]), sub(ONE, rho)), mul(r[j], rho)))
        lo_w, hi_w = (rho, sub(ONE, rho)) if fault == "F5" else (sub(ONE, rho), rho)
        H = [[add(mul(lo_w, E[k]), mul(hi_w, O[k])) for k in range(len(E))] for E, O in EO]
        for en in entries:
            odd = en[1] & 1
            en[2] = mul(en[2], add(smul(1 - odd, sub(ONE, rho)), smul(odd, rho)))
            en[1] >>= 1
    n0 = ZERO
    n1 = ZERO
    for b, _u, w in entries:
        n0 = add(n0, smul(1 - b, w))
        n1 = add(n1, smul(b, w))
    return msgs, (n0, n1, H[0][0], H[1][0])


def eq1_small(r, X):  # eq(r, X) = X*r + (1 - X)(1 - r) at a small public X, by additions
    if X <= 1:
        return r if X else sub(ONE, r)
    return sub(small(X, r), small(X - 1, sub(ONE, r)))


def structured_level_cost(h, t, U):
    """Closed form of structured_level's counted multiplications (honest, no fault)."""
    B = 1 << (h - t - 1)
    full = scalar = 0
    for _b in (0, 1):  # poly_linear_product with B factors
        for size in range(1, B + 1):
            full += size
            scalar += size
    full += U * B  # entry weights
    scalar += U * B
    K = 2 * B
    if t:  # suffix moments over t bits
        for _ in range(t):
            scalar += (K + 1) + (K + 1) * (K + 2) // 2
            full += 2 * (K + 1)
    for j in range(t):
        rest = t - j - 1
        full += U * rest  # eq products
        scalar += U * rest * 2  # eq1 selects
        scalar += 2 * 2 * (B + 1)  # even_odd scalings for two polynomials
        full += 4 * ((B + 1) ** 2 + (2 * B + 1))  # products and moment dot
        scalar += U * 4 * (B + 1)  # four Horner evaluations
        full += U * (1 + 4 * 2)  # weq, then two mults per X
        scalar += U * 4 * 2  # the pair-bit select per X
        full += 4 * 3  # message: prefix * eq, lam * dd, product
        full += 3  # prefix update
        full += 2 * 2 * (B + 1)  # polynomial fold
        full += U  # weight fold
        scalar += U * 2
    scalar += 2 * U  # final sums
    return full, scalar


# ---------- independent transcript checker ----------
def check_level(msgs, evals, claim, r, lam, chal):
    """Sees only messages, the incoming claim and the challenges (RFC-0010 step 2)."""
    cur = claim
    for j, s in enumerate(msgs):
        if add(s[0], s[1]) != cur:
            return False
        cur = lagrange3(s, chal[j])
    n0, n1, d0, d1 = evals
    e = ONE
    for ri, pi in zip(r, chal):
        e = mul(e, add(mul(sub(ONE, ri), sub(ONE, pi)), mul(ri, pi)))
    g = add(add(mul(n0, d1), mul(n1, d0)), mul(lam, mul(d0, d1)))
    return cur == mul(e, g)


def lagrange3(s, x):
    # interpolate the cubic through (0..3, s) at x; denominators are public constants
    out = ZERO
    for i in range(4):
        num, den = ONE, 1
        for k in range(4):
            if k != i:
                num = mul(num, sub(x, fe(k)))
                den = den * (i - k)
        out = add(out, mul(s[i], smul(pow(den % P, P - 2, P), num)))
    return out


def mle_at(vals, r):
    table = list(vals)
    for ri in r:  # lowest variable first
        table = [add(table[2 * u], mul(ri, sub(table[2 * u + 1], table[2 * u]))) for u in range(len(table) // 2)]
    return table[0]


# ---------- the grid ----------
PATTERNS = ("uniform", "zeros", "top", "single")


def make_keys(rng, h, ha, pattern):
    T, U = 1 << h, 1 << ha
    real = rng.below(U + 1)  # lookups before padding
    if pattern == "uniform":
        keys = [rng.below(T) for _ in range(real)]
    elif pattern == "zeros":
        keys = [0] * real
    elif pattern == "top":
        keys = [T - 1] * real
    else:
        k = rng.below(T)
        keys = [k] * real
    return keys + [0] * (U - real)  # RFC-0008: padding slots repeat row 0


def run_case(rng, h, c, ha, pattern, alpha_kind, fault=None, record=None):
    keys = make_keys(rng, h, ha, pattern)
    alpha = rng.fp2() if alpha_kind == "random" else fe(rng.below(1 << h))
    n, d = table_leaves(h, alpha, keys)
    levels = all_levels(n, d)
    ok = True
    for t in range(h):
        r = [rng.fp2() for _ in range(t)]
        chal = [rng.fp2() for _ in range(t)]
        lam = rng.fp2()
        COUNT["full"] = COUNT["scalar"] = 0
        ref = dense_level(*levels[t + 1], r, lam, chal)
        if fault is None and record is not None and (COUNT["full"], COUNT["scalar"]) != dense_level_cost(t):
            COUNTS[1] += 1
        if t < c:
            continue  # both arms run the same dense code on the cached levels
        COUNT["full"] = COUNT["scalar"] = 0
        try:
            got = structured_level(h, t, alpha, keys, r, lam, chal, fault)
        except ZeroDivisionError:
            got = None
        if fault is None and record is not None:
            COUNTS[0] += 1
            if (COUNT["full"], COUNT["scalar"]) != structured_level_cost(h, t, len(keys)):
                COUNTS[1] += 1
        if got != ref:
            ok = False
            if fault is None and record is not None:
                record.append({"h": h, "c": c, "ha": ha, "pattern": pattern, "alpha": alpha_kind, "t": t})
            if fault is not None:
                return False
            continue
        if fault is None and record is not None:
            nt, dt = levels[t]
            claim = add(mle_at(nt, r), mul(lam, mle_at(dt, r)))
            if not check_level(*got, claim, r, lam, chal):
                record.append({"checker": "rejected honest", "h": h, "t": t})
            if got[0]:
                bad = [list(s) for s in got[0]]
                bad[0][0] = add(bad[0][0], ONE)
                if check_level(bad, got[1], claim, r, lam, chal):
                    record.append({"checker": "accepted tampered", "h": h, "t": t})
            CHECKED[0] += 1
    if fault is None and record is not None:
        # root identity against the lookup tree: n_a d_t + n_t d_a = 0
        na, da = ZERO, ONE
        for a in keys:
            dl = sub(alpha, fe(a))
            na, da = add(mul(na, dl), da), mul(da, dl)
        nt, dt = levels[0][0][0], levels[0][1][0]
        if add(mul(na, dt), mul(nt, da)) != ZERO:
            record.append({"root identity": "failed", "h": h})
    return ok


CHECKED = [0]
COUNTS = [0, 0]  # structured calls compared, count mismatches (dense and structured)


def grid(max_h):
    for h in range(1, max_h + 1):
        for c in range(0, h + 1):
            for ha in sorted({0, 1, 3, h, h + 2}):
                for pattern in PATTERNS:
                    for alpha_kind in ("random", "in-table"):
                        yield h, c, ha, pattern, alpha_kind


def count_check(max_h):
    rng = SplitMix64(0x0691C0)
    rows = []
    for h in range(1, max_h + 1):
        for ha in sorted({0, 1, 3, h}):
            keys = make_keys(rng, h, ha, "uniform")
            alpha = rng.fp2()
            for t in range(h):
                r = [rng.fp2() for _ in range(t)]
                chal = [rng.fp2() for _ in range(t)]
                lam = rng.fp2()
                for name, fn, closed in (
                    ("structured", lambda: structured_level(h, t, alpha, keys, r, lam, chal), structured_level_cost(h, t, 1 << ha)),
                ):
                    COUNT["full"] = COUNT["scalar"] = 0
                    fn()
                    got = (COUNT["full"], COUNT["scalar"])
                    rows.append({"h": h, "ha": ha, "t": t, "arm": name, "instrumented": got, "closed": closed})
            n, d = table_leaves(h, alpha, keys)
            levels = all_levels(n, d)
            for t in range(h):
                r = [rng.fp2() for _ in range(t)]
                COUNT["full"] = COUNT["scalar"] = 0
                dense_level(*levels[t + 1], r, rng.fp2(), [rng.fp2() for _ in range(t)])
                rows.append({"h": h, "ha": ha, "t": t, "arm": "dense", "instrumented": (COUNT["full"], COUNT["scalar"]), "closed": dense_level_cost(t)})
    return rows


def real_counts(out):
    data = json.load(gzip.open(ROOT / "research/experiments/EXP-0030-real-runtime-sizing/results/counts.json.gz"))
    rec = [r for r in data["records"] if r["case"] == "real" and r["policy"] == 30 and r["positions"] == 2][0]
    rows = []
    for inst in rec["instances"]:
        m = re.match(r"Range \{ bits: (\d+) \}", inst["kind"])
        if not m:
            continue
        h, ha = int(m.group(1)), inst["height"]
        assert h == inst["table_log2_rows"]
        c = min(h - 4, 20) if h > 4 else 0
        T = 1 << h
        scan = lambda depth: T + 3 * (T - (1 << depth))  # leaf generations + recombination
        cache_read = lambda t: (1 << c) + 3 * ((1 << c) - (1 << (t + 1)))  # level t+1 from the cache
        shared = (scan(c) if c else scan(0))  # root / cache build
        shared += sum(4 * cache_read(t) + sum(dense_level_cost(t)) for t in range(c))
        base = shared + sum(4 * scan(t + 1) + sum(dense_level_cost(t)) for t in range(c, h))
        cand = shared + sum(sum(structured_level_cost(h, t, 1 << ha)) for t in range(c, h))
        rows.append({"table": inst["table"], "bits": h, "lookup_height": ha, "entries": inst["entries"], "c": c,
                     "baseline": base, "candidate": cand, "ratio": cand / base})
    (out / "real-range.json").write_text(json.dumps(rows, indent=1) + "\n")
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--max-h", type=int, default=10)
    ap.add_argument("--only", default="all", choices=["all", "exact", "faults", "counts"])
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    env = {"python": sys.version.split()[0], "utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
    report = {"environment": env, "field_vectors": check_field_vectors(), "max_h": args.max_h}
    if args.only in ("all", "exact"):
        rng = SplitMix64(0x0691)
        mismatches = []
        cases = 0
        for case in grid(args.max_h):
            run_case(rng, *case, record=mismatches)
            cases += 1
        report["exact"] = {"cases": cases, "levels_checked": CHECKED[0], "problems": mismatches,
                           "structured_calls_counted": COUNTS[0], "count_mismatches": COUNTS[1]}
        print("exact:", cases, "cases,", CHECKED[0], "levels checked,", len(mismatches), "problems;",
              COUNTS[0], "structured calls counted,", COUNTS[1], "count mismatches", flush=True)
    if args.only in ("all", "faults"):
        faults = {}
        for fault in ("F1", "F2", "F3", "F4", "F5", "F6", "F7"):
            rng = SplitMix64(0x0691)
            caught = 0
            first = None
            for case in grid(args.max_h):
                h, c, ha, pattern, alpha_kind = case
                if fault == "F7":
                    keys_ok = run_case_f7(rng, *case)
                else:
                    keys_ok = run_case(rng, *case, fault=fault)
                if not keys_ok:
                    caught += 1
                    first = first or list(case)
            faults[fault] = {"cases_caught": caught, "first": first}
            print("fault", fault, "caught in", caught, "cases", flush=True)
        report["faults"] = faults
    if args.only in ("all", "counts"):
        rows = count_check(args.max_h)
        bad = [r for r in rows if tuple(r["instrumented"]) != tuple(r["closed"])]
        report["count_check"] = {"rows": len(rows), "mismatches": bad[:10], "mismatch_count": len(bad)}
        print("count check:", len(rows), "rows,", len(bad), "mismatches", flush=True)
        real = real_counts(out)
        b, c = sum(r["baseline"] for r in real), sum(r["candidate"] for r in real)
        report["real"] = {"tables": len(real), "baseline": b, "candidate": c, "ratio": c / b}
        print("real Range:", len(real), "tables, ratio", c / b, flush=True)
    (out / f"report-{args.only}.json").write_text(json.dumps(report, indent=1) + "\n")


def run_case_f7(rng, h, c, ha, pattern, alpha_kind):
    """F7: the candidate omits the lookup column's padding slots."""
    keys = make_keys(rng, h, ha, pattern)
    alpha = rng.fp2() if alpha_kind == "random" else fe(rng.below(1 << h))
    levels = all_levels(*table_leaves(h, alpha, keys))
    trimmed = list(keys)
    while trimmed and trimmed[-1] == 0:  # the padding is the trailing row-0 slots
        trimmed.pop()
    for t in range(h):
        r = [rng.fp2() for _ in range(t)]
        chal = [rng.fp2() for _ in range(t)]
        lam = rng.fp2()
        ref = dense_level(*levels[t + 1], r, lam, chal)
        if t >= c and structured_level(h, t, alpha, trimmed, r, lam, chal) != ref:
            return False
    return True


if __name__ == "__main__":
    main()
