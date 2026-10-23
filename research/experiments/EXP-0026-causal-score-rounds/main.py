# SPDX-License-Identifier: Apache-2.0 OR MIT
"""EXP-0026: SoftmaxRow's key and forest sums without the score tensor.

A literal RFC-0010 sumcheck prover over the materialized `[layers, heads, queries, keys]`
tensors, against a candidate that answers the key-bit rounds from the queries, prefix sums of
the folded keys and the causal mask's closed form, then switches to row tables. Exact
Goldilocks-extension arithmetic; see README.md for the frozen criteria.
"""

import json
import platform
import sys
import time

P = 2**64 - 2**32 + 1
W = 7  # Fp2 = Fp[X]/(X^2 - 7)
MULS = [0]


def norm(a):
    return (a[0] % P, a[1] % P)


def add(a, b):
    return ((a[0] + b[0]) % P, (a[1] + b[1]) % P)


def sub(a, b):
    return ((a[0] - b[0]) % P, (a[1] - b[1]) % P)


def mul(a, b):
    MULS[0] += 1
    return ((a[0] * b[0] + W * a[1] * b[1]) % P, (a[0] * b[1] + a[1] * b[0]) % P)


ZERO, ONE = (0, 0), (1, 0)


def base(v):
    return (v % P, 0)


class Rng:
    """SplitMix64."""

    def __init__(self, seed):
        self.s = seed

    def next(self):
        self.s = (self.s + 0x9E3779B97F4A7C15) % 2**64
        z = self.s
        z = ((z ^ (z >> 30)) * 0xBF58476D1CE4E5B9) % 2**64
        z = ((z ^ (z >> 27)) * 0x94D049BB133111EB) % 2**64
        return z ^ (z >> 31)

    def below(self, m):
        return self.next() % m

    def int_in(self, lo, hi):
        return lo + self.below(hi - lo + 1)

    def fp2(self):
        while True:
            a, b = self.next(), self.next()
            if a < P and b < P:
                return (a, b)


def pow2(x):
    p = 1
    while p < x:
        p *= 2
    return p


def log2(x):
    return x.bit_length() - 1


def eq1(z, x):
    """eq for one coordinate: z·x + (1 − z)(1 − x), x a field element."""
    return add(mul(z, x), mul(sub(ONE, z), sub(ONE, x)))


def eq_bit(z, bit):
    return z if bit else sub(ONE, z)


def eq_bits(zs, value):
    out = ONE
    for t, z in enumerate(zs):
        out = mul(out, eq_bit(z, (value >> t) & 1))
    return out


def lt(point, a):
    """Σ_{h < a} eq(point, h) over len(point)-bit h, 0 ≤ a ≤ 2^len, point lowest bit first."""
    j = len(point)
    if a >= 1 << j:
        return ONE
    out, prefix = ZERO, ONE
    for t in reversed(range(j)):
        bit = (a >> t) & 1
        if bit:
            out = add(out, mul(prefix, sub(ONE, point[t])))
        prefix = mul(prefix, eq_bit(point[t], bit))
    return out


# --- The instance ------------------------------------------------------------------------------


class Instance:
    def __init__(self, rng, n, heads, kv, layers, d):
        self.n, self.H, self.Hkv, self.L, self.d = n, heads, kv, layers, d
        self.np, self.Hp, self.Lp = pow2(n), pow2(heads), pow2(layers)
        self.s = log2(self.np)
        self.a, self.b = log2(self.Hp), log2(self.Lp)
        self.N = 2 * self.s + self.a + self.b
        # Arbitrary values on the padding too, which only the mask removes: the identity must
        # hold for any tables, not only for the honest prover's zeros.
        self.Q = [[[[rng.int_in(-128, 127) for _ in range(d)] for q in range(self.np)]
                   for h in range(self.Hp)] for l in range(self.Lp)]
        self.K = [[[[rng.int_in(-128, 127) for _ in range(d)] for i in range(self.np)]
                   for g in range(kv)] for l in range(self.Lp)]
        self.c = [[[rng.int_in(-2**20, 2**20) for q in range(self.np)] for h in range(self.Hp)]
                  for l in range(self.Lp)]
        self.z = [rng.fp2() for _ in range(self.N)]

    def g(self, h):
        return min(h * self.Hkv // self.H, self.Hkv - 1)

    def real(self, l, h, q):
        return l < self.L and h < self.H and q < self.n

    def score(self, l, h, q, i):
        return sum(x * y for x, y in zip(self.Q[l][h][q], self.K[l][self.g(h)][i]))

    def index(self, l, h, q, i):
        return i + self.np * (q + self.np * (h + self.Hp * l))


# --- The literal prover ------------------------------------------------------------------------


def rounds(tables, challenges, start, count, counted):
    """RFC-0010 rounds of Σ eq·M·(c − s) over `tables` = [eq, M, c, s], lowest variable first.
    Returns the round evaluations at 0..3 and the folded tables. `counted` gates the counter."""
    out = []
    for j in range(count):
        before = MULS[0]
        half = len(tables[0]) // 2
        evals = []
        for X in range(4):
            x = base(X)
            total = ZERO
            for k in range(half):
                vals = [add(t[2 * k], mul(x, sub(t[2 * k + 1], t[2 * k]))) for t in tables]
                total = add(total, mul(mul(vals[0], vals[1]), sub(vals[2], vals[3])))
            evals.append(total)
        out.append(evals)
        r = challenges[start + j]
        tables = [[add(t[2 * k], mul(r, sub(t[2 * k + 1], t[2 * k]))) for k in range(half)]
                  for t in tables]
        if not counted:
            MULS[0] = before
    return out, tables


def literal(inst, challenges, key_rounds_only=False):
    size = 1 << inst.N
    eqt, mt, ct, st = [ZERO] * size, [ZERO] * size, [ZERO] * size, [ZERO] * size
    saved = MULS[0]
    for l in range(inst.Lp):
        for h in range(inst.Hp):
            for q in range(inst.np):
                for i in range(inst.np):
                    x = inst.index(l, h, q, i)
                    bits = x
                    e = ONE
                    for t in range(inst.N):
                        e = mul(e, eq_bit(inst.z[t], (bits >> t) & 1))
                    eqt[x] = e
                    mt[x] = ONE if (inst.real(l, h, q) and i <= q) else ZERO
                    ct[x] = base(inst.c[l][h][q])
                    st[x] = base(inst.score(l, h, q, i))
    MULS[0] = saved  # building the tables is not the rounds' work
    count = inst.s if key_rounds_only else inst.N
    before = MULS[0]
    out, tables = rounds([eqt, mt, ct, st], challenges, 0, count, True)
    work = MULS[0] - before
    final = None if key_rounds_only else (tables[2][0], tables[3][0])
    return out, final, work


# --- The candidate -----------------------------------------------------------------------------


def candidate(inst, challenges, fault=None):
    s, np_, d = inst.s, inst.np, inst.d
    zk, zq = inst.z[:s], inst.z[s:2 * s]
    zh, zl = inst.z[2 * s:2 * s + inst.a], inst.z[2 * s + inst.a:]
    # The keys as field vectors, folded lowest bit first as the rounds bind them.
    folded = [[[base(v) for v in inst.K[l][g][i]] for i in range(np_)]
              for l in range(inst.Lp) for g in range(inst.Hkv)]

    def kf(l, g):
        return folded[l * inst.Hkv + g]

    eq_rest = {}
    for l in range(inst.Lp):
        for h in range(inst.Hp):
            for q in range(np_):
                eq_rest[(l, h, q)] = mul(mul(eq_bits(zq, q), eq_bits(zh, h)), eq_bits(zl, l))
    out = []
    before = MULS[0]
    bound = ONE  # Π eq(z_t, r_t) over the bound key bits
    r_low = []
    for j in range(1, s + 1):
        hi_bits = s - j
        zhi = zk[j:]
        eq_hi = [eq_bits(zhi, t) for t in range(1 << hi_bits)]
        if fault == "high eq dropped":
            eq_hi = [ONE] * (1 << hi_bits)
        evals = []
        for X in range(4):
            x = base(X)
            point = r_low + [x]
            ex = mul(bound, eq1(zk[j - 1], x))
            # Prefix sums over the key blocks below each q_hi: of eq_hi, and of eq_hi·K̃.
            pa = [ZERO] * ((1 << hi_bits) + 1)
            for t in range(1 << hi_bits):
                pa[t + 1] = add(pa[t], eq_hi[t])
            pb, kx = {}, {}
            for l in range(inst.Lp):
                for g in range(inst.Hkv):
                    f = kf(l, g)
                    if fault == "fold swapped":
                        kt = [[add(f[2 * t + 1][e], mul(x, sub(f[2 * t][e], f[2 * t + 1][e])))
                               for e in range(d)] for t in range(1 << hi_bits)]
                    else:
                        kt = [[add(f[2 * t][e], mul(x, sub(f[2 * t + 1][e], f[2 * t][e])))
                               for e in range(d)] for t in range(1 << hi_bits)]
                    kx[(l, g)] = kt
                    acc = [[ZERO] * d]
                    for t in range(1 << hi_bits):
                        acc.append([add(acc[t][e], mul(eq_hi[t], kt[t][e])) for e in range(d)])
                    pb[(l, g)] = acc
            total = ZERO
            for l in range(inst.Lp):
                for h in range(inst.Hp):
                    for q in range(np_):
                        if fault == "padded rows unmasked":
                            if not (l < inst.L and h < inst.H):
                                continue
                        elif fault == "padded heads unmasked":
                            if not (l < inst.L and q < inst.n):
                                continue
                        elif not inst.real(l, h, q):
                            continue
                        qlo, qhi = q & ((1 << j) - 1), q >> j
                        bound_lt = qlo if fault == "LT at q_lo" else qlo + 1
                        edge = mul(eq_hi[qhi], lt(point, bound_lt))
                        upto = qhi + 1 if fault == "prefix includes q_hi" else qhi
                        if fault == "row maximum on the edge only":
                            a_q = edge
                        else:
                            a_q = add(pa[upto], edge)
                        g = (h % inst.Hkv) if fault == "key head mod" else inst.g(h)
                        vec = [add(pb[(l, g)][upto][e], mul(edge, kx[(l, g)][qhi][e]))
                               for e in range(d)]
                        dot = ZERO
                        order = reversed(range(d)) if fault == "neutral" else range(d)
                        for e in order:
                            dot = add(dot, mul(base(inst.Q[l][h][q][e]), vec[e]))
                        term = sub(mul(base(inst.c[l][h][q]), a_q), dot)
                        total = add(total, mul(eq_rest[(l, h, q)], term))
            evals.append(mul(ex, total))
        out.append(evals)
        r = challenges[j - 1]
        bound = mul(bound, eq1(zk[j - 1], r))
        r_low.append(r)
        for l in range(inst.Lp):
            for g in range(inst.Hkv):
                f = kf(l, g)
                folded[l * inst.Hkv + g] = [
                    [add(f[2 * t][e], mul(r, sub(f[2 * t + 1][e], f[2 * t][e]))) for e in range(d)]
                    for t in range(len(f) // 2)]
    work = MULS[0] - before
    # Row tables over (q, h, l), the query index lowest.
    rows = np_ * inst.Hp * inst.Lp
    eqt, mt, ct, st = [ZERO] * rows, [ZERO] * rows, [ZERO] * rows, [ZERO] * rows
    for l in range(inst.Lp):
        for h in range(inst.Hp):
            for q in range(np_):
                x = q + np_ * (h + inst.Hp * l)
                eqt[x] = mul(bound, eq_rest[(l, h, q)])
                if inst.real(l, h, q):
                    mt[x] = lt(r_low, q if fault == "row mask at q" else q + 1)
                ct[x] = base(inst.c[l][h][q])
                kvec = kf(l, inst.g(h))[0]
                acc = ZERO
                for e in range(d):
                    acc = add(acc, mul(base(inst.Q[l][h][q][e]), kvec[e]))
                st[x] = acc
    more, tables = rounds([eqt, mt, ct, st], challenges, s, inst.N - s, True)
    final = (tables[2][0], tables[3][0])
    return out + more, final, work


# --- The runs ----------------------------------------------------------------------------------

FAULTS = [
    "prefix includes q_hi",
    "LT at q_lo",
    "padded rows unmasked",
    "key head mod",
    "fold swapped",
    "high eq dropped",
    "padded heads unmasked",
    "row maximum on the edge only",
    "row mask at q",
]

DOMAINS = [
    (n, H, kv, L, d)
    for n in (1, 2, 3, 5, 8, 9, 16)
    for (H, kv) in ((1, 1), (2, 1), (3, 1), (4, 2))
    for L in (1, 3)
    for d in (1, 3)
]


def check(inst, challenges, fault=None):
    a, fa, _ = literal(inst, challenges)
    b, fb, _ = candidate(inst, challenges, fault)
    return a == b and fa == fb


def run_domains(seed, fault=None, stop_at_first=False):
    rng = Rng(seed)
    results = []
    for (n, H, kv, L, d) in DOMAINS:
        for _ in range(3):
            inst = Instance(rng, n, H, kv, L, d)
            challenges = [rng.fp2() for _ in range(inst.N)]
            ok = check(inst, challenges, fault)
            results.append({"n": n, "H": H, "Hkv": kv, "L": L, "d": d, "equal": ok})
            if stop_at_first and not ok:
                return results
    return results


def counts(seed):
    rng = Rng(seed)
    rows = []
    for e in range(6, 11):
        n = 1 << e
        inst = Instance(rng, n, 2, 1, 1, 4)
        challenges = [rng.fp2() for _ in range(inst.N)]
        MULS[0] = 0
        _, _, lit = literal(inst, challenges, key_rounds_only=True)
        MULS[0] = 0
        _, _, cand = candidate_key_rounds_only(inst, challenges)
        rows.append({"n": n, "literal": lit, "candidate": cand, "ratio": cand / lit})
        print(f"counts n={n}: literal {lit}, candidate {cand}, ratio {cand / lit:.4f}", flush=True)
    return rows


def candidate_key_rounds_only(inst, challenges):
    """The candidate's key-bit rounds, its count; the row-table rounds are the literal's."""
    out, _, work = candidate(inst, challenges)
    return out, None, work


def main():
    mode, path = sys.argv[1], sys.argv[2]
    seed = 0x45800260001
    report = {
        "experiment": "EXP-0026",
        "seed": hex(seed),
        "python": platform.python_version(),
        "platform": platform.platform(),
        "started": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    if mode in ("all", "equality"):
        control = run_domains(seed)
        report["control_before"] = {"runs": len(control), "equal": sum(r["equal"] for r in control)}
        print("control before:", report["control_before"], flush=True)
        detected = {}
        for fault in FAULTS:
            res = run_domains(seed, fault, stop_at_first=True)
            caught = not all(r["equal"] for r in res)
            detected[fault] = {"detected": caught, "first": res[-1] if caught else None}
            print(f"fault {fault}: {'DETECTED' if caught else 'SURVIVED'}", flush=True)
        report["faults"] = detected
        neutral = run_domains(seed, "neutral")
        report["neutral"] = {"runs": len(neutral), "equal": sum(r["equal"] for r in neutral)}
        print("neutral:", report["neutral"], flush=True)
        after = run_domains(seed)
        report["control_after"] = {"runs": len(after), "equal": sum(r["equal"] for r in after)}
        print("control after:", report["control_after"], flush=True)
    if mode in ("all", "counts"):
        report["counts"] = counts(seed + 1)
    report["finished"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    with open(path, "w") as f:
        json.dump(report, f, indent=2)
        f.write("\n")


if __name__ == "__main__":
    main()
