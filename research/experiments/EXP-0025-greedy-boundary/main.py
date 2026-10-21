#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0 OR MIT
"""Exact E0 oracle for original-source greedy boundaries; no production imports."""

import argparse
import hashlib
import itertools
import json
import math
import platform
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

P = 0xFFFFFFFF00000001
SEED = 0x12800370025


@dataclass(frozen=True, slots=True)
class E:
    a: int
    b: int = 0

    def __add__(self, other):
        other = other if isinstance(other, E) else E(other)
        return E((self.a + other.a) % P, (self.b + other.b) % P)

    __radd__ = __add__

    def __neg__(self):
        return E(-self.a % P, -self.b % P)

    def __sub__(self, other):
        return self + -other if isinstance(other, E) else self + E(-other % P)

    def __mul__(self, other):
        other = other if isinstance(other, E) else E(other)
        return E((self.a * other.a + 7 * self.b * other.b) % P,
                 (self.a * other.b + self.b * other.a) % P)

    __rmul__ = __mul__


ZERO, ONE = E(0), E(1)


class Rng:
    def __init__(self):
        self.state = SEED

    def next(self):
        mask = (1 << 64) - 1
        self.state = (self.state + 0x9E3779B97F4A7C15) & mask
        z = self.state
        z = ((z ^ (z >> 30)) * 0xBF58476D1CE4E5B9) & mask
        z = ((z ^ (z >> 27)) * 0x94D049BB133111EB) & mask
        return z ^ (z >> 31)

    def field(self):
        return E(2 + self.next() % (P - 2), 1 + self.next() % (P - 1))

    def point(self, width):
        return [self.field() for _ in range(width)]


def product(values):
    out = ONE
    for value in values:
        out *= value
    return out


def bits(n):
    assert n > 0
    return (n - 1).bit_length()


def cube_point(width, value):
    return [E((value >> j) & 1) for j in range(width)]


def basis(point, value):
    return product(r if (value >> j) & 1 else ONE - r for j, r in enumerate(point))


def eq(a, b):
    assert len(a) == len(b)
    return product(x * y + (ONE - x) * (ONE - y) for x, y in zip(a, b))


def mle(table, point):
    assert len(table) == 1 << len(point)
    values = list(table)
    for r in point:
        values = [a + r * (b - a) for a, b in zip(values[::2], values[1::2])]
    return values[0]


def prefix(points, bound):
    """MSB prefix partition of a product of equality weights."""
    width = len(points[0])
    assert all(len(p) == width for p in points) and 0 <= bound <= 1 << width
    weights = [(product(ONE - p[j] for p in points), product(p[j] for p in points))
               for j in range(width)]
    lower = [ONE]
    for a, b in weights:
        lower.append(lower[-1] * (a + b))
    if bound == 1 << width:
        return lower[-1]
    answer, equal = ZERO, ONE
    for j in reversed(range(width)):
        a, b = weights[j]
        if (bound >> j) & 1:
            answer += equal * a * lower[j]
            equal *= b
        else:
            equal *= a
    return answer


def compare_bit(a, b, previous):
    return previous if a == b else (-1 if a < b else 1)


def row(output, source, offset, count):
    """Carry DP for sum(u<count) eq(output,u) eq(source,offset+u)."""
    assert count <= 1 << len(output) and offset + count <= 1 << len(source)
    states = {(0, 0): ONE}
    for j in range(max(len(output), len(source)) + 1):
        following = {}
        for (carry, comparison), weight in states.items():
            for u in range(2 if j < len(output) else 1):
                total = u + ((offset >> j) & 1) + carry
                t, next_carry = total & 1, total >> 1
                if j >= len(source) and t:
                    continue
                factor = weight
                if j < len(output):
                    factor *= output[j] if u else ONE - output[j]
                if j < len(source):
                    factor *= source[j] if t else ONE - source[j]
                key = (next_carry, compare_bit(u, (count >> j) & 1, comparison))
                following[key] = following.get(key, ZERO) + factor
        states = following
    return states.get((0, -1), ZERO)


def suffix(target, source, vocab):
    """Joint order/bound DP for sum(w<vocab) eq(source,w) sum(i<w) eq(target,i)."""
    assert len(target) == len(source) and 0 <= vocab <= 1 << len(source)
    states = {(0, 0): ONE}
    for j in range(len(target) + 1):
        following = {}
        for (order, bound), weight in states.items():
            for i in range(2 if j < len(target) else 1):
                for w in range(2 if j < len(target) else 1):
                    factor = weight
                    if j < len(target):
                        factor *= target[j] if i else ONE - target[j]
                        factor *= source[j] if w else ONE - source[j]
                    key = (compare_bit(i, w, order), compare_bit(w, (vocab >> j) & 1, bound))
                    following[key] = following.get(key, ZERO) + factor
        states = following
    return states.get((-1, -1), ZERO)


def shift_terms(point, vocabulary_bits):
    """Cyclic time increment as weighted ORIGINAL MLE points, never a new source tensor."""
    v, time = point[:vocabulary_bits], point[vocabulary_bits:]
    terms, carry = [], ONE
    for k in range(len(time)):
        terms.append((carry * (ONE - time[k]), v + [ZERO] * k + [ONE] + time[k + 1:]))
        carry *= time[k]
    terms.append((carry, v + [ZERO] * len(time)))
    return terms


def shifted(table, point, vocabulary_bits):
    return sum((coefficient * mle(table, original)
                for coefficient, original in shift_terms(point, vocabulary_bits)), ZERO)


def width(lo, hi):
    assert -(1 << 28) < lo <= hi < 1 << 28
    return max(1, (hi - lo).bit_length())


def accepts(logits, selected, maximum, range_bits):
    keys = [maximum - value - int(v < selected) for v, value in enumerate(logits)]
    return maximum == logits[selected] and all(0 <= key < 1 << range_bits for key in keys)


def sources(n, vocab, prompt, family, rng):
    nv, nt = bits(vocab), bits(n)
    vp, tp = 1 << nv, 1 << nt
    if family == 3:
        return ([rng.field() for _ in range(tp * vp)],
                [rng.field() for _ in range(tp * vp)],
                [rng.field() for _ in range(tp)], [rng.field() for _ in range(tp)])
    logits, h, maximum, tokens = [ZERO] * (tp * vp), [ZERO] * (tp * vp), [ZERO] * tp, [ZERO] * tp
    integers = []
    for t in range(n):
        values = [int(rng.next() % 15) - 7 for _ in range(vocab)]
        integers.append(values)
        for v, value in enumerate(values):
            logits[t * vp + v] = E(value % P)
    for t in range(n):
        y = (integers[t - 1].index(max(integers[t - 1])) if t >= prompt else rng.next() % vocab)
        if family == 1 and t == prompt and vocab > 1:
            y = (y + 1) % vocab
        tokens[t], h[t * vp + y] = E(y), ONE
    for t in range(prompt - 1, n - 1):
        maximum[t] = E(max(integers[t]) % P)
    if family == 1 and vocab == 1:
        maximum[prompt - 1] += ONE
    if family == 2:
        maximum[n - 1 if (n + vocab + prompt) % 2 else prompt - 1] += ONE
    return logits, h, maximum, tokens


def finish(kind, left, right, maximum, coefficients):
    if kind == 'key':
        cm, cl, ch = coefficients
        return cm * maximum - cl * left - ch * right
    if kind == 'selected':
        active, cm = coefficients
        return active * left * right - cm * maximum
    assert kind == 'tokens'
    ch, ct = coefficients
    return ch * left - ct * right


def native(kind, n, vocab, prompt, target_v, target_o, zero, point, data):
    nv = bits(vocab)
    rv, rt = point[:nv], point[nv:]
    logits, h, maximum, tokens = data
    l, hp, m = mle(logits, point), shifted(h, point, nv), mle(maximum, rt)
    delta_v = product(ONE - x for x in rv)
    if kind == 'key':
        time = row(target_o, rt, prompt - 1, n - prompt)
        return finish(kind, l, hp, m, (time * delta_v * prefix([target_v], vocab),
                      time * prefix([target_v, rv], vocab), time * suffix(target_v, rv, vocab)))
    if kind == 'selected':
        active = prefix([zero, rt], n - 1) - prefix([zero, rt], prompt - 1)
        return finish(kind, l, hp, m, (active * prefix([rv], vocab), eq(zero, rt) * delta_v))
    assert kind == 'tokens'
    index = sum((r * (1 << j) for j, r in enumerate(rv)), ZERO)
    return finish(kind, mle(h, point), mle(tokens, rt), ZERO,
                  (eq(zero, rt) * index, eq(zero, rt) * delta_v))


def check_domain(n, vocab, prompt, rng, counters, degree):
    nv, nt, no = bits(vocab), bits(n), bits(n - prompt)
    vp, tp, op = 1 << nv, 1 << nt, 1 << no
    families = [sources(n, vocab, prompt, f, rng) for f in range(4)]
    for _ in range(4):
        pv, po, z, rv, rt = rng.point(nv), rng.point(no), rng.point(nt), rng.point(nv), rng.point(nt)
        assert row(po, rt, prompt - 1, n - prompt) == sum(
            (basis(po, u) * basis(rt, prompt - 1 + u) for u in range(n - prompt)), ZERO)
        assert prefix([pv, rv], vocab) == sum((basis(pv, v) * basis(rv, v) for v in range(vocab)), ZERO)
        assert prefix([rv], vocab) == sum((basis(rv, v) for v in range(vocab)), ZERO)
        assert suffix(pv, rv, vocab) == sum((basis(rv, w) * sum(
            (basis(pv, i) for i in range(w)), ZERO) for w in range(vocab)), ZERO)
        counters['kernel_equalities'] += 4
        # Candidate coefficients on the source cube; cache them across the four source families.
        wr = [row(po, cube_point(nt, t), prompt - 1, n - prompt) for t in range(tp)]
        wl = [prefix([pv, cube_point(nv, v)], vocab) for v in range(vp)]
        wh = [suffix(pv, cube_point(nv, v), vocab) for v in range(vp)]
        wm = prefix([pv], vocab)
        wz = [prefix([z, cube_point(nt, t)], n - 1)
              - prefix([z, cube_point(nt, t)], prompt - 1) for t in range(tp)]
        ez = [basis(z, t) for t in range(tp)]
        for family, data in enumerate(families):
            logits, h, maximum, tokens = data
            literal_shift = [h[((t + 1) % tp) * vp + v] for t in range(tp) for v in range(vp)]
            assert shifted(h, rv + rt, nv) == mle(literal_shift, rv + rt)
            counters['shift_equalities'] += 1
            # Judge the actual final-value path as well as its cube sum and degree.
            time = sum((basis(po, u) * basis(rt, prompt - 1 + u) for u in range(n - prompt)), ZERO)
            tv = sum((basis(pv, v) * basis(rv, v) for v in range(vocab)), ZERO)
            hv = sum((basis(rv, w) * sum((basis(pv, i) for i in range(w)), ZERO)
                      for w in range(vocab)), ZERO)
            mv = sum((basis(pv, v) for v in range(vocab)), ZERO)
            active = sum((basis(z, t) * basis(rt, t) for t in range(prompt - 1, n - 1)), ZERO)
            dz, dv = eq(z, rt), basis(rv, 0)
            lv, hn, mm = mle(logits, rv + rt), mle(literal_shift, rv + rt), mle(maximum, rt)
            actual = [native(k, n, vocab, prompt, pv, po, z, rv + rt, data)
                      for k in ['key', 'selected', 'tokens']]
            expected = [time * (dv * mv * mm - tv * lv - hv * hn),
                        active * sum((basis(rv, v) for v in range(vocab)), ZERO) * hn * lv - dz * dv * mm,
                        dz * (mle(h, rv + rt) * sum((basis(rv, v) * v for v in range(vp)), ZERO)
                              - dv * mle(tokens, rt))]
            assert actual == expected, ('final', n, vocab, prompt, family)
            counters['native_final_equalities'] += 3
            keys = [ZERO] * (op * vp)
            for u in range(n - prompt):
                t = prompt - 1 + u
                for v in range(vocab):
                    keys[u * vp + v] = maximum[t] - logits[t * vp + v] - sum(
                        (h[(t + 1) * vp + w] for w in range(v + 1, vocab)), ZERO)
            key_target = mle(keys, pv + po)
            key_sum = selected_sum = token_sum = ZERO
            for t in range(tp):
                for v in range(vp):
                    at, following = t * vp + v, ((t + 1) % tp) * vp + v
                    key_sum += finish('key', logits[at], h[following], maximum[t],
                                      (wr[t] * wm if v == 0 else ZERO, wr[t] * wl[v], wr[t] * wh[v]))
                    selected_sum += finish('selected', logits[at], h[following], maximum[t],
                                           (wz[t] if v < vocab else ZERO, ez[t] if v == 0 else ZERO))
                    token_sum += finish('tokens', h[at], tokens[t], ZERO,
                                        (ez[t] * v, ez[t] if v == 0 else ZERO))
            selected_target = sum((ez[t] * ((sum(
                (h[(t + 1) * vp + v] * logits[t * vp + v] for v in range(vocab)), ZERO)
                if prompt - 1 <= t < n - 1 else ZERO) - maximum[t]) for t in range(tp)), ZERO)
            token_target = sum((ez[t] * (sum((h[t * vp + v] * v for v in range(vp)), ZERO)
                                        - tokens[t]) for t in range(tp)), ZERO)
            assert key_sum == key_target, ('key', n, vocab, prompt, family)
            assert selected_sum == selected_target, ('selected', n, vocab, prompt, family)
            assert token_sum == token_target, ('tokens', n, vocab, prompt, family)
            counters['cube_equalities'] += 3
            if family == 0:
                assert selected_target == token_target == ZERO
                counters['honest_zero_checks'] += 2
        if degree:
            data = families[3]
            for kind, limit in [('key', 2), ('selected', 3), ('tokens', 2)]:
                for coordinate in range(nv + nt):
                    evaluations = []
                    for value in range(limit + 2):
                        changed = list(rv + rt)
                        changed[coordinate] = E(value)
                        evaluations.append(native(kind, n, vocab, prompt, pv, po, z, changed, data))
                    difference = sum((e * ((-1) ** (limit + 1 - i) * math.comb(limit + 1, i))
                                      for i, e in enumerate(evaluations)), ZERO)
                    assert difference == ZERO, ('degree', kind, coordinate, n, vocab, prompt)
                    counters['degree_checks'] += 1


def integers(max_vocab, counters):
    for vocab in range(1, max_vocab + 1):
        for logits in itertools.product(range(-2, 3), repeat=vocab):
            highest = max(logits)
            first = logits.index(highest)
            for y in range(vocab):
                for maximum in range(-3, 4):
                    expected = y == first and maximum == highest
                    assert accepts(logits, y, maximum, 29) == expected
                    assert accepts(logits, y, maximum, width(-2, 2)) == expected
                    counters['integer_cases'] += 1
    for lo, hi in [(0, 0), (-1, 0), (0, 1), (-128, 127),
                   (-(1 << 28) + 1, (1 << 28) - 1), (-(1 << 28) + 1, -1)]:
        k = width(lo, hi)
        assert 1 <= k <= 29
        for logits in [(lo, hi), (hi, lo), (lo, lo), (hi, lo, hi)]:
            for y in range(len(logits)):
                for m in {lo, hi, lo - 1, hi + 1}:
                    expected = y == logits.index(max(logits)) and m == max(logits)
                    assert accepts(logits, y, m, k) == expected
                    for v, value in enumerate(logits):
                        key = m - value - int(v < y)
                        if -(1 << 29) <= key < 0:
                            assert key % P >= 1 << k
                    counters['boundary_cases'] += 1


def frozen():
    here = Path(__file__).resolve().parent
    root = here.parents[2]
    plan = here / 'results/README.preregistered.txt'
    digest = hashlib.sha256(plan.read_bytes()).hexdigest()
    assert digest in (here / 'results/preregistration.txt').read_text()
    assert plan.read_bytes() == subprocess.check_output(['git', 'show', 'HEAD:' + str(plan.relative_to(root))], cwd=root)
    return digest


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--quick', action='store_true', help='bounded fault-injection control only')
    args = parser.parse_args()
    registration = frozen()
    rng = Rng()
    counters = dict(domains=0, kernel_equalities=0, shift_equalities=0,
                    cube_equalities=0, honest_zero_checks=0, degree_checks=0,
                    native_final_equalities=0, integer_cases=0, boundary_cases=0)
    domains = ([(2, 1, 1), (3, 3, 1), (4, 2, 2), (5, 4, 2), (5, 3, 2)] if args.quick else
               [(n, v, p) for n in range(2, 10) for v in range(1, 10) for p in range(1, n)])
    for n, vocab, prompt in domains:
        check_domain(n, vocab, prompt, rng, counters, n <= 5 and vocab <= 4)
        counters['domains'] += 1
    integers(3 if args.quick else 5, counters)
    assert counters['domains'] == (5 if args.quick else 324)
    assert counters['integer_cases'] == (3010 if args.quick else 129885)
    print(json.dumps(dict(evidence='E0 finite exact arithmetic, not full protocol validation',
                         preregistration_sha256=registration, seed=SEED, quick=args.quick,
                         counters=counters, environment=dict(python=sys.version, system=platform.platform(),
                                                            machine=platform.machine())), indent=2))


if __name__ == '__main__':
    main()
