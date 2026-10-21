#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0 OR MIT
"""Literal-index oracle for RFC-0053's proposed native boundaries.

No research, prover or verifier imports. Kernels and MLEs are literal sums,
not the dynamic programs or folding used by the implementations.
"""
import json
from dataclasses import dataclass
from pathlib import Path

P = 0xFFFFFFFF00000001


@dataclass(frozen=True)
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


Z, O = E(0), E(1)


class Rng:
    def __init__(self):
        self.state = 0x53_128_0037

    def next(self):
        mask = (1 << 64) - 1
        self.state = (self.state + 0x9E3779B97F4A7C15) & mask
        z = self.state
        z = ((z ^ (z >> 30)) * 0xBF58476D1CE4E5B9) & mask
        z = ((z ^ (z >> 27)) * 0x94D049BB133111EB) & mask
        return z ^ (z >> 31)

    def field(self):
        return E(2 + self.next() % (P - 2), 1 + self.next() % (P - 1))

    def point(self, n):
        return [self.field() for _ in range(n)]


def product(xs):
    result = O
    for x in xs:
        result *= x
    return result


def bitlen(n):
    return (n - 1).bit_length()


def basis(point, x):
    return product(r if (x >> j) & 1 else O - r for j, r in enumerate(point))


def mle(table, point):
    assert len(table) == 1 << len(point)
    return sum((value * basis(point, i) for i, value in enumerate(table)), Z)


def encode(value):
    if isinstance(value, E):
        return [str(value.a), str(value.b)]
    if isinstance(value, (list, tuple)):
        return [encode(v) for v in value]
    if isinstance(value, dict):
        return {k: encode(v) for k, v in value.items()}
    return value


def case(n, vocab, prompt, shape, arbitrary, rng):
    nv, nt, no = bitlen(vocab), bitlen(n), bitlen(n - prompt)
    vp, tp, op = 1 << nv, 1 << nt, 1 << no
    l, h, m, tokens = [Z] * (vp * tp), [Z] * (vp * tp), [Z] * tp, [Z] * tp
    if arbitrary:
        l, h = [rng.field() for _ in l], [rng.field() for _ in h]
        m, tokens = [rng.field() for _ in m], [rng.field() for _ in tokens]
    else:
        rows = []
        for t in range(n):
            values = ([-2] * vocab if shape % 3 == 0 else
                      [-(v % 3) - 1 for v in range(vocab)] if shape % 3 == 1 else
                      [int(rng.next() % 15) - 7 for _ in range(vocab)])
            rows.append(values)
            for v, value in enumerate(values):
                l[t * vp + v] = E(value % P)
        for t in range(n):
            y = rows[t - 1].index(max(rows[t - 1])) if t >= prompt else (t + 1) % vocab
            h[t * vp + y], tokens[t] = O, E(y)
        for t in range(prompt - 1, n - 1):
            m[t] = E(max(rows[t]) % P)
    pv, po, z, r = rng.point(nv), rng.point(no), rng.point(nt), rng.point(nv + nt)
    rv, rt = r[:nv], r[nv:]
    hp = [h[((t + 1) % tp) * vp + v] for t in range(tp) for v in range(vp)]
    ell, maximum, shifted = mle(l, r), mle(m, rt), mle(hp, r)
    carry, originals = O, []
    for j in range(nt):
        original = rv + [Z] * j + [O] + rt[j + 1:]
        originals.append(dict(point=original, coefficient=carry * (O - rt[j]), value=mle(h, original)))
        carry *= rt[j]
    original = rv + [Z] * nt
    originals.append(dict(point=original, coefficient=carry, value=mle(h, original)))
    assert sum((i['coefficient'] * i['value'] for i in originals), Z) == shifted
    time = sum((basis(po, u) * basis(rt, prompt - 1 + u) for u in range(n - prompt)), Z)
    tv = sum((basis(pv, v) * basis(rv, v) for v in range(vocab)), Z)
    sv = sum((basis(rv, w) * sum((basis(pv, i) for i in range(w)), Z) for w in range(vocab)), Z)
    mv = sum((basis(pv, v) for v in range(vocab)), Z)
    active = sum((basis(z, t) * basis(rt, t) for t in range(prompt - 1, n - 1)), Z)
    ez = sum((basis(z, t) * basis(rt, t) for t in range(tp)), Z)
    dv = basis(rv, 0)
    lt = sum((basis(rv, v) for v in range(vocab)), Z)
    index = sum((basis(rv, v) * v for v in range(vp)), Z)
    key = [Z] * (op * vp)
    for u in range(n - prompt):
        t = prompt - 1 + u
        for v in range(vocab):
            key[u * vp + v] = m[t] - l[t * vp + v] - sum((h[(t + 1) * vp + w] for w in range(v + 1, vocab)), Z)
    raw = [ell, maximum] + [i['value'] for i in originals]
    return dict(n=n, vocab=vocab, prompt=prompt, honest=not arbitrary,
                target_v=pv, target_o=po, zero=z, point=r,
                logits=l, onehot=h, maximum=m, tokens=tokens,
                key_claim=mle(key, pv + po), raw=raw,
                token_raw=[mle(h, r), mle(tokens, rt)],
                key_final=time * (dv * mv * maximum - tv * ell - sv * shifted),
                selected_final=active * lt * shifted * ell - ez * dv * maximum,
                token_final=ez * (mle(h, r) * index - dv * mle(tokens, rt)),
                original_onehot=originals)


def main():
    rng = Rng()
    shapes = [(2, 1, 1), (3, 3, 1), (3, 5, 2), (4, 4, 2),
              (5, 3, 2), (8, 5, 1), (9, 9, 4), (9, 8, 8)]
    cases = [case(n, v, p, i, arbitrary, rng)
             for i, (n, v, p) in enumerate(shapes) for arbitrary in [False, True] for _ in range(2)]
    intervals = [(0, 0), (-1, 0), (-128, 127), (-2048, 2047),
                 (-(1 << 28) + 1, (1 << 28) - 1)]
    document = dict(description='Literal Python sums for proposed RFC-0053 boundaries; no complete proof verdict.',
                    cases=cases, ranges=[dict(lo=str(lo), hi=str(hi), bits=max(1, (hi-lo).bit_length())) for lo, hi in intervals])
    path = Path(__file__).resolve().with_name('greedy-boundary.json')
    path.write_text(json.dumps(encode(document), indent=1) + '\n')
    print(f'wrote {path.name}: {len(cases)} native cases')


if __name__ == '__main__':
    main()
