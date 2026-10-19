#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0 OR MIT
"""EXP-0021: exact finite audit. This is not a production proof verifier."""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import itertools
import json
import os
from pathlib import Path
import platform
import resource
import signal
import subprocess
import sys
import time
from collections import Counter
from dataclasses import dataclass
from datetime import datetime, timezone
from fractions import Fraction

sys.dont_write_bytecode = True
HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
P = (1 << 64) - (1 << 32) + 1
FROZEN = "90b7307ed2a2162bbd7a56a426c959068c83500c8111a6432e72981f641252bc"
COMMIT = "e3a2a51"
SEED = 0x0021003400000001


class AuditFailure(Exception):
    pass


class AuditLimit(Exception):
    pass


class Meter:
    def __init__(self):
        self.start = time.monotonic()
        self.ops = 0
        self.judgments = 0
        self.physical = 0
        self.counts = Counter()
        self.maxima = Counter()
        self.arm = "setup"
        self.stream = None
        self.failures = []
        self.last_check = 0

    def operation(self, count=1):
        self.ops += count
        if self.ops - self.last_check >= 65536:
            self.last_check = self.ops
            self.check()

    def check(self):
        if self.ops > 3_000_000_000 or self.judgments > 100_000_000:
            raise AuditLimit("arithmetic/judgment cap")
        if time.monotonic() - self.start > 900:
            raise AuditLimit("15 minute watchdog")
        rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        if sys.platform != "darwin":
            rss *= 1024
        self.maxima["process_rss_bytes"] = max(self.maxima["process_rss_bytes"], rss)
        # This is stronger than the retained-fixture memory limit: whole-process RSS.
        if rss > 256 * 1024 * 1024:
            raise AuditLimit("256 MiB process RSS (stronger than retained-fixture bound)")

    def observe(self, condition):
        self.judgments += 1
        self.physical += 1
        self.counts[f"judgments/{self.arm}"] += 1
        if self.judgments % 65536 == 0:self.check()
        return condition

    def judge(self, condition, name, detail=None):
        self.observe(condition)
        if not condition:
            row = {"arm": self.arm, "name": name, "detail": clean(detail)}
            self.failures.append(row)
            self.event("failure", row)
            raise AuditFailure(json.dumps(row))
        if self.judgments % 65536 == 0:
            self.check()

    def cached(self, delta):
        self.judgments += delta
        self.counts[f"judgments/{self.arm}"] += delta
        self.counts["cached_logical_judgments"] += delta
        self.check()

    def event(self, kind, data):
        if self.stream is not None:
            self.stream.write(json.dumps({"kind": kind, "data": clean(data)}, sort_keys=True) + "\n")
            self.stream.flush()


M = Meter()


def clean(v):
    if isinstance(v, F):
        return [v.a, v.b]
    if isinstance(v, R):
        return str(v.v)
    if isinstance(v, Fraction):
        return str(v)
    if isinstance(v, Desc):
        return list(v.identity)
    if isinstance(v, dict):
        return {str(k): clean(x) for k, x in v.items()}
    if isinstance(v, (list, tuple)):
        return [clean(x) for x in v]
    return v


class F:
    """Schoolbook Fp[u]/(u^2-7), using arbitrary precision integers."""
    __slots__ = ("a", "b")

    def __init__(self, a=0, b=0):
        if isinstance(a, F):
            self.a, self.b = a.a, a.b
        else:
            self.a, self.b = int(a) % P, int(b) % P

    def __add__(self, other):
        other = coerce(other, F)
        M.operation(2)
        return F(self.a + other.a, self.b + other.b)

    __radd__ = __add__

    def __sub__(self, other):
        other = coerce(other, F)
        M.operation(2)
        return F(self.a - other.a, self.b - other.b)

    def __rsub__(self, other):
        return coerce(other, F) - self

    def __neg__(self):
        M.operation(2)
        return F(-self.a, -self.b)

    def __mul__(self, other):
        other = coerce(other, F)
        M.operation(7)
        return F(self.a * other.a + 7 * self.b * other.b,
                 self.a * other.b + self.b * other.a)

    __rmul__ = __mul__

    def __eq__(self, other):
        if isinstance(other, (int, F)):
            other = coerce(other, F)
            return self.a == other.a and self.b == other.b
        return False

    def __hash__(self):
        return hash((self.a, self.b))

    def __repr__(self):
        return f"F({self.a},{self.b})"


class R:
    __slots__ = ("v",)

    def __init__(self, value=0):
        self.v = value.v if isinstance(value, R) else Fraction(value)

    def __add__(self, other):
        M.operation()
        return R(self.v + coerce(other, R).v)

    __radd__ = __add__

    def __sub__(self, other):
        M.operation()
        return R(self.v - coerce(other, R).v)

    def __rsub__(self, other):
        return coerce(other, R) - self

    def __mul__(self, other):
        M.operation()
        return R(self.v * coerce(other, R).v)

    __rmul__ = __mul__

    def __neg__(self):
        M.operation()
        return R(-self.v)

    def __eq__(self, other):
        return self.v == coerce(other, R).v

    def __repr__(self):
        return str(self.v)


def iadd(a,b):
    M.operation(); return a+b


def isub(a,b):
    M.operation(); return a-b


def imul(a,b):
    M.operation(); return a*b


def idiv(a,b):
    M.operation(); return a//b


def isum(values):
    out=0
    for v in values:out=iadd(out,v)
    return out


def coerce(x, cls):
    return x if isinstance(x, cls) else cls(x)


ZERO, ONE, U = F(), F(1), F(0, 1)
HALF = F((P + 1) // 2)
INV6 = F((5 * P + 1) // 6)  # 6 divides 5*P+1.


def pad(n):
    return 1 << (n - 1).bit_length()


@dataclass(frozen=True)
class Desc:
    d: int
    n: int
    W: int
    L: int = 1
    H: int = 1
    format: str = "e0-local-window/1"

    @property
    def N(self):
        return 1 << self.d

    @property
    def identity(self):
        return self.format, self.d, self.n, self.W, self.L, self.H


def descriptor(fields):
    names = {"format", "d", "n", "W", "L", "H"}
    if not isinstance(fields, dict) or set(fields) != names:
        return "descriptor", None
    if fields["format"] != "e0-local-window/1":
        return "descriptor", None
    if any(type(fields[k]) is not int for k in names - {"format"}):
        return "descriptor", None
    d, n, w, l, h = (fields[k] for k in ("d", "n", "W", "L", "H"))
    if not (0 <= d <= 3 and 0 <= n <= 1 << d and 1 <= w <= 1 << 17
            and 1 <= l <= 3 and 1 <= h <= 3):
        return "descriptor", None
    return None, Desc(d, n, w, l, h)


def windows(n):
    return sorted(set(range(1, n + 1)) | {n + 1, 1024})


def positional(q, i, n, w):
    # Direct integer predicate, independent of all three subtraction chains.
    return int(isub(q,i)<w) if 0<=i<=q<n else 0


def dp(rq, ri, n, w, omit=None, arithmetic=None):
    cls = arithmetic or (type(rq[0]) if rq else F)
    zero, one = cls(0), cls(1)
    d = len(rq)
    if n == 0:
        return zero
    if d == 0:
        return one
    ww, aa = isub(min(w,1<<d),1), isub(n,1)
    states = {(0, 0, 0): one}
    transitions = 0
    maxstates = 1
    for bit, (r, s) in enumerate(zip(rq, ri)):
        nxt = {}
        weights = ((one-r)*(one-s), r*(one-s), (one-r)*s, r*s)
        for (b0, b1, b2), value in states.items():
            for Q, I in ((0, 0), (1, 0), (0, 1), (1, 1)):
                transitions += 1
                M.operation(7)  # six borrow subtractions and digit remainder/division
                t = Q - I - b0
                digit = t % 2
                key = (int(t < 0), int(((ww >> bit) & 1) - digit - b1 < 0),
                       int(((aa >> bit) & 1) - Q - b2 < 0))
                contribution = value * weights[Q + 2 * I]
                nxt[key] = nxt.get(key, zero) + contribution
        states = nxt
        maxstates = max(maxstates, len(states))
    M.maxima["dp_states"] = max(M.maxima["dp_states"], maxstates)
    M.maxima["dp_transitions"] = max(M.maxima["dp_transitions"], transitions)
    if maxstates > 8 or transitions > 32 * d:
        raise AuditFailure("DP work bound")
    if omit is not None:
        return sum((v for key, v in states.items()
                    if all(key[j] == 0 for j in range(3) if j != omit)), zero)
    return states.get((0, 0, 0), zero)


def weights(point, cls=None):
    """Direct Boolean basis, without a borrow recurrence."""
    cls = cls or (type(point[0]) if point else F)
    out = [cls(1)]
    for r in point:
        out = [v*(cls(1)-r) for v in out] + [v*r for v in out]
    return out


def mle_ref(table, point, cls=None):
    cls = cls or (type(point[0]) if point else F)
    if len(table) != 1 << len(point):
        raise AuditFailure("reference MLE dimension")
    return sum((coerce(v, cls)*a for v, a in zip(table, weights(point, cls))), cls(0))


def mle_fold(table, point):
    cls = type(point[0]) if point else F
    table = [coerce(v, cls) for v in table]
    for r in point:
        table = [table[i] + r*(table[i+1]-table[i]) for i in range(0, len(table), 2)]
    if len(table) != 1:
        raise AuditFailure("MLE dimension")
    return table[0]


def lt_ref(point, bound):
    cls = type(point[0]) if point else F
    return mle_ref([int(i < bound) for i in range(1 << len(point))], point, cls)


def full_p(desc, point):
    d, hb = desc.d, (pad(desc.H).bit_length()-1)
    ri, rq = point[:d], point[d:2*d]
    rh, rl = point[2*d:2*d+hb], point[2*d+hb:]
    return dp(rq, ri, desc.n, desc.W)*lt_ref(rh, desc.H)*lt_ref(rl, desc.L)


def splitmix(seed):
    while True:
        M.operation(3)  # one addition and two multiplications in fixture PRNG
        seed = (seed + 0x9E3779B97F4A7C15) & ((1 << 64)-1)
        z = seed
        z = ((z ^ (z >> 30))*0xBF58476D1CE4E5B9) & ((1 << 64)-1)
        z = ((z ^ (z >> 27))*0x94D049BB133111EB) & ((1 << 64)-1)
        yield z ^ (z >> 31)


def catalog(k, length):
    rng = splitmix(SEED)
    if k == 0: return [ZERO]*length
    if k == 1: return [ONE]*length
    if k == 2: return [HALF]*length
    if k == 3: return [U]*length
    if k == 4: return [F(i % 2) for i in range(length)]
    if k == 5: return [(HALF, U, ONE+U, ZERO)[i % 4] for i in range(length)]
    if k == 6: return [(F(2, 1), F(1, -1))[i % 2] for i in range(length)]
    return [F(next(rng), next(rng)) for _ in range(length)]


def positional_table(desc):
    return [positional(q, i, desc.n, desc.W) for q in range(desc.N) for i in range(desc.N)]


def selector_audit():
    M.arm = "selector"
    for d in range(4):
        nsize = 1 << d
        for n in range(nsize+1):
            for w in windows(nsize):
                desc = Desc(d, n, w)
                table = positional_table(desc)
                for cls in (R, F):
                    for q in range(nsize):
                        rq = [cls((q >> j) & 1) for j in range(d)]
                        for i in range(nsize):
                            ri = [cls((i >> j) & 1) for j in range(d)]
                            got = dp(rq, ri, n, w, arithmetic=cls)
                            M.judge(got == cls(positional(q, i, n, w)), "boolean selector", [desc, q, i])
                            M.counts["selector/boolean"] += 1
                if d <= 2:
                    for cls, alphabet in ((R, [R(0), R(1), R(Fraction(1,2)), R(-1)]),
                                          (F, [ZERO, ONE, U, ONE+U])):
                        for pt in itertools.product(alphabet, repeat=2*d):
                            got = dp(pt[d:], pt[:d], n, w, arithmetic=cls)
                            expected = mle_ref(table, pt, cls)
                            M.judge(got == expected, "offcube selector", [desc, pt, got, expected])
                            M.counts[f"selector/offcube/{cls.__name__}"] += 1
                else:
                    rational = [([0]*3,[0]*3),([1]*3,[1]*3),([Fraction(1,2)]*3,[Fraction(1,2)]*3),
                                ([-1]*3,[-1]*3),([Fraction(1,2)]*3,[0]*3),([0]*3,[Fraction(1,2)]*3),
                                ([0,1,Fraction(1,2)],[Fraction(1,2),0,1]),
                                ([Fraction(1,2),-1,2],[2,Fraction(1,2),-1])]
                    def image(x):
                        mapping={Fraction(1,2): F((P+1)//2,1), -1:F(-1,1), 2:F(2,1)}
                        return mapping[x] if x in mapping else F(x)
                    for cls in (R,F):
                        for rq0, ri0 in rational:
                            rq = [R(x) if cls is R else image(x) for x in rq0]
                            ri = [R(x) if cls is R else image(x) for x in ri0]
                            got = dp(rq,ri,n,w)
                            expected = mle_ref(table,ri+rq,cls)
                            M.judge(got == expected,"d3 selector catalog",[desc,rq,ri])
                            M.counts["selector/d3_catalog"] += 1
                M.event("selector_descriptor", {"descriptor": desc, "counts": dict(M.counts), "ops":M.ops})
    for l,h,n,w in itertools.product((1,2,3),(1,2,3),(0,1,3),(1,2,4,1024)):
        desc=Desc(2,n,w,l,h)
        hp,lp=pad(h),pad(l)
        table=[int(ll<l and hh<h)*positional(q,i,n,w)
               for ll in range(lp) for hh in range(hp) for q in range(4) for i in range(4)]
        bits=(len(table)-1).bit_length()
        for index,value in enumerate(table):
            pt=[F((index>>j)&1) for j in range(bits)]
            M.judge(full_p(desc,pt)==value,"full boolean selector",[desc,index])
            M.counts["selector/full_boolean"]+=1
        for k in (2,3,4,5):
            pt=catalog(k,bits)
            M.judge(full_p(desc,pt)==mle_ref(table,pt),"full offcube selector",[desc,k])
            M.counts["selector/full_offcube"]+=1
    trap=R(Fraction(1,2))
    causal=mle_ref([1,1,0,1],[trap,trap],R)
    lower=mle_ref([1,0,1,1],[trap,trap],R)
    joint=dp([trap],[trap],2,1)
    M.judge(causal*lower==R(Fraction(9,16)) and joint==trap and causal*lower!=joint,"registered MLE trap")
    M.event("representative",{"name":"MLE trap","wrong":causal*lower,"joint":joint})


def normative_table():
    path=REPO/'spec/vectors/tables/generate.py'
    spec=importlib.util.spec_from_file_location('exp21_table_oracle',path)
    module=importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    table=[module.exp_value(i,16,1,1) for i in range(16)]
    M.counts['normative_oracle_requested_entries']=16
    M.judge(table[0]==65536 and table[15]==0 and table[3]!=0 and table[7]!=0,'normative table boundary',table)
    M.event('normative_table',table)
    return table


def membership(k, e, table):
    return k.b == 0 and k.a < len(table) and e == table[k.a]


def row_reason(scores, selected, real, c, e, reciprocal, table, sigma,
               omit=None, masked_zero=False):
    k0=len(table)-1
    for i,x in enumerate(scores):
        key=c-F(x) if selected[i] else F(k0)
        if not selected[i] and omit=='excluded-e':
            continue
        if not membership(key,e[i],table):
            return 'excluded-e' if not selected[i] else 'lookup'
    root=ONE
    for i,x in enumerate(scores):
        leaf=c-F(x) if selected[i] else F(0 if masked_zero else 1)
        root=root*leaf
    if omit!='false-max' and root != F(0 if real else 1):
        return 'false-max'
    if real:
        if not isinstance(reciprocal,int) or isinstance(reciprocal,bool) or not 0<=reciprocal<1<<(sigma+1):
            return 'range-R'
    elif not isinstance(reciprocal,int) or isinstance(reciprocal,bool) or not 0<=reciprocal<1<<(sigma+1):
        return 'range-R'
    total=isum(e)
    a=isub(imul(int(real),1<<(2*sigma)),imul(reciprocal,total))
    b=isub(isub(total,int(real)),a)
    if not (0<=a<1<<(sigma+18) and 0<=b<1<<(sigma+18)):
        return 'reciprocal'
    if not real and omit!='padding-c' and c!=ZERO:
        return 'padding-c'
    if not real and omit!='padding-R' and reciprocal!=0:
        return 'padding-R'
    return None


def row_core(scores, selected, real):
    M.counts['row_core/physical_inputs']+=1
    table=[4,1,1,0]
    maximum=max((x for x,s in zip(scores,selected) if s),default=0)
    cs=list(dict.fromkeys([F(x) for x in (-2,-1,0,1,2)]+[U,ONE+U,F(maximum,1)]))
    honest_e=[table[isub(maximum,x)] if s else 0 for x,s in zip(scores,selected)]
    honest_r=idiv(16,isum(honest_e)) if real else 0
    r_candidates=list(range(8))
    passing=[]
    for c in cs:
        possible=[]
        valid=True
        for x,s in zip(scores,selected):
            k=c-F(x) if s else F(3)
            expected=table[k.a] if k.b==0 and k.a<len(table) else None
            outputs=[]
            for e in range(5):
                got=membership(k,e,table)
                M.judge(got==(expected is not None and e==expected),'functional pointwise lookup')
                M.counts['row/e_probes']+=1
                M.counts['physical_row/e_probes']+=1
                if got:outputs.append(e)
            if len(outputs)!=1:
                valid=False
            possible.append(outputs[0] if outputs else 0)
        # Gate R probes on the key/product/padding-c checks, independently of R/slacks.
        root=ONE
        for x,s in zip(scores,selected):
            root=root*(c-F(x) if s else ONE)
        anchored=valid and root==F(0 if real else 1) and (real or c==ZERO)
        M.judge(anchored==(c==F(maximum) if real else c==ZERO),'maximum anchor',[scores,selected,real,c])
        M.counts['row/c_probes']+=1
        M.counts['physical_row/c_probes']+=1
        if anchored:
            for reciprocal in r_candidates:
                reason=row_reason(scores,selected,real,c,possible,reciprocal,table,2)
                M.judge((reason is None)==(reciprocal==honest_r),'reciprocal witness',[scores,real,reciprocal,reason])
                M.counts['row/R_probes']+=1
                M.counts['physical_row/R_probes']+=1
                if reason is None:passing.append((c,possible,reciprocal))
            for bad in (U,ONE+U):
                M.judge(row_reason(scores,selected,real,c,possible,bad,table,2)=='range-R','nonbase reciprocal range')
                M.counts['row/nonbase_R_probes']+=1
                M.counts['physical_row/nonbase_R_probes']+=1
    M.judge(len(passing)==1,'unique local witness',[scores,selected,real,passing])
    c,e,r=passing[0]
    total=isum(e);a=isub(imul(int(real),16),imul(r,total));b=isub(isub(total,int(real)),a)
    expected=(F(maximum) if real else ZERO,honest_e,honest_r)
    M.judge(c==expected[0],'direct c oracle',[scores,selected,real])
    M.judge(len(e)==len(expected[1]),'direct e length oracle')
    for got,want in zip(e,expected[1]):M.judge(got==want,'direct e scalar oracle')
    M.judge(r==expected[2],'direct R scalar oracle')
    oracle_s=isum(honest_e);oracle_a=isub(imul(int(real),16),imul(honest_r,oracle_s));oracle_b=isub(isub(oracle_s,int(real)),oracle_a)
    for got,want in zip((total,a,b),(oracle_s,oracle_a,oracle_b)):M.judge(got==want,'direct sum/slack scalar oracle')


def pointwise_audit(table16):
    M.arm='pointwise'
    cache={}
    expected_contexts=0
    for size in (1,2,4):
        for n,w in itertools.product(range(size+1),windows(size)):
            begin=Counter(M.counts)
            for q in range(size):
                for scores in itertools.product((-1,0,1),repeat=size):
                    for l,h in itertools.product((1,3),(1,3)):
                        for ll,hh in itertools.product(range(pad(l)),range(pad(h))):
                            real=ll<l and hh<h and q<n
                            selected=tuple(bool(real and positional(q,i,n,w)) for i in range(size))
                            key=(scores,selected,real)
                            expected_contexts+=1
                            M.counts['row/contexts']+=1
                            if key in cache:
                                delta,counters=cache[key]
                                M.cached(delta)
                                M.counts.update(counters)
                                M.counts['row/cached_contexts']+=1
                            else:
                                old=M.judgments;before=Counter(M.counts)
                                row_core(scores,selected,real)
                                counters={k:v-before[k] for k,v in M.counts.items() if k.startswith('row/') and v!=before[k]}
                                cache[key]=(M.judgments-old,counters)
                            if expected_contexts%4096==0:M.check()
            M.event('row_descriptor',{'N':size,'n':n,'W':w,'contexts':M.counts['row/contexts']-begin['row/contexts'],'cache_entries':len(cache),'judgments':M.judgments,'ops':M.ops})
    M.judge(expected_contexts==248850,'frozen pointwise context count',expected_contexts)
    M.maxima['pointwise_cache_entries']=len(cache)
    M.arm='normative_rows'
    for n,w,scores in itertools.product((1,3,4),(1,2,4,1024),((0,0,0,0),(-1,0,1,1),(1,1,0,-1),(1,-1,1,-1))):
        for q in range(4):
            real=q<n
            selected=[bool(real and positional(q,i,n,w)) for i in range(4)]
            maximum=max((x for x,s in zip(scores,selected) if s),default=0)
            e=[table16[isub(maximum,x)] if s else 0 for x,s in zip(scores,selected)]
            rr=idiv(1<<32,isum(e)) if real else 0
            for r in sorted({rr,max(0,rr-1),rr+1,0,(1<<17)-1}):
                expected=r==rr
                M.judge((row_reason(scores,selected,real,F(maximum),e,r,table16,16) is None)==expected,'normative reciprocal fixture',[n,w,q,scores,r,rr])
                M.counts['normative_row/probes']+=1
    for context,sigma in ((8,2),(8,16),(1<<17,16),(1<<17,23),(1<<17,24)):
        M.operation(3)
        guard=(2*context-1)*(1<<(2*sigma))<P
        M.judge(guard==(sigma!=24),'Goldilocks reciprocal guard',[context,sigma,guard])
        M.counts['Goldilocks/guards']+=1
    M.event('derivation',{'maximum':'Masked leaves1 and zero field product anchor c=x at a selected coordinate. Require true integer scores of span Sspan with Sspan+K0<P: delta-k is a multiple of P strictly between -P and P, hence delta=k>=0, so c is the selected maximum. No malicious-c integrality assumption. Spread<P alone is insufficient.','scope':'Finite domains and this stated algebraic implication; no protocol/range-proof or full-field exhaustion.'})


def poly_eval(coeffs, x):
    value=ZERO
    for c in reversed(coeffs):
        value=value*x+c
    return value


def interpolate4(values, x):
    # Newton differences; direct reference uses explicit coefficient convolution instead.
    y0,y1,y2,y3=values
    d1=y1-y0
    d2=y2-2*y1+y0
    d3=y3-3*y2+3*y1-y0
    return y0+x*d1+x*(x-ONE)*HALF*d2+x*(x-ONE)*(x-F(2))*INV6*d3


def reference_coefficients(factors):
    """Expand products of affine polynomials, then sum exact coefficients."""
    coeff=[ZERO]*4
    for i in range(0,len(factors[0]),2):
        product=[ONE]
        for table in factors:
            linear=[table[i],table[i+1]-table[i]]
            out=[ZERO]*(len(product)+1)
            for j,a in enumerate(product):
                for k,b in enumerate(linear):out[j+k]=out[j+k]+a*b
            product=out
        for j,v in enumerate(product):coeff[j]=coeff[j]+v
    return coeff


def round_grid(factors):
    out=[]
    for x in (ZERO,ONE,F(2),F(3)):
        value=ZERO
        for i in range(0,len(factors[0]),2):
            product=ONE
            for table in factors:
                product=product*(table[i]+x*(table[i+1]-table[i]))
            value=value+product
        out.append(value)
    return out


def round_claim_reason(values,incoming):
    return None if values[0]+values[1]==incoming else 'transport-target'


def initial_reason(factors,target):
    return round_claim_reason(round_grid(factors),target)


def sumcheck(factors, target, challenges, degrees, tag):
    M.judge(len(challenges)==len(degrees),'challenge/degree dimensions',tag)
    candidate=[[F(v) for v in t] for t in factors]
    reference=[list(t) for t in candidate]
    M.judge(all(len(t)==1<<len(challenges) for t in candidate),'factor table shape',tag)
    incoming=target
    ref_incoming=target
    for j,(challenge,degree) in enumerate(zip(challenges,degrees)):
        values=round_grid(candidate)
        coeff=reference_coefficients(reference)
        for x,v in enumerate(values):
            M.judge(v==poly_eval(coeff,F(x)),'round grid versus expanded reference',[tag,j,x])
        M.judge(all(c==ZERO for c in coeff[degree+1:]),'degree bound',[tag,j,degree,coeff])
        M.judge(round_claim_reason(values,incoming) is None,'round sum candidate',[tag,j])
        M.judge(poly_eval(coeff,ZERO)+poly_eval(coeff,ONE)==ref_incoming,'round sum reference',[tag,j])
        incoming=interpolate4(values,challenge)
        ref_incoming=poly_eval(coeff,challenge)
        M.judge(incoming==ref_incoming,'round challenge interpolation',[tag,j])
        candidate=[[t[i]+challenge*(t[i+1]-t[i]) for i in range(0,len(t),2)] for t in candidate]
        # Separately written direct two-point weighted sum, not the candidate recurrence.
        reference=[[(ONE-challenge)*t[i]+challenge*t[i+1] for i in range(0,len(t),2)] for t in reference]
        M.counts['transport/rounds']+=1
    M.judge(all(len(t)==1 for t in candidate),'terminal factor dimensions',tag)
    final=ONE
    for t in candidate:final=final*t[0]
    M.judge(final==incoming,'terminal product',tag)
    for a,b in zip(candidate,reference):M.judge(a==b,'terminal factor reference',tag)
    M.counts['transport/reductions']+=1
    return [t[0] for t in candidate]


def evaluate_eq(a,b):
    out=ONE
    for x,y in zip(a,b):out=out*(x*y+(ONE-x)*(ONE-y))
    return out


def consume(desc, claims, expected, tables):
    """Each oracle/identity/coordinate equality is a counted audit judgment."""
    if not M.observe(len(claims)==len(expected)):return 'claim-inventory'
    for supplied,want in zip(claims,expected):
        if not M.observe(set(supplied)=={'tensor','point','value','descriptor'}):return 'claim-inventory'
        ident=supplied['descriptor']
        if not M.observe(len(ident)==len(desc.identity)):return 'descriptor'
        for got,need in zip(ident,desc.identity):
            if not M.observe(got==need):return 'descriptor'
        if not M.observe(supplied['tensor']==want[0]):return 'claim-coordinate'
        if not M.observe(len(supplied['point'])==len(want[1])):return 'claim-coordinate'
        for got,need in zip(supplied['point'],want[1]):
            if not M.observe(got==need):return 'claim-coordinate'
        oracle=mle_ref(tables[supplied['tensor']],supplied['point'])
        if not M.observe(supplied['value']==oracle):return 'claim-value'
    return None


def claim(desc,name,pt,table):
    return {'tensor':name,'point':list(pt),'value':mle_fold(table,pt),'descriptor':desc.identity}


def transport_fixture(desc,family,table16):
    nsize=desc.N;hp=pad(desc.H);lp=pad(desc.L);db=desc.d
    hb=(hp-1).bit_length();lb=(lp-1).bit_length()
    def raw(l,h,p,c,offset):
        if l>=desc.L or h>=desc.H or p>=desc.n:return 0
        if family==0:return 0
        if family==1:return 1
        if family==2:return -1 if isum((l,h,p,c))%2 else 1
        z=isum((l,imul(2,h),imul(3,p),c,offset));M.operation()
        return isub(z%3,1)
    Q=[raw(l,h,p,c,0) for l in range(lp) for h in range(hp) for p in range(nsize) for c in range(2)]
    K=[raw(l,h,p,c,1) for l in range(lp) for h in range(hp) for p in range(nsize) for c in range(2)]
    V=[raw(l,h,p,c,2) for l in range(lp) for h in range(hp) for p in range(nsize) for c in range(2)]
    def operand(data,l,h,p,c):return data[c+2*(p+nsize*(h+hp*l))]
    x=[];ct=[];ee=[];pt=[];kap=[];leaves=[];yy=[]
    for l,h in itertools.product(range(lp),range(hp)):
        for q in range(nsize):
            scores=[isum(imul(operand(Q,l,h,q,c),operand(K,l,h,i,c)) for c in range(2)) for i in range(nsize)]
            real=l<desc.L and h<desc.H and q<desc.n
            selected=[bool(real and positional(q,i,desc.n,desc.W)) for i in range(nsize)]
            maximum=max((v for v,s in zip(scores,selected) if s),default=0)
            es=[table16[isub(maximum,v)] if selected[i] else 0 for i,v in enumerate(scores)]
            ct.append(maximum);x.extend(scores);ee.extend(es);pt.extend(map(int,selected))
            for i,v in enumerate(scores):
                kap.append(isub(maximum,v) if selected[i] else 15)
                leaves.append(isub(maximum,v) if selected[i] else 1)
            for c in range(2):
                full=isum(imul(es[i],operand(V,l,h,i,c)) for i in range(nsize))
                selected_only=isum(imul(es[i],operand(V,l,h,i,c)) for i in range(nsize) if selected[i])
                M.judge(full==selected_only,'full-domain PV equals selected-only', [desc,family,l,h,q,c])
                yy.append(full)
    tables={name:[F(z) for z in values] for name,values in
            [('Q',Q),('K',K),('V',V),('x',x),('c',ct),('e',ee),('P',pt),('kappa',kap),('leaf',leaves),('y',yy)]}
    return tables,hb,lb


def transport_audit(table16,controls,smoke=False):
    M.arm='transport'
    specimen=None
    fixtures=[(3,3,3,2,3)] if smoke else itertools.product((1,3),(1,3),(0,1,3,4),(1,2,4,1024),range(4))
    for l,h,n,w,family in fixtures:
        desc=Desc(2,n,w,l,h)
        tables,hb,lb=transport_fixture(desc,family,table16)
        hp,lp=pad(h),pad(l);bits=4+hb+lb
        for cat in range(8):
            r=catalog(cat,bits);s=catalog((cat+1)%8,bits)
            ri,rq,rh,rl=r[:2],r[2:4],r[4:4+hb],r[4+hb:]
            eqt=weights(r)
            ktable=[tables['c'][index//4]-v for index,v in enumerate(tables['x'])]
            pr=full_p(desc,r)
            directA=sum((eqt[j]*tables['P'][j]*ktable[j] for j in range(len(ktable))),ZERO)
            targets=[mle_ref(tables['kappa'],r)-F(15)*(ONE-pr),mle_ref(tables['leaf'],r)-ONE+pr]
            for label,target in zip(('key','max-leaf'),targets):
                M.judge(target==directA,'direct key/leaf identity',[desc,family,cat,label])
                final=sumcheck([eqt,tables['P'],ktable],target,s,[3]*bits,label)
                public=evaluate_eq(r,s)*full_p(desc,s)
                cclaim=claim(desc,'c',s[2:],tables['c'])
                xclaim=claim(desc,'x',s,tables['x'])
                M.judge(final[0]==evaluate_eq(r,s) and final[1]==full_p(desc,s),'terminal public factors',[desc,cat])
                M.judge(public*(cclaim['value']-xclaim['value'])==final[0]*final[1]*final[2],'terminal c/x split',[desc,cat])
                pool=[cclaim,xclaim]
                expected=[('c',s[2:]),('x',s)]
                M.judge(consume(desc,pool,expected,tables) is None,'consume c/x terminal claims',[desc,cat])
                if public!=ZERO and specimen is None and family==3 and l==3 and h==3 and n==3:
                    specimen=(desc,pool,expected,tables)
            # Dense x transport: sum over shared layer/head and channel, not target-point product.
            xp=catalog(cat,1+hb+lb)
            eqh=weights(rh+rl)
            qfactor=[];kfactor=[];efactor=[]
            for ll,hh,c in itertools.product(range(lp),range(hp),range(2)):
                qarr=[tables['Q'][c+2*(q+4*(hh+hp*ll))] for q in range(4)]
                karr=[tables['K'][c+2*(i+4*(hh+hp*ll))] for i in range(4)]
                qfactor.append(mle_ref(qarr,rq));kfactor.append(mle_ref(karr,ri));efactor.append(eqh[hh+hp*ll])
            xtarget=mle_ref(tables['x'],r)
            xf=sumcheck([efactor,qfactor,kfactor],xtarget,xp,[2]+[3]*(hb+lb),'dense-QK')
            qpt=xp[:1]+rq+xp[1:1+hb]+xp[1+hb:]
            kpt=xp[:1]+ri+xp[1:1+hb]+xp[1+hb:]
            qc=claim(desc,'Q',qpt,tables['Q']);kc=claim(desc,'K',kpt,tables['K'])
            M.judge(xf[0]*qc['value']*kc['value']==xf[0]*xf[1]*xf[2],'dense Q/K final transport')
            M.judge(consume(desc,[qc,kc],[('Q',qpt),('K',kpt)],tables) is None,'consume Q/K terminal claims')
            # PV target order: low channel, query, head, layer; key cube remains full.
            yr=catalog(cat,3+hb+lb);ra,yrq,yrh,yrl=yr[:1],yr[1:3],yr[3:3+hb],yr[3+hb:]
            yp=catalog((cat+2)%8,2+hb+lb)
            yw=weights(yrh+yrl);e_f=[];v_f=[];z_f=[]
            for ll,hh,i in itertools.product(range(lp),range(hp),range(4)):
                earr=[tables['e'][i+4*(q+4*(hh+hp*ll))] for q in range(4)]
                varr=[tables['V'][c+2*(i+4*(hh+hp*ll))] for c in range(2)]
                e_f.append(mle_ref(earr,yrq));v_f.append(mle_ref(varr,ra));z_f.append(yw[hh+hp*ll])
            ytarget=mle_ref(tables['y'],yr)
            yf=sumcheck([z_f,e_f,v_f],ytarget,yp,[2,2]+[3]*(hb+lb),'full-PV')
            ept=yp[:2]+yrq+yp[2:2+hb]+yp[2+hb:]
            vpt=ra+yp[:2]+yp[2:2+hb]+yp[2+hb:]
            ec=claim(desc,'e',ept,tables['e']);vc=claim(desc,'V',vpt,tables['V'])
            M.judge(yf[0]*ec['value']*vc['value']==yf[0]*yf[1]*yf[2],'PV e/V final transport')
            M.judge(consume(desc,[ec,vc],[('e',ept),('V',vpt)],tables) is None,'consume e/V terminal claims')
            # Save independent nonzero inventory specimens for every tensor, not just c/x.
            for tag,cl,exp in [('QK',[qc,kc],[('Q',qpt),('K',kpt)]),('PV',[ec,vc],[('e',ept),('V',vpt)])]:
                if family==3 and l==3 and h==3 and n==3 and (xf[0] if tag=='QK' else yf[0])!=ZERO and all(x['value']!=ZERO for x in cl) and tag not in controls['specimens']:
                    controls['specimens'][tag]=(desc,cl,exp,tables)
            M.counts['transport/runs']+=1
        M.event('transport_fixture',{'descriptor':desc,'family':family,'runs':M.counts['transport/runs'],'ops':M.ops})
    M.judge(M.counts['transport/runs']==(8 if smoke else 2048),'frozen transport run count')
    M.judge(specimen is not None,'nonzero c/x omission specimen')
    controls['specimens']['key']=specimen


def record_control(controls,name,good,mutated,detail):
    M.judge(good,'control unchanged passes',name)
    M.judge(mutated,'registered mutation/attack sensitivity',name)
    row={'name':name,'unchanged_control_passed':good,'sensitive':mutated,'fixture':clean(detail)}
    controls['records'].append(row)
    M.event('control',row)


def adversarial(controls,table16):
    M.arm='controls'
    for label,span,k0 in [('toy',2,3),('transport',4,15),('normative-fixtures',2,15),('RFC-score-bound',1<<22,(1<<32)-1)]:
        M.judge(iadd(span,k0)<P,'quantitative integer lift',[label,span,k0])
    S=idiv(isub(P,3),2)
    wrap_scores=[-S,S]
    wrapped=row_reason(wrap_scores,[True,True],True,F(-S),[4,0],4,[4,1,1,0],2)
    M.judge(wrapped is None,'outside-domain counterexample to spread-only lemma')
    M.judge(imul(2,S)<P and iadd(imul(2,S),3)==P and -S!=S,'outside-domain missing range condition')
    M.event('outside_domain_derivation_control',{'scores':wrap_scores,'c':-S,'keys':[0,3],'e':[4,0],'R':4,'a':0,'b':3,'relation_accepts':wrapped is None,'score_span':2*S,'K0':3,'span_plus_K0':P,'inside_frozen_score_alphabets':False,'refutes':'spread<P alone implies integer lift','required':'Sspan+K0<P or stronger Lemma0 quantitative bounds','source':'independent reviewer preflight hand counterexample; no production/spec bug asserted'})
    base=dict(format='e0-local-window/1',d=2,n=3,W=2,L=3,H=3)
    M.judge(descriptor(base)[0] is None,'descriptor positive')
    bads=[{},dict(base,extra=0),dict(base,format='other')]
    for k,v in [('d',-1),('d',4),('n',-1),('n',5),('W',0),('W',-1),('W',(1<<17)+1),('L',0),('L',4),('H',0),('H',4)]:bads.append(dict(base,**{k:v}))
    for k in ('d','n','W','L','H'):bads.append(dict(base,**{k:True}))
    for k in base:
        z=dict(base);del z[k];bads.append(z)
    bads.extend([dict(base,global_flag=True),dict(base,layer_windows=[1,2,3])])
    for fields in bads:
        record_control(controls,'descriptor',descriptor(base)[0] is None,descriptor(fields)[0]=='descriptor',fields)
    # Skipping validation admits W0 as an arbitrary tuple: explicitly a malformed descriptor.
    record_control(controls,'skip descriptor validation',descriptor(dict(base,W=0))[0]=='descriptor',Desc(2,3,0).W==0,dict(base,W=0))
    borrow_cases=[(0,1,2,2,0),(1,0,2,1,1),(1,1,1,1,2)]
    for q,i,n,w,drop in borrow_cases:
        rq,ri=[F(q)],[F(i)]
        record_control(controls,f'drop final borrow{drop}',dp(rq,ri,n,w)==ZERO,dp(rq,ri,n,w,omit=drop)==ONE,[q,i,n,w,drop])
    for d,n,w in ((0,1,1024),(1,2,2),(2,4,4),(2,4,1024),(3,8,1024)):
        for q in range(1<<d):
            for i in range(1<<d):
                got=dp([F((q>>b)&1) for b in range(d)],[F((i>>b)&1) for b in range(d)],n,w)
                M.judge(got==int(i<=q<n),'clamp/full-width edge',[d,n,w,q,i])
    M.judge(dp([HALF],[HALF],0,1)==ZERO,'n0 zero branch')
    desc=Desc(2,3,2,3,3)
    for axis in ('layer','head'):
        point=[ZERO]*8
        # Score order i2,q2,h2,l2; choose padded coordinate3 in the selected axis.
        off=6 if axis=='layer' else 4
        point[off:off+2]=[ONE,ONE]
        good=full_p(desc,point)
        wrong=dp(point[2:4],point[:2],3,2)  # missing the selected padded axis factor.
        record_control(controls,'omit '+axis+' LT',good==ZERO,wrong==ONE,point)
    # Complete local cheating fixtures, including reciprocal brackets.
    scores=[0,0];selected=[True,False];e=[4,1]
    record_control(controls,'excluded-e',row_reason(scores,selected,True,ZERO,e,3,[4,1,1,0],2)=='excluded-e',
                   row_reason(scores,selected,True,ZERO,e,3,[4,1,1,0],2,omit='excluded-e') is None,
                   {'scores':scores,'selected':selected,'c':0,'e':e,'R':3,'s':5,'a':1,'b':3,'V':[0,1],'wrong_PV':1,'oracle_PV':0})
    scores=[0]*4;selected=[True,True,True,False];e=[1,1,1,0]
    for mode in ('omit-root','masked-leaf-zero'):
        mutant=row_reason(scores,selected,True,ONE,e,5,[4,1,1,0],2,
                          omit='false-max' if mode=='omit-root' else None,masked_zero=mode=='masked-leaf-zero')
        record_control(controls,mode,row_reason(scores,selected,True,ONE,e,5,[4,1,1,0],2)=='false-max',mutant is None,
                       {'scores':scores,'selected':selected,'c':1,'e':e,'R':5,'s':3,'a':1,'b':1})
    for name,c,r in [('padding-c',ONE,0),('padding-R',ZERO,1)]:
        record_control(controls,name,row_reason([0]*4,[False]*4,False,c,[0]*4,r,[4,1,1,0],2)==name,
                       row_reason([0]*4,[False]*4,False,c,[0]*4,r,[4,1,1,0],2,omit=name) is None,[c,r])
    # Changed-D control: key0 is excluded from row1 by W1, though its dense exponential is positive.
    record_control(controls,'original-D difference',True,4*1+4*0 != 0*1+4*0,
                   {'N':2,'n':2,'W':1,'q':1,'scores':[0,0],'dense_e':[4,4],'window_e':[0,4],'V':[1,0],'dense_PV':4,'window_PV':0})
    for tag,(desc,cl,expected,tables) in controls['specimens'].items():
        M.judge(consume(desc,cl,expected,tables) is None,'inventory no-op',tag)
        for j,c in enumerate(cl):
            record_control(controls,'omit terminal '+c['tensor'],True,consume(desc,cl[:j]+cl[j+1:],expected,tables)=='claim-inventory',{'tag':tag,'claims':cl})
            altered=[dict(x) for x in cl];altered[j]['value']=altered[j]['value']+ONE
            record_control(controls,'alter terminal '+c['tensor'],True,consume(desc,altered,expected,tables)=='claim-value',{'tag':tag,'index':j,'claim':c})
            altered=[dict(x) for x in cl];altered[j]['descriptor']=Desc(desc.d,desc.n,desc.W+1,desc.L,desc.H).identity
            record_control(controls,'mismatched descriptor '+c['tensor'],True,consume(desc,altered,expected,tables)=='descriptor',{'tag':tag,'index':j})
            # Coordinate identity is part of the audit even if a particular constant table hides a swap.
            altered=[dict(x) for x in cl];altered[j]['point']=list(c['point']);altered[j]['point'][0]=altered[j]['point'][0]+ONE
            record_control(controls,'alter coordinate '+c['tensor'],True,consume(desc,altered,expected,tables)=='claim-coordinate',{'tag':tag,'index':j})
    # Direct target and axis/padding omission controls use nontrivial exact tables, independent of supplied claims.
    desc=Desc(2,3,2,3,3);tables,hb,lb=transport_fixture(desc,3,table16)
    r=catalog(7,8)
    for name in ('kappa','leaf'):
        correct=mle_ref(tables[name],r)
        selector=full_p(desc,r)
        target=correct-F(15)*(ONE-selector) if name=='kappa' else correct-ONE+selector
        kt=[tables['c'][j//4]-x for j,x in enumerate(tables['x'])]
        grid=round_grid([weights(r),tables['P'],kt])
        record_control(controls,'alter '+name+' target',round_claim_reason(grid,target) is None,round_claim_reason(grid,target+ONE)=='transport-target',{'target':target,'reason':'transport-target'})
    swapped=r[2:4]+r[:2]+r[4:]
    cl=claim(desc,'x',r,tables['x']);bad=dict(cl,point=swapped)
    record_control(controls,'swap q/i',consume(desc,[cl],[('x',r)],tables) is None,consume(desc,[bad],[('x',r)],tables)=='claim-coordinate',{'point':r,'swapped':swapped,'reason':'claim-coordinate'})
    for axis,start in [('head',4),('layer',6)]:
        changed=list(r);changed[start:start+2]=[ONE-r[start],r[start+1]]
        cl=claim(desc,'x',r,tables['x']);bad=dict(cl,point=changed)
        record_control(controls,'swap '+axis+' identity',consume(desc,[cl],[('x',r)],tables) is None,consume(desc,[bad],[('x',r)],tables)=='claim-coordinate',{'point':r,'changed':changed,'reason':'claim-coordinate'})
    selected=[0,0,4,0];value=[1,2,3,4];full=isum(imul(a,b) for a,b in zip(selected,value));short=imul(4,value[0])
    factors=[[F(v) for v in selected],[F(v) for v in value],[ONE]*4]
    record_control(controls,'renumber compacted e domain',initial_reason(factors,F(full)) is None,initial_reason(factors,F(short))=='transport-target',{'logical_e':selected,'V':value,'full':full,'wrong_shortened':short,'reason':'transport-target'})


def arithmetic_controls():
    M.arm='arithmetic'
    M.judge(U*U==F(7),'Fp2 defining relation')
    M.judge(F(6)*INV6==ONE and HALF+HALF==ONE,'fixed rational embeddings')
    rng=splitmix(SEED)
    for _ in range(32):
        a,b,c=(F(next(rng),next(rng)) for _ in range(3))
        M.judge((a+b)*c==a*c+b*c,'Fp2 distributive')
        M.judge(a*ONE==a and a+ZERO==a and a-a==ZERO,'Fp2 identities')
        product=a*b
        # Independent four-integer coefficient expansion, using a three-product identity.
        aa=a.a*b.a % P;bb=a.b*b.b % P
        cross=((a.a+a.b)*(b.a+b.b)-aa-bb)%P
        M.operation(9)
        M.judge(product==F((aa+7*bb)%P,cross),'Fp2 alternate product')
    for i in range(-4,5):
        M.judge(F(i)*HALF==F((i*((P+1)//2))%P),'rational half embedding')


def preflight(table16,controls):
    arithmetic_controls()
    M.arm='preflight'
    for d in range(4):
        for n,w in itertools.product(range((1<<d)+1),windows(1<<d)):
            point=[HALF]*(2*d)
            desc=Desc(d,n,w)
            M.judge(dp(point[d:],point[:d],n,w)==mle_ref(positional_table(desc),point),'half-point preflight',[d,n,w])
    # Explicit nonzero claims exercise the inventory code without claiming the full transport arm.
    desc=Desc(2,3,2,3,3);tables,_,_=transport_fixture(desc,3,table16)
    for tag,names in [('key',('c','x')),('QK',('Q','K')),('PV',('e','V'))]:
        pool=[];expected=[]
        for name in names:
            bits=(len(tables[name])-1).bit_length();pt=catalog(7,bits)
            pool.append(claim(desc,name,pt,tables[name]));expected.append((name,pt))
        controls['specimens'][tag]=(desc,pool,expected,tables)
    transport_audit(table16,controls,smoke=True)
    adversarial(controls,table16)


def verify_registration():
    path=HERE/'results/README.preregistered.txt'
    data=path.read_bytes()
    if hashlib.sha256(data).hexdigest()!=FROZEN:raise AuditFailure('frozen file differs')
    relative=str(path.relative_to(REPO))
    committed=subprocess.check_output(['git','show',f'HEAD:{relative}'],cwd=REPO)
    if committed!=data:raise AuditFailure('pre-registration not identical in HEAD')
    pins=json.loads((HERE/'results/source-pins.json').read_text())
    drift=[]
    for f,h in pins['sources_sha256'].items():
        if hashlib.sha256((REPO/f).read_bytes()).hexdigest()!=h:drift.append(f)
    # Idea prose may change only after judgment, so report that drift separately on reproduction.
    semantic=[x for x in drift if not x.startswith('research/ideas/')]
    if semantic:raise AuditFailure('normative source drift: '+repr(semantic))
    return {'frozen_sha256':FROZEN,'first_commit':COMMIT,'source_drift':drift,
            'head':subprocess.check_output(['git','rev-parse','HEAD'],cwd=REPO,text=True).strip(),
            'script_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--preflight',action='store_true',help='small implementation controls only; no full-domain verdict')
    args=parser.parse_args()
    attempt=datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S.%fZ')+'-'+str(os.getpid())
    out=HERE/'results'/'attempts'/attempt
    out.mkdir(parents=True,exist_ok=False)
    (out/'audit.source.py').write_bytes(Path(__file__).read_bytes())
    controls={'records':[],'specimens':{}}
    code=0;verdict='inconclusive';error=None;completed=[]
    def alarm(_signum,_frame):raise AuditLimit('15 minute process watchdog')
    signal.signal(signal.SIGALRM,alarm)
    signal.alarm(900)
    with (out/'events.jsonl').open('x') as stream:
        M.stream=stream
        env={'attempt':attempt,'preflight_only':args.preflight,'utc':datetime.now(timezone.utc).isoformat(),
             'python':sys.version,'platform':platform.platform(),'machine':platform.machine(),'cpu':platform.processor(),
             'pid':os.getpid(),'scope':'E0 finite relation audit; no benchmark or full protocol/ZK/fidelity approval'}
        try:
            env.update(verify_registration());M.event('environment',env)
            table16=normative_table()
            if args.preflight:
                preflight(table16,controls);completed=['preflight controls only']
            else:
                arithmetic_controls();completed.append('arithmetic')
                selector_audit();completed.append('selector')
                pointwise_audit(table16);completed.append('pointwise and normative rows')
                transport_audit(table16,controls);completed.append('transport')
                adversarial(controls,table16);completed.append('adversarial controls')
                verdict='validated'
        except AuditLimit as exc:
            error=str(exc);code=2;M.event('resource_limit',error)
        except AuditFailure as exc:
            error=str(exc);code=1;verdict='inconclusive' if args.preflight else 'refuted';M.event('audit_failure',error)
        except Exception as exc:  # Preserve every failed implementation attempt.
            import traceback
            error=traceback.format_exc();code=3;M.event('implementation_error',error)
        finally:
            signal.alarm(0)
            summary={'environment':env,'verdict':verdict,'exit_code':code,'error':error,'completed_arms':completed,
                     'counts':dict(M.counts),'judged_comparisons':M.judgments,'physical_judgments':M.physical,
                     'primitive_exact_operations':M.ops,'maxima':dict(M.maxima),'controls':len(controls['records']),
                     'finite_scope_only':True,'no_performance_claim':True,
                     'row_cache_scope':'row_core depends only on immutable scores, selected bits and real-row flag; sigma2/table/field are fixed for this arm. Layer/head/query/window affect it only through selected and real. Logical replay is not a new physical test.',
                     'limits':{'comparisons':100000000,'operations':3000000000,'watchdog_seconds':900,'whole_process_memory_bytes':256*1024*1024}}
            (out/'summary.json').write_text(json.dumps(clean(summary),indent=2)+'\n')
            (out/'controls.json').write_text(json.dumps(controls['records'],indent=2)+'\n')
            (out/'counterexamples.json').write_text(json.dumps(M.failures,indent=2)+'\n')
            (out/'environment.json').write_text(json.dumps(env,indent=2)+'\n')
            # Unique attempt paths never overwrite earlier records.
            print(json.dumps({'attempt':str(out),'verdict':verdict,'exit_code':code,'error':error,
                              'completed':completed,'comparisons':M.judgments,'ops':M.ops},indent=2),flush=True)
    return code


if __name__=='__main__':
    raise SystemExit(main())
