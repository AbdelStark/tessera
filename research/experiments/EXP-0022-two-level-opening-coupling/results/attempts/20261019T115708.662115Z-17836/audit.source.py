#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0 OR MIT
"""EXP-0022 conditional finite audit; not a cryptographic proof verifier.

The candidate never receives complete U or the row-mask oracle. Registry
membership and exact terminal evaluation are explicitly ideal assumptions.
"""
from __future__ import annotations

import argparse
from collections import Counter
import copy
from dataclasses import dataclass, fields
from datetime import datetime, timezone
from fractions import Fraction
import hashlib
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

sys.dont_write_bytecode = True
HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
P = (1 << 64) - (1 << 32) + 1
COMMIT = "f344d76"
FROZEN = "6676608fac008fc20b9bb7594dbe6463acb54f5bdc0fe2f9a1cb4a20fb22fd24"
SEED = 0x0022003500000001
SESSIONS = itertools.count(1)  # local finite-model identities, not cryptographic randomness


class Failure(Exception):
    pass


class Limit(Exception):
    pass


class Reject(Exception):
    def __init__(self, reason):
        self.reason = reason
        super().__init__(reason)


class Meter:
    def __init__(self):
        self.start = time.monotonic()
        self.ops = 0
        self.judgments = 0
        self.counts = Counter()
        self.maxima = Counter()
        self.arm = "setup"
        self.stream = None
        self.next_check = 65536

    def op(self, n=1):
        self.ops += n
        if self.ops >= self.next_check:
            self.next_check = self.ops + 65536
            self.check()

    def check(self):
        if self.ops > 3_000_000_000 or self.judgments > 100_000_000:
            raise Limit("operation/judgment cap")
        if time.monotonic() - self.start > 900:
            raise Limit("15-minute watchdog")
        rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        if sys.platform != "darwin":
            rss *= 1024
        self.maxima["rss_bytes"] = max(self.maxima["rss_bytes"], rss)
        if rss > 256 * 1024 * 1024:
            raise Limit("256 MiB process RSS")

    def observe(self, condition):
        self.judgments += 1
        self.counts["judgments/" + self.arm] += 1
        if self.judgments % 65536 == 0:
            self.check()
        return bool(condition)

    def judge(self, condition, name, detail=None):
        if not self.observe(condition):
            self.event("failure", {"name": name, "detail": detail})
            raise Failure(name)

    def event(self, kind, value):
        if self.stream:
            self.stream.write(json.dumps({"kind": kind, "data": clean(value)}, sort_keys=True) + "\n")
            self.stream.flush()


M = Meter()


class F:
    __slots__ = ("a", "b")

    def __init__(self, a=0, b=0):
        aa,bb=(a.a,a.b) if isinstance(a,F) else (int(a)%P,int(b)%P)
        object.__setattr__(self,"a",aa)
        object.__setattr__(self,"b",bb)

    def __setattr__(self, name, value):
        raise AttributeError("immutable exact scalar")

    def __deepcopy__(self, memo):
        # Scalars are immutable; registry snapshots cannot alias mutable field words.
        return self

    def __add__(self, v):
        v = F(v); M.op(2)
        return F(self.a + v.a, self.b + v.b)

    __radd__ = __add__

    def __sub__(self, v):
        v = F(v); M.op(2)
        return F(self.a - v.a, self.b - v.b)

    def __rsub__(self, v):
        return F(v) - self

    def __neg__(self):
        M.op(2); return F(-self.a, -self.b)

    def __mul__(self, v):
        v = F(v); M.op(7)
        return F(self.a * v.a + 7 * self.b * v.b, self.a * v.b + self.b * v.a)

    __rmul__ = __mul__

    def __pow__(self, n):
        assert type(n) is int and n >= 0
        x, y = F(self), F(1)
        while n:
            if n & 1:
                y = y * x
            n >>= 1
            if n:
                x = x * x
        return y

    def inv(self):
        if self == 0:
            raise Reject("zero-inverse")
        M.op(4)
        norm = (self.a * self.a - 7 * self.b * self.b) % P
        inv = basepow(norm, P - 2)
        M.op(3)
        return F(self.a * inv, -self.b * inv)

    def __truediv__(self, v):
        return self * F(v).inv()

    def __eq__(self, v):
        if isinstance(v, (int, F)):
            v = F(v)
            return self.a == v.a and self.b == v.b
        return False

    def __hash__(self):
        return hash((self.a, self.b))

    def __repr__(self):
        return f"F({self.a},{self.b})"


ZERO, ONE, U = F(), F(1), F(0, 1)
HALF = F((P + 1) // 2)


def basepow(x, n):
    y = 1
    while n:
        if n & 1:
            M.op(); y = y * x % P
        n >>= 1
        if n:
            M.op(); x = x * x % P
    return y


def pairadd(x, y):
    M.op(2)
    return ((x[0] + y[0]) % P, (x[1] + y[1]) % P)


def pairmul(x, y):
    # Independent pair oracle, not a call through F operators.
    M.op(7)
    ac, bd, ad, bc = x[0]*y[0], x[1]*y[1], x[0]*y[1], x[1]*y[0]
    return ((ac + 7*bd) % P, (ad + bc) % P)


def clean(x):
    if isinstance(x, F):
        return [x.a, x.b]
    if isinstance(x, Fraction):
        return str(x)
    if hasattr(x, "__dataclass_fields__"):
        return {v.name: clean(getattr(x, v.name)) for v in fields(x)}
    if isinstance(x, dict):
        return {str(k): clean(v) for k, v in x.items()}
    if isinstance(x, (tuple, list)):
        return [clean(v) for v in x]
    return x


def digest(x):
    return hashlib.sha256(json.dumps(clean(x), sort_keys=True).encode()).hexdigest()


def require(cond, reason):
    if not M.observe(cond):
        raise Reject(reason)


def same_nested(a, b, reason):
    """Count each logical scalar plus each sequence/schema-length assertion.

    Fp2 is one semantic scalar. Tuple/list storage is not a wire encoding here.
    Dictionary key identity is one shape assertion; its actual values are checked
    recursively rather than counted as one opaque nested equality.
    """
    if isinstance(a, (tuple, list)) or isinstance(b, (tuple, list)):
        require(isinstance(a, (tuple, list)) and isinstance(b, (tuple, list)), reason)
        require(len(a) == len(b), reason)
        for x, y in zip(a, b):
            same_nested(x, y, reason)
    elif isinstance(a, dict) or isinstance(b, dict):
        require(isinstance(a, dict) and isinstance(b, dict), reason)
        require(set(a) == set(b), reason)
        for key in sorted(a):
            same_nested(a[key], b[key], reason)
    elif hasattr(a, "__dataclass_fields__") or hasattr(b, "__dataclass_fields__"):
        require(type(a) is type(b), reason)
        for field in fields(a):
            same_nested(getattr(a, field.name), getattr(b, field.name), reason)
    else:
        require(a == b, reason)


def same_vector(a, b, reason):
    require(len(a) == len(b), reason + "/length")
    for x, y in zip(a, b):
        same_nested(x, y, reason)


def pow2(n):
    return 1 << (max(1, n) - 1).bit_length()


def bits(n):
    assert n > 0 and n & (n - 1) == 0
    return n.bit_length() - 1


def eq_boolean(point, index):
    ans = ONE
    for b, r in enumerate(point):
        ans = ans * (r if (index >> b) & 1 else ONE - r)
    return ans


def mle_ref(table, point):
    require(len(table) == 1 << len(point), "reference-table-shape")
    ans = ZERO
    for index, value in enumerate(table):
        ans = ans + value * eq_boolean(point, index)
    return ans


def mle_fold(table, point):
    require(len(table) == 1 << len(point), "fold-table-shape")
    row = list(table)
    for r in point:
        row = [a + (b-a)*r for a, b in zip(row[::2], row[1::2])]
    require(len(row) == 1, "fold-terminal-shape")
    return row[0]


def dot(a, b):
    require(len(a) == len(b), "dot-shape")
    value = ZERO
    for x, y in zip(a, b):
        value = value + x*y
    return value


def poly_eval(coeffs, x):
    ans = ZERO
    for c in reversed(coeffs):
        ans = ans*x + c
    return ans


def catalog(cat, length, offset=0):
    if cat == 0:
        return (ZERO,) * length
    if cat == 1:
        return (ONE,) * length
    if cat == 2:
        return tuple(F((b+offset) % 2) for b in range(length))
    choices = (HALF, U, ONE+U, F(2)-U)
    return tuple(choices[(b+offset) % 4] for b in range(length))


def draws(cat, n):
    return ((0,), (0, 0), (n-1, 0, n-1), tuple(range(n)))[cat]


def gamma(cat):
    return (ZERO, ONE, F(2), ONE+U)[cat]


def splitmix(index):
    # PRNG bit/machine-word operations are separately declared fixture work.
    M.counts["fixture_prng_words"] += 1
    x = (SEED + 0x9E3779B97F4A7C15 * (index+1)) & ((1 << 64)-1)
    x = ((x ^ (x >> 30)) * 0xBF58476D1CE4E5B9) & ((1 << 64)-1)
    x = ((x ^ (x >> 27)) * 0x94D049BB133111EB) & ((1 << 64)-1)
    return x ^ (x >> 31)


def fixture_value(family, namespace, row, column):
    if family == 0:
        return ZERO
    if family == 1:
        return ONE
    # Namespace/index arithmetic selects an input. The value-generating
    # remainder/subtraction and sign negation are metered algebraic work.
    index = namespace * 4096 + row * 64 + column
    if family == 2:
        M.op()
        return -ONE if index % 2 else ONE
    M.op(2)
    return F(splitmix(index) % 5 - 2)


@dataclass(frozen=True)
class Desc:
    k1: int
    D: int
    G: int
    mode: str
    P1: int
    k2: int
    P2: int
    format: str = "e22-two-level/1"
    basis: str = "coeff-natural/experimental"
    lanes: tuple = (0, 1)

    @property
    def m(self):
        return self.G + 1

    @property
    def groups(self):
        return pow2(self.m)

    @property
    def identity(self):
        return tuple(getattr(self, f.name) for f in fields(self))


def parse_desc(record):
    expected = {f.name for f in fields(Desc)}
    require(type(record) is dict and set(record) == expected, "descriptor/fields")
    for name, domain in (("k1", (1,2,4)), ("D", (1,3)), ("G", (0,1,2)),
                         ("P1", (0,1)), ("k2", (2,4)), ("P2", (0,1))):
        require(type(record[name]) is int and record[name] in domain, "descriptor/"+name)
    require(record["mode"] in ("plain", "blinded-algebra"), "descriptor/mode")
    require(record["format"] == "e22-two-level/1", "descriptor/format")
    require(record["basis"] == "coeff-natural/experimental", "descriptor/basis")
    require(type(record["lanes"]) is tuple and len(record["lanes"]) == 2 and
            all(type(v) is int for v in record["lanes"]) and record["lanes"] == (0,1),
            "descriptor/lanes")
    return Desc(**record)


def encoded(rows, masks, k):
    n = 2*k
    omega = basepow(7, (P-1)//n)
    points = tuple(F(7) * F(omega)**j for j in range(n))
    columns = []
    for x in points:
        columns.append(tuple(poly_eval(row,x) + (x**k)*poly_eval(mask,x)
                             for row,mask in zip(rows,masks)))
    return points, tuple(columns)


class IdealRegistry:
    """Immutable ideal association: root -> tensor, masks, row map, columns.

    The candidate has no read-tensor/read-mask/honest-combination method.
    Registry methods never compare a committed U/V with A-derived U.
    """
    def __init__(self):
        self.__records = {}

    def commit(self, descriptor, tensor, rows, masks, physical, k):
        tensor = tuple(tensor)
        rows, masks = tuple(map(tuple,rows)), tuple(map(tuple,masks))
        require(type(descriptor) is tuple and len(descriptor)>0,"registry/descriptor")
        if descriptor[0]=="original":
            require(len(descriptor)==9,"registry/descriptor")
            _,width,data,groups,mode,mask_width,family,cat,session=descriptor
            require(type(width) is int and width in (1,2,4),"registry/width")
            require(type(data) is int and data in (1,3),"registry/data-rows")
            require(type(groups) is int and groups in (0,1,2),"registry/groups")
            require(mode in ("plain","blinded-algebra"),"registry/mode")
            require(type(mask_width) is int and mask_width in (0,1),"registry/mask-width")
            require(type(family) is int and family in range(4) and type(cat) is int and cat in range(4),
                    "registry/fixture")
            require(type(session) is str and len(session)>0,"registry/session")
            count=pow2(data)+(2*(groups+1) if mode=="blinded-algebra" else 0)
            allocated=pow2(count)
            expected_physical=tuple(range(data))+(tuple(range(pow2(data),count)) if mode=="blinded-algebra" else ())
        else:
            require(descriptor[0]=="combined-U" and len(descriptor)==5,"registry/descriptor")
            identity=descriptor[1]
            require(type(identity) is tuple and len(identity)==len(fields(Desc)),"registry/descriptor")
            spec=parse_desc(dict(zip((f.name for f in fields(Desc)),identity)))
            require(any(rec[0]==descriptor[2] for rec in self.__records.values()),"registry/original-association")
            width,mask_width=spec.k2,spec.P2
            allocated=max(2*spec.groups*spec.k1,spec.k2)//spec.k2
            expected_physical=tuple(r for r in range(allocated) if r*width<2*spec.m*spec.k1)
        require(type(k) is int and k==width,"registry/width")
        require(len(rows)==allocated,"registry/row-count")
        require(len(tensor)==allocated*width,"registry/tensor-length")
        require(all(type(v) is F and v.b == 0 for v in tensor), "registry/nonbase-coefficient")
        require(len(rows) == len(masks), "registry/row-mask-length")
        for mask in masks:
            require(len(mask)==mask_width,"registry/mask-width")
            for value in mask:require(type(value) is F and value.b==0,"registry/nonbase-mask")
        require(all(type(r) is int and 0<=r<allocated for r in physical),"registry/physical-index")
        require(tuple(physical)==expected_physical,"registry/row-map")
        require(tuple(sorted(set(physical))) == tuple(physical), "registry/physical-order")
        require(all(len(row) == k for row in rows), "registry/row-width")
        require(tuple(v for row in rows for v in row) == tensor, "registry/tensor-row-map")
        present = set(physical)
        require(all(all(v == 0 for v in rows[r]) and all(v == 0 for v in masks[r])
                    for r in range(len(rows)) if r not in present), "registry/public-zero")
        points, columns = encoded(rows,masks,k)
        root = len(self.__records) + 1
        self.__records[root] = (descriptor,tensor,rows,masks,tuple(physical),points,columns)
        M.counts["registry_commits"] += 1
        return root, points, columns

    def verify_columns(self, root, descriptor, draw, response):
        require(root in self.__records, "membership/root")
        registered, _, _, _, physical, points, columns = self.__records[root]
        require(registered == descriptor, "membership/descriptor")
        require(len(draw) > 0 and all(type(j) is int and 0 <= j < len(points) for j in draw),
                "membership/query")
        indices = tuple(sorted(set(draw)))
        require(tuple(j for j,_ in response) == indices, "membership/column-order")
        for j, symbols in response:
            same_vector(symbols,tuple(columns[j][r] for r in physical),"membership/symbol")
        return True

    def verify_mle(self, root, descriptor, point, value):
        require(root in self.__records, "opening/root")
        registered, tensor, *_ = self.__records[root]
        require(registered == descriptor, "opening/descriptor")
        require(len(tensor) == 1 << len(point), "opening/point-length")
        require(mle_ref(tensor,point) == value, "opening/value")
        return True


@dataclass
class Original:
    k: int
    D: int
    G: int
    mode: str
    P1: int
    family: int
    cat: int
    data_cube: int
    rows: tuple
    masks: tuple
    physical: tuple
    coefficients: tuple
    values: tuple
    q: tuple
    points: tuple
    Urows: tuple
    combined_masks: tuple
    row_ownership: tuple


def original_fixture(k,D,G,mode,P1,family,cat):
    data_cube = pow2(D)
    m = G+1
    count = data_cube + (2*m if mode == "blinded-algebra" else 0)
    allocated = pow2(count)
    physical = tuple(range(D)) + (tuple(range(data_cube,count)) if mode == "blinded-algebra" else ())
    rows = tuple(tuple(fixture_value(family,10,r,i) if r in physical else ZERO for i in range(k))
                 for r in range(allocated))
    masks = tuple(tuple(fixture_value(family,20,r,j) if r in physical else ZERO for j in range(P1))
                  for r in range(allocated))
    coeffs,values,q,points,Urows,combined,ownership = [],[],[],[],[],[],[]
    for g in range(m):
        p = catalog(cat,bits(k),g+1)
        c = [ZERO]*allocated
        vals = []
        if g < G:
            for t in range(2):
                rp = catalog(cat,bits(data_cube),2*g+t)
                value = mle_ref(tuple(v for row in rows[:data_cube] for v in row),p+rp)
                vals.append(value)
                power = gamma(cat)**(t+1)
                for r in range(data_cube):
                    c[r] = c[r] + power*eq_boolean(rp,r)
        else:
            for r in range(data_cube):
                c[r] = eq_boolean(catalog(cat,bits(data_cube)),r)
        qg = ZERO
        if mode == "blinded-algebra":
            b0,b1 = data_cube+2*g,data_cube+2*g+1
            c[b0],c[b1] = ONE,U
            ownership.append((g,b0,b1))
            if g < G:
                qg = mle_ref(tuple(a+U*b for a,b in zip(rows[b0],rows[b1])),p)
        urow = tuple(sum((c[r]*rows[r][i] for r in range(allocated)),ZERO) for i in range(k))
        mg = tuple(sum((c[r]*masks[r][j] for r in range(allocated)),ZERO) for j in range(P1))
        coeffs.append(tuple(c)); values.append(tuple(vals)); q.append(qg); points.append(p)
        Urows.append(urow); combined.append(mg)
    return Original(k,D,G,mode,P1,family,cat,data_cube,rows,masks,physical,
                    tuple(coeffs),tuple(values),tuple(q),tuple(points),tuple(Urows),tuple(combined),tuple(ownership))


@dataclass(frozen=True)
class Equation:
    kind: str
    group: int
    index: int
    ordinal: int
    parameter: tuple | F
    target: F
    identity: tuple


def equation_weights(eq,k):
    if eq.kind == "evaluation":
        return tuple(eq_boolean(eq.parameter,i) for i in range(k))
    return tuple(eq.parameter**i for i in range(k))


def public_coefficients(desc,cat):
    # Derived only from descriptor/points/challenges, never A/U or row masks.
    data = pow2(desc.D)
    allocated = pow2(data+(2*desc.m if desc.mode == "blinded-algebra" else 0))
    result=[]
    for g in range(desc.m):
        c=[]
        for r in range(allocated):
            value=ZERO
            if r < data:
                if g < desc.G:
                    for t in range(2):
                        value=value+(gamma(cat)**(t+1))*eq_boolean(catalog(cat,bits(data),2*g+t),r)
                else:
                    value=eq_boolean(catalog(cat,bits(data)),r)
            if desc.mode == "blinded-algebra":
                if r == data+2*g:value=ONE
                if r == data+2*g+1:value=U
            c.append(value)
        result.append(tuple(c))
    return tuple(result)


def derive_equations(source,desc,cat,registry,skip_membership=False,phi_old=False,allow_plain_q=False):
    require(source["identity"] == desc.identity,"source/descriptor")
    require(len(source["values"]) == desc.G,"source/claim-count")
    require(len(source["q"]) == desc.G,"source/q-count")
    if desc.mode=="plain" and not allow_plain_q:
        for value in source["q"]:require(value==ZERO,"source/plain-q")
    omega=F(basepow(7,(P-1)//(2*desc.k1)))
    expected_points=tuple(F(7)*(omega**j) for j in range(2*desc.k1))
    same_vector(source["column_points"],expected_points,"source/column-point")
    require(len(source["M"]) == desc.m,"source/mask-count")
    for row in source["M"]:
        require(len(row)==desc.P1,"source/mask-width")
    expected_physical=tuple(range(desc.D))
    if desc.mode=="blinded-algebra":
        expected_physical+=tuple(range(pow2(desc.D),pow2(desc.D)+2*desc.m))
    require(source["physical"]==expected_physical,"source/row-map")
    for _,symbols in source["columns"]:
        require(len(symbols)==len(expected_physical),"source/column-symbol-count")
    coefficients=public_coefficients(desc,cat)
    same_vector(source["ownership"],tuple((g,pow2(desc.D)+2*g,pow2(desc.D)+2*g+1)
                for g in range(desc.m)) if desc.mode=="blinded-algebra" else (),"source/blinding-ownership")
    if not skip_membership:
        registry.verify_columns(source["root"],source["original_descriptor"],source["draw"],source["columns"])
    same_nested(source["draw"],draws(cat,2*desc.k1),"source/query-draw")
    equations=[]
    for g in range(desc.G):
        require(len(source["values"][g])==2,"source/claim-length")
        target=source["q"][g]
        for t,v in enumerate(source["values"][g]):
            target=target+(gamma(cat)**(t+1))*v
        equations.append(Equation("evaluation",g,-1,len(equations),catalog(cat,bits(desc.k1),g+1),target,desc.identity))
    physical=source["physical"]
    expected=tuple(range(desc.D))
    if desc.mode=="blinded-algebra":
        expected+=tuple(range(pow2(desc.D),pow2(desc.D)+2*desc.m))
    same_nested(physical,expected,"source/row-map")
    for j,symbols in source["columns"]:
        x=source["column_points"][j]
        for g in range(desc.m):
            target=sum((coefficients[g][r]*v for r,v in zip(physical,symbols)),ZERO)
            phi=x**desc.k1
            if phi_old:phi=phi-ONE
            target=target-phi*poly_eval(source["M"][g],x)
            equations.append(Equation("column",g,j,len(equations),x,target,desc.identity))
    return tuple(equations)


def inventory(received,expected,skip=False):
    if skip:return
    require(len(received)==len(expected),"equation-inventory/length")
    for a,b in zip(received,expected):
        same_nested((a.kind,a.group,a.index,a.ordinal,a.parameter,a.identity),
                    (b.kind,b.group,b.index,b.ordinal,b.parameter,b.identity),"equation-inventory/metadata")
        require(a.target==b.target,"equation-inventory/target-from-source")


def weight_table(equations,alpha,desc):
    table=[ZERO]*(desc.groups*desc.k1)
    for eq in equations:
        lam=alpha**eq.ordinal
        for i,w in enumerate(equation_weights(eq,desc.k1)):
            at=eq.group*desc.k1+i
            table[at]=table[at]+lam*w
    return tuple(table)


def weight_factored(equations,alpha,desc,point,wrong_exponent=False):
    require(len(point)==bits(desc.groups*desc.k1),"weight/point-length")
    si,sg=point[:bits(desc.k1)],point[bits(desc.k1):]
    value=ZERO
    for eq in equations:
        w=ONE
        if eq.kind=="evaluation":
            for p,s in zip(eq.parameter,si):
                w=w*((ONE-p)*(ONE-s)+p*s)
        else:
            for b,s in enumerate(si):
                exponent=(1<<b)+(1 if wrong_exponent else 0)
                w=w*(ONE-s+s*(eq.parameter**exponent))
        value=value+(alpha**eq.ordinal)*eq_boolean(sg,eq.group)*w
    return value


def ref_residuals(actual_u,equations,desc):
    return tuple(dot(actual_u[e.group],equation_weights(e,desc.k1))-e.target for e in equations)


def round_direct(a,b):
    # Reference affine-coefficient convolution. Does not interpolate samples.
    coeff=[ZERO,ZERO,ZERO]
    for a0,a1,b0,b1 in zip(a[::2],a[1::2],b[::2],b[1::2]):
        da,db=a1-a0,b1-b0
        coeff[0]=coeff[0]+a0*b0
        coeff[1]=coeff[1]+a0*db+da*b0
        coeff[2]=coeff[2]+da*db
    return tuple(coeff)


def make_sumcheck(actual_u,W,point):
    # Prover: arbitrary committed U is legal input; no A-derived U check.
    a=list(actual_u);b=list(W);rounds=[]
    require(len(a)==len(b)==1<<len(point),"prover/table-shape")
    for r in point:
        samples=[]
        for t in (ZERO,ONE,F(2)):
            samples.append(sum(((a0+(a1-a0)*t)*(b0+(b1-b0)*t)
                               for a0,a1,b0,b1 in zip(a[::2],a[1::2],b[::2],b[1::2])),ZERO))
        q0=samples[0]
        q2=(samples[2]-F(2)*samples[1]+samples[0])*HALF
        q1=samples[1]-q0-q2
        q=(q0,q1,q2)
        reference=round_direct(a,b)
        same_vector(q,reference,"reference/round-polynomial")
        rounds.append(q)
        a=[v+(w-v)*r for v,w in zip(a[::2],a[1::2])]
        b=[v+(w-v)*r for v,w in zip(b[::2],b[1::2])]
    require(len(a)==len(b)==1,"prover/final-table-length")
    return {"rounds":tuple(rounds),"point":tuple(point),"terminal_u":a[0],"terminal_point":tuple(point)}


def check_sumcheck(proof,equations,alpha,desc,expected_point,wrong_weight=False,wrong_weight_axis=None):
    require(len(proof["rounds"])==len(expected_point),"batch/round-count")
    same_nested(proof["point"],expected_point,"batch/challenge-point")
    same_nested(proof["terminal_point"],expected_point,"batch/terminal-point")
    claim=sum(((alpha**e.ordinal)*e.target for e in equations),ZERO)
    for index,(coeff,r) in enumerate(zip(proof["rounds"],expected_point)):
        require(len(coeff)<=3,"batch/degree")
        require(len(coeff)==3,"batch/polynomial-shape")
        require(poly_eval(coeff,ZERO)+poly_eval(coeff,ONE)==claim,
                "batch/initial" if index==0 else "batch/round-identity")
        claim=poly_eval(coeff,r)
    weight_point=expected_point
    if wrong_weight_axis is not None:
        coordinates=list(expected_point)
        axis=0 if wrong_weight_axis=="coefficient" else bits(desc.k1)
        coordinates[axis]=coordinates[axis]+ONE
        weight_point=tuple(coordinates)
    value=weight_factored(equations,alpha,desc,weight_point,wrong_weight)
    require(claim==proof["terminal_u"]*value,"batch/final")
    return proof["terminal_u"]


@dataclass
class Case:
    desc: Desc
    original: Original
    registry: IdealRegistry
    source: dict
    actual_u: tuple
    V: tuple
    level2_rows: tuple
    level2_masks: tuple
    level2_physical: tuple
    root: int
    level2_points: tuple
    level2_columns: tuple
    level2_descriptor: tuple
    identity: str
    session: str
    original_root: int
    original_descriptor: tuple


def build_case(original,desc,actual_u=None):
    registry=IdealRegistry()
    session="e22-local-session/"+str(next(SESSIONS))
    orig_desc=("original",desc.k1,desc.D,desc.G,desc.mode,desc.P1,original.family,original.cat,session)
    oroot,opoints,ocols=registry.commit(orig_desc,tuple(v for row in original.rows for v in row),
                                       original.rows,original.masks,original.physical,desc.k1)
    source={"identity":desc.identity,"session":session,"root":oroot,"original_descriptor":orig_desc,
            "draw":draws(original.cat,2*desc.k1),"physical":original.physical,
            "columns":tuple((j,tuple(ocols[j][r] for r in original.physical))
                            for j in sorted(set(draws(original.cat,2*desc.k1)))),
            "column_points":opoints,"values":original.values[:desc.G],
            "q":original.q[:desc.G],"M":original.combined_masks,"ownership":original.row_ownership}
    actual_u=tuple(tuple(row) for row in (original.Urows if actual_u is None else actual_u))
    require(len(actual_u)==desc.m and all(len(row)==desc.k1 for row in actual_u),"commit/U-shape")
    padded_u=actual_u+tuple((ZERO,)*desc.k1 for _ in range(desc.groups-desc.m))
    V=tuple(F(v.a if lane==0 else v.b) for row in padded_u for lane in range(2) for v in row)
    V=V+(ZERO,)*(max(len(V),desc.k2)-len(V))
    rows=tuple(tuple(V[start:start+desc.k2]) for start in range(0,len(V),desc.k2))
    # Structural physical rows, independent of sampled values.
    physical=tuple(r for r in range(len(rows)) if r*desc.k2 < 2*desc.m*desc.k1)
    masks=tuple(tuple(fixture_value(original.family,30,r,j) if r in physical else ZERO
                      for j in range(desc.P2)) for r in range(len(rows)))
    l2desc=("combined-U",desc.identity,orig_desc,original.cat,original.family)
    root,points,columns=registry.commit(l2desc,V,rows,masks,physical,desc.k2)
    source["cu"]=(root,l2desc)
    identity=digest((desc,original.family,original.cat,source,V))
    M.maxima["U_table_size"]=max(M.maxima["U_table_size"],len(padded_u)*desc.k1)
    M.maxima["V_table_size"]=max(M.maxima["V_table_size"],len(V))
    M.maxima["allocated_R2"]=max(M.maxima["allocated_R2"],len(rows))
    return Case(desc,original,registry,source,actual_u,V,rows,masks,physical,root,points,columns,l2desc,identity,session,oroot,orig_desc)


def lane_points(desc,point):
    require(len(point)==bits(desc.groups*desc.k1),"terminal/dimensions")
    si,sg=point[:bits(desc.k1)],point[bits(desc.k1):]
    extra=bits(max(2*desc.groups*desc.k1,desc.k2))-bits(2*desc.groups*desc.k1)
    return tuple(si+(F(lane),)+sg+(ZERO,)*extra for lane in (0,1))


def level2_groups(claims,k):
    groups=[]
    for claim in claims:
        p=claim["point"][:bits(k)]
        found=next((g for g in groups if g[0]==p),None)
        if found is None:
            found=(p,[]);groups.append(found)
        found[1].append(claim)
    return tuple((p,tuple(cs)) for p,cs in groups)


def l2_coefficients(claims,k,rows,cat):
    groups=level2_groups(claims,k)
    coeffs=[]
    for p,cs in groups:
        coeff=[]
        for r in range(rows):
            coeff.append(sum((((F(2)+U)**(t+1))*eq_boolean(c["point"][bits(k):],r)
                              for t,c in enumerate(cs)),ZERO))
        coeffs.append(tuple(coeff))
    coeffs.append(tuple(eq_boolean(catalog(cat,bits(rows),2),r) for r in range(rows)))
    return groups,tuple(coeffs)


def make_level2(case,claims):
    d=case.desc;cat=case.original.cat
    groups,coeffs=l2_coefficients(claims,d.k2,len(case.level2_rows),cat)
    combined=tuple(tuple(sum((c[r]*case.level2_rows[r][i] for r in range(len(c))),ZERO)
                         for i in range(d.k2)) for c in coeffs)
    masks=tuple(tuple(sum((c[r]*case.level2_masks[r][j] for r in range(len(c))),ZERO)
                      for j in range(d.P2)) for c in coeffs)
    query=draws(cat,2*d.k2)
    response=tuple((j,tuple(case.level2_columns[j][r] for r in case.level2_physical)) for j in sorted(set(query)))
    M.maxima["level2_evaluation_groups"]=max(M.maxima["level2_evaluation_groups"],len(groups))
    return {"root":case.root,"descriptor":case.level2_descriptor,"claims":tuple(claims),
            "combined":combined,"masks":masks,"draw":query,"columns":response,
            "physical":case.level2_physical,"split":(bits(d.k2),bits(len(case.level2_rows))),"consumed":True}


def check_level2(record,claims,case,skip_ideal=False,skip_shape=False,old_phi=False):
    # Only public descriptor/dimensions plus ideal services. No V/row-mask reads.
    d=case.desc
    require(record is not None,"terminal/missing-opening")
    require(record["root"]==case.root,"level2/root")
    require(record["descriptor"]==case.level2_descriptor,"level2/descriptor")
    same_nested(record["claims"],tuple(claims),"level2/claim-inventory")
    R=max(2*d.groups*d.k1,d.k2)//d.k2
    require(record.get("split")== (bits(d.k2),bits(R)),"level2/split")
    groups,coeffs=l2_coefficients(claims,d.k2,R,case.original.cat)
    if not skip_shape:
        require(len(record["combined"])==len(groups)+1,"level2/combined-count")
        require(len(record["masks"])==len(groups)+1,"level2/mask-count")
        for row in record["combined"]:require(len(row)==d.k2,"level2/combined-width")
        for row in record["masks"]:require(len(row)==d.P2,"level2/mask-width")
    for g,(p,cs) in enumerate(groups):
        target=sum((((F(2)+U)**(t+1))*c["value"] for t,c in enumerate(cs)),ZERO)
        require(dot(record["combined"][g],tuple(eq_boolean(p,i) for i in range(d.k2)))==target,
                "level2/evaluation")
    same_nested(record["draw"],draws(case.original.cat,2*d.k2),"level2/query-draw")
    physical=tuple(r for r in range(R) if r*d.k2 < 2*d.m*d.k1)
    same_nested(record["physical"],physical,"level2/row-map")
    case.registry.verify_columns(case.root,case.level2_descriptor,record["draw"],record["columns"])
    for j,symbols in record["columns"]:
        x=case.level2_points[j]
        for g,c in enumerate(coeffs):
            target=sum((c[r]*v for r,v in zip(physical,symbols)),ZERO)
            phi=x**d.k2
            if old_phi:phi=phi-ONE
            require(poly_eval(record["combined"][g],x)+phi*poly_eval(record["masks"][g],x)==target,
                    "level2/column")
    if not skip_ideal:
        for claim in claims:
            case.registry.verify_mle(case.root,case.level2_descriptor,claim["point"],claim["value"])
    require(record["consumed"] is True,"terminal/unconsumed")
    return True


def reference_level2(case, claims, record):
    """Complete-table oracle; never calls level2_groups or l2_coefficients."""
    width=case.desc.k2
    column_bits=width.bit_length()-1
    partitions=[]
    for claim in claims:
        key=tuple(claim["point"][j] for j in range(column_bits))
        index=None
        for j,(prior,_) in enumerate(partitions):
            if prior==key:
                index=j
                break
        if index is None:
            partitions.append((key,[claim]))
        else:
            partitions[index][1].append(claim)
    M.judge(len(record["combined"])==len(partitions)+1,"reference/l2-group-count")
    M.judge(len(record["masks"])==len(partitions)+1,"reference/l2-mask-count")
    row_count=len(case.level2_rows)
    weights=[]
    for _,group_claims in partitions:
        row_weights=[ZERO for _ in range(row_count)]
        power=F(2)+U
        for claim in group_claims:
            row_point=claim["point"][column_bits:]
            for row in range(row_count):
                factor=ONE
                for bit,r in enumerate(row_point):
                    factor=factor*(r if (row>>bit)&1 else ONE-r)
                row_weights[row]=row_weights[row]+power*factor
            power=power*(F(2)+U)
        weights.append(tuple(row_weights))
    proximity=catalog(case.original.cat,bits(row_count),2)
    proximity_weights=[]
    for row in range(row_count):
        factor=ONE
        for bit,r in enumerate(proximity):
            factor=factor*(r if (row>>bit)&1 else ONE-r)
        proximity_weights.append(factor)
    weights.append(tuple(proximity_weights))
    for group,coefficients in enumerate(weights):
        expected=[]
        for column in range(width):
            value=ZERO
            for row,c in enumerate(coefficients):
                value=value+c*case.level2_rows[row][column]
            expected.append(value)
        same_vector(record["combined"][group],expected,"reference/l2-combined")
        expected_masks=[]
        for column in range(case.desc.P2):
            value=ZERO
            for row,c in enumerate(coefficients):
                value=value+c*case.level2_masks[row][column]
            expected_masks.append(value)
        same_vector(record["masks"][group],expected_masks,"reference/l2-mask")
    # Direct monomial evaluation uses no encoded()/poly_eval() reference shortcut.
    for index,symbols in record["columns"]:
        x=case.level2_points[index]
        expected=[]
        for row in case.level2_physical:
            power=ONE
            value=ZERO
            for coefficient in case.level2_rows[row]:
                value=value+coefficient*power
                power=power*x
            mask=ZERO
            mask_power=ONE
            for coefficient in case.level2_masks[row]:
                mask=mask+coefficient*mask_power
                mask_power=mask_power*x
            expected.append(value+power*mask)
        same_vector(symbols,expected,"reference/l2-column")
    M.counts["independent_level2_references"]+=1


def events_for(desc, source, round_count, mode, record=None, alpha=None, cat=None):
    """Deep-copied local message receipts, not a cryptographic FS transcript.

    A full candidate supplies record/alpha/catalog. Standalone order controls
    may omit the arithmetic record; those traces claim order coverage only.
    """
    proof = {} if record is None else record.get("proof", {})
    rounds = proof.get("rounds", ())
    challenges = proof.get("point", ())
    opening = None if record is None else record.get("opening")
    def part(key):
        return None if opening is None else opening.get(key)
    g1 = None if cat is None else gamma(cat)
    xi1 = None if cat is None else catalog(cat, bits(pow2(desc.D)))
    R2 = max(2*desc.groups*desc.k1, desc.k2)//desc.k2
    xi2 = None if cat is None else catalog(cat, bits(R2), 2)
    events = [
        ("descriptor", (desc.identity, source["session"])),
        ("original-root", source["root"]),
        ("claims", {"values": source["values"], "q": source["q"]}),
    ]
    if desc.mode == "blinded-algebra":
        events.append(("blinding", {"q": source["q"], "ownership": source["ownership"]}))
    events += [
        ("e22/l1/gamma", g1), ("e22/l1/xi", xi1), ("C_U", source["cu"]),
        ("M", source["M"]), ("queries1", (mode, source["draw"])),
        ("columns1", {"columns": source["columns"], "physical": source["physical"],
                       "column_points": source["column_points"]}),
        ("equations", None if record is None else record.get("equations")),
        ("e22/batch/alpha", alpha),
    ]
    for r in range(round_count):
        events += [(f"e22/batch/round/{r}", rounds[r] if r < len(rounds) else None),
                   (f"e22/batch/challenge/{r}", challenges[r] if r < len(challenges) else None)]
    events += [
        ("terminal-requests", {"lanes": None if record is None else record.get("claims"),
                               "terminal_u": proof.get("terminal_u"),
                               "terminal_point": proof.get("terminal_point")}),
        ("e22/l2/gamma", F(2)+U), ("e22/l2/xi", xi2),
        ("combined2", {"combined": part("combined"), "masks": part("masks")}),
        ("queries2", (mode, part("draw"))),
        ("columns2", {"columns": part("columns"), "physical": part("physical")}),
        ("opening2", {"root": part("root"), "descriptor": part("descriptor"),
                       "claims": part("claims"), "split": part("split")}),
        ("consume", part("consumed")),
    ]
    # A later mutation of a source/record cannot mutate its earlier receipt.
    return copy.deepcopy(tuple(events))


def check_events(events, desc, source, round_count, mode, record=None, alpha=None, cat=None):
    require(mode in ("live", "ni"), "transcript/mode")
    expected = events_for(desc, source, round_count, mode, record, alpha, cat)
    require(len(events) == len(expected), "transcript/count")
    for got, want in zip(events, expected):
        require(isinstance(got, (tuple, list)) and len(got) == 2, "transcript/event-shape")
        require(got[0] == want[0], "transcript/order-label")
        same_nested(got[1], want[1], "transcript/identity-mode")


def make_record(case, equations, alpha, source=None):
    source = case.source if source is None else source
    desc = case.desc
    point = catalog(case.original.cat, bits(desc.groups*desc.k1), 1)
    W = weight_table(equations, alpha, desc)
    u = tuple(v for row in case.actual_u for v in row)+(ZERO,)*((desc.groups-desc.m)*desc.k1)
    proof = make_sumcheck(u, W, point)
    claims = tuple({"lane": lane, "root": case.root, "descriptor": case.level2_descriptor,
                    "point": p, "value": mle_ref(case.V, p)}
                   for lane, p in enumerate(lane_points(desc, point)))
    record = {"equations": equations, "proof": proof, "claims": claims,
              "opening": make_level2(case, claims), "mode": "live"}
    record["events"] = events_for(desc, source, len(point), "live", record, alpha, case.original.cat)
    return record


def candidate_fixture(case, source, alpha, record, mutant=None):
    """Explicit malformed-message fixture fixed before its dependent challenge.

    This is ONLY a corpus adapter, never candidate acceptance. Rebind content
    receipts to the actual intentionally malformed message so its own typed
    semantic guard is exercised. Keep event order and trusted identity anchors;
    this cannot repair a swapped/dropped event or a different-session replay.
    Unchanged-receipt attacks below call candidate directly instead.
    """
    frame = copy.deepcopy(record)
    expected = dict(events_for(case.desc, source, bits(case.desc.groups*case.desc.k1),
                               frame["mode"], frame, alpha, case.original.cat))
    anchors = {"descriptor", "original-root", "C_U"}
    frame["events"] = tuple((label, payload if label in anchors or label not in expected
                            else copy.deepcopy(expected[label]))
                           for label, payload in frame["events"])
    return candidate(case, source, alpha, frame, mutant)


def candidate(case,source,alpha,record,mutant=None):
    desc=parse_desc({f.name:getattr(case.desc,f.name) for f in fields(Desc)})
    require(source.get("root")==case.original_root,"source/original-root")
    require(source.get("original_descriptor")==case.original_descriptor,"source/original-descriptor")
    require(source.get("session")==case.session,"source/session")
    require(source.get("cu")== (case.root,case.level2_descriptor),"source/C_U")
    if mutant!="skip-order":
        check_events(record["events"],desc,source,bits(desc.groups*desc.k1),record["mode"],
                     record,alpha,case.original.cat)
    expected=derive_equations(source,desc,case.original.cat,case.registry,
                              skip_membership=mutant=="skip-membership",phi_old=mutant=="old-phi",
                              allow_plain_q=mutant=="allow-plain-q")
    inventory(record["equations"],expected,skip=mutant=="skip-inventory")
    value=check_sumcheck(record["proof"],record["equations"],alpha,desc,
                        catalog(case.original.cat,bits(desc.groups*desc.k1),1),mutant=="wrong-weight",
                        "group" if mutant=="wrong-weight-group" else
                        "coefficient" if mutant=="wrong-weight-coefficient" else None)
    if mutant=="value-only":return True
    require(len(record["claims"])==2,"terminal/lane-count")
    expected_points=lane_points(desc,record["proof"]["point"])
    for lane,claim in enumerate(record["claims"]):
        require(claim["lane"]==lane,"terminal/lane-order")
        same_nested(claim["point"],expected_points[lane],"terminal/lane-point")
        require(claim["root"]==case.root,"terminal/root")
        require(claim["descriptor"]==case.level2_descriptor,"terminal/descriptor")
    require(value==record["claims"][0]["value"]+U*record["claims"][1]["value"],"terminal/reconstruction")
    check_level2(record["opening"],record["claims"],case,skip_ideal=mutant=="skip-ideal")
    return True


def outcome(fn):
    try:
        fn();return "accept"
    except Reject as error:
        return error.reason


def exercise(case,source=None,label="honest",save=False):
    source=case.source if source is None else source
    equations=derive_equations(source,case.desc,case.original.cat,case.registry)
    residual=ref_residuals(case.actual_u,equations,case.desc)
    M.judge(len(residual)==len(equations),"reference/residual-length")
    polynomial=tuple(residual)  # One unique coefficient per canonical ordinal.
    results=[]
    for alpha in (ZERO,ONE,F(2),U):
        record=make_record(case,equations,alpha,source=source)
        reference_level2(case,record["claims"],record["opening"])
        W=weight_table(equations,alpha,case.desc)
        direct=sum(((alpha**e.ordinal)*dot(case.actual_u[e.group],equation_weights(e,case.desc.k1))
                    for e in equations),ZERO)
        target=sum(((alpha**e.ordinal)*e.target for e in equations),ZERO)
        D=poly_eval(polynomial,alpha)
        M.judge(direct-target==D,"reference/batch-residual")
        M.judge(dot(tuple(v for row in case.actual_u for v in row)+(ZERO,)*((case.desc.groups-case.desc.m)*case.desc.k1),W)==direct,
                "reference/full-batch-sum")
        M.judge(weight_factored(equations,alpha,case.desc,record["proof"]["point"])==mle_ref(W,record["proof"]["point"]),
                "reference/factored-weight")
        got=outcome(lambda:candidate(case,source,alpha,record))
        expected="accept" if D==ZERO else ("batch/initial" if record["proof"]["rounds"] else "batch/final")
        M.judge(got==expected,"candidate/root-classification",{"label":label,"alpha":alpha,"D":D,"got":got,"expected":expected})
        M.event("reduction", {"case_identity":case.identity,"descriptor":case.desc,
                "family":case.original.family,"catalog":case.original.cat,"label":label,
                "alpha":alpha,"residual_at_alpha":D,"expected":expected,"actual":got,
                "independent_level2_reference_completed":True})
        M.counts["reductions/"+label]+=1
        M.counts[("root_acceptances/" if D==ZERO else "nonroot_rejections/")+label]+=1
        if save:results.append({"alpha":alpha,"residual_at_alpha":D,"expected":expected,"actual":got,
                                "full_record":record})
    if save:
        M.event("balanced-control",{"label":label,"source":source,"committed_U":case.actual_u,
                                    "equations":equations,"residual_coefficients":residual,"runs":results})
    return residual


CONTROLS=[]


def control(name,positive,negative,expected,inputs=None,mutant=None):
    good=outcome(positive)
    bad=outcome(negative)
    M.judge(good=="accept",name+"/positive",good)
    M.judge(bad==expected,name+"/negative",{"expected":expected,"actual":bad})
    sensitivity=None
    if mutant is not None:
        sensitivity=outcome(mutant)
        M.judge(sensitivity=="accept",name+"/mutant",sensitivity)
    row={"name":name,"positive":good,"expected":expected,"actual":bad,"mutant":sensitivity}
    CONTROLS.append(row)
    M.event("control",dict(row,inputs=inputs))


def arithmetic_controls():
    M.arm="arithmetic"
    values=(ZERO,ONE,-ONE,U,ONE+U,F(2,3))
    for x,y,z in itertools.product(values,repeat=3):
        pair=pairadd((x.a,x.b),pairmul((y.a,y.b),(z.a,z.b)))
        value=x+y*z
        M.judge((value.a,value.b)==pair,"field/pair-oracle")
        M.judge(x*(y+z)==x*y+x*z,"field/distributivity")
    for x in values:
        if x!=ZERO:M.judge(x*x.inv()==ONE,"field/inverse")
    # Every Fraction expression is one declared rational operation.
    def add(a,b):M.op();return a+b
    def sub(a,b):M.op();return a-b
    def mul(a,b):M.op();return a*b
    half=Fraction(1,2)
    # Independent integer and Fraction embeddings of the registered row point.
    M.op(); twice_half=(2*HALF.a) % P
    M.judge(twice_half==1,"field/canonical-half-integer")
    M.judge(F(2)*HALF==ONE,"field/canonical-half")
    embedded_half=F(half.numerator)/F(half.denominator)
    M.judge(embedded_half==HALF,"fraction/half-base-embedding")
    c=add(Fraction(2),Fraction(4))
    u0,u1=mul(c,Fraction(2)),mul(c,Fraction(-1))
    mask=mul(c,Fraction(3))
    ev=add(mul(u0,half),mul(u1,half))
    col=add(sub(Fraction(2),Fraction(3)),mul(Fraction(9),Fraction(3)))
    corrected=sub(mul(c,col),mul(Fraction(9),mask))
    expected=(Fraction(6),Fraction(12),Fraction(-6),Fraction(18),Fraction(3),Fraction(26),Fraction(-6))
    same_vector((c,u0,u1,mask,ev,col,corrected),expected,"fraction/hand")
    for value in (c,u0,u1,mask,ev,col,corrected):
        M.op();num=F(value.numerator);den=F(value.denominator)
        M.judge(num/den==F(int(value)),"fraction/base-embedding")
    lane0=mle_ref((ONE,F(2),F(3),F(4)),(U,HALF))
    lane1=mle_ref((F(5),F(6),F(7),F(8)),(U,HALF))
    same_vector((lane0,lane1,lane0+U*lane1),(F(2)+U,F(6)+U,F(9,7)),"lane/hand")
    control("lane-not-real-imag",lambda:require(lane0+U*lane1==F(9,7),"lane/reconstruction"),
            lambda:require(F(9)+U*F(7)==lane0+U*lane1 and (F(9),F(7))==(lane0,lane1),"lane/components"),
            "lane/components",{"point":(U,HALF),"lanes":(lane0,lane1),"combined":F(9,7)})


def fixed_case(k1=4,k2=2,G=2,mode="blinded-algebra",cat=3,P1=1,P2=1,family=3,D=3):
    orig=original_fixture(k1,D,G,mode,P1,family,cat)
    return build_case(orig,Desc(k1,D,G,mode,P1,k2,P2))


def mutations():
    M.arm="controls"
    case=fixed_case();alpha=F(2)
    equations=derive_equations(case.source,case.desc,case.original.cat,case.registry)
    base=make_record(case,equations,alpha)
    positive=lambda:candidate_fixture(case,case.source,alpha,base)
    inputs={"descriptor":case.desc,"source":case.source,"record":base,"committed_U":case.actual_u}
    # Field-by-field descriptor validation, before any allocation.
    record={f.name:getattr(case.desc,f.name) for f in fields(Desc)}
    def good_desc():parse_desc(record)
    for name,value in (("format","unknown"),("basis","evaluation-natural/old"),("lanes",(0,)),
                       ("lanes",(1,0)),("mode","hiding-approved"),("k1",8),("D",0),("G",3),
                       ("P1",2),("k2",1),("P2",2)):
        bad=dict(record);bad[name]=value
        control("descriptor/"+name+"/"+str(value),good_desc,lambda b=bad:parse_desc(bad if b is None else b),
                "descriptor/"+name,{"descriptor":bad})
    for name in ("k1","D","G","P1","k2","P2"):
        bad=dict(record);bad[name]=True
        control("descriptor/bool/"+name,good_desc,lambda b=bad:parse_desc(b),"descriptor/"+name,{"descriptor":bad})
    for name in ("missing","extra","m","groups"):
        bad=dict(record)
        if name=="missing":bad.pop("format")
        else:bad[name]=7
        control("descriptor/"+name,good_desc,lambda b=bad:parse_desc(b),"descriptor/fields",{"descriptor":bad})
    # Every original equation must have its canonical position/source target.
    for index in range(len(equations)):
        changed=copy.deepcopy(base);changed["equations"]=equations[:index]+equations[index+1:]
        control(f"inventory/drop/{index}",positive,lambda r=changed:candidate_fixture(case,case.source,alpha,r),
                "equation-inventory/length",dict(inputs,record=changed))
        # The actual omitted-list reduction passes if the inventory guard is
        # disabled: use a consistent proof/opening for that list, not bare inequality.
        omitted=changed["equations"]
        mutant=make_record(case,omitted,alpha)
        M.judge(outcome(lambda:candidate_fixture(case,case.source,alpha,mutant,"skip-inventory"))=="accept",
                "inventory/drop-mutant-sensitive",index)
    for label,change in (
            ("duplicate",lambda es:es+(es[0],)),
            ("reorder",lambda es:(es[1],es[0])+es[2:]),
            ("proximity-evaluation",lambda es:(Equation("evaluation",case.desc.G,-1,0,(),ZERO,case.desc.identity),)+es[1:])):
        r=copy.deepcopy(base);r["equations"]=change(equations)
        reason="equation-inventory/length" if label=="duplicate" else "equation-inventory/metadata"
        control("inventory/"+label,positive,lambda r=r:candidate_fixture(case,case.source,alpha,r),reason,dict(inputs,record=r))
    for field,value in (("ordinal",99),("group",99),("kind","invented"),("target",equations[0].target+ONE)):
        r=copy.deepcopy(base);a=equations[0]
        kwargs={f.name:getattr(a,f.name) for f in fields(Equation)};kwargs[field]=value
        r["equations"]=(Equation(**kwargs),)+equations[1:]
        reason="equation-inventory/target-from-source" if field=="target" else "equation-inventory/metadata"
        control("inventory/change/"+field,positive,lambda r=r:candidate_fixture(case,case.source,alpha,r),reason,dict(inputs,record=r))
    # Original column membership/shape cannot be replaced by the U commitment.
    for label,mutate,reason in (
        ("root",lambda s:s.__setitem__("root",999999),"source/original-root"),
        ("descriptor",lambda s:s.__setitem__("original_descriptor",("different",)),"source/original-descriptor"),
        ("omit-column",lambda s:s.__setitem__("columns",s["columns"][1:]),"membership/column-order"),
        ("repeat-column",lambda s:s.__setitem__("columns",s["columns"]+(s["columns"][0],)),"membership/column-order"),
        ("unsort-column",lambda s:s.__setitem__("columns",tuple(reversed(s["columns"]))),"membership/column-order"),
        ("row-map",lambda s:s.__setitem__("physical",tuple(range(len(s["physical"])))),"source/row-map")):
        source=copy.deepcopy(case.source);mutate(source)
        frame=copy.deepcopy(base)
        if label=="root":
            # Fix the altered claimed root in its event before testing membership.
            # A separate session/event control checks an inconsistent payload.
            frame["events"]=events_for(case.desc,source,len(base["proof"]["point"]),"live")
        control("original/"+label,positive,lambda s=source,r=frame:candidate_fixture(case,s,alpha,r),reason,dict(inputs,source=source,record=frame))
    source=copy.deepcopy(case.source);j,symbols=source["columns"][0]
    source["columns"]=((j,(symbols[0]+ONE,)+symbols[1:]),)+source["columns"][1:]
    control("original/symbol",positive,lambda:candidate_fixture(case,source,alpha,base),"membership/symbol",dict(inputs,source=source))
    # Omitted combined masks are a shape fault; false same-shaped masks must
    # reach reduction, not a comparison against an oracle's honest masks.
    source=copy.deepcopy(case.source);source["M"]=source["M"][:-1]
    control("original/missing-mask",positive,lambda:candidate_fixture(case,source,alpha,base),"source/mask-count",dict(inputs,source=source))
    source=copy.deepcopy(case.source);source["M"]=((),)+source["M"][1:]
    control("original/mask-width",positive,lambda:candidate_fixture(case,source,alpha,base),"source/mask-width",dict(inputs,source=source))
    # Nonroot false-M attack, including blinding mask component, has a full
    # valid-to-committed-V terminal record and unchanged ideal columns.
    for group,label in ((0,"combined-mask"),(case.desc.G,"blinding-row-mask")):
        source=copy.deepcopy(case.source);masks=list(source["M"]);masks[group]=(masks[group][0]+ONE,);source["M"]=tuple(masks)
        es=derive_equations(source,case.desc,case.original.cat,case.registry)
        r=make_record(case,es,alpha,source=source)
        control("original/alter/"+label,positive,lambda s=source,r=r:candidate_fixture(case,s,alpha,r),"batch/initial",dict(inputs,source=source,record=r))
    old=derive_equations(case.source,case.desc,case.original.cat,case.registry,phi_old=True)
    r=make_record(case,old,alpha)
    control("original/old-vanishing-factor",positive,lambda:candidate_fixture(case,case.source,alpha,r),
            "equation-inventory/target-from-source",dict(inputs,record=r))
    # Round and public final-weight checks.
    for label,mutate,reason in (
        ("degree",lambda r:r["proof"].__setitem__("rounds",(r["proof"]["rounds"][0]+(ONE,),)+r["proof"]["rounds"][1:]),"batch/degree"),
        ("identity",lambda r:r["proof"].__setitem__("rounds",((r["proof"]["rounds"][0][0]+ONE,)+r["proof"]["rounds"][0][1:],)+r["proof"]["rounds"][1:]),"batch/initial"),
        ("challenge-length",lambda r:r["proof"].__setitem__("point",r["proof"]["point"][:-1]),"batch/challenge-point"),
        ("round-count",lambda r:r["proof"].__setitem__("rounds",r["proof"]["rounds"][:-1]),"batch/round-count"),
        ("terminal-point",lambda r:r["proof"].__setitem__("terminal_point",tuple(ZERO for _ in r["proof"]["point"])),"batch/terminal-point"),
        ("terminal-value",lambda r:r["proof"].__setitem__("terminal_u",r["proof"]["terminal_u"]+ONE),"batch/final")):
        r=copy.deepcopy(base);mutate(r)
        control("batch/"+label,positive,lambda r=r:candidate_fixture(case,case.source,alpha,r),reason,dict(inputs,record=r))
    control("batch/wrong-monomial-exponent",positive,lambda:candidate_fixture(case,case.source,alpha,base,"wrong-weight"),
            "batch/final",inputs)
    # Named terminal obligations, including unfinished transcript acceptance mutant.
    for label,mutate,reason in (
        ("drop-lane0",lambda r:r.__setitem__("claims",r["claims"][1:]),"terminal/lane-count"),
        ("drop-lane1",lambda r:r.__setitem__("claims",r["claims"][:1]),"terminal/lane-count"),
        ("drop-both",lambda r:r.__setitem__("claims",()),"terminal/lane-count"),
        ("missing-opening",lambda r:r.__setitem__("opening",None),"terminal/missing-opening"),
        ("unconsumed",lambda r:r["opening"].__setitem__("consumed",False),"terminal/unconsumed"),
        ("lane-order",lambda r:r["claims"][0].__setitem__("lane",1),"terminal/lane-order"),
        ("value",lambda r:r["claims"][0].__setitem__("value",r["claims"][0]["value"]+ONE),"terminal/reconstruction"),
        ("root",lambda r:r["claims"][0].__setitem__("root",999999),"terminal/root"),
        ("descriptor",lambda r:r["claims"][0].__setitem__("descriptor",("different",)),"terminal/descriptor")):
        r=copy.deepcopy(base);mutate(r)
        control("terminal/"+label,positive,lambda r=r:candidate_fixture(case,case.source,alpha,r),reason,dict(inputs,record=r),
                mutant=(lambda r=r:candidate_fixture(case,case.source,alpha,r,"value-only")) if label.startswith("drop-") or label=="missing-opening" else None)
    for axis in range(len(base["claims"][0]["point"])):
        r=copy.deepcopy(base);point=list(r["claims"][0]["point"]);point[axis]=point[axis]+ONE;r["claims"][0]["point"]=tuple(point)
        control("terminal/point-axis/"+str(axis),positive,lambda r=r:candidate_fixture(case,case.source,alpha,r),
                "terminal/lane-point",dict(inputs,record=r))
    # Direct ideal service mutation: this is not concrete PCS soundness evidence.
    claim=base["claims"][0]
    control("ideal/false-terminal-value",lambda:case.registry.verify_mle(case.root,case.level2_descriptor,claim["point"],claim["value"]),
            lambda:case.registry.verify_mle(case.root,case.level2_descriptor,claim["point"],claim["value"]+ONE),
            "opening/value",claim)
    # Level2 explicit ordinary record guards.
    for label,mutate,reason in (
        ("missing-proximity",lambda r:r["opening"].__setitem__("combined",r["opening"]["combined"][:-1]),"level2/combined-count"),
        ("missing-mask",lambda r:r["opening"].__setitem__("masks",r["opening"]["masks"][:-1]),"level2/mask-count"),
        ("missing-group",lambda r:r["opening"].__setitem__("combined",r["opening"]["combined"][1:]),"level2/combined-count"),
        ("missing-column",lambda r:r["opening"].__setitem__("columns",r["opening"]["columns"][1:]),"membership/column-order"),
        ("root",lambda r:r["opening"].__setitem__("root",999999),"level2/root"),
        ("descriptor",lambda r:r["opening"].__setitem__("descriptor",("different",)),"level2/descriptor"),
        ("row-map",lambda r:r["opening"].__setitem__("physical",r["opening"]["physical"][:-1]),"level2/row-map")):
        r=copy.deepcopy(base);mutate(r)
        control("level2/"+label,positive,lambda r=r:candidate_fixture(case,case.source,alpha,r),reason,dict(inputs,record=r))
    for group in range(len(base["opening"]["masks"])):
        r=copy.deepcopy(base);mm=list(r["opening"]["masks"]);mm[group]=(mm[group][0]+ONE,);r["opening"]["masks"]=tuple(mm)
        control("level2/alter-mask/"+str(group),positive,lambda r=r:candidate_fixture(case,case.source,alpha,r),"level2/column",dict(inputs,record=r))
    control("level2/old-vanishing-factor",positive,
            lambda:check_level2(base["opening"],base["claims"],case,old_phi=True),"level2/column",inputs)
    # Public zero high bit occurs at k1=1,k2=4,G0, and the lane may be a column bit.
    small=fixed_case(k1=1,k2=4,G=0,cat=3)
    es=derive_equations(small.source,small.desc,small.original.cat,small.registry)
    sr=make_record(small,es,alpha);altered=copy.deepcopy(sr)
    pt=list(altered["claims"][0]["point"]);pt[-1]=ONE;altered["claims"][0]["point"]=tuple(pt)
    control("terminal/public-zero-high-bit",lambda:candidate_fixture(small,small.source,alpha,sr),
            lambda:candidate_fixture(small,small.source,alpha,altered),"terminal/lane-point",{"record":altered,"source":small.source})
    # Padded group coefficients must stay zero in the registry tensor.
    fake=list(case.V);fake[-1]=ONE
    control("registry/padded-group",positive,
            lambda:case.registry.commit(case.level2_descriptor,tuple(fake),
                  tuple(tuple(fake[i:i+case.desc.k2]) for i in range(0,len(fake),case.desc.k2)),
                  case.level2_masks,case.level2_physical,case.desc.k2),"registry/public-zero",{"fakeV":fake})
    # All temporal faults exercise the actual state checker in both tagged modes.
    for mode in ("live","ni"):
        events=events_for(case.desc,case.source,len(base["proof"]["point"]),mode)
        good=lambda e=events,m=mode:check_events(e,case.desc,case.source,len(base["proof"]["point"]),m)
        index={name:i for i,(name,_) in enumerate(events)}
        for before,after in (("C_U","queries1"),("C_U","e22/batch/alpha"),("M","e22/batch/alpha"),
                             ("columns1","e22/batch/alpha"),("equations","e22/batch/alpha"),
                             ("combined2","queries2"),("blinding","e22/l1/gamma")):
            e=list(events);a,b=index[before],index[after];e[a],e[b]=e[b],e[a]
            control("order/"+mode+"/"+before+"/"+after,good,
                    lambda e=tuple(e),m=mode:check_events(e,case.desc,case.source,len(base["proof"]["point"]),m),
                    "transcript/order-label",{"events":e})
        for label,mutate,reason in (
            ("drop",lambda e:e.pop(),"transcript/count"),
            ("duplicate",lambda e:e.append(e[-1]),"transcript/count"),
            ("replay",lambda e:e.__setitem__(0,("descriptor",("different-session",))),"transcript/identity-mode"),
            ("reuse-alpha",lambda e:e.__setitem__(index["e22/l2/gamma"],("e22/batch/alpha",None)),"transcript/order-label"),
            ("reuse-xi",lambda e:e.__setitem__(index["e22/l2/xi"],("e22/l1/xi",None)),"transcript/order-label")):
            e=list(events);mutate(e)
            control("order/"+mode+"/"+label,good,
                    lambda e=tuple(e),m=mode:check_events(e,case.desc,case.source,len(base["proof"]["point"]),m),reason,{"events":e})
    # Algebraic late-q compensation, with order failure and a skip-order mutant.
    bc=fixed_case(cat=1);ss=copy.deepcopy(bc.source)
    vv=[list(v) for v in ss["values"]];vv[0][0]=vv[0][0]+ONE;ss["values"]=tuple(map(tuple,vv))
    qq=list(ss["q"]);qq[0]=qq[0]-ONE;ss["q"]=tuple(qq)
    eqs=derive_equations(ss,bc.desc,bc.original.cat,bc.registry);br=make_record(bc,eqs,alpha,source=ss)
    late=list(br["events"]);a=next(i for i,e in enumerate(late) if e[0]=="blinding");b=next(i for i,e in enumerate(late) if e[0]=="e22/l1/gamma");late[a],late[b]=late[b],late[a];br["events"]=tuple(late)
    clean_record=copy.deepcopy(br);clean_record["events"]=events_for(bc.desc,ss,len(br["proof"]["point"]),"live")
    control("blinding/late-q-compensation",lambda:candidate_fixture(bc,ss,alpha,clean_record),
            lambda:candidate_fixture(bc,ss,alpha,br),"transcript/order-label",{"source":ss,"record":br},
            mutant=lambda:candidate_fixture(bc,ss,alpha,br,"skip-order"))


def session_controls():
    """Actual same-descriptor session replay, plus bound C_U event payload."""
    M.arm="controls"
    first=fixed_case()
    second=build_case(first.original,first.desc)
    alpha=F(2)
    e1=derive_equations(first.source,first.desc,first.original.cat,first.registry)
    e2=derive_equations(second.source,second.desc,second.original.cat,second.registry)
    r1=make_record(first,e1,alpha)
    r2=make_record(second,e2,alpha)
    M.judge(first.desc.identity==second.desc.identity,"session/same-valid-descriptor")
    M.judge(first.session!=second.session,"session/distinct-contexts")
    good=lambda:candidate_fixture(second,second.source,alpha,r2)
    control("session/same-descriptor-replay",good,
            lambda:candidate_fixture(second,second.source,alpha,r1),"transcript/identity-mode",
            {"from_session":first.session,"to_session":second.session,"record":r1})
    changed=copy.deepcopy(r2)
    at=next(i for i,e in enumerate(changed["events"]) if e[0]=="C_U")
    wire=list(changed["events"])
    wire[at]=("C_U",(999999,second.level2_descriptor))
    changed["events"]=tuple(wire)
    control("session/C_U-payload",good,lambda:candidate_fixture(second,second.source,alpha,changed),
            "transcript/identity-mode",{"record":changed})
    desc={f.name:getattr(second.desc,f.name) for f in fields(Desc)}
    invalid=dict(desc);invalid["lanes"]=(False,True)
    control("descriptor/bool-lanes",lambda:parse_desc(desc),lambda:parse_desc(invalid),
            "descriptor/lanes",{"descriptor":invalid})


def _classified_source_case(case, source, name, *, nonzero=True, balanced=False):
    """Reference residuals classify actual full candidate records at all four alphas."""
    residual=exercise(case,source,label=name,save=True)
    present=any(v!=ZERO for v in residual)
    M.judge(present==nonzero,"corpus/nonzero-residual/"+name)
    if balanced:
        total=sum(residual,ZERO)
        M.judge(total==ZERO,"corpus/balanced-unweighted/"+name)
        # Explicit defective aggregation: all equation coefficients are one,
        # irrespective of the requested fresh alpha. Invoke the full checker
        # on that defective aggregation; do not merely assert total==0.
        equations=derive_equations(source,case.desc,case.original.cat,case.registry)
        defective_record=make_record(case,equations,ONE,source=source)
        M.judge(outcome(lambda:candidate(case,source,ONE,defective_record))=="accept",
                "corpus/naive-unweighted-mutant/"+name)
        M.event("naive-unweighted-mutant",{"name":name,"requested_alpha":F(2),
                "defective_equation_weights":"all one (ignores requested alpha)",
                "record":defective_record,"source":source})
    if present:
        M.judge(any(poly_eval(residual,a)!=ZERO for a in (ZERO,F(2),U)),
                "corpus/nonroot-sensitive/"+name)
    return residual


def balanced_corpus():
    M.arm="balanced-controls"
    groups_checked=0
    for cat in (0,1,3):
        case=fixed_case(cat=cat)
        for delta in (ONE,U):
            source=copy.deepcopy(case.source)
            values=[list(row) for row in source["values"]]
            values[0][0]=values[0][0]+delta
            values[1][0]=values[1][0]-delta
            source["values"]=tuple(tuple(row) for row in values)
            label=f"evaluation-pair/cat{cat}/delta{clean(delta)}"
            _classified_source_case(case,source,label,nonzero=cat!=0,balanced=True)
            if cat==0:M.event("original-gamma-degeneracy",{"source":source,"label":label})
            groups_checked+=1
    case=fixed_case(cat=3)
    for g,h in ((0,1),(1,case.desc.G)):
        for delta in (ONE,U):
            source=copy.deepcopy(case.source)
            masks=list(source["M"])
            masks[g]=(masks[g][0]+delta,)
            masks[h]=(masks[h][0]-delta,)
            source["M"]=tuple(masks)
            _classified_source_case(case,source,f"column-pair/{g}/{h}/delta{clean(delta)}",balanced=True)
            groups_checked+=1
    # One distinct public index0 (catalog1 draw[0,0]); constant chosen before
    # queries. This tests a targeted finite event, not guessing-query security.
    case=fixed_case(cat=1)
    for delta in (ONE,U):
        source=copy.deepcopy(case.source)
        values=[list(row) for row in source["values"]]
        values[0][0]=values[0][0]+delta
        source["values"]=tuple(tuple(row) for row in values)
        masks=list(source["M"])
        masks[case.desc.G]=(masks[case.desc.G][0]+delta/(F(7)**case.desc.k1),)
        source["M"]=tuple(masks)
        _classified_source_case(case,source,f"evaluation-proximity/delta{clean(delta)}",balanced=True)
        groups_checked+=1
    # Binding permits arbitrary well-shaped U: swap whole groups before C_U.
    original=original_fixture(4,3,2,"blinded-algebra",1,3,3)
    desc=Desc(4,3,2,"blinded-algebra",1,2,1)
    swapped=list(original.Urows)
    swapped[0],swapped[1]=swapped[1],swapped[0]
    M.judge(any(a!=b for a,b in zip(original.Urows[0],original.Urows[1])),
            "corpus/group-swap-sensitive")
    changed=build_case(original,desc,swapped)
    for column in range(desc.k1):
        M.judge(original.Urows[0][column]+original.Urows[1][column]==
                swapped[0][column]+swapped[1][column],"corpus/swapped-vector-aggregate")
    _classified_source_case(changed,changed.source,"swapped-committed-groups")
    groups_checked+=1
    # Isolate missing base blinding components in a nonzero all-one fixture.
    original=original_fixture(4,3,2,"blinded-algebra",1,1,3)
    for g in range(desc.m):
        for lane in (0,1):
            altered=[list(row) for row in original.Urows]
            row=original.data_cube+2*g+lane
            for column in range(desc.k1):
                term=original.rows[row][column]*(ONE if lane==0 else U)
                altered[g][column]=altered[g][column]-term
            changed=build_case(original,desc,altered)
            _classified_source_case(changed,changed.source,f"missing-blinding-component/{g}/{lane}")
            groups_checked+=1
    return groups_checked


def additional_omission_controls():
    M.arm="controls"
    # Every full-query inventory omission is already checked in mutations().
    # Here a one-query fixture additionally demonstrates false source equations
    # accepted by an omission-check-disabled variant, rather than treating all
    # redundant full-query equations as independent forgery opportunities.
    case=fixed_case(cat=1)
    alpha=F(2)
    honest=derive_equations(case.source,case.desc,case.original.cat,case.registry)
    good_record=make_record(case,honest,alpha)
    positive=lambda:candidate_fixture(case,case.source,alpha,good_record)
    for kind,index in (("evaluation",0),("evaluation",1),("column",0),("column",1),("column",2)):
        source=copy.deepcopy(case.source)
        if kind=="evaluation":
            values=[list(row) for row in source["values"]]
            values[index][0]=values[index][0]+ONE
            source["values"]=tuple(tuple(row) for row in values)
            ordinal=index
        else:
            masks=list(source["M"])
            masks[index]=(masks[index][0]+ONE,)
            source["M"]=tuple(masks)
            ordinal=case.desc.G+index
        equations=derive_equations(source,case.desc,case.original.cat,case.registry)
        residual=ref_residuals(case.actual_u,equations,case.desc)
        for j,value in enumerate(residual):
            M.judge((value!=ZERO)==(j==ordinal),"omission/isolated-false-equation")
        omitted=equations[:ordinal]+equations[ordinal+1:]
        record=make_record(case,omitted,alpha,source=source)
        control(f"omission/false-source/{kind}/{index}",positive,
                lambda s=source,r=record:candidate_fixture(case,s,alpha,r),"equation-inventory/length",
                {"source":source,"omitted_ordinal":ordinal,"record":record},
                mutant=lambda s=source,r=record:candidate_fixture(case,s,alpha,r,"skip-inventory"))
    # Ownership/q inventory is checked without revealing rows to candidate.
    for kind in ("missing-q","pair-reuse","wrong-pair"):
        source=copy.deepcopy(case.source)
        if kind=="missing-q":
            source["q"]=source["q"][:-1]
            expected="source/q-count"
        else:
            ownership=list(source["ownership"])
            g,a,b=ownership[1]
            ownership[1]=(g,ownership[0][1] if kind=="pair-reuse" else a,b+1)
            source["ownership"]=tuple(ownership)
            expected="source/blinding-ownership"
        control("blinding/"+kind,positive,lambda s=source:candidate_fixture(case,s,alpha,good_record),
                expected,{"source":source,"record":good_record})
    return 8


def runner_case_reference(case):
    equations=derive_equations(case.source,case.desc,case.original.cat,case.registry)
    record=make_record(case,equations,ZERO)
    reference_level2(case,record["claims"],record["opening"])
    return {"checked":True,"method":"independent direct level2 partition/row weights/row masks/encoded symbols",
            "additional_per_reduction_reference":True}



def receipt_payload_controls():
    """Change a real message while retaining its earlier immutable receipt."""
    M.arm="controls/receipts"
    case=fixed_case()
    alpha=F(2)
    equations=derive_equations(case.source,case.desc,case.original.cat,case.registry)
    honest=make_record(case,equations,alpha)
    positive=lambda:candidate(case,case.source,alpha,honest)
    for label in ("values","q","M","columns"):
        source=copy.deepcopy(case.source)
        if label=="values":
            rows=[list(row) for row in source[label]];rows[0][0]=rows[0][0]+ONE
            source[label]=tuple(tuple(row) for row in rows)
        elif label in ("q","M"):
            rows=list(source[label])
            rows[0]=rows[0]+ONE if label=="q" else (rows[0][0]+ONE,)
            source[label]=tuple(rows)
        else:
            rows=list(source[label]);index,symbols=rows[0]
            rows[0]=(index,(symbols[0]+ONE,)+symbols[1:]);source[label]=tuple(rows)
        control("receipt/source-"+label,positive,
                lambda s=source:candidate(case,s,alpha,honest),
                "transcript/identity-mode",{"source":source,"record":honest})
    def changed_equation(frame):
        row=frame["equations"][0]
        changed=Equation(row.kind,row.group,row.index,row.ordinal,row.parameter,row.target+ONE,row.identity)
        frame["equations"]=(changed,)+frame["equations"][1:]
    def changed_round(frame):
        rounds=list(frame["proof"]["rounds"])
        rounds[0]=(rounds[0][0]+ONE,)+rounds[0][1:]
        frame["proof"]["rounds"]=tuple(rounds)
    def changed_opening_vector(frame,key):
        rows=list(frame["opening"][key]);rows[0]=(rows[0][0]+ONE,)+rows[0][1:]
        frame["opening"][key]=tuple(rows)
    def changed_opening_column(frame):
        rows=list(frame["opening"]["columns"]);index,symbols=rows[0]
        rows[0]=(index,(symbols[0]+ONE,)+symbols[1:]);frame["opening"]["columns"]=tuple(rows)
    for label,change in (
        ("equations",changed_equation), ("round",changed_round),
        ("terminal",lambda r:r["claims"][0].__setitem__("value",r["claims"][0]["value"]+ONE)),
        ("terminal-scalar",lambda r:r["proof"].__setitem__("terminal_u",r["proof"]["terminal_u"]+ONE)),
        ("combined2",lambda r:changed_opening_vector(r,"combined")),
        ("masks2",lambda r:changed_opening_vector(r,"masks")),
        ("columns2",changed_opening_column),
        ("consume",lambda r:r["opening"].__setitem__("consumed",False)),
    ):
        frame=copy.deepcopy(honest);change(frame)
        control("receipt/record-"+label,positive,
                lambda r=frame:candidate(case,case.source,alpha,r),
                "transcript/identity-mode",{"record":frame})
    frame=copy.deepcopy(honest);frame["mode"]="unknown"
    control("receipt/unknown-challenge-mode",positive,
            lambda:candidate(case,case.source,alpha,frame),"transcript/mode",{"record":frame})
    return 13



def ideal_omission_controls():
    """False lane requests with sampled-column-consistent level2 rows.

    This shows why the explicitly ideal final opening remains an obligation;
    sparse toy column tests alone do not replace it. No concrete PCS bound.
    """
    M.arm="controls/ideal-omission"
    case=fixed_case(k1=2,k2=4,cat=1)
    alpha=F(2)
    equations=derive_equations(case.source,case.desc,case.original.cat,case.registry)
    honest=make_record(case,equations,alpha)
    frame=copy.deepcopy(honest)
    deltas=(ONE,-ONE/U)
    for lane,delta in enumerate(deltas):
        frame["claims"][lane]["value"]=frame["claims"][lane]["value"]+delta
    frame["opening"]["claims"]=copy.deepcopy(frame["claims"])
    groups=level2_groups(frame["claims"],case.desc.k2)
    rows=list(frame["opening"]["combined"])
    x=case.level2_points[case.source["draw"][0]]
    # Catalog1 has one distinct level2 query at index0. X-x and X*(X-x)
    # preserve that queried column while changing the required off-cube MLE.
    for group,(point,claims) in enumerate(groups):
        q=(-x,ONE,ZERO,ZERO) if claims[0]["lane"]==0 else (ZERO,-x,ONE,ZERO)
        at_point=mle_ref(q,point)
        require(at_point!=ZERO,"control/nonzero-ideal-omission-direction")
        lane=claims[0]["lane"]
        scale=(F(2)+U)*deltas[lane]/at_point
        rows[group]=tuple(v+scale*c for v,c in zip(rows[group],q))
    frame["opening"]["combined"]=tuple(rows)
    control("ideal/final-opening-required",
            lambda:candidate(case,case.source,alpha,honest),
            lambda:candidate_fixture(case,case.source,alpha,frame),"opening/value",
            {"source":case.source,"honest":honest,"record":frame},
            mutant=lambda:candidate_fixture(case,case.source,alpha,frame,"skip-ideal"))


def runner_diagnostic_corpus():
    before=len(CONTROLS)
    mutations()
    session_controls()
    additional_omission_controls()
    source_parameter_controls()
    receipt_payload_controls()
    original_binding_controls()
    registry_shape_controls()
    remaining_parameter_controls()
    ideal_omission_controls()
    count=balanced_corpus()
    return {"negative_controls":True,"balanced_attacks":True,"session_replay":True,
            "corpus_complete":True,"named_controls":len(CONTROLS)-before,
            "balanced_source_or_committed_tensor_cases":count,
            "root_exceptions":"actual full candidate runs, honest-to-committed-V records; no dishonest-transcript exhaustion"}


def source_parameter_controls():
    M.arm="controls"
    case=fixed_case(mode="plain",cat=1)
    alpha=F(2)
    honest=derive_equations(case.source,case.desc,case.original.cat,case.registry)
    record=make_record(case,honest,alpha)
    positive=lambda:candidate_fixture(case,case.source,alpha,record)
    source=copy.deepcopy(case.source)
    values=[list(row) for row in source["values"]]
    values[0][0]=values[0][0]+ONE
    source["values"]=tuple(tuple(row) for row in values)
    q=list(source["q"]);q[0]=q[0]-ONE;source["q"]=tuple(q)
    # At gamma1 this unallowed plain-mode term would compensate a false claim.
    equations=derive_equations(source,case.desc,case.original.cat,case.registry,allow_plain_q=True)
    defective=make_record(case,equations,alpha,source=source)
    control("source/no-plain-blinder",positive,
            lambda:candidate_fixture(case,source,alpha,defective),"source/plain-q",
            {"source":source,"record":defective},
            mutant=lambda:candidate_fixture(case,source,alpha,defective,"allow-plain-q"))
    source=copy.deepcopy(case.source)
    points=list(source["column_points"]);points[0]=points[0]+ONE
    source["column_points"]=tuple(points)
    control("source/derived-column-point",positive,
            lambda:candidate_fixture(case,source,alpha,record),"source/column-point",
            {"source":source,"record":record})


# SPDX-License-Identifier: Apache-2.0 OR MIT
"""Append-only EXP22 runner fragment; never import/execute this file standalone.

Required existing interfaces: HERE, REPO, COMMIT, FROZEN, SEED, M, Meter,
Failure, Limit, Reject, CONTROLS, F, ZERO, clean, digest, original_fixture,
Desc, build_case, exercise, arithmetic_controls.

Root integration hooks (must be real checks, not placeholder attestations):
  runner_case_reference(case) -> {"checked": True, ...}
    Independently audits the level2 partition/row weights/combined rows/masks
    from complete inputs, without level2_groups/l2_coefficients. Once per case
    suffices because its target/challenge catalog is fixed across the four
    alphas; if implementation changes that fact, audit all four records.
  runner_diagnostic_corpus() -> {
    "negative_controls": True, "balanced_attacks": True,
    "session_replay": True, "corpus_complete": True, ...}
    Runs the full final named corpus (including existing mutations and new
    root/nonroot/source/session cases). Adds raw records via M.event/CONTROLS.

Missing hooks prevent a full validated verdict. No verification/algebra
routine is replaced by this fragment. The fragment has not been run.
"""

def original_binding_controls():
    M.arm="controls"
    case=fixed_case(k1=2,k2=2,D=1,G=0,mode="plain",P1=0,P2=0,cat=1)
    alpha=F(2)
    equations=derive_equations(case.source,case.desc,case.original.cat,case.registry)
    record=make_record(case,equations,alpha)
    positive=lambda:candidate_fixture(case,case.source,alpha,record)
    source=copy.deepcopy(case.source)
    source["root"]=case.root
    source["original_descriptor"]=case.level2_descriptor
    source["columns"]=tuple((j,tuple(case.level2_columns[j][r] for r in case.level2_physical))
                            for j in sorted(set(source["draw"])))
    frame=copy.deepcopy(record)
    frame["events"]=events_for(case.desc,source,len(record["proof"]["point"]),"live")
    control("binding/original-root-to-C_U",positive,
            lambda:candidate_fixture(case,source,alpha,frame),"source/original-root",
            {"source":source,"record":frame})
    extra=copy.deepcopy(case.source)
    j,symbols=extra["columns"][0]
    extra["columns"]=((j,symbols+(ZERO,)),)+extra["columns"][1:]
    control("binding/no-column-zip-truncation",positive,
            lambda:candidate_fixture(case,extra,alpha,record),"source/column-symbol-count",
            {"source":extra,"record":record})
    control("membership/unknown-original-root",positive,
            lambda:case.registry.verify_columns(999999,case.original_descriptor,
                     case.source["draw"],case.source["columns"]),"membership/root",
            {"root":999999})


def registry_shape_controls():
    M.arm="controls"
    case=fixed_case(k1=2,k2=2,D=1,G=0,mode="plain",P1=1,P2=1,cat=1)
    rows,masks=case.original.rows,case.original.masks
    tensor=tuple(v for row in rows for v in row)
    registry=case.registry
    good=lambda:registry.commit(case.original_descriptor,tensor,rows,masks,case.original.physical,2)
    nonbase=((U,),)+masks[1:]
    control("registry/base-mask",good,
            lambda:registry.commit(case.original_descriptor,tensor,rows,nonbase,case.original.physical,2),
            "registry/nonbase-mask",{"mask":nonbase})
    badwidth=((),)+masks[1:]
    control("registry/mask-width",good,
            lambda:registry.commit(case.original_descriptor,tensor,rows,badwidth,case.original.physical,2),
            "registry/mask-width",{"mask":badwidth})
    for bad in ((True,),(9,)):
        control("registry/physical-index/"+str(bad),good,
                lambda bad=bad:registry.commit(case.original_descriptor,tensor,rows,masks,bad,2),
                "registry/physical-index",{"physical":bad})
    control("registry/row-geometry",good,
            lambda:registry.commit(case.original_descriptor,tensor+rows[0],rows+(rows[0],),masks+(masks[0],),case.original.physical,2),
            "registry/row-count",{"row_count":len(rows)+1})
    scalar=F(3)
    try:
        scalar.a=4
    except AttributeError:
        refused=True
    else:
        refused=False
    M.judge(refused and scalar==F(3),"registry/immutable-scalar")


def remaining_parameter_controls():
    M.arm="controls"
    case=fixed_case()
    alpha=F(2)
    equations=derive_equations(case.source,case.desc,case.original.cat,case.registry)
    record=make_record(case,equations,alpha)
    positive=lambda:candidate_fixture(case,case.source,alpha,record)
    for axis in ("group","coefficient"):
        control("weight/wrong-"+axis+"-point",positive,
                lambda axis=axis:candidate_fixture(case,case.source,alpha,record,"wrong-weight-"+axis),
                "batch/final",{"record":record,"axis":axis})
    altered=copy.deepcopy(record)
    cb,rb=altered["opening"]["split"]
    altered["opening"]["split"]=(cb+1,rb-1)
    control("level2/row-column-split",positive,
            lambda:candidate_fixture(case,case.source,alpha,altered),"level2/split",{"record":altered})
    # Generate the exact wrong t0=1 row/mask combination, retaining the
    # prescribed public target weights. This must not pass as the honest U.
    original=case.original
    coefficients=[]
    row_point=catalog(original.cat,bits(original.data_cube),0)
    for row in range(len(original.rows)):
        coefficients.append((ONE-gamma(original.cat))*eq_boolean(row_point,row)
                            if row<original.data_cube else ZERO)
    changed=[list(row) for row in original.Urows]
    for column in range(case.desc.k1):
        changed[0][column]=changed[0][column]+sum((c*row[column] for c,row in zip(coefficients,original.rows)),ZERO)
    wrong=build_case(original,case.desc,changed)
    source=copy.deepcopy(wrong.source)
    masks=list(source["M"])
    masks[0]=tuple(masks[0][j]+sum((c*row[j] for c,row in zip(coefficients,original.masks)),ZERO)
                   for j in range(case.desc.P1))
    source["M"]=tuple(masks)
    _classified_source_case(wrong,source,"wrong-t0-coefficient-one")


_RUNNER_ENTERED = False
_RUNNER_EXPECTED = {
    "original_shapes": 72,
    "original_fixtures": 1152,
    "two_level_fixtures": 4608,
    "honest_batched_reductions": 18432,
}
_RUNNER_HOOK_FLAGS = (
    "negative_controls", "balanced_attacks", "session_replay", "corpus_complete",
)


class RunnerIncomplete(Exception):
    pass


def _runner_write_json(path, value):
    # All artifacts in a newly created attempt use exclusive creation. Nothing
    # overwrites an earlier attempt or republishes mutable top-level pointers.
    with path.open("x", encoding="utf-8") as handle:
        json.dump(clean(value), handle, sort_keys=True, indent=2)
        handle.write("\n")


def _runner_sha(data):
    return hashlib.sha256(data).hexdigest()


def _runner_git(*args):
    # Read-only git: no checkout/index/config/ref writes.
    result = subprocess.run(
        ["git", *args], cwd=REPO, check=False,
        stdout=subprocess.PIPE, stderr=subprocess.PIPE,
    )
    if result.returncode:
        raise RunnerIncomplete("read-only git command failed: " + " ".join(args[:2]))
    return result.stdout


def _runner_attempt():
    started = datetime.now(timezone.utc)
    prefix = started.strftime("%Y%m%dT%H%M%S.%fZ") + "-" + str(os.getpid())
    parent = HERE / "results" / "attempts"
    parent.mkdir(parents=True, exist_ok=True)
    for serial in range(10000):
        suffix = "" if serial == 0 else "-" + str(serial)
        destination = parent / (prefix + suffix)
        try:
            destination.mkdir(exist_ok=False)
        except FileExistsError:
            continue
        # This is the first artifact, before provenance checks or any audit arm.
        source = Path(__file__).resolve().read_bytes()
        with (destination / "audit.source.py").open("xb") as handle:
            handle.write(source)
        _runner_write_json(destination / "invocation.json", {
            "started_utc": started.isoformat(), "pid": os.getpid(),
            "argv": sys.argv, "source_path": str(Path(__file__).resolve()),
            "source_sha256": _runner_sha(source),
            "scope": "synthetic exact E0 only; not a performance measurement",
        })
        return destination, source
    raise RunnerIncomplete("cannot reserve a unique attempt directory")


def _runner_provenance(attempt, source):
    freeze = HERE / "results" / "README.preregistered.txt"
    pins_file = HERE / "results" / "source-pins.json"
    freeze_relative = freeze.relative_to(REPO).as_posix()
    pins_relative = pins_file.relative_to(REPO).as_posix()
    frozen_bytes = freeze.read_bytes()
    pin_bytes = pins_file.read_bytes()
    committed_freeze = _runner_git("show", COMMIT + ":" + freeze_relative)
    committed_pins = _runner_git("show", COMMIT + ":" + pins_relative)
    # Save original bytes even on a failed provenance check.
    with (attempt / "README.preregistered.txt").open("xb") as handle:
        handle.write(frozen_bytes)
    with (attempt / "source-pins.json").open("xb") as handle:
        handle.write(pin_bytes)
    pins = json.loads(pin_bytes)
    rows = []
    violations = []
    if frozen_bytes != committed_freeze or _runner_sha(frozen_bytes) != FROZEN:
        violations.append("freeze absent/different from committed preregistration")
    if pin_bytes != committed_pins:
        violations.append("source-pin manifest differs from committed manifest")
    if pins.get("preregistration_sha256") != FROZEN:
        violations.append("pin manifest does not name the frozen report")
    expected_sources = pins.get("sources_sha256")
    if not isinstance(expected_sources, dict) or len(expected_sources) != 12:
        violations.append("expected exactly the twelve committed source pins")
        expected_sources = {}
    for relative, expected in sorted(expected_sources.items()):
        target = REPO / relative
        actual = _runner_sha(target.read_bytes()) if target.is_file() else None
        allowed_note = relative == "research/ideas/IDEA-0035-combined-row-second-commitment.md"
        status = "match" if actual == expected else (
            "idea35-note-only-drift" if actual is not None and allowed_note else "blocking-drift"
        )
        rows.append({"path": relative, "expected_sha256": expected,
                     "actual_sha256": actual, "status": status})
        if status == "blocking-drift":
            violations.append("pinned source changed: " + relative)
    # The only allowed pin drift is IDEA35's separately disclosed report/status
    # update. The parent must review its diff; never grant a normative exception.
    note_path = "research/ideas/IDEA-0035-combined-row-second-commitment.md"
    if any(row["status"] == "idea35-note-only-drift" for row in rows):
        with (attempt / "IDEA-0035.current.txt").open("xb") as handle:
            handle.write((REPO / note_path).read_bytes())
        with (attempt / "IDEA-0035.preregistered.txt").open("xb") as handle:
            # Source pin describes the pre-decision-link note at COMMIT's parent.
            # Exact historical bytes are optional; do not invent a matching file.
            previous = _runner_git("show", COMMIT + "^:" + note_path)
            handle.write(previous)
        saved = _runner_sha(previous)
        if saved != expected_sources[note_path]:
            violations.append("IDEA35 historical bytes do not match its frozen source pin")
    report = {
        "registration_commit": COMMIT,
        "head": _runner_git("rev-parse", "HEAD").decode().strip(),
        "freeze_sha256": _runner_sha(frozen_bytes), "freeze_verified": not any(
            "freeze" in violation for violation in violations),
        "source_sha256": _runner_sha(source), "pins": rows,
        "allowed_drift": "IDEA35 note-only report/decision-link drift; parent review required",
        "blocking_violations": violations,
    }
    _runner_write_json(attempt / "provenance.json", report)
    M.event("provenance", report)
    if violations:
        raise RunnerIncomplete("; ".join(violations))
    return report


def _runner_environment():
    return {
        "utc": datetime.now(timezone.utc).isoformat(),
        "python": sys.version, "executable": sys.executable,
        "os": platform.platform(), "machine": platform.machine(),
        "processor": platform.processor(), "pid": os.getpid(),
        "evidence_level": "E0", "performance_claim": False,
        "prng_seed": hex(SEED),
        "prng_scope": "synthetic fixture generation only, not uniform challenges",
    }


def _runner_shapes():
    # Frozen Cartesian product: 3*2*3*2*2=72. Geometry is crossed below.
    return itertools.product(
        (1, 2, 4), (1, 3), (0, 1, 2),
        ("plain", "blinded-algebra"), (0, 1),
    )


def _runner_one_original(shape, family, catalog_index, coverage, reference_hook):
    k1, data_rows, groups, mode, mask1 = shape
    before_ops, before_judgments = M.ops, M.judgments
    M.arm = "domain/original-generation"
    original = original_fixture(k1, data_rows, groups, mode, mask1, family, catalog_index)
    original_digest = digest(original)
    coverage["original_fixtures"] += 1
    M.counts["domain/original_fixtures"] += 1
    M.event("original-fixture", {
        "shape": shape, "family": family, "catalog": catalog_index,
        "input_digest": original_digest,
        "generated_original_number": coverage["original_fixtures"],
    })
    for k2, mask2 in itertools.product((2, 4), (0, 1)):
        M.check()
        desc = Desc(k1, data_rows, groups, mode, mask1, k2, mask2)
        M.arm = "domain/two-level-generation"
        case = build_case(original, desc)
        case_start_ops, case_start_judgments = M.ops, M.judgments
        M.event("two-level-fixture-start", {
            "descriptor": desc, "family": family, "catalog": catalog_index,
            "input_digest": case.identity, "original_digest": original_digest,
        })
        if reference_hook is not None:
            M.arm = "domain/independent-level2-reference"
            reference = reference_hook(case)
            M.judge(isinstance(reference, dict) and reference.get("checked") is True,
                    "runner/independent-level2-reference-result", reference)
            coverage["independent_level2_references"] += 1
            M.counts["domain/independent_level2_references"] += 1
        else:
            reference = {"checked": False, "reason": "integration hook missing"}
        M.arm = "domain/honest-coupling"
        reductions_before = M.counts["reductions/honest"]
        acceptances_before = M.counts["root_acceptances/honest"]
        # exercise() itself traverses alpha=(0,1,2,u). Do not loop alpha again.
        residuals = exercise(case, label="honest", save=False)
        M.judge(len(residuals) > 0, "runner/nonempty-equation-list")
        for residual in residuals:
            M.judge(residual == ZERO, "runner/honest-equation-residual")
        reductions = M.counts["reductions/honest"] - reductions_before
        acceptances = M.counts["root_acceptances/honest"] - acceptances_before
        M.judge(reductions == 4, "runner/four-alpha-reductions", reductions)
        M.judge(acceptances == 4, "runner/four-honest-acceptances", acceptances)
        coverage["two_level_fixtures"] += 1
        coverage["honest_batched_reductions"] += reductions
        M.counts["domain/two_level_fixtures"] += 1
        M.counts["domain/honest_batched_reductions"] += reductions
        M.event("two-level-fixture-complete", {
            "descriptor": desc, "family": family, "catalog": catalog_index,
            "input_digest": case.identity, "alpha_catalog": (ZERO, F(1), F(2), F(0, 1)),
            "reduction_count": reductions, "honest_acceptances": acceptances,
            "independent_reference": reference,
            "physical_judgments_since_case_generation": M.judgments-case_start_judgments,
            "metered_ops_since_case_generation": M.ops-case_start_ops,
            "coverage": dict(coverage),
        })
        # No Case or registry is retained by this runner after its fixture.
        del case
        M.check()
    M.event("original-fixture-complete", {
        "shape": shape, "family": family, "catalog": catalog_index,
        "input_digest": original_digest,
        "physical_judgments": M.judgments-before_judgments,
        "metered_ops": M.ops-before_ops,
    })


def _runner_domain(full, reference_hook):
    coverage = Counter()
    if full:
        for shape in _runner_shapes():
            coverage["original_shapes"] += 1
            M.counts["domain/original_shapes"] += 1
            for family, catalog_index in itertools.product(range(4), range(4)):
                _runner_one_original(shape, family, catalog_index, coverage, reference_hook)
            M.event("shape-complete", {"shape": shape, "coverage": dict(coverage)})
            M.check()
        M.arm = "domain/coverage"
        for name, expected in _RUNNER_EXPECTED.items():
            M.judge(coverage[name] == expected, "runner/full-domain/"+name,
                    {"actual": coverage[name], "expected": expected})
        M.judge(coverage["independent_level2_references"] == 4608,
                "runner/full-domain/independent-level2-references")
    else:
        # Explicit subset of frozen cases, never a reduced full-domain verdict.
        # Covers zero-dimensional U, padded groups, nonbase points, both modes,
        # the lane-in-column and lane-in-row geometries, and P1/P2 zero/nonzero.
        selected = (
            ((1, 1, 0, "plain", 0), 0, 0),
            ((1, 3, 2, "blinded-algebra", 1), 3, 3),
            ((4, 3, 2, "blinded-algebra", 1), 3, 3),
            ((2, 3, 1, "plain", 1), 1, 1),
        )
        for shape, family, catalog_index in selected:
            coverage["original_shapes"] += 1
            M.counts["domain/original_shapes"] += 1
            _runner_one_original(shape, family, catalog_index, coverage, reference_hook)
    return dict(coverage)


def _runner_main():
    global _RUNNER_ENTERED
    if _RUNNER_ENTERED:
        raise RunnerIncomplete("one attempt per interpreter process")
    _RUNNER_ENTERED = True
    # Snapshot comes before argument/provenance checks and every math arm.
    attempt, source = _runner_attempt()
    stream = (attempt / "events.jsonl").open("x", encoding="utf-8")
    M.stream = stream
    status = "inconclusive"
    reason = "attempt did not reach judgment"
    phase = "arguments"
    full = False
    coverage = {}
    provenance = None
    corpus = {}
    arithmetic_complete = False
    coverage_complete = False
    error = None
    exit_code = 2
    previous_handler = None
    alarm_installed = False
    args_record = None
    M.event("attempt-start", {"directory": str(attempt), "argv": sys.argv})
    print("EXP22 attempt: " + str(attempt), flush=True)
    try:
        def watchdog(signum, frame):
            del signum, frame
            raise Limit("15-minute alarm watchdog")
        if hasattr(signal, "SIGALRM"):
            previous_handler = signal.signal(signal.SIGALRM, watchdog)
            # Include provenance/argument handling in the whole-attempt guard.
            remaining = max(1, int(900-(time.monotonic()-M.start)))
            signal.alarm(remaining)
            alarm_installed = True
        parser = argparse.ArgumentParser(description="EXP22 exact E0 audit; no timing claim")
        modes = parser.add_mutually_exclusive_group(required=True)
        modes.add_argument("--preflight", action="store_true")
        modes.add_argument("--all", action="store_true")
        arguments = parser.parse_args()
        full = arguments.all
        args_record = {"preflight": arguments.preflight, "all": full}
        _runner_write_json(attempt / "environment.json", _runner_environment())
        phase = "provenance"
        provenance = _runner_provenance(attempt, source)
        M.check()
        reference_hook = globals().get("runner_case_reference")
        corpus_hook = globals().get("runner_diagnostic_corpus")
        reference_hook = reference_hook if callable(reference_hook) else None
        corpus_hook = corpus_hook if callable(corpus_hook) else None
        if full and (reference_hook is None or corpus_hook is None):
            raise RunnerIncomplete("full audit requires independent-level2 and complete diagnostic hooks")
        phase = "arithmetic"
        arithmetic_controls()
        arithmetic_complete = True
        M.check()
        phase = "diagnostic-corpus"
        if corpus_hook is not None:
            corpus = corpus_hook()
            if not isinstance(corpus, dict):
                raise RunnerIncomplete("diagnostic hook must return a completion record")
            for flag in _RUNNER_HOOK_FLAGS:
                if corpus.get(flag) is not True:
                    raise RunnerIncomplete("diagnostic corpus incomplete: " + flag)
            M.event("diagnostic-corpus-complete", corpus)
        else:
            corpus = {"corpus_complete": False, "reason": "integration hook missing"}
            M.event("diagnostic-corpus-not-run", corpus)
        M.check()
        phase = "domain"
        coverage = _runner_domain(full, reference_hook)
        coverage_complete = full and all(coverage.get(k) == v for k,v in _RUNNER_EXPECTED.items())
        if full:
            # No validation is available from a partial corpus or absent oracle.
            M.judge(arithmetic_complete and coverage_complete and
                    all(corpus.get(flag) is True for flag in _RUNNER_HOOK_FLAGS),
                    "runner/full-arm-conjunction")
            phase = "final-resource-check"
            M.check()  # Must remain immediately before final validated status.
            status, reason = "validated", "all frozen finite arms completed"
        else:
            M.check()
            status, reason = "inconclusive", "preflight-only; no full-domain verdict"
        exit_code = 0
    except SystemExit as exception:
        # Even argument errors/help preserve the attempt/source; no math result.
        reason = "argument parser ended before a judged audit"
        error = {"type": "SystemExit", "code": exception.code}
        exit_code = 0 if exception.code == 0 else 2
    except Failure as exception:
        # Preserve a failed exact comparison without pretending its cause has
        # already received independent mathematical/harness classification.
        # In particular a preflight never assigns a full-domain verdict.
        status = "inconclusive"
        reason = "failed comparison; independent classification required: " + str(exception)
        error = {"type": type(exception).__name__, "message": str(exception)}
        exit_code = 1
    except (Limit, RunnerIncomplete) as exception:
        status, reason = "inconclusive", str(exception)
        error = {"type": type(exception).__name__, "message": str(exception)}
        exit_code = 2
    except Reject as exception:
        # Unexpected local refusal outside a negative/control outcome is a
        # harness/arm discrepancy, not an independently established theorem refutation.
        status, reason = "inconclusive", "unexpected unclassified local rejection"
        error = {"type": "Reject", "reason": exception.reason}
        exit_code = 2
    except Exception as exception:
        status, reason = "inconclusive", "unresolved harness exception"
        error = {"type": type(exception).__name__, "message": str(exception)}
        exit_code = 2
    finally:
        if alarm_installed:
            signal.alarm(0)
            signal.signal(signal.SIGALRM, previous_handler)
        cached = M.counts.get("cached_logical_judgments", 0)
        physical = getattr(M, "physical", M.judgments-cached)
        if not coverage:
            # Preserve partial traversal when a failure interrupts a callee
            # before it can return its local coverage record.
            coverage = {name: M.counts.get("domain/"+name, 0) for name in (
                *_RUNNER_EXPECTED, "independent_level2_references",
            )}
        report = {
            "status": status, "reason": reason, "phase": phase,
            "mode": "all" if full else "preflight-or-argument-only",
            "arguments": args_record, "attempt": str(attempt),
            "source_sha256": _runner_sha(source), "frozen_sha256": FROZEN,
            "registration_commit": COMMIT, "provenance": provenance,
            "coverage": coverage, "expected_full_coverage": _RUNNER_EXPECTED,
            "arithmetic_complete": arithmetic_complete,
            "full_domain_complete": coverage_complete, "corpus": corpus,
            "judgments": M.judgments, "physical_judgments": physical,
            "memoized_logical_judgments": cached, "metered_algebraic_operations": M.ops,
            "counts": dict(M.counts), "maxima": dict(M.maxima),
            "controls_recorded": len(CONTROLS), "error": error,
            "limits": {"judgments": 100000000, "metered_ops": 3000000000,
                       "seconds_watchdog": 900, "process_rss_bytes": 256*1024*1024},
            "evidence_level": "E0", "performance_claim": False,
            "input_reuse": "one immutable original input per family/catalog, crossed with four geometries; no runner oracle-judgment caching",
            "scope": "conditional finite algebra and obligation checks under ideal PCS/experimental coefficient basis",
            "not_established": ["full extractor", "G-4 budget", "joint ZK", "concrete PCS security",
                                "arbitrary dishonest-transcript exhaustion", "FS security",
                                "current group/layout/payload counts", "performance"],
        }
        # Summary/corpus files are exclusive within this immutable attempt.
        _runner_write_json(attempt / "summary.json", report)
        _runner_write_json(attempt / "controls.json", CONTROLS)
        M.event("attempt-terminal", report)
        stream.close()
        M.stream = None
        print(json.dumps({"status": status, "reason": reason,
                          "summary": str(attempt / "summary.json"),
                          "exit_code": exit_code}, sort_keys=True), flush=True)
    return exit_code


if __name__ == "__main__":
    raise SystemExit(_runner_main())
