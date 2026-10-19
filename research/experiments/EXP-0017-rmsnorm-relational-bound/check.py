#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0 OR MIT
"""Independent exact-integer oracle and saved-record verifier for EXP-0017."""

import argparse
from collections import Counter
from functools import lru_cache
from hashlib import sha256
from itertools import product
import json
from math import isqrt
from pathlib import Path
import subprocess

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
RESULTS = HERE / "results"
P = 2**64 - 2**32 + 1
SEED = 0x7E55_E7A0_0011_0017


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"))


def binary_sqrt(n):
    assert n >= 0
    lo, hi = 0, 1 << ((n.bit_length() + 1) // 2)
    while lo < hi:
        mid = (lo + hi + 1) // 2
        if mid * mid <= n:
            lo = mid
        else:
            hi = mid - 1
    return lo


def width(lo, hi):
    return 1 + max(max(0, hi).bit_length(), max(0, ~lo).bit_length())


def precision(d, e):
    assert e > 0
    return max(k for k in range(32) if (d << (2 * k)) // e < 2**64)


@lru_cache(maxsize=None)
def extrema(n, e, lo, hi, step):
    def endpoint(top):
        best, witness, rho = 0, 0, isqrt(n // e)
        for x in range(0, top + 1, step):
            r = isqrt(n // (x * x + e))
            if x * r > best:
                best, witness, rho = x * r, x, r
        assert binary_sqrt(n // (witness**2 + e)) == rho
        assert rho**2 * (witness**2 + e) <= n < (rho + 1)**2 * (witness**2 + e)
        return dict(magnitude=best, x=witness, rho=rho, last_x=top, step=step)
    negative, positive = endpoint(-lo), endpoint(hi)
    a, b = -negative["magnitude"], positive["magnitude"]
    return dict(negative=negative, positive=positive, lo=a, hi=b, width=width(a, b))


def expected_selection(lo, hi, shifts):
    b = width(lo, hi)
    cb = (len(shifts) - 1).bit_length()
    rq = lambda bits, rs: dict(tag="Requant", b=bits, shifts=rs, lo=-128, hi=127)
    if b + cb <= 32:
        return dict(tag="Requant1", split=None, middle=None, commits_per_element=1,
                    lookups_per_element=1, kinds=[rq(b, shifts)])
    assert b <= 62
    least = b + cb - 32
    most = min(min(shifts) - 1, 32)
    assert least <= most and len(shifts) * 2**b < P
    s = max(least, min((b + cb + 1) // 2, most))
    return dict(tag="RequantSplit", split=s, middle=None, commits_per_element=2,
                lookups_per_element=2,
                kinds=[dict(tag="Range", bits=s), rq(b - s, [r - s for r in shifts])])


def rows(kind):
    tag = kind["tag"]
    if tag == "Range":
        return 2**kind["bits"]
    if tag in ("Requant", "Silu"):
        return 2 ** (kind["b"] + (len(kind["shifts"]) - 1).bit_length())
    if tag == "Exp":
        return 2**kind["key_bits"]
    if tag == "Rope":
        return 2 ** (kind["max_context"] * kind["d_head"] // 2 - 1).bit_length()
    raise AssertionError(tag)


def check_gadget(got, expected):
    for key in ("tag", "split", "middle", "commits_per_element", "lookups_per_element"):
        assert got[key] == expected[key], (key, got, expected)
    assert [t["kind"] for t in got["tables"]] == expected["kinds"]
    for table in got["tables"]:
        assert table["rows"] == rows(table["kind"]) == 2**table["log2_rows"]


def tally(records, positions, checkpoint):
    """Count every recorded gadget independently, including unchanged row gadgets."""
    t = dict(lookup_outputs=0,split_limbs=0,row_scalars=0,one_hot=0,checkpoints=checkpoint)
    lookups = 0
    for nd in records:
        e = nd["elements"] // positions
        assert e * positions == nd["elements"]
        g = nd["gadget"]
        tag = g["tag"]
        if tag in ("Lookup","Requant1","RequantSplit","RequantWide"):
            t["lookup_outputs"] += e
            parts = 0 if tag in ("Lookup","Requant1") else 1 if tag == "RequantSplit" else len(g["tables"])-2
            t["split_limbs"] += parts*e
            lookups += (parts+1+(tag == "RequantWide"))*e
        elif tag == "RsqrtRow":
            t["row_scalars"] += 20*e
            lookups += 30*e
        elif tag == "SoftmaxRow":
            assert e % positions == 0
            real_rows = e//positions
            # sigma16, limb width16: c,R plus two three-limb brackets.
            t["row_scalars"] += 6*real_rows
            lookups += 7*real_rows
        elif tag == "OneHot":
            t["one_hot"] += e
        else:
            raise AssertionError(tag)
    t["total"] = sum(t.values())
    return dict(committed=t,lookups=lookups)


def inspect_compiler(compiler):
    assert compiler["schema"] == "EXP-0017/compiler/1"
    assert len(compiler["cases"]) == 6
    committed = json.loads((ROOT / "docs/perf/v0.2/commitment-report-405b-servable-c2-b32.json").read_text())
    seen = set()
    summaries = []
    for c in compiler["cases"]:
        d, mode, e, k = c["d"], c["mode"], c["epsilon"], c["kappa"]
        assert (d, mode) not in seen
        seen.add((d, mode))
        assert d in (4096, 16384) and mode in (16, 30, 31)
        assert c["layers"] == (32 if d == 4096 else 126)
        assert e == (d if mode == 31 else (d * 2**14 + 99999) // 100000)
        assert k == (16 if mode == 16 else precision(d, e)) == mode
        n = d << (2 * k)
        assert int(c["N"]) == n
        root_bits = isqrt(n // e).bit_length()
        assert root_bits <= 32
        assert len(c["norms"]) == 2 * c["layers"] + 1
        if (d, mode) == (16384, 30):
            for key in ("counts", "per_layer_per_token", "per_token", "tables", "multiplicities", "estimates"):
                assert c["baseline"][key] == committed[key], key
        norm_map = {v["node"]: v for v in c["norms"]}
        first_end = next(nd["node"] for nd in c["all_gadgets"] if nd["point"] == "Layer { layer: 0, site: MlpResidual }")
        first_start = next(nd["node"] for nd in c["all_gadgets"] if nd["gadget"]["tag"] == "RsqrtRow")
        layer_nodes = [nd for nd in c["all_gadgets"] if first_start<=nd["node"]<=first_end]
        for records, report_key, checkpoint in ((c["all_gadgets"],"per_token",d),(layer_nodes,"per_layer_per_token",0)):
            counted = tally(records,c["positions"],checkpoint)
            for key in ("committed","lookups"):
                assert counted[key] == c["baseline"][report_key][key], (d,mode,report_key,key,counted,c["baseline"][report_key])
        change_c = change_k = layer_c = layer_k = 0
        uniform = []
        for index, norm in enumerate(c["norms"]):
            a, b = norm["residual_range"]
            assert (a, b) == ((-254, 254) if index == 0 else (-2**19, 2**19 - 1))
            assert norm["first_embedding"] == (index == 0)
            assert norm["step"] == (2 if index == 0 else 1)
            assert norm["root_range"] == [0, 2**root_bits - 1]
            assert norm["key_range"] == [a * (2**root_bits - 1), b * (2**root_bits - 1)]
            assert norm["baseline_width"] == width(*norm["key_range"])
            assert norm["shifts"] == [k - 3]
            assert d * max(a*a, b*b) + e < 2**55
            assert max(map(abs, norm["key_range"])) < (P - 1) // 2
            ex = extrema(n, e, a, b, norm["step"])
            assert norm["extrema"] == ex
            universal = isqrt(n - e) if n >= e else 0
            assert max(-ex["lo"], ex["hi"]) <= universal
            assert max(-ex["lo"], ex["hi"])**2 < n
            check_gadget(norm["baseline"], expected_selection(*norm["key_range"], norm["shifts"]))
            check_gadget(norm["candidate"], expected_selection(ex["lo"], ex["hi"], norm["shifts"]))
            dc = (norm["candidate"]["commits_per_element"] - norm["baseline"]["commits_per_element"]) * d
            dk = (norm["candidate"]["lookups_per_element"] - norm["baseline"]["lookups_per_element"]) * d
            change_c += dc
            change_k += dk
            if index < 2:
                layer_c += dc
                layer_k += dk
            if index:
                uniform.append((ex["width"], norm["candidate"]["tag"]))
        # Rebuild the complete unique table set from initial public tables and all gadget uses.
        unique = {}
        def add(table):
            assert table["rows"] == rows(table["kind"])
            unique.setdefault(canonical(table["kind"]), table["rows"])
        for t in c["initial_tables"]:
            add(t)
        for nd in c["all_gadgets"]:
            g = norm_map[nd["node"]]["candidate"] if nd["node"] in norm_map else nd["gadget"]
            for t in g["tables"]:
                add(t)
        actual = {canonical(t["kind"]): t["rows"] for t in c["candidate_tables"]}
        assert len(actual) == len(c["candidate_tables"])
        assert actual == unique
        for arm in ("baseline", "candidate"):
            ts = c[arm + "_tables"]
            assert len({canonical(t["kind"]) for t in ts}) == len(ts)
            for t in ts:
                assert t["rows"] == rows(t["kind"])
            assert sum(t["rows"] for t in ts) == c[arm]["multiplicities"]["total"] == c[arm]["counts"]["table_rows"]
        assert c["candidate"]["counts"]["commits_per_token"] == c["baseline"]["counts"]["commits_per_token"] + change_c
        assert c["candidate"]["counts"]["lookups_per_token"] == c["baseline"]["counts"]["lookups_per_token"] + change_k
        assert c["candidate"]["per_layer_per_token"]["committed"]["total"] == c["baseline"]["per_layer_per_token"]["committed"]["total"] + layer_c
        assert c["candidate"]["per_layer_per_token"]["lookups"] == c["baseline"]["per_layer_per_token"]["lookups"] + layer_k
        assert len(set(uniform)) == 1
        summaries.append(dict(d=d, mode=mode, kappa=k, epsilon=e, root_width=root_bits,
                              universal_magnitude=isqrt(n - e), first=c["norms"][0]["extrema"],
                              ordinary=c["norms"][1]["extrema"],
                              baseline_width=c["norms"][1]["baseline_width"],
                              baseline_split=c["norms"][1]["baseline"]["split"],
                              candidate_split=c["norms"][1]["candidate"]["split"],
                              norm_count=len(c["norms"]), C_delta=change_c, K_delta=change_k,
                              first_layer_C_delta=layer_c, first_layer_K_delta=layer_k,
                              C=int(c["candidate"]["counts"]["commits_per_token"]),
                              K=int(c["candidate"]["counts"]["lookups_per_token"]),
                              M_before=int(c["baseline"]["counts"]["table_rows"]),
                              M_after=sum(unique.values()),
                              M_delta=sum(unique.values())-int(c["baseline"]["counts"]["table_rows"])))
        for key, norm in (("first_witnesses",c["norms"][0]),("ordinary_witnesses",c["norms"][1])):
            witnesses = []
            for sign, endpoint in ((-1,"negative"),(1,"positive")):
                point = norm["extrema"][endpoint]
                x, r = sign*point["x"], point["rho"]
                u = x*x+e
                witnesses.append(dict(dimension=d,default=0,nonzero={"0":x},rho=r,u=u,
                                      lower_slack=n-r*r*u,upper_slack=(r+1)**2*u-n-1))
            summaries[-1][key] = witnesses
    assert compiler["root_controls"][0]["accepted"] is False
    assert compiler["root_controls"][1]["accepted"] is False
    assert compiler["root_controls"][2]["accepted"] is True
    assert isqrt((4096 << 52) // 1).bit_length() == 33
    return summaries


def exhaust():
    tally = Counter()
    digest = sha256()
    representative = []
    for d in range(1, 5):
        for row in product(range(-4, 4), repeat=d):
            s = sum(x*x for x in row)
            for e in range(1, 5):
                for k in range(5):
                    n, u = d << (2*k), s+e
                    rho = isqrt(n // u)
                    assert binary_sqrt(n // u) == rho
                    accepted = []
                    for r in range(isqrt(n // e) + 2):
                        tally["candidate_roots"] += 1
                        if r*r*u <= n < (r+1)**2*u:
                            accepted.append(r)
                    assert accepted == [rho]
                    bound = isqrt(n-e) if n >= e else 0
                    ex = extrema(n, e, -4, 3, 1)
                    for x in row:
                        z = rho*x
                        assert abs(z) <= bound and ex["lo"] <= z <= ex["hi"]
                    digest.update(canonical([d,row,e,k,rho]).encode())
                    tally["rows"] += 1
                    tally["keys"] += d
                    if len(representative) < 12:
                        representative.append(dict(row=list(row),e=e,k=k,rho=rho))
    # Dropping an inequality permits a wrong root; dropping e>0 breaks existence at u=0.
    controls = dict(
        no_lower=dict(N=4,u=1,rho=3,upper=True,lower=False),
        no_upper=dict(N=4,u=1,rho=0,upper=False,lower=True),
        nonpositive_epsilon=dict(epsilon0_zero_row="no finite maximal root; upper bracket impossible",negative_epsilon="not encodable as public u64; integer oracle rejects e<=0"),
        padding=dict(psi=0,u=0,roots_without_pin=[0,1],roots_with_pin=[0]),
    )
    for v in (controls["no_lower"], controls["no_upper"]):
        assert (v["rho"]**2*v["u"]<=v["N"]) == v["lower"]
        assert (v["N"]<(v["rho"]+1)**2*v["u"]) == v["upper"]
    for e in (0, -1):
        try:
            precision(4, e)
        except AssertionError:
            pass
        else:
            raise AssertionError("nonpositive e accepted")
    assert all(r*r*0 <= 0 and 0 <= (r+1)**2*0 for r in (0,1))
    assert [r for r in (0,1) if r == 0] == [0]
    return dict(coverage=dict(tally), SHA256=digest.hexdigest(), representative=representative, controls=controls)


class SplitMix64:
    def __init__(self, seed):
        self.state = seed
    def next(self):
        mask = 2**64 - 1
        self.state = (self.state + 0x9E3779B97F4A7C15) & mask
        x = self.state
        x = ((x ^ (x >> 30)) * 0xBF58476D1CE4E5B9) & mask
        x = ((x ^ (x >> 27)) * 0x94D049BB133111EB) & mask
        return x ^ (x >> 31)


def rq(z, shift):
    return max(-128, min(127, (z + 2**(shift-1)) // 2**shift))


def synthetic(compiler):
    rng = SplitMix64(SEED)
    keys = {d: [] for d in (4,16,128,4096,16384)}
    counts = Counter()
    row_hash = sha256()
    for d, count in ((4,1000),(16,1000),(128,1000),(4096,100),(16384,100)):
        e = (d*2**14+99999)//100000
        k = precision(d,e)
        n = d << (2*k)
        bound = isqrt(n-e)
        for index in range(count):
            if d <= 128:
                row = [rng.next()%256-128 for _ in range(d)]
                if index == 0: row = [0]*d
                if index == 1: row = [-128]*d
                if index == 2: row = [127]*d
                if index == 3: row = [-128 if i%2 else 127 for i in range(d)]
            elif index % 4 == 0:
                row = [-2**19 if i%2 else 2**19-1 for i in range(d)]
            elif index % 4 == 1:
                row = [0]*d
                row[rng.next()%d] = -2**19
            elif index % 4 == 2:
                row = [0]*d
                row[rng.next()%d] = 2**19-1
            else:
                row = [0 if rng.next()%16 else rng.next()%2**20-2**19 for _ in range(d)]
            u = sum(x*x for x in row)+e
            r = isqrt(n//u)
            assert r == binary_sqrt(n//u) and r*r*u<=n<(r+1)**2*u
            zs = [x*r for x in row]
            assert all(abs(z)<=bound for z in zs)
            row_hash.update(canonical([d,index,row,r]).encode())
            counts[str(d)] += 1
            keys[d].extend(zs)
    groups = []
    for c in compiler["cases"]:
        for category, norm in (("ordinary",c["norms"][1]),("embedding",c["norms"][0])):
            lo, hi = norm["extrema"]["lo"], norm["extrema"]["hi"]
            shift = c["kappa"]-3
            test = {lo,lo+1,hi-1,hi,0,1,-1}
            for y in range(-130,131):
                tie = y*2**shift - 2**(shift-1)
                test.update(z for z in (tie-1,tie,tie+1) if lo<=z<=hi)
            if category == "ordinary" and c["mode"] == 30:
                test.update(keys[c["d"]])
            test = sorted(z for z in test if lo<=z<=hi)
            groups.append(dict(name=f'{c["d"]}/{c["mode"]}/{category}',lo=lo,hi=hi,shift=shift,keys=test))
    for d in (4,16,128):
        e = (d*2**14+99999)//100000
        k = precision(d,e)
        ex = extrema(d<<(2*k),e,-128,127,1)
        groups.append(dict(name=f'{d}/synthetic8bit',lo=ex["lo"],hi=ex["hi"],shift=k-3,keys=sorted(set(keys[d]))))
    return groups, dict(seed=SEED,rows=dict(counts),row_SHA256=row_hash.hexdigest())


def decide(summaries):
    current = next(s for s in summaries if (s["d"],s["mode"]) == (16384,30))
    if current["ordinary"]["width"] > 32 and current["candidate_split"] is not None and current["first_layer_C_delta"] == current["first_layer_K_delta"] == 0:
        return "refuted"
    if current["candidate_split"] is None and current["first_layer_C_delta"] == current["first_layer_K_delta"] == -32768:
        return "validated"
    return "inconclusive"


def verify(audit, compiler):
    summaries = inspect_compiler(compiler)
    assert audit["summaries"] == summaries
    assert audit["verdict"] == decide(summaries)
    assert audit["exhaustion"] == exhaust()
    groups, coverage = synthetic(compiler)
    assert audit["synthetic"] == coverage
    requests = json.loads((RESULTS / "lookup-requests.json").read_text())
    responses = json.loads((RESULTS / "lookup-responses.json").read_text())
    assert requests == groups and len(requests) == len(responses) == 15
    checked = 0
    for request, response in zip(requests,responses):
        assert request["name"] == response["name"]
        assert response["values"] == [rq(z,request["shift"]) for z in request["keys"]]
        check_gadget(response["gadget"],expected_selection(request["lo"],request["hi"],[request["shift"]]))
        checked += len(request["keys"])
    assert audit["requant_keys"] == checked
    assert audit["compiler_SHA256"] == sha256((RESULTS / "compiler.json").read_bytes()).hexdigest()
    return summaries


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--audit", type=Path)
    args = parser.parse_args()
    compiler = json.loads((RESULTS / "compiler.json").read_text())
    if args.audit:
        audit = json.loads(args.audit.read_text())
    else:
        summaries = inspect_compiler(compiler)
        groups, coverage = synthetic(compiler)
        (RESULTS / "lookup-requests.json").write_text(json.dumps(groups)+"\n")
        exe = HERE / "target/release/normbound"
        result = subprocess.run([str(exe),"lookup"],input=json.dumps(groups),text=True,capture_output=True,check=True)
        (RESULTS / "lookup-responses.json").write_text(result.stdout)
        audit = dict(schema="EXP-0017/audit/1",summaries=summaries,verdict=decide(summaries),
                     exhaustion=exhaust(),synthetic=coverage,requant_keys=sum(len(g["keys"]) for g in groups),
                     compiler_SHA256=sha256((RESULTS / "compiler.json").read_bytes()).hexdigest())
        (RESULTS / "audit.json").write_text(json.dumps(audit,indent=2)+"\n")
    verify(audit,compiler)
    print(json.dumps(dict(verdict=audit["verdict"],summaries=audit["summaries"],coverage=audit["exhaustion"]["coverage"],requant_keys=audit["requant_keys"]),indent=2))


if __name__ == "__main__":
    main()
