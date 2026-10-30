# SPDX-License-Identifier: Apache-2.0 OR MIT
"""RFC-0076 public selection oracle, written before either production implementation.

Only Python integers and the normative RFCs are used. Split points are searched
by table cost, rather than implementing the RFC's clamp expression.
"""
import json
import random
from pathlib import Path

P = 2**64 - 2**32 + 1


def ceil_log(value):
    return (value - 1).bit_length()


def tables(family, b, s, t, shifts, lo, hi):
    def rq(bits, rs, low=lo, high=hi):
        return ["requant", bits, rs, low, high]
    if family == "one":
        return [rq(b, shifts)]
    if family == "split":
        return [["range", s], rq(b - s, [r - s for r in shifts])]
    k = b - s - t
    return ([["range", s]] if s else []) + [
        ["range", t], ["range", k], rq(k + 1, [1], -2, 1),
        rq(t + 2, [r - s for r in shifts])]


def plan(family, b, s, t, node, maximum):
    shifts, lo, hi = node["shifts"], node["lo"], node["hi"]
    if not (1 <= b <= 62 and -(2**31) <= lo < 0 <= hi < 2**31):
        return None
    if family == "split" and not (1 <= s < b and s < min(shifts)
                                   and len(shifts) * 2**b < P):
        return None
    if family == "wide" and not (0 <= s < min(shifts) and t >= 1 and b-s-t >= 1):
        return None
    requests = tables(family, b, s, t, shifts, lo, hi)
    for row in requests:
        if row[0] == "range":
            if not (1 <= row[1] <= maximum):
                return None
        elif not (1 <= row[1] and row[1] + ceil_log(len(row[2])) <= maximum
                  and all(1 <= r <= 63 for r in row[2])):
            return None
    return dict(family=family, bits=b, split=s, middle=t, requests=requests)


def best_split(node, maximum, widened):
    b = max(node["bits"], 2) if widened else node["bits"]
    feasible = []
    for s in range(1, b):
        p = plan("split", b, s, 0, node, maximum)
        if p is not None:
            # RFC-0028: favor the larger low table at a tie.
            feasible.append((2**s + 2**(b+ceil_log(len(node["shifts"]))-s), -s, p))
    return min(feasible, key=lambda x: x[:2])[2] if feasible else None


def wide(node, maximum, force_zero=False, widened=False):
    s = 0 if force_zero else min(min(node["shifts"])-1, maximum)
    t = max(node["shifts"])-s+ceil_log(max(abs(node["lo"]), abs(node["hi"]), 1))
    b = max(node["bits"], s+t+1) if widened else node["bits"]
    return plan("wide", b, s, t, node, maximum)


def legacy(node, maximum):
    if not (1 <= len(node["shifts"]) <= 4
            and all(1 <= r <= 63 for r in node["shifts"])):
        return None
    return (plan("one", node["bits"], 0, 0, node, maximum)
            or best_split(node, maximum, False) or wide(node, maximum))


def select(nodes, maximum):
    old = [legacy(n, maximum) for n in nodes]
    if any(p is None for p in old):
        return dict(error="legacy", layer=old.index(None))
    classes = [len(n["shifts"]) > 1 for n in nodes]
    if any(c != classes[0] for c in classes):
        return dict(error="template", layer=next(i for i,c in enumerate(classes) if c != classes[0]))
    signatures = [(p["family"], p["split"] > 0 if p["family"] == "wide" else False) for p in old]
    if all(s == signatures[0] for s in signatures):
        return dict(promoted=False, plans=old)
    split = [best_split(n, maximum, True) for n in nodes]
    if all(p is not None for p in split):
        return dict(promoted=True, plans=split)
    force_zero = any(min(n["shifts"]) == 1 for n in nodes)
    result = [wide(n, maximum, force_zero, True) for n in nodes]
    if any(p is None for p in result):
        return dict(error="wide", layer=result.index(None))
    return dict(promoted=True, plans=result)


def main():
    def node(bits, shifts, lo=-128, hi=127):
        return dict(bits=bits, shifts=shifts, lo=lo, hi=hi)
    cases = []
    def add(name, maximum, nodes):
        cases.append(dict(name=name, maximum=maximum, nodes=nodes, expected=select(nodes, maximum)))
    add("unchanged-one", 32, [node(16,[12]), node(20,[4])])
    add("unchanged-split-different-widths", 16, [node(28,[16]), node(30,[18])])
    add("one-split", 16, [node(12,[12]), node(28,[16])])
    add("one-wide-zero-low", 16, [node(1,[1],-2,1), node(17,[1],-2,1)])
    add("split-wide-real-shifts", 32, [node(50,[2],-2**19,2**19-1),node(50,[25],-2**19,2**19-1)])
    add("mixed-low-presence", 24, [node(40,[1]), node(40,[2])])
    add("wide-top-one", 16, [node(1,[1],-2,1), node(17,[1],-2,1)])
    add("four-class-wrap", 32, [node(62,[31,32,33,34]), node(60,[31,32,33,34])])
    add("four-class-split-no-wrap-bound", 32, [node(62,[33,34,35,36]), node(60,[33,34,35,36])])
    add("wide-mixed-low-valid", 24, [node(30,[1]),node(30,[2])])
    add("minimum-split-envelope", 8, [node(1,[2],-1,1), node(12,[12],-1,1)])
    add("wide-envelope-past-62", 32, [node(32,[33],-2**29,2**29-1), node(62,[25])])
    add("class-order", 32, [node(40,[4,2,3]),node(38,[12,10,11])])
    add("mixed-class-refusal", 32, [node(20,[10]),node(20,[10,11])])
    add("legacy-refusal", 16, [node(62,[63]),node(28,[1])])
    rng=random.Random(7617)
    for i in range(2048):
        maximum=rng.choice([8,12,16,20,24,28,30,31,32])
        count=rng.choice([1,2,3,4])
        nodes=[node(rng.randint(1,62),rng.sample(range(1,40),count),
                    -2**rng.choice([0,7,15,19]),2**rng.choice([0,7,15,19])-1)
               for _ in range(rng.choice([1,2,3,5]))]
        add(f"seeded-{i}",maximum,nodes)
    # Dedicated exhaustive small widths retain degenerate-width and tie boundaries.
    for b in range(1,17):
        for r in range(1,9):
            add(f"small-{b}-{r}",8,[node(b,[r],-1,1),node(12,[1],-1,1)])
    out=dict(rfc="RFC-0076",oracle="Python integers; split search by cost",cases=cases)
    (Path(__file__).parent/'cases.json').write_text(
        '{\n  \"rfc\": \"RFC-0076\",\n  \"oracle\": \"Python integers; split search by cost\",\n  \"cases\": [\n'
        + ',\n'.join('    '+json.dumps(c,separators=(',', ':')) for c in out['cases'])
        + '\n  ]\n}\n')
    promoted=[c for c in cases if c['expected'].get('promoted')]
    assert any(p['bits']>n['bits'] for c in promoted for n,p in zip(c['nodes'],c['expected']['plans']))
    assert any(p['family']=='wide' and p['bits']-p['split']-p['middle']==1 for c in promoted for p in c['expected']['plans'])
    print(f"{len(cases)} cases, {len(promoted)} promoted groups")


if __name__ == '__main__':
    main()
