#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0 OR MIT
"""RFC-0006 range oracle, written before the independent verifier range implementation.

Input graphs are the independent RFC-0044 lowering oracle's public metadata, never Rust output.
All arithmetic uses Python integers; only the specified i128 interval boundary is imposed.

anchor.json applies the same analysis, with RFC-0066 'Ranges' and RFC-0006 'Committed graph
inputs', to the SHA-256 graphs of the independent anchor-graphs oracle, and to small graphs that
each break one of those rules once. Its part of this script was written from RFC-0006 and
RFC-0066 alone, by an author who read neither crates/ nor verifier/crates/.
"""
import json
import math
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
LIMIT = 1 << 127


def bounded(lo, hi):
    return [lo, hi] if -LIMIT <= lo <= hi < LIMIT else None


def scale(a, n):
    return bounded(a[0] * n, a[1] * n) if a is not None else None


def product(a, b):
    if a is None or b is None:
        return None
    ends = [x * y for x in a for y in b]
    return bounded(min(ends), max(ends))


def analyze(case):
    ranges = {}
    shapes = {}
    producers = {}
    nodes = {n[0]: n for n in case['nodes']}
    tables = {t[0]: t[1] for t in case['tables']}
    constants = case['constants']
    for tid, _, shape, _, _, lo, hi in case['inputs']:
        ranges[tid] = [lo, hi]
        shapes[tid] = shape
    facts = []
    for node in case['nodes']:
        nid, op, param, inputs, outputs, _ = node
        a = ranges[inputs[0]]
        result = None
        fact = None
        if op == 'OneHot':
            result = [0, 1]
        elif op == 'MatMulWeight':
            parent = nodes.get(producers.get(inputs[0]))
            if parent is not None and parent[1] == 'OneHot':
                result = [-127, 127]
                fact = 'one-hot'
            elif a is not None:
                fan_in = shapes[inputs[1]][-1]
                maximum = 127 * fan_in * max(abs(x) for x in a)
                result = bounded(-maximum, maximum)
        elif op == 'ConstMul':
            c = constants[param['c']][0]
            result = product(a, [c, c])
        elif op == 'Add':
            b = ranges[inputs[1]]
            if a is not None and b is not None:
                result = bounded(a[0] + b[0], a[1] + b[1])
        elif op == 'SumSquares':
            if a is not None:
                square_lo = 0 if a[0] <= 0 <= a[1] else min(x * x for x in a)
                result = scale(bounded(square_lo, max(x * x for x in a)), shapes[inputs[0]][param['axis']])
        elif op == 'RsqrtRow':
            root = math.isqrt((param['dim'] << (2 * param['kappa'])) // param['eps_int'])
            result = [0, (1 << max(1, root.bit_length())) - 1]
        elif op == 'Requant':
            result = [param['lo'], param['hi']]
        elif op in ('Reshape', 'Transpose', 'Expand'):
            result = a
        elif op == 'Rotate':
            if a is not None:
                maximum = (1 << (tables[param['rope']]['Rope']['bits'] + 1)) * max(abs(x) for x in a)
                result = bounded(-maximum, maximum)
        elif op == 'MatMulAct':
            axis = -1 if param['transpose_rhs'] else -2
            result = scale(product(a, ranges[inputs[1]]), shapes[inputs[1]][axis])
        elif op == 'Mul':
            result = product(a, ranges[inputs[1]])
            matmul = nodes.get(producers.get(inputs[0]))
            softmax = nodes.get(producers.get(inputs[1]))
            if (matmul is not None and softmax is not None
                    and matmul[1] == 'MatMulAct' and not matmul[2]['transpose_rhs']
                    and softmax[1] == 'SoftmaxRow'
                    and matmul[3][0] == softmax[4][0][0]
                    and inputs[1] == softmax[4][1][0]):
                values = ranges[matmul[3][1]]
                if values is not None:
                    mass = 1 << (2 * softmax[2]['sigma'])
                    result = bounded(mass * min(values[0], 0), mass * max(values[1], 0))
                    fact = 'softmax-mass'
        elif op == 'SoftmaxRow':
            result = [0, 1 << param['sigma']]
        elif op == 'TableMap':
            assert 'Silu' in tables[param['table']]
            result = [-128, 127]
        else:
            raise AssertionError(op)
        for j, (tid, shape, _) in enumerate(outputs):
            ranges[tid] = ([0, (1 << (param['sigma'] + 1)) - 1]
                           if op == 'SoftmaxRow' and j == 1 else result)
            producers[tid] = nid
            shapes[tid] = shape
        if fact is not None:
            facts.append([nid, fact])
    assert len(ranges) == case['tensors']
    return {'model': case['model'], 'n': case['n'],
            'ranges': [ranges[i] for i in range(case['tensors'])], 'facts': facts}


def main():
    cases = json.loads((ROOT / 'spec/vectors/lowering/lowering.json').read_text())['cases']
    data = {'description': 'RFC-0006 inclusive i128 interval ranges, from independent RFC-0044 graphs; null means unbounded.',
            'cases': [analyze(case) for case in cases]}
    assert all(any(f[1] == 'one-hot' for f in c['facts']) for c in data['cases'])
    assert all(any(f[1] == 'softmax-mass' for f in c['facts']) for c in data['cases'])
    Path(__file__).with_name('ranges.json').write_text(json.dumps(data, indent=2) + '\n')
    write_anchor()



# ----- The anchor circuits (RFC-0066 'Ranges', RFC-0006 'Committed graph inputs') -------------

P = 2**64 - 2**32 + 1
HALF = (P - 1) // 2  # the centered field range is open, (-HALF, HALF): RFC-0006 'True-integer rule'
MOVES = ('Reshape', 'Slice', 'Transpose', 'GatherPublic', 'Expand')


def size_of(shape):
    out = 1
    for n in shape:
        out *= n
    return out


def moved(op, param, shape, elements, constants):
    """The source elements of a move's output, row-major, from its input's: Reshape keeps
    them, Transpose permutes them, Expand repeats them, Slice and GatherPublic keep the rows they
    select."""
    if op == 'Reshape':
        return elements
    strides = [size_of(shape[a + 1:]) for a in range(len(shape))]
    if op == 'Transpose':
        perm = param['perm']
        out = []

        def walk(axis, offset):
            if axis == len(perm):
                out.append(elements[offset])
                return
            for i in range(shape[perm[axis]]):
                walk(axis + 1, offset + i * strides[perm[axis]])
        walk(0, 0)
        return out
    if op == 'Slice':
        axis, start, length = param['axis'], param['start'], param['len']
        outer, n, inner = size_of(shape[:axis]), shape[axis], size_of(shape[axis + 1:])
        return [elements[(o * n + start + j) * inner + i]
                for o in range(outer) for j in range(length) for i in range(inner)]
    if op == 'GatherPublic':
        row = size_of(shape[1:])
        return [e for i in constants[param['index']] for e in elements[i * row:(i + 1) * row]]
    if op == 'Expand':
        axis, length = param['axis'], param['len']
        outer, inner = size_of(shape[:axis]), size_of(shape[axis:])
        return [elements[o * inner + i] for o in range(outer) for _ in range(length) for i in range(inner)]
    raise AssertionError(op)


def analyze_graph(case):
    """Ranges of every tensor, and the first refusal in node order of RFC-0066 'Ranges' and
    RFC-0006's domain table, then of RFC-0006 'Committed graph inputs' (#527), or None."""
    ranges, shapes, origin = {}, {}, {}
    constants = case['constants']
    committed = []
    for tid, _, shape, kind, _, lo, hi in case['inputs']:
        ranges[tid], shapes[tid] = [lo, hi], shape
        if kind == 'A':
            committed.append(tid)
            origin[tid] = (tid, list(range(size_of(shape))))
    covered = {t: set() for t in committed}
    refusal, checked = None, {'BitDecompose': 0, 'BitFn': 0, 'AssertEqual': 0}

    def refuse(code, nid):
        nonlocal refusal
        if refusal is None:
            refusal = {'code': code, 'node': nid}

    for nid, op, param, inputs, outputs, _ in case['nodes']:
        rs = [ranges[t] for t in inputs]
        a = rs[0] if rs else None
        if op in MOVES:
            result = a
            if inputs[0] in origin:
                source, elements = origin[inputs[0]]
                origin[outputs[0][0]] = (source, moved(op, param, shapes[inputs[0]], elements, constants))
        elif op == 'Add':
            result = bounded(a[0] + rs[1][0], a[1] + rs[1][1]) if a is not None and rs[1] is not None else None
        elif op == 'Sub':
            result = bounded(a[0] - rs[1][1], a[1] - rs[1][0]) if a is not None and rs[1] is not None else None
        elif op == 'Mul':
            result = product(a, rs[1])
        elif op == 'ConstMul':
            c = constants[param['c']]
            result = product(a, [min(c), max(c)])
        elif op == 'ConstAdd':
            c = constants[param['c']]
            result = bounded(a[0] + min(c), a[1] + max(c)) if a is not None else None
        elif op == 'BitDecompose':
            checked[op] += 1
            width = (1 << (param['low'] + param['high'])) - 1
            if inputs[0] in origin:
                # A committed graph input, directly or moved: its bits prove its range, which
                # its declared range must hold.
                source, elements = origin[inputs[0]]
                lo, hi = ranges[source]
                if not (lo <= 0 and width <= hi):
                    refuse('ERR-COMPILE-RANGE', nid)
                covered[source].update(elements)
            elif a is None:
                refuse('ERR-COMPILE-WRAP', nid)
            elif not (0 <= a[0] and a[1] <= width):
                refuse('ERR-COMPILE-RANGE', nid)
            result = [0, 1]
        elif op == 'BitFn':
            checked[op] += 1
            for t, r in zip(inputs, rs):
                if t in origin:
                    refuse('ERR-COMPILE-RANGE', nid)  # its declared [0, 1] proves no bit
                elif r is None:
                    refuse('ERR-COMPILE-WRAP', nid)
                elif not (0 <= r[0] and r[1] <= 1):
                    refuse('ERR-COMPILE-RANGE', nid)
            result = [0, 1]
        elif op == 'AssertEqual':
            checked[op] += 1
            for r in rs:
                if r is None:
                    refuse('ERR-COMPILE-WRAP', nid)
                elif not (-HALF < r[0] and r[1] < HALF):
                    refuse('ERR-COMPILE-RANGE', nid)
            result = None
        elif op == 'OneHot':
            if inputs[0] in committed:
                covered[inputs[0]].update(range(size_of(shapes[inputs[0]])))
            result = [0, 1]
        else:
            raise AssertionError(op)
        for tid, shape, _ in outputs:
            ranges[tid], shapes[tid] = result, shape
    for t in committed:
        if refusal is None and len(covered[t]) != size_of(shapes[t]):
            refusal = {'code': 'ERR-COMPILE-RANGE', 'input': t}
    assert len(ranges) == case['tensors']
    return [ranges[i] for i in range(case['tensors'])], checked, refusal


def graph(inputs, nodes, constants=()):
    # The IR's constants are i64 values (RFC-0006 'Operation set').
    assert all(-2**63 <= v < 2**63 for c in constants for v in c), constants
    tensors = len(inputs) + sum(len(n[4]) for n in nodes)
    return {'inputs': inputs, 'nodes': nodes, 'constants': [list(c) for c in constants], 'tensors': tensors}


def byte_input(kind='A', lo=0, hi=255, shape=(4,), tid=0, role='committed'):
    return [tid, role, list(shape), kind, 'u8' if hi <= 255 else f'u{hi.bit_length()}', lo, hi]


# Hand-built graphs, each with one fault or none, and the verdict RFC-0006 and RFC-0066 give.
RULES = [
    ('a byte declared [0, 255] under BitDecompose { 8, 1 }, read directly',
     graph([byte_input()], [[0, 'BitDecompose', {'low': 8, 'high': 1}, [0], [[1, [4, 8], 'A'], [2, [4, 1], 'A']], 'nine bits']]),
     {'code': 'ERR-COMPILE-RANGE', 'node': 0}),
    ('a byte declared [0, 255] under BitDecompose { 8, 1 }, reshaped',
     graph([byte_input()], [[0, 'Reshape', {}, [0], [[1, [2, 2], 'v']], 'reshape'],
                            [1, 'BitDecompose', {'low': 8, 'high': 1}, [1], [[2, [2, 2, 8], 'A'], [3, [2, 2, 1], 'A']], 'nine bits']]),
     {'code': 'ERR-COMPILE-RANGE', 'node': 1}),
    ('a byte declared [0, 255] under BitDecompose { 8, 1 }, gathered',
     graph([byte_input()], [[0, 'GatherPublic', {'index': 0}, [0], [[1, [4], 'v']], 'gather'],
                            [1, 'BitDecompose', {'low': 8, 'high': 1}, [1], [[2, [4, 8], 'A'], [3, [4, 1], 'A']], 'nine bits']],
           [[3, 0, 1, 2]]),
     {'code': 'ERR-COMPILE-RANGE', 'node': 1}),
    ('a committed input declared [0, 511] under BitDecompose { 8, 1 }',
     graph([byte_input(hi=511)], [[0, 'BitDecompose', {'low': 8, 'high': 1}, [0], [[1, [4, 8], 'A'], [2, [4, 1], 'A']], 'nine bits']]),
     None),
    ('a byte declared [0, 255], gathered and transposed under BitDecompose { 8, 0 }',
     graph([byte_input(shape=(2, 2))],
           [[0, 'Transpose', {'perm': [1, 0]}, [0], [[1, [2, 2], 'v']], 'transpose'],
            [1, 'GatherPublic', {'index': 0}, [1], [[2, [2, 2], 'v']], 'gather'],
            [2, 'BitDecompose', {'low': 8, 'high': 0}, [2], [[3, [2, 2, 8], 'A']], 'bits']],
           [[1, 0]]),
     None),
    ('a sum of two bytes under BitDecompose { 8, 0 }',
     graph([byte_input('P', role='a'), byte_input('P', tid=1, role='b')],
           [[0, 'Add', {}, [0, 1], [[2, [4], 'v']], 'sum'],
            [1, 'BitDecompose', {'low': 8, 'high': 0}, [2], [[3, [4, 8], 'A']], 'bits']]),
     {'code': 'ERR-COMPILE-RANGE', 'node': 1}),
    ('a ConstAdd of -1 to [1, 256] under BitDecompose { 8, 0 }',
     graph([[0, 'p', [4], 'P', 'u9', 1, 256]],
           [[0, 'ConstAdd', {'c': 0}, [0], [[1, [4], 'v']], 'shift'],
            [1, 'BitDecompose', {'low': 8, 'high': 0}, [1], [[2, [4, 8], 'A']], 'bits']],
           [[-1]]),
     None),
    ('62 bits of [0, 2^62 - 1]',
     graph([[0, 'x', [3], 'P', 'u62', 0, 2**62 - 1]],
           [[0, 'BitDecompose', {'low': 32, 'high': 30}, [0], [[1, [3, 32], 'A'], [2, [3, 30], 'A']], 'bits']]),
     None),
    ('BitFn on a sum of bits',
     graph([[0, 'a', [4], 'P', 'u1', 0, 1], [1, 'b', [4], 'P', 'u1', 0, 1]],
           [[0, 'Add', {}, [0, 1], [[2, [4], 'v']], 'sum'],
            [1, 'BitFn', {'f': 'Xor'}, [0, 2], [[3, [4], 'v']], 'xor']]),
     {'code': 'ERR-COMPILE-RANGE', 'node': 1}),
    ('BitFn on a committed input, read directly',
     graph([[0, 'c', [4], 'A', 'u1', 0, 1], [1, 'p', [4], 'P', 'u1', 0, 1]],
           [[0, 'BitFn', {'f': 'And'}, [1, 0], [[2, [4], 'v']], 'and'],
            [1, 'BitDecompose', {'low': 1, 'high': 0}, [0], [[3, [4, 1], 'A']], 'range check']]),
     {'code': 'ERR-COMPILE-RANGE', 'node': 0}),
    ('BitFn on a committed input, reshaped',
     graph([[0, 'c', [4], 'A', 'u1', 0, 1], [1, 'p', [2, 2], 'P', 'u1', 0, 1]],
           [[0, 'Reshape', {}, [0], [[2, [2, 2], 'v']], 'reshape'],
            [1, 'BitFn', {'f': 'Xor'}, [2, 1], [[3, [2, 2], 'v']], 'xor'],
            [2, 'BitDecompose', {'low': 1, 'high': 0}, [0], [[4, [4, 1], 'A']], 'range check']]),
     {'code': 'ERR-COMPILE-RANGE', 'node': 1}),
    ('BitFn on the bits of a committed input',
     graph([[0, 'c', [4], 'A', 'u1', 0, 1], [1, 'p', [4], 'P', 'u1', 0, 1]],
           [[0, 'BitDecompose', {'low': 1, 'high': 0}, [0], [[2, [4, 1], 'A']], 'bits'],
            [1, 'Reshape', {}, [2], [[3, [4], 'v']], 'reshape'],
            [2, 'BitFn', {'f': 'Xor'}, [3, 1], [[4, [4], 'v']], 'xor']]),
     None),
    ('AssertEqual of an unbounded tensor',
     graph([[0, 'x', [2], 'P', 'u8', 0, 255]],
           [[0, 'ConstMul', {'c': 0}, [0], [[1, [2], 'v']], 'scale'],
            [1, 'ConstMul', {'c': 1}, [1], [[2, [2], 'v']], 'scale'],
            [2, 'ConstMul', {'c': 2}, [2], [[3, [2], 'v']], 'scale'],
            [3, 'AssertEqual', {}, [3, 0], [], 'binding']],
           [[2**62], [2**62], [2**62]]),
     {'code': 'ERR-COMPILE-WRAP', 'node': 3}),
    ('AssertEqual of [0, (P - 1)/2], just outside the centered field range',
     graph([[0, 'x', [2], 'P', 'u1', 0, 1]],
           [[0, 'ConstMul', {'c': 0}, [0], [[1, [2], 'v']], 'scale'],
            [1, 'AssertEqual', {}, [0, 1], [], 'binding']],
           [[HALF]]),
     {'code': 'ERR-COMPILE-RANGE', 'node': 1}),
    ('AssertEqual of [-(P - 3)/2, (P - 3)/2], the centered field range\'s edges',
     graph([[0, 'x', [2], 'P', 'i2', -1, 1]],
           [[0, 'ConstMul', {'c': 0}, [0], [[1, [2], 'v']], 'scale'],
            [1, 'AssertEqual', {}, [0, 1], [], 'binding']],
           [[HALF - 1]]),
     None),
    ('a committed input half read by a Slice',
     graph([byte_input()],
           [[0, 'Slice', {'axis': 0, 'start': 0, 'len': 2}, [0], [[1, [2], 'v']], 'half'],
            [1, 'BitDecompose', {'low': 8, 'high': 0}, [1], [[2, [2, 8], 'A']], 'bits']]),
     {'code': 'ERR-COMPILE-RANGE', 'input': 0}),
    ('a committed input that nothing reads',
     graph([byte_input(), [1, 'p', [4], 'P', 'u8', 0, 255]],
           [[0, 'BitDecompose', {'low': 8, 'high': 0}, [1], [[2, [4, 8], 'A']], 'bits of the public input']]),
     {'code': 'ERR-COMPILE-RANGE', 'input': 0}),
    ('a committed input read by two Slices that cover it',
     graph([byte_input()],
           [[0, 'Slice', {'axis': 0, 'start': 0, 'len': 2}, [0], [[1, [2], 'v']], 'low half'],
            [1, 'BitDecompose', {'low': 8, 'high': 0}, [1], [[2, [2, 8], 'A']], 'bits'],
            [2, 'Slice', {'axis': 0, 'start': 2, 'len': 2}, [0], [[3, [2], 'v']], 'high half'],
            [3, 'BitDecompose', {'low': 8, 'high': 0}, [3], [[4, [2, 8], 'A']], 'bits']]),
     None),
]


def write_anchor():
    graphs = json.loads((ROOT / 'spec/vectors/anchor-graphs/sha256-graphs.json').read_text())['cases']
    circuits = []
    for case in graphs:
        ranges, checked, refusal = analyze_graph(case)
        assert refusal is None, (case['name'], refusal)
        circuits.append({'name': case['name'], 'ranges': ranges, 'checked': checked})
    rules = []
    for name, case, verdict in RULES:
        ranges, _, refusal = analyze_graph(case)
        assert refusal == verdict, (name, refusal)
        rules.append({'name': name, **case, 'ranges': ranges, 'verdict': verdict})
    text = '{\n "description": ' + json.dumps(
        'RFC-0006 inclusive i128 interval ranges with RFC-0066 \'Ranges\', of the independent '
        'SHA-256 graphs of spec/vectors/anchor-graphs/ and of small graphs that each break one rule; '
        'null means unbounded. README.md gives the fields.') + ',\n "circuits": [\n'
    text += ',\n'.join('  ' + json.dumps(c, separators=(',', ':')) for c in circuits)
    text += '\n ],\n "rules": [\n'
    text += ',\n'.join('  ' + json.dumps(r, separators=(',', ':')) for r in rules)
    text += '\n ]\n}\n'
    json.loads(text)
    Path(__file__).with_name('anchor.json').write_text(text)

if __name__ == '__main__':
    main()
