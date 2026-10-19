#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0 OR MIT
"""RFC-0006 range oracle, written before the independent verifier range implementation.

Input graphs are the independent RFC-0044 lowering oracle's public metadata, never Rust output.
All arithmetic uses Python integers; only the specified i128 interval boundary is imposed.
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


if __name__ == '__main__':
    main()
