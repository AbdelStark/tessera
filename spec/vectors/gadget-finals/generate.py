#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0 OR MIT
"""RFC-0009/0050 comparison oracle: whole Python integers and direct cube sums.

No Rust implementation is read or called. Normalized digits/carries come from
weighted integer prefixes rather than the production limb recurrence. The two
normalization chains and subtraction are checked independently as whole integers.
"""
import json
import re
from pathlib import Path

P = 2**64 - 2**32 + 1
ONE = (1, 0)
ZERO = (0, 0)


def add(a, b):
    return ((a[0] + b[0]) % P, (a[1] + b[1]) % P)


def mul(a, b):
    return ((a[0]*b[0] + 7*a[1]*b[1]) % P,
            (a[0]*b[1] + a[1]*b[0]) % P)


def eq(point, index):
    out = ONE
    for bit, coordinate in enumerate(point):
        out = mul(out, coordinate if index >> bit & 1 else
                  ((1-coordinate[0]) % P, -coordinate[1] % P))
    return out


def mle(values, point):
    assert len(values) == 2**len(point)
    out = ZERO
    for i, value in enumerate(values):
        out = add(out, mul((value % P, 0), eq(point, i)))
    return list(out)


def whole(coefficients, radix):
    return sum(v * radix**i for i, v in enumerate(coefficients))


def normalized(coefficients, radix):
    value = whole(coefficients, radix)
    digits = [(value // radix**i) % radix for i in range(len(coefficients))]
    carries = [sum(v*radix**k for k, v in enumerate(coefficients[:j+1])) // radix**(j+1)
               for j in range(len(coefficients))]
    assert carries[-1] == 0 and whole(digits, radix) == value
    return digits, carries


def witness(left, right, bits):
    radix = 2**bits
    x, y = whole(left, radix), whole(right, radix)
    assert x <= y
    ld, lc = normalized(left, radix)
    rd, rc = normalized(right, radix)
    difference = [(y-x) // radix**i % radix for i in range(len(left))]
    # A borrow out of a prefix is exactly the strict comparison of its two values.
    borrows = [int(x % radix**(j+1) > y % radix**(j+1)) for j in range(len(left))]
    assert borrows[-1] == 0 and whole(difference, radix) == y-x
    for j in range(len(left)):
        assert left[j] + (lc[j-1] if j else 0) == ld[j] + radix*lc[j]
        assert right[j] + (rc[j-1] if j else 0) == rd[j] + radix*rc[j]
        assert rd[j]-ld[j]-(borrows[j-1] if j else 0) == difference[j]-radix*borrows[j]
    return [ld, lc, rd, rc, difference, borrows]


def carry_widths(bounds, bits):
    radix = 2**bits
    result = []
    previous_box = 0
    for j, bound in enumerate(bounds):
        assert 0 <= bound < 2**40
        maximum = sum(v*radix**k for k, v in enumerate(bounds[:j+1])) // radix**(j+1)
        width = max(1, maximum.bit_length())
        box = 2**width - 1
        assert bound + previous_box < 2**41
        assert radix-1 + radix*box < 2**41
        result.append(width)
        previous_box = box
    assert maximum == 0
    return result


def case(name, bits, rows, bounds=None):
    m = len(rows[0][0])
    assert all(len(l) == len(r) == m for l, r in rows)
    assert len(rows) in (1, 2, 4, 8)
    left_bounds, right_bounds = bounds if bounds is not None else (
        [max(l[j] for l, _ in rows) for j in range(m)],
        [max(r[j] for _, r in rows) for j in range(m)])
    widths = carry_widths(left_bounds, bits), carry_widths(right_bounds, bits)
    point = [[(0xfffffff1*(i+1)+19) % P, (P-17*(i+1)) % P]
             for i in range((len(rows)-1).bit_length())]
    records = []
    for left, right in rows:
        arrays = witness(left, right, bits)
        for side, values in enumerate((left, right)):
            assert all(0 <= v <= b for v, b in zip(values, (left_bounds, right_bounds)[side]))
        expected_widths = [[bits]*m, widths[0], [bits]*m, widths[1], [bits]*m, [1]*m]
        for array, bs in zip(arrays, expected_widths):
            assert all(0 <= v < 2**b for v, b in zip(array, bs))
        records.append(dict(left=left, right=right, arrays=arrays,
                            x=str(whole(left, 2**bits)), y=str(whole(right, 2**bits))))
    evaluations = {
        "left": [mle([r['left'][j] for r in records], point) for j in range(m)],
        "right": [mle([r['right'][j] for r in records], point) for j in range(m)],
        "arrays": [[mle([r['arrays'][a][j] for r in records], point) for j in range(m)]
                   for a in range(6)],
    }
    return dict(name=name, bits=bits, bounds=[left_bounds, right_bounds], widths=list(widths),
                rows=records, point=point, evaluations=evaluations)


def product_coefficients(root, u, width, increment):
    root_digits = [(root >> i) & 255 for i in range(0, width, 8)]
    root_digits[0] += int(increment)
    input_digits = [(u >> (16*i)) & (127 if i == 3 else 65535) for i in range(4)]
    out = [0] * ((2*width+55+7)//8)
    for i, a in enumerate(root_digits):
        for j, b in enumerate(root_digits):
            for k, c in enumerate(input_digits):
                out[i+j+2*k] += a*b*c
    assert whole(out, 256) == (root+int(increment))**2*u
    return out


def root_cases():
    from math import isqrt
    cases = []
    parameters = [(f'root-{w}', w, 2**(2*(w-1)), 1) for w in range(1, 33)]
    parameters += [('root-constant-at-P', 32, P-1, 1),
                   ('root-constant-above-P', 32, 4096*2**60, 672)]
    for name, width, n, epsilon in parameters:
        m = (2*width+55+7)//8
        rows_lower, rows_upper = [], []
        for u in [epsilon, epsilon+1, 2**32-2, 2**55-1]:
            root = isqrt(n//u)
            rows_lower.append((product_coefficients(root, u, width, False),
                               [(n >> (8*i)) & 255 for i in range(m)]))
            rows_upper.append(([(n+1) >> (8*i) & 255 for i in range(m)],
                               product_coefficients(root, u, width, True)))
        # Padded rows: both constants zero, and (rho+1)^2*u remains zero.
        for _ in range(4):
            rows_lower.append(([0]*m, [0]*m))
            rows_upper.append(([0]*m, [0]*m))
        maxima_lower = product_coefficients(2**width-1, 2**55-1, width, False)
        maxima_upper = product_coefficients(2**width-1, 2**55-1, width, True)
        const_l = [(n >> (8*i)) & 255 for i in range(m)]
        const_u = [((n+1) >> (8*i)) & 255 for i in range(m)]
        cases += [case(f'{name}-lower', 8, rows_lower, (maxima_lower, const_l)),
                  case(f'{name}-upper', 8, rows_upper, (const_u, maxima_upper))]
    return cases


def main():
    cases = []
    for bits in [1, 2, 7, 8, 16, 31, 32]:
        radix = 2**bits
        m = max(3, (42+bits-1)//bits)
        rows = []
        for left, right in [([radix-1, 0], [0, 1]), ([0, 0], [0, 0]),
                            ([radix+1, 0], [radix+1, 0]), ([radix-1, radix-1], [0, radix])]:
            rows.append((left+[0]*(m-2), right+[0]*(m-2)))
        cases.append(case(f'radix-{bits}-borrow-and-carry', bits, rows))
    cases.extend(root_cases())
    out = {"schema_version": 1,
           "description": "RFC-0009/0050 grouped comparison arrays; independent integer prefixes and direct MLE sums",
           "cases": cases}
    rendered = re.sub(r'\[\s+[\d,\s]+\]',
                      lambda match: json.dumps(json.loads(match[0])),
                      json.dumps(out, indent=2))
    Path(__file__).with_name('comparison.json').write_text(rendered+'\n')
    print(f'{len(cases)} comparison cases, {sum(len(c["rows"]) for c in cases)} integer witnesses')


if __name__ == '__main__':
    main()
