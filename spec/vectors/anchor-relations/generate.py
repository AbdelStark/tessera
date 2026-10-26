#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0 OR MIT
"""Reference values for the public factors of the anchor circuits' relations, RFC-0072 with
RFC-0066: the selector psi and LT, ConstAdd's shift c-hat, the selection kernels S and K, the
recomposition weights W_b and W_h with their zero-extensions, the BitFn polynomials, and the
recomposition identity on honest and false witnesses.

This script is the independent oracle. It was written from RFC-0072, RFC-0066, RFC-0010
'Multilinear extensions' and RFC-0002 alone, by an author who read neither crates/ nor
verifier/crates/. Its arithmetic is Python's arbitrary-precision integers, with
Fp2 = Fp[X]/(X^2 - 7) multiplied by the schoolbook formula of RFC-0002. Every value is computed
twice before it is written: once by the closed form RFC-0072 gives the verifier, and once as the
multilinear extension of the function's table, enumerated over the whole padded cube. The two
must agree, or the script stops.

Usage: python3 -I spec/vectors/anchor-relations/generate.py
(rewrites relations.json)

Inputs come from SplitMix64 with a fixed seed, words at or above P rejected, so the file is
reproducible byte for byte.
"""

import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
SEED = 0x7E55E7A_2026_1026  # fixed; changing it changes every vector
P = 2**64 - 2**32 + 1  # Goldilocks (RFC-0002 'Constants')
W = 7  # Fp2 = Fp[X]/(X^2 - W)
POINTS = 2  # random points per case, where a case takes several

# FIPS 180-4 section 4.2.2: the first round constant, used as a realistic ConstAdd shift.
K0 = 0x428A2F98


def require(condition, message):
    """A check that holds under `python -O` too, unlike `assert`."""
    if not condition:
        raise SystemExit(f"anchor-relations/generate.py: {message}")


def splitmix64(state):
    """Yields 64-bit outputs of SplitMix64 (Steele, Lea, Flood 2014) from `state`."""
    mask = (1 << 64) - 1
    while True:
        state = (state + 0x9E3779B97F4A7C15) & mask
        z = state
        z = ((z ^ (z >> 30)) * 0xBF58476D1CE4E5B9) & mask
        z = ((z ^ (z >> 27)) * 0x94D049BB133111EB) & mask
        yield z ^ (z >> 31)


# ----- Fp2 = Fp[X]/(X^2 - 7) on pairs (c0, c1); Fp embeds as (x, 0) ---------------------------

ZERO, ONE = (0, 0), (1, 0)


def add(a, b):
    return ((a[0] + b[0]) % P, (a[1] + b[1]) % P)


def sub(a, b):
    return ((a[0] - b[0]) % P, (a[1] - b[1]) % P)


def mul(a, b):
    """(a0 + a1 X)(b0 + b1 X) with X^2 = W, schoolbook."""
    return ((a[0] * b[0] + W * a[1] * b[1]) % P, (a[0] * b[1] + a[1] * b[0]) % P)


def lift(x):
    """An integer's image in Fp2: RFC-0002's signed embedding, a negative x to P - |x|."""
    return (x % P, 0)


def total(values):
    out = ZERO
    for v in values:
        out = add(out, v)
    return out


def product(values):
    out = ONE
    for v in values:
        out = mul(out, v)
    return out


class Inputs:
    """Uniform field elements and integers from SplitMix64, by rejection."""

    def __init__(self, seed):
        self.words = splitmix64(seed)

    def fp(self):
        while True:
            x = next(self.words)
            if x < P:
                return x

    def fp2(self):
        return (self.fp(), self.fp())

    def point(self, n):
        return [self.fp2() for _ in range(n)]

    def below(self, bound):
        """A uniform integer in [0, bound), bound <= 2^64."""
        while True:
            x = next(self.words)
            if x < bound * ((1 << 64) // bound):
                return x % bound

    def signed(self, bits):
        """A uniform integer in [-2^(bits-1), 2^(bits-1))."""
        return self.below(1 << bits) - (1 << (bits - 1))


# ----- Multilinear extensions (RFC-0010 'Multilinear extensions') ------------------------------


def bit(x, i):
    return (x >> i) & 1


def log2_ceil(n):
    """The variables of an axis of length n >= 1: ceil(log2 n), 0 for n = 1."""
    require(n >= 1, "an axis has at least one position")
    return (n - 1).bit_length()


def eq_at(r, x):
    """eq(r, x) for an integer x read as the Boolean point (x_1, ..., x_n), x_1 its lowest bit:
    the product of r_i where x_i = 1 and 1 - r_i where x_i = 0."""
    out = ONE
    for i, ri in enumerate(r):
        out = mul(out, ri if bit(x, i) else sub(ONE, ri))
    return out


def eq_points(a, b):
    """eq(a, b) = prod_i (a_i b_i + (1 - a_i)(1 - b_i)) for two field points."""
    out = ONE
    for ai, bi in zip(a, b, strict=True):
        out = mul(out, add(mul(ai, bi), mul(sub(ONE, ai), sub(ONE, bi))))
    return out


def eq_table(r):
    """[eq(r, x) for x in 0 .. 2^n - 1], each entry the product of its n factors."""
    return [eq_at(r, x) for x in range(1 << len(r))]


def mle(table, r):
    """f(r) = sum over the hypercube of f[x] eq(r, x), the table's entries in Fp2: the direct
    enumeration every closed form below is checked against."""
    require(len(table) == 1 << len(r), "a table of 2^n entries takes n coordinates")
    weights = eq_table(r)
    return total(mul(t, w) for t, w in zip(table, weights) if t != ZERO)


def lt_direct(r, a):
    """LT(r, a) = sum of eq(r, h) over the integers h < a (RFC-0010), 0 <= a <= 2^n."""
    require(0 <= a <= 1 << len(r), "LT's bound lies in [0, 2^n]")
    return total(eq_at(r, h) for h in range(a))


def lt_closed(r, a):
    """LT(r, a) in O(n): for each set bit j of a, the block of integers that agree with a above
    bit j and have bit j clear weighs (1 - r_j) times the eq factors of a's bits above j."""
    n = len(r)
    if a == 1 << n:
        return ONE
    out = ZERO
    for j in range(n):
        if bit(a, j):
            block = sub(ONE, r[j])
            for i in range(j + 1, n):
                block = mul(block, r[i] if bit(a, i) else sub(ONE, r[i]))
            out = add(out, block)
    return out


def lt(r, a):
    value = lt_closed(r, a)
    require(value == lt_direct(r, a), f"LT's closed form is its sum, n = {len(r)}, a = {a}")
    return value


# ----- Padded tensors ------------------------------------------------------------------------
#
# A tensor of shape (n_0, ..., n_{d-1}) is flattened row-major after each axis is padded to a
# power of two, so its last axis takes the lowest variables. A point lists low variables first:
# the last axis's coordinates, then the axis before it's, up to axis 0's.


def axis_vars(shape):
    return [log2_ceil(n) for n in shape]


def total_vars(shape):
    return sum(axis_vars(shape))


def split_point(shape, r):
    """The coordinates of each axis of `shape` in the point r, by axis."""
    vs = axis_vars(shape)
    require(len(r) == sum(vs), "a point has one coordinate per variable of the shape")
    out, at = [None] * len(shape), 0
    for a in reversed(range(len(shape))):
        out[a] = r[at:at + vs[a]]
        at += vs[a]
    return out


def join_point(shape, coords):
    """The point of `shape` whose axis a has the coordinates coords[a]."""
    out = []
    for a in reversed(range(len(shape))):
        require(len(coords[a]) == log2_ceil(shape[a]), "one coordinate per variable of the axis")
        out += coords[a]
    return out


def flat(shape, idx):
    """The index of the multi-index idx in the padded layout of `shape`."""
    out = 0
    for i, v in zip(idx, axis_vars(shape), strict=True):
        out = (out << v) | i
    return out


def positions(shape):
    """Every position of the padded cube of `shape` as (index, multi-index), index order."""
    vs = axis_vars(shape)
    out = []
    for x in range(1 << sum(vs)):
        idx, rest = [0] * len(shape), x
        for a in reversed(range(len(shape))):
            idx[a] = rest & ((1 << vs[a]) - 1)
            rest >>= vs[a]
        out.append((x, idx))
    return out


def padded(shape, value):
    """The padded table of `shape`, value(multi-index) at real positions and 0 at padded ones."""
    out = []
    for _, idx in positions(shape):
        real = all(i < n for i, n in zip(idx, shape, strict=True))
        out.append(value(idx) if real else ZERO)
    return out


def psi_closed(shape, r):
    """psi(r) = prod_a LT(r_a, n_a) (RFC-0072 'Real positions')."""
    return product(lt(ra, n) for ra, n in zip(split_point(shape, r), shape, strict=True))


def psi(shape, r):
    value = psi_closed(shape, r)
    require(value == mle(padded(shape, lambda idx: ONE), r), f"psi's closed form, shape {shape}")
    return value


# ----- Rendering ---------------------------------------------------------------------------------


def hex64(x):
    return f"0x{x:016x}"


def fp2(a):
    return [hex64(a[0]), hex64(a[1])]


def points(values):
    return [fp2(v) for v in values]


def section(cases):
    return "[\n" + ",\n".join(json.dumps(case, separators=(",", ":")) for case in cases) + "\n  ]"


# ----- LT and psi ------------------------------------------------------------------------------

# (n, bounds): the anchor's axis lengths, SHA-256's carry widths and the SHR masks' supports
# (SHR^3 keeps rows i < 29, SHR^10 rows i < 22).
LT_CASES = [
    (0, [0, 1]),
    (1, [0, 1, 2]),
    (2, [0, 1, 2, 3, 4]),
    (3, [0, 3, 5, 6, 7, 8]),
    (5, [1, 22, 26, 29, 31, 32]),
    (6, [35, 53, 63]),
    (8, [208, 255]),
    (10, [512, 536, 1023]),
]

PSI_SHAPES = [
    [1], [3], [5], [3, 5, 3], [5, 3], [3, 3], [4, 3], [3, 1], [5, 1], [4, 4], [32, 3],
    [26, 2], [22, 3], [32, 1], [512, 3], [2, 3, 5],
]


def lt_cases(rng):
    cases = []
    for n, bounds in LT_CASES:
        r = rng.point(n)
        cases.append({"n": n, "point": points(r), "values": [[a, fp2(lt(r, a))] for a in bounds]})
    return cases


def psi_cases(rng):
    cases = []
    for shape in PSI_SHAPES:
        for _ in range(POINTS):
            r = rng.point(total_vars(shape))
            cases.append({"shape": shape, "point": points(r), "value": fp2(psi(shape, r))})
    return cases


# ----- ConstAdd's shift (RFC-0072 'Linear operations that are not coordinate maps') ------------


def const_add_cases(rng):
    """c-hat, the extension of psi*c: c*psi(r) for a constant of one value, and
    c(r_c)*prod_{a != last} LT(r_a, n_a) for a constant per channel, against the direct
    extension of the table psi*c over the padded cube. A constant per channel also gives
    c(r_c)*psi(r), the wrong value RFC-0072 names, which differs off the cube when the last
    axis's length is not a power of two."""
    specs = [
        ([3], [7]),
        ([5, 3], [-12]),
        ([3, 5, 3], [K0]),
        ([5, 1], [2**32 - 1]),
        ([3, 1], [1]),
        ([4, 4], [-1]),
        ([5, 3], [1, -2, 3]),
        ([3, 5, 3], [rng.signed(63), rng.signed(63), rng.signed(63)]),
        ([3, 3], [K0, 0, -K0]),
        ([4, 3], [rng.signed(32), rng.signed(32), rng.signed(32)]),
        ([3, 5], [rng.signed(16) for _ in range(5)]),
        ([3, 2], [5, -9]),
    ]
    cases = []
    for shape, constant in specs:
        require(len(constant) == 1 or (len(constant) == shape[-1] and len(constant) > 1),
                "a constant has one value, or one per channel of the last axis and more than one")
        per_channel = len(constant) > 1
        for _ in range(POINTS):
            r = rng.point(total_vars(shape))
            axes = split_point(shape, r)
            psi_r = psi(shape, r)
            table = padded(shape, lambda idx: lift(constant[idx[-1]] if per_channel else constant[0]))
            direct = mle(table, r)
            case = {"shape": shape, "constant": constant, "point": points(r), "psi": fp2(psi_r)}
            if per_channel:
                channels = [lift(c) for c in constant] + [ZERO] * ((1 << len(axes[-1])) - len(constant))
                c_tilde = mle(channels, axes[-1])
                others = product(lt(ra, n) for ra, n in zip(axes[:-1], shape[:-1], strict=True))
                closed = mul(c_tilde, others)
                wrong = mul(c_tilde, psi_r)
                differs = wrong != direct
                padded_last = shape[-1] & (shape[-1] - 1) != 0
                require(differs == padded_last,
                        f"c(r_c) psi(r) differs from c-hat exactly when the last axis pads, {shape}")
                case.update({"c_tilde": fp2(c_tilde), "wrong": fp2(wrong), "wrong_differs": differs})
            else:
                closed = mul(lift(constant[0]), psi_r)
            require(closed == direct, f"c-hat's closed form is the extension of psi*c, {shape} {constant}")
            case["value"] = fp2(closed)
            cases.append(case)
    return cases


# ----- The selection relation (RFC-0072 'The selection relation', 'Combined selections') -------


def member_rows(member, n_a):
    """(len, idx) of a selection along an axis of length n_a: GatherPublic's index vector, or
    Slice's rows start + j."""
    if member["op"] == "GatherPublic":
        idx = list(member["index"])
    else:
        require(member["len"] >= 1 and member["start"] + member["len"] <= n_a, "a slice in its axis")
        idx = [member["start"] + j for j in range(member["len"])]
    require(all(0 <= i < n_a for i in idx), "every index a row of the axis")
    return len(idx), idx


def kernel_closed(members, n_a, r_a, r_k):
    """K(r_a, r_k) = sum_t a_t sum_{j < len_t} eq(r_a, j) eq(r_k, idx_{t,j})."""
    out = ZERO
    for member in members:
        length, idx = member_rows(member, n_a)
        term = total(mul(eq_at(r_a, j), eq_at(r_k, idx[j])) for j in range(length))
        out = add(out, mul(member["coefficient"], term))
    return out


def kernel_direct(members, n_a, r_a, r_k):
    """The extension of the table K(j, k) = sum_t a_t [j < len_t][idx_{t,j} = k] over the whole
    cube of (j, k), j on r_a's variables and k on r_k's."""
    cube_j, cube_k = 1 << len(r_a), 1 << len(r_k)
    table = [[ZERO] * cube_k for _ in range(cube_j)]
    for member in members:
        length, idx = member_rows(member, n_a)
        require(length <= cube_j, "the selected rows fit y's axis variables")
        for j in range(length):
            table[j][idx[j]] = add(table[j][idx[j]], member["coefficient"])
    e_j, e_k = eq_table(r_a), eq_table(r_k)
    out = ZERO
    for j in range(cube_j):
        row = total(mul(table[j][k], e_k[k]) for k in range(cube_k) if table[j][k] != ZERO)
        out = add(out, mul(e_j[j], row))
    return out


def select(x, shape, axis, member):
    """y = the rows of x along `axis` that the member selects, as a dict of real positions."""
    length, idx = member_rows(member, shape[axis])
    out_shape = list(shape)
    out_shape[axis] = length
    return out_shape, lambda pos: x[tuple(pos[:axis] + [idx[pos[axis]]] + pos[axis + 1:])]


def rotation(r):
    return [(i + r) % 32 for i in range(32)]


def assembly(n_blocks, offsets):
    """RFC-0066 'The message': the gather that puts committed byte k's bit t, row 8k + t of the
    decomposed bytes, at its word position, and row 0 at the others. Bit i of word j is bit
    i mod 8 of byte 4j + 3 - floor(i/8), so byte o's bit t sits at row 32 (o // 4) +
    8 (3 - o % 4) + t."""
    idx = [0] * (512 * n_blocks)
    for k, o in enumerate(offsets):
        for t in range(8):
            idx[32 * (o // 4) + 8 * (3 - o % 4) + t] = 8 * k + t
    return idx


def gather(index, coefficient=ONE):
    return {"op": "GatherPublic", "index": index, "coefficient": coefficient}


def slice_(start, length, coefficient=ONE):
    return {"op": "Slice", "start": start, "len": length, "coefficient": coefficient}


def selection_specs(rng):
    """(name, x's shape, axis, members): the selections RFC-0072 'Testing Strategy' names."""
    specs = []
    for m in (1, 3):
        for r in (2, 13, 22, 6, 11, 25, 7, 18, 17, 19, 3, 10):
            name = f"ROTR^{r}" if r not in (3, 10) else f"SHR^{r}'s gather"
            specs.append((f"{name}, M = {m}", [32, m], 0, [gather(rotation(r))]))
    for level in range(5):
        rows = 32 >> level
        even, odd = list(range(0, rows, 2)), list(range(1, rows, 2))
        weight = lift(2 ** (2**level))
        specs.append((f"value level {level}, even rows", [rows, 3], 0, [gather(even)]))
        specs.append((f"value level {level}, odd rows", [rows, 3], 0, [gather(odd)]))
        specs.append((f"value level {level}, even + 2^(2^{level}) odd, combined", [rows, 3], 0,
                      [gather(even), gather(odd, weight)]))
        a = rng.fp2()
        specs.append((f"value level {level}, combined under a claim coefficient", [rows, 1], 0,
                      [gather(even, a), gather(odd, mul(a, weight))]))
    key_offsets = list(range(21, 53))
    frame_offsets = list(range(41, 67))
    specs.append(("message assembly, the key commitment (N = 1, K = 32)", [8 * 32, 1], 0,
                  [gather(assembly(1, key_offsets))]))
    specs.append(("message assembly, a frame of one token (N = 2, K = 26)", [8 * 26, 2], 0,
                  [gather(assembly(2, frame_offsets))]))
    for start in (0, 160, 480):
        specs.append((f"Slice axis 0, a message word at row {start}", [512, 1], 0, [slice_(start, 32)]))
    specs.append(("Slice axis 0, an initial word", [256, 3], 0, [slice_(224, 32)]))
    specs.append(("Slice axis 0, len 3", [5, 3], 0, [slice_(2, 3)]))
    specs.append(("Slice axis 0, len 5", [8, 2], 0, [slice_(0, 5)]))
    specs.append(("Slice axis 1, len 3", [3, 6, 5], 1, [slice_(2, 3)]))
    specs.append(("Slice axis 1, len 29", [2, 32, 3], 1, [slice_(3, 29)]))
    specs.append(("Slice axis 1, len 4 of 5", [4, 5], 1, [slice_(1, 4)]))
    specs.append(("Slice axis 2, len 3", [2, 3, 7], 2, [slice_(4, 3)]))
    specs.append(("GatherPublic, len 5 with repeats", [5, 2], 0, [gather([4, 0, 0, 3, 1])]))
    specs.append(("GatherPublic, len 3", [6, 3], 0, [gather([5, 2, 2])]))
    specs.append(("combined, two slices of lengths 3 and 4", [8, 3], 0,
                  [slice_(0, 3, rng.fp2()), slice_(3, 4, rng.fp2())]))
    specs.append(("combined, a gather and a slice", [6, 2], 0,
                  [gather([1, 5, 0], lift(3)), slice_(2, 4, lift(P - 5))]))
    specs.append(("combined, three slices on axis 1", [3, 9, 2], 1,
                  [slice_(0, 3, lift(1)), slice_(3, 3, lift(2)), slice_(6, 3, lift(4))]))
    return specs


# x's tables up to this many padded entries are checked through the relation and written out.
RELATION_MAX = 1 << 12


def selection_cases(rng):
    cases = []
    for name, shape, axis, members in selection_specs(rng):
        n_a = shape[axis]
        lengths = [member_rows(m, n_a)[0] for m in members]
        y_vars = {log2_ceil(n) for n in lengths}
        require(len(y_vars) == 1, "a combination's members share the point's coordinates")
        y_axis_vars = y_vars.pop()
        out_shape = list(shape)
        out_shape[axis] = max(lengths)
        r = rng.point(total_vars(out_shape))
        r_axes = split_point(out_shape, r)
        r_a = r_axes[axis]
        r_k = rng.point(log2_ceil(n_a))
        require(len(r_a) == y_axis_vars, "r_a holds y's axis variables")
        value = kernel_closed(members, n_a, r_a, r_k)
        require(value == kernel_direct(members, n_a, r_a, r_k), f"the kernel's closed form, {name}")
        case = {
            "name": name,
            "x_shape": shape,
            "axis": axis,
            "members": [
                {key: (fp2(v) if key == "coefficient" else v) for key, v in m.items()} for m in members
            ],
            "r_a": points(r_a),
            "r_k": points(r_k),
            "kernel": fp2(value),
        }
        if 1 << total_vars(shape) <= RELATION_MAX:
            case.update(relation(rng, shape, axis, members, out_shape, r, r_k))
        cases.append(case)
    return cases


def relation(rng, shape, axis, members, out_shape, r, r_k):
    """The relation on a random x: y = sum_t a_t y_t, its claim at r, the sum over k of
    K(r_a, k) x(r_lo || k || r_hi), and the final value K(r_a, r_k) x(r_lo || r_k || r_hi)."""
    values = {}
    for _, idx in positions(shape):
        if all(i < n for i, n in zip(idx, shape, strict=True)):
            values[tuple(idx)] = rng.fp()
    x_table = padded(shape, lambda idx: lift(values[tuple(idx)]))
    claim = ZERO
    for member in members:
        y_shape, entry = select(values, shape, axis, member)
        # A member shorter than the combination's axis is 0 on the rows past its length.
        y_table = padded(out_shape, lambda idx: lift(entry(idx)) if idx[axis] < y_shape[axis] else ZERO)
        claim = add(claim, mul(member["coefficient"], mle(y_table, r)))
    r_axes = split_point(out_shape, r)
    r_a = r_axes[axis]

    def x_at(k_coords):
        coords = list(r_axes)
        coords[axis] = k_coords
        return join_point(shape, coords)

    # x(r_lo || k || r_hi) at each Boolean k: the entries whose axis-a index is k, weighed by eq
    # at the claim's coordinates of the other axes.
    k_vars = log2_ceil(shape[axis])
    lo_shape, hi_shape = shape[axis + 1:], shape[:axis]
    r_lo, r_hi = join_point(lo_shape, r_axes[axis + 1:]), join_point(hi_shape, r_axes[:axis])
    at_k = [ZERO] * (1 << k_vars)
    for pos, v in values.items():
        weight = mul(eq_at(r_lo, flat(lo_shape, pos[axis + 1:])), eq_at(r_hi, flat(hi_shape, pos[:axis])))
        at_k[pos[axis]] = add(at_k[pos[axis]], mul(lift(v), weight))
    summed = ZERO
    for k in range(1 << k_vars):
        k_point = [lift(bit(k, i)) for i in range(k_vars)]
        summed = add(summed, mul(kernel_closed(members, shape[axis], r_a, k_point), at_k[k]))
    require(summed == claim, "the selection relation holds on the cube")
    x_point = x_at(r_k)
    x_value = mle(x_table, x_point)
    require(x_value == total(mul(eq_at(r_k, k), at_k[k]) for k in range(1 << k_vars)),
            "x at the final point is its Boolean slices' extension")
    final = mul(kernel_closed(members, shape[axis], r_a, r_k), x_value)
    return {
        "x": [hex64(values[tuple(idx)]) for _, idx in positions(shape)
              if all(i < n for i, n in zip(idx, shape, strict=True))],
        "claim_point": points(r),
        "claim": fp2(claim),
        "x_point": points(x_point),
        "x_value": fp2(x_value),
        "final": fp2(final),
    }


# ----- Recomposition's weights and zero-extensions (RFC-0072 'The bit zero-checks') ------------

# The brief's widths, then two whose carry has more bit variables than the result, the only
# ones where b's zero-extension z_b is not 1.
WIDTHS = [(8, 0), (32, 1), (32, 2), (32, 3), (3, 0), (5, 3), (1, 0), (3, 5), (1, 2)]


def bit_vars(low, high):
    """(k_b, k_h, k): the outputs' bit variables and recomposition's, k_b when high = 0."""
    k_b = log2_ceil(low)
    k_h = log2_ceil(high) if high else None
    return k_b, k_h, (max(k_b, k_h) if high else k_b)


def weight_table(n, k_out, k):
    """W(i) = 2^j at i = j 2^{k - k_out} for j < n, 0 elsewhere, over k bit variables."""
    out = [ZERO] * (1 << k)
    for j in range(n):
        out[j << (k - k_out)] = lift(2**j)
    return out


def zero_extension_table(k_out, k):
    """Z(i) = prod_{t < k - k_out} (1 - i_t): 1 where i's low k - k_out bits are 0."""
    return [ONE if i % (1 << (k - k_out)) == 0 else ZERO for i in range(1 << k)]


def weights(n, k_out, k, r):
    """(Z, sum_{j < n} 2^j eq(s, j), W) at the bit point r of k coordinates, s its high k_out
    coordinates, W = Z times the sum: each closed form against its table's extension."""
    z = product(sub(ONE, r[t]) for t in range(k - k_out))
    s = r[k - k_out:]
    digits = total(mul(lift(2**j), eq_at(s, j)) for j in range(n))
    if n == 1 << k_out:
        require(digits == product(add(sub(ONE, s[t]), mul(s[t], lift(2 ** (2**t)))) for t in range(k_out)),
                "a full bit axis's weights are the product of 1 - s_t + s_t 2^(2^t)")
    w = mul(z, digits)
    require(z == mle(zero_extension_table(k_out, k), r), "the zero-extension's closed form")
    require(w == mle(weight_table(n, k_out, k), r), "the weights' closed form")
    return z, digits, w


def bit_weight_cases(rng):
    cases = []
    for low, high in WIDTHS:
        k_b, k_h, k = bit_vars(low, high)
        for _ in range(POINTS):
            r = rng.point(k)
            z_b, sum_b, w_b = weights(low, k_b, k, r)
            case = {"low": low, "high": high, "k_b": k_b, "k_h": k_h, "k": k, "point": points(r),
                    "z_b": fp2(z_b), "sum_b": fp2(sum_b), "w_b": fp2(w_b)}
            if high:
                z_h, sum_h, w_h = weights(high, k_h, k, r)
                case.update({"z_h": fp2(z_h), "sum_h": fp2(sum_h), "w_h": fp2(w_h)})
            cases.append(case)
    return cases


# ----- BitFn (RFC-0066 '`BitFn { f }`') -------------------------------------------------------


def two(x):
    return mul(lift(2), x)


BITFNS = {
    # f: (inputs, the polynomial of RFC-0066's table, the Boolean function)
    "Xor": (2, lambda a, b: sub(add(a, b), two(mul(a, b))), lambda a, b: a ^ b),
    "Xor3": (3, lambda a, b, c: add(sub(add(add(a, b), c), two(add(add(mul(a, b), mul(a, c)), mul(b, c)))),
                                    mul(lift(4), mul(mul(a, b), c))),
             lambda a, b, c: a ^ b ^ c),
    "And": (2, lambda a, b: mul(a, b), lambda a, b: a & b),
    "Ch": (3, lambda e, f, g: sub(add(mul(e, f), g), mul(e, g)), lambda e, f, g: (e & f) ^ ((1 - e) & g)),
    "Maj": (3, lambda a, b, c: sub(add(add(mul(a, b), mul(a, c)), mul(b, c)), two(mul(mul(a, b), c))),
            lambda a, b, c: (a & b) ^ (a & c) ^ (b & c)),
}


def bitfn_cases(rng):
    cases = []
    for name, (arity, poly, logic) in BITFNS.items():
        rows = []
        for x in range(1 << arity):
            inputs = [bit(x, arity - 1 - i) for i in range(arity)]  # first input most significant
            out = logic(*inputs)
            require(poly(*[lift(v) for v in inputs]) == lift(out), f"{name} is its Boolean function on bits")
            rows.append(inputs + [out])
        # The truth table indexed with the first input on the lowest variable, for the extension.
        table = [lift(logic(*[bit(x, i) for i in range(arity)])) for x in range(1 << arity)]
        evaluations = []
        for _ in range(4):
            r = rng.point(arity)
            value = poly(*r)
            require(value == mle(table, r), f"{name}'s polynomial is the extension of its truth table")
            evaluations.append({"inputs": points(r), "value": fp2(value)})
        cases.append({"f": name, "inputs": arity, "truth_table": rows, "evaluations": evaluations})
    return cases


# ----- Recomposition on a witness (RFC-0072 'Recomposition') -----------------------------------

RECOMPOSITION_SHAPES = {
    (8, 0): [[3], [5, 3]],
    (32, 1): [[3]],
    (32, 2): [[3]],
    (32, 3): [[3], [1]],
    (3, 0): [[3]],
    (5, 3): [[3]],
    (1, 0): [[3]],
    (3, 5): [[3]],
    (1, 2): [[2, 3]],
}


def bits_of(x, low, high):
    return [bit(x, i) for i in range(low)], [bit(x, low + j) for j in range(high)]


def recomposition(rng, low, high, shape, x, flip):
    """The identity of one recomposition on a witness: x's real values, its bits, and `flip`,
    None or (output, position, bit) to set one bit wrong. Returns the case's fields."""
    k_b, k_h, k = bit_vars(low, high)
    n_y = total_vars(shape)
    real = [tuple(idx) for _, idx in positions(shape) if all(i < n for i, n in zip(idx, shape, strict=True))]
    b_bits = {pos: bits_of(x[pos], low, high)[0] for pos in real}
    h_bits = {pos: bits_of(x[pos], low, high)[1] for pos in real}
    if flip is not None:
        output, pos, t = flip
        target = b_bits if output == "b" else h_bits
        target[pos] = list(target[pos])
        target[pos][t] ^= 1
    x_table = padded(shape, lambda idx: lift(x[tuple(idx)]))
    # b and h have x's shape with the bit axis innermost, padded to 2^{k_b} and 2^{k_h} slots.
    b_table = padded(shape + [low], lambda idx: lift(b_bits[tuple(idx[:-1])][idx[-1]]))
    h_table = padded(shape + [high], lambda idx: lift(h_bits[tuple(idx[:-1])][idx[-1]])) if high else None

    def hat(table, k_out):
        """The output zero-extended to k bit variables on its high coordinates."""
        out = []
        for i_y in range(1 << (k + n_y)):
            i, y = i_y & ((1 << k) - 1), i_y >> k
            low_bits = i & ((1 << (k - k_out)) - 1)
            out.append(table[(y << k_out) | (i >> (k - k_out))] if low_bits == 0 else ZERO)
        return out

    b_hat = hat(b_table, k_b)
    h_hat = hat(h_table, k_h) if high else None
    w_b_table, w_h_table = weight_table(low, k_b, k), (weight_table(high, k_h, k) if high else None)
    inv = lift(pow(2, P - 1 - k, P))  # 2^{-k}
    two_low = lift(2**low)

    def bracket(i, y):
        """2^{-k} x(y) - W_b(i) b-hat(i || y) - 2^low W_h(i) h-hat(i || y), on the cube."""
        out = sub(mul(inv, x_table[y]), mul(w_b_table[i], b_hat[(y << k) | i]))
        if high:
            out = sub(out, mul(two_low, mul(w_h_table[i], h_hat[(y << k) | i])))
        return out

    # For each y the sum over the bit axis is RFC-0066's relation x - sum 2^i b_i - 2^low sum 2^j h_j.
    sums = []
    for y_index, idx in positions(shape):
        summed = total(bracket(i, y_index) for i in range(1 << k))
        pos = tuple(idx)
        if pos in b_bits:
            relation = x[pos] - sum(2**i * v for i, v in enumerate(b_bits[pos])) \
                - 2**low * sum(2**j * v for j, v in enumerate(h_bits[pos]))
        else:
            relation = 0
        require(summed == lift(relation), "the bit axis's sum is RFC-0066's recomposition")
        sums.append(relation)
    z_y = rng.point(n_y)
    eq_y = eq_table(z_y)
    cube_sum = total(mul(eq_y[y], bracket(i, y)) for y in range(1 << n_y) for i in range(1 << k))
    require((cube_sum == ZERO) == (flip is None), "the zero-check sums to 0 exactly on the honest witness")

    # The instance's final check at r' = r'_i || r'_y, from the evaluations and the closed forms.
    r = rng.point(k + n_y)
    r_i, r_y = r[:k], r[k:]
    x_eval = mle(x_table, r_y)
    b_eval = mle(b_table, r_i[k - k_b:] + r_y)
    z_b, _, w_b = weights(low, k_b, k, r_i)
    eq_eval = eq_points(z_y, r_y)
    inner = sub(mul(inv, x_eval), mul(w_b, mul(z_b, b_eval)))
    fields = {}
    if high:
        h_eval = mle(h_table, r_i[k - k_h:] + r_y)
        z_h, _, w_h = weights(high, k_h, k, r_i)
        inner = sub(inner, mul(two_low, mul(w_h, mul(z_h, h_eval))))
        fields.update({"h_eval": fp2(h_eval), "z_h": fp2(z_h), "w_h": fp2(w_h)})
    final = mul(eq_eval, inner)
    # The same value from the tables: eq times 2^{-k} x - W_b b-hat - 2^low W_h h-hat, each the
    # extension of its own table at r'.
    direct = sub(mul(inv, x_eval), mul(mle(w_b_table, r_i), mle(b_hat, r)))
    if high:
        direct = sub(direct, mul(two_low, mul(mle(w_h_table, r_i), mle(h_hat, r))))
    require(final == mul(eq_eval, direct), "the final value's closed forms are the extensions")
    fields.update({"x_eval": fp2(x_eval), "b_eval": fp2(b_eval), "z_b": fp2(z_b), "w_b": fp2(w_b)})
    return {
        "z_y": points(z_y),
        "bit_axis_sums": sums,
        "cube_sum": fp2(cube_sum),
        "final_point": points(r),
        "eq": fp2(eq_eval),
        **fields,
        "final": fp2(final),
    }


def recomposition_cases(rng):
    cases = []
    for low, high in WIDTHS:
        for shape in RECOMPOSITION_SHAPES[(low, high)]:
            real = [tuple(idx) for _, idx in positions(shape) if all(i < n for i, n in zip(idx, shape, strict=True))]
            x = {pos: rng.below(1 << (low + high)) for pos in real}
            # The edges of the width at the first two positions, when there are two.
            x[real[0]] = (1 << (low + high)) - 1
            if len(real) > 1:
                x[real[1]] = 0
            k_b, k_h, k = bit_vars(low, high)
            base = {"low": low, "high": high, "k": k, "shape": shape, "x": [x[pos] for pos in real]}
            cases.append({**base, "witness": "honest", **recomposition(rng, low, high, shape, x, None)})
            # A false bit: the top carry when there is one, else the lowest result bit.
            flip = ("h", real[-1], high - 1) if high else ("b", real[-1], 0)
            cheat = {"witness": f"{flip[0]}[{', '.join(map(str, flip[1]))}][{flip[2]}] flipped"}
            cases.append({**base, **cheat, **recomposition(rng, low, high, shape, x, flip)})
    return cases


def main():
    rng = Inputs(SEED)
    header = {
        "description": "RFC-0072's and RFC-0066's public factors of the anchor relations, each "
        "checked against the direct multilinear extension of its table, written by "
        "spec/vectors/anchor-relations/generate.py; see README.md for the fields of each section",
        "modulus": hex64(P),
        "non_residue": W,
        "seed": hex64(SEED),
    }
    sections = {
        "lt": lt_cases(rng),
        "psi": psi_cases(rng),
        "const_add": const_add_cases(rng),
        "selections": selection_cases(rng),
        "bit_weights": bit_weight_cases(rng),
        "bitfn": bitfn_cases(rng),
        "recomposition": recomposition_cases(rng),
    }
    text = json.dumps(header, indent=2)[:-2]
    for name, cases in sections.items():
        text += f',\n  "{name}": ' + section(cases)
    text += "\n}\n"
    json.loads(text)  # the file must parse
    (HERE / "relations.json").write_text(text)


if __name__ == "__main__":
    main()
