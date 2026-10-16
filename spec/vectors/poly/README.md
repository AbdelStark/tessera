# Multilinear-extension vectors

`poly.json` pins the conventions of [RFC-0010 'Multilinear extensions'](../../../docs/rfcs/RFC-0010-proof-schedule.md#multilinear-extensions): how a table of `2^n` values is a multilinear polynomial, what binding its lowest variable gives, the `eq` tables, the less-than predicates and the causal selector, and the evaluation of a round polynomial from its message. `crates/tessera-poly/tests/vectors.rs` checks every case.

[`generate.py`](generate.py) writes it. The script is the independent oracle: it computes with Python's arbitrary-precision integers, multiplies in `Fp2` by the schoolbook formula of [RFC-0002](../../../docs/rfcs/RFC-0002-field-arithmetic.md#quadratic-extension), and computes every value from its definition rather than by the algorithms of `tessera-poly`. Its inputs come from SplitMix64 with the seed the file states, words at or above `P` rejected, so `python3 spec/vectors/poly/generate.py` reproduces the file byte for byte. The file is about 340 KB.

## Conventions

- `Fp` is Goldilocks, `P = 2^64 − 2^32 + 1`, and an element is written `"0x"` and 16 hex digits. `Fp2 = Fp[X]/(X^2 − 7)`, and an element is `[c0, c1]`.
- A table `f[0], …, f[2^n − 1]` is the multilinear `f̃(x_1, …, x_n)` equal to `f[Σ_i x_i·2^{i−1}]` on `{0, 1}^n`: `x_1` is the least significant bit of the index. A point is the list `r_1, …, r_n`.
- `eq(r, x) = Π_i (r_i·x_i + (1 − r_i)·(1 − x_i))`, and `f̃(r) = Σ_x f[x]·eq(r, x)`.

## Sections

| Section | A case | Cases |
|---|---|---|
| `multilinear` | `field` (`Fp` or `Fp2`), `n`, `table` (`2^n` entries of that field), `point` (`n` coordinates in `Fp2`), `value` `= f̃(point)`, and for `1 ≤ n ≤ 8` `bound`: the `Fp2` table `f[2k] + r_1·(f[2k+1] − f[2k])` after binding `x_1` to the point's first coordinate | `Fp` tables of `2^0` to `2^10` and `2^12` entries; `Fp2` tables of `2^0` to `2^9` |
| `eq_tables` | `n`, `point`, `table`: the `2^n` values `eq(point, x)` in index order | points of 0 to 10 coordinates |
| `less_than` | `n`, `point`, `values`: pairs `[a, LT(point, a)]`, with `LT(r, a) = Σ_{h < a} eq(r, h)` over the `n`-bit integers `h` and `a` written like an `Fp` element | every `a` in `0..=2^n` for `n ≤ 4`; for `n` in 5, 8, 12, 16, 32, 63, 64, 65 and 80, with `m = min(2^n, 2^64)`: `0`, `1`, `2`, `m − 1`, `m/2`, `m/2 + 1`, three random `a < m`, and `2^n` itself when `n < 64` |
| `less_than_points` | `n`, positions `q` and `i` of `n` coordinates, `lt_iq = LT2(i, q)`, `lt_qi = LT2(q, i)`, `causal = C(q, i)` and `eq = eq(q, i)`, with `LT2(x, y) = Σ_{x' < y'} eq(x, x')·eq(y, y')` and `C(q, i) = Σ_{i' ≤ q'} eq(q, q')·eq(i, i')` | `n` from 0 to 10 |
| `round_polys` | `degree` `d`, `evaluations` `p(0), p(2), …, p(d)` as a round message sends them, `claim = p(0) + p(1)`, a challenge `r` and `value = p(r)` | four per degree from 1 to 7, the last at a node: 0, 1 or `d` |

## How the oracle computes

- `f̃(r)` is the sum over the hypercube of `f[x]·eq(r, x)`, with each `eq(r, x)` the product of its factors. A bound table comes from the formula of RFC-0010.
- Each `eq` table entry is the product of its factors.
- `LT(r, a)` is the sum of `eq(r, h)` over every `h < a` for `n ≤ 16`. Above that it is the sum over the dyadic blocks of `[0, a)`, which the script checks against the enumeration for `n ≤ 12`.
- `LT2`, `C` and `eq(q, i)` are sums over all `4^n` pairs of positions.
- A round polynomial is drawn by its `d + 1` coefficients in `Fp2`, and every value comes from Horner's rule.

Before it writes the file, the script checks:
- that binding `x_1` then evaluating at the remaining coordinates gives `f̃(point)`;
- that each `eq` table sums to 1;
- that `LT(r, 0) = 0` and `LT(r, 2^n) = 1`;
- that the pair sums split all pairs, `LT2(i, q) + LT2(q, i) + eq(q, i) = 1`;
- that the diagonal sum equals `eq(q, i)` computed from its product formula.
