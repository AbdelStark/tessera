# Anchor relation vectors

`relations.json` pins the public factors that the anchor circuits' relations need from the verifier, [RFC-0072](../../../docs/rfcs/RFC-0072-anchor-relations.md) with [RFC-0066](../../../docs/rfcs/RFC-0066-bits-and-sha256.md):
- the selector `ψ̃` and `LT`;
- `ConstAdd`'s shift `ĉ`;
- the selection kernels `S̃` and `K̃`;
- recomposition's weights `W̃_b`, `W̃_h` and its zero-extensions;
- the `BitFn` polynomials;
- recomposition's identity and final value on honest and false witnesses.

The prover (`crates/`) and the independent verifier (`verifier/`) each check their own evaluations against it ([#150](https://github.com/AbdelStark/tessera/issues/150)).

[`generate.py`](generate.py) writes it. The script is the independent oracle:
- **Sources.** It was written from RFC-0072, RFC-0066, [RFC-0010 'Multilinear extensions'](../../../docs/rfcs/RFC-0010-proof-schedule.md#multilinear-extensions) and [RFC-0002](../../../docs/rfcs/RFC-0002-field-arithmetic.md#quadratic-extension) alone, by an author who read neither implementation.
- **Arithmetic.** Python's arbitrary-precision integers, `Fp2` multiplied by the schoolbook formula.
- **Two computations per value.** Each value is computed twice: by the closed form RFC-0072 gives the verifier, and as the multilinear extension of the function's table, enumerated over the whole padded cube. The script stops unless they agree.
- **Inputs.** They come from SplitMix64 with the seed the file states, words at or above `P` rejected.
- **Reproducible.** `python3 -I spec/vectors/anchor-relations/generate.py` rewrites the file byte for byte in about 3 seconds, and `cargo xtask gen-vectors --check` fails when it differs.

## Conventions

The conventions are those of [`../poly/`](../poly/README.md):
- **Field elements.** `Fp` is Goldilocks, an element written `"0x"` and 16 hex digits. `Fp2 = Fp[X]/(X^2 − 7)`, an element `[c0, c1]`. Every point and value below is in `Fp2`.
- **Constants.** Integer constants, such as a `ConstAdd`'s, are decimal and may be negative; they enter the field by RFC-0002's signed embedding.
- **Tensors and points.** A tensor of shape `(n_0, …, n_{d−1})` is flattened row-major after each axis pads to `2^{⌈log2 n_a⌉}` positions, an axis of length 1 taking no variable. Its last axis takes the lowest variables. A point lists low variables first: the last axis's coordinates, then those of the axis before it, up to axis 0's.
- **Predicates.** `eq(r, x) = Π_i (r_i·x_i + (1 − r_i)·(1 − x_i))`, and `LT(r, a) = Σ_{h < a} eq(r, h)`.

## Sections

| Section | A case | Cases |
|---|---|---|
| `lt` | `n`, `point` (`n` coordinates), `values`: pairs `[a, LT(point, a)]` | the anchor's axis lengths, carry widths and masks: `n` from 0 to 3, 5, 6, 8, 10. `LT(r, 29)` and `LT(r, 22)` on 5 coordinates are the extensions of the `SHR^3` and `SHR^{10}` masks, 1 on rows `i < 32 − r` |
| `psi` | `shape`, `point`, `value = ψ̃(point) = Π_a LT(r_a, n_a)` | 16 shapes, two points each |
| `const_add` | `shape`, `constant` (one value, or one per channel of the last axis), `point`, `psi`, `value = ĉ(point)`. A constant per channel adds `c_tilde = c̃(r_c)`, `wrong = c̃(r_c)·ψ̃(point)`, and `wrong_differs`, whether `wrong` differs from `ĉ` | scalars on `[3]`, `[5, 3]`, `[3, 5, 3]`, `[5, 1]`, `[3, 1]`, `[4, 4]`; per channel on last axes of 3, of 5 and of 2, where `wrong` is right |
| `selections` | `name`, `x_shape`, `axis` `a`, `members`, each `{op, index}` for `GatherPublic` or `{op, start, len}` for `Slice`, with its `coefficient`; `r_a` (`⌈log2 len⌉` coordinates), `r_k` (`⌈log2 n_a⌉`), `kernel = K̃(r_a, r_k)`. Where `x` has at most `2^{12}` padded entries, also the relation on a random `x`: `x` (its real entries, row-major, in `Fp`), `claim_point` (a point of `y`), `claim = Σ_t a_t·ỹ_t(claim_point)`, `x_point = r_lo ‖ r_k ‖ r_hi`, `x_value = x̃(x_point)` and `final = kernel·x_value` | the rotations of `Σ0`, `Σ1`, `σ0` and `σ1` and the `SHR` gathers at `M = 1` and 3; the even and odd rows of each value level, alone and combined (`1` and `2^{2^ℓ}`, and under a claim's coefficient in `Fp2`); the message assembly of the key commitment and of a frame of one token; `Slice` on axes 0, 1 and 2, of lengths 3, 4, 5 and 29; gathers of lengths 3 and 5 with repeated indices; combinations of slices of lengths 3 and 4, of a gather and a slice, and of three slices on axis 1 |
| `bit_weights` | `low`, `high`, `k_b`, `k_h` (null when `high = 0`), `k`, `point` (`k` coordinates), `z_b = Π_{t < k − k_b}(1 − r_t)`, `sum_b = Σ_{j < low} 2^j·eq(s, j)`, `w_b = z_b·sum_b = W̃_b(point)`, and the same for `h` when `high > 0` | `(low, high)` in `(8, 0)`, `(32, 1)`, `(32, 2)`, `(32, 3)`, `(3, 0)`, `(5, 3)`, `(1, 0)`, and `(3, 5)` and `(1, 2)`, whose carry has more bit variables than the result so that `z_b` is not 1; two points each |
| `bitfn` | `f`, `inputs`, `truth_table` (rows `[inputs…, output]`, the first input most significant), `evaluations` (`inputs` in `Fp2`, `value` the polynomial) | `Xor`, `Xor3`, `And`, `Ch`, `Maj` |
| `recomposition` | `low`, `high`, `k`, `shape` (`x`'s), `x` (its real values, row-major, integers in `[0, 2^{low + high})`), `witness`, `z_y`, `bit_axis_sums`, `cube_sum`, `final_point`, `eq`, `x_eval`, `b_eval`, `z_b`, `w_b`, and `h_eval`, `z_h`, `w_h` when `high > 0`; then `final` | each width above, `x` of shape `[3]`, and also `[5, 3]` for `(8, 0)` and `[1]` for `(32, 3)`, and `[2, 3]` alone for `(1, 2)`; each with the honest bits and with one bit flipped |

### `ĉ`, RFC-0072 'Linear operations that are not coordinate maps'

- **A constant of one value** gives `ĉ(r) = c·ψ̃(r)`, whatever the last axis's length.
- **A constant per channel** gives `ĉ(r) = c̃(r_c)·Π_{a ≠ last} LT(r_a, n_a)`. Here `c̃` is the extension of the constant's values over the last axis's coordinates `r_c`, 0 at padded channels.
- **The direct value.** Each is checked against the extension of the table `ψ·c`, the constant at real positions and 0 elsewhere, enumerated over the padded cube.
- **`wrong`.** `c̃(r_c)·ψ̃(r)` agrees with `ĉ` on the cube only. A test must reject a prover or verifier that uses it. It differs from `ĉ` exactly when the last axis's length is not a power of two, which the script checks.

### Selections, RFC-0072 'The selection relation' and 'Combined selections'

- **Rows.** `idx_j` is `index[j]` for `GatherPublic`, which selects axis 0, and `start + j` for `Slice`. `n_a` is `x_shape[axis]`, and `len` is the member's row count.
- **The kernel.** `kernel = K̃(r_a, r_k) = Σ_t a_t Σ_{j < len_t} eq(r_a, j)·eq(r_k, idx_{t,j})`, with `a_t` the members' coefficients; a single selection has one member of coefficient 1.
- **The direct value.** Each kernel is checked against the extension of the table `K(j, k) = Σ_t a_t·[j < len_t]·[idx_{t,j} = k]` over the whole cube of `(j, k)`.
- **Combinations.** Their members may differ in length, but take the same number of variables on `y`'s axis, at one point.
- **The relation.**
  - The claim point lists `y`'s coordinates, `y` with axis `a` of length `max len_t`, so `r_a` is that point's coordinates of axis `a`.
  - `r_lo` are the coordinates of the axes after `a`, the lower variables, and `r_hi` those of the axes before it.
  - The script checks `claim = Σ_{k ∈ {0,1}^{⌈log2 n_a⌉}} K̃(r_a, k)·x̃(r_lo ‖ k ‖ r_hi)`.
  - `final` is the selection instance's final value at `r_k`.
- **The message assembly.** It is RFC-0066's gather of the committed bits, `[8K, M] → [512·N, M]`. It puts committed byte `k`'s bit `t`, row `8k + t`, at row `32·⌊o/4⌋ + 8·(3 − o mod 4) + t` for the byte's offset `o`, and row 0 at every other row.
  - The key commitment's committed bytes are at offsets 21 to 52, `N = 1`.
  - The frame's are at offsets 41 to 66, `N = 2`.

### Recomposition, RFC-0072 'The bit zero-checks'

- **Variables.** `k_b = ⌈log2 low⌉`, `k_h = ⌈log2 high⌉`, and `k = max(k_b, k_h)`, or `k_b` when `high = 0`. The bit coordinates are the lowest `k` of a point, and `s` is the high `k_b` of them, low first.
- **The weights.** `W_b(i) = 2^j` at `i = j·2^{k − k_b}` for `j < low`, and 0 elsewhere. Its extension `W̃_b = z_b·sum_b` is checked against the extension of that table. When `low = 2^{k_b}`, `sum_b` is also checked equal to `Π_t (1 − s_t + s_t·2^{2^t})`. `W_h` is the same with `high` and `k_h`.
- **The zero-extensions.** `b̂(i ‖ y) = b̃(i_{≥ k − k_b} ‖ y)·z_b(i)`, checked against the table of `b` placed at `i = j·2^{k − k_b}` and 0 elsewhere.
- **The witness.**
  - `b` and `h` are `x`'s bits: `b` has `x`'s shape with an innermost axis of `low`, padded to `2^{k_b}`, and `h` one of `high`.
  - The cheating witness flips one bit: `h`'s top bit at `x`'s last real position when `high > 0`, else `b`'s lowest bit there. `witness` names it.
  - `x`'s first real value is `2^{low + high} − 1`, and its second, if any, 0.
- **`bit_axis_sums`.**
  - For each `y` of `x`'s padded cube, in index order, the integer `Σ_i [2^{−k}·x(y) − W_b(i)·b̂(i ‖ y) − 2^{low}·W_h(i)·ĥ(i ‖ y)]` over the `2^k` bit positions.
  - The script checks that it is RFC-0066's recomposition `x − Σ_i 2^i·b_i − 2^{low}·Σ_j 2^j·h_j`: 0 everywhere on the honest witness, and nonzero at the flipped position.
- **`cube_sum`.** `Σ_{(i, y)} eq(z_y, y)·[…]`, the zero-check's sum at `z_y`, the instance's `eq` coordinates. It is 0 exactly on the honest witness, which the script checks.
- **The final check.**
  - At `final_point = r'_i ‖ r'_y`, with `r'_i` its lowest `k` coordinates:
    - `eq = eq(z_y, r'_y)`;
    - `x_eval = x̃(r'_y)`;
    - `b_eval = b̃(r'_{i, ≥ k − k_b} ‖ r'_y)` and `h_eval = h̃(r'_{i, ≥ k − k_h} ‖ r'_y)`;
    - `z_b`, `w_b`, `z_h` and `w_h` at `r'_i`.
  - The final value is `final = eq·[2^{−k}·x_eval − w_b·z_b·b_eval − 2^{low}·w_h·z_h·h_eval]`.
  - `z_b` appears twice: in `W̃_b`, and in `b̂`'s zero-extension. RFC-0072 says both parties use that polynomial.
  - The script checks the value against `eq` times the extensions of the tables `W_b`, `b̂`, `W_h` and `ĥ` at `final_point`.

## Who checks them

The prover's and the verifier's tests of the anchor relations (#526, #150) read this file. Each implementation evaluates every value with its own closed forms and compares it with the case. No test reads it yet: the verifier's relations land in #150's later slices.
