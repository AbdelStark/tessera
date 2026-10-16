# Reed–Solomon encoding vectors

`encoding.json` pins the conventions of [RFC-0011 'Encoding'](../../../docs/rfcs/RFC-0011-advice-commitment.md#encoding): which root of unity each transform uses, natural order on both sides, the coset `7·L` of the codeword, and where the vanishing mask's coefficients go and in which order they are drawn. `crates/tessera-pcs/tests/vectors.rs` checks every case.

[`generate.py`](generate.py) writes it. The script is the independent oracle: it computes with Python's arbitrary-precision integers, and every value from its definition rather than by a fast transform or any algorithm of `tessera-pcs`. Its inputs come from SplitMix64 with the seed the file states, words at or above `P` rejected, so `python3 spec/vectors/encoding/generate.py` reproduces the file byte for byte. The file is about 81 KB.

## Conventions

- `Fp` is Goldilocks, `P = 2^64 − 2^32 + 1`, and an element is written `"0x"` and 16 hex digits.
- `ω_m = 7^((P − 1)/m)` is the primitive `m`-th root of unity of [RFC-0002](../../../docs/rfcs/RFC-0002-field-arithmetic.md#constants), for `m` a power of two.
- A polynomial is the list of its coefficients, constant first.
- A transform of `n` points is in natural order on both sides: position `j` holds the value at `ω_n^j`, or at `7·ω_n^j` on the coset.
- A row of `k` values holds `f(ω_k^0), …, f(ω_k^{k−1})` for a polynomial `f` of degree below `k`.
- The mask is `m = Σ_{i < P} m_i·X^i` with `0 ≤ P ≤ k`, its coefficients drawn in the order `m_0, m_1, …`. The encoded polynomial is `g = f + (X^k − 1)·m`, and symbol `j` of the codeword of `n = 2k` symbols is `g(7·ω_n^j)`.

## Sections

| Section | A case | Cases |
|---|---|---|
| `ntt` | `n`, `coefficients` (`n` of them), `values`: `f(ω_n^j)` for `j < n`, and `coset_values`: `f(7·ω_n^j)` | `n` from `2^1` to `2^8` |
| `encoding` | `k`, `mask_coeffs` (`P`), `f` (`k` coefficients), `row` (`k` values), `mask` (`m_0, …, m_{P−1}`) and `codeword` (`2k` symbols) | `k` from `2^0` to `2^6`, each with `P` in `{0, 1, k/2, k}` where distinct: 25 cases |

## How the oracle computes

- `ω_m` is `pow(7, (P − 1)/m, P)`, checked to have order exactly `m`; and 7 is checked to generate `Fp^*`, by `7^((P − 1)/q) ≠ 1` for every prime `q` dividing `P − 1`.
- Every evaluation is Horner's rule at its point, each point `s·ω_n^j` computed with its own exponentiation.
- A row is the values of random coefficients `f` on the subgroup `H` of order `k`.
- `g` comes from polynomial arithmetic on coefficient lists: `X^k·m` shifts `m` up by `k` places, and `−m` subtracts it at the bottom.

Before it writes the file, the script checks for each encoding case:
- that `g` takes the row's values on `H`, where the mask vanishes;
- that no codeword point `7·ω_n^j` lies in `H`;
- that `g` has degree below `k + P`, with `m` as its coefficients `k` to `k + P − 1`.
