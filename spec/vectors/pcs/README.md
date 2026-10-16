# Advice opening vectors

`opening.json` holds openings of the advice commitment of [RFC-0011 'Opening protocol'](../../../docs/rfcs/RFC-0011-advice-commitment.md#opening-protocol), at small sizes. Each opening carries its commitment (the layout, `commitment_id` and root), its claims grouped by column point with their batching challenges, the proximity point `ξ*`, the combined rows, the drawn column indices and the answer to them: the distinct columns and their multiproof. The `openings` are honest, and a verifier of RFC-0011 steps 6 and 8 accepts each of them; every one also records the values its checks compute, so an implementation can locate the first step where it diverges. The `rejections` are honest openings changed in exactly one place, and the verifier rejects each with the `reason`, `group` and `column` it states.

[`generate.py`](generate.py) writes it. The script is the independent oracle, an honest prover of RFC-0011 commitments and openings written from the RFC text alone, with Python's arbitrary-precision integers. It imports [`../hash/generate.py`](../hash/generate.py) for BLAKE3, `H`, Merkle trees, canonical multiproofs and column chains, and [`../encoding/generate.py`](../encoding/generate.py) for roots of unity and the masked encoding, as [`../sumcheck/generate.py`](../sumcheck/generate.py) imports the transcript oracle. Before anything is written it reruns the official BLAKE3 test vectors, and checks its `Fp2` products and inverses and its roots of unity against [`../field/`](../field/). Its inputs come from SplitMix64 with the seed the file states, words at or above `P` rejected, so `python3 spec/vectors/pcs/generate.py` reproduces the file byte for byte. The file is about 205 KB.

## Conventions

- `Fp` is Goldilocks, `P = 2^64 − 2^32 + 1`, and an element is written `"0x"` and 16 hex digits. `Fp2 = Fp[X]/(X^2 − 7)`, and an element is `[c0, c1]`. A digest or `commitment_id` is 64 lowercase hex digits. Indices and counts are JSON numbers.
- `ω_m = 7^((P − 1)/m)` is the primitive `m`-th root of unity of [RFC-0002](../../../docs/rfcs/RFC-0002-field-arithmetic.md#constants). A row has `k = 2^log2_k` values and its codeword `n = k·2^log2_inv_rate = 2k` symbols. `H` is the subgroup of order `k`, and column `j` sits at `x_j = 7·ω_n^j`, on the coset `7·L` of [RFC-0011 'Encoding'](../../../docs/rfcs/RFC-0011-advice-commitment.md#encoding). `Z_H(X) = X^k − 1`.
- A point lists its coordinates lowest variable first, and coordinate `t` sets bit `t` of the index: `eq(r, x) = Π_t (r_t·x_t + (1 − r_t)·(1 − x_t))`, with `x_t` bit `t` of `x`.
- The matrix has `2^row_vars` rows. Row `A_r` holds `f_r(ω_k^0), …, f_r(ω_k^{k−1})` for `f_r` of degree below `k`; its mask is `m_r = Σ_{i < P_A} m_{r,i}·X^i`; symbol `j` of its codeword is `g_r(x_j) = f_r(x_j) + Z_H(x_j)·m_r(x_j)`. A zero row holds only zeros and has no mask and no codeword.
- The matrix's extension takes the column point as its low `log2_k` coordinates and the row point as its high ones: `Ã(column_point ‖ row_point) = Σ_row eq(row_point, row)·Σ_i A_row[i]·eq(column_point, i)`.
- Column digests follow [RFC-0004 'Column digests'](../../../docs/rfcs/RFC-0004-hashing-and-transcript.md#column-digests), one chain per segment over that segment's non-zero rows, `m` of them, in layout order: `d_j[0] = H("column-init", commitment_id ‖ u64(j))`, `d_j[t+1] = H("column-step", d_j[t] ‖ the symbols of rows tB … min((t+1)B, m) − 1 at column j)`, each symbol its 8 little-endian bytes, and `d_j^(s) = d_j[⌈m/B⌉]`. An empty segment's digest is therefore `d_j[0]`. Then `leaf_j = H("column-merge", d_j^(1) ‖ … ‖ d_j^(S))`, for every `S ≥ 1` ([RFC-0011 'Column digests and root'](../../../docs/rfcs/RFC-0011-advice-commitment.md#column-digests-and-root)), and the root is the Merkle root over the `n` leaves, with `merkle-node` nodes and canonical multiproofs ([RFC-0004 'Merkle trees'](../../../docs/rfcs/RFC-0004-hashing-and-transcript.md#merkle-trees)).
- No value comes from a transcript. The batching challenges `γ`, the point `ξ*` and the draws are inputs, as an opening check receives them. RFC-0011 does not order the groups; the vectors fix the order as listed. `P_A` and the number of draws `τ` are independent here: RFC-0013's hiding condition `τ_A ≤ P_A` is a rule of the parameter sets, not an opening check.

## The file

| Field | Meaning |
|---|---|
| `description` | what the file holds |
| `modulus`, `non_residue`, `coset_shift` | `P`, the `7` of `X^2 − 7`, the `7` of `7·L` |
| `seed` | the SplitMix64 seed of every input |
| `official_blake3_cases_checked` | how many official BLAKE3 test vectors the hash oracle reproduced before the file was written (35, in all three modes) |
| `openings` | the honest openings, in the order of the table below |
| `rejections` | the changed openings, in the order of the table below |

## Openings

An opening is one honest advice opening, which the verifier must accept.

| Field | Meaning |
|---|---|
| `name` | what the case exercises |
| `log2_k` | row width `k = 2^log2_k` |
| `log2_inv_rate` | `n = k·2^log2_inv_rate` columns, the codeword length; 1 throughout, rate 1/2 |
| `mask_coeffs` | `P_A`, the number of mask coefficients per row, `1 ≤ P_A ≤ k` |
| `row_block` | `B`, rows per chain step of a column digest |
| `row_vars` | the row index has `row_vars` bits; rows are `0 … 2^row_vars − 1` |
| `commitment_id` | the 32 bytes bound into `column-init` |
| `segments` | the non-zero rows of each segment, as row indices, in layout order. Rows increase within and across segments, and every row not listed is a zero row. A segment may be empty |
| `root` | the Merkle root over the `n` leaves `leaf_j` |
| `groups` | in group order, each `{"column_point": [Fp2 × log2_k], "claims": [{"row_point": [Fp2 × row_vars], "gamma": Fp2, "value": Fp2}, …]}`. A claim says `Ã(column_point ‖ row_point) = value`, with `γ` its batching challenge |
| `xi_point` | `ξ*`, `row_vars` coordinates: `ξ_row = eq(ξ*, row)` |
| `combined` | the `G + 1` combined rows in group order, the proximity row last, each `{"u": [Fp2 × k], "mask": [Fp2 × P_A]}`: `u_g = Σ_row c_g(row)·A_row` and `M_g = Σ_row c_g(row)·m_row` over the non-zero rows, where `c_g(row) = Σ_{claims in g} γ·eq(row_point, row)`, then `u_ξ` and `M_ξ` with `ξ_row` (RFC-0011 steps 2 to 5) |
| `queries` | the `τ` drawn column indices, in draw order. They may repeat and need not be increasing |
| `columns` | one per distinct queried index, in increasing index order. Each holds one `Fp` symbol per non-zero row, segment by segment, in layout order |
| `siblings` | the canonical multiproof of the distinct indices |
| `intermediate` | for locating a divergence: `row_coefficients`, the `G + 1` lists of `c_g(row)`, `ξ_row` last, each over the non-zero rows in layout order; `points`, `x_j` for each distinct query; `leaves`, `leaf_j` for each distinct query; `combined_at_queries`, the `G + 1` lists of `enc(u)(x_j) + Z_H(x_j)·M(x_j)`, each over the distinct queries. `enc(u)` is the polynomial of degree below `k` that takes the values `u` on `H` |

An honest opening satisfies step 6, `Σ_{claims in g} γ·value = Σ_i u_g[i]·eq(column_point, i)` for every group, and step 8.3, `Σ_row c(row)·col_j[row] = enc(u)(x_j) + Z_H(x_j)·M(x_j)` for every distinct `j` and every combined row.

## Rejections

A rejection is `{"name", "opening", "reason", "group", "column"}`. `opening` is a complete opening without `intermediate`, which an honest opening of `openings` became after exactly the one change that `name` describes; its own `name` is that honest opening's, so the two can be compared. The checks run in this order, and the first failure decides the verdict:

1. RFC-0011 step 6: the shape of `combined` (`G + 1` rows, each `u` of `k` values and `mask` of `P_A` coefficients), then each group's sum in group order. A failure is `"Evaluation"`, with `group` the index of the first group whose sums differ, or `null` when the shape is wrong; `column` is `null`.
2. Step 8.1: one column per distinct query, each with one symbol per non-zero row.
3. Step 8.2: the leaves, recomputed from the columns, and the multiproof of the distinct indices against the root. A failure of 8.1 or 8.2 is `"MerklePath"`, with `column` the smallest queried index, wherever the fault lies: a multiproof fails as a whole. `group` is `null`.
4. Step 8.3: consistency at each distinct `j` in increasing order, the groups in order, then `ξ`. A failure is `"Proximity"`, with `column` the first `j` at which some combined row is inconsistent with the columns. `group` is `null`.

The reasons are `Reject(Evaluation)`, `Reject(MerklePath)` and `Reject(Proximity)` of [04 §2](../../../docs/spec/04-error-model.md#2-verdicts), as RFC-0011 steps 6 and 8 assign them; its ['Failure modes'](../../../docs/rfcs/RFC-0011-advice-commitment.md#failure-modes) table lists the same three.

## Cases

| # | Opening | `k`, `n` | `P_A` | `B` | `row_vars` | `segments` | Claims per group | Draws |
|---|---|---|---|---|---|---|---|---|
| 1 | one row: the row points and `ξ*` are empty, and `ξ_0 = 1` | 4, 8 | 1 | 1 | 0 | `[0]` | 1 | `7`, that is `n − 1` |
| 2 | the main opening, which most rejections change: zero rows between the segments (3), inside the second (5) and after it (7); claims on whole tensors through Boolean row coordinates, so that `c_0` vanishes on rows 4 and 6, `c_1` on 0, 1, 2 and 6, and `c_2` on 6: no claim touches row 6. The draw repeats 6, and its distinct indices in draw order, 6, 1, 4, are not increasing | 4, 8 | 2 | 2, a partial last block in the first segment | 3 | `[0, 1, 2]`, `[4, 6]` | 2, 1, 3 | `6, 1, 6, 4` |
| 3 | three segments after a leading zero row, the middle one empty, whose digest is `d_j[0]`; `P_A = k`; one chain step in each other segment; row points that select single rows | 4, 8 | 4 | 32 | 2 | `[1]`, `[]`, `[2, 3]` | 1, 4 | `5, 0` |
| 4 | whole blocks of `B = 2`; every draw the same column | 8, 16 | 1 | 2 | 2 | `[0, 1, 2, 3]` | 5, 2 | `15, 15, 15` |
| 5 | zero rows inside, between and after the segments; `c_1` vanishes on row 12, which a claim of group 2 selects alone | 8, 16 | 3 | 32 | 4 | `[0, 2]`, `[5, 6, 7]`, `[12]` | 1, 2, 3, 5 | 5 uniform: `2, 0, 6, 0, 4` |
| 6 | two segments of one row each; `P_A = k` | 8, 16 | 8 | 1 | 1 | `[0]`, `[1]` | 3 | 6 uniform: `5, 0, 6, 3, 5, 4` |
| 7 | zero rows inside and after one segment | 16, 32 | 1 | 32 | 3 | `[0, 1, 3, 4, 6]` | 4, 1, 2 | 7 uniform: `22, 1, 1, 3, 22, 0, 16` |
| 8 | a zero row between the segments | 16, 32 | 5 | 1 | 2 | `[0, 1]`, `[3]` | 2, 5 | 8 uniform: `24, 28, 10, 27, 27, 15, 27, 0` |
| 9 | three segments after two leading zero rows, the first one empty; `P_A = k`; partial last blocks; `c_1` vanishes on the whole second segment | 16, 32 | 16 | 2 | 4 | `[]`, `[2, 3, 5]`, `[8, 9, 10, 11, 13]` | 5, 1, 2, 3 | 6 uniform: `27, 9, 23, 11, 28, 14` |
| 10 | every column queried, so the multiproof is empty | 4, 8 | 3 | 1 | 1 | `[0, 1]` | 1, 1 | `3, 7, 0, 5, 1, 6, 2, 4` |

Together they cover `log2_k` of 2, 3 and 4; `P_A` of 1, of 2, 3 and 5, and of `k`; `B` of 1, 2 and 32; one, two and three segments, with an empty segment in the middle (opening 3) and first (opening 9); zero rows before, between, inside and after segments; 1 to 4 groups of 1 to 5 claims; every `τ` from 1 to 8; draws that repeat an index; draws whose distinct indices are not increasing; and columns 0 and `n − 1`.

The rejections change opening 2 unless the table says otherwise. Its smallest queried index is 1.

| Change | Reason | `group` | `column` | Decided by |
|---|---|---|---|---|
| opening 1: 1 added to `c0` of the only claim's value | `Evaluation` | 0 | `null` | step 6, sums |
| claim 0 of group 1: its value changed by `X/γ`, so that the group's claimed sum moves by `X` and its `c0` is unchanged | `Evaluation` | 1 | `null` | step 6, sums |
| entry 2 of group 2's `u` changed by `1/eq(column_point, 2)`, so that the group's opened sum moves by 1 and its `c1` is unchanged | `Evaluation` | 2 | `null` | step 6, sums |
| the proximity row missing: `G` combined rows | `Evaluation` | `null` | `null` | step 6, shape |
| group 0's `u` of `k − 1` values, its last dropped | `Evaluation` | `null` | `null` | step 6, shape |
| group 1's mask of `P_A + 1` coefficients, a zero appended: the same polynomial `M` | `Evaluation` | `null` | `null` | step 6, shape |
| the proximity row's `u` of `k + 1` values, a zero appended | `Evaluation` | `null` | `null` | step 6, shape |
| the proximity row's mask of `P_A − 1` coefficients, its last dropped | `Evaluation` | `null` | `null` | step 6, shape |
| an extra combined row, the proximity row sent twice: `G + 2` rows | `Evaluation` | `null` | `null` | step 6, shape |
| group 1's mask: 1 added to `c1` of coefficient 0 | `Proximity` | `null` | 1 | step 8.3 |
| the proximity row's `u`: 1 added to `c1` of entry 3 | `Proximity` | `null` | 1 | step 8.3 |
| the proximity row's mask: 1 added to `c0` of coefficient 1 | `Proximity` | `null` | 1 | step 8.3 |
| a column symbol: 1 added to row 6's symbol in column 6, the largest queried index | `MerklePath` | `null` | 1 | step 8.2 |
| a sibling: one bit of sibling 1 flipped | `MerklePath` | `null` | 1 | step 8.2 |
| siblings 0 and 1 swapped, out of canonical order | `MerklePath` | `null` | 1 | step 8.2 |
| a sibling missing: the last one, at the second level | `MerklePath` | `null` | 1 | step 8.2 |
| an extra sibling: a random digest appended | `MerklePath` | `null` | 1 | step 8.2 |
| the root: one bit flipped | `MerklePath` | `null` | 1 | step 8.2 |
| the distinct columns in draw order, 6, 1, 4 | `MerklePath` | `null` | 1 | step 8.2 |
| the repeated column, 6, sent twice, next to itself | `MerklePath` | `null` | 1 | step 8.1 |
| one column per draw, in draw order: 6, 1, 6, 4 | `MerklePath` | `null` | 1 | step 8.1 |
| column 4 one symbol short, its last dropped | `MerklePath` | `null` | 1 | step 8.1 |
| column 6 one symbol long, a random symbol appended | `MerklePath` | `null` | 1 | step 8.1 |
| a column missing: the last one, column 6 | `MerklePath` | `null` | 1 | step 8.1 |
| an extra column: the committed column 2, not queried, in increasing order | `MerklePath` | `null` | 1 | step 8.1 |
| opening 10: an extra sibling given to its empty multiproof | `MerklePath` | `null` | 0 | step 8.2 |
| a cheating prover: row 4's codeword changed at `D = {4, 6, 7}` and committed; `c_0(4) = 0`, while `c_1(4)` and `c_2(4)` are not | `Proximity` | `null` | 4 | step 8.3 |
| a cheating prover: row 6's codeword changed at `D = {2, 6}` and committed; no claim touches row 6, so only `ξ` sees it | `Proximity` | `null` | 6 | step 8.3 |

A cheating prover's root, columns and multiproof are those of the changed codewords, so steps 8.1 and 8.2 pass, while its claims and combined rows are the honest ones, from the row's message. Its changed codeword is not in the code. The draws hit `D`, but not at their smallest index, 1, so the expected column, the smallest queried index in `D`, differs from it; for row 4 it also differs from 6, the first index of `D` in draw order.

Some changes target specific verifier bugs. The claim's value moves its group's claimed sum by `X`, which a verifier comparing `c0` alone misses, and the `u` entry moves its group's opened sum by 1, which one comparing `c1` alone misses. The mask of `P_A + 1` coefficients is the same polynomial `M`, so only the shape check rejects it, and the extra combined row is caught only by counting the rows.

## How the oracle computes

- Each non-zero row draws its `k` values, then its `P_A` mask coefficients `m_0, m_1, …`, in layout order. `f_r` is interpolated on `H` by Lagrange's formula on coefficient lists, and checked against the inverse DFT from its definition, `f_i = k^{−1}·Σ_t A[t]·ω_k^{−it}`. Each symbol `f_r(x_j) + Z_H(x_j)·m_r(x_j)` is evaluated on its own by Horner's rule, and checked against the encoding oracle's `g = f + (X^k − 1)·m` evaluated on `7·L`. No fast transform is used anywhere.
- Each segment's chains are the hash oracle's `column_leaves` over that segment's codewords; the leaves merge them, and the hash oracle builds the tree and the multiproof.
- A claim's value is the extension of the whole matrix of `2^{log2_k + row_vars}` entries, zero rows included, at `column_point ‖ row_point`, with `eq` tabulated one coordinate at a time; it is checked against the per-row sum of the conventions above. A row point's coordinates are random, or Boolean where a claim covers one tensor's rows.
- `c_g`, `ξ_row`, `u` and `M` follow their definitions, and `enc(u)(x_j)` interpolates `u` on `H`, coordinate by coordinate, then evaluates by Horner's rule.
- A separate checker then runs the checks above on every opening, from the fields written to the file: `eq` from its product formula, `enc(u)(x)` from the barycentric formula on `H`, `L_i(x) = (x^k − 1)·ω^i / (k·(x − ω^i))`, `M(x)` as a sum of powers, each leaf from its opened column, and the multiproof by rebuilding the root from the opened leaves, each sibling taken as it is needed.

Before it writes the file, the script checks:
- the official BLAKE3 vectors, in all three modes; the 4145 products and inverses of [`../field/goldilocks_ext2.json`](../field/goldilocks_ext2.json); and every root of unity of [`../field/constants.json`](../field/constants.json) as `7^((P − 1)/m)`;
- that every codeword lies in the code: interpolating all `n` symbols on `7·L` gives exactly `f + Z_H·m`, of degree below `k + P_A`, and, when `k + P_A < n`, interpolating `k + P_A` of them reproduces the others, for the first, the last and two random sets of positions. When `P_A = k`, `k + P_A = n`, so only the first check applies;
- that the mask vanishes on `H`, where `g` takes the row's values, and that `7·L` misses `H`;
- that each claim's value is the per-row sum, that each group's evaluation identity holds, and that every combined row agrees with the columns at every one of the `n` columns, not only the queried ones;
- that the checker accepts every honest opening as written in the file and recomputes its `intermediate` exactly;
- that every rejection differs from its honest opening at exactly the paths of its stated change, and that the checker rejects it with its `reason`, `group` and `column` at the step the table above names;
- for a cheating prover, that the codeword changes at exactly `D`, is not in the code, and that the opening differs from the honest one only in its root, that row's symbols in `D` and its multiproof;
- that the file parses back to the values computed, and stays under 1 MB.
