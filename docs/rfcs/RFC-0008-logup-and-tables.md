# RFC-0008: LogUp-GKR Lookups and the Deterministic Table Registry

- Status: Accepted
- Authors: @AbdelStark
- Created: 2026-10-15
- Target milestone: v0.1 (argument), v0.2 (registry)
- Amended by: [RFC-0032](RFC-0032-zero-knowledge-tree-levels.md) ('Fraction tree', in `tessera-protocol/2`: the roots are not sent, and the leaf checks read masked keys, values and multiplicities)

## Summary

Every non-arithmetic relation in Tessera reduces to lookups `(key, value) ∈ T` proven with LogUp-GKR: two fraction trees per table instance, one for the lookups and one for the table, batched across all layers, heads and tokens, with challenges in `Fp2`. Keys are virtual, values and multiplicities are committed. Tables are public, defined by integer-only generators whose outputs are bit-identical across the prover, the verifier and the Python reference, and pinned by digests in the claim list. Large tables (up to 2^32 rows) are cheap for the verifier because every v1 table is a piecewise function whose multilinear extension is evaluated in time proportional to its non-constant core, at most 2^24 entries.

## Motivation

Lookups are half the cost of a proof (source §5: 40 ps per lookup against 30 ps per committed element) and every uniqueness argument leans on them: the exponential lookup proves `c_q ≥ x_qi` (source §7, "Softmax by max certificate"); requantization tables absorb rescales (source §6, principle 4). The source specifies LogUp-GKR with one instance per table batched across layers and degree-2 extension challenges, bringing the LogUp error from VerInf's 2^-28 to about 2^-90 (source §7, "Lookups everywhere are LogUp-GKR"). Tables must also be generated deterministically: a table built with a platform's floating-point `exp` would differ across machines, and the verifier, which regenerates every table, would reject honest proofs.

## Goals

- A LogUp-GKR prover and an independent verifier with soundness error `(N + T)/|Fp2|` plus sumcheck terms.
- Batching of all lookups into a table, across every layer, into one instance.
- A registry of table kinds sufficient for `D`, scoring and the anchor circuits.
- Integer-only generators with pinned digests.
- Verifier table evaluation in `O(n · core)` time.

## Non-Goals

- One-hot or decomposable-table arguments (Lasso, Shout).
- LogUp micro-architecture optimizations beyond the v1 kernels (NG-6).
- Tables whose contents are committed rather than public. `D`'s embedding uses a one-hot matmul instead (RFC-0006).

## Proposed Design

### Relation

For a table instance with rows `t_0 … t_{T−1}` and lookups `a_0 … a_{N−1}`, each a tuple compressed to one `Fp2` element with challenge `β` (`a = key + β·value`, `t = key_t + β·value_t`), the prover commits multiplicities `m_j` and proves

```math
\sum_{i<N} \frac{1}{\alpha - a_i} \;-\; \sum_{j<T} \frac{m_j}{\alpha - t_j} \;=\; 0
```

for a challenge `α ∈ Fp2` drawn after the values and multiplicities are committed. If some `a_i` is not a row, the identity holds for at most `N + T − 1` values of `α`, so the error is at most `(N + T)/|Fp2|`.

### Fraction tree

An instance, with at least one lookup, proves its identity with two fraction trees, one per side, so that neither side is padded to the other's size:

- **The lookup tree** has `2^{h_a}` leaves, `h_a = ⌈log2 N⌉`. Leaf `i < N` is `(1, α − a_i)`. The `2^{h_a} − N` padding leaves repeat row 0, `(1, α − t_0)`, and the multiplicity of row 0 counts them. In the leaf stage the lookups are an instance's column, its padding included, in the leaf order of [RFC-0045](RFC-0045-leaf-layout.md#the-leaves).
- **The table tree** has `2^{h_t}` leaves, `h_t = ⌈log2 T⌉`. Leaf `j < T` is `(−m_j, α − t_j)`. The `2^{h_t} − T` padding leaves repeat row 0, `(−m_j, α − t_0)`, and an honest prover gives them multiplicity 0. A padding row is a copy of row 0, so a multiplicity there only counts more lookups of row 0, and the padding cannot admit a tuple that is not a row. Padding with a row of key and value 0 would admit the lookup `(0, 0)`: nothing checks a committed multiplicity there, and a non-zero one would balance it.
- **Internal nodes.** `(n, d) = (n_L·d_R + n_R·d_L, d_L·d_R)`. Node `x` of level `ℓ` combines entries `2x` (left) and `2x + 1` (right) of level `ℓ + 1`, with the pair bit lowest, as in a product tree ([RFC-0010](RFC-0010-proof-schedule.md#tree-levels)). A root's `d` is the product of its leaves' denominators. When `d ≠ 0`, no leaf denominator is zero, and `n/d` is the sum of the leaves' fractions `Σ n_i/d_i`.
- **Roots.** For every instance, the prover sends the roots `(n_a, d_a)` of the lookup tree and `(n_t, d_t)` of the table tree, in schedule order, before the first tree level (label `logup/roots`, RFC-0014; absorbed even when empty). The instances' schedule order is ascending `TableId` ([RFC-0045](RFC-0045-leaf-layout.md#the-split) 'The check'). The verifier checks `d_a ≠ 0`, `d_t ≠ 0` and `n_a·d_t + n_t·d_a = 0`, else `Reject(Lookup)`. The lookups' fractions and the rows' fractions then sum to zero, which is the identity of 'Relation'. In `tessera-protocol/2` the roots are not sent: a masked sumcheck proves this identity, each tree takes a padding tree, and the leaf checks below read the masked extensions of the keys, values and multiplicities ([RFC-0032](RFC-0032-zero-knowledge-tree-levels.md)).
- **Levels.** Each tree joins the batched levels where its height matches ('Batching'), with its root as the claims on its level 0, on no variable. A level runs one sumcheck of degree 3 per variable over the tree's numerator and denominator claims, combined by a challenge `λ` (Papini and Haböck). It reduces the two claims on level `ℓ` to two claims on level `ℓ + 1`, at one point ([RFC-0010 'Tree levels'](RFC-0010-proof-schedule.md#tree-levels)).
- **Leaf claims.** After its last level, each tree holds claims `Ñ(ρ) = p` on its leaves' numerators and `D̃(ρ) = q` on their denominators. The verifier checks them instance by instance, the lookup tree first:
  - **Lookup tree.**
    - Its numerators are all 1, so the verifier checks `p = 1`, else `Reject(Lookup)`.
    - The prover sends `v = valũe(ρ)`, the value polynomial's evaluation, one per instance whose table has values, in schedule order (label `logup/values`). The message follows the last tree level, or `logup/roots` when no tree has a level, and is absorbed even when empty. A table without values (`Range`) sends none, and `v = 0`.
    - Two claims follow: `kẽy(ρ) = α − q − β·v` on the lookup-side key polynomial, which is virtual and goes to the stage that produces the keys ([RFC-0010](RFC-0010-proof-schedule.md#stages)); and `valũe(ρ) = v` on the value polynomial, which is committed advice.
    - Both polynomials have `2^{h_a}` entries, and their padding slots hold row 0's key and value.
  - **Table tree.**
    - `p = −m̃(ρ)` leaves the claim `m̃(ρ) = −p` on the multiplicity polynomial, committed advice of `2^{h_t}` entries.
    - The verifier evaluates the table polynomial `t̃ = kẽy_t + β·valũe_t` itself ('Structured evaluation'), its padding rows copies of row 0. It checks `q = α − t̃(ρ)`, else `Reject(Lookup)`.

### Batching

- All lookups into one table definition, across every layer, head, token and channel, form one instance. Its lookups are blocks, one per lookup of a site at a layer, aligned and sorted by size ([RFC-0045](RFC-0045-leaf-layout.md#the-column)).
- All instances run their GKR layers in one batched sumcheck per level, with a random combination: the lookup tree and the table tree of each instance, beside the product forests of the leaf stage. Trees of different heights join the batch at the level where their height matches, as in Stwo's LogUp-GKR batching ([RFC-0010 'Tree levels'](RFC-0010-proof-schedule.md#tree-levels)).
- `α` and `β` are shared by all instances and derived from the transcript immediately after Round 1 (RFC-0014, label `logup/alpha`, `logup/beta`).

### Table kinds

| Kind | Key | Value | Rows | Used by |
|---|---|---|---|---|
| `Range { bits }` | `x ∈ [0, 2^bits)` | none | `2^bits` | Split remainders, limbs, brackets |
| `Requant { b, r_by_class, lo, hi }` | `(class, k)`, `k ∈ [−2^{b−1}, 2^{b−1})` | `clamp(floor((k + 2^{r−1}) / 2^r), lo, hi)` | `\|r_by_class\| · 2^b` | `Requant1` (RFC-0009) |
| `Silu { b, r_by_class, in_exp, out_exp }` | `(class, k)` | `clamp(round(silu(h · 2^{in_exp}) / 2^{out_exp}), −128, 127)`, `h = k >> r_c` | `\|r_by_class\| · 2^b` | MLP |
| `Exp { key_bits, sigma, scale }` | `k ∈ [0, 2^{key_bits})` | `round(2^σ · exp(−k · scale))` | `2^{key_bits}` | Softmax, vocabulary softmax |
| `Exp2Frac { frac_bits, out_bits }` | `f ∈ [0, 2^{frac_bits})` | `⌊2^{f/2^{frac_bits}} · 2^{out_bits}⌋`, from below | `2^{frac_bits}` | Scoring bound (RFC-0017) |
| `Pow2 { max_exp }` | `i ∈ [0, max_exp)` | `2^i` | `max_exp` | Scoring bound (RFC-0017) |
| `Bitwise8 { op }` | `(a, b)`, 8-bit each | `a ⊕ b`, `a ∧ b` | 2^16 | SHA-256 |
| `AesSbox`, `Gf256Mul { c }` | `x`, 8-bit | S-box, `c · x` in GF(2^8) | 2^8 | AES-128 |
| `Rope { theta, factor, low_freq_factor, high_freq_factor, original_max, bits, max_context, d_head }` | `(position, frequency)` | `(C, S)`, 14-bit | `max_context · d_head / 2` | Public constants for `Rotate`, not a lookup table |

Signed keys are offset-encoded (`k + 2^{b−1}`); the compressed key adds `2^b · class` so tuples pack into one integer below `2^{B_max}`.

### Table specifications

A claim list names each table by a `TableSpec` (RFC-0007), whose TCE layout is tier A ([02 §1](../spec/02-public-api.md#1-stability-tiers)). An enum's TCE tag is its variant's declaration index (RFC-0003), so kinds are appended, never reordered.

```rust
pub struct TableId(pub u32);
pub struct TableSpec {
    pub id: TableId,
    pub kind: TableKind,
    pub log2_rows: u8,                 // ⌈log2 T⌉ for the kind's T rows, at most 32
    pub digest: Digest,                // 'Generators and digests'
}
pub enum TableKind {
    Range { bits: u8 },
    Requant { b: u8, r_by_class: Vec<u8>, lo: i32, hi: i32 },
    Pow2 { max_exp: u8 },
    Bitwise8 { op: BitOp },
    AesSbox,
    Gf256Mul { c: u8 },
    Exp { key_bits: u8, sigma: u8, scale: Rational },
    Exp2Frac { frac_bits: u8, out_bits: u8 },
    Silu { b: u8, r_by_class: Vec<u8>, in_exp: i8, out_exp: i8 },
    Rope { theta: u32, factor: u16, low_freq_factor: u16, high_freq_factor: u16,
           original_max: u32, bits: u8, max_context: u32, d_head: u16 },
}
pub enum BitOp { Xor, And }
```

Row `j` of each kind, for `j < T`, is below; rows `T` to `2^{log2_rows} − 1` repeat row 0. Keys and values are true integers, mapped to `Fp` in the centered representation ([RFC-0002](RFC-0002-field-arithmetic.md)).

| Kind | `T` | Key of row `j` | Value of row `j` | Parameters allowed |
|---|---|---|---|---|
| `Range` | `2^bits` | `j` | 0 | `1 ≤ bits ≤ 32` |
| `Requant` | `n · 2^b`, `n = \|r_by_class\|` | `j` | `rq(k, r_c, lo, hi)` of [RFC-0005](RFC-0005-integer-model.md#integer-semantics), for the class `c = ⌊j / 2^b⌋` and `k = (j mod 2^b) − 2^{b−1}` | `b ≥ 1`, `1 ≤ n ≤ 4`, `b + ⌈log2 n⌉ ≤ 32`, `1 ≤ r_c ≤ 63`, `lo ≤ hi` |
| `Pow2` | `max_exp` | `j` | `2^j` | `1 ≤ max_exp ≤ 63`, so that every value is in the centered range |
| `Bitwise8` | `2^16` | `j` | `(j mod 2^8) ⊕ ⌊j / 2^8⌋` for `Xor`, `∧` for `And` | any |
| `AesSbox` | `2^8` | `j` | `S(j)`, the S-box of FIPS-197 §5.1.1 | none |
| `Gf256Mul` | `2^8` | `j` | `c • j` in GF(2^8) modulo `x^8 + x^4 + x^3 + x + 1` (FIPS-197 §4.2) | any |
| `Exp` | `2^key_bits` | `j` | `round(2^σ · e^{−j · scale})` as 'Fixed-point generators' computes it | `1 ≤ key_bits ≤ 32`, `σ ≤ 32`, `scale > 0`; the zero tail starts by key `2^24` |
| `Exp2Frac` | `2^frac_bits` | `j` | `⌊2^out_bits · 2^{j / 2^frac_bits}⌋` from below, as 'Fixed-point generators' computes it | `1 ≤ frac_bits ≤ 16`, `1 ≤ out_bits ≤ 56` |
| `Rope` | `max_context · d_head / 2` pairs | the pair of position `⌊j / (d_head/2)⌋` and frequency `j mod (d_head/2)` | `(C, S) = (round(2^bits · cos(p·ω_i)), round(2^bits · sin(p·ω_i)))` as 'Fixed-point generators' computes them | `θ ≥ 2`, `factor ≥ 1`, `1 ≤ low_freq_factor < high_freq_factor`, `original_max ≥ 1`, `1 ≤ bits ≤ 30`, an even `d_head ≥ 2`, `1 ≤ max_context`, at most `2^24` pairs |
| `Silu` | `n · 2^b`, `n = \|r_by_class\|` | `j` | `clamp(round(silu(h · 2^{in_exp}) / 2^{out_exp}), −128, 127)` as 'Fixed-point generators' computes it, for `c = ⌊j / 2^b⌋`, `k = (j mod 2^b) − 2^{b−1}` and `h = k >> r_c` | `b ≥ 1`, `1 ≤ n ≤ 4`, `b + ⌈log2 n⌉ ≤ 32`, `r_c < b`, `Σ_c 2^{b − r_c} ≤ 2^24`, `−64 ≤ in_exp ≤ 0`, `1 ≤ 63 − in_exp + out_exp ≤ 126` |

A lookup's key is the row's key:

- `pack(c, z) = (z + 2^{b−1}) + 2^b · c` for `Requant` and `Silu`, as `Requant1` forms it ([RFC-0009](RFC-0009-gadgets.md#requant1));
- `a + 2^8 · b` for the operands `a` and `b` of `Bitwise8`;
- the operand itself for the other kinds.

`Requant`'s and `Silu`'s classes are indices into `r_by_class`. The shifts they stand for are public; which class an output channel takes is hidden ([RFC-0007](RFC-0007-claim-list.md#public-and-hidden-quantization-data)).

Claim-list validation regenerates each spec from its kind and id. Any of the following is `Reject(StatementMismatch)`:

- parameters outside the table above;
- a `log2_rows` other than `⌈log2 T⌉`;
- another digest.

### Structured evaluation

The verifier needs `t̃(ρ)` at the random point where the table tree ends, over its `2^{h_t}` rows, whose padding rows are copies of row 0. With `t = key + β·value` over the `T` rows, `t̃(ρ) = Σ_{j<T} eq(ρ, j)·t_j + t_0·(1 − LT(ρ, T))`: the padding term costs one `LT`, and vanishes when `T` is a power of two. On the `T` rows:

- `key` is an affine function of the row index bits (offset encoding and class packing), so `kẽy(ρ)` is a closed-form `O(n)` expression.
- The sum over a prefix of an affine function of the bits, `Σ_{h<a} eq(ρ, h)·(w_0 + Σ_i w_i·h_i)`, has an `O(n)` closed form: at the highest bit `k` where `h` and `a` differ, `a_k = 1` and `h_k = 0`, the bits above are `a`'s, and over the free bits below, `eq` sums to 1 and each `h_i` to `ρ_i`. With all `w_i = 0` and `w_0 = 1` it is `LT(ρ, a) = Σ_{h<a} eq(ρ, h)`.
- A class of `Requant` has `x = k + 2^{b−1}`, the offset key. Before its clamp, `⌊(k + 2^{r−1})/2^r⌋ = (x >> r) + x_{r−1} − 2^{b−1−r}` for `r < b`, and 0 for `r ≥ b`. That is affine in the bits of `x`: its rounding reads bit `r − 1`, so it does not depend on `x >> r` alone. It never decreases with `x`, so the clamp holds `lo` on a prefix of the class and `hi` on a suffix. `valũe` is then `lo·LT(ρ, a)`, plus the affine prefix sum's difference between `c` and `a`, plus `hi·(1 − LT(ρ, c))`: `O(n)` per class, whatever `b`, `lo` and `hi`.
- For `Silu`, `value(x)` depends only on its class and the high bits `h = x >> r`, so `valũe(ρ) = f̃(ρ_high)`: the low variables sum out because `Σ_{x_low} eq(ρ_low, x_low) = 1`. `Exp` has no shift.
- Their `f` is piecewise: constant below `a`, a dense core on `[a, c)` of at most 2^24 explicit values, and constant on `[c, 2^{n_h})`. Then `f̃(ρ) = A·LT(ρ, a) + Σ_{h=a}^{c−1} f(h)·eq(ρ, h) + C·(1 − LT(ρ, c))`. The dense middle term costs `O(core)`. `[a, c)` is cut into at most `2n` aligned blocks, each an inner product with an `eq` table built by doubling. No step divides: updating `eq` along a Gray code would divide by `ρ_i` and `1 − ρ_i`, which fails at a coordinate 0 or 1.
- `Range` has no value, so `t̃` is `kẽy`'s closed form. `Pow2`, `Bitwise8`, `AesSbox`, `Gf256Mul` and `Exp2Frac` are small enough to evaluate by enumeration.

Verifier cost per table: `O(n · core)`, at most a few times 2^24 field operations, and `O(n)` per class for `Range` and `Requant`. It does not depend on the 2^32 row count the prover enumerates.

### Generators and digests

- Generators use integers only: no floating point, no platform math library, no big-float dependency.
- `Exp`, `Exp2Frac`, `Silu` and `Rope` compute in the fixed point of 'Fixed-point generators', with 128-bit intermediates. Range reduction uses a pinned `ln 2` (respectively `2π`), and a fixed-degree Taylor series is evaluated by Horner's rule with floor division at each step. The constants are the floors of the true constants times `2^64`. `spec/vectors/tables/generate.py` computes them with 100 significant digits (about 330 bits) and stores them in `spec/vectors/tables/constants.json`; every implementation writes them as literals.
- **The generator is the definition.** A table's correctness for the protocol is its exact output, not its closeness to the real function. Accuracy (at most one output unit from the true value for every key) is a fidelity property, tested separately against references of at least 60 significant digits (about 200 bits).
- `TableSpec.digest = H("table", TCE(spec without digest) ‖ TCE(piece parameters) ‖ TCE(core values))`. Hashing the defining data rather than all rows keeps digest computation `O(core)`.
  - The piece parameters are a `Vec<u64>`, and the core values a `Vec<Fp>` in row order, without padding.
  - `Exp` has one piece, `z`, the first key of its zero tail ('Fixed-point generators'), and its core is the values of keys `0` to `z − 1`.
  - `Pow2`, `Bitwise8`, `AesSbox`, `Gf256Mul` and `Exp2Frac` have no piece, and their core is their whole value column: the verifier evaluates them by enumeration.
  - `Silu` has no piece. A row's value depends only on its class and `h = k >> r_c`, so the core is the value at every `h` of every class: class by class, `h` ascending from `−2^{b−1−r_c}` to `2^{b−1−r_c} − 1`, `Σ_c 2^{b − r_c}` values in all.
  - `Rope` has no piece, and its core is every pair in row order, `C` then `S`: `2 · max_context · d_head / 2` values.
  - `Range` and `Requant` have neither: their values are closed forms of their parameters.
- Three implementations exist: `tessera-tables` (prover), `tv-tables` (verifier), `tessera_quant.tables` (Python reference). All are checked against pinned digests.

### Fixed-point generators

The transcendental kinds are defined by the integers below, not by the functions they approximate: an implementation computes exactly these. Division is floor division and `>>` is the arithmetic shift. A Q64 number `v` stands for `v / 2^64`, and every intermediate stays below `2^128`.

- **Constants.** `LN2 = ⌊ln 2 · 2^64⌋ = 0xB17217F7D1CF79AB`, `TAU = ⌊2π · 2^64⌋ = 0x6487ED5110B4611A6` and `HALF_PI = ⌊π/2 · 2^64⌋ = 0x1921FB54442D18469`.
- **`exp_neg(x)`**, for a Q64 `x ≥ 0`, returns `(y, n)` with `e^{−x/2^64} ≈ y · 2^{−64−n}`:
  1. Set `n = ⌊x / LN2⌋` and `r = x − n · LN2`.
  2. Set `y = 2^64`. Then, for `i` from 20 down to 1, set `y ← 2^64 − ⌊r · y / (i · 2^64)⌋`.

  Throughout, `0 < y ≤ 2^64` and `r · y < 2^128`.
- **`exp_pos(x)`**, for a Q64 `x` with `0 ≤ x ≤ LN2`, returns `e^{x/2^64}` in Q62:
  1. Set `y = 2^62`.
  2. Then, for `i` from 20 down to 1, set `y ← 2^62 + ⌊x · y / (i · 2^64)⌋`.

  `y ≤ 2^63` throughout. Every step rounds down and the dropped terms of the series are positive, so `y / 2^62` never exceeds `e^{x/2^64}`.
- **`Exp`** with `scale = num / den` evaluates key `k` as follows:
  1. Write `k · num = q · den + m`.
  2. If `q ≥ 2^20`, the value is 0.
  3. Otherwise set `x = q · 2^64 + ⌊m · 2^64 / den⌋`, the floor of `k · scale · 2^64`, and `(y, n) = exp_neg(x)`.
  4. The value is 0 when `n ≥ σ + 2`. Otherwise it is `(y · 2^σ + 2^{63+n}) >> (64 + n)`, the value rounded half up.

  Key 0 gives exactly `2^σ`.
- **`Exp`'s zero tail.** The keys with `q ≥ 2^20` or `n ≥ σ + 2` form a suffix, because `x` never decreases with `k`. On them the true value is below `1/4`, and the table holds 0. Its first key `z` is the least `k` with `x ≥ (σ + 2) · LN2` or `q ≥ 2^20`, or `T` when no key qualifies. `z ≤ 2^24` bounds the core ('Structured evaluation').
- **`Exp2Frac`** gives key `f` the value `exp_pos(⌊f · LN2 / 2^frac_bits⌋) >> (62 − out_bits)`.
  - It never exceeds `⌊2^out_bits · 2^{f / 2^frac_bits}⌋`, because `LN2`, the argument and every step of `exp_pos` round down. RFC-0017 relies on this when it says that `E2` rounds down.
  - The rounding inside `exp_pos` stays below four units of `2^{−62}`, which `out_bits ≤ 56` keeps far below one output unit.
- **`Silu`** gives `h` (with `|h| < 2^31`) its value as follows:
  1. Set `(y, n) = exp_neg(|h| · 2^{64 + in_exp})`, the Q64 argument being `|x|` for `x = h · 2^{in_exp}`. Set `E = y >> n`, or 0 when `n ≥ 128`: `e^{−|x|}` in Q64.
  2. Set `σ = ⌊2^127 / (2^64 + E)⌋` when `h ≥ 0`, and `⌊E · 2^63 / (2^64 + E)⌋` when `h < 0`. This is the logistic `σ(x)` in Q63.
  3. The value is `clamp((h · σ + 2^{s−1}) >> s, −128, 127)` with `s = 63 − in_exp + out_exp`: `silu(x) / 2^{out_exp}` rounded half up.

  The argument stays below `2^95`, and `|h · σ| < 2^94`.
- **`ln(n)`**, for an integer `1 ≤ n < 2^64`, returns `ln n` in Q64. Write `n = 2^m · u` with `u ∈ [1, 2)`, so that `ln n = m · LN2 + 2 · atanh(z)` with `z = (u − 1)/(u + 1) < 1/3`:
  1. Set `z = ⌊(n − 2^m) · 2^64 / (n + 2^m)⌋` and `w = ⌊z² / 2^64⌋`.
  2. Set `a = ⌊2^64 / 45⌋`. Then, for `k` from 21 down to 0, set `a ← ⌊2^64 / (2k + 1)⌋ + ⌊w · a / 2^64⌋`.
  3. The result is `m · LN2 + ⌊z · a / 2^63⌋`.
- **`cos_sin(θ)`**, for a Q64 angle `θ < 2^100`, returns `(cos θ, sin θ)` in Q62, signed:
  1. Set `t = θ mod TAU`, `q = ⌊t / HALF_PI⌋` and `r = (t − q · HALF_PI) >> 2`, Q62 of the angle within its quadrant.
  2. Set `s = ⌊r² / 2^62⌋` and `c = t' = 2^62`. Then, for `k` from 13 down to 1, set `c ← 2^62 − ⌊s · c / ((2k−1) · 2k · 2^62)⌋` and `t' ← 2^62 − ⌊s · t' / (2k · (2k+1) · 2^62)⌋`, with signed floor division.
  3. With `sin r = ⌊r · t' / 2^62⌋`, quadrant `q mod 4` gives `(c, sin r)`, `(−sin r, c)`, `(−c, −sin r)` or `(sin r, −c)`.
- **`Rope`'s frequencies.** For frequency `i`, set `(y, n) = exp_neg(⌊2i · ln(θ) / d_head⌋)` and `ω = y >> n`, the Q64 of `θ^{−2i/d_head}`. Llama-3 then scales it by its wavelength `2π/ω` against `L = original_max`, and these comparisons decide the bands exactly:
  - kept when `TAU · high_freq_factor < L · ω`;
  - divided, `⌊ω / factor⌋`, when `TAU · low_freq_factor > L · ω`;
  - otherwise interpolated: with `smooth = ⌊(L · ω − low_freq_factor · TAU) · 2^40 / ((high_freq_factor − low_freq_factor) · TAU)⌋`, it becomes `⌊ω / factor⌋ + ⌊⌊ω · smooth / 2^40⌋ · (factor − 1) / factor⌋`.
- **`Rope`'s pairs.** The pair of position `p` and frequency `i` is `(c, s) = cos_sin(p · ω_i)`, each rounded half up as `(v · 2^bits + 2^61) >> 62`.
- **Accuracy**, a fidelity property rather than part of the definition. The oracle checks every key of `Exp`'s core at the test parameters and at the production ones (`σ = 16`, `key_bits = 22`), every key of `Exp2Frac` and every `h` of `Silu`, against 60-digit references, and RoPE at 40 digits:
  - every `Exp` value lies within one unit of `2^σ · e^{−k·scale}`;
  - every `Exp2Frac` value lies at most one unit below the true floor, and never above it;
  - every `Silu` value lies within one unit of `silu(h · 2^{in_exp}) / 2^{out_exp}` clamped to INT8. This holds at every `h` of the test parameters and of a 405B-like gate: `b = 30`, four classes of shifts 14 to 17, `in_exp = −10`, `out_exp = −4`, 122,880 values.
  - every `Rope` constant lies within one unit of `2^bits` times the cosine or sine of Hugging Face's Llama-3 RoPE angle. The oracle checks every pair of the test tables and 4,008 pairs of Llama-3.1's 131,072-position table against 40-digit references. `tessera-tables` checks 10^6 more against an f64 reference.

### Interfaces

```rust
pub trait Table {
    fn spec(&self) -> &TableSpec;
    fn log2_rows(&self) -> u8;
    fn row(&self, index: u64) -> Option<(Fp, Fp)>;         // (key, value); None past the padded rows
    fn eval_mle(&self, rho: &[Fp2], beta: Fp2) -> Result<Fp2, TableError>; // structured evaluation of key + beta*value
    fn index(&self, key: i64, class: u8) -> Option<u64>;   // the row a lookup hits, for multiplicities
    fn lookup(&self, key: i64, class: u8) -> Option<i64>;  // witness-side integer lookup
}

pub struct LogUpInstance<'a> { pub table: &'a dyn Table, pub keys: VirtualPoly, pub values: CommittedPoly,
                               pub multiplicities: CommittedPoly }
pub fn prove_logup(tr: &mut Transcript, instances: &[LogUpInstance], forests: &[Forest],
                   schedule: &[TreeId]) -> ProvenLogUp;    // roots, levels, values, leaf claims
```

The multiplicities are an input: they are committed before `α` ('Relation'). A schedule of `TreeId`s orders the instances' lookup and table trees beside the leaf stage's product forests, which share their levels ('Batching'). The v0.1 prover runs on dense test tables ('Migration / Rollout'): its instance takes the table's rows and the lookups as columns, and `Table` takes the rows' place when the table registry lands.

### Failure modes

| Failure | Response |
|---|---|
| Honest key outside the table | `ERR-INTERNAL` (range analysis bug, INV-IR-2) |
| A root denominator `d_a = 0` or `d_t = 0`, or roots with `n_a·d_t + n_t·d_a ≠ 0` | `Reject(Lookup)` |
| A lookup tree's leaf numerator claim other than 1 | `Reject(Lookup)` |
| Final table evaluation mismatch | `Reject(Lookup)` |
| Generator divergence between implementations | Digest mismatch; `Reject(StatementMismatch)` at claim-list validation; CI conformance failure |
| Multiplicity counter overflow | Impossible: counts are `< N < 2^{48}` and multiplicities are field elements |

## Alternatives Considered

1. **One-hot lookups (Lasso/Shout, as in Jolt Atlas).** Considered for huge decomposable tables. Rejected: with hash-based commitments each lookup commits several one-hot chunks, so per-lookup cost rises, and the separation result cited by the source shows the activation function's structure cannot be exploited to lower it (source §3). The piecewise structured evaluation already makes large tables cheap for the verifier.
2. **Univariate lookups (Plookup, Caulk, cq).** Rejected: they assume univariate polynomial commitments, often pairing-based, incompatible with the multilinear hash-based design.
3. **Base-field LogUp challenges (VerInf).** Rejected: about 2^-28 error at frontier volume.
4. **One combined tree for both sides.** Rejected after the first verifier was written (#62).
   - It pads both sides into one power of two, up to twice the leaves.
   - Its leaf claims mix the two sides under a selector, so either the smaller side is padded to the larger one's size, or the lookup-side key polynomial carries a hole where the table sits.

   Two trees cost one more root to check and one more instance per level.
5. **Shipping table contents with the claim list.** Rejected: a 2^32-row table is 64 GB, and the verifier would trust prover-supplied data instead of regenerating it.
6. **Floating-point or big-float generators (libm, MPFR).** Rejected: libm differs across platforms; MPFR is a C dependency under a copyleft license, outside the verifier allowlist.

## Drawbacks

- Fixed-point generators must be implemented three times with identical rounding.
- The table tree holds all `T` rows, so a small table used rarely still pays its full row count per proof; tables are therefore shared across layers wherever definitions coincide.

## Migration / Rollout

The LogUp-GKR prover and verifier land in v0.1 (with dense test tables) for the unit-cost measurement. The registry, generators and structured evaluation land in v0.2.

## Testing Strategy

- Completeness: random dense tables (2^8 to 2^16 rows) and lookups (2^4 to 2^20) verify; batched instances of different heights verify.
- Soundness: one lookup outside the table, one wrong value, one multiplicity off by one, each rejected; over the `F65537` test field, the empirical false-accept rate of a single random-`α` trial stays at or below `(N + T)/|F|` within a 99% binomial interval. The bound is reached only by an instance whose difference polynomial has `N + T − 1` roots in the field; random bad instances accept far less often.
- Padding: instance sizes `2^k − 1`, `2^k`, `2^k + 1` verify, on both sides.
- Trees and roots: lookup trees of `2^0` to `2^20` leaves beside table trees of `2^0` to `2^16`, joining where their heights match, verify. Each of these is rejected with `Reject(Lookup)`: a zero root denominator on either side; roots whose fractions do not sum to zero; a lookup tree's leaf numerator claim other than 1; a table evaluation off by one.
- Structured evaluation: for each table kind at small parameters (`b ≤ 16`), `eval_mle` equals brute-force `Σ_x eq(ρ, x) t(x)` over the `2^{h_t}` padded rows at 100 random points, `T` both a power of two and not.
- Generators: cross-implementation digests agree; accuracy against a 256-bit reference is within one output unit for every key at test parameters and on 10^6 sampled keys at production parameters.
- Performance: CPU throughput per lookup in `cargo bench` (`tessera-logup`'s throughput benchmark, and `logup/prove` among the microbenchmarks of [08 §6](../spec/08-performance-budget.md#6-regression-gates)); GPU throughput is the v0.1 unit-cost measurement (RFC-0020).

## Open Questions

None.

## References

- Source §3 (separation result), §5 (lookup unit cost), §6 principles 4 and 6, §7 ("Lookups everywhere are LogUp-GKR"), §9 problem 1.
- Haböck, "Multivariate lookups based on logarithmic derivatives", 2022.
- Papini and Haböck, "Improving logarithmic derivative lookups using GKR", 2023.
- Setty and Thaler, "Twist and Shout", 2025 (one-hot alternative).
