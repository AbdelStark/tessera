// SPDX-License-Identifier: Apache-2.0 OR MIT

//! Integer-only table generators and structured multilinear evaluation of tables (RFC-0008),
//! written from RFC-0008 and the specifications it cites, not from the prover's tables crate,
//! so that the two agree only because both follow the specification (SP-6, RFC-0001). One
//! exposure is disclosed: while writing this crate, its author glimpsed doc comments of the
//! prover's `tessera-tables`, and then switched its computation of `Exp`'s zero tail to the
//! closed form described under 'Parameters' below.
//!
//! The verifier regenerates every table of a claim list from its definition and compares
//! digests (RFC-0007 'Verifier validation' step 4), then evaluates each lookup table's extension
//! at the point where its table tree ends, for the leaf check `q = α − t̃(ρ)` (RFC-0008
//! 'Fraction tree'). Where the specification admits several algorithms this crate takes its
//! own: `LT` and the affine prefix sums come from one recursion over the bits, the lowest first,
//! rather than a sum over the highest bit where `h` and the bound differ; dense cores are folded
//! one variable at a time, a chunk of `2^10` values at a time, rather than cut into aligned
//! blocks against tables of `eq`; and `Exp`'s zero tail comes from a closed form of its two
//! rules rather than a search over keys. The S-box is no such choice: it inverts by `a^254` in
//! GF(2^8), the usual method, which the prover's generator uses too. The oracle of
//! `spec/vectors/tables/` finds the inverses by search instead, and checks its S-box against
//! FIPS-197.
//!
//! | Item | What it computes or checks |
//! |---|---|
//! | [`TableSpec`], [`TableKind`], [`BitOp`], [`TableId`] | the claim list's table specifications, decoded and encoded as TCE |
//! | [`Table::from_spec`] | claim-list validation of one specification: its parameters, `log2_rows` and digest |
//! | [`Table::generate`] | the table a kind defines under an identifier, and the specification that names it |
//! | [`Table::eval_mle`] | `t̃(ρ) = kẽy(ρ) + β·valũe(ρ)` over the padded rows |
//! | [`Table::row`], [`Table::value`] | a padded row as field elements, a row's integer value |
//! | [`rope::Rope`], [`rope::constants`] | RoPE's frequencies and pairs, a `Rope` table's constants |
//! | [`fixed`] | the pinned constants and the functions of RFC-0008 'Fixed-point generators' |
//!
//! # Specification
//!
//! What this crate computes and rejects, written down from RFC-0008 ('Table kinds', 'Table
//! specifications', 'Structured evaluation', 'Generators and digests', 'Fixed-point
//! generators', 'Interfaces', 'Failure modes'), RFC-0003 ('Primitive types', 'Composite
//! types'), RFC-0002 ('Signed integers'), RFC-0004 ('Hash function and domains'), RFC-0005
//! ('Integer semantics'), RFC-0010 ('Multilinear extensions') and RFC-0007 ('Verifier
//! validation') before the code. The tests check every item against `spec/vectors/tables/`,
//! written by an independent Python oracle, and with inputs that break exactly one rule.
//!
//! ## Conventions
//!
//! - A table has `T` rows `(key_j, value_j)`, `j < T`, padded to `2^{log2_rows}` rows with
//!   copies of row 0, `log2_rows = ⌈log2 T⌉`. Keys and values are true integers, mapped to `Fp`
//!   in the centered representation: `v` for `v ≥ 0`, `P − |v|` for `v < 0` (RFC-0002 'Signed
//!   integers').
//! - A point `ρ` lists its coordinates lowest variable first: coordinate `i` binds bit `i` of the
//!   row index, and `eq(ρ, j) = Π_i (ρ_i·j_i + (1 − ρ_i)·(1 − j_i))` (RFC-0010 'Multilinear
//!   extensions'). `LT(ρ, a) = Σ_{h<a} eq(ρ, h)` for `0 ≤ a ≤ 2^{|ρ|}`.
//! - `/` is floor division and `>>` the arithmetic shift. A Q64 number `v` stands for `v / 2^64`.
//!
//! ## Specifications
//!
//! `TableSpec { id: TableId(u32), kind: TableKind, log2_rows: u8, digest: Digest }`, its fields
//! in this order (RFC-0003 'Composite types'). `TableKind` is an enum whose `u16` tag is the
//! variant's declaration index, followed by the variant's fields in order:
//!
//! | Tag | Kind | Fields |
//! |---|---|---|
//! | 0 | `Range` | `bits: u8` |
//! | 1 | `Requant` | `b: u8`, `r_by_class: Vec<u8>`, `lo: i32`, `hi: i32` |
//! | 2 | `Pow2` | `max_exp: u8` |
//! | 3 | `Bitwise8` | `op: BitOp`, an enum: `Xor` 0, `And` 1 |
//! | 4 | `AesSbox` | none |
//! | 5 | `Gf256Mul` | `c: u8` |
//! | 6 | `Exp` | `key_bits: u8`, `sigma: u8`, `scale: Rational` |
//! | 7 | `Exp2Frac` | `frac_bits: u8`, `out_bits: u8` |
//! | 8 | `Silu` | `b: u8`, `r_by_class: Vec<u8>`, `in_exp: i8`, `out_exp: i8` |
//! | 9 | `Rope` | `theta: u32`, `factor: u16`, `low_freq_factor: u16`, `high_freq_factor: u16`, `original_max: u32`, `bits: u8`, `max_context: u32`, `d_head: u16` |
//!
//! [`TableSpec::decode`] reads one with `tv-codec`'s cursor and its errors: a kind tag above 9
//! or an operation tag above 1 is `ERR-CODEC-NONCANONICAL` at the tag, a scale not in lowest
//! terms or with a zero denominator `ERR-CODEC-NONCANONICAL` at the rational, an input cut short
//! `ERR-CODEC-TRUNCATED` at the value it cuts. [`TableSpec::to_tce`] is its inverse
//! (INV-TCE-1, INV-TCE-2). Decoding checks no parameter: validation does. The classes of
//! `Requant` and `Silu`, which a hostile claim list may make as long as its limits allow, are
//! copied out of the input into memory reserved fallibly: memory that cannot be reserved is
//! `ERR-RES-MEMORY` at their length.
//!
//! ## Rows
//!
//! Row `j < T` of each kind; `c = ⌊j / 2^b⌋` is the class and `k = (j mod 2^b) − 2^{b−1}` the
//! signed key of `Requant` and `Silu`, whose `n = |r_by_class|`. Every key is `j`: a lookup packs
//! its operands into the row index (`(k + 2^{b−1}) + 2^b·c`, `a + 2^8·b` for `Bitwise8`).
//!
//! | Kind | `T` | Value of row `j` |
//! |---|---|---|
//! | `Range` | `2^bits` | 0 |
//! | `Requant` | `n·2^b` | `rq(k, r_c, lo, hi) = clamp(⌊(k + 2^{r_c−1}) / 2^{r_c}⌋, lo, hi)` |
//! | `Pow2` | `max_exp` | `2^j` |
//! | `Bitwise8` | `2^16` | `(j mod 2^8) ⊕ ⌊j / 2^8⌋`, or `∧` for `And` |
//! | `AesSbox` | `2^8` | `S(j)`: the inverse in GF(2^8) (0 for 0), then FIPS-197 §5.1.1's affine map |
//! | `Gf256Mul` | `2^8` | `c • j` modulo `x^8 + x^4 + x^3 + x + 1` |
//! | `Exp` | `2^key_bits` | 'Exp' below |
//! | `Exp2Frac` | `2^frac_bits` | `exp_pos(⌊j · LN2 / 2^frac_bits⌋) >> (62 − out_bits)` |
//! | `Silu` | `n·2^b` | 'Silu' below, at `h = k >> r_c` |
//! | `Rope` | `max_context · d_head / 2` | the pair `(C, S)` of position `⌊j / (d_head/2)⌋` and frequency `j mod (d_head/2)`, 'Rope' below |
//!
//! The functions `exp_neg`, `exp_pos`, `ln` and `cos_sin` and the constants `LN2`, `TAU` and
//! `HALF_PI` are those of RFC-0008 'Fixed-point generators', in [`fixed`].
//!
//! - **Exp.** With `scale = num / den`, key `k`: `k·num = q·den + m`; the value is 0 when
//!   `q ≥ 2^20`; otherwise `x = q·2^64 + ⌊m·2^64 / den⌋` and `(y, n) = exp_neg(x)`, and the value
//!   is 0 when `n ≥ σ + 2` and `(y·2^σ + 2^{63+n}) >> (64 + n)` otherwise. The zero tail is the
//!   suffix of keys with `q ≥ 2^20` or `x ≥ (σ + 2)·LN2`; its first key `z`, or `T` when it is
//!   empty, is the table's one piece, and its core the values of keys `0` to `z − 1`.
//! - **Silu.** `(y, n) = exp_neg(|h|·2^{64 + in_exp})`, `E = y >> n` or 0 when `n ≥ 128`;
//!   `σ = ⌊2^127 / (2^64 + E)⌋` for `h ≥ 0` and `⌊E·2^63 / (2^64 + E)⌋` for `h < 0`; the value is
//!   `clamp((h·σ + 2^{s−1}) >> s, −128, 127)` with `s = 63 − in_exp + out_exp`. The core is the
//!   value at every `h` of every class, class by class, `h` from `−2^{b−1−r_c}` up.
//! - **Rope.** `ω_i = y >> n` for `(y, n) = exp_neg(⌊2i·ln(θ) / d_head⌋)`, then Llama-3's bands
//!   against `L = original_max`: kept when `TAU·high_freq_factor < L·ω`, `⌊ω / factor⌋` when
//!   `TAU·low_freq_factor > L·ω`, else `⌊ω / factor⌋ + ⌊⌊ω·smooth / 2^40⌋·(factor − 1) / factor⌋`
//!   with `smooth = ⌊(L·ω − low_freq_factor·TAU)·2^40 / ((high_freq_factor − low_freq_factor)·TAU)⌋`.
//!   The pair of position `p` at frequency `i` is `cos_sin(p·ω_i)`, each part rounded half up as
//!   `(v·2^bits + 2^61) >> 62`. The core is every pair in row order, `C` then `S`.
//!
//! ## Parameters
//!
//! [`Table::generate`] and [`Table::from_spec`] check a kind's parameters (RFC-0008 'Table
//! specifications', 'Parameters allowed') in this order, before anything is allocated, and the
//! first rule that fails is the error:
//!
//! | Kind | Rule, in order | Error |
//! |---|---|---|
//! | `Range` | `1 ≤ bits ≤ 32` | `RangeBits` |
//! | `Requant` | `b ≥ 1`; `1 ≤ n ≤ 4`; `b + ⌈log2 n⌉ ≤ 32`; class by class `1 ≤ r_c ≤ 63`; `lo ≤ hi` | `KeyBits`, `Classes`, `RowBits`, `RequantShift`, `RequantClamp` |
//! | `Pow2` | `1 ≤ max_exp ≤ 63` | `Pow2Exponent` |
//! | `Bitwise8`, `AesSbox`, `Gf256Mul` | none | |
//! | `Exp` | `1 ≤ key_bits ≤ 32`; `σ ≤ 32`; `scale > 0`; `z ≤ 2^24` | `ExpKeyBits`, `ExpSigma`, `ExpScale`, `ExpTail` |
//! | `Exp2Frac` | `1 ≤ frac_bits ≤ 16`; `1 ≤ out_bits ≤ 56` | `Exp2FracBits`, `Exp2FracOutBits` |
//! | `Silu` | `b ≥ 1`; `1 ≤ n ≤ 4`; `b + ⌈log2 n⌉ ≤ 32`; class by class `r_c < b`; `Σ_c 2^{b − r_c} ≤ 2^24`; `−64 ≤ in_exp ≤ 0`; `1 ≤ 63 − in_exp + out_exp ≤ 126` | `KeyBits`, `Classes`, `RowBits`, `SiluShift`, `SiluCore`, `SiluInExp`, `SiluOutShift` |
//! | `Rope` | `θ ≥ 2`; `factor ≥ 1`; `1 ≤ low_freq_factor < high_freq_factor`; `original_max ≥ 1`; `1 ≤ bits ≤ 30`; an even `d_head ≥ 2`; `max_context ≥ 1`; at most `2^24` pairs | `RopeTheta`, `RopeFactor`, `RopeFrequencyFactors`, `RopeOriginalMax`, `RopeBits`, `RopeHeadDim`, `RopeContext`, `RopePairs` |
//!
//! `Exp`'s `z` is computed before its core is generated. With `scale = num / den`, `q ≥ 2^20`
//! exactly from `k_1 = ⌈2^20·den / num⌉` on, and `x ≥ (σ + 2)·LN2` exactly from
//! `k_2 = ⌈(σ + 2)·LN2·den / (num·2^64)⌉` on, since `x = ⌊k·num·2^64 / den⌋` below `k_1`: so
//! `z = min(T, k_1, k_2)`, and the rule, applied to keys `z − 1` and `z`, confirms it. Every
//! limit on a core, `z ≤ 2^24`, `Σ_c 2^{b − r_c} ≤ 2^24` and `2^24` pairs, is checked before its
//! memory is reserved, and memory that cannot be reserved is `Memory`.
//!
//! ## Validation
//!
//! [`Table::from_spec`]`(spec)` regenerates the table from `spec.id` and `spec.kind` and checks,
//! in this order (RFC-0008 'Table specifications'):
//!
//! 1. the parameters, as above;
//! 2. `spec.log2_rows = ⌈log2 T⌉`, else `Log2Rows { expected, found }`;
//! 3. `spec.digest` equals the regenerated digest, else `Digest`.
//!
//! The first two read the specification where it lies: nothing is copied, allocated or
//! generated before they pass, and the number of classes is read from their length before any
//! class is. A specification is copied into its table only once all three have passed, with at
//! most four classes.
//!
//! The regenerated digest is `H("table", TCE(id, kind, log2_rows) ‖ TCE(pieces) ‖ TCE(core))`
//! (RFC-0004 'Hash function and domains'), the pieces a `Vec<u64>` and the core a `Vec<Fp>` in
//! row order without padding, each a `u64` length and then its elements (RFC-0008 'Generators and
//! digests'). `Exp` has the piece `z` and its core; `Pow2`, `Bitwise8`, `AesSbox`, `Gf256Mul` and
//! `Exp2Frac` no piece and every value; `Silu` and `Rope` no piece and their cores; `Range` and
//! `Requant` neither. A digest binds the identifier: the same kind under another id is another
//! table.
//!
//! ## Structured evaluation
//!
//! [`Table::eval_mle`]`(ρ, β)` returns `t̃(ρ) = Σ_{j < 2^{log2_rows}} eq(ρ, j)·(key_j + β·value_j)`
//! over the padded rows, which is `Σ_{j<T} eq(ρ, j)·t_j + t_0·(1 − LT(ρ, T))` (RFC-0008
//! 'Structured evaluation'). It first checks that the table is a lookup table, else
//! `NotLookupTable` (`Rope` holds constants for `Rotate`, RFC-0008 'Table kinds'), then that `ρ`
//! has `log2_rows` coordinates, else `PointLength`.
//!
//! - `kẽy(ρ)` is the prefix sum below `T` of the affine function `j = Σ_i 2^i·j_i`; row 0's key
//!   is 0, so the padding adds nothing to it.
//! - `LT(ρ, a)` and `Σ_{h<a} eq(ρ, h)·(w_0 + Σ_i w_i·h_i)` take one pass over the bits of `a`,
//!   lowest first, `O(n)`, for any bound `0 ≤ a ≤ 2^n`.
//! - `Range` has no value. A `Requant` class `c` with shift `r` holds `lo` on a prefix of its
//!   offset keys `x`, `hi` on a suffix, and between them `⌊(x − 2^{b−1} + 2^{r−1}) / 2^r⌋`, which
//!   is `(x >> r) + x_{r−1} − 2^{b−1−r}` for `r < b` and 0 for `r ≥ b`: two `LT` and two affine
//!   prefix sums over its `b` variables, `O(b)` per class whatever `b`, `lo` and `hi`, times
//!   `eq` of the class bits.
//! - A `Silu` class's values depend only on `x >> r_c`, so its sum is its core's extension at
//!   `ρ_{r_c}, …, ρ_{b−1}`: the variables below `r_c` sum out. `Exp`'s core is zero from `z` on.
//!   `Pow2`, `Bitwise8`, `AesSbox`, `Gf256Mul` and `Exp2Frac` hold every value. Each of these
//!   dense vectors is folded variable by variable, `O(core)` in all, with no division, so
//!   coordinates 0 and 1 need no special case.
//!
//! ## Errors
//!
//! [`TableError`] says whose fault an error is ([`is_rejection`](TableError::is_rejection)):
//!
//! | Errors | Fault | Verdict (04 §2) |
//! |---|---|---|
//! | the parameter errors above, `Log2Rows`, `Digest` | the claim list's | `Reject(StatementMismatch)` at claim-list validation (RFC-0007 step 4) |
//! | `NotLookupTable`, `PointLength` | the verifier's: what it passes in | `Inconclusive(ERR-INTERNAL)` |
//! | `Memory` | the verifier's: memory for a core that the limits admit could not be reserved | `Inconclusive(ERR-RES-MEMORY)` |
//! | `Internal` | the verifier's: a bound the parameter checks guarantee failed, a bug of this crate that no input reaches | `Inconclusive(ERR-INTERNAL)` |
//!
//! `NotLookupTable` is the verifier's fault under a precondition on its caller. A claim list
//! that looks up a `Rope` table, which RFC-0008 'Table kinds' makes no lookup table, is at fault
//! itself, and `tv-core` must refuse it with `Reject(StatementMismatch)` at claim-list
//! validation, where every lookup is checked against its table (RFC-0007 'Verifier validation'
//! step 5), before any call of [`Table::eval_mle`]. `eval_mle` on a `Rope` table then means that
//! the verifier skipped that check. `PointLength` is the verifier's for the same reason: the
//! point's length is the height of a table tree, which the verifier derives from `log2_rows`.
//!
//! ## Robustness and cost
//!
//! Every input yields a value or a typed error: nothing here panics, indexes unchecked or
//! computes unchecked, and every intermediate of the fixed-point functions is checked against
//! `2^128` (RFC-0019 'Robustness'). A specification of any parameters costs at most:
//!
//! - its checks, with two `Exp` keys examined to confirm `z`, and nothing copied, however many
//!   classes it holds;
//! - a dense core of at most `2^24` values, one `exp_neg` each, or `2^24` RoPE pairs, one
//!   `cos_sin` each, held as `i64` (`i8` for `Silu`), or never held for `Rope`, whose pairs are
//!   hashed as they are computed. `Exp`'s values reach `2^σ`, `2^32` at `σ = 32`, which no
//!   32-bit type holds, so its core takes 128 MiB at its limit;
//! - an evaluation in `O(core)` field operations with `O(core / 2^10)` values of memory beside
//!   the core, and `O(log2_rows)` per class for `Range` and `Requant`, whose `2^32` rows are
//!   never enumerated.

#![forbid(unsafe_code)]
// A claim list comes from a hostile prover, and its table parameters must be rejected with typed
// errors, never by panicking (RFC-0019 'Robustness'). Integer arithmetic is checked or
// saturating, conversions are explicit, and field arithmetic is exempt through its own types
// (`clippy.toml`).
#![deny(
    clippy::unwrap_used,
    clippy::expect_used,
    clippy::panic,
    clippy::unreachable,
    clippy::indexing_slicing,
    clippy::arithmetic_side_effects,
    clippy::as_conversions
)]
#![cfg_attr(
    test,
    allow(
        clippy::unwrap_used,
        clippy::expect_used,
        clippy::panic,
        clippy::unreachable,
        clippy::indexing_slicing,
        clippy::arithmetic_side_effects,
        clippy::as_conversions
    )
)]

mod error;
mod evaluation;
pub mod fixed;
mod galois;
pub mod rope;
mod spec;
mod table;

pub use error::TableError;
pub use spec::{BitOp, TableId, TableKind, TableSpec};
pub use table::{MAX_CORE, Table};
